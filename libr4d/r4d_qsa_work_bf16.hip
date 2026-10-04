// The QSA indexer's WORK LIST: which blocks complete this step, where their raw keys are, and
// where their block key goes.
//
// `qsa_block_key` takes a flat list of blocks and reads each one's `ratio` raw keys from
// consecutive rows of one plane. Nothing in a batch hands it that. The keys of a block completing
// at DECODE are three rows of the tail and one row of this step; the keys of a block completing
// inside a PREFILL chunk are four rows of the chunk; and a block straddling the start of a chunk
// is some of each. The blocks themselves belong to different sequences at different positions.
//
// So this op gathers. It walks the sequences, works out which blocks became complete, copies each
// one's `ratio` raw keys into consecutive rows of a staging plane, and writes the block's
// destination page and first-token position beside it. What comes out is exactly the three
// operands `qsa_block_key` wants, and that kernel does not learn that two sources existed.
// docs/QSA.md is the decomposition.
//
// THE TAIL IS A LINEAR KV STATE. `state` is the tail group's cache for ONE layer,
// `[slots, 1, ring, n]`, and `sidx` names this sequence's slot -- not the batch row, which is
// step-local and would hand a long-lived request the keys of a shorter one that used to sit
// there. r4d_qsa_tail_store_bf16.hip's header is the whole argument, including why the ring is
// `ratio + n_spec` rather than `ratio`; this side only has to read `q % ring` to match.
//
// THE COPY IS THE POINT AND IT IS CHEAP. A kilobyte a completing block: 512 KiB for a full
// 2048-token prefill chunk a layer, four kilobytes for a decode step. The alternative is a
// block-key kernel that takes two sources, a per-key source selector and a tail layout -- three
// more things to get wrong inside the kernel that does the arithmetic.
//
// THE LAUNCH IS BOUNDED ON THE HOST, NOT COUNTED ON THE DEVICE. How many blocks complete is
// device data, but `ceil(T / ratio) + n_seq` bounds it from numbers the caller already has -- one
// block a `ratio` tokens, plus at most one straddling block a sequence. Slots past the real count
// get `page = -1`, and `qsa_block_key` skips those, so the bound costs empty workgroups that exit
// on their first load and never a wrong answer.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
/* The sequence loop's bound. It is NOT a constraint on the declared `seqs`, which is a deployment
 * maximum -- rad-convert declares at 256 and a row that refused that would not resolve at convert
 * and its weights would never be written. The real count comes off `cu_seqlens` at issue and the
 * launcher checks it there; the loop only ever runs to that. */
constexpr int kMaxSeq  = 256;

__global__ __launch_bounds__(kThreads) void qsa_work_kernel(
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const uint16_t* __restrict__ state,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ bt,
    uint16_t* __restrict__ stage, int32_t* __restrict__ page, int32_t* __restrict__ bpos,
    int32_t* __restrict__ nc,
    int n, int ratio, int ring, int seqs, int max_work, int k_ld, int s_slot, int s_row,
    int sidx_ld, int bt_ld, int g_ld, const int32_t* __restrict__ rope, int rplane,
    int bstep) {

    const int w = (int)blockIdx.x;
    /* COMPLETE BLOCKS A SEQUENCE, written by the first `seqs` workgroups on their way past. The
     * scorer and the selection both need it and it is one division of a number this kernel has
     * already loaded; a launch a layer a step for that alone would be five microseconds of
     * dispatch for four instructions. */
    if (w < seqs && threadIdx.x == 0) {
        const int t1 = cu[w + 1];
        nc[w] = t1 > cu[w] ? (pos[t1 - 1] + 1) / ratio : 0;
    }
    if (w >= max_work) return;

    /* WHICH SEQUENCE OWNS WORK ITEM w. A running count over the sequences, recomputed by every
     * workgroup: `seqs` is a handful and the alternative is a second launch to build a prefix. */
    int s = -1, j = 0;
    int run = 0;
    for (int i = 0; i < seqs && i < kMaxSeq; ++i) {
        const int t0 = cu[i], t1 = cu[i + 1];
        if (t1 <= t0) continue;
        const int p0 = pos[t0], p1 = pos[t1 - 1];
        /* A block is complete when its LAST token exists: 4b + ratio - 1 in [p0, p1]. Both
         * bounds are taken on non-negative numerators, because C division truncates toward zero:
         * `(p1 - (ratio - 1)) / ratio` is 0 rather than -1 for p1 < ratio - 1, which would count
         * block 0 complete from its first token and gather keys past the end of the sequence. */
        const int lo = p0 / ratio;                                  /* ceil((p0-ratio+1)/ratio) */
        const int hi = (p1 + 1) / ratio - 1;                        /* floor((p1-ratio+1)/ratio) */
        const int cnt = hi >= lo ? hi - lo + 1 : 0;
        if (w < run + cnt) { s = i; j = w - run + lo; break; }
        run += cnt;
    }
    if (s < 0) {
        /* Past the real count. `page = -1` is what makes the host's bound safe. */
        if (threadIdx.x == 0) {
            page[w] = -1;
            for (int c = 0; c < (rope ? 3 : 1); ++c) bpos[(size_t)w * bstep + c] = 0;
        }
        return;
    }

    const int b  = j;                       /* the block index within the sequence */
    const int t0 = cu[s];
    const int p0 = pos[t0];
    /* Column 0 is the committed slot; -1 is the pad for a sequence with no state. A block whose
     * earlier keys are in a state that does not exist cannot be built, and writing a page out of
     * whatever the pointer arithmetic reached is the failure this guard exists to prevent. */
    const int si = sidx[(size_t)s * sidx_ld];

    if (threadIdx.x == 0) {
        page[w] = si < 0 ? -1 : bt[(size_t)s * bt_ld + b];
        if (!rope) {
            bpos[(size_t)w * bstep] = b * ratio;
        } else {
            /* THE ROTARY POSITION OF THE BLOCK'S FIRST TOKEN, three components, a work item's
             * three side by side and items `bstep` apart. A first token in this step reads its
             * own components. One committed in an earlier step is TEXT -- a prefill chunk ends on a
             * whole block, so only a decode row or a prompt's last chunk leaves a block open, and a
             * prompt ends in text -- and text rotates at its index plus the shift this step's first
             * row of the sequence carries, the same shift, because nothing but text lies between.
             * */
            const int f = b * ratio;
            for (int c = 0; c < 3; ++c)
                bpos[(size_t)w * bstep + c] =
                    f >= p0 ? rope[(size_t)c * rplane + t0 + (f - p0)] : f + (rope[t0] - p0);
        }
    }
    if (si < 0) return;

    /* The block's `ratio` raw keys, from whichever side of the chunk boundary each one is on. */
    for (int i = 0; i < ratio; ++i) {
        const int q = b * ratio + i;                 /* this key's position */
        const uint16_t* src;
        if (q >= p0) src = k + (size_t)(t0 + (q - p0)) * k_ld;
        else         src = state + (size_t)si * s_slot + (size_t)(q % ring) * s_row;
        uint16_t* dst = stage + (size_t)(w * ratio + i) * g_ld;
        for (int d = (int)threadIdx.x; d < n; d += kThreads) dst[d] = src[d];
    }
}

}  // namespace

//   k      [T][n]                  bf16, this step's raw index keys
//   pos    [T]                     i32
//   cu     [seqs + 1]              i32
//   state  [slots][ring][n]        bf16, the tail group's cache for this layer
//   sidx   [seqs][sidx_ld]         i32, the tail group's state row; column 0 is the slot
//   bt     [seqs][bt_ld]           i32, the BLOCK-KEY group's block table
//   stage  [max_work * ratio][n]   bf16
//   page   [max_work]              i32, -1 past the real count
//   bpos   [max_work][bstep]       i32, a work item's first-token position at [w * bstep]
//   rope   [3][rplane]             i32, optional: this step's rotary components (RadBatch::
//                                  rope_pos); present, a work item's three components are
//                                  bpos[w * bstep + 0..2]
static int r4d_qsa_work_bf16_emit(long k, long pos, long cu, long state, long sidx, long bt,
                                 long stage, long page, long bpos, long nc, int n, int ratio,
                                 int ring, int seqs, int max_work, int k_ld, int s_slot,
                                 int s_row, int sidx_ld, int bt_ld, int g_ld, long rope,
                                 int rplane, int bstep, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!k || !pos || !cu || !state || !sidx || !bt || !stage || !page || !bpos || !nc) return -1;
    if (n <= 0 || ratio <= 0 || ring < ratio || seqs <= 0 || max_work <= 0) return -2;
    if (seqs > kMaxSeq) return -5;
    if (!k_ld) k_ld = n;
    if (!s_row) s_row = n;
    if (!g_ld) g_ld = n;
    if (!sidx_ld) sidx_ld = 1;
    emit_rc = R4D_EMIT(sk, qsa_work_kernel, dim3((unsigned)max_work), dim3(kThreads), 0, (const uint16_t*)k, (const int32_t*)pos,
                       (const int32_t*)cu, (const uint16_t*)state, (const int32_t*)sidx,
                       (const int32_t*)bt, (uint16_t*)stage, (int32_t*)page, (int32_t*)bpos,
                       (int32_t*)nc, n, ratio, ring, seqs, max_work, k_ld, s_slot, s_row,
                       sidx_ld, bt_ld, g_ld, (const int32_t*)rope, rplane, bstep);
    return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
}

extern "C" int r4d_qsa_work_bf16_emit(long k, long pos, long cu, long state, long sidx, long bt,
                                 long stage, long page, long bpos, long nc, int n, int ratio,
                                 int ring, int seqs, int max_work, int k_ld, int s_slot,
                                 int s_row, int sidx_ld, int bt_ld, int g_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return r4d_qsa_work_bf16_emit(k, pos, cu, state, sidx, bt, stage, page, bpos, nc, n, ratio, ring, seqs, max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, 0, 0, 1, sk);
}

// ============================================================== the plugin ABI entry point

enum { QW_K = 0, QW_POS, QW_CU, QW_STATE, QW_SIDX, QW_BT, QW_ROPE, QW_STAGE, QW_PAGE, QW_BPOS,
       QW_NC };

static int r4d_rad_qsa_work_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* k  = r4d_opt(a, QW_K);
    const RadTensor* po = r4d_opt(a, QW_POS);
    const RadTensor* cu = r4d_opt(a, QW_CU);
    const RadTensor* tl = r4d_opt(a, QW_STATE);
    const RadTensor* sx = r4d_opt(a, QW_SIDX);
    const RadTensor* bt = r4d_opt(a, QW_BT);
    const RadTensor* rp = r4d_opt(a, QW_ROPE);       /* OPTIONAL: multi-component positions */
    const RadTensor* st = r4d_opt(a, QW_STAGE);
    const RadTensor* pg = r4d_opt(a, QW_PAGE);
    const RadTensor* bp = r4d_opt(a, QW_BPOS);
    const RadTensor* nc = r4d_opt(a, QW_NC);
    if (!k || !po || !cu || !tl || !sx || !bt || !st || !pg || !bp || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (k->rank != 2 || bt->rank != 2 || st->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* The tail state arrives as its group declared it: [slots, heads, ring, n], heads 1. See
     * r4d_qsa_tail_store_bf16.hip. */
    if (tl->rank != 4 || tl->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (tl->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    /* `bt` here is the BLOCK-KEY group's table, not the attention group's. They are different
     * pages for the same sequence and the join between them is the block INDEX, which is the same
     * number in both because both groups are paged at `ratio` (docs/QSA.md). */
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long ring  = r4d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R4D_NO(RAD_E_INVAL);
    const long long seqs = r4d_numel(cu) - 1;
    const long long T    = k->shape[0];
    if (seqs < 1 || T < 1 || r4d_numel(po) < T) return R4D_NO(RAD_E_SHAPE);
    if (k->shape[1] != n || st->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (tl->shape[2] < ring || tl->shape[3] != n) return R4D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* THE WORK COUNT COMES OFF `page` AND `stage` MUST BE `ratio` TIMES IT. Taking it off `stage`
     * instead would let a caller pass a plane sized for more blocks than it has slots to name,
     * and the overflow would be silent -- the kernel writes keys for a block whose page nobody
     * reads. */
    const long long mw = r4d_numel(pg);
    if (mw < 1 || st->shape[0] < mw * ratio) return R4D_NO(RAD_E_SHAPE);
    /* `bpos` is one position a work item, or a rank-2 [>= work, 1 or 3] column slice of a buffer
     * declared [max_work, 3]: a work item's components side by side and the items a row pitch
     * apart, so the buffer's rows are its work items and a smaller step is fewer of them. Three
     * columns are required with `rope`. `rope` itself is [3, rplane] component-major, a plane at
     * least as long as the step. */
    long long bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return R4D_NO(RAD_E_SHAPE);
        bstep = bp->stride[0];
    } else if (rp || r4d_numel(bp) < mw) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T || !r4d_contig(rp) ||
               rp->dtype != RAD_I32))
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(nc) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* The per-sequence output is written by the first `seqs` workgroups, so a launch narrower
     * than the batch would leave some of it stale. */
    if (mw < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k);
    const long long bp2 = r4d_row_pitch(bt), sp = r4d_row_pitch(st);
    if (kp < 0 || bp2 < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || tl->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_qsa_work_bf16_emit(r4d_addr(k), r4d_addr(po), r4d_addr(cu), r4d_addr(tl),
                                    r4d_addr(sx), r4d_addr(bt), r4d_addr(st), r4d_addr(pg),
                                    r4d_addr(bp), r4d_addr(nc), (int)n, (int)ratio, (int)ring,
                                    (int)seqs, (int)mw, (int)kp, (int)tl->stride[0],
                                    (int)tl->stride[2], (int)sxp, (int)bp2, (int)sp,
                                    r4d_addr(rp), rp ? (int)rp->shape[1] : 0, (int)bstep, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

R4D_ENTRY(r4d_rad_qsa_work, r4d_rad_qsa_work_body)
