// r4d_ngram_ids.hip -- the hashed row ids of Qwen4-Exp's Per-Layer Embedding.
//
// ============================== WHAT IT COMPUTES ==============================
//
//   mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]        int64, wrapping
//   ids[h]   = mixed(b+1) % vocab_sizes[h] + offsets[h]          h in block b
//
// with `heads` laid out as `ngram - 1` equal blocks, block b using the (b+2)-gram, and `t_s` the
// token `s` positions back OR the EOS id if the shift would cross one. libref/ref_ngram.cpp holds
// the derivation of that EOS rule -- it collapses a cummax over segment starts into "once a shift
// falls back, every deeper one does" -- and docs/OPS.md holds the operand contract.
//
// ============================== ONE THREAD A TOKEN, NOT A HEAD ==============================
//
// The obvious grid is one thread per (token, head): 16x the parallelism and every thread doing one
// modulo. It is the wrong one. The sixteen heads of a token share the THREE shifted tokens and the
// `ngram - 1` hash mixes -- the whole EOS walk and every multiply -- and differ only in which
// constant they reduce against. A thread per head redoes that work sixteen times, and at decode the
// grid is one token: sixteen threads either way, and the per-head form is sixteen partial waves
// instead of one.
//
// So a thread owns a token and loops its heads. At prefill the grid is the token count, which is
// thousands; at decode it is 1 to 8 and the kernel is a rounding error beside the 5 KB gather that
// follows it. Neither end wants more threads.
//
// ============================== THE MODULO IS THE COST, AND IT IS FINE ==============================
//
// `mixed % vocab_sizes[h]` is a 64-bit division by a runtime value -- no reciprocal, no strength
// reduction, ~70 cycles on this part -- sixteen per token. At decode that is ~1100 cycles for the
// whole step's ids. Magic-number division would remove it and would mean the ARCHITECTURE computing
// a multiplier and a shift per head at declare and passing them as a second weight; it is not worth
// a second source of truth for sixteen divisions a token. If a model ever hashes at prefill width
// with a hundred heads, that is the change to make.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <cstdint>

namespace {

/* THE SAME TWO OPERATIONS libref SPELLS OUT, and for the same reason: a signed overflow is UB and
 * torch's `remainder` FLOORS where C truncates. On a real checkpoint neither can fire -- the
 * multipliers are bounded so a real token's product has its sign bit clear -- but the kernel and
 * the oracle have to agree on inputs a harness draws, not only on the ones a model produces. */
__device__ __forceinline__ long long ng_wrap_mul(long long a, long long b) {
    return (long long)((unsigned long long)a * (unsigned long long)b);
}
__device__ __forceinline__ long long ng_floor_mod(long long a, long long b) {
    const long long r = a % b;
    return r < 0 ? r + b : r;
}

enum { kNgMaxShift = 8, kNgThreads = 256 };

/* ONE WORKGROUP A (TOKEN TILE, SEQUENCE), AND THE WINDOW WRITE-BACK IS IN IT.
 *
 * `s` is blockIdx.y, so the sequence scan is gone and so is the second launch. What makes the
 * fold legal is WHICH THREADS READ THE WINDOW: only a token with `t - bos < ctx` reaches back
 * past its own sequence's first row, and those are the first `ctx` tokens, which live in tile
 * ZERO. So tile 0's block is the only reader of state[slot], its threads are ordered against each
 * other by __syncthreads(), and thread 0 writes the new window after that barrier. No other block
 * touches those bytes and no grid-wide barrier is needed.
 *
 * AT DECODE THIS OP IS NOTHING BUT LAUNCHES: its cost is flat from M=1 to M=512, which is the
 * dispatches and none of the work. Folding the write-back into the same launch, rather than
 * taking a second one for it, is therefore most of what the op costs. */
__global__ __launch_bounds__(kNgThreads) void ngram_ids_kernel(
    const int32_t* __restrict__ tok, int32_t* __restrict__ state,
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,
    const long long* __restrict__ vsz, const long long* __restrict__ off,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ hini,
    const int32_t* __restrict__ nacc, int32_t* __restrict__ ids,
    uint32_t heads, uint32_t ngram, uint32_t per, uint32_t W,
    long long eos, uint32_t ids_ld, int32_t* __restrict__ ahead, uint32_t n_ahead,
    uint32_t ahead_ld) {

    const uint32_t s   = blockIdx.y;
    const uint32_t bos = (uint32_t)cu[s];
    const uint32_t eot = (uint32_t)cu[s + 1];
    const uint32_t ctx = ngram - 1u;
    /* UNIFORM EXITS ONLY before the barrier below: an empty sequence and a negative slot are
     * properties of the block, not of the thread, so every thread of the block leaves together. */
    if (eot <= bos) return;
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;

    const uint32_t slen = eot - bos;
    const bool warm = !hini || hini[s] != 0;
    const long long o = nacc ? (long long)nacc[s] - 1 : 0;
    const uint32_t lt = blockIdx.x * kNgThreads + threadIdx.x;   /* this thread's row, in-sequence */

    /* THE PROMPT THAT FOLLOWS continues the LAST sequence: its tokens sit in `tok` right after
     * the step's, so a row past the sequence's end hashes exactly as that row will when the next
     * step runs it -- same history, same EOS rule -- and lands in `ahead` instead of `ids`. The
     * window below is the step's own and never sees them. */
    const uint32_t reach = slen + ((ahead && s + 1u == gridDim.y) ? n_ahead : 0u);
    if (lt < reach) {
        const uint32_t t = bos + lt;
        /* The `ctx` committed ids in front of this sequence's first row. A COLD slot's window is
         * EOS and not what the slot last held -- and not zero, which is a real token id. */
        long long prev[kNgMaxShift];
        for (uint32_t i = 0; i < ctx; ++i)
            prev[i] = warm ? (long long)state[(long long)slot * W + o + i] : eos;

        /* The shifted tokens. `p` indexes the history `prev ++ this sequence's rows`, so row `lt`
         * sits at `ctx + lt` and every shift is in range without a bounds test. */
        const uint32_t p = ctx + lt;
        long long sh[kNgMaxShift];
        sh[0] = (long long)tok[t];
        bool cut = false;
        for (uint32_t k = 1; k < ngram; ++k) {
            if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
            const uint32_t q = p - k;
            sh[k] = q < ctx ? prev[q] : (long long)tok[bos + (q - ctx)];
        }

        /* One mix per n-gram order, extended in place: the (b+2)-gram's mix is the (b+1)-gram's
         * XOR'd with one more term, which is the reference's own inner loop. */
        long long mixed = 0;
        int blk = -1;
        for (uint32_t k = 0; k < ngram; ++k) {
            const long long term = ng_wrap_mul(sh[k], mult[k]);
            mixed = k == 0 ? term : (mixed ^ term);
            if (k == 0) continue;
            ++blk;
            int32_t* const row = lt < slen ? ids + (long long)t * ids_ld
                                           : ahead + (long long)(lt - slen) * ahead_ld;
            for (uint32_t j = 0; j < per; ++j) {
                const uint32_t h = (uint32_t)blk * per + j;
                row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
            }
        }
    }

    /* THE WINDOW THIS STEP LEAVES BEHIND, in libref/ref_ngram.cpp's two layouts.
     *
     * WITH `num_accepted` the step is a decode step whose drafts a verify may reject, so the
     * window is shifted down by one and EVERY id of the step follows it: slot i holds position
     * i + 1 of (window read ++ step). A next step that kept k of these ids reads at k - 1 and
     * finds the `ctx` ids ending at the k-th. Slot i is read from o + i + 1, which is past every
     * slot already written, so the ascending walk is safe in place. Slots past W are not written;
     * the manager sizes the window ctx + n_spec, which covers a step of 1 + n_spec ids.
     *
     * WITHOUT it every id commits and the last `ctx` land at offset zero.
     *
     * ONE THREAD, not one per slot: a thread per (sequence, k) races within the sequence -- at
     * ctx = 2 and slen = 1, thread 0 reads slot o+1 while thread 1 writes slot 1 -- and at decode
     * the serial form is a loop of ctx - 1 + slen stores, so removing the class costs nothing. */
    __syncthreads();
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int32_t* const win = state + (long long)slot * W;
    if (nacc) {
        for (uint32_t k = 0; k + 1 < ctx; ++k)
            win[k] = warm ? win[o + k + 1] : (int32_t)eos;
        for (uint32_t t = 0; t < slen && ctx - 1u + t < W; ++t) win[ctx - 1u + t] = tok[bos + t];
        return;
    }
    long long keep[kNgMaxShift];
    for (uint32_t k = 0; k < ctx; ++k) {
        const long long src = (long long)slen - (long long)ctx + (long long)k;
        keep[k] = src >= 0 ? (long long)tok[bos + src]
                           : (warm ? (long long)win[o + src + ctx] : eos);
    }
    for (uint32_t k = 0; k < ctx; ++k) win[k] = (int32_t)keep[k];
}

}  // namespace

enum { NG_TOK = 0, NG_STATE, NG_CU, NG_MULT, NG_VSZ, NG_OFF, NG_CIDX, NG_HINIT, NG_NACC, NG_IDS,
       NG_AHEAD };

static int ngram_ids_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* tok  = r4d_opt(a, NG_TOK);
    const RadTensor* st   = r4d_opt(a, NG_STATE);
    const RadTensor* cu   = r4d_opt(a, NG_CU);
    const RadTensor* mult = r4d_opt(a, NG_MULT);
    const RadTensor* vsz  = r4d_opt(a, NG_VSZ);
    const RadTensor* off  = r4d_opt(a, NG_OFF);
    const RadTensor* cidx = r4d_opt(a, NG_CIDX);
    const RadTensor* hini = r4d_opt(a, NG_HINIT);
    const RadTensor* nacc = r4d_opt(a, NG_NACC);
    const RadTensor* ids  = r4d_opt(a, NG_IDS);
    const RadTensor* ahd  = r4d_opt(a, NG_AHEAD);
    if (!tok || !st || !cu || !mult || !vsz || !off || !ids) return R4D_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || st->dtype != RAD_I32 || ids->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (mult->dtype != RAD_I64 || vsz->dtype != RAD_I64 || off->dtype != RAD_I64)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tok) || !r4d_contig(st) || !r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    const long long M     = r4d_geti_or(a, "M", 0);
    const long long heads = r4d_geti_or(a, "heads", 0);
    const long long ngram = r4d_geti_or(a, "ngram", 0);
    const long long eos   = r4d_geti_or(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return R4D_NO(RAD_E_SHAPE);
    if (ngram > kNgMaxShift) return R4D_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return R4D_NO(RAD_E_SHAPE);
    const long long per = heads / (ngram - 1), ctx = ngram - 1;

    long long rows = 0;
    const long long ids_ld = r4d_rows_pitch(ids, heads, &rows);
    if (ids_ld < 0 || rows <= 0) return R4D_NO(RAD_E_STRIDE);
    /* THE ROW COUNT COMES FROM THE OPERAND. `M` is RANGED, so the value in the args is the band's
     * upper bound and a grid sized by it walks off the arena. */
    const long long T = rows;
    /* The prompt that follows: `ahead`'s rows, whose tokens `tok` holds after the step's. */
    long long A = 0, ahead_ld = heads;
    if (ahd) {
        if (ahd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
        ahead_ld = r4d_rows_pitch(ahd, heads, &A);
        if (ahead_ld < 0) return R4D_NO(RAD_E_STRIDE);
    }
    if (r4d_numel(tok) < T + A) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(mult) < ngram || r4d_numel(vsz) < heads || r4d_numel(off) < heads)
        return R4D_NO(RAD_E_SHAPE);

    const long long N = r4d_numel(cu) - 1;
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long W = st->rank >= 2 ? r4d_dim(st, (int)st->rank - 1) : ctx;
    if (W < ctx) return R4D_NO(RAD_E_SHAPE);

    /* (token tiles, sequences). The tile count is taken from the batch's TOTAL token count,
     * which is an upper bound on any one sequence's -- a short sequence in a mixed batch launches
     * a few blocks that find their rows past its end, and the waste is bounded by the sequence
     * count. Reading a per-sequence tile count would mean reading `cu` on the host. */
    const uint32_t tiles = (uint32_t)((T + A + kNgThreads - 1) / kNgThreads);
    return R4D_EMIT(sk, ngram_ids_kernel, dim3(tiles, (uint32_t)N), dim3(kNgThreads), 0,
                       (const int32_t*)r4d_addr(tok), (int32_t*)r4d_addr(st),
                       (const int32_t*)r4d_addr(cu), (const long long*)r4d_addr(mult),
                       (const long long*)r4d_addr(vsz), (const long long*)r4d_addr(off),
                       cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,
                       hini ? (const int32_t*)r4d_addr(hini) : nullptr,
                       nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,
                       (int32_t*)r4d_addr(ids), (uint32_t)heads, (uint32_t)ngram,
                       (uint32_t)per, (uint32_t)W, eos, (uint32_t)ids_ld,
                       A > 0 ? (int32_t*)r4d_addr(ahd) : nullptr, (uint32_t)A,
                       (uint32_t)ahead_ld);
}

/* ABI v10: the launch and its description, from the body above. */
R4D_ENTRY(r4d_rad_ngram_ids, ngram_ids_body)
