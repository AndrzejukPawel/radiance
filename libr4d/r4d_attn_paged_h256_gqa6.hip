// R4D attention dispatch. One translation unit so the KV-dtype variants share the device code:
// each is a template instantiation, not a copy. The entry points are named for the geometry they
// are compiled for (head_dim 256, 6 queries per KV head, paged block 16) because that geometry is
// a hard gate -- everything else is rejected below rather than run.
//
// The query is bf16 in every variant. A single-term 8-bit query is an order of magnitude worse
// against an fp32 reference (2.58e-02 relative against ~2.3e-03), and the two-term e4m3 form that
// DOES reach f16 accuracy is 4-10% slower end to end, so no 8-bit query variant is compiled. Two
// families ship, differing only in KV cache dtype: bf16 KV (all 16-bit) and fp8 KV (16/8 mixed,
// the production configuration).
//
// `opt` is a bitmask of the optimisation steps, all of them on:
//     prefill 3955211 = f16 operands | mask skip | folded query scale | contiguous-k fragments
//                     | prefetch 4 + scheduling barrier | dot2 row sum | block table in SGPRs
//                     | wide K fetch | V prefetch across the m-tile loop | LDS-scoped barrier
//     decode  3429947 = the same, minus the wide K fetch (its K never reaches LDS), with the K
//                       prefetch at 2 rather than 4, plus reading K straight from the paged cache
//                       (bit 16) and f16 split-KV partials (32).
// Each bit is defined and justified where it is used, in r4d_attn_prefill_h256_gqa6.hip and
// r4d_attn_decode_h256_gqa6.hip.
// They are kept separable so any one of them can be turned off on its own after a geometry change.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

// The geometry every kernel in this translation unit is compiled for. HEAD_DIM and GQA appear in
// the entry-point names; BLOCK_SIZE does not, because it is 16 for the whole library and so
// discriminates nothing between them.
#define A_HEAD_DIM   256
#define A_GQA          6
#define A_BLOCK_SIZE  16
#define A_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold

extern "C" void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows) {
    if (head_dim)        *head_dim        = A_HEAD_DIM;
    if (gqa)             *gqa             = A_GQA;
    if (block_size)      *block_size      = A_BLOCK_SIZE;
    if (max_decode_rows) *max_decode_rows = A_MAX_DECODE_ROWS;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS, head_dim 256, 16-bit K and V: sK = TILE*264*2, sV = 256*(TILE+8)*2. bf16 KV at TILE 64
// would need 68096 B > 65536, so the geometry stops at 48.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211

// KVP: 0 = fp8 e4m3 cache, 1 = bf16 cache. Reached only through the two named wrappers below.
template <int KVP>
static int prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / A_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,
                                                KVP, P_OPT>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<1>(a, s);
}

// ---- decode ----------------------------------------------------------------------------------
#ifndef D_TILE
#define D_TILE 16
#endif
// The decode leg prefetches TWO K fragments per WMMA group, not four.  Four costs 8 more VGPRs at
// the peak of the QK loop and that is exactly the amount this kernel is over: it allocates 256
// VGPRs, spills 18 of them, and reloads three 16-byte groups of the query FROM SCRATCH ON EVERY
// TILE.  Two fits in 255 with no spill and is 1.7-7.5% faster across concurrency and depth, for
// bit-identical output.
#ifndef D_OPT
#define D_OPT  3429947
#endif
// Output rows per block in the vectorised merge: 32 threads own a row, so this is the block width
// in waves.  1, 2 and 4 are within 1% of each other everywhere the vectorised form is taken; 2 is
// the best of them on average across concurrency and depth.
#ifndef D_JPB
#define D_JPB 2
#endif
// Output rows below which the vectorised merge is NOT taken.  It gives a row 32 lanes where the
// general form gives it 256, so it trades parallelism for eight-times-fewer load instructions --
// and that trade is only good while there are enough rows to fill the part.  At one request there
// are 96 rows and the segment count is at its largest (the split law scales splits with 1/num_seqs),
// so each of 96 waves walks 64 segments on an idle machine and the vectorised form LOSES up to
// 23%.  The crossover is around 192 rows, and above 288 the vectorised form wins by 1.2-20%; the
// cut is set at four rows per CU.
#ifndef D_VECMIN
#define D_VECMIN 256
#endif

// Split law: AS MANY WORKGROUPS AS FIT UNDER 128.  `splits * kv_heads * num_seqs` is the whole
// grid, and 128 of them is the argmin at every concurrency this serve reaches on this geometry.
//
// THE COUNT IS `128 / units`, NOT THE LARGEST POWER OF TWO UNDER IT.  A power of two lands exactly
// on 128 only when units is one too -- one, two, four and eight requests -- and at three, five,
// six and seven it lands on 96, 80, 96 and 112 workgroups instead, which is a third of the part
// idle in the last round.  `floor(128 / units)` is the argmin at all eight, worth 2-20% over a
// power-of-two law at the four counts where they differ.
//
// Nearby counts are not interchangeable: at five requests 13 splits is 130 workgroups, three
// rounds of 64 for two rounds of work, and is slower than 12.  UNDER a multiple of 64 is what
// matters, not near it.
//
// What the count buys is partial-buffer traffic, not parallelism.  The split-KV scratch is
// `seqs x q_len x q_heads x splits` rows of head_dim f16, written once by the decode kernel and
// read back once by the combine: at eight requests that is 12.8 MB at splits 32 against 3.2 at 8,
// which is 19% of a 16K step's KV and 1.2% of a 128K one.  So the win shrinks with depth -- a few
// percent up to 16K, under 2% at 64K and a wash at 128K -- and never reverses enough to be worth a
// context term.
static int decode_splits(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + D_TILE - 1) / D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    int s = 128 / units;                     // as many workgroups as fit under 128
    const int cap = tiles / 2;               // >= 2 tiles per segment
    if (s > cap) s = cap;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

// Same for both cache dtypes: the split law does not read the dtype, and the partials are f16
// either way (opt bit 32).
extern "C" long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a) {
    const int  splits = decode_splits(a);
    const long rows   = (long)r4d_q_total(*a) * a->q_heads;
    return (D_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)       // f16 O/l + m + l
                        : rows * splits * (long)(a->head_dim + 2) * 4L;     // f32 O partials + m + l
}

template <int KVP>
static int decode_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    const int rows = a->q_len * A_GQA;
    if (rows > A_MAX_DECODE_ROWS) return -4;    // verify width <= 10 at gqa 6
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = decode_splits(a);

#define D_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, D_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,    \
                                               KVP, D_OPT>),                                  \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    // The vectorised merge owns a row per WAVE, so its wave-wide sum only reproduces the general
    // path's LDS tree while the segment count fits 64 lanes-worth. Past that -- reachable only
    // when kv_heads * num_seqs is 1, where the law asks for 128 splits -- fall back rather than
    // change the merge order. Both forms produce the same bytes; which one runs is a speed choice
    // only.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= D_VECMIN) {
        const int blocks = (jobs3 + D_JPB - 1) / D_JPB;
        if (D_OPT & 32)
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<A_HEAD_DIM, 4, 1, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
        else
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<A_HEAD_DIM, 4, 0, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
    } else if (D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 1>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 0>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_decode_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<1>(a, s);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader is r4d_attn_args.h, a header this kernel family shares because the paged
// geometries implement one op and take one operand list. See libr4d/r4d_args.h for why there is no
// adapter directory between the core and these launches.
//
// THEY GO IN THE TRANSLATION UNIT AND NOT IN THE FILES IT #includes. r4d_attn_prefill_*.hip and
// r4d_attn_decode_*.hip are textual inclusions of this one -- appending an entry point to either
// compiles it into nothing, and the .so then carries an UNDEFINED symbol that no build step
// reports. The plugin still links; only r4d_selftest catches the missing symbol.
/* ------------------------------------------------------------------ paged prefill */

extern "C" int r4d_rad_attn_prefill_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    /* ABSENT DESCALES MEAN 1.0, they are not an error. The kernels already spell it that way --
     * `a.k_descale ? a.k_descale[seq * kv_heads + kvh] : 1.0f` -- and 1.0 is what vLLM defaults
     * both scales to, so a cache written by kv_store_fp8 at its default scale is read back by
     * exactly this path with no descale plane to allocate, fill or keep in step. Refusing here
     * would make the operands mandatory in the shim while they stay optional in the schema and
     * optional in the kernel, which is three answers to one question. A CALIBRATED pair still
     * works: pass the operands and they are used. */
    return r4d_attn_prefill_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_prefill_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

/* ------------------------------------------------------------------ paged decode */

/* The split-KV partial buffer. The decode kernel writes it and the combine reads it back, and
 * nothing else touches it -- so it is arena scratch, sized at declare from the band's upper bound
 * exactly as RadScratchFn intends.
 *
 * IT IS CALLED WITH PARAMETERS AND NO TENSORS. That is what declare is: the op is known, the batch
 * is not, and there is nothing to point at yet. So the parameters are the primary source here and
 * the tensors are the fallback, not the other way round -- reading tensors first makes this hook
 * refuse every attn_paged row in the model with "invalid argument", which presents as a pile of
 * missing kernels rather than as one hook asking the wrong question.
 *
 * q_len is the band's upper bound, deliberately: this region is sized once for the worst step the
 * arena will ever hold, so the bound is the right number and a per-step count would be the wrong
 * one.
 *
 * THE SEQUENCE COUNT IS EVERY COUNT UP TO max_seqs, NOT max_seqs ALONE. Both split laws hand a
 * step of S sequences a split count that falls as S grows -- about 128 / (kv_heads * S) -- so the
 * partials are S * splits(S) rows a (query position, head), and that product is not monotone in
 * S: at kv_heads 2 the h256 law gives 60 at twelve sequences and 63 at nine, and the h128 law's
 * step off its pothole doubles the splits at one count and not at the next. A step may carry any
 * count up to the bound, so the region is the largest over all of them. The laws are closed forms
 * and this runs at declare, so the sweep costs nothing a step. */
static int64_t decode_scratch(const RadArgs* a, int head_dim, int which) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim  = head_dim;
        r.q_heads   = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads  = (int)(gqa > 0 ? n_head / gqa : n_head);
        r.num_seqs  = (int)max_seqs;
        r.q_len     = (int)r4d_geti_or(a, "q_len", 1);
        r.max_ctx   = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits    = sp;
        /* max_ctx feeds the split law only. Zero would make it choose one split, which UNDERSIZES
         * the buffer for the launch that later picks more, so refuse rather than guess low. */
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n) {
            r.num_seqs = (int)n;
            const int64_t need = which == 0 ? r4d_attn_decode_h256_gqa6_scratch_bytes(&r)
                                            : r4d_attn_decode_h128_gqa4_scratch_bytes(&r);
            if (need > worst) worst = need;
        }
        return worst;
    }

    int rc = fill_paged(a, &r, head_dim);
    if (rc != RAD_OK) return rc;
    r.splits = sp;
    return which == 0 ? r4d_attn_decode_h256_gqa6_scratch_bytes(&r)
                      : r4d_attn_decode_h128_gqa4_scratch_bytes(&r);
}

extern "C" int64_t r4d_rad_attn_decode_h256_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0);
}

extern "C" int64_t r4d_rad_attn_decode_h128_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1);
}

static int decode_common(const RadArgs* a, R4DArgs* r, int head_dim, int which) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    int rc = fill_paged(a, r, head_dim);
    if (rc != RAD_OK) return rc;
    r->splits = sp;
    const int64_t need = (which == 0) ? r4d_attn_decode_h256_gqa6_scratch_bytes(r)
                                      : r4d_attn_decode_h128_gqa4_scratch_bytes(r);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

extern "C" int r4d_rad_attn_decode_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h128_gqa4_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_bf16kv(&r, (hipStream_t)s);
}
