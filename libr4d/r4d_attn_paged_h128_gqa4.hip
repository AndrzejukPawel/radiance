// R4D attention dispatch for the DFlash2 drafter's geometry: head_dim 128, 4 queries per KV head,
// paged block 16, and a SLIDING WINDOW. It is the same device code as the 27B target's h256/gqa6
// dispatch -- the kernels are templates on head_dim and gqa, so this file is a second set of
// instantiations and not a second kernel -- with the window bit turned on.
//
// Decode only, deliberately. A query long enough to want the query-tiled prefill kernel is rare on
// a drafter (it happens on a prefill step and nowhere else), and the caller has a working fallback
// for it; instantiating a second prefill kernel to serve that case would cost build time and a
// second masking path to keep correct for no measurable return. The caller asks
// r4d_attn_h128_gqa4_have_prefill() and routes the whole batch elsewhere when it is 0.
//
// WHY THIS EXISTS. A general unified-attention kernel (vLLM's) spends 30.9 us a layer on this
// shape at one request and 203.7 at eight -- against a KV stream that the window caps at 2048
// positions, i.e. 2.1 MB and 16.8 MB, which is 3.3 us and 26.3 us of DRAM. The gap is grid shape,
// not bandwidth: it launches (q_blocks, kv_heads,
// segments) with the query in the GRID and pads q_blocks for capture, so the great majority of its
// workgroups start and immediately return. R4D carries the query INSIDE the workgroup (up to 64
// rows of q_len * gqa), so its grid is (splits, kv_heads, num_seqs) and has nothing dead in it.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel shares its query conversion and fragment helpers with the prefill one,
// so the prefill header comes first here for the same reason it does in the h256 dispatch.
// Nothing in this translation unit instantiates r4d_attn_prefill_kernel.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   128
#define B_GQA          4
#define B_BLOCK_SIZE  16
#define B_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold
#define B_TILE 16
// The same option set the h256 decode kernel ships, K prefetch of 2 included. This geometry does
// not need it -- head_dim 128 halves both the accumulator and the query, so it sits at 162-179
// VGPRs and does not spill at 4 -- but 2 is still 0.3-1.3% faster at every concurrency and window,
// for bit-identical output, so the two legs stay on one number.
#define B_OPT  3429947
#ifndef B_JPB
#define B_JPB 2
#endif
#ifndef B_VECMIN
#define B_VECMIN 256
#endif

extern "C" int r4d_attn_h128_gqa4_have_prefill(void) { return 0; }

// Split law. It aims at 192 workgroups, snaps DOWN to a power of two but never below 16, clamps to
// at least two tiles a segment, and then steps UP off the exact grid size `splits * units == 256`,
// which is a pothole in this geometry's timing.
//
// The one thing it does that the h256 law does not: under a sliding window the context the kernel
// actually walks is the window, not the sequence, so the tile count that feeds the machine-fill
// target and the thin-segment clamp comes from the window. Without that the law reads a 16K
// sequence and asks for segments that are two tiles of a 129-tile walk.
static int decode_splits_h128(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int span  = (a->window > 0 && a->window + a->q_len < a->max_ctx)
                    ? a->window + a->q_len : a->max_ctx;
    const int tiles = (span + B_TILE - 1) / B_TILE;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    if (s * units == 256 && s * 2 <= tiles) s <<= 1;   // step off the pothole, see above
    return s;
}

extern "C" long r4d_attn_decode_h128_gqa4_scratch_bytes(const R4DArgs* a) {
    const int  splits = decode_splits_h128(a);
    const long rows   = (long)r4d_q_total(*a) * a->q_heads;
    return (B_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)
                        : rows * splits * (long)(a->head_dim + 2) * 4L;
}

template <int KVP>
static int decode_launch_h128(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    const int rows = a->q_len * B_GQA;
    if (rows > B_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = decode_splits_h128(a);

#define B_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, B_TILE, B_HEAD_DIM, B_GQA, B_BLOCK_SIZE,    \
                                               KVP, B_OPT, 1>),                               \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) B_LAUNCH(1);
    else if (nw == 2) B_LAUNCH(2);
    else if (nw == 3) B_LAUNCH(3);
    else              B_LAUNCH(4);
#undef B_LAUNCH
    CHK4();

    // Same two merge forms, and the same rule for choosing between them, as the h256 dispatch:
    // a wave a row while there are enough rows to fill the part and the segment count fits a
    // wave, the general LDS-tree form otherwise. Both produce the same bytes.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= B_VECMIN) {
        const int blocks = (jobs3 + B_JPB - 1) / B_JPB;
        if (B_OPT & 32)
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<B_HEAD_DIM, 4, 1, B_JPB, 1>),
                               dim3(blocks), dim3(32 * B_JPB), 0, s, *a, splits, B_TILE);
        else
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<B_HEAD_DIM, 4, 0, B_JPB, 1>),
                               dim3(blocks), dim3(32 * B_JPB), 0, s, *a, splits, B_TILE);
    } else if (B_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<B_HEAD_DIM, 4, 1, 1>), dim3(jobs3),
                           dim3(B_HEAD_DIM), splits * sizeof(float), s, *a, splits, B_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<B_HEAD_DIM, 4, 0, 1>), dim3(jobs3),
                           dim3(B_HEAD_DIM), splits * sizeof(float), s, *a, splits, B_TILE);
    CHK4();
    return 0;
}

extern "C" int r4d_attn_decode_h128_gqa4_fp8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch_h128<0>(a, s);
}
extern "C" int r4d_attn_decode_h128_gqa4_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch_h128<1>(a, s);
}
