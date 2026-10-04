// r4d_gdn_conv_w4_h128_bf16.hip -- the gated-delta-net depthwise causal convolution and the work fused around
// it, one entry point per serving phase.
//
// PREFILL (r4d_gdn_conv_prep): everything the layer does between the qkv projection and the
// chunked scan, in one kernel:
//
//     conv(x)        depthwise causal convolution, width 4, silu, varlen, with the conv-state
//                    cache read at a sequence's first chunk and rewritten from its last tokens
//     split          the packed [q | k | v] channel range into three contiguous tensors
//     l2norm         over each 128-dim q and k head
//     gate           g = -exp(A_log) . softplus(a + dt_bias), beta = sigmoid(b)
//     cumsum         g summed along the chunk, which is what the scan actually consumes
//
// It replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum. The pair it replaces
// spends most of its time moving the conv output: causal_conv1d_fn writes all conv_dim channels of
// every token to HBM and fused_post_conv_prep reads every one of them straight back, which at this
// model's shape is 34 MB out and 34 MB in per layer per rank for a tensor that is dead two
// microseconds later. Here the conv output only ever exists in registers.
//
// The layout falls out of the l2norm. A workgroup is HPB waves, one wave per 128-dim head, one
// lane per four dims -- so the norm of a head is a wave reduction with no LDS and no barrier, and
// four contiguous bf16 per lane is one 8-byte access on both sides. Channel groups are counted in
// heads rather than in channels so that a group never straddles the q/k/v boundaries (they fall on
// head multiples at every tensor-parallel size).
//
// Tokens are blocked by the chunk, 64, because the cumsum is per chunk: the whole scan is then a
// wave-level prefix over 32 lanes holding two tokens each, and no workgroup needs another's
// partial sums. The conv reads three tokens of halo per block, 4.7% of the traffic, which is
// cheaper than the barrier a shared staging buffer would need.
//
// The conv output is rounded to bf16 BEFORE the norm and the store, which is what the two kernels
// it replaces do (the conv writes a bf16 tensor and the prep reads it back). Keeping the fp32
// accumulator through the norm would be slightly finer and slightly different; matching is worth
// more than the last half ULP here, because it makes this kernel a bit-for-bit drop-in for the
// pair it replaces.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"
#include "r4d_gdn_conv_tap.h"

#define CP_HD    128            // dims per head, q k and v alike
#define CP_BT     64            // token block = the scan's chunk, so the cumsum stays wave-local
#define CP_W       R4D_GDN_CONV_W   // conv width
#define CP_ST   (CP_W - 1)      // conv state depth
// CU_MAXT: tokens the DECODE update pre-loads into registers. 16 is about 13% slower than 8: the
// sixteen ushort4 cost more in register pressure and predicated-load exec-mask work than the round
// trips they hide, and the compiler already batches the loop's loads.
// 1 disables it (slen is num_spec+1 > 1 on every speculative step).
// CU_STATIC=1: index the xb[] preload with a COMPILE-TIME t.  The preload exists so a token tap is
// not an HBM round trip, but a register array under a RUNTIME index is not a register read: the
// compiler builds a select tree for it.  With both readers -- the token loop and the cache rewrite
// -- indexing at runtime, the HPB=1 ISA census is 2033 instructions for 41 loads and 29 stores,
// 193 s_cselect_b32 + 138 s_cmp_eq_u32 + 132 v_cndmask_b16 of it: about a third of the kernel
// choosing which register to read.  That is also why a larger CU_MAXT loses -- twice the array is
// twice the tree, not twice the reuse.  Unrolling both readers over CU_MAXT with a wave-uniform
// `t < slen` guard turns every access into a static index; `slen` is uniform, so the guard is a
// scalar branch, not a per-lane select.
#ifndef CU_STATIC
#define CU_STATIC 1
#endif
// CU_ISSUE=1: put the input preload AHEAD of the history load.  `xrow` depends only on the token
// range and this lane's channels, while `hp` depends on `slot`, which has to come back from
// cache_idx first -- and the history block consumes its loads immediately (bf2f), so the compiler
// waits on them before it issues a single input load and the two HBM round trips run back to back.
// One wave per SIMD means nothing else hides either of them.
#ifndef CU_ISSUE
#define CU_ISSUE 1
#endif
#ifndef CU_MAXT
#define CU_MAXT 8
#endif
// WVEC: read a channel's CP_W conv weights as one vector load instead of CP_W scalar ones.
#ifndef WVEC
#define WVEC 1
#endif
#define CP_DPL     4            // dims per lane (CP_HD / 32)

// cp_silu, and the reason it is a reciprocal and not a divide, is r4d_gdn_conv_tap.h: the
// recurrent update folds the decode convolution in and has to compute the same bits.

// softplus, in the two-sided form FLA uses, with the same large-x cutoff
__device__ __forceinline__ float cp_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x)))
                    : __logf(1.0f + __expf(x));
}

// a and b arrive as the layer holds them: a chunk() of the ba projection, so bf16 and strided by
// 2*H, not the fp32 contiguous pair FLA's own kernels are usually handed. Reading them where they
// lie is what keeps this a single kernel -- a .float().contiguous() per tensor per layer would put
// two more launches and 0.6 MB back on the path this exists to shorten.
template <int ABF16>
__device__ __forceinline__ float cp_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// `has_init` SAYS WHETHER THE SLOT HOLDS A HISTORY, and its width is the caller's -- the flag is
// the caller's answer, like ab_is_bf16 above. Guessing the width is not safe. The engine hands
// over RadBatch::ctx_lens, which is int32 like every derived buffer the core computes; reading
// that as `unsigned char` takes byte n of an int32 array. At n = 0 with a cold sequence that is 0
// by luck and correct; at the SECOND chunk of a long prompt, ctx_len 512 has a zero low byte, so
// the conv window would silently start from zeros and the sequence would lose its history. libref
// reads the operand through its own dtype and is right either way, so a width mismatch shows up
// only as the two implementations disagreeing, with the fast path wrong.
__device__ __forceinline__ bool r4d_has_init(const void* p, int i32, int n) {
  if (!p) return false;
  return i32 ? ((const int*)p)[n] != 0 : ((const unsigned char*)p)[n] != 0;
}

template <int HPB, int ABF16>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_prep_kernel(const unsigned short* __restrict__ x, long xpitch,
                              const unsigned short* __restrict__ wgt,
                              const unsigned short* __restrict__ bias,
                              unsigned short* __restrict__ cstate,
                              long cs_seq, long cs_dim, long cs_tok,
                              const int* __restrict__ cache_idx, long ci_stride,
                              const void* __restrict__ has_init, int hi_i32,
                              const void* __restrict__ av, const void* __restrict__ bv,
                              long ab_stride,
                              const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                              unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                              unsigned short* __restrict__ v,
                              float* __restrict__ go, float* __restrict__ beta,
                              const int* __restrict__ cu, int N, int H, int Hg, float sp_thr)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // flat chunk id -> (sequence, chunk); see r4d_gdn_kkt_solve_k128_c64_bf16.hip for why the grid is an upper
  // bound rather than the exact count
  int n = -1, t0 = 0, bos = 0, rows = 0, slen = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + CP_BT - 1) / CP_BT;
      if (want < acc + nc) { n = i; bos = b; slen = e - b; t0 = (want - acc) * CP_BT;
                             rows = min(CP_BT, slen - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }
  const bool first = (t0 == 0);

  // ---- the gating half, on one workgroup per chunk ----------------------------------------
  // g and beta come from a and b alone, so this is independent of the convolution; it rides on
  // the y == 0 workgroup rather than costing a launch of its own.
  if (blockIdx.y == 0) {
    for (int hv = warp; hv < H; hv += HPB) {
      const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];
      const int i0 = 2 * lane, i1 = i0 + 1;
      float g0 = 0.0f, g1 = 0.0f;
      if (i0 < rows)
        g0 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i0) * ab_stride + hv)
                                 + dtb, sp_thr);
      if (i1 < rows)
        g1 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i1) * ab_stride + hv)
                                 + dtb, sp_thr);
      // inclusive wave scan of the per-lane pair sums, made exclusive by subtracting the pair
      float p = g0 + g1;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float o = __shfl_up(p, d, 32);
        if (lane >= d) p += o;
      }
      const float pre = p - (g0 + g1);            // sum of all lanes before this one
      if (i0 < rows) {
        go[(size_t)(bos + t0 + i0) * H + hv] = pre + g0;
        beta[(size_t)(bos + t0 + i0) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i0) * ab_stride + hv)));
      }
      if (i1 < rows) {
        go[(size_t)(bos + t0 + i1) * H + hv] = pre + g0 + g1;
        beta[(size_t)(bos + t0 + i1) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i1) * ab_stride + hv)));
      }
    }
  }

  // ---- the convolution half ----------------------------------------------------------------
  const int unit = blockIdx.y * HPB + warp;       // global 128-dim head unit
  const int d0 = unit * CP_HD + lane * CP_DPL;    // this lane's first channel
  const unsigned short* xp = x + (size_t)(bos + t0) * xpitch + d0;

  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
    for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(wgt[(size_t)(d0 + c) * CP_W + i]);
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  // window[i] holds x[t - (W-1) + i]; seed it from the previous tokens, or from the conv state
  // cache when this is the first chunk of the sequence
  float win[CP_W][CP_DPL];
  {
    const bool init = first && r4d_has_init(has_init, hi_i32, n);
    const unsigned short* sp = init ? (cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                                       + (size_t)d0 * cs_dim) : nullptr;
#pragma unroll
    for (int i = 0; i < CP_ST; ++i) {              // i = 0,1,2  ->  x[t0-3+i]
      const int src = t0 - CP_ST + i;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) {
        float val = 0.0f;
        if (src >= 0) val = bf2f(x[(size_t)(bos + src) * xpitch + d0 + c]);
        else if (init) val = bf2f(sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok]);
        win[i][c] = val;
      }
    }
  }

  // ---- write the new conv state, from x, at the sequence's first chunk ---------------------
  // Ordering: the cache line is updated in place, so the read above and this write must be in the
  // same workgroup for the same channels -- which they are, and this is why the FIRST chunk owns
  // the write even though the tokens it stores belong to the last one.
  if (first && cache_idx) {
    unsigned short* sp = cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                       + (size_t)d0 * cs_dim;
    const bool init = r4d_has_init(has_init, hi_i32, n);
    // Channel-major, because the cache is [dim][slot] with the slot contiguous: this lane owns
    // CP_DPL consecutive channels, so its CP_DPL*CP_ST shorts are one contiguous run and the wave
    // covers 32 of those back to back. A slot-major order would make every store a strided 2-byte
    // scalar.
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
      for (int i = 0; i < CP_ST; ++i) {
        const int src = slen - CP_ST + i;
        unsigned short val = 0;
        if (src >= 0) val = x[(size_t)(bos + src) * xpitch + d0 + c];
        else if (init) val = sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok];
        sp[(size_t)c * cs_dim + (size_t)i * cs_tok] = val;
      }
    }
  }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  for (int t = 0; t < rows; ++t) {
    float o[CP_DPL];
    {
      const unsigned short* xr = xp + (size_t)t * xpitch;
      const ushort4 raw = *(const ushort4*)xr;
      win[CP_W - 1][0] = bf2f(raw.x); win[CP_W - 1][1] = bf2f(raw.y);
      win[CP_W - 1][2] = bf2f(raw.z); win[CP_W - 1][3] = bf2f(raw.w);
    }
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
      float acc = bs[c];
#pragma unroll
      for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];
      // the pair this replaces rounds here, on the way to HBM; keep that
      o[c] = bf2f(f2bf(cp_silu(acc)));
    }

    if (region < 2) {
      float ss = 0.0f;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) ss += o[c] * o[c];
      ss += r4d_xor_lane<16>(ss); ss += r4d_xor_lane<8>(ss); ss += r4d_xor_lane<4>(ss);
      ss += r4d_xor_lane<2>(ss);  ss += r4d_xor_lane<1>(ss);
      const float inv = 1.0f / sqrtf(ss + 1e-6f);   // L2NORM_EPS, as in the kernel it replaces
      unsigned short* dst = (region ? k : q)
                          + ((size_t)(bos + t0 + t) * Hg + (region ? hk : hq)) * CP_HD
                          + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0] * inv), f2bf(o[1] * inv),
                                    f2bf(o[2] * inv), f2bf(o[3] * inv));
    } else {
      unsigned short* dst = v + ((size_t)(bos + t0 + t) * H + hv) * CP_HD + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0]), f2bf(o[1]), f2bf(o[2]), f2bf(o[3]));
    }

#pragma unroll
    for (int i = 0; i < CP_W - 1; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];
  }
}

extern "C" int r4d_gdn_conv_prep_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
    const void* has_init, int hi_i32,
    const void* a, const void* b, long ab_stride, int ab_is_bf16, const void* A_log,
    const void* dt_bias, void* q, void* k, void* v, void* g, void* beta, const void* cu,
    int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  const int nchunk_ub = (T + CP_BT - 1) / CP_BT + N;
  dim3 grid(nchunk_ub, units / hpb), blk(hpb * 32);
#define CP_LAUNCH(P, AB)                                                                      \
  hipLaunchKernelGGL((r4d_gdn_conv_prep_kernel<P, AB>), grid, blk, 0, (hipStream_t)stream,     \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,            \
                     (const unsigned short*)bias, (unsigned short*)cstate,                    \
                     cs_seq, cs_dim, cs_tok, (const int*)cache_idx, ci_stride,                \
                     has_init, hi_i32, a, b, ab_stride,                                     \
                     (const float*)A_log, (const float*)dt_bias, (unsigned short*)q,          \
                     (unsigned short*)k, (unsigned short*)v, (float*)g, (float*)beta,         \
                     (const int*)cu, N, H, Hg, softplus_thr)
#define CP_SWITCH(AB)                                                                         \
  switch (hpb) {                                                                              \
    case 8: CP_LAUNCH(8, AB); break;                                                          \
    case 4: CP_LAUNCH(4, AB); break;                                                          \
    case 2: CP_LAUNCH(2, AB); break;                                                          \
    default: CP_LAUNCH(1, AB); break;                                                         \
  }
  if (ab_is_bf16) { CP_SWITCH(1); } else { CP_SWITCH(0); }
#undef CP_SWITCH
#undef CP_LAUNCH
  return (int)hipGetLastError();
}

// ---------------------------------------------------------------------------------------------
// The decode-time convolution. Same depthwise causal filter, but the tokens of a step are a
// speculative window rather than a chunk, and the state cache is a rolling buffer of
// width-1 + num_spec entries rather than width-1: a step reads its window starting at the slot
// the last ACCEPTED token left behind, and rewrites the buffer shifted by one so the next step
// can do the same. That offset is the only thing the caller has to get right, and it is why this
// is a separate entry point rather than a flag on the one above.
//
// It writes q, k and v in their own layouts, which also removes the torch.cat the reference layer
// needs to make the three views contiguous for the recurrent kernel.
// ---------------------------------------------------------------------------------------------
template <int HPB>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_update_kernel(const unsigned short* __restrict__ x, long xpitch,
                                const unsigned short* __restrict__ wgt,
                                const unsigned short* __restrict__ bias,
                                unsigned short* __restrict__ cstate,
                                long cs_seq, long cs_dim, long cs_tok, int state_len_max,
                                const int* __restrict__ cache_idx, long ci_stride,
                                const int* __restrict__ naccept,
                                unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                                unsigned short* __restrict__ v,
                                const int* __restrict__ cu, int H, int Hg, int maxq)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.x;
  const int bos = cu[n], slen = cu[n + 1] - bos;
  if (slen <= 0) return;
  const int slot = cache_idx[(size_t)n * ci_stride];
  // A NEGATIVE slot is the pad and ZERO IS A REAL SLOT. `slot <= 0` would be vLLM's NULL_BLOCK_ID
  // convention, where 0 is reserved; radiance's block manager hands out 0-based state slots, so
  // that test would skip the FIRST sequence on the card entirely -- no convolution at all, on
  // every decode step and every linear-attention layer, silently. libref skips only a negative
  // slot (libref/ref_gdn.cpp) and so does every index in libr4d.
  if (slot < 0) return;

  const int off = naccept ? (naccept[n] - 1) : 0;
  // the cache depth this step actually uses; FLA shrinks the caller's max_query_len bound by the
  // tokens this sequence did not bring
  const int slen_eff = state_len_max - (maxq - slen);

  const int unit = blockIdx.y * HPB + warp;
  const int d0 = unit * CP_HD + lane * CP_DPL;

  // The weight of one channel is CP_W contiguous shorts, so a lane's four channels are four
  // vector loads, not sixteen scalar ones.
  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
    if (WVEC && CP_W == 4) {
      const ushort4 w4 = *(const ushort4*)(wgt + (size_t)(d0 + c) * CP_W);
      const unsigned short* ws = (const unsigned short*)&w4;
#pragma unroll
      for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(ws[i]);
    } else {
#pragma unroll
      for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(wgt[(size_t)(d0 + c) * CP_W + i]);
    }
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  // ADDRESS ARITHMETIC IS WHAT THIS KERNEL SPENDS ITS TIME ON.  Indexing with (size_t)i * cs_tok
  // and (size_t)(bos + ti) * xpitch gives every one of the ~40 cache stores and every input tap a
  // 64-bit multiply: that form costs 1297 instructions -- 252 of them s_wait_alu and about 330
  // scalar address ALU -- to move 290 KB.  The strides are small (cs_dim is the per-channel row, a
  // handful of tokens; xpitch is conv_dim), so every offset below is an int taken from a base
  // pointer that already carries the 64-bit part, and the loops walk their pointers rather than
  // re-deriving them.
  const int csd = (int)cs_dim, cst = (int)cs_tok, xpi = (int)xpitch;
  unsigned short* const sp = cstate + (size_t)slot * cs_seq + (size_t)d0 * cs_dim;
  const unsigned short* const xrow = x + (size_t)bos * xpitch + d0;
#if CU_ISSUE
  ushort4 xb[CU_MAXT];
  const bool xpre = (slen <= CU_MAXT);
#pragma unroll
  for (int t = 0; t < CU_MAXT; ++t)
    if (xpre && t < slen) xb[t] = *(const ushort4*)(xrow + t * xpi);
#endif
  float win[CP_W][CP_DPL];
  unsigned short hist[CP_ST][CP_DPL];             // the raw history, for the shifted rewrite
  {
    const unsigned short* hp = sp + off * cst;
#pragma unroll
    for (int i = 0; i < CP_ST; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) {
        const unsigned short raw = hp[c * csd + i * cst];
        hist[i][c] = raw;
        win[i][c] = bf2f(raw);
      }
  }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  // This step's inputs, all issued before the first is waited on. With the load inside the loop
  // each of the (num_spec + 1) four-tap taps pays an HBM round trip that nothing hides: a decode
  // step launches one wave per channel group, so there is no other wave on the SIMD to run
  // meanwhile. They are also exactly the values the cache rewrite at the bottom re-reads.
#if !CU_ISSUE
  ushort4 xb[CU_MAXT];
  const bool xpre = (slen <= CU_MAXT);
#pragma unroll
  for (int t = 0; t < CU_MAXT; ++t)
    if (xpre && t < slen) xb[t] = *(const ushort4*)(xrow + t * xpi);
#endif

  // The destination of a token is the previous one plus a fixed stride, so it is walked rather
  // than rebuilt: recomputing ((bos + t) * H + hv) * CP_HD per token is a 64-bit multiply each
  // time.
  unsigned short* op = (region == 2)
      ? (v + ((size_t)bos * H + hv) * CP_HD + lane * CP_DPL)
      : ((region ? k : q) + ((size_t)bos * Hg + (region ? hk : hq)) * CP_HD + lane * CP_DPL);
  const int ostep = (region == 2) ? H * CP_HD : Hg * CP_HD;

#define CU_TOKEN_BODY(RAW)                                                       \
    {                                                                            \
      const ushort4 raw_ = (RAW);                                                \
      win[CP_W - 1][0] = bf2f(raw_.x); win[CP_W - 1][1] = bf2f(raw_.y);           \
      win[CP_W - 1][2] = bf2f(raw_.z); win[CP_W - 1][3] = bf2f(raw_.w);           \
      unsigned short out[CP_DPL];                                                \
      _Pragma("unroll")                                                          \
      for (int c = 0; c < CP_DPL; ++c) {                                         \
        float acc = bs[c];                                                       \
        _Pragma("unroll")                                                        \
        for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];              \
        out[c] = f2bf(cp_silu(acc));                                             \
      }                                                                          \
      *(ushort4*)op = make_ushort4(out[0], out[1], out[2], out[3]);               \
      op += ostep;                                                               \
      _Pragma("unroll")                                                          \
      for (int i = 0; i < CP_W - 1; ++i)                                         \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];              \
    }
#if CU_STATIC
  if (xpre) {
    // A GUARD, NOT A BREAK.  `#pragma unroll` needs a constant trip count; a data-dependent exit
    // leaves the loop rolled and `t` a runtime variable again, which is the whole thing being fixed.
#pragma unroll
    for (int t = 0; t < CU_MAXT; ++t) {
      if (t < slen) CU_TOKEN_BODY(xb[t])          // static index: a register read, not a tree
    }
  } else {
    for (int t = 0; t < slen; ++t) CU_TOKEN_BODY(*(const ushort4*)(xrow + t * xpi))
  }
#else
  for (int t = 0; t < slen; ++t)
    CU_TOKEN_BODY(xpre ? xb[t] : *(const ushort4*)(xrow + t * xpi))
#endif

  // Rewrite the cache shifted by one: the tail of the old history, then this step's tokens. The
  // first CP_ST-1 entries are exactly the history already in registers, so the rewrite costs no
  // extra reads -- and reading before writing is what makes the in-place update safe.
  // Channel-major for the same reason as the prefill path, and it matters more here: this is
  // width-1 + num_spec slots per channel, so the slot-major order made it 44 strided 2-byte
  // stores per lane where channel-major is one 88-byte run.
  // The two halves of this run have different sources -- the tail of the old history, then this
  // step's tokens -- and testing for it per store put a branch and a select in front of every one
  // of the ~40.  The split point is a loop bound instead.  `nh` is CP_W-2 by construction
  // (state_len_max is (CP_W-1)+(maxq-1) and slen_eff takes off the tokens this sequence did not
  // bring), but it is computed rather than assumed so a caller that shrinks the window differently
  // still gets the shipped answer.
  const int VAL = CP_W - 2;
  const int nh = slen_eff - slen < 0 ? 0 : (slen_eff - slen);
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
    unsigned short* dst = sp + c * csd;
#if CU_STATIC
    // The history half is bounded by the array it reads: `hist[i + 1]` with i < nh already required
    // nh <= CP_ST - 1, so unrolling to that bound reads exactly what the runtime loop did.
#pragma unroll
    for (int i = 0; i + 1 < CP_ST; ++i)
      if (i < nh && i < slen_eff) dst[i * cst] = hist[i + 1][c];
    // The token half, walked by TOKEN rather than by slot, so the index into xb is the loop counter.
    if (xpre) {
#pragma unroll
      for (int t = 0; t < CU_MAXT; ++t) {
        const int i = t + VAL;
        if (i >= nh && i < slen_eff && t < slen) {
          // NOT `((const unsigned short*)&xb[t])[c]`.  Taking the address of an element makes the
          // whole array address-taken, and an address-taken array does not become registers however
          // static the index is -- that one `&` costs 80 bytes a thread of SCRATCH under the
          // preload.  With t and c both compile-time this ternary folds to a register read.
          const ushort4 vv = xb[t];
          dst[i * cst] = (c == 0) ? vv.x : (c == 1) ? vv.y : (c == 2) ? vv.z : vv.w;
        }
      }
    }
    // Whatever the two loops above did not cover: a slot whose token is outside [0, slen), and the
    // whole range when the caller brought more tokens than the preload holds.  Both are empty on the
    // shipped path (slen_eff is slen + CP_ST - 1 and nh is CP_W - 2), so this stores nothing.
    for (int i = nh; i < slen_eff; ++i) {
      const int ti = i - VAL;
      if (!xpre || ti < 0 || ti >= slen) dst[i * cst] = xrow[ti * xpi + c];
    }
#else
    int i = 0;
    for (; i < nh && i < slen_eff; ++i) dst[i * cst] = hist[i + 1][c];
    for (; i < slen_eff; ++i) {
      const int ti = i - VAL;
      dst[i * cst] = (xpre && ti >= 0 && ti < slen)
                         ? ((const unsigned short*)&xb[ti])[c]
                         : xrow[ti * xpi + c];
    }
#endif
  }
#undef CU_TOKEN_BODY
}

extern "C" int r4d_gdn_conv_update_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
    int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  // A decode step is a handful of sequences, so the channel grouping is what decides whether the
  // grid covers the device: at 8 sequences and 8 heads per block this launches 40 workgroups for
  // 64 CUs. Narrow the block until there are enough of them.
  while (hpb > 1 && (long)N * (units / hpb) < 256) hpb >>= 1;
  dim3 grid(N, units / hpb), blk(hpb * 32);
#define CU_LAUNCH(P)                                                                          \
  hipLaunchKernelGGL((r4d_gdn_conv_update_kernel<P>), grid, blk, 0, (hipStream_t)stream,      \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,            \
                     (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim,    \
                     cs_tok, state_len_max, (const int*)cache_idx, ci_stride,                 \
                     (const int*)num_accepted, (unsigned short*)q, (unsigned short*)k,        \
                     (unsigned short*)v, (const int*)cu, H, Hg, max_query_len)
  switch (hpb) {
    case 8: CU_LAUNCH(8); break;
    case 4: CU_LAUNCH(4); break;
    case 2: CU_LAUNCH(2); break;
    default: CU_LAUNCH(1); break;
  }
#undef CU_LAUNCH
  return (int)hipGetLastError();
}

// THE OPERAND ENUMS CARRY AN `OPD_` PREFIX: this kernel's own tile constants are already called
// CP_W and CP_ST, and an unprefixed operand index named CP_W in the same translation unit is a
// redefinition -- caught by the compiler, but only because the spellings happen to collide. The
// prefix says which namespace a name is in.
// ============================================================== the plugin ABI entry points
//
// The shared operand helpers are r4d_gdn_args.h. See libr4d/r4d_args.h for why there is no
// adapter directory between the core and these launches.
/* ------------------------------------------------------------------ conv prep */


/* Inputs first, outputs last -- docs/OPS.md's resolved table. */
enum { OPD_CP_X = 0, OPD_CP_W, OPD_CP_B, OPD_CP_ALOG, OPD_CP_DTB, OPD_CP_CSTATE, OPD_CP_CU,
       OPD_CP_A, OPD_CP_BGATE, OPD_CP_CACHE_IDX, OPD_CP_HAS_INIT,
       OPD_CP_Q, OPD_CP_K, OPD_CP_V, OPD_CP_G, OPD_CP_BETA };

extern "C" int r4d_rad_gdn_conv_prep(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x    = r4d_opt(a, OPD_CP_X);
    const RadTensor* w    = r4d_opt(a, OPD_CP_W);
    const RadTensor* b    = r4d_opt(a, OPD_CP_B);
    const RadTensor* alog = r4d_opt(a, OPD_CP_ALOG);
    const RadTensor* dtb  = r4d_opt(a, OPD_CP_DTB);
    const RadTensor* cst  = r4d_opt(a, OPD_CP_CSTATE);
    const RadTensor* cu   = r4d_opt(a, OPD_CP_CU);
    const RadTensor* q    = r4d_opt(a, OPD_CP_Q);
    const RadTensor* k    = r4d_opt(a, OPD_CP_K);
    const RadTensor* v    = r4d_opt(a, OPD_CP_V);
    const RadTensor* g    = r4d_opt(a, OPD_CP_G);
    const RadTensor* beta = r4d_opt(a, OPD_CP_BETA);
    const RadTensor* ta   = r4d_opt(a, OPD_CP_A);
    const RadTensor* tb   = r4d_opt(a, OPD_CP_BGATE);
    const RadTensor* cidx = r4d_opt(a, OPD_CP_CACHE_IDX);
    const RadTensor* hin  = r4d_opt(a, OPD_CP_HAS_INIT);

    if (!x || !w || !alog || !dtb || !cst || !cu || !q || !k || !v || !g || !beta)
        return R4D_NO(RAD_E_INVAL);
    /* Optional in the SCHEMA so docs/OPS.md's shorter list stays a positional prefix, required by
     * the KERNEL: it dereferences all three unconditionally. Named rather than crashed. */
    if (!ta || !tb || !cidx) return R4D_NO(RAD_E_INVAL);

    if (x->rank < 2 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (g->rank != 2 || beta->rank != 2 || cst->rank != 3 || ta->rank != 2) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    /* x may be a padded VIEW -- the qkvz split the layer hands it -- so its row pitch is an
     * argument. Its inner axes still have to be tight. */
    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0) return R4D_NO(RAD_E_STRIDE);
    if (r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);

    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(g) ||
        !r4d_contig(beta) || !r4d_contig(w) || !r4d_contig(alog) || !r4d_contig(dtb))
        return R4D_NO(RAD_E_STRIDE);
    if (cidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R4D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    if (g->shape[0] != T || g->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (beta->shape[0] != T || beta->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    const int width = (int)r4d_geti_or(a, "conv_width", 4);
    const float thr = (float)r4d_getf(a, "softplus_thr", 20.0);
    /* libr4d's l2 norm is rsqrt(ss + 1e-6f) with the epsilon COMPILED IN, not an argument. A
     * caller that names a different one is asking for arithmetic this kernel does not do, and
     * running it anyway would be a silently different normalisation. Refused by name instead. */
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    /* `a` and `b` may be bf16 or f32; libr4d takes a flag rather than two entry points. */
    const int ab_is_bf16 = (ta->dtype == RAD_BF16) ? 1 : 0;
    if (tb->dtype != ta->dtype) return R4D_NO(RAD_E_DTYPE);
    if (!ab_is_bf16 && ta->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);

    /* "bf16 x / weights / conv state / q,k,v out, fp32 a,b and gate/beta out". The CONV WEIGHT is
     * bf16: the kernel takes it as `const unsigned short*` and reads it through bf2f. */
    R4D_DT(x, RAD_BF16); R4D_DT(w, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(g, RAD_F32); R4D_DT(beta, RAD_F32);

    return r4d_gdn_conv_prep_w4_h128_bf16(
        x->data, (long)xpitch, w->data, b ? b->data : nullptr, cst->data,
        (long)cst->stride[0], (long)cst->stride[1], (long)cst->stride[2],
        cidx->data, (long)cidx->stride[0], hin ? hin->data : nullptr,
        /* The WIDTH of has_init, not a guess: the engine hands over RadBatch::ctx_lens, which is
         * int32 like every derived buffer. Reading it as bytes is silent for a cold first chunk
         * and wrong for every one after it. */
        hin && (hin->dtype == RAD_I32 || hin->dtype == RAD_U32),
        ta->data, tb->data, (long)ab_stride, ab_is_bf16,
        alog->data, dtb->data, q->data, k->data, v->data, g->data, beta->data,
        cu->data, N, (int)T, (int)H, (int)Hg, (int)K, (int)V, width, thr, s);
}

/* ------------------------------------------------------------------ conv update */

enum { CUP_X = 0, CUP_W, CUP_B, CUP_CSTATE, CUP_SIDX, CUP_NACC, CUP_CU, CUP_Q, CUP_K, CUP_V };

extern "C" int r4d_rad_gdn_conv_update(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x    = r4d_opt(a, CUP_X);
    const RadTensor* w    = r4d_opt(a, CUP_W);
    const RadTensor* b    = r4d_opt(a, CUP_B);
    const RadTensor* cst  = r4d_opt(a, CUP_CSTATE);
    const RadTensor* sidx = r4d_opt(a, CUP_SIDX);
    const RadTensor* nacc = r4d_opt(a, CUP_NACC);
    const RadTensor* cu   = r4d_opt(a, CUP_CU);
    const RadTensor* q    = r4d_opt(a, CUP_Q);
    const RadTensor* k    = r4d_opt(a, CUP_K);
    const RadTensor* v    = r4d_opt(a, CUP_V);
    if (!x || !w || !cst || !sidx || !cu || !q || !k || !v) return R4D_NO(RAD_E_INVAL);
    if (cst->rank != 3 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (sidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(w))
        return R4D_NO(RAD_E_STRIDE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || k->shape[0] != T) return R4D_NO(RAD_E_SHAPE);

    /* "bf16 x / weights / conv state / q,k,v out" -- same casts as the prefill conv. */
    R4D_DT(x, RAD_BF16); R4D_DT(w, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16);

    const int width = (int)r4d_geti_or(a, "conv_width", 4);
    /* THE TWO WINDOW ARGUMENTS DESCRIBE ONE WINDOW, SO ONLY ONE OF THEM IS FREE. The state cache
     * is a rolling window read at the slot the last ACCEPTED token left, which is how a
     * speculative rejection becomes a change of read offset rather than a recompute (spec 10).
     * The kernel lays it out as `width - 2` history entries followed by the step's tokens, and it
     * finds the split as `state_len_max - max_query_len` -- so that difference MUST be width - 2,
     * whatever the two of them are individually. The cache depth fixes one; the other follows.
     *
     * DO NOT DERIVE max_query_len FROM THE STEP. The mean tokens per sequence is wrong twice over:
     * it is not the max when one request in the step drafts and another does not, and even the max
     * is not what this kernel wants, because the pool is sized for the deepest draft the
     * DEPLOYMENT allows. A decode step that drafts nothing would then put the split at
     * width - 2 + n_spec instead of width - 2. The history half runs past what the kernel has in
     * registers, this step's tokens land outside the range the next step reads, and the
     * convolution advances over a window that never changes: the model stops hearing its own last
     * three tokens and repeats one phrase forever. */
    const long long state_len_max = cst->shape[2];
    const long long mql = state_len_max - (width - 2);
    if (width < 2 || mql < 1) return R4D_NO(RAD_E_SHAPE);

    return r4d_gdn_conv_update_w4_h128_bf16(
        x->data, (long)xpitch, w->data, b ? b->data : nullptr, cst->data,
        (long)cst->stride[0], (long)cst->stride[1], (long)cst->stride[2],
        (int)state_len_max, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, q->data, k->data, v->data, cu->data,
        N, (int)H, (int)Hg, (int)K, (int)V, width, (int)mql, s);
}

