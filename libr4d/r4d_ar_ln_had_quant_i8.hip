// r4d_ar_ln_had_quant_i8.hip -- r4d_ar_ln_had_quant_i8: the tensor-parallel all-reduce and the
// residual add, RMS norm, block Hadamard and int8 quantiser that always follow it, in ONE kernel.
//
// WHY.  Every row-parallel linear in the model is followed by an all-reduce, and every one of those
// all-reduces is followed by a norm -- 128 of each on a 64-layer decode step, always adjacent and
// always in that order:
//
//     out_proj / o_proj / down_proj  ->  all_reduce  ->  (residual add, RMS norm, rotate, quantise)
//
// and the norm's output is the next quantised linear's A.  A decode step is launch-bound: over a
// thousand dispatches with ~2.3 us of unavoidable gap between each is several milliseconds of pure
// gap, and the standalone norm is itself latency- rather than bandwidth-bound (4.1 us for an
// 80 KiB row set, against a ~2.6 us launch floor).  Merging the pair removes 128 dispatches AND
// nearly all of the norm's fixed cost, because the norm's work is bolted onto a kernel that has
// already paid it.
//
// The all-reduce OUTPUT never exists.  In the unfused pair it is a bf16 [M, K] tensor written by
// the all-reduce and read straight back by the norm; here the peer's contribution is simply a
// second operand on the norm's load -- `t = mine + peer + residual` -- so the round trip is gone
// as well.
//
// WHY THE PARTITION IS BY ROW.  r4d_ar_oneshot_2rank_exact gives block b a contiguous chunk of 16B
// WORDS and an independent flag, with no grid barrier anywhere: b pushes its chunk to the peer,
// releases peer_flags[b], spins on my_flags[b], then reduces its own chunk.  A norm cannot use that
// partition -- a sum of squares closes only over a whole row -- but it does not need a grid barrier
// either, because rows are contiguous: give block b a contiguous range of ROWS and its chunk of
// words is exactly the bytes of those rows.  Both ranks compute the same range from (M, gridDim.x),
// so the handshake stays per-block and the kernel keeps the no-grid-barrier structure.
//
// The block count therefore has a ceiling the standalone all-reduce does not have (nb <= M), which
// costs nothing at the sizes this runs at: the standalone all-reduce asks for 4 blocks on an 80 KiB
// decode message and the row count is 8 or more.  The number of threads actually pushing is n16
// either way.
//
// Everything below the handshake is r4d_rmsnorm_had_quant_i8 unchanged -- the same one-pass
// argument (the FWHT is linear and inv_rms is a per-row scalar, so the rotation runs before the
// norm closes and the scale absorbs it), the same register staging rather than an LDS round trip,
// the same `wadd` added inside the fp32 arithmetic for GemmaRMSNorm's `x_hat * (1 + w)`, and the
// same optional unrotated bf16 output for a GDN block's bf16 in_proj_ba.  The only change is that
// the row is loaded from two places and that LDS is re-used across the rows a block owns, which
// needs one more barrier per row.
#include "r4d.h"
#include "r4d_ar_seq.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>

// The compressed all-reduce this fuses stores its fp32 sum to a bf16 tensor and the norm reads
// that back, so the fused kernel rounds the sum to bf16 too and reproduces the unfused pair's
// output exactly.  Build with -DARN_W6_MATCH=0 to keep the fp32 sum, which is strictly more
// accurate and NOT token-exact against the pair.
#ifndef ARN_W6_MATCH
#define ARN_W6_MATCH 1
#endif

#define R4D_ARN_SPIN_MAX 4000000000ULL
#define R4D_ARN_MAXVALS 72

__device__ __forceinline__ void arn_store_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arn_load_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---------------------------------------------------------------------------------------------
// THE ROTATED 6-BIT WIRE.  Same format and the same arithmetic as r4d_ar_oneshot_2rank_wht6 --
// orthonormal Walsh-Hadamard over a group of 64, amax-scaled, 6 uniform bits plus a bf16 scale --
// but packed at a CHUNK of 1024 elements instead of 2048, so that a chunk never straddles a row.
// That is what lets the norm keep its row partition: block b owns rows [r0, r1), and because K is
// a whole number of chunks its element range is a whole number of chunks too.
//
// The wire is why this exists.  Above ~200 KiB the message is bandwidth-bound and the compressed
// payload is 2.6x smaller, which is worth far more than the launch the fusion saves.  Carrying the
// payload in here lets the two wins compose: a message large enough to want the compressed wire
// still avoids a second dispatch and a bf16 round trip through memory.
//
// The helpers are duplicated rather than included from r4d_ar_oneshot_2rank_wht6.h on purpose:
// that header also carries the standalone kernel and its launcher, so including it would
// instantiate a second copy of a shipped kernel in this translation unit.
#define ARN_GROUP 64
#define ARN_BITS 6
#define ARN_EPL 8
#define ARN_LPG (ARN_GROUP / ARN_EPL)                   // 8 lanes span one group
#define ARN_WV 32
#define ARN_GPW (ARN_WV * ARN_EPL / ARN_GROUP)          // 4 groups per wave-iteration
// GROUPS PER CHUNK, and the only real tuning knob on the wire.  8 here, against the standalone
// kernel's 32: a wave owns a whole chunk, so the chunk size is what decides how many waves have
// work.  A 5120-group message is 160 chunks at 32 and 640 at 8, and the block count cannot make up
// the difference, because every block pays its own flag round trip over the fabric -- raising the
// count from 24 to 32 costs ~3%, to 64 ~4%, and to 160 ~24%.  Shrinking the chunk buys the same
// parallelism for nothing, and the wire's cost falls monotonically from 32 groups to 8.
#ifndef ARN_CG
#define ARN_CG 8
#endif
                                                        // groups per chunk
#define ARN_CE (ARN_CG * ARN_GROUP)                     // 1024 elements per chunk
#define ARN_ITERS (ARN_CE / (ARN_WV * ARN_EPL))         // 4 iterations to cover a chunk
#define ARN_NWORD (ARN_ITERS * ARN_EPL * ARN_BITS / 32) // 6 dwords a lane holds for a chunk
#define ARN_CBYTES (ARN_CE * ARN_BITS / 8)              // 768 wire bytes a chunk
#define ARN_QMAX ((1 << (ARN_BITS - 1)) - 1)
#define ARN_QMIN (-(1 << (ARN_BITS - 1)))
#define ARN_MASK ((1u << ARN_BITS) - 1u)

// RTNE to bf16 and back: the scale ships as its top 16 bits, so the sender's own copy has to be
// exactly what the receiver reconstructs.
__device__ __forceinline__ float arn_bf16_round(float v) {
  unsigned u = __float_as_uint(v);
  const unsigned lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return __uint_as_float(u & 0xffff0000u);
}


// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`.  The sign flip is one XOR against a
// mask that is uniform over the whole stage, so the select disappears: two instructions an element
// (xor, then the DPP-sourced add) instead of an add, a subtract and a v_cndmask.  Bit-identical --
// IEEE negation is exact and addition commutes.
template <int M>
__device__ __forceinline__ void arn_xstage(float* v, int lig) {
  const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
  for (int i = 0; i < ARN_EPL; ++i) {
    const float o = r4d_xor_lane<M>(v[i]);
    v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
  }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this routine also undoes it.
// A group is ARN_LPG lanes x ARN_EPL elements: the in-register butterflies are free ALU, and with
// ROW_XMASK the cross-lane stages are free too.
__device__ __forceinline__ void arn_wht(float* v, int lig) {
  static_assert(ARN_LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
  for (int h = 1; h < ARN_EPL; h <<= 1)
#pragma unroll
    for (int i = 0; i < ARN_EPL; ++i)
      if ((i & h) == 0) { const float x = v[i], y = v[i ^ h]; v[i] = x + y; v[i ^ h] = x - y; }
  arn_xstage<1>(v, lig);
  arn_xstage<2>(v, lig);
  arn_xstage<4>(v, lig);
  const float nrm = rsqrtf((float)ARN_GROUP);
#pragma unroll
  for (int i = 0; i < ARN_EPL; ++i) v[i] *= nrm;
}

template <int HAD, int THR, bool WOUT, int WIRE>
__global__ __launch_bounds__(THR) void r4d_ar_ln_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ X, __hip_bfloat16* __restrict__ RES,
    const __hip_bfloat16* __restrict__ Wg, __hip_bfloat16* __restrict__ OUTB,
    signed char* __restrict__ Q, float* __restrict__ S,
    int4* peer_base, const int4* my_base,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int M, int K, int slot_stride16, int drain, int acq,
    float eps, float wadd, float rnorm, unsigned int nbw) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  constexpr int MAXB = R4D_ARN_MAXVALS / VPL;
  extern __shared__ float lds[];
  // One staging slice per row the block owns.  The exact wire needs only one (the norm reads the
  // row straight from memory), but the compressed wire has to REDUCE first, and a per-row reduce
  // would put cpr chunks -- five at K = 5120 -- in front of NW waves behind a barrier, three times
  // over.  With the whole block's rows staged, the reduce is one pass over every chunk the block
  // owns and there is one barrier, not one per row.
  const int r0_ = (int)(((long)blockIdx.x * M) / gridDim.x);
  const int nrows = (int)(((long)(blockIdx.x + 1) * M) / gridDim.x) - r0_;
  float* red = lds + (size_t)(WIRE ? nrows : 1) * K;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int nblk = K / HAD;
  // Contiguous row range. Integer arithmetic only, so both ranks land on the same split.
  const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);

  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int sq = s_seq;
  int4* ps = peer_base + (size_t)s_slot * slot_stride16;
  const int4* ms = my_base + (size_t)s_slot * slot_stride16;

  // WIRE == 6 only.  The slot is split exactly as the standalone compressed all-reduce splits it:
  // packed symbols from the base, bf16 group scales from half way up.  Chunk indices are GLOBAL
  // over the whole [M, K] message, so both ranks address the same bytes without exchanging
  // anything; a block simply owns the chunks its rows cover.
  const int cpr = K / ARN_CE;                    // chunks a row spans (K % ARN_CE == 0, checked)
  const int c00 = r0 * cpr, cpb = (r1 - r0) * cpr;
  char* p_pay = (char*)ps;
  const char* m_pay = (const char*)ms;
  unsigned short* p_sc = (unsigned short*)((char*)ps + (size_t)(slot_stride16 >> 1) * 16);
  const unsigned short* m_sc =
      (const unsigned short*)((const char*)ms + (size_t)(slot_stride16 >> 1) * 16);
  // This rank's own packed half, kept in LDS so the reduce folds exactly the bytes it sent
  // without a global round trip -- the standalone kernel needs a side buffer for this because it
  // has no LDS to spare; here the norm's staging array is already the reason there is a barrier.
  unsigned int* Lpk = (unsigned int*)(red + 2 * NW);
  float* Lsc = (float*)(Lpk + (size_t)cpb * ARN_WV * ARN_NWORD);
  const int lig = lane & (ARN_LPG - 1), gw = lane / ARN_LPG;

  if constexpr (WIRE == 0) {
    // push my rows -> peer slot.  K bf16 is K/8 16B words, and rows are contiguous, so a row range
    // is a word range.
    const int wpr = K >> 3;
    const int wstart = r0 * wpr, wend = r1 * wpr;
    const int4* in4 = (const int4*)X;
    int k = wstart + tid;
    for (; k + 3 * THR < wend; k += 4 * THR) {
      int4 v0 = in4[k], v1 = in4[k + THR], v2 = in4[k + 2 * THR], v3 = in4[k + 3 * THR];
      ps[k] = v0; ps[k + THR] = v1; ps[k + 2 * THR] = v2; ps[k + 3 * THR] = v3;
    }
    for (; k < wend; k += THR) ps[k] = in4[k];
  } else {
    // rotate, amax, quantise and pack straight into the peer's slot, keeping the same words in
    // LDS.  A wave owns whole chunks, so every lane emits exactly ARN_NWORD dwords per chunk and
    // the flush is lane-uniform: one 512B uint4 store and one 256B uint2 store a wave a chunk.
    for (int ci = wave; ci < cpb; ci += NW) {
      const int c = c00 + ci;
      unsigned int acc[ARN_NWORD];
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w) acc[w] = 0u;
#pragma unroll
      for (int t = 0; t < ARN_ITERS; ++t) {
        const int gl = t * ARN_GPW + gw;                 // group within the chunk
        const long off = (long)(c * ARN_CG + gl) * ARN_GROUP + (long)lig * ARN_EPL;
        float v[ARN_EPL];
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) v[i] = __bfloat162float(X[off + i]);
        arn_wht(v, lig);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
        a = fmaxf(a, r4d_xor_lane<1>(a));
        a = fmaxf(a, r4d_xor_lane<2>(a));
        a = fmaxf(a, r4d_xor_lane<4>(a));
        float scl = (a > 0.f) ? arn_bf16_round(a * (1.0f / (float)ARN_QMAX)) : 1.0f;
        if (!(scl > 0.f)) scl = 1.0f;
        if (lig == 0) {
          p_sc[c * ARN_CG + gl] = (unsigned short)(__float_as_uint(scl) >> 16);
          Lsc[(size_t)ci * ARN_CG + gl] = scl;
        }
        const float inv = 1.0f / scl;
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) {
          const unsigned int sym =
              (unsigned int)(min(max((int)rintf(v[i] * inv), ARN_QMIN), ARN_QMAX) - ARN_QMIN);
          const int bit = (t * ARN_EPL + i) * ARN_BITS, wi = bit >> 5, bo = bit & 31;
          acc[wi] |= sym << bo;
          if (bo + ARN_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
        }
      }
      // Flush lane-contiguously region by region, widest first, so each region is one coalesced
      // store of ARN_WV x width bytes.  ARN_NWORD is 6 at ARN_CG 16 and 3 at 8.
      char* pb = p_pay + (size_t)c * ARN_CBYTES;
#pragma unroll
      for (int w4 = 0; w4 + 4 <= ARN_NWORD; w4 += 4)
        *(uint4*)(pb + w4 * 4 * ARN_WV + lane * 16) =
            make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]);
      if constexpr (ARN_NWORD % 4 >= 2)
        *(uint2*)(pb + (ARN_NWORD & ~3) * 4 * ARN_WV + lane * 8) =
            make_uint2(acc[ARN_NWORD & ~3], acc[(ARN_NWORD & ~3) + 1]);
      if constexpr (ARN_NWORD % 2 == 1)
        *(unsigned int*)(pb + (ARN_NWORD - 1) * 4 * ARN_WV + lane * 4) = acc[ARN_NWORD - 1];
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w)
        Lpk[((size_t)ci * ARN_NWORD + w) * ARN_WV + lane] = acc[w];
    }
  }
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    arn_store_rel(&peer_flags[b], sq);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation: norming a slot the peer never
    // filled would hand stale bytes on as a sum, so it traps instead.
    while (r4d_ar_seq_behind(arn_load_acq(&my_flags[b]), sq)) {
      if (++z > R4D_ARN_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();

  // The peer's rows are now in MY scratch, in the same [M, K] layout as X.
  const __hip_bfloat16* P = (const __hip_bfloat16*)ms;

  if constexpr (WIRE == 6) {
    // REDUCE: unpack both halves of every chunk this block owns, add in fp32, inverse-rotate and
    // leave the reduced value in the norm's own staging slice.  The norm then reads it exactly
    // where it would have read `x + p`, and overwrites the same slot with its rotated result --
    // one lane owns a slot in both passes, so the aliasing needs no second array.
    for (int ci = wave; ci < cpb; ci += NW) {
      const int c = c00 + ci;
      const char* pb = m_pay + (size_t)c * ARN_CBYTES;
      unsigned int pac[ARN_NWORD], lac[ARN_NWORD];
#pragma unroll
      for (int w4 = 0; w4 + 4 <= ARN_NWORD; w4 += 4) {
        const uint4 pq = *(const uint4*)(pb + w4 * 4 * ARN_WV + lane * 16);
        pac[w4] = pq.x; pac[w4 + 1] = pq.y; pac[w4 + 2] = pq.z; pac[w4 + 3] = pq.w;
      }
      if constexpr (ARN_NWORD % 4 >= 2) {
        const uint2 pu = *(const uint2*)(pb + (ARN_NWORD & ~3) * 4 * ARN_WV + lane * 8);
        pac[ARN_NWORD & ~3] = pu.x; pac[(ARN_NWORD & ~3) + 1] = pu.y;
      }
      if constexpr (ARN_NWORD % 2 == 1)
        pac[ARN_NWORD - 1] = *(const unsigned int*)(pb + (ARN_NWORD - 1) * 4 * ARN_WV + lane * 4);
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w)
        lac[w] = Lpk[((size_t)ci * ARN_NWORD + w) * ARN_WV + lane];
#pragma unroll
      for (int t = 0; t < ARN_ITERS; ++t) {
        const int gl = t * ARN_GPW + gw;
        const float psc = __uint_as_float((unsigned int)m_sc[c * ARN_CG + gl] << 16);
        const float lsc = Lsc[(size_t)ci * ARN_CG + gl];
        float v[ARN_EPL];
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) {
          const int bit = (t * ARN_EPL + i) * ARN_BITS, wi = bit >> 5, bo = bit & 31;
          unsigned int psy = pac[wi] >> bo, lsy = lac[wi] >> bo;
          if (bo + ARN_BITS > 32) {
            psy |= pac[wi + 1] << (32 - bo);
            lsy |= lac[wi + 1] << (32 - bo);
          }
          // The two ranks fold the identical pair in OPPOSITE order; fp32 add commutes exactly, so
          // they stay bit-identical as long as neither product is contracted into an FMA.
          // libr4d/CMakeLists.txt passes -ffp-contract=off for exactly this.
          v[i] = (float)((int)(psy & ARN_MASK) + ARN_QMIN) * psc
               + (float)((int)(lsy & ARN_MASK) + ARN_QMIN) * lsc;
        }
        arn_wht(v, lig);
        const int e0 = (int)((long)(c * ARN_CG + gl) * ARN_GROUP + (long)lig * ARN_EPL
                             - (long)r0 * K);
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i)
#if ARN_W6_MATCH
          lds[e0 + i] = __bfloat162float(__float2bfloat16(v[i]));
#else
          lds[e0 + i] = v[i];
#endif
      }
    }
    __syncthreads();
  }

  for (int row = r0; row < r1; ++row) {
    float* rl = lds + (size_t)(WIRE ? (row - r0) : 0) * K;
    const __hip_bfloat16* x = X + (size_t)row * K;
    const __hip_bfloat16* p = P + (size_t)row * K;
    __hip_bfloat16* res = RES ? RES + (size_t)row * K : nullptr;
    (void)x; (void)p;
    float u[WOUT ? MAXB : 1][VPL];
    float ss = 0.f, m = 0.f;
    int nbk = 0;
    for (int blk = wave; blk < nblk; blk += NW, nbk++) {
      const int base = blk * HAD + lane * VPL;
      float v[VPL];
#pragma unroll
      for (int j = 0; j < VPL; j++) {
        // WIRE == 0: ROUNDED TO bf16 ON PURPOSE.  The standalone all-reduce accumulates in fp32
        // and STORES bf16, and the norm then reads that back -- so folding the reduce into the
        // norm's load would otherwise drop a rounding step and change the token stream.  It would
        // change it for the better, but keeping the fused form bit-identical to the pair it
        // replaces is worth more than the accuracy.  The price is one software RTNE per element
        // (gfx1201 has no bf16 convert instruction) on a kernel that is launch-bound rather than
        // ALU-bound.  WIRE == 6 has no such pair to match: the compressed all-reduce it
        // replaces stores the fp32 sum straight to bf16 and the norm reads that, so here the sum
        // is taken from LDS in fp32 and the bf16 round trip is simply gone.
        float t;
        if constexpr (WIRE == 0) {
          t = __bfloat162float(__float2bfloat16(
              __bfloat162float(x[base + j]) + __bfloat162float(p[base + j])));
        } else {
          t = rl[base + j];
        }
        if (res) {
          t += __bfloat162float(res[base + j]);
          res[base + j] = __float2bfloat16(t);
        }
        ss += t * t;
        v[j] = t * (__bfloat162float(Wg[base + j]) + wadd);
        if (WOUT) u[nbk][j] = v[j];
      }
      r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
      for (int j = 0; j < VPL; j++) {
        rl[base + j] = v[j];
        m = fmaxf(m, fabsf(v[j]));
      }
    }
    ss = r4d_wave_sum(ss);
    m = r4d_wave_max(m);
    if (lane == 0) { red[wave] = ss; red[NW + wave] = m; }
    __syncthreads();
    ss = 0.f; m = 0.f;
#pragma unroll
    for (int i = 0; i < NW; i++) { ss += red[i]; m = fmaxf(m, red[NW + i]); }
    const float inv_rms = rsqrtf(ss / (float)K + eps);
    m = fmaxf(m, 1e-8f);
    const float inv = 127.f / m;
    if (tid == 0) S[row] = m * inv_rms * rnorm / 127.f;

    if (WOUT) {
      nbk = 0;
      for (int blk = wave; blk < nblk; blk += NW, nbk++) {
        const int base = blk * HAD + lane * VPL;
#pragma unroll
        for (int j = 0; j < VPL; j++)
          OUTB[(size_t)row * K + base + j] = __float2bfloat16(u[nbk][j] * inv_rms);
      }
    }

    signed char* q = Q + (size_t)row * K;
    for (int i = tid; i < K; i += THR) {
      const int s = i >> 4, j = i & 15;
      const int sr = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
      const int t = __float2int_rn(rl[sr] * inv);
      q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
    // lds and red are re-used by the next row this block owns.
    __syncthreads();
  }
}

// What the compressed wire costs in LDS, so the caller can pick its block count instead of
// guessing.  A block owns whole rows and holds, per row, a staging slice at `stage_bits` bits an
// element plus the packed half of its own chunks; getting this wrong is not a slow launch but a
// refused one.
void r4d_ar_ln_wire_dims(int* stage_bits, int* chunk_elems, int* chunk_bytes, int* scale_bytes) {
  // 32: the reduced values are staged as fp32 IN THE NORM'S OWN ARRAY -- the norm reads a slot
  // and overwrites it with its rotated result, so the staging is not extra LDS at all.  Staging
  // at bf16 instead would fit half again as many rows per block, but all that buys is a LOWER
  // block count and this kernel wants a higher one; at the row counts it actually runs, the two
  // occupy the same LDS and bf16 only adds conversions.
  if (stage_bits)  *stage_bits  = 32;
  if (chunk_elems) *chunk_elems = ARN_CE;
  if (chunk_bytes) *chunk_bytes = ARN_WV * ARN_NWORD * (int)sizeof(unsigned int);
  if (scale_bytes) *scale_bytes = ARN_CG * (int)sizeof(float);
}

// The LDS one launch asks for: the norm's fp32 staging -- one row on the exact wire, every row the
// block owns on the compressed one -- the two reduction planes, and on the compressed wire this
// rank's packed half of every chunk the block owns. One formula for the launcher and for
// r4d_ar_ln_blocks_min, so the block count a caller is told fits is the one the launcher accepts.
static size_t ar_ln_lds_bytes(int rows_max, int K, int thr, int wire) {
  size_t b = (size_t)(wire ? rows_max : 1) * K * sizeof(float) + 2 * (thr / 32) * sizeof(float);
  if (wire) {
    const size_t cpb = (size_t)rows_max * (K / ARN_CE);
    b += cpb * ARN_WV * ARN_NWORD * sizeof(unsigned int) + cpb * ARN_CG * sizeof(float);
  }
  return b;
}
// The ceiling on that dynamic allocation: a workgroup's 64 KiB less the kernel's own static words
// (the sequence and the slot), which the runtime places in front of it.
static constexpr size_t kArnLdsCap = 65536 - 16;

// The fewest blocks at which this shape fits: every block owns ceil(M / nb) rows, and on the
// compressed wire each of them costs a staging slice and its chunks' packed half. -1 when no
// block count serves it -- a row too wide to stage even alone, or too wide for the register
// staging at this width -- which is the case the launcher would otherwise throw on.
int r4d_ar_ln_blocks_min(int M, int K, int had, int nthreads, int wire) {
  if (M < 1 || K < 1 || had < 32 || K % had) return -1;
  if (wire != 0 && wire != 6) return -1;
  if (wire && K % ARN_CE) return -1;
  const int thr = nthreads ? nthreads : r4d_act_threads(M, K);
  if (thr < 32) return -1;
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_ARN_MAXVALS) return -1;
  if (!wire) return ar_ln_lds_bytes(1, K, thr, 0) <= kArnLdsCap ? 1 : -1;
  const size_t fixed = ar_ln_lds_bytes(0, K, thr, wire);
  const size_t row = ar_ln_lds_bytes(1, K, thr, wire) - fixed;
  if (fixed + row > kArnLdsCap) return -1;
  const long fit = (long)((kArnLdsCap - fixed) / row);
  return (int)(((long)M + fit - 1) / fit);
}

void r4d_ar_ln_had_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16,
                            long x, long residual, long wgain, long out_bf16, long q, long s,
                            int M, int K, int had, float eps, float wadd,
                            long stream, int nblocks, int nthreads, int drain, int acq, int wire) {
  if (K % 16) throw std::runtime_error("r4d ar_ln_had_quant_i8: K must be a multiple of 16");
  if (K % 8) throw std::runtime_error("r4d ar_ln_had_quant_i8: K*2 must be 16B-aligned");
  if (M < 1) throw std::runtime_error("r4d ar_ln_had_quant_i8: M must be positive");
  if (had < 2 || (had & (had - 1))) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: had must be a power of two");
  if (K % had) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: had must divide K -- a rotation block cannot straddle a "
      "tensor-parallel shard boundary");
  int nb = nblocks < 1 ? 1 : nblocks;
  if (nb > M) nb = M;                       // row-aligned: a block owns whole rows
  const int nbw = r4d_ar_max_blocks();      // the counter capacity; see r4d_ar_seq.h
  if (nb > nbw) nb = nbw;
  const int thr = nthreads ? nthreads : r4d_act_threads(M, K);
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_ARN_MAXVALS)
    throw std::runtime_error(
        "r4d ar_ln_had_quant_i8: K too large for the register staging at this width");
  const int rows_max = (M + nb - 1) / nb;
  if (wire) {
    if (wire != 6) throw std::runtime_error("r4d ar_ln_had_quant_i8: wire must be 0 or 6");
    // A chunk may not straddle a row, or a block's rows would not be a whole number of chunks
    // and the two ranks would disagree about who owns which bytes.
    if (K % ARN_CE) throw std::runtime_error(
        "r4d ar_ln_had_quant_i8: the 6-bit wire needs K to be a multiple of 1024");
  }
  const size_t shmem = ar_ln_lds_bytes(rows_max, K, thr, wire);
  if (shmem > kArnLdsCap) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: the rows a block owns do not fit LDS at this block count");
  const float rn = 1.f / sqrtf((float)had);
#define R4D_ARN_LAUNCH(H, T, W, R) \
    hipLaunchKernelGGL((r4d_ar_ln_had_quant_i8_kernel<H, T, W, R>), \
    dim3(nb), dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)x, \
    (__hip_bfloat16*)residual, (const __hip_bfloat16*)wgain, (__hip_bfloat16*)out_bf16, \
    (signed char*)q, (float*)s, (int4*)peer_scratch, (const int4*)my_scratch, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, M, K, (int)slot_stride16, \
    drain, acq, eps, wadd, rn, (unsigned)nbw)
#define R4D_ARN_T(H, W, R) switch (thr) { \
    case 128: R4D_ARN_LAUNCH(H, 128, W, R); break; \
    case 256: R4D_ARN_LAUNCH(H, 256, W, R); break; \
    case 512: R4D_ARN_LAUNCH(H, 512, W, R); break; \
    case 1024: R4D_ARN_LAUNCH(H, 1024, W, R); break; \
    default: throw std::runtime_error( \
        "r4d ar_ln_had_quant_i8: workgroup width must be 128, 256, 512 or 1024"); }
#define R4D_ARN_W(W, R) switch (had) { \
    case 128: R4D_ARN_T(128, W, R); break; \
    case 256: R4D_ARN_T(256, W, R); break; \
    case 512: R4D_ARN_T(512, W, R); break; \
    default: throw std::runtime_error( \
        "r4d ar_ln_had_quant_i8: only widths 128, 256 and 512 are compiled"); }
  if (wire) {
    if (out_bf16) { R4D_ARN_W(true, 6); } else { R4D_ARN_W(false, 6); }
  } else {
    if (out_bf16) { R4D_ARN_W(true, 0); } else { R4D_ARN_W(false, 0); }
  }
#undef R4D_ARN_W
#undef R4D_ARN_T
#undef R4D_ARN_LAUNCH
}
