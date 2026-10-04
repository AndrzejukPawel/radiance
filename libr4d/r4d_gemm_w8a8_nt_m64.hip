// r4d_gemm_w8a8_nt_m64.hip - r4d_gemm_w8a8_nt_m64: 8-bit-weight, 8-bit-activation skinny GEMM for
// M up to a prefill chunk, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row and
// W per (row, 128-wide group).
//
// WHY THIS EXISTS.  The INT4 target promotes 80 of its 256 linears -- every down_proj and the
// sixteen attention o_proj -- to eight bits, because those two groups are where the 4-bit grid
// costs the most KL.  Serving those through a library int8 matmul means a composite -- rotate the
// activation with a bf16 matmul, quantise it, call the library, then scale an int32 [M,N] back to
// bf16 -- which is eight launches and three materialised intermediates per site, 80 sites a step.
// It also THROWS AWAY the fused rms+Hadamard+quantise producer the 4-bit sites read: the int8 the
// norm has already written is ignored and the bf16 is re-rotated and re-quantised.
//
// This is one launch that reads exactly what the 4-bit GEMM reads: the SAME permuted int8
// activation buffer, the SAME per-row scale, and a weight in the same fragment order at twice the
// radix.  Nothing about the producer changes; only the GEMM does.
//
// IT IS r4d_gemm_w4a8_nt_m64 WITH THE UNPACK DELETED.  That kernel spends two v_and_b32 and one
// v_lshlrev_b32 turning a dword of nibbles into eight int8 lanes; here the eight bytes ARE the
// fragment, so the B operand is two dwords straight out of the load and the dequant is nothing at
// all.  Everything else -- the A fragment read, the split-K reduction, the per-group scale folded
// with one v_cvt_f32_i32 and one fma, the per-row activation scale applied once at the store -- is
// the same code.
//
// WHY NOT gemm_w8a16_nt_m64, WHICH ALREADY EXISTS.  It runs v_wmma_f32_16x16x16_f16 at 207 TF/s
// where v_wmma_i32_16x16x16_iu8 runs at 407.  For lm_head at M=8 that is invisible -- the kernel is
// at the DRAM roofline and the matrix pipe is free -- but the promoted sites run at M=8 AND at
// M=72, and the f16 form is where the M=64 cliff in that kernel comes from.  An 8-bit weight has
// no reason to widen to 16 bits to be multiplied by an 8-bit activation.
//
// THE WEIGHT LAYOUT.  Per n-tile t (16 output rows) and k-block kb (KPB=32 columns), 32 lanes hold
// 16 bytes each: 512 bytes covering 16 rows x 32 k, which is TWO WMMA k steps.  Byte j of lane l is
//
//     W[t*16 + (l & 15)][kb*32 + 16*(j >> 3) + 8*((j & 7) >> 2) + 4*(l >> 4) + (j & 3)]
//
// -- the iu8 16x16x16 fragment order the A side already uses, so one permutation serves both
// operands and the packer is the same routine at a different radix.  A group is GROUP/KPB blocks
// and its f16 scale is read once per group per n-plane, as in the 4-bit kernel.
//
// THE SCALE PLANE IS f16-ONLY, NOT THE SHARED (scale, zero) DWORD.  There is no zero point on a
// symmetric grid and this kernel is DRAM-bound at the decode batch, so the dead half of that dword
// would be pure traffic: at group 128 it is 32 bytes per 2048 of weight, 1.5% of everything read.
// The 4-bit kernel keeps the dword because r4d_gemm_w4a16_nt_m64 shares its bytes; nothing shares
// these.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

// Not structural: mt0 = blockIdx.y * MT tiles M and the row index is clamped into 0..M-1, so any M
// is correct.  It is a guard against a caller handing over a garbage shape.
#define R4D_GEMM_W8A8_MAX_M 32768
#ifndef R4D_GEMM_W8A8_GROUP
#define R4D_GEMM_W8A8_GROUP 128
#endif
#define R4D_GEMM_W8A8_KPB   32       // K per packed block: 2 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i    __attribute__((ext_vector_type(8)));
typedef float    v4f    __attribute__((ext_vector_type(4)));
typedef int      v2i    __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const uint4v*        r4d_gw8;
typedef __attribute__((address_space(1))) const unsigned short* r4d_gs2;
typedef __attribute__((address_space(1))) const uint2v*        r4d_ga2;
typedef __attribute__((address_space(1))) const unsigned*      r4d_gu1;

// 0 reads A row-major, which only a harness feeding an unpermuted buffer needs; the serving path
// always feeds the permuted buffer the quantiser writes.
#ifndef R4D_W8A8_APERM
#define R4D_W8A8_APERM 1
#endif

// COST ABLATIONS.  Wrong answers by construction; they exist to say what the time is made of.
//   1  no scale load        (sc = 1)
//   2  no activation load   (the A fragment is a constant)
//   4  no weight load       (the B fragment is a constant; leaves the A stream and the WMMA)
#ifndef R4D_W8A8_ABL
#define R4D_W8A8_ABL 0
#endif

// SOFTWARE-PIPELINE THE WEIGHT: hold the NEXT group's weight in registers so its load is issued a
// whole group ahead of the wait, instead of issuing two loads, waiting, and then spending the WMMA
// block with no memory in flight.  Costs bpg*NPW*4 VGPRs.
#ifndef R4D_W8A8_PF
#define R4D_W8A8_PF 0
#endif

// STAGGER THE GROUP ORDER PER WAVE.  Every plane pitch here is a multiple of 4096 and a 256-bit
// GDDR6 part sweeps all sixteen channels in exactly 4096 bytes, so without this every wave in the
// grid is at the same address modulo 4096 and aimed at the same channel group.  Starting each wave
// at a different group inside its OWN k-slice and wrapping changes only the order.
// NOTE this reorders an fp32 accumulation, so results move by ~1 ULP.
#ifndef R4D_W8A8_STAGGER
#define R4D_W8A8_STAGGER 0
#endif

#if R4D_W8A8_STAGGER
#if R4D_W8A8_APERM
#define AOFF8(g) ((g) * (R4D_GEMM_W8A8_GROUP / 8))
#else
#define AOFF8(g) ((g) * (R4D_GEMM_W8A8_GROUP / 4))
#endif
#else
#define AOFF8(g) 0
#endif

// HOIST THE ACTIVATION LOADS OFF THEIR WMMA.
// Read `afrag[i]` inside the k-step loop immediately before the WMMA that consumes it and the group
// loop gets one wait per load: at MT=5, NPW=1 that is 40 `global_load_b64`, 40 WMMA and 37
// `s_wait_loadcnt`.  The wave issues one load, stalls on it, runs one matrix instruction and issues
// the next, so there is no memory-level parallelism and the matrix pipe is fully exposed instead of
// co-issued.  The ablations below price that exposure at a significant fraction of the kernel.
//
// A lane's eight A bytes for consecutive k steps are SIXTEEN bytes apart -- the intervening eight
// belong to the other lane half -- so they cannot be merged into one b128, and the layout is fixed:
// the same buffer feeds r4d_gemm_w4a8_nt_m64.  What can change is when they are
// issued.  1 loads the two k steps of a weight block up front (2*MT loads, then 2*NPW*MT WMMA);
// 2 loads the whole group (2*bpg*MT), which costs bpg times the registers.
#ifndef R4D_W8A8_AHOIST
#define R4D_W8A8_AHOIST 2
#endif

// The A fragment, byte for byte what r4d_gemm_w4a8_nt_m64 reads -- same buffer, same producer.
__device__ __forceinline__ v2i fragA8w(r4d_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

__device__ __forceinline__ v2i fragA8row(r4d_gu1 p, int u) {
  v2i r;
  r[0] = p[4 * u];
  r[1] = p[4 * u + 2];
  return r;
}

__device__ __forceinline__ v8i mmai8(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// __launch_bounds__ IS A REGISTER BUDGET, AND IT HAS TO SEE MT.
// A block of T threads is T/32 waves spread over four SIMDs, so the compiler budgets 1536/(T/128)
// VGPRs per wave: 1024 threads leaves 192, 512 leaves 256 (the architectural cap), 256 the same.
// The accumulator is what runs out -- `acc` is MT*NPW*8 VGPRs and `g_acc` another MT*NPW*8 while a
// group is in flight -- so MT*NPW is the thing that decides, and a bound written in NPW alone is
// blind to half of it.  A flat bound of 1024 leaves 192 VGPRs, and under that bound
// `-Rpass-analysis=kernel-resource-usage` reports every wide tile SPILLING:
//
//     MT=5..8, NPW=2   192 VGPRs, 15 / 51 / 87 / 123 spilled
//     MT=3,4,  NPW=4   256 VGPRs, 11 / 65 spilled
//     MT=2,3,4 NPW=8   256 VGPRs, 99 / 287 / 530 spilled
//
// and those are the configs M=72 wants, because MB=5 or 6 is what covers five row tiles in one
// block.  A wide tile has plenty of independent WMMA to hide latency with and does not need the
// waves; it needs the registers.  Bounding it at 256 threads gives it all 256 VGPRs.
//
// The bound is a REAL constraint on the launch, not a hint: a kernel built for 256 threads cannot
// be launched with more, so the host checks WV*SK against the same expression and refuses instead
// of failing at dispatch.
#ifndef R4D_GEMM_W8A8_LB
#define R4D_GEMM_W8A8_LB 1
#endif
#if R4D_GEMM_W8A8_LB
#define R4D_W8A8_BOUNDS(MT, NPW) ((MT) * (NPW) >= 8 ? 256 : ((MT) * (NPW) >= 4 ? 512 : 1024))
#else
#define R4D_W8A8_BOUNDS(MT, NPW) 1024
#endif

template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(R4D_W8A8_BOUNDS(MT, NPW)) void r4d_gemm_w8a8_nt_m64_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned short* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W8A8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Rows clamped, never masked: a tile past M recomputes the last valid row and the store
    // discards it.  A is int8, so the row stride is K bytes.
#if R4D_W8A8_APERM
    r4d_ga2 pa[MT];
#else
    r4d_gu1 pa[MT];
#endif
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
#if R4D_W8A8_APERM
      const size_t off = (size_t)r * K + kbase + 8 * (lane >> 4);
      pa[i] = (r4d_ga2)(const uint2v*)(A + off);
#else
      const size_t off = (size_t)r * K + kbase + 4 * (lane >> 4);
      pa[i] = (r4d_gu1)(const unsigned*)(A + off);
#endif
    }

    const int kpb = K / R4D_GEMM_W8A8_KPB;
    const int nsz = K / R4D_GEMM_W8A8_GROUP;
    r4d_gw8 pw[NPW];
    r4d_gs2 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw8)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W8A8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gs2)(const unsigned short*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W8A8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB;

#if R4D_W8A8_PF
    uint4v nx[NPW][R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int b = 0; b < bpg; b++)
        nx[j][b] = NT ? __builtin_nontemporal_load(&pw[j][(size_t)b * WAVE32])
                      : pw[j][(size_t)b * WAVE32];
#endif

#if R4D_W8A8_STAGGER
    const int gstag = groups > 1 ? (wave * R4D_W8A8_STAGGER) % groups : 0;
#endif
    for (int gi = 0; gi < groups; gi++) {
#if R4D_W8A8_STAGGER
      const int g = gi + gstag < groups ? gi + gstag : gi + gstag - groups;
#else
      const int g = gi;
#endif
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#if R4D_W8A8_ABL & 1
        sc[j] = 1.f;
#else
        sc[j] = __half2float(__ushort_as_half(ps[j][(size_t)g * 16]));
#endif

      v8i g_acc[NPW][MT];
#if R4D_W8A8_PF
      uint4v cw[NPW][R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int b = 0; b < bpg; b++) cw[j][b] = nx[j][b];
      if (g + 1 < groups) {
#pragma unroll
        for (int j = 0; j < NPW; j++)
#pragma unroll
          for (int b = 0; b < bpg; b++) {
            const size_t o = (size_t)((g + 1) * bpg + b) * WAVE32;
            nx[j][b] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          }
      }
#endif
#if R4D_W8A8_AHOIST == 2
      // The whole group's activation, issued before any of it is consumed.
      v2i afg[2 * (R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB)][MT];
#pragma unroll
      for (int u = 0; u < 2 * bpg; u++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#if R4D_W8A8_ABL & 2
        { afg[u][i][0] = 0x01010101; afg[u][i][1] = 0x01010101; }
#elif R4D_W8A8_APERM
          afg[u][i] = fragA8w(pa[i] + AOFF8(g), u);
#else
          afg[u][i] = fragA8row(pa[i] + AOFF8(g), u);
#endif
#endif
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
#if R4D_W8A8_PF
          wv[j] = cw[j][b];
#else
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
#if R4D_W8A8_ABL & 4
          (void)o;
          wv[j] = (uint4v)(0x01010101u);
#else
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
#endif
#endif
        }
#if R4D_W8A8_AHOIST == 1
        // Both k steps of this weight block, issued before either is consumed.
        v2i afb[2][MT];
#pragma unroll
        for (int s = 0; s < 2; s++)
#pragma unroll
          for (int i = 0; i < MT; i++)
#if R4D_W8A8_ABL & 2
          { afb[s][i][0] = 0x01010101; afb[s][i][1] = 0x01010101; }
#elif R4D_W8A8_APERM
            afb[s][i] = fragA8w(pa[i] + AOFF8(g), b * 2 + s);
#else
            afb[s][i] = fragA8row(pa[i] + AOFF8(g), b * 2 + s);
#endif
#endif
        // Two WMMA k steps per 16-byte weight load: dwords (0,1) then (2,3).
#pragma unroll
        for (int s = 0; s < 2; s++) {
          const int u = b * 2 + s;
          v2i afrag[MT];
#pragma unroll
#if R4D_W8A8_AHOIST == 2
          for (int i = 0; i < MT; i++) afrag[i] = afg[u][i];
#elif R4D_W8A8_AHOIST == 1
          for (int i = 0; i < MT; i++) afrag[i] = afb[s][i];
#elif R4D_W8A8_ABL & 2
          for (int i = 0; i < MT; i++) { afrag[i][0] = 0x01010101; afrag[i][1] = 0x01010101; }
#elif R4D_W8A8_APERM
          for (int i = 0; i < MT; i++) afrag[i] = fragA8w(pa[i] + AOFF8(g), u);
#else
          for (int i = 0; i < MT; i++) afrag[i] = fragA8row(pa[i] + AOFF8(g), u);
#endif
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            // The eight bytes ARE the fragment: no mask, no shift, no zero point.
            v2i bfrag;
            bfrag[0] = (int)wv[j][2 * s];
            bfrag[1] = (int)wv[j][2 * s + 1];
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? mmai8(afrag[i], bfrag, (v8i)(0))
                                     : mmai8(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf((float)g_acc[j][i][e], sc[j], acc[j][i][e]);
#if !R4D_W8A8_STAGGER
#pragma unroll
#if R4D_W8A8_APERM
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W8A8_GROUP / 8;
#else
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W8A8_GROUP / 4;
#endif
#endif
    }
  }

  // Split-K reduction and store, one row tile at a time.  The per-row activation scale is applied
  // here and only here: it is constant over n and over the group index, so it never entered the
  // k loop.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;

  // AT SK == 1 THERE IS NOTHING TO REDUCE.  The general path below writes eight floats per
  // accumulator to LDS and reads them straight back, and pays two __syncthreads() per row tile to
  // do it -- MB row tiles is MB barrier pairs on a kernel whose whole job is to keep a load in
  // flight.  With one k-slice a lane already holds the finished value, and the same (m, n) it
  // would have landed on is just the LDS index arithmetic with off = e*32 + lane substituted in.
  if (SK == 1) {
    if (n0 >= N) return;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + (lane & 15);
      if (n >= N) continue;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16 + 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int m = mb + e;
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m]);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m]);
    }
  }
}

void r4d_gemm_w8a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream) {
  if (M < 1 || M > R4D_GEMM_W8A8_MAX_M)
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: M out of range");
  if (N % 16) throw std::runtime_error("r4d gemm_w8a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W8A8_GROUP))
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > R4D_W8A8_BOUNDS(MB, NPW))
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: WV*SK*32 exceeds this tile's launch bound");
  // MB RUNS TO EIGHT, AND THAT IS THE WHOLE POINT AT M = 72.  A block covers MB row tiles and
  // grid.y takes the rest, so every extra grid.y walks the WHOLE weight again -- the same
  // "seventeen rows is two macro tiles" that makes a library GEMM read a 636 MB lm_head twice.  At
  // eight concurrent sequences the target verifies 8 x 9 = 72 rows, which is mtiles = 5, so a cap
  // of 4 would force grid.y = 2 and read the whole weight twice to do LESS work than M = 64 does
  // in one pass.  Five is the shape the model actually runs.
  if (MB < 1 || MB > 8) throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB must be 1..8");
  if (MB * NPW > 10) throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 10 would spill");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w8a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned short*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH8(MT, NP, TT) hipLaunchKernelGGL((r4d_gemm_w8a8_nt_m64_kernel<MT, NP, TT>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
// WHICH (MT, NPW) EXIST AT ALL.  `acc` is MT*NPW*8 VGPRs and `g_acc` another MT*NPW*8 while a
// group is in flight, so a tile costs 16*MT*NPW registers plus about forty of addressing against
// an architectural cap of 256 -- and `-Rpass-analysis=kernel-resource-usage` puts the boundary
// exactly where that predicts: MT*NPW = 10 lands at 207 VGPRs and does not spill, 12 spills 33,
// and 16 spills 119 to 530.  A spilling tile can never win, so it is not built: it would only cost
// compile time and give the tuner configs to reject.
#define LAUNCH8_MB8(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    case 3: LAUNCH8(3, NP, TT); break; \
    case 4: LAUNCH8(4, NP, TT); break; \
    case 5: LAUNCH8(5, NP, TT); break; \
    case 6: LAUNCH8(6, NP, TT); break; \
    case 7: LAUNCH8(7, NP, TT); break; \
    case 8: LAUNCH8(8, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB unsupported"); }
#define LAUNCH8_MB5(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    case 3: LAUNCH8(3, NP, TT); break; \
    case 4: LAUNCH8(4, NP, TT); break; \
    case 5: LAUNCH8(5, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 10 would spill"); }
#define LAUNCH8_MB2(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 10 would spill"); }
#define LAUNCH8_MB1(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 10 would spill"); }
#define LAUNCH8_NPW(TT) switch (NPW) { \
    case 1: LAUNCH8_MB8(1, TT); break; \
    case 2: LAUNCH8_MB5(2, TT); break; \
    case 4: LAUNCH8_MB2(4, TT); break; \
    case 8: LAUNCH8_MB1(8, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: NPW must be 1, 2, 4 or 8"); }
  if (NT) { LAUNCH8_NPW(true); } else { LAUNCH8_NPW(false); }
#undef LAUNCH8_NPW
#undef LAUNCH8_MB1
#undef LAUNCH8_MB2
#undef LAUNCH8_MB5
#undef LAUNCH8_MB8
#undef LAUNCH8
}

int r4d_gemm_w8a8_nt_m64_max_m(void) { return R4D_GEMM_W8A8_MAX_M; }
int r4d_gemm_w8a8_nt_m64_group(void) { return R4D_GEMM_W8A8_GROUP; }
int r4d_gemm_w8a8_nt_m64_kpb(void) { return R4D_GEMM_W8A8_KPB; }
int r4d_gemm_w8a8_nt_m64_aperm(void) { return R4D_W8A8_APERM; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
extern "C" int r4d_rad_gemm_w8a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_w8a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_w8a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_MBNPW, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w8a8_nt_m64",
        r4d_gemm_w8a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                             r4d_addr(g.bscale), r4d_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                             (long)(uintptr_t)s));
    return RAD_OK;
}

