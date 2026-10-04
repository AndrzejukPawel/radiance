// r4d_gemm_bf16_nt_tiled.hip - r4d_gemm_bf16_nt_tiled: the bf16 GEMM at prefill row counts, gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate: gemm_bf16_nt_m64's arithmetic in a
// block shape that reuses what it loads.
//
// WHY THE SKINNY KERNEL IS THE WRONG SHAPE ABOVE 64 ROWS. It gives a wave one 16-wide N tile and
// reads every fragment straight from global memory, so each WMMA costs an activation fragment
// load, and every N tile re-reads the whole activation. That is the right trade at decode, where
// the weight is the cost; at a 2048-row prefill chunk the activation is the larger operand and it
// is read N/16 times. Measured at M = 2048, K = 2560: the MoE router (N 512) 224 us, ~24 TF/s, and
// the PLE key projection (N 10240) 3.5 ms, ~31 TF/s, at the kernel's untuned point -- and no point
// of its space reaches 50.
//
// HERE a block of eight waves stages a BM x 64 activation tile and a BN x 64 weight tile through
// LDS, and each wave computes a (BM/2) x (BN/4) patch from them: every staged fragment feeds
// TM or TN WMMA instead of one. The next K tile is loaded into registers while the current one is
// computed, so the global latency sits behind four k steps of WMMA.
//
// BYTE-IDENTICAL TO gemm_bf16_nt_m64, and that is a contract rather than a coincidence. That
// kernel splits K into SK slices, accumulates each slice from zero with one WMMA per 16 K in
// ascending order, and sums the slices in ascending order starting from 0.f. This one issues the
// same WMMA sequence for every output tile, with the same fragment layout, and at the end of each
// slice adds the slice into a running fp32 sum that started at 0.f -- the same additions in the
// same order. SK comes from the same automatic rule (cfg_for), so an op whose band moves from one
// row to the other changes no bit of its output. What differs is only where the operands sit
// between global memory and the matrix unit.
//
// Fragment rows are CLAMPED, as in the skinny kernel: a tile that runs past M or N re-reads the
// last valid row, computes a result nobody stores, and never touches memory it does not own.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>

#define BT_BK 64                // K per staged tile: four WMMA k steps
// THE LDS ROW PITCH, in elements: 144 bytes. A fragment read has sixteen lanes on sixteen rows at
// one k, each reading 8 bytes; at 36 dwords a row the sixteen land on distinct bank pairs, and the
// upper half-wave's +2 dwords lands between them. 144 is also a multiple of 16, so a staged
// 16-byte chunk is one aligned ds_store_b128.
#define BT_PITCH (BT_BK + 8)
#define BT_WM 2                 // waves over M
#define BT_WN 4                 // waves over N

// TM x TN WMMA tiles a wave; BM = 2*TM*16, BN = 4*TN*16. SPLIT carries the slice sum.
template <int TM, int TN, bool SPLIT>
__global__ __launch_bounds__(256) void r4d_gemm_bf16_nt_tiled_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int spk) {
  constexpr int BM = BT_WM * TM * 16, BN = BT_WN * TN * 16;
  constexpr int CPR = BT_BK / 8;                  // 16-byte chunks a staged row
  constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
  static_assert(AL >= 1 && WL >= 1 && BM * CPR % 256 == 0 && BN * CPR % 256 == 0,
                "a tile is whole 16-byte chunks for every thread");
  __shared__ __attribute__((aligned(16))) unsigned short sA[BM * BT_PITCH];
  __shared__ __attribute__((aligned(16))) unsigned short sW[BN * BT_PITCH];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int wm = wave / BT_WN, wn = wave % BT_WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  // Staging: chunk c = tid + 256 i is row c / CPR, elements (c % CPR) * 8 of it. Eight threads
  // cover a row's 128 bytes, so a warp's loads are four whole lines.
  unsigned aoff[AL], woff[WL];
  unsigned short* adst[AL];
  unsigned short* wdst[WL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
    const int gr = m0 + r < M ? m0 + r : M - 1;
    aoff[i] = (unsigned)gr * (unsigned)K + (unsigned)e;
    adst[i] = sA + r * BT_PITCH + e;
  }
#pragma unroll
  for (int i = 0; i < WL; ++i) {
    const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
    const int gr = n0 + r < N ? n0 + r : N - 1;
    woff[i] = (unsigned)gr * (unsigned)K + (unsigned)e;
    wdst[i] = sW + r * BT_PITCH + e;
  }

  // Fragment sources: this lane's row of each tile at its half-K offset, as in rowP.
  const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * BT_PITCH + 4 * (lane >> 4);
  const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * BT_PITCH + 4 * (lane >> 4);

  v8f acc[TM][TN], tot[SPLIT ? TM : 1][SPLIT ? TN : 1];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);
  if constexpr (SPLIT) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) tot[i][j] = (v8f)(0.f);
  }

  uint4 ra[AL], rw[WL];
#pragma unroll
  for (int i = 0; i < AL; ++i) ra[i] = *(const uint4*)(A + aoff[i]);
#pragma unroll
  for (int i = 0; i < WL; ++i) rw[i] = *(const uint4*)(W + woff[i]);

  const int nk = K / BT_BK;
  int left = spk;                                 // k steps to the end of this split slice
  for (int kt = 0; kt < nk; ++kt) {
    __syncthreads();                              // the previous tile's fragments are all read
#pragma unroll
    for (int i = 0; i < AL; ++i) *(uint4*)adst[i] = ra[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) *(uint4*)wdst[i] = rw[i];
    __syncthreads();
    if (kt + 1 < nk) {
      const unsigned k1 = (unsigned)(kt + 1) * BT_BK;
#pragma unroll
      for (int i = 0; i < AL; ++i) ra[i] = *(const uint4*)(A + aoff[i] + k1);
#pragma unroll
      for (int i = 0; i < WL; ++i) rw[i] = *(const uint4*)(W + woff[i] + k1);
    }
#pragma unroll
    for (int s = 0; s < BT_BK / 16; ++s) {
      v8bf af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * BT_PITCH + s * 16);
#pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * BT_PITCH + s * 16);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
      if constexpr (SPLIT) {
        if (--left == 0) {
          left = spk;
#pragma unroll
          for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              tot[i][j] += acc[i][j];
              acc[i][j] = (v8f)(0.f);
            }
        }
      }
    }
  }

  // The skinny kernel's store: lane l holds column (l & 15) of a tile and rows 8 (l >> 4) + e.
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
      const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float v = 0.f;
        if constexpr (SPLIT) v = tot[i][j][e];
        else v += acc[i][j][e];
        const int m = mb + e;
        if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
      }
    }
}

// THE TWO TILES, and the rule between them. A block walks the whole of K alone, so its time is a
// floor -- ~40 us at K = 2560 -- and a grid too small to fill the part pays it with idle CUs:
// at M = 200, N = 512 the 128 x 128 tile is an eight-block grid. Measured at K = 2560 (us, 128 x
// 128 / 64 x 64): M 2048 N 512 66 / 114, N 2560 341 / 644, N 10240 1627 / 2661; M 500 N 512 41 / 31,
// N 2560 92 / 163; M 200 N 512 39 / 26, N 2560 66 / 72; M 80 N 2560 50 / 45. The large tile wins
// once its grid is 32 blocks, half the part's CUs, and loses by less than the small one does
// below that. (64 x 128 and 128 x 64 gain no more than a few percent over these.)
enum { kBtLargeGrid = 32 };

int r4d_gemm_bf16_nt_tiled_arm(int M, int N) {
  const long long g = (long long)((N + 127) / 128) * ((M + 127) / 128);
  return g >= kBtLargeGrid ? 1 : 2;
}

// `tile`: 1 = 128 x 128, 2 = 64 x 64 (M x N).
int r4d_gemm_bf16_nt_tiled(long a, long w, long c, int M, int K, int N, int SK, int tile,
                           long stream) {
  if (M < 1 || N < 1 || K < BT_BK || K % BT_BK) return -1;
  if (SK < 1 || K % (SK * 16)) return -1;
  if (tile != 1 && tile != 2) return -1;
  // 32-bit element offsets: the last row of either operand has to be addressable.
  if ((long long)M * K >= (1ll << 32) || (long long)N * K >= (1ll << 32)) return -1;
  const int spk = K / SK / 16;
  const int bt = tile == 1 ? 128 : 64;
  dim3 grid((N + bt - 1) / bt, (M + bt - 1) / bt), block(256);
  auto Ap = (const unsigned short*)a; auto Wp = (const unsigned short*)w;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(TM, TN, S) hipLaunchKernelGGL((r4d_gemm_bf16_nt_tiled_kernel<TM, TN, S>), grid, \
                                             block, 0, st, Ap, Wp, Cp, M, K, N, spk)
  const bool split = SK > 1;
  if (tile == 1) { if (split) LAUNCH(4, 2, true); else LAUNCH(4, 2, false); }
  else           { if (split) LAUNCH(2, 1, true); else LAUNCH(2, 1, false); }
#undef LAUNCH
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_bf16_tiled(const RadArgs* a, RadStream s) {
  GemmOps g;
  int rc = gemm_ops(a, &g, 0);
  if (rc != RAD_OK) return rc;
  // THE SPLIT IS THE SKINNY ROW'S, resolved by the same rule from the same arguments, which is
  // what keeps the two rows' outputs identical.
  Cfg c;
  rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);
  if (rc != RAD_OK) return rc;
  int tile = (int)r4d_geti_or(a, "tile", 0);
  if (tile == 0) tile = r4d_gemm_bf16_nt_tiled_arm((int)g.M, (int)g.N);
  const int e = r4d_gemm_bf16_nt_tiled(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y), (int)g.M,
                                       (int)g.K, (int)g.N, c.sk, tile, (long)(uintptr_t)s);
  return e == 0 ? RAD_OK : R4D_NO(e < 0 ? RAD_E_SHAPE : RAD_E_DEVICE);
}
