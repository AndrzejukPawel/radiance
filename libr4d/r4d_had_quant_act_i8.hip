// r4d_had_quant_act_i8.hip - r4d_had_quant_act_i8: a block Hadamard along K, then per-row symmetric
// int8 quantisation, written in the order r4d_gemm_w4a8_nt_m64's A fragment wants.
//
// WHY THE ROTATION LIVES HERE.  Plain per-token int8 activations are unusable on these models: the
// KL against the bf16 model is an order of magnitude worse than per-token e4m3.  A block Hadamard
// first puts int8 several times BETTER than e4m3 instead.  The mechanism is that a rotation fixes
// RANGE, not precision: int8 spends its whole grid covering a few outlier channels, and mixing each
// value with 127 others removes them.  The same rotation does nothing for e4m3 -- marginally worse,
// in fact -- because four exponent bits already carry the outliers and Gaussianising does not add
// mantissa bits.
//
// It has to happen BEFORE the absmax, which is why it belongs in the quantiser rather than in front
// of it: the entire benefit is that the row maximum the int8 grid is scaled by gets smaller.
//
// THE ROTATION IS FREE, THE LAUNCH IS NOT.  A quantiser has to run either way, so seven butterfly
// stages ride along on a launch already being paid -- ~14 ops per element on a kernel that already
// touches every element twice.  What costs is the number of these a decode step needs: one per
// rotated site per layer.  That cost comes out by fusing the rotation into the RMSNorm or SwiGLU
// that produces the input, not by making the transform cheaper.
//
// WIDTH.  128, 256 and 512 are all buildable; 1024 is NOT, and the reason is sharding rather than
// quality.  A rotation block cannot straddle a tensor-parallel boundary, so the width has to divide
// the PER-RANK K, and a down_proj of 17408 at TP=2 is 8704 = 2^9 * 17.  512 divides it, 1024 does
// not.  Width is nearly free to choose because the weight error barely moves across 128 through
// 1024, so it is picked on the activation's terms alone.
//
// THE TRANSFORM IS IN REGISTERS, NOT LDS.  See r4d_fwht.h for why staging the butterfly in LDS
// costs several times the whole quantiser.  One wave owns a `had`-wide block in registers and the
// only shared traffic left is a single staging write, because the row absmax still has to see the
// whole row.
//
// NORMALISATION IS FOLDED INTO THE SCALE.  The stages below compute the UNNORMALISED transform, whose
// values are sqrt(had) times the orthonormal ones.  Rather than a pass to divide them, the row scale
// carries it: quantising with inv = 127/absmax_unnorm gives exactly the codes the normalised
// transform would, and S[row] = absmax_unnorm / (sqrt(had) * 127) is the scale that undoes both.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

// 0 = choose the workgroup width from M at launch (see r4d_act_threads in r4d_common.h); a
// non-zero build-time value pins one width for every M.
#ifndef R4D_HQAI8_THREADS
#define R4D_HQAI8_THREADS 0
#endif

template <int HAD, int THR>
__global__ __launch_bounds__(THR) void r4d_had_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K, float rnorm) {
  constexpr int VPL = HAD / 32;
  extern __shared__ float lds[];                 // K floats of transformed row, then the reduction
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  constexpr int NW = THR / 32;

  float m = 0.f;
  for (int blk = wave; blk < K / HAD; blk += NW) {
    const int base = blk * HAD + lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) v[j] = __bfloat162float(a[base + j]);
    r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      lds[base + j] = v[j];
      m = fmaxf(m, fabsf(v[j]));
    }
  }
  m = r4d_wave_max(m);
  if (lane == 0) red[wave] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < NW; i++) m = fmaxf(m, red[i]);
  m = fmaxf(m, 1e-8f);
  const float inv = 127.f / m;
  if (threadIdx.x == 0) S[row] = m * rnorm / 127.f;

  // Same fragment permutation as r4d_quant_act_i8: byte j of a k step comes from lane-half j>>3 and
  // element j&7, i.e. source 8*((j&7)>>2) + 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[src] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The launch, as the plugin ABI's RadLaunchFn. See the note at the head of r4d_quant_act_i8.hip on
// the argument list and the status return.
//
// THE WIDTH SET IS A PROPERTY OF THIS FILE, not of the row that selects it: the rows bound the
// rotation with `had <= 512` and nothing else, and the kernel compiles exactly 128, 256 and 512.
// 1024 cannot tile a down_proj of 8704 per rank, and nothing narrower than 128 is instantiated, so
// a legal-looking 64 is refused here.
enum { HQ_X = 0, HQ_Q, HQ_S };

extern "C" int r4d_rad_had_quant_act_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* x  = r4d_opt(args, HQ_X);
  const RadTensor* q  = r4d_opt(args, HQ_Q);
  const RadTensor* sc = r4d_opt(args, HQ_S);
  if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  const int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);

  const long long had = r4d_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return R4D_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard boundary, so the PER-RANK K is what
  // it has to divide -- which is why 512 is the widest that down_proj's 8704 admits.
  if (K % had) return R4D_NO(RAD_E_SHAPE);

  const int thr = R4D_HQAI8_THREADS ? R4D_HQAI8_THREADS : r4d_act_threads((int)M, (int)K);
  const size_t shmem = (size_t)K * sizeof(float) + (thr / 32) * sizeof(float);
  if (shmem > 65536) return R4D_NO(RAD_E_SCRATCH);   // K too large to stage in LDS
  const float rn = 1.f / sqrtf((float)had);
#define R4D_HQ_LAUNCH(H, T) hipLaunchKernelGGL((r4d_had_quant_act_i8_kernel<H, T>), dim3((int)M), \
    dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)r4d_p(x), \
    (signed char*)r4d_wp(q), (float*)r4d_wp(sc), (int)K, rn)
#define R4D_HQ_T(H) switch (thr) { \
    case 128: R4D_HQ_LAUNCH(H, 128); break; \
    case 256: R4D_HQ_LAUNCH(H, 256); break; \
    case 512: R4D_HQ_LAUNCH(H, 512); break; \
    case 1024: R4D_HQ_LAUNCH(H, 1024); break; \
    default: return R4D_NO(RAD_E_INVAL); }
  switch (had) {
    case 128: R4D_HQ_T(128); break;
    case 256: R4D_HQ_T(256); break;
    default:  R4D_HQ_T(512); break;
  }
#undef R4D_HQ_T
#undef R4D_HQ_LAUNCH
  return RAD_OK;
}
