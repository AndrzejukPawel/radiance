// r4d_quant_act_i8.hip - r4d_quant_act_i8: per-row symmetric int8 quantisation of a bf16 activation,
// written in the order r4d_gemm_w4a8_nt_m64's A fragment wants.
//
// One row per workgroup: absmax over K, then a second pass writing the bytes.  K is a few thousand
// and M is a speculative-decode batch, so the whole thing is well under a microsecond -- one launch
// in place of the several an unfused quantiser would cost.
//
// THE BYTE ORDER.  Fragment element e of lane l is A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], so a
// lane's eight bytes are two dwords eight bytes apart in a row-major activation.  Writing each k
// step's sixteen bytes as [half 0's eight][half 1's eight] instead makes them contiguous and the
// GEMM reads a fragment with one global_load_b64.  Rows are untouched, so every lane of a wave
// stays inside the same sixteen rows it would have read anyway -- laying A out as two separate
// PLANES is the version that loses, see the note in the GEMM.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <cstdint>

// 0 = choose the workgroup width from M at launch (see r4d_act_threads in r4d_common.h).  This
// kernel is not on the W4A8 path -- the rotated sibling is faster -- but it has the same
// one-workgroup-per-row shape, so it gets the same knob rather than being the odd one out.
#ifndef R4D_QAI8_THREADS
#define R4D_QAI8_THREADS 0
#endif

template <int THR>
__global__ __launch_bounds__(THR) void r4d_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = r4d_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  // Byte j of a k step comes from lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) +
  // 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const float v = __bfloat162float(a[src]) * inv;
    const int t = __float2int_rn(v);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}



// ---------------------------------------------------------------------------------------------
// Same quantisation, plus the per-row per-group SUM of the int8 codes that an asymmetric weight
// grid needs: sum_k a_k (nibble_k - z_g) = sum_k a_k nibble_k - z_g * sum_k a_k, and the second
// factor depends only on the activation, so it is computed once here rather than N times in the
// GEMM.  Output layout is [K/group][Mpad] -- GROUP-MAJOR, Mpad = ceil(M/16)*16 -- because in the
// GEMM a lane's eight accumulator elements are eight CONTIGUOUS output rows at one group, and the
// padding is what lets the GEMM read them as two aligned v4f instead of eight scalars.  The caller
// allocates it zeroed; nothing writes the pad rows and the GEMM discards their outputs anyway.
//
// The fragment permutation is a shuffle INSIDE each 16-element step and a group is a whole number
// of steps, so it does not change which elements belong to which group and the sum can be taken on
// either side of it.
#define R4D_QAI8_MAXG 256

template <int THR>
__global__ __launch_bounds__(THR) void r4d_quant_act_i8_asum_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    float* __restrict__ Asum, int K, int group, int Mpad) {
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;
  const int G = K / group;

  __shared__ int gs[R4D_QAI8_MAXG];
  for (int g = threadIdx.x; g < G; g += THR) gs[g] = 0;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = r4d_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  for (int i = threadIdx.x; i < K; i += THR) {
    const int st = i >> 4, j = i & 15;
    const int src = (st << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(__bfloat162float(a[src]) * inv);
    const int c = t < -128 ? -128 : (t > 127 ? 127 : t);
    q[i] = (signed char)c;
    atomicAdd(&gs[src / group], c);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < G; g += THR)
    Asum[(size_t)g * Mpad + row] = (float)gs[g];
}

// The two launches, as the plugin ABI's RadLaunchFn: `(const RadArgs*, RadStream) -> int`.  A
// refused shape is a status code, not a throw; see libr4d/r4d_args.h for why the entry point takes
// the ABI's argument list directly and needs no adapter in front of it.
//
// THE THIRD OUTPUT SELECTS THE FORM.  Whether the caller wants the per-group row sums that an
// asymmetric weight grid subtracts is a property of the CALL, not of the model, so this is one
// row with a branch rather than two rows a constraint cannot tell apart.
enum { QA_X = 0, QA_Q, QA_S, QA_ASUM };

static int rad_quant_act_i8_body(const RadArgs* args, const R4dSink& sk) {
  int emit_rc = RAD_OK;
  const RadTensor* x  = r4d_opt(args, QA_X);
  const RadTensor* q  = r4d_opt(args, QA_Q);
  const RadTensor* sc = r4d_opt(args, QA_S);
  const RadTensor* as = r4d_opt(args, QA_ASUM);
  if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  const int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);          // a k step is sixteen bytes of one row
  if (r4d_numel(q) != M * K) return R4D_NO(RAD_E_SHAPE);

  const int thr = R4D_QAI8_THREADS ? R4D_QAI8_THREADS : r4d_act_threads((int)M, (int)K);

  if (!as) {
#define R4D_QAI8_LAUNCH(T) emit_rc = R4D_EMIT(sk, (r4d_quant_act_i8_kernel<T>), dim3((int)M),    \
    dim3(T), 0, (const __hip_bfloat16*)r4d_p(x), (signed char*)r4d_wp(q),                         \
    (float*)r4d_wp(sc), (int)K)
    switch (thr) {
      case 128:  R4D_QAI8_LAUNCH(128);  break;
      case 256:  R4D_QAI8_LAUNCH(256);  break;
      case 512:  R4D_QAI8_LAUNCH(512);  break;
      case 1024: R4D_QAI8_LAUNCH(1024); break;
      default:   return R4D_NO(RAD_E_INVAL);       // r4d_act_threads answers only these four
    }
#undef R4D_QAI8_LAUNCH
    return emit_rc;
  }

  if (!r4d_contig(as)) return R4D_NO(RAD_E_STRIDE);
  const long long group = r4d_geti_or(args, "group", R4D_W4_GROUP);
  if (group < 1 || K % group) return R4D_NO(RAD_E_SHAPE);
  if (K / group > R4D_QAI8_MAXG) return R4D_NO(RAD_E_SHAPE);   // the LDS accumulator's bound
  // asum is [K/group][Mpad] with Mpad = ceil(M/16)*16 -- GROUP-MAJOR, and padded, because the
  // asymmetric GEMM reads it a whole n-tile at a time.  Sizing it from M alone overruns.
  const long long mpad = ((M + 15) / 16) * 16;
  if (r4d_numel(as) < (K / group) * mpad) return R4D_NO(RAD_E_SCRATCH);

#define R4D_QAI8A_LAUNCH(T) emit_rc = R4D_EMIT(sk, (r4d_quant_act_i8_asum_kernel<T>),            \
    dim3((int)M), dim3(T), 0, (const __hip_bfloat16*)r4d_p(x), (signed char*)r4d_wp(q),          \
    (float*)r4d_wp(sc), (float*)r4d_wp(as), (int)K, (int)group, (int)mpad)
  switch (thr) {
    case 128:  R4D_QAI8A_LAUNCH(128);  break;
    case 256:  R4D_QAI8A_LAUNCH(256);  break;
    case 512:  R4D_QAI8A_LAUNCH(512);  break;
    case 1024: R4D_QAI8A_LAUNCH(1024); break;
    default:   return R4D_NO(RAD_E_INVAL);
  }
#undef R4D_QAI8A_LAUNCH
  return emit_rc;
}
R4D_ENTRY(r4d_rad_quant_act_i8, rad_quant_act_i8_body)
