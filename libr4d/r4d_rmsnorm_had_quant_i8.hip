// r4d_rmsnorm_had_quant_i8.hip - r4d_rmsnorm_had_quant_i8: the residual add, the RMS norm, the block
// Hadamard and the int8 activation quantiser, in one pass over the row.
//
// WHY FUSE.  W4A8 needs a rotated int8 activation in front of every quantised linear, and there are
// four such inputs per layer -- 256 per decode step at 64 layers.  As separate kernels that is
// 0.4-1.0 ms of pure launch on a step the whole project is trying to get to ~19 ms, and it is launch
// count rather than bandwidth: the arithmetic is ~14 ops per element and the traffic is 50 MB.
// Production pays none of it today because vLLM already fuses its fp8 quantiser into the norm, so a
// standalone rotate-and-quantise would be a REGRESSION against the thing it is trying to beat.
// Fused, the rotation and the quantiser both ride on a pass that already has the row in registers.
//
// WHAT IT WRITES.  Two consumers, not one.  The int8 output feeds r4d_gemm_w4a8_nt_m64 in the byte
// order its A fragment reads.  The bf16 output is optional and exists because a GDN block's
// in_proj_a and in_proj_b are in modules_to_not_convert, stay bf16, and read the SAME post-norm
// tensor -- pass a null pointer on the layers that have no such consumer and the write is skipped.
// The bf16 written is the UNROTATED norm output, so those two projections keep their original
// weights; rotating them offline instead would round W_a into bf16 for no gain.
//
// `wadd` IS ADDED TO EVERY GAIN, AND IT IS NOT A CONVENIENCE.  Qwen3.5/Qwen3-Next norms are
// GemmaRMSNorm, which is `x_hat * (1 + w)` with a ZERO-CENTRED gamma, and vLLM forms that gain as
// `weight.float() + 1.0` -- in fp32.  Rounding `1 + w` to bf16 first throws away most of w: near 1
// a bf16 ULP is 2^-7, so a gain that should be 1.0031 becomes 1.0000, a 0.3% per-channel error that
// is systematic rather than random and is far larger than anything the int8 quantiser does.  Adding
// it here, after the bf16 load and inside the fp32 arithmetic, is bit-identical to what vLLM
// computes and costs one v_add_f32 per element.  Pass 0.0f for a plain (Llama-style) RMS norm.
//
// The residual is updated in place exactly as vLLM's fused_add_rms_norm does: residual += x first,
// then the norm is taken of the sum.  Pass a null residual for the non-additive form.
//
// THE TRANSFORM DOES NOT HAVE TO WAIT FOR THE NORM.  A sum of squares closes only after the whole
// row is seen, which looks like it forces the rotation behind a barrier.  But the FWHT is LINEAR
// and `inv_rms` is a per-row SCALAR, so FWHT(x * Wg * inv_rms) = inv_rms * FWHT(x * Wg): the
// transform can run before the norm is known.  And since the int8 codes are
// round(value * 127 / absmax(value)), a positive scalar cancels out of them entirely -- so inv_rms
// touches nothing but S[row], exactly like the 1/sqrt(had) it already carries.  The sum of squares
// and the row max then reduce in the SAME barrier.
//
// THE ROW NEVER GOES THROUGH LDS TWICE.  The reduction still splits the kernel in two, and staging
// the row in LDS across it costs an LDS write, read and rewrite per element -- enough to put this
// kernel at 3.5x the plain quantiser at M=4096, where the sibling `gated_had_quant_i8`, which has
// no such pass, is 1.4x.  The wave instead KEEPS its blocks in registers across the reduction: ~34
// floats per lane at the widest per-rank K, and LDS is touched once, for the permuted store.
//
// Re-reading the residual from memory in pass 2 is NOT an alternative: it is stored as bf16, and
// normalising from a bf16-rounded sum produces thousands of +/-1 code errors and a 1e-3 scale
// error against 5e-7.  Storing bf16 is correct -- it is what the next layer reads -- normalising
// from it is not.
//
// NORMALISATION.  As in r4d_had_quant_act_i8, the butterfly computes the UNNORMALISED transform and
// the row scale carries the 1/sqrt(had), so no pass is spent scaling by a constant.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

// 0 = choose the workgroup width from M at launch (see r4d_act_threads in r4d_common.h); a
// non-zero build-time value pins one width instead.
#ifndef R4D_RHQ_THREADS
#define R4D_RHQ_THREADS 0
#endif

// Register staging bound: blocks-per-wave * VPL floats per lane.  ~K/256 at NW=8, so 72 covers
// every per-rank K up to 18432 and every compiled width.
#define R4D_RHQ_MAXVALS 72

template <int HAD, int THR, bool WOUT>
__global__ __launch_bounds__(THR) void r4d_rmsnorm_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ X, __hip_bfloat16* __restrict__ RES,
    const __hip_bfloat16* __restrict__ Wg, __hip_bfloat16* __restrict__ OUTB,
    signed char* __restrict__ Q, float* __restrict__ S, int K, float eps, float wadd,
    float rnorm) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  constexpr int MAXB = R4D_RHQ_MAXVALS / VPL;
  extern __shared__ float lds[];
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* x = X + (size_t)row * K;
  __hip_bfloat16* res = RES ? RES + (size_t)row * K : nullptr;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int nblk = K / HAD;

  // Kept only when a bf16 consumer wants the unrotated norm output, since that one DOES need
  // inv_rms and so cannot be written before the reduction.
  float u[WOUT ? MAXB : 1][VPL];

  float ss = 0.f, m = 0.f;
  int nb = 0;
  for (int blk = wave; blk < nblk; blk += NW, nb++) {
    const int base = blk * HAD + lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      float t = __bfloat162float(x[base + j]);
      if (res) {
        t += __bfloat162float(res[base + j]);
        res[base + j] = __float2bfloat16(t);
      }
      ss += t * t;
      v[j] = t * (__bfloat162float(Wg[base + j]) + wadd);
      if (WOUT) u[nb][j] = v[j];
    }
    r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      lds[base + j] = v[j];
      m = fmaxf(m, fabsf(v[j]));
    }
  }

  // one barrier for both reductions
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
  if (threadIdx.x == 0) S[row] = m * inv_rms * rnorm / 127.f;

  if (WOUT) {
    nb = 0;
    for (int blk = wave; blk < nblk; blk += NW, nb++) {
      const int base = blk * HAD + lane * VPL;
#pragma unroll
      for (int j = 0; j < VPL; j++)
        OUTB[(size_t)row * K + base + j] = __float2bfloat16(u[nb][j] * inv_rms);
    }
  }

  signed char* q = Q + (size_t)row * K;
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int sr = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[sr] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The launch, as the plugin ABI's RadLaunchFn. See the note at the head of r4d_quant_act_i8.hip.
//
// THE RESIDUAL AND THE BF16 ECHO ARE OPTIONAL OPERANDS, and each one changes what this computes:
// with a residual it is `add` then `rmsnorm` then the quantiser, without it just the last two.
// That is why the fusion declaration (RadFuseFn) has to be handed the present-operand mask -- the
// CHAIN LENGTH depends on an operand rather than on a parameter.
enum { RN_X = 0, RN_RESID, RN_W, RN_Q, RN_S, RN_OUTBF16 };

extern "C" int r4d_rad_rmsnorm_had_quant_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* x  = r4d_opt(args, RN_X);
  const RadTensor* rs = r4d_opt(args, RN_RESID);
  const RadTensor* w  = r4d_opt(args, RN_W);
  const RadTensor* q  = r4d_opt(args, RN_Q);
  const RadTensor* sc = r4d_opt(args, RN_S);
  const RadTensor* ob = r4d_opt(args, RN_OUTBF16);
  if (!x || !w || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc) || !r4d_contig(w))
    return R4D_NO(RAD_E_STRIDE);
  if (rs && !r4d_contig(rs)) return R4D_NO(RAD_E_STRIDE);
  if (ob && !r4d_contig(ob)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(w) != K) return R4D_NO(RAD_E_SHAPE);

  const long long had = r4d_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return R4D_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard boundary, and this is the check that
  // catches a width the split cannot carry: down_proj is 8704 per rank, so 1024 is unavailable.
  if (K % had) return R4D_NO(RAD_E_SHAPE);

  const float eps = (float)r4d_getf(args, "eps", 1e-6);
  // GemmaRMSNorm stores a zero-centred gamma and uses `1 + w`; forming that in bf16 first costs
  // 0.3% per channel, so it is added in fp32 after the load. 0 is a plain RMS norm.
  const float wadd = (float)r4d_getf(args, "wadd", 0.0);

  const int thr = R4D_RHQ_THREADS ? R4D_RHQ_THREADS : r4d_act_threads((int)M, (int)K);
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_RHQ_MAXVALS)
    return R4D_NO(RAD_E_SHAPE);                    // K too large for the register staging here
  const size_t shmem = (size_t)K * sizeof(float) + 2 * (thr / 32) * sizeof(float);
  if (shmem > 65536) return R4D_NO(RAD_E_SCRATCH); // K too large to stage in LDS
  const float rn = 1.f / sqrtf((float)had);
#define R4D_RHQ_LAUNCH(H, T, W) hipLaunchKernelGGL((r4d_rmsnorm_had_quant_i8_kernel<H, T, W>), \
    dim3((int)M), dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)r4d_p(x), \
    (__hip_bfloat16*)r4d_wp(rs), (const __hip_bfloat16*)r4d_p(w), (__hip_bfloat16*)r4d_wp(ob), \
    (signed char*)r4d_wp(q), (float*)r4d_wp(sc), (int)K, eps, wadd, rn)
#define R4D_RHQ_T(H, W) switch (thr) { \
    case 128: R4D_RHQ_LAUNCH(H, 128, W); break; \
    case 256: R4D_RHQ_LAUNCH(H, 256, W); break; \
    case 512: R4D_RHQ_LAUNCH(H, 512, W); break; \
    case 1024: R4D_RHQ_LAUNCH(H, 1024, W); break; \
    default: return R4D_NO(RAD_E_INVAL); }
#define R4D_RHQ_W(W) switch (had) { \
    case 128: R4D_RHQ_T(128, W); break; \
    case 256: R4D_RHQ_T(256, W); break; \
    default:  R4D_RHQ_T(512, W); break; }
  if (ob) { R4D_RHQ_W(true); } else { R4D_RHQ_W(false); }
#undef R4D_RHQ_W
#undef R4D_RHQ_T
#undef R4D_RHQ_LAUNCH
  return RAD_OK;
}
