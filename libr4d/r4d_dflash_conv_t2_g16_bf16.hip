// DFlash2 grouped dynamic depthwise convolution, fused.
//
// The device math lives in r4d_dflash_conv_body.h, deliberately free of HIP headers so the same
// source compiles standalone to gfx1201 assembly with a stock clang for ISA review.
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"

#define DFC_THREADS 128
#define DFC_VEC     8      // bf16 lanes per thread: one dwordx4 in and one out

template <int TAPS, int GROUP>
__global__ __launch_bounds__(DFC_THREADS) void r4d_dflash_conv_kernel(
    const r4d_u16* __restrict__ x, const r4d_u16* __restrict__ delta,
    const r4d_u16* __restrict__ base, r4d_u16* __restrict__ out,
    int H, int dpitch, int NG, int vpr, int blockmask)
{
  const int hv = blockIdx.x * DFC_THREADS + threadIdx.x;
  r4d_dflash_conv_body<TAPS, GROUP>(x, delta, base, out, H, dpitch, NG, vpr,
                                    blockmask, blockIdx.y, hv);
}

// x     [T, H]        bf16, row pitch H
// delta [T, taps, NG] bf16, row pitch dpitch -- this is a SLICE of the [T, 2, taps, NG] kernel
//                     projection, so dpitch is 2*taps*NG, not taps*NG. Passing the packed pitch
//                     silently reads the other side's coefficients; it is not a shape error.
// base  [taps, H]     bf16, contiguous
// out   [T, H]        bf16. Must not alias x -- tap 1 of row t reads row t-1 of the INPUT.
extern "C" int r4d_dflash_conv_t2_g16_bf16(
    const void* x, const void* delta, const void* base, void* out,
    int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream)
{
  if (taps != 2 || group != 16) return -1;
  if (H % (group * DFC_VEC) || NG != H / group) return -1;
  if (block_size < 1 || (block_size & (block_size - 1))) return -1;  // body masks: pow2 only
  if (T <= 0) return 0;

  const int vpr = H / DFC_VEC;
  dim3 grid((vpr + DFC_THREADS - 1) / DFC_THREADS, T), blk(DFC_THREADS);
  hipLaunchKernelGGL((r4d_dflash_conv_kernel<2, 16>), grid, blk, 0, (hipStream_t)stream,
                     (const r4d_u16*)x, (const r4d_u16*)delta, (const r4d_u16*)base,
                     (r4d_u16*)out, H, dpitch, NG, vpr, block_size - 1);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
//
// The shared operand helpers are r4d_gdn_args.h. See libr4d/r4d_args.h for why there is no
// adapter directory between the core and these launches.
/* ------------------------------------------------------------------ dflash conv */

enum { DC_X = 0, DC_DELTA, DC_BASE, DC_OUT };

extern "C" int r4d_rad_dflash_conv(const RadArgs* a, RadStream s) {
    const RadTensor* x  = r4d_opt(a, DC_X);
    const RadTensor* d  = r4d_opt(a, DC_DELTA);
    const RadTensor* bs = r4d_opt(a, DC_BASE);
    const RadTensor* o  = r4d_opt(a, DC_OUT);
    if (!x || !d || !bs || !o) return R4D_NO(RAD_E_INVAL);
    if (x->rank != 2 || o->rank != 2 || d->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(x) || !r4d_contig(o) || !r4d_contig(bs)) return R4D_NO(RAD_E_STRIDE);

    /* "out must not alias x" is in the entry point's own contract and nothing downstream checks
     * it -- the kernel reads x while writing out, so an alias is silent corruption. */
    if (x->data == o->data) return R4D_NO(RAD_E_INVAL);

    const long long T = x->shape[0];
    const long long H = x->shape[1];
    if (o->shape[0] != T || o->shape[1] != H) return R4D_NO(RAD_E_SHAPE);

    /* delta is a SLICE of the [T, 2, taps, NG] kernel projection, so its row pitch is 2*taps*NG
     * and not the slice's own width. Reading it off stride[0] is what makes the slice legal. */
    const long long dpitch = r4d_row_pitch(d);
    if (dpitch < 0) return R4D_NO(RAD_E_STRIDE);

    const int taps  = (int)r4d_geti_or(a, "taps", 2);
    const int group = (int)r4d_geti_or(a, "group_size", 16);
    const int NG    = (int)r4d_geti_or(a, "NG", 0);
    const int bsz   = (int)r4d_geti_or(a, "block_size", 0);
    if (NG <= 0 || bsz <= 0) return R4D_NO(RAD_E_INVAL);
    /* block_size is masked with block_size-1 in the kernel body, so only a power of two is
     * admissible. That is a property of the VALUE and not a bound a constraint can express, which
     * is why libr4d's registry leaves it out of the predicates and the entry point rejects it --
     * except that libr4d's entry point does not, so this shim does. */
    if (bsz & (bsz - 1)) return R4D_NO(RAD_E_INVAL);
    if (dpitch != 2LL * taps * NG) return R4D_NO(RAD_E_STRIDE);

    return r4d_dflash_conv_t2_g16_bf16(x->data, d->data, bs->data, o->data,
                                       (int)T, (int)H, (int)dpitch, NG, taps, group, bsz, s);
}

