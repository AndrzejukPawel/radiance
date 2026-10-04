// r4d_gdn_conv_tap.h -- the arithmetic of one gated-delta-net convolution output, shared by the
// standalone conv kernels (r4d_gdn_conv_w4_h128_bf16.hip) and the recurrent update that folds the
// decode convolution into its prologue (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip).
// One definition is what makes the fold byte-identical to the standalone pair: the activation's
// reciprocal form below changes the last bit of a bf16 output, so two copies that drift apart
// would disagree on a real fraction of channels.
#pragma once

#define R4D_GDN_CONV_W 4        // conv width; the state holds R4D_GDN_CONV_W - 1 taps of history

#ifndef CU_RCP
#define CU_RCP 1
#endif

// The divide here is IEEE-exact and the result is rounded to bf16 on the very next instruction, so
// the exactness buys nothing: v_rcp_f32 is 1 ULP (~1e-7 relative) against bf16 s 2^-8 quantum.  The
// exact form costs a v_div_scale pair, a v_rcp, a Newton chain, v_div_fmas and v_div_fixup -- about
// ten instructions and a long dependent chain -- where this costs two.  The conv update runs four
// waves on five workgroups, so its instruction count IS its time.
__device__ __forceinline__ float cp_silu(float z) {
#if CU_RCP
  return z * __builtin_amdgcn_rcpf(1.0f + __expf(-z));
#else
  return z / (1.0f + __expf(-z));
#endif
}
