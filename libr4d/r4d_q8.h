// r4d_q8.h -- the 8-bit activation codes the trunk GEMMs read, written by every quantising producer:
// E4M3 for the fp8a8 GEMMs, int8 for their int8 twins (gemm_i8a8_*). One scale per (row, 128
// columns) either way; only the largest code and the conversion differ.
//
// THE E4M3 FORM IS EXACTLY WHAT EACH PRODUCER WROTE INLINE BEFORE: the scale `amax * (1 / 448)`,
// the clamp to +-448 (v_cvt_pk_fp8_f32 answers an out-of-range input with NaN rather than
// saturating, and `amax * (1 / scale)` can land an ulp above 448), and two packed conversions. So a
// producer switched to this header writes the same bytes it did.
//
// THE INT8 FORM IS quant_act_i8g's (r4d_quant_act_fp8.hip): the scale `amax * (1 / 127)` and the
// value over it rounded to nearest-even, -127..127 -- so a producer that writes int8 itself writes
// exactly what the standalone quantiser would have written from the bf16 plane.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

template <bool I8>
__device__ __forceinline__ constexpr float q8_top() {
    return I8 ? 127.0f : 448.0f;
}

// Four values times `inv` (the reciprocal of the scale), as four codes in one dword, value 0 in
// the low byte.
template <bool I8>
__device__ __forceinline__ uint32_t q8_pack4(float a, float b, float c, float d, float inv) {
    if constexpr (I8) {
        const int ia = (int)rintf(fminf(fmaxf(a * inv, -127.0f), 127.0f));
        const int ib = (int)rintf(fminf(fmaxf(b * inv, -127.0f), 127.0f));
        const int ic = (int)rintf(fminf(fmaxf(c * inv, -127.0f), 127.0f));
        const int id = (int)rintf(fminf(fmaxf(d * inv, -127.0f), 127.0f));
        return ((uint32_t)ia & 0xFFu) | (((uint32_t)ib & 0xFFu) << 8) |
               (((uint32_t)ic & 0xFFu) << 16) | (((uint32_t)id & 0xFFu) << 24);
    } else {
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(a * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(b * inv, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(c * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(d * inv, -448.0f), 448.0f), pk, true);
        return (uint32_t)pk;
    }
}
