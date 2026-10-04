// r4d_fwht.h - a wave-local fast Walsh-Hadamard transform, in registers, with no LDS and no barrier.
//
// WHY IT IS IN REGISTERS.  Staging the row in LDS and running the butterfly there costs seven
// stages, each a read-modify-write of every element plus a __syncthreads() -- roughly 28 LDS
// operations and 7 barriers per element.  The transform's own ARITHMETIC is a fraction of a percent
// of a rotated quantiser; the LDS traffic and the barriers around it are what a butterfly staged in
// LDS actually costs, and they are several times the rest of the kernel.  None of that is the
// transform, so none of it is paid here.
//
// THE MAPPING.  Give one wave a `HAD`-wide block, VPL = HAD/32 contiguous values per lane, so lane l
// holds indices l*VPL .. l*VPL+VPL-1.  A butterfly at stride h pairs index i with i+h:
//
//   h < VPL    both indices are in the SAME lane      -- plain register arithmetic, no communication
//   h >= VPL   the partner is lane l + h/VPL          -- one lane exchange with mask h/VPL
//
// Since HAD/VPL is always 32, the cross-lane masks are exactly 1, 2, 4, 8, 16 whatever HAD is: five
// exchanges per register (r4d_xor_lane, DPP rather than LDS) and no barrier at any width.  The lane holding the LOW index of a pair keeps
// (v + partner) and the high one keeps (partner - v), which is the same butterfly written twice.
//
// The caller still needs one barrier before a whole-row reduction, but ONE, not seven.
#pragma once
#include <hip/hip_runtime.h>
#include "r4d_common.h"

// `hi ? (o - v) : (v + o)` written as `o + (hi ? -v : v)`: the sign flip is one XOR against a mask
// that is uniform over the stage, so the per-element select disappears.  Bit-identical -- IEEE
// negation is exact and addition commutes.
template <int VPL, int MASK>
__device__ __forceinline__ void r4d_fwht_xstage(float* v, int lane) {
    const unsigned int sm = (lane & MASK) ? 0x80000000u : 0u;
#pragma unroll
    for (int j = 0; j < VPL; j++) {
        const float o = r4d_xor_lane<MASK>(v[j]);
        v[j] = __int_as_float(__float_as_int(v[j]) ^ sm) + o;
    }
}

// In-register FWHT of a HAD-wide block held VPL = HAD/32 values per lane, lane-major.
// Unnormalised: the caller folds 1/sqrt(HAD) into whatever scale it already applies.
template <int HAD>
__device__ __forceinline__ void r4d_fwht_wave(float* v, int lane) {
    constexpr int VPL = HAD / 32;
    // in-lane stages: stride 1 .. VPL/2, entirely within this lane's registers
#pragma unroll
    for (int h = 1; h < VPL; h <<= 1) {
#pragma unroll
        for (int j = 0; j < VPL; j++) {
            if ((j & h) == 0) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    // cross-lane stages: stride VPL*mask for mask = 1, 2, 4, 8, 16
    r4d_fwht_xstage<VPL, 1>(v, lane);
    r4d_fwht_xstage<VPL, 2>(v, lane);
    r4d_fwht_xstage<VPL, 4>(v, lane);
    r4d_fwht_xstage<VPL, 8>(v, lane);
    r4d_fwht_xstage<VPL, 16>(v, lane);
}
