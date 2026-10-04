// r4d_gemm_bf16_nt_m16.hip -- THE ROW THAT FORWARDS. There is no kernel in this file.
//
// This row owns the M <= 16 band -- the MoE router gate's band -- and serves it with the WMMA
// kernel behind the M <= 64 row, which is faster than a scalar split-K kernel at every point of it.
//
// THE FORWARD IS NOT BIT-NEUTRAL against a scalar kernel of the same band. Both formulations are
// deterministic fp32 reductions over the same split, but they are different reduction TREES, so the
// last bit differs. At N = 1280 and N = 256, which are the DFlash2 drafter's shapes, a bit change
// is not a text change; at N = 48, the trunk's GDN a|b projection and 48 calls a step, the emitted
// text does move where M <= 16. A reduction order is not a quality bar, so that is accepted.
//
// THE ROW STAYS EVEN THOUGH THERE IS NO KERNEL. Deleting the registry row would leave the M <= 16
// band to whatever else claims it, and a band that resolves to a kernel which cannot serve it is
// invisible until a request lands in it. One forwarding entry is cheap and the band stays
// explicitly owned.
#include "r4d.h"
#include "r4d_gemm_args.h"

extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs*, RadStream);

extern "C" int r4d_rad_gemm_bf16_m16(const RadArgs* a, RadStream s) {
    return r4d_rad_gemm_bf16_m64(a, s);
}
