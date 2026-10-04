// H += X^T X, in f32, over an E4M3 activation with a per-(row, group) scale, or over a plain bf16
// one.
//
// WHAT IT IS FOR. GPTQ needs a Gram matrix of the layer's real input -- the second moment
// `H = sum_t x_t x_t^T` over a calibration corpus -- and this is the device half of producing one;
// `rad_quantize` alone cannot, because it has no calibration harness to run the activations
// through. The 4-bit expert format is the caller: see docs/MOE-W4.md.
//
// IT IS NOT A GEMM AND CANNOT BE ONE. `gemm_nt` contracts the LAST axis of both operands; this
// contracts the FIRST axis of both. Declaring it as a GEMM would need X transposed in memory,
// which is a 26 MB scratch buffer and a transpose kernel for an op whose whole body is an FMA
// loop. So it is its own row.
//
// WHY THE INPUT IS E4M3 AND NOT bf16. The two taps are the MoE block's own quantiser outputs --
// `had_quant_act_fp8`'s (q, scale) for gate_up, and `gated_had_quant_fp8`'s for down. Those are
// literally the bytes the expert GEMM contracts, so a Hessian built from them is a Hessian of the
// arithmetic that actually runs rather than of the bf16 that preceded it. It is also the only
// form the rotated path HAS: nothing stores the rotated activation as bf16.
//
// AND A bf16 FORM BESIDE IT, for the same reason pointing the other way. A model served at bf16
// contracts its experts against the plain bf16 activation, so that IS the operand there, and the
// bf16 instantiation reads it with no scale. Everything past the staging is shared.
//
// THE TOP-K WEIGHTING IS FREE AND THIS IS WHY THE gate_up TAP IS THE UNSORTED ACTIVATION.
// The expert GEMM reads token rows through a `sorted` index list, so its true input distribution
// is every token repeated top_k times. Every token is routed to EXACTLY top_k experts, so that
// distribution is the token distribution scaled by top_k -- H_sorted = top_k * H_token, exactly --
// and GPTQ is invariant to a positive scale on H (the damping is a fraction of the trace and the
// LDLQ update uses ratios). So the tap takes the M token rows once and not the M*top_k gathered
// ones, at a top_k-fold saving, and loses nothing.
//
// ACCUMULATION IS f32 AND DELIBERATELY NOT f64. A block keeps its whole tile in registers across
// the M loop and touches global once, so the error over a 2048-row launch is a register tree, not
// a 2048-deep serial sum. Across launches the global f32 accumulator sees ~one add per launch.
// The consumer damps H by a percent of its trace before inverting it, which is four orders of
// magnitude above anything this rounds away; f64 would double 1.3 GB of state to buy nothing.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_fp8_wmma.h"

namespace {

// A 64-wide C tile, 256 threads, 4x4 accumulators a thread: 16x16 threads x 16 = 4096 = 64*64.
// 64 is chosen against the SCALE GRID as much as against the register file -- a 64-column tile
// whose start is a multiple of 64 lies entirely inside one 128-wide scale group, so a row's scale
// is ONE load a tile rather than one a column. The launcher enforces that by refusing a group the
// tile does not divide.
constexpr int kTile    = 64;
constexpr int kThreads = 256;
constexpr int kMB      = 32;    // M rows staged a pass; 2 * 32 * 64 * 4 = 16 KiB of LDS
constexpr int kTT      = 4;

/* BF16 reads a plain bf16 activation and no scale: a staging slot is then four elements in two
 * dwords, and everything after the staging -- the tile, the accumulation order, the store -- is the
 * fp8 form's. The fp8 instantiation carries none of the bf16 staging. */
template <bool BF16>
__global__ __launch_bounds__(kThreads) void gram_accum_kernel(
    const void* __restrict__ qv, const float* __restrict__ sc, float* __restrict__ h,
    uint32_t M, uint32_t K, uint32_t q_ld, uint32_t s_ld, uint32_t h_ld, uint32_t grp) {

    __shared__ float sa[kMB][kTile];
    __shared__ float sb[kMB][kTile];

    const uint32_t k1_0 = blockIdx.x * (uint32_t)kTile;
    const uint32_t k2_0 = blockIdx.y * (uint32_t)kTile;
    const uint32_t tx = threadIdx.x & 15u, ty = threadIdx.x >> 4;   // 16 x 16
    // Both tiles sit inside one scale group each -- see kTile's note. bf16 has no groups.
    const uint32_t g1 = BF16 ? 0u : k1_0 / grp, g2 = BF16 ? 0u : k2_0 / grp;

    float acc[kTT][kTT];
#pragma unroll
    for (int i = 0; i < kTT; ++i)
#pragma unroll
        for (int j = 0; j < kTT; ++j) acc[i][j] = 0.0f;

    for (uint32_t m0 = 0; m0 < M; m0 += (uint32_t)kMB) {
        __syncthreads();
        // 32 rows x 16 dwords = 512 staging slots for 256 threads, twice, for each of the two
        // tiles. A dword is four E4M3 codes and decodes in two v_cvt_pk_f32_fp8.
        for (uint32_t idx = threadIdx.x; idx < (uint32_t)(kMB * kTile / 4);
             idx += (uint32_t)kThreads) {
            const uint32_t mi = idx >> 4, cq = (idx & 15u) * 4u;
            const uint32_t m = m0 + mi;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            if (m < M) {
                const size_t qrow = (size_t)m * q_ld;
                const uint32_t c1 = k1_0 + cq, c2 = k2_0 + cq;
                if constexpr (BF16) {
                    // K is a multiple of four, so a slot never runs past the row.
                    if (c1 < K) {
                        const uint2 w = *(const uint2*)((const uint16_t*)qv + qrow + c1);
                        a0 = bf16_to_f32((uint16_t)w.x); a1 = bf16_to_f32((uint16_t)(w.x >> 16));
                        a2 = bf16_to_f32((uint16_t)w.y); a3 = bf16_to_f32((uint16_t)(w.y >> 16));
                    }
                    if (c2 < K) {
                        const uint2 w = *(const uint2*)((const uint16_t*)qv + qrow + c2);
                        b0 = bf16_to_f32((uint16_t)w.x); b1 = bf16_to_f32((uint16_t)(w.x >> 16));
                        b2 = bf16_to_f32((uint16_t)w.y); b3 = bf16_to_f32((uint16_t)(w.y >> 16));
                    }
                } else {
                    if (c1 < K) {
                        // The row is owned out to K rounded up to four bytes -- quant_act_fp8's
                        // own contract -- so a dword load at c1 < K is inside the caller's
                        // allocation even when the last group is ragged.
                        const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c1);
                        const float s = sc[(size_t)m * s_ld + g1];
                        const v2f lo = r4d_fp8::e4m3_pk<false>(w), hi = r4d_fp8::e4m3_pk<true>(w);
                        a0 = lo.x * s; a1 = lo.y * s; a2 = hi.x * s; a3 = hi.y * s;
                    }
                    if (c2 < K) {
                        const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c2);
                        const float s = sc[(size_t)m * s_ld + g2];
                        const v2f lo = r4d_fp8::e4m3_pk<false>(w), hi = r4d_fp8::e4m3_pk<true>(w);
                        b0 = lo.x * s; b1 = lo.y * s; b2 = hi.x * s; b3 = hi.y * s;
                    }
                }
            }
            // Columns past K stage as zeros and contribute nothing; the store masks them anyway.
            sa[mi][cq + 0] = a0; sa[mi][cq + 1] = a1; sa[mi][cq + 2] = a2; sa[mi][cq + 3] = a3;
            sb[mi][cq + 0] = b0; sb[mi][cq + 1] = b1; sb[mi][cq + 2] = b2; sb[mi][cq + 3] = b3;
        }
        __syncthreads();

        const uint32_t rows = M - m0 < (uint32_t)kMB ? M - m0 : (uint32_t)kMB;
        for (uint32_t r = 0; r < rows; ++r) {
            float u[kTT], v[kTT];
#pragma unroll
            for (int i = 0; i < kTT; ++i) u[i] = sa[r][ty * kTT + i];
#pragma unroll
            for (int j = 0; j < kTT; ++j) v[j] = sb[r][tx * kTT + j];
#pragma unroll
            for (int i = 0; i < kTT; ++i)
#pragma unroll
                for (int j = 0; j < kTT; ++j) acc[i][j] = fmaf(u[i], v[j], acc[i][j]);
        }
    }

    // One block owns one C tile, so this read-modify-write races with nothing. It is an ACCUMULATE
    // and not a store: the caller zeroes H once and then drives the whole corpus through it.
#pragma unroll
    for (int i = 0; i < kTT; ++i) {
        const uint32_t k1 = k1_0 + ty * kTT + (uint32_t)i;
        if (k1 >= K) continue;
#pragma unroll
        for (int j = 0; j < kTT; ++j) {
            const uint32_t k2 = k2_0 + tx * kTT + (uint32_t)j;
            if (k2 < K) h[(size_t)k1 * h_ld + k2] += acc[i][j];
        }
    }
}

}  // namespace

extern "C" int r4d_gram_accum_fp8_tile(void) { return kTile; }

//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), owned out to a multiple of 4
//   s  [M][ceil(K/grp)]      f32,  `s_ld` elements a row (0 = ceil(K/grp))
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K) -- READ AND WRITTEN
extern "C" int r4d_gram_accum_fp8(long q, long s, long h, int M, int K, int grp,
                                  int q_ld, int s_ld, int h_ld, long stream) {
    if (!q || !s || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is one dword of four codes
    if (grp <= 0 || (grp % kTile)) return -5;   // a tile must lie inside one scale group
    if (!q_ld) q_ld = K;
    if (q_ld % 4) return -6;                    // ...and each row's dwords must stay aligned
    if (!s_ld) s_ld = (K + grp - 1) / grp;
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kTile - 1) / kTile);
    hipLaunchKernelGGL(gram_accum_kernel<false>, dim3(nt, nt), dim3(kThreads), 0,
                       (hipStream_t)stream, (const void*)q, (const float*)s, (float*)h,
                       (uint32_t)M, (uint32_t)K, (uint32_t)q_ld, (uint32_t)s_ld, (uint32_t)h_ld,
                       (uint32_t)grp);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

//   a  [M][K]                bf16, `a_ld` elements a row (0 = K)
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K) -- READ AND WRITTEN
extern "C" int r4d_gram_accum_bf16(long a, long h, int M, int K, int a_ld, int h_ld, long stream) {
    if (!a || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is four elements of one row
    if (!a_ld) a_ld = K;
    if (a_ld % 4 || (a % 8)) return -6;         // ...loaded as two aligned dwords
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kTile - 1) / kTile);
    hipLaunchKernelGGL(gram_accum_kernel<true>, dim3(nt, nt), dim3(kThreads), 0,
                       (hipStream_t)stream, (const void*)a, (const float*)nullptr, (float*)h,
                       (uint32_t)M, (uint32_t)K, (uint32_t)a_ld, 0u, (uint32_t)h_ld, 1u);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry point

enum { GA_Q = 0, GA_S = 1, GA_H = 2 };

extern "C" int r4d_rad_gram_accum(const RadArgs* a, RadStream s) {
    const RadTensor* q = r4d_opt(a, GA_Q);
    const RadTensor* sc = r4d_opt(a, GA_S);
    const RadTensor* h = r4d_opt(a, GA_H);
    if (!q || !sc || !h) return R4D_NO(RAD_E_INVAL);
    /* ROWS ARE DIM 0 AND EVERYTHING AFTER IT IS THE ROW, the same reading r4d_rad_quant_act_fp8
     * takes: a rank-3 activation is one row a token and its K is the product of the rest. `sc`
     * and `h` stay rank 2 because this op's meaning fixes them. */
    if (q->rank < 2 || sc->rank != 2 || h->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = q->shape[0];
    if (M < 1 || r4d_numel(q) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(q) / M;
    const long long grp = r4d_geti_or(a, "group", R4D_FP8_BLOCK);
    const long long nkb = (K + grp - 1) / grp;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (h->shape[0] != K || h->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    const long long qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc), hp = r4d_row_pitch(h);
    if (qp < 0 || sp < 0 || hp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = r4d_gram_accum_fp8(r4d_addr(q), r4d_addr(sc), r4d_addr(h), (int)M, (int)K,
                                     (int)grp, (int)qp, (int)sp, (int)hp, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gram_accum_bf16(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, GA_Q);
    const RadTensor* h = r4d_opt(a, GA_H);
    if (!x || !h) return R4D_NO(RAD_E_INVAL);
    /* No scale: the bf16 form reads the activation as it stands. One handed over anyway is a
     * caller that believes the input is quantised, and it is refused rather than ignored. */
    if (r4d_opt(a, GA_S)) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || h->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (x->rank < 2 || h->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0];
    if (M < 1 || r4d_numel(x) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(x) / M;
    if (h->shape[0] != K || h->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), hp = r4d_row_pitch(h);
    if (xp < 0 || hp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = r4d_gram_accum_bf16(r4d_addr(x), r4d_addr(h), (int)M, (int)K, (int)xp, (int)hp,
                                      (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : R4D_NO(RAD_E_SHAPE);
}
