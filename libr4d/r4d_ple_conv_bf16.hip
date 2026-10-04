// r4d_ple_conv_bf16.hip -- Per-Layer Embedding's dilated depthwise convolution, its silu, and the
// residual add that finishes the layer.
//
// ============================== WHAT IT COMPUTES ==============================
//
//   y[t][c] = resid[t][c] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation][c] )
//
// libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. THE TAPS ARE `dilation` APART, so
// at this model's width 4 and dilation 3 they sit 9, 6, 3 and 0 timesteps back and the history a
// sequence carries is (width-1)*dilation = NINE, not three. Every off-by-one in a convolution cache
// is a model that generates fluently from a slightly wrong context, so the indexing here mirrors
// ref's line for line: state slot `i` holds the value at relative position `i - hist`, a read at
// `src < 0` becomes `cs[slot][c][off + src + hist]`, and `has_init` 0 means that window is ZEROS
// rather than whatever the slot last held.
//
// ============================== ONE THREAD A CHANNEL ==============================
//
// A depthwise convolution never mixes channels, so the only axis that has to stay together is time.
// A thread owns one channel and walks a tile of tokens; consecutive threads own consecutive
// channels, so a wave's load is one contiguous 512-byte slice of a token row on both sides, and the
// time axis -- the one with the dependence -- never crosses a thread.
//
// The grid is (channel blocks, token tiles, sequences). Token tiles are counted from the batch's
// TOTAL token count, which is an upper bound on any one sequence's: a short sequence in a mixed
// batch launches a few blocks that find `t0 >= slen` and return, and the waste is bounded by the
// sequence count. Reading a per-sequence tile count would mean reading `cu` on the host.
//
// ============================== FOUR TAP READS AND NOT A SHIFT REGISTER ==============================
//
// The taps move by ONE timestep per output while they are spaced `dilation` apart, so a thread that
// kept its window in registers would need `hist + 1` of them and would index that array by a value
// the compiler cannot see -- and a register array under a runtime index is not a register read, it
// is a select tree. The GDN conv update's CU_STATIC note records exactly that
// (r4d_gdn_conv_w4_h128_bf16.hip: a third of the kernel becomes s_cselect/v_cndmask choosing which
// register to read). A compile-time window would mean templating on (width, dilation) and a
// specialisation per model.
//
// So the taps are read where they lie. They are re-reads of the same 512-byte slice a handful of
// iterations apart -- a tile's whole working set is (BT + hist) * 512 bytes, 21 KB at BT 32, which
// is L0-resident -- and this op is bandwidth-bound on the three arrays it must touch anyway. If a
// profile ever says otherwise, the templated window is the change to make, not a runtime ring.
//
// ============================== THE WINDOW WRITE-BACK IS IN THE SAME LAUNCH ==============================
//
// The obvious reading says it cannot be: the write-back overwrites the window other blocks read,
// and a kernel boundary is the only grid-wide barrier this tree has. What makes it legal here is
// WHICH BLOCK READS THE STATE. History is reached only where `t - hist + j*dilation < 0`, so only
// tokens below `hist` -- nine of them -- read it at all, and they are all inside the 32-token TILE
// ZERO. One block reads state[slot][c], one thread inside it owns the whole channel, and its own
// reads are ordered before its own writes by program order. No barrier and no second dispatch.
//
// That matters because at decode this op is nothing but dispatch: its cost is flat from M=1 to
// M=8, which is the launches and none of the work. A second dispatch for the write-back would
// roughly double it.
//
// It only holds while `hist < kPcBT`, which the constraint table already guarantees -- the
// static_assert below is where that dependency is written down.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cstdint>

namespace {

enum {
    kPcThreads = 256,
    kPcBT      = 32,   /* tokens a thread walks; the tile's working set is (BT + hist) * 512 B */
    kPcMaxW    = 8,    /* taps held in registers, so the weight loop unrolls to a static bound */
    kPcMaxHist = 16,   /* (width-1)*dilation, the write-back's register window; see cPleConv */
};

/* THE WINDOW WRITE-BACK LIVES IN TILE ZERO and that is only sound while the deepest tap is inside
 * the first tile. At hist >= BT a second block would read history too, and the fold would overwrite
 * bytes another block is still reading -- a wrong prefix on the first tokens of a sequence, which
 * nothing downstream reports. The constraint table caps width and dilation so that hist <= 16; this
 * is the line that says the tile size is the OTHER half of that bound. */
static_assert(kPcMaxHist < kPcBT, "the conv history must fit inside tile 0 for the fused write-back");

/* x * sigmoid(x), and expf rather than __expf. The result is rounded to bf16 two instructions
 * later, so the fast intrinsic's couple of ULP would be invisible in the bytes -- but this op is
 * bound by the three arrays it moves, not by its transcendental, and the oracle it is checked
 * against uses std::exp. The GDN conv, which is instruction-bound, makes the opposite trade. */
__device__ __forceinline__ float pc_silu(float z) { return z / (1.0f + expf(-z)); }

__global__ __launch_bounds__(kPcThreads) void ple_conv_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,
    uint16_t* __restrict__ y, uint32_t n, uint32_t wid, uint32_t dil, uint32_t hist,
    uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len) {

    const uint32_t c = blockIdx.x * kPcThreads + threadIdx.x;
    if (c >= n) return;
    const uint32_t s = blockIdx.z;
    const int32_t bos = cu[s];
    const int32_t slen = cu[s + 1] - bos;
    if (slen <= 0) return;
    const int32_t t0 = (int32_t)(blockIdx.y * kPcBT);
    if (t0 >= slen) return;
    const int32_t t1 = min(t0 + (int32_t)kPcBT, slen);

    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;
    /* A cold slot's window is ZEROS and not what the slot last held: a fresh sequence landing on a
     * recycled slot would otherwise convolve the previous sequence's tail into its first `hist`
     * tokens -- fluent output with the wrong prefix. */
    const bool warm = cs && (!hini || hini[s] != 0);
    const long long off = nacc ? (long long)nacc[s] - 1 : 0;

    /* This channel's taps, read once for the whole tile. */
    float wt[kPcMaxW];
#pragma unroll
    for (uint32_t j = 0; j < kPcMaxW; ++j) wt[j] = j < wid ? bf16_to_f32(w[(size_t)c * wld + j]) : 0.0f;

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c;
    const uint16_t* __restrict__ rc = resid ? resid + (size_t)bos * rld + c : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c;

    for (int32_t t = t0; t < t1; ++t) {
        float acc = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kPcMaxW; ++j) {
            if (j >= wid) break;
            const int32_t src = t - (int32_t)hist + (int32_t)(j * dil);
            const float v = src >= 0 ? bf16_to_f32(xc[(size_t)src * xld])
                                     : (hc ? bf16_to_f32(hc[(src + (int32_t)hist) * cs_tok]) : 0.0f);
            acc += v * wt[j];
        }
        const float r = rc ? bf16_to_f32(rc[(size_t)t * rld]) : 0.0f;
        yc[(size_t)t * yld] = f32_to_bf16(r + pc_silu(acc));
    }

    /* THE WINDOW THIS STEP LEAVES BEHIND, in the same launch and with no barrier at all.
     *
     * WHICH BLOCK READ THE STATE decides that this is legal: history is read only where
     * `t - hist + j*dilation < 0`, so only tokens below `hist` reach back -- and `hist` is 9
     * against a 32-token tile, so TILE ZERO is the only block that ever touches state[slot][c].
     * Within it one thread owns a whole channel, so its own reads are ordered before its own
     * writes by program order and there is nothing to synchronise.
     *
     * Folding it in matters because at decode this op is nothing but launches: its cost is flat
     * from M=1 to M=8, which is the dispatches and none of the work.
     *
     * THE LAYOUT IS libref/ref_ple.cpp's, and `num_accepted` picks it. Present, this is a decode
     * step whose drafts a verify may reject: the history read is shifted down by one and every
     * input of the step follows it, slot i holding position i + 1 of (window read ++ step), so a
     * next step that kept k tokens reads at k - 1 and finds the history ending at the k-th.
     * Absent, every token commits and the last `hist` inputs land at offset zero. The two agree
     * at one token. Slots past `state_len` are not written; the manager sizes the window
     * hist + n_spec, which covers a step of 1 + n_spec tokens.
     *
     * The values are copied as bf16 BITS -- the same tensor's bytes on both sides, and a round
     * trip through f32 would be a second definition of the state's precision. The step's rows
     * come out of `x`, which is read-only here, so reading rows another tile is also reading is
     * not a hazard. Every history value is loaded before the first store, as in the tail form. */
    if (!dst || blockIdx.y != 0) return;
    uint16_t keep[kPcMaxHist];
    if (nacc) {
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i) {
            if (i + 1 >= hist) break;
            keep[i] = hc ? hc[(size_t)(i + 1) * cs_tok] : (uint16_t)0;
        }
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i) {
            if (i + 1 >= hist) break;
            dst[(size_t)i * cs_tok] = keep[i];
        }
        for (int32_t t = 0; t < slen && (long long)hist - 1 + t < state_len; ++t)
            dst[(size_t)(hist - 1 + (uint32_t)t) * cs_tok] = xc[(size_t)t * xld];
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i >= hist) break;
        const int32_t src = slen - (int32_t)hist + (int32_t)i;
        keep[i] = src >= 0 ? xc[(size_t)src * xld]
                           : (hc ? hc[(src + (int32_t)hist) * cs_tok] : (uint16_t)0);
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i >= hist) break;
        dst[(size_t)i * cs_tok] = keep[i];
    }
}

enum { PC_X = 0, PC_W, PC_CS, PC_CU, PC_RESID, PC_CIDX, PC_HINIT, PC_NACC, PC_Y };

}  // namespace

extern "C" int r4d_rad_ple_conv(const RadArgs* a, RadStream s) {
    const RadTensor* x     = r4d_opt(a, PC_X);
    const RadTensor* w     = r4d_opt(a, PC_W);
    const RadTensor* cs    = r4d_opt(a, PC_CS);
    const RadTensor* cu    = r4d_opt(a, PC_CU);
    const RadTensor* resid = r4d_opt(a, PC_RESID);
    const RadTensor* cidx  = r4d_opt(a, PC_CIDX);
    const RadTensor* hini  = r4d_opt(a, PC_HINIT);
    const RadTensor* nacc  = r4d_opt(a, PC_NACC);
    const RadTensor* y     = r4d_opt(a, PC_Y);
    if (!x || !w || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (resid && resid->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cs && cs->data && cs->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (cidx && cidx->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    /* `has_init` IS THE CALLER'S WIDTH and reading it as bytes is how the GDN conv lost a long
     * prompt's history: ctx_len 512 has a zero low byte. The engine hands over RadBatch::ctx_lens,
     * which is int32 like every derived buffer, so anything else is refused rather than guessed. */
    if (hini && hini->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (nacc && nacc->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    const long long n   = r4d_geti_or(a, "n", 0);
    const long long wid = r4d_geti_or(a, "width", 0);
    const long long dil = r4d_geti_or(a, "dilation", 1);
    if (n <= 0 || wid <= 0 || dil <= 0) return R4D_NO(RAD_E_SHAPE);
    if (wid > kPcMaxW) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long hist = (wid - 1) * dil;
    if (hist > kPcMaxHist) return R4D_NO(RAD_E_UNSUPPORTED);

    long long T = 0, ty = 0, tr = 0, wrows = 0;
    const long long xld = r4d_rows_pitch(x, n, &T);
    const long long yld = r4d_rows_pitch(y, n, &ty);
    const long long rld = resid ? r4d_rows_pitch(resid, n, &tr) : 0;
    const long long wld = r4d_rows_pitch(w, wid, &wrows);
    if (xld < 0 || yld < 0 || wld < 0 || (resid && rld < 0)) return R4D_NO(RAD_E_STRIDE);
    /* THE ROW COUNT COMES FROM THE OPERANDS AND EVERY ONE HAS TO AGREE. `M` is ranged, so the value
     * in the args is the band's upper bound and a grid sized by it walks off the arena. */
    if (T <= 0 || ty != T || (resid && tr != T)) return R4D_NO(RAD_E_SHAPE);
    if (wrows < n) return R4D_NO(RAD_E_SHAPE);

    const long long N = r4d_numel(cu) - 1;
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    /* [n_slots, n, state_len]. A state too shallow for the deepest tap is refused here; the OFFSET
     * a rollback reads at cannot be checked without reading `num_accepted` on the host, and the
     * manager sizes the window as hist + n_spec for exactly that reason. */
    long long cs_slot = 0, cs_ch = 0, cs_tok = 0, state_len = 0;
    const bool have_cs = cs && cs->data && hist > 0;
    if (have_cs) {
        if (cs->rank != 3) return R4D_NO(RAD_E_SHAPE);
        if (r4d_dim(cs, 1) < n || r4d_dim(cs, 2) < hist) return R4D_NO(RAD_E_SHAPE);
        cs_slot   = r4d_stride(cs, 0);
        cs_ch     = r4d_stride(cs, 1);
        cs_tok    = r4d_stride(cs, 2);
        state_len = r4d_dim(cs, 2);
    }

    const uint32_t cb = (uint32_t)((n + kPcThreads - 1) / kPcThreads);
    const uint32_t tb = (uint32_t)((T + kPcBT - 1) / kPcBT);
    hipLaunchKernelGGL(ple_conv_kernel, dim3(cb, tb, (uint32_t)N), dim3(kPcThreads), 0,
                       (hipStream_t)s,
                       (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w),
                       have_cs ? (uint16_t*)r4d_addr(cs) : nullptr,
                       (const int32_t*)r4d_addr(cu),
                       resid ? (const uint16_t*)r4d_addr(resid) : nullptr,
                       cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,
                       hini ? (const int32_t*)r4d_addr(hini) : nullptr,
                       nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,
                       (uint16_t*)r4d_addr(y), (uint32_t)n, (uint32_t)wid, (uint32_t)dil,
                       (uint32_t)hist, (uint32_t)xld, (uint32_t)rld, (uint32_t)yld, (uint32_t)wld,
                       cs_slot, cs_ch, cs_tok, state_len);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
