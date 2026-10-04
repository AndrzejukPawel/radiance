/* ref_ple.cpp -- the gate of Qwen4-Exp's Per-Layer Embedding: everything between the n-gram
 * lookup's two projections and the dilated convolution.
 *
 * ============================== WHAT IT IS ==============================
 *
 * PLE mixes a hashed n-gram embedding into the residual stream, and the mixing coefficient is a
 * DOT PRODUCT between the embedding and the stream -- one scalar per hyper-connection stream, per
 * token. Written out, with `n` the model width and `hc` the number of streams:
 *
 *   K[c] = grouped_rmsnorm(k)[c] * (w_key[c]   + wadd)      k is key_proj(embedding)
 *   Q[c] = grouped_rmsnorm(q)[c] * (w_query[c] + wadd)      q is the INCOMING hc stream
 *   s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
 *   s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))                 <- the signed square root
 *   gv[c] = sigmoid(s[c]) * v                               v is value_proj(embedding), ONE row
 *   gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)
 *
 * and the caller finishes with `out = gv + silu(dilated_conv(gvn))`. BOTH gv AND gvn leave this op
 * because the convolution reads the NORMED copy and the residual add reads the UN-normed one --
 * two different tensors, and recomputing either outside would be a second definition of the norm.
 *
 * ============================== THE SIGNED SQUARE ROOT ==============================
 *
 * `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` is not a monotone rescaling of the dot product
 * and it is not decoration: it compresses a score whose magnitude spans orders of magnitude into
 * something sigmoid can resolve, and it KEEPS THE SIGN, so a stream whose embedding opposes the
 * residual gets a gate below a half rather than a gate of zero.
 *
 * THE CLAMP IS ON THE MAGNITUDE AND THE SIGN IS TAKEN FROM THE ORIGINAL, which matters at exactly
 * one value: torch's `sign(0)` is 0, so a dot product of exactly zero comes out as zero and not as
 * sqrt(gate_eps). Clamping first and then re-signing the clamped value would give 1e-3 there.
 * Zero is not a measure-zero curiosity in a quantised model -- an fp8 activation row can produce
 * it -- so the order is written out rather than left to the arithmetic.
 *
 * ============================== ONE OP AND NOT SIX ==============================
 *
 * Three grouped norms, a reduction, a scalar nonlinearity and a broadcast multiply would be six
 * launches over tensors that are all [M, hc*n] -- 10240 wide at this model's shape. Every launch
 * costs a fixed floor and every one of those ops is memory-bound, so the fused form is the only one
 * worth writing. `hc_read` is the same argument at the same width.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <cmath>

using namespace ref;

namespace {

/* A value narrowed to a tensor's own dtype and read back -- the same helper ref_hc.cpp carries and
 * for the same reason. The reference model's arithmetic is NOT f32 throughout: torch's grouped norm
 * returns `.type_as(x)`, so everything downstream of a norm reads a bf16 value. Rounding once, here,
 * is what lets a fused kernel match this to a tolerance that means something. */
inline float narrow(uint32_t dtype, float v) {
    uint32_t buf[2] = {0, 0};
    st_dt(dtype, buf, 0, v);
    return ld_dt(dtype, buf, 0);
}

/* One group's reciprocal RMS: the rsqrt is per `n`-wide stream while the gain below it spans all
 * hc*n. That asymmetry is the whole of what `group_size` means on Qwen4ExpTextRMSNorm. */
inline float group_rsqrt(const RadTensor* x, int64_t base, int64_t stride, int64_t n, float eps) {
    double acc = 0;
    for (int64_t d = 0; d < n; ++d) {
        const double v = ldt(x, base + d * stride);
        acc += v * v;
    }
    return (float)(1.0 / std::sqrt(acc / (double)n + (double)eps));
}

}  /* namespace */

/* PLE's gate.
 *
 *   k        [M][hc*n]   key_proj(embedding), UN-normed
 *   q        [M][hc*n]   the incoming hyper-connection stream, UN-normed
 *   v        [M][n]      value_proj(embedding) -- ONE row, gated differently per stream
 *   w_key    [hc*n]   w
 *   w_query  [hc*n]   w
 *   w_conv   [hc*n]   w
 *   gv       [M][hc*n]   out, UN-normed: what the residual add reads
 *   gvn      [M][hc*n]   out, normed:    what the convolution reads
 */
int ref_ple_gate(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor* k   = t_in(a, 0);
    const RadTensor* q   = t_in(a, 1);
    const RadTensor* v   = t_in(a, 2);
    const RadTensor* wk  = t_in(a, 3);
    const RadTensor* wq  = t_in(a, 4);
    const RadTensor* wc  = t_in(a, 5);
    const RadTensor* gv  = t_in(a, 6);
    const RadTensor* gvn = t_in(a, 7);
    if (!k || !q || !v || !wk || !wq || !wc || !gv || !gvn) return RAD_E_INVAL;

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    /* `v` is [M, n] and it is the call; `M` is the band. See band_rows in ref_common.h. */
    const int64_t M  = n > 0 ? band_rows(numel(v) / n, p_int(a, "M", 0)) : 0;
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    /* The floor the gate's magnitude is clamped to before the square root. Named rather than
     * written into the expression: it is a model constant, and a constant a reader cannot see is
     * one a kernel can disagree with in silence. */
    const float geps = p_f32(a, "gate_eps", 1e-6f);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    if (numel(k) < M * hn || numel(q) < M * hn || numel(v) < M * n) return RAD_E_SHAPE;
    if (numel(wk) < hn || numel(wq) < hn || numel(wc) < hn) return RAD_E_SHAPE;
    if (numel(gv) < M * hn || numel(gvn) < M * hn) return RAD_E_SHAPE;

    const int64_t kls = laststride(k), qls = laststride(q), vls = laststride(v);
    const int64_t wks = laststride(wk), wqs = laststride(wq), wcs = laststride(wc);
    const int64_t gls = laststride(gv), gnls = laststride(gvn);

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t kb = rowoff(k, m, hn), qb = rowoff(q, m, hn), vb = rowoff(v, m, n);
        const int64_t gb = rowoff(gv, m, hn), nb = rowoff(gvn, m, hn);

        for (int64_t c = 0; c < hc; ++c) {
            /* The two normed streams, and the dot between them. Neither is materialised: every
             * element is used exactly once here, and at 2560 wide per stream a buffer would be the
             * only allocation in the op. */
            const float rk = group_rsqrt(k, kb + c * n * kls, kls, n, eps);
            const float rq = group_rsqrt(q, qb + c * n * qls, qls, n, eps);
            double dot = 0;
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                const float kk = narrow(k->dtype, ldt(k, kb + j * kls) * rk *
                                                  (ldt(wk, j * wks) + wadd));
                const float qq = narrow(q->dtype, ldt(q, qb + j * qls) * rq *
                                                  (ldt(wq, j * wqs) + wadd));
                dot += (double)kk * (double)qq;
            }
            float s = (float)(dot / std::sqrt((double)n));

            /* THE SIGNED SQUARE ROOT, in the reference's own order: the magnitude is clamped, the
             * root is taken, and the SIGN COMES FROM THE ORIGINAL. torch's sign(0) is 0, so a dot
             * of exactly zero stays zero instead of becoming sqrt(gate_eps). */
            const float mag = std::sqrt(std::fabs(s) < geps ? geps : std::fabs(s));
            s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);

            const float g = act_sigmoid(s);
            for (int64_t d = 0; d < n; ++d)
                stt(gv, gb + (c * n + d) * gls, narrow(gv->dtype, g * ldt(v, vb + d * vls)));
        }

        /* And the third norm, over what was just written -- READ BACK, not carried in registers,
         * because `gv` has been narrowed to its own dtype and the convolution's input is a norm of
         * THOSE bytes. A fused kernel that normed the f32 it still held would disagree in the last
         * bits -- the same epilogue-fold rule the GDN out-projection follows. */
        for (int64_t c = 0; c < hc; ++c) {
            const float r = group_rsqrt(gv, gb + c * n * gls, gls, n, eps);
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                stt(gvn, nb + j * gnls,
                    narrow(gvn->dtype, ldt(gv, gb + j * gls) * r * (ldt(wc, j * wcs) + wadd)));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ple_conv */
/* PLE's depthwise convolution: width 4 at DILATION 3, silu AFTER it, and the residual add folded
 * in. That is the last piece of the layer -- `out = gv + silu(conv(gvn))`.
 *
 * ============================== THE DILATION IS THE WHOLE DIFFERENCE ==============================
 *
 * A dilated causal convolution reads `width` taps spaced `dilation` apart:
 *
 *     y[t] = silu( sum_j w[c][j] * x[t - (width-1-j)*dilation] )
 *
 * so at Qwen4-Exp's width 4 and dilation 3 the taps are 9, 6, 3 and 0 timesteps back, and the
 * history a sequence has to carry is (width-1)*dilation = NINE, not three. Every off-by-one in a
 * convolution cache is a model that generates fluently from a slightly wrong context, so the
 * indexing is written once here and mirrors gdn_conv_prep's exactly: state slot `i` holds the
 * value at relative position `i - state_len`, and a read at `src < 0` becomes `cs[slot][c][src +
 * state_len]`.
 *
 * The reference reaches the same place by a different route -- it pads by `state_len`, slices the
 * last `state_len + seq_len`, and convolves with no padding -- and that pad-and-slice is a no-op
 * on a warm cache. What it exists for is the COLD one, where the window in front of the first
 * token must be zeros.
 *
 * ============================== ONE OP WHERE THE GDN HAS TWO ==============================
 *
 * gdn_conv_prep and gdn_conv_update are separate because the work AROUND their convolutions
 * differs -- one splits a fused projection and takes an l2 norm, the other does not. PLE's
 * convolution is the same arithmetic at prefill and at decode, so it is one op, and it takes the
 * union of the two contracts: `has_init` says whether the slot holds a history (prefill's
 * question) and `num_accepted` shifts the read offset within a deeper window (decode's).
 *
 * `num_accepted` ABSENT MEANS EVERY TOKEN COMMITS, which is what a prefill chunk is: the read is at
 * offset zero and nothing this step ran can be rejected. gdn_conv_update makes it required and
 * argues that an optional rollback contract is not a contract; the difference here is that this op
 * serves prefill too, where there is no previous step to have rejected anything. The state must
 * still be sized (width-1)*dilation + n_spec by the manager, and a window too shallow for the
 * offset asked for is refused by shape rather than clamped.
 *
 * ============================== WHAT THE WINDOW HOLDS AFTER A STEP ==============================
 *
 * The next step reads at ITS `num_accepted - 1`, and what that offset must find is the `hist`
 * values ending at the last token this step COMMITTED. Which token that is depends on the kind of
 * step, so the presence of `num_accepted` picks the layout:
 *
 *   PRESENT -- a decode step of `slen` tokens, any of which past the first a verify may reject.
 *     The window is rewritten the way gdn_conv_update rewrites its own: slot i holds position
 *     i + 1 of (the window this step read ++ this step's inputs), so the old history shifted down
 *     by one and then token t at slot hist - 1 + t. A next step that kept k of these tokens reads
 *     at k - 1 and finds the history ending at token k - 1. It needs hist - 1 + slen slots, which
 *     is the manager's hist + n_spec for a step of 1 + n_spec tokens.
 *
 *   ABSENT -- every token commits, so the next step reads at offset zero (its `num_accepted` is
 *     1) and the window is the last `hist` values of the step, there.
 *
 * The two agree when the step has one token. Writing the tail at offset zero after a verify of
 * several would hand a step that accepted only the first of them a history ending at a rejected
 * draft, and one that accepted two a slot nothing wrote.
 */
int ref_ple_conv(const RadArgs* a, RadStream) {
    /* NINE positions, four of them optional, so the count is checked and the optionals are tested
     * one by one -- `have()` would refuse the op for an absent operand that is allowed to be. */
    if (a->n_t < 9) return RAD_E_INVAL;
    const RadTensor* x     = t_in(a, 0);
    const RadTensor* w     = t_in(a, 1);
    const RadTensor* cs    = t_in(a, 2);
    const RadTensor* cu    = t_in(a, 3);
    const RadTensor* resid = t_in(a, 4);      /* optional */
    const RadTensor* sidx  = t_in(a, 5);      /* optional */
    const RadTensor* hini  = t_in(a, 6);      /* optional */
    const RadTensor* nacc  = t_in(a, 7);      /* optional */
    const RadTensor* y     = t_in(a, 8);
    if (!x || !w || !cs || !cu || !y) return RAD_E_INVAL;

    const int64_t n    = p_int(a, "n", 0);
    const int64_t wid  = p_int(a, "width", 0);
    /* `x` is [M, n] and it is the call; `M` is the band. See band_rows in ref_common.h. */
    const int64_t M    = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    const int64_t dil  = p_int(a, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_SHAPE;
    const int64_t hist_len = (wid - 1) * dil;
    /* The read window lives in a stack frame. Nine at this model's shape; the bound is stated so a
     * caller with a wider one is refused rather than overrunning it. */
    enum { kHistMax = 64 };
    if (hist_len > kHistMax) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    if (numel(x) < M * n || numel(y) < M * n || numel(w) < n * wid) return RAD_E_SHAPE;
    if (resid && numel(resid) < M * n) return RAD_E_SHAPE;

    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : hist_len;
    if (cs->data && state_len < hist_len) return RAD_E_SHAPE;

    /* Checked for every sequence BEFORE any of them is written: a state too shallow for the offset
     * asked for, or for the window a decode step leaves behind, is a caller bug whose symptom
     * would otherwise be a silently truncated history. */
    for (int64_t s = 0; s < N; ++s) {
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + hist_len > state_len) return RAD_E_SHAPE;
        const int64_t slen = ld_int(cu->dtype, cu->data, offlin(cu, s + 1)) -
                             ld_int(cu->dtype, cu->data, offlin(cu, s));
        if (nacc && cs->data && hist_len > 0 && slen > 0 && hist_len - 1 + slen > state_len)
            return RAD_E_SHAPE;
    }

    const int64_t xrow = rowoff(x, 1, n), xcol = laststride(x);
    const int64_t yrow = rowoff(y, 1, n), ycol = laststride(y);
    const int64_t rrow = resid ? rowoff(resid, 1, n) : 0;
    const int64_t rcol = resid ? laststride(resid) : 0;
    const int64_t wrow = rowoff(w, 1, wid), wcol = laststride(w);
    /* [n_slots, n, state_len], tight, the layout gdn's conv state already uses. */
    const int64_t cs_slot = n * state_len, cs_ch = state_len;

    #pragma omp parallel for schedule(static)
    for (int64_t sc = 0; sc < N * n; ++sc) {
        const int64_t s = sc / n, c = sc % n;
        const int64_t bos = ld_int(cu->dtype, cu->data, offlin(cu, s));
        const int64_t slen = ld_int(cu->dtype, cu->data, offlin(cu, s + 1)) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        /* A zero `has_init` means the slot holds no history: the window in front of the first
         * token is ZEROS and not whatever the slot last held. Without it a fresh sequence landing
         * on a recycled slot convolves the previous sequence's tail into its first nine tokens --
         * fluent output with the wrong prefix. */
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, s)) != 0;
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;

        float hist[kHistMax];
        for (int64_t i = 0; i < hist_len; ++i)
            hist[i] = (warm && cs->data) ? ldt(cs, slot * cs_slot + c * cs_ch + off + i) : 0.0f;

        for (int64_t t = 0; t < slen; ++t) {
            float acc = 0.0f;
            for (int64_t j = 0; j < wid; ++j) {
                const int64_t src = t - hist_len + j * dil;
                const float v = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                         : hist[src + hist_len];
                acc += v * ldt(w, c * wrow + j * wcol);
            }
            const float r = resid ? ldt(resid, (bos + t) * rrow + c * rcol) : 0.0f;
            stt(y, (bos + t) * yrow + c * ycol, r + act_silu(acc));
        }
    }

    /* The window this step leaves behind, written after every read is done -- a sequence longer
     * than the window overwrites history it was still reading otherwise. The layout is the one
     * the header describes: shifted by one with every input of the step behind it when
     * `num_accepted` is present, the last `hist` values at offset zero when it is not. */
    if (cs->data && hist_len > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t sc = 0; sc < N * n; ++sc) {
            const int64_t s = sc / n, c = sc % n;
            const int64_t bos = ld_int(cu->dtype, cu->data, offlin(cu, s));
            const int64_t slen = ld_int(cu->dtype, cu->data, offlin(cu, s + 1)) - bos;
            if (slen <= 0) continue;
            const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, s)) : s;
            if (slot < 0) continue;
            const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, s)) != 0;
            const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
            const int64_t base = slot * cs_slot + c * cs_ch;

            float win[kHistMax];
            for (int64_t i = 0; i < hist_len; ++i)
                win[i] = warm ? ldt(cs, base + off + i) : 0.0f;

            if (nacc) {
                for (int64_t i = 0; i + 1 < hist_len; ++i) stt(cs, base + i, win[i + 1]);
                for (int64_t t = 0; t < slen; ++t)
                    stt(cs, base + hist_len - 1 + t, ldt(x, (bos + t) * xrow + c * xcol));
            } else {
                float keep[kHistMax];
                for (int64_t i = 0; i < hist_len; ++i) {
                    const int64_t src = slen - hist_len + i;
                    keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                       : win[src + hist_len];
                }
                for (int64_t i = 0; i < hist_len; ++i) stt(cs, base + i, keep[i]);
            }
        }
    }
    return RAD_OK;
}
