/* ref_gdn.cpp -- the gated delta net: the conv preamble, the chunk preamble, the chunked scan, the
 * recurrent update and the gated norm.
 *
 * The algorithm is libr4d's, stated in its own comments and reproduced here without any of the
 * fusion. Where libr4d says the state lives in WMMA accumulators across a whole chunk, this keeps
 * it in a per-thread workspace and writes every intermediate out; where libr4d recomputes the gram
 * once per k head for three v heads, this recomputes it per v head. Both are the same arithmetic.
 *
 * ============================== THE ALGORITHM ==============================
 *
 * Per token, per v head, with q and k l2-normalised and the state S shaped [head_v, head_k]:
 *
 *     g    = -exp(A_log) . softplus(a + dt_bias)        beta = sigmoid(b)
 *     S   *= e^g
 *     u    = beta . (v - S k)          S += u k^T          o = S q
 *
 * That is the recurrent form, which is what DECODE runs. Prefill runs the chunked form, which is
 * the same recurrence blocked over `chunk` tokens with the intra-chunk dependencies solved in one
 * triangular inverse (`gdn_kkt_solve`) instead of sequentially:
 *
 *     ge[j] = e^{g_j}   gs[j] = e^{g_last - g_j}     (g is the INTRA-CHUNK CUMSUM of the gate)
 *     U  = K S^T        W  = beta . (V - ge . U)     V' = A W
 *     P  = Q K^T        sA = scale . e^{g_i - g_j} . tril(P)
 *     O  = scale . ge . (Q S^T) + sA V'
 *     S  = e^{g_last} S + V'^T (gs . K)
 *
 * EVERY DECAY FACTOR FORMED HERE IS <= 1, and that is a correctness requirement rather than a
 * style. Intra-chunk gate spans on real prefill inputs reach several hundred nats, so
 * any factorisation that puts half the exponent on its own -- e^{g_i - c} and e^{c - g_j} about
 * some per-chunk reference c -- sends one half past fp32's e^88 while the other underflows to
 * zero, whatever c is chosen. The pairs e^{g_i - g_j} (j <= i) and e^{g_last - g_j} never leave
 * (0, 1]. A reference that took the convenient factorisation would produce infinities on inputs
 * the fast kernel handles, which is the wrong way round for an oracle.
 *
 * ============================== THE LAYOUTS ==============================
 *
 *     q, k     [T, head_k_count, head_k]        v, o     [T, head_v_count, head_v]
 *     g, beta  [T, head_v_count]                A        [T, head_v_count, chunk]
 *     h0, ht   [N, head_v_count, head_v, head_k]         cu  [N+1]
 *     conv w   [conv_dim, conv_width]           conv b   [conv_dim]
 *     conv_state [n_slots, conv_dim, state_len]
 *     state      [n_slots, head_v_count, head_v, head_k]
 *
 * with conv_dim = 2 * head_k_count * head_k + head_v_count * head_v, and `x` the fused projection
 * row [q | k | v | a | b] -- conv_dim convolved channels followed by one `a` and one `b` per v
 * head. gdn_conv_prep takes `a` and `b_gate` as OPTIONAL operands: present, they are the layer's
 * separate gate and beta projections; absent, they are the last 2 * head_v_count columns of `x`.
 * gdn_recurrent_update always names them separately, and a caller with a fused row passes strided
 * views of it.
 *
 * The speculative rollback contract for the two state caches is in ref_ops.h and is obeyed here.
 */
#include "ref_common.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* The deepest convolution history this reference holds in a stack frame. libr4d's width is 4 and
 * no linear-attention model in circulation exceeds it by much; a wider request returns
 * RAD_E_UNSUPPORTED by name rather than convolving a truncated window, which would be numerically
 * plausible and completely wrong. */
enum { REF_CONV_HIST_MAX = 16 };

struct V3 { const RadTensor* t; int64_t s0, s1, s2; };
struct V2 { const RadTensor* t; int64_t s0, s1; };

static inline V3 v3(const RadTensor* t, int64_t d1, int64_t d2) {
    V3 v{ t, d1 * d2, d2, 1 };
    if (t->rank >= 3) {
        v.s0 = t->stride[t->rank - 3];
        v.s1 = t->stride[t->rank - 2];
        v.s2 = t->stride[t->rank - 1];
    }
    return v;
}
/* The same [tokens, d1, d2] reading for an operand that may be a COLUMN VIEW of a wider row: the
 * fp8 delta net's output gate is the tail of a fused [q|k|v|z] projection, so its token stride is
 * its own rather than d1*d2. A rank-3 spelling already carries all three strides; a rank-2 one
 * carries the token stride in stride[0] and v3() would have guessed d1*d2 for it. */
static inline V3 v3_rows(const RadTensor* t, int64_t d1, int64_t d2) {
    if (t->rank != 2) return v3(t, d1, d2);
    return V3{ t, t->stride[0], d2 * t->stride[1], t->stride[1] };
}
static inline V2 v2(const RadTensor* t, int64_t d1) {
    V2 v{ t, d1, 1 };
    if (t->rank >= 2) { v.s0 = t->stride[t->rank - 2]; v.s1 = t->stride[t->rank - 1]; }
    return v;
}
static inline int64_t at3(const V3& v, int64_t i, int64_t j, int64_t k) {
    return i * v.s0 + j * v.s1 + k * v.s2;
}
static inline int64_t at2(const V2& v, int64_t i, int64_t j) { return i * v.s0 + j * v.s1; }

/* [N, heads, head_v, head_k] -- the shape both the chunked scan's h0/ht and the paged recurrent
 * state have. The slot and head strides are read off the tensor and NOT derived from the shape:
 * a paged state cache is padded so its page matches the attention page, and a kernel that derived
 * the stride would read every slot but the first at the wrong offset, which looks exactly like a
 * correct kernel producing noise (libr4d makes the same point about vLLM's mamba page). */
struct V4 { const RadTensor* t; int64_t s0, s1, s2, s3; };
static inline V4 v4(const RadTensor* t, int64_t d1, int64_t d2, int64_t d3) {
    V4 v{ t, d1 * d2 * d3, d2 * d3, d3, 1 };
    if (t->rank >= 4) {
        v.s0 = t->stride[t->rank - 4];
        v.s1 = t->stride[t->rank - 3];
        v.s2 = t->stride[t->rank - 2];
        v.s3 = t->stride[t->rank - 1];
    }
    return v;
}
static inline int64_t at4(const V4& v, int64_t i, int64_t j, int64_t k, int64_t l) {
    return i * v.s0 + j * v.s1 + k * v.s2 + l * v.s3;
}

static inline int64_t cu_at(const RadTensor* cu, int64_t i) {
    return ld_int(cu->dtype, cu->data, offlin(cu, i));
}

/* The conv state is [n_slots, conv_dim, state_len]; a rank the caller flattened is read as the
 * tight layout that shape implies. */
static inline V3 conv_state_view(const RadTensor* cs, int64_t conv_dim, int64_t width,
                                 int64_t* state_len) {
    *state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    return v3(cs, conv_dim, *state_len);
}

}  /* namespace */

/* ------------------------------------------------------------------ gdn_conv_prep */
/* Everything between the qkv projection and the chunked scan: the width-`conv_width` depthwise
 * causal convolution with silu and its state cache, the q/k/v split, the l2 norm over each q and k
 * head, the gate with its per-chunk cumsum, and beta.
 *
 * THE CONV OUTPUT IS ROUNDED BEFORE THE NORM. A fused kernel rounds to bf16 there because the pair
 * of kernels it replaces does -- the conv writes a bf16 tensor and the prep reads it back -- and
 * matching that is what lets the fused form stand in for the unfused pair on real weights. The
 * reference does the same by writing the conv result into its output tensor and reading it back
 * for the norm.
 *
 * The l2 norm is rsqrt(sum(x^2) + l2_eps) with NO division by the head width -- it is a norm, not
 * an RMS -- and l2_eps is 1e-6, which is FLA's and libr4d's L2NORM_EPS. */
int ref_gdn_conv_prep(const RadArgs* a, RadStream) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[5], *cu = &a->t[6];
    const RadTensor *q = &a->t[11], *k = &a->t[12], *v = &a->t[13];
    const RadTensor *g = &a->t[14], *beta = &a->t[15];
    const RadTensor* bias = t_in(a, 2);
    const RadTensor* A_log = t_in(a, 3);
    const RadTensor* dt_bias = t_in(a, 4);
    /* The gate and beta projections, when the layer hands them separately from the qkv slice.
     * Absent, they are the last 2*H columns of the fused `x` row, which is what a caller holding
     * a single fused projection passes. */
    const RadTensor* ta = t_in(a, 7);
    const RadTensor* tb = t_in(a, 8);
    const RadTensor* sidx = t_in(a, 9);
    const RadTensor* hini = t_in(a, 10);
    if (!x->data || !w->data || !cu->data || !q->data || !k->data || !v->data ||
        !g->data || !beta->data) return RAD_E_INVAL;
    /* One of the two, not one of each: a caller that names the gate has to name beta as well, or
     * the second read silently falls back into a fused row that may not be there. */
    if ((ta != nullptr) != (tb != nullptr)) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* `chunk` is RAD_DERIVED on this op, and NOTHING HERE CARRIES IT: the gate cumsum is cut at
     * the chunk boundary and no operand's shape records where that is. Ref declares no RAD_C_EQ
     * for it, so a caller who both omitted it and resolved to ref arrives with the key unset and
     * gets RAD_E_SHAPE by name. The alternative -- treating the sequence as one chunk -- is a
     * decay summed over the wrong span, which is numerically plausible and completely wrong. A
     * caller pins `chunk` here (which is what the architecture plugins do) or resolves to a kernel
     * that declares one. */
    const int64_t chunk = p_int(a, "chunk", 0), width = p_int(a, "conv_width", 0);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    if (K <= 0 || V <= 0 || chunk <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > REF_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;                       /* an empty step is not an error */
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H  = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;
    if (numel(g) < T * H || numel(beta) < T * H) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : (conv_dim + (ta ? 0 : 2 * H));
    const int64_t xcol = laststride(x);
    /* THE BOUND IS THE VIEW'S AND NOT THE BUFFER'S. `x` may be a column slice of a wider row --
     * the fp8 block hands the convolution the left-hand `conv_dim` of a fused [q|k|v|z]
     * projection -- and a slice's numel counts only the elements it names, so the flat product
     * refuses a call that is perfectly in range. Rank 2 or more spells the extent in the shape;
     * only a flattened operand has to be bounded by numel. */
    const int64_t xneed = conv_dim + (ta ? 0 : 2 * H);
    if (x->rank >= 2) {
        if (x->shape[x->rank - 1] < xneed || x->shape[x->rank - 2] < T) return RAD_E_SHAPE;
    } else if (numel(x) < (T - 1) * xrow + xneed) {
        return RAD_E_SHAPE;
    }
    if (ta && (numel(ta) < T * H || numel(tb) < T * H)) return RAD_E_SHAPE;

    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);
    if (cs->data && state_len < width - 1) return RAD_E_SHAPE;

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, H, V);
    const V2 wv = v2(w, width);
    const V2 gv = v2(g, H), bv = v2(beta, H);
    const V2 av = ta ? v2(ta, H) : V2{ nullptr, 0, 0 };
    const V2 bgv = tb ? v2(tb, H) : V2{ nullptr, 0, 0 };

    /* ---- phase 1: the convolution, straight into q / k / v ---------------------------------- */
    #pragma omp parallel for schedule(static)
    for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
        const int64_t n = nc / conv_dim, c = nc % conv_dim;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
        /* A zero `has_init` means the slot holds no history: the window in front of the sequence's
         * first token is ZEROS, not whatever the slot last held. Without it a fresh sequence
         * landing on a recycled slot convolves the previous sequence's tail into its first
         * conv_width-1 tokens, which is fluent output with the wrong prefix. */
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, n)) != 0;

        /* Which of the three outputs this channel belongs to. Channel groups are counted in HEADS,
         * so a group never straddles the q/k/v boundaries at any tensor-parallel width. */
        const RadTensor* out;
        V3 ov;
        int64_t oh, od;
        if (c < Hg * K)            { out = q; ov = qv; oh = c / K; od = c % K; }
        else if (c < 2 * Hg * K)   { out = k; ov = kv; oh = (c - Hg * K) / K; od = (c - Hg * K) % K; }
        else                       { out = v; ov = vv; oh = (c - 2 * Hg * K) / V;
                                     od = (c - 2 * Hg * K) % V; }

        const float bs = bias ? ldt(bias, c * laststride(bias)) : 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            float acc = bs;
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                float val = 0.0f;
                if (src >= 0) val = ldt(x, (bos + src) * xrow + c * xcol);
                else if (warm && cs->data && slot >= 0)
                    val = ldt(cs, at3(cvs, slot, c, src + width - 1));
                acc += val * ldt(w, at2(wv, c, j));
            }
            stt(out, at3(ov, bos + t, oh, od), act_silu(acc));
        }
    }

    /* ---- phase 2: the l2 norm over each q and k head, reading the ROUNDED conv output -------- */
    #pragma omp parallel for schedule(static)
    for (int64_t th = 0; th < T * Hg * 2; ++th) {
        const int64_t which = th / (T * Hg);
        const int64_t rest = th % (T * Hg);
        const int64_t t = rest / Hg, h = rest % Hg;
        const RadTensor* z = which ? k : q;
        const V3& zv = which ? kv : qv;
        float ss = 0.0f;
        for (int64_t d = 0; d < K; ++d) { float u = ldt(z, at3(zv, t, h, d)); ss += u * u; }
        const float inv = 1.0f / std::sqrt(ss + l2eps);
        for (int64_t d = 0; d < K; ++d)
            stt(z, at3(zv, t, h, d), ldt(z, at3(zv, t, h, d)) * inv);
    }

    /* ---- phase 3: the gate, its per-chunk cumsum, and beta ----------------------------------- */
    /* The cumsum is per CHUNK and per sequence, because that is what the scan consumes: g[j] is
     * the decay from the start of j's chunk up to and including j. */
    #pragma omp parallel for schedule(static)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb  = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        float run = 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            if (t % chunk == 0) run = 0.0f;
            const float av_ = ta ? ldt(ta, at2(av, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + h) * xcol);
            const float bvl = tb ? ldt(tb, at2(bgv, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + H + h) * xcol);
            run += -ref_exp(alog) * act_softplus(av_ + dtb, spthr);
            stt(g, at2(gv, bos + t, h), run);
            stt(beta, at2(bv, bos + t, h), act_sigmoid(bvl));
        }
    }

    /* ---- phase 4: rewrite the conv state from the sequence's last width-1 tokens -------------- */
    /* After phase 1, so the history it overwrites has already been read. A sequence shorter than
     * width-1 keeps what the old state held in front of its tokens, which is what makes chunked
     * prefill work: the second chunk of a sequence continues from the first. */
    if (cs->data && width > 1) {
        #pragma omp parallel for schedule(static)
        for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
            const int64_t n = nc / conv_dim, c = nc % conv_dim;
            const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
            const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
            if (slen <= 0 || slot < 0) continue;
            const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, n)) != 0;
            float keep[REF_CONV_HIST_MAX];
            const int64_t st = width - 1;
            for (int64_t i = 0; i < st; ++i) {
                const int64_t src = slen - (width - 1) + i;
                keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                   : (warm ? ldt(cs, at3(cvs, slot, c, src + width - 1)) : 0.0f);
            }
            for (int64_t i = 0; i < st; ++i) stt(cs, at3(cvs, slot, c, i), keep[i]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_update */
/* The same convolution for a decode step, against the rolling window. See ref_ops.h for the
 * contract; the two things that matter are that the READ starts at num_accepted-1 and that the
 * rewrite shifts the buffer down by one so the next step can do the same. A rejection is a change
 * of read offset, not a recompute (spec §10), and that is the whole reason this is a separate
 * entry point rather than a flag on the prefill one.
 *
 * No l2 norm here: on the decode path the recurrent update does it, which is where libr4d puts it
 * too. */
int ref_gdn_conv_update(const RadArgs* a, RadStream) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[3], *cu = &a->t[6];
    const RadTensor *q = &a->t[7], *k = &a->t[8], *v = &a->t[9];
    const RadTensor* bias = t_in(a, 2);
    const RadTensor* sidx = t_in(a, 4);
    const RadTensor* nacc = t_in(a, 5);
    if (!x->data || !w->data || !cs->data || !cu->data || !q->data || !k->data || !v->data)
        return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t width = p_int(a, "conv_width", 0);
    if (K <= 0 || V <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > REF_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H  = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : conv_dim;
    const int64_t xcol = laststride(x);
    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);

    /* The window this step needs: the largest read offset plus the history, and the largest write
     * slot. Both are checked here rather than clamped, because a cache that is too shallow is a
     * caller bug whose symptom would otherwise be a silently truncated history. */
    for (int64_t n = 0; n < N; ++n) {
        const int64_t slen = cu_at(cu, n + 1) - cu_at(cu, n);
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + width - 1 > state_len) return RAD_E_SHAPE;
        if (width > 1 && (width - 2) + slen > state_len) return RAD_E_SHAPE;
    }

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, H, V);
    const V2 wv = v2(w, width);

    #pragma omp parallel for schedule(static)
    for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
        const int64_t n = nc / conv_dim, c = nc % conv_dim;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
        if (slot < 0) continue;
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) - 1 : 0;

        float hist[REF_CONV_HIST_MAX];
        const int64_t st = width - 1;
        for (int64_t i = 0; i < st; ++i) hist[i] = ldt(cs, at3(cvs, slot, c, off + i));

        const RadTensor* out;
        V3 ov;
        int64_t oh, od;
        if (c < Hg * K)          { out = q; ov = qv; oh = c / K; od = c % K; }
        else if (c < 2 * Hg * K) { out = k; ov = kv; oh = (c - Hg * K) / K; od = (c - Hg * K) % K; }
        else                     { out = v; ov = vv; oh = (c - 2 * Hg * K) / V;
                                   od = (c - 2 * Hg * K) % V; }

        const float bs = bias ? ldt(bias, c * laststride(bias)) : 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            float acc = bs;
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                const float val = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                           : hist[src + width - 1];
                acc += val * ldt(w, at2(wv, c, j));
            }
            stt(out, at3(ov, bos + t, oh, od), act_silu(acc));
        }

        /* Shift down by one: the tail of the old history, then this step's tokens. Reading before
         * writing is what makes the in-place update safe, and the history is already in registers. */
        if (width > 1) {
            for (int64_t i = 0; i + 1 < width - 1; ++i)
                stt(cs, at3(cvs, slot, c, i), hist[i + 1]);
            for (int64_t t = 0; t < slen; ++t)
                stt(cs, at3(cvs, slot, c, t + width - 2),
                    ldt(x, (bos + t) * xrow + c * xcol));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_kkt_solve */
/* A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}, per chunk, per v head.
 *
 * The decay is applied PER ELEMENT as e^{g_i - g_j} rather than split about a per-chunk reference.
 * Only i > j is kept and g decreases along the chunk, so the difference is always <= 0 and the
 * direct form cannot overflow at any span -- which the split form can, on its own, at spans this
 * model actually reaches (libr4d's r4d_gdn_kkt_solve says the same at length).
 *
 * The inverse is a forward substitution rather than libr4d's blocked 16 -> 32 -> 64 WMMA merge.
 * The two agree exactly: I + M is unit lower triangular, so its inverse is unique and the
 * substitution computes it in fp32, which is what libr4d's diagonal blocks do too. */
int ref_gdn_kkt_solve(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *beta = &a->t[1], *g = &a->t[2], *cu = &a->t[3], *A = &a->t[4];

    const int64_t K = p_int(a, "head_k", 0);
    /* `chunk` is RAD_DERIVED, so a caller who did not pin it and resolved to ref (which declares
     * no RAD_C_EQ for it) arrives without the key. Here the operand says: A is [T, heads, chunk]
     * and the caller sized it, which is this plugin's rule for every other extent too. */
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    if (K <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(k) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (numel(A) < T * H * chunk) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;

    const V3 kv = v3(k, Hg, K), Av = v3(A, H, chunk);
    const V2 gv = v2(g, H), bv = v2(beta, H);

    #pragma omp parallel for schedule(static)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        float* wsp = ws_get((size_t)(chunk * chunk + chunk));
        float* Am = wsp;                 /* [chunk, chunk] the inverse, in fp32 */
        float* row = wsp + chunk * chunk;

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float bi = ldt(beta, at2(bv, ti, h));
                const float gi = ldt(g, at2(gv, ti, h));
                for (int64_t j = 0; j < i; ++j) {
                    const int64_t tj = bos + c0 + j;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < K; ++d)
                        dot += ldt(k, at3(kv, ti, hk, d)) * ldt(k, at3(kv, tj, hk, d));
                    row[j] = bi * dot * ref_exp(gi - ldt(g, at2(gv, tj, h)));
                }
                /* (I + M) A = I, row by row: A[i][j] = delta - sum_{j<=l<i} M[i][l] A[l][j]. */
                for (int64_t j = 0; j <= i; ++j) {
                    float s = (i == j) ? 1.0f : 0.0f;
                    for (int64_t l = j; l < i; ++l) s -= row[l] * Am[l * chunk + j];
                    Am[i * chunk + j] = s;
                }
                for (int64_t j = i + 1; j < chunk; ++j) Am[i * chunk + j] = 0.0f;
                for (int64_t j = 0; j < chunk; ++j)
                    stt(A, at3(Av, ti, h, j), Am[i * chunk + j]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_chunk_scan */
int ref_gdn_chunk_scan(const RadArgs* a, RadStream) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *A = &a->t[3];
    const RadTensor *g = &a->t[4], *beta = &a->t[5], *cu = &a->t[7];
    const RadTensor *o = &a->t[8], *ht = &a->t[9];
    const RadTensor* h0 = t_in(a, 6);
    /* THE STATE SLOT, when the caller keeps its recurrent states in a pool the batch does not
     * index positionally. Absent, a row owns slot n, which is correct only where slots are handed
     * out in batch order. */
    const RadTensor* sidx = t_in(a, 10);
    if (!q->data || !k->data || !v->data || !A->data || !g->data || !beta->data ||
        !cu->data || !o->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* RAD_DERIVED, and A is [T, heads, chunk] -- see gdn_kkt_solve, which writes the same tensor. */
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (K <= 0 || V <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(q) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;
    if (numel(o) < T * H * V || numel(A) < T * H * chunk) return RAD_E_SHAPE;
    if (!sidx) {
        if (ht->data && numel(ht) < N * H * V * K) return RAD_E_SHAPE;
        if (h0 && numel(h0) < N * H * V * K) return RAD_E_SHAPE;
    } else if (sidx->rank < 1 || sidx->shape[0] < N) {
        return RAD_E_SHAPE;
    }

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, H, V), ov = v3(o, H, V);
    const V3 Av = v3(A, H, chunk);
    const V2 gv = v2(g, H), bv = v2(beta, H);
    const V4 h0v = h0 ? v4(h0, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };
    const V4 htv = ht->data ? v4(ht, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };

    #pragma omp parallel for schedule(static)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* COLUMN 0 IS THE STATE. A row may be wider -- a decode kernel serving speculation asks
         * for scratch beside it -- and the scan writes the state and nothing else. */
        const int64_t sw = sidx && sidx->rank >= 1 ? sidx->stride[0] : 1;
        const int64_t sn = sidx ? ld_int(sidx->dtype, sidx->data, n * sw) : n;
        if (sn < 0) continue;                  /* a padded row owns no state */

        /* One workspace request, carved up by hand -- a second ws_get() may reallocate and
         * invalidate the first pointer (ref_common.h). */
        const size_t need = (size_t)(V * K + chunk * V + chunk * V + chunk);
        float* wsp = ws_get(need);
        float* S  = wsp;                       /* [V, K] the recurrent state */
        float* Wm = S + V * K;                 /* [chunk, V] */
        float* Vp = Wm + chunk * V;            /* [chunk, V] */
        float* row = Vp + chunk * V;           /* [chunk] one row of sA */

        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d)
                S[e * K + d] = h0 ? ldt(h0, at4(h0v, sn, h, e, d)) : 0.0f;

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            const float glast = ldt(g, at2(gv, bos + c0 + L - 1, h));

            /* W = beta . (V - e^{g_j} . (K S^T)) */
            for (int64_t j = 0; j < L; ++j) {
                const int64_t tj = bos + c0 + j;
                const float ge = ref_exp(ldt(g, at2(gv, tj, h)));
                const float bt = ldt(beta, at2(bv, tj, h));
                for (int64_t e = 0; e < V; ++e) {
                    float u = 0.0f;
                    for (int64_t d = 0; d < K; ++d)
                        u += ldt(k, at3(kv, tj, hk, d)) * S[e * K + d];
                    Wm[j * V + e] = bt * (ldt(v, at3(vv, tj, h, e)) - ge * u);
                }
            }
            /* V' = A W, with A unit lower triangular so only j <= i contributes. */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                for (int64_t e = 0; e < V; ++e) {
                    float s = 0.0f;
                    for (int64_t j = 0; j <= i; ++j)
                        s += ldt(A, at3(Av, ti, h, j)) * Wm[j * V + e];
                    Vp[i * V + e] = s;
                }
            }
            /* O = scale . e^{g_i} . (Q S^T) + sA V' */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float gi = ldt(g, at2(gv, ti, h));
                const float ge = ref_exp(gi);
                for (int64_t j = 0; j <= i; ++j) {
                    const int64_t tj = bos + c0 + j;
                    float p = 0.0f;
                    for (int64_t d = 0; d < K; ++d)
                        p += ldt(q, at3(qv, ti, hk, d)) * ldt(k, at3(kv, tj, hk, d));
                    row[j] = scale * ref_exp(gi - ldt(g, at2(gv, tj, h))) * p;
                }
                for (int64_t e = 0; e < V; ++e) {
                    float inter = 0.0f;
                    for (int64_t d = 0; d < K; ++d)
                        inter += ldt(q, at3(qv, ti, hk, d)) * S[e * K + d];
                    float s = scale * ge * inter;
                    for (int64_t j = 0; j <= i; ++j) s += row[j] * Vp[j * V + e];
                    stt(o, at3(ov, ti, h, e), s);
                }
            }
            /* S = e^{g_last} S + V'^T (e^{g_last - g_j} . K) */
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d) {
                    float s = ref_exp(glast) * S[e * K + d];
                    for (int64_t j = 0; j < L; ++j) {
                        const int64_t tj = bos + c0 + j;
                        const float gs = ref_exp(glast - ldt(g, at2(gv, tj, h)));
                        s += Vp[j * V + e] * (gs * ldt(k, at3(kv, tj, hk, d)));
                    }
                    S[e * K + d] = s;
                }
        }

        if (ht->data)
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    stt(ht, at4(htv, sn, h, e, d), S[e * K + d]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_recurrent_update */
/* What decode runs where prefill runs the chunked scan, against the PAGED state cache: the initial
 * state comes from the slot the last accepted token wrote and every candidate token writes its own
 * slot, because which of them survives verification is not known until after this layer has run
 * (ref_ops.h). The gated RMS norm is folded into the epilogue, which is where libr4d puts it too --
 * its workgroup owns the whole row. */
int ref_gdn_recurrent_update(const RadArgs* a, RadStream) {
    if (a->n_t < 14) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *av = &a->t[3], *bvt = &a->t[4];
    const RadTensor *state = &a->t[7], *o = &a->t[13];
    const RadTensor* A_log = t_in(a, 5);
    const RadTensor* dt_bias = t_in(a, 6);
    const RadTensor* sidx = t_in(a, 8);
    const RadTensor* nacc = t_in(a, 9);
    const RadTensor* cu = t_in(a, 10);
    const RadTensor* z = t_in(a, 11);
    const RadTensor* nw = t_in(a, 12);
    /* The out projection's quantiser, folded in. Optional and all-or-nothing; see the
     * operand list in r4d_rows.cpp for why the group is the row. */
    const RadTensor* oq  = (a->n_t > 14 && a->t[14].data) ? &a->t[14] : nullptr;
    const RadTensor* osc = (a->n_t > 15 && a->t[15].data) ? &a->t[15] : nullptr;
    if (!q->data || !k->data || !v->data || !av->data || !bvt->data || !state->data || !o->data ||
        !cu) return RAD_E_INVAL;

    /* `cu` BOUNDS THE SEQUENCES, exactly as it does on the sibling gdn_conv_update, which runs off
     * the same tensor on the same step. Carrying the sequence count in the state index instead
     * works only for a rectangular batch, and leaves a fast kernel -- which reads `cu[n]` before
     * any null test -- with a null pointer. Ref reads the token spans from cu and falls back to
     * the rectangular reading only for a caller whose cu is degenerate. */
    /* THE OTHER STATE CONTRACT IS DECLINED, NOT APPROXIMATED. Under "anchor" the cache holds one
     * committed state a sequence plus a block of replay factors, and the accepted prefix is
     * replayed onto it next step -- two slots whatever the draft depth. The `o` it produces is the
     * same, so a comparison against this reference AGREES on every output and disagrees on the
     * whole state, which reads as a broken kernel and is not one. A named refusal is the honest
     * answer: RAD_E_UNSUPPORTED says "not this contract", where a number would say "wrong". */
    const char* sf = p_str(a, "state_form", "per_candidate");
    if (sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float neps  = p_f32(a, "eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int nact = p_enum(a, "act", kActs, 2, 0);
    if (K <= 0 || V <= 0) return RAD_E_SHAPE;

    /* The head count and the token count come off the operands: v is [T, head_v_count, head_v],
     * so a rank-3 value tensor names both. `n_head_v` overrides it for a caller that hands a
     * flattened view. */
    int64_t Hv = p_int(a, "n_head_v", 0);
    if (Hv <= 0) Hv = v->rank >= 3 ? v->shape[v->rank - 2] : 0;
    if (Hv <= 0) return RAD_E_SHAPE;
    const int64_t Tk = numel(v) / (Hv * V);
    if (Tk <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (Tk * K);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = Hv / Hg;

    /* N comes from cu and nothing else: cu is [N+1] and its length minus one IS the sequence
     * count. Deriving N from state_idx and trusting cu to be at least as long is an off-by-one
     * waiting to happen, because a state index may be wider than one column per sequence. */
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) != Tk) return RAD_E_SHAPE;
    const int64_t sw = sidx ? (sidx->rank >= 2 ? sidx->shape[sidx->rank - 1]
                                               : (numel(sidx) / n_seq > 0 ? numel(sidx) / n_seq
                                                                          : 1))
                            : 1;

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, Hv, V), ov = v3(o, Hv, V);
    const V3 zv = z ? v3_rows(z, Hv, V) : vv;
    const V2 a2 = v2(av, Hv), b2 = v2(bvt, Hv);
    const V2 si = sidx ? v2(sidx, sw) : V2{ nullptr, 0, 0 };

    #pragma omp parallel for schedule(static)
    for (int64_t nh = 0; nh < n_seq * Hv; ++nh) {
        const int64_t n = nh / Hv, h = nh % Hv, hk = h / grp;
        const int64_t bos = cu_at(cu, n), q_len = cu_at(cu, n + 1) - bos;
        if (q_len <= 0) continue;
        const int64_t na = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) : 1;
        const int64_t rd = na > 0 ? na - 1 : 0;
        const int64_t slot_in = sidx ? ld_int(sidx->dtype, sidx->data, at2(si, n, rd < sw ? rd : sw - 1))
                                     : n;
        if (slot_in < 0) continue;

        const size_t need = (size_t)(V * K + 2 * K + 2 * V);
        float* wsp = ws_get(need);
        float* S  = wsp;
        float* qn = S + V * K;
        float* kn = qn + K;
        float* uu = kn + K;
        float* orow = uu + V;

        const int64_t sstride = state->rank >= 4 ? state->stride[0] : (int64_t)Hv * V * K;
        const int64_t hstride = state->rank >= 4 ? state->stride[1] : (int64_t)V * K;
        const int64_t vstride = state->rank >= 4 ? state->stride[2] : K;
        const int64_t dstride = state->rank >= 4 ? state->stride[3] : 1;

        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d)
                S[e * K + d] = ldt(state, slot_in * sstride + h * hstride + e * vstride +
                                          d * dstride);

        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb  = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;

        for (int64_t t = 0; t < q_len; ++t) {
            const int64_t tok = bos + t;
            const float gv = -ref_exp(alog) *
                             act_softplus(ldt(av, at2(a2, tok, h)) + dtb, spthr);
            const float bt = act_sigmoid(ldt(bvt, at2(b2, tok, h)));

            float sq = 0.0f, sk = 0.0f;
            for (int64_t d = 0; d < K; ++d) {
                const float u = ldt(q, at3(qv, tok, hk, d)), w = ldt(k, at3(kv, tok, hk, d));
                sq += u * u;
                sk += w * w;
            }
            const float qs = 1.0f / std::sqrt(sq + l2eps) * scale;
            const float ks = 1.0f / std::sqrt(sk + l2eps);
            for (int64_t d = 0; d < K; ++d) {
                qn[d] = ldt(q, at3(qv, tok, hk, d)) * qs;
                kn[d] = ldt(k, at3(kv, tok, hk, d)) * ks;
            }

            const float dec = ref_exp(gv);
            for (int64_t e = 0; e < V; ++e) {
                float sk2 = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    S[e * K + d] *= dec;
                    sk2 += S[e * K + d] * kn[d];
                }
                uu[e] = bt * (ldt(v, at3(vv, tok, h, e)) - sk2);
            }
            for (int64_t e = 0; e < V; ++e) {
                float ss = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    S[e * K + d] += uu[e] * kn[d];
                    ss += S[e * K + d] * qn[d];
                }
                orow[e] = ss;
            }

            /* The gated RMS norm, over the head's V channels. */
            float sq2 = 0.0f;
            for (int64_t e = 0; e < V; ++e) sq2 += orow[e] * orow[e];
            const float sc = 1.0f / std::sqrt(sq2 / (float)V + neps);
            for (int64_t e = 0; e < V; ++e) {
                float r = orow[e] * sc;
                if (nw) r *= ldt(nw, e * laststride(nw));
                if (z) {
                    const float zz = ldt(z, at3(zv, tok, h, e));
                    r *= (nact == 1 ? act_sigmoid(zz) : act_silu(zz));
                }
                stt(o, at3(ov, tok, h, e), r);
            }

            /* QUANTISE WHAT WAS WRITTEN, not what produced it: the standalone quant_act_fp8 this
             * replaces read `o` back out of memory, so it saw the rounded store and not the f32.
             * One group per (token, head), because head_v is the fp8 block width. */
            if (oq && osc) {
                float amax = 0.0f;
                for (int64_t e = 0; e < V; ++e) {
                    const float m = std::fabs(ldt(o, at3(ov, tok, h, e)));
                    if (m > amax) amax = m;
                }
                const float qsc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float inv = 1.0f / qsc;
                const int64_t qb = rowoff(oq, tok, Hv * V), qs = laststride(oq);
                for (int64_t e = 0; e < V; ++e) {
                    float u = ldt(o, at3(ov, tok, h, e)) * inv;
                    if (u > 448.0f) u = 448.0f;
                    if (u < -448.0f) u = -448.0f;
                    st_dt(oq->dtype, oq->data, qb + (h * V + e) * qs, u);
                }
                stt(osc, rowoff(osc, tok, Hv) + h * laststride(osc), qsc);
            }

            const int64_t so = sidx ? ld_int(sidx->dtype, sidx->data, at2(si, n, t < sw ? t : sw - 1))
                                    : n;
            if (so >= 0)
                for (int64_t e = 0; e < V; ++e)
                    for (int64_t d = 0; d < K; ++d)
                        stt(state, so * sstride + h * hstride + e * vstride + d * dstride,
                            S[e * K + d]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_gated_rmsnorm */
/* out = x * rsqrt(mean_channels(x^2) + eps) * w * act(z), one row per (token, head). The prefill
 * path's form of the epilogue the recurrent update folds in. */
/* The gate has a third spelling the shared rowoff() cannot see. [tokens, heads, ch] it walks, and
 * a flat [rows, ch] projection is its product; but a COLUMN SLICE of a wider projection row -- the
 * fp8 delta net fuses [q|k|v|z] into one GEMM -- has heads*ch as its last axis, so rowoff reads it
 * as a dense row and lands in the next token. Spelled out here rather than generalised there,
 * because it is a property of THIS operand and not of every one. */
static inline int64_t gate_rowoff(const RadTensor* t, int64_t r, int64_t ch) {
    if (t->rank == 2 && ch > 0 && t->shape[1] != ch && t->shape[1] % ch == 0) {
        const int64_t heads = t->shape[1] / ch;
        return (r / heads) * t->stride[0] + (r % heads) * ch * t->stride[1];
    }
    return rowoff(t, r, ch);
}

int ref_gdn_gated_rmsnorm(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *z = &a->t[1], *w = &a->t[2], *o = &a->t[3];
    const int64_t ch = p_int(a, "channels", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (ch <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / ch;
    if (rows <= 0 || numel(o) < rows * ch || numel(z) < rows * ch || numel(w) < ch)
        return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, ch), zb = gate_rowoff(z, r, ch), ob = rowoff(o, r, ch);
        const int64_t xs = laststride(x), zs = laststride(z), os = laststride(o);
        const int64_t ws = laststride(w);
        float ss = 0.0f;
        for (int64_t i = 0; i < ch; ++i) { float u = ldt(x, xb + i * xs); ss += u * u; }
        const float sc = 1.0f / std::sqrt(ss / (float)ch + eps);
        for (int64_t i = 0; i < ch; ++i) {
            const float zz = ldt(z, zb + i * zs);
            stt(o, ob + i * os, ldt(x, xb + i * xs) * sc * ldt(w, i * ws) *
                                (act == 1 ? act_sigmoid(zz) : act_silu(zz)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_recurrent_update */
/* gdn_conv_update into a staging q, k and v, then gdn_recurrent_update over them: the fused op is
 * DEFINED as the pair, so the reference is the pair and nothing is restated. The head geometry the
 * staging needs comes off `o` ([T, Hv, head_v]) and `x` (2 Hg head_k + Hv head_v columns). */
int ref_gdn_conv_recurrent_update(const RadArgs* a, RadStream st) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *o = &a->t[15];
    if (!x->data || !o->data || o->rank != 3 || x->rank < 2) return RAD_E_SHAPE;
    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t T = o->shape[0], Hv = o->shape[1];
    if (K <= 0 || V <= 0 || T <= 0 || Hv <= 0 || o->shape[2] != V) return RAD_E_SHAPE;
    const int64_t conv_dim = x->shape[x->rank - 1];
    if (conv_dim <= Hv * V || (conv_dim - Hv * V) % (2 * K)) return RAD_E_SHAPE;
    const int64_t Hg = (conv_dim - Hv * V) / (2 * K);

    std::vector<uint16_t> qb((size_t)(T * Hg * K)), kb((size_t)(T * Hg * K)), vb((size_t)(T * Hv * V));
    auto stage = [&](uint16_t* d, int64_t h, int64_t w) {
        RadTensor t{};
        t.data = d; t.dtype = x->dtype; t.rank = 3;
        t.shape[0] = T; t.shape[1] = h; t.shape[2] = w;
        t.stride[0] = h * w; t.stride[1] = w; t.stride[2] = 1;
        return t;
    };
    const RadTensor tq = stage(qb.data(), Hg, K), tk = stage(kb.data(), Hg, K),
                    tv = stage(vb.data(), Hv, V);

    RadParam cp[4] = {};
    const char* ck[4] = { "q_len", "head_k", "head_v", "conv_width" };
    for (int i = 0; i < 4; ++i) {
        const RadParam* f = rad_param_find(a->p, a->n_p, ck[i]);
        if (!f) return RAD_E_INVAL;
        cp[i] = *f;
    }
    /* x, w, b, conv_state, state_idx, num_accepted, cu, q, k, v */
    RadTensor ct[10] = { a->t[0], a->t[1], a->t[2], a->t[3], a->t[4], a->t[11], a->t[12], tq, tk,
                         tv };
    RadArgs ca = *a;
    ca.t = ct; ca.n_t = 10; ca.p = cp; ca.n_p = 4;
    int rc = ref_gdn_conv_update(&ca, st);
    if (rc != RAD_OK) return rc;

    std::vector<RadParam> rp;
    for (int i = 0; i < a->n_p; ++i)
        if (std::strcmp(a->p[i].key, "conv_width")) rp.push_back(a->p[i]);
    /* q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu, z, norm_w, o, o_q,
     * o_scale */
    RadTensor rt[16] = { tq, tk, tv, a->t[5], a->t[6], a->t[7], a->t[8], a->t[9], a->t[10],
                         a->t[11], a->t[12], a->t[13], a->t[14], a->t[15],
                         a->n_t > 16 ? a->t[16] : RadTensor{}, a->n_t > 17 ? a->t[17] : RadTensor{} };
    RadArgs ra = *a;
    ra.t = rt; ra.n_t = 16; ra.p = rp.data(); ra.n_p = (int)rp.size();
    return ref_gdn_recurrent_update(&ra, st);
}
