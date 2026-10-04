/* avx_gdn.cpp -- the gated delta net: the two convolution forms, the triangular solve, the chunked
 * scan, the decode recurrence, and the gated norm.
 *
 * ============================== WHAT IS ACTUALLY SLOW HERE ==============================
 *
 * These are not elementwise ops and the win is not the vector width. Every one of them contracts a
 * HEAD -- 128 elements of k against 128 of q, or a [128, 128] recurrent state against a key -- and
 * libref reads both sides of every one of those products through `ldt`, which is a dtype switch per
 * element. In the quadratic ops that is the whole cost:
 *
 *   gdn_kkt_solve  reads k[ti] and k[tj] for every (i, j) pair of a chunk. At chunk 64 and
 *                  head_k 128 that is 64*64*2*128 = 1.0 M switched loads to compute 64*64 dots
 *                  over data that is 64*128 = 8 K elements. THE K ROWS OF A CHUNK ARE READ ONCE
 *                  HERE, into an f32 panel, and every dot is then a vector reduction over it.
 *
 *   gdn_chunk_scan does the same for q, k and v, and additionally re-reads `g` inside three
 *                  separate double loops. Same treatment: one panel per chunk.
 *
 * That is a change of constant, not of asymptotics -- the algorithms below ARE libref's, step for
 * step, because every one of them is a definition rather than a choice:
 *
 *   THE DECAY IS APPLIED PER ELEMENT as e^{g_i - g_j}, not split about a per-chunk reference. Only
 *   i > j is kept and g decreases along the chunk, so the difference is always <= 0 and the direct
 *   form cannot overflow at any span -- which the split form can, at spans this model reaches.
 *
 *   THE TRIANGULAR INVERSE IS A FORWARD SUBSTITUTION. I + M is unit lower triangular, so its
 *   inverse is unique and the substitution computes it exactly; libr4d's blocked WMMA merge agrees
 *   with it because there is only one answer to agree with.
 *
 *   THE CONV OUTPUT IS ROUNDED BEFORE THE L2 NORM. A fused kernel rounds to bf16 there because the
 *   pair of kernels it replaces does -- the conv writes a tensor and the prep reads it back -- and
 *   matching that is what lets the fused form stand in for the unfused pair on real weights. So the
 *   conv result is WRITTEN to q/k/v and READ BACK for the norm, rather than carried in registers.
 *
 *   THE L2 NORM HAS NO DIVISION BY THE HEAD WIDTH. It is a norm, not an RMS:
 *   rsqrt(sum(x^2) + l2_eps), with l2_eps 1e-6, which is FLA's and libr4d's L2NORM_EPS.
 *
 * ============================== THE ROLLBACK CONTRACT ==============================
 *
 * A linear-attention kernel pair must support speculative rejection WITHOUT RECOMPUTE (spec §10).
 * The recurrent state and the conv window have already absorbed the rejected tokens, so
 * `num_accepted` is a required operand and a rejection is a CHANGE OF READ OFFSET:
 *
 *   conv state   read at `num_accepted - 1`, rewritten shifted DOWN BY ONE so slots
 *                0..conv_width-3 take the tail of the old history and this step's token t lands at
 *                slot t + conv_width - 2.
 *   linear state read from state_idx[s, num_accepted - 1]; EVERY candidate token writes its own
 *                state to state_idx[s, t], because which of them survives verification is not
 *                known until after this layer has run.
 *
 * `num_accepted` IS ONE-BASED: it counts the tokens the previous step committed, so 1 is an
 * ordinary decode that drafted nothing and 0 is not a value either op can read.
 *
 * THE OP DECLARES WHICH STATE CONTRACT IT RUNS UNDER, in its `state_form` parameter. libref and
 * this plugin implement "per_candidate", the reading above; libr4d also serves "anchor" -- one
 * committed state plus a small replay block, two slots a sequence instead of 1 + n_spec. The two
 * produce the same `o` from different states, so both host plugins decline "anchor" by name
 * rather than run the per-candidate reading over a cache laid out the other way. libref is the
 * oracle for this plugin under per_candidate, and that is what rad-avx-check exercises.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <vector>

using namespace avx;

/* THE VIEW HELPERS ARE `view2` / `view3` / `view4` AND NOT `v2` / `v3` / `v4`, which is what
 * libref calls them. The per-level inline namespace in avx_vec.h is named `sc` / `v1` / `v2` / `v3`
 * (AVX_SUFFIX), so a file-scope function called `v3` is ambiguous with the namespace at AVX-512 and
 * with a different one at each other level -- three of the four levels would fail to compile and
 * the fourth would build cleanly, so a build of that one level alone would not show it.
 *
 * The deepest convolution history held in a stack frame. libr4d's width is 4 and no linear-
 * attention model in circulation exceeds it by much; a wider request returns RAD_E_UNSUPPORTED by
 * name rather than convolving a truncated window, which would be numerically plausible and
 * completely wrong. Same number as libref's, so the two refuse the same calls. */
enum { AVX_CONV_HIST_MAX = 16 };

struct V3 { const RadTensor* t; int64_t s0, s1, s2; };
struct V2 { const RadTensor* t; int64_t s0, s1; };
struct V4 { const RadTensor* t; int64_t s0, s1, s2, s3; };

static inline V3 view3(const RadTensor* t, int64_t d1, int64_t d2) {
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
 * its own rather than d1*d2. A rank-3 spelling carries all three strides; a rank-2 one carries the
 * token stride in stride[0] and view3() would have guessed d1*d2 for it. */
static inline V3 view3_rows(const RadTensor* t, int64_t d1, int64_t d2) {
    if (t->rank != 2) return view3(t, d1, d2);
    return V3{ t, t->stride[0], d2 * t->stride[1], t->stride[1] };
}
static inline V2 view2(const RadTensor* t, int64_t d1) {
    V2 v{ t, d1, 1 };
    if (t->rank >= 2) { v.s0 = t->stride[t->rank - 2]; v.s1 = t->stride[t->rank - 1]; }
    return v;
}
static inline int64_t at3(const V3& v, int64_t i, int64_t j, int64_t k) {
    return i * v.s0 + j * v.s1 + k * v.s2;
}
static inline int64_t at2(const V2& v, int64_t i, int64_t j) { return i * v.s0 + j * v.s1; }

/* [N, heads, head_v, head_k] -- the shape both the chunked scan's h0/ht and the paged recurrent
 * state have. THE SLOT AND HEAD STRIDES ARE READ OFF THE TENSOR and not derived from the shape: a
 * paged state cache is padded so its page matches the attention page, and a kernel that derived
 * the stride would read every slot but the first at the wrong offset -- which looks exactly like a
 * correct kernel producing noise. */
static inline V4 view4(const RadTensor* t, int64_t d1, int64_t d2, int64_t d3) {
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

static inline int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }

/* The conv state is [n_slots, conv_dim, state_len]; a rank the caller flattened is read as the
 * tight layout that shape implies. */
static inline V3 conv_state_view(const RadTensor* cs, int64_t conv_dim, int64_t width,
                                 int64_t* state_len) {
    *state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    return view3(cs, conv_dim, *state_len);
}

/* softplus in the two-sided form, with the large-x cutoff FLA and libr4d both use: above the
 * threshold log1p(e^x) IS x to every bit f32 holds, and evaluating it costs an overflow. */
static inline float softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

static inline float dotf(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero();
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
        a1 = vf_fma(vf_loadu(a + i + VF_N), vf_loadu(b + i + VF_N), a1);
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
    float s = vf_hsum(vf_add(a0, a1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* dst += c * src */
/* Widen-transpose a [cbw, width] tap block starting at channel c0 into tapT[j*cbw + c]:
 * the token loops read tap COLUMNS, and this makes them contiguous. 8 rows at a time:
 * vector-widen into a 32-float staging block, then scatter the 32 values into the four
 * columns (~15ns a block, L1-hot). Loads/stores only, so bit-exact; anything the vector
 * form cannot spell -- another width, unpacked rows, an exotic dtype, levels 0/1 -- and
 * the ragged tail take the scalar form below it. */
static inline void fill_tapT(const RadTensor* w, V2 wv, int64_t c0, int64_t cbw, int64_t width,
                             float* tapT) {
#if AVX_LEVEL >= 2
    if (width == 4 && wv.s0 == 4 && wv.s1 == 1 &&
        (w->dtype == RAD_F32 || w->dtype == RAD_BF16)) {
        const bool f32 = (w->dtype == RAD_F32);
        int64_t c = 0;
        for (; c + 8 <= cbw; c += 8) {
            alignas(32) float stg[8 * 4];
            if (f32) {
                const float* base = (const float*)w->data + (c0 + c) * 4;
                _mm256_store_ps(stg, _mm256_loadu_ps(base));
                _mm256_store_ps(stg + 8, _mm256_loadu_ps(base + 8));
                _mm256_store_ps(stg + 16, _mm256_loadu_ps(base + 16));
                _mm256_store_ps(stg + 24, _mm256_loadu_ps(base + 24));
            } else {
                /* bf16 -> f32 is a zero-extend and a shift, NOT cvtph (which reads FP16).
                 * Same values as cvt_bf16_to_f32, same order. */
                const uint16_t* base = (const uint16_t*)w->data + (c0 + c) * 4;
                for (int k = 0; k < 4; ++k) {
                    __m128i h = _mm_loadu_si128((const __m128i*)(base + k * 8));
                    __m256i e = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
                    _mm256_store_ps(stg + k * 8, _mm256_castsi256_ps(e));
                }
            }
            for (int64_t j = 0; j < 4; ++j)
                for (int64_t k = 0; k < 8; ++k) tapT[j * cbw + c + k] = stg[k * 4 + j];
        }
        for (; c < cbw; ++c)
            for (int64_t j = 0; j < width; ++j)
                tapT[j * cbw + c] = rad_load_f32(w->data, w->dtype, at2(wv, c0 + c, j));
        return;
    }
#endif
    for (int64_t c = 0; c < cbw; ++c)
        for (int64_t j = 0; j < width; ++j)
            tapT[j * (size_t)cbw + c] = ldt(w, at2(wv, c0 + c, j));
}

static inline void axpy(float* dst, const float* src, float c, int64_t n) {
    const vf vc = vf_set1(c);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(dst + i, vf_fma(vc, vf_loadu(src + i), vf_loadu(dst + i)));
    for (; i < n; ++i) dst[i] += c * src[i];
}

/* A [rows, width] panel of an operand read as [token, head, d], widened into f32 once. This is the
 * whole optimisation in the quadratic ops -- see the file header. */
static void panel3(const RadTensor* t, const V3& v, int64_t t0, int64_t rows, int64_t h,
                   int64_t width, float* dst) {
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = at3(v, t0 + r, h, 0);
        if (v.s2 == 1) row_to_f32(t->dtype, byte_at(t->data, t->dtype, b), dst + r * width, width);
        else for (int64_t d = 0; d < width; ++d) dst[r * width + d] = ldt(t, b + d * v.s2);
    }
}

/* One token's convolution output over a channel panel [c0, c0 + cbw), split into q, k and v by
 * region and written to each at its head AND ITS OFFSET WITHIN THE HEAD. A panel is 1024 channels
 * and a head is K or V of them, so a panel edge falls mid-head whenever the head width does not
 * divide 1024 -- 96 or 192, say -- and the piece that starts there begins part way into a head
 * and may end part way into another. Each run below is cut at the next head boundary, so it is
 * always contiguous within one head's row. */
static inline void store_pieces(const float* acc, int64_t cbw, int64_t c0, int64_t bos, int64_t t,
                                const RadTensor* q, const RadTensor* k, const RadTensor* v,
                                V3 qv, V3 kv, V3 vv, int64_t Hg, int64_t K, int64_t V,
                                int64_t conv_dim) {
    const RadTensor* outs[3] = { q, k, v };
    const V3 ovs[3] = { qv, kv, vv };
    const int64_t edges[4] = { 0, Hg * K, 2 * Hg * K, conv_dim };
    for (int r = 0; r < 3; ++r) {
        const int64_t lo = c0 > edges[r] ? c0 : edges[r];
        const int64_t hi = c0 + cbw < edges[r + 1] ? c0 + cbw : edges[r + 1];
        const int64_t kd = (r < 2) ? K : V;
        for (int64_t ch = lo; ch < hi;) {
            const int64_t oh = (ch - edges[r]) / kd, d0 = (ch - edges[r]) % kd;
            const int64_t len = hi - ch < kd - d0 ? hi - ch : kd - d0;
            if (ovs[r].s2 == 1)
                row_from_f32(outs[r]->dtype, acc + (ch - c0),
                             (void*)byte_at(outs[r]->data, outs[r]->dtype,
                                            at3(ovs[r], bos + t, oh, d0)),
                             len);
            else
                for (int64_t d = 0; d < len; ++d)
                    stt(outs[r], at3(ovs[r], bos + t, oh, d0 + d), acc[ch - c0 + d]);
            ch += len;
        }
    }
}

/* ------------------------------------------------------------------ gdn_conv_prep */
/* Everything between the qkv projection and the chunked scan: the width-`conv_width` depthwise
 * causal convolution with silu and its state cache, the q/k/v split, the l2 norm over each q and k
 * head, the gate with its per-chunk cumsum, and beta. */
AVX_KERNEL(gdn_conv_prep) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[5], *cu = &a->t[6];
    const RadTensor *q = &a->t[11], *k = &a->t[12], *v = &a->t[13];
    const RadTensor *g = &a->t[14], *beta = &a->t[15];
    const RadTensor* bias = rad_arg_in(a, 2);
    const RadTensor* A_log = rad_arg_in(a, 3);
    const RadTensor* dt_bias = rad_arg_in(a, 4);
    const RadTensor* ta = rad_arg_in(a, 7);
    const RadTensor* tb = rad_arg_in(a, 8);
    const RadTensor* sidx = rad_arg_in(a, 9);
    const RadTensor* hini = rad_arg_in(a, 10);
    if (!x->data || !w->data || !cu->data || !q->data || !k->data || !v->data ||
        !g->data || !beta->data) return RAD_E_INVAL;
    /* One of the two, not one of each: a caller that names the gate has to name beta as well, or
     * the second read silently falls back into a fused row that may not be there. */
    if ((ta != nullptr) != (tb != nullptr)) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* `chunk` is RAD_DERIVED and NOTHING HERE CARRIES IT: the gate cumsum is cut at the chunk
     * boundary and no operand's shape records where that is. A caller who omitted it gets
     * RAD_E_SHAPE by name; the alternative -- treating the sequence as one chunk -- is a decay
     * summed over the wrong span, numerically plausible and completely wrong. */
    const int64_t chunk = p_int(a, "chunk", 0), width = p_int(a, "conv_width", 0);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    if (K <= 0 || V <= 0 || chunk <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > AVX_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;                       /* an empty step is not an error */
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;
    if (numel(g) < T * H || numel(beta) < T * H) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : (conv_dim + (ta ? 0 : 2 * H));
    const int64_t xcol = laststride(x);
    /* THE BOUND IS THE VIEW'S AND NOT THE BUFFER'S. `x` may be a column slice of a wider row, and
     * a slice's numel counts only the elements it names -- so the flat product would refuse a call
     * that is perfectly in range. Rank 2 or more spells the extent in the shape. */
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

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V);
    const V2 wv = view2(w, width);
    const V2 gv = view2(g, H), bv = view2(beta, H);
    const V2 av = ta ? view2(ta, H) : V2{ nullptr, 0, 0 };
    const V2 bgv = tb ? view2(tb, H) : V2{ nullptr, 0, 0 };

    /* ---- phase 1: the convolution, straight into q / k / v ---------------------------------- */
    /* TOKENS OUTER, CHANNELS VECTOR. The other nest -- each channel walked down the tokens --
     * reads x strided (a cache line per bf16) and issues a dtype switch plus a scalar exp per
     * element. Walking tokens outer over channel panels turns the tap loop into vector FMA
     * over contiguous rows. Per element the order is unchanged -- bias, taps ascending, silu
     * last -- and the multiply and add stay SEPARATE ops: an FMA would fuse one rounding the
     * reference keeps. The silu goes through vf_silu (the exact minimax, ~1 ulp from libm),
     * which the bf16 store that follows cannot carry. Panels are 1024 channels, and a panel
     * edge may fall inside a head; store_pieces places each piece at its offset in the head. */
    const int64_t CB = 1024;
    const int64_t npan = (conv_dim + CB - 1) / CB;
    AVX_PARALLEL_FOR_IF(N * npan >= 4)
    for (int64_t ncb = 0; ncb < N * npan; ++ncb) {
        const int64_t n = ncb / npan, pan = ncb % npan;
        const int64_t c0 = pan * CB, cbw = (c0 + CB < conv_dim) ? CB : conv_dim - c0;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
        /* A zero `has_init` means the slot holds no history: the window in front of the
         * sequence's first token is ZEROS, not whatever the slot last held. Without it a
         * fresh sequence landing on a recycled slot convolves the previous sequence's tail
         * into its first conv_width-1 tokens, which is fluent output with the wrong prefix. */
        const bool warm = !hini || ldi(hini, offlin(hini, n)) != 0;

        float* acc = scratch_f32((size_t)cbw);
        float* xr = scratch_f32_b((size_t)cbw);
        /* Taps transposed to [width][cbw], widened once per panel: the token loop reads tap
         * columns, which are contiguous in this layout. Launch-constant, thread-local. */
        float* tapT = scratch_f32_c((size_t)width * (size_t)cbw);
        fill_tapT(w, wv, c0, cbw, width, tapT);
        /* Bias, widened once per panel into the raw slot (the three f32 slots are taken);
         * each token starts its accumulator as a copy of this. */
        float* bs = (float*)scratch_raw((size_t)cbw * sizeof(float) + 64);
        if (bias) {
            if (laststride(bias) == 1)
                row_to_f32(bias->dtype, byte_at(bias->data, bias->dtype, c0), bs, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c) bs[c] = ldt(bias, (c0 + c) * laststride(bias));
        } else {
            for (int64_t c = 0; c < cbw; ++c) bs[c] = 0.0f;
        }

        for (int64_t t = 0; t < slen; ++t) {
            /* Fresh accumulator per token: bias copy, then taps ascending. */
            int64_t c = 0;
            for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_loadu(bs + c));
            for (; c < cbw; ++c) acc[c] = bs[c];
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                if (src >= 0) {
                    if (xcol == 1)
                        row_to_f32(x->dtype,
                                   byte_at(x->data, x->dtype, (bos + src) * xrow + c0), xr,
                                   cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(x, (bos + src) * xrow + (c0 + c) * xcol);
                } else if (warm && cs->data && slot >= 0) {
                    if (cvs.s1 == 1)
                        row_to_f32(cs->dtype,
                                   byte_at(cs->data, cs->dtype, at3(cvs, slot, c0, src + width - 1)),
                                   xr, cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(cs, at3(cvs, slot, c0 + c, src + width - 1));
                } else {
                    for (int64_t c = 0; c < cbw; c += VF_N) vf_storeu(xr + c, vf_zero());
                    for (int64_t c = (cbw / VF_N) * VF_N; c < cbw; ++c) xr[c] = 0.0f;
                }
                /* Separate mul and add -- NOT an FMA -- so the rounding matches the reference's
                 * scalar loop bit for bit. */
                const float* tc = tapT + j * cbw;
                c = 0;
                for (; c + VF_N <= cbw; c += VF_N)
                    vf_storeu(acc + c, vf_add(vf_mul(vf_loadu(xr + c), vf_loadu(tc + c)),
                                              vf_loadu(acc + c)));
                for (; c < cbw; ++c) acc[c] += xr[c] * tc[c];
            }
            /* Silu in place, then split into q/k/v, through the same helper the decode entry
             * uses (below), which places a piece at its head AND its offset within the head. */
            c = 0;
            for (; c + VF_N <= cbw; c += VF_N)
                vf_storeu(acc + c, vf_silu(vf_loadu(acc + c)));
            for (; c < cbw; ++c) acc[c] /= (1.0f + std::exp(-acc[c]));
            store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
        }
    }

    /* ---- phase 2: the l2 norm over each q and k head, reading the ROUNDED conv output -------- */
    AVX_PARALLEL_FOR_IF(T * Hg * 2 >= 32)
    for (int64_t th = 0; th < T * Hg * 2; ++th) {
        const int64_t which = th / (T * Hg);
        const int64_t rest = th % (T * Hg);
        const int64_t t = rest / Hg, h = rest % Hg;
        const RadTensor* z = which ? k : q;
        const V3& zv = which ? kv : qv;
        const int64_t b = at3(zv, t, h, 0);
        float* buf = scratch_f32((size_t)K);
        const float* u = (zv.s2 == 1) ? row_f32_in(z->dtype, byte_at(z->data, z->dtype, b), buf, K)
                                      : buf;
        if (zv.s2 != 1) for (int64_t d = 0; d < K; ++d) buf[d] = ldt(z, b + d * zv.s2);
        /* No division by K: this is an L2 norm, not an RMS. */
        const float inv = 1.0f / std::sqrt(row_sumsq(u, K) + l2eps);
        if (zv.s2 == 1) {
            float* o2 = scratch_f32_b((size_t)K);
            const vf vi = vf_set1(inv);
            int64_t d = 0;
            for (; d + VF_N <= K; d += VF_N) vf_storeu(o2 + d, vf_mul(vf_loadu(u + d), vi));
            for (; d < K; ++d) o2[d] = u[d] * inv;
            out_row(z, b, K, o2);
        } else {
            for (int64_t d = 0; d < K; ++d) stt(z, b + d * zv.s2, buf[d] * inv);
        }
    }

    /* ---- phase 3: the gate, its per-chunk cumsum, and beta ----------------------------------- */
    /* The cumsum is per CHUNK and per sequence, because that is what the scan consumes: g[j] is the
     * decay from the start of j's chunk up to and including j. Sequential in t by construction --
     * a running sum is the one shape no vectorisation reaches, and the exp/softplus per token is
     * what it costs. */
    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        const float negexp = -std::exp(alog);
        float run = 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            if (t % chunk == 0) run = 0.0f;
            const float av_ = ta ? ldt(ta, at2(av, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + h) * xcol);
            const float bvl = tb ? ldt(tb, at2(bgv, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + H + h) * xcol);
            run += negexp * softplus(av_ + dtb, spthr);
            stt(g, at2(gv, bos + t, h), run);
            stt(beta, at2(bv, bos + t, h), 1.0f / (1.0f + std::exp(-bvl)));
        }
    }

    /* ---- phase 4: rewrite the conv state from the sequence's last width-1 tokens -------------- */
    /* After phase 1, so the history it overwrites has already been read. A sequence shorter than
     * width-1 keeps what the old state held in front of its tokens, which is what makes chunked
     * prefill work: the second chunk of a sequence continues from the first. */
    if (cs->data && width > 1) {
        AVX_PARALLEL_FOR_IF(N * conv_dim >= 32)
        for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
            const int64_t n = nc / conv_dim, c = nc % conv_dim;
            const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
            const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
            if (slen <= 0 || slot < 0) continue;
            const bool warm = !hini || ldi(hini, offlin(hini, n)) != 0;
            float keep[AVX_CONV_HIST_MAX];
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

/* Silu in place over a channel panel, then split into q/k/v by region. Shared by the two
 * convolution entry points; the split itself is store_pieces, above both. */
static inline void silu_store_pieces(float* acc, int64_t cbw, int64_t c0, int64_t bos, int64_t t,
                                     const RadTensor* q, const RadTensor* k, const RadTensor* v,
                                     V3 qv, V3 kv, V3 vv, int64_t Hg, int64_t K, int64_t V,
                                     int64_t conv_dim) {
    /* Exact silu only: the cheap form (vf_silu_fast, as silu_mul uses), keyed on all-bf16
     * outputs, measures unmeasurably different here -- the token loop is frontend-bound and the
     * extra branch buys nothing. Do not switch to it without measuring. */
    int64_t c = 0;
    for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_silu(vf_loadu(acc + c)));
    for (; c < cbw; ++c) acc[c] /= (1.0f + std::exp(-acc[c]));
    store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
}

/* ------------------------------------------------------------------ gdn_conv_update */
/* The same convolution for a decode step, against the rolling window. The two things that matter
 * are that the READ starts at num_accepted-1 and that the rewrite shifts the buffer DOWN BY ONE so
 * the next step can do the same -- a rejection is a change of read offset and not a recompute,
 * which is the whole reason this is a separate entry point rather than a flag on the prefill one.
 *
 * No l2 norm here: on the decode path the recurrent update does it, which is where libr4d puts it. */
AVX_KERNEL(gdn_conv_update) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[3], *cu = &a->t[6];
    const RadTensor *q = &a->t[7], *k = &a->t[8], *v = &a->t[9];
    const RadTensor* bias = rad_arg_in(a, 2);
    const RadTensor* sidx = rad_arg_in(a, 4);
    const RadTensor* nacc = rad_arg_in(a, 5);
    if (!x->data || !w->data || !cs->data || !cu->data || !q->data || !k->data || !v->data)
        return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t width = p_int(a, "conv_width", 0);
    if (K <= 0 || V <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > AVX_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_OK;
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : conv_dim;
    const int64_t xcol = laststride(x);
    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);

    /* Checked rather than clamped: a cache too shallow for this step's offset is a caller bug
     * whose symptom would otherwise be a silently truncated history. Before the loop, because an
     * OpenMP body cannot return. */
    for (int64_t n = 0; n < N; ++n) {
        const int64_t slen = cu_at(cu, n + 1) - cu_at(cu, n);
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, n)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + width - 1 > state_len) return RAD_E_SHAPE;
        if (width > 1 && (width - 2) + slen > state_len) return RAD_E_SHAPE;
    }

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V);
    const V2 wv = view2(w, width);

    /* Tokens outer, channels vector -- the prefill entry's form. The channel-outer nest reads a
     * cache line per bf16 and pays a switch and a scalar exp per element; at decode slen is 1 but
     * conv_dim is 10240, so the problem is the same one. */
    const int64_t CB = 1024;
    const int64_t npan = (conv_dim + CB - 1) / CB;
    AVX_PARALLEL_FOR_IF(N * npan >= 4)
    for (int64_t ncb = 0; ncb < N * npan; ++ncb) {
        const int64_t n = ncb / npan, pan = ncb % npan;
        const int64_t c0 = pan * CB, cbw = (c0 + CB < conv_dim) ? CB : conv_dim - c0;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
        if (slot < 0) continue;
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, n)) - 1 : 0;

        float* acc = scratch_f32((size_t)cbw);
        float* xr = scratch_f32_b((size_t)cbw);
        float* tapT = scratch_f32_c((size_t)width * (size_t)cbw);
        fill_tapT(w, wv, c0, cbw, width, tapT);
        /* ONE raw block, partitioned by hand (single arena slot): bias panel, then the
         * (width-1) history rows over this panel, widened once -- only the first width-1
         * tokens read history. */
        float* raw = (float*)scratch_raw(((size_t)cbw + (size_t)(width - 1) * (size_t)cbw) *
                                         sizeof(float) + 64);
        float* bs = raw;
        float* hseg = raw + cbw;
        if (bias) {
            if (laststride(bias) == 1)
                row_to_f32(bias->dtype, byte_at(bias->data, bias->dtype, c0), bs, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c) bs[c] = ldt(bias, (c0 + c) * laststride(bias));
        } else {
            for (int64_t c = 0; c < cbw; ++c) bs[c] = 0.0f;
        }
        for (int64_t i = 0; i < width - 1; ++i) {
            if (cvs.s1 == 1)
                row_to_f32(cs->dtype, byte_at(cs->data, cs->dtype, at3(cvs, slot, c0, off + i)),
                           hseg + i * cbw, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c)
                    hseg[i * cbw + c] = ldt(cs, at3(cvs, slot, c0 + c, off + i));
        }
        for (int64_t t = 0; t < slen; ++t) {
            int64_t c = 0;
            for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_loadu(bs + c));
            for (; c < cbw; ++c) acc[c] = bs[c];
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                if (src >= 0) {
                    if (xcol == 1)
                        row_to_f32(x->dtype,
                                   byte_at(x->data, x->dtype, (bos + src) * xrow + c0), xr,
                                   cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(x, (bos + src) * xrow + (c0 + c) * xcol);
                } else {
                    /* src + width - 1 lies in [0, width - 2]: the history segment above. */
                    const float* hs = hseg + (src + width - 1) * cbw;
                    for (int64_t c = 0; c < cbw; ++c) xr[c] = hs[c];
                }
                /* Separate mul and add -- NOT an FMA -- matching the scalar reference's two
                 * roundings. */
                const float* tc = tapT + j * cbw;
                c = 0;
                for (; c + VF_N <= cbw; c += VF_N)
                    vf_storeu(acc + c, vf_add(vf_mul(vf_loadu(xr + c), vf_loadu(tc + c)),
                                              vf_loadu(acc + c)));
                for (; c < cbw; ++c) acc[c] += xr[c] * tc[c];
            }
            silu_store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
        }
        /* Shift down by one: the tail of the history this step READ -- the window at
         * `off`, not the one at zero -- then this step's tokens. Row memcpys where both
         * sides are channel-contiguous and the dtypes match (a move, not a conversion);
         * the scalar form otherwise. Slot i takes slot off + i + 1, which no earlier
         * iteration has written, so the ascending walk is safe in place. */
        if (width > 1) {
            const uint32_t csbits = rad_dtype_bits(cs->dtype);
            const size_t elbytes = (size_t)csbits / 8;
            /* memmove, not memcpy: the shift reads row off+i+1 after row i was written, so the
             * ranges can overlap when the history stride is short. Byte dtypes only -- a
             * sub-byte element has no whole-byte row to move. */
            if (cvs.s1 == 1 && xcol == 1 && x->dtype == cs->dtype && csbits % 8 == 0 &&
                elbytes > 0) {
                for (int64_t i = 0; i + 1 < width - 1; ++i)
                    std::memmove((char*)cs->data + (at3(cvs, slot, c0, i) * (int64_t)elbytes),
                                 (const char*)cs->data +
                                     (at3(cvs, slot, c0, off + i + 1) * (int64_t)elbytes),
                                 (size_t)cbw * elbytes);
                for (int64_t t = 0; t < slen; ++t)
                    std::memmove((char*)cs->data +
                                     (at3(cvs, slot, c0, t + width - 2) * (int64_t)elbytes),
                                 (const char*)x->data +
                                     (((bos + t) * xrow + c0) * (int64_t)elbytes),
                                 (size_t)cbw * elbytes);
            } else {
                for (int64_t c = 0; c < cbw; ++c) {
                    for (int64_t i = 0; i + 1 < width - 1; ++i)
                        stt(cs, at3(cvs, slot, c0 + c, i),
                            ldt(cs, at3(cvs, slot, c0 + c, off + i + 1)));
                    for (int64_t t = 0; t < slen; ++t)
                        stt(cs, at3(cvs, slot, c0 + c, t + width - 2),
                            ldt(x, (bos + t) * xrow + (c0 + c) * xcol));
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_kkt_solve */
/* A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}, per chunk, per v head.
 *
 * THE K ROWS OF A CHUNK ARE WIDENED ONCE. libref reads k[ti] and k[tj] through `ldt` for every
 * (i, j) pair, which at chunk 64 and head_k 128 is a million switched loads over eight thousand
 * distinct elements. One panel per chunk makes every dot a vector reduction, and `g` and `beta`
 * are pulled into the same frame for the same reason. */
AVX_KERNEL(gdn_kkt_solve) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *beta = &a->t[1], *g = &a->t[2], *cu = &a->t[3], *A = &a->t[4];

    const int64_t K = p_int(a, "head_k", 0);
    /* RAD_DERIVED: A is [T, heads, chunk] and the caller sized it, which is this plugin's rule for
     * every other extent too. */
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

    const V3 kv = view3(k, Hg, K), Av = view3(A, H, chunk);
    const V2 gv = view2(g, H), bv = view2(beta, H);

    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* ONE scratch call, carved by hand: a second may reallocate and invalidate the first. */
        const size_t need = (size_t)(chunk * chunk + chunk + chunk * K + 2 * chunk + chunk) + 16;
        float* wsp = scratch_f32(need);
        float* Am = wsp;                          /* [chunk, chunk] the inverse, in f32 */
        float* row = Am + chunk * chunk;          /* [chunk] one row of M */
        float* kp = row + chunk;                  /* [chunk, K] the chunk's k rows */
        float* gp = kp + chunk * K;               /* [chunk] */
        float* bp = gp + chunk;                   /* [chunk] */
        float* ev = bp + chunk;                   /* [chunk] e^{g_i - g_j} for this i */

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            panel3(k, kv, bos + c0, L, hk, K, kp);
            for (int64_t i = 0; i < L; ++i) {
                gp[i] = ldt(g, at2(gv, bos + c0 + i, h));
                bp[i] = ldt(beta, at2(bv, bos + c0 + i, h));
            }
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                /* e^{g_i - g_j} over the row, vectorised (vf_exp ~1 ulp from libm -- the
                 * gate's question): one L1 pass instead of i scalar exps. */
                {
                    const vf vg = vf_set1(gp[i]);
                    int64_t j = 0;
                    for (; j + VF_N <= i; j += VF_N)
                        vf_storeu(ev + j, vf_exp(vf_sub(vg, vf_loadu(gp + j))));
                    for (; j < i; ++j) ev[j] = std::exp(gp[i] - gp[j]);
                }
                for (int64_t j = 0; j < i; ++j)
                    row[j] = bp[i] * dotf(kp + i * K, kp + j * K, K) * ev[j];
                /* (I + M) A = I, row by row: A[i][j] = delta - sum_{j<=l<i} M[i][l] A[l][j].
                 * l-outer so the inner run over j is contiguous vector work (Am rows are).
                 * Each (i, j) lane still subtracts l ascending, and mul+sub rather than FMA
                 * keeps the rounding identical to the scalar loop -- bit-identical, only
                 * wider. Terms with l < j read Am[l][j] = 0, so extending the sum down to 0
                 * changes nothing. */
                for (int64_t j = 0; j <= i; ++j) Am[i * chunk + j] = (i == j) ? 1.0f : 0.0f;
                for (int64_t l = 0; l < i; ++l) {
                    const vf rl = vf_set1(row[l]);
                    const float* Aml = Am + l * chunk;
                    float* Ami = Am + i * chunk;
                    int64_t j = 0;
                    for (; j + VF_N <= i + 1; j += VF_N)
                        vf_storeu(Ami + j,
                                  vf_sub(vf_loadu(Ami + j), vf_mul(rl, vf_loadu(Aml + j))));
                    for (; j <= i; ++j) Ami[j] -= row[l] * Aml[j];
                }
                for (int64_t j = i + 1; j < chunk; ++j) Am[i * chunk + j] = 0.0f;
                if (Av.s2 == 1)
                    row_from_f32(A->dtype, Am + i * chunk,
                                 (void*)byte_at(A->data, A->dtype, at3(Av, ti, h, 0)), chunk);
                else
                    for (int64_t j = 0; j < chunk; ++j)
                        stt(A, at3(Av, ti, h, j), Am[i * chunk + j]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_chunk_scan */
AVX_KERNEL(gdn_chunk_scan) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *A = &a->t[3];
    const RadTensor *g = &a->t[4], *beta = &a->t[5], *cu = &a->t[7];
    const RadTensor *o = &a->t[8], *ht = &a->t[9];
    const RadTensor* h0 = rad_arg_in(a, 6);
    /* THE STATE SLOT, when the caller keeps its recurrent states in a pool the batch does not
     * index positionally. Absent, a row owns slot n. */
    const RadTensor* sidx = rad_arg_in(a, 10);
    if (!q->data || !k->data || !v->data || !A->data || !g->data || !beta->data ||
        !cu->data || !o->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
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

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V), ov = view3(o, H, V);
    const V3 Av = view3(A, H, chunk);
    const V2 gv = view2(g, H), bv = view2(beta, H);
    const V4 h0v = h0 ? view4(h0, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };
    const V4 htv = ht->data ? view4(ht, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };

    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* COLUMN 0 IS THE STATE. A row may be wider -- a decode kernel serving speculation asks
         * for scratch beside it -- and the scan writes the state and nothing else. */
        const int64_t sw = sidx && sidx->rank >= 1 ? sidx->stride[0] : 1;
        const int64_t sn = sidx ? ldi(sidx, n * sw) : n;
        if (sn < 0) continue;                  /* a padded row owns no state */

        /* One scratch request, carved by hand. The three panels (q, k, v) are what makes the
         * quadratic loops below vector work instead of a dtype switch per element. Ap/expv do
         * the same for the A weights and the gate exponentials: at chunk 64 the O-loop issues
         * L*L/2 scalar exps a chunk, which is most of the wall time at model shape. */
        const size_t need = (size_t)(V * K + 3 * chunk * V + 2 * chunk * K + 3 * chunk +
                                     V + K + chunk * chunk + chunk + K * V) + 32;
        float* wsp = scratch_f32(need);
        float* S = wsp;                            /* [V, K] the recurrent state */
        float* Wm = S + V * K;                     /* [chunk, V] */
        float* Vp = Wm + chunk * V;                /* [chunk, V] */
        float* row = Vp + chunk * V;               /* [chunk] one row of sA */
        float* qp = row + chunk;                   /* [chunk, K] */
        float* kp = qp + chunk * K;                /* [chunk, K] */
        float* vp = kp + chunk * K;                /* [chunk, V] */
        float* gp = vp + chunk * V;                /* [chunk] */
        float* bp = gp + chunk;                    /* [chunk] */
        /* Two more, and they get their own names rather than reusing a dead buffer: the output row
         * and the decayed key. Carving them out of `Wm`'s tail and out of `row` would be correct
         * only while chunk*V >= V and K <= chunk -- two conditions that hold at every shape this is
         * checked at and neither of which the op promises. A buffer aliased under a size
         * assumption fails first on a geometry nobody checked. */
        float* acc = bp + chunk;                   /* [V] the output row being formed */
        float* gk = acc + V;                       /* [K] e^{g_last - g_j} . k[j] */
        float* Ap = gk + K;                        /* [chunk, chunk] A weights, one row per i */
        float* expv = Ap + chunk * chunk;          /* [chunk] e^{g_j} */

        /* The initial state, widened a row at a time rather than switched per element -- the
         * recurrent update's form, same values. */
        if (h0 && h0v.s3 == 1) {
            for (int64_t e = 0; e < V; ++e)
                row_to_f32(h0->dtype,
                           byte_at(h0->data, h0->dtype, at4(h0v, sn, h, e, 0)), S + e * K, K);
        } else {
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    S[e * K + d] = h0 ? ldt(h0, at4(h0v, sn, h, e, d)) : 0.0f;
        }

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            panel3(q, qv, bos + c0, L, hk, K, qp);
            panel3(k, kv, bos + c0, L, hk, K, kp);
            panel3(v, vv, bos + c0, L, h, V, vp);
            for (int64_t j = 0; j < L; ++j) {
                gp[j] = ldt(g, at2(gv, bos + c0 + j, h));
                bp[j] = ldt(beta, at2(bv, bos + c0 + j, h));
            }
            const float glast = gp[L - 1];
            /* e^{g_j} once per j, vectorised: the W-loop and the decay below each spent a
             * scalar exp per (j) and per (i, j). vf_exp's minimax is ~1 ulp from libm --
             * a tolerance question the gate answers, not a second definition. */
            {
                int64_t j = 0;
                for (; j + VF_N <= L; j += VF_N)
                    vf_storeu(expv + j, vf_exp(vf_loadu(gp + j)));
                for (; j < L; ++j) expv[j] = std::exp(gp[j]);
            }
            /* The A rows for this chunk's tokens, widened once: the triangle below reads
             * A[ti, h, 0..i] 2k times a chunk through a dtype switch. */
            if (Av.s2 == 1) {
                for (int64_t i = 0; i < L; ++i)
                    row_to_f32(A->dtype,
                               byte_at(A->data, A->dtype, at3(Av, bos + c0 + i, h, 0)),
                               Ap + i * chunk, chunk);
            }

            /* W = beta . (V - e^g . (K S)). Two transposed-S k-outer forms -- Wm-streaming and
             * register-blocked over a 4-chain -- both measure slower than this dot loop. */
            for (int64_t j = 0; j < L; ++j) {
                const float ge = expv[j];
                for (int64_t e = 0; e < V; ++e)
                    Wm[j * V + e] =
                        bp[j] * (vp[j * V + e] - ge * dotf(kp + j * K, S + e * K, K));
            }
            /* V' = A W, with A unit lower triangular so only j <= i contributes. Accumulated over
             * e as a vector, which is the same j order libref sums in. */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                for (int64_t e = 0; e < V; ++e) Vp[i * V + e] = 0.0f;
                if (Av.s2 == 1) {
                    for (int64_t j = 0; j <= i; ++j)
                        axpy(Vp + i * V, Wm + j * V, Ap[i * chunk + j], V);
                } else {
                    for (int64_t j = 0; j <= i; ++j)
                        axpy(Vp + i * V, Wm + j * V, ldt(A, at3(Av, ti, h, j)), V);
                }
            }
            /* O = scale . e^{g_i} . (Q S^T) + sA V' */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float gi = gp[i];
                const float ge = std::exp(gi);
                /* e^{gi - g_j} over the row, vectorised, then the dots: splitting the exp
                 * out of the dot loop costs one L1 pass over 64 floats and removes L
                 * scalar exps per i. */
                {
                    const vf vg = vf_set1(gi), vs = vf_set1(scale);
                    int64_t j = 0;
                    for (; j + VF_N <= i + 1; j += VF_N)
                        vf_storeu(row + j,
                                  vf_mul(vs, vf_exp(vf_sub(vg, vf_loadu(gp + j)))));
                    for (; j <= i; ++j) row[j] = scale * std::exp(gi - gp[j]);
                }
                for (int64_t j = 0; j <= i; ++j)
                    row[j] *= dotf(qp + i * K, kp + j * K, K);
                for (int64_t e = 0; e < V; ++e)
                    acc[e] = scale * ge * dotf(qp + i * K, S + e * K, K);
                for (int64_t j = 0; j <= i; ++j) axpy(acc, Vp + j * V, row[j], V);
                if (ov.s2 == 1) out_row(o, at3(ov, ti, h, 0), V, acc);
                else for (int64_t e = 0; e < V; ++e) stt(o, at3(ov, ti, h, e), acc[e]);
            }
            /* S = e^{g_last} S + V'^T (e^{g_last - g_j} . K) */
            {
                const float el = std::exp(glast);
                for (int64_t e = 0; e < V; ++e) {
                    const vf ve = vf_set1(el);
                    int64_t d = 0;
                    for (; d + VF_N <= K; d += VF_N)
                        vf_storeu(S + e * K + d, vf_mul(vf_loadu(S + e * K + d), ve));
                    for (; d < K; ++d) S[e * K + d] *= el;
                }
                /* e^{g_last - g_j} over the chunk, vectorised into `row` (dead here): one
                 * pass instead of L scalar exps. */
                {
                    const vf vg = vf_set1(glast);
                    int64_t j = 0;
                    for (; j + VF_N <= L; j += VF_N)
                        vf_storeu(row + j, vf_exp(vf_sub(vg, vf_loadu(gp + j))));
                    for (; j < L; ++j) row[j] = std::exp(glast - gp[j]);
                }
                for (int64_t j = 0; j < L; ++j) {
                    const float gs = row[j];
                    /* gs . k[j] formed ONCE per j rather than per (e, d), and in THAT ASSOCIATION:
                     * libref writes `Vp * (gs * k)`, and f32 multiplication is not associative, so
                     * forming (Vp * gs) * k instead would move the answer in the last bits on
                     * every element of the state. */
                    const vf vg = vf_set1(gs);
                    int64_t d = 0;
                    for (; d + VF_N <= K; d += VF_N)
                        vf_storeu(gk + d, vf_mul(vf_loadu(kp + j * K + d), vg));
                    for (; d < K; ++d) gk[d] = kp[j * K + d] * gs;
                    for (int64_t e = 0; e < V; ++e) axpy(S + e * K, gk, Vp[j * V + e], K);
                }
            }
        }

        if (ht->data) {
            if (htv.s3 == 1) {
                for (int64_t e = 0; e < V; ++e)
                    row_from_f32(ht->dtype, S + e * K,
                                 (void*)byte_at(ht->data, ht->dtype, at4(htv, sn, h, e, 0)),
                                 K);
            } else {
                for (int64_t e = 0; e < V; ++e)
                    for (int64_t d = 0; d < K; ++d)
                        stt(ht, at4(htv, sn, h, e, d), S[e * K + d]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_recurrent_update */
/* What decode runs where prefill runs the chunked scan, against the PAGED state cache: the initial
 * state comes from the slot the last accepted token wrote and EVERY candidate token writes its own
 * slot, because which of them survives verification is not known until after this layer has run.
 * The gated RMS norm is folded into the epilogue, which is where libr4d puts it too. */
AVX_KERNEL(gdn_recurrent_update) {
    if (a->n_t < 14) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *av = &a->t[3], *bvt = &a->t[4];
    const RadTensor *state = &a->t[7], *o = &a->t[13];
    const RadTensor* A_log = rad_arg_in(a, 5);
    const RadTensor* dt_bias = rad_arg_in(a, 6);
    const RadTensor* sidx = rad_arg_in(a, 8);
    const RadTensor* nacc = rad_arg_in(a, 9);
    const RadTensor* cu = rad_arg_in(a, 10);
    const RadTensor* z = rad_arg_in(a, 11);
    const RadTensor* nw = rad_arg_in(a, 12);
    /* The out projection's quantiser, folded in. Optional and all-or-nothing. */
    const RadTensor* oq = (a->n_t > 14 && a->t[14].data) ? &a->t[14] : nullptr;
    const RadTensor* osc = (a->n_t > 15 && a->t[15].data) ? &a->t[15] : nullptr;
    if (!q->data || !k->data || !v->data || !av->data || !bvt->data || !state->data || !o->data ||
        !cu) return RAD_E_INVAL;

    /* ONLY THE PER-CANDIDATE STATE CONTRACT, AND ANY OTHER IS DECLINED BY NAME, as libref declines
     * it. Under "anchor" the cache holds one committed state a sequence plus a block of replay
     * factors and the accepted prefix is replayed onto it next step; the `o` that contract
     * produces is the same, the state is not, so running this reading over an anchor cache would
     * compute the right output once and resume every later step from the wrong state. */
    const char* sf = p_str(a, "state_form", "per_candidate");
    if (sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float neps = p_f32(a, "eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int nact = p_enum(a, "act", kActs, 2, 0);
    if (K <= 0 || V <= 0) return RAD_E_SHAPE;

    /* The head count and the token count come off the operands: v is [T, head_v_count, head_v], so
     * a rank-3 value tensor names both. `n_head_v` overrides it for a flattened view. */
    int64_t Hv = p_int(a, "n_head_v", 0);
    if (Hv <= 0) Hv = v->rank >= 3 ? v->shape[v->rank - 2] : 0;
    if (Hv <= 0) return RAD_E_SHAPE;
    const int64_t Tk = numel(v) / (Hv * V);
    if (Tk <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (Tk * K);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = Hv / Hg;

    /* N COMES FROM cu AND NOTHING ELSE: cu is [N+1] and its length minus one IS the sequence
     * count. Deriving it from state_idx and trusting cu to be at least as long is an off-by-one
     * waiting to happen. */
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) != Tk) return RAD_E_SHAPE;
    const int64_t sw = sidx ? (sidx->rank >= 2 ? sidx->shape[sidx->rank - 1]
                                               : (numel(sidx) / n_seq > 0 ? numel(sidx) / n_seq
                                                                          : 1))
                            : 1;

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, Hv, V), ov = view3(o, Hv, V);
    const V3 zv = z ? view3_rows(z, Hv, V) : vv;
    const V2 a2 = view2(av, Hv), b2 = view2(bvt, Hv);
    const V2 si = sidx ? view2(sidx, sw) : V2{ nullptr, 0, 0 };

    AVX_PARALLEL_FOR_IF(n_seq * Hv >= 4)
    for (int64_t nh = 0; nh < n_seq * Hv; ++nh) {
        const int64_t n = nh / Hv, h = nh % Hv, hk = h / grp;
        const int64_t bos = cu_at(cu, n), q_len = cu_at(cu, n + 1) - bos;
        if (q_len <= 0) continue;
        const int64_t na = nacc ? ldi(nacc, offlin(nacc, n)) : 1;
        const int64_t rd = na > 0 ? na - 1 : 0;
        const int64_t slot_in = sidx ? ldi(sidx, at2(si, n, rd < sw ? rd : sw - 1)) : n;
        if (slot_in < 0) continue;

        /* vrow/zrow widen the per-token value and gate rows once: the loops below read them
         * V times each, and per-element ldt there is a dtype switch per element. */
        const size_t need = (size_t)(V * K + 2 * K + 5 * V) + 16;
        float* wsp = scratch_f32(need);
        float* S = wsp;
        float* qn = S + V * K;
        float* kn = qn + K;
        float* uu = kn + K;
        float* orow = uu + V;
        float* gainv = orow + V;
        float* vrow = gainv + V;
        float* zrow = vrow + V;

        const int64_t sstride = state->rank >= 4 ? state->stride[0] : (int64_t)Hv * V * K;
        const int64_t hstride = state->rank >= 4 ? state->stride[1] : (int64_t)V * K;
        const int64_t vstride = state->rank >= 4 ? state->stride[2] : K;
        const int64_t dstride = state->rank >= 4 ? state->stride[3] : 1;

        /* ONE SWITCH PER ROW, NOT PER ELEMENT (avx_vec.h): the state tile is V*K scalar
         * dtype-switched loads below, and the widened row loop is what the hardware
         * prefetcher can see through. Same f32 values either way, so the arithmetic is
         * untouched. */
        if (dstride == 1) {
            for (int64_t e = 0; e < V; ++e)
                row_to_f32(state->dtype,
                           byte_at(state->data, state->dtype,
                                   slot_in * sstride + h * hstride + e * vstride),
                           S + e * K, K);
        } else {
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    S[e * K + d] = ldt(state, slot_in * sstride + h * hstride + e * vstride +
                                              d * dstride);
        }

        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        const float negexp = -std::exp(alog);
        /* The norm gain, read once per head rather than once per (token, channel). */
        if (nw) {
            if (laststride(nw) == 1) row_to_f32(nw->dtype, nw->data, gainv, V);
            else for (int64_t e = 0; e < V; ++e) gainv[e] = ldt(nw, e * laststride(nw));
        }

        for (int64_t t = 0; t < q_len; ++t) {
            const int64_t tok = bos + t;
            const float gval = negexp * softplus(ldt(av, at2(a2, tok, h)) + dtb, spthr);
            const float bt = 1.0f / (1.0f + std::exp(-ldt(bvt, at2(b2, tok, h))));

            if (qv.s2 == 1 && kv.s2 == 1) {
                row_to_f32(q->dtype, byte_at(q->data, q->dtype, at3(qv, tok, hk, 0)), qn, K);
                row_to_f32(k->dtype, byte_at(k->data, k->dtype, at3(kv, tok, hk, 0)), kn, K);
            } else {
                for (int64_t d = 0; d < K; ++d) {
                    qn[d] = ldt(q, at3(qv, tok, hk, d));
                    kn[d] = ldt(k, at3(kv, tok, hk, d));
                }
            }
            /* The value row, widened once per token: the decay loop below reads it V times. */
            if (vv.s2 == 1)
                row_to_f32(v->dtype, byte_at(v->data, v->dtype, at3(vv, tok, h, 0)), vrow, V);
            else
                for (int64_t e = 0; e < V; ++e) vrow[e] = ldt(v, at3(vv, tok, h, e));
            /* The l2 norms: no division by K, and `scale` folded into q's -- libref multiplies
             * them in that order and the product is not associative in f32. */
            const float qs = 1.0f / std::sqrt(row_sumsq(qn, K) + l2eps) * scale;
            const float ks = 1.0f / std::sqrt(row_sumsq(kn, K) + l2eps);
            {
                const vf vq = vf_set1(qs), vk = vf_set1(ks);
                int64_t d = 0;
                for (; d + VF_N <= K; d += VF_N) {
                    vf_storeu(qn + d, vf_mul(vf_loadu(qn + d), vq));
                    vf_storeu(kn + d, vf_mul(vf_loadu(kn + d), vk));
                }
                for (; d < K; ++d) { qn[d] *= qs; kn[d] *= ks; }
            }

            const float dec = std::exp(gval);
            const vf vdec = vf_set1(dec);
            for (int64_t e = 0; e < V; ++e) {
                float* Se = S + e * K;
                int64_t d = 0;
                for (; d + VF_N <= K; d += VF_N)
                    vf_storeu(Se + d, vf_mul(vf_loadu(Se + d), vdec));
                for (; d < K; ++d) Se[d] *= dec;
                uu[e] = bt * (vrow[e] - dotf(Se, kn, K));
            }
            for (int64_t e = 0; e < V; ++e) {
                float* Se = S + e * K;
                axpy(Se, kn, uu[e], K);
                orow[e] = dotf(Se, qn, K);
            }

            /* The gate row, widened once per token like the value row above. */
            if (z) {
                if (zv.s2 == 1)
                    row_to_f32(z->dtype, byte_at(z->data, z->dtype, at3(zv, tok, h, 0)), zrow,
                               V);
                else
                    for (int64_t e = 0; e < V; ++e) zrow[e] = ldt(z, at3(zv, tok, h, e));
            }
            /* The gated RMS norm, over the head's V channels. */
            const float sc = 1.0f / std::sqrt(row_sumsq(orow, V) / (float)V + neps);
            for (int64_t e = 0; e < V; ++e) {
                float r = orow[e] * sc;
                if (nw) r *= gainv[e];
                if (z) {
                    const float zz = zrow[e];
                    const float sg = 1.0f / (1.0f + std::exp(-zz));
                    r *= (nact == 1 ? sg : zz * sg);
                }
                orow[e] = r;
            }
            if (ov.s2 == 1) out_row(o, at3(ov, tok, h, 0), V, orow);
            else for (int64_t e = 0; e < V; ++e) stt(o, at3(ov, tok, h, e), orow[e]);

            /* QUANTISE WHAT WAS WRITTEN, not what produced it: the standalone quant_act_fp8 this
             * replaces read `o` back out of memory, so it saw the rounded store and not the f32.
             * One group per (token, head), because head_v is the fp8 block width. */
            if (oq && osc) {
                if (ov.s2 == 1)
                    row_to_f32(o->dtype, byte_at(o->data, o->dtype, at3(ov, tok, h, 0)), uu, V);
                else
                    for (int64_t e = 0; e < V; ++e) uu[e] = ldt(o, at3(ov, tok, h, e));
                const float amax = row_amax(uu, V);
                const float qsc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float inv = 1.0f / qsc;
                const int64_t qb = rowoff(oq, tok, Hv * V), qs2 = laststride(oq);
                if (qs2 == 1) {
                    const vf vi = vf_set1(inv), vlo = vf_set1(-448.0f), vhi = vf_set1(448.0f);
                    int64_t e = 0;
                    for (; e + VF_N <= V; e += VF_N) {
                        vf u = vf_mul(vf_loadu(uu + e), vi);
                        vf_storeu(uu + e, vf_min(vf_max(u, vlo), vhi));
                    }
                    for (; e < V; ++e) {
                        float u = uu[e] * inv;
                        uu[e] = u > 448.0f ? 448.0f : (u < -448.0f ? -448.0f : u);
                    }
                    row_from_f32(oq->dtype, uu,
                                 (void*)byte_at(oq->data, oq->dtype, qb + h * V), V);
                } else {
                    for (int64_t e = 0; e < V; ++e) {
                        float u = uu[e] * inv;
                        u = u > 448.0f ? 448.0f : (u < -448.0f ? -448.0f : u);
                        rad_store_f32(oq->data, oq->dtype, qb + (h * V + e) * qs2, u);
                    }
                }
                stt(osc, rowoff(osc, tok, Hv) + h * laststride(osc), qsc);
            }

            const int64_t so = sidx ? ldi(sidx, at2(si, n, t < sw ? t : sw - 1)) : n;
            if (so >= 0) {
                if (dstride == 1) {
                    for (int64_t e = 0; e < V; ++e)
                        row_from_f32(state->dtype, S + e * K,
                                     (void*)byte_at(state->data, state->dtype,
                                                    so * sstride + h * hstride + e * vstride),
                                     K);
                } else {
                    for (int64_t e = 0; e < V; ++e)
                        for (int64_t d = 0; d < K; ++d)
                            stt(state, so * sstride + h * hstride + e * vstride + d * dstride,
                                S[e * K + d]);
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_recurrent_update */
/* gdn_conv_update into a staging q, k and v, then gdn_recurrent_update over them. The fused op is
 * DEFINED as that pair, so this is the pair -- this level's own two kernels, called with the
 * operands and parameters each one takes -- and nothing is restated. The head geometry the staging
 * needs comes off `o` ([T, Hv, head_v]) and `x` (2 Hg head_k + Hv head_v columns).
 *
 * The staging is sized from x's dtype, a whole element each; a sub-byte activation is refused
 * rather than given a buffer the convolution would write past. It lives in a buffer the calling
 * thread keeps, so a call allocates nothing once the thread has seen its widest. */
AVX_KERNEL(gdn_conv_recurrent_update) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *o = &a->t[15];
    if (!x->data || !o->data || o->rank != 3 || x->rank < 2) return RAD_E_SHAPE;
    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t T = o->shape[0], Hv = o->shape[1];
    if (K <= 0 || V <= 0 || T <= 0 || Hv <= 0 || o->shape[2] != V) return RAD_E_SHAPE;
    const int64_t conv_dim = x->shape[x->rank - 1];
    if (conv_dim <= Hv * V || (conv_dim - Hv * V) % (2 * K)) return RAD_E_SHAPE;
    const int64_t Hg = (conv_dim - Hv * V) / (2 * K);
    const int64_t eb = (int64_t)rad_dtype_bits(x->dtype) / 8;
    if (eb <= 0 || (int64_t)rad_dtype_bits(x->dtype) % 8) return RAD_E_DTYPE;

    static thread_local std::vector<uint8_t> stage;
    const int64_t nq = T * Hg * K, nv = T * Hv * V;
    if (stage.size() < (size_t)((2 * nq + nv) * eb)) stage.resize((size_t)((2 * nq + nv) * eb));
    auto staged = [&](int64_t off, int64_t h, int64_t w) {
        RadTensor t{};
        t.data = stage.data() + off * eb;
        t.dtype = x->dtype;
        t.rank = 3;
        t.shape[0] = T; t.shape[1] = h; t.shape[2] = w;
        t.stride[0] = h * w; t.stride[1] = w; t.stride[2] = 1;
        return t;
    };
    const RadTensor tq = staged(0, Hg, K), tk = staged(nq, Hg, K), tv = staged(2 * nq, Hv, V);

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
    const int rc = AVX_FN(avx_gdn_conv_update)(&ca, stream);
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
    return AVX_FN(avx_gdn_recurrent_update)(&ra, stream);
}

/* ------------------------------------------------------------------ gdn_gated_rmsnorm */
/* out = x * rsqrt(mean_channels(x^2) + eps) * w * act(z), one row per (token, head). The prefill
 * path's form of the epilogue the recurrent update folds in.
 *
 * THE GATE HAS A THIRD SPELLING the shared rowoff() cannot see. It walks [tokens, heads, ch], and a
 * flat [rows, ch] projection is its product; but a COLUMN SLICE of a wider projection row -- the
 * fp8 delta net fuses [q|k|v|z] into one GEMM -- has heads*ch as its last axis, so rowoff reads it
 * as a dense row and lands in the next token. Spelled out here rather than generalised into
 * avx_common.h, because it is a property of THIS operand and not of every one. */
static inline int64_t gate_rowoff(const RadTensor* t, int64_t r, int64_t ch) {
    if (t->rank == 2 && ch > 0 && t->shape[1] != ch && t->shape[1] % ch == 0) {
        const int64_t heads = t->shape[1] / ch;
        return (r / heads) * t->stride[0] + (r % heads) * ch * t->stride[1];
    }
    return rowoff(t, r, ch);
}

AVX_KERNEL(gdn_gated_rmsnorm) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *z = &a->t[1], *w = &a->t[2], *o = &a->t[3];
    const int64_t ch = p_int(a, "channels", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (ch <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / ch;
    if (rows <= 0 || numel(o) < rows * ch || numel(z) < rows * ch || numel(w) < ch)
        return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)ch * sizeof(float) + 64);
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, ch);
        else for (int64_t i = 0; i < ch; ++i) gain[i] = ldt(w, i * ws);
    }

    AVX_PARALLEL_FOR_IF(rows * ch >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, ch), zb = gate_rowoff(z, r, ch), ob = rowoff(o, r, ch);
        const int64_t zs = laststride(z), os = laststride(o);
        float* xbuf = scratch_f32((size_t)ch);
        float* zbuf = scratch_f32_b((size_t)ch);
        float* obuf = scratch_f32_c((size_t)ch);
        const float* xv = in_row(x, xb, ch, xbuf);
        const float* zv2 = (zs == 1) ? in_row(z, zb, ch, zbuf) : zbuf;
        if (zs != 1) for (int64_t i = 0; i < ch; ++i) zbuf[i] = ldt(z, zb + i * zs);

        const float sc = 1.0f / std::sqrt(row_sumsq(xv, ch) / (float)ch + eps);
        const vf vs = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= ch; i += VF_N) {
            const vf zz = vf_loadu(zv2 + i);
            const vf gate = (act == 1) ? vf_sigmoid(zz) : vf_silu(zz);
            vf_storeu(obuf + i,
                      vf_mul(vf_mul(vf_mul(vf_loadu(xv + i), vs), vf_loadu(gain + i)), gate));
        }
        for (; i < ch; ++i) {
            const float zz = zv2[i];
            const float sg = 1.0f / (1.0f + std::exp(-zz));
            obuf[i] = xv[i] * sc * gain[i] * (act == 1 ? sg : zz * sg);
        }
        if (os == 1) out_row(o, ob, ch, obuf);
        else for (int64_t j = 0; j < ch; ++j) stt(o, ob + j * os, obuf[j]);
    }
    return RAD_OK;
}
