/* kfixture.cpp -- reading and writing the recorded reference. See kfixture.h for why it exists.
 *
 * The format is binary and little-endian, because the payload is float samples and a text format
 * for those is either lossy or three times the size. It is versioned in the header and a reader
 * refuses a version it does not know rather than reading a struct that has moved -- the failure
 * mode of getting that wrong is a report full of confident wrong numbers.
 */
#include "kfixture.h"
#include "opshapes.h"
#include "format/encoding.h"
#include "rad_plugin.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>

namespace rad {
namespace {

const char KF_MAGIC[4] = { 'R', 'K', 'B', '1' };

/* Every write goes through one of these and every read through its mirror, so a field added to
 * one side and not the other is a compile error rather than a silent misparse. */
struct W {
    FILE* f;
    bool  ok = true;
    void raw(const void* p, size_t n) { if (ok && std::fwrite(p, 1, n, f) != n) ok = false; }
    void u8(uint8_t v)   { raw(&v, 1); }
    void u32(uint32_t v) { raw(&v, 4); }
    void i64(int64_t v)  { raw(&v, 8); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f64(double v)   { raw(&v, 8); }
    void str(const std::string& s) { u32((uint32_t)s.size()); raw(s.data(), s.size()); }
    void f32v(const std::vector<float>& v) {
        u32((uint32_t)v.size()); raw(v.data(), v.size() * sizeof(float));
    }
    void i64v(const std::vector<int64_t>& v) {
        u32((uint32_t)v.size()); raw(v.data(), v.size() * sizeof(int64_t));
    }
};

struct R {
    FILE* f;
    bool  ok = true;
    void raw(void* p, size_t n) { if (ok && std::fread(p, 1, n, f) != n) ok = false; }
    uint8_t  u8()  { uint8_t v = 0;  raw(&v, 1); return v; }
    uint32_t u32() { uint32_t v = 0; raw(&v, 4); return v; }
    int64_t  i64() { int64_t v = 0;  raw(&v, 8); return v; }
    uint64_t u64() { uint64_t v = 0; raw(&v, 8); return v; }
    double   f64() { double v = 0;   raw(&v, 8); return v; }
    std::string str() {
        const uint32_t n = u32();
        /* A length out of a corrupt file must not become an allocation. Nothing here is longer
         * than an op name or a geometry key. */
        if (!ok || n > (1u << 20)) { ok = false; return {}; }
        std::string s(n, '\0');
        raw(s.data(), n);
        return s;
    }
    std::vector<float> f32v() {
        const uint32_t n = u32();
        if (!ok || n > (1u << 24)) { ok = false; return {}; }
        std::vector<float> v(n);
        raw(v.data(), (size_t)n * sizeof(float));
        return v;
    }
    std::vector<int64_t> i64v() {
        const uint32_t n = u32();
        if (!ok || n > RAD_MAX_RANK) { ok = false; return {}; }
        std::vector<int64_t> v(n);
        raw(v.data(), (size_t)n * sizeof(int64_t));
        return v;
    }
};

}  /* namespace */

uint64_t kf_operand_seed(uint64_t case_seed, int k) {
    uint64_t x = case_seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(k + 1));
    return rad_splitmix(x);
}

void kf_sample_positions(uint64_t seed, int64_t n, std::vector<int64_t>* out) {
    out->clear();
    if (n <= 0) return;
    /* Small operands are sampled ENTIRELY rather than approximately: below the sample count there
     * is nothing to gain by drawing, and an exhaustive check of a 256-element scale plane is
     * strictly better than 1024 draws with repeats. */
    if (n <= KF_SAMPLES) {
        out->reserve((size_t)n);
        for (int64_t i = 0; i < n; ++i) out->push_back(i);
        return;
    }
    out->reserve(KF_SAMPLES);
    uint64_t s = seed;
    for (int i = 0; i < KF_SAMPLES; ++i)
        out->push_back((int64_t)(rad_splitmix(s) % (uint64_t)n));
}

int Kfixture::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return RAD_E_IO;
    W w{ f };
    w.raw(KF_MAGIC, 4);
    w.u32(version);
    w.u32((uint32_t)KF_SAMPLES);
    w.str(source);
    w.str(oracle);
    w.str(recorded);
    w.u32((uint32_t)cases.size());
    for (const KfCase& c : cases) {
        w.str(c.op);
        w.u64(c.seed);
        w.i64(c.m);
        w.str(c.band);
        w.u8(c.speed_only ? 1 : 0);
        w.u32((uint32_t)c.params.size());
        for (const KfParam& p : c.params) {
            w.str(p.key);
            w.u32((uint32_t)p.kind);
            w.u64((uint64_t)p.i);
            w.f64(p.d);
            w.str(p.s);
        }
        w.u32((uint32_t)c.opd.size());
        for (const KfOperand& o : c.opd) {
            w.u32((uint32_t)o.role);
            w.u32(o.dtype);
            w.i64v(o.shape);
            w.u8(o.absent ? 1 : 0);
            w.u32((uint32_t)o.fill);
            w.u8((uint8_t)((o.is_index ? 1 : 0) | (o.idx_unique ? 2 : 0) | (o.idx_cu ? 4 : 0)));
            w.i64(o.idx_max);
            w.i64(o.idx_const);
            w.i64(o.fill_chunk);
            w.u8(o.has_enc ? 1 : 0);
            if (o.has_enc) {
                w.raw(&o.enc, sizeof o.enc);
                w.u32((uint32_t)o.sel.size());
                for (size_t k = 0; k < o.sel.size(); ++k) {
                    w.u32((uint32_t)o.sel[k]);
                    w.i64(o.sel_rows[k]);
                    w.i64(o.sel_cols[k]);
                }
            }
            w.u8(o.have_ref ? 1 : 0);
            if (o.have_ref) {
                w.i64(o.ref_n);
                w.f64(o.ref_sumsq);
                w.i64(o.ref_nonfinite);
                w.f32v(o.sample);
            }
        }
    }
    const bool ok = w.ok;
    std::fclose(f);
    return ok ? RAD_OK : RAD_E_IO;
}

int Kfixture::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return RAD_E_NOTFOUND;
    R r{ f };
    char magic[4];
    r.raw(magic, 4);
    if (!r.ok || std::memcmp(magic, KF_MAGIC, 4) != 0) { std::fclose(f); return RAD_E_FORMAT; }
    version = r.u32();
    const uint32_t samples = r.u32();
    /* A file recorded with a different sample count is not readable as this one: the positions
     * are DERIVED from the count, so every sample would be compared against the wrong element and
     * the report would be a page of confident failures. */
    if (version != 4 || samples != (uint32_t)KF_SAMPLES) {
        std::fclose(f);
        return RAD_E_FORMAT;
    }
    source   = r.str();
    oracle   = r.str();
    recorded = r.str();
    const uint32_t n = r.u32();
    if (!r.ok || n > (1u << 20)) { std::fclose(f); return RAD_E_FORMAT; }
    cases.clear();
    cases.reserve(n);
    for (uint32_t ci = 0; ci < n && r.ok; ++ci) {
        KfCase c;
        c.op   = r.str();
        c.seed = r.u64();
        c.m    = r.i64();
        c.band = r.str();
        c.speed_only = r.u8() != 0;
        const uint32_t np = r.u32();
        if (!r.ok || np > 256) { r.ok = false; break; }
        for (uint32_t i = 0; i < np && r.ok; ++i) {
            KfParam p;
            p.key  = r.str();
            p.kind = (int)r.u32();
            p.i    = (long long)r.u64();
            p.d    = r.f64();
            p.s    = r.str();
            c.params.push_back(std::move(p));
        }
        const uint32_t no = r.u32();
        if (!r.ok || no > 64) { r.ok = false; break; }
        for (uint32_t i = 0; i < no && r.ok; ++i) {
            KfOperand o;
            o.role   = (int)r.u32();
            o.dtype  = r.u32();
            o.shape  = r.i64v();
            o.absent = r.u8() != 0;
            o.fill   = (int)r.u32();
            const uint8_t flags = r.u8();
            o.is_index   = (flags & 1) != 0;
            o.idx_unique = (flags & 2) != 0;
            o.idx_cu     = (flags & 4) != 0;
            o.idx_max    = r.i64();
            o.idx_const  = r.i64();
            o.fill_chunk = r.i64();
            o.has_enc    = r.u8() != 0;
            if (o.has_enc) {
                r.raw(&o.enc, sizeof o.enc);
                const uint32_t ns = r.u32();
                if (!r.ok || ns == 0 || ns > RAD_ENC_MAX_PLANES || !rad_enc_valid(&o.enc)) {
                    r.ok = false;
                    break;
                }
                for (uint32_t k = 0; k < ns; ++k) {
                    const uint32_t ix = r.u32();
                    if (ix >= (uint32_t)o.enc.n_planes) r.ok = false;
                    o.sel.push_back((int)ix);
                    o.sel_rows.push_back(r.i64());
                    o.sel_cols.push_back(r.i64());
                }
            }
            o.have_ref   = r.u8() != 0;
            if (o.have_ref) {
                o.ref_n         = r.i64();
                o.ref_sumsq     = r.f64();
                o.ref_nonfinite = r.i64();
                o.sample        = r.f32v();
            }
            c.opd.push_back(std::move(o));
        }
        cases.push_back(std::move(c));
    }
    const bool ok = r.ok;
    std::fclose(f);
    return ok ? RAD_OK : RAD_E_FORMAT;
}

}  /* namespace rad */

namespace rad {

void kf_reduce(const float* v, int64_t n, uint64_t seed, double* sumsq, int64_t* nonfinite,
               std::vector<float>* sample) {
    /* THREADED, AND SUMMED IN BLOCK ORDER SO IT IS STILL DETERMINISTIC.
     *
     * This walks every element of every output of every case, which at a 248K-vocabulary
     * logits_gemm is sixteen million elements and at gdn_recurrent_update rather more -- single
     * threaded it dominates the CPU time of a replay that does almost no GPU work. Partial sums
     * are accumulated per block and added back in block index order, so the answer does not depend
     * on how many threads ran: a recorded sum of squares that moved with the core count would make
     * every recorded reference machine-specific. */
    const int64_t nb = (n + KF_REDUCE_BLOCK - 1) / KF_REDUCE_BLOCK;
    std::vector<double> part((size_t)std::max<int64_t>(nb, 1), 0.0);
    std::vector<int64_t> partnf((size_t)std::max<int64_t>(nb, 1), 0);
    auto do_block = [&](int64_t b) {
        const int64_t lo = b * KF_REDUCE_BLOCK;
        const int64_t hi = std::min<int64_t>(lo + KF_REDUCE_BLOCK, n);
        double ss = 0;
        int64_t nf = 0;
        for (int64_t i = lo; i < hi; ++i) {
            if (!std::isfinite(v[i])) { ++nf; continue; }
            ss += (double)v[i] * (double)v[i];
        }
        part[(size_t)b] = ss;
        partnf[(size_t)b] = nf;
    };
    if (nb <= 2) {
        for (int64_t b = 0; b < nb; ++b) do_block(b);
    } else {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        const int64_t nt = std::min<int64_t>(nb, (int64_t)hw);
        std::vector<std::thread> th;
        th.reserve((size_t)nt);
        for (int64_t t = 0; t < nt; ++t)
            th.emplace_back([&, t]() { for (int64_t b = t; b < nb; b += nt) do_block(b); });
        for (auto& x : th) x.join();
    }
    double ss = 0;
    int64_t nf = 0;
    for (int64_t b = 0; b < nb; ++b) { ss += part[(size_t)b]; nf += partnf[(size_t)b]; }
    *sumsq = ss;
    *nonfinite = nf;

    std::vector<int64_t> pos;
    kf_sample_positions(seed, n, &pos);
    sample->clear();
    sample->reserve(pos.size());
    for (int64_t p : pos) sample->push_back(v[p]);
}

void kf_draw(const std::vector<KfOperand>& opd, size_t k, int64_t M, int64_t vocab_rows,
             uint64_t seed, std::vector<uint8_t>* out) {
    const KfOperand& d = opd[k];
    int64_t n = 1;
    for (int64_t x : d.shape) n *= x;
    out->assign((size_t)rad_dtype_bytes(d.dtype, n), 0);
    if (n <= 0 || d.absent) return;
    if (d.role != RAD_OPD_IN && d.role != RAD_OPD_INOUT && d.role != RAD_OPD_WEIGHT) return;

    /* A WEIGHT'S CONTENT DOES NOT DEPEND ON THE BATCH SIZE, so its seed must not either.
     *
     * The M sweep runs the same GEMM at eight or ten query lengths. Under a per-case seed each of
     * those points would re-draw the same 178-million-element weight from scratch -- ten
     * Box-Muller passes over 178M elements for one (N, K), which dominates a replay's wall clock.
     * Deriving the seed from the operand's own identity instead -- its position, dtype and extents
     * -- makes every point of the sweep ask for the SAME bytes, which a caller can then draw once
     * and keep. kf_weight_seed is public for exactly that: it is the cache key.
     *
     * It costs nothing in coverage. Which pseudorandom weight a kernel is checked against was
     * never the point; that two implementations agree about it is. */
    if (d.role == RAD_OPD_WEIGHT) seed = kf_weight_seed(d, k);

    std::vector<float> tmp;
    /* A BIG OPERAND IS DRAWN IN BLOCKS AND NEVER MATERIALISES AS FLOATS.
     *
     * The plain-normal path -- which is every weight and every ordinary activation, so almost all
     * of the bytes here -- would otherwise allocate one float vector the size of the whole tensor
     * and narrow it in one pass. At an fp8 GEMM weight that is 713 MB of scratch to produce 178 MB
     * of fp8, touched twice, for one case. In blocks it is a few hundred kilobytes that stays in
     * cache, and the result is identical because rad_fill_normal is block-seeded at exactly this
     * width.
     *
     * Only this path blocks. The shaped fills below (a triangular inverse, a chunk-local cumulative
     * sum) read across the whole tensor by construction and are small by construction, so blocking
     * them would buy nothing and get the shapes wrong. */
    /* A WEIGHT CAN BE AN INDEX, so `role == WEIGHT` is not the whole answer here. An operand whose
     * VALUES are sizes or offsets -- a weight by role, an index by content -- must not be drawn as
     * standard normals cast to its integer dtype, which is zeros and negatives. ngram_ids is the
     * case: it refuses `vocab_size <= 0` and `offset < 0` by name, so drawing it as a plain weight
     * makes the ORACLE refuse every geometry and the op is never checked on any plugin. That
     * refusal reads exactly like a shape the reference does not serve, so it is easy to miss.
     *
     * The index path runs values through the float `tmp` below, so an index constant past 2^24
     * does not survive the round trip -- fine for a vocabulary and not for a token id at a 250k
     * vocab, which is why the constant here is the vocabulary and not the id. */
    const bool plain = (d.role == RAD_OPD_WEIGHT && !d.is_index) ||
                       (!d.is_index && d.fill == KF_FILL_NORMAL);
    if (plain && n > RAD_FILL_BLOCK && rad_dtype_bytes(d.dtype, 1) >= 1) {
        const float sigma = d.role == RAD_OPD_WEIGHT ? 0.02f : 1.0f;
        tmp.resize((size_t)RAD_FILL_BLOCK);
        for (int64_t lo = 0; lo < n; lo += RAD_FILL_BLOCK) {
            const int64_t len = std::min<int64_t>(RAD_FILL_BLOCK, n - lo);
            rad_fill_normal_block(tmp.data(), len, seed, lo / RAD_FILL_BLOCK, sigma);
            rad_narrow(tmp.data(), d.dtype, len,
                       out->data() + rad_dtype_bytes(d.dtype, lo));
        }
        return;
    }
    tmp.assign((size_t)n, 0.0f);
    if (d.role == RAD_OPD_WEIGHT && !d.is_index) {
        /* A WEIGHT IS A SMALL DRAW, and the 0.02 sigma is not cosmetic: a checkpoint's weights
         * are that size, and drawing them at unit variance puts an fp8 quantiser's amax three
         * orders of magnitude off where a real one sits, so the case would measure the
         * quantiser's behaviour on data no model produces. */
        rad_fill_normal(tmp.data(), n, seed, 0.02f);
        rad_narrow(tmp.data(), d.dtype, n, out->data());
        return;
    }
    if (d.is_index && d.idx_cu) {
        /* Cumulative sequence lengths, [0 .. T]: the batch's shape, not a draw. Every GDN kernel
         * opens by reading cu[n] before any null test, so a wrong one is a fault and not a wrong
         * answer. */
        const int64_t S = n > 1 ? n - 1 : 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)((i * M) / S);
    } else if (d.is_index && d.idx_const >= 0) {
        /* A length, not a choice. `seqused` is the case: a random context shorter than the query
         * makes causal attention read a prefix that does not exist, and both implementations
         * agree about the garbage. */
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)d.idx_const;
    } else if (d.is_index && d.idx_unique) {
        /* A scatter's destinations, distinct: a partial Fisher-Yates over the range. Two tokens
         * on one slot is a race the two implementations resolve differently, so the comparison
         * would measure the scheduler. */
        uint64_t s2 = seed;
        const int64_t range = d.idx_max > 0 ? d.idx_max : n;
        std::vector<int32_t> pool((size_t)range);
        for (int64_t i = 0; i < range; ++i) pool[(size_t)i] = (int32_t)i;
        for (int64_t i = 0; i < n && i < range; ++i) {
            const int64_t j = i + (int64_t)(rad_splitmix(s2) % (uint64_t)(range - i));
            std::swap(pool[(size_t)i], pool[(size_t)j]);
            tmp[(size_t)i] = (float)pool[(size_t)i];
        }
        for (int64_t i = range; i < n; ++i) tmp[(size_t)i] = -1.0f;
    } else if (d.is_index) {
        /* An index operand indexes the operand before it, whose leading extent is the only legal
         * range -- unless the recipe named the range itself, which the paged pair has to: a KV
         * slot indexes n_blocks * block_size and a block-table entry indexes n_blocks, and neither
         * is any operand's dim 0. */
        uint64_t s2 = seed;
        int64_t rows = vocab_rows > 0 ? vocab_rows : 32000;
        if (k > 0 && !opd[k - 1].shape.empty()) rows = opd[k - 1].shape[0];
        if (d.idx_max > 0) rows = d.idx_max;
        if (rows <= 0) rows = 1;
        for (int64_t i = 0; i < n; ++i)
            tmp[(size_t)i] = (float)(rad_splitmix(s2) % (uint64_t)rows);
    } else {
        rad_fill_normal(tmp.data(), n, seed, 1.0f);
        /* INTO THE OPERAND'S DOMAIN, where a normal draw is outside it. */
        if (d.fill == KF_FILL_SIGMOID) {
            for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = 1.0f / (1.0f + std::exp(-tmp[(size_t)i]));
        } else if (d.fill == KF_FILL_TRI_LOWER) {
            /* UNIT DIAGONAL, SMALL OFF-DIAGONAL, and the diagonal is the whole point.
             *
             * `A` is the chunked delta rule's intra-chunk matrix -- I + tril(diag(beta) K K^T, -1)
             * -- so it is UNIT lower triangular by construction and its inverse is bounded. Drawn
             * with a normal on the diagonal it is not: a diagonal element near zero makes the
             * inverse enormous, and `gdn_kkt_solve` then returns values of order 1e20 with a
             * rel_l2 that grows with the chunk count. Two implementations summing an
             * ill-conditioned product in different orders disagree by however ill-conditioned it
             * is, so that number is the CONDITION NUMBER OF THE TEST, not an error in the kernel.
             *
             * This is the same correction FillSigmoid and FillGateCumsum already are: put the
             * operand inside the domain the op is only ever handed, or the comparison measures
             * the harness. The strictly-upper part stays zeroed -- whether it is read at all is a
             * convention the two implementations never disagree about on a real matrix, because
             * the same library's kkt_solve produced it. */
            const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : 1;
            const int64_t heads = d.shape.size() >= 3 ? d.shape[1] : 1;
            const int64_t rows = (heads * ch) != 0 ? n / (heads * ch) : n;
            for (int64_t t = 0; t < rows; ++t)
                for (int64_t h = 0; h < heads; ++h)
                    for (int64_t j = 0; j < ch; ++j) {
                        const size_t i = (size_t)((t * heads + h) * ch + j);
                        const int64_t diag = t % ch;
                        if (j > diag)       tmp[i] = 0.0f;
                        else if (j == diag) tmp[i] = 1.0f;
                        else                tmp[i] *= 0.25f;
                    }
        } else if (d.fill == KF_FILL_GATE_CUMSUM) {
            /* Rows are [tokens, heads] and the cumsum runs down TOKENS within a chunk, resetting
             * at each boundary -- which is what chunk_local_cumsum produces and what the scan
             * assumes. */
            const int64_t heads = d.shape.size() >= 2 ? d.shape.back() : 1;
            const int64_t rows = heads > 0 ? n / heads : n;
            const int64_t ch = d.fill_chunk > 0 ? d.fill_chunk : rows;
            for (int64_t h = 0; h < heads; ++h) {
                float acc = 0.0f;
                for (int64_t t = 0; t < rows; ++t) {
                    if (t % ch == 0) acc = 0.0f;
                    acc -= std::log1p(std::exp(tmp[(size_t)(t * heads + h)]));
                    tmp[(size_t)(t * heads + h)] = acc;
                }
            }
        } else if (d.fill == KF_FILL_UNIT_ROW) {
            /* UNIT L2 NORM ALONG THE LAST DIMENSION, which is what the model's own qk_norm hands
             * the gated-delta-net path. Drawn as plain normals over a 128-wide head the key has
             * norm ~11, the solve's matrix I + tril(beta k k^T, -1) then has off-diagonals two
             * orders of magnitude too large, and its inverse is astronomical -- gdn_kkt_solve
             * returns values of order 1e18 and a rel_l2 that moves with the draw. That is not a
             * defect in the kernel; it is an input outside the domain the op is ever handed. */
            const int64_t w = d.shape.empty() ? n : d.shape.back();
            if (w > 0) {
                for (int64_t r = 0; r + w <= n; r += w) {
                    double ss = 0;
                    for (int64_t i = 0; i < w; ++i)
                        ss += (double)tmp[(size_t)(r + i)] * (double)tmp[(size_t)(r + i)];
                    const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
                    for (int64_t i = 0; i < w; ++i) tmp[(size_t)(r + i)] *= inv;
                }
            }
        }
    }
    rad_narrow(tmp.data(), d.dtype, n, out->data());
}

}  /* namespace rad */

namespace rad {


uint64_t kf_weight_seed(const KfOperand& d, size_t k) {
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(k + 1) ^ ((uint64_t)d.dtype << 17);
    for (int64_t x : d.shape) { s ^= (uint64_t)x; s = rad_splitmix(s); }
    return rad_splitmix(s);
}

}  /* namespace rad */

namespace rad {

void kf_operand_enc(const WeightInfo& w, KfOperand* d) {
    d->has_enc = true;
    d->enc = w.enc;
    d->sel.assign(w.sel, w.sel + w.n_sel);
    d->sel_rows.assign(w.sel_rows, w.sel_rows + w.n_sel);
    d->sel_cols.assign(w.sel_cols, w.sel_cols + w.n_sel);
}

namespace {

/* The entries of the codebook `codes` index, or 0 when they index none. */
int64_t codebook_entries(const RadEncoding& e) {
    if (const RadEncPlane* g = rad_enc_plane(&e, "grid")) return g->extent[0];
    if (const RadEncPlane* t = rad_enc_plane(&e, "table")) return t->extent[1];
    return 0;
}

/* One plane, rows of `rb` bytes. */
void draw_plane(const RadEncoding& e, const RadEncPlane& p, int64_t rows, int64_t cols,
                int64_t rb, uint64_t seed, uint8_t* out) {
    const int64_t n = rows * cols;
    uint64_t s = seed;
    if (!std::strcmp(p.role, "t.perm")) {
        /* a permutation of the columns, each row its own: Fisher-Yates over the identity */
        for (int64_t r = 0; r < rows; ++r) {
            std::vector<int64_t> v((size_t)cols);
            for (int64_t j = 0; j < cols; ++j) v[(size_t)j] = j;
            for (int64_t j = cols - 1; j > 0; --j)
                std::swap(v[(size_t)j], v[(size_t)(rad_splitmix(s) % (uint64_t)(j + 1))]);
            for (int64_t j = 0; j < cols; ++j)
                rad_store_f32(out + r * rb, p.dtype, j, (float)v[(size_t)j]);
        }
        return;
    }
    int sgn = 0;
    const int pb = rad_dtype_packed_int(p.dtype, &sgn);
    const bool ints = pb || p.dtype == RAD_I8 || p.dtype == RAD_U8 || p.dtype == RAD_I16 ||
                      p.dtype == RAD_U16 || p.dtype == RAD_I32 || p.dtype == RAD_U32;
    const bool codes = !std::strcmp(p.role, "codes");
    if (ints || p.dtype == RAD_FP4E2M1 || p.dtype == RAD_FP6E2M3 || p.dtype == RAD_FP6E3M2) {
        /* EVERY BIT PATTERN OF THESE IS A VALUE, so a uniform draw over the patterns is the
         * widest test -- except a code into a codebook, which is held inside it, and a wide
         * integer, which a reference would sum past f32's exact range. */
        const int bits = rad_dtype_bits(p.dtype);
        uint64_t range = bits >= 16 ? 1u << 12 : 1ull << bits;
        const int64_t entries = codes ? codebook_entries(e) : 0;
        if (entries > 0 && (uint64_t)entries < range) range = (uint64_t)entries;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < cols; ++j) {
                uint32_t c = (uint32_t)(rad_splitmix(s) % range);
                if (pb) { rad_store_bits(out + r * rb, pb, j, c); continue; }
                const bool neg = (p.dtype == RAD_I8 || p.dtype == RAD_I16 || p.dtype == RAD_I32);
                const float v = neg ? (float)((int64_t)c - (int64_t)(range / 2)) : (float)c;
                rad_store_f32(out + r * rb, p.dtype, j, v);
            }
        return;
    }
    if (p.dtype == RAD_E8M0) {
        /* 2^-4 .. 2^4: a block scale, which a draw over every exponent would make an overflow */
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < cols; ++j)
                out[r * rb + j] = (uint8_t)(123 + rad_splitmix(s) % 9);
        return;
    }
    /* A float plane: a normal draw, and a magnitude for what scales (a scale or a minimum is
     * read as one, and a negative one is a sign flip no quantiser writes). */
    std::vector<float> f((size_t)n);
    rad_fill_normal(f.data(), n, seed, 1.0f);
    const bool mag = rad_enc_role_in_chain(p.role, "scale") || !std::strcmp(p.role, "t.scale");
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < cols; ++j) {
            float v = f[(size_t)(r * cols + j)];
            if (mag) v = 0.25f + std::fabs(v);
            rad_store_f32(out + r * rb, p.dtype, j, v);
        }
}

}  /* namespace */

void kf_draw_planes(const KfOperand& d, uint64_t seed, std::vector<std::vector<uint8_t>>* out) {
    out->assign(d.sel.size(), {});
    for (size_t k = 0; k < d.sel.size(); ++k) {
        const RadEncPlane& p = d.enc.plane[d.sel[k]];
        const int64_t rows = d.sel_rows[k], cols = d.sel_cols[k];
        const int64_t rb = rad_enc_row_bytes(p.dtype, cols);
        std::vector<uint8_t>& b = (*out)[k];
        b.assign((size_t)(rows * rb), 0);
        uint64_t ps = seed ^ (0xA24BAED4963EE407ull * (uint64_t)(d.sel[k] + 1));
        draw_plane(d.enc, p, rows, cols, rb, rad_splitmix(ps), b.data());
    }
}

int kf_planes_logical(const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                      std::vector<uint8_t>* out, std::string* why) {
    const int64_t n = rad_numel(d.shape);
    out->assign((size_t)rad_dtype_bytes(d.dtype, n), 0);
    const void* data[RAD_ENC_MAX_PLANES] = {};
    for (size_t k = 0; k < planes.size() && k < RAD_ENC_MAX_PLANES; ++k)
        data[k] = planes[k].data();
    return enc_selection_logical(d.enc, d.sel.data(), (int)d.sel.size(), d.sel_rows.data(),
                                 d.sel_cols.data(), data, d.dtype, n, out->data(), why);
}

int kf_lay_weight(const RadKernelInfo* info, const RadParam* p, int n_p, int operand,
                  const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                  RadLayout* lay, std::vector<uint8_t>* out) {
    if (!info || !info->layout || !d.has_enc) return RAD_E_UNSUPPORTED;
    RadTensor pt[RAD_ENC_MAX_PLANES];
    const int n = (int)d.sel.size();
    for (int k = 0; k < n; ++k) {
        pt[k] = RadTensor{};
        pt[k].data  = const_cast<uint8_t*>(planes[(size_t)k].data());
        pt[k].dtype = d.enc.plane[d.sel[(size_t)k]].dtype;
        pt[k].rank  = 2;
        pt[k].shape[0] = d.sel_rows[(size_t)k];
        pt[k].shape[1] = d.sel_cols[(size_t)k];
        rad_tensor_pack(&pt[k]);
    }
    *lay = RadLayout{};
    const int rc = info->layout(p, n_p, operand, &d.enc, d.sel.data(), pt, n, lay);
    if (rc != RAD_OK) return rc;
    /* A layout with no relayout behind it is a row that would fail at load; said so here rather
     * than handed on as bytes nobody wrote. */
    if (lay->bytes <= 0 || !info->relayout) return RAD_E_STATE;
    out->assign((size_t)lay->bytes, 0);
    return info->relayout(p, n_p, operand, &d.enc, d.sel.data(), pt, n, out->data(), lay->bytes);
}

}  /* namespace rad */

namespace rad {

int64_t kf_vocab_rows(const std::vector<KfOperand>& opd) {
    int64_t v = 0;
    for (const KfOperand& d : opd)
        if (!d.shape.empty() && d.shape[0] > v) v = d.shape[0];
    return v;
}

}  /* namespace rad */

namespace rad {
/* HOW THE ENGINE DESCRIBES A RE-LAID WEIGHT, AND THEREFORE HOW THIS TOOL MUST.
 *
 * STORED DTYPE, DECLARED SHAPE -- which is what Ctx::refresh_weights does, and not what the
 * RadLayout says. The two differ for any layout that changes the width: libr4d's fp8 lm_head
 * appends each row's 128-block scales to its own row, so the stored plane is [N, K + 2*sk] while
 * the operand the kernel is handed is still [N, K] of E4M3 and the kernel finds the scales past
 * the end of its own row. Described by the stored shape instead, the kernel reads K = K + 2*sk,
 * finds it is not the K it was compiled for, and refuses the geometry it was selected for.
 *
 * Describing the operand any other way here would be testing a call the engine never makes. */
void kf_describe_laid(const KfOperand& d, const RadLayout& lay, void* data, RadTensor* out) {
    *out = RadTensor{};
    out->data  = data;
    out->dtype = lay.dtype;
    out->rank  = (uint32_t)d.shape.size();
    for (size_t i = 0; i < d.shape.size() && i < RAD_MAX_RANK; ++i)
        out->shape[i] = d.shape[i];
    rad_tensor_pack(out);
}

}  /* namespace rad */
