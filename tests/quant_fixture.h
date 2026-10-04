/* quant_fixture.h -- quantising a weight through a quantiser plugin exactly as rad-convert does.
 *
 * Shared by the tests that hold a quantiser's output to something: quant_test reads it back
 * through the core's decoder, relayout_parity_test hands it to the kernel libraries' relayouts.
 * Both have to quantise the way the converter will -- ask the encoding, size the planes, pass the
 * rows over in the quantiser's own blocks -- or what they check is a path no container is made by.
 */
#pragma once
#include "rad_test.h"
#include "rad_plugin.h"
#include "rad_quant.h"
#include "format/encoding.h"
#include "plugin/registry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qfix {

inline std::string home() {
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}

inline rad::Registry& reg() {
    static rad::Registry r;
    static bool tried = false;
    if (!tried) {
        tried = true;
        r.load_plugin(home() + "/quantizers/libquant.so", 0);
    }
    return r;
}

inline const RadQuantizerInfo* quantizer(const char* name) {
    const rad::Registry::QuantRow* q = reg().quantizer(name);
    return q ? q->info : nullptr;
}

#define NEED_LIBQUANT()                                                                       \
    do {                                                                                      \
        if (!::qfix::quantizer("rtn")) {                                                      \
            std::fprintf(stderr, "  SKIP no libquant under %s\n", ::qfix::home().c_str());    \
            return;                                                                           \
        }                                                                                     \
    } while (0)

/* Parameters as a caller hands them over: text for strings, typed for the rest. The keys and
 * strings are owned here and the RadParam pointers patched on every data(), so a copy is safe. */
struct Opts {
    std::vector<std::string> keys, vals;
    std::vector<RadParam> p;
    Opts& s(const char* k, const char* v) {
        keys.emplace_back(k);
        vals.emplace_back(v);
        p.push_back(RAD_STR(nullptr, nullptr));
        return *this;
    }
    Opts& i(const char* k, long long v) {
        keys.emplace_back(k);
        vals.emplace_back("");
        p.push_back(RAD_INT(nullptr, v));
        return *this;
    }
    Opts& f(const char* k, double v) {
        keys.emplace_back(k);
        vals.emplace_back("");
        p.push_back(RAD_F64(nullptr, v));
        return *this;
    }
    const RadParam* data() {
        for (size_t j = 0; j < p.size(); ++j) {
            p[j].key = keys[j].c_str();
            if (p[j].kind == RAD_P_STR) p[j].sval = vals[j].c_str();
        }
        return p.data();
    }
    int n() const { return (int)p.size(); }
};

/* A splitmix draw, normal-ish via the sum of four uniforms, with a few outliers. Integer state and
 * one double product a value, so the same seed is the same weight on every machine. */
inline std::vector<float> draw(int64_t n, uint64_t seed, float outlier = 0.0f) {
    std::vector<float> v((size_t)n);
    uint64_t z = seed;
    auto next = [&]() {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return (double)((x ^ (x >> 31)) >> 11) / 9007199254740992.0;
    };
    for (int64_t i = 0; i < n; ++i) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += next() - 0.5;
        v[(size_t)i] = (float)(s * 0.05);
        if (outlier > 0 && next() < 0.002) v[(size_t)i] *= outlier;
    }
    return v;
}

struct Quantised {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> planes;
    int status = RAD_OK;
};

/* Quantise as rad-convert does: ask the encoding, size the planes, hand the rows over in the
 * quantiser's own blocks. `importance` is the imatrix's column importances the converter would
 * hand over, or null -- and null when the encoding is asked, as at declare. */
inline Quantised quantise(const RadQuantizerInfo* q, Opts& o, const char* name, int64_t rows,
                          int64_t cols, const std::vector<float>& w,
                          const float* importance = nullptr) {
    Quantised out;
    RadQuantWeight qw{};
    qw.name = name;
    qw.rank = 2;
    qw.shape[0] = rows;
    qw.shape[1] = cols;
    qw.rows = rows;
    qw.cols = cols;
    qw.layer = qw.expert = -1;
    out.status = q->encoding(o.data(), o.n(), &qw, &out.enc);
    if (out.status != RAD_OK) return out;
    qw.importance = importance;
    for (int i = 0; i < out.enc.n_planes; ++i)
        out.planes.emplace_back((size_t)rad_enc_plane_bytes(&out.enc.plane[i], rows, cols), 0);
    int64_t rb = q->row_block(o.data(), o.n(), &qw);
    if (rb <= 0) rb = rows;
    for (int64_t r0 = 0; r0 < rows; r0 += rb) {
        const int64_t n = r0 + rb < rows ? rb : rows - r0;
        std::vector<float> blk(w.begin() + r0 * cols, w.begin() + (r0 + n) * cols);
        std::vector<void*> pp;
        for (int i = 0; i < out.enc.n_planes; ++i) {
            const RadEncPlane& p = out.enc.plane[i];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&p, rows, cols, &pr, &pc);
            const bool banded = p.kind == RAD_PLANE_TILED && p.block[0] > 0;
            const int64_t prow = banded ? r0 / p.block[0] : 0;
            pp.push_back(out.planes[(size_t)i].data() + prow * rad_enc_row_bytes(p.dtype, pc));
        }
        out.status = q->quantize(o.data(), o.n(), &qw, blk.data(), r0, n, pp.data());
        if (out.status != RAD_OK) return out;
    }
    return out;
}

inline std::vector<float> decode(const Quantised& q, int64_t rows, int64_t cols) {
    std::vector<const void*> d;
    for (auto& p : q.planes) d.push_back(p.data());
    rad::EncodedView v;
    rad::enc_view(q.enc, rows, cols, d.data(), &v);
    std::vector<float> out((size_t)(rows * cols));
    std::string why;
    if (rad::enc_decode_rows(v, 0, rows, out.data(), &why) != RAD_OK) {
        std::fprintf(stderr, "    decode: %s\n", why.c_str());
        out.clear();
    }
    return out;
}

inline double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double)(a[i] - b[i]) * (a[i] - b[i]);
        den += (double)a[i] * a[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0;
}

/* A calibration file as the engine's tap writes one: "RADGRAM2", k, rank, rows, domain, a pad word,
 * then k*k f32 row-major. `domain` 0 is the model's own input, N the input rotated by the width-N
 * Hadamard. False if it could not be written. */
inline bool write_gram(const std::string& path, int64_t k, int64_t rows,
                       const std::vector<float>& h, uint32_t domain = 0) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    char hdr[32] = { 'R', 'A', 'D', 'G', 'R', 'A', 'M', '2' };
    const uint32_t kk = (uint32_t)k, rank = 0;
    const uint64_t rr = (uint64_t)rows;
    std::memcpy(hdr + 8, &kk, 4);
    std::memcpy(hdr + 12, &rank, 4);
    std::memcpy(hdr + 16, &rr, 8);
    std::memcpy(hdr + 24, &domain, 4);
    bool ok = std::fwrite(hdr, 1, 32, f) == 32;
    ok = ok && std::fwrite(h.data(), 4, h.size(), f) == h.size();
    return std::fclose(f) == 0 && ok;
}

}  /* namespace qfix */
