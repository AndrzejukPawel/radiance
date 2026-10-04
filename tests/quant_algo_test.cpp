/* quant_algo_test.cpp -- libquant's calibrated algorithms: GPTQ's act order, AWQ, AutoRound and
 * ParoQuant, each against round-to-nearest on the same grid.
 *
 * WHAT THEY ARE HELD TO is the number every one of them minimises: the layer's output error, the
 * Hessian's quadratic form sum_r (w_hat_r - w_r) H (w_hat_r - w_r)^T, with H the Gram of a
 * synthetic input whose channels are as uneven as a real layer's -- a few carry most of the
 * energy, and neighbours are correlated. The decoded weight is read back through the core's
 * decoder, so a transform or a column scale the quantiser wrote and the reader did not undo is an
 * error of order one, not a near miss. Each is also run twice at two thread counts and must give
 * the same bytes: its output is a file.
 */
#include "quant_fixture.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
using namespace qfix;

namespace {

constexpr int64_t kR = 64, kC = 256, kT = 4096;

/* The input: per-channel energy spread over two decades, a few channels ten times the rest, and
 * each channel mixed with its neighbours. */
std::vector<float> gram() {
    const std::vector<float> z = draw(kT * kC, 101);
    std::vector<float> amp((size_t)kC);
    for (int64_t c = 0; c < kC; ++c)
        amp[(size_t)c] = (float)std::pow(10.0, (double)((c * 37) % kC) / kC * 2.0) *
                         ((c % 31 == 3) ? 10.0f : 1.0f);
    std::vector<float> x((size_t)(kT * kC));
    for (int64_t t = 0; t < kT; ++t)
        for (int64_t c = 0; c < kC; ++c) {
            const float* zr = &z[(size_t)(t * kC)];
            const float v = zr[c] + 0.5f * zr[(c + 1) % kC] + 0.25f * zr[(c + 7) % kC];
            x[(size_t)(t * kC + c)] = v * amp[(size_t)c];
        }
    std::vector<double> h((size_t)(kC * kC), 0.0);
    for (int64_t t = 0; t < kT; ++t)
        for (int64_t i = 0; i < kC; ++i) {
            const double xi = x[(size_t)(t * kC + i)];
            for (int64_t j = 0; j < kC; ++j) h[(size_t)(i * kC + j)] += xi * x[(size_t)(t * kC + j)];
        }
    std::vector<float> hf(h.size());
    for (size_t i = 0; i < h.size(); ++i) hf[i] = (float)h[i];
    return hf;
}

const std::vector<float>& H() {
    static const std::vector<float> h = gram();
    return h;
}

/* A calibration directory holding the Gram for `stem`, made once. */
const std::string& calib_dir() {
    static std::string d;
    if (d.empty()) {
        char tmpl[] = "/tmp/rad_quant_algo_XXXXXX";
        if (mkdtemp(tmpl)) {
            d = tmpl;
            write_gram(d + "/gram.r0.blk.2.attn_q.bin", kC, kT, H());
        }
    }
    return d;
}

double out_err(const std::vector<float>& w, const std::vector<float>& d) {
    double e = 0.0;
    const std::vector<float>& h = H();
    for (int64_t r = 0; r < kR; ++r)
        for (int64_t i = 0; i < kC; ++i) {
            const double di = (double)d[(size_t)(r * kC + i)] - w[(size_t)(r * kC + i)];
            if (di == 0.0) continue;
            double hi = 0.0;
            for (int64_t j = 0; j < kC; ++j)
                hi += (double)h[(size_t)(i * kC + j)] *
                      ((double)d[(size_t)(r * kC + j)] - w[(size_t)(r * kC + j)]);
            e += di * hi;
        }
    return e;
}

const std::vector<float>& weight() {
    static const std::vector<float> w = draw(kR * kC, 102, 6.0f);
    return w;
}

const char* kName = "blk.2.attn_q.weight";

/* The thread count of the OpenMP runtime the plugin loaded, so a run can be repeated at another. */
void threads(int n) {
    using Fn = void (*)(int);
    if (Fn f = (Fn)dlsym(RTLD_DEFAULT, "omp_set_num_threads")) f(n);
}

/* rtn, then the algorithm, on the same grid; the algorithm decoded, its output error below rtn's
 * by `margin`, and its bytes the same at one thread and at four. */
void beats_rtn(const char* algo, Opts grid, Opts extra, double margin, const char* want_enc) {
    const std::vector<float>& w = weight();
    Quantised r = quantise(quantizer("rtn"), grid, kName, kR, kC, w);
    REQUIRE_EQ(r.status, RAD_OK);
    Opts o = grid;
    for (size_t i = 0; i < extra.p.size(); ++i) {
        if (extra.p[i].kind == RAD_P_STR) o.s(extra.keys[i].c_str(), extra.vals[i].c_str());
        else if (extra.p[i].kind == RAD_P_INT) o.i(extra.keys[i].c_str(), extra.p[i].ival);
        else o.f(extra.keys[i].c_str(), extra.p[i].dval);
    }
    o.s("calib", calib_dir().c_str());
    threads(1);
    Quantised a = quantise(quantizer(algo), o, kName, kR, kC, w);
    threads(4);
    Quantised b = quantise(quantizer(algo), o, kName, kR, kC, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    if (want_enc) CHECK_EQ(enc_name(a.enc), std::string(want_enc));
    int same = 1;
    for (int i = 0; i < a.enc.n_planes; ++i) same &= a.planes[(size_t)i] == b.planes[(size_t)i];
    CHECK(same);
    const std::vector<float> dr = decode(r, kR, kC), da = decode(a, kR, kC);
    REQUIRE(!da.empty());
    const double er = out_err(w, dr), ea = out_err(w, da);
    std::fprintf(stderr, "    %s: output error %.4g of rtn's, weight error %.4f (rtn %.4f)\n",
                 algo, ea / er, rel_err(w, da), rel_err(w, dr));
    CHECK(ea < er * margin);
}

}  /* namespace */

TEST(the_calibrated_quantisers_are_in_the_registry) {
    NEED_LIBQUANT();
    for (const char* q : { "awq", "autoround", "paroquant", "cast" }) CHECK(quantizer(q) != nullptr);
}

TEST(gptq_act_order_stores_the_permutation_it_quantised_in) {
    NEED_LIBQUANT();
    const std::vector<float>& w = weight();
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "f16").i("act_order", 1)
     .s("calib", calib_dir().c_str());
    Quantised q = quantise(quantizer("gptq"), o, kName, kR, kC, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("i4*f16[1x32]/perm"));
    /* the permutation is the columns by descending Hessian diagonal */
    std::vector<int32_t> perm((size_t)kC);
    std::memcpy(perm.data(), q.planes[2].data(), (size_t)kC * 4);
    int bad = 0;
    for (int64_t j = 1; j < kC; ++j)
        bad += H()[(size_t)(perm[(size_t)j - 1] * (kC + 1))] < H()[(size_t)(perm[(size_t)j] * (kC + 1))];
    CHECK_EQ(bad, 0);
    Opts n;
    n.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", calib_dir().c_str());
    Opts m;
    m.s("codes", "i4").i("group", 32).s("scale", "f16");
    Quantised g = quantise(quantizer("gptq"), n, kName, kR, kC, w);
    Quantised r = quantise(quantizer("rtn"), m, kName, kR, kC, w);
    const double ea = out_err(w, decode(q, kR, kC)), eg = out_err(w, decode(g, kR, kC));
    const double er = out_err(w, decode(r, kR, kC));
    std::fprintf(stderr, "    output error of rtn's: gptq %.4f, gptq act order %.4f\n", eg / er,
                 ea / er);
    CHECK(ea < er * 0.8);
}

/* The width-n Hadamard over each n-block of a row, unnormalised: libquant's own butterflies, in
 * its order, so a file this writes is the bytes libquant would make of the model-domain one. */
void fwht_rows(std::vector<double>& h, int64_t k, int64_t n) {
    for (int64_t r = 0; r < k; ++r) {
        double* v = h.data() + r * k;
        for (int64_t b0 = 0; b0 < k; b0 += n)
            for (int64_t s = 1; s < n; s *= 2)
                for (int64_t i = b0; i < b0 + n; i += 2 * s)
                    for (int64_t j = i; j < i + s; ++j) {
                        const double x = v[j], y = v[j + s];
                        v[j] = x + y;
                        v[j + s] = x - y;
                    }
    }
}

TEST(a_hessian_is_brought_into_the_domain_its_grid_quantises_in) {
    /* The same Gram twice over: as the model's input (domain 0, what a bf16 model's tap writes)
     * and rotated by the width-128 Hadamard (domain 128, what a rotated container's tap writes).
     * A rotated grid quantised against either must come out byte for byte the same -- the reader
     * rotates the first exactly as the second was rotated -- and an unrotated grid against the
     * rotated file must too, after the reader rotates it back. */
    NEED_LIBQUANT();
    const std::vector<float>& hf = H();
    std::vector<double> h(hf.begin(), hf.end());
    fwht_rows(h, kC, 128);
    std::vector<double> t((size_t)(kC * kC));
    for (int64_t r = 0; r < kC; ++r)
        for (int64_t c = 0; c < kC; ++c) t[(size_t)(c * kC + r)] = h[(size_t)(r * kC + c)];
    fwht_rows(t, kC, 128);
    std::vector<float> rot((size_t)(kC * kC));
    for (int64_t r = 0; r < kC; ++r)
        for (int64_t c = 0; c < kC; ++c) rot[(size_t)(r * kC + c)] = (float)t[(size_t)(c * kC + r)];

    char tmpl[] = "/tmp/rad_quant_dom_XXXXXX";
    REQUIRE(mkdtemp(tmpl) != nullptr);
    const std::string dm = tmpl;
    REQUIRE(write_gram(dm + "/gram.r0.blk.2.attn_q.bin", kC, kT, rot, 128));

    auto run = [&](const std::string& dir, bool rotated) {
        Opts o;
        o.s("codes", "i4").i("group", 128).s("scale", "f16").s("calib", dir.c_str());
        if (rotated) o.s("transform", "fwht128");
        return quantise(quantizer("gptq"), o, kName, kR, kC, weight());
    };
    const Quantised a = run(calib_dir(), true), b = run(dm, true);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    int same = a.enc.n_planes == b.enc.n_planes;
    for (int i = 0; same && i < a.enc.n_planes; ++i) same &= a.planes[(size_t)i] == b.planes[(size_t)i];
    CHECK(same);

    /* And the way back: a rotated file read for an unrotated grid is the model's Gram up to the
     * rounding of a rotation there and back, so the two quantise within a hair of each other. */
    const Quantised c = run(calib_dir(), false), d = run(dm, false);
    REQUIRE_EQ(c.status, RAD_OK);
    REQUIRE_EQ(d.status, RAD_OK);
    const double ec = out_err(weight(), decode(c, kR, kC)), ed = out_err(weight(), decode(d, kR, kC));
    std::fprintf(stderr, "    unrotated grid: model-domain %.6g, back from rotated %.6g\n", ec, ed);
    CHECK(std::fabs(ec - ed) <= 1e-3 * ec);
    std::remove((dm + "/gram.r0.blk.2.attn_q.bin").c_str());
    rmdir(dm.c_str());
}

TEST(an_expert_reads_its_own_hessian_before_its_layers) {
    /* A layer's pooled Gram and one expert's own, which differ (the own one is the pooled one with
     * its channels reversed). The expert with a file of its own must quantise exactly as against
     * that Gram alone, wherever its rank put it; an expert with no file of its own, and one whose
     * file says fewer rows than libquant trusts an own Gram with, exactly as against the pooled
     * one. And the two answers must differ, or the test could not tell them apart. */
    NEED_LIBQUANT();
    const std::vector<float>& pool = H();
    std::vector<float> own((size_t)(kC * kC));
    for (int64_t i = 0; i < kC; ++i)
        for (int64_t j = 0; j < kC; ++j)
            own[(size_t)(i * kC + j)] = pool[(size_t)((kC - 1 - i) * kC + (kC - 1 - j))];

    char t1[] = "/tmp/rad_quant_exp_XXXXXX", t2[] = "/tmp/rad_quant_own_XXXXXX",
         t3[] = "/tmp/rad_quant_pool_XXXXXX";
    REQUIRE(mkdtemp(t1) && mkdtemp(t2) && mkdtemp(t3));
    const std::string both = t1, only_own = t2, only_pool = t3;
    const std::vector<std::string> files = {
        both + "/gram.r0.blk.2.ffn_down_exps.bin", both + "/gram.r1.blk.2.ffn_down_exps.7.bin",
        both + "/gram.r0.blk.2.ffn_down_exps.9.bin", only_own + "/gram.r0.blk.2.ffn_down_exps.bin",
        only_pool + "/gram.r0.blk.2.ffn_down_exps.bin" };
    REQUIRE(write_gram(files[0], kC, kT, pool));
    /* libquant's lq::kOwnRowsPerK: the rows an own Gram needs, in units of K */
    constexpr int64_t kOwnRowsPerK = 4;
    REQUIRE(write_gram(files[1], kC, kOwnRowsPerK * kC, own));
    REQUIRE(write_gram(files[2], kC, kOwnRowsPerK * kC - 1, own));
    REQUIRE(write_gram(files[3], kC, kT, own));
    REQUIRE(write_gram(files[4], kC, kT, pool));

    auto run = [&](const std::string& dir, const char* name) -> Quantised {
        Opts o;
        o.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", dir.c_str());
        Quantised q = quantise(quantizer("gptq"), o, name, kR, kC, weight());
        CHECK_EQ(q.status, RAD_OK);
        return q;
    };
    auto same = [](const Quantised& a, const Quantised& b) {
        int s = a.enc.n_planes == b.enc.n_planes;
        for (int i = 0; s && i < a.enc.n_planes; ++i) s &= a.planes[(size_t)i] == b.planes[(size_t)i];
        return s != 0;
    };
    const char* e7 = "blk.2.ffn_down_exps.7.weight";
    const char* e9 = "blk.2.ffn_down_exps.9.weight";
    const char* e11 = "blk.2.ffn_down_exps.11.weight";
    CHECK(same(run(both, e7), run(only_own, e7)));
    CHECK(!same(run(both, e7), run(only_pool, e7)));
    CHECK(same(run(both, e9), run(only_pool, e9)));
    CHECK(same(run(both, e11), run(only_pool, e11)));
    for (const std::string& f : files) std::remove(f.c_str());
    for (const std::string& d : { both, only_own, only_pool }) rmdir(d.c_str());
}

TEST(awq_scales_the_columns_the_input_says_matter) {
    NEED_LIBQUANT();
    Opts g, x;
    g.s("codes", "i4").i("group", 64).s("scale", "f16");
    beats_rtn("awq", g, x, 0.9, "i4*f16[1x64]*f16[*x1]");
    /* asymmetric, and over a codebook */
    Opts g2, g3;
    g2.s("codes", "u3").s("zero", "u4").i("group", 64).s("scale", "f16");
    beats_rtn("awq", g2, x, 0.95, nullptr);
    g3.s("table", "nf4").i("group", 64).s("scale", "bf16");
    beats_rtn("awq", g3, x, 0.95, nullptr);
}

TEST(autoround_descends_below_round_to_nearest) {
    NEED_LIBQUANT();
    Opts g, x, g2;
    g.s("codes", "i4").i("group", 32).s("scale", "f16");
    x.i("iters", 60);
    beats_rtn("autoround", g, x, 0.8, "i4*f16[1x32]");
    g2.s("codes", "u2").s("zero", "u4").i("group", 64).s("scale", "f16");
    beats_rtn("autoround", g2, x, 0.9, nullptr);
}

TEST(paroquant_learns_a_rotation_the_reader_undoes) {
    NEED_LIBQUANT();
    Opts g, x;
    g.s("codes", "u4").s("zero", "u4").i("group", 128).s("scale", "f16");
    x.i("steps", 60).i("iters", 40).i("sample_rows", 32);
    beats_rtn("paroquant", g, x, 0.85, "u4*f16[1x128]-u4[1x128]/givens");
    /* the rotation alone, stage two off, is already below rounding to nearest */
    Opts x1;
    x1.i("steps", 60).i("iters", 0).i("sample_rows", 32);
    beats_rtn("paroquant", g, x1, 0.95, nullptr);
    /* the pairs: every stage a perfect matching inside each 128-channel group */
    Opts ident = g;
    ident.i("steps", 0).i("iters", 0);
    Quantised q = quantise(quantizer("paroquant"), ident, kName, kR, kC, weight());
    REQUIRE_EQ(q.status, RAD_OK);
    const int32_t* p = (const int32_t*)q.planes[4].data();
    int bad = 0;
    for (int s = 0; s < 8; ++s) {
        std::vector<int> seen((size_t)kC, 0);
        for (int64_t k = 0; k < kC; k += 2) {
            const int32_t a = p[s * kC + k], b = p[s * kC + k + 1];
            bad += a / 128 != b / 128 || a == b;
            ++seen[(size_t)a];
            ++seen[(size_t)b];
        }
        for (int v : seen) bad += v != 1;
    }
    CHECK_EQ(bad, 0);
}

TEST(a_calibrated_quantiser_refuses_a_grid_it_cannot_serve) {
    NEED_LIBQUANT();
    RadQuantWeight qw{};
    qw.name = kName;
    qw.rank = 2; qw.shape[0] = kR; qw.shape[1] = kC; qw.rows = kR; qw.cols = kC;
    RadEncoding e;
    auto refused = [&](const char* q, Opts o) {
        return quantizer(q)->encoding(o.data(), o.n(), &qw, &e) != RAD_OK;
    };
    { Opts o; o.s("codes", "i4").s("transform", "fwht128");              CHECK(refused("awq", o)); }
    { Opts o; o.s("codes", "fp4_e2m1").i("group", 32);                   CHECK(refused("autoround", o)); }
    { Opts o; o.s("codes", "i4").s("block", "128x128");                  CHECK(refused("autoround", o)); }
    { Opts o; o.s("codes", "i4").s("transform", "fwht128");              CHECK(refused("paroquant", o)); }
    { Opts o; o.s("codes", "i4").i("rot_group", 96);                     CHECK(refused("paroquant", o)); }
    { Opts o; o.s("codes", "i4").i("act_order", 1).s("transform", "fwht128"); CHECK(refused("gptq", o)); }
}

RAD_TEST_MAIN()
