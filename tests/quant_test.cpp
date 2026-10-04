/* quant_test.cpp -- libquant through the plugin registry, read back through the core's decoder.
 *
 * Every case quantises through the quantiser's own hooks, exactly as rad-convert will, and checks
 * the result through enc_decode_rows rather than through anything libquant says about itself: the
 * point of an encoding is that a reader who never saw the quantiser recovers the weight. Where a
 * rule is exact -- a block scale that is the bf16 of amax/448, an MX exponent that is a power of
 * two, a fixed scale -- the exact value is checked; where it is approximate, the error is bounded
 * by the grid's own step, which a wrong block, a wrong zero or a missing rotation all exceed.
 */
#include "quant_fixture.h"

#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
using namespace qfix;

TEST(libquant_offers_its_quantisers_through_the_registry) {
    NEED_LIBQUANT();
    CHECK(quantizer("rtn") != nullptr);
    CHECK(quantizer("gptq") != nullptr);
    CHECK(quantizer("no-such-quantiser") == nullptr);
    /* a second copy of the same plugin is refused, not merged */
    Registry r2;
    CHECK_EQ(r2.load_plugin(home() + "/quantizers/libquant.so", 0), RAD_OK);
    CHECK(r2.load_plugin(home() + "/quantizers/libquant.so", 1) < 0);
}

TEST(block_fp8_is_the_bf16_of_amax_over_448_and_codes_against_it) {
    NEED_LIBQUANT();
    const int64_t R = 256, C = 384;
    const std::vector<float> w = draw(R * C, 1, 20.0f);
    Opts o;
    o.s("codes", "fp8_e4m3").s("block", "128x128").s("scale", "bf16");
    Quantised q = quantise(quantizer("rtn"), o, "blk.0.attn_q.weight", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp8_e4m3*bf16[128x128]"));
    REQUIRE_EQ(q.planes[1].size(), (size_t)(2 * 3 * 2));
    /* block (1, 2) */
    float amax = 0;
    for (int64_t r = 128; r < 256; ++r)
        for (int64_t c = 256; c < 384; ++c)
            amax = std::fmax(amax, std::fabs(w[(size_t)(r * C + c)]));
    const uint16_t want = rad_f32_to_bf16(amax * (1.0f / 448.0f));
    uint16_t got;
    std::memcpy(&got, q.planes[1].data() + (1 * 3 + 2) * 2, 2);
    CHECK_EQ((int)got, (int)want);
    const std::vector<float> d = decode(q, R, C);
    REQUIRE(!d.empty());
    CHECK(rel_err(w, d) < 0.04);
}

TEST(a_rotated_int4_group_decodes_back_through_the_hadamard) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 512;
    const std::vector<float> w = draw(R * C, 2, 30.0f);
    Opts plain, rot;
    plain.s("codes", "i4").i("group", 128).s("scale", "bf16");
    rot.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    Quantised a = quantise(quantizer("rtn"), plain, "x", R, C, w);
    Quantised b = quantise(quantizer("rtn"), rot, "x", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK_EQ(enc_name(b.enc), std::string("i4*bf16[1x128]/fwht128"));
    const double ea = rel_err(w, decode(a, R, C)), eb = rel_err(w, decode(b, R, C));
    std::fprintf(stderr, "    int4 g128: plain %.4f, rotated %.4f\n", ea, eb);
    /* Both are a 4-bit grid's error, and the rotated one is decoded THROUGH the rotation, so a
     * reader that forgot it would see noise, ~1.0. On weights with outliers the rotation is also
     * the better grid: it spreads one large value over the group instead of letting it set the
     * scale every other value is rounded against. */
    CHECK(ea < 0.35);
    CHECK(eb < 0.15);
    CHECK(eb < ea);
}

TEST(the_w4nl_table_is_exact_in_e4m3_and_beats_minus_8_to_7_after_the_rotation) {
    NEED_LIBQUANT();
    /* The sixteen levels libr4d's w4nla8h kernel decodes, in its code order, as a container stores
     * them; and against -8..7 at the same bytes, under the same rotation and scale search, they
     * are the better grid for the near-normal weight they were fitted to. Not with outliers: the
     * rotation turns one outlier into a +-o pattern across its whole group, which is bimodal. */
    const int64_t R = 32, C = 512;
    const std::vector<float> w = draw(R * C, 2);
    Opts i4, nl;
    i4.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
      .s("rule", "search");
    nl.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
      .s("rule", "search");
    Quantised a = quantise(quantizer("rtn"), i4, "x", R, C, w);
    Quantised b = quantise(quantizer("rtn"), nl, "x", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK_EQ(enc_name(b.enc),
             std::string("affine:codes=u4[1x1],scale=bf16[1x128],table=f32{1x16}/fwht128"));
    const int t = rad_enc_find(&b.enc, "table");
    REQUIRE(t >= 0);
    const float want[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                             -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };
    float got[16];
    std::memcpy(got, b.planes[(size_t)t].data(), sizeof got);
    for (int i = 0; i < 16; ++i) {
        CHECK_EQ(got[i], want[i]);
        CHECK_EQ(rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(want[i])), want[i]);
    }
    const double ea = rel_err(w, decode(a, R, C)), eb = rel_err(w, decode(b, R, C));
    std::fprintf(stderr, "    rotated g128, searched scale: -8..7 %.4f, w4nl %.4f\n", ea, eb);
    CHECK(eb < ea * 0.95);
}

TEST(an_asymmetric_two_bit_grid_keeps_its_zero_in_range) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 256;
    const std::vector<float> w = draw(R * C, 3);
    Opts o;
    o.s("codes", "u2").i("group", 128).s("scale", "f16").s("zero", "u8");
    Quantised q = quantise(quantizer("rtn"), o, "head", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("u2*f16[1x128]-u8[1x128]"));
    int bad = 0;
    for (uint8_t z : q.planes[2]) bad += z > 3;
    CHECK_EQ(bad, 0);
    CHECK(rel_err(w, decode(q, R, C)) < 0.6);
}

TEST(mxfp4_shares_a_power_of_two_a_32) {
    NEED_LIBQUANT();
    const int64_t R = 8, C = 256;
    const std::vector<float> w = draw(R * C, 4, 10.0f);
    Opts o;
    o.s("codes", "fp4_e2m1").i("group", 32).s("scale", "e8m0");
    Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp4_e2m1*e8m0[1x32]"));
    /* the OCP rule: floor(log2 amax) - 2, so amax / 2^e lands in [4, 8) */
    for (int64_t r = 0; r < R; ++r)
        for (int64_t g = 0; g < C / 32; ++g) {
            float amax = 0;
            for (int64_t c = g * 32; c < g * 32 + 32; ++c)
                amax = std::fmax(amax, std::fabs(w[(size_t)(r * C + c)]));
            const float s = rad_e8m0_to_f32(q.planes[1][(size_t)(r * (C / 32) + g)]);
            CHECK(amax / s >= 4.0f && amax / s < 8.0f);
        }
    CHECK(rel_err(w, decode(q, R, C)) < 0.2);
}

TEST(a_fixed_per_tensor_scale_streams_a_row_at_a_time) {
    NEED_LIBQUANT();
    const int64_t R = 64, C = 160;
    const std::vector<float> w = draw(R * C, 5);
    Opts o;
    o.s("codes", "fp8_e4m3").s("block", "*x*").s("scale", "bf16").s("rule", "fixed")
     .f("scale_value", 1.99317932128906e-4);
    const RadQuantizerInfo* q = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "t";
    qw.rank = 2; qw.shape[0] = R; qw.shape[1] = C; qw.rows = R; qw.cols = C;
    CHECK_EQ(q->row_block(o.data(), o.n(), &qw), 1);
    Quantised a = quantise(q, o, "t", R, C, w);
    REQUIRE_EQ(a.status, RAD_OK);
    CHECK_EQ(enc_name(a.enc), std::string("fp8_e4m3*bf16[*x*]"));
    uint16_t s;
    std::memcpy(&s, a.planes[1].data(), 2);
    CHECK_EQ((int)s, 0x3951);
    /* the rule's reciprocal is the unrounded value's, in double */
    const double inv = 1.0 / 1.99317932128906e-4;
    int off = 0;
    for (int64_t i = 0; i < R * C; ++i)
        off += a.planes[0][(size_t)i] != rad_f32_to_fp8e4m3((float)((double)w[(size_t)i] * inv));
    CHECK_EQ(off, 0);
}

TEST(gptq_without_a_hessian_is_round_to_nearest_byte_for_byte) {
    NEED_LIBQUANT();
    const int64_t R = 64, C = 256;
    const std::vector<float> w = draw(R * C, 6, 20.0f);
    Opts a, b;
    a.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    b.s("codes", "i4").i("group", 128).s("scale", "bf16").s("transform", "fwht128");
    Quantised r = quantise(quantizer("rtn"), a, "blk.0.ffn_down_exps.0.weight", R, C, w);
    Quantised g = quantise(quantizer("gptq"), b, "blk.0.ffn_down_exps.0.weight", R, C, w);
    REQUIRE_EQ(r.status, RAD_OK);
    REQUIRE_EQ(g.status, RAD_OK);
    for (int i = 0; i < r.enc.n_planes; ++i) CHECK(r.planes[(size_t)i] == g.planes[(size_t)i]);
}

TEST(gptq_with_a_hessian_lowers_the_output_error) {
    NEED_LIBQUANT();
    /* Correlated activations: x = A z, so H = sum x x^T is far from diagonal and error feedback
     * has something to exploit that rounding each column alone cannot. */
    const int64_t R = 48, C = 128, T = 2048;
    const std::vector<float> w = draw(R * C, 7, 10.0f);
    const std::vector<float> z = draw(T * C, 8);
    const std::vector<float> mix = draw(C * C, 9);
    std::vector<float> x((size_t)(T * C), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < C; ++i) {
            double s = z[(size_t)(t * C + i)] * 4.0;
            for (int64_t j = 0; j < 8; ++j)
                s += mix[(size_t)(i * C + j)] * z[(size_t)(t * C + j)] * 40.0;
            x[(size_t)(t * C + i)] = (float)s;
        }
    std::vector<float> h((size_t)(C * C), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < C; ++i)
            for (int64_t j = 0; j < C; ++j)
                h[(size_t)(i * C + j)] += x[(size_t)(t * C + i)] * x[(size_t)(t * C + j)];

    char dir[] = "/tmp/rad_quant_test_XXXXXX";
    REQUIRE(mkdtemp(dir) != nullptr);
    const std::string path = std::string(dir) + "/gram.r0.blk.3.ffn_down_exps.bin";
    REQUIRE(write_gram(path, C, T, h));

    Opts a, b;
    a.s("codes", "i4").i("group", 32).s("scale", "f32");
    b.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", dir);
    Quantised r = quantise(quantizer("rtn"), a, "blk.3.ffn_down_exps.17.weight", R, C, w);
    Quantised g = quantise(quantizer("gptq"), b, "blk.3.ffn_down_exps.17.weight", R, C, w);
    REQUIRE_EQ(r.status, RAD_OK);
    REQUIRE_EQ(g.status, RAD_OK);
    const std::vector<float> dr = decode(r, R, C), dg = decode(g, R, C);
    auto out_err = [&](const std::vector<float>& d) {
        double e = 0, n = 0;
        for (int64_t t = 0; t < T; t += 4)
            for (int64_t o = 0; o < R; ++o) {
                double a1 = 0, a2 = 0;
                for (int64_t i = 0; i < C; ++i) {
                    a1 += (double)w[(size_t)(o * C + i)] * x[(size_t)(t * C + i)];
                    a2 += (double)d[(size_t)(o * C + i)] * x[(size_t)(t * C + i)];
                }
                e += (a1 - a2) * (a1 - a2);
                n += a1 * a1;
            }
        return std::sqrt(e / n);
    };
    const double er = out_err(dr), eg = out_err(dg);
    std::fprintf(stderr, "    output error: rtn %.4f, gptq %.4f\n", er, eg);
    CHECK(eg < er * 0.9);

    /* COORDINATE DESCENT AFTER IT never raises the form it descends, which on the corpus the
     * Gram came from is the output error, and on a correlated Hessian it finds codes to move. */
    Opts c;
    c.s("codes", "i4").i("group", 32).s("scale", "f32").s("calib", dir).i("cd", 3);
    Quantised gc = quantise(quantizer("gptq"), c, "blk.3.ffn_down_exps.17.weight", R, C, w);
    REQUIRE_EQ(gc.status, RAD_OK);
    const double ec = out_err(decode(gc, R, C));
    std::fprintf(stderr, "    output error: gptq + cd=3 %.4f\n", ec);
    CHECK(ec < eg);
    /* The scales are GPTQ's, untouched: only codes move. */
    CHECK(gc.planes[1] == g.planes[1]);
    std::remove(path.c_str());
    rmdir(dir);
}

TEST(w4nl_at_64_with_e4m3_scales_under_a_row_scale_beats_128_with_bf16_at_the_same_bytes) {
    NEED_LIBQUANT();
    /* The rotated expert grid two ways at the same bytes a row: a bf16 scale a 128 columns, or an
     * E4M3 one a 64 under a bf16 scale a row (whose bytes are a row's, not a group's). The second
     * searches its sub-scales as they are stored -- rounded to E4M3 relative to the row's -- so
     * its rounding does not give back what the finer groups won. Rows whose magnitude varies
     * along K, so a 64-column group is worth having. */
    const int64_t R = 64, C = 1024;
    std::vector<float> w = draw(R * C, 21, 1.0f);
    for (int64_t r = 0; r < R; ++r)
        for (int64_t c = 0; c < C; ++c)
            w[(size_t)(r * C + c)] *= 1.0f + 3.0f * (float)((c / 64 + r) % 5) / 4.0f;
    Opts a, b;
    a.s("table", "w4nl").i("group", 128).s("scale", "bf16").s("transform", "fwht128")
     .s("rule", "search");
    b.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "bf16")
     .s("block2", "1x*").s("transform", "fwht128").s("rule", "search");
    Quantised qa = quantise(quantizer("rtn"), a, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    std::fprintf(stderr, "    %s\n    %s\n", enc_name(qa.enc).c_str(), enc_name(qb.enc).c_str());
    size_t ba = 0, bb = 0;
    for (const auto& p : qa.planes) ba += p.size();
    for (const auto& p : qb.planes) bb += p.size();
    const double ea = rel_err(w, decode(qa, R, C)), eb = rel_err(w, decode(qb, R, C));
    std::fprintf(stderr, "    g128 bf16: %.4f (%zu bytes)   g64 e4m3 + row bf16: %.4f (%zu bytes)\n",
                 ea, ba, eb, bb);
    CHECK(eb < ea * 0.97);
    CHECK(bb <= ba + (size_t)R * 2 + 64);

    /* THE SAME GRID WITH THE SECOND LEVEL FIXED, one f32 for the tensor: the E4M3 sub-scales
     * carry every row's magnitude themselves, which they can because this weight's scales sit well
     * inside E4M3's range -- and the error is the per-row form's, at a row's two bytes fewer. */
    Opts c;
    c.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "f32")
     .s("block2", "*x*").f("scale2_value", 0x1p-13).s("transform", "fwht128").s("rule", "search");
    Quantised qc = quantise(quantizer("rtn"), c, "blk.0.ffn_gate_up_exps.0.weight", R, C, w);
    REQUIRE_EQ(qc.status, RAD_OK);
    std::fprintf(stderr, "    %s\n", enc_name(qc.enc).c_str());
    float s2 = 0.0f;
    std::memcpy(&s2, qc.planes[2].data(), 4);
    CHECK_EQ(s2, 0x1p-13f);
    const double ec = rel_err(w, decode(qc, R, C));
    std::fprintf(stderr, "    g64 e4m3 under a fixed 2^-13: %.4f\n", ec);
    CHECK(ec < ea * 0.97);
    CHECK(std::fabs(ec - eb) < 0.002);
    /* A value the second level's dtype cannot hold exactly is refused. */
    Opts d;
    d.s("table", "w4nl").i("group", 64).s("scale", "fp8_e4m3").s("scale2", "bf16")
     .s("block2", "*x*").f("scale2_value", 0.1).s("transform", "fwht128");
    CHECK(quantise(quantizer("rtn"), d, "x", R, C, w).status != RAD_OK);
}

TEST(gptq_cd_and_act_order_are_refused_together) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 128;
    const std::vector<float> w = draw(R * C, 12, 4.0f);
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "f32").i("act_order", 1).i("cd", 2);
    Quantised q = quantise(quantizer("gptq"), o, "blk.0.ffn_down_exps.0.weight", R, C, w);
    CHECK(q.status != RAD_OK);
}

TEST(a_codebook_indexes_its_table_and_the_scale_multiplies_it) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 256;
    const std::vector<float> w = draw(R * C, 11, 8.0f);
    /* NF4: sixteen f32 quantiles, a u4 code each, the extremes at +-1 so the scale is absmax */
    {
        Opts o;
        o.s("table", "nf4").i("group", 64).s("scale", "bf16");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(enc_name(q.enc),
                 std::string("affine:codes=u4[1x1],scale=bf16[1x64],table=f32{1x16}"));
        float t[16];
        std::memcpy(t, q.planes[2].data(), sizeof t);
        CHECK_EQ(t[0], -1.0f);
        CHECK_EQ(t[7], 0.0f);
        CHECK_EQ(t[15], 1.0f);
        const double e = rel_err(w, decode(q, R, C));
        std::fprintf(stderr, "    nf4 g64: %.4f\n", e);
        CHECK(e < 0.15);
    }
    /* IQ4_NL's values: unequal ends, so a block's scale takes its largest value's sign and that
     * value lands on -127 exactly (to the bf16 of the scale) */
    {
        Opts o;
        o.s("table", "iq4nl").i("group", 32).s("scale", "f32");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        const std::vector<float> d = decode(q, R, C);
        REQUIRE(!d.empty());
        int off = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t g = 0; g < C / 32; ++g) {
                int64_t at = g * 32;
                for (int64_t c = g * 32; c < g * 32 + 32; ++c)
                    if (std::fabs(w[(size_t)(r * C + c)]) > std::fabs(w[(size_t)(r * C + at)]))
                        at = c;
                off += std::fabs(d[(size_t)(r * C + at)] - w[(size_t)(r * C + at)]) >
                       1e-6f * std::fabs(w[(size_t)(r * C + at)]);
            }
        CHECK_EQ(off, 0);
        CHECK(rel_err(w, d) < 0.15);
    }
    /* one bit: every value decodes to plus or minus its block's scale */
    {
        Opts o;
        o.s("table", "binary").i("group", 128).s("scale", "bf16");
        Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(q.enc.plane[0].dtype, (uint32_t)RAD_U1);
        const std::vector<float> d = decode(q, R, C);
        int bad = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t c = 0; c < C; ++c) {
                const float s = rad_bf16_to_f32(((const uint16_t*)q.planes[1].data())[r * 2 + c / 128]);
                bad += std::fabs(d[(size_t)(r * C + c)]) != s ||
                       (w[(size_t)(r * C + c)] < 0) != (d[(size_t)(r * C + c)] < 0);
            }
        CHECK_EQ(bad, 0);
    }
}

TEST(nvfp4_is_two_scale_levels_the_second_a_tensor) {
    NEED_LIBQUANT();
    const int64_t R = 16, C = 256;
    const std::vector<float> w = draw(R * C, 12, 30.0f);
    Opts o;
    o.s("codes", "fp4_e2m1").i("group", 16).s("scale", "fp8_e4m3").s("scale2", "f32");
    Quantised q = quantise(quantizer("rtn"), o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]"));
    /* the second level is the widest block's ideal scale, amax / 6, over E4M3's top, so that
     * block's sub-scale is 448 */
    float amax = 0;
    for (float v : w) amax = std::fmax(amax, std::fabs(v));
    float s2;
    std::memcpy(&s2, q.planes[2].data(), 4);
    CHECK_EQ(s2, (amax / 6.0f) / 448.0f);
    int top = 0;
    for (uint8_t b : q.planes[1]) top += b == 0x7e;
    CHECK(top >= 1);
    const double e = rel_err(w, decode(q, R, C));
    std::fprintf(stderr, "    nvfp4: %.4f\n", e);
    CHECK(e < 0.15);
}

TEST(an_integer_sub_scale_under_a_superblock_scale) {
    NEED_LIBQUANT();
    /* a K-quant's shape: 4-bit codes a group of 32, a six-bit sub-scale each, an f16 a 256 */
    const int64_t R = 8, C = 512;
    const std::vector<float> w = draw(R * C, 13, 6.0f);
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "u6").s("scale2", "f16").i("group2", 256);
    const RadQuantizerInfo* rtn = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "x";
    qw.rank = 2; qw.shape[0] = R; qw.shape[1] = C; qw.rows = R; qw.cols = C;
    CHECK_EQ(rtn->row_block(o.data(), o.n(), &qw), 1);
    Quantised q = quantise(rtn, o, "x", R, C, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("i4*u6[1x32]*f16[1x256]"));
    /* each superblock's widest group sits at 63 and none is 0 */
    for (int64_t r = 0; r < R; ++r)
        for (int64_t sb = 0; sb < 2; ++sb) {
            int mx = 0, mn = 64;
            for (int64_t g = 0; g < 8; ++g) {
                const int v = (int)rad_load_f32(q.planes[1].data() + r * 12, RAD_U6, sb * 8 + g);
                mx = std::max(mx, v);
                mn = std::min(mn, v);
            }
            CHECK_EQ(mx, 63);
            CHECK(mn >= 1);
        }
    CHECK(rel_err(w, decode(q, R, C)) < 0.2);
}

TEST(sixteen_and_thirty_two_bit_codes_are_nearly_exact) {
    NEED_LIBQUANT();
    const int64_t R = 8, C = 128;
    const std::vector<float> w = draw(R * C, 14, 5.0f);
    Opts a, b;
    a.s("codes", "i16").s("scale", "f32");
    b.s("codes", "i32").s("scale", "f32");
    Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    CHECK_EQ(enc_name(qb.enc), std::string("i32*f32[1x*]"));
    const double ea = rel_err(w, decode(qa, R, C)), eb = rel_err(w, decode(qb, R, C));
    std::fprintf(stderr, "    i16 %.3g, i32 %.3g\n", ea, eb);
    CHECK(ea < 1e-4);
    CHECK(eb < 1e-6);
}

TEST(the_searched_scale_never_loses_to_absmax_and_weighs_by_importance) {
    NEED_LIBQUANT();
    const int64_t R = 32, C = 256;
    const std::vector<float> w = draw(R * C, 15, 12.0f);
    std::vector<float> imp((size_t)C);
    for (int64_t c = 0; c < C; ++c) imp[(size_t)c] = (c % 7 == 0) ? 50.0f : 1.0f;
    for (const char* codes : { "i4", "i3" }) {
        Opts a, b;
        a.s("codes", codes).i("group", 32).s("scale", "bf16");
        b.s("codes", codes).i("group", 32).s("scale", "bf16").s("rule", "search");
        Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w, imp.data());
        Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w, imp.data());
        REQUIRE_EQ(qa.status, RAD_OK);
        REQUIRE_EQ(qb.status, RAD_OK);
        const std::vector<float> da = decode(qa, R, C), db = decode(qb, R, C);
        /* per block, in the importance-weighted error the search minimises */
        int worse = 0;
        double ta = 0, tb = 0;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t g = 0; g < C / 32; ++g) {
                double ea = 0, eb = 0;
                for (int64_t c = g * 32; c < g * 32 + 32; ++c) {
                    const size_t i = (size_t)(r * C + c);
                    ea += imp[(size_t)c] * (double)(w[i] - da[i]) * (w[i] - da[i]);
                    eb += imp[(size_t)c] * (double)(w[i] - db[i]) * (w[i] - db[i]);
                }
                worse += eb > ea * (1 + 1e-9);
                ta += ea;
                tb += eb;
            }
        std::fprintf(stderr, "    %s weighted error: absmax %.4g, search %.4g\n", codes, ta, tb);
        CHECK_EQ(worse, 0);
        CHECK(tb < ta * 0.9);
    }
    /* and asymmetric */
    Opts a, b;
    a.s("codes", "u3").s("zero", "u4").i("group", 32).s("scale", "f16");
    b.s("codes", "u3").s("zero", "u4").i("group", 32).s("scale", "f16").s("rule", "search");
    Quantised qa = quantise(quantizer("rtn"), a, "x", R, C, w);
    Quantised qb = quantise(quantizer("rtn"), b, "x", R, C, w);
    REQUIRE_EQ(qa.status, RAD_OK);
    REQUIRE_EQ(qb.status, RAD_OK);
    CHECK(rel_err(w, decode(qb, R, C)) <= rel_err(w, decode(qa, R, C)));
}

TEST(cast_converts_to_nearest_even_with_no_scale) {
    NEED_LIBQUANT();
    const int64_t R = 4, C = 64;
    const std::vector<float> w = draw(R * C, 16, 100.0f);
    const struct { const char* dt; uint32_t code; } cases[] = {
        { "bf16", RAD_BF16 }, { "f16", RAD_F16 }, { "fp8_e4m3", RAD_F8E4M3 }, { "f32", RAD_F32 } };
    for (const auto& k : cases) {
        Opts o;
        o.s("dtype", k.dt);
        Quantised q = quantise(quantizer("cast"), o, "x", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(enc_name(q.enc), std::string(k.dt));
        int off = 0;
        for (int64_t i = 0; i < R * C; ++i) {
            std::vector<uint8_t> one(4);
            rad_store_f32(one.data(), k.code, 0, w[(size_t)i]);
            off += std::memcmp(one.data(), q.planes[0].data() + i * rad_dtype_bytes(k.code, 1),
                               (size_t)rad_dtype_bytes(k.code, 1)) != 0;
        }
        CHECK_EQ(off, 0);
    }
    Opts bad;
    bad.s("dtype", "i4");
    CHECK(quantise(quantizer("cast"), bad, "x", R, C, w).status != RAD_OK);
}

TEST(a_grid_the_options_do_not_describe_is_refused) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("rtn");
    RadQuantWeight qw{};
    qw.name = "x";
    qw.rank = 2; qw.shape[0] = 4; qw.shape[1] = 256; qw.rows = 4; qw.cols = 256;
    RadEncoding e;
    auto refused = [&](Opts o) { return q->encoding(o.data(), o.n(), &qw, &e) != RAD_OK; };
    { Opts o; o.s("codes", "u4").i("group", 32);                          CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("zero", "u8");                         CHECK(refused(o)); }
    { Opts o; o.s("codes", "fp4_e2m1").s("scale", "e8m0").s("rule", "absmax"); CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group", 32).s("block", "1x32");       CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("transform", "fwht96");                CHECK(refused(o)); }
    { Opts o; o.s("codes", "bf16");                                       CHECK(refused(o)); }
    { Opts o; o.s("table", "nf4").s("zero", "u8");                        CHECK(refused(o)); }
    { Opts o; o.s("table", "nf4").s("codes", "u8");                       CHECK(refused(o)); }
    { Opts o; o.s("table", "1;2;x");                                      CHECK(refused(o)); }
    { Opts o; o.s("table", "iq4nl").s("scale", "u6").s("scale2", "f16");  CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("scale", "u6");                        CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group", 32).s("scale2", "f16").i("group2", 48);
                                                                          CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").i("group2", 256);                        CHECK(refused(o)); }
    { Opts o; o.s("codes", "fp4_e2m1").s("scale", "e8m0").s("scale2", "f32"); CHECK(refused(o)); }
    { Opts o; o.s("codes", "i4").s("rule", "fixed").f("scale_value", 1.0).s("scale2", "f32");
                                                                          CHECK(refused(o)); }
    /* a vector is declined, not an error */
    RadQuantWeight v = qw;
    v.rank = 1; v.shape[0] = 256; v.rows = 1;
    {
        Opts o;
        o.s("codes", "i4");
        CHECK_EQ(q->encoding(o.data(), o.n(), &v, &e), RAD_E_UNSUPPORTED);
    }
}

RAD_TEST_MAIN()

/* A SCHEME OF A QUANTISER'S OWN IS READ THROUGH THAT QUANTISER. The core decodes plain and affine
 * and nothing else; for any other scheme enc_decode_rows asks the loaded quantisers' decode hooks
 * (abi/rad_quant.h) -- which is how the oracle, rad-kbench and rad-convert's error report read a
 * weight whose format the engine has never heard of. Unloaded, the same weight is refused by name. */
TEST(a_quantisers_own_scheme_decodes_through_its_hook_and_only_while_it_is_loaded) {
    const std::string so = home() + "/testquant/radtest_quant.so";
    struct stat st {};
    if (stat(so.c_str(), &st) != 0) {
        std::fprintf(stderr, "  SKIP no %s\n", so.c_str());
        return;
    }
    const int64_t R = 8, C = 64;
    const std::vector<float> w = draw(R * C, 7, 4.0f);
    Quantised q;
    {
        Registry r;
        REQUIRE_EQ(r.load_plugin(so, 0), RAD_OK);
        const Registry::QuantRow* qr = r.quantizer("radtest-halves");
        REQUIRE(qr && qr->info);
        Opts o;
        q = quantise(qr->info, o, "blk.0.ffn_down.weight", R, C, w);
        REQUIRE_EQ(q.status, RAD_OK);
        CHECK_EQ(std::string(q.enc.scheme), std::string("radtest-halves"));
        const std::vector<float> d = decode(q, R, C);
        REQUIRE_EQ(d.size(), w.size());
        for (size_t i = 0; i < w.size(); ++i) CHECK(std::fabs(d[i] - w[i]) <= 0.25f + 1e-6f);
        /* rows from the middle, as the oracle and rad-convert ask for them */
        std::vector<const void*> pd;
        for (auto& p : q.planes) pd.push_back(p.data());
        EncodedView v;
        enc_view(q.enc, R, C, pd.data(), &v);
        std::vector<float> part((size_t)(2 * C));
        REQUIRE_EQ(enc_decode_rows(v, 3, 2, part.data()), RAD_OK);
        for (int64_t i = 0; i < 2 * C; ++i) CHECK_EQ(part[(size_t)i], d[(size_t)(3 * C + i)]);
    }
    /* the registry closed: its decoder went with it */
    std::vector<const void*> pd;
    for (auto& p : q.planes) pd.push_back(p.data());
    EncodedView v;
    enc_view(q.enc, R, C, pd.data(), &v);
    std::vector<float> out((size_t)(R * C));
    std::string why;
    CHECK_EQ(enc_decode_rows(v, 0, R, out.data(), &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("radtest-halves") != std::string::npos);
}
