/* share_test.cpp -- which bytes of a container entry one rank takes (core/format/share.h).
 *
 * The loader reads exactly the rectangles weight_share names, so a wrong one is a rank serving
 * another rank's rows -- a model that loads, runs and is wrong in the way spec §17 exists to
 * prevent. Each case is a shape tensor parallelism actually produces, worked by hand.
 */
#include "rad_test.h"
#include "format/share.h"

#include <cstring>
#include <string>

using namespace rad;

namespace {

/* A declaration of `planes` of `e` over a logical [R, C] weight at rank `world`, as the builder
 * records it: one plane at its own extents, several at the logical ones, this rank's share of
 * them under a shard. */
WeightInfo decl(const RadEncoding& e, std::initializer_list<int> planes, int64_t R, int64_t C,
                int shard, int world, std::initializer_list<int64_t> parts = {}) {
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = shard;
    w.decl.rank = 2;
    int64_t r = R, c = C;
    if (shard == RAD_SHARD_ROW && world > 1) r = R / world;
    if (shard == RAD_SHARD_COL && world > 1) c = C / world;
    w.decl.shape[0] = r;
    w.decl.shape[1] = c;
    int n = 0;
    for (int64_t p : parts) w.decl.row_parts[n++] = p;
    w.decl.n_row_parts = n;
    for (int k : planes) {
        w.sel[w.n_sel] = k;
        rad_enc_plane_dims(&e.plane[k], r, c, &w.sel_rows[w.n_sel], &w.sel_cols[w.n_sel]);
        ++w.n_sel;
    }
    return w;
}

RadEncoding fp8_block() {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_BF16, 128, 128);
    return e;
}

int share(const WeightInfo& w, int64_t R, int64_t C, int world, int index, PlaneShare* out,
          std::string* why = nullptr) {
    const int64_t shape[2] = { R, C };
    return weight_share(w, w.enc, 2, shape, world, index, out, why);
}

}  /* namespace */

/* Unsharded, a rank takes every plane whole: one rectangle, every byte of every row. */
TEST(an_unsharded_weight_is_every_plane_whole) {
    const RadEncoding e = fp8_block();
    const WeightInfo w = decl(e, { 0, 1 }, 256, 384, RAD_SHARD_NONE, 2);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 256, 384, 2, 1, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)0);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)256);
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)384);
    CHECK_EQ(s[0].dense_bytes(), (int64_t)(256 * 384));
    CHECK_EQ(s[1].rows, (int64_t)2);
    CHECK_EQ(s[1].cols, (int64_t)3);
    CHECK_EQ(s[1].dense_bytes(), (int64_t)(2 * 3 * 2));
}

/* A ROW shard of a block-fp8 linear is rows of every plane, the scales a block of 128 rows each. */
TEST(a_row_shard_takes_rows_of_the_codes_and_their_scale_blocks) {
    const RadEncoding e = fp8_block();
    for (int index = 0; index < 2; ++index) {
        const WeightInfo w = decl(e, { 0, 1 }, 512, 256, RAD_SHARD_ROW, 2);
        PlaneShare s[RAD_ENC_MAX_PLANES];
        REQUIRE_EQ(share(w, 512, 256, 2, index, s), RAD_OK);
        CHECK_EQ(s[0].rects[0].row0, (int64_t)(index * 256));
        CHECK_EQ(s[0].rects[0].rows, (int64_t)256);
        CHECK_EQ(s[1].rects[0].row0, (int64_t)(index * 2));
        CHECK_EQ(s[1].rects[0].rows, (int64_t)2);
        CHECK_EQ(s[1].rects[0].bytes, (int64_t)(2 * 2));
    }
    /* the scale plane declared on its own, as gemm_nt_q's b_scale operand: its own extents */
    const WeightInfo ws = decl(e, { 1 }, 512, 256, RAD_SHARD_ROW, 2);
    CHECK_EQ(ws.sel_rows[0], (int64_t)2);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(ws, 512, 256, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)2);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)2);
}

/* A STACKED [gate|up] IS CUT IN EACH PART: rank 1 of 2 takes the second half of gate AND the second
 * half of up, never "all of up". */
TEST(a_stacked_weight_is_cut_in_every_part) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    /* entry [2 * 512, 64]: gate rows 0..511, up rows 512..1023; a rank's parts are 256 rows each */
    const WeightInfo w = decl(e, { 0 }, 1024, 64, RAD_SHARD_ROW, 2, { 256, 256 });
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 1024, 64, 2, 1, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)2);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)256);
    CHECK_EQ(s[0].rects[1].row0, (int64_t)768);
    CHECK_EQ(s[0].rows, (int64_t)512);
    /* parts that do not sum to the rank's share are refused */
    const WeightInfo bad = decl(e, { 0 }, 1024, 64, RAD_SHARD_ROW, 2, { 200, 256 });
    std::string why;
    CHECK_EQ(share(bad, 1024, 64, 2, 0, s, &why), RAD_E_SHAPE);
    CHECK(why.find("row parts") != std::string::npos);
}

/* A COL shard is columns of each row, and a column cut has to land on a whole byte. */
TEST(a_col_shard_takes_a_byte_range_of_every_row) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    const WeightInfo w = decl(e, { 0 }, 8, 128, RAD_SHARD_COL, 4);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 8, 128, 4, 3, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)8);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(96 * 2));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(32 * 2));
    CHECK_EQ(s[0].row_bytes, (int64_t)(128 * 2));

    RadEncoding i4;
    rad_enc_clear(&i4);
    rad_enc_copy_str(i4.scheme, "plain");
    rad_enc_add_plane(&i4, "codes", RAD_I4, 1, 1);
    const WeightInfo odd = decl(i4, { 0 }, 4, 6, RAD_SHARD_COL, 2);   /* 3 nibbles a rank */
    std::string why;
    CHECK_EQ(share(odd, 4, 6, 2, 1, s, &why), RAD_E_SHAPE);
    CHECK(why.find("whole bytes") != std::string::npos);
}

/* A cut that splits a scale block belongs to neither rank, and is refused. */
TEST(a_shard_that_splits_a_block_is_refused) {
    const RadEncoding e = fp8_block();
    const WeightInfo w = decl(e, { 0, 1 }, 384, 256, RAD_SHARD_ROW, 2);   /* 192 rows a rank */
    PlaneShare s[RAD_ENC_MAX_PLANES];
    std::string why;
    CHECK_EQ(share(w, 384, 256, 2, 0, s, &why), RAD_E_SHAPE);
}

/* AN UNEVEN SPAN: the plugin names this rank's rows of each part, and the scale plane takes the
 * same slice in its own blocks. */
TEST(an_uneven_span_is_the_rows_the_plugin_names) {
    const RadEncoding e = fp8_block();
    WeightInfo w = decl(e, { 0, 1 }, 640, 128, RAD_SHARD_ROW, 2);
    w.decl.shape[0] = 384;
    w.shard_lo = 256;
    w.shard_hi = 640;
    rad_enc_plane_dims(&e.plane[0], 384, 128, &w.sel_rows[0], &w.sel_cols[0]);
    rad_enc_plane_dims(&e.plane[1], 384, 128, &w.sel_rows[1], &w.sel_cols[1]);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 640, 128, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)256);
    CHECK_EQ(s[0].rows, (int64_t)384);
    CHECK_EQ(s[1].rects[0].row0, (int64_t)2);
    CHECK_EQ(s[1].rows, (int64_t)3);
}

/* A per-tensor scale is one value every rank carries whole, once, whatever the codes' cut. */
TEST(a_per_tensor_scale_is_carried_whole_by_every_rank) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_BF16, 0, 0);
    const WeightInfo w = decl(e, { 0, 1 }, 1000, 64, RAD_SHARD_ROW, 2, { 250, 250 });
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 1000, 64, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects.size(), (size_t)2);
    REQUIRE_EQ(s[1].rects.size(), (size_t)1);
    CHECK_EQ(s[1].rows, (int64_t)1);
    CHECK_EQ(s[1].cols, (int64_t)1);
}

/* A rank-1 ROW shard is a contiguous slice of the vector: one value per head. */
TEST(a_sharded_vector_is_a_slice_of_it) {
    const RadEncoding e = rad_enc_plain(RAD_F32);
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = RAD_SHARD_ROW;
    w.decl.rank = 1;
    w.decl.shape[0] = 16;
    w.n_sel = 1;
    w.sel[0] = 0;
    w.sel_rows[0] = 1;
    w.sel_cols[0] = 16;
    const int64_t shape[1] = { 64 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 1, shape, 4, 2, s, nullptr), RAD_OK);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(32 * 4));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(16 * 4));
}

/* A RESHAPE: a conv kernel [O, C, T, H, W] declared as the [O, C*T*H*W] matrix it is. The bytes
 * do not move, so the share is the whole plane in the declared view -- and a shard of a reshape
 * is a different cut, so it is refused. */
TEST(a_whole_plane_may_be_declared_under_another_shape) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w = decl(e, { 0 }, 1152, 1536, RAD_SHARD_NONE, 2);
    const int64_t conv[5] = { 1152, 3, 2, 16, 16 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 5, conv, 2, 1, s, nullptr), RAD_OK);
    CHECK_EQ(s[0].rows, (int64_t)1152);
    CHECK_EQ(s[0].cols, (int64_t)1536);
    CHECK_EQ(s[0].dense_bytes(), (int64_t)(1152 * 1536 * 2));

    WeightInfo r = decl(e, { 0 }, 1152, 1536, RAD_SHARD_ROW, 2);
    std::string why;
    CHECK_EQ(weight_share(r, e, 5, conv, 2, 0, s, &why), RAD_E_SHAPE);
}

/* A depthwise conv kernel [C, 1, K] is the [C, K] matrix its inner 1 leaves, so a row shard of
 * it is rows of that matrix -- contiguous, K elements each. An inner dimension that is not 1 is
 * still a rank-3 weight, and which of its dimensions a rank owns is not the loader's to guess. */
TEST(a_unit_inner_dimension_is_sharded_as_the_matrix_it_leaves) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w = decl(e, { 0 }, 10240, 4, RAD_SHARD_ROW, 2);
    const int64_t conv[3] = { 10240, 1, 4 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 3, conv, 2, 1, s, nullptr), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)5120);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)5120);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)0);
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(4 * 2));

    const int64_t stack[3] = { 10240, 2, 2 };
    std::string why;
    CHECK_EQ(weight_share(w, e, 3, stack, 2, 0, s, &why), RAD_E_SHAPE);
}

RAD_TEST_MAIN()
