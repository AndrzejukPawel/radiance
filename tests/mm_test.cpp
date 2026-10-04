/* mm_test.cpp -- the multimodal processor against the reference, number for number.
 *
 * Every expected value in mm_golden.h came out of transformers' own code (tests/mkgolden.py says
 * which function made which table): smart_resize in both forms, the frames a video keeps, the
 * patches of a picture and of a clip, the timestamps between a clip's groups, and get_rope_index's
 * positions over a prompt carrying both. A model reads its prompt through all of these at once, so
 * a mistake in any one is a model that still answers -- about a slightly different picture, or
 * with its positions a row off -- and nothing downstream of this file can tell.
 *
 * The pixel inputs are a splitmix hash of their coordinates rather than a ramp: a patch taken with
 * the wrong stride from a ramp is often another plausible ramp, and from this it is not.
 */
#include "rad_test.h"
#include "rad_types.h"
#include "mm/media.h"
#include "mm/processor.h"
#include "mm_golden.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::mm;

namespace {

uint8_t pix(uint64_t f, uint64_t y, uint64_t x, uint64_t c) {
    uint64_t z = f * 0x9E3779B97F4A7C15ull + y * 0xBF58476D1CE4E5B9ull + x * 0x94D049BB133111EBull +
                 c * 0x2545F4914F6CDD1Dull + 12345ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint8_t)((z ^ (z >> 31)) & 255);
}

Frame frame(int f, int h, int w) {
    Frame fr;
    fr.w = w;
    fr.h = h;
    fr.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                fr.rgb[((size_t)y * w + x) * 3 + c] = pix((uint64_t)f, (uint64_t)y, (uint64_t)x, (uint64_t)c);
    return fr;
}

uint64_t fnv(const std::vector<uint16_t>& v) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint16_t u : v)
        for (int b = 0; b < 2; ++b) h = (h ^ (uint8_t)(u >> (8 * b))) * 0x100000001b3ull;
    return h;
}

/* The Qwen3.8 processor configuration, as a container converted from the checkpoint states it. */
VisionConfig qwen_cfg() {
    VisionConfig c;
    c.patch = 16;
    c.temporal = 2;
    c.merge = 2;
    c.image_min_pixels = 65536;
    c.image_max_pixels = 16777216;
    c.video_min_pixels = 4096;
    c.video_max_pixels = 25165824;
    c.tok_image = 248056;
    c.tok_video = 248057;
    c.tok_start = 248053;
    c.tok_end = 248054;
    c.max_patches = 16384;
    return c;
}

}  /* namespace */

TEST(smart_resize_is_the_references_for_pictures) {
    for (const GoldImageResize& g : kGoldImageResize) {
        int64_t oh = 0, ow = 0;
        std::string why;
        CHECK_OK(smart_resize_image(g.h, g.w, 32, g.min_px, g.max_px, &oh, &ow, &why));
        if (oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld x %lld in [%lld, %lld]: %lld x %lld, reference %lld x %lld\n",
                         (long long)g.h, (long long)g.w, (long long)g.min_px, (long long)g.max_px,
                         (long long)oh, (long long)ow, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

TEST(smart_resize_is_the_references_for_video) {
    for (const GoldVideoResize& g : kGoldVideoResize) {
        int64_t oh = 0, ow = 0;
        std::string why;
        CHECK_OK(smart_resize_video(g.t, g.h, g.w, 2, 32, g.min_px, g.max_px, &oh, &ow, &why));
        if (oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld frames of %lld x %lld: %lld x %lld, reference %lld x %lld\n",
                         (long long)g.t, (long long)g.h, (long long)g.w, (long long)oh,
                         (long long)ow, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

TEST(the_frames_a_video_keeps_are_the_references) {
    for (const GoldSample& g : kGoldSample) {
        VideoInfo info;
        info.n_frames = g.total;
        info.fps = g.fps;
        const std::vector<int64_t> idx = sample_frames(info, 2.0, 4, 768);
        REQUIRE_EQ((int32_t)idx.size(), g.n);
        int off = 0;
        for (int32_t i = 0; i < g.n; ++i) off += idx[(size_t)i] != kGoldSampleIdx[g.first + i];
        if (off)
            std::fprintf(stderr, "    %lld frames at %g fps: %d of %d indices differ\n",
                         (long long)g.total, g.fps, off, g.n);
        CHECK_EQ(off, 0);
    }
}

TEST(a_picture_patchifies_as_the_reference_does) {
    Processor p;
    REQUIRE(p.init(qwen_cfg(), [](const std::string&) { return std::vector<int32_t>{ 1 }; }) >= 0);
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.image_from(frame(0, 256, 320), &it, &why));
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_t, kGoldImageGrid[0]);
    CHECK_EQ(it->grid_h, kGoldImageGrid[1]);
    CHECK_EQ(it->grid_w, kGoldImageGrid[2]);
    REQUIRE_EQ(it->pixels.size(), (size_t)it->n_patch * 1536);
    for (int i = 0; i < 16; ++i) CHECK_EQ(it->pixels[(size_t)i], kGoldImageHead[i]);
    CHECK_EQ(fnv(it->pixels), kGoldImageFnv);

    /* <|vision_start|>, a pad a merged row, <|vision_end|>: the template's triple with the pad run
     * at its real length, which is what transformers substitutes. */
    const int32_t rows = (kGoldImageGrid[1] / 2) * (kGoldImageGrid[2] / 2);
    REQUIRE_EQ(it->expand.size(), (size_t)rows + 2);
    CHECK_EQ(it->expand.front(), 248053);
    CHECK_EQ(it->expand.back(), 248054);
    for (int32_t i = 1; i <= rows; ++i) CHECK_EQ(it->expand[(size_t)i], 248056);
    REQUIRE_EQ(it->seg_at.size(), (size_t)1);
    CHECK_EQ(it->seg_at[0], 1);
    CHECK_EQ(it->n_rows(), (int64_t)rows);
}

TEST(a_clip_patchifies_and_is_stamped_as_the_reference_does) {
    std::vector<std::string> stamps;
    Processor p;
    REQUIRE(p.init(qwen_cfg(), [&](const std::string& s) {
                stamps.push_back(s);
                return std::vector<int32_t>(s.size(), 7);   /* one token a byte: the length shows */
            }) >= 0);
    Video v;
    v.info.n_frames = 6;
    v.info.fps = 2.0;
    v.info.w = 96;
    v.info.h = 64;
    v.index = sample_frames(v.info, 2.0, 4, 768);
    for (int64_t i : v.index) v.frames.push_back(frame((int)i, 64, 96));
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.video_from(v, &it, &why));
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_t, kGoldVideoGrid[0]);
    CHECK_EQ(it->grid_h, kGoldVideoGrid[1]);
    CHECK_EQ(it->grid_w, kGoldVideoGrid[2]);
    REQUIRE_EQ(it->pixels.size(), (size_t)it->n_patch * 1536);
    for (int i = 0; i < 16; ++i) CHECK_EQ(it->pixels[(size_t)i], kGoldVideoHead[i]);
    CHECK_EQ(fnv(it->pixels), kGoldVideoFnv);

    /* One stamp a group, in the reference's words and rounding: 0.25 s is "<0.2 seconds>". */
    const size_t n_stamps = sizeof(kGoldVideoStamps) / sizeof(kGoldVideoStamps[0]);
    REQUIRE_EQ(stamps.size(), n_stamps);
    for (size_t i = 0; i < n_stamps; ++i) CHECK_EQ(stamps[i], std::string(kGoldVideoStamps[i]));

    /* <stamp> <|vision_start|> pads <|vision_end|>, a group at a time, and nothing around them. */
    const int32_t rows = (kGoldVideoGrid[1] / 2) * (kGoldVideoGrid[2] / 2);
    size_t at = 0;
    REQUIRE_EQ(it->seg_at.size(), (size_t)kGoldVideoGrid[0]);
    for (int32_t g = 0; g < kGoldVideoGrid[0]; ++g) {
        at += stamps[(size_t)g].size();
        REQUIRE(at < it->expand.size());
        CHECK_EQ(it->expand[at], 248053);
        CHECK_EQ(it->seg_at[(size_t)g], (int32_t)(at + 1));
        for (int32_t r = 0; r < rows; ++r) CHECK_EQ(it->expand[at + 1 + (size_t)r], 248057);
        CHECK_EQ(it->expand[at + 1 + (size_t)rows], 248054);
        at += (size_t)rows + 2;
    }
    CHECK_EQ(at, it->expand.size());
}

TEST(the_rotary_layout_is_get_rope_index) {
    /* The prompt mkgolden.py built: an image of 2 x 3 merged rows in its vision pair, then a clip of
     * two groups of 2 x 4 rows, each group behind five tokens of timestamp. Only the placement and
     * the grids reach the layout, so the items carry nothing else. */
    auto img = std::make_shared<Item>();
    img->kind = Kind::Image;
    img->grid_t = 1; img->grid_h = 4; img->grid_w = 6;
    img->rows_h = 2; img->rows_w = 3; img->seg_rows = 6; img->n_seg = 1;
    img->seg_at = { 1 };
    auto vid = std::make_shared<Item>();
    vid->kind = Kind::Video;
    vid->grid_t = 2; vid->grid_h = 4; vid->grid_w = 8;
    vid->rows_h = 2; vid->rows_w = 4; vid->seg_rows = 8; vid->n_seg = 2;
    vid->seg_at = { kGoldRopeVideoSeg[0], kGoldRopeVideoSeg[1] };

    PromptMedia pm;
    Placed a; a.item = img; a.tok = kGoldRopeImageAt;
    Placed b; b.item = vid; b.tok = kGoldRopeVideoAt;
    pm.items = { a, b };
    pm.layout();
    REQUIRE_EQ(pm.rope.runs.size(), (size_t)3);

    int off = 0;
    for (int32_t i = 0; i < kGoldRopeN; ++i) {
        int32_t p[3];
        pm.rope.at(i, p);
        if (p[0] != kGoldRopeT[i] || p[1] != kGoldRopeH[i] || p[2] != kGoldRopeW[i]) {
            if (off < 4)
                std::fprintf(stderr, "    token %d: (%d, %d, %d), reference (%d, %d, %d)\n", i,
                             p[0], p[1], p[2], kGoldRopeT[i], kGoldRopeH[i], kGoldRopeW[i]);
            ++off;
        }
    }
    CHECK_EQ(off, 0);

    /* Generation continues from the reference's delta: position = index + delta, in all three. */
    CHECK_EQ(pm.rope.shift_at(kGoldRopeN), kGoldRopeDelta);
    int32_t p[3];
    pm.rope.at(kGoldRopeN + 5, p);
    CHECK_EQ(p[0], kGoldRopeN + 5 + kGoldRopeDelta);
    CHECK_EQ(p[1], p[0]);
    CHECK_EQ(p[2], p[0]);
}

/* THE DECODER ON EVERY FORMAT A REQUEST MAY CARRY. tests/data/media holds a solid red picture in
 * each still format media.cpp admits and a short solid red clip in each container and codec
 * (its README has the commands that made them); each must decode, at its size, to red. That is
 * also what holds a reduced FFmpeg build -- the container image's -- to the formats the server
 * accepts. A build without libavformat has no decoder, and the cases say so. */
namespace {

std::vector<uint8_t> media_file(const char* name) {
    std::ifstream f(std::string(RAD_MEDIA_FIXTURES) + "/" + name, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

/* Solid red through a YUV codec and back comes out near (254, 0, 0), not exactly. */
bool is_red(const Frame& f) {
    if (f.rgb.size() != (size_t)f.w * f.h * 3 || f.rgb.empty()) return false;
    for (size_t i = 0; i < f.rgb.size(); i += 3)
        if (f.rgb[i] < 200 || f.rgb[i + 1] > 60 || f.rgb[i + 2] > 60) return false;
    return true;
}

std::vector<int64_t> first_and_last(const VideoInfo& info, const void*) {
    if (info.n_frames <= 1) return {0};
    return {0, info.n_frames - 1};
}

}  /* namespace */

TEST(every_admitted_picture_format_decodes) {
    if (!decoder_available()) { std::printf("  SKIP no media decoder in this build\n"); return; }
    for (const char* name : { "red.png", "red.apng", "red.jpg", "red.webp", "red.gif", "red.bmp",
                              "red.tiff", "red.qoi", "red.ppm", "red.jls", "red.avif" }) {
        const std::vector<uint8_t> b = media_file(name);
        CHECK(!b.empty());
        Frame f;
        std::string why;
        const int rc = decode_image(b.data(), b.size(), int64_t(1) << 24, &f, &why);
        if (rc != RAD_OK) std::printf("  %s: %s\n", name, why.c_str());
        CHECK_EQ(rc, RAD_OK);
        const int want = std::string(name) == "red.avif" ? 64 : 32;
        CHECK_EQ(f.w, want);
        CHECK_EQ(f.h, want);
        CHECK(is_red(f));
    }
}

TEST(every_admitted_video_format_decodes) {
    if (!decoder_available()) { std::printf("  SKIP no media decoder in this build\n"); return; }
    for (const char* name : { "red-h264.mp4", "red-hevc.mp4", "red-av1.mp4", "red-vp8.webm",
                              "red-vp9.webm", "red-h264.mkv", "red-h264.ts", "red-mpeg4.avi",
                              "red-theora.ogv" }) {
        const std::vector<uint8_t> b = media_file(name);
        CHECK(!b.empty());
        Video v;
        std::string why;
        const int rc = decode_video(b.data(), b.size(), first_and_last, nullptr, int64_t(1) << 24,
                                    &v, &why);
        if (rc != RAD_OK) std::printf("  %s: %s\n", name, why.c_str());
        CHECK_EQ(rc, RAD_OK);
        CHECK(v.info.n_frames >= 1);
        CHECK(!v.frames.empty());
        CHECK_EQ(v.frames.size(), v.index.size());
        for (const Frame& f : v.frames) {
            CHECK_EQ(f.w, 64);
            CHECK_EQ(f.h, 64);
            CHECK(is_red(f));
        }
    }
}

/* THE BANDS ARE CHECKED AS A WHOLE, because the command line can replace one end of a band and
 * leave the other to the container: a minimum stated alone can land above the container's maximum. */
TEST(a_media_configuration_with_an_empty_band_names_it) {
    mm::VisionConfig c;
    c.image_min_pixels = 65536;
    c.image_max_pixels = 16777216;
    c.video_min_pixels = 4096;
    c.video_max_pixels = 25165824;
    std::string why;
    CHECK_EQ(c.check(&why), RAD_OK);

    mm::VisionConfig b = c;
    b.image_min_pixels = 32 << 20;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("image_max_pixels") != std::string::npos);
    b = c;
    b.video_max_pixels = 1000;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("video_max_pixels") != std::string::npos);
    b = c;
    b.fps = 0.0;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    b = c;
    b.min_frames = 10;
    b.max_frames = 5;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("max_frames") != std::string::npos);
}

RAD_TEST_MAIN()
