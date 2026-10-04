/* gen_config_test.cpp -- the sampling a checkpoint states for itself.
 *
 * WHAT THIS IS HOLDING DOWN. SamplingParams' defaults are an untruncated sampler, and a model
 * tuned against a truncated tail served that way draws, now and then, a token from far down its
 * distribution. End-of-turn is one of those tokens, so the symptom is not prose that reads badly
 * -- it is a reply that ends early, which from outside the server is indistinguishable from a
 * dropped request. Nothing about a wrong sampler default announces itself, which is exactly why
 * the reading of it is a component with a test rather than a helper inside the engine.
 *
 * No model and no container: the three sources are a string, a file and a key lookup, and all
 * three are reachable directly.
 */
#include "rad_test.h"

#include "gen_config.h"

#include <cstdio>
#include <map>
#include <string>

using namespace rad;

namespace {

/* The container's metadata table as gen_sampling_from_kv sees it: bare field names, and every
 * value a string, which is how rad-convert writes free-form metadata. */
std::function<const char*(const char*)> kv_of(const std::map<std::string, std::string>& m) {
    return [&m](const char* k) -> const char* {
        auto it = m.find(k);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

std::string write_temp(const char* name, const std::string& body) {
    std::string path = std::string("/tmp/radiance_gen_config_test_") + name + ".json";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return std::string();
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
    return path;
}

/* Qwen3.8-Flash-Next's, verbatim -- the shape this was built for. */
const char* const kQwenGenConfig = R"({
    "bos_token_id": 248044,
    "do_sample": true,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95
})";

}  /* namespace */

TEST(gen_config_reads_a_real_generation_config) {
    GenSampling g;
    CHECK(gen_sampling_parse(kQwenGenConfig, "test", &g));
    CHECK_NEAR(g.temp, 1.0f, 1e-6);
    CHECK_EQ(g.top_k, 20);
    CHECK_NEAR(g.top_p, 0.95f, 1e-6);
    CHECK_NEAR(g.min_p, -1.0f, 1e-6);      /* the file does not say, so nothing is claimed */

    /* Overlaid on the built-in defaults it truncates the tail, which is the whole point: the
     * untruncated sampler is what lets an early end-of-turn be drawn at all. */
    const SamplingParams sp = g.onto(SamplingParams{});
    CHECK_NEAR(sp.temp, 1.0f, 1e-6);
    CHECK_EQ(sp.top_k, 20);
    CHECK_NEAR(sp.top_p, 0.95f, 1e-6);
    CHECK_NEAR(sp.min_p, 0.0f, 1e-6);      /* untouched, so still the struct's own */
    CHECK(!sp.greedy());
}

/* A FIELD A SOURCE DID NOT STATE MUST NOT BE CLAIMED, or the sources cannot stack: a sidecar that
 * sets only top_k would otherwise wipe the temperature underneath it. */
TEST(gen_config_claims_only_what_a_source_states) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"top_k": 40})", "test", &g));
    CHECK_EQ(g.top_k, 40);
    CHECK(g.temp < 0.0f);
    CHECK(g.top_p < 0.0f);

    SamplingParams base;
    base.temp = 0.6f;
    base.top_p = 0.8f;
    const SamplingParams sp = g.onto(base);
    CHECK_NEAR(sp.temp, 0.6f, 1e-6);
    CHECK_NEAR(sp.top_p, 0.8f, 1e-6);
    CHECK_EQ(sp.top_k, 40);

    GenSampling none;
    CHECK(!gen_sampling_parse(R"({"bos_token_id": 1})", "test", &none));
    CHECK(!none.any());
}

/* `do_sample: false` is greedy and transformers ignores every other sampler field under it. */
TEST(gen_config_reads_do_sample_false_as_greedy) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"do_sample": false, "temperature": 0.7, "top_k": 20})",
                             "test", &g));
    CHECK_NEAR(g.temp, 0.0f, 1e-6);
    CHECK_EQ(g.top_k, -1);                 /* not taken: greedy does not have one */
    CHECK(g.onto(SamplingParams{}).greedy());

    /* do_sample true is the ordinary case and says nothing by itself. */
    GenSampling t;
    CHECK(gen_sampling_parse(R"({"do_sample": true, "temperature": 0.7})", "test", &t));
    CHECK_NEAR(t.temp, 0.7f, 1e-6);
}

/* transformers spells "no top_k" as -1 and this sampler spells it 0. */
TEST(gen_config_translates_top_k_minus_one_to_off) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"top_k": -1})", "test", &g));
    CHECK_EQ(g.top_k, 0);
}

/* A VALUE OUT OF RANGE IS DROPPED, NOT REFUSED, and it drops alone: these arrive with the model,
 * and one implausible entry is not a reason to serve the rest wrongly or not at all. */
TEST(gen_config_drops_an_out_of_range_field_and_keeps_the_rest) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"temperature": 9.0, "top_p": 0.95, "top_k": 20})", "test", &g));
    CHECK(g.temp < 0.0f);                  /* 9.0 is not a temperature; nothing was claimed */
    CHECK_NEAR(g.top_p, 0.95f, 1e-6);
    CHECK_EQ(g.top_k, 20);

    /* top_p is a half-open range: 0 is not a nucleus, 1 is. */
    GenSampling z;
    CHECK(!gen_sampling_parse(R"({"top_p": 0.0})", "test", &z));
    GenSampling one;
    CHECK(gen_sampling_parse(R"({"top_p": 1.0})", "test", &one));
    CHECK_NEAR(one.top_p, 1.0f, 1e-6);
}

TEST(gen_config_survives_text_that_is_not_a_config) {
    GenSampling g;
    CHECK(!gen_sampling_parse("{not json", "test", &g));
    CHECK(!gen_sampling_parse("[1, 2, 3]", "test", &g));
    CHECK(!gen_sampling_parse("", "test", &g));
    CHECK(!g.any());
}

/* The container's copy and the original file must resolve to the same sampler: rad-convert
 * flattens every value to a string, and two readings of one schema is how the two come to
 * disagree about a checkpoint. */
TEST(gen_config_reads_the_container_table_as_the_file) {
    GenSampling from_file;
    CHECK(gen_sampling_parse(kQwenGenConfig, "test", &from_file));

    const std::map<std::string, std::string> table = {
        { "do_sample",   "1" },
        { "temperature", "1" },
        { "top_k",       "20" },
        { "top_p",       "0.95" },
    };
    GenSampling from_kv;
    CHECK(gen_sampling_from_kv(kv_of(table), &from_kv));
    CHECK_NEAR(from_kv.temp, from_file.temp, 1e-6);
    CHECK_EQ(from_kv.top_k, from_file.top_k);
    CHECK_NEAR(from_kv.top_p, from_file.top_p, 1e-6);

    /* An empty table states nothing rather than stating zeros. */
    GenSampling empty;
    CHECK(!gen_sampling_from_kv(kv_of({}), &empty));
    CHECK(!empty.any());

    /* A value that is not a number is dropped, and the rest of the table still reads. */
    GenSampling junk;
    CHECK(gen_sampling_from_kv(kv_of({ { "temperature", "warm" }, { "top_k", "20" } }), &junk));
    CHECK(junk.temp < 0.0f);
    CHECK_EQ(junk.top_k, 20);
}

TEST(gen_config_reads_a_file_and_treats_absence_as_silence) {
    const std::string path = write_temp("sidecar", kQwenGenConfig);
    CHECK(!path.empty());
    GenSampling g;
    CHECK(gen_sampling_from_file(path, /*required=*/false, &g));
    CHECK_EQ(g.top_k, 20);
    std::remove(path.c_str());

    /* A checkpoint that ships no generation config is served from whatever lies underneath, not
     * refused -- so a missing file takes nothing and reports nothing taken. */
    GenSampling missing;
    CHECK(!gen_sampling_from_file("/nonexistent/generation_config.json", false, &missing));
    CHECK(!missing.any());
}

/* The sidecar is the container's path with its extension replaced, and the extension is the LAST
 * dot in the basename -- a directory with a dot in its name must not eat it. */
TEST(gen_config_derives_the_sidecar_path) {
    CHECK_EQ(gen_sidecar_path("/m/q38-flashnext-unc-w4-mtp.rad"),
             std::string("/m/q38-flashnext-unc-w4-mtp.generation.json"));
    CHECK_EQ(gen_sidecar_path("model.rad"), std::string("model.generation.json"));
    CHECK_EQ(gen_sidecar_path("/a.b/model"), std::string("/a.b/model.generation.json"));
    CHECK_EQ(gen_sidecar_path("/a.b/model.rad"), std::string("/a.b/model.generation.json"));
}

RAD_TEST_MAIN()
