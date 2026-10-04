/* tokenizer_perf_test.cpp -- the tokeniser's speed, held to budgets.
 *
 * Tokenisation runs on the request path: an agent re-sends its whole conversation every turn and
 * every byte of it is encoded before the first token can be generated. So speed is a property
 * with a test:
 *
 *   - throughput on prose-and-code text, against an absolute floor and against the reference
 *     encoder (tools/tokref) on the same text -- the ratio holds on a loaded machine where an
 *     absolute number would not;
 *   - LINEAR SCALING: twice the text takes about twice the time, on ordinary text and on the
 *     inputs a pre-tokeniser is quadratic on when it goes wrong (one enormous word, one
 *     enormous run of whitespace, digits, alternating classes);
 *   - the segment cache: the last turn of a long conversation costs a fraction of encoding it.
 *
 * Budgets are set well inside what this build does, so a failure is a regression and not noise.
 * Every time is the best of several runs.
 */
#include "rad_test.h"
#include "tokenizer_fixture.h"
#include "tokref.h"

#include "rad_internal.h"

#include <chrono>
#include <cstdio>
#include <functional>

using namespace rad;
using namespace tokfix;

namespace {

double best_of(int n, const std::function<void()>& f) {
    double best = 1e30;
    for (int i = 0; i < n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

/* Prose and code in the proportions an agent's conversation has them. */
std::string document(uint32_t seed, size_t bytes) {
    static const char* words[] = {
        "the", "cache", "is", "sharded", "by", "key", "and", "each", "shard", "has", "its", "own",
        "lock", "so", "that", "readers", "never", "wait", "on", "writers", "of", "another", "we",
        "measured", "it", "at", "requests", "per", "second", "with", "p99", "latency", "under",
        "load", "tokenizer", "encode", "returns", "ids", "for", "text", "value", "struct", "func",
        "return", "if", "else", "nil", "err", "string", "int", "len", "make", "map", "append",
        "更新", "缓存", "naïve", "café", "über", "Straße", "ünïcode", "don't", "it's", "we'll",
    };
    static const char* punct[] = { ".", ",", ";", ":", "(", ")", "{", "}", "[", "]", " =", " :=",
                                   " ==", " !=", " //", " ->", "\"", "'", "?", "!" };
    std::mt19937 rng(seed);
    std::string s;
    while (s.size() < bytes) {
        const int n = 3 + rng() % 14;
        for (int k = 0; k < n; ++k) {
            if (k) s += ' ';
            s += words[rng() % (sizeof(words) / sizeof(words[0]))];
            if (rng() % 5 == 0) s += punct[rng() % (sizeof(punct) / sizeof(punct[0]))];
            if (rng() % 17 == 0) s += std::to_string(rng() % 100000);
        }
        s += rng() % 3 ? ".\n" : (rng() % 2 ? "\n\t" : "\n\n    ");
    }
    return s;
}

std::shared_ptr<const Vocab> vocab() {
    static std::shared_ptr<const Vocab> v = [] {
        VocabBuild vb = train_bytelevel(document(1, 300000), k_qwen35, 1500);
        vb.steps = { simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() };
        add_specials(vb, { { "<|im_start|>", RAD_TT_CONTROL }, { "<|im_end|>", RAD_TT_CONTROL } });
        return load(std::move(vb));
    }();
    return v;
}

double encode_seconds(const Tokenizer& t, const std::string& text, int runs = 5) {
    std::vector<int32_t> ids;
    return best_of(runs, [&] { ids.clear(); t.encode(text, ids, false, true); });
}

}  /* namespace */

TEST(throughput_has_a_floor_and_beats_the_reference) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const std::string text = document(2, 2 << 20);
    const double s = encode_seconds(t, text);
    const double mbs = text.size() / s / 1e6;
    std::vector<int32_t> ids;
    const double r = best_of(2, [&] { ids.clear(); tokref::encode(*v, text, ids, false, true); });
    std::fprintf(stderr, "    %.1f MB/s (reference %.1f MB/s, %.1fx)\n", mbs, text.size() / r / 1e6, r / s);
    CHECK(mbs > 25.0);
    CHECK(r / s > 4.0);
}

/* TWICE THE TEXT, ABOUT TWICE THE TIME. The texts differ, so a cache that remembered one would
 * not help the other; the ratio is allowed well past 2 because the larger one also walks more of
 * memory. */
TEST(encoding_time_is_linear_in_the_text) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const std::string one = document(3, 1 << 20), two = document(4, 2 << 20);
    const double a = encode_seconds(t, one), b = encode_seconds(t, two);
    std::fprintf(stderr, "    1 MB %.2f ms, 2 MB %.2f ms, ratio %.2f\n", a * 1e3, b * 1e3, b / a);
    CHECK(b / a > 1.3);
    CHECK(b / a < 2.8);
}

/* THE INPUTS A SPLITTER GOES QUADRATIC ON. A backtracking `\s+(?!\S)` rescans a whitespace run
 * from every position in it, and a merge that is quadratic in the word shows on one enormous
 * word; both must stay linear. */
TEST(pathological_inputs_stay_linear) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const struct { const char* name; std::function<std::string(size_t)> make; } inputs[] = {
        { "one word", [](size_t n) { return std::string(n, 'a'); } },
        { "whitespace", [](size_t n) { return std::string(n, ' ') + "x"; } },
        { "mixed whitespace", [](size_t n) { std::string s; while (s.size() < n) s += " \t \n"; return s + "x"; } },
        { "digits", [](size_t n) { std::string s; while (s.size() < n) s += "0123456789"; return s; } },
        { "alternating", [](size_t n) { std::string s; while (s.size() < n) s += "a1!"; return s; } },
        { "apostrophes", [](size_t n) { std::string s; while (s.size() < n) s += "''s"; return s; } },
    };
    /* A megabyte and two, best of five: at a few hundred kilobytes one of these runs in a couple of
     * milliseconds, which one preemption under a parallel ctest turns into a false quadratic. */
    for (const auto& in : inputs) {
        const std::string one = in.make(1000000), two = in.make(2000000);
        const double a = encode_seconds(t, one, 5), b = encode_seconds(t, two, 5);
        std::fprintf(stderr, "    %-17s 1 MB %7.2f ms, 2 MB %7.2f ms, ratio %.2f\n", in.name,
                     a * 1e3, b * 1e3, b / a);
        CHECK(b / a < 3.0);
    }
}

/* THE LAST TURN OF A LONG CONVERSATION costs what is new in it. */
TEST(a_resent_conversation_costs_its_new_turn) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer plain, cached;
    plain.init(v);
    plain.set_segment_cache(false);
    cached.init(v);
    cached.clear_cache();
    std::string conv;
    std::vector<std::string> prompts;
    for (int k = 0; k < 40; ++k) {
        conv += std::string("<|im_start|>") + (k % 2 ? "assistant" : "user") + "\n" +
                document(100 + k, 15000) + "<|im_end|>\n";
        prompts.push_back(conv + "<|im_start|>assistant\n");
    }
    std::vector<int32_t> ids;
    for (size_t k = 0; k + 1 < prompts.size(); ++k) { ids.clear(); cached.encode(prompts[k], ids, false, true); }
    const std::string& last = prompts.back();
    const double warm = best_of(1, [&] { ids.clear(); cached.encode(last, ids, false, true); });
    const double cold = encode_seconds(plain, last, 3);
    std::fprintf(stderr, "    %zu bytes: uncached %.2f ms, cached %.2f ms (%.0fx)\n", last.size(),
                 cold * 1e3, warm * 1e3, cold / warm);
    CHECK(warm * 5 < cold);
}

/* Compiling a shipped pattern is a load-time cost, paid once; it stays small. */
TEST(compiling_a_shipped_pattern_is_cheap) {
    for (const char* p : { k_qwen35, k_minicpm, k_llama3, k_gpt2 }) {
        std::unique_ptr<TokRegex> re;
        std::string why;
        const double s = best_of(3, [&] { re.reset(); TokRegex::compile(p, &re, &why); });
        CHECK(re != nullptr);
        CHECK(s < 0.05);
    }
}

RAD_TEST_MAIN()
