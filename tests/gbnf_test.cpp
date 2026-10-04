/* gbnf_test.cpp -- the GBNF mask engine, held to upstream's candidate walk.
 *
 * fill_mask serves every mask from a plan precomputed when the grammar is compiled
 * (core/sample/gbnf_mask.cpp); tests/gbnf_oracle.cpp computes the same mask the way llama.cpp
 * does, by rescanning the vocabulary. They are two implementations of one pure function of the
 * grammar's state, so every case here is the same assertion: at every state a walk reaches, the
 * two agree bit for bit and status for status.
 *
 * WHAT MAKES A STATE WORTH COMPARING is that the engine reached it the way the sampler does. The
 * walks below drive a grammar exactly as Sampler::build_step drives a constrained sequence under
 * speculation -- push the state, advance over a window of drafts, mask every position, roll back,
 * then accept the verified prefix and one more token -- with drafts mostly admissible and sometimes
 * not, so that dead windows, rollbacks over a failed accept and the mask cache are all on the path.
 *
 * WHAT MAKES A VOCABULARY WORTH TESTING AGAINST is that it is shaped like a real one. The plan's
 * keys and remainder tries are properties of piece lengths and byte distributions, and a
 * vocabulary of single letters exercises neither. bpe_like() builds one the way a byte-level BPE
 * looks: every byte as a piece, pieces that end or begin mid-character, multi-byte characters
 * whole and split, words with and without a leading space, long runs, special and
 * end-of-generation tokens, control tokens with no text. It is still not the real thing, and
 * rad-gbnf-check runs this same comparison against a real tokenizer.json.
 *
 * The grammars are hand-written edge cases, JSON-schema conversions, the tool-call grammars of the
 * three chat templates checked in under data/gbnf, and a generator of random grammars for what
 * nobody thought to write. Beside the comparison, the cases pin what the plan promises about its
 * own size and what it refuses; the last holds the engine to latency budgets, at a scale that runs
 * in seconds and a margin that survives a loaded machine, but far below what a vocabulary walk per
 * mask costs.
 */
#include "rad_test.h"

#include "gbnf_oracle.h"
#include "sample/gbnf.h"
#include "sample/gbnf_internal.h"
#include "sample/grammar.h"
#include "sample/vocab_view.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace rad;
using ojson = nlohmann::ordered_json;

namespace {

/* ================================================================== randomness */

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull) {
        if (!s) s = 1;
    }
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    size_t below(size_t n) { return n ? (size_t)(next() % n) : 0; }
    bool   chance(int pct) { return (int)(next() % 100) < pct; }
    template <class T> const T& pick(const std::vector<T>& v) { return v[below(v.size())]; }
};

/* ======================================================== a vocabulary like a real one */

const char* kMultibyte[] = {
    "\xc3\xa9", "\xc3\xbc", "\xc3\x9f", "\xc3\xb1", "\xce\xa9", "\xd1\x8f",    /* é ü ß ñ Ω я */
    "\xe6\x97\xa5", "\xe6\x9c\xac", "\xe4\xb8\xad", "\xe6\x96\x87", "\xe5\xad\x97",  /* 日 本 中 文 字 */
    "\xe2\x80\x94", "\xe2\x80\x9c", "\xe2\x82\xac", "\xe2\x9c\x93",                  /* — “ € ✓ */
    "\xf0\x9f\x98\x80", "\xf0\x9f\x99\x8f", "\xf0\x9f\x9a\x80",                /* three emoji */
};

struct BpeVocab {
    SimpleVocab              v;
    std::vector<int32_t>     structural;   /* pieces carrying structure, to steer walks by */
    std::vector<char>        nul;          /* id -> the piece decodes to U+0000 before its end */
    std::unordered_map<std::string, int32_t> by_piece;
    size_t                   longest = 0;
    double                   mean_bytes = 0.0;

    int32_t add(const std::string& p, bool eog = false) {
        auto it = by_piece.find(p);
        if (it != by_piece.end() && !eog) return it->second;
        const int32_t id = v.add(p, eog);
        by_piece.emplace(p, id);
        longest = std::max(longest, p.size());
        return id;
    }

    /* Longest match, as a greedy tokenizer does: enough to turn a realistic text into a token
     * sequence the grammar will see, which is all a replay needs. */
    std::vector<int32_t> tokenize(const std::string& text) const {
        std::vector<int32_t> out;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t n = std::min(longest, text.size() - pos);
            for (; n > 0; --n) {
                auto it = by_piece.find(text.substr(pos, n));
                if (it != by_piece.end() && !v.is_eog(it->second)) {
                    out.push_back(it->second);
                    break;
                }
            }
            if (n == 0) return {};
            pos += n;
        }
        return out;
    }
};

std::unique_ptr<BpeVocab> bpe_like(uint64_t seed, size_t n_target) {
    auto b = std::make_unique<BpeVocab>();
    Rng rng(seed);

    /* Every byte, as byte-level BPE has: continuation bytes and lead bytes alone are pieces that
     * begin or end mid-character. NUL is a control token. */
    for (int c = 1; c < 256; ++c) b->add(std::string(1, (char)c));

    /* Control tokens with no text, and one that starts with a NUL; special and end-of-generation
     * tokens, whose pieces are rendered. */
    b->v.add("");
    b->v.add("");
    b->v.add(std::string("\0x", 2));
    /* Pieces with a NUL inside, which the decoder reads as ending there: after plain text, after
     * the head of a character, after the bytes that finish one. */
    b->add(std::string("a\0b", 3));
    b->add(std::string("\xe6\0z", 3));
    b->add(std::string("\x97\0y", 3));
    b->add(std::string("\x97\xa5\0q", 4));
    /* Overlong sequences: heads that carry no bits in, and pieces that finish them as U+0000. */
    for (const char* o : { "\xc0\x80", "\xc0\x80z", "\xe0\x80", "\x80\x80", "\x80\x80q",
                           "\xf0\x80\x80", "\x80z" })
        b->add(o);
    for (const char* s : { "<|im_start|>", "<tool_call>", "</tool_call>", "<think>", "</think>" })
        b->add(s);
    b->add("<|im_end|>", true);
    b->add("<|endoftext|>", true);

    /* The pieces structured output is made of. */
    for (const char* s : {
             "{\"", "\":", "\": \"", "\",", "\", \"", "\"}", "\"]", "},", "],", "{}", "[]", "[{",
             "}]", "}}", "\"", "\\\"", "\\\\", "\\n", "\\t", "\\u", "\\u00", "\n", "\n\n", "\n\n\n",
             "\t", "\t\t", " ", "  ", "   ", "    ", "        ", ">\n", "</", "/>", "<", ">",
             "=", "==", "====", "----", "//", "/*", "*/", "();", "),", "):", "=>", "->", "::",
             "&&", "||", "!=",
             "<=", ">=", "...", "e+", "E-", ".5", "0.", "true", "false", "null", " true", " false",
             " null", "<function=", "<parameter=", "</parameter>", "</function>", "parameter",
             "function", "tool", "_call", "\">", "</param>", "<param name=\"", "<function name=\"",
             " {", " }", " [", " ]", "\":\"", "\",\"", ",\"", "\":{", "\":[", ", ", ": " })
        b->add(s);
    for (const char* s : { "ab", "abc", "xy", "the", " the", "dog", "held", "other" }) b->add(s);
    for (int i = 0; i < 10; ++i) b->add(std::to_string(i));
    for (const char* s : { "00", "10", "12", "99", "123", "2024", "-1", "0x", "1.5" }) b->add(s);

    /* Multi-byte characters: whole, after a space, beside ASCII, in pairs, and split at every
     * byte -- the head alone ends mid-character and the tail alone begins mid-character. */
    for (const char* m : kMultibyte) {
        const std::string s(m);
        b->add(s);
        b->add(" " + s);
        b->add("a" + s);
        b->add(s + "s");
        for (size_t k = 1; k < s.size(); ++k) {
            b->add(s.substr(0, k));
            b->add(s.substr(k));
            b->add("x" + s.substr(0, k));
            b->add(s.substr(k) + "y");
        }
        for (const char* m2 : kMultibyte) b->add(s + m2);
    }
    b->add("caf\xc3\xa9");
    b->add("r\xc3\xa9sum\xc3\xa9");
    b->add("na\xc3\xafve");

    /* Long pieces: whitespace and rule runs, as code vocabularies have. */
    for (int n : { 12, 16, 24, 32, 48, 64 }) b->add(std::string((size_t)n, ' '));
    for (int n : { 8, 16, 40 }) {
        b->add(std::string((size_t)n, '='));
        b->add(std::string((size_t)n, '-'));
    }
    b->add(std::string(20, '\n'));

    /* Words, from syllables, with the length spread of real subwords: with and without a
     * leading space, capitalised, with a suffix, with punctuation after. */
    const std::vector<std::string> onset = { "", "b", "c", "d", "f", "g", "h", "k", "l", "m", "n",
                                             "p", "r", "s", "t", "v", "w", "st", "th", "ch", "sh",
                                             "tr", "pr", "gr", "qu" };
    const std::vector<std::string> vowel = { "a", "e", "i", "o", "u", "ou", "ea", "ai", "io", "y" };
    const std::vector<std::string> coda  = { "", "", "n", "r", "s", "t", "l", "nd", "st", "ng",
                                             "ck", "x" };
    const std::vector<std::string> tail  = { "ing", "ed", "s", "tion", "er", "ly", "_id", "()",
                                             "\",", "\":", ",", ".", ":", "\"" };
    while (b->v.n_tokens() < (int32_t)n_target) {
        std::string w;
        const size_t syl = 1 + rng.below(3);
        for (size_t k = 0; k < syl; ++k) w += rng.pick(onset) + rng.pick(vowel) + rng.pick(coda);
        if (rng.chance(15)) w[0] = (char)std::toupper((unsigned char)w[0]);
        if (rng.chance(12)) w += rng.pick(tail);
        if (rng.chance(50)) w = " " + w;
        if (rng.chance(3)) w += rng.pick(std::vector<std::string>(std::begin(kMultibyte),
                                                                  std::end(kMultibyte)));
        b->add(w);
    }
    /* Two pieces that are the same text, which a real vocabulary also has. */
    b->v.add("dup");
    b->v.add("dup");

    size_t bytes = 0, n = 0;
    b->nul.assign((size_t)b->v.n_tokens(), 0);
    for (int32_t id = 0; id < b->v.n_tokens(); ++id) {
        const std::string& p = b->v.token_piece(id);
        if (p.empty()) continue;
        const auto dec = gbnf::decode_utf8(p, PartialUtf8{});
        for (size_t i = 0; i + 1 < dec.first.size(); ++i)
            if (dec.first[i] == 0) b->nul[(size_t)id] = 1;
        bytes += p.size();
        ++n;
        if (p.find_first_of("\"<>{}[],:\n\\") != std::string::npos) b->structural.push_back(id);
    }
    b->mean_bytes = (double)bytes / (double)n;
    return b;
}

/* ================================================================== the comparison */

struct Tally {
    int64_t masks = 0;       /* compared with the oracle */
    int64_t partial = 0;     /* of them, at a state carrying a split character */
    int64_t dead = 0;        /* of them, where both said no token is admissible */
    int64_t skipped = 0;     /* the oracle ran out of its work bound */
    int64_t bounded = 0;     /* the engine refused a state past its bounds that the oracle,
                              * rescanning one stack at a time, could still answer */
    int64_t grammars = 0;
    int64_t mismatches = 0;
};

std::string show_piece(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7f && c != '\'') { o += (char)c; continue; }
        char b[8];
        std::snprintf(b, sizeof b, "\\x%02x", c);
        o += b;
    }
    return o;
}

/* What the oracle may spend on one mask: enough for every grammar here that is not
 * pathological, and a tenth of a second of rescanning where one is. */
constexpr uint64_t kOracleWork = 1ull << 24;

/* The engine's mask at g's state against the oracle's. Returns the engine's status.
 *
 * ONE DIFFERENCE IS NOT A DISAGREEMENT. The engine answers a state through unions of its stacks'
 * successors, and in a grammar ambiguous enough -- a nullable star under a repetition forks a
 * stack per way of splitting the text -- those unions can outgrow the bound on one stack set,
 * where the oracle, rescanning the vocabulary once per stack, never holds more than one stack's.
 * The engine then fails the request, which is what its bounds promise. It is counted, and only
 * the random grammars below may reach it: every grammar anyone would write is held to zero. */
int compare(const GbnfGrammar& g, const VocabView& v, std::vector<uint32_t>& a,
            std::vector<uint32_t>& b, Tally& t, const std::string& what) {
    const int64_t n_words = (int64_t)a.size();
    const int sa = g.fill_mask(a.data(), n_words);
    const int sb = gbnf_oracle_mask(g, b.data(), n_words, kOracleWork);
    if (sb == RAD_E_FULL || sb == RAD_E_NOMEM) { ++t.skipped; return sa; }
    if ((sa == RAD_E_FULL || sa == RAD_E_NOMEM) && sb >= 0) {
        ++t.bounded;
        return sa;
    }
    ++t.masks;
    if (g.partial().n_remain > 0) ++t.partial;
    if (sa == RAD_E_STATE && sb == RAD_E_STATE) ++t.dead;
    const bool same = sa == sb && (sa < 0 || std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
    CHECK(same);
    if (!same && ++t.mismatches <= 4) {
        std::string ex;
        int shown = 0;
        for (int32_t id = 0; id < v.n_tokens() && shown < 4; ++id) {
            const bool x = (a[(size_t)id >> 5] >> (id & 31)) & 1u;
            const bool y = (b[(size_t)id >> 5] >> (id & 31)) & 1u;
            if (x == y) continue;
            ex += std::string(" ") + (x ? "+" : "-") + std::to_string(id) + "'" +
                  show_piece(v.token_piece(id)) + "'";
            ++shown;
        }
        std::fprintf(stderr,
                     "    %s: engine status %d, oracle %d, partial {%u,%d}, %zu stacks;%s\n",
                     what.c_str(), sa, sb, g.partial().value, g.partial().n_remain,
                     g.stacks().size(), ex.c_str());
    }
    return sa;
}

bool has(const std::vector<uint32_t>& m, int32_t id) {
    return id >= 0 && ((m[(size_t)id >> 5] >> (id & 31)) & 1u);
}

/* MAY THE MASK ADMIT A TOKEN THAT ACCEPTING THEN REFUSES? Upstream's mask and upstream's accept
 * read a piece differently in two places, and the engine reproduces the mask exactly, so an
 * admitted token can kill the parse:
 *
 *   - a token terminal (`<[id]>`) is applied by the mask wherever it is reached, in the middle of
 *     a piece included, and by accept only at a token's start;
 *   - a piece that decodes to U+0000 before its end -- an overlong sequence, finished or carried
 *     in -- is read by the mask as ending at that 0, and fed whole to the grammar by accept.
 *
 * Anywhere else, an admitted token must be accepted. */
bool accept_may_refuse(const GbnfGrammar& g, const std::string& gbnf, const VocabView& v,
                       int32_t id) {
    if (gbnf.find("<[") != std::string::npos) return true;
    const auto dec = gbnf::decode_utf8(v.token_piece(id), g.partial());
    for (size_t i = 0; i + 1 < dec.first.size(); ++i)
        if (dec.first[i] == 0) return true;
    return false;
}

/* A token the mask admits: now and then one that carries structure, so a walk gets through a
 * grammar's delimiters rather than circling in its widest loop; otherwise any. -1 if only the end
 * of generation is admissible. A piece that decodes to U+0000 is admitted almost everywhere and
 * accepted almost nowhere (accept_may_refuse), so it is compared in every mask and taken rarely,
 * or walks would end on it. */
int32_t pick_admitted(const std::vector<uint32_t>& m, const BpeVocab& bv, Rng& rng) {
    if (rng.chance(40)) {
        std::vector<int32_t> s;
        for (int32_t id : bv.structural) if (has(m, id)) s.push_back(id);
        if (!s.empty()) return rng.pick(s);
    }
    const bool take_nul = rng.chance(2);
    std::vector<int32_t> ids;
    for (int32_t id = 0; id < bv.v.n_tokens(); ++id)
        if (has(m, id) && !bv.v.is_eog(id) && (take_nul || !bv.nul[(size_t)id])) ids.push_back(id);
    return ids.empty() ? -1 : rng.pick(ids);
}

/* WALKS, AS THE SAMPLER DRIVES THEM. Each step drafts a window of up to three tokens -- mostly
 * admissible, sometimes any token at all -- pushes the state, advances over the drafts masking
 * every position, rolls back, and then accepts what verification would: the drafts up to the
 * first one its position's mask refused, and one token more. */
void walk_and_compare(const std::string& gbnf, const BpeVocab& bv, uint64_t seed, int n_walks,
                      int n_steps, Tally& t, const std::string& name) {
    const VocabView& v = bv.v;
    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<std::vector<uint32_t>> m(4, std::vector<uint32_t>((size_t)n_words));
    std::vector<uint32_t> ref((size_t)n_words);
    for (int w = 0; w < n_walks; ++w) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        REQUIRE(GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err) >= 0);
        Rng rng(seed * 1000 + (uint64_t)w);
        for (int step = 0; step < n_steps; ++step) {
            const int n_pos = 1 + (int)rng.below(4);
            std::vector<int32_t> d;
            g->push_state();
            int live = 0;
            for (int k = 0; k < n_pos; ++k) {
                if (k > 0 && g->accept_token(d[(size_t)k - 1]) < 0) {
                    /* A draft the grammar refuses leaves no parse; the mask there must say so. */
                    compare(*g, v, m[(size_t)k], ref, t, name + " after a refused draft");
                    break;
                }
                const int st = compare(*g, v, m[(size_t)k], ref, t,
                                       name + " step " + std::to_string(step) + " pos " +
                                           std::to_string(k));
                if (st < 0) break;
                ++live;
                if (k + 1 < n_pos) {
                    int32_t x = rng.chance(85) ? pick_admitted(m[(size_t)k], bv, rng)
                                               : (int32_t)rng.below((size_t)v.n_tokens());
                    if (x < 0) break;
                    d.push_back(x);
                }
            }
            CHECK_OK(g->rollback());
            if (live == 0) break;
            size_t j = 0;
            while ((int)j + 1 < live && j < d.size() && has(m[j], d[j])) ++j;
            bool ok = true;
            for (size_t k = 0; k < j && ok; ++k) {
                const bool quirk = accept_may_refuse(*g, gbnf, v, d[k]);
                const int  as = g->accept_token(d[k]);
                ok = as >= 0;
                CHECK(ok || quirk || as == RAD_E_FULL || as == RAD_E_NOMEM);
            }
            if (!ok) break;
            const int32_t bonus = pick_admitted(m[j], bv, rng);
            if (bonus < 0) break;
            const bool quirk = accept_may_refuse(*g, gbnf, v, bonus);
            const int  as = g->accept_token(bonus);
            if (as == RAD_E_FULL || as == RAD_E_NOMEM) break;     /* the parse outgrew its bounds */
            CHECK(as >= 0 || quirk);
            if (as < 0) break;
        }
    }
    ++t.grammars;
}

/* A REPLAY: a realistic output, tokenised, driven at depth-3 speculation with the true
 * continuation as the drafts -- every draft verifies, so each step masks four positions and
 * accepts three tokens, which is the steady state of a speculating constrained request. */
void replay_and_compare(const std::string& gbnf, const BpeVocab& bv,
                        const std::vector<int32_t>& toks, Tally& t, const std::string& name) {
    const VocabView& v = bv.v;
    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words), ref((size_t)n_words);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    REQUIRE(GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err) >= 0);
    size_t i = 0;
    while (i < toks.size()) {
        const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
        g->push_state();
        for (int k = 0; k < n_pos; ++k) {
            if (k > 0 && g->accept_token(toks[i + (size_t)k - 1]) < 0) { CHECK(false); break; }
            const std::string where = name + " token " + std::to_string(i + (size_t)k);
            if (compare(*g, v, m, ref, t, where) < 0) break;
            CHECK(has(m, toks[i + (size_t)k]));
        }
        CHECK_OK(g->rollback());
        const size_t n_acc = std::min<size_t>(3, toks.size() - i);
        for (size_t k = 0; k < n_acc; ++k) REQUIRE(g->accept_token(toks[i + k]) >= 0);
        i += n_acc;
    }
    ++t.grammars;
}

void report(const char* what, const Tally& t) {
    std::fprintf(stderr, "    %s: %lld grammars, %lld masks compared (%lld at a split character, "
                 "%lld dead); %lld beyond the oracle's bound, %lld beyond the engine's\n", what,
                 (long long)t.grammars, (long long)t.masks, (long long)t.partial, (long long)t.dead,
                 (long long)t.skipped, (long long)t.bounded);
}

/* ================================================================== grammars */

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string fixture(const char* name) {
    return std::string(RAD_GBNF_TEST_DATA) + "/" + name;
}

const char* kTools = R"([
 {"type":"function","function":{"name":"bash","description":"Run a shell command.",
  "parameters":{"type":"object","properties":{"command":{"type":"string"},
   "timeout":{"type":"integer"},"description":{"type":"string"}},"required":["command"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Write a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"edit_file","description":"Replace a string.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "old_string":{"type":"string"},"new_string":{"type":"string"},
   "replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}}},
 {"type":"function","function":{"name":"todo_write","description":"Update the todo list.",
  "parameters":{"type":"object","properties":{"todos":{"type":"array","items":{
   "type":"object","properties":{"content":{"type":"string"},
    "status":{"type":"string","enum":["pending","in_progress","completed"]}},
   "required":["content","status"]}}},"required":["todos"]}}}
])";

struct TemplateCase { const char* file; const char* bos; const char* eos; };
const TemplateCase kTemplates[] = {
    { "qwen3.8.jinja", "", "<|im_end|>" },
    { "qwen3.6.jinja", "", "<|im_end|>" },
    { "minicpm5.jinja", "<s>", "</s>" },
};

/* The tool-call grammar a template generates for kTools: parallel calls off and on, with
 * tool_choice "auto" or, where auto leaves the reply unconstrained, "required". */
std::vector<std::pair<std::string, std::string>> template_grammars() {
    std::vector<std::pair<std::string, std::string>> out;
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        if (ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) < 0) continue;
        for (int par = 0; par < 2; ++par) {
            for (const char* choice : { "auto", "required" }) {
                ChatOptions opt;
                opt.parallel_tool_calls = par != 0;
                opt.tool_choice = choice;
                ChatPrompt cp;
                if (ct.apply(messages, ojson::parse(kTools), opt, &cp) < 0) continue;
                if (cp.grammar.empty()) continue;
                out.push_back({ std::string(tc.file) + (par ? " parallel " : " single ") + choice,
                                cp.grammar });
                break;
            }
        }
    }
    return out;
}

const char* kSchemas[] = {
    R"({"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"},
        "env":{"type":"object","additionalProperties":{"type":"string"}}},"required":["command"]})",
    R"({"type":"object","properties":{"answer":{"type":"string"},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "sources":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},
            "year":{"type":"integer"}},"required":["title"]},"maxItems":4},
        "lang":{"enum":["en","de","ja"]}},"required":["answer","sources"]})",
    R"({"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},
        "name":{"type":"string","minLength":1,"maxLength":12},"tags":{"type":"array",
        "items":{"type":"string"}},"score":{"type":["number","null"]},
        "kind":{"oneOf":[{"const":"user"},{"const":"group"}]}},"required":["id","name","kind"]}})",
    R"({"anyOf":[{"type":"object","properties":{"name":{"const":"a"},"arguments":{"type":"object",
        "properties":{"x":{"type":"boolean"}}}},"required":["name"]},
        {"type":"string","pattern":"^[a-z]{2,5}-[0-9]+$"}]})",
    R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
        "kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},
        "$ref":"#/$defs/node"})",
    R"({"type":"object","properties":{"s":{"type":"string","maxLength":200},
        "code":{"type":"string","minLength":90}},"required":["s","code"]})",
};

/* Hand-written grammars at the edges of what the engine has to get right. `%T`/`%N` are replaced
 * with two token ids of the vocabulary, for the token syntax. */
const char* kTricky[][2] = {
    { "recursion",       "root ::= \"(\" root \")\" | \"x\"\n" },
    { "stack growth",    "root ::= \"[\" root \"]\" root | \"\"\n" },
    { "nested repeats",  "root ::= (\"a\" (\"b\" \"c\"?)* )+ \"d\"\n" },
    { "nullable star",   "root ::= (\"a\"?)* \"b\"\n" },
    { "ambiguity",       "root ::= \"ab\" \"c\" | \"a\" [b-c] \"c\" | \"a\" \"b\"?\n" },
    { "empty alts",      "root ::= \"a\" x \"b\"\nx ::= | \"c\" | y\ny ::= \"d\" |\n" },
    { "counted",         "root ::= [0-9]{2,4} \"-\" [a-f]{0,3} \"x\"{3} \" \"?\n" },
    { "ranges",          "root ::= [a-zA-Z0-9_]+ ([\\u00C0-\\u00FF] | [\\u4E00-\\u9FFF] | "
                         "\"\xc3\xa9\")* \"!\"\n" },
    { "negated",         "root ::= \"\\\"\" [^\"\\\\\\n]* \"\\\"\"\n" },
    { "negated wide",    "root ::= [^a-z\\u4E00-\\u9FFF]+ \".\"\n" },
    { "any char",        "root ::= . . (\"x\" | .) \"\\n\"\n" },
    { "multibyte",       "root ::= (\"\xe6\x97\xa5\xe6\x9c\xac\" | [\\U0001F600-\\U0001F64F] | "
                         "\"\xc3\xbc\")+ \"s\"?\n" },
    { "above bmp",       "root ::= [\\u0080-\\U0010FFFF]+ \"a\"\n" },
    { "escapes",         "root ::= \"\\x41\\u00e9\" [\\x01-\\x1F]? \"\\n\" \"\\t\"*\n" },
    { "long literal",    "root ::= \"the quick brown fox jumps over the lazy dog\" "
                         "\" \"? \"!\"\n" },
    { "whitespace",      "root ::= \"a\" [ \\t\\n]* \"b\" [ \\t]{0,20} \"c\" ws\nws ::= \" \"*\n" },
    { "deep nesting",    "root ::= ((((((((((((\"a\" | \"b\"))))))))))))+ ((((\"c\"))))\n" },
    { "json by hand",
      "root ::= value\n"
      "value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
      "object ::= \"{\" ws (string ws \":\" ws value (ws \",\" ws string ws \":\" ws value)*)? "
      "ws \"}\"\n"
      "array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n"
      "string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt])* \"\\\"\"\n"
      "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\nws ::= [ \\t\\n]*\n" },
    { "token syntax",    "root ::= <[%T]> \"x\"* | !<[%N]> \"y\" | \"z\" <[%N]>\n" },
    { "token mid-piece", "root ::= \"a\" <[%T]> | \"a\" !<[%N]> \"b\"\n" },
    { "long bounded",    "root ::= \"\\\"\" [^\"]{0,300} \"\\\"\"\n" },
    { "long required",   "root ::= [a-z]{150} \";\"\n" },
};

std::string with_tokens(std::string g, const BpeVocab& bv) {
    const std::string t = std::to_string(bv.by_piece.at("ab"));
    const std::string n = std::to_string(bv.by_piece.at("x"));
    for (size_t p; (p = g.find("%T")) != std::string::npos;) g.replace(p, 2, t);
    for (size_t p; (p = g.find("%N")) != std::string::npos;) g.replace(p, 2, n);
    return g;
}

/* ================================================================== random grammars

 * A grammar generator for what nobody thought to write by hand: rules of literals, character
 * classes (negated, multi-byte, above the BMP), `.`, groups under every repetition, rule
 * references forward and back, token terminals and empty alternatives. A reference in first
 * position only ever points forward, so most of what comes out has no left recursion; what the
 * compiler refuses anyway is counted, not failed. */
struct GrammarGen {
    Rng& rng;
    int  n_rules;
    const BpeVocab& bv;

    std::string literal() {
        static const char* atoms[] = { "a", "b", "c", "ab", "the", " ", "\\n", "\\\"", "<", "/",
                                       ">", "{", "}", ":", ",", "0", "7", "\xc3\xa9",
                                       "\xe6\x97\xa5", "\xf0\x9f\x98\x80", "x", "y" };
        std::string s = "\"";
        const size_t n = 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += atoms[rng.below(sizeof atoms / sizeof *atoms)];
        return s + "\"";
    }
    std::string cls() {
        static const char* parts[] = { "a-c", "x", "0-9", " ", "\\n", "\"", "\\\\", "<", "/",
                                       "\\u00C0-\\u00FF", "\\u4E00-\\u9FFF",
                                       "\\U0001F600-\\U0001F64F",
                                       "a-z", "{}", ":", "b" };
        std::string s = rng.chance(30) ? "[^" : "[";
        const size_t n = 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += parts[rng.below(sizeof parts / sizeof *parts)];
        return s + "]";
    }
    std::string repeat() {
        switch (rng.below(7)) {
            case 0: return "*";
            case 1: return "+";
            case 2: return "?";
            case 3:
                return "{" + std::to_string(rng.below(3)) + "," +
                       std::to_string(2 + rng.below(4)) + "}";
            case 4: return "{" + std::to_string(1 + rng.below(3)) + "}";
            default: return "";
        }
    }
    std::string item(int self, bool first, int depth) {
        const size_t k = rng.below(depth > 2 ? 8 : 10);
        switch (k) {
            case 0: case 1: case 2: return literal() + repeat();
            case 3: case 4: return cls() + repeat();
            case 5:
                if (rng.chance(50)) return ".";
                return "\"" + std::string(1, (char)('a' + rng.below(3))) + "\"";
            case 6: {
                if (rng.chance(15)) {
                    const int32_t id = rng.pick(bv.structural);
                    return std::string(rng.chance(50) ? "<[" : "!<[") + std::to_string(id) + "]>";
                }
                const int lo = first ? self + 1 : 0;
                if (lo >= n_rules) return literal();
                return "r" + std::to_string(lo + (int)rng.below((size_t)(n_rules - lo)));
            }
            case 7:
                return "r" + std::to_string(std::min(n_rules - 1, self + 1 + (int)rng.below(2)));
            default: return "(" + alts(self, depth + 1) + ")" + repeat();
        }
    }
    /* An empty sequence at a rule's top level is written `""`: GBNF skips newlines after `::=`
     * and `|`, so an empty last alternative at the end of a line would run into the next rule. */
    std::string seq(int self, int depth) {
        std::string s;
        const size_t n = rng.chance(10) ? 0 : 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += (i ? " " : "") + item(self, i == 0, depth);
        return s.empty() && depth == 0 ? "\"\"" : s;
    }
    std::string alts(int self, int depth) {
        std::string s = seq(self, depth);
        const size_t n = rng.below(3);
        for (size_t i = 0; i < n; ++i) s += " | " + seq(self, depth);
        return s;
    }
    std::string grammar() {
        std::string g = "root ::= r0 (\"!\" | \"\")\n";
        for (int r = 0; r < n_rules; ++r)
            g += "r" + std::to_string(r) + " ::= " + alts(r, 0) + "\n";
        return g;
    }
};

/* ================================================================== timing */

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string go_file() {
    std::string s = "package cache\n\nimport (\n\t\"container/list\"\n\t\"sync\"\n)\n\n";
    for (int i = 0; i < 6; ++i) {
        const std::string n = std::to_string(i);
        s += "// Shard" + n + " holds a fixed-capacity LRU keyed by string.\n"
             "type Shard" + n + " struct {\n\tmu    sync.Mutex\n\tll    *list.List\n}\n\n"
             "func (s *Shard" + n + ") Get(key string) (any, bool) {\n"
             "\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\tif e, ok := s.items[key]; ok {\n"
             "\t\treturn e.Value.(*entry).val, true\n\t}\n\treturn nil, false\n}\n\n";
    }
    return s;
}

std::vector<std::string> tool_call_texts() {
    return {
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "cd /repo && go test ./... 2>&1 | head -50\n</parameter>\n"
        "<parameter=description>\nRun the test suite\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=read_file>\n<parameter=path>\n/repo/cache/lru.go\n</parameter>\n"
        "<parameter=limit>\n200\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/repo/cache/shards.go\n"
        "</parameter>\n"
        "<parameter=content>\n" + go_file() + "\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=edit_file>\n<parameter=path>\n"
        "/repo/\xe6\x97\xa5\xe6\x9c\xac/lru.go\n"
        "</parameter>\n<parameter=old_string>\n\tif s.ll.Len() >= s.cap {\n</parameter>\n"
        "<parameter=new_string>\n\tfor s.ll.Len() >= s.cap && s.cap > 0 { "
        "// caf\xc3\xa9 \xf0\x9f\x98\x80\n"
        "</parameter>\n</function>\n</tool_call>",
    };
}

}  /* namespace */

/* ================================================================== the cases */

TEST(gbnf_the_synthetic_vocabulary_is_shaped_like_a_real_one) {
    auto bv = bpe_like(7, 4000);
    /* The properties the engine's plan is sensitive to, held so that a change to the generator
     * cannot quietly make every other case here easier. */
    CHECK(bv->v.n_tokens() >= 4000);
    CHECK(bv->mean_bytes > 4.0 && bv->mean_bytes < 10.0);
    CHECK(bv->longest >= 40);
    int ends_mid = 0, starts_mid = 0, eog = 0, control = 0;
    for (int32_t id = 0; id < bv->v.n_tokens(); ++id) {
        const std::string& p = bv->v.token_piece(id);
        if (bv->v.is_eog(id)) { ++eog; continue; }
        if (p.empty() || p[0] == 0) { ++control; continue; }
        if (((uint8_t)p[0] >> 6) == 2) ++starts_mid;
        const bool head_last = (uint8_t)p.back() >= 0xC0;
        const bool head_second = p.size() >= 2 && (uint8_t)p[p.size() - 2] >= 0xE0;
        if (head_last || head_second) ++ends_mid;
    }
    CHECK(ends_mid > 50);
    CHECK(starts_mid > 50);
    CHECK_EQ(eog, 2);
    CHECK_EQ(control, 3);
}

TEST(gbnf_hand_written_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(11, 3000);
    Tally t;
    uint64_t seed = 1;
    for (const auto& c : kTricky) {
        const std::string g = with_tokens(c[1], *bv);
        std::unique_ptr<GbnfGrammar> probe;
        std::string err;
        const int st = GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &probe, &err);
        if (st < 0) std::fprintf(stderr, "    %s: %s\n", c[0], err.c_str());
        CHECK_OK(st);
        if (st < 0) continue;
        walk_and_compare(g, *bv, seed++, 6, 40, t, c[0]);
    }
    report("hand-written", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.masks > 2500);
    CHECK(t.partial > 20);
}

TEST(gbnf_json_schema_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(13, 3000);
    Tally t;
    uint64_t seed = 100;
    for (const char* schema : kSchemas) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        if (gbnf.empty()) continue;
        ++seed;
        walk_and_compare(gbnf, *bv, seed, 6, 50, t, "schema " + std::to_string(seed - 100));
    }
    report("json schema", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.masks > 1200);
}

/* A SCHEMA NO VALUE SATISFIES, which vendor/patches/0012 represents rather than refuses. oh-my-pi's
 * `task` tool declares `"model?": "never"` in its batch form and Arktype exports that as
 * `{"not": {}}`, the same schema as JSON Schema's `false`. Refused, it failed the tool-call grammar
 * of every request that carried the tool. Represented, it is a key the object never holds. */
TEST(gbnf_a_property_no_value_satisfies_is_a_key_never_written) {
    auto bv = bpe_like(23, 1500);
    /* A whole value, from a fresh grammar: every byte taken and the parse at an end. */
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };

    for (const char* never : { "{\"not\":{}}", "false" }) {
        const std::string schema = std::string("{\"type\":\"object\",\"properties\":{"
                                               "\"context\":{\"type\":\"string\"},\"model\":") +
                                   never + "},\"required\":[\"context\"]}";
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"context\": \"x\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": \"y\"}"));
        CHECK(!whole(gbnf, "{\"model\": \"y\", \"context\": \"x\"}"));
    }

    /* additionalProperties must not readmit the key with a value of its own. */
    {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(
            R"({"type":"object","properties":{"model":{"not":{}}},"additionalProperties":true})",
            "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"other\": 1}"));
        CHECK(!whole(gbnf, "{\"model\": 1}"));
    }

    /* A union alternative of that kind contributes nothing to the union. */
    {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(R"({"anyOf":[{"type":"string"},{"not":{}}]})", "root",
                                          &gbnf, &err));
        CHECK(whole(gbnf, "\"a\""));
        CHECK(!whole(gbnf, "1"));
    }

    /* Where a value is required there is nothing to write, and that is refused by name. */
    for (const char* schema : {
             R"({"type":"object","properties":{"model":{"not":{}}},"required":["model"]})",
             R"({"not":{}})",
             R"({"anyOf":[{"not":{}},false]})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(schema, "root", &gbnf, &err) < 0);
        CHECK(!err.empty());
    }

    /* And the tool as oh-my-pi sends it, through each served template's tool-call grammar. */
    const char* tools = R"([{"type":"function","function":{"name":"task",
      "description":"Spawn subagents.","parameters":{"type":"object","properties":{
       "context":{"type":"string"},"model":{"not":{}},
       "tasks":{"type":"array","items":{"type":"object","properties":{
        "name":{"type":"string"},"agent":{"type":"string","default":"task"},
        "task":{"type":"string"},"solutionSpace":{"type":"string"},
        "model":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]},
        "schemaMode":{"enum":["permissive","strict"]},
        "tools":{"type":"array","items":{"type":"string"}}},
        "required":["agent","solutionSpace","task"]}}},
       "required":["context","tasks"]}}}])";
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    /* "auto" is the choice the server serves (it refuses "required"), with parallel calls on --
     * the server's default -- and off. Under "auto" a template may leave the reply unconstrained
     * and produce no grammar at all; what must not happen is a refusal. */
    int built = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            ChatOptions opt;
            opt.tool_choice = "auto";
            opt.parallel_tool_calls = par != 0;
            ChatPrompt cp;
            CHECK(ct.apply(messages, ojson::parse(tools), opt, &cp) >= 0);
            if (!cp.grammar.empty()) ++built;
        }
    }
    CHECK(built >= 3);
}

/* THE `not` OF A SCHEMA EVERY VALUE SATISFIES is `never` too, whatever spelling the universal
 * schema takes. oh-my-pi 18.6.0 sends its `task` tool's optional `model` as the `not` of a union
 * of all six JSON types, not as `{"not": {}}`; refused, it failed the tool-call grammar of every
 * request carrying the tool under every template that constrains arguments. A `not` of anything
 * narrower is a real complement and stays refused. */
TEST(gbnf_a_not_of_every_json_type_is_never_too) {
    auto bv = bpe_like(29, 1500);
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };
    auto optional_model = [](const std::string& never) {
        return std::string(R"({"type":"object","properties":{"context":{"type":"string"},"model":)") +
               never + R"(},"required":["context"]})";
    };

    for (const char* never : {
             R"({"not":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},)"
             R"({"type":"object"},{"type":"array"},{"type":"null"}]}})",
             R"({"not":{"type":["string","number","boolean","object","array","null"]}})",
             R"({"not":{"anyOf":[{"type":["string","number"]},{"type":["boolean","object"]},)"
             R"({"type":"array","description":"a list"},{"type":"null"}]}})",
             R"({"not":{"anyOf":[{"type":"string","minLength":1},{}]}})",
             R"({"not":true})",
             R"({"not":{"description":"anything"}})" }) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(optional_model(never), "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"context\": \"x\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": \"y\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": null}"));
    }

    for (const char* complement : {
             /* no null among the six */
             R"({"not":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},)"
             R"({"type":"object"},{"type":"array"}]}})",
             /* integer is not number */
             R"({"not":{"type":["string","integer","boolean","object","array","null"]}})",
             /* an alternative that constrains more than its type is not all of that type */
             R"({"not":{"anyOf":[{"type":"string","minLength":1},{"type":"number"},)"
             R"({"type":"boolean"},{"type":"object"},{"type":"array"},{"type":"null"}]}})",
             /* a keyword beside the type narrows it */
             R"({"not":{"type":["string","number","boolean","object","array","null"],"minimum":0}})",
             R"({"not":{"type":"string"}})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(optional_model(complement), "root", &gbnf, &err) < 0);
        CHECK(err.find("not") != std::string::npos);
    }
}

/* OH-MY-PI 18.6.0's TOOLS AS IT SENDS THEM, captured from an unmodified client and kept whole but
 * for their descriptions (tests/data/chat/oh-my-pi-18.6.0-tools.json). Every served template must
 * render them and every grammar it writes must compile, which is what admission does with it. */
TEST(gbnf_oh_my_pi_tools_build_and_compile_under_every_template) {
    auto bv = bpe_like(31, 3000);
    const ojson tools = ojson::parse(slurp(fixture("../chat/oh-my-pi-18.6.0-tools.json")));
    REQUIRE(tools.is_array() && tools.size() == 11);
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    int built = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            ChatOptions opt;
            opt.tool_choice = "auto";
            opt.parallel_tool_calls = par != 0;
            opt.template_kwargs["preserve_thinking"] = "true";
            ChatPrompt cp;
            CHECK(ct.apply(messages, tools, opt, &cp) >= 0);
            if (cp.grammar.empty()) continue;
            ++built;
            std::unique_ptr<GbnfGrammar> g;
            std::string err;
            CHECK(GbnfGrammar::create(cp.grammar, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
            if (!err.empty()) fprintf(stderr, "    %s: %s\n", tc.file, err.c_str());
        }
    }
    CHECK(built >= 2);
}

/* The grammars a chat template writes for its own tool-call syntax, from the three templates the
 * engine serves: walked, and replayed over four agent calls wherever the syntax is the one the
 * calls are written in. */
TEST(gbnf_tool_call_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(17, 3000);
    const auto grammars = template_grammars();
    CHECK(grammars.size() >= 4);
    Tally t;
    uint64_t seed = 200;
    int replays = 0;
    for (const auto& ng : grammars) {
        walk_and_compare(ng.second, *bv, seed++, 4, 40, t, ng.first);
        for (const std::string& text : tool_call_texts()) {
            std::unique_ptr<GbnfGrammar> probe;
            std::string err;
            REQUIRE(GbnfGrammar::create(ng.second, "root", &bv->v, false, {}, {}, &probe,
                                        &err) >= 0);
            if (probe->accept_str(text) < 0) continue;
            const std::vector<int32_t> toks = bv->tokenize(text);
            REQUIRE(!toks.empty());
            replay_and_compare(ng.second, *bv, toks, t, ng.first + " replay");
            ++replays;
        }
    }
    report("tool calls", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(replays >= 8);
    CHECK(t.masks > 1200);
}

TEST(gbnf_random_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(19, 1500);
    Tally t;
    int compiled = 0, refused = 0;
    std::map<std::string, int> why;
    for (uint64_t seed = 1; seed <= 250; ++seed) {
        Rng rng(seed);
        GrammarGen gen{ rng, 2 + (int)rng.below(5), *bv };
        const std::string g = gen.grammar();
        std::unique_ptr<GbnfGrammar> probe;
        std::string err;
        if (GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &probe, &err) < 0) {
            ++refused;
            CHECK(!err.empty());
            ++why[err.substr(0, err.find_first_of(";:"))];
            continue;
        }
        ++compiled;
        walk_and_compare(g, *bv, seed, 2, 24, t, "random grammar " + std::to_string(seed));
        if (t.mismatches) {
            std::fprintf(stderr, "    the grammar:\n%s", g.c_str());
            break;
        }
    }
    report("random", t);
    CHECK(t.bounded * 100 <= t.masks);
    for (const auto& kv : why)
        std::fprintf(stderr, "      refused %d: %s\n", kv.second, kv.first.c_str());
    std::fprintf(stderr, "    %d of %d random grammars compiled\n", compiled, compiled + refused);
    CHECK_EQ(t.mismatches, 0);
    /* Most of what the generator writes is runnable; what is refused is refused for a reason the
     * refusal names -- left recursion, mostly. */
    CHECK(compiled > 150);
    CHECK(t.masks > 8000);
}

/* A STATE CARRYING A SPLIT CHARACTER is served by its own index, not by the plan: only pieces
 * that begin with the bytes finishing the character are admissible. Walked into deliberately --
 * accept the head of a character, mask -- over grammars whose classes cut through the ranges the
 * carried bits could complete to. */
TEST(gbnf_masks_after_a_split_character_agree_with_the_oracle) {
    auto bv = bpe_like(23, 3000);
    const char* grammars[] = {
        "root ::= [\\u0080-\\U0010FFFF]+\n",
        "root ::= ([\\u4E00-\\u9FFF] | [\\u00C0-\\u00FF] \"s\"?)+ \"!\"\n",
        "root ::= (\"\xe6\x97\xa5\" | \"\xe6\x9c\xac\" [^\\u4E00-\\u4E2F])* [a-z]\n",
        "root ::= [^\\u00E0-\\u00FF]* [\\U0001F600-\\U0001F602] \"x\"\n",
        "root ::= \"\\\"\" [^\"\\\\\\x00-\\x1F]* \"\\\"\"\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    for (const char* gt : grammars) {
        for (const char* m : kMultibyte) {
            const std::string s(m);
            for (size_t k = 1; k < s.size(); ++k) {
                std::unique_ptr<GbnfGrammar> g;
                std::string err;
                REQUIRE(GbnfGrammar::create(gt, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
                /* The head of the character, then its middle a byte at a time. */
                if (g->accept_token(bv->by_piece.at(s.substr(0, k))) < 0) continue;
                compare(*g, bv->v, a, b, t,
                        std::string(gt) + " after " + show_piece(s.substr(0, k)));
                if (k + 1 < s.size() && g->accept_token(bv->by_piece.at(s.substr(k, 1))) >= 0)
                    compare(*g, bv->v, a, b, t, std::string(gt) + " a byte later");
            }
        }
        walk_and_compare(gt, *bv, 300, 6, 40, t, gt);
    }
    report("split characters", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.partial > 60);
}

/* A CHARACTER FINISHED AS U+0000 ENDS THE PIECE, as upstream reads it: decoded sequences are
 * NUL-terminated, so a piece that completes an overlong head to 0 -- or decodes to 0 itself --
 * reads as having no characters, whatever bytes follow. The mask after such a head is served by
 * its own path in the engine, and is held to the oracle here state by state. */
TEST(gbnf_a_character_finished_as_nul_agrees_with_the_oracle) {
    auto bv = bpe_like(59, 1500);
    const char* grammars[] = {
        "root ::= [^a]* \"!\"\n",
        "root ::= [\\x00-\\x7F]+\n",
        "root ::= (\"x\" | [\\u0080-\\U0010FFFF])+ \"z\"?\n",
        "root ::= \"q\" | !<[5]> \"y\"\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    for (const char* gt : grammars) {
        for (const char* head : { "\xc0", "\xe0", "\xe0\x80", "\xf0", "\xf0\x80\x80" }) {
            std::unique_ptr<GbnfGrammar> g;
            std::string err;
            if (GbnfGrammar::create(gt, "root", &bv->v, false, {}, {}, &g, &err) < 0) continue;
            if (g->accept_token(bv->by_piece.at(head)) < 0) continue;
            compare(*g, bv->v, a, b, t, std::string(gt) + " after " + show_piece(head));
        }
        walk_and_compare(gt, *bv, 700, 6, 30, t, gt);
    }
    report("characters finished as NUL", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK(t.partial > 10);
}

/* A token terminal matches an id, not text, and upstream's walk applies it wherever it is reached
 * -- in the middle of a piece included, where it admits the token with that id, or every other
 * token for `!<[id]>`. The engine has to reproduce that exactly, oddity and all. */
TEST(gbnf_token_terminals_agree_with_the_oracle) {
    auto bv = bpe_like(29, 2000);
    const int32_t ab = bv->by_piece.at("ab"), a = bv->by_piece.at("a"), x = bv->by_piece.at("x");
    const std::string grammars[] = {
        "root ::= <[" + std::to_string(ab) + "]> \"c\" | \"d\"\n",
        "root ::= !<[" + std::to_string(x) + "]> \"y\"*\n",
        "root ::= \"a\" <[" + std::to_string(ab) + "]> | \"a\" !<[" + std::to_string(a) + "]>\n",
        "root ::= (\"x\" !<[" + std::to_string(ab) + "]>)+ \"z\"?\n",
        "root ::= \"<think>\" [^<]* <[" + std::to_string(bv->by_piece.at("</think>")) + "]>\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> ma((size_t)n_words), mb((size_t)n_words);
    Tally t;
    uint64_t seed = 400;
    for (const std::string& g : grammars) {
        std::unique_ptr<GbnfGrammar> gg;
        std::string err;
        REQUIRE(GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &gg, &err) >= 0);
        compare(*gg, bv->v, ma, mb, t, g);
        walk_and_compare(g, *bv, seed++, 6, 30, t, g);
    }
    report("token terminals", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
}

/* The statuses, which are part of the contract: no parse left, nothing admissible but the end,
 * and a grammar that has completed. */
TEST(gbnf_dead_and_complete_states_agree_with_the_oracle) {
    auto bv = bpe_like(31, 1500);
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    REQUIRE(GbnfGrammar::create("root ::= \"ab\" | \"c\"\n", "root", &bv->v, false, {}, {}, &g,
                                &err) >= 0);
    CHECK_OK(compare(*g, bv->v, a, b, t, "start"));
    CHECK_OK(g->accept_token(bv->by_piece.at("ab")));
    CHECK(g->complete());
    CHECK_OK(compare(*g, bv->v, a, b, t, "complete"));       /* only the end is admissible */
    CHECK(g->accept_token(bv->by_piece.at("x")) < 0);         /* no parse left */
    CHECK_EQ(compare(*g, bv->v, a, b, t, "dead"), RAD_E_STATE);
    CHECK_EQ(t.mismatches, 0);
}

/* A REPETITION LONGER THAN ANY TOKEN IS A STAR TO THE PLAN. `x{0,n}` is n rules and `x{n}` is n
 * copies, each a key the plan would walk the vocabulary for; with n past the longest token, every
 * one past that point answers alike and they are one key. So the plan stays the size of the
 * vocabulary's longest token whatever n is, and the masks deep into the repetition are still
 * exact. */
TEST(gbnf_a_repetition_longer_than_any_token_costs_what_the_token_does) {
    auto bv = bpe_like(37, 1500);
    std::vector<GbnfProgramInfo> info;
    for (int n : { 200, 400, 1600 }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        const std::string g = "root ::= \"\\\"\" [^\"]{0," + std::to_string(n) +
                              "} \"\\\"\" [a-z]{" +
                              std::to_string(n) + "} \";\"\n";
        REQUIRE(gbnf_compile(g, "root", &bv->v, &p, &err) >= 0);
        info.push_back(gbnf_program_info(*p));
    }
    CHECK_EQ(info[0].keys, info[1].keys);
    CHECK_EQ(info[1].keys, info[2].keys);
    CHECK_EQ(info[0].dense_masks, info[2].dense_masks);

    Tally t;
    walk_and_compare("root ::= \"\\\"\" [^\"]{0,400} \"\\\"\" [a-z]{400} \";\"\n", *bv, 500, 4, 120,
                     t, "long repetitions");
    report("long repetitions", t);
    CHECK_EQ(t.mismatches, 0);
}

/* A LENGTH BOUND PAST 2000 IS HELD, EXACTLY. Upstream refused any repetition count past 2000, so
 * a tool whose string argument declared `maxLength: 8000` failed every request carrying it. The
 * expansion is bounded by what it costs to compile, not by the count: the bound is the string's
 * length to the character, the plan is the one a count under 2000 compiles to, and only a count
 * the compile budget cannot hold is refused -- naming the rule it is in. */
TEST(gbnf_a_length_bound_past_two_thousand_is_held_exactly) {
    auto bv = bpe_like(59, 1500);
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };
    auto from_schema = [](const std::string& schema) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        return gbnf;
    };
    auto quoted = [](size_t n) { return "\"" + std::string(n, 'a') + "\""; };

    for (int n : { 2001, 8000, 65536 }) {
        const std::string g =
            from_schema("{\"type\":\"string\",\"maxLength\":" + std::to_string(n) + "}");
        CHECK(whole(g, quoted(0)));
        CHECK(whole(g, quoted((size_t)n)));
        CHECK(!whole(g, quoted((size_t)n + 1)));
    }
    {
        const std::string g = from_schema(R"({"type":"string","minLength":3000,"maxLength":8000})");
        CHECK(!whole(g, quoted(2999)));
        CHECK(whole(g, quoted(3000)));
        CHECK(whole(g, quoted(8000)));
        CHECK(!whole(g, quoted(8001)));
    }
    {
        const std::string g =
            from_schema(R"({"type":"array","items":{"type":"integer"},"maxItems":5000})");
        auto ones = [](int k) {
            std::string s = "[";
            for (int i = 0; i < k; ++i) s += i ? ",1" : "1";
            return s + "]";
        };
        CHECK(whole(g, ones(5000)));
        CHECK(!whole(g, ones(5001)));
    }

    /* The plan does not grow with the bound: past the longest token, no token tells the counts
     * apart. */
    std::vector<GbnfProgramInfo> info;
    for (int n : { 1999, 8000, 65536 }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        REQUIRE(gbnf_compile(from_schema("{\"type\":\"string\",\"maxLength\":" +
                                         std::to_string(n) + "}"),
                             "root", &bv->v, &p, &err) >= 0);
        info.push_back(gbnf_program_info(*p));
    }
    CHECK_EQ(info[0].keys, info[1].keys);
    CHECK_EQ(info[1].keys, info[2].keys);
    CHECK_EQ(info[0].dense_masks, info[2].dense_masks);

    /* The tool from the report, four 8000-character fields, through every served template's
     * tool-call grammar and compiled: the request it is in is served. */
    const char* tools = R"([{"type":"function","function":{"name":"clarify",
      "description":"Ask the user to choose.","parameters":{"type":"object","properties":{
       "question":{"type":"string","maxLength":8000},
       "choices":{"type":"array","maxItems":4,"items":{"type":"string","maxLength":8000}}},
       "required":["question","choices"]}}}])";
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    int compiled = 0, carried = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            for (const char* choice : { "auto", "required" }) {
                ChatOptions opt;
                opt.tool_choice = choice;
                opt.parallel_tool_calls = par != 0;
                ChatPrompt cp;
                CHECK(ct.apply(messages, ojson::parse(tools), opt, &cp) >= 0);
                if (cp.grammar.empty()) continue;
                std::shared_ptr<const GbnfProgram> p;
                std::string err;
                const auto t0 = std::chrono::steady_clock::now();
                CHECK(gbnf_compile(cp.grammar, "root", &bv->v, &p, &err) >= 0);
                if (!err.empty()) std::fprintf(stderr, "    %s: %s\n", tc.file, err.c_str());
                const double ms = ms_since(t0);
                CHECK(ms < 2000.0);
                ++compiled;
                if (cp.grammar.find("8000}") != std::string::npos) {
                    ++carried;
                    std::fprintf(stderr, "    %s, %s, parallel %s: the bound is in the grammar\n",
                                 tc.file, choice, par ? "on" : "off");
                }
            }
        }
    }
    CHECK(compiled >= 6);
    CHECK(carried >= 1);

    /* A group under a repetition is a reference to the group's rule, not a copy of it, so its
     * cost is the two counts added, not multiplied; upstream refused it as multiplied. */
    {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile("root ::= ([a-z]{3000}){3000} \";\"\n", "root", &bv->v, &p, &err) >= 0);
    }

    /* A schema's bound past INT_MAX -- Number.MAX_SAFE_INTEGER is a common "no limit" -- is no
     * bound, which no context the engine serves can tell from the bound; it used to wrap to -1 and
     * fail to parse. A negative or non-numeric bound is refused by the converter, naming it. */
    for (const char* big : { "9007199254740991", "1e300", "2147483648" }) {
        const std::string g =
            from_schema(std::string("{\"type\":\"string\",\"maxLength\":") + big + "}");
        CHECK(whole(g, quoted(3000)));
        const std::string a = from_schema(std::string("{\"type\":\"array\",\"maxItems\":") + big +
                                          ",\"items\":{\"type\":\"integer\"}}");
        CHECK(whole(a, "[1,2,3]"));
    }
    for (const char* bad : { R"({"type":"string","maxLength":-5})",
                             R"({"type":"string","minLength":"3"})",
                             R"({"type":"array","items":{},"maxItems":-1})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(bad, "root", &gbnf, &err) < 0);
        CHECK(err.find("must be a non-negative number") != std::string::npos);
    }

    /* A bound the compile budget cannot hold is refused naming the property's rule and the
     * count, which is what a client needs to find it in its tool list. */
    {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile(from_schema(R"({"type":"object","properties":{"content":)"
                                       R"({"type":"string","maxLength":1000000}}})"),
                           "root", &bv->v, &p, &err) < 0);
        CHECK(err.find("rule 'content'") != std::string::npos);
        CHECK(err.find("{0,1000000}") != std::string::npos);
    }

    /* What the budget cannot hold is refused, by rule, and a count past 2^64 - 2 is refused as
     * one; none of them wraps into a small expansion that compiles. A repetition of a repetition
     * copies the expanded run, so `{3000}{3000}` is nine million elements. */
    for (const char* g : { "root ::= \"a\" item\nitem ::= [a-z]{0,100000000}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{0,18446744073709551614}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{18446744073709551614}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{0,99999999999999999999999}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{3000}{3000}\n" }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile(g, "root", &bv->v, &p, &err) < 0);
        CHECK(err.find("rule 'item'") != std::string::npos);
    }
}

/* EVERY TOOL'S COPY OF THE SAME CLAUSE IS ONE CLAUSE TO THE PLAN. A tool-calling template writes
 * the same undeclared-parameter clause, the same free-text value and the same closing tags once
 * per tool, as rules of their own; the plan hash-conses rules by body, so its size -- and the
 * number of vocabulary walks the compile makes -- does not grow with the number of tools. */
TEST(gbnf_rule_copies_share_one_plan) {
    auto bv = bpe_like(41, 1500);
    ChatTemplate ct;
    REQUIRE(ct.load(slurp(fixture("qwen3.8.jinja")), "", "<|im_end|>") >= 0);
    std::vector<size_t> dense;
    for (int n_tools : { 3, 30 }) {
        ojson tools = ojson::array();
        for (int i = 0; i < n_tools; ++i)
            tools.push_back({ { "type", "function" }, { "function", {
                { "name", "tool_" + std::to_string(i) }, { "description", "d" },
                { "parameters", { { "type", "object" }, { "properties", {
                    { "path", { { "type", "string" } } }, { "n", { { "type", "integer" } } } } },
                    { "required", { "path" } } } } } } });
        ChatPrompt cp;
        REQUIRE(ct.apply(ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } }), tools,
                         ChatOptions(), &cp) >= 0);
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        REQUIRE(gbnf_compile(cp.grammar, "root", &bv->v, &p, &err) >= 0);
        dense.push_back(gbnf_program_info(*p).dense_masks);
    }
    CHECK_EQ(dense[0], dense[1]);
}

/* A GRAMMAR WHOSE MASK NEEDS UNBOUNDED CONTEXT IS REFUSED AT COMPILE TIME, by name. Here every
 * level of recursion leaves an optional space below it, so how many spaces a token may start with
 * depends on how deep the parse is -- and no stack suffix of bounded depth decides a mask. The
 * engine serves nothing slowly; it says which rule it cannot precompute. */
TEST(gbnf_a_grammar_whose_masks_need_unbounded_context_is_refused) {
    auto bv = bpe_like(43, 24000);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    const int st = GbnfGrammar::create("root ::= [a-z] root s | \"\"\ns ::= [^a-z]?\n", "root",
                                       &bv->v, false, {}, {}, &g, &err);
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
    CHECK(err.find("'root'") != std::string::npos || err.find("'s'") != std::string::npos);
    std::fprintf(stderr, "    refused: %s\n", err.c_str());
}

/* A RULE THAT CAN NEVER MATCH ANYTHING IS REFUSED, by name. `r ::= r` passes the left-recursion
 * check -- the reference is in tail position, so the stack never grows -- but every way into it
 * leads back into it, and upstream's walk reads the empty stack set that leaves as admitting every
 * token. A grammar that merely never ends is a different thing and still compiles. */
TEST(gbnf_a_rule_that_can_never_match_is_refused) {
    auto bv = bpe_like(61, 800);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" r | \"b\"\nr ::= r\n", "root", &bv->v, false,
                                 {}, {}, &g, &err), RAD_E_UNSUPPORTED);
    CHECK(err.find("'r'") != std::string::npos);
    err.clear();
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" s\ns ::= t\nt ::= (\"\")? s\n", "root", &bv->v,
                                 false, {}, {}, &g, &err), RAD_E_UNSUPPORTED);
    CHECK(!err.empty());
    CHECK_OK(GbnfGrammar::create("root ::= \"a\" root\n", "root", &bv->v, false, {}, {}, &g, &err));
    CHECK_OK(GbnfGrammar::create("root ::= [a-z] root | \"\"\n", "root", &bv->v, false, {}, {},
                                 &g, &err));
}

/* THE SCHEDULER THREAD NEVER COMPILES A GRAMMAR THE SERVER ADMITTED. Admission compiles it and the
 * request holds the program; starting the request finds that program by its text however many
 * other grammars were compiled since, because a program is found for as long as anything holds
 * it, and not only while the cache pins it. */
TEST(gbnf_a_held_program_is_found_whatever_was_compiled_since) {
    auto bv = bpe_like(47, 800);
    std::shared_ptr<const GbnfProgram> held;
    std::string err;
    REQUIRE(gbnf_compile("root ::= \"held\" [a-z]*\n", "root", &bv->v, &held, &err) >= 0);
    for (int i = 0; i < 80; ++i) {
        std::shared_ptr<const GbnfProgram> p;
        REQUIRE(gbnf_compile("root ::= \"other" + std::to_string(i) + "\"\n", "root", &bv->v, &p,
                             &err) >= 0);
    }
    std::unique_ptr<GbnfGrammar> g;
    REQUIRE(GbnfGrammar::create("root ::= \"held\" [a-z]*\n", "root", &bv->v, false, {}, {}, &g,
                                &err) >= 0);
    CHECK(&g->program() == held.get());
}

/* WHAT A MASK MAY COST, measured, at a scale that runs in seconds: a 48,000-piece vocabulary shaped
 * like a real one, the tool-call grammar the Qwen template writes, and four agent calls replayed
 * the way the sampler drives them. Every mask is timed on a program compiled for the run, so the
 * first request's masks are all in it -- none of them from a cache warmed by an earlier one. Each
 * run is repeated and the fastest kept per mask, which is what the code costs rather than what a
 * busy machine did to it once.
 *
 * The budgets are an order of magnitude above what the engine takes and an order of magnitude
 * below what one walk of this vocabulary takes, so they fail on a regression to walking the
 * vocabulary per mask and not on a slow machine. */
TEST(gbnf_mask_latency_stays_within_budget) {
    auto bv = bpe_like(53, 48000);
    ChatTemplate ct;
    REQUIRE(ct.load(slurp(fixture("qwen3.8.jinja")), "", "<|im_end|>") >= 0);
    ChatPrompt cp;
    REQUIRE(ct.apply(ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } }),
                     ojson::parse(kTools), ChatOptions(), &cp) >= 0);
    REQUIRE(!cp.grammar.empty());
    std::vector<std::vector<int32_t>> calls;
    for (const std::string& text : tool_call_texts()) calls.push_back(bv->tokenize(text));

    GbnfVocabInfo vi;
    CHECK_OK(gbnf_prepare_vocab(&bv->v, &vi));
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words);

    std::vector<double> best_mask, best_step;
    double best_compile = 1e30;
    for (int run = 0; run < 3; ++run) {
        /* A trailing comment is a new program: compiled from nothing, every mask computed. */
        const std::string text = cp.grammar + "\n# run " + std::to_string(run) + "\n";
        std::shared_ptr<const GbnfProgram> prog;
        std::string err;
        const auto tc = std::chrono::steady_clock::now();
        REQUIRE(gbnf_compile(text, "root", &bv->v, &prog, &err) >= 0);
        best_compile = std::min(best_compile, ms_since(tc));

        std::vector<double> masks, steps;
        for (const auto& toks : calls) {
            std::unique_ptr<GbnfGrammar> g;
            REQUIRE(GbnfGrammar::create(text, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
            size_t i = 0;
            while (i < toks.size()) {
                const auto ts = std::chrono::steady_clock::now();
                const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
                g->push_state();
                for (int k = 0; k < n_pos; ++k) {
                    if (k > 0) REQUIRE(g->accept_token(toks[i + (size_t)k - 1]) >= 0);
                    const auto tm = std::chrono::steady_clock::now();
                    REQUIRE(g->fill_mask(m.data(), n_words) >= 0);
                    masks.push_back(ms_since(tm));
                }
                g->rollback();
                const size_t n_acc = std::min<size_t>(3, toks.size() - i);
                for (size_t k = 0; k < n_acc; ++k) REQUIRE(g->accept_token(toks[i + k]) >= 0);
                steps.push_back(ms_since(ts));
                i += n_acc;
            }
        }
        if (best_mask.empty()) { best_mask = masks; best_step = steps; }
        for (size_t k = 0; k < masks.size() && k < best_mask.size(); ++k)
            best_mask[k] = std::min(best_mask[k], masks[k]);
        for (size_t k = 0; k < steps.size() && k < best_step.size(); ++k)
            best_step[k] = std::min(best_step[k], steps[k]);
    }

    std::vector<double> sm = best_mask;
    std::sort(sm.begin(), sm.end());
    const double p50 = sm[sm.size() / 2], p99 = sm[sm.size() * 99 / 100], worst = sm.back();
    double step_sum = 0.0;
    for (double x : best_step) step_sum += x;
    const double step_mean = step_sum / (double)best_step.size();
    std::fprintf(stderr, "    %d-token vocabulary: index %.0f ms; compile %.1f ms; %zu masks on a "
                 "fresh program: p50 %.3f p99 %.3f max %.3f ms; %zu steps, %.3f ms mean\n",
                 bv->v.n_tokens(), vi.build_ms, best_compile, sm.size(), p50, p99, worst,
                 best_step.size(), step_mean);
    CHECK(best_compile < 2000.0);
    CHECK(p99 < 0.25);
    CHECK(worst < 1.0);
    CHECK(step_mean < 0.3);
}

RAD_TEST_MAIN()
