/* tokenizer_fixture.h -- vocabularies and text for the tokenizer tests, made on the spot.
 *
 * The tests need no model file. A byte-level BPE vocabulary is TRAINED here, deterministically,
 * on generated text -- the textbook algorithm: count adjacent pairs over the pre-tokenised words,
 * merge the most frequent, repeat -- so it has the shape of a real one: every byte is a token,
 * common words are single tokens, rare ones take several merges, and some merges produce text
 * other merges build on. The generators draw from every general category, the scripts and
 * sequences that stress a pre-tokeniser (combining marks, ZWJ emoji, CJK, Hangul, RTL), digit runs
 * of every length, every kind of whitespace, contractions in mixed case, and optionally bytes that
 * are not UTF-8.
 */
#pragma once
#include "text/tokenizer.h"
#include "text/tokenizer_regex.h"
#include "text/tokenizer_ucd.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokfix {

using namespace rad;

/* The patterns real tokenizer.json files carry, in the spelling they carry them. */
inline const char* k_qwen35 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
inline const char* k_minicpm_digits = "\\p{N}{1,3}";
inline const char* k_minicpm =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
inline const char* k_llama3 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
inline const char* k_gpt2 =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/* The GPT-2 byte alphabet: byte -> its mapped character's UTF-8. */
inline std::string bytelevel_char(uint8_t b) {
    static std::vector<std::string> map = [] {
        std::vector<std::string> m(256);
        bool taken[256] = {};
        auto utf8 = [](uint32_t c) {
            std::string s;
            if (c < 0x80) s += (char)c;
            else { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
            return s;
        };
        for (int c = 0x21; c <= 0x7E; ++c) { m[c] = utf8(c); taken[c] = true; }
        for (int c = 0xA1; c <= 0xAC; ++c) { m[c] = utf8(c); taken[c] = true; }
        for (int c = 0xAE; c <= 0xFF; ++c) { m[c] = utf8(c); taken[c] = true; }
        uint32_t n = 0;
        for (int c = 0; c < 256; ++c) if (!taken[c]) m[c] = utf8(256 + n++);
        return m;
    }();
    return map[b];
}

inline std::string cp_utf8(uint32_t c) {
    std::string s;
    if (c < 0x80) { s += (char)c; }
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) {
        s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
    } else {
        s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F));
        s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
    }
    return s;
}

/* The general category of `c` from the compiler's own tables. */
inline uint32_t gc_of(uint32_t c) {
    size_t lo = 0, hi = ucd::k_gc_runs_n;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (ucd::k_gc_runs[mid].first <= c) lo = mid; else hi = mid;
    }
    return ucd::k_gc_runs[lo].gc;
}

/* Random text. `junk` adds bytes that are not UTF-8; `exclude(c)` removes codepoints the caller
 * does not want (the reference encoder's known disagreements, for one). */
class TextGen {
public:
    explicit TextGen(uint32_t seed) : rng_(seed) {
        std::vector<std::vector<uint32_t>> by_gc(ucd::kGcCount);
        for (size_t i = 0; i < ucd::k_gc_runs_n; ++i) {
            const uint32_t a = ucd::k_gc_runs[i].first;
            const uint32_t b = i + 1 < ucd::k_gc_runs_n ? ucd::k_gc_runs[i + 1].first : 0x110000;
            for (uint32_t c = a; c < b && c < a + 64; ++c) by_gc[ucd::k_gc_runs[i].gc].push_back(c);
        }
        by_gc_ = std::move(by_gc);
    }

    template <class Exclude>
    std::string text(size_t n_cps, bool junk, Exclude exclude) {
        std::string out;
        size_t n = 0;
        while (n < n_cps) {
            const int k = pick(100);
            std::string piece;
            if (k < 30)      piece = word();
            else if (k < 40) piece = pick_of(k_space);
            else if (k < 47) piece = digits();
            else if (k < 54) piece = contraction();
            else if (k < 62) piece = pick_of(k_sequences);
            else if (k < 70) piece = pick_of(k_punct);
            else if (k < 92) {
                const uint32_t g = pick((int)by_gc_.size());
                if (g == ucd::Cs || by_gc_[g].empty()) continue;
                const uint32_t c = by_gc_[g][pick((int)by_gc_[g].size())];
                if (exclude(c)) continue;
                piece = cp_utf8(c);
            } else if (junk) {
                piece = pick_of(k_junk);
            } else {
                piece = " ";
            }
            out += piece;
            ++n;
        }
        return out;
    }
    std::string text(size_t n_cps, bool junk = false) {
        return text(n_cps, junk, [](uint32_t) { return false; });
    }

    int pick(int n) { return (int)(rng_() % (uint32_t)n); }

private:
    std::mt19937 rng_;
    std::vector<std::vector<uint32_t>> by_gc_;

    template <size_t N> std::string pick_of(const char* const (&a)[N]) { return a[pick((int)N)]; }

    std::string word() {
        static const char* const words[] = {
            "the", "cache", "shard", "value", "func", "return", "struct", "Hello", "WORLD", "tokenizer",
            "naïve", "café", "Straße", "über", "мир", "привет", "κόσμος", "世界", "更新", "東京", "한국어",
            "مرحبا", "שלום", "नमस्ते", "สวัสดี", "ǅemal", "ﬁnance", "aaaa", "abab", "xyzzy", "Z",
        };
        std::string w = pick_of(words);
        if (pick(4) == 0) w = " " + w;
        return w;
    }
    std::string digits() {
        static const char* const sys[] = { "0123456789", "٠١٢٣٤٥٦٧٨٩", "०१२३४५६७८९" };
        const std::string s = sys[pick(8) == 0 ? 1 + pick(2) : 0];
        const size_t cp = s.size() / 10;
        std::string out;
        const int n = 1 + pick(pick(4) == 0 ? 25 : 7);
        for (int i = 0; i < n; ++i) out += s.substr((size_t)pick(10) * cp, cp);
        return out;
    }
    std::string contraction() {
        static const char* const base[] = { "it", "IT", "We", "they", "YOU", "i", "He" };
        static const char* const suf[] = { "'s", "'S", "'t", "'T", "'re", "'RE", "'Re", "'rE", "'ve",
                                           "'VE", "'m", "'M", "'ll", "'LL", "'lL", "'d", "'D", "'x", "'" };
        return std::string(pick_of(base)) + pick_of(suf);
    }

    static constexpr const char* k_space[] = {
        " ", " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r", " \n", "\t\t", "\n  ", "  \n ",
        "\xc2\xa0", "\xe3\x80\x80", "\xe2\x80\x80", "\xe2\x80\xa8", "\xe2\x80\xa9", "\xc2\x85",
        "\x0b", "\x0c", "\xe1\x9a\x80", "\xe2\x80\xaf",
    };
    static constexpr const char* k_sequences[] = {
        "e\xcc\x81", "a\xcc\x88\xcc\x81", "\xcc\x81", "\xcc\x81\xcc\x81",           /* combining marks */
        "\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7", /* ZWJ family */
        "\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd", "\xf0\x9f\x87\xaf\xf0\x9f\x87\xb5",     /* skin tone, flag */
        "\xe2\x9d\xa4\xef\xb8\x8f", "\xe2\x80\x8d", "\xe2\x80\x8b", "\xef\xbb\xbf",
        "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8",                                     /* Hangul jamo */
        "\xed\x95\x9c\xea\xb8\x80", "\xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb7",          /* Hangul, Devanagari */
        "\xd7\xa9\xd7\x9c\xd7\x95\xd7\x9d", "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7", /* RTL */
        "\xe2\x80\x8f", "\xe2\x80\xae", "\xef\xbc\xa1\xef\xbc\x91",                 /* marks, fullwidth */
        "\xc5\xbf", "'\xc5\xbf",                                                    /* long s */
    };
    static constexpr const char* k_punct[] = {
        ".", ",", "!", "?", "...", "->", "::", "();", "{}", "[]", "//", "/*", "*/", "#", "@", "$",
        "%", "^", "&", "*", "_", "=", "+", "-", "`", "~", "|", "\\", "\"", "<", ">", "'", "\xe2\x80\x94",
        "\xe2\x80\x9c", "\xe2\x80\x9d", "\xc2\xab", "\xe3\x80\x82", "\xc2\xbd", "\xc2\xb2", "\xe2\x91\xa0",
    };
    static constexpr const char* k_junk[] = {
        "\xc3", "\xe4\xb8", "\xf0\x9f\x98", "\x80", "\xbf\xbf", "\xc0\x80", "\xc1\xbf", "\xe0\x80\x80",
        "\xed\xa0\x80", "\xed\xbf\xbf", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80", "\xf8\x88\x80\x80\x80",
        "\xfe", "\xff", "\xf0\x80\x80\x80",
    };
};

/* A byte-level BPE vocabulary trained on `corpus` pre-tokenised by `pattern`: 256 byte tokens,
 * then one token per merge, `n_merges` of them (fewer if the pairs run out). Ties between equally
 * frequent pairs go to the smaller (left id, right id), so the result is a function of the
 * inputs. */
inline VocabBuild train_bytelevel(const std::string& corpus, const char* pattern, int n_merges) {
    VocabBuild vb;
    vb.kind = RAD_TOK_BPE;
    for (int b = 0; b < 256; ++b) vb.tokens.push_back(bytelevel_char((uint8_t)b));

    std::unique_ptr<TokRegex> re;
    std::string why;
    TokRegex::compile(pattern, &re, &why);
    std::string canon;
    std::string_view text = corpus;
    if (!utf8_is_canonical(text)) { utf8_canonicalize(text, canon); text = canon; }
    std::vector<TokSpan> pieces, scratch;
    tok_split(*re, text, TOK_SPLIT_ISOLATED, false, pieces, scratch);
    std::map<std::string, int> counts;
    for (const TokSpan& p : pieces) ++counts[std::string(text.substr(p.begin, p.end - p.begin))];

    std::vector<std::pair<std::vector<int32_t>, int>> words;
    for (const auto& [w, c] : counts) {
        std::vector<int32_t> ids;
        for (unsigned char b : w) ids.push_back(b);
        words.push_back({ ids, c });
    }
    std::unordered_map<uint64_t, long> pairs;
    for (int m = 0; m < n_merges; ++m) {
        pairs.clear();
        for (const auto& [ids, c] : words)
            for (size_t i = 0; i + 1 < ids.size(); ++i)
                pairs[((uint64_t)(uint32_t)ids[i] << 32) | (uint32_t)ids[i + 1]] += c;
        if (pairs.empty()) break;
        uint64_t best = 0;
        long best_n = -1;
        for (const auto& [k, n] : pairs)
            if (n > best_n || (n == best_n && k < best)) { best = k; best_n = n; }
        const int32_t l = (int32_t)(best >> 32), r = (int32_t)(best & 0xFFFFFFFF);
        const int32_t id = (int32_t)vb.tokens.size();
        vb.tokens.push_back(vb.tokens[l] + vb.tokens[r]);
        vb.merges.push_back({ (uint32_t)l, (uint32_t)r });
        for (auto& [ids, c] : words) {
            std::vector<int32_t> next;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i + 1 < ids.size() && ids[i] == l && ids[i + 1] == r) { next.push_back(id); ++i; }
                else next.push_back(ids[i]);
            }
            ids.swap(next);
        }
    }
    vb.types.assign(vb.tokens.size(), (uint8_t)RAD_TT_NORMAL);
    return vb;
}

inline VocabStep split_step(const char* pattern) {
    VocabStep s;
    s.kind = RAD_PRE_SPLIT;
    s.arg0 = pattern;
    s.iarg = TOK_SPLIT_ISOLATED;
    return s;
}
inline VocabStep bytelevel_step(bool prefix = false, bool regex = false) {
    VocabStep s;
    s.kind = RAD_PRE_BYTELEVEL;
    s.flags = (prefix ? 1u : 0u) | (regex ? 2u : 0u);
    return s;
}
inline VocabStep simple_step(uint32_t kind) {
    VocabStep s;
    s.kind = kind;
    return s;
}

/* Add control tokens to a vocabulary. */
inline void add_specials(VocabBuild& vb, const std::vector<std::pair<std::string, uint8_t>>& specials) {
    for (const auto& [text, type] : specials) {
        vb.tokens.push_back(text);
        vb.types.push_back(type);
    }
}

inline std::shared_ptr<const Vocab> load(VocabBuild vb) {
    auto v = std::make_shared<Vocab>();
    if (v->load(std::move(vb)) < 0) return nullptr;
    return v;
}

}  /* namespace tokfix */
