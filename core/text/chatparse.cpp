#include "text/chatparse.h"
#include "text/chatfmt.h"

/* Bind time only: the schema resolution that decides which arguments are strings. The same
 * function the grammar generator uses, so the parser and the grammar never disagree about a
 * type. Nothing from vendor/ runs while a reply is being read. */
#include "llama/common/json-schema-to-grammar.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <deque>
#include <set>
#include <unordered_map>

namespace rad {

using ojson = nlohmann::ordered_json;

namespace {

/* The whitespace set std::isspace accepts in the C locale, which is the set the format's grammar
 * skips between tags. */
inline bool is_ws(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

/* The whitespace a reasoning block may consist of and still count as empty. */
inline bool is_blank(unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

/* ================================================================== delimiter automaton
 *
 * A deterministic Aho-Corasick automaton over a handful of delimiters. The state is the longest
 * suffix of the text scanned so far that is a prefix of some delimiter, so its depth is exactly
 * how many trailing bytes might still become one -- the bytes a streaming scan must hold -- and a
 * state whose string IS a delimiter is a match.
 *
 * That last step is exact only because chat_format_check refuses delimiter sets in which one
 * delimiter occurs inside another at a non-zero offset. Then a match ends at the first byte where
 * any delimiter completes, and no earlier-starting match can still be in progress: the automaton
 * reports matches in the order of their START, which is the order the grammar reads them in. */
class Dfa {
public:
    void build(const std::vector<std::string>& pats) {
        pats_ = pats;
        struct Node { int next[256]; int depth; int word; };
        std::vector<Node> t(1);
        std::fill(std::begin(t[0].next), std::end(t[0].next), -1);
        t[0].depth = 0; t[0].word = -1;
        for (size_t i = 0; i < pats.size(); ++i) {
            int cur = 0;
            for (unsigned char c : pats[i]) {
                if (t[(size_t)cur].next[c] < 0) {
                    Node n;
                    std::fill(std::begin(n.next), std::end(n.next), -1);
                    n.depth = t[(size_t)cur].depth + 1;
                    n.word  = -1;
                    t.push_back(n);
                    t[(size_t)cur].next[c] = (int)t.size() - 1;
                }
                cur = t[(size_t)cur].next[c];
            }
            if (cur != 0 && t[(size_t)cur].word < 0) t[(size_t)cur].word = (int)i;
        }
        const size_t n = t.size();
        tr_.assign(n * 256, 0);
        depth_.assign(n, 0);
        word_.assign(n, -1);
        std::vector<int> fail(n, 0), order;
        order.reserve(n);
        for (int c = 0; c < 256; ++c) {
            const int v = t[0].next[c];
            if (v > 0) { fail[(size_t)v] = 0; tr_[(size_t)c] = (uint16_t)v; order.push_back(v); }
        }
        for (size_t k = 0; k < order.size(); ++k) {
            const int u = order[k];
            for (int c = 0; c < 256; ++c) {
                const int v = t[(size_t)u].next[c];
                const uint16_t via = tr_[(size_t)fail[(size_t)u] * 256 + (size_t)c];
                if (v > 0) {
                    fail[(size_t)v] = via;
                    tr_[(size_t)u * 256 + (size_t)c] = (uint16_t)v;
                    order.push_back(v);
                } else {
                    tr_[(size_t)u * 256 + (size_t)c] = via;
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            depth_[i] = (uint16_t)t[i].depth;
            word_[i]  = (int16_t)t[i].word;
        }
        std::fill(std::begin(first_), std::end(first_), false);
        n_first_ = 0;
        for (const std::string& p : pats)
            if (!p.empty() && !first_[(unsigned char)p[0]]) {
                first_[(unsigned char)p[0]] = true;
                only_ = (unsigned char)p[0];
                ++n_first_;
            }
    }

    uint16_t step(uint16_t s, unsigned char c) const { return tr_[(size_t)s * 256 + c]; }
    size_t   depth(uint16_t s) const { return depth_[s]; }
    int      word(uint16_t s) const { return word_[s]; }
    const std::string& pattern(int i) const { return pats_[(size_t)i]; }

    /* The first byte in [p, e) that can begin a delimiter. Plain text between delimiters is
     * skipped here without running the automaton at all. */
    const char* skip(const char* p, const char* e) const {
        if (n_first_ == 1) {
            const void* q = std::memchr(p, only_, (size_t)(e - p));
            return q ? (const char*)q : e;
        }
        if (n_first_ == 0) return e;
        while (p < e && !first_[(unsigned char)*p]) ++p;
        return p;
    }

private:
    std::vector<std::string> pats_;
    std::vector<uint16_t>    tr_;
    std::vector<uint16_t>    depth_;
    std::vector<int16_t>     word_;
    bool                     first_[256] = {};
    int                      n_first_ = 0;
    unsigned char            only_ = 0;
};

/* ================================================================== JSON
 *
 * A JSON value, recognised one byte at a time with an explicit stack. It accepts exactly what the
 * format's grammar accepts as a JSON-typed argument, quirks included, because the answer decides
 * whether the argument is typed or text:
 *
 *   * no whitespace before the value, any whitespace after it and inside it;
 *   * a number may not be followed by a character that could have continued it -- `01`, `1.`,
 *     `1-` are not numbers -- while `true` and `null` may be followed by anything;
 *   * a string may hold raw control characters, and any escape but the JSON ones fails it.
 *
 * push() answers More while the bytes are still a JSON value or its trailing whitespace, Fail at
 * the first byte that cannot be, and Done at the first byte after a complete value -- which is
 * NOT consumed. */
class JsonScan {
public:
    enum Res : uint8_t { More, Fail, Done };

    void reset() { st_ = Value; stack_.clear(); lit_ = nullptr; lpos_ = 0; need_ = 0; }

    Res push(unsigned char c) {
        for (;;) {
            switch (st_) {
            case Value:
                return start_value(c);
            case ValueWs:
                if (is_ws(c)) return More;
                st_ = Value;
                continue;
            case Str:
                if (need_ > 0) {
                    if ((c & 0xC0) != 0x80) return Fail;
                    --need_;
                    return More;
                }
                if (c == '"') { st_ = key_ ? KeyAfter : After; return More; }
                if (c == '\\') { st_ = StrEsc; return More; }
                if (c < 0x80) return More;
                if (!(c & 0x40)) return Fail;
                if (!(c & 0x20)) { need_ = 1; return More; }
                if (!(c & 0x10)) { need_ = 2; return More; }
                if (!(c & 0x08)) { need_ = 3; return More; }
                return Fail;
            case StrEsc:
                if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                    c == 'r' || c == 't') { st_ = Str; return More; }
                if (c == 'u') { st_ = StrHex; need_ = 4; return More; }
                return Fail;
            case StrHex:
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                    return Fail;
                if (--need_ == 0) st_ = Str;
                return More;
            case NumMinus:
                if (c == '0') { st_ = NumZero; return More; }
                if (c >= '1' && c <= '9') { st_ = NumInt; return More; }
                return Fail;
            case NumZero:
                if (c == '.') { st_ = NumDot; return More; }
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumInt:
                if (c >= '0' && c <= '9') return More;
                if (c == '.') { st_ = NumDot; return More; }
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumDot:
                if (c >= '0' && c <= '9') { st_ = NumFrac; return More; }
                return Fail;
            case NumFrac:
                if (c >= '0' && c <= '9') return More;
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumE:
                if (c == '+' || c == '-') { st_ = NumSign; return More; }
                if (c >= '0' && c <= '9') { st_ = NumExp; return More; }
                return Fail;
            case NumSign:
                if (c >= '0' && c <= '9') { st_ = NumExp; return More; }
                return Fail;
            case NumExp:
                if (c >= '0' && c <= '9') return More;
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case Lit:
                if ((unsigned char)lit_[lpos_] != c) return Fail;
                if (lit_[++lpos_] == '\0') st_ = After;
                return More;
            case After:
                if (is_ws(c)) return More;
                if (stack_.empty()) return Done;
                if (stack_.back() == '{') {
                    if (c == ',') { st_ = ObjKeyWs; return More; }
                    if (c == '}') { stack_.pop_back(); st_ = After; return More; }
                    return Fail;
                }
                if (c == ',') { st_ = ValueWs; return More; }
                if (c == ']') { stack_.pop_back(); st_ = After; return More; }
                return Fail;
            case ObjOpen:
                if (is_ws(c)) return More;
                if (c == '}') { stack_.pop_back(); st_ = After; return More; }
                if (c == '"') { st_ = Str; key_ = true; need_ = 0; return More; }
                return Fail;
            case ObjKeyWs:
                if (is_ws(c)) return More;
                if (c == '"') { st_ = Str; key_ = true; need_ = 0; return More; }
                return Fail;
            case KeyAfter:
                if (is_ws(c)) return More;
                if (c == ':') { st_ = ValueWs; return More; }
                return Fail;
            case ArrOpen:
                if (is_ws(c)) return More;
                if (c == ']') { stack_.pop_back(); st_ = After; return More; }
                st_ = Value;
                continue;
            }
            return Fail;
        }
    }

private:
    enum St : uint8_t {
        Value, ValueWs, Str, StrEsc, StrHex, NumMinus, NumZero, NumInt, NumDot, NumFrac, NumE,
        NumSign, NumExp, Lit, After, ObjOpen, ObjKeyWs, KeyAfter, ArrOpen,
    };
    /* Deeper than any schema describes. The recognizer does not recurse, so this bounds memory
     * rather than the stack; a value nested past it is read as text. */
    static constexpr size_t kMaxDepth = 1u << 16;

    static bool num_continues(unsigned char c) {
        return (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
    }

    Res start_value(unsigned char c) {
        switch (c) {
        case '{':
            if (stack_.size() >= kMaxDepth) return Fail;
            stack_.push_back('{'); st_ = ObjOpen; return More;
        case '[':
            if (stack_.size() >= kMaxDepth) return Fail;
            stack_.push_back('['); st_ = ArrOpen; return More;
        case '"': st_ = Str; key_ = false; need_ = 0; return More;
        case '-': st_ = NumMinus; return More;
        case '0': st_ = NumZero; return More;
        case 't': st_ = Lit; lit_ = "true";  lpos_ = 1; return More;
        case 'f': st_ = Lit; lit_ = "false"; lpos_ = 1; return More;
        case 'n': st_ = Lit; lit_ = "null";  lpos_ = 1; return More;
        default:
            if (c >= '1' && c <= '9') { st_ = NumInt; return More; }
            return Fail;
        }
    }

    St                st_ = Value;
    std::vector<char> stack_;
    const char*       lit_ = nullptr;
    size_t            lpos_ = 0;
    int               need_ = 0;
    bool              key_ = false;
};

/* ================================================================== JSON text helpers */

/* A string's JSON encoding without its quotes, byte for byte what nlohmann::json::dump writes:
 * the two-character escapes where JSON has one, \u00XX in lower-case hex for the other control
 * characters, and everything else -- DEL and UTF-8 included -- verbatim. */
void json_escape_into(std::string& out, const char* p, size_t n) {
    static const char hex[] = "0123456789abcdef";
    const char* run = p;
    const char* e = p + n;
    for (; p < e; ++p) {
        const unsigned char c = (unsigned char)*p;
        if (c >= 0x20 && c != '"' && c != '\\') continue;
        out.append(run, (size_t)(p - run));
        run = p + 1;
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    out.append(run, (size_t)(e - run));
}

std::string json_quote(std::string_view s) {
    std::string out = "\"";
    json_escape_into(out, s.data(), s.size());
    out += '"';
    return out;
}

/* How many trailing bytes of `s` from `from` on form an unfinished UTF-8 sequence. A delta
 * becomes a JSON string on the wire, and half a code point there is a malformed payload. */
size_t utf8_tail(const std::string& s, size_t from) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n - from; ++back) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        return back < need ? back : 0;
    }
    return 0;
}

/* The value transforms the format's grammar applies, on a whole value. Used for values short
 * enough to be held entire and for the held ends of long ones. */
std::string_view strip_leading_newline(std::string_view v) {
    if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
    else if (v.size() >= 2 && v[0] == '\r' && v[1] == '\n') v.remove_prefix(2);
    return v;
}

std::string_view strip_trailing_newline(std::string_view v) {
    if (!v.empty() && v.back() == '\n') {
        v.remove_suffix(1);
        if (!v.empty() && v.back() == '\r') v.remove_suffix(1);
    }
    return v;
}

constexpr std::string_view kCdOpen  = "<![CDATA[";
constexpr std::string_view kCdClose = "]]>";
/* A response-format document may be fenced as a Markdown code block. */
constexpr std::string_view kFenceOpen  = "```json";
constexpr std::string_view kFenceClose = "```";

/* A CDATA section is punctuation, not content, and its two ends are stripped independently. A
 * value still arriving holds back what could yet become the opener, and a trailing `]` or `]]`
 * that could be the start of the closer. */
void unwrap_cdata(std::string_view& v, bool partial) {
    if (v.size() < kCdOpen.size()) {
        if (partial && !v.empty() && kCdOpen.compare(0, v.size(), v) == 0) v = std::string_view();
        return;
    }
    if (v.compare(0, kCdOpen.size(), kCdOpen) != 0) return;
    v.remove_prefix(kCdOpen.size());
    if (v.size() >= kCdClose.size() &&
        v.compare(v.size() - kCdClose.size(), kCdClose.size(), kCdClose) == 0) {
        v.remove_suffix(kCdClose.size());
        return;
    }
    if (!partial) return;
    size_t hold = 0;
    while (hold + 1 < kCdClose.size() && v.size() > hold && v[v.size() - 1 - hold] == ']') ++hold;
    v.remove_suffix(hold);
}

/* The name of an argument, as the grammar reads it: one leading whitespace character and every
 * trailing one are not part of it. */
std::string_view trim_arg_name(std::string_view v) {
    if (!v.empty() && is_ws((unsigned char)v.front())) v.remove_prefix(1);
    while (!v.empty() && is_ws((unsigned char)v.back())) v.remove_suffix(1);
    return v;
}

std::string_view trim_trailing_ws(std::string_view v) {
    while (!v.empty() && is_ws((unsigned char)v.back())) v.remove_suffix(1);
    return v;
}

/* ================================================================== an argument value
 *
 * The bytes of one value as the delimiters bound it, turned into its JSON encoding in the
 * arguments object. Two encodings: a STRING (quoted, escaped) and RAW JSON text, which is either
 * a value that parsed as JSON or a non-string argument that did not and is passed through --
 * with a Python-style container's single quotes normalised to double ones.
 *
 * The transforms touch only the ends of a value -- one newline at each end, the CDATA opener and
 * closer -- so the middle streams straight through and only a short head and tail are held: the
 * first kHead bytes until the head is decided, and the last kTail after that. */
class ValueOut {
public:
    enum Mode : uint8_t { String, Raw };
    static constexpr size_t kHead = 16;   /* a newline pair and the CDATA opener, and the tail */
    static constexpr size_t kTail = 5;    /* a newline pair and the CDATA closer */

    void begin(Mode m, bool cdata, bool newlines, std::string& out) {
        mode_ = m; cdata_ = cdata; newlines_ = newlines;
        buf_.clear(); head_ = false; opened_ = false;
        raw_started_ = false; container_ = false;
        in_single_ = in_double_ = bs_ = false;
        if (mode_ == String) out += '"';
    }

    void add(const char* p, size_t n, std::string& out) {
        buf_.append(p, n);
        if (!head_) {
            if (buf_.size() < kHead) return;
            std::string_view v(buf_);
            size_t drop = 0;
            if (newlines_) drop = v.size() - strip_leading_newline(v).size();
            if (cdata_ && v.compare(drop, kCdOpen.size(), kCdOpen) == 0) {
                opened_ = true;
                drop += kCdOpen.size();
            }
            buf_.erase(0, drop);
            head_ = true;
        }
        if (buf_.size() > kTail) {
            const size_t k = buf_.size() - kTail;
            emit(buf_.data(), k, out);
            buf_.erase(0, k);
        }
    }

    void finish(bool partial, std::string& out) {
        std::string_view v(buf_);
        if (!head_) {
            if (newlines_) v = strip_trailing_newline(strip_leading_newline(v));
            if (cdata_) unwrap_cdata(v, partial);
        } else {
            if (newlines_) v = strip_trailing_newline(v);
            if (opened_) {
                if (v.size() >= kCdClose.size() &&
                    v.compare(v.size() - kCdClose.size(), kCdClose.size(), kCdClose) == 0) {
                    v.remove_suffix(kCdClose.size());
                } else if (partial) {
                    size_t hold = 0;
                    while (hold + 1 < kCdClose.size() && v.size() > hold &&
                           v[v.size() - 1 - hold] == ']')
                        ++hold;
                    v.remove_suffix(hold);
                }
            }
        }
        emit(v.data(), v.size(), out);
        if (bs_) { out += '\\'; bs_ = false; }
        buf_.clear();
    }

    size_t held() const { return buf_.size() + (bs_ ? 1 : 0); }

private:
    void emit(const char* p, size_t n, std::string& out) {
        if (n == 0) return;
        if (mode_ == String) { json_escape_into(out, p, n); return; }
        if (!raw_started_) {
            raw_started_ = true;
            container_ = p[0] == '[' || p[0] == '{';
        }
        if (!container_) { out.append(p, n); return; }
        for (size_t i = 0; i < n; ++i) normalise(p[i], out);
    }

    /* Python-style quoting to JSON, one character at a time: a single-quoted string becomes a
     * double-quoted one, with the quotes inside it re-escaped to match. A backslash is held until
     * the character after it arrives, since the pair is read together. */
    void normalise(char c, std::string& out) {
        if (bs_) {
            bs_ = false;
            if (in_single_) {
                if (c == '\'') { out += '\''; return; }
                if (c == '"')  { out += "\\\""; return; }
                out += '\\'; out += c; return;
            }
            if (in_double_) { out += '\\'; out += c; return; }
            out += '\\';
            /* Outside a string the backslash stands alone and the next character is read
             * normally. */
        } else if (c == '\\') {
            bs_ = true;
            return;
        }
        if (c == '"') {
            if (in_single_) out += "\\\"";
            else { in_double_ = !in_double_; out += '"'; }
        } else if (c == '\'') {
            if (in_double_) out += '\'';
            else if (in_single_) { in_single_ = false; out += '"'; }
            else { in_single_ = true; out += '"'; }
        } else {
            out += c;
        }
    }

    Mode        mode_ = String;
    bool        cdata_ = true, newlines_ = true;
    std::string buf_;
    bool        head_ = false, opened_ = false;
    bool        raw_started_ = false, container_ = false;
    bool        in_single_ = false, in_double_ = false, bs_ = false;
};

/* ================================================================== the answer text
 *
 * Content is trimmed at the end: trailing spaces and newlines, then one turn terminator. Trimming
 * the END of a stream means holding what might be the end -- the trailing whitespace, and a
 * terminator or the part of one that has arrived -- until something that is not whitespace
 * follows it. The terminator automaton tracks the longest suffix that could still become one. */
class ContentOut {
public:
    void init(const Dfa* terms, const std::vector<std::string>* list) { terms_ = terms; list_ = list; }

    void add(const char* p, size_t n, std::string& full) {
        for (size_t i = 0; i < n; ++i) {
            const unsigned char c = (unsigned char)p[i];
            if (c == ' ' || c == '\n') {
                if (ws_ == 0)
                    term_ = terms_ && terms_->word(ts_) >= 0 ? terms_->depth(ts_) : 0;
                ++ws_;
            } else {
                ws_ = 0;
            }
            if (terms_) ts_ = terms_->step(ts_, c);
        }
        pend_.append(p, n);
        const size_t hold = ws_ ? ws_ + term_ : (terms_ ? terms_->depth(ts_) : 0);
        if (pend_.size() > hold) {
            full.append(pend_, 0, pend_.size() - hold);
            pend_.erase(0, pend_.size() - hold);
        }
    }

    /* The exact trim: trailing spaces and newlines, then the first terminator in the list that
     * ends what is left. */
    void finish(std::string& full) {
        while (!pend_.empty() && (pend_.back() == '\n' || pend_.back() == ' ')) pend_.pop_back();
        if (list_)
            for (const std::string& t : *list_)
                if (!t.empty() && pend_.size() >= t.size() &&
                    pend_.compare(pend_.size() - t.size(), t.size(), t) == 0) {
                    pend_.erase(pend_.size() - t.size());
                    break;
                }
        full += pend_;
        pend_.clear();
    }

    size_t held_space() const { return std::min(ws_, pend_.size()); }
    size_t held_other() const { return pend_.size() - held_space(); }

private:
    const Dfa*                      terms_ = nullptr;
    const std::vector<std::string>* list_ = nullptr;
    std::string                     pend_;
    size_t                          ws_ = 0, term_ = 0;
    uint16_t                        ts_ = 0;
};

}  /* namespace */

/* ================================================================== the compiled format */

struct ReplyFormat::Impl {
    ChatFormat  f;
    /* Each tag split into its core and the whitespace around it that is optional to match. */
    std::string open_core, open_trail, close_core, close_trail, ns_core, ns_trail, end_core,
                end_trail, co_core, co_trail;
    /* What begins a call in content. The first is the format's own opener: call_open's core, or
     * name_prefix when calls are not wrapped. A wrapped format also takes name_prefix alone,
     * second: a model writes the call without its wrapper often enough that its calls must not
     * be lost to it, and a wrapper written without its trailing whitespace is read too. */
    std::vector<std::string> openers;
    std::string name_end;     /* what ends an argument name */
    /* The same two, whitespace-trimmed. A loose call is one the grammar never opened, so the
     * model wrote it unconstrained, and its argument tags are read with the whitespace around
     * them optional -- as its name and its close are. */
    std::string name_end_core, value_core;
    Dfa think, think_breaks, text, arg_name, value, junk, arg_name_loose, value_loose;
    size_t longest = 0;       /* the longest delimiter */
};

ReplyFormat::ReplyFormat() : impl_(std::make_unique<Impl>()) {}
ReplyFormat::~ReplyFormat() = default;
const ChatFormat& ReplyFormat::format() const { return impl_->f; }

static void split_ws(const std::string& tag, std::string* core, std::string* trail) {
    size_t a = 0, b = tag.size();
    while (a < b && is_ws((unsigned char)tag[a])) ++a;
    while (b > a && is_ws((unsigned char)tag[b - 1])) --b;
    *core = tag.substr(a, b - a);
    *trail = tag.substr(b);
}

int ReplyFormat::compile(const ChatFormat& f, std::shared_ptr<const ReplyFormat>* out,
                         std::string* why) {
    const int st = chat_format_check(f, why);
    if (st < 0) return st;
    std::shared_ptr<ReplyFormat> r(new ReplyFormat());
    Impl& m = *r->impl_;
    m.f = f;
    split_ws(f.reasoning_open, &m.open_core, &m.open_trail);
    split_ws(f.reasoning_close, &m.close_core, &m.close_trail);
    split_ws(f.name_suffix, &m.ns_core, &m.ns_trail);
    split_ws(f.call_end, &m.end_core, &m.end_trail);
    split_ws(f.call_open, &m.co_core, &m.co_trail);
    if (f.call_open.empty()) m.openers = { f.name_prefix };
    else m.openers = { m.co_core, f.name_prefix };
    m.name_end = f.arg_name_suffix.empty() ? f.arg_value_prefix : f.arg_name_suffix;

    if (f.has_reasoning()) {
        m.think.build({ m.close_core });
        std::vector<std::string> b{ m.close_core };
        for (const std::string& s : f.reasoning_breaks) b.push_back(s);
        m.think_breaks.build(b);
    }
    if (f.has_tools()) {
        m.text.build(m.openers);
        m.arg_name.build({ m.name_end });
        m.value.build({ f.arg_value_suffix });
        std::string trail;
        split_ws(m.name_end, &m.name_end_core, &trail);
        split_ws(f.arg_value_suffix, &m.value_core, &trail);
        m.arg_name_loose.build({ m.name_end_core });
        m.value_loose.build({ m.value_core });
        m.junk.build({ f.arg_prefix, m.end_core });
    }
    const std::string* const delims[] = { &f.reasoning_open, &f.reasoning_close, &f.call_open,
                                          &f.call_close, &f.name_prefix, &f.name_suffix,
                                          &f.call_end, &f.arg_prefix, &m.name_end,
                                          &f.arg_value_suffix };
    for (const std::string* s : delims) m.longest = std::max(m.longest, s->size());
    for (const std::string& s : f.reasoning_breaks) m.longest = std::max(m.longest, s.size());
    *out = std::move(r);
    return RAD_OK;
}

/* ================================================================== the bound request */

namespace {
struct Param {
    bool typed = false;       /* read as JSON when it parses as JSON */
};
struct Tool {
    std::string name;
    std::unordered_map<std::string, Param> params;
    /* The declared names in the order the grammar tries them: required ones, then the rest,
     * each in schema order. */
    std::vector<std::string> arms;
    bool has_props = false;
    bool has_required = false;
};
}  /* namespace */

struct ReplyParser::Impl {
    std::shared_ptr<const ReplyFormat> fmt;
    ReplyOptions opt;
    std::string  tail;             /* the part of the generation prompt the reply continues */
    bool         calls = false;    /* the request sent tools and the format has calls */
    bool         breaks = false;   /* an unclosed reasoning block ends at a call */
    bool         pure = false;     /* no reasoning, calls or document: every byte is content */
    std::vector<Tool> tools;
    std::unordered_map<std::string, size_t> by_name;
    std::vector<std::string> terminators;
    Dfa          terms;
};

ReplyParser::ReplyParser() : impl_(std::make_unique<Impl>()) {}
ReplyParser::~ReplyParser() = default;
const ChatFormat& ReplyParser::format() const { return impl_->fmt->format(); }

size_t ReplyParser::hold_bound() const {
    size_t term = 0;
    for (const std::string& t : impl_->terminators) term = std::max(term, t.size());
    return impl_->fmt->impl().longest + std::max(term, kFenceOpen.size()) + ValueOut::kHead + 1 +
           3 * 3;
}

/* Read the tools the way the grammar generator reads them, so a name or a type means the same
 * thing to the parser as to the grammar that constrained the reply. */
static int bind_tools(const ReplyFormat::Impl& F, const ojson& tools, ReplyParser::Impl* P,
                      std::string* why) {
    for (const auto& tool : tools) {
        if (!tool.is_object() || !tool.contains("type") || tool.at("type") != "function" ||
            !tool.contains("function"))
            continue;
        try {
            const auto& func = tool.at("function");
            Tool t;
            t.name = func.at("name").get<std::string>();
            auto params = func.contains("parameters") ? func.at("parameters") : ojson::object();
            const auto properties = params.contains("properties") ? params.at("properties")
                                                                 : ojson::object();
            std::set<std::string> required;
            if (params.contains("required")) params.at("required").get_to(required);
            common_schema_info info;
            info.resolve_refs(params);
            std::vector<std::string> optional;
            for (const auto& [pname, pschema] : properties.items()) {
                Param p;
                p.typed = F.f.other_value == ChatValue::Json && !info.resolves_to_string(pschema);
                t.has_props = true;
                if (required.count(pname)) {
                    t.has_required = true;
                    t.arms.push_back(pname);
                } else {
                    optional.push_back(pname);
                }
                /* A name the argument's own delimiter could end early would be read as a
                 * different name. */
                const std::string probe = pname + F.name_end;
                if (probe.find(F.name_end) != pname.size()) {
                    *why = "parameter '" + pname + "' of tool '" + t.name + "' contains '" +
                           F.name_end + "', which this model's tool-call format uses to end an "
                           "argument name";
                    return RAD_E_INVAL;
                }
                t.params.emplace(pname, p);
            }
            t.arms.insert(t.arms.end(), optional.begin(), optional.end());
            if (t.name.find(F.ns_core) != std::string::npos ||
                (t.name + F.f.name_suffix).find(F.f.name_suffix) != t.name.size()) {
                *why = "tool '" + t.name + "' has a name containing '" + F.ns_core +
                       "', which this model's tool-call format uses to end a function name";
                return RAD_E_INVAL;
            }
            if (!P->by_name.count(t.name)) P->by_name.emplace(t.name, P->tools.size());
            P->tools.push_back(std::move(t));
        } catch (const std::exception& e) {
            *why = std::string("a tool definition could not be read: ") + e.what();
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

int ReplyParser::create(std::shared_ptr<const ReplyFormat> fmt, const ReplyOptions& opt,
                        std::shared_ptr<const ReplyParser>* out, std::string* why) {
    std::string dummy;
    if (!why) why = &dummy;
    if (!fmt) { *why = "no reply format"; return RAD_E_INVAL; }
    std::shared_ptr<ReplyParser> r(new ReplyParser());
    Impl& P = *r->impl_;
    const ReplyFormat::Impl& F = fmt->impl();
    P.fmt = fmt;
    P.opt = opt;
    P.opt.tools = nullptr;

    const bool want_calls = opt.tools && opt.tools->is_array() && !opt.tools->empty();
    if (want_calls && !F.f.has_tools()) {
        *why = "this model's reply format has no tool calls";
        return RAD_E_UNSUPPORTED;
    }
    if (want_calls) {
        const int st = bind_tools(F, *opt.tools, &P, why);
        if (st < 0) return st;
    }
    /* A response format replaces the call grammar outright; the calls are not parsed. */
    P.calls  = want_calls && !opt.response_format;
    P.breaks = want_calls && F.f.has_reasoning() && !F.f.reasoning_breaks.empty();
    P.pure   = !F.f.has_reasoning() && !want_calls && !opt.response_format;

    if (F.f.has_reasoning()) {
        switch (F.f.reasoning_in_prompt) {
        case ChatThink::FromPrompt:
            if (!F.f.reasoning_open.empty() && !opt.generation_prompt.empty()) {
                const size_t at = opt.generation_prompt.find(F.f.reasoning_open);
                if (at != std::string::npos) P.tail = opt.generation_prompt.substr(at);
            }
            break;
        case ChatThink::Always: P.tail = F.f.reasoning_open; break;
        case ChatThink::Never:  break;
        }
    }
    for (const std::string& t : opt.terminators)
        if (!t.empty()) P.terminators.push_back(t);
    P.terms.build(P.terminators);
    *out = std::move(r);
    return RAD_OK;
}

/* ================================================================== the stream */

namespace {
enum class St : uint8_t {
    LeadWs,       /* whitespace before the reply proper: dropped */
    ThinkOpen,    /* the core of reasoning_open */
    ThinkOpenWs,  /* the optional whitespace that ends reasoning_open */
    Think,        /* reasoning text */
    ThinkCloseWs, /* the optional whitespace that ends reasoning_close */
    Text,         /* content, watching for the call opener when calls are on */
    OpenCoTail,   /* the whitespace that ends call_open, after its core */
    OpenWs,       /* after call_open: whitespace */
    OpenPrefix,   /* name_prefix, after call_open */
    OpenName,     /* the function name, to the core of name_suffix */
    OpenTail,     /* the whitespace that ends name_suffix */
    OpenPeek,     /* a loose call's name is read: whitespace, then an argument or the close */
    OpenPeekArg,  /* ... and the first argument's name, which must close as a name */
    Body,         /* inside a call: whitespace, then an argument or the call's end */
    BodyLit,      /* arg_prefix or call_end */
    ArgName,      /* an argument name, to its end */
    ValText,      /* a string (or raw) value, to arg_value_suffix */
    ValJson,      /* a JSON-typed value */
    ValJsonEnd,   /* arg_value_suffix, exactly, after a complete JSON value */
    EndTail,      /* the whitespace that ends call_end */
    Junk,         /* text inside a call that is no argument: content */
    AfterEnd,     /* after call_end: whitespace, then call_close */
    AfterEndLit,  /* call_close */
    AfterCall,    /* after a call: whitespace, dropped */
    RfWs,         /* before the document: whitespace, dropped */
    RfFence,      /* "```json" */
    RfFenceWs,    /* whitespace after the fence, dropped */
    RfJson,       /* the document */
    RfClose,      /* "```" */
    RfEnd,        /* after the closing fence */
    RfRaw,        /* a reply that is not the document it was asked for: content */
};

/* A function name the grammar does not know is still read as a call, up to this long. */
constexpr size_t kMaxLooseName = 1024;
/* And what a loose opener may hold in all before it is known to be a call. */
constexpr size_t kMaxLooseHold = 2 * kMaxLooseName;
}  /* namespace */

struct ReplyStream::Impl {
    const ReplyParser::Impl& P;
    const ReplyFormat::Impl& F;
    explicit Impl(const ReplyParser::Impl& p) : P(p), F(p.fmt->impl()) {
        content.init(P.terminators.empty() ? nullptr : &P.terms, &P.terminators);
        st = P.pure ? St::Text : St::LeadWs;
    }

    ReplyMessage msg;
    ContentOut   content;
    bool         think_text = false;    /* the reasoning has had a non-blank byte */
    std::string  think_pend;            /* its leading blanks, until then */
    std::vector<ReplyEvent>* out = nullptr;

    St   st;
    bool strict = true, finished = false, fed_tail = false;
    bool post_call = false;             /* content after a call */

    /* delimiter scan */
    uint16_t    ds = 0;
    std::string hold;
    /* literal match */
    std::string lhold;
    size_t      lpos = 0, trail = 0;
    /* a call being opened */
    St          origin = St::Text;
    /* LOOSE: opened by what the format's grammar does not open a call with -- name_prefix alone,
     * or call_open without its trailing whitespace. Such an opener is also ordinary text often
     * enough ("the `<function=f>` syntax") that it becomes a call only once what follows the
     * name tag is an argument or the call's close; until then it is held, and if it is not a
     * call it is content, as the grammar says, and the reply is still within it. WRAPPED: the
     * call was opened with call_open, so its close is followed by call_close. */
    bool        loose = false, wrapped = false;
    size_t      tag_end = 0;            /* where a loose call's name tag ends in `att` */
    bool        name_lt = false;        /* a loose call's argument name holds a '<' */
    std::string att, name;
    int         tool = -1;
    /* the open call */
    size_t      argc = 0, calls = 0;
    int         depth = 0;              /* unclosed braces in the arguments so far */
    bool        in_str = false, esc = false;
    const Param* param = nullptr;
    /* a value */
    ValueOut    val;
    ValueOut::Mode vmode = ValueOut::String;
    JsonScan    js;
    std::string vheld;
    size_t      vs_pos = 0;
    bool        fenced = false;

    /* published offsets, for the deltas */
    size_t sent_r = 0, sent_c = 0, sent_a = 0;
    int    cur = -1;                    /* the channel last published: 0 r, 1 c, 2 args */

    /* bytes to read again, ahead of the rest */
    struct Seg { const char* p; size_t n; };
    std::vector<Seg>        stack;
    std::deque<std::string> owned;
    std::string             refeed;

    void lenient() { strict = false; }

    /* ------------------------------------------------------------ output channels */

    void event(ReplyEvent::Kind k, uint32_t call, std::string_view text) {
        if (!out) return;
        if ((k == ReplyEvent::Reasoning || k == ReplyEvent::Content || k == ReplyEvent::CallArgs) &&
            !out->empty() && out->back().kind == k && out->back().call == call) {
            out->back().text.append(text);
            return;
        }
        ReplyEvent e;
        e.kind = k;
        e.call = call;
        e.text.assign(text);
        out->push_back(std::move(e));
    }

    void publish(int ch, bool all) {
        std::string* s; size_t* sent; ReplyEvent::Kind k; uint32_t call = 0;
        if (ch == 0) { s = &msg.reasoning; sent = &sent_r; k = ReplyEvent::Reasoning; }
        else if (ch == 1) { s = &msg.content; sent = &sent_c; k = ReplyEvent::Content; }
        else {
            if (msg.calls.empty()) return;
            s = &msg.calls.back().arguments; sent = &sent_a; k = ReplyEvent::CallArgs;
            call = (uint32_t)msg.calls.size() - 1;
        }
        const size_t end = all ? s->size() : s->size() - utf8_tail(*s, *sent);
        if (end > *sent) {
            event(k, call, std::string_view(*s).substr(*sent, end - *sent));
            *sent = end;
        }
    }

    void touch(int ch) {
        if (cur != ch && cur >= 0) publish(cur, false);
        cur = ch;
    }

    void think_out(const char* p, size_t n) {
        if (n == 0) return;
        touch(0);
        if (think_text) { msg.reasoning.append(p, n); return; }
        for (size_t i = 0; i < n; ++i)
            if (!is_blank((unsigned char)p[i])) { think_text = true; break; }
        think_pend.append(p, n);
        if (think_text) { msg.reasoning += think_pend; think_pend.clear(); }
    }

    void text_out(const char* p, size_t n) {
        if (n == 0) return;
        if (post_call && (!F.f.call_open.empty() || !P.opt.parallel_tool_calls)) lenient();
        touch(1);
        content.add(p, n, msg.content);
    }

    void args_out(std::string_view s) {
        std::string& a = msg.calls.back().arguments;
        for (char ch : s) {
            if (esc) { esc = false; continue; }
            if (ch == '\\' && in_str) { esc = true; continue; }
            if (ch == '"') { in_str = !in_str; continue; }
            if (!in_str) { if (ch == '{') ++depth; else if (ch == '}') --depth; }
        }
        touch(2);
        a.append(s);
    }

    /* The value encoder writes into a scratch string; the arguments see it through args_out so
     * the brace count stays right. */
    std::string scratch;
    void flush_scratch() { if (!scratch.empty()) { args_out(scratch); scratch.clear(); } }

    /* ------------------------------------------------------------ scanning */

    /* Scan text under `d` from the current state, handing plain text to `sink`. Returns the
     * position after a matched delimiter with *hit its index, or `e` with *hit = -1. */
    template <class Sink>
    const char* scan(const Dfa& d, const char* p, const char* e, int* hit, Sink&& sink) {
        *hit = -1;
        while (p < e) {
            if (ds == 0) {
                const char* q = d.skip(p, e);
                if (q > p) sink(p, (size_t)(q - p));
                p = q;
                if (p == e) break;
            }
            const unsigned char c = (unsigned char)*p++;
            const uint16_t ns = d.step(ds, c);
            const size_t keep = d.depth(ns);
            const size_t win = hold.size() + 1;
            if (keep < win) {
                const size_t rel = win - keep;
                if (rel <= hold.size()) {
                    sink(hold.data(), rel);
                    hold.erase(0, rel);
                    hold.push_back((char)c);
                } else {
                    if (!hold.empty()) sink(hold.data(), hold.size());
                    hold.clear();
                    const char cc = (char)c;
                    sink(&cc, 1);
                }
            } else {
                hold.push_back((char)c);
            }
            ds = ns;
            if (d.word(ds) >= 0) {
                *hit = d.word(ds);
                ds = 0;
                hold.clear();
                return p;
            }
        }
        return p;
    }

    void read_again(std::string_view bytes) { refeed.append(bytes); }

    /* ------------------------------------------------------------ calls */

    bool declared_prefix(std::string_view n) const {
        for (const Tool& t : P.tools) {
            const std::string full = t.name + F.f.name_suffix;
            if (full.size() >= n.size() && full.compare(0, n.size(), n) == 0) return true;
        }
        return false;
    }

    /* The call opener has matched at `att`; whether a call follows is decided below. */
    void begin_attempt(St from) {
        origin = from;
        name.clear();
        tool = -1;
        lpos = 0;
        loose = false;
        wrapped = !F.f.call_open.empty();
        if (!P.calls) { fail_attempt(); return; }
        st = F.f.call_open.empty() ? St::OpenName : St::OpenWs;
    }

    /* An opener matched in content. In a wrapped format the first is call_open's core, whose
     * trailing whitespace decides whether the grammar opened a call; the second is name_prefix
     * without the wrapper, which is always loose. */
    void begin_text_attempt(int which) {
        origin = St::Text;
        name.clear();
        tool = -1;
        lpos = 0;
        trail = 0;
        att = F.openers[(size_t)which];
        if (F.f.call_open.empty()) {
            loose = false;
            wrapped = false;
            st = St::OpenName;
        } else if (which == 0) {
            loose = false;
            wrapped = true;
            st = St::OpenCoTail;
        } else {
            loose = true;
            wrapped = false;
            st = St::OpenName;
        }
    }

    /* The whitespace a loose opener may be followed by: whatever sorts at or below a space, as a
     * model's stray control characters do. */
    static bool loose_ws(unsigned char c) { return c <= ' '; }

    /* Not a call after all: the opener's first byte is text of the region it was found in, and
     * everything after it is read again from there. */
    void fail_attempt() {
        /* The grammar committed to a call at its own opener, so a call that does not follow
         * leaves it; a loose opener that turns out to be text is text, within it. */
        if (!loose) lenient();
        loose = false;
        st = origin;
        ds = 0;
        hold.clear();
        if (origin == St::Think) think_out(att.data(), 1);
        else text_out(att.data(), 1);
        read_again(std::string_view(att).substr(1));
        att.clear();
    }

    void begin_call(bool exact) {
        if (!exact || tool < 0) lenient();
        if (!P.opt.parallel_tool_calls && calls > 0) lenient();
        if (cur >= 0) publish(cur, false);
        if (cur == 2) publish(2, true);
        const std::string_view nm = trim_trailing_ws(std::string_view(name).substr(
            0, name.size() - F.ns_core.size()));
        ReplyCall c;
        c.name.assign(nm);
        msg.calls.push_back(std::move(c));
        sent_a = 0;
        cur = -1;
        event(ReplyEvent::CallBegin, (uint32_t)msg.calls.size() - 1, msg.calls.back().name);
        depth = 0; in_str = false; esc = false;
        args_out("{");
        argc = 0;
        ++calls;
        att.clear();
        post_call = false;
        st = St::Body;
    }

    /* The core of call_end has matched. The whitespace the format writes after it is part of the
     * tag, so the call closes when that has arrived too -- a reply that ends in between leaves the
     * call open, as the grammar does -- and a reply that goes straight on without it has closed
     * the call all the same. */
    void end_core_done() {
        trail = 0;
        if (F.end_trail.empty() || loose) end_call();
        else st = St::EndTail;
    }

    /* The call's close tag: the arguments object is closed with as many braces as it has open
     * outside strings. A string-typed value is always closed by its own suffix before this can
     * match, so no quote is ever pending here; a raw value that left a string open is passed
     * through as the model wrote it. */
    void end_call() {
        std::string close((size_t)std::max(depth, 0), '}');
        args_out(close);
        publish(2, true);
        msg.calls.back().closed = true;
        event(ReplyEvent::CallEnd, (uint32_t)msg.calls.size() - 1, {});
        cur = -1;
        lpos = 0;
        lhold.clear();
        if (F.f.call_open.empty()) st = St::AfterCall;
        else if (wrapped) st = St::AfterEnd;
        else after_loose_call();
    }

    /* A loose call whose argument name runs into a tag was a call only as far as its arguments
     * before it: the call ends there, and the argument's opener and all after it are content. */
    void end_loose_call_at_arg(bool closed_name) {
        std::string replay = F.f.arg_prefix + name;
        if (closed_name) replay += name_end();
        name.clear();
        name_lt = false;
        wrapped = false;
        end_call();
        read_again(replay);
    }

    /* After a loose call the text goes on as content, whitespace included: the grammar never
     * opened the call, so nothing of its punctuation claims the spacing around it. */
    void after_loose_call() {
        loose = false;
        post_call = true;
        ds = 0;
        hold.clear();
        st = St::Text;
    }

    const Tool* cur_tool() const {
        return tool >= 0 ? &P.tools[(size_t)tool] : nullptr;
    }

    /* The argument name is complete. */
    void begin_value() {
        const std::string_view nm = trim_arg_name(
            std::string_view(name).substr(0, name.size() - name_end().size()));
        std::string key = argc > 0 ? "," : "";
        key += json_quote(nm);
        key += ':';
        args_out(key);
        ++argc;
        param = nullptr;
        if (const Tool* t = cur_tool()) {
            auto it = t->params.find(std::string(nm));
            /* The grammar binds a declared name to its own arm by the raw spelling; a name that
             * only trims to a declared one is read by the generic arm, as text. */
            if (it != t->params.end() && nm.size() + name_end().size() == name.size())
                param = &it->second;
        }
        name.clear();
        if (param && param->typed) {
            js.reset();
            vheld.clear();
            st = St::ValJson;
        } else {
            start_text_value(ValueOut::String);
        }
    }

    /* What ends an argument name and a value in the call being read: the tags as the format
     * writes them, or their cores in a loose call. */
    const std::string& name_end() const { return loose ? F.name_end_core : F.name_end; }
    const std::string& value_suffix() const { return loose ? F.value_core : F.f.arg_value_suffix; }

    /* A CDATA section is punctuation around any value the format lets carry one, typed or not. */
    bool cdata() const { return F.f.string_value == ChatValue::Cdata; }

    void start_text_value(ValueOut::Mode m) {
        vmode = m;
        val.begin(m, cdata(), F.f.value_newlines, scratch);
        flush_scratch();
        ds = 0;
        hold.clear();
        st = St::ValText;
    }

    /* A JSON-typed value that parsed and then met its suffix: the value is the JSON text and the
     * whitespace after it. */
    void json_value_done(std::string_view text) {
        val.begin(ValueOut::Raw, cdata(), F.f.value_newlines, scratch);
        val.add(text.data(), text.size(), scratch);
        val.finish(false, scratch);
        flush_scratch();
    }

    /* ------------------------------------------------------------ the machine
     *
     * One step consumes at least one byte, or queues bytes to be read again. */
    const char* step(const char* p, const char* e) {
        int hit = -1;
        switch (st) {
        case St::LeadWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) st = after_lead();
            return p;

        case St::ThinkOpen: {
            const unsigned char c = (unsigned char)*p;
            if (lpos < F.open_core.size() && (unsigned char)F.open_core[lpos] == c) {
                lhold.push_back((char)c);
                if (++lpos == F.open_core.size()) { lhold.clear(); trail = 0; st = St::ThinkOpenWs; }
                return p + 1;
            }
            /* No block: what was held is read again as whatever follows one. */
            st = after_think();
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::ThinkOpenWs:
            if (trail >= F.open_trail.size()) { st = St::Think; ds = 0; hold.clear(); return p; }
            if (*p == F.open_trail[trail]) { ++trail; return p + 1; }
            ++trail;
            return p;

        case St::Think: {
            const Dfa& d = P.breaks ? F.think_breaks : F.think;
            p = scan(d, p, e, &hit, [this](const char* q, size_t n) { think_out(q, n); });
            if (hit == 0) { trail = 0; st = St::ThinkCloseWs; }
            else if (hit > 0) { att = d.pattern(hit); begin_attempt(St::Think); }
            return p;
        }

        case St::ThinkCloseWs:
            if (trail >= F.close_trail.size()) { st = after_think(); return p; }
            if (*p == F.close_trail[trail]) { ++trail; return p + 1; }
            ++trail;
            return p;

        case St::Text:
            if (!P.calls) {
                text_out(p, (size_t)(e - p));
                return e;
            }
            p = scan(F.text, p, e, &hit, [this](const char* q, size_t n) { text_out(q, n); });
            if (hit >= 0) begin_text_attempt(hit);
            return p;

        case St::OpenCoTail:
            if (trail < F.co_trail.size() && *p == F.co_trail[trail]) {
                att.push_back(*p);
                if (++trail == F.co_trail.size()) st = St::OpenWs;
                return p + 1;
            }
            /* call_open's core without its whitespace: a loose opener. */
            loose = trail < F.co_trail.size();
            st = St::OpenWs;
            return p;

        case St::OpenWs:
            if (loose) {
                while (p < e && loose_ws((unsigned char)*p)) att.push_back(*p++);
            } else {
                while (p < e && is_ws((unsigned char)*p)) att.push_back(*p++);
            }
            if (p < e) { lpos = 0; st = St::OpenPrefix; }
            return p;

        case St::OpenPrefix: {
            const char c = *p;
            att.push_back(c);
            if (c != F.f.name_prefix[lpos]) { fail_attempt(); return p + 1; }
            if (++lpos == F.f.name_prefix.size()) { name.clear(); st = St::OpenName; }
            return p + 1;
        }

        case St::OpenName: {
            const char c = *p;
            att.push_back(c);
            name.push_back(c);
            const size_t k = F.ns_core.size();
            if (name.size() >= k && name.compare(name.size() - k, k, F.ns_core) == 0) {
                const std::string nm = name.substr(0, name.size() - k);
                auto it = P.by_name.find(nm);
                tool = it == P.by_name.end() ? -1 : (int)it->second;
                if (tool < 0 && (nm.empty() || nm.find('<') != std::string::npos)) {
                    fail_attempt();
                    return p + 1;
                }
                /* The call exists from the moment its opening tag is complete: a reply that ends
                 * right there is a call with no arguments yet. A loose call waits for what
                 * follows the tag. */
                trail = 0;
                lpos = 0;
                lhold.clear();
                tag_end = att.size();
                if (loose) st = St::OpenPeek;
                else if (F.ns_trail.empty()) begin_call(true);
                else st = St::OpenTail;
                return p + 1;
            }
            if (name.size() > kMaxLooseName || (c == '<' && !declared_prefix(name))) {
                fail_attempt();
                return p + 1;
            }
            return p + 1;
        }

        case St::OpenTail:
            if (trail < F.ns_trail.size() && *p == F.ns_trail[trail]) {
                att.push_back(*p);
                if (++trail == F.ns_trail.size()) begin_call(true);
                return p + 1;
            }
            begin_call(trail == F.ns_trail.size());
            return p;

        case St::OpenPeek: {
            if (lpos == 0) {
                while (p < e && loose_ws((unsigned char)*p) && att.size() <= kMaxLooseHold)
                    att.push_back(*p++);
                if (att.size() > kMaxLooseHold) { fail_attempt(); return p; }
                if (p == e) return p;
            }
            const char c = *p;
            att.push_back(c);
            const std::string& ap = F.f.arg_prefix;
            const std::string& fc = F.end_core;
            const bool a_ok = lpos < ap.size() && ap[lpos] == c && ap.compare(0, lpos, lhold) == 0;
            const bool f_ok = lpos < fc.size() && fc[lpos] == c && fc.compare(0, lpos, lhold) == 0;
            if (!a_ok && !f_ok) { fail_attempt(); return p + 1; }
            lhold.push_back(c);
            ++lpos;
            if (a_ok && lpos == ap.size()) {
                lhold.clear();
                st = St::OpenPeekArg;
            } else if (f_ok && lpos == fc.size()) {
                begin_call(false);
                end_call();
            }
            return p + 1;
        }

        case St::OpenPeekArg: {
            /* The first argument's name, which is where prose that merely looks like a call
             * shows itself: a name that runs into a tag is no name. On one that closes, the call
             * is read -- the bytes after the name tag replayed through the call itself. */
            const char c = *p;
            att.push_back(c);
            lhold.push_back(c);
            const std::string& ne = F.name_end_core;
            if (lhold.size() > kMaxLooseName) { fail_attempt(); return p + 1; }
            if (lhold.size() >= ne.size() &&
                lhold.compare(lhold.size() - ne.size(), ne.size(), ne) == 0) {
                const std::string_view nm(lhold.data(), lhold.size() - ne.size());
                if (nm.empty() || nm.find('<') != std::string_view::npos) {
                    fail_attempt();
                    return p + 1;
                }
                const std::string replay = att.substr(tag_end);
                lhold.clear();
                begin_call(false);
                read_again(replay);
            }
            return p + 1;
        }

        case St::Body:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { lpos = 0; lhold.clear(); st = St::BodyLit; }
            return p;

        case St::BodyLit: {
            const char c = *p;
            const std::string& ap = F.f.arg_prefix;
            const std::string& fc = F.end_core;
            const bool a_ok = lpos < ap.size() && ap[lpos] == c && ap.compare(0, lpos, lhold) == 0;
            const bool f_ok = lpos < fc.size() && fc[lpos] == c && fc.compare(0, lpos, lhold) == 0;
            if (!a_ok && !f_ok) {
                lenient();
                st = St::Junk;
                ds = 0;
                hold.clear();
                lhold.push_back(c);
                read_again(lhold);
                lhold.clear();
                return p + 1;
            }
            lhold.push_back(c);
            ++lpos;
            if (a_ok && lpos == ap.size()) { start_arg(); return p + 1; }
            if (f_ok && lpos == fc.size()) { finish_call_strictness(); end_core_done(); return p + 1; }
            return p + 1;
        }

        case St::ArgName:
            p = scan(loose ? F.arg_name_loose : F.arg_name, p, e, &hit,
                     [this](const char* q, size_t n) {
                         name.append(q, n);
                         if (loose && std::memchr(q, '<', n)) name_lt = true;
                     });
            if (name_lt) { end_loose_call_at_arg(hit >= 0); return p; }
            if (hit >= 0) { name += name_end(); begin_value(); }
            return p;

        case St::ValText:
            p = scan(loose ? F.value_loose : F.value, p, e, &hit, [this](const char* q, size_t n) {
                val.add(q, n, scratch);
                flush_scratch();
            });
            if (hit >= 0) {
                val.finish(false, scratch);
                if (vmode == ValueOut::String) scratch += '"';
                flush_scratch();
                st = St::Body;
            }
            return p;

        case St::ValJson: {
            for (; p < e; ++p) {
                const unsigned char c = (unsigned char)*p;
                const JsonScan::Res r = js.push(c);
                if (r == JsonScan::More) { vheld.push_back((char)c); continue; }
                if (r == JsonScan::Fail) {
                    /* Not JSON: the argument is its text, read again from its first byte. */
                    vheld.push_back((char)c);
                    start_text_value(ValueOut::Raw);
                    read_again(vheld);
                    vheld.clear();
                    return p + 1;
                }
                vs_pos = 0;
                st = St::ValJsonEnd;
                return p;
            }
            return p;
        }

        case St::ValJsonEnd: {
            const std::string& vs = value_suffix();
            const char c = *p;
            vheld.push_back(c);
            if (c == vs[vs_pos]) {
                if (++vs_pos == vs.size()) {
                    json_value_done(std::string_view(vheld).substr(0, vheld.size() - vs.size()));
                    vheld.clear();
                    st = St::Body;
                }
                return p + 1;
            }
            /* JSON followed by something else: the grammar reads the argument as text, all of
             * it, from its first byte. */
            start_text_value(ValueOut::String);
            read_again(vheld);
            vheld.clear();
            return p + 1;
        }

        case St::EndTail:
            if (trail < F.end_trail.size() && *p == F.end_trail[trail]) {
                if (++trail == F.end_trail.size()) end_call();
                return p + 1;
            }
            /* The close's own whitespace is missing: still the close, read leniently. */
            lenient();
            end_call();
            return p;

        case St::Junk:
            p = scan(F.junk, p, e, &hit, [this](const char* q, size_t n) { text_out(q, n); });
            if (hit == 0) start_arg();
            else if (hit == 1) end_core_done();
            return p;

        case St::AfterEnd:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) {
                lpos = 0;
                lhold.clear();
                st = F.f.call_close.empty() ? St::AfterCall : St::AfterEndLit;
            }
            return p;

        case St::AfterEndLit: {
            const char c = *p;
            if (c == F.f.call_close[lpos]) {
                lhold.push_back(c);
                if (++lpos == F.f.call_close.size()) {
                    lhold.clear();
                    if (loose) after_loose_call();
                    else st = St::AfterCall;
                }
                return p + 1;
            }
            lenient();
            if (loose) after_loose_call();
            else st = St::AfterCall;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::AfterCall:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { post_call = true; ds = 0; hold.clear(); st = St::Text; }
            return p;

        case St::RfWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) {
                if (*p == '`') { lpos = 0; lhold.clear(); st = St::RfFence; }
                else { js.reset(); fenced = false; st = St::RfJson; }
            }
            return p;

        case St::RfFence: {
            const char c = *p;
            if (c == kFenceOpen[lpos]) {
                lhold.push_back(c);
                if (++lpos == kFenceOpen.size()) { lhold.clear(); st = St::RfFenceWs; }
                return p + 1;
            }
            lenient();
            st = St::RfRaw;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::RfFenceWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { js.reset(); fenced = true; st = St::RfJson; }
            return p;

        case St::RfJson: {
            const char* run = p;
            for (; p < e; ++p) {
                const JsonScan::Res r = js.push((unsigned char)*p);
                if (r == JsonScan::More) continue;
                text_out(run, (size_t)(p - run));
                if (r == JsonScan::Fail || !fenced) { lenient(); st = St::RfRaw; return p; }
                lpos = 0;
                lhold.clear();
                st = St::RfClose;
                return p;
            }
            text_out(run, (size_t)(p - run));
            return p;
        }

        case St::RfClose: {
            const char c = *p;
            if (c == kFenceClose[lpos]) {
                lhold.push_back(c);
                if (++lpos == kFenceClose.size()) { lhold.clear(); st = St::RfEnd; }
                return p + 1;
            }
            lenient();
            st = St::RfRaw;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::RfEnd:
            lenient();
            st = St::RfRaw;
            return p;

        case St::RfRaw:
            text_out(p, (size_t)(e - p));
            return e;
        }
        return e;
    }

    St after_lead() const {
        if (F.f.has_reasoning() && P.opt.extract_reasoning)
            return F.open_core.empty() ? St::Think : St::ThinkOpen;
        return after_think();
    }
    St after_think() const { return P.opt.response_format ? St::RfWs : St::Text; }

    /* An argument opener matched. The generic argument arm exists only for a tool that declares
     * parameters; a tool that declares none takes none. */
    void start_arg() {
        const Tool* t = cur_tool();
        if (t && !t->has_props) lenient();
        name.clear();
        name_lt = false;
        ds = 0;
        hold.clear();
        st = St::ArgName;
    }

    /* The reply ended inside an argument name. A name is only a name once it is closed -- except
     * that where the format writes nothing between a name and its value's own prefix
     * (arg_name_suffix empty), the grammar's opening tag for a DECLARED argument is complete as
     * soon as the declared name is: the reply `<param name="path` already carries the key. The
     * first declared name in the grammar's order that the text spells, followed by at most part
     * of the value prefix, is the key. */
    void finish_arg_name() {
        const Tool* t = cur_tool();
        if (!t || !F.f.arg_name_suffix.empty()) return;
        for (const std::string& a : t->arms) {
            if (name.size() < a.size() || name.compare(0, a.size(), a) != 0) continue;
            const std::string_view rest = std::string_view(name).substr(a.size());
            if (rest.size() >= F.name_end.size() ||
                F.name_end.compare(0, rest.size(), rest) != 0)
                continue;
            std::string key = argc > 0 ? "," : "";
            key += json_quote(trim_arg_name(a));
            key += ':';
            args_out(key);
            ++argc;
            return;
        }
    }

    /* call_end matched: a tool with required parameters needed at least one argument first. */
    void finish_call_strictness() {
        const Tool* t = cur_tool();
        if (t && t->has_required && argc == 0) lenient();
    }

    /* ------------------------------------------------------------ driving */

    void run(std::string_view in) {
        stack.push_back({ in.data(), in.size() });
        while (!stack.empty()) {
            Seg s = stack.back();
            stack.pop_back();
            const char* p = s.p;
            const char* e = s.p + s.n;
            while (p < e) {
                p = step(p, e);
                if (!refeed.empty()) {
                    if (p < e) stack.push_back({ p, (size_t)(e - p) });
                    owned.push_back(std::move(refeed));
                    refeed.clear();
                    stack.push_back({ owned.back().data(), owned.back().size() });
                    break;
                }
            }
        }
        owned.clear();
    }

    void feed_tail() {
        if (fed_tail) return;
        fed_tail = true;
        if (!P.tail.empty()) run(P.tail);
    }

    /* Would the grammar still be waiting for more, rather than have failed, on what an opener
     * has held? Asked only at the end of the reply. */
    bool open_viable() const {
        switch (st) {
        case St::OpenWs:
        case St::OpenPrefix:
            return true;
        case St::OpenName:
            return declared_prefix(name);
        case St::OpenTail:
            return tool >= 0;
        default:
            return true;
        }
    }

    /* The reply ended inside a loose opener, so it was never a call: its bytes are content, read
     * again from its second byte -- which may hold another opener, and so on. */
    bool in_loose_attempt() const {
        switch (st) {
        case St::OpenWs: case St::OpenPrefix: case St::OpenName: case St::OpenPeek:
        case St::OpenPeekArg:
            return loose;
        default:
            return false;
        }
    }

    void finish_all() {
        while (in_loose_attempt()) {
            fail_attempt();
            std::string again = std::move(refeed);
            refeed.clear();
            run(again);
        }
        switch (st) {
        case St::OpenWs: case St::OpenPrefix: case St::OpenName: case St::OpenTail:
            if (!open_viable()) lenient();
            break;
        case St::BodyLit: {
            const Tool* t = cur_tool();
            const bool args_ok = !t || t->has_props;
            const bool end_ok  = !t || !t->has_required || argc > 0;
            const bool a = args_ok && F.f.arg_prefix.compare(0, lhold.size(), lhold) == 0;
            const bool f = end_ok && F.end_core.compare(0, lhold.size(), lhold) == 0;
            if (t && !a && !f) lenient();
            break;
        }
        case St::ArgName:
            finish_arg_name();
            break;
        case St::ValText:
            /* A partly-matched suffix at the end means the value itself was complete. */
            val.finish(hold.empty(), scratch);
            flush_scratch();
            break;
        case St::ValJson:
            val.begin(ValueOut::Raw, cdata(), F.f.value_newlines, scratch);
            val.add(vheld.data(), vheld.size(), scratch);
            val.finish(true, scratch);
            flush_scratch();
            break;
        case St::ValJsonEnd:
            json_value_done(std::string_view(vheld).substr(0, vheld.size() - vs_pos));
            break;
        default:
            break;
        }
        hold.clear();
        ds = 0;
        if (!think_text) think_pend.clear();
        content.finish(msg.content);
        if (cur >= 0) publish(cur, true);
        publish(0, true);
        publish(1, true);
        publish(2, true);
    }
};

ReplyStream::ReplyStream() = default;
ReplyStream::~ReplyStream() = default;

std::unique_ptr<ReplyStream> ReplyParser::open() const {
    std::unique_ptr<ReplyStream> s(new ReplyStream());
    s->impl_ = std::make_unique<ReplyStream::Impl>(*impl_);
    return s;
}

void ReplyStream::feed(std::string_view bytes, std::vector<ReplyEvent>* out) {
    Impl& m = *impl_;
    if (m.finished) return;
    m.out = out;
    m.feed_tail();
    if (!bytes.empty()) m.run(bytes);
    if (m.cur >= 0) m.publish(m.cur, false);
    m.out = nullptr;
}

void ReplyStream::finish(std::vector<ReplyEvent>* out) {
    Impl& m = *impl_;
    if (m.finished) return;
    m.out = out;
    m.feed_tail();
    m.finish_all();
    m.finished = true;
    m.out = nullptr;
}

const ReplyMessage& ReplyStream::message() const { return impl_->msg; }
bool ReplyStream::strict() const { return impl_->strict; }

ReplyHeld ReplyStream::held() const {
    const Impl& m = *impl_;
    ReplyHeld h;
    h.delimiter = m.hold.size() + m.lhold.size() + m.val.held() + m.content.held_other() +
                  (m.msg.reasoning.size() - m.sent_r) + (m.msg.content.size() - m.sent_c) +
                  (m.msg.calls.empty() ? 0 : m.msg.calls.back().arguments.size() - m.sent_a);
    h.space = m.content.held_space() + m.think_pend.size();
    h.structure = m.att.size() + m.name.size() + m.vheld.size();
    return h;
}

ReplyMessage reply_parse(const ReplyParser& p, std::string_view text, bool* strict) {
    std::unique_ptr<ReplyStream> s = p.open();
    s->feed(text, nullptr);
    s->finish(nullptr);
    if (strict) *strict = s->strict();
    return s->message();
}

}  /* namespace rad */
