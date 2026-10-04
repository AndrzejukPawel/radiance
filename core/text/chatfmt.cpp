#include "text/chatfmt.h"
#include "text/chat.h"

#include "rad_builder.h"

/* The analyser and the grammar builder, at load and admission time only -- never while a reply
 * is being read. */
#include "llama/common/chat.h"
#include "llama/common/chat-auto-parser.h"
#include "llama/common/json-schema-to-grammar.h"
#include "shim/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <set>

namespace rad {

using ojson = nlohmann::ordered_json;

namespace {

bool is_ws(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws((unsigned char)s[a])) ++a;
    while (b > a && is_ws((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

/* `inner` occurs inside `outer` somewhere other than at its start. */
bool occurs_inside(const std::string& inner, const std::string& outer) {
    return !inner.empty() && outer.size() > inner.size() && outer.find(inner, 1) != std::string::npos;
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::string show(const std::string& s) { return nlohmann::json(s).dump(); }

}  /* namespace */

/* ================================================================== validity */

int chat_format_check(const ChatFormat& f, std::string* why) {
    auto no = [&](const std::string& s) {
        if (why) *why = s;
        return RAD_E_UNSUPPORTED;
    };

    if (!f.reasoning_open.empty() && f.reasoning_close.empty())
        return no("reasoning_open " + show(f.reasoning_open) + " has no reasoning_close");
    if (f.has_reasoning() && trim_ws(f.reasoning_close).empty())
        return no("reasoning_close is only whitespace");
    if ((uint32_t)f.reasoning_in_prompt > (uint32_t)ChatThink::Never)
        return no("reasoning_in_prompt is not a RAD_CHAT_THINK_* value");

    if (!f.has_tools()) {
        for (const std::string* s : { &f.section_open, &f.section_close, &f.call_separator,
                                      &f.call_open, &f.call_close, &f.name_suffix, &f.call_end,
                                      &f.arg_prefix, &f.arg_name_suffix, &f.arg_value_prefix,
                                      &f.arg_value_suffix })
            if (!s->empty()) return no("tool-call delimiters are set but name_prefix is empty");
        if (!f.reasoning_breaks.empty()) return no("reasoning_breaks without tool calls");
        if (!f.triggers.empty()) return no("grammar triggers without tool calls");
        return RAD_OK;
    }

    if (!f.section_open.empty() || !f.section_close.empty())
        return no("a section around all calls (section_open/section_close) is not read by the "
                  "streaming parser");
    if (!f.call_separator.empty())
        return no("a separator between calls (call_separator " + show(f.call_separator) +
                  ") is not read by the streaming parser; tagged calls separate on whitespace");
    if (f.call_open.empty() && !f.call_close.empty())
        return no("call_close without call_open");
    if (f.name_suffix.empty() || trim_ws(f.name_suffix).empty())
        return no("name_suffix must end the function name with something besides whitespace");
    if (is_ws((unsigned char)f.name_suffix[0]))
        return no("name_suffix " + show(f.name_suffix) + " may not begin with whitespace");
    if (f.call_end.empty()) return no("call_end is empty: a call must close with its own tag");
    if (f.arg_prefix.empty()) return no("arg_prefix is empty");
    if (f.arg_value_suffix.empty()) return no("arg_value_suffix is empty");
    if (f.arg_name_suffix.empty() && f.arg_value_prefix.empty())
        return no("nothing ends an argument name: arg_name_suffix and arg_value_prefix are empty");
    if (!f.arg_name_suffix.empty() && !f.arg_value_prefix.empty())
        return no("arg_name_suffix and arg_value_prefix are both set; the streaming parser reads "
                  "one delimiter between an argument's name and its value");
    if (f.string_value != ChatValue::Raw && f.string_value != ChatValue::Cdata)
        return no("string_value must be RAD_CHAT_VALUE_RAW or RAD_CHAT_VALUE_CDATA");
    if ((uint32_t)f.other_value > (uint32_t)ChatValue::Json)
        return no("other_value is not a RAD_CHAT_VALUE_* value");

    /* A wrapped format's content is scanned for the wrapper's core and for name_prefix alone,
     * which opens a call written without its wrapper; the two must not be confusable. */
    if (!f.call_open.empty()) {
        const std::string co = trim_ws(f.call_open);
        if (co.empty()) return no("call_open is only whitespace");
        if (starts_with(co, f.name_prefix) || starts_with(f.name_prefix, co) ||
            occurs_inside(co, f.name_prefix) || occurs_inside(f.name_prefix, co))
            return no("call_open " + show(f.call_open) + " and name_prefix " +
                      show(f.name_prefix) + " cannot be told apart");
    }
    const std::string opener = f.call_open.empty() ? f.name_prefix : f.call_open;
    for (const std::string& b : f.reasoning_breaks)
        if (b != opener)
            return no("reasoning break " + show(b) + " is not the call opener " + show(opener));
    if (!f.reasoning_breaks.empty() && !f.has_reasoning())
        return no("reasoning_breaks without a reasoning block");
    for (const std::string& t : f.triggers)
        if (t.empty()) return no("an empty grammar trigger");

    /* The delimiter sets scanned together must be read in the order they START, which an
     * automaton reports correctly only when no delimiter occurs inside another. */
    if (f.has_reasoning()) {
        const std::string close = trim_ws(f.reasoning_close);
        for (const std::string& b : f.reasoning_breaks)
            if (occurs_inside(close, b) || occurs_inside(b, close) || starts_with(b, close) ||
                starts_with(close, b))
                return no("reasoning break " + show(b) + " overlaps reasoning_close " +
                          show(close));
    }
    /* Inside a call the next tag is an argument or the close, and the close is recognised by its
     * core as well: the whitespace around it is optional when a reply leaves the format. */
    const std::string end = trim_ws(f.call_end);
    if (end.empty()) return no("call_end is only whitespace");
    for (const std::string* e : { &f.call_end, &end })
        if (starts_with(f.arg_prefix, *e) || starts_with(*e, f.arg_prefix) ||
            occurs_inside(f.arg_prefix, *e) || occurs_inside(*e, f.arg_prefix))
            return no("arg_prefix " + show(f.arg_prefix) + " and call_end " + show(f.call_end) +
                      " cannot be told apart");
    return RAD_OK;
}

/* ================================================================== the plugin's declaration */

int chat_format_from_abi(const RadChatFormat* in, ChatFormat* out, std::string* why) {
    if (!in || !out) {
        if (why) *why = "no format";
        return RAD_E_INVAL;
    }
    if (in->struct_size < offsetof(RadChatFormat, name) + sizeof(in->name)) {
        if (why) *why = "struct_size " + std::to_string(in->struct_size) + " is too small";
        return RAD_E_INVAL;
    }
    RadChatFormat c;
    std::memset(&c, 0, sizeof c);
    std::memcpy(&c, in, std::min<size_t>(in->struct_size, sizeof c));

    auto s = [](const char* p) { return p ? std::string(p) : std::string(); };
    auto list = [](const char* const* v, uint32_t n, std::vector<std::string>* o) {
        o->clear();
        for (uint32_t i = 0; v && i < n; ++i)
            if (v[i]) o->emplace_back(v[i]);
    };

    ChatFormat f;
    f.name            = s(c.name);
    f.reasoning_open  = s(c.reasoning_open);
    f.reasoning_close = s(c.reasoning_close);
    list(c.reasoning_breaks, c.n_reasoning_breaks, &f.reasoning_breaks);
    f.reasoning_in_prompt = (ChatThink)c.reasoning_in_prompt;
    f.section_open    = s(c.section_open);
    f.section_close   = s(c.section_close);
    f.call_separator  = s(c.call_separator);
    f.call_open       = s(c.call_open);
    f.call_close      = s(c.call_close);
    f.name_prefix     = s(c.name_prefix);
    f.name_suffix     = s(c.name_suffix);
    f.call_end        = s(c.call_end);
    f.arg_prefix      = s(c.arg_prefix);
    f.arg_name_suffix = s(c.arg_name_suffix);
    f.arg_value_prefix = s(c.arg_value_prefix);
    f.arg_value_suffix = s(c.arg_value_suffix);
    f.string_value    = (ChatValue)c.string_value;
    f.other_value     = (ChatValue)c.other_value;
    f.value_newlines  = c.value_newlines != 0;
    list(c.triggers, c.n_triggers, &f.triggers);
    *out = std::move(f);
    return RAD_OK;
}

/* ================================================================== the template's format */

int chat_format_from_template(const std::string& jinja, const std::string& bos,
                              const std::string& eos, ChatFormat* out,
                              std::string* tools_refused, std::string* why) {
    std::string dummy;
    if (!tools_refused) tools_refused = &dummy;
    if (!why) why = &dummy;
    tools_refused->clear();
    const std::string src = jinja.empty() ? std::string(ChatTemplate::chatml_source()) : jinja;

    ChatFormat f;
    f.name = "template";
    try {
        common_chat_template tmpl(src, bos, eos);

        /* A template llama.cpp reads with a hand-written handler is one the analysis never
         * describes, and its markers would be read as if it had. */
        autoparser::generation_params probe;
        probe.messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
        probe.add_generation_prompt = true;
        if (common_chat_try_specialized_template(tmpl, src, probe)) {
            *why = "the chat template is one llama.cpp reads with a hand-written handler, whose "
                   "reply format the template analysis does not describe";
            return RAD_E_UNSUPPORTED;
        }

        autoparser::autoparser a;
        a.analyze_template(tmpl);

        if (a.reasoning.mode != autoparser::reasoning_mode::NONE && !a.reasoning.end.empty()) {
            f.reasoning_open  = a.reasoning.start;
            f.reasoning_close = a.reasoning.end;
        }
        if (a.content.mode != autoparser::content_mode::PLAIN) {
            *why = "the template wraps its answer in content markers, which the streaming parser "
                   "does not read";
            return RAD_E_UNSUPPORTED;
        }

        const auto& t = a.tools;
        const bool calls = t.format.mode != autoparser::tool_format::NONE &&
                           a.jinja_caps.supports_tool_calls;
        std::string refuse;
        if (calls) {
            if (t.format.mode == autoparser::tool_format::JSON_NATIVE)
                refuse = "the template writes each call as a JSON object";
            else if (t.format.mode == autoparser::tool_format::TAG_WITH_JSON)
                refuse = "the template writes a call's arguments as one JSON object inside a tag";
            else if (t.call_id.pos != autoparser::call_id_position::NONE)
                refuse = "the template writes a call id";
            else if (!t.arguments.start.empty() || !t.arguments.end.empty())
                refuse = "the template wraps a call's arguments in " + show(t.arguments.start) +
                         " .. " + show(t.arguments.end);
            else if (!t.arguments.separator.empty())
                refuse = "the template separates arguments with " + show(t.arguments.separator);
        }
        if (calls && refuse.empty()) {
            ChatFormat c = f;
            c.section_open     = t.format.section_start;
            c.section_close    = t.format.section_end;
            c.call_open        = t.format.per_call_start;
            c.call_close       = t.format.per_call_end;
            c.name_prefix      = t.function.name_prefix;
            c.name_suffix      = t.function.name_suffix;
            c.call_end         = t.function.close;
            c.arg_prefix       = t.arguments.name_prefix;
            c.arg_name_suffix  = t.arguments.name_suffix;
            c.arg_value_prefix = t.arguments.value_prefix;
            c.arg_value_suffix = t.arguments.value_suffix;
            c.string_value     = ChatValue::Cdata;
            c.other_value      = ChatValue::Json;
            c.value_newlines   = true;
            /* What opens a call is what both the lazy grammar waits for and what ends an
             * unclosed reasoning block -- the section or per-call opener. A format with neither
             * gets no grammar and no break: its calls open with the function name prefix, which
             * a model also writes in prose. */
            const std::string& marker = !t.format.section_start.empty() ? t.format.section_start
                                                                         : t.format.per_call_start;
            if (!marker.empty()) {
                c.triggers.push_back(marker);
                if (c.has_reasoning()) c.reasoning_breaks.push_back(marker);
            }
            std::string bad;
            if (chat_format_check(c, &bad) == RAD_OK) f = std::move(c);
            else refuse = bad;
        }
        if (!refuse.empty()) *tools_refused = refuse;
    } catch (const std::exception& e) {
        *why = std::string("the chat template could not be analysed: ") + e.what();
        return RAD_E_FORMAT;
    }
    *out = std::move(f);
    return RAD_OK;
}

std::string chat_format_describe(const ChatFormat& f) {
    std::string s = f.name + ":";
    if (f.has_reasoning())
        s += " reasoning " + show(f.reasoning_open) + ".." + show(f.reasoning_close);
    else
        s += " no reasoning";
    if (f.has_tools()) {
        s += "; calls " + show(f.call_open) + show(f.name_prefix) + "NAME" + show(f.name_suffix) +
             " " + show(f.arg_prefix) + "ARG" + show(f.arg_name_suffix) + show(f.arg_value_prefix) +
             "VALUE" + show(f.arg_value_suffix) + " " + show(f.call_end) + show(f.call_close);
        if (!f.triggers.empty()) {
            s += "; grammar on";
            for (const std::string& t : f.triggers) s += " " + show(t);
        } else {
            s += "; no grammar";
        }
    } else {
        s += "; no tool calls";
    }
    return s;
}

std::vector<std::string> chat_format_markers(const ChatFormat& f) {
    std::vector<std::string> out;
    for (const std::string* s : { &f.reasoning_open, &f.reasoning_close, &f.section_open,
                                  &f.section_close, &f.call_open, &f.call_close, &f.name_prefix,
                                  &f.name_suffix, &f.call_end, &f.arg_prefix, &f.arg_name_suffix,
                                  &f.arg_value_prefix, &f.arg_value_suffix }) {
        const std::string t = trim_ws(*s);
        if (!t.empty() && std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    return out;
}

/* ================================================================== grammar
 *
 * The grammar is built as an expression tree and rendered by the rules the vendored PEG-to-GBNF
 * pass uses, so a format derived from a template yields the same text the vendored generator
 * produces for it: sequences and alternations are flattened as they are built, a sequence or
 * alternation nested in a sequence is parenthesised, only an alternation nested in an
 * alternation is, and a repeated sequence or alternation is. */
namespace {

struct G {
    enum K : uint8_t { Eps, Lit, Seq, Alt, Rep, Leaf, Wrap } k = Eps;
    std::string s;               /* Lit: the text; Leaf: rendered GBNF */
    std::vector<G> kids;
    int min = 0, max = -1;
    bool compound = false;       /* Leaf/Wrap: renders like a sequence for parenthesising */
};

G eps() { return G{}; }
G lit(const std::string& s) { G g; g.k = G::Lit; g.s = s; return g; }
G leaf(const std::string& s) { G g; g.k = G::Leaf; g.s = s; return g; }
G space() { return leaf("space"); }
G rule_ref(const std::string& name) { return leaf(name); }

/* A tag or an atomic around a node: it renders as the node, is parenthesised as the node would
 * be, but a sequence built around it does not flatten into it. */
G wrap(G inner) { G g; g.k = G::Wrap; g.kids.push_back(std::move(inner)); return g; }

G seq(std::vector<G> parts) {
    G g; g.k = G::Seq;
    for (G& p : parts) {
        if (p.k == G::Seq) for (G& c : p.kids) g.kids.push_back(std::move(c));
        else g.kids.push_back(std::move(p));
    }
    return g;
}
G alt(std::vector<G> parts) {
    G g; g.k = G::Alt;
    for (G& p : parts) {
        if (p.k == G::Alt) for (G& c : p.kids) g.kids.push_back(std::move(c));
        else g.kids.push_back(std::move(p));
    }
    return g;
}
G rep(G child, int min, int max) {
    G g; g.k = G::Rep; g.min = min; g.max = max; g.kids.push_back(std::move(child)); return g;
}
G opt(G child) { return rep(std::move(child), 0, 1); }

const G& effective(const G& g) { return g.k == G::Wrap ? effective(g.kids[0]) : g; }

std::string render(const G& g);

std::string render_rep_suffix(int min, int max) {
    if (min == 0 && max == 1) return "?";
    if (min == 0 && max == -1) return "*";
    if (min == 1 && max == -1) return "+";
    if (max == -1) return "{" + std::to_string(min) + ",}";
    if (min == max) return min == 1 ? "" : "{" + std::to_string(min) + "}";
    return "{" + std::to_string(min) + "," + std::to_string(max) + "}";
}

std::string render(const G& g) {
    switch (g.k) {
    case G::Eps: return "";
    case G::Lit: return gbnf_format_literal(g.s);
    case G::Leaf: return g.s;
    case G::Wrap: return render(g.kids[0]);
    case G::Seq: {
        std::string s;
        for (const G& c : g.kids) {
            const std::string r = render(c);
            if (r.empty()) continue;
            if (!s.empty()) s += " ";
            const G& e = effective(c);
            if (e.k == G::Seq || e.k == G::Alt) s += "(" + r + ")";
            else s += r;
        }
        return s;
    }
    case G::Alt: {
        std::string s;
        for (const G& c : g.kids) {
            if (!s.empty()) s += " | ";
            const std::string r = render(c);
            if (effective(c).k == G::Alt) s += "(" + r + ")";
            else s += r;
        }
        return s;
    }
    case G::Rep: {
        std::string r = render(g.kids[0]);
        const G& e = effective(g.kids[0]);
        if (e.k == G::Seq || e.k == G::Alt) r = "(" + r + ")";
        return r + render_rep_suffix(g.min, g.max);
    }
    }
    return "";
}

/* A character in a GBNF class, escaped as the vendored generator escapes it. */
std::string class_char(uint32_t c) {
    if (c == '-' || c == ']' || c == '[' || c == '\\') return std::string("\\") + (char)c;
    if (c == '\n') return "\\n";
    if (c == '\t') return "\\t";
    if (c == '\r') return "\\r";
    if (c >= 0x20 && c <= 0x7E) return std::string(1, (char)c);
    static const char* hex = "0123456789ABCDEF";
    std::string s;
    if (c <= 0xFF) { s = "\\x"; for (int i = 1; i >= 0; --i) s += hex[(c >> (4 * i)) & 15]; }
    else if (c <= 0xFFFF) { s = "\\u"; for (int i = 3; i >= 0; --i) s += hex[(c >> (4 * i)) & 15]; }
    else { s = "\\U"; for (int i = 7; i >= 0; --i) s += hex[(c >> (4 * i)) & 15]; }
    return s;
}

std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp = c;
        size_t   n = 1;
        if (c >= 0xF0)      { cp = c & 0x07; n = 4; }
        else if (c >= 0xE0) { cp = c & 0x0F; n = 3; }
        else if (c >= 0xC0) { cp = c & 0x1F; n = 2; }
        if (i + n > s.size()) break;
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

/* A rule name as the vendored builder spells it: every run of characters outside
 * [a-zA-Z0-9-] becomes one '-'. */
std::string rule_name(const std::string& s) {
    std::string out;
    bool run = false;
    for (char ch : s) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '-';
        if (ok) { out += ch; run = false; }
        else if (!run) { out += '-'; run = true; }
    }
    return out;
}

std::string utf8(const std::vector<uint32_t>& cps) {
    std::string s;
    for (uint32_t c : cps) {
        if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) {
            s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F));
            s += (char)(0x80 | (c & 0x3F));
        } else {
            s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F));
            s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
        }
    }
    return s;
}

/* "Any text not containing one of these", as a repetition over the trie of the delimiters:
 * for each proper prefix, that prefix followed by a character that does not continue it. */
std::string until_pattern(const std::vector<std::string>& delims) {
    if (delims.empty()) return ".*";
    struct Node { std::map<uint32_t, size_t> kids; bool word = false; };
    std::vector<Node> t(1);
    for (const std::string& d : delims) {
        size_t cur = 0;
        for (uint32_t c : codepoints(d)) {
            auto it = t[cur].kids.find(c);
            if (it == t[cur].kids.end()) {
                t.emplace_back();
                t[cur].kids[c] = t.size() - 1;
                cur = t.size() - 1;
            } else {
                cur = it->second;
            }
        }
        t[cur].word = true;
    }
    std::string pattern;
    std::vector<uint32_t> prefix;
    std::function<void(size_t)> walk = [&](size_t n) {
        if (!t[n].word && !t[n].kids.empty()) {
            std::string cls;
            for (const auto& kv : t[n].kids) cls += class_char(kv.first);
            if (!pattern.empty()) pattern += " | ";
            if (!prefix.empty()) pattern += gbnf_format_literal(utf8(prefix)) + " [^" + cls + "]";
            else pattern += "[^" + cls + "]";
        }
        for (const auto& kv : t[n].kids) {
            prefix.push_back(kv.first);
            walk(kv.second);
            prefix.pop_back();
        }
    };
    walk(0);
    return "(" + pattern + ")*";
}

/* The rules the vendored generator emits for its generic JSON parser whenever an argument or a
 * document is typed by a schema. Nothing references them -- the schema's own rules do the
 * constraining -- but the vendored grammar carries them, and so does this one, so the two can be
 * compared byte for byte. */
void add_json_rules(const common_grammar_builder& b) {
    b.add_rule("json-array", "\"[\" space (\"]\" | json-value (\",\" space json-value)* space "
                             "\"]\") space");
    b.add_rule("json-bool", "(\"true\" | \"false\") space");
    b.add_rule("json-null", "\"null\" space");
    b.add_rule("json-number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") "
                              "[+-]? [0-9]+)? space");
    b.add_rule("json-object", "\"{\" space (\"}\" | json-string space \":\" space json-value "
                              "(space \",\" space json-string space \":\" space json-value)* "
                              "space \"}\") space");
    b.add_rule("json-string", "\"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" "
                              "[0-9a-fA-F]{4} ) )* \"\\\"\" space");
    b.add_rule("json-value", "json-object | json-array | json-string | json-number | json-bool | "
                             "json-null");
}

G optspace(const std::string& tag) {
    size_t a = 0, b = tag.size();
    while (a < b && is_ws((unsigned char)tag[a])) ++a;
    while (b > a && is_ws((unsigned char)tag[b - 1])) --b;
    std::vector<G> parts{ eps() };
    for (size_t i = 0; i < a; ++i) parts.push_back(opt(lit(std::string(1, tag[i]))));
    parts.push_back(lit(tag.substr(a, b - a)));
    for (size_t i = b; i < tag.size(); ++i) parts.push_back(opt(lit(std::string(1, tag[i]))));
    return seq(std::move(parts));
}

/* The reasoning block as the vendored generator builds it. */
G reasoning_expr(const ChatFormat& f, const std::vector<std::string>& breaks) {
    const std::string end = trim_ws(f.reasoning_close);
    if (!breaks.empty()) {
        std::vector<std::string> stops{ end };
        stops.insert(stops.end(), breaks.begin(), breaks.end());
        G unclosed = seq({ wrap(leaf(until_pattern(stops))), eps() });
        G closed = seq({ wrap(leaf(until_pattern({ end }))), optspace(f.reasoning_close) });
        if (!f.reasoning_open.empty())
            return opt(alt({ seq({ optspace(f.reasoning_open), unclosed }),
                             seq({ optspace(f.reasoning_open), closed }) }));
        return opt(alt({ unclosed, closed }));
    }
    if (!f.reasoning_open.empty())
        return opt(seq({ optspace(f.reasoning_open), wrap(leaf(until_pattern({ end }))),
                         optspace(f.reasoning_close) }));
    return opt(seq({ wrap(leaf(until_pattern({ end }))), optspace(f.reasoning_close) }));
}

}  /* namespace */

int chat_format_grammar(const ChatFormat& f, const ChatGrammarRequest& req, ChatGrammar* out,
                        std::string* why) {
    *out = ChatGrammar{};
    const ojson none;
    const ojson& tools = req.tools ? *req.tools : none;
    const bool has_tools = f.has_tools() && tools.is_array() && !tools.empty();
    const bool has_rf = req.json_schema && req.json_schema->is_object() && !req.json_schema->empty();
    if (!has_rf && !(has_tools && !f.triggers.empty())) return RAD_OK;

    const std::string opener = f.call_open.empty() ? f.name_prefix : f.call_open;
    std::vector<std::string> breaks;
    if (has_tools && req.extract_reasoning && f.has_reasoning()) breaks = f.reasoning_breaks;

    try {
        out->text = build_grammar([&](const common_grammar_builder& b) {
            for (const auto& tool : tools) {
                if (!tool.is_object() || !tool.contains("type") || tool.at("type") != "function" ||
                    !tool.contains("function"))
                    continue;
                const auto& fn = tool.at("function");
                auto schema = fn.contains("parameters") ? fn.at("parameters") : ojson::object();
                b.resolve_refs(schema);
            }
            if (has_rf) {
                auto schema = *req.json_schema;
                b.resolve_refs(schema);
            }

            if (has_rf) {
                /* The whole reply: what follows the generation prompt, the reasoning block, and
                 * the document, fenced or not. */
                b.add_rule("response-format", b.add_schema("response-format-schema",
                                                           *req.json_schema));
                add_json_rules(b);
                std::string gp = req.generation_prompt;
                if (f.has_reasoning() && !f.reasoning_open.empty()) {
                    const size_t at = gp.find(f.reasoning_open);
                    if (at != std::string::npos) gp = gp.substr(0, at);
                }
                std::vector<G> root{ gp.empty() ? eps() : lit(gp), space() };
                if (f.has_reasoning() && req.extract_reasoning) root.push_back(reasoning_expr(f, breaks));
                root.push_back(space());
                root.push_back(alt({ seq({ lit("```json"), space(), rule_ref("response-format"), space(),
                                           lit("```") }),
                                     rule_ref("response-format") }));
                b.add_rule("root", render(seq(std::move(root))));
                return;
            }

            /* The lazy call grammar: only what follows a trigger. */
            const std::string name_end = f.arg_name_suffix.empty() ? f.arg_value_prefix
                                                                   : f.arg_name_suffix;
            std::vector<G> choices;
            bool typed = false;
            for (const auto& tool : tools) {
                if (!tool.is_object() || !tool.contains("type") || tool.at("type") != "function" ||
                    !tool.contains("function"))
                    continue;
                const auto& fn = tool.at("function");
                const std::string name = fn.at("name");
                auto params = fn.contains("parameters") ? fn.at("parameters") : ojson::object();
                const auto properties = params.contains("properties") ? params.at("properties")
                                                                     : ojson::object();
                std::set<std::string> required;
                if (params.contains("required")) params.at("required").get_to(required);
                common_schema_info info;
                info.resolve_refs(params);

                std::vector<G> req_arms, opt_arms;
                for (const auto& [pname, pschema] : properties.items()) {
                    G value;
                    if (f.other_value != ChatValue::Json || info.resolves_to_string(pschema)) {
                        value = wrap(rule_ref("until-suffix"));
                    } else {
                        typed = true;
                        const std::string sch = b.add_schema(
                            "tool-" + name + "-arg-" + pname + "-schema", pschema);
                        value = alt({ seq({ wrap(leaf(sch)), space() }), wrap(rule_ref("until-suffix")) });
                    }
                    G arm = wrap(seq({ wrap(seq({ lit(f.arg_prefix), wrap(lit(pname)),
                                                  lit(f.arg_name_suffix) })),
                                       lit(f.arg_value_prefix), value,
                                       wrap(lit(f.arg_value_suffix)) }));
                    const std::string rule = rule_name("tool-" + name + "-arg-" + pname);
                    b.add_rule(rule, render(arm));
                    (required.count(pname) ? req_arms : opt_arms).push_back(rule_ref(rule));
                }
                std::vector<G> any = req_arms;
                any.insert(any.end(), opt_arms.begin(), opt_arms.end());
                /* The generic arm for an undeclared name. A tool that declares no parameters
                 * takes no arguments at all, so its arm is never reached and not emitted. */
                if (!any.empty()) {
                    G arm = wrap(seq({ wrap(seq({ lit(f.arg_prefix),
                                                  wrap(leaf(until_pattern({ name_end }))),
                                                  lit(f.arg_name_suffix) })),
                                       lit(f.arg_value_prefix), wrap(rule_ref("until-suffix")),
                                       wrap(lit(f.arg_value_suffix)) }));
                    const std::string rule = rule_name("tool-" + name + "-arg-undeclared");
                    b.add_rule(rule, render(arm));
                    any.push_back(rule_ref(rule));
                }
                G any_arg = alt(any);
                G args = eps();
                if (!req_arms.empty()) args = seq({ any_arg, rep(seq({ space(), any_arg }), 0, -1) });
                else if (!opt_arms.empty()) args = seq({ eps(), rep(seq({ space(), any_arg }), 0, -1) });

                G open = wrap(seq({ lit(f.name_prefix), wrap(lit(name)), lit(f.name_suffix) }));
                G func = seq({ open, eps(), space(), args, space(), wrap(lit(f.call_end)) });
                b.add_rule(rule_name("tool-" + name), render(func));
                choices.push_back(rule_ref(rule_name("tool-" + name)));
            }
            if (typed) add_json_rules(b);
            if (!choices.empty()) b.add_rule("until-suffix", until_pattern({ f.arg_value_suffix }));

            G choice = alt(choices);
            G calls;
            if (!f.call_open.empty()) {
                G wrapped = seq({ lit(f.call_open), space(), choice, space(), lit(f.call_close) });
                calls = req.parallel_tool_calls
                            ? seq({ wrapped, rep(seq({ space(), wrapped }), 0, -1), space() })
                            : seq({ wrapped, space() });
            } else {
                G next = seq({ space(), opt(wrap(leaf(until_pattern({ opener })))), choice });
                calls = req.parallel_tool_calls
                            ? seq({ lit(""), space(), choice, rep(next, 0, -1), space(), lit("") })
                            : seq({ lit(""), space(), choice, space(), lit("") });
            }
            b.add_rule("tool-call", render(calls));
            b.add_rule("root", "tool-call");
        });
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return RAD_E_INVAL;
    }
    out->lazy = !has_rf;
    if (out->lazy) out->trigger_words = f.triggers;
    return RAD_OK;
}

}  /* namespace rad */
