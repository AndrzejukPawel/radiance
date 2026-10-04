#include "toolparse_legacy.h"
#include "text/chat.h"

#include "llama/common/chat.h"
#include "llama/common/peg-parser.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>

namespace rad {
namespace legacy {

/* ================================================================== the PEG parser */

struct ToolParser::Impl {
    common_chat_parser_params params;
    bool                      ready = false;
};

ToolParser::ToolParser() : impl_(std::make_unique<Impl>()) {}
ToolParser::~ToolParser() = default;

bool ToolParser::ready() const { return impl_ && impl_->ready; }

int ToolParser::init(const ChatPrompt& p, const std::string& reasoning_format,
                     bool parse_tool_calls) {
    impl_->ready = false;
    impl_->params = common_chat_parser_params{};
    impl_->params.format = (common_chat_format)p.format;
    impl_->params.generation_prompt = p.generation_prompt;
    impl_->params.parse_tool_calls = parse_tool_calls;
    impl_->params.reasoning_format = common_reasoning_format_from_name(reasoning_format);
    impl_->params.reasoning_in_content =
        impl_->params.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    if (!p.parser.empty()) {
        try {
            impl_->params.parser.load(p.parser);
        } catch (const std::exception&) {
            return RAD_E_FORMAT;
        }
    }
    impl_->ready = true;
    return RAD_OK;
}

int ToolParser::parse(const std::string& text, bool is_partial, ParsedReply* out) const {
    if (!ready()) return RAD_E_STATE;
    if (!out) return RAD_E_INVAL;
    out->content.clear();
    out->reasoning_content.clear();
    out->tool_calls.clear();

    common_chat_msg msg;
    try {
        if (!impl_->params.parser.empty())
            msg = common_chat_peg_parse(impl_->params.parser, text, is_partial, impl_->params);
        else
            msg = common_chat_parse(text, is_partial, impl_->params);
    } catch (const std::exception&) {
        return RAD_E_FORMAT;
    }
    out->content           = std::move(msg.content);
    out->reasoning_content = std::move(msg.reasoning_content);
    for (common_chat_tool_call& c : msg.tool_calls)
        out->tool_calls.push_back({ std::move(c.name), std::move(c.arguments), std::move(c.id) });
    return RAD_OK;
}

/* ================================================================== the by-hand lifter
 *
 * Two spellings, whatever the model's own format:
 *
 *   attr  <function name="N"><param name="K">V</param></function>
 *   eq    <tool_call><function=N><parameter=K>V</parameter></function></tool_call>
 */
namespace {

struct XmlCallSyntax {
    const char* wrap_open;
    const char* wrap_close;
    const char* fn_open;
    const char* fn_name_end;
    const char* fn_close;
    const char* arg_open;
    const char* arg_name_end;
    const char* arg_close;
    bool        trim_value;
};

const XmlCallSyntax kAttr = { "", "", "<function name=\"", "\">", "</function>",
                              "<param name=\"", "\">", "</param>", false };
const XmlCallSyntax kEq   = { "<tool_call>", "</tool_call>", "<function=", ">", "</function>",
                              "<parameter=", ">", "</parameter>", true };

size_t skip_ws(const std::string& s, size_t p) {
    while (p < s.size() && (unsigned char)s[p] <= ' ') ++p;
    return p;
}

std::string trimmed(const std::string& v) {
    size_t a = 0, b = v.size();
    while (a < b && (unsigned char)v[a] <= ' ') ++a;
    while (b > a && (unsigned char)v[b - 1] <= ' ') --b;
    return v.substr(a, b - a);
}

size_t read_call(const std::string& s, size_t f, const XmlCallSyntax& x,
                 std::string* name, nlohmann::ordered_json* args) {
    const std::string fn_open(x.fn_open), fn_name_end(x.fn_name_end), fn_close(x.fn_close);
    const std::string arg_open(x.arg_open), arg_name_end(x.arg_name_end), arg_close(x.arg_close);

    if (s.compare(f, fn_open.size(), fn_open) != 0) return std::string::npos;
    const size_t ns = f + fn_open.size();
    const size_t ne = s.find(fn_name_end, ns);
    if (ne == std::string::npos) return std::string::npos;
    *name = s.substr(ns, ne - ns);
    if (name->empty() || name->find('<') != std::string::npos) return std::string::npos;

    *args = nlohmann::ordered_json::object();
    size_t p = ne + fn_name_end.size();
    while (true) {
        p = skip_ws(s, p);
        if (s.compare(p, fn_close.size(), fn_close) == 0) return p + fn_close.size();
        if (s.compare(p, arg_open.size(), arg_open) != 0) return std::string::npos;

        const size_t ks = p + arg_open.size();
        const size_t ke = s.find(arg_name_end, ks);
        if (ke == std::string::npos) return std::string::npos;
        const std::string key = s.substr(ks, ke - ks);
        if (key.empty() || key.find('<') != std::string::npos) return std::string::npos;

        size_t vs = ke + arg_name_end.size();
        std::string val;
        if (s.compare(vs, 9, "<![CDATA[") == 0) {
            const size_t ce = s.find("]]>", vs + 9);
            if (ce == std::string::npos) return std::string::npos;
            val = s.substr(vs + 9, ce - (vs + 9));
            p = ce + 3;
            if (s.compare(p, arg_close.size(), arg_close) != 0) return std::string::npos;
            p += arg_close.size();
        } else {
            const size_t bound = std::min(s.find(arg_open, vs), s.find(fn_close, vs));
            const size_t ve = s.find(arg_close, vs);
            if (ve != std::string::npos && (bound == std::string::npos || ve <= bound)) {
                val = s.substr(vs, ve - vs);
                p = ve + arg_close.size();
            } else if (bound != std::string::npos) {
                const size_t lt = bound > vs ? s.rfind("</", bound - 1) : std::string::npos;
                if (lt != std::string::npos && lt >= vs) {
                    const size_t gt = s.find('>', lt);
                    if (gt == std::string::npos || gt >= bound) return std::string::npos;
                    val = s.substr(vs, lt - vs);
                    p = gt + 1;
                } else {
                    val = s.substr(vs, bound - vs);
                    p = bound;
                }
            } else {
                return std::string::npos;
            }
        }
        (*args)[key] = x.trim_value ? trimmed(val) : val;
    }
}

std::string mint_call_id() {
    static std::atomic<uint64_t> seq{0};
    char buf[40];
    std::snprintf(buf, sizeof buf, "call_%023llx",
                  (unsigned long long)seq.fetch_add(1, std::memory_order_relaxed));
    return std::string(buf);
}

}  /* namespace */

size_t toolparse_find_call_open(const std::string& s) {
    static const char* const kMarks[] = { "<tool_call>", "<function name=\"", "<function=" };
    size_t at = std::string::npos;
    for (const char* m : kMarks) at = std::min(at, s.find(m));
    if (at != std::string::npos) return at;
    for (const char* m : kMarks) {
        const size_t len = std::strlen(m);
        for (size_t k = std::min(len - 1, s.size()); k >= 1; --k) {
            if (s.compare(s.size() - k, k, m, k) == 0) {
                at = std::min(at, s.size() - k);
                break;
            }
        }
    }
    return at;
}

size_t toolparse_lift_xml_calls(std::string& s, std::vector<ToolCall>& out) {
    std::string kept;
    size_t cursor = 0, scan = 0, found = 0;
    while (true) {
        const size_t rel = toolparse_find_call_open(s.substr(scan));
        const size_t at = rel == std::string::npos ? std::string::npos : scan + rel;
        if (at == std::string::npos) break;

        const XmlCallSyntax* x = nullptr;
        size_t f = at, wrap = std::string::npos;
        if (s.compare(at, strlen(kEq.wrap_open), kEq.wrap_open) == 0) {
            wrap = at;
            f = skip_ws(s, at + strlen(kEq.wrap_open));
            x = &kEq;
        } else if (s.compare(at, strlen(kAttr.fn_open), kAttr.fn_open) == 0) {
            x = &kAttr;
        } else if (s.compare(at, strlen(kEq.fn_open), kEq.fn_open) == 0) {
            x = &kEq;
        }
        if (!x) { scan = at + 1; continue; }

        std::string name;
        nlohmann::ordered_json args;
        size_t end = read_call(s, f, *x, &name, &args);
        if (end == std::string::npos) { scan = at + 1; continue; }
        if (wrap != std::string::npos) {
            const size_t w = skip_ws(s, end);
            if (s.compare(w, strlen(kEq.wrap_close), kEq.wrap_close) != 0) { scan = at + 1; continue; }
            end = w + strlen(kEq.wrap_close);
        }

        const size_t span_start = (wrap != std::string::npos) ? wrap : f;
        kept.append(s, cursor, span_start - cursor);
        ToolCall tc;
        tc.id        = mint_call_id();
        tc.name      = name;
        tc.arguments = args.dump();
        out.push_back(std::move(tc));
        cursor = end;
        scan   = end;
        ++found;
    }
    if (found == 0) return 0;
    kept.append(s, cursor, std::string::npos);
    while (!kept.empty() && (unsigned char)kept.back() <= ' ') kept.pop_back();
    s = std::move(kept);
    return found;
}

/* ================================================================== the bridge */

static void strip_terminator(std::string& s, const std::vector<std::string>& terms) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    for (const std::string& t : terms) {
        if (!t.empty() && s.size() >= t.size() &&
            s.compare(s.size() - t.size(), t.size(), t) == 0) {
            s.erase(s.size() - t.size());
            return;
        }
    }
}

Answer legacy_parse(const ToolParser& tp, const std::string& text, bool is_partial,
                    bool want_calls, const std::vector<std::string>& terminators) {
    Answer a;
    ParsedReply r;
    if (tp.parse(text, is_partial, &r) < 0) {
        a.strict_failed = true;
        a.content = text;
    } else {
        a.content   = std::move(r.content);
        a.reasoning = std::move(r.reasoning_content);
        a.calls     = std::move(r.tool_calls);
    }
    strip_terminator(a.content, terminators);
    if (!want_calls || a.content.empty()) return a;
    const size_t at = toolparse_find_call_open(a.content);
    if (at == std::string::npos) return a;
    if (is_partial) { a.content.resize(at); return a; }
    if (!a.calls.empty()) return a;
    a.lifted = toolparse_lift_xml_calls(a.content, a.calls);
    return a;
}

/* ================================================================== the streaming diff */

namespace {

size_t utf8_cut(const std::string& s) {
    size_t n = s.size();
    size_t k = n, back = 0;
    while (k > 0 && back < 4) {
        const unsigned char c = (unsigned char)s[k - 1];
        if ((c & 0xC0) == 0x80) { --k; ++back; continue; }
        size_t need = 1;
        if ((c & 0x80) == 0x00) need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return n;
        return n - (k - 1) >= need ? n : k - 1;
    }
    return n;
}

bool extends(const std::string& sent, const std::string& cand) {
    return cand.size() >= sent.size() && cand.compare(0, sent.size(), sent) == 0;
}

std::string chunk(const nlohmann::ordered_json& delta) {
    nlohmann::ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = delta;
    ch["finish_reason"] = nullptr;
    nlohmann::ordered_json o;
    o["id"] = "chatcmpl-legacy";
    o["object"] = "chat.completion.chunk";
    o["created"] = 0;
    o["model"] = "legacy";
    o["choices"] = nlohmann::ordered_json::array({ ch });
    return "data: " + o.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace) +
           "\n\n";
}

}  /* namespace */

size_t LegacyDeltaStream::push(const std::string& full_text, bool is_final) {
    Answer p = legacy_parse(tp_, full_text, !is_final, want_calls_, terms_);
    size_t out = 0;
    if (p.reasoning.size() > reasoning_.size() && extends(reasoning_, p.reasoning)) {
        const size_t safe = is_final ? p.reasoning.size() : utf8_cut(p.reasoning);
        if (safe > reasoning_.size()) {
            nlohmann::ordered_json d;
            d["reasoning_content"] = p.reasoning.substr(reasoning_.size(), safe - reasoning_.size());
            out += chunk(d).size();
            reasoning_ = p.reasoning.substr(0, safe);
        }
    }
    if (p.content.size() > content_.size() && extends(content_, p.content)) {
        const size_t safe = is_final ? p.content.size() : utf8_cut(p.content);
        if (safe > content_.size()) {
            nlohmann::ordered_json d;
            d["content"] = p.content.substr(content_.size(), safe - content_.size());
            out += chunk(d).size();
            content_ = p.content.substr(0, safe);
        }
    }
    for (size_t i = 0; i < p.calls.size(); ++i) {
        const ToolCall& tc = p.calls[i];
        if (tc.name.empty()) continue;
        if (names_.size() <= i) {
            names_.resize(i + 1);
            args_.resize(i + 1);
            names_[i] = tc.name;
            nlohmann::ordered_json f;
            f["name"] = tc.name;
            f["arguments"] = "";
            nlohmann::ordered_json e;
            e["index"] = (int64_t)i;
            e["function"] = f;
            nlohmann::ordered_json d;
            d["tool_calls"] = nlohmann::ordered_json::array({ e });
            out += chunk(d).size();
        }
        if (!extends(args_[i], tc.arguments)) continue;
        const size_t safe = is_final ? tc.arguments.size() : utf8_cut(tc.arguments);
        if (safe <= args_[i].size()) continue;
        nlohmann::ordered_json f;
        f["arguments"] = tc.arguments.substr(args_[i].size(), safe - args_[i].size());
        nlohmann::ordered_json e;
        e["index"] = (int64_t)i;
        e["function"] = f;
        nlohmann::ordered_json d;
        d["tool_calls"] = nlohmann::ordered_json::array({ e });
        out += chunk(d).size();
        args_[i] = tc.arguments.substr(0, safe);
    }
    return out;
}

}  /* namespace legacy */
}  /* namespace rad */
