/* toolparse_legacy.h -- a whole-reply parser built on llama.cpp's PEG auto-parser, kept as the
 * ORACLE the differential tests compare the streaming parser (core/text/chatparse.h) against. It
 * is test code: it is built into the test binaries and the acceptance tool, never into the engine.
 *
 * It is the whole legacy path, layer for layer, because its answer is what all of them produce
 * together:
 *
 *   * ToolParser -- llama.cpp's PEG parser, generated per request by the vendored auto-parser from
 *     the chat template (vendor/llama/common/chat-auto-parser*.cpp) and run over the WHOLE reply;
 *   * legacy_parse -- the bridge's reading of that: a turn terminator trimmed off the content,
 *     and, when the strict parse produced no call but the content still carries call syntax,
 *     calls lifted out of the text by hand;
 *   * LegacyDeltaStream -- a streaming diff that re-parses the accumulated reply on every chunk and
 *     sends what has grown.
 */
#pragma once
#include "rad_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {

struct ChatPrompt;

namespace legacy {

struct ToolCall {
    std::string name;
    std::string arguments;
    std::string id;
};

struct ParsedReply {
    std::string           content;
    std::string           reasoning_content;
    std::vector<ToolCall> tool_calls;
};

/* The by-hand lifter: `<function name="N"><param name="K">V</param></function>` and
 * `<tool_call><function=N><parameter=K>V</parameter></function></tool_call>` spans read out of
 * `text` whatever the model's format, appended to `out` and removed. */
size_t toolparse_lift_xml_calls(std::string& text, std::vector<ToolCall>& out);

/* The earliest byte of `text` that could begin a call in either spelling, or npos. */
size_t toolparse_find_call_open(const std::string& text);

/* The vendored PEG parser for one rendered request. */
class ToolParser {
public:
    ToolParser();
    ~ToolParser();
    int init(const ChatPrompt& p, const std::string& reasoning_format = "auto",
             bool parse_tool_calls = true);
    /* RAD_OK, or RAD_E_FORMAT when the reply does not match the format at all. */
    int parse(const std::string& text, bool is_partial, ParsedReply* out) const;
    bool ready() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* What the legacy path answers for a reply, and which of its layers produced the answer. */
struct Answer {
    std::string content, reasoning;
    std::vector<ToolCall> calls;
    bool strict_failed = false;   /* the PEG parse failed; the whole reply became content */
    size_t lifted = 0;            /* calls the by-hand lifter took out of the content */
};

/* The bridge's reading of a reply: the PEG parse, the terminator trim and, for a request that
 * offered tools, the lifter. `is_partial` is the streaming path's question. */
Answer legacy_parse(const ToolParser& tp, const std::string& text, bool is_partial,
                    bool want_calls, const std::vector<std::string>& terminators);

/* The legacy streaming diff over the accumulated text: returns how many bytes of deltas it sends,
 * and fills the concatenation of what it sent. */
class LegacyDeltaStream {
public:
    LegacyDeltaStream(const ToolParser& tp, bool want_calls, std::vector<std::string> terminators)
        : tp_(tp), want_calls_(want_calls), terms_(std::move(terminators)) {}
    size_t push(const std::string& full_text, bool is_final);
    const std::string& content() const { return content_; }
    const std::string& reasoning() const { return reasoning_; }
    const std::vector<std::string>& arguments() const { return args_; }
    const std::vector<std::string>& names() const { return names_; }
private:
    const ToolParser& tp_;
    bool want_calls_;
    std::vector<std::string> terms_;
    std::string content_, reasoning_;
    std::vector<std::string> names_, args_;
};

}  /* namespace legacy */
}  /* namespace rad */
