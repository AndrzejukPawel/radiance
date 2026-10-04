/* chatfmt.h -- how a model spells a turn: reasoning, tool calls, arguments. As DATA.
 *
 * Everything the engine knows about a model's reply syntax is in one plain struct, ChatFormat,
 * and everything that depends on that syntax is DERIVED from it: the streaming reply parser
 * (chatparse.h), the lazy tool-call grammar and the literal words that trigger it (below), and the
 * tokens the decoder must render as text so the parser can see the structure. A new model is
 * supported by stating its format once.
 *
 * TWO PRODUCERS, IN ORDER OF PRECEDENCE.
 *
 *   1. The architecture plugin, through the optional `rad_arch_chat_format` export
 *      (abi/rad_builder.h). A plugin that knows its model's syntax says so and nothing is guessed.
 *   2. Analysis of the chat template. llama.cpp's differential analyser renders the template with
 *      and without a reasoning block, a call and its arguments, and diffs the renders to find the
 *      delimiters (vendor/llama/common/chat-diff-analyzer.cpp). Its derived MARKERS are read here;
 *      its PEG parser is not used.
 *
 * If neither yields a format the parser can run, the model still chats -- the reply is content --
 * but a request that sends tools is refused, naming why. There is no second parser to fall back
 * to: a reply format the engine cannot describe is a reply it cannot read, and saying so at
 * admission is better than returning a call as prose.
 *
 * THE FAMILY THIS DESCRIBES is the tagged one, which covers every model this engine serves:
 *
 *   Qwen3.6 / Qwen3.8    <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n
 *                        </function>\n</tool_call>
 *   MiniCPM5             <function name="NAME"><param name="ARG">VALUE</param></function>
 *
 * A call's name is written between a prefix and a suffix and it ends with its own close; each
 * argument is a name between a prefix and a suffix and a value ending at its own suffix; calls may
 * be wrapped one by one. Arguments are typed by the tool's JSON schema: a string-typed argument is
 * the raw text (optionally CDATA-wrapped), any other type is JSON, and text that does not parse as
 * JSON is taken raw -- a model writing into an XML tag writes Python as often as JSON.
 *
 * The analyser also recognises formats outside this family (a JSON object per call, JSON
 * arguments inside a tag, call ids). For those chat_format_from_template keeps the reasoning split
 * and names why tool calls are refused, and a model that needs one gets it by adding the shape to
 * the parser, not by a fallback. */
#pragma once
#include "rad_internal.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct RadChatFormat;

namespace rad {

/* How a value is written between its delimiters. The numbering is the ABI's (RAD_CHAT_VALUE_*). */
enum class ChatValue : uint32_t {
    Raw   = 0,   /* the text, verbatim */
    Cdata = 1,   /* the text, unless wrapped in <![CDATA[ ]]>, which is punctuation and removed */
    Json  = 2,   /* a JSON value; text that does not parse as one is taken raw */
};

/* Whether the reply starts inside a reasoning block. The numbering is the ABI's. */
enum class ChatThink : uint32_t {
    /* Search the rendered generation prompt for `reasoning_open`: whatever follows its first
     * occurrence is the start of the reply's own text. A template that opens the block for the
     * model (Qwen writes "<think>\n" at the end of the prompt) and one that closes it again when
     * thinking is off ("<think>\n\n</think>\n\n") are both read correctly this way. */
    FromPrompt = 0,
    Always     = 1,  /* the reply is inside a reasoning block from its first byte */
    Never      = 2,  /* the model opens its own block, if any */
};

struct ChatFormat {
    std::string name;                 /* for logs: "plugin:<arch>" or "template" */

    /* ------------------------------------------------------------------ reasoning
     * A block runs from `reasoning_open` to `reasoning_close`. Whitespace at either end of either
     * tag is optional when matching, so "<think>\n" also matches "<think>". An empty close means
     * the format has no reasoning block. An empty open with a close means the reply starts in one
     * and the close ends it. */
    std::string reasoning_open;
    std::string reasoning_close;
    /* Literals that end an UNCLOSED block when a call starts there. A model that forgets to close
     * its reasoning and goes straight to a call must not have the call swallowed as reasoning.
     * Must name the call opener (`call_open`) when non-empty. */
    std::vector<std::string> reasoning_breaks;
    ChatThink   reasoning_in_prompt = ChatThink::FromPrompt;

    /* ------------------------------------------------------------------ tool calls
     * `name_prefix` is what opens a call (after `call_open`, when there is one). Empty means the
     * format has no tool calls. */
    std::string section_open, section_close;  /* around ALL calls; not read yet, must be empty */
    /* Between two calls beyond whitespace. Tagged calls are self-delimiting, so this is empty;
     * a JSON-array format would say ", ". Not read yet, must be empty. */
    std::string call_separator;
    /* Around EACH call: "<tool_call>\n", "</tool_call>". When set, a call written without it --
     * name_prefix alone -- is still read as a call, leniently (chatparse.h). */
    std::string call_open, call_close;
    std::string name_prefix, name_suffix;     /* "<function=", ">\n" */
    std::string call_end;                     /* "</function>\n" */

    std::string arg_prefix;                   /* "<parameter=" */
    std::string arg_name_suffix;              /* ">\n"   (MiniCPM: empty, the value prefix ends it) */
    std::string arg_value_prefix;             /* ""      (MiniCPM: "\">") */
    std::string arg_value_suffix;             /* "</parameter>\n" */

    ChatValue   string_value = ChatValue::Cdata;  /* a string-typed argument: Raw or Cdata */
    ChatValue   other_value  = ChatValue::Json;   /* any other type: Json */
    /* One newline at each end of a value is punctuation: the template writes the value on its own
     * lines, and exactly the first and last newline belong to the tags. */
    bool        value_newlines = true;

    /* The literal words that start constrained decoding of a call. Empty means no lazy grammar:
     * the reply is unconstrained and the parser reads whatever arrives. */
    std::vector<std::string> triggers;

    bool has_reasoning() const { return !reasoning_close.empty(); }
    bool has_tools() const { return !name_prefix.empty(); }
};

/* Can the streaming parser run this format? RAD_OK, or RAD_E_UNSUPPORTED with *why naming the
 * field. A format with no tool calls is valid (reasoning split only); a format with tool calls
 * must describe them completely enough that every delimiter is unambiguous. */
int chat_format_check(const ChatFormat& f, std::string* why);

/* The plugin's declaration, copied out of the ABI struct. Fields past the plugin's
 * `struct_size` read as their zero. Returns RAD_E_INVAL with *why when the struct is unusable. */
int chat_format_from_abi(const RadChatFormat* in, ChatFormat* out, std::string* why);

/* Derive the format from a chat template by the vendored differential analysis.
 *
 * RAD_OK with *out filled. A template whose tool calls the analysis found but this family cannot
 * express still yields its reasoning split, with the tool fields empty and *tools_refused saying
 * why -- the model chats, and a request with tools is refused with that sentence. RAD_E_UNSUPPORTED
 * with *why when the template routes to one of llama.cpp's hand-written handlers, whose formats
 * the analysis does not describe at all; RAD_E_FORMAT when the template does not compile. An
 * empty source is the ChatML fallback, which has neither reasoning nor calls. */
int chat_format_from_template(const std::string& jinja, const std::string& bos,
                              const std::string& eos, ChatFormat* out,
                              std::string* tools_refused, std::string* why);

/* Human-readable one-liner for the startup log. */
std::string chat_format_describe(const ChatFormat& f);

/* ================================================================== grammar
 *
 * The GBNF that constrains a reply to the format, generated from the description. For a tool
 * request it is LAZY: the reply is free until one of `triggers` appears, and from there the
 * grammar admits the call syntax with each argument constrained by its schema. For a response
 * format it is the whole reply. Rule names and structure follow the vendored generator so the
 * two produce the same grammar for a template-analysed format. */
struct ChatGrammarRequest {
    const nlohmann::ordered_json* tools = nullptr;        /* the request's tools, or null */
    const nlohmann::ordered_json* json_schema = nullptr;  /* a response format's schema, or null */
    std::string generation_prompt;
    bool        parallel_tool_calls = true;
    bool        extract_reasoning = true;  /* reasoning_format != "none" */
};

struct ChatGrammar {
    std::string text;                     /* empty when the request is unconstrained */
    bool        lazy = false;
    std::vector<std::string> trigger_words;
};

/* RAD_OK, or RAD_E_INVAL with *why when a tool or schema cannot be expressed. */
int chat_format_grammar(const ChatFormat& f, const ChatGrammarRequest& req, ChatGrammar* out,
                        std::string* why);

/* The markers the decoder must render as text for the parser to see them: a format built from
 * control tokens reaches the parser with its structure gone otherwise, because the decoder
 * suppresses control tokens. Every non-empty delimiter, whitespace-trimmed, deduplicated, in the
 * order the vendored analyser lists them -- a marker that is not a single token is harmless, the
 * decoder only compares it with the spellings of control tokens. */
std::vector<std::string> chat_format_markers(const ChatFormat& f);

}  /* namespace rad */
