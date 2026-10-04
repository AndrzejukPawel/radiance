/* chatparse.h -- reading a model's reply back as it streams: reasoning, content, tool calls.
 *
 * ONE PASS, ONE PARSER. A reply is fed as it is generated, in pieces of any size, and every byte
 * is examined a bounded number of times: the whole reply costs O(its length), whether it arrives
 * in one piece or one byte at a time. Nothing is re-parsed, nothing is matched with a regular
 * expression, and there is no second pass that repairs what the first one missed -- the malformed
 * shapes a model actually writes are states of this machine, not fallbacks around it.
 *
 * It is driven by a ChatFormat (chatfmt.h) and built from three pieces:
 *
 *   * DELIMITER RECOGNITION is a deterministic Aho-Corasick automaton over the delimiters that can
 *     end the current region -- "</think>" and the call opener inside reasoning, the call opener
 *     in content, the value suffix inside a value. In the automaton's root state the scan skips
 *     straight to the next byte that can begin a delimiter (memchr for the common one-byte case),
 *     so a long run of prose or file content costs a memchr and a copy.
 *   * THE TAG STRUCTURE is a transducer over those regions: call opener, name, arguments, close.
 *   * A JSON-TYPED ARGUMENT is recognised by an incremental JSON scanner with an explicit stack.
 *
 * WHAT IT PRODUCES is exactly what the format's grammar says the reply means, including the
 * places where the grammar is lenient on purpose: an argument the schema does not declare is
 * still an argument, a non-string argument that is not valid JSON is taken as text, a reasoning
 * block the model never closed ends where a call begins, prose before, between and after calls is
 * content. Where the reply leaves the grammar altogether -- a call to a function that is not a
 * tool, text inside a call, a missing close -- it recovers locally and says so (strict()).
 *
 * That includes a call written WITHOUT its wrapper. In a format whose calls are wrapped (Qwen's
 * `<tool_call>\n`) the grammar, and constrained decoding with it, begins only at the wrapper, so a
 * call the model writes as a bare `<function=NAME>`, or with the wrapper run straight into it, is
 * unconstrained -- and it is the one most likely to be lost. In content, name_prefix alone and
 * call_open without its trailing whitespace also open a call, with every tag's surrounding
 * whitespace optional inside it. Because such an opener is also ordinary text, it becomes a call
 * only once the name tag is closed and an argument or the call's close follows; until then it is
 * held (a bounded number of bytes), and if it is not a call it is content.
 *
 * THE STREAMING CONTRACT: what has been emitted is final. A delta is never taken back, so every
 * emitted prefix of content, reasoning or arguments is a prefix of the finished value, and the
 * concatenated deltas ARE the finished values. That forces some bytes to be held until their
 * meaning is known, and the holds are bounded:
 *
 *   * a possible delimiter prefix -- at most the longest delimiter, less one byte;
 *   * an unfinished UTF-8 sequence -- at most three bytes, because a delta goes into a JSON string;
 *   * whitespace whose fate depends on what follows -- the trailing whitespace of content, which
 *     is trimmed if nothing follows it, and the whitespace that opens a reasoning block, which is
 *     dropped if the block holds nothing else;
 *   * a tag's own name while it is being read, since a name is only a name once it is closed;
 *   * a JSON-typed argument value until its closing delimiter: `30` and `30 seconds` differ from
 *     their first byte (a number, or the string "30 seconds"), so the value is emitted whole once
 *     its type is known. A value that stops being JSON part-way is streamed as text from there.
 *
 * Streaming emits in the reply's order; a call's arguments are streamed as they are parsed. */
#pragma once
#include "rad_internal.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad {

struct ChatFormat;

/* One increment of a parsed reply. Consecutive increments of the same kind are merged. */
struct ReplyEvent {
    enum Kind : uint8_t {
        Reasoning,   /* text: a reasoning delta */
        Content,     /* text: a content delta */
        CallBegin,   /* text: the function name; `call` is the new call's index */
        CallArgs,    /* text: an arguments delta for call `call` */
        CallEnd,     /* call `call` closed */
    };
    Kind        kind = Content;
    uint32_t    call = 0;
    std::string text;
};

struct ReplyCall {
    std::string name;
    std::string arguments;    /* JSON text, as the OpenAI API carries it */
    bool        closed = false;
};

/* A reply, taken apart. After finish() it is the whole reply; before, what has been emitted. */
struct ReplyMessage {
    std::string            content;
    std::string            reasoning;
    std::vector<ReplyCall> calls;
};

/* What held bytes are waiting on, by kind. See the header comment for the bound on each. */
struct ReplyHeld {
    size_t delimiter = 0;     /* a possible delimiter prefix, and an unfinished UTF-8 sequence */
    size_t space     = 0;     /* whitespace that may yet be trimmed */
    size_t structure = 0;     /* a tag name being read, or a JSON value awaiting its type */
};

/* The per-request inputs. */
struct ReplyOptions {
    /* The assistant prefix the template already rendered. It decides whether the reply starts
     * inside a reasoning block (ChatFormat::reasoning_in_prompt). */
    std::string generation_prompt;
    /* reasoning_format "none" keeps a reasoning block inline in the content. */
    bool        extract_reasoning = true;
    /* The request's tools, OpenAI-shaped; null or empty parses no calls, and call syntax in the
     * reply is then content. Read once, when the parser is created. */
    const nlohmann::ordered_json* tools = nullptr;
    bool        parallel_tool_calls = true;
    /* A response_format was requested: the reply is one JSON document, optionally fenced as
     * ```json ... ```, and the document is the content. */
    bool        response_format = false;
    /* The model's turn terminators. The decoder renders one only when the format needs it, and
     * one left on the end of content is not part of the answer. */
    std::vector<std::string> terminators;
};

class ReplyStream;

/* A format compiled once: its automata and its delimiters. Shared by every request. */
class ReplyFormat {
public:
    ~ReplyFormat();
    /* RAD_OK, or RAD_E_UNSUPPORTED with *why when chat_format_check refuses the format. */
    static int compile(const ChatFormat& f, std::shared_ptr<const ReplyFormat>* out,
                       std::string* why);
    const ChatFormat& format() const;

    struct Impl;
    const Impl& impl() const { return *impl_; }
private:
    ReplyFormat();
    std::unique_ptr<Impl> impl_;
};

/* One request's parser: the format bound to the tools the request sent and the prompt it
 * rendered. Immutable once made; each choice opens its own stream from it. */
class ReplyParser {
public:
    ~ReplyParser();
    /* RAD_OK, or RAD_E_INVAL with *why when a tool cannot be read unambiguously in this format --
     * a function or parameter name containing the delimiter that ends it -- and
     * RAD_E_UNSUPPORTED when the request sends tools and the format has no tool calls. */
    static int create(std::shared_ptr<const ReplyFormat> fmt, const ReplyOptions& opt,
                      std::shared_ptr<const ReplyParser>* out, std::string* why);
    std::unique_ptr<ReplyStream> open() const;
    const ChatFormat& format() const;
    /* The most bytes ReplyHeld::delimiter can reach for this request: a delimiter or terminator
     * prefix in each place one can be pending at once, a value's undecided head, and an
     * unfinished UTF-8 sequence on each output. A constant of the format, not of the reply. */
    size_t hold_bound() const;

    struct Impl;
    const Impl& impl() const { return *impl_; }
private:
    ReplyParser();
    std::unique_ptr<Impl> impl_;
};

class ReplyStream {
public:
    ~ReplyStream();

    /* Append bytes. The increments they complete are appended to *out (which may be null). */
    void feed(std::string_view bytes, std::vector<ReplyEvent>* out);
    /* The reply is complete. Whatever was held is resolved and emitted. Idempotent. */
    void finish(std::vector<ReplyEvent>* out);

    const ReplyMessage& message() const;
    /* False once the reply left the format's grammar and was read by a recovery rule. */
    bool strict() const;
    ReplyHeld held() const;

    struct Impl;
private:
    friend class ReplyParser;
    ReplyStream();
    std::unique_ptr<Impl> impl_;
};

/* Parse a complete reply in one call. */
ReplyMessage reply_parse(const ReplyParser& p, std::string_view text, bool* strict = nullptr);

}  /* namespace rad */
