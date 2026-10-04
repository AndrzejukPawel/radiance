/* core/server/oai.h -- the OpenAI schema, in and out.
 *
 * Two rules shape everything here.
 *
 * Errors are values. Nothing in this file throws; every entry point returns RAD_OK or a negative
 * status and fills an ApiError that already knows its HTTP status, its `param`, and a message
 * that names what was wrong. The HTTP layer turns that into a status code and a body. An
 * exception crossing the transport is a connection closed with no body, which is the least
 * diagnosable thing a client can be handed.
 *
 * A parameter we do not support is a 400 that names it. Not a silent ignore. A client that sends
 * `logit_bias` and gets a 200 has been told its bias was applied, and the resulting bug report
 * ("the model ignores my bias") is unanswerable from the server side. The cost of strictness is
 * that a client sending a field we have not heard of gets a 400 rather than a shrug; the
 * allowlist below is where the genuinely inert OpenAI metadata fields (`user`, `store`, ...)
 * are named, so that cost falls only on fields that actually mean something.
 */
#pragma once
#include "iface.h"
#include "rad_core.h"
#include "sink.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* ================================================================== errors */

struct ApiError {
    int         status = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param;      /* "" when the error is not about one field */
    std::string code;       /* "" when there is nothing more specific to say */

    std::string dump() const;    /* the OpenAI error envelope, as a JSON string */
};

ApiError err_bad(const std::string& param, const std::string& why);
ApiError err_unsupported(const std::string& param, const std::string& why);
ApiError err_not_implemented(const std::string& what);

/* SERIALISE OUT OF LINE. nlohmann's dump() inlines a shared_ptr release that GCC 16 speculatively
 * devirtualises onto std::mutex and then reports as an out-of-bounds destroy -- a false positive
 * in a vendored header, which appears or not depending on how deeply the CALLING function
 * inlined. That makes "is this file warning-clean" a property of how short its handlers are,
 * which is not something anybody can act on. One definition, in one translation unit. */
std::string json_dump(const json& j);
ApiError err_overloaded(const std::string& why);

/* ================================================================== a parsed request */

enum class Endpoint { Chat, Completion, Embedding };

struct PromptInput {
    std::vector<int32_t> tokens;
    std::string          text;    /* the rendered prompt; needed for `echo` and for the logs */
    /* The media parts, processed and placed; null for a text prompt (Request::media). */
    std::shared_ptr<mm::PromptMedia> media;
};

struct OaiRequest {
    Endpoint    endpoint = Endpoint::Chat;
    std::string model;
    std::string id;               /* "chatcmpl-..." / "cmpl-..." */
    int64_t     created = 0;      /* unix seconds */

    bool stream = false;
    bool include_usage = false;   /* stream_options.include_usage */

    int             n = 1;
    SamplingParams  sp;
    int32_t         max_tokens = 0;
    std::vector<std::string> stop;
    /* Generate past end-of-text, to max_tokens. See Request::ignore_eos. */
    bool        ignore_eos = false;
    int             priority = 0;

    std::vector<PromptInput> prompts;   /* /v1/completions accepts a batch; chat has exactly one */
    bool echo = false;

    int  n_logprobs = 0;          /* 0 = off */
    bool want_logprobs = false;

    bool tools_present = false;
    bool parse_tools = false;
    /* The reply parser this request's render produced. Per request and not per server: it is the
     * model's format bound to the tools THIS call sent, so two concurrent requests against the
     * same model can legitimately need different parsers. */
    std::shared_ptr<IReplyParser> reply_parser;

    /* The control tokens this request's reply must be decoded AS TEXT, so `tool_parser` can see
     * the structure it is looking for. See ITokenizer::detokenizer. */
    std::vector<std::string> preserved_tokens;

    /* /v1/embeddings */
    bool base64_embeddings = false;

    int  n_choices() const { return (int)prompts.size() * n; }
};

/* What the server needs to know about itself to validate a request. */
struct OaiLimits {
    std::string model_id = "radiance";
    int64_t     max_ctx = 0;              /* 0 = unknown, no length check */
    int64_t     default_max_tokens = 512;
    int         max_n = 8;
    bool        allow_image = false;
    bool        allow_video = false;
    bool        allow_audio = false;
    bool        supports_logprobs = false;

    /* THE DEPLOYMENT'S DEFAULT REASONING EFFORT, empty for "whatever the template's own default
     * is". It exists because the template's default is not always a servable one: Qwen3.8's is
     * `xhigh`, which on a short task reliably spends the whole token budget inside <think> and
     * returns a turn with EMPTY content and finish_reason "length" -- an agent turn that carries
     * no answer and no tool call. The per-request field is the right control and most OpenAI
     * clients never send it, so without this an operator serving such a client cannot reach the
     * dial at all. Seeded only when the request says nothing, so a caller always wins. */
    std::string default_reasoning_effort;

    /* WHAT A REQUEST THAT SENDS NO SAMPLER FIELDS IS SERVED WITH, field by field.
     *
     * The struct's own defaults are an untruncated sampler -- temperature 1, top_k off, top_p 1
     * -- which is the right thing to fall back to and the wrong thing to serve a checkpoint
     * with: a model tuned against a truncated tail, sampled without one, will occasionally draw
     * a token from far down its distribution. End-of-turn is one of those tokens, and drawing it
     * early ends an agent's turn with prose and no tool call, which reads as the server dropping
     * the request. The deployment resolves this from what the checkpoint itself states and hands
     * the result over here.
     *
     * Only the sampler knobs are read: the grammar fields are per request and are filled in
     * after this is copied. A request that names a field always wins, exactly as with
     * default_reasoning_effort. */
    SamplingParams default_sampling;
};

struct OaiDeps {
    ITokenizer*       tok = nullptr;
    IChatTemplate*    chat = nullptr;
    IGrammarCompiler* grammar = nullptr;
    IMultimodal*      mm = nullptr;
};

/* Parse. `body` is the raw request body; a malformed body is a 400 that says where. */
int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err);
int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);
/* Embeddings: fills `out->prompts` with one entry per input. */
int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);

/* ================================================================== responses */

struct LogprobEntry {
    int32_t     token = 0;
    std::string text;
    float       logprob = 0.0f;
};

struct ChoiceOut {
    int           index = 0;
    std::string   text;          /* raw generated text */
    ParsedMessage msg;           /* chat: content / reasoning / tool_calls after parsing */
    Finish        finish = Finish::None;
    std::vector<LogprobEntry> logprobs;
};

struct UsageOut {
    int64_t prompt = 0;
    int64_t completion = 0;
    int64_t cached = 0;
};

std::string chat_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u);
std::string text_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u);
std::string embeddings_body(const OaiRequest& r, const std::vector<std::vector<float>>& vecs,
                            const UsageOut& u);
std::string models_body(const std::string& model_id, int64_t created);

/* The usage chunk of each endpoint's stream: an SSE event whose choices array is empty and whose
 * only payload is `usage`. Emitted when the request asked for stream_options.include_usage, ONCE
 * per stream and after every choice's final chunk -- `n` choices share one prompt and one total,
 * so a chunk per choice would count the prompt n times and interleave with the choices. */
std::string text_usage_chunk(const OaiRequest& r, const UsageOut& u);
std::string chat_usage_chunk(const OaiRequest& r, const UsageOut& u);

/* ================================================================== streaming
 *
 * One of these per choice. It feeds the generated text to the choice's reply reader as it grows
 * and forwards each increment the reader completes as an SSE chunk. The reader never revises
 * what it has delivered, so nothing here diffs, holds back or reconciles: a delta is the next
 * piece of the answer, and the chunks a client concatenates are the answer the buffered path
 * returns.
 */
class ChatDeltaStream {
public:
    ChatDeltaStream(const OaiRequest& r, int index, const IReplyParser* parser);

    /* The `role: assistant` opener OpenAI clients expect before any content. */
    std::string first_chunk();

    /* Feed the text generated since the last push; returns zero or more SSE `data:` lines.
     * `is_final` says the generation is over, which resolves whatever the reader was holding. */
    std::string push(std::string_view appended, bool is_final);

    /* The terminating chunk carrying finish_reason. The usage chunk is the stream's, not the
     * choice's: chat_usage_chunk. */
    std::string final_chunk(Finish f);

    /* What the whole generation parsed to -- the non-streaming shape. Complete once push has
     * been called with is_final. */
    const ParsedMessage& parsed() const { return last_; }
    Finish adjusted_finish(Finish f) const;

private:
    std::string chunk(json delta, const char* finish_reason);
    std::string emit(const std::vector<ReplyDelta>& ev);

    std::unique_ptr<IReplyReader> reader_;
    std::string id_, model_;
    int64_t     created_;
    int         index_;
    bool        finished_ = false;

    ParsedMessage            last_;
    std::vector<ReplyDelta>  ev_;
};

/* The SSE terminator. Written once at the end of every stream, including a failed one, because a
 * client blocked on the reader has no other way to learn the stream is over. */
extern const char* const SSE_DONE;

std::string sse_event(const std::string& json_text);
std::string sse_error(const ApiError& e);

/* ================================================================== small shared helpers */

std::string base64_encode(const void* data, size_t len);
bool        base64_decode(const std::string& in, std::vector<uint8_t>* out);
/* A token id as a request spells it: an integer in [0, INT32_MAX]. False for anything else, and
 * the caller refuses the request -- narrowing instead would make 2^32 + 5 an alias of token 5. */
bool        token_id(const json& v, int32_t* out);

/* Parse a request body into a tree of at most json_value_budget(max_ctx) values, nested at most
 * kJsonMaxDepth deep. A syntax error comes back discarded, as json::parse's would; a body with
 * more values than the budget sets `*over`, and one nested deeper than the bound sets `*deep`,
 * and either comes back holding only what fitted, which the caller refuses. See the definition
 * for why the tree is bounded rather than the body alone. */
json        parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep);
int64_t     json_value_budget(int64_t max_ctx);
constexpr int kJsonMaxDepth = 1024;

std::string random_id(const char* prefix);
int64_t     unix_now();

}  /* namespace server */
}  /* namespace rad */
