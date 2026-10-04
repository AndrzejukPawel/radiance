/* chat_test.cpp -- the chat template and the reply parser, against the REAL template.
 *
 * server_test covers the endpoint with a fake IChatTemplate and a fake IReplyParser, which is the
 * right shape for testing the endpoint and tests nothing about the model we actually serve. The
 * two halves that decide whether an agent turn works -- what the template renders, and whether
 * the reply comes back apart into reasoning, content and tool calls -- are what this file covers.
 *
 * The templates are read from disk rather than baked: tests/data/chat holds each one byte for
 * byte as the model ships it, and RAD_TEST_CHAT_JINJA / RAD_TEST_MINICPM_JINJA point the cases at
 * a checkpoint's own copy instead -- which is how a revised template is checked before the
 * fixture is replaced. The reply is read the way the server reads it: the format derived from
 * the template (chatfmt.h), bound to the request's tools, streamed through chatparse.h.
 *
 * WHAT THE CASES ARE FOR. Every one of them is a shape the model actually emitted into a served
 * agent turn, not a shape invented here. The tool-call syntax is Qwen's XML
 * (<tool_call><function=..><parameter=..>), the reasoning block is opened by the TEMPLATE and
 * closed by the MODEL, and the interesting cases are the ones where those two interact: a call
 * inside an unclosed think block, a second </think>, a call the model wrote with no reasoning at
 * all. A parser that handles only "reasoning, </think>, then a clean call" passes a happy-path
 * test and drops a tool call on the floor in production, which is what this file exists to stop.
 */
#include "rad_test.h"

#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace rad;
using json = nlohmann::ordered_json;

/* ------------------------------------------------------------------ the template under test */

/* The committed fixture unless a checkpoint's copy is named. A template that cannot be read
 * skips every case that needs it, and says which file it wanted -- a template test with no
 * template must not read as a pass. */
static std::string jinja_path() {
    if (const char* p = std::getenv("RAD_TEST_CHAT_JINJA")) return p;
    return std::string(RAD_CHAT_FIXTURES) + "/qwen3.8/chat_template.jinja";
}

static const std::string& jinja_source() {
    static const std::string s = [] {
        std::ifstream f(jinja_path());
        if (!f) return std::string();
        std::ostringstream ss; ss << f.rdbuf();
        return ss.str();
    }();
    return s;
}

/* One compiled template, shared: loading it parses 9 KB of Jinja and probes its capabilities,
 * which is slow enough to notice once per case. */
static ChatTemplate* tmpl() {
    static ChatTemplate* t = [] () -> ChatTemplate* {
        if (jinja_source().empty()) return nullptr;
        auto* c = new ChatTemplate();
        if (c->load(jinja_source(), "", "") < 0) { delete c; return nullptr; }
        return c;
    }();
    return t;
}

static bool need_template(const char* what) {
    if (tmpl() && tmpl()->ready()) return true;
    fprintf(stderr, "  SKIP %s: no chat template at %s (set RAD_TEST_CHAT_JINJA)\n",
            what, jinja_path().c_str());
    return false;
}

static json tools_json() {
    return json::parse(R"([{
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"limit":{"type":"integer"}},
                       "required":["path"]}}},
     {"type": "function",
      "function": {
        "name": "edit",
        "description": "Replace an exact string in a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"old_string":{"type":"string"},
                                     "new_string":{"type":"string"}},
                       "required":["path","old_string","new_string"]}}}])");
}

/* Render one turn. */
static bool render(ChatPrompt* p, bool with_tools = true) {
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Read /work/a.py and summarise it."}});
    ChatOptions opt;
    opt.add_generation_prompt = true;
    opt.enable_thinking = true;
    return tmpl()->apply(msgs, with_tools ? tools_json() : json(), opt, p) >= 0;
}

/* The reply parser for a render, built the way the server builds it: the format derived from
 * the template, bound to the request's tools and to the generation prompt the render produced --
 * which is what says the reply starts inside a reasoning block. */
static std::shared_ptr<const ReplyParser> parser_for(const std::string& jinja, const char* bos,
                                                     const char* eos, const ChatPrompt& p,
                                                     const json* tools, bool parallel) {
    ChatFormat f;
    std::string refused, why;
    if (chat_format_from_template(jinja, bos, eos, &f, &refused, &why) < 0) {
        fprintf(stderr, "    no reply format: %s\n", why.c_str());
        return nullptr;
    }
    std::shared_ptr<const ReplyFormat> rf;
    if (ReplyFormat::compile(f, &rf, &why) < 0) {
        fprintf(stderr, "    reply format does not compile: %s\n", why.c_str());
        return nullptr;
    }
    ReplyOptions ro;
    ro.generation_prompt   = p.generation_prompt;
    ro.tools               = tools;
    ro.parallel_tool_calls = parallel;
    ro.terminators         = { "<|im_end|>", "<|endoftext|>" };
    std::shared_ptr<const ReplyParser> rp;
    if (ReplyParser::create(rf, ro, &rp, &why) < 0) {
        fprintf(stderr, "    no reply parser: %s\n", why.c_str());
        return nullptr;
    }
    return rp;
}

/* Read a reply whole, or -- `partial` -- as far as a stream has delivered it so far. */
static ReplyMessage read_reply(const ReplyParser& rp, const std::string& text, bool partial) {
    std::unique_ptr<ReplyStream> s = rp.open();
    s->feed(text, nullptr);
    if (!partial) s->finish(nullptr);
    return s->message();
}

/* ------------------------------------------------------------------ what the template renders */

TEST(template_opens_the_think_block_itself) {
    if (!need_template("template_opens_the_think_block_itself")) return;
    ChatPrompt p;
    CHECK(render(&p));
    /* The model never writes the opening tag: the template does, at the end of the prompt, and
     * the reply therefore starts INSIDE a reasoning block. A parser that does not know this
     * treats the whole turn as content and leaks the model's first </think> to the user. */
    CHECK(p.prompt.find("<|im_start|>assistant") != std::string::npos);
    CHECK(p.prompt.size() >= 7 &&
          p.prompt.compare(p.prompt.size() - 8, 8, "<think>\n") == 0);
    CHECK(p.generation_prompt.find("<think>") != std::string::npos);
}

TEST(template_renders_the_xml_tool_preamble) {
    if (!need_template("template_renders_the_xml_tool_preamble")) return;
    ChatPrompt p;
    CHECK(render(&p));
    CHECK(p.prompt.find("<tool_call>") != std::string::npos);
    CHECK(p.prompt.find("<function=example_function_name>") != std::string::npos);
    CHECK(tmpl()->caps().tools);
    CHECK(tmpl()->caps().tool_calls);
}

/* A tool call the CLIENT sends back arrives with `arguments` as a JSON string, because that is
 * what the OpenAI schema says it is. The stock template feeds it to `|items` regardless, which
 * is a type error on a string, and patched templates in the wild exist to fix exactly that.
 * Whatever this stack does about it, a multi-turn tool loop has to render the parameters the
 * model was trained on, and this is the assertion that says so. */
TEST(replayed_tool_call_arguments_render_as_parameter_xml) {
    if (!need_template("replayed_tool_call_arguments_render_as_parameter_xml")) return;
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Read /work/a.py."}});
    json call = {{"id", "c1"}, {"type", "function"},
                 {"function", {{"name", "read"},
                               {"arguments", "{\"path\": \"/work/a.py\", \"limit\": 40}"}}}};
    msgs.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({call})}});
    msgs.push_back({{"role", "tool"}, {"tool_call_id", "c1"}, {"content", "print(1)\n"}});
    msgs.push_back({{"role", "user"}, {"content", "Now with limit 80."}});

    ChatPrompt p;
    ChatOptions opt;
    CHECK_OK(tmpl()->apply(msgs, tools_json(), opt, &p));
    CHECK(p.prompt.find("<function=read>") != std::string::npos);
    CHECK(p.prompt.find("<parameter=path>") != std::string::npos);
    CHECK(p.prompt.find("/work/a.py") != std::string::npos);
    /* The arguments must NOT survive as the raw JSON string the client sent. Match the string
     * the client actually sent, spaces and all -- the tool SCHEMA in the preamble legitimately
     * contains `"path":`, so a looser needle finds the schema and passes for the wrong reason. */
    CHECK(p.prompt.find("{\"path\": \"/work/a.py\"") == std::string::npos);
}

/* ------------------------------------------------------------------ reading the reply back */

static ReplyMessage parse_reply(const std::string& text, bool partial = false) {
    ChatPrompt p;
    if (!render(&p)) return ReplyMessage{};
    const json tools = tools_json();
    std::shared_ptr<const ReplyParser> rp = parser_for(jinja_source(), "", "", p, &tools, false);
    if (!rp) return ReplyMessage{};
    return read_reply(*rp, text, partial);
}

TEST(reply_splits_reasoning_from_content) {
    if (!need_template("reply_splits_reasoning_from_content")) return;
    ReplyMessage r = parse_reply("The user wants a summary.\n</think>\n\nIt prints one.");
    CHECK(r.reasoning.find("The user wants a summary.") != std::string::npos);
    CHECK_EQ(r.content, std::string("It prints one."));
    CHECK_EQ(r.calls.size(), (size_t)0);
    /* The closing tag is a delimiter, not text. Leaking it is what the user sees when the
     * reasoning split is wrong. */
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.reasoning.find("</think>") == std::string::npos);
}

TEST(reply_with_tool_call_after_reasoning) {
    if (!need_template("reply_with_tool_call_after_reasoning")) return;
    ReplyMessage r = parse_reply(
        "I should read it.\n</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK_EQ(r.calls[0].name, std::string("read"));
        CHECK(r.calls[0].arguments.find("/work/a.py") != std::string::npos);
    }
    CHECK(r.content.find("<tool_call>") == std::string::npos);
    CHECK(r.content.find("</think>") == std::string::npos);
}

/* A REAL AGENT TURN: the model reasons at length, closes the block, and calls a tool. When this
 * fails the user gets the literal </think> followed by the literal <tool_call> XML and no tool
 * call at all. Long text before the tag is the only thing separating this from the case above, so
 * if it behaves differently the parser is length- or newline-sensitive. */
TEST(long_reasoning_then_tool_call_still_parses) {
    if (!need_template("long_reasoning_then_tool_call_still_parses")) return;
    std::string reasoning;
    for (int i = 0; i < 60; ++i)
        reasoning += "Checking the file again, because the previous edit may not have applied. ";
    ReplyMessage r = parse_reply(
        reasoning + "\nNext, I'll fix the test with the exact old text. Let me check the rest:\n"
        "</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/test_a.py\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.content.find("<tool_call>") == std::string::npos);
    CHECK(r.content.find("<function=") == std::string::npos);
}

/* The model skips the closing tag and goes straight to the call from inside the think block.
 * Patched templates carry an explicit self-heal for exactly this shape, so it is a real one. The
 * call must still come out: dropping it strands the agent, and the tool XML is unambiguous with
 * or without the reasoning tag around it. */
TEST(tool_call_inside_an_unclosed_think_block) {
    if (!need_template("tool_call_inside_an_unclosed_think_block")) return;
    ReplyMessage r = parse_reply(
        "I need to look at the file first.\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("<tool_call>") == std::string::npos);
}

/* A reply with no reasoning at all: the block the template opened is closed immediately. */
TEST(reply_with_empty_reasoning) {
    if (!need_template("reply_with_empty_reasoning")) return;
    ReplyMessage r = parse_reply("\n</think>\n\nIt prints one.");
    CHECK_EQ(r.content, std::string("It prints one."));
    CHECK(r.content.find("</think>") == std::string::npos);
}

/* Streaming asks the same parser the same question with half a reply. A partial tool call is not
 * an error; a partial reply must never leak the delimiters either. */
TEST(partial_reply_does_not_leak_delimiters) {
    if (!need_template("partial_reply_does_not_leak_delimiters")) return;
    const std::string full =
        "Reading it.\n</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "</function>\n</tool_call>";
    for (size_t n = 1; n <= full.size(); ++n) {
        ReplyMessage r = parse_reply(full.substr(0, n), /*partial=*/true);
        if (r.content.find("</think>") != std::string::npos ||
            r.content.find("<tool_call>") != std::string::npos ||
            r.content.find("<function=") != std::string::npos) {
            ::radtest::fail(__FILE__, __LINE__,
                            "partial prefix of " + std::to_string(n) + " bytes leaked a "
                            "delimiter into content: \"" + r.content + "\"");
            break;
        }
    }
}

/* ------------------------------------------------------------- the value is not text to tidy
 *
 * An agent's edit tool matches `old_string` against the file BYTE FOR BYTE. The XML carries the
 * value between two delimiting newlines --
 *
 *     <parameter=old_string>\n    return a + b\n</parameter>
 *
 * -- and exactly those two newlines are the delimiters. Everything between them, INCLUDING the
 * leading indentation, is the value. A parser that strips "one leading whitespace character"
 * without checking that it is the delimiter eats the first space of every indented line and
 * every such edit fails to apply, on a model that wrote a perfectly good call. */
TEST(indented_argument_keeps_every_leading_space) {
    if (!need_template("indented_argument_keeps_every_leading_space")) return;
    ReplyMessage r = parse_reply(
        "Fixing it.\n</think>\n\n"
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=old_string>\n    return a + b\n</parameter>\n"
        "<parameter=new_string>\n    return a + b + 1\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() != 1) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("path", std::string()), std::string("/work/a.py"));
    CHECK_EQ(a.value("old_string", std::string()), std::string("    return a + b"));
    CHECK_EQ(a.value("new_string", std::string()), std::string("    return a + b + 1"));
}

/* Multi-line, blank lines and a tab: a hunk is not one line, and the blank line in the middle of
 * one must not be collapsed either. Only the first and last newline belong to the XML. */
TEST(multiline_argument_survives_verbatim) {
    if (!need_template("multiline_argument_survives_verbatim")) return;
    const std::string body = "def add(a, b):\n\n\t# sum\n    return a + b\n  ";
    ReplyMessage r = parse_reply(
        "Fixing it.\n</think>\n\n"
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=old_string>\n" + body + "\n</parameter>\n"
        "<parameter=new_string>\nx\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() != 1) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("old_string", std::string()), body);
}


/* ------------------------------------------------------------- reasoning effort
 *
 * THE TWO VOCABULARIES. OpenAI's `reasoning_effort` ladder is none/minimal/low/medium/high;
 * THIS model's template accepts xhigh/medium/low and `raise_exception`s on everything else, with
 * xhigh as its DEFAULT. A server that polices OpenAI's list instead gets both ends wrong:
 * `"high"` -- what an ordinary client sends -- reaches the template and is refused, and `"xhigh"`
 * -- this model's own documented maximum -- is refused by the server for a value the template
 * would have taken. Either is a hard 400 on every request of an agent loop.
 */
static int render_effort(ChatPrompt* p, const char* effort) {
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Read /work/a.py and summarise it."}});
    ChatOptions opt;
    opt.add_generation_prompt = true;
    opt.enable_thinking = true;
    if (effort) opt.template_kwargs["reasoning_effort"] = json(effort).dump();
    return tmpl()->apply(msgs, tools_json(), opt, p);
}

TEST(the_templates_own_effort_values_render) {
    if (!need_template("the_templates_own_effort_values_render")) return;
    for (const char* e : {"xhigh", "medium", "low"}) {
        ChatPrompt p;
        CHECK_OK(render_effort(&p, e));
        CHECK(p.error.empty());
        /* medium is the template's "say nothing" rung: it injects no instruction at all. */
        if (std::string(e) == "medium")
            CHECK(p.prompt.find("Reasoning effort is set to") == std::string::npos);
        else
            CHECK(p.prompt.find(std::string("Reasoning effort is set to ") + e) !=
                  std::string::npos);
    }
    /* And the default, which is what every client that says nothing gets. */
    ChatPrompt d;
    CHECK_OK(render_effort(&d, nullptr));
    CHECK(d.prompt.find("Reasoning effort is set to xhigh") != std::string::npos);
}

TEST(a_refused_effort_comes_back_in_the_templates_own_words) {
    if (!need_template("a_refused_effort_comes_back_in_the_templates_own_words")) return;
    ChatPrompt p;
    CHECK(render_effort(&p, "high") < 0);
    /* The template's own sentence has to SURVIVE to the caller. Logged and dropped, the 400 says
     * only "the model's chat template refused this request" and names neither the field nor the
     * values it would have accepted -- which is not something a caller can act on. */
    CHECK(!p.error.empty());
    CHECK(p.error.find("reasoning effort") != std::string::npos);
    CHECK(p.error.find("xhigh") != std::string::npos);
}

/* ================================================================== MiniCPM5: the tagged format
 *
 * A SECOND TEMPLATE, because the first exercises none of this. Qwen's calls are
 * `<tool_call><function=f><parameter=p>`; MiniCPM5's are `<function name="f"><param name="p">`,
 * which the auto-parser classifies as TAG_WITH_TAGGED and drives down a different generator. A
 * failure anywhere on that path looks to the user like a model that cannot call tools: the
 * tool-call rule ends in p.end(), so ANY mismatch anywhere fails EVERYTHING and the optional
 * tool_calls matches none. There is no partial credit, which is why these are replays of what the
 * model actually emitted rather than shapes invented here.
 *
 * ONE CALL WITH ONE SCALAR ARGUMENT PASSES EVERY ONE OF THESE CASES, so a happy-path case is no
 * evidence at all.
 */

static std::string mc_jinja_path() {
    if (const char* p = std::getenv("RAD_TEST_MINICPM_JINJA")) return p;
    return std::string(RAD_CHAT_FIXTURES) + "/minicpm5/chat_template.jinja";
}

static const std::string& mc_jinja_source() {
    static const std::string s = [] {
        std::ifstream f(mc_jinja_path());
        if (!f) return std::string();
        std::ostringstream ss; ss << f.rdbuf();
        return ss.str();
    }();
    return s;
}

static ChatTemplate* mc_tmpl() {
    static ChatTemplate* t = [] () -> ChatTemplate* {
        if (mc_jinja_source().empty()) return nullptr;
        auto* c = new ChatTemplate();
        if (c->load(mc_jinja_source(), "<s>", "</s>") < 0) { delete c; return nullptr; }
        return c;
    }();
    return t;
}

static bool mc_need(const char* what) {
    if (mc_tmpl() && mc_tmpl()->ready()) return true;
    fprintf(stderr, "  SKIP %s: no chat template at %s (set RAD_TEST_MINICPM_JINJA)\n",
            what, mc_jinja_path().c_str());
    return false;
}

/* `edits` is an ARRAY on purpose: a non-string argument is matched by a schema-constrained JSON
 * parser, and that is the one an agent's edit tool actually declares. */
static json mc_tools_json() {
    return json::parse(R"([{
      "type": "function",
      "function": {
        "name": "ls",
        "description": "List a directory",
        "parameters": {"type":"object","properties":{"path":{"type":"string"}},
                       "required":["path"]}}},
     {"type": "function",
      "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {"type":"object","properties":{"command":{"type":"string"}},
                       "required":["command"]}}},
     {"type": "function",
      "function": {
        "name": "write",
        "description": "Write a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"content":{"type":"string"}},
                       "required":["path","content"]}}},
     {"type": "function",
      "function": {
        "name": "edit",
        "description": "Apply edits to a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},
                                     "edits":{"type":"array","items":{"type":"object",
                                       "properties":{"oldText":{"type":"string"},
                                                     "newText":{"type":"string"}},
                                       "required":["oldText","newText"]}}},
                       "required":["path","edits"]}}}])");
}

static ReplyMessage mc_parse(const std::string& text, bool partial = false) {
    ChatPrompt p;
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Implement tic tac toe in python."}});
    ChatOptions opt;
    opt.add_generation_prompt = true;
    opt.enable_thinking = true;
    /* As the server renders it: ChatRenderOptions defaults this to true, and it decides whether
     * the generator emits a repetition over calls at all. */
    opt.parallel_tool_calls = true;
    /* SAY WHICH STEP FAILED. A silent early return here makes every case in the section report
     * zero tool calls, which reads as a parser that found nothing rather than a fixture that
     * never ran. */
    const int st = mc_tmpl()->apply(msgs, mc_tools_json(), opt, &p);
    if (st < 0) {
        fprintf(stderr, "    mc_parse: apply failed (%d): %s\n", st, p.error.c_str());
        return ReplyMessage{};
    }
    const json tools = mc_tools_json();
    std::shared_ptr<const ReplyParser> rp =
        parser_for(mc_jinja_source(), "<s>", "</s>", p, &tools, true);
    if (!rp) return ReplyMessage{};
    return read_reply(*rp, text, partial);
}

/* No marker of the format may survive into content. Every failure in this family looks the same
 * from the outside: the tags come back as prose. */
static void mc_content_is_clean(const ReplyMessage& r) {
    CHECK(r.content.find("<function") == std::string::npos);
    CHECK(r.content.find("<param") == std::string::npos);
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.content.find("CDATA") == std::string::npos);
    /* The turn terminator is an eog token: the decoder renders it only when the format needs
     * it, and the parser trims one off the end of content when it does. Neither should ever
     * leave it here. */
    CHECK(r.content.find("<|im_end|>") == std::string::npos);
}

/* ===================================== replies that leave the grammar
 *
 * Every case is a real reply shape that the grammar the template implies does not match -- a
 * function that is not a tool, a call written from inside prose. The parser reads them as calls
 * anyway, because the layer that can answer "no such tool" in words the model can act on is the
 * agent, and a call that comes back as prose with the XML in it is a turn the model loses. */
static void mc_recovered(const ReplyMessage& r, size_t calls) {
    CHECK_EQ(r.calls.size(), calls);
    mc_content_is_clean(r);
}

TEST(a_call_whose_function_is_not_a_tool_at_all_is_still_a_call) {
    if (!mc_need("a_call_whose_function_is_not_a_tool_at_all_is_still_a_call")) return;
    /* A shell command in the function name. No schema-driven grammar can match this. */
    ReplyMessage r = mc_parse("Checking.\n</think>\n\nLet me check the compiler version.\n\n"
                              "<function name=\"which gcc cc clang make\"></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("which gcc cc clang make"));
    CHECK_EQ(r.calls[0].arguments, std::string("{}"));
    CHECK_EQ(r.content, std::string("Let me check the compiler version."));
}

TEST(a_call_with_no_arguments_is_an_empty_object) {
    if (!mc_need("a_call_with_no_arguments_is_an_empty_object")) return;
    ReplyMessage r = mc_parse("Notes.\n</think>\n\nI'll check my notes.\n\n"
                              "<function name=\"memory\"></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("memory"));
    CHECK_EQ(r.calls[0].arguments, std::string("{}"));
}

TEST(a_cdata_value_arrives_verbatim_without_its_delimiters) {
    if (!mc_need("a_cdata_value_arrives_verbatim_without_its_delimiters")) return;
    ReplyMessage r = mc_parse("Writing.\n</think>\n\nWriting it.\n\n<function name=\"write\">"
                              "<param name=\"path\">/work/a.c</param>"
                              "<param name=\"content\"><![CDATA[int main(void) { return 0; }\n"
                              "/* <not a tag> & \"quotes\" */\n]]></param></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("content", std::string()),
             std::string("int main(void) { return 0; }\n/* <not a tag> & \"quotes\" */\n"));
    CHECK_EQ(r.content, std::string("Writing it."));
}

TEST(two_calls_keep_the_text_between_them) {
    if (!mc_need("two_calls_keep_the_text_between_them")) return;
    ReplyMessage r = mc_parse("Both.\n</think>\n\nFirst.\n"
                              "<function name=\"bash\"><param name=\"cmd\">ls</param></function>"
                              "\nThen.\n<function name=\"bash\"><param name=\"cmd\">pwd</param>"
                              "</function>");
    mc_recovered(r, 2);
    if (r.calls.size() != 2) return;
    CHECK_EQ(r.calls[0].arguments, std::string("{\"cmd\":\"ls\"}"));
    CHECK_EQ(r.calls[1].arguments, std::string("{\"cmd\":\"pwd\"}"));
    CHECK_EQ(r.content, std::string("First.\nThen."));
}

/* QWEN'S FORMAT, with a sentence after the call -- which Qwen's grammar does not admit, so a parse
 * held to that grammar fails the whole reply. */
TEST(a_qwen_call_followed_by_a_sentence_is_still_a_call) {
    if (!need_template("a_qwen_call_followed_by_a_sentence_is_still_a_call")) return;
    ReplyMessage r = parse_reply("Looking.\n</think>\n\n<tool_call>\n<function=read>\n"
                                 "<parameter=path>\n/work/a.py\n</parameter>\n</function>\n"
                                 "</tool_call>\nThat is the file.");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("read"));
    CHECK_EQ(r.calls[0].arguments, std::string("{\"path\":\"/work/a.py\"}"));
    CHECK_EQ(r.content, std::string("That is the file."));
    CHECK_EQ(r.reasoning, std::string("Looking.\n"));
}

/* A TRUNCATED CALL is a call in progress, never content: the stream has already announced it.
 * Its arguments are what arrived, unterminated -- the server reports finish_reason "length" or
 * "stop" rather than "tool_calls" for it, because they do not parse (json_partial.h). */
TEST(a_half_written_call_is_a_call_whose_arguments_did_not_finish) {
    if (!mc_need("a_half_written_call_is_a_call_whose_arguments_did_not_finish")) return;
    ReplyMessage r = mc_parse("Writing.\n</think>\n\nWriting.\n<function name=\"write\">"
                              "<param name=\"content\"><![CDATA[int main(void) {");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.empty()) return;
    CHECK(!r.calls[0].closed);
    CHECK_EQ(r.calls[0].arguments, std::string("{\"content\":\"int main(void) {"));
    CHECK(json::parse(r.calls[0].arguments, nullptr, false).is_discarded());
    CHECK_EQ(r.content, std::string("Writing."));
}

TEST(a_reply_that_only_mentions_the_syntax_is_left_alone) {
    if (!mc_need("a_reply_that_only_mentions_the_syntax_is_left_alone")) return;
    const std::string text = "A hash map stores key-value pairs. The `<function>` syntax is how "
                             "this model writes a call, but this sentence is not one.";
    ReplyMessage r = mc_parse("Hm.\n</think>\n\n" + text);
    CHECK_EQ(r.calls.size(), (size_t)0);
    CHECK_EQ(r.content, text);
}

/* THE STREAMING CONTRACT: content is sent as it grows and cannot be retracted, so the opening of a
 * call must never reach it -- at any prefix, including one that ends inside the opener. */
TEST(a_streamed_call_never_shows_its_opening_as_content) {
    if (!mc_need("a_streamed_call_never_shows_its_opening_as_content")) return;
    const std::string full = "Writing.\n</think>\n\nWriting it now.\n<function name=\"write\">"
                             "<param name=\"path\">/work/a.c</param></function>";
    for (size_t n = 1; n <= full.size(); ++n) {
        ReplyMessage r = mc_parse(full.substr(0, n), /*partial=*/true);
        if (std::string("Writing it now.").compare(0, r.content.size(), r.content) != 0) {
            ::radtest::fail(__FILE__, __LINE__, "prefix of " + std::to_string(n) +
                            " bytes emitted content \"" + r.content + "\"");
            break;
        }
    }
    CHECK_EQ(mc_parse(full).content, std::string("Writing it now."));
}

/* AN ARGUMENT NAME THE SCHEMA DOES NOT DECLARE. Every argument arm matches its parameter name as
 * a literal, so one name outside the schema matches nothing, the call matches nothing, and
 * p.end() makes that the whole reply. A real shape: `read` takes `path`, the model writes
 *
 *     <function name="read"><param name="action">read</param></function>
 *
 * and the turn comes back as prose with the XML in it. The second case shows what that really
 * costs -- a call that is entirely VALID is lost because of one extra argument beside it. Both
 * must reach the tool, which can reject an argument properly; a reply that never arrives
 * cannot. */
TEST(mc_an_undeclared_argument_does_not_lose_the_call) {
    if (!mc_need("mc_an_undeclared_argument_does_not_lose_the_call")) return;
    ReplyMessage r = mc_parse("Reading it.\n</think>\n\n"
                             "<function name=\"ls\"><param name=\"action\">read</param></function>");
    CHECK_EQ((int)r.calls.size(), 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("ls"));
    CHECK(r.calls[0].arguments.find("action") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_an_extra_argument_does_not_lose_a_valid_call) {
    if (!mc_need("mc_an_extra_argument_does_not_lose_a_valid_call")) return;
    ReplyMessage r = mc_parse("Listing.\n</think>\n\n"
                             "<function name=\"ls\"><param name=\"path\">/work</param>"
                             "<param name=\"action\">read</param></function>");
    CHECK_EQ((int)r.calls.size(), 1);
    if (r.calls.empty()) return;
    /* The declared argument still binds to its OWN schema-checked arm: the generic one is tried
     * last, so nothing that parsed before parses differently. */
    CHECK(r.calls[0].arguments.find("\"path\"") != std::string::npos);
    CHECK(r.calls[0].arguments.find("/work") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_one_call_with_one_argument) {
    if (!mc_need("mc_one_call_with_one_argument")) return;
    ReplyMessage r = mc_parse(
        "I will look at the directory.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK_EQ(r.calls[0].name, std::string("ls"));
        CHECK(r.calls[0].arguments.find("/work") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* TWO CALLS. A parallel-call rule that separates them with a literal ", " -- the separator of a
 * JSON array of calls -- never matches the second, and the whole reply comes back as prose. */
TEST(mc_two_calls_separated_by_a_newline) {
    if (!mc_need("mc_two_calls_separated_by_a_newline")) return;
    ReplyMessage r = mc_parse(
        "Both, then.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>\n"
        "<function name=\"bash\"><param name=\"command\">pwd</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)2);
    if (r.calls.size() == 2) {
        CHECK_EQ(r.calls[0].name, std::string("ls"));
        CHECK_EQ(r.calls[1].name, std::string("bash"));
        CHECK(r.calls[1].arguments.find("pwd") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* A SENTENCE BEFORE THE CALL. content_before_tools reads up to the section or per-call opener,
 * and this format declares neither -- so it falls back to p.eps(), which permits no content at
 * all. */
TEST(mc_a_sentence_before_the_call) {
    if (!mc_need("mc_a_sentence_before_the_call")) return;
    ReplyMessage r = mc_parse(
        "The workspace is empty.\n</think>\n\n"
        "The workspace is empty. I will create the file.\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\">print(1)</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("I will create the file.") != std::string::npos);
    mc_content_is_clean(r);
}

/* And after it, and between two of them: the same trap with the ends swapped. */
TEST(mc_a_sentence_after_the_call) {
    if (!mc_need("mc_a_sentence_after_the_call")) return;
    ReplyMessage r = mc_parse(
        "Writing it.\n</think>\n\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\">print(1)</param></function>\n"
        "Created the file.");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("Created the file.") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_a_sentence_between_two_calls) {
    if (!mc_need("mc_a_sentence_between_two_calls")) return;
    ReplyMessage r = mc_parse(
        "First one, then the other.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>\n"
        "Next, I will check the current directory.\n"
        "<function name=\"bash\"><param name=\"command\">pwd</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)2);
    CHECK(r.content.find("Next, I will check") != std::string::npos);
    mc_content_is_clean(r);
}

/* ARGUMENT ORDER. A tagged argument carries its own name, so its position means nothing --
 * requiring the arguments in SCHEMA order loses the call. The model writes the content first
 * because that is the part it was thinking about. */
TEST(mc_arguments_in_any_order) {
    if (!mc_need("mc_arguments_in_any_order")) return;
    ReplyMessage r = mc_parse(
        "Writing.\n</think>\n\n"
        "<function name=\"write\"><param name=\"content\">print(1)</param>"
        "<param name=\"path\">/work/t.py</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK(r.calls[0].arguments.find("/work/t.py") != std::string::npos);
        CHECK(r.calls[0].arguments.find("print(1)") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* CDATA IS PUNCTUATION. The template tells the model to wrap any value holding <, & or a newline
 * in one, so this is the normal shape of every file an agent writes -- and a delimiter that
 * reaches the tool is a delimiter written into the file. */
TEST(mc_cdata_is_not_part_of_the_value) {
    if (!mc_need("mc_cdata_is_not_part_of_the_value")) return;
    ReplyMessage r = mc_parse(
        "Writing.\n</think>\n\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\"><![CDATA[def f():\n    return \"<b>\"\n]]></param>"
        "</function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        const std::string& a = r.calls[0].arguments;
        CHECK(a.find("CDATA") == std::string::npos);
        CHECK(a.find("]]>") == std::string::npos);
        CHECK(a.find("def f()") != std::string::npos);
    }
}

/* A STREAMED value is re-parsed from the top on every chunk and only what is NEW is sent, so a
 * partial parse must never emit something it would later take back. `<![CDATA` is eight
 * characters and matches nothing yet; emitted once, it is in the user's file forever. */
TEST(mc_a_partial_cdata_value_emits_no_delimiter) {
    if (!mc_need("mc_a_partial_cdata_value_emits_no_delimiter")) return;
    for (const char* head : { "<", "<![CDATA", "<![CDATA[", "<![CDATA[def f():" }) {
        ReplyMessage r = mc_parse(
            std::string("Writing.\n</think>\n\n"
                        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
                        "<param name=\"content\">") + head, /*partial=*/true);
        if (r.calls.empty()) continue;   /* nothing emitted yet is also correct */
        CHECK(r.calls[0].arguments.find("CDATA") == std::string::npos);
    }
}

/* A NON-STRING ARGUMENT THAT IS NOT JSON. An array parameter is matched by a schema-constrained
 * JSON parser, and the model writes Python. The JSON parser cannot begin on a single quote, so
 * the argument does not match, so the call does not, so the reply comes back as prose -- and the
 * model, reading that as a rejection, rewrites the same call over and over. */
TEST(mc_a_python_quoted_array_argument_still_parses) {
    if (!mc_need("mc_a_python_quoted_array_argument_still_parses")) return;
    ReplyMessage r = mc_parse(
        "Fixing the typo.\n</think>\n\n"
        "<function name=\"edit\"><param name=\"edits\">"
        "[{'oldText': 'def minimise:', 'newText': 'def minimax:'}]</param>"
        "<param name=\"path\">/work/t.py</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        const std::string& a = r.calls[0].arguments;
        CHECK_EQ(r.calls[0].name, std::string("edit"));
        /* Normalised to JSON on the way out, because that is what `arguments` is. */
        CHECK(a.find("\"oldText\"") != std::string::npos);
        CHECK(a.find("def minimax:") != std::string::npos);
        CHECK(a.find("'") == std::string::npos);
    }
    mc_content_is_clean(r);
}

/* And the plain answer still comes back plain, with no tool call invented and no marker left. */
TEST(mc_a_reply_with_no_call_is_just_content) {
    if (!mc_need("mc_a_reply_with_no_call_is_just_content")) return;
    ReplyMessage r = mc_parse("Thinking about it.\n</think>\n\nA hash table maps keys to values.");
    CHECK_EQ(r.calls.size(), (size_t)0);
    CHECK(r.content.find("A hash table maps keys to values.") != std::string::npos);
    mc_content_is_clean(r);
}

/* ===================================== the template face, with no checkpoint in the room
 *
 * EVERY CASE BELOW RUNS WITHOUT A VENDOR TEMPLATE, so the seam between core and the vendored
 * Jinja engine is covered even where no template file can be read and every case above skips.
 * ChatML ships in the binary (`ChatTemplate::chatml_source`), which is enough to exercise the
 * three things chat.h says this face is for, and those are the three a vendor template cannot
 * test any better:
 *
 *   ERRORS ARE VALUES. Upstream throws on a bad template, a message it rejects, a malformed
 *   request -- and a throw crossing the server's request loop is not a failure mode this project
 *   accepts. Every one of those has to come back as a negative RAD_E_*.
 *
 *   THE FALLBACK IS ANNOUNCED. An empty source is a model whose prompt format we are guessing,
 *   and guessing quietly is how a chat endpoint produces fluent nonsense.
 *
 *   THE CAPABILITIES ARE PROBED, not assumed -- rendered against synthetic inputs to see what the
 *   template does with them. A probe that answered from a default would report `tools: true` for
 *   a template with no tool support, and the adapter above it would then hand it tools.
 */

TEST(an_empty_template_is_the_chatml_fallback_and_says_which_one_it_is) {
    ChatTemplate t;
    CHECK_OK(t.load("", "", ""));
    CHECK(t.ready());
    /* Named rather than inlined precisely so the server can report which template it is running:
     * an operator looking at /props has to be able to tell a guess from the model's own. */
    CHECK_EQ(t.source(), std::string(ChatTemplate::chatml_source()));
}

TEST(a_template_is_not_ready_until_one_is_loaded) {
    ChatTemplate t;
    CHECK(!t.ready());
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});
    ChatPrompt p;
    /* A value, not a throw and not a crash: this is reached from the request loop. */
    CHECK(t.apply(msgs, json(), ChatOptions{}, &p) < 0);
}

TEST(chatml_renders_every_turn_in_order_and_marks_the_assistant_prefix) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));

    json msgs = json::array();
    msgs.push_back({{"role", "system"}, {"content", "You are terse."}});
    msgs.push_back({{"role", "user"}, {"content", "First question."}});
    msgs.push_back({{"role", "assistant"}, {"content", "First answer."}});
    msgs.push_back({{"role", "user"}, {"content", "Second question."}});

    ChatOptions opt;
    opt.add_generation_prompt = true;
    ChatPrompt p;
    CHECK_OK(t.apply(msgs, json(), opt, &p));

    const size_t sys = p.prompt.find("You are terse.");
    const size_t q1  = p.prompt.find("First question.");
    const size_t a1  = p.prompt.find("First answer.");
    const size_t q2  = p.prompt.find("Second question.");
    CHECK(sys != std::string::npos);
    CHECK(q1 != std::string::npos);
    CHECK(a1 != std::string::npos);
    CHECK(q2 != std::string::npos);
    /* In the order they were sent. A template that rendered them in any other order would still
     * contain every string, which is why the positions and not the presence are the check. */
    CHECK(sys < q1);
    CHECK(q1 < a1);
    CHECK(a1 < q2);

    /* THE GENERATION PROMPT IS ALREADY INSIDE THE PROMPT, and saying so is the contract the
     * scheduler and the reply parser both read: the parser cuts the reply at it, and a copy that
     * was not a suffix of the prompt would cut at the wrong byte. */
    CHECK(!p.generation_prompt.empty());
    CHECK(p.prompt.size() >= p.generation_prompt.size());
    CHECK_EQ(p.prompt.compare(p.prompt.size() - p.generation_prompt.size(),
                              p.generation_prompt.size(), p.generation_prompt), 0);
}

TEST(without_a_generation_prompt_nothing_is_appended_for_the_assistant) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});

    ChatOptions on;
    on.add_generation_prompt = true;
    ChatPrompt with;
    CHECK_OK(t.apply(msgs, json(), on, &with));

    ChatOptions off;
    off.add_generation_prompt = false;
    ChatPrompt without;
    CHECK_OK(t.apply(msgs, json(), off, &without));

    CHECK(without.prompt.size() < with.prompt.size());
    CHECK_EQ(with.prompt.compare(0, without.prompt.size(), without.prompt), 0);
}

/* THE FLAG STRIPS A BOS, IT DOES NOT ADD ONE, and the difference is a whole token of prompt.
 *
 * The container says whether the TOKENISER adds BOS. When it does and the template also renders
 * `bos_token`, the prompt carries two -- so the render drops the one the template wrote. Read the
 * other way round, the flag would put a second one in rather than take one out, and a doubled BOS
 * is a prompt the model was never trained on with nothing in the output to say so.
 *
 * ChatML renders no bos_token, so this needs a template that does. Four lines of Jinja rather
 * than a checkpoint, because the behaviour under test is the render's and not the vendor's. */
TEST(a_bos_the_tokeniser_will_add_is_not_also_rendered_into_the_prompt) {
    static const char* src =
        "{{- bos_token -}}"
        "{%- for message in messages -%}{{- message.content -}}{%- endfor -%}";

    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});

    /* The tokeniser does not add one, so the template's own has to stay. */
    ChatTemplate keeps;
    CHECK_OK(keeps.load(src, "<s>", "</s>", false, false));
    ChatPrompt a;
    CHECK_OK(keeps.apply(msgs, json(), ChatOptions{}, &a));
    CHECK_EQ(a.prompt.compare(0, 3, "<s>"), 0);

    /* The tokeniser does add one, so the template's own is dropped -- exactly one BOS reaches the
     * model either way. */
    ChatTemplate drops;
    CHECK_OK(drops.load(src, "<s>", "</s>", true, false));
    ChatPrompt b;
    CHECK_OK(drops.apply(msgs, json(), ChatOptions{}, &b));
    CHECK(b.prompt.compare(0, 3, "<s>") != 0);
    CHECK_EQ(a.prompt.size() - b.prompt.size(), (size_t)3);

    /* And the request may say so for a container that did not: the two are ORed, so a caller can
     * turn it on and cannot turn it off. */
    ChatPrompt c;
    ChatOptions opt;
    opt.add_bos = true;
    CHECK_OK(keeps.apply(msgs, json(), opt, &c));
    CHECK_EQ(c.prompt, b.prompt);
}
/* THE PROBE HAS TO ANSWER FROM THE TEMPLATE. ChatML reads `message.content` and nothing else --
 * no tool loop, no tool_calls -- so a probe reporting tool support for it is a probe reporting a
 * default, and the adapter above would then hand this template a tool list it cannot render. */
TEST(the_capability_probe_reads_what_the_template_actually_does) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    const ChatCaps& c = t.caps();
    CHECK(!c.tools);
    CHECK(!c.tool_calls);
    /* It does read a string content, and it does render whatever role it is given. */
    CHECK(c.string_content);
    CHECK(c.system_role);
}

/* Jinja that does not parse is the ordinary case for a container somebody built by hand, and it
 * arrives at load time rather than at the first request -- which is the whole point of loading
 * once at startup. It must be a value: a throw here crosses the bringup path. */
TEST(a_template_that_does_not_parse_is_a_value_and_not_a_throw) {
    ChatTemplate t;
    CHECK(t.load("{%- for message in messages -%}{{ message.content }}", "", "") < 0);
    CHECK(!t.ready());
}

/* The messages come off an HTTP request, so every shape a client can send has to come back as a
 * refusal rather than as an exception through the request loop. */
TEST(a_message_list_that_is_not_one_is_refused_rather_than_thrown) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    ChatPrompt p;

    /* An object where an array belongs. */
    CHECK(t.apply(json::object({{"role", "user"}}), json(), ChatOptions{}, &p) < 0);
    /* A message with no role. */
    json no_role = json::array();
    no_role.push_back({{"content", "hello"}});
    CHECK(t.apply(no_role, json(), ChatOptions{}, &p) < 0);
    /* And a null, which is what a missing field deserialises to. */
    CHECK(t.apply(json(), json(), ChatOptions{}, &p) < 0);
}

/* A client's tool and response schemas reach the JSON-schema converter, which recurses once per
 * level of nesting; a schema some thousands of levels deep -- tens of kilobytes of request --
 * exhausts the thread rendering it. It is refused on its depth before the template runs. */
TEST(a_schema_nested_past_the_limit_is_refused_before_it_is_rendered) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    std::string deep;
    for (int i = 0; i < 20000; ++i) deep += R"({"type":"array","items":)";
    deep += R"({"type":"integer"})";
    for (int i = 0; i < 20000; ++i) deep += "}";

    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hi"}});
    ChatPrompt p;
    ChatOptions opt;
    opt.json_schema = deep;
    CHECK_EQ(t.apply(msgs, json(), opt, &p), RAD_E_INVAL);
    CHECK(p.error.find("levels deep") != std::string::npos);

    json tools = json::array();
    tools.push_back({{"type", "function"},
                     {"function", {{"name", "f"}, {"parameters", json::parse(deep)}}}});
    ChatPrompt q;
    CHECK_EQ(t.apply(msgs, tools, ChatOptions{}, &q), RAD_E_INVAL);
}

RAD_TEST_MAIN()
