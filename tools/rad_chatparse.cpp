/* rad-chatparse -- the reply parser's acceptance run, against a checkpoint's real tokenizer and
 * chat template.
 *
 *   rad-chatparse <fixtures root>                      the three served models by their folders
 *   rad-chatparse --model qwen3.8=<dir> [--model ...]  one folder per model, named explicitly
 *       [--fuzz N]           random replies per model and request (default 300)
 *       [--long a,b,c]       reply sizes the legacy path is timed at (default 4096,16384,65536)
 *       [--new-long a,b]     sizes only the streaming parser is timed at (default 262144,1048576)
 *
 * A folder holds the checkpoint's tokenizer.json and chat_template.jinja. The model names are
 * the ones tests/legacy/chatdiff.h knows -- qwen3.8, qwen3.6, minicpm5 -- which fix the BOS/EOS
 * spellings and the turn terminators; under a fixtures root they are found as
 * Qwen3.8-Flash-Next (or Qwen3.8-27B-FP8), Qwen3.6-35B-A3B-FP8 and MiniCPM5-2B.
 *
 * WHAT THE CTEST SUITE CANNOT DO, and this does. tests/chatparse_test cuts replies at synthetic
 * boundaries because the tokenizers are too big to commit. Here every reply is tokenised by the
 * model's own tokenizer and decoded token by token through the same Detokenizer, with the same
 * preserved control tokens, as the server -- so the parser is fed the pieces it is fed in
 * production, split exactly where the model's tokens split them. On every reply:
 *
 *   * the differential verdict against the legacy path (tests/legacy/, "old" in the report;
 *     chatdiff::judge);
 *   * the streaming contract at the token boundaries (chatdiff::check_stream);
 *   * the legacy path's own stream -- the accumulated reply re-parsed at every token, diffed
 *     against what it had sent -- against its buffered answer, counting where the two disagree.
 *
 * And the cost: a long reply streamed token by token through the SSE layer, legacy ("old")
 * against streaming ("new"), and the streaming parser alone at sizes the legacy path cannot reach
 * in reasonable time.
 *
 * Exit status is non-zero when any reply breaks the contract or a verdict is not allowed. */
#include "chatdiff.h"
#include "server/adapters.h"
#include "server/oai.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace rad;
using namespace rad::chatdiff;
using clk = std::chrono::steady_clock;

namespace {

std::string slurp(const std::string& p) {
    std::ifstream f(p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool is_dir(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

std::vector<size_t> sizes(const char* s) {
    std::vector<size_t> v;
    for (const char* p = s; *p;) {
        v.push_back((size_t)std::strtoull(p, nullptr, 10));
        const char* c = std::strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return v;
}

/* A model's real tokenizer, and the pieces the server's decoder hands a reply to the parser in. */
struct Tok {
    std::shared_ptr<Vocab> vocab;
    Tokenizer tok;

    int load(const std::string& dir, std::string* why) {
        VocabBuild vb;
        if (parse_tokenizer_json((dir + "/tokenizer.json").c_str(), vb) < 0) {
            *why = "tokenizer.json did not load";
            return RAD_E_FORMAT;
        }
        vocab = std::make_shared<Vocab>();
        if (vocab->load(std::move(vb)) < 0) { *why = "the vocabulary did not load"; return RAD_E_FORMAT; }
        return tok.init(vocab);
    }

    /* Tokenise as the model would have written the reply -- control tokens as themselves -- and
     * decode it back one token at a time the way the server does. */
    std::vector<std::string> pieces(const std::string& text,
                                    const std::vector<std::string>& preserved) const {
        std::vector<int32_t> ids;
        tok.encode(text, ids, /*add_special=*/false, /*parse_special=*/true);
        Detokenizer d(vocab, {}, /*render_special=*/false, preserved);
        std::vector<std::string> out;
        for (int32_t id : ids) {
            std::string p = d.push(id);
            if (!p.empty()) out.push_back(std::move(p));
        }
        std::string tail = d.flush();
        if (!tail.empty()) out.push_back(std::move(tail));
        return out;
    }
};

struct Tally {
    int replies = 0, same = 0, recovered = 0, bare_lift = 0, violations = 0;
    int decoder_changed = 0;    /* the decoded text is not the text: a control token not preserved */
    int old_stream_disagrees = 0;
    int shown = 0, markers_lost = 0;
    size_t tokens = 0, bytes = 0;
    double t_new = 0, t_old = 0;
};

/* The legacy path's stream: the accumulated reply re-parsed at every token and the growth
 * sent. Returns whether what it sent adds up to its own buffered answer. */
bool old_stream_agrees(const Bench& b, const std::vector<std::string>& pieces, double* secs) {
    legacy::LegacyDeltaStream s(b.old_parser(), b.request().tools, b.model().terminators);
    std::string acc;
    const auto t0 = clk::now();
    for (size_t i = 0; i < pieces.size(); ++i) {
        acc += pieces[i];
        s.push(acc, false);
    }
    s.push(acc, true);
    *secs += since(t0);
    const legacy::Answer a = b.old_parse(acc);
    if (s.content() != a.content || s.reasoning() != a.reasoning || s.names().size() != a.calls.size())
        return false;
    for (size_t i = 0; i < a.calls.size(); ++i)
        if (s.names()[i] != a.calls[i].name || s.arguments()[i] != a.calls[i].arguments) return false;
    return true;
}

void check_reply(const Bench& b, const Tok& t, const std::string& written, const char* what,
                 Tally* ty) {
    const std::vector<std::string> pieces = t.pieces(written, b.prompt().preserved_tokens);
    std::string text;
    std::vector<size_t> cuts;
    for (const std::string& p : pieces) {
        text += p;
        cuts.push_back(text.size());
    }
    ++ty->replies;
    ty->tokens += pieces.size();
    ty->bytes += text.size();
    if (text != written) {
        ++ty->decoder_changed;
        /* The decoder suppresses control tokens by design, and a reply may hold ones the format
         * does not use. What must never go missing is a marker the format is MADE of -- that is
         * the failure that turns a call into prose -- so a reply that lost one is shown. The turn
         * terminators are control tokens the decoder never renders, and a marker overlapping one
         * (`>` inside `<|im_end|>`) is not a marker the reply wrote. */
        std::vector<std::pair<size_t, size_t>> terms;
        for (const std::string& t : b.model().terminators)
            for (size_t at = written.find(t); at != std::string::npos; at = written.find(t, at + 1))
                terms.push_back({ at, at + t.size() });
        auto count = [&terms](const std::string& s, const std::string& m, bool skip_terms) {
            size_t n = 0;
            for (size_t at = s.find(m); at != std::string::npos; at = s.find(m, at + 1)) {
                bool inside = false;
                for (const auto& [a, e] : terms)
                    inside = inside || (skip_terms && at < e && a < at + m.size());
                if (!inside) ++n;
            }
            return n;
        };
        for (const std::string& m : b.prompt().preserved_tokens) {
            if (count(text, m, false) >= count(written, m, true)) continue;
            ++ty->markers_lost;
            if (ty->shown++ < 5)
                std::fprintf(stderr, "  the decoder lost the marker %s:\n    written: %s\n",
                             nlohmann::json(m).dump().c_str(),
                             nlohmann::json(written).dump(-1, ' ', false,
                                                          nlohmann::json::error_handler_t::replace).c_str());
            break;
        }
    }

    const Verdict v = compare(b, text);
    const std::string j = judge(v);
    if (!j.empty()) {
        ++ty->violations;
        std::fprintf(stderr, "  VIOLATION %s [%s, %s]: %s\n  reply: %s\n%s\n", what,
                     b.model().name.c_str(), describe(b.request()).c_str(), j.c_str(),
                     nlohmann::json(text).dump(-1, ' ', false,
                                               nlohmann::json::error_handler_t::replace).c_str(),
                     v.detail.c_str());
    }
    if (v.same) ++ty->same;
    else if (v.bare_lift) ++ty->bare_lift;
    else ++ty->recovered;

    const ReplyMessage expect = reply_parse(b.parser(), text);
    const auto t0 = clk::now();
    const std::string s = check_stream(b.parser(), text, cuts, expect);
    ty->t_new += since(t0);
    if (!s.empty()) {
        ++ty->violations;
        std::fprintf(stderr, "  STREAM %s [%s, %s]: %s\n", what, b.model().name.c_str(),
                     describe(b.request()).c_str(), s.c_str());
    }
    if (!old_stream_agrees(b, pieces, &ty->t_old)) ++ty->old_stream_disagrees;
}

/* ------------------------------------------------------------------ the long-reply timing */

struct Timing { double sse = 0, parser = 0; };

Timing time_new(const Bench& b, const std::vector<std::string>& pieces, int reps) {
    Timing best{ 1e30, 1e30 };
    server::ReplyParserBridge bridge(b.parser_ptr());
    for (int r = 0; r < reps; ++r) {
        server::OaiRequest req;
        req.id = "chatcmpl-acceptance";
        req.model = b.model().name;
        req.parse_tools = true;
        auto t0 = clk::now();
        {
            server::ChatDeltaStream st(req, 0, &bridge);
            size_t n = st.first_chunk().size();
            for (const std::string& p : pieces) n += st.push(p, false).size();
            n += st.final_chunk(server::Finish::Stop).size();
            if (n == 0) std::fprintf(stderr, "no output\n");
        }
        best.sse = std::min(best.sse, since(t0));
        t0 = clk::now();
        std::unique_ptr<ReplyStream> s = b.parser().open();
        std::vector<ReplyEvent> ev;
        for (const std::string& p : pieces) {
            s->feed(p, &ev);
            ev.clear();
        }
        s->finish(&ev);
        best.parser = std::min(best.parser, since(t0));
    }
    return best;
}

Timing time_old(const Bench& b, const std::vector<std::string>& pieces) {
    Timing t;
    auto t0 = clk::now();
    {
        legacy::LegacyDeltaStream s(b.old_parser(), true, b.model().terminators);
        std::string acc;
        for (const std::string& p : pieces) {
            acc += p;
            s.push(acc, false);
        }
        s.push(acc, true);
    }
    t.sse = since(t0);
    t0 = clk::now();
    std::string acc;
    for (const std::string& p : pieces) {
        acc += p;
        (void)b.old_parse(acc, true);
    }
    (void)b.old_parse(acc, false);
    t.parser = since(t0);
    return t;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::vector<std::pair<std::string, std::string>> want;   /* model name, folder */
    int fuzz = 300;
    std::vector<size_t> old_sizes = { 4096, 16384, 65536 };
    std::vector<size_t> new_sizes = { 262144, 1048576 };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            const std::string v = argv[++i];
            const size_t eq = v.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "--model name=dir\n"); return 2; }
            want.push_back({ v.substr(0, eq), v.substr(eq + 1) });
        } else if (a == "--fuzz" && i + 1 < argc) {
            fuzz = std::atoi(argv[++i]);
        } else if (a == "--long" && i + 1 < argc) {
            old_sizes = sizes(argv[++i]);
        } else if (a == "--new-long" && i + 1 < argc) {
            new_sizes = sizes(argv[++i]);
        } else if (a[0] != '-') {
            const std::pair<const char*, std::vector<const char*>> layout[] = {
                { "qwen3.8", { "Qwen3.8-Flash-Next", "Qwen3.8-27B-FP8" } },
                { "qwen3.6", { "Qwen3.6-35B-A3B-FP8" } },
                { "minicpm5", { "MiniCPM5-2B" } },
            };
            for (const auto& [name, dirs] : layout)
                for (const char* d : dirs)
                    if (is_dir(a + "/" + d)) { want.push_back({ name, a + "/" + d }); break; }
        } else {
            std::fprintf(stderr, "usage: rad-chatparse <fixtures root> | --model name=dir ... "
                                 "[--fuzz N] [--long a,b] [--new-long a,b]\n");
            return 2;
        }
    }
    if (want.empty()) {
        std::fprintf(stderr, "rad-chatparse: no model folders found\n");
        return 2;
    }
    log_set_level(Log::Warn);

    int violations = 0;
    for (const auto& [name, dir] : want) {
        const Model* m = nullptr;
        for (const Model& k : models()) if (k.name == name) m = &k;
        if (!m) { std::fprintf(stderr, "unknown model '%s'\n", name.c_str()); return 2; }
        std::printf("== %s (%s)\n", m->name.c_str(), dir.c_str());

        Tok tk;
        std::string why;
        if (tk.load(dir, &why) < 0) { std::fprintf(stderr, "  %s\n", why.c_str()); return 1; }
        const std::string jinja = slurp(dir + "/chat_template.jinja");
        std::printf("   tokenizer: %d tokens; template: %zu bytes\n", tk.vocab->n_tokens(),
                    jinja.size());

        const std::vector<ChatRequest> reqs = [] {
            std::vector<ChatRequest> v(4);
            v[1].thinking = false;
            v[2].parallel = false;
            v[3].reasoning = false;
            return v;
        }();
        Tally ty;
        std::mt19937 rng(20260924);
        const std::vector<std::string> frag = fragments(*m);
        for (const ChatRequest& rq : reqs) {
            Bench b;
            if (b.load(*m, jinja, &why) < 0 || b.bind(rq, agent_tools(), &why) < 0) {
                std::fprintf(stderr, "  %s: %s\n", describe(rq).c_str(), why.c_str());
                return 1;
            }
            std::vector<std::string> seeds;
            for (const ojson& msg : rendered_corpus()) {
                const std::string r = b.render_reply(msg);
                if (r.empty()) continue;
                seeds.push_back(r);
                check_reply(b, tk, r, "rendered", &ty);
            }
            for (const Adversarial& a : adversarial(*m)) {
                seeds.push_back(a.text);
                check_reply(b, tk, a.text, a.name.c_str(), &ty);
            }
            for (int i = 0; i < fuzz && !seeds.empty(); ++i) {
                std::string text;
                if (i % 2 == 0) {
                    const int n = (int)(rng() % 24);
                    for (int k = 0; k < n; ++k) text += frag[rng() % frag.size()];
                } else {
                    text = mutate(rng, seeds[rng() % seeds.size()], frag);
                }
                check_reply(b, tk, sanitize(text), "fuzz", &ty);
            }
        }
        violations += ty.violations;
        std::printf("   %d replies, %zu tokens, %zu bytes: %d identical to the old path, %d "
                    "recovered where the old grammar failed, %d bare calls the old lifter took "
                    "from content; %d contract violations\n",
                    ty.replies, ty.tokens, ty.bytes, ty.same, ty.recovered, ty.bare_lift,
                    ty.violations);
        std::printf("   the decoder dropped a control token outside the format from %d replies and "
                    "a format marker from %d; the OLD path's own stream disagreed with its "
                    "buffered answer on %d\n",
                    ty.decoder_changed - ty.markers_lost, ty.markers_lost, ty.old_stream_disagrees);
        violations += ty.markers_lost;
        std::printf("   corpus fed token by token: new %.1f ms, old %.1f ms\n", ty.t_new * 1e3,
                    ty.t_old * 1e3);

        /* The cost of a long reply, token by token. */
        Bench b;
        if (b.load(*m, jinja, &why) < 0 || b.bind(ChatRequest{}, agent_tools(), &why) < 0) return 1;
        std::printf("   %9s %8s | %12s %12s %9s | %12s %12s\n", "bytes", "tokens", "old SSE ms",
                    "new SSE ms", "speedup", "old parse ms", "new parse ms");
        for (size_t n : old_sizes) {
            const std::string text = long_reply(*m, n);
            const std::vector<std::string> p = tk.pieces(text, b.prompt().preserved_tokens);
            const Timing o = time_old(b, p);
            const Timing nw = time_new(b, p, 5);
            std::printf("   %9zu %8zu | %12.2f %12.3f %8.0fx | %12.2f %12.3f\n", text.size(),
                        p.size(), o.sse * 1e3, nw.sse * 1e3, o.sse / nw.sse, o.parser * 1e3,
                        nw.parser * 1e3);
        }
        for (size_t n : new_sizes) {
            const std::string text = long_reply(*m, n);
            const std::vector<std::string> p = tk.pieces(text, b.prompt().preserved_tokens);
            const Timing nw = time_new(b, p, 3);
            std::printf("   %9zu %8zu | %12s %12.3f %9s | %12s %12.3f   (%.1f ns/byte, %.2f us/token)\n",
                        text.size(), p.size(), "-", nw.sse * 1e3, "", "-", nw.parser * 1e3,
                        nw.parser / text.size() * 1e9, nw.parser / p.size() * 1e6);
        }
    }
    std::printf("%s\n", violations ? "FAILED" : "ok");
    return violations ? 1 : 0;
}
