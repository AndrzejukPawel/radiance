/* grammar.cpp -- the GBNF backend that ships, and the schema seam onto it.
 */
#include "grammar.h"
#include "gbnf.h"

#include <json-schema-to-grammar.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <new>
#include <stdexcept>

namespace rad {

int64_t grammar_mask_words(int64_t n_vocab) { return bitmask_words(n_vocab); }

/* HOW DEEPLY A SCHEMA MAY NEST, in objects and arrays. The converter recurses once per level and
 * names each rule after its whole path, so its output grows with the square of the depth: a
 * 72 KB schema 3,000 levels deep compiles to 67 MB of grammar, and deeper ones exhaust the
 * memory or the stack of the thread that admits the request. A schema nested as deep as this one
 * allows converts in tens of milliseconds to a couple of megabytes, and a schema written to
 * describe data nests far less. */
static constexpr size_t kMaxSchemaDepth = 512;

/* Walked with an explicit stack, because the values it exists to catch are the ones too deep to
 * recurse over. */
static size_t json_depth(const nlohmann::ordered_json& root) {
    std::vector<std::pair<const nlohmann::ordered_json*, size_t>> todo{ { &root, 1 } };
    size_t deepest = 0;
    while (!todo.empty()) {
        const auto [j, d] = todo.back();
        todo.pop_back();
        if (!j->is_structured()) continue;
        deepest = std::max(deepest, d);
        for (const auto& c : *j) todo.push_back({ &c, d + 1 });
    }
    return deepest;
}

/* ONE JSON-SCHEMA COMPILER, AND IT IS THE VENDORED ONE.
 *
 * `response_format` on /v1/chat/completions goes to the chat template, which compiles it with the
 * vendored converter; on /v1/completions it is refused by the accept-list (kAcceptedCompletion)
 * before parse_output_format runs, and that is the only path to this seam. A second, separately
 * maintained copy of the same upstream compiler would therefore be the copy no request reaches,
 * while the vendored one serves every schema. The hardening that belongs on a schema compiler
 * fed client data -- vendor/patches 0003 and 0004, which close a hang and two out-of-bounds
 * reads and refuse three constructs upstream accepts and then gets wrong -- lives in the
 * vendored copy, which is the copy that runs.
 *
 * This function is the seam response_format would go through if /v1/completions accepted it.
 * No request reaches it.
 *
 * The vendored entry point throws; errors are values by the time a caller sees them. */
int grammar_from_json_schema(const std::string& schema_json, const std::string& root,
                             std::string* out_gbnf, std::string* err) {
    if (!out_gbnf) return RAD_E_INVAL;
    /* The converter names its top rule "root" and takes no say in it. Every caller in the tree
     * asks for "root"; anything else would silently get that instead, so it is refused. */
    if (!root.empty() && root != "root") {
        if (err) *err = "this converter always names the top rule 'root'";
        return RAD_E_UNSUPPORTED;
    }
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        if (json_depth(schema) > kMaxSchemaDepth) {
            if (err) *err = "the schema nests more than " + std::to_string(kMaxSchemaDepth) +
                            " levels deep";
            return RAD_E_INVAL;
        }
        *out_gbnf = json_schema_to_grammar(schema);
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return RAD_E_INVAL;
    }
    if (out_gbnf->empty()) {
        if (err) *err = "the schema compiled to an empty grammar";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* ================================================================== the GBNF backend */

namespace {

class GbnfBackend : public GrammarBackend {
public:
    explicit GbnfBackend(std::unique_ptr<GbnfGrammar> g) : g_(std::move(g)) {}

    const char* name() const override { return "gbnf"; }
    int  accept_token(int32_t token) override { return g_->accept_token(token); }
    int  fill_mask(uint32_t* m, int64_t n) const override { return g_->fill_mask(m, n); }
    bool complete() const override { return g_->complete(); }
    bool constrains() const override { return !g_->awaiting_trigger(); }
    uint64_t masks_walked() const override { return g_->masks_walked(); }

    /* The GBNF machine's state is a stack set and a carried UTF-8 partial, both copyable, so a
     * rewind is a copy back -- there is nothing to reconstruct and no cost that scales with how
     * far the window went. */
    bool can_speculate() const override { return true; }
    int  push_state() override { g_->push_state(); return RAD_OK; }
    int  rollback() override   { return g_->rollback(); }

private:
    std::unique_ptr<GbnfGrammar> g_;
};

int gbnf_ctor(const GrammarSpec& spec, const VocabView& vocab,
              std::unique_ptr<GrammarBackend>* out, std::string* err) {
    std::string text = spec.text;

    /* A JSON Schema is compiled to GBNF here rather than being a second backend, because it is
     * not a different state machine -- it is the same one fed different text. Carrying it as a
     * backend would mean two places that can disagree about what a schema admits. */
    if (spec.kind == GrammarKind::JsonSchema) {
        std::string gbnf;
        const int st = grammar_from_json_schema(spec.text, spec.root, &gbnf, err);
        if (st < 0) return st;
        text = std::move(gbnf);
    }

    std::unique_ptr<GbnfGrammar> g;
    const int st = GbnfGrammar::create(text, spec.root.empty() ? "root" : spec.root, &vocab,
                                       spec.lazy, spec.trigger_words, spec.trigger_tokens,
                                       &g, err);
    if (st < 0) return st;

    if (!spec.prefix.empty()) {
        const int ps = g->accept_str(spec.prefix);
        if (ps < 0) {
            /* The grammar does not admit the prompt it was supposed to have been generating.
             * Constraining from here would mask against a state that does not describe this
             * request, so it is refused rather than run in the wrong place. */
            if (err) *err = "the grammar does not admit this template's generation prompt, so it "
                            "cannot be positioned at the start of the reply";
            return ps;
        }
    }

    *out = std::make_unique<GbnfBackend>(std::move(g));
    return RAD_OK;
}

}  /* namespace */

int grammar_create(const GrammarSpec& spec, const VocabView& vocab,
                   std::unique_ptr<GrammarBackend>* out, std::string* err) {
    if (!out) return RAD_E_INVAL;
    if (spec.text.empty()) {
        if (err) *err = "empty grammar";
        return RAD_E_INVAL;
    }

    /* The compiler below turns its own failures into statuses; what is left is running out of
     * memory on the way, and this is called on the admission thread and the scheduler thread,
     * where an escaping exception ends every request on the server rather than this one. */
    try {
        return gbnf_ctor(spec, vocab, out, err);
    } catch (const std::bad_alloc&) {
        if (err) *err = "out of memory building the grammar";
        return RAD_E_NOMEM;
    }
}

}  /* namespace rad */
