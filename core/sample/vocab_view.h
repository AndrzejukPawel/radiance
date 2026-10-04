/* vocab_view.h -- what the sampler needs to know about a vocabulary, and nothing else.
 *
 * The tokeniser lives in core/text/ and belongs to the text frontend. The sampler needs exactly
 * three facts about a token -- how many there are, its printable piece, and whether it ends
 * generation -- plus, for DRY's sequence breakers and GBNF's `<token>` syntax, the ability to
 * tokenise a short literal. Taking a dependency on the whole tokeniser for that would put a
 * component boundary on the step path and make every sampler test drag a real vocabulary behind
 * it. An abstract view costs one virtual call per token per grammar step, which is host work the
 * device is not waiting on.
 *
 * core/text/ implements this over the real vocab; SimpleVocab below implements it over a table,
 * which is what the tests use.
 */
#pragma once
#include "../rad_internal.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class VocabView {
public:
    virtual ~VocabView() = default;

    virtual int32_t n_tokens() const = 0;

    /* The token's bytes as they would be appended to the output stream. The grammar walks these,
     * so a view that returns a display form rather than the exact bytes will admit the wrong
     * token set -- and the failure mode is plausible text with wrong structure, which is the
     * hardest kind to notice. */
    virtual const std::string& token_piece(int32_t id) const = 0;

    /* End-of-generation: EOS, EOT, and whatever else this model ends on. A grammar admits one
     * only when some stack has run to completion. */
    virtual bool is_eog(int32_t id) const = 0;

    /* Tokenise a short literal. Only DRY's sequence breakers and GBNF's `<token>` form need it,
     * and both are optional features, so a view that has no tokeniser returns
     * RAD_E_UNSUPPORTED rather than pretending. */
    virtual int tokenize(const std::string& text, std::vector<int32_t>* out) const {
        (void)text; (void)out;
        return RAD_E_UNSUPPORTED;
    }

    /* The same with special tokens recognised in the text, so `<tool_call>` comes back as the
     * one control token a template means by it. Grammar triggers are resolved through this. */
    virtual int tokenize_special(const std::string& text, std::vector<int32_t>* out) const {
        (void)text; (void)out;
        return RAD_E_UNSUPPORTED;
    }

    /* ---------------- what a grammar backend derives from this vocabulary ----------------
     *
     * A GBNF mask is built over a DECODED form of the whole vocabulary -- every piece as code
     * points, plus a trie over them -- which costs a full vocabulary scan and is therefore
     * computed once and kept. It is kept HERE, with the vocabulary's own lifetime, because the
     * only other place to keep it is a table outside, and the only key such a table has is this
     * object's ADDRESS. An address is not an identity: a view destroyed and another constructed
     * in the same storage answers to the same key, and the second one is then served what was
     * derived for the first -- the decoded vocabulary, the trie over it, and the grammars
     * compiled against both. The mask that comes out is a mask over another model's tokens: it
     * admits the wrong set, or nothing at all, and nothing downstream can tell. Tying the cache
     * to the object makes the question unaskable.
     *
     * Opaque and mutable: what is cached is the backend's business and this header is the
     * sampler's interface, and filling it does not change what the vocabulary says. A backend
     * that caches nothing pays one null pointer. */
    mutable std::shared_ptr<void> gbnf_cache;
};

/* A vocabulary held as a table. Not a toy: it is what the sampler tests are written against, and
 * being able to state a vocabulary in four lines is what makes a grammar test readable. */
class SimpleVocab : public VocabView {
public:
    SimpleVocab() = default;
    explicit SimpleVocab(std::vector<std::string> pieces) : pieces_(std::move(pieces)) {}

    int32_t add(std::string piece, bool eog = false) {
        int32_t id = (int32_t)pieces_.size();
        pieces_.push_back(std::move(piece));
        if (eog) eog_.push_back(id);
        return id;
    }

    void mark_eog(int32_t id) { eog_.push_back(id); }

    int32_t n_tokens() const override { return (int32_t)pieces_.size(); }

    const std::string& token_piece(int32_t id) const override {
        static const std::string empty;
        if (id < 0 || id >= (int32_t)pieces_.size()) return empty;
        return pieces_[id];
    }

    bool is_eog(int32_t id) const override {
        for (int32_t e : eog_) if (e == id) return true;
        return false;
    }

    /* Longest-match over the table, which is not BPE and does not pretend to be. It exists so a
     * DRY sequence breaker or a `<token>` literal resolves in a test; a real vocabulary's merge
     * machinery lives in core/text/. */
    /* A table has no special tokens of its own: every piece is matched as written. */
    int tokenize_special(const std::string& text, std::vector<int32_t>* out) const override {
        return tokenize(text, out);
    }

    int tokenize(const std::string& text, std::vector<int32_t>* out) const override {
        if (!out) return RAD_E_INVAL;
        size_t pos = 0;
        while (pos < text.size()) {
            int32_t best = -1;
            size_t  best_len = 0;
            for (int32_t id = 0; id < (int32_t)pieces_.size(); ++id) {
                const std::string& p = pieces_[id];
                if (p.empty() || p.size() <= best_len) continue;
                if (text.compare(pos, p.size(), p) == 0) { best = id; best_len = p.size(); }
            }
            if (best < 0) return RAD_E_NOTFOUND;
            out->push_back(best);
            pos += best_len;
        }
        return RAD_OK;
    }

private:
    std::vector<std::string> pieces_;
    std::vector<int32_t>     eog_;
};

}  /* namespace rad */
