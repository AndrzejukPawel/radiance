/* grammar.h -- structured output: the grammar backend interface, and the bitmask that crosses to
 * the device.
 *
 * Structured output is GBNF, with JSON-schema compiled to a grammar. THE GRAMMAR STATE MACHINE
 * STAYS ON HOST, where it belongs -- it is a pushdown automaton over a rule set, it branches per
 * candidate, and none of that is work a GPU is good at. What crosses is a TOKEN BITMASK PER
 * SEQUENCE, which at a 151K vocab is 19 KiB, cheap enough to upload every step (spec §13). The
 * mask lands as the `sample_mask` op (docs/OPS.md), which is an ordinary declared op resolved
 * through the ordinary hierarchy.
 *
 * The backend is a small interface so an alternative can be dropped in, and it is small on
 * purpose: three calls. GBNF is the shipped one and there is no reason to carry xgrammar for it.
 * If a device-side grammar ever earns its place, it implements this and nothing above it changes
 * -- at which point it gets a way to be selected, which is a thing to add then rather than a
 * registry to carry now for an entry that never arrived.
 */
#pragma once
#include "../rad_internal.h"
#include "vocab_view.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {

enum class GrammarKind {
    Gbnf       = 0,   /* the text is GBNF */
    JsonSchema = 1    /* the text is a JSON Schema, compiled to GBNF at create time */
};

struct GrammarSpec {
    GrammarKind kind = GrammarKind::Gbnf;
    std::string text;
    std::string root = "root";

    /* Lazy grammars constrain nothing until a trigger fires, which is how a tool-calling template
     * stays free-form until the model starts emitting a call. A trigger is a token id or a literal
     * word in the emitted bytes; see sampler.cpp for how a template's triggers become these. */
    bool                     lazy = false;
    std::vector<std::string> trigger_words;
    std::vector<int32_t>     trigger_tokens;

    /* TEXT THE GRAMMAR HAS ALREADY SEEN, replayed into it before it constrains anything.
     *
     * A chat template's grammar describes the whole assistant turn, generation prompt included:
     * its root really does begin `"<|im_start|>assistant\n"`, because that is where the turn
     * begins. But those bytes are in the PROMPT, not in what the model is about to generate, so a
     * grammar started at its root demands them again -- and the model, obediently masked, emits
     * the literal text `<|im_start|>assistant` as its answer. Replaying the prefix is what puts
     * the automaton where the sampler actually is. Empty for a caller-supplied grammar, which
     * describes the reply and nothing before it. */
    std::string              prefix;
};

class GrammarBackend {
public:
    virtual ~GrammarBackend() = default;

    virtual const char* name() const = 0;

    /* Advance over the token the sampler just emitted. Returns RAD_OK, or RAD_E_STATE if that
     * token was not admissible -- which means the mask and the pick disagreed, a bug on this side
     * rather than a bad request. RAD_E_FULL or RAD_E_NOMEM means the parse outgrew the backend's
     * bounds; the backend is then dead and its next fill_mask fails. */
    virtual int accept_token(int32_t token) = 0;

    /* Fill `bitmask` ([n_words] words, bit set means admissible). RAD_E_STATE if nothing at all is
     * admissible, RAD_E_FULL or RAD_E_NOMEM if the state is past the backend's bounds; any of them
     * fails the request rather than emitting a token the grammar forbids. */
    virtual int fill_mask(uint32_t* bitmask, int64_t n_words) const = 0;

    /* Some parse has run to completion, so the sequence may stop here. */
    virtual bool complete() const = 0;

    /* INSTRUMENT: how many masks this backend has had to COMPUTE rather than serve from a memo,
     * monotonic over the life of whatever the memo belongs to. Diff it around a fill_mask to learn
     * which of the two that call was.
     *
     * It is a counter and not a duration. Inferring computed-versus-hit from how long a fill_mask
     * took needs a cost threshold, and no threshold holds: a mask computed on a small grammar
     * finishes inside the time a memo hit on a large one takes, so the classification is wrong and
     * any per-mask cost derived from it is wrong by the same factor. A backend with no memo counts
     * every mask. */
    virtual uint64_t masks_walked() const { return 0; }

    /* False while a lazy grammar is still waiting for its trigger. The driver skips the mask op
     * for a sequence that is not being constrained, which is the only reason this is on the
     * interface rather than folded into fill_mask returning all-ones. */
    virtual bool constrains() const = 0;

    /* ---------------------------------------------------------- speculation

     * A CONSTRAINED SEQUENCE SPECULATES BY WALKING THE WINDOW AND REWINDING. The mask for draft
     * position i is the mask of the state reached by accepting drafts 0..i-1, so the sampler
     * pushes the state, advances over each proposal in turn to mask the position behind it, and
     * pops back to where the sequence really is. Without it a grammar-constrained request decodes
     * one token at a time, at several times the per-token cost of the same model generating
     * unconstrained prose.
     *
     * A backend that cannot do this says so and is driven unspeculated -- the sampler decodes
     * such a request a token at a time rather than masking a draft position against a state the
     * sequence is not in, which would admit the wrong token set and read as a model that writes
     * malformed calls. */
    virtual bool can_speculate() const { return false; }

    /* Push the current state. Nestable; `rollback` pops the most recent. */
    virtual int push_state() { return RAD_E_UNSUPPORTED; }
    virtual int rollback()   { return RAD_E_UNSUPPORTED; }
};

/* Words a mask needs for a vocabulary of this size. */
int64_t grammar_mask_words(int64_t n_vocab);

/* Compile a JSON Schema to GBNF text. Separated from create() so a caller that wants to SEE the
 * grammar -- `--debug-graph`, or a user asking why their schema behaves as it does -- can get it
 * without building a state machine. */
int grammar_from_json_schema(const std::string& schema_json, const std::string& root,
                             std::string* out_gbnf, std::string* err);

/* Build the grammar backend. Returns RAD_OK, or a negative status with a human-readable reason
 * in *err. There is ONE backend and it is GBNF: a name-keyed registry over a single entry with no
 * second registration site is a lookup that can only ever resolve to that entry. */
int grammar_create(const GrammarSpec& spec, const VocabView& vocab,
                   std::unique_ptr<GrammarBackend>* out, std::string* err);

}  /* namespace rad */
