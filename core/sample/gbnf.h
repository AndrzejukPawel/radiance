/* gbnf.h -- the GBNF grammar state machine.
 *
 * LIFTED from llama.cpp `src/llama-grammar.{h,cpp}` at upstream commit 06938ac12 (MIT). It lives
 * under core/sample/ rather than vendor/ because it is on the sampling path and the sampler owns
 * it; vendor/ belongs to the text frontend. It is vendor-SHAPED all the same: the algorithm is
 * upstream's, element for element, and the changes are listed here so a future re-lift knows what
 * to re-apply.
 *
 * What changed and why:
 *   - `llama_vocab` became `VocabView` (vocab_view.h), so the sampler does not depend on the
 *     tokeniser's concrete type and a grammar test can state its vocabulary in four lines.
 *   - `GGML_ASSERT` / `GGML_ABORT` became this project's status returns and `fatal()`. Errors are
 *     values (docs/IMPLEMENTATION.md); upstream's aborts inside the stack walk stay aborts,
 *     because they are broken invariants rather than runtime conditions.
 *   - The parser still throws internally -- it is a recursive-descent parser and that is what
 *     upstream does -- but nothing escapes: `parse()` catches and returns a status with the
 *     message in an out-parameter.
 *   - `llama_grammar_apply_impl`, which edited a candidate array in place, became `fill_mask`,
 *     which writes a token bitmask. That is the whole reason the state machine stays on host: at
 *     a 151K vocab the mask is 19 KiB, cheap enough to upload every step, where a candidate array
 *     is not (spec §13).
 *   - Everything a client's grammar can make the machine do is BOUNDED (gbnf_internal.h, "what a
 *     grammar may cost"): what a repetition expands to, how deep parentheses nest, how large a
 *     stack set grows, how much an epsilon closure explores, how much work one mask, one accepted
 *     token or one compile does, and how large the automaton and the mask plan grow. A grammar
 *     past a bound is refused at compile time or fails its request at run time; it never takes
 *     the process with it.
 *   - Left recursion is found with nullability computed as a fixed point and an iterative
 *     strongly-connected-components pass. Upstream's recursive check missed recursion through an
 *     indirectly nullable rule, and recursed once per rule of a chain.
 *   - A closure's explored stacks are shared by every closure that builds the same set, the
 *     stacks that match code points are advanced together rather than one at a time, and a
 *     large set is de-duplicated through an index. Each gives the same set as upstream; without
 *     them, stacks that converge on one continuation re-expand it once each.
 *   - The token mask is not upstream's candidate walk. It is served from a plan precomputed when
 *     the grammar is compiled (gbnf_mask.cpp): per stack suffix, the tokens it admits whatever lies
 *     below it, and a small trie of what depends on what does. Upstream's walk survives only as
 *     the test oracle (tests/gbnf_oracle.cpp), and every mask is held to it bit for bit.
 *
 * THE COST OF THE MASK IS ON THE STEP. The mask is built on the scheduler thread with the rank
 * threads parked at the barrier, so whatever it takes is added to the step time of a constrained
 * request -- up to four times a step under speculation. A mask that walks the vocabulary costs
 * more than the decode step it sits inside, and a constrained tool call then runs at many times the
 * per-token cost of the same model writing prose.
 *
 * What it costs as written:
 *
 *   - Compiling a grammar builds its mask plan, which walks the vocabulary once per place in the
 *     grammar a mask can start. That is milliseconds to a few hundred of them, bounded, and done on
 *     the thread that ADMITS the request (gbnf_compile), never on the scheduler thread.
 *   - A mask is then a key lookup per live stack, an OR of precomputed token sets, and a walk of
 *     the few tokens whose admissibility depends on the stack below the key -- the same cost for
 *     a state seen before and one never seen. A state seen before is also a memcpy from the
 *     program's mask cache, shared by every request and rank that uses the grammar.
 *   - The vocabulary index the plan is built over is computed once per vocabulary and shared.
 *     Nothing is per request and nothing is on the device.
 *
 * So the state machine stays on host: the cost that would argue for a device-side grammar is a
 * mask computed by walking the vocabulary per step, and the answer to that is to precompute the
 * walk (XGrammar's adaptive token-mask cache, carried over to this machine's stacks), not a
 * different place to run it (spec §13).
 */
#pragma once
#include "../rad_internal.h"
#include "vocab_view.h"
#include "word_scan.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {

/* ================================================================== grammar elements */

enum GreType {
    GRE_END            = 0,   /* end of rule definition */
    GRE_ALT            = 1,   /* start of an alternate definition */
    GRE_RULE_REF       = 2,   /* reference to a rule */
    GRE_CHAR           = 3,   /* terminal: a code point */
    GRE_CHAR_NOT       = 4,   /* inverse char set: [^a], [^a-b] */
    GRE_CHAR_RNG_UPPER = 5,   /* modifies the preceding CHAR into an inclusive range */
    GRE_CHAR_ALT       = 6,   /* adds an alternate char to the preceding set */
    GRE_CHAR_ANY       = 7,   /* '.' */
    GRE_TOKEN          = 8,   /* terminal: a token id, `<[id]>` */
    GRE_TOKEN_NOT      = 9    /* inverse token, `!<[id]>` */
};

struct GbnfElement {
    GreType  type;
    uint32_t value;           /* code point, rule id, or token id */
};

/* A UTF-8 sequence split across token boundaries. Byte-level BPE emits partial sequences all the
 * time, so a grammar that could not carry one between tokens would reject perfectly good text. */
struct PartialUtf8 {
    uint32_t value    = 0;    /* bits so far, unshifted */
    int      n_remain = 0;    /* bytes still needed; -1 marks an invalid sequence */
};

using GbnfRule   = std::vector<GbnfElement>;
using GbnfRules  = std::vector<GbnfRule>;
using GbnfStack  = std::vector<const GbnfElement*>;
using GbnfStacks = std::vector<GbnfStack>;

/* THE COMPILED GRAMMAR, SHARED BY EVERY REQUEST THAT ASKS FOR IT, with its mask plan and its mask
 * cache. Defined in gbnf.cpp, which says what it is worth; the short version is that nothing
 * inside it is per-request. `GbnfState` is the mask cache's key: the whole of what fill_mask
 * reads. */
struct GbnfProgram;
struct GbnfState;

/* What compiling a grammar cost and what the compiled program holds. */
struct GbnfProgramInfo {
    double   compile_ms    = 0.0;   /* all of it: parse, checks and the mask plan */
    double   plan_ms       = 0.0;   /* of which, the mask plan */
    uint64_t plan_work     = 0;     /* units the plan charged against its bound */
    size_t   rules         = 0;
    size_t   elements      = 0;
    size_t   keys          = 0;     /* stack suffixes with a precomputed token set */
    size_t   dense_masks   = 0;     /* of those, held as a whole-vocabulary bitmask */
    size_t   max_key_depth = 0;     /* the deepest suffix, in stack entries */
    size_t   rem_nodes     = 0;     /* nodes across every remainder trie */
    size_t   max_rem_nodes = 0;     /* the largest remainder trie */
    size_t   bytes         = 0;     /* rules and plan; the mask cache is extra */
};

/* What the vocabulary index cost and holds. */
struct GbnfVocabInfo {
    double build_ms = 0.0;
    size_t bytes    = 0;
};

/* ================================================================== the parser */

class GbnfParser {
public:
    explicit GbnfParser(const VocabView* vocab = nullptr) : vocab_(vocab) {}

    /* Returns RAD_OK, or RAD_E_FORMAT with the parse error in *err. Never throws out. */
    int parse(const char* src, std::string* err);

    const std::map<std::string, uint32_t>& symbol_ids() const { return symbol_ids_; }
    const GbnfRules& rules() const { return rules_; }


private:
    const VocabView*                vocab_ = nullptr;
    std::map<std::string, uint32_t> symbol_ids_;
    GbnfRules                       rules_;

    /* What the parse has built beyond the text itself, and how deep the parentheses are right
     * now. Both are bounded (gbnf_internal.h, "what a grammar may cost"), because the text is
     * client data and the parser runs on shared threads. */
    uint64_t budget_used_ = 0;
    uint32_t depth_       = 0;
    void     charge(uint64_t bytes);

    uint32_t get_symbol_id(const char* src, size_t len);
    uint32_t generate_symbol_id(const std::string& base);
    void     add_rule(uint32_t id, const GbnfRule& rule);

    const char* parse_alternates(const char* src, const std::string& rule_name,
                                 uint32_t rule_id, bool is_nested);
    const char* parse_sequence(const char* src, const std::string& rule_name,
                               GbnfRule& rule, bool is_nested);
    const char* parse_rule(const char* src);
};

/* ================================================================== the state machine */

class GbnfGrammar {
public:
    /* Build from grammar text. `root` names the start symbol. Returns RAD_OK, or RAD_E_FORMAT /
     * RAD_E_NOTFOUND / RAD_E_UNSUPPORTED with the reason in *err -- left recursion is
     * RAD_E_UNSUPPORTED, because it is a grammar this machine cannot run rather than one that is
     * malformed, and the distinction is what the user needs to fix it.
     *
     * A LAZY grammar constrains nothing until a trigger fires, which is how a tool-calling
     * template stays free-form until the model starts writing a call. A trigger is a token id, or
     * a literal word found in the generated bytes; parsing resumes from where the word begins. */
    static int create(const std::string& text, const std::string& root, const VocabView* vocab,
                      bool lazy,
                      const std::vector<std::string>& trigger_words,
                      const std::vector<int32_t>& trigger_tokens,
                      std::unique_ptr<GbnfGrammar>* out, std::string* err);

    /* Advance the state machine over one accepted token. Returns RAD_OK, or RAD_E_STATE if the
     * token left no stack alive -- which means the sampler emitted a token the grammar forbade,
     * a bug in the mask rather than in the request. RAD_E_FULL (or RAD_E_NOMEM) means the parse
     * outgrew the bounds in gbnf_internal.h: the grammar is left dead, and the request cannot go
     * on. */
    int accept_token(int32_t token);

    /* Advance over a raw string, for replaying a prompt prefix into the grammar. */
    int accept_str(const std::string& piece);

    /* Fill a token bitmask: bit set means admissible. `n_words` must be at least
     * ceil(n_tokens/32). Returns RAD_OK, or RAD_E_STATE if the grammar admits no token at all,
     * which is a dead grammar and must fail the request rather than produce a random token, or
     * RAD_E_FULL / RAD_E_NOMEM if the mask outgrew its bounds, which must fail it the same way. */
    int fill_mask(uint32_t* bitmask, int64_t n_words) const;

    /* Some stack has run to completion, so an end-of-generation token is admissible. */
    bool complete() const;

    /* Masks this grammar's PROGRAM has had to compute rather than serve from its memo. Shared
     * with every other request on the same program, which is the point: see grammar.h. */
    uint64_t masks_walked() const;

    /* A lazy grammar that has not seen its trigger yet constrains nothing. */
    bool awaiting_trigger() const { return awaiting_trigger_; }

    /* The state as it stands, read-only: what a mask is a function of. The test oracle computes
     * the same mask from these by upstream's candidate walk. */
    const GbnfStacks&  stacks() const { return stacks_; }
    PartialUtf8        partial() const { return partial_utf8_; }
    const VocabView*   vocab() const { return vocab_; }
    const GbnfProgram& program() const { return *prog_; }

    /* PUSH AND POP THE WHOLE MUTABLE STATE, so a speculative window can be walked forward and
     * rewound (grammar.h says why). Everything create() fixes -- the rules, the vocabulary, the
     * decoded vocabulary, the trigger words -- is absent from the snapshot by construction;
     * what is here is exactly what accepting one token can change. The trigger fields are in it
     * because a draft token may fire the trigger mid-window, which is a state change like any
     * other and has to rewind with the rest.
     *
     * Nestable, though the sampler only ever needs one level. */
    void push_state();
    int  rollback();

private:
    /* A token that may still hold the start of a trigger word: its id and the stream offset of its
     * first byte. It ends where the next one starts, or at the scan's current offset. */
    struct Recent {
        int32_t  token;
        uint64_t start;
    };

    struct Snapshot {
        GbnfStacks  stacks;
        PartialUtf8 partial;
        bool        awaiting_trigger = false;
        WordScanner::State      scan;
        std::vector<Recent>     recent;
    };

    const VocabView* vocab_ = nullptr;

    /* THE RULES ARE SHARED AND THE POSITION IN THEM IS NOT. `stacks_` holds pointers INTO
     * prog_->rules, so the program has to outlive every grammar positioned in it and must never
     * be mutated once published -- which is what the shared_ptr and the const are here to say,
     * and also why the mask cache can be keyed on those pointers across requests at all. */
    std::shared_ptr<const GbnfProgram> prog_;
    GbnfStacks       stacks_;
    PartialUtf8      partial_utf8_{};

    std::vector<Snapshot>   saved_;

    GbnfState state_key(bool allow_eog, int64_t n_words) const;

    /* The mask itself, from the program's plan, with no memo. */
    int mask_compute(uint32_t* bitmask, int64_t n_words) const;

    /* THE TRIGGER, while the grammar is lazy and waiting. The words are scanned as the bytes
     * arrive and nothing already scanned is kept except the few tokens a word could still be
     * starting in -- at most the longest word's length of them -- which is also all a match
     * replays. So waiting costs the same on the ten-thousandth token of a reply as on the first,
     * and so does copying it into a snapshot. The scanner is fixed at create() and shared by
     * every copy of the state. */
    bool                                awaiting_trigger_ = false;
    std::vector<int32_t>                trigger_tokens_;
    std::shared_ptr<const WordScanner>  scanner_;
    WordScanner::State                  scan_;
    std::vector<Recent>                 recent_;

    int accept_token_piece(int32_t token, const std::string& piece);
    int accept_token_piece_impl(int32_t token, const std::string& piece);
};

/* COMPILE A GRAMMAR, on the calling thread, into the program every GbnfGrammar::create for the
 * same (text, root, vocabulary) will be served from while `out` -- or any other holder -- keeps it
 * alive. This is the whole of the host work a grammar costs before its first mask: the parse, the
 * checks and the mask plan. The server calls it on the thread that admits the request, so the
 * scheduler thread never compiles. Returns what GbnfGrammar::create returns. */
int gbnf_compile(const std::string& text, const std::string& root, const VocabView* vocab,
                 std::shared_ptr<const GbnfProgram>* out, std::string* err);

GbnfProgramInfo gbnf_program_info(const GbnfProgram& p);

/* Build the vocabulary index every grammar's plan is compiled against, if it is not built yet.
 * Worth calling once at startup: otherwise the first grammar to arrive pays for it. */
int gbnf_prepare_vocab(const VocabView* vocab, GbnfVocabInfo* info);

/* Words a bitmask needs for this vocabulary. 19 KiB at 151K tokens, which is the number spec §13
 * quotes and the reason the state machine stays on host. */
inline int64_t bitmask_words(int64_t n_vocab) { return (n_vocab + 31) / 32; }

}  /* namespace rad */
