/* gbnf_internal.h -- what the GBNF state machine's three translation units share, and nothing
 * else includes.
 *
 *   gbnf.cpp        the lifted machine: parser, stack machinery, accepting tokens, the compiled
 *                   program and its caches.
 *   gbnf_mask.cpp   the mask engine: the vocabulary index, the stack-set automaton and the
 *                   precomputed mask plan every fill_mask is served from.
 *   tests/gbnf_oracle.cpp
 *                   upstream's candidate walk, kept as the oracle the mask engine is tested
 *                   against and nowhere on a serving path.
 *
 * The stack machinery lives here rather than in one of them because all three must run the SAME
 * closure and the same character tests: the oracle is a second implementation of the mask, not of
 * the parse, and a difference between the two has to mean a defect in the mask engine.
 */
#pragma once
#include "gbnf.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rad {
namespace gbnf {

/* ============================================================ what a grammar may cost

 * THE GRAMMAR TEXT IS CLIENT DATA, AND IT IS COMPILED AND WALKED ON SHARED THREADS: the HTTP
 * thread that admits the request, and the scheduler thread every other request's tokens come
 * from. So nothing a grammar says may make either one use unbounded memory, time or stack -- a
 * few hundred bytes of text that exhausts the machine is a request that ends every other
 * request with it. Every bound below is one of those, and each is set well above what real
 * grammars reach: a JSON schema of 500 tool calls compiles to about 1.3 MB of text, 22,000 rules
 * and a few megabytes of compiled grammar, nests parentheses two deep, and never holds more than
 * about 1,500 stacks at once; a schema nested 128 objects deep holds stacks about 130 frames
 * tall.
 *
 * WHAT A GRAMMAR MAY COMPILE TO, BEYOND ITS OWN TEXT. The parse itself is linear in the text
 * except where a construct amplifies it: a repetition copies its operand once per count, and a
 * generated rule's name repeats the name of the rule it came from. Those are charged here, so a
 * 100 KB literal under `{1999}` is refused before it allocates the 1.6 GB it asks for, and what
 * a program in the cache can hold is bounded by its text plus this. */
inline constexpr uint64_t kMaxCompileBytes = 32ull << 20;

/* How deeply parentheses may nest. The parser is recursive descent, a pair of frames per level,
 * so an unbounded depth is a stack overflow that 40 KB of '(' reaches. */
inline constexpr uint32_t kMaxNesting = 256;

/* One SET OF STACKS, which is the whole state of the pushdown automaton at a position: how many
 * distinct stacks it may hold, and how many elements across all of them. A grammar can make the
 * set grow without end -- ambiguity that forks a stack per token, or a closure whose paths
 * multiply -- and every accept, every mask and every cached state copies it. */
inline constexpr size_t kMaxStacks   = 1u << 14;
inline constexpr size_t kMaxSetElems = 1u << 18;

/* Stack elements one epsilon closure may push while exploring, and one set may hold as explored,
 * which bounds the memory a closure holds. A grammar with no left recursion still has closures
 * whose PATHS multiply (each level of `r ::= s "x" | s "y"` doubles them), so the set bound above
 * is reached only after the exploring has been done; this is what stops it. */
inline constexpr size_t kMaxClosureWork = 1u << 20;

/* THE WORK ONE OPERATION MAY DO: one accepted token, one mask. The bounds above cap sizes, and
 * sizes do not cap time: a closure pays for every stack it builds, and a walk pays for every trie
 * node it visits. So the machinery counts what it does against one budget per operation, in units
 * of roughly a nanosecond of the thread it runs on: a stack element copied or a char-range element
 * tested is one; creating a stack -- an allocation and an insertion into the explored set -- costs
 * kStackCost more. A grammar that needs more than this at one position fails the request rather
 * than holding the scheduler thread for more than about half a second. */
inline constexpr uint64_t kStackCost = 64;
inline constexpr uint64_t kMaxOpWork = 1ull << 29;

/* THE WORK ONE COMPILE MAY DO, in the same units. Compiling precomputes the mask plan (see
 * gbnf_mask.cpp), which walks the vocabulary once for every place in the grammar where a mask can
 * start: on a quarter-million-piece vocabulary a tool-calling grammar needs a few tens of millions
 * of units, 200 tools about two hundred million, and a string bounded past the length of the
 * vocabulary's longest token -- the most a grammar can make the plan walk -- under two billion.
 * It runs on the thread that admits the request, never on the scheduler thread, and a grammar
 * that needs more than this is refused there, in a few seconds at most. */
inline constexpr uint64_t kMaxCompileWork = 1ull << 31;

/* WHAT ONE COMPILED PLAN MAY HOLD: its precomputed masks and remainder tries. A plan is shared by
 * every request that uses the grammar and is kept for later ones, so this is what one grammar may
 * cost the process while it is cached. A 200-tool grammar holds about 16 MiB on a
 * quarter-million-piece vocabulary, and a string bounded past the longest token about 40. */
inline constexpr size_t kMaxPlanBytes = 64ull << 20;

/* What the programs a vocabulary's cache keeps PINNED may hold together, beside the count it pins.
 * A program some request still holds is kept whatever this says; one that nothing holds is what
 * this bounds. */
inline constexpr size_t kMaxPinnedBytes = 256ull << 20;

/* The stack-set automaton is a memo that grows as states are discovered, and a grammar whose
 * stacks deepen with every token discovers new states forever. Past the soft bound it is dropped
 * and rebuilt lazily at the next mask, which costs warm-up and nothing else; the hard bound is
 * what one operation may add before it is refused. */
inline constexpr size_t kMaxDfaElems     = 1u << 22;
inline constexpr size_t kMaxDfaWalkElems = 1u << 23;

/* A bound above was reached. Thrown from inside the stack machinery, which has no status path
 * (it is lifted code, recursive in places), and caught at every public entry point, where it
 * becomes a status and the request is refused or failed -- never the process. */
struct GbnfLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void throw_set_limit();

/* The work counter for one operation, passed to everything the operation calls. */
struct OpWork {
    uint64_t used  = 0;
    uint64_t limit = kMaxOpWork;

    void add(uint64_t n) {
        used += n;
        if (used > limit) over();
    }

private:
    [[noreturn]] void over() const;
};

uint64_t stack_hash(const GbnfStack& s);

struct StackLess {
    bool operator()(const GbnfStack& a, const GbnfStack& b) const {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
            [](const GbnfElement* pa, const GbnfElement* pb) { return pa < pb; });
    }
};

/* ONE STACK SET WHILE IT IS BUILT, shared by every call that adds to that set: its running size,
 * which is what makes the set bounds exact rather than per call; the operation it is built for,
 * whose work every addition is charged to; and the stacks already explored for it.
 *
 * THE EXPLORED STACKS ARE THE SET'S, NOT ONE CLOSURE'S. Every stack a closure pops has had its
 * whole closure added to the set, so a later closure into the same set that reaches it has
 * nothing left to add there. Keeping `seen` per call instead re-derives it every time -- and when
 * many stacks converge on one continuation, as every alternative of a wide rule does once it
 * matches, that is one full re-expansion per stack, quadratic in a width the bounds allow. The
 * set that comes out, and the order it comes out in, are the same either way. */
struct SetBuild {
    explicit SetBuild(OpWork& w) : work(w) {}

    OpWork& work;
    size_t  stacks = 0, elems = 0;

    std::set<GbnfStack, StackLess> seen;
    size_t                         seen_elems = 0;

    void add(size_t stack_size) {
        stacks += 1;
        elems  += stack_size + 1;
        if (stacks > kMaxStacks || elems > kMaxSetElems) throw_set_limit();
    }

    /* Append `s` to `out` -- the set this belongs to -- unless it is already there.
     *
     * A SET OF A FEW STACKS IS SEARCHED, A LARGE ONE IS INDEXED. The search is what the lifted
     * code does and is the fastest thing at the sizes real states have; but many stacks that
     * converge on one continuation each re-expand it into the same set, and a linear search per
     * insertion makes that quadratic in a set size the bounds allow. Past kIndexFrom the set is
     * indexed by hash, and the order stacks land in `out` is the same either way. */
    void insert(GbnfStacks& out, GbnfStack s);

private:
    static constexpr size_t kIndexFrom = 16;
    std::unordered_multimap<uint64_t, size_t> index;
};

/* ============================================================ UTF-8 */

/* One code point from NUL-terminated text. Assumes valid UTF-8 but checks for overrun. */
std::pair<uint32_t, const char*> decode_utf8(const char* src);

/* Decode a piece, carrying an incomplete sequence in from the previous token and out to the next.
 * The returned code-point vector is 0-terminated. An invalid sequence leaves no code points and
 * a carried state of {0, -1}. */
std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8(const std::string& src,
                                                         PartialUtf8 partial_start);

/* ============================================================ elements */

inline bool is_end_of_sequence(const GbnfElement* pos) {
    return pos->type == GRE_END || pos->type == GRE_ALT;
}

inline bool is_char_elem(const GbnfElement* pos) {
    return pos->type == GRE_CHAR || pos->type == GRE_CHAR_NOT || pos->type == GRE_CHAR_ANY;
}

inline bool is_token_elem(const GbnfElement* pos) {
    return pos->type == GRE_TOKEN || pos->type == GRE_TOKEN_NOT;
}

/* True iff chr satisfies the char range at pos (regular or inverse), and the element after the
 * range. pos must point at a char range element. */
inline std::pair<bool, const GbnfElement*> match_char(const GbnfElement* pos, uint32_t chr) {
    bool found = false;
    const bool is_positive = pos->type == GRE_CHAR || pos->type == GRE_CHAR_ANY;
    do {
        if (pos[1].type == GRE_CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == GRE_CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == GRE_CHAR_ALT);
    return { found == is_positive, pos };
}

/* True iff SOME completion of a partial UTF-8 sequence could satisfy the range at pos. This is
 * what lets a byte-level BPE token that ends mid-codepoint stay admissible. */
inline bool match_partial_char(const GbnfElement* pos, PartialUtf8 partial) {
    const bool is_positive = pos->type == GRE_CHAR || pos->type == GRE_CHAR_ANY;

    const uint32_t partial_value = partial.value;
    const int      n_remain      = partial.n_remain;
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) return false;

    uint32_t low  = partial_value << (n_remain * 6);
    const uint32_t high = low | (uint32_t)((1 << (n_remain * 6)) - 1);
    if (low == 0) {
        if (n_remain == 2)      low = 1u << 11;
        else if (n_remain == 3) low = 1u << 16;
    }

    do {
        if (pos[1].type == GRE_CHAR_RNG_UPPER) {
            if (pos->value <= high && low <= pos[1].value) return is_positive;
            pos += 2;
        } else if (pos->type == GRE_CHAR_ANY) {
            return true;
        } else {
            if (low <= pos->value && pos->value <= high) return is_positive;
            pos += 1;
        }
    } while (pos->type == GRE_CHAR_ALT);

    return !is_positive;
}

/* How many elements the char range at pos spans, which is what one match_char on it costs: it
 * walks the whole range whatever the code point. */
inline size_t char_range_len(const GbnfElement* pos) {
    return (size_t)(match_char(pos, 0).second - pos);
}

inline bool match_token(const GbnfElement* pos, int32_t token) {
    if (pos->type == GRE_TOKEN)     return pos->value == (uint32_t)token;
    if (pos->type == GRE_TOKEN_NOT) return pos->value != (uint32_t)token;
    return false;
}

/* ============================================================ the compiled rules

 * EVERY RULE IN ONE ARENA, so an element's address is also its index. The stacks hold element
 * pointers, and the mask engine needs a fact about every element a stack names -- which class of
 * continuation it is (gbnf_mask.cpp) -- on every mask; with the rules held as separate vectors
 * that is a search per element, with one arena it is a subtraction. The arena is final once the
 * program is published: every stack points into it. */
struct GbnfRuleSet {
    std::vector<GbnfElement> elems;
    std::vector<uint32_t>    start;     /* rule id -> index of its first element */

    const GbnfElement* rule(size_t id) const { return elems.data() + start[id]; }
    uint32_t index(const GbnfElement* e) const { return (uint32_t)(e - elems.data()); }
    size_t   n_rules() const { return start.size(); }
};

/* Expand a stack into the N stacks that each end at a terminal. This is the pushdown automaton's
 * epsilon closure and it is where rule references are resolved. `build` is `out` while it is
 * built; see SetBuild. Throws GbnfLimit past the bounds above. */
void advance_stack(const GbnfRuleSet& rules, const GbnfStack& stack, GbnfStacks& out,
                   SetBuild& build);

/* The rules a compiled program's stacks point into. */
const GbnfRuleSet& program_rules(const GbnfProgram& p);

/* ============================================================ the mask engine (gbnf_mask.cpp) */

/* Everything the mask engine derives from a vocabulary: each piece decoded to code points, a
 * trie over them, and the index of pieces that resume a split character. Built once per
 * vocabulary and shared by every grammar compiled against it. */
struct VocabIndex;

/* Throws std::bad_alloc; a vocabulary with no text in it builds an empty index. */
std::shared_ptr<const VocabIndex> vocab_index_build(const VocabView& v);
size_t vocab_index_bytes(const VocabIndex& x);

/* What compiling a grammar's mask plan cost and holds, for the log line and the acceptance
 * tool. */
struct PlanStats {
    double   ms          = 0.0;   /* wall time of the plan build */
    uint64_t work        = 0;     /* units charged against kMaxCompileWork */
    size_t   classes     = 0;     /* distinct stack-entry classes */
    size_t   keys        = 0;     /* stack suffixes with a precomputed mask */
    size_t   max_depth   = 0;     /* the deepest of them, in stack entries */
    size_t   dense       = 0;     /* of them, held as a full bitmask */
    size_t   rem_nodes   = 0;     /* nodes across every remainder trie */
    size_t   max_rem     = 0;     /* the largest one remainder trie */
    size_t   bytes       = 0;     /* what the plan holds */
};

/* A grammar's compiled mask plan. Its runtime half -- an automaton over stack sets and the scratch
 * a mask needs -- is a memo guarded by the plan's own lock, so a plan is shared by every request
 * and every rank that uses the grammar. */
struct MaskPlan;
struct MaskPlanDeleter { void operator()(MaskPlan* p) const; };
using MaskPlanPtr = std::unique_ptr<MaskPlan, MaskPlanDeleter>;

/* Build the plan for a grammar whose rules are final. Returns RAD_OK or RAD_E_UNSUPPORTED with
 * the reason in *err, naming the rule from `rule_names` (by rule id) where it can: a grammar whose
 * masks cannot be precomputed within the bounds above is refused here rather than served slowly
 * later. Throws GbnfLimit or std::bad_alloc, which the caller turns into a status. */
int plan_build(const GbnfRuleSet& rules, uint32_t root, const std::shared_ptr<const VocabIndex>& vx,
               const std::vector<std::string>* rule_names, MaskPlanPtr* out, PlanStats* stats,
               std::string* err);

/* The mask for a stack set: bit set means admissible. `n_words` covers the vocabulary. Returns
 * RAD_OK, or RAD_E_STATE if no token at all is admissible; throws GbnfLimit past the per-operation
 * bounds. Thread-safe: it takes the plan's lock. */
int plan_fill(const MaskPlan& plan, const GbnfStacks& stacks, PartialUtf8 partial, bool allow_eog,
              uint32_t* bitmask, int64_t n_words);

}  /* namespace gbnf */
}  /* namespace rad */
