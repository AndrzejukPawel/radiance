/* gbnf_oracle.cpp -- upstream's candidate walk (llama.cpp `llama_grammar_reject_candidates` and
 * `llama_grammar_apply_impl`, at the commit gbnf.h names), over the engine's own stack machinery.
 * See gbnf_oracle.h for why it exists and where it may be used.
 *
 * The closure, the character tests and the UTF-8 decoding are the engine's (gbnf_internal.h),
 * because they are the PARSE and this is a second implementation of the MASK: the thing under
 * test is how a mask is assembled from a stack set, and sharing everything below that is what
 * makes a disagreement mean a defect in the assembly.
 */
#include "gbnf_oracle.h"

#include "sample/gbnf_internal.h"

#include <algorithm>
#include <new>

namespace rad {

namespace {

using namespace gbnf;

struct Candidate {
    size_t          index;
    const uint32_t* code_points;
    PartialUtf8     partial_utf8;
    int32_t         id;
};
using Candidates = std::vector<Candidate>;

/* What carrying one candidate through one element of a character range costs, against the same
 * per-operation bound the engine is held to. */
constexpr uint64_t kCandidateCost = 4;

Candidates reject_for_stack(const GbnfRuleSet& rules, const GbnfStack& stack,
                            const Candidates& candidates, OpWork& work);

Candidates reject_candidates(const GbnfRuleSet& rules, const GbnfStacks& stacks,
                             const Candidates& candidates, OpWork& work) {
    if (candidates.empty() || stacks.empty()) return {};
    Candidates rejects = reject_for_stack(rules, stacks.front(), candidates, work);
    for (size_t i = 1; i < stacks.size(); ++i)
        rejects = reject_for_stack(rules, stacks[i], rejects, work);
    return rejects;
}

Candidates reject_for_stack(const GbnfRuleSet& rules, const GbnfStack& stack,
                            const Candidates& candidates, OpWork& work) {
    {
        const GbnfElement* t = stack.empty() ? nullptr : stack.back();
        const bool range = t && !is_token_elem(t);
        work.add(candidates.size() * kCandidateCost * (range ? char_range_len(t) : 1) +
                 kStackCost);
    }
    Candidates rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        /* A completed stack accepts nothing further, so anything with content left is rejected. */
        for (const auto& t : candidates)
            if (*t.code_points != 0 || t.partial_utf8.n_remain != 0) rejects.push_back(t);
        return rejects;
    }

    const GbnfElement* top = stack.back();

    if (is_token_elem(top)) {
        for (const auto& t : candidates) {
            if (*t.code_points == 0) {
                if (t.partial_utf8.n_remain != 0) rejects.push_back(t);
            } else if (!match_token(top, t.id)) {
                rejects.push_back(t);
            }
        }
        return rejects;
    }

    Candidates next;
    next.reserve(candidates.size());
    for (const auto& t : candidates) {
        if (*t.code_points == 0) {
            if (t.partial_utf8.n_remain != 0 && !match_partial_char(top, t.partial_utf8))
                rejects.push_back(t);
        } else if (match_char(top, *t.code_points).first) {
            next.push_back({ t.index, t.code_points + 1, t.partial_utf8, t.id });
        } else {
            rejects.push_back(t);
        }
    }

    const GbnfElement* after = match_char(top, 0).second;
    GbnfStack stack_after(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(after)) stack_after.push_back(after);

    GbnfStacks next_stacks;
    SetBuild   next_set(work);
    advance_stack(rules, stack_after, next_stacks, next_set);

    for (const auto& t : reject_candidates(rules, next_stacks, next, work))
        rejects.push_back({ t.index, t.code_points - 1, t.partial_utf8, t.id });

    return rejects;
}

int oracle_impl(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words, uint64_t work_limit) {
    const VocabView&   vocab = *g.vocab();
    const GbnfRuleSet& rules = program_rules(g.program());
    const int32_t      n_tok = vocab.n_tokens();
    const bool         allow_eog = g.complete();

    /* Decode every piece against the carried partial, then run the whole vocabulary through the
     * reject walk in one pass: every candidate admitted, then the rejects cleared. */
    std::vector<std::pair<std::vector<uint32_t>, PartialUtf8>> decoded;
    decoded.reserve((size_t)n_tok);
    Candidates cands;
    cands.reserve((size_t)n_tok);
    std::vector<int32_t> index_to_id;
    index_to_id.reserve((size_t)n_tok);

    for (int32_t id = 0; id < n_tok; ++id) {
        if (vocab.is_eog(id)) {
            if (allow_eog) bitmask[id >> 5] |= 1u << ((uint32_t)id & 31u);
            continue;
        }
        const std::string& piece = vocab.token_piece(id);
        if (piece.empty() || piece[0] == 0) continue;
        decoded.push_back(decode_utf8(piece, g.partial()));
    }
    size_t k = 0;
    for (int32_t id = 0; id < n_tok; ++id) {
        if (vocab.is_eog(id)) continue;
        const std::string& piece = vocab.token_piece(id);
        if (piece.empty() || piece[0] == 0) continue;
        cands.push_back({ index_to_id.size(), decoded[k].first.data(), decoded[k].second, id });
        index_to_id.push_back(id);
        ++k;
    }

    for (int32_t id : index_to_id) bitmask[id >> 5] |= 1u << ((uint32_t)id & 31u);

    OpWork work;
    if (work_limit) work.limit = work_limit;
    for (const auto& r : reject_candidates(rules, g.stacks(), cands, work)) {
        const int32_t id = index_to_id[r.index];
        bitmask[id >> 5] &= ~(1u << ((uint32_t)id & 31u));
    }

    for (int64_t w = 0; w < n_words; ++w)
        if (bitmask[w]) return RAD_OK;
    return RAD_E_STATE;
}

}  /* namespace */

int gbnf_oracle_mask(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words,
                     uint64_t work_limit) {
    if (!bitmask || !g.vocab()) return RAD_E_INVAL;
    if (n_words < bitmask_words(g.vocab()->n_tokens())) return RAD_E_SHAPE;
    if (g.awaiting_trigger()) {
        std::fill(bitmask, bitmask + n_words, 0xFFFFFFFFu);
        return RAD_OK;
    }
    std::fill(bitmask, bitmask + n_words, 0u);
    if (g.stacks().empty()) return RAD_E_STATE;
    try {
        return oracle_impl(g, bitmask, n_words, work_limit);
    } catch (const gbnf::GbnfLimit&) {
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        return RAD_E_NOMEM;
    }
}

}  /* namespace rad */
