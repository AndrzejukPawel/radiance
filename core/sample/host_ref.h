/* host_ref.h -- the host reference sampling chain.
 *
 * Sampling runs on device (spec §13): at a 151K vocab and batch 256 the logits are ~77 MiB a
 * step, and moving them across the link to run a host chain costs the bandwidth AND a
 * synchronisation, on the step path, at exactly the concurrency the engine exists to serve. So
 * every sampler is a kernel, not a lift.
 *
 * This file is the other half of that sentence. A fallback you cannot test against is just an
 * untested slow path (spec §17), so the device samplers need an oracle, and the oracle is
 * llama.cpp's chain ported to plain C++ over a float array. Ported, not linked: there are no
 * llama.cpp headers here and no ggml, because a reference that drags a second engine in is a
 * dependency, and this is a reference. Each individual sampler below reproduces the semantics of
 * its counterpart in llama.cpp `src/llama-sampler.cpp` at upstream commit 06938ac12 (MIT),
 * including its edge cases, because reproducing the happy path only is how a comparison harness
 * agrees with the thing it is checking and both are wrong.
 *
 * TWO THINGS DEVIATE FROM UPSTREAM, DELIBERATELY.
 *
 * 1. THE CHAIN ORDER. llama.cpp's default chain is penalties, DRY, top-n-sigma, top-k, typical,
 *    top-p, min-p, XTC, TEMPERATURE LAST. Radiance's order is the spec's and vLLM's: penalties,
 *    temperature, grammar mask, top-k, then top-p / min-p / typical, then the pick. The
 *    difference is not cosmetic -- top-p, min-p and typical all cut on probabilities, so whether
 *    temperature ran first changes which tokens survive. Radiance's device kernels run the
 *    radiance order (it is the order every OpenAI-compatible server is measured against), so
 *    ChainOrder::Radiance is what rad-kbench compares against. ChainOrder::LlamaCpp exists so
 *    this file can also be checked against llama.cpp itself, which is the only way to know the
 *    port is faithful.
 *
 * 2. THE RNG. Upstream carries a stateful std::mt19937 per sampler. That cannot be a device
 *    kernel: a kernel has no place to keep a Mersenne twister's 624 words of state per sequence,
 *    and a stateful generator makes the answer depend on how many draws happened earlier, which
 *    makes a comparison run irreproducible the moment the batch composition changes. Here the
 *    randomness is a pure function of (seed, position, stream) -- counter-based, stateless, and
 *    reproducible across a run with different batching. It is abi/rad_sample.h's draw, bit for
 *    bit, with the same per-stream keys, so the pick and XTC's coin here see exactly the uniforms
 *    the device stages see.
 */
#pragma once
#include "../rad_core.h"
#include "vocab_view.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== the candidate set */

/* One candidate. `p` is meaningless until a softmax has run over the set; the samplers that need
 * it call one themselves, exactly as upstream's do. */
struct Candidate {
    int32_t id    = 0;
    float   logit = 0.0f;
    float   p     = 0.0f;
};

/* The set the chain narrows. `sorted` tracks descending-by-logit order, because half the
 * samplers upstream have a fast path that depends on it and a slow path that does not, and
 * dropping the flag silently changes which one runs. */
struct CandidateSet {
    std::vector<Candidate> c;
    bool sorted   = false;
    int  selected = -1;      /* index into c, set by the pick; -1 until then */

    void from_logits(const float* logits, int64_t n_vocab);
    void sort_desc();        /* descending logit, ties broken by ascending id */
    size_t size() const { return c.size(); }
};

/* ================================================================== reproducible randomness */

/* A counter-based generator: bits are a pure function of the three coordinates, so the same draw
 * comes out on the host reference and in the device kernel, in any batch order, at any batch
 * size. It IS rad_sample_uniform01 from abi/rad_sample.h -- splitmix64 over
 * ((seed ^ key) * golden ratio + pos) -- because a reference with its own generator agrees with
 * the device on no seeded draw at all.
 *
 * `pos` is the sequence's output position -- token index within this request's generation -- so
 * a request's draws depend on where it is, not on when it was scheduled. `stream` separates
 * independent draws taken at the same position (the pick, XTC's coin, speculative acceptance's
 * uniforms) so they are not the same number; each maps to its RAD_SAMPLE_KEY_* key. */
enum : uint32_t {
    RAD_RNG_PICK    = 0,   /* the final categorical draw */
    RAD_RNG_XTC     = 1,   /* XTC's "does this fire at all" coin */
    RAD_RNG_ACCEPT  = 2,   /* speculative acceptance, one per draft position (see accept.h) */
    RAD_RNG_RESID   = 3    /* the residual draw after a speculative rejection */
};

uint64_t rng_bits(uint64_t seed, uint64_t pos, uint32_t stream);
/* Uniform in [0,1). 24 bits of mantissa: a float cannot resolve more, and pretending otherwise
 * would make the host and the device disagree in the last place. */
float    rng_uniform(uint64_t seed, uint64_t pos, uint32_t stream);

/* ================================================================== individual samplers */

/* Upstream: llama_sampler_softmax_impl. Fills `p` and normalises. */
void host_softmax(CandidateSet& s, bool do_sort);

/* Upstream: llama_sampler_temp_impl. temp <= 0 collapses to the argmax by setting every other
 * logit to -inf, which is what keeps a temperature-0 request bit-identical to a greedy one. */
void host_temp(CandidateSet& s, float temp);

/* Upstream: llama_sampler_top_k_impl. k <= 0 is off. */
void host_top_k(CandidateSet& s, int32_t k);

/* Upstream: llama_sampler_top_p_apply. Ties AT the threshold are all kept, which can overshoot
 * p by the mass of the tie -- the alternative is inventing an order among equal probabilities. */
void host_top_p(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_min_p_apply. A floor relative to the mode, not an absolute one. */
void host_min_p(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_typical_apply. Keeps the candidates whose surprisal is closest to the
 * distribution's entropy, which is not the same set as the most probable ones. */
void host_typical(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_penalties_apply, with the history passed in rather than accumulated in
 * a ring buffer. The scheduler already owns each request's token history, and a sampler that
 * kept its own copy would be a second place for it to be wrong.
 *
 * `history` is the sequence's tokens, oldest first; only the last `last_n` are counted. */
void host_penalties(CandidateSet& s, const int32_t* history, int64_t n_history,
                    int32_t last_n, float rep, float freq, float pres);

/* DRY's sequence breakers, resolved to token sequences once per request rather than per step.
 * Upstream calls this `dry_processed_breakers` and builds it from the vocabulary; the cost is a
 * full vocabulary scan per breaker string, which is why it does not belong on the step path.
 *
 * `seqs` maps a HEAD token to the tails that complete a breaker after it; an empty tail means the
 * head token contains a whole breaker by itself. The two bitsets over the vocabulary answer the
 * two questions the step path asks per token -- "does anything start here" and "is this token a
 * whole breaker" -- with one load, where the multimap is a hash lookup a token. */
struct DryBreakers {
    std::unordered_multimap<int32_t, std::vector<int32_t>> seqs;
    std::vector<uint32_t> heads;     /* bit t: token t heads at least one breaker sequence */
    std::vector<uint32_t> singles;   /* bit t: token t is a whole breaker by itself */

    bool empty() const { return seqs.empty(); }

    /* Build the two bitsets from `seqs`. dry_prepare_breakers calls it; a set filled by hand is
     * answered from the multimap until it is indexed, so the two ways of building one agree. */
    void index(int32_t n_vocab);
    bool is_head(int32_t t) const;
    bool is_single(int32_t t) const;

private:
    static bool bit(const std::vector<uint32_t>& v, int32_t t) {
        if (t < 0 || (size_t)(t >> 5) >= v.size()) return false;
        return (v[(size_t)(t >> 5)] >> ((uint32_t)t & 31u)) & 1u;
    }
};

/* THE BOUNDS ON WHAT A REQUEST MAY ASK THE BREAKER SCAN TO DO. Each breaker costs a pass over the
 * whole vocabulary on the scheduler thread, once per rank, so an unbounded list stalls every
 * sequence on the server for as long as the scan takes. Upstream truncates a breaker to 40 bytes
 * and its token tail to 20 tokens (llama_sampler_init_dry); the count bound is this engine's own,
 * well above the four breakers clients send by default, and a request over it is refused at
 * admission by name rather than silently shortened. */
constexpr size_t kDryMaxBreakers      = 16;
constexpr size_t kDryBreakerMaxBytes  = 40;
constexpr int    kDryBreakerMaxTail   = 20;

/* Upstream: get_overlapping_token_sequences, per breaker, with upstream's truncation of each
 * breaker to kDryBreakerMaxBytes (cut back to a UTF-8 boundary, so the tail still tokenises) and
 * of each tail to `max_tail_len` tokens. Refuses more than kDryMaxBreakers breakers with
 * RAD_E_INVAL. Returns RAD_E_UNSUPPORTED if the view has no tokeniser, because a DRY request
 * whose breakers silently became empty is a DRY request that quietly stopped working. */
int dry_prepare_breakers(const VocabView& vocab, const std::vector<std::string>& breakers,
                         DryBreakers* out, int max_tail_len = kDryBreakerMaxTail);

/* Upstream's step 1 of llama_sampler_dry_apply: walking back from the end of `window` (n tokens,
 * oldest first), the first token that heads a COMPLETE breaker sequence caps how long a repeat
 * may be counted. Returns that cap, or n when no breaker completes inside the window. Shared by
 * host_dry and the sampler's per-step staging, so the reference and the device stage are handed
 * the same limit. */
int dry_rep_limit(const DryBreakers& breakers, const int32_t* window, int n);

struct DryParams {
    float   multiplier     = 0.0f;
    float   base           = 1.75f;
    int32_t allowed_length = 2;
    int32_t penalty_last_n = -1;    /* -1 = the whole context */
};

/* Upstream: llama_sampler_dry_apply, Z-algorithm and all. Ported from Koboldcpp originally
 * (LostRuins/koboldcpp#982, author pi6am) by way of llama.cpp. */
void host_dry(CandidateSet& s, const int32_t* history, int64_t n_history,
              int64_t total_context, const DryParams& p, const DryBreakers& breakers);

/* Upstream: llama_sample_xtc_apply. Fires with probability `probability`; when it fires it drops
 * every candidate above the threshold except the least probable of them, which is the point --
 * it removes the obvious continuation rather than the unlikely ones. */
void host_xtc(CandidateSet& s, float probability, float threshold, size_t min_keep,
              uint64_t seed, uint64_t pos);

/* The grammar mask. One bit per token, set means admissible; a cleared bit takes the logit to
 * -inf rather than removing the candidate, so the mask composes with everything downstream
 * without renumbering anything. `n_words` is ceil(n_vocab/32). */
void host_mask(CandidateSet& s, const uint32_t* bitmask, int64_t n_words);

/* The pick. Inverse CDF over the surviving candidates in DESCENDING LOGIT ORDER, ties to the lower
 * id -- a total order, so the answer is a property of the uniform rather than of how the previous
 * stage happened to permute the set, and the order `sample_pick` walks the candidate set the
 * top-k leaves (and the order upstream's dist sampler walks a sorted set in). A candidate at
 * -infinity -- a grammar-masked logit -- is never returned, including by the fall-through a draw
 * at the very top of the range takes. Returns the token id, or -1 for an empty or fully masked
 * set. */
int32_t host_pick(CandidateSet& s, uint64_t seed, uint64_t pos);

/* Argmax over raw logits, lowest id winning a tie. The greedy path skips the chain entirely, so
 * this must agree with what host_temp(0) + host_pick would produce or a temperature-0 request
 * stops matching a greedy one. */
int32_t host_argmax(const float* logits, int64_t n_vocab);

/* ================================================================== the chain */

enum class ChainOrder {
    Radiance = 0,   /* penalties, DRY, temperature, mask, top-k, top-p/min-p/typical, XTC, pick */
    LlamaCpp = 1    /* penalties, DRY, top-k, typical, top-p, min-p, XTC, temperature, pick */
};

/* Everything the chain needs that is not in SamplingParams. Kept separate because SamplingParams
 * is frozen in rad_core.h and these are per-step facts, not per-request settings. */
struct HostRefInput {
    const int32_t* history   = nullptr;   /* the sequence's tokens so far, oldest first */
    int64_t        n_history = 0;
    int64_t        total_context = 0;     /* DRY's clamp; 0 means use n_history */
    const uint32_t* bitmask  = nullptr;   /* grammar mask, or null */
    int64_t        n_words   = 0;
    uint64_t       pos       = 0;         /* output position: the RNG's second coordinate */
    const DryBreakers* dry_breakers = nullptr;
};

/* Run the whole chain over one row of logits and return the token. This is what rad-kbench
 * compares the device sampler against, per sequence, on the same logits. */
int32_t host_sample(const float* logits, int64_t n_vocab, const SamplingParams& sp,
                    const HostRefInput& in, ChainOrder order = ChainOrder::Radiance);

/* The same, leaving the narrowed candidate set behind for a caller that wants to inspect it --
 * rad-kbench reports which stage diverged, and it cannot do that from a token id alone. */
int32_t host_sample_into(CandidateSet& s, const SamplingParams& sp, const HostRefInput& in,
                         ChainOrder order = ChainOrder::Radiance);

}  /* namespace rad */
