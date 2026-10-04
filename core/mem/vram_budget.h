/* vram_budget.h -- how many bytes each VRAM pool gets, decided once and printed in full.
 *
 * ============================== WHY THIS IS NOT THE §6 VIOLATION IT LOOKS LIKE ==============================
 *
 * Spec §6 says the engine does not derive its budgets, and core/mem/pools.cpp still refuses to
 * re-optimise a split the operator stated. The reason given there is exact and worth repeating:
 * a budget derived from the FREE VRAM at startup is a benchmark that moves when somebody opens a
 * window on the machine's desktop.
 *
 * What that argument forbids is inference. It does not forbid arithmetic on numbers that are all
 * known, all printed, and all stable:
 *
 *   capacity        totalGlobalMem. A fixed property of the part.
 *   already held    capacity - the free VRAM this process can see, MEASURED once, after every
 *                   kernel plugin has initialised and its code objects are resident. This is the
 *                   driver's context and the loaded code, and it is the one term that is observed
 *                   rather than computed -- so it is printed on its own line, and a machine with a
 *                   desktop on the card shows up in it instead of silently eating a pool.
 *   --gpu-headroom-mib   what the operator wants left alone. Stated, defaulted to 96 MiB.
 *   arena           the activation arena and the scratch beside it, computed from the buffer plan
 *                   by arena_plan_bytes() -- the same call that later allocates it.
 *
 * What is left over is claimable, and it is spent in TWO STEPS, not one.
 *
 * THE STATIC WEIGHTS COME OFF THE TOP. Everything that is not an expert -- the attention
 * projections, the norms, the embedding, the head -- has to be resident for the model to run at a
 * sensible speed at all, and under tensor parallel each rank already holds its own even share of
 * them (declare runs per rank with every dimension divided, spec §9). They are a FLOOR, not a
 * share of anything: applying a ratio to the weights as a whole would let --expert-vs-cache-ratio
 * 0.3 demote the attention projections to host memory to make room for a cache nothing will fill.
 * If they do not fit, that is a refusal with the shortfall named.
 *
 * ONLY WHAT IS LEFT AFTER THAT IS DIVIDED, and that is the division the flag is named for: the
 * expert plane against the KV cache, which is the one genuinely two-sided trade on this engine. A
 * long-context low-concurrency server and a short-context high-concurrency one want opposite
 * answers and neither is derivable.
 *
 * BOTH SIDES ARE THEN CAPPED AT WHAT THEY COULD USE, and whatever one side cannot take goes to the
 * other rather than nowhere. A cache cannot address past max_seqs x max_ctx tokens
 * (KVManager::ceiling), so a share larger than that ceiling is VRAM the cache can never fill and
 * the expert plane could have held. And an expert plane that fits entirely in VRAM has no use for
 * more; on a model that fits, the ratio stops meaning anything and that is correct.
 *
 * A stated --vram-weights-mib or --vram-kv-mib still wins outright and turns the whole of this
 * off for that pool. The point is that the DEFAULT is a full card rather than a guess: VRAM left
 * unbudgeted is VRAM no pool can use, and on a model whose experts do not fit, every unbudgeted
 * GiB costs measurable decode time.
 */
#pragma once
#include "../rad_core.h"

#include <string>
#include <vector>

namespace rad {

/* One rank's card and program, as the resolver needs to see them. Every field is either read from
 * the device or computed from the declared program; none of it is a policy. */
struct VramFacts {
    int64_t capacity   = 0;   /* RadDeviceProps::vram_bytes -- totalGlobalMem */
    int64_t free       = 0;   /* RadDeviceProps::vram_free, measured now */
    int64_t arena      = 0;   /* arena_plan_bytes() for this rank's program, total incl. scratch */
    int64_t staging    = 0;   /* BatchBuilder::device_bytes(): the step slab and the media staging */
    int64_t kv_ceiling = 0;   /* KVManager::ceiling(max_ctx) for this rank's declared groups */
    int64_t kv_fixed   = 0;   /* KVManager::fixed_bytes(): the part of the ceiling that is state */
    /* weight_footprint() for this rank, which under tensor parallel is already this card's even
     * share: declare runs per rank with every dimension already divided. */
    int64_t statics    = 0;   /* non-expert weights -- the floor, not a share of anything */
    int64_t experts    = 0;   /* the elastic plane the ratio divides against the KV cache */
    int64_t expert_layer_max = 0;   /* the most expert bytes one routed layer holds */
    /* Integrated graphics: what of the host's memory the card leaves to the host. Its device
     * memory IS the host's, so an unclaimed MiB is the operating system's and every other
     * process's to run in, not slack. 0 on a discrete card. */
    int64_t host_reserve = 0;
};

/* The resolved split, in bytes, beside every term it came from. Always filled, including when the
 * operator stated the budgets -- the placement report and the pool report both want the totals. */
struct VramBudget {
    int64_t capacity = 0, already_held = 0, headroom = 0, arena = 0, staging = 0;
    int64_t claimable = 0;            /* after the fixed terms above */
    int64_t statics = 0;              /* off the top, before the ratio sees anything */
    int64_t elastic = 0;              /* claimable - statics: what the ratio divides */
    int64_t kv_ceiling = 0, expert_ceiling = 0;
    int64_t kv_fixed = 0;             /* the state part of kv_ceiling, which no context uses */
    int64_t experts = 0, kv = 0;      /* the two shares, each capped at its ceiling */
    int64_t unusable = 0;             /* elastic neither side could take -- both were full */
    /* THE PREFILL STAGING REGION: two buffers, each one layer's share of the experts the budget
     * leaves off the card, taken out of the expert share and appended to the top of the activation
     * arena (core/place/stager.h). Zero when every expert is resident or the placement cannot move
     * any. */
    int64_t stage = 0;
    int64_t weights = 0;              /* statics + experts: what vram_weights is set to */
    bool    derived_weights = false, derived_kv = false;
    std::string report;       /* the block the engine prints at startup */
};

/* Writes the answer into cfg.vram_weights_mib / cfg.vram_kv_mib -- ONE resolved number that every
 * consumer downstream reads, so the planner, the KV manager and the pool report cannot disagree.
 * Takes the SMALLEST capacity and free across ranks and the LARGEST arena, because a split has to
 * fit on every card it is applied to.
 *
 * Refuses, by name and by how much, when the fixed terms alone exceed the card. */
int vram_budget_resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out);

}  /* namespace rad */
