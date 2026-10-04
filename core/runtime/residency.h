/* residency.h -- the seam the placement component plugs into.
 *
 * The run phase takes weight HANDLES, never pointers (spec §3.2). It resolves a handle to a
 * pointer at issue time, which is the only reason the mover is allowed to relocate a weight while
 * the model is serving. This header is the whole of the contract between the two: the run phase
 * asks a question that never blocks, and the placement component answers it out of a table it
 * owns.
 *
 * The degenerate case matters as much as the interesting one. In the all-in-VRAM deployment the
 * planner assigned everything to tier 0 and the mover never runs, so no ResidencyTable is
 * installed at all and the lookup collapses to WeightInfo::ptr -- a pointer the run phase already
 * baked into its per-weight tensor template. There is no virtual call, no branch on a table entry,
 * and nothing to pay for a feature that deployment is not using.
 */
#pragma once
#include "../rad_core.h"

#include <vector>

namespace rad {

/* Where a weight's bytes are RIGHT NOW, and whether they have arrived.
 *
 * `ready` is the mover's event. Null means resident and settled: nothing to wait on, which is the
 * common case even under expert offload because a hot expert stays put for many steps. Non-null
 * means a transfer into `ptr` is in flight and the compute stream must be ordered behind it -- a
 * STREAM wait, never a host sync, which is what lets placement be dynamic without serialising the
 * step (spec §5.4).
 *
 * `generation` is what makes that wait cheap, and it is a contract the mover has to keep: it MUST
 * be bumped whenever a new transfer into this slot begins. The run phase memoises "the compute
 * stream has already been made to wait on generation g of weight w" and skips the wait every time
 * afterwards. That is sound because a stream wait orders everything issued after it, so one wait
 * per generation is sufficient and per-issue waits are pure cost -- at 64 layers and eight experts
 * a step they would be thousands of runtime calls buying nothing. A mover that reuses an event
 * without bumping the generation silently loses the ordering, and the symptom is garbage output
 * three layers later with no way to bisect it.
 */
struct WeightSlot {
    void*    ptr;
    RadEvent ready;
    uint32_t generation;
};

class ResidencyTable {
public:
    virtual ~ResidencyTable() = default;

    /* NEVER blocks. Not "usually does not block" -- this is called once per weight operand per
     * issue, inside the step, and a lock here would serialise the mover against compute, which is
     * the exact failure this design exists to avoid. Returning null means "I have nothing for this
     * handle" and the run phase falls back to WeightInfo::ptr. */
    virtual const WeightSlot* lookup(rad_weight) const = 0;

    /* A count that changes whenever any slot's pointer, event or generation does. Everything
     * lookup() returns is a function of it, so a caller that resolved a run of weights at one
     * epoch may reuse that answer for as long as the epoch stands -- which is what keeps a routed
     * layer's expert table from being walked, entry by entry, at every issue. */
    virtual uint64_t epoch() const = 0;

    /* THE SLOTS WHOSE POINTER OR EVENT CHANGED since the run phase last cleared the list, each
     * named once. A routed layer's weight table holds its whole expert set and a step moves a
     * handful of experts, so the run phase updates the entries named here rather than resolving
     * every entry of every table whenever the epoch moves -- which, with experts moving on most
     * steps, would be every entry of every table on most steps.
     *
     * One reader: the run phase of the rank this table belongs to, on the thread that also drives
     * the writers, so neither call races the other. A slot that only SETTLED -- its transfer
     * landed, the pointer stayed -- is not named, because nothing a reader holds becomes wrong. */
    virtual const std::vector<rad_weight>& changes() const = 0;
    virtual void clear_changes() = 0;
};

}  /* namespace rad */
