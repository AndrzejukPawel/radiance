/* arena.h -- ONE contiguous allocation, sized by the buffer plan at load and never allocated from
 * at runtime.
 *
 * There is deliberately no allocate() on this class. If there were one somebody would eventually
 * call it from inside rad_arch_step, and an allocator on the step path is precisely the class of
 * dynamic behaviour this engine exists to remove (spec §3.1). Offsets come from the buffer plan,
 * which computed liveness over the declared op list and is conservative on purpose: a plan that is
 * correct for the worst step is correct for every step. The arena only owns the block those
 * offsets index into.
 *
 * The scratch region sits at the tail and is shared by every op on a stream. That is safe and not
 * merely convenient: launches on one stream execute in order, so no two kernels' scratch lifetimes
 * overlap. A context's second lane is a second stream, where that order does not hold, so a program
 * that declares one gets a second region and each lane's kernels are handed their own
 * (Program::scratch_regions). A lane used without that declaration shares region 0, and there
 * Ctx::issue orders a scratch user behind one on the other lane that nothing has ordered it after
 * yet. Program::scratch_bytes is the largest single kernel's requirement, so a region of that size
 * serves all of them -- and if a kernel asks for more than the plan reserved, prepare refuses by
 * name rather than handing it a short buffer.
 */
#pragma once
#include "../rad_core.h"

namespace rad {

class Arena {
public:
    Arena() = default;
    ~Arena() { release(); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    /* `kind` is a RAD_MEM_* class. The device arena is RAD_MEM_DEVICE; a program with
     * RAD_DOMAIN_HOST buffers gets a second arena in RAD_MEM_HOST. `regions` scratch regions of
     * `scratch_bytes` each sit at the tail, one per lane (Program::scratch_regions). */
    int  init(int64_t plan_bytes, int64_t scratch_bytes, int kind, int regions = 1);
    void release();

    char*   base()          const { return base_; }
    /* Region `r`, or region 0 when the arena holds fewer. */
    void*   scratch(int r = 0) const {
        if (!base_) return nullptr;
        return base_ + scratch_off_ + (r > 0 && r < scratch_regions_ ? r * scratch_pitch_ : 0);
    }
    int64_t scratch_bytes() const { return scratch_bytes_; }
    int     scratch_regions() const { return scratch_regions_; }
    int64_t bytes()         const { return bytes_; }
    bool    live()          const { return base_ != nullptr; }

private:
    char*   base_ = nullptr;
    int     kind_ = RAD_MEM_DEVICE;
    int64_t bytes_ = 0;
    int64_t scratch_off_ = 0;
    int64_t scratch_bytes_ = 0;
    int64_t scratch_pitch_ = 0;     /* scratch_bytes_ padded to RAD_ALIGN_UNIT */
    int     scratch_regions_ = 1;
};


/* THE ARENA'S SIZE, COMPUTED WITHOUT ALLOCATING IT. Two callers need the same answer at two
 * different times: Ctx::bind_arena, which allocates it at the end of startup, and the VRAM budget
 * resolver, which has to subtract it from the card BEFORE the pools are allocated near the top of
 * startup. Two copies of this walk would agree today and drift the first time the plan grows a
 * domain -- and the failure would be a budget that overcommits the card by exactly the difference.
 *
 * `dev_out` is what the device arena's init() will be given as plan_bytes; `total_out` is what it
 * will actually allocate, plan and scratch both padded to RAD_ALIGN_UNIT as Arena::init pads them.
 * Returns the total. */
int64_t arena_plan_bytes(const Program& P, int64_t* dev_out, int64_t* total_out);
}  /* namespace rad */
