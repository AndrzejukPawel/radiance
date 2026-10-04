/* arena.cpp -- one allocation, sized once. See arena.h for why there is no allocate(). */
#include "arena.h"

#include <algorithm>

namespace rad {

int Arena::init(int64_t plan_bytes, int64_t scratch_bytes, int kind, int regions) {
    release();
    if (plan_bytes < 0 || scratch_bytes < 0 || regions < 1) return RAD_E_INVAL;

    /* The scratch region starts on a page boundary. Kernels that want 4 KiB alignment for a
     * workspace get it for free, and the ones that do not are not charged for it -- the padding is
     * one page against an arena measured in hundreds of megabytes. */
    const int64_t off   = align_up(plan_bytes, RAD_ALIGN_UNIT);
    const int64_t pitch = align_up(scratch_bytes, RAD_ALIGN_UNIT);
    const int64_t tail  = pitch * regions;
    int64_t       total = off + tail;

    /* A zero-byte arena would hand every buffer a null base and turn a plan bug into a segfault
     * with no name on it. One page instead, so a stray offset lands somewhere reportable. */
    if (total == 0) total = RAD_ALIGN_UNIT;

    void* p = rad_dev_alloc(total, kind);
    if (!p) {
        RAD_ERR("arena: %s of %s failed: %s",
                kind == RAD_MEM_DEVICE ? "device allocation" : "host allocation",
                humanb(total).c_str(), rad_dev_last_error());
        return RAD_E_NOMEM;
    }

    base_          = static_cast<char*>(p);
    kind_          = kind;
    bytes_         = total;
    scratch_off_   = off;
    scratch_bytes_ = scratch_bytes;
    scratch_pitch_ = pitch;
    scratch_regions_ = regions;
    return RAD_OK;
}

void Arena::release() {
    if (base_) rad_dev_free(base_, kind_);
    base_          = nullptr;
    bytes_         = 0;
    scratch_off_   = 0;
    scratch_bytes_ = 0;
    scratch_pitch_ = 0;
    scratch_regions_ = 1;
}


/* See arena.h. The walk is Ctx::bind_arena's, lifted here whole so the budget resolver and the
 * allocation cannot disagree.
 *
 * THIS ASSUMES the plan gives host-domain buffers their own offset space. If it interleaves them
 * with device buffers instead, both arenas come out sparse: correct, but paying VRAM for host
 * activations. BufferInfo carries one arena_offset and one total, so the contract has to be stated
 * somewhere and this is it. */
int64_t arena_plan_bytes(const Program& P, int64_t* dev_out, int64_t* total_out) {
    int64_t dev_hw = 0, host_hw = 0;
    for (size_t i = 1; i < P.buffers.size(); ++i) {
        const BufferInfo& bi = P.buffers[i];
        if (bi.arena_offset < 0) continue;      /* the plan did not run; bind_arena refuses by name */
        int64_t bytes = bi.bytes;
        if (bytes <= 0) {
            int64_t n = 1;
            for (uint32_t d = 0; d < bi.decl.rank; ++d) n *= bi.decl.shape[d];
            bytes = rad_dtype_bytes(bi.decl.dtype, n);
        }
        int64_t& hw = (bi.decl.domain == RAD_DOMAIN_HOST) ? host_hw : dev_hw;
        hw = std::max(hw, bi.arena_offset + bytes);
    }
    /* With no host-domain buffers the plan's own total is authoritative: it may carry alignment
     * slack the high-water mark does not see. */
    int64_t dev_bytes = host_hw > 0 ? dev_hw : std::max(dev_hw, P.arena_bytes);
    /* The staging region sits above every buffer, so it is its own term, whichever of the two
     * above set the high-water mark. */
    if (P.arena_stage_bytes > 0 && P.arena_stage_off >= 0)
        dev_bytes = std::max(dev_bytes, P.arena_stage_off + P.arena_stage_bytes);
    int64_t total = align_up(dev_bytes, RAD_ALIGN_UNIT) +
                    align_up(P.scratch_bytes, RAD_ALIGN_UNIT) * std::max<int64_t>(P.scratch_regions, 1);
    if (total == 0) total = RAD_ALIGN_UNIT;     /* Arena::init's floor, for the same reason */
    if (dev_out)   *dev_out = dev_bytes;
    if (total_out) *total_out = total;
    return total;
}
}  /* namespace rad */
