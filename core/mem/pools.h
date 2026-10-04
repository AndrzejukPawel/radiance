/* pools.h -- explicit pool budgets (spec §6).
 *
 * Every pool is budgeted EXPLICITLY. No profiling pass, no utilisation fraction, no inference
 * about what the workload will be. The trade between KV blocks and resident weights is real and
 * workload-dependent -- a long-context low-concurrency server and a short-context high-concurrency
 * one want opposite splits -- but it is the operator's to make. An engine that quietly
 * re-optimises the split is an engine whose benchmarks do not reproduce.
 *
 * Two properties this file exists to make STRUCTURAL rather than conventional:
 *
 *   1. A Pool holds a base, a size and a cursor, and has no pointer to any other Pool. There is
 *      no allocate-from-wherever call anywhere in this header. Therefore no code path exists
 *      along which the mover's expert promotion (spec §5.4) can reach a KV block: the mover is
 *      handed Pools::weights() and it is not possible to reach Pools::kv() from it. That is the
 *      whole reason the weight slab is a fixed budget.
 *
 *   2. A reservation that does not fit returns null and RECORDS the shortfall by name and by how
 *      much. It never falls through to another pool, never shrinks an earlier reservation, and
 *      never silently rounds the request down. The engine refuses to start instead.
 */
#pragma once
#include "rad_core.h"

#include "../device/device.h"

#include <string>
#include <string_view>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the allocator seam */
/* Everything here goes through an allocator object rather than calling rad_dev_alloc directly.
 * The host backend is the primary development target and not a fallback
 * (docs/IMPLEMENTATION.md), and a pool test that cannot run without ROCm is a pool test that does
 * not run. The device allocator is the default; the malloc one is what the no-device build and
 * the tests use. */
class MemAllocator {
public:
    virtual ~MemAllocator() = default;
    virtual void* alloc(int64_t bytes, int kind) = 0;   /* kind is RAD_MEM_* */
    virtual void  free(void* p, int kind) = 0;
    virtual const char* name() const = 0;
};

MemAllocator& rad_device_allocator();   /* rad_dev_alloc / rad_dev_free */
MemAllocator& rad_malloc_allocator();   /* aligned_alloc; host-only builds and tests */

/* ------------------------------------------------------------------ shortfalls */
/* What did not fit, by name and by how much. Every refusal in this component produces one of
 * these rather than a bare RAD_E_NOMEM, because "it returned -6" is not a diagnosis. */
struct Shortfall {
    std::string pool;       /* which budget: "vram_weights", "vram_kv", ... */
    std::string what;       /* what was being reserved when it ran out */
    int64_t     wanted = 0;
    int64_t     available = 0;
    int64_t     missing() const { return wanted - available; }
    std::string str() const;
};

/* ------------------------------------------------------------------ Pool */
/* One reservation, made once, that never grows, never shrinks and never borrows.
 *
 * `accounting_only` pools hold no memory: the headroom budget is memory we promise NOT to touch,
 * for activations and the arena (spec §6). The arena owner still reserves against it, so an arena
 * that would exceed the headroom refuses at startup by name instead of succeeding here and
 * failing the first prefill on the device. */
class Pool {
public:
    Pool() = default;
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    const char* name()  const { return name_.c_str(); }
    int64_t     bytes() const { return bytes_; }
    int64_t     used()  const { return used_; }
    int64_t     avail() const { return bytes_ - used_; }
    void*       base()  const { return base_; }
    int         kind()  const { return kind_; }
    bool        accounting_only() const { return accounting_; }

    /* Carve `bytes` at `align`. Returns the pointer (null for an accounting-only pool, which
     * still tracks the cursor), or null with a recorded shortfall if it does not fit.
     * `off_out`, when given, receives the byte offset from base -- which is what a slab of
     * fixed-stride slots actually wants, since the offset survives a base that is a device
     * pointer nobody may dereference on the host. */
    void* reserve(std::string_view what, int64_t bytes, int64_t align, int64_t* off_out = nullptr);

    /* True when nothing was refused. Checked once, at the end of startup, so the report names
     * every failure rather than the first one at the top (spec §3.1's rule, applied here). */
    bool ok() const { return shortfalls_.empty(); }

    /* ---- elastic pools ----------------------------------------------------------------------
     *
     * An elastic pool reserves its ADDRESS RANGE whole and backs only the part somebody has asked
     * for. Every offset reserve() hands out is final from the first call, so a slab stride or a
     * layer stride computed against the full range stays correct while the physical memory behind
     * the unused part goes back to the card for another pool to take. That is the only way two
     * pools on one card can trade: the addresses cannot move, so the pages have to.
     *
     * The two pools this exists for are the KV cache and the expert slab, and the trade is not
     * symmetric. A KV block holds a sequence's state and can only be given back once nothing
     * holds it; an expert slot holds a COPY of a weight that also lives in the host pool or the
     * container, so it can be given back whenever the mover can spare it. That asymmetry is the
     * policy's, not this file's -- what is here is the mechanism and the refusal.
     *
     * A pool is elastic only where the device layer has virtual memory management (VMem in
     * core/device/device.h). Where it does not, init_elastic() allocates the whole range exactly
     * as init() does and commit()/decommit() are no-ops that succeed: a caller written for an
     * elastic pool runs unchanged against a fixed one, holding the maximum it was reserved. That
     * is the degradation spec §6 allows, because the pool is still one explicit budget -- it is
     * simply one that cannot give anything back. */
    bool    elastic()   const { return elastic_; }
    int64_t committed() const { return elastic_ ? vm_.committed() : bytes_; }

    /* Back [off, off+bytes) so a kernel may touch it. Rounds OUTWARD to the commit granule, so
     * what was asked for is certainly backed, and is idempotent: a pool growing to n asks for
     * [0, n) every time n changes and pays only for the part that is new. */
    int  commit(int64_t off, int64_t bytes);

    /* Hand the pages back. Rounds INWARD, so a granule holding one byte the caller still wants
     * stays -- a caller that shrinks to exactly the boundary it is holding is therefore safe, and
     * one that shrinks between boundaries simply keeps a granule it will use next time. */
    int  decommit(int64_t off, int64_t bytes);

    bool is_committed(int64_t off, int64_t bytes) const;

    /* The commit quantum, so a caller can pick grow steps that do not strand a fraction of one in
     * every stripe. 0 when the pool is not elastic. */
    static int64_t granule();

    const std::vector<Shortfall>& shortfalls() const { return shortfalls_; }

    std::string report() const;

private:
    /* Reserve the address range and back nothing. `bytes` is the MAXIMUM the pool may ever hold
     * and is what reserve() carves against; what it costs the card is committed(). Falls back to
     * a whole allocation where the device layer has no virtual memory management. */
    int init_elastic(std::string_view nm, int64_t bytes, int kind, MemAllocator* a);

    friend class Pools;
    int init(std::string_view nm, int64_t bytes, int kind, MemAllocator* a, bool accounting);
    void release();

    std::string   name_;
    int64_t       bytes_ = 0;
    int64_t       used_  = 0;
    void*         base_  = nullptr;
    int           kind_  = RAD_MEM_DEVICE;
    bool          accounting_ = false;
    bool          elastic_ = false;
    VMem          vm_;
    MemAllocator* alloc_ = nullptr;
    std::vector<Shortfall> shortfalls_;
    struct Res { std::string what; int64_t off, bytes; };
    std::vector<Res> res_;
};

/* ------------------------------------------------------------------ Pools */
class Pools {
public:
    /* `vram_capacity` is what the device reports free. Pass 0 to skip the capacity check, which
     * is what the host backend wants -- it has no meaningful answer and inventing one would be
     * exactly the inference §6 forbids.
     *
     * Refuses, by name and by how much, rather than shrinking anything:
     *   - if the operator did not state the weight and KV budgets at all;
     *   - if weights + kv exceeds the card;
     *   - if a pool's backing allocation fails.
     */
    int configure(const Config& cfg, int64_t vram_capacity,
                  MemAllocator* device = nullptr, MemAllocator* host = nullptr);

    Pool& weights()     { return weights_; }
    Pool& kv()          { return kv_; }
    Pool& host_pinned() { return host_; }

    const Pool& weights()     const { return weights_; }
    const Pool& kv()          const { return kv_; }
    const Pool& host_pinned() const { return host_; }

    bool configured() const { return configured_; }

    /* The line the engine prints at startup and the reason it refuses. Always populated, even on
     * success, because the placement report wants the pool totals (spec §16). */
    const std::string& fit_report() const { return fit_report_; }

    /* Every shortfall from every pool, gathered. */
    std::vector<Shortfall> shortfalls() const;

private:
    Pool weights_, kv_, host_;
    std::string fit_report_;
    bool configured_ = false;
};

}  /* namespace rad */
