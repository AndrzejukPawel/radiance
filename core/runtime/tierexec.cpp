#include "runtime/tierexec.h"

#include "mem/sstier.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

TierExec::~TierExec() { close(); }

/* The two ops are resolved ONCE, at configure, and at the width they will actually be called at.
 * `n` is the fragment in bytes and it is a per-group constant, so nothing here re-resolves on the
 * maintenance path -- a selector call between steps would be affordable and is still the wrong
 * shape: the kernel that serves a transfer must not be able to change underneath a half-finished
 * batch. */
int TierExec::configure(Registry* reg, KVManager* kv, IdleTiers* tiers, SSDTier* disk,
                        const std::vector<int32_t>& groups, int max_batch, int64_t max_copy,
                        int64_t max_chain, bool leader, int device, int64_t slot_off,
                        int64_t ck_slot_off, RadStream shared) {
    close();
    if (!reg || !kv || !tiers || groups.empty() || max_batch <= 0 || max_copy <= 0 ||
        max_chain <= 0)
        return RAD_E_INVAL;
    reg_ = reg; kv_ = kv; tiers_ = tiers; disk_ = disk; leader_ = leader; device_ = device;
    max_batch_ = max_batch; max_copy_ = max_copy; max_chain_ = max_chain;
    slot_off_ = slot_off;
    slot_bytes_    = tiers->entry_bytes();
    ck_bytes_      = kv->checkpoint_bytes();
    ck_slot_bytes_ = tiers->ckpt_bytes();
    ck_slot_off_   = ck_slot_off;
    shared_        = shared;
    RAD_TRY(rad_dev_set(device_));

    int64_t off = 0;
    for (int32_t g : groups) {
        const KVGroupPlan* p = kv->plan(g);
        if (!p || !p->paged()) continue;
        GroupInfo gi;
        gi.g = g;
        gi.n_layers = p->n_layers;
        gi.frag = p->layer_bytes_per_block();
        gi.layer_stride = p->layer_stride();
        gi.n_blocks = p->n_blocks;
        gi.base = p->base;
        gi.off = off;
        /* THE FRAGMENT WIDTH IS THE CONSTRAINT THE KERNEL IS PICKED ON. libr4d's byte-row pair
         * takes n as a multiple of 4; a geometry that does not satisfy it resolves to nothing
         * here rather than failing at launch inside a batch. */
        Geometry q;
        q.set_s("dtype", "u8");
        q.set_i("n", gi.frag);
        q.set_i("M", max_batch);
        gi.gather  = reg->resolve("gather_rows",  q, RAD_DOMAIN_DEVICE);
        gi.scatter = reg->resolve("scatter_rows", q, RAD_DOMAIN_DEVICE);
        if (!gi.gather || !gi.scatter) {
            RAD_WARN("kv tiers: no device gather/scatter for group %d at a %lld-byte fragment, "
                     "so a finished conversation cannot leave VRAM. The tiers are off.",
                     (int)g, (long long)gi.frag);
            close();
            return RAD_E_UNSUPPORTED;
        }
        off += gi.n_layers * gi.frag;
        gi_.push_back(gi);
    }
    if (gi_.empty()) return RAD_E_UNSUPPORTED;
    entry_bytes_ = off;

    /* NOTHING IS BUILT ON THE DEVICE HERE -- see device_init(). A tier that is configured but has
     * no job to run must cost the step nothing, and on this engine that is not a figure of speech:
     * building the staging buffer and the stream at bringup costs 13.3 ms of a 20.5 ms decode step
     * on a two-card R9700 Flash-Next server, whether or not a single byte ever moves. */
    verify_ = leader && std::getenv("RADIANCE_DEBUG_TIER_VERIFY") != nullptr;

    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "kv tier transfer: %zu paged group(s), %s an entry, batches of %d through "
                  "gather_rows/scatter_rows at u8%s%s",
                  gi_.size(), humanb(entry_bytes_).c_str(), max_batch_,
                  ck_bytes_ > 0 ? "; linear state as one contiguous copy of " : "",
                  ck_bytes_ > 0 ? humanb(ck_bytes_).c_str() : "");
    report_ = buf;
    enabled_ = true;
    return RAD_OK;
}

void TierExec::close() {
    /* Nothing may still be reading the buffers about to be freed. */
    if (stream_) (void)rad_stream_sync(stream_);
    if (stream_ && own_stream_) rad_stream_destroy(stream_);
    stream_ = nullptr; shared_ = nullptr; own_stream_ = false;
    if (stage_)    { rad_dev_free(stage_, RAD_MEM_DEVICE); stage_ = nullptr; }
    if (idx_dev_)  { rad_dev_free(idx_dev_, RAD_MEM_DEVICE); idx_dev_ = nullptr; }
    if (idx_host_) { rad_dev_free(idx_host_, RAD_MEM_HOST_PINNED); idx_host_ = nullptr; }
    for (int i = 0; i < kRing; ++i) {
        if (ring_ev_[i]) rad_event_destroy(ring_ev_[i]);
        ring_ev_[i] = nullptr;
        ring_used_[i] = false;
    }
    if (ev_compute_) { rad_event_destroy(ev_compute_); ev_compute_ = nullptr; }
    if (ev_copy_)    { rad_event_destroy(ev_copy_);    ev_copy_ = nullptr; }
    if (ev_restore_) { rad_event_destroy(ev_restore_); ev_restore_ = nullptr; }
    copy_out_ = false;
    gi_.clear();
    order_.clear();
    verify_pending_.clear();
    entry_bytes_ = 0;
    ck_bytes_ = ck_slot_bytes_ = ck_slot_off_ = 0;
    enabled_ = false;
    leader_ = false;
    reg_ = nullptr; kv_ = nullptr; tiers_ = nullptr; disk_ = nullptr;
    report_.clear();
}

/* THE DEVICE SIDE, BUILT ON FIRST USE AND NOT BEFORE.
 *
 * Ctx::lane_init() makes the same argument for the second compute lane: a stream is cheap
 * enough that its cost hides inside the run-to-run spread of a warming part, so the only honest
 * way to claim the lane costs nothing when unused is to not build it. The tiers follow the same
 * rule, and here the cost is not small enough to hide in anything. Decode step time with the
 * device side built at bringup, on a two-card R9700 Flash-Next server at 200K context:
 *
 *   no tiers                                             20.54 ms/step   collect 14.66
 *   tiers on, 512 MiB host arena, no disk                33.94 ms/step   collect 26.78
 *   tiers on, idle threshold a day -- NOTHING EVER MOVED 33.88 ms/step   collect 26.66
 *
 * In the third line not one byte crosses the link, and the step is still 65% longer. The cost is
 * in `collect` -- device time -- and a configured tier that has nothing to do pays it on every
 * step. It is not the transfers, not the pinned host arena (512 MiB costs exactly what 12 GiB
 * does) and not the VRAM the staging takes (with no tiers and --gpu-headroom-mib lowered to leave
 * the same 17 MiB free, the step runs at full speed).
 *
 * So the device side is built when the first job needs it, and a server whose conversations never
 * finish pays nothing for the tiers. */
int TierExec::device_init() {
    if (stream_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    const int64_t ng = (int64_t)gi_.size();
    copy_idx_  = max_copy_;
    chain_idx_ = max_chain_;
    const int64_t idx_n = ng * (copy_idx_ + (int64_t)kRing * chain_idx_);
    stage_    = rad_dev_alloc((int64_t)max_batch_ * entry_bytes_, RAD_MEM_DEVICE);
    idx_dev_  = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_DEVICE);
    idx_host_ = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED);
    bool ev = rad_event_create(&ev_compute_) == RAD_OK && rad_event_create(&ev_copy_) == RAD_OK &&
              rad_event_create(&ev_restore_) == RAD_OK;
    for (int i = 0; i < kRing && ev; ++i) ev = rad_event_create(&ring_ev_[i]) == RAD_OK;
    auto undo = [&] {
        if (stage_)    { rad_dev_free(stage_, RAD_MEM_DEVICE); stage_ = nullptr; }
        if (idx_dev_)  { rad_dev_free(idx_dev_, RAD_MEM_DEVICE); idx_dev_ = nullptr; }
        if (idx_host_) { rad_dev_free(idx_host_, RAD_MEM_HOST_PINNED); idx_host_ = nullptr; }
        for (int i = 0; i < kRing; ++i) { if (ring_ev_[i]) rad_event_destroy(ring_ev_[i]); ring_ev_[i] = nullptr; }
        if (ev_compute_) { rad_event_destroy(ev_compute_); ev_compute_ = nullptr; }
        if (ev_copy_)    { rad_event_destroy(ev_copy_);    ev_copy_ = nullptr; }
        if (ev_restore_) { rad_event_destroy(ev_restore_); ev_restore_ = nullptr; }
    };
    if (!stage_ || !idx_dev_ || !idx_host_ || !ev) { undo(); return RAD_E_NOMEM; }
    /* THE MOVER'S STREAM IF THERE IS ONE, AND A NEW ONE ONLY IF THERE IS NOT. Both move bytes
     * nobody is waiting for, so one maintenance stream a rank serves both -- and the fourth
     * hardware queue a second one would claim costs about 13 ms a decode step (see
     * device_init). Never the step's own stream: a copy out would then be a dependency of the
     * next token. */
    if (shared_) {
        stream_ = shared_;
        own_stream_ = false;
    } else if (rad_stream_create(&stream_, /*high_priority=*/0) == RAD_OK) {
        own_stream_ = true;
    } else {
        stream_ = nullptr;
        undo();
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

int64_t tier_entry_bytes(const KVManager& kv, const std::vector<int32_t>& groups) {
    int64_t n = 0;
    for (int32_t g : groups) {
        const KVGroupPlan* p = kv.plan(g);
        if (p && p->paged()) n += p->bytes_per_block;
    }
    return n;
}

/* One (group, layer) launch. `stage_base` is the first entry's staging slot; the rows this call
 * writes (or reads) are at a stride of entry_bytes_ from it, which is what puts each entry's
 * fragment inside its own contiguous staging entry. */
int TierExec::run_rows(const Resolved& r, bool scatter, const GroupInfo& gi, int64_t layer,
                       void* stage_base, const int32_t* d_idx, int64_t n_jobs) {
    RadTensor pool{};
    pool.dtype = RAD_U8;
    pool.rank = 2;
    pool.shape[0] = gi.n_blocks;  pool.shape[1] = gi.frag;
    pool.stride[0] = gi.frag;     pool.stride[1] = 1;
    pool.data = (char*)gi.base + layer * gi.layer_stride;

    RadTensor idx{};
    idx.dtype = RAD_I32; idx.rank = 1;
    idx.shape[0] = n_jobs; idx.stride[0] = 1;
    idx.data = (void*)d_idx;

    RadTensor pack{};
    pack.dtype = RAD_U8;
    pack.rank = 2;
    pack.shape[0] = n_jobs;      pack.shape[1] = gi.frag;
    pack.stride[0] = entry_bytes_;   /* <-- the whole trick: one entry's stride, not one row's */
    pack.stride[1] = 1;
    pack.data = (char*)stage_base + gi.off + layer * gi.frag;

    RadTensor t3[3];
    if (scatter) { t3[0] = pack; t3[1] = idx; t3[2] = pool; }
    else         { t3[0] = pool; t3[1] = idx; t3[2] = pack; }

    /* THE RESOLVED GEOMETRY, not a hand-built one. It carries whatever the winning kernel
     * supplied for a derived key as well as the query, and M in it is the band's upper bound --
     * which is correct and is the convention: both implementations take the row count from the
     * OPERAND extent, because M is ranged and the value in the args is the worst case. */
    RadArgs a{};
    a.t = t3; a.n_t = 3;
    a.p = r.geom.params(); a.n_p = r.geom.n_params();
    a.instance = r.instance;
    return r.row->info->launch(&a, stream_);
}

/* ------------------------------------------------------------------ RADIANCE_DEBUG_TIER_VERIFY */

static uint64_t tier_fnv(const void* p, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* b = (const unsigned char*)p;
    for (int64_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

void TierExec::verify_note(const BlockHash& key, const void* p, int64_t n, bool snapshot) {
    if (!verify_ || !leader_ || !p || n <= 0) return;
    (snapshot ? out_ck_ : out_sum_)[key] = tier_fnv(p, n);
}

void TierExec::verify_check(const BlockHash& key, const void* p, int64_t n, bool snapshot) {
    if (!verify_ || !leader_ || !p || n <= 0) return;
    auto& m = snapshot ? out_ck_ : out_sum_;
    auto it = m.find(key);
    if (it == m.end()) return;                      /* never went out through this process */
    const uint64_t back = tier_fnv(p, n);
    if (back == it->second) return;
    RAD_ERR("kv tiers: the %s for %s came back DIFFERENT (%016llx out, %016llx in). The tier is "
            "moving the wrong bytes, not the engine choosing differently downstream.",
            snapshot ? "linear-state snapshot" : "attention blocks", key.str().c_str(),
            (unsigned long long)it->second, (unsigned long long)back);
}

/* ------------------------------------------------------------------ helpers */

/* WHY THIS IS NOT RAD_TRY. A tier that fails is invisible: the entry stays where it was, the pool
 * keeps its blocks, and everything still works -- just slower, forever, with a counter nobody is
 * looking at. The first failure names the call it came from, because "5824 failures" and nothing
 * else is a day of bisecting. */
#define TIER_TRY(call, what)                                                                  \
    do { const int _s = (call);                                                               \
         if (_s != RAD_OK) { warn_once(what, _s); return _s; } } while (0)

void TierExec::warn_once(const char* what, int st) {
    if (warned_) return;
    warned_ = true;
    RAD_WARN("kv tiers: %s failed with %s. The entry stays where it was; this is said once, and "
             "the running failure count is in /sessions.", what, rad_strerror(st));
}

bool TierExec::blocks_backed(const std::vector<int32_t>& blocks, const char* what) {
    if (!kv_) return false;
    for (size_t k = 0; k < gi_.size(); ++k) {
        if (k >= blocks.size()) continue;          /* an absent group is the gather's -1 row */
        const int32_t b = blocks[k];
        if (b < 0) continue;
        const int64_t live = kv_->live_blocks(gi_[k].g);
        if (b < live) continue;
        if (warned_unbacked_) return false;
        warned_unbacked_ = true;
        RAD_ERR("kv tier %s: block %d of group %d is at or past this rank's live mark of %lld, so "
                "it is lent to the expert slab and holds weights. The transfer is refused; the "
                "entry stays where it is.", what, (int)b, (int)gi_[k].g, (long long)live);
        return false;
    }
    return true;
}

int TierExec::copy_runs(const std::vector<TierJob*>& order, size_t first, int64_t n, bool out) {
    int64_t k = 0;
    while (k < n) {
        const int32_t s0 = order[first + (size_t)k]->host_slot;
        int64_t run = 1;
        while (k + run < n && order[first + (size_t)(k + run)]->host_slot == s0 + run) ++run;
        char* host = (char*)tiers_->arena().at(s0);
        if (!host) return RAD_E_STATE;
        host += slot_off_;
        char* dev = (char*)stage_ + k * entry_bytes_;
        const int st = out
            ? rad_memcpy_2d_async(host, slot_bytes_, dev, entry_bytes_, entry_bytes_, run, stream_)
            : rad_memcpy_2d_async(dev, entry_bytes_, host, slot_bytes_, entry_bytes_, run, stream_);
        if (st != RAD_OK) return st;
        k += run;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ copy on idle */

int TierExec::issue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const int st = enqueue_copy(jobs, compute);
    /* A COPY THAT FAILED PART OF THE WAY IN IS DRAINED BEFORE ITS SLOTS GO BACK. The report frees
     * every host slot the batch was handed, and a transfer already queued into one would land on
     * whatever the arena gives that slot to next. */
    if (st != RAD_OK) (void)rad_stream_sync(stream_);
    return st;
}

int TierExec::enqueue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    order_.clear();
    for (TierJob& j : jobs) if (!j.blocks.empty()) order_.push_back(&j);
    const int64_t n = (int64_t)order_.size();
    if (n > copy_idx_) return RAD_E_INVAL;
    /* EVERY ID FIRST, BEFORE ANY OF THEM IS LAUNCHED AGAINST. A gather over an unbacked id faults
     * the whole card rather than failing the transfer. */
    for (TierJob* j : order_)
        if (!blocks_backed(j->blocks, "copy to host")) return RAD_E_STATE;

    /* AFTER EVERYTHING THE COMPUTE STREAM HAS BEEN GIVEN, which includes the step that wrote these
     * blocks and the save that wrote the snapshots. A stream wait, not a host wait. */
    TIER_TRY(rad_event_record(ev_compute_, compute), "ordering the copy after the step");
    TIER_TRY(rad_event_wait(stream_, ev_compute_), "ordering the copy after the step");

    if (n > 0) {
        const int64_t ng = (int64_t)gi_.size();
        int32_t* h = idx_host_;                           /* the copy region */
        for (int64_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < n; ++k) {
                const std::vector<int32_t>& b = order_[(size_t)k]->blocks;
                h[g * n + k] = (size_t)g < b.size() ? b[(size_t)g] : -1;
            }
        TIER_TRY(rad_memcpy_async(idx_dev_, h, ng * n * (int64_t)sizeof(int32_t), stream_),
                 "the block-index upload");
        for (int64_t s = 0; s < n; s += max_batch_) {
            const int64_t c = std::min<int64_t>(max_batch_, n - s);
            for (size_t gidx = 0; gidx < gi_.size(); ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers; ++l)
                    TIER_TRY(run_rows(gi_[gidx].gather, /*scatter=*/false, gi_[gidx], l, stage_,
                                      idx_dev_ + (int64_t)gidx * n + s, c),
                             "the gather out of the pool");
            TIER_TRY(copy_runs(order_, (size_t)s, c, /*out=*/true), "the copy to host memory");
        }
        st_.bytes_out += n * entry_bytes_;
    }

    /* The linear half: one contiguous copy each, out of the device slot the snapshot sits in. */
    for (TierJob& j : jobs) {
        if (j.ck_host < 0 || j.ck_dev < 0 || ck_bytes_ <= 0) continue;
        if (kv_->back_checkpoint(j.ck_dev) < 0) { j.ck_ok = false; continue; }
        const void* src = kv_->checkpoint_ptr(j.ck_dev);
        char* dst = (char*)tiers_->ckpt_arena().at(j.ck_host);
        if (!src || !dst) { j.ck_ok = false; continue; }
        const int st = rad_memcpy_async(dst + ck_slot_off_, src, ck_bytes_, stream_);
        if (st != RAD_OK) {
            warn_once("the snapshot copy to host memory", st);
            j.ck_ok = false;
            continue;
        }
        st_.bytes_out += ck_bytes_;
    }

    TIER_TRY(rad_event_record(ev_copy_, stream_), "marking the copy");
    copy_out_ = true;
    ++st_.passes;
    st_.jobs += (int64_t)jobs.size();
    if (verify_)
        for (const TierJob& j : jobs)
            verify_pending_.push_back({j.key, j.blocks.empty() ? -1 : j.host_slot,
                                       j.ck_ok ? j.ck_host : -1});
    return RAD_OK;
}

int TierExec::copy_state() {
    if (!copy_out_) return 1;
    (void)rad_dev_set(device_);
    const int q = rad_event_query(ev_copy_);
    if (q == 0) return 0;
    copy_out_ = false;
    if (q > 0 && verify_) {
        for (const VerifyNote& v : verify_pending_) {
            if (v.host_slot >= 0)
                verify_note(v.key, (char*)tiers_->arena().at(v.host_slot) + slot_off_,
                            entry_bytes_, false);
            if (v.ck_host >= 0)
                verify_note(v.key, (char*)tiers_->ckpt_arena().at(v.ck_host) + ck_slot_off_,
                            ck_bytes_, true);
        }
    }
    verify_pending_.clear();
    if (q < 0) warn_once("the copy to host memory", q);
    return q > 0 ? 1 : q;
}

/* ------------------------------------------------------------------ restore */

int TierExec::issue_restore(std::vector<TierJob>& jobs, RadStream compute,
                            std::vector<char>* ok) {
    ok->assign(jobs.size(), 0);
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const size_t ng = gi_.size();

    /* WHO CHOOSES THE BLOCKS, AND WHO KEEPS THE BOOKS. Ids are shared across ranks -- one block
     * table indexes every rank's shard -- so exactly one rank may choose, and that same rank is
     * the only one whose allocator means anything: the prefix cache holds the KV references and
     * the prefix cache is rank 0's. A follower writes into the ids it is handed and reserves
     * nothing.
     *
     * THE WHOLE CHAIN IS RESERVED BEFORE ANY OF IT MOVES, one recall of the loan for all of it.
     * Where the pool runs short the chain is cut at that point and the rest stay unstarted: a hit
     * needs a contiguous prefix, so an entry past a gap would be restored for nothing. */
    if (leader_) {
        const int64_t n = (int64_t)jobs.size();
        std::vector<std::vector<int32_t>> ids(ng);
        int64_t m = n;
        for (size_t k = 0; k < ng; ++k) {
            ids[k].reserve((size_t)n);
            m = std::min(m, kv_->reserve_blocks(gi_[k].g, m, &ids[k]));
        }
        for (size_t k = 0; k < ng; ++k)
            for (size_t i = (size_t)m; i < ids[k].size(); ++i) kv_->release_block(gi_[k].g, ids[k][i]);
        for (size_t i = 0; i < jobs.size(); ++i) {
            jobs[i].blocks.clear();
            if ((int64_t)i >= m) continue;
            jobs[i].blocks.resize(ng);
            for (size_t k = 0; k < ng; ++k) jobs[i].blocks[k] = ids[k][i];
        }
    }
    size_t m = 0;
    while (m < jobs.size() && jobs[m].blocks.size() == ng) ++m;
    /* AND WHETHER THIS RANK CAN ADDRESS WHAT IT WAS HANDED. The leader chose these ids against its
     * own live mark and the recall moved every follower's with it; this is the check that it did. */
    for (size_t i = 0; i < m; ++i)
        if (!blocks_backed(jobs[i].blocks, "restore")) { m = i; break; }

    /* THE CUT IS WHAT THE FOLLOWERS READ. They find their chain by which jobs carry block ids,
     * so every job past the point the leader stopped gives its ids back and carries none. A
     * follower handed ids for an entry the leader could not address would copy into blocks that
     * are not the cache's. */
    if (leader_) {
        for (size_t i = m; i < jobs.size(); ++i) {
            for (size_t k = 0; k < ng && k < jobs[i].blocks.size(); ++k)
                if (jobs[i].blocks[k] >= 0) kv_->release_block(gi_[k].g, jobs[i].blocks[k]);
            jobs[i].blocks.clear();
        }
    }
    if (m == 0) return RAD_OK;

    order_.clear();
    for (size_t i = 0; i < m; ++i) order_.push_back(&jobs[i]);
    for (size_t i = 0; i < m; ++i)
        verify_check(jobs[i].key, (char*)tiers_->arena().at(jobs[i].host_slot) + slot_off_,
                     entry_bytes_, false);

    /* FROM HERE ON A FAILURE LEAVES A MIXTURE, so every job about to be written is marked started
     * first -- report() drops a started job that failed rather than trusting its host copy to
     * describe blocks some layers of which were overwritten. And the compute stream is made to
     * wait on whatever was enqueued, success or not: a failed job's blocks go back to the pool,
     * and nothing may be written into them before the transfers already queued against them. */
    for (size_t i = 0; i < m; ++i) jobs[i].started = true;
    const int ri = ring_next_;
    ring_next_ = (ring_next_ + 1) % kRing;
    int32_t* h = idx_host_ + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int32_t* d = idx_dev_  + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int st = RAD_OK;
    size_t done = 0;
    for (size_t s = 0; s < m && st == RAD_OK; s += (size_t)chain_idx_) {
        const int64_t c = std::min<int64_t>(chain_idx_, (int64_t)(m - s));
        if (ring_used_[ri] && (st = rad_event_sync(ring_ev_[ri])) != RAD_OK) break;
        for (size_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < c; ++k) h[(int64_t)g * c + k] = jobs[s + (size_t)k].blocks[g];
        if ((st = rad_memcpy_async(d, h, (int64_t)ng * c * (int64_t)sizeof(int32_t), stream_)) != RAD_OK)
            break;
        if ((st = rad_event_record(ring_ev_[ri], stream_)) != RAD_OK) break;
        ring_used_[ri] = true;
        for (int64_t b = 0; b < c && st == RAD_OK; b += max_batch_) {
            const int64_t cc = std::min<int64_t>(max_batch_, c - b);
            if ((st = copy_runs(order_, s + (size_t)b, cc, /*out=*/false)) != RAD_OK) break;
            for (size_t gidx = 0; gidx < ng && st == RAD_OK; ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers && st == RAD_OK; ++l)
                    st = run_rows(gi_[gidx].scatter, /*scatter=*/true, gi_[gidx], l, stage_,
                                  d + (int64_t)gidx * c + b, cc);
        }
        if (st == RAD_OK) done = s + (size_t)c;
    }
    if (st != RAD_OK) warn_once("the restore into the pool", st);
    for (size_t i = 0; i < done; ++i) (*ok)[i] = 1;
    st_.bytes_in += (int64_t)done * entry_bytes_;

    /* THE LINEAR HALF, into a slot the leader allocates. The same rule as the blocks: the id is
     * rank 0's to choose, and a follower takes the id it is handed and backs its own pages. A
     * snapshot that cannot be placed is not a failure of the restore -- the session keeps its
     * attention hit and replays its linear layers, which is what it would have done with no
     * tiers at all.
     *
     * THE DEEPEST ONE IS WORTH A SLOT SOMEBODY ELSE HAS. A hit resumes from the last checkpoint at
     * or below its attention hit, which on a restored chain is the chain's last snapshot; the
     * earlier ones take a slot only where one is free. A full pool is then made room in for that
     * one snapshot by detaching another session's, which that session's own next restore brings
     * back -- rather than for every snapshot on the chain, which would push out as many other
     * sessions' states as this one has checkpoints. */
    size_t deepest = done;
    for (size_t i = 0; i < done; ++i)
        if (jobs[i].ck_want && jobs[i].ck_ok && jobs[i].ck_host >= 0) deepest = i;
    for (size_t i = 0; i < done; ++i) {
        TierJob& j = jobs[i];
        if (!j.ck_want || !j.ck_ok || j.ck_host < 0 || ck_bytes_ <= 0) continue;
        if (leader_) {
            int32_t slot = -1;
            if (kv_->alloc_checkpoint(&slot) != RAD_OK || slot < 0) {
                slot = -1;
                if (i != deepest || !ck_room_ || !ck_room_() ||
                    kv_->alloc_checkpoint(&slot) != RAD_OK || slot < 0)
                    continue;
            }
            j.ck_dev = slot;
        }
        if (j.ck_dev < 0) continue;
        if (kv_->back_checkpoint(j.ck_dev) < 0) { j.ck_ok = false; continue; }
        const char* src = (const char*)tiers_->ckpt_arena().at(j.ck_host);
        void* dst = kv_->checkpoint_ptr(j.ck_dev);
        if (!src || !dst) { j.ck_ok = false; continue; }
        verify_check(j.key, src + ck_slot_off_, ck_bytes_, true);
        const int cs = rad_memcpy_async(dst, src + ck_slot_off_, ck_bytes_, stream_);
        if (cs != RAD_OK) { warn_once("the snapshot copy back into the pool", cs); j.ck_ok = false; continue; }
        st_.bytes_in += ck_bytes_;
    }

    /* THE STEP THAT READS THESE WAITS FOR THEM, on the device. If even the marker cannot be
     * enqueued the only ordering left is the host's. */
    if (rad_event_record(ev_restore_, stream_) != RAD_OK ||
        rad_event_wait(compute, ev_restore_) != RAD_OK) {
        (void)rad_stream_sync(stream_);
    }
    ++st_.passes;
    st_.jobs += (int64_t)m;
    return st;
}

int TierExec::fence(RadEvent* out) {
    *out = nullptr;
    if (!enabled_ || !stream_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    RadEvent e = nullptr;
    RAD_TRY(rad_event_create(&e));
    const int st = rad_event_record(e, stream_);
    if (st != RAD_OK) { rad_event_destroy(e); return st; }
    *out = e;
    return RAD_OK;
}

/* ------------------------------------------------------------------ the disk store */

void TierExec::run_disk(std::vector<TierJob>& jobs, std::vector<char>* ok) {
    ok->assign(jobs.size(), 1);
    for (size_t i = 0; i < jobs.size(); ++i) (*ok)[i] = run_write(jobs[i]) ? 1 : 0;
}

bool TierExec::run_write(TierJob& j) {
    if (j.kind != TierJob::Kind::ToDisk) return true;   /* the drops move nothing */
    bool ok = true;
    /* THE WHOLE SLOT, every rank's shard of it, because a disk entry read back must be able to
     * serve all of them. A job may carry only the snapshot half, when the blocks are already on
     * disk or their write went out earlier. */
    if (j.host_slot >= 0) {
        const void* src = tiers_->arena().at(j.host_slot);
        ok = disk_ && src && disk_->put_block(j.key, nullptr, 0, src, slot_bytes_) == RAD_OK;
    }
    if (j.ck_host >= 0) {
        const void* cs = tiers_->ckpt_arena().at(j.ck_host);
        /* A snapshot store that was never opened -- no room for one inside the disk cap -- refuses
         * this, which is the right answer and not an error: the host copy stays the only one, as
         * it was. */
        j.ck_ok = disk_ && cs && disk_->put_checkpoint(j.key, j.ck_pos, cs, ck_slot_bytes_) == RAD_OK;
    }
    return ok;
}

void TierExec::run_fetch(std::vector<TierJob>& jobs, std::vector<char>* ok) {
    ok->assign(jobs.size(), 1);
    std::vector<BlockHash> keys;
    std::vector<void*>     dst;
    std::vector<size_t>    at;
    for (size_t i = 0; i < jobs.size(); ++i) {
        TierJob& j = jobs[i];
        j.started = true;
        if (j.host_slot < 0) continue;              /* only the snapshot half */
        keys.push_back(j.key);
        dst.push_back(tiers_->arena().at(j.host_slot));
        at.push_back(i);
    }
    if (!keys.empty()) {
        std::vector<char> got(keys.size(), 0);
        if (disk_) (void)disk_->get_blocks(keys.data(), (int64_t)keys.size(), dst.data(),
                                           slot_bytes_, got.data());
        for (size_t k = 0; k < keys.size(); ++k) (*ok)[at[k]] = got[k];
        st_.bytes_in += (int64_t)keys.size() * slot_bytes_;
    }
    for (TierJob& j : jobs) {
        if (j.ck_host < 0) continue;
        void* cs = tiers_->ckpt_arena().at(j.ck_host);
        j.ck_ok = disk_ && cs && disk_->get_checkpoint(j.key, j.ck_pos, cs, ck_slot_bytes_) == RAD_OK;
    }
}

/* ------------------------------------------------------------------ the report */

void TierExec::report_pass(std::vector<TierJob>& jobs, const std::vector<char>& ok,
                           uint64_t now_ns) {
    if (!enabled_ || !leader_) return;
    for (size_t i = 0; i < jobs.size(); ++i) {
        TierJob& j = jobs[i];
        const bool good = i < ok.size() && ok[i];
        /* A SNAPSHOT IS RESTORED ONLY IF EVERY RANK RESTORED ITS SHARD, and the leader is the only
         * one that can act on that. It allocates the checkpoint slot and fills its own window
         * first, so a follower that then fails leaves the slot holding this rank's new state beside
         * another rank's stale state -- a checkpoint the cache would go on to serve, one card
         * resuming the conversation and the other resuming whatever it last held.
         *
         * A PROMOTION THAT FAILED AS A WHOLE gives back everything the leader took for it: report()
         * drops or keeps the record, and nothing will ever name these blocks or this slot again. */
        if (j.kind == TierJob::Kind::ToDevice && j.ck_dev >= 0 && (!j.ck_ok || !good)) {
            kv_->free_checkpoint(j.ck_dev);
            j.ck_dev = -1;
        }
        if (j.kind == TierJob::Kind::ToDevice && !good) {
            for (size_t k = 0; k < gi_.size() && k < j.blocks.size(); ++k)
                if (j.blocks[k] >= 0) kv_->release_block(gi_[k].g, j.blocks[k]);
            j.blocks.clear();
        }
        tiers_->report(j, good, now_ns);
    }
    std::vector<int32_t> frees;
    tiers_->take_checkpoint_frees(&frees);
    for (int32_t s : frees) kv_->free_checkpoint(s);
}

bool TierExec::plan_promote(const BlockHash& key, TierJob* out) {
    if (!enabled_ || !leader_) return false;
    return tiers_->plan_promote(key, out);
}

/* ------------------------------------------------------------------ TierIO */

TierIO::~TierIO() { stop(); }

void TierIO::start(TierExec* leader) {
    stop();
    x_ = leader;
    stop_ = false;
    th_ = std::thread([this] { loop(); });
}

void TierIO::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    has_work_ = has_done_ = false;
    work_.clear();
    done_.clear();
    ok_.clear();
    work_at_ = 0;
    fetch_q_.clear();
    fetch_done_.clear();
}

bool TierIO::submit(std::vector<TierJob>&& jobs) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (has_work_ || has_done_ || !th_.joinable()) return false;
        work_ = std::move(jobs);
        ok_.assign(work_.size(), 1);
        work_at_ = 0;
        has_work_ = true;
    }
    cv_.notify_all();
    return true;
}

bool TierIO::submit_fetch(Fetch&& f) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!th_.joinable()) return false;
        fetch_q_.push_back(std::move(f));
    }
    cv_.notify_all();
    return true;
}

bool TierIO::take_fetch(Fetch* out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (fetch_done_.empty()) return false;
    *out = std::move(fetch_done_.front());
    fetch_done_.pop_front();
    return true;
}

bool TierIO::take(std::vector<TierJob>* jobs, std::vector<char>* ok) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!has_done_) return false;
    *jobs = std::move(done_);
    *ok = std::move(ok_);
    done_.clear();
    ok_.clear();
    has_done_ = false;
    return true;
}

bool TierIO::busy() const {
    std::lock_guard<std::mutex> lk(mu_);
    return has_work_ || has_done_;
}

void TierIO::loop() {
    for (;;) {
        Fetch f;
        bool read = false;
        size_t i = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !fetch_q_.empty() || (has_work_ && !has_done_); });
            if (stop_) return;
            if (!fetch_q_.empty()) {
                f = std::move(fetch_q_.front());
                fetch_q_.pop_front();
                read = true;
            } else {
                i = work_at_;
            }
        }
        if (read) {
            /* NOT A BYTE UNTIL EVERY RANK'S RESTORES THAT COULD READ THESE SLOTS HAVE RUN. A fence
             * that cannot be waited on leaves the read unstarted, and the records stay on disk. */
            bool fenced = true;
            for (RadEvent e : f.fences) if (e && rad_event_sync(e) != RAD_OK) fenced = false;
            if (fenced) x_->run_fetch(f.jobs, &f.ok);
            else f.ok.assign(f.jobs.size(), 0);
            std::lock_guard<std::mutex> lk(mu_);
            fetch_done_.push_back(std::move(f));
            continue;
        }
        /* One write at a time, so a read that arrives behind a long batch waits for one job. */
        const bool ok = i < work_.size() ? x_->run_write(work_[i]) : true;
        std::lock_guard<std::mutex> lk(mu_);
        if (i < ok_.size()) ok_[i] = ok ? 1 : 0;
        work_at_ = i + 1;
        if (work_at_ >= work_.size()) {
            done_ = std::move(work_);
            work_.clear();
            has_work_ = false;
            has_done_ = true;
        }
    }
}
}  /* namespace rad */
