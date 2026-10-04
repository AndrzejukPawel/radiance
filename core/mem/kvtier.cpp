#include "mem/kvtier.h"

#include "mem/sstier.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace rad {

const char* kv_tier_name(KVTier t) {
    switch (t) {
        case KVTier::Device: return "device";
        case KVTier::Host:   return "host";
        case KVTier::Disk:   return "disk";
    }
    return "?";
}

/* ------------------------------------------------------------------ HostArena */

HostArena::~HostArena() { close(); }

int HostArena::open(int64_t slot_bytes, int64_t cap_bytes) {
    close();
    if (slot_bytes <= 0 || cap_bytes < slot_bytes) return RAD_E_INVAL;
    const int64_t n = cap_bytes / slot_bytes;
    /* PINNED, and the allocation is refused rather than downgraded. A pageable host buffer on
     * this path does not fail, it just collapses -- an async copy to pageable memory serialises
     * against the stream and the whole point of moving instead of dropping is gone. Better to
     * start without the tier and say so than to run it at a tenth of the rate it was budgeted at. */
    base_ = rad_dev_alloc(n * slot_bytes, RAD_MEM_HOST_PINNED);
    if (!base_) return RAD_E_NOMEM;
    slot_bytes_ = slot_bytes;
    used_ = 0;
    cap_slots_ = n;
    cursor_ = 0;
    bits_.assign((size_t)((n + 63) / 64), 0ull);
    return RAD_OK;
}

void HostArena::close() {
    if (base_) rad_dev_free(base_, RAD_MEM_HOST_PINNED);
    base_ = nullptr;
    slot_bytes_ = 0;
    used_ = 0;
    cap_slots_ = 0;
    cursor_ = 0;
    bits_.clear();
}

int32_t HostArena::alloc() {
    int32_t got = 0;
    return alloc_run(1, &got);
}

/* THE FIRST FREE RUN AT OR AFTER THE CURSOR, wrapping once. A whole word of used slots is skipped
 * in one compare, so a nearly full arena costs a scan of its bitmap -- a few thousand words at the
 * sizes an operator gives this -- and an arena with room costs nothing. The cursor makes
 * consecutive calls hand out consecutive slots, which is what lets a batch be one copy a run. */
int32_t HostArena::alloc_run(int32_t want, int32_t* got) {
    *got = 0;
    if (want <= 0 || !base_ || full()) return -1;
    int64_t s = -1;
    for (int pass = 0; pass < 2 && s < 0; ++pass) {
        const int64_t lo = pass == 0 ? cursor_ : 0;
        const int64_t hi = pass == 0 ? cap_slots_ : cursor_;
        for (int64_t i = lo; i < hi;) {
            const uint64_t w = bits_[(size_t)(i >> 6)];
            if ((i & 63) == 0 && w == ~0ull) { i += 64; continue; }
            if (!is_used(i)) { s = i; break; }
            ++i;
        }
    }
    if (s < 0) return -1;
    int64_t n = 0;
    while (n < want && s + n < cap_slots_ && !is_used(s + n)) {
        bits_[(size_t)((s + n) >> 6)] |= 1ull << ((s + n) & 63);
        ++n;
    }
    used_ += n;
    cursor_ = s + n >= cap_slots_ ? 0 : s + n;
    *got = (int32_t)n;
    return (int32_t)s;
}

void HostArena::free(int32_t slot) {
    if (slot < 0 || slot >= cap_slots_) return;
    /* A slot that is already free is refused rather than counted twice: the used count is what the
     * sessions view reports and what `full` answers, and a negative one hides the next leak. */
    if (!is_used(slot)) return;
    bits_[(size_t)(slot >> 6)] &= ~(1ull << (slot & 63));
    --used_;
}

void* HostArena::at(int32_t slot) const {
    if (!base_ || slot < 0 || slot >= cap_slots_) return nullptr;
    return (char*)base_ + (int64_t)slot * slot_bytes_;
}

/* ------------------------------------------------------------------ IdleTiers */

int IdleTiers::configure(const Config& c, SSDTier* disk) {
    std::lock_guard lk(mu_);
    reset();
    cfg_ = c;
    disk_ = disk;

    /* THE SNAPSHOT STORE IS CARVED OUT OF THE TIER, NOT ADDED TO IT. engine_bringup decides how
     * much; what matters here is that the block arena is opened against what is left, so the two
     * together never exceed what --prefix-cache-host-mib asked for. */
    if (cfg_.ckpt_bytes > 0 && cfg_.ckpt_cap_bytes >= cfg_.ckpt_bytes && cfg_.host_cap_bytes > 0) {
        if (ck_arena_.open(cfg_.ckpt_bytes, cfg_.ckpt_cap_bytes) != RAD_OK) {
            RAD_WARN("kv tiers: no pinned host memory for %s of linear-state snapshots; a hybrid "
                     "session's blocks will tier and its recurrent state will not",
                     humanb(cfg_.ckpt_cap_bytes).c_str());
            cfg_.ckpt_cap_bytes = 0;
        } else {
            cfg_.host_cap_bytes -= ck_arena_.cap_bytes();
        }
    } else {
        cfg_.ckpt_cap_bytes = 0;
    }

    if (cfg_.host_cap_bytes > 0 && cfg_.entry_bytes > 0) {
        const int st = arena_.open(cfg_.entry_bytes, cfg_.host_cap_bytes);
        if (st != RAD_OK) {
            RAD_WARN("kv tiers: no pinned host memory for %s, the host tier is off",
                     humanb(cfg_.host_cap_bytes).c_str());
            cfg_.host_cap_bytes = 0;
            /* The snapshot store rides on the host tier: nothing reaches disk except through it,
             * so a snapshot arena under no block arena would fill once and never drain. */
            ck_arena_.close();
            cfg_.ckpt_cap_bytes = 0;
        }
    } else {
        cfg_.host_cap_bytes = 0;
        ck_arena_.close();
        cfg_.ckpt_cap_bytes = 0;
    }
    if (!disk_ || !disk_->enabled()) cfg_.disk_cap_bytes = 0;

    /* A disk tier under no host tier is legal and it is not a mistake: the path to disk runs
     * through host memory, so with no host tier nothing ever reaches disk, and saying so at
     * startup is cheaper than an operator reading a disk tier's zero counters for an afternoon. */
    if (!host_on() && disk_on())
        RAD_WARN("kv tiers: --prefix-cache-disk-mib is set but --prefix-cache-host-mib is not, and the "
                 "path to disk runs through host memory -- nothing will be written");

    char buf[512];
    if (!any_on()) {
        report_ = "kv tiers: off (no --prefix-cache-host-mib), so a finished conversation's KV stays "
                  "in VRAM until the cache evicts it under pressure -- it is never copied to host "
                  "memory, and what the pool is holding is mostly cache rather than live requests";
    } else {
        std::snprintf(buf, sizeof buf,
                      "kv tiers: a finished conversation's KV is copied to host memory at once "
                      "(%s, %lld entries) and its VRAM is given up when the expert loan or a "
                      "request wants it; %s. One entry is %s",
                      humanb(arena_.cap_bytes()).c_str(), (long long)arena_.capacity(),
                      disk_on() ? ("every host copy is written on to disk (" +
                                   humanb(cfg_.disk_cap_bytes) + ") and its slot is given up "
                                   "when another entry needs it").c_str()
                                : "no disk tier, so a full host tier drops its oldest entries",
                      humanb(cfg_.entry_bytes).c_str());
        report_ = buf;
        /* SAID EVEN WHEN IT IS ZERO, on a model that has recurrent state. A hybrid server whose
         * snapshots do not tier looks identical from every block counter there is, and the
         * difference it makes is a full linear replay of the transcript on what the hit rate will
         * happily report as a hit. */
        if (cfg_.ckpt_bytes > 0) {
            std::snprintf(buf, sizeof buf,
                          "\nkv tiers: a session's linear state moves with its blocks -- %lld "
                          "snapshot(s) of %s in host memory, %lld on disk",
                          (long long)ck_arena_.capacity(), humanb(cfg_.ckpt_bytes).c_str(),
                          (long long)cfg_.ckpt_disk_slots);
            report_ += buf;
            if (!ckpt_on())
                report_ += " -- NONE, so an idle hybrid session comes back needing a full "
                           "linear replay; give --prefix-cache-host-mib more room";
        }
    }
    return RAD_OK;
}

void IdleTiers::close() {
    std::lock_guard lk(mu_);
    reset();
}

/* The body of close(), for the caller that already holds the lock. */
void IdleTiers::reset() {
    recs_.clear();
    sessions_.clear();
    roots_.clear();
    host_lru_.clear();
    disk_lru_.clear();
    wb_q_.clear();
    dk_q_.clear();
    clean_q_.clear();
    ck_frees_.clear();
    starved_ = 0;
    wb_stuck_ = false;
    arena_.close();
    ck_arena_.close();
    disk_ = nullptr;
    cfg_ = Config{};
    report_.clear();
}

std::string IdleTiers::report() const { return report_; }

void IdleTiers::unlist(Rec& r) {
    if (!r.listed) return;
    list_for(r.tier).erase(r.it);
    r.listed = false;
}

/* ONLY A RECORD WHOSE BYTES ARE ONLY OFF THE CARD. A device record -- clean or not -- is ordered by
 * the prefix cache's own LRU, and a second ordering of the same entries is the two-owner shape the
 * header's one-record rule exists to rule out. */
void IdleTiers::relist(const BlockHash& key, Rec& r) {
    if (r.pending || r.tier == KVTier::Device || r.listed) return;
    std::list<BlockHash>& l = list_for(r.tier);
    l.push_back(key);
    r.it = std::prev(l.end());
    r.listed = true;
}

bool IdleTiers::wants_copy(const Rec& r) const {
    if (r.pending || r.doomed) return false;
    const bool blocks = host_on() && r.tier == KVTier::Device && r.host_slot < 0 && !r.on_disk &&
                        !r.blocks.empty();
    return blocks || wants_ck_copy(r);
}

bool IdleTiers::wants_ck_copy(const Rec& r) const {
    /* A snapshot on a disk record stays where it is: the record's blocks are already past the
     * host tier, and the device slot it still holds is reachable the moment they come back. */
    return ckpt_on() && r.ck_dev >= 0 && r.ck_host < 0 && !r.ck_on_disk && !r.ck_skip &&
           r.tier != KVTier::Disk;
}

bool IdleTiers::wants_disk(const Rec& r) const {
    if (!disk_on() || r.doomed) return false;
    const bool blocks = r.host_slot >= 0 && !r.on_disk && r.disk_src < 0;
    const bool snap = cfg_.ckpt_disk_slots > 0 && r.ck_host >= 0 && !r.ck_on_disk &&
                      r.ck_disk_src < 0;
    return blocks || snap;
}

void IdleTiers::queue_copy(const BlockHash& key, Rec& r) {
    if (r.queued || !wants_copy(r)) return;
    wb_q_.push_back(key);
    r.queued = true;
}

void IdleTiers::queue_disk(const BlockHash& key, Rec& r) {
    if (r.disk_q || !wants_disk(r)) return;
    dk_q_.push_back(key);
    r.disk_q = true;
}

void IdleTiers::give_slot(int32_t slot) {
    if (slot < 0) return;
    arena_.free(slot);
    wb_stuck_ = false;
}

void IdleTiers::free_host(Rec& r) {
    if (r.host_slot < 0) return;
    if (r.host_slot != r.disk_src) give_slot(r.host_slot);
    r.host_slot = -1;
}

void IdleTiers::free_ck_host(Rec& r) {
    if (r.ck_host < 0) return;
    if (r.ck_host != r.ck_disk_src) ck_arena_.free(r.ck_host);
    r.ck_host = -1;
}

/* THE ONE PLACE `on_disk` CHANGES, because the disk figure in the occupancy view is kept beside it
 * rather than counted, and an entry landing on disk is also what unblocks a stuck write-back. */
void IdleTiers::set_on_disk(Rec& r, bool v) {
    if (r.on_disk == v) return;
    r.on_disk = v;
    stats_.disk_bytes += v ? cfg_.entry_bytes : -cfg_.entry_bytes;
    if (v) wb_stuck_ = false;
}

void IdleTiers::queue_clean(const BlockHash& key, Rec& r) {
    if (r.clean_q) return;
    clean_q_.push_back(key);
    r.clean_q = true;
}

/* EVERYTHING THIS RECORD HOLDS OFF THE CARD, both halves. The blocks' arena slot and the
 * snapshot's are different arenas with different slot sizes; SSDTier owns its own slots either way
 * and reclaims them by LRU, so there is nothing to give back for a disk-resident record.
 *
 * A snapshot still in its device slot is the CACHE'S, not this object's: the cache retires device
 * snapshots by its own retention policy and the slot is still correct and indexed. Only a snapshot
 * that exists nowhere but here is lost with the record, and only that loss is announced -- the
 * cache learns about a lost ENTRY from set_drop_sink at the sites that erase one, and this is also
 * called from forget(), where the cache is the caller and already knows. */
void IdleTiers::release_storage(const BlockHash& key, Rec& r) {
    free_host(r);
    set_on_disk(r, false);
    const bool lost_ck = r.ck_dev < 0 && (r.ck_host >= 0 || r.ck_on_disk);
    free_ck_host(r);
    r.ck_on_disk = false;
    if (lost_ck) { ++stats_.ck_drops; if (on_ck_drop_) on_ck_drop_(key); }
}

/* ------------------------------------------------------------------ the linear half */

void IdleTiers::note_snapshot(const BlockHash& key, int32_t dev_slot, int64_t pos) {
    if (dev_slot < 0) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    /* A FRESH DEVICE SNAPSHOT REPLACES WHATEVER COPY THERE WAS, never joins it. The host copy
     * described the state as it was last written; the slot now holds it as it was just written,
     * and a record that called the two the same bytes would restore the older one. The copy goes
     * and the snapshot is queued for a new one.
     *
     * Never while a copy is outstanding, for the same reason: report() owns the slot the job was
     * handed and frees it when it sees the record's device slot has changed under it. */
    if (!r.pending) free_ck_host(r);
    r.ck_on_disk = false;
    r.ck_dev = dev_slot;
    r.ck_pos = pos;
    r.ck_skip = false;
    /* COPIED ONCE ITS RECORD IS PUBLISHED, like the blocks: note_used queues it then. A request
     * commits its checkpoints as its prefill passes them, so copying each at commit would send the
     * shallow ones across first, one at a time, while the tip does not exist yet -- and the tip is
     * the one the next turn resumes from. Published together, they are planned together, and the
     * plan takes the deepest first. A record the request had already published takes a new
     * commit at once. */
    if (r.tier != KVTier::Device || !r.blocks.empty()) queue_copy(key, r);
}

bool IdleTiers::release_snapshot(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    Rec& r = it->second;
    /* A COPY IS READING THE DEVICE SLOT. The record forgets the snapshot now -- the cache no longer
     * has it -- but the slot itself is held until the copy reports, because freeing it here lets
     * the pool decommit a page an asynchronous transfer is still reading. */
    if (r.ck_reading && r.ck_dev >= 0) {
        r.ck_free_after = r.ck_dev;
        r.ck_dev = -1;
        free_ck_host(r);
        r.ck_on_disk = false;
        return true;
    }
    /* The host copy is the record's to free unless a job is carrying it: a pending host or disk
     * record is being restored out of that very slot, and report() gives it back. A pending device
     * record is a copy of its blocks, whose snapshot half -- if it has one -- is the record's own. */
    if (!r.pending || r.tier == KVTier::Device) free_ck_host(r);
    r.ck_on_disk = false;
    r.ck_dev = -1;
    /* A RECORD THAT ONLY EVER EXISTED FOR THE SNAPSHOT GOES WITH IT. note_snapshot creates one at
     * commit time, which is mid-request and long before the chain is published; a request that
     * never publishes -- a rebased block table, a cancel before the first whole block -- would
     * otherwise leave a record with no blocks, no storage and no way to be reached again. */
    if (!r.pending && r.tier == KVTier::Device && r.blocks.empty() && r.host_slot < 0)
        recs_.erase(it);
    return false;
}

void IdleTiers::take_checkpoint_frees(std::vector<int32_t>* out) {
    std::lock_guard lk(mu_);
    out->insert(out->end(), ck_frees_.begin(), ck_frees_.end());
    ck_frees_.clear();
}

bool IdleTiers::snapshot_reading(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it != recs_.end() && it->second.ck_reading;
}

int32_t IdleTiers::evict_snapshot_slot(bool lossless) {
    /* EVERY RECORD HOLDING A SNAPSHOT'S HOST COPY IS A CANDIDATE, on the lists or not. A session
     * still in VRAM holds its snapshots' host copies from the moment they land, and a search that
     * looked only at the off-device lists would not see them: the first snapshot to land would keep
     * the slot and every later one -- the running session's own tip among them -- would be refused
     * for good. The arena is a few slots deep, so the walk is over the records, once, and only
     * when a copy has found the arena full.
     *
     * THREE PRICES, cheapest first. A copy the disk store also holds costs nothing. A copy of a
     * snapshot still in its device slot costs that slot its backing: the checkpoint pool can no
     * longer detach it, only retire it, and a restore that needs a device slot can no longer take
     * it. A copy that is the snapshot's only one costs the state. A restore takes only the first
     * kind -- anything else trades one session's state for another's.
     *
     * THE TIP IS THE ONE WORTH KEEPING, so within a price the earliest POSITION loses: within a
     * session that is exactly "not the tip", which is the one position an appending conversation
     * will ask for and the one that makes its replay short. Across sessions it is a choice between
     * snapshots that are equally not tips, and the older one goes.
     *
     * The slot is returned to the caller rather than freed, because the caller is about to
     * allocate one: going through the free list would be the same slot and two more operations. */
    Rec* best = nullptr;
    const BlockHash* best_key = nullptr;
    int best_price = 3;
    const uint64_t now = rad_mono_ns();
    for (auto& kv : recs_) {
        Rec& r = kv.second;
        if (r.pending || r.doomed || r.ck_host < 0 || r.ck_host == r.ck_disk_src ||
            held_by_fetch(r, now))
            continue;
        const int price = r.ck_on_disk ? 0 : r.ck_dev >= 0 ? 1 : 2;
        if (lossless && price > 0) continue;
        if (best && (price > best_price ||
                     (price == best_price &&
                      (r.ck_pos > best->ck_pos ||
                       (r.ck_pos == best->ck_pos && r.used_ns >= best->used_ns)))))
            continue;
        best = &r;
        best_key = &kv.first;
        best_price = price;
    }
    if (!best) return -1;
    Rec& r = *best;
    const int32_t s = r.ck_host;
    r.ck_host = -1;
    if (best_price == 2) {
        ++stats_.ck_drops;
        if (on_ck_drop_) on_ck_drop_(*best_key);
    } else if (best_price == 1) {
        /* Still in its device slot and nowhere else off the card: its index entry is as good as
         * it was, and it is not queued for another copy (Rec::ck_skip). */
        r.ck_skip = true;
    }
    return s;
}

/* ------------------------------------------------------------------ the cache's notices */

void IdleTiers::note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns) {
    if (!any_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    r.used_ns = now_ns;
    if (r.tier == KVTier::Device && !r.pending) r.blocks = blocks;
    /* A record only moves to the back of its list when it is IN one. A device record is a
     * timestamp and nothing else, which is what keeps this cheap enough to sit on the publish
     * path -- the hash insert is the whole cost. */
    if (r.listed) {
        list_for(r.tier).erase(r.it);
        r.listed = false;
        relist(key, r);
    }
    queue_copy(key, r);
}

void IdleTiers::note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns) {
    if (!any_on() || !keys || n <= 0) return;
    std::lock_guard lk(mu_);
    for (int64_t i = 0; i < n; ++i) {
        auto it = recs_.find(keys[i]);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        r.used_ns = now_ns;
        if (r.listed) {
            list_for(r.tier).erase(r.it);
            r.listed = false;
            relist(keys[i], r);
        }
    }
}

void IdleTiers::rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns) {
    if (!any_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    if (r.pending) return;
    unlist(r);
    release_storage(key, r);
    r.tier = KVTier::Device;
    r.blocks = blocks;
    r.used_ns = now_ns;
    queue_copy(key, r);
}

bool IdleTiers::doomed(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() || it->second.doomed;
}

IdleTiers::Stats IdleTiers::stats() const {
    std::lock_guard lk(mu_);
    Stats s = stats_;
    /* What the block arena holds, clean copies included: counted off the arena rather than kept
     * beside it, because every path that takes or returns a slot would otherwise have to remember
     * to move a second number. */
    s.host_bytes = arena_.used_slots() * cfg_.entry_bytes;
    return s;
}

void IdleTiers::forget(const BlockHash& key) {
    std::lock_guard lk(mu_);
    /* AND EVERY CONVERSATION THAT STARTED HERE. This is the only notice the cache gives that an
     * entry is gone for good, and a chain whose first block has gone cannot be hit at any length
     * -- a prefix hit starts at position zero. Leaving the row behind would report blocks the
     * pool has already handed to somebody else as this session's, and would hold its whole chain
     * in memory for a conversation that can never come back. */
    auto ro = roots_.find(key);
    if (ro != roots_.end()) {
        const std::vector<uint64_t> ids = ro->second;   /* drop_session rewrites the list */
        for (uint64_t id : ids) drop_session(id);
    }
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.pending) {
        /* The engine is mid-transfer against this record. Erasing it now is exactly the
         * use-after-free the one-record rule exists to prevent, so mark it and let report()
         * finish the job and clean up. */
        r.doomed = true;
        return;
    }
    unlist(r);
    release_storage(key, r);
    recs_.erase(it);
}

KVTier IdleTiers::tier_of(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() ? KVTier::Device : it->second.tier;
}

bool IdleTiers::resident_off_device(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    return !it->second.pending && it->second.tier != KVTier::Device;
}

/* ------------------------------------------------------------------ copy on idle */

bool IdleTiers::writeback_waiting() const {
    std::lock_guard lk(mu_);
    /* NOT WHILE NOTHING CAN MOVE. An arena full of copies the disk does not hold yet has nothing
     * to give up, so a plan would pop the queue, find no slot and put it back, every tick, until
     * a slot is freed or a disk copy lands -- which is what ends the wait. */
    if (wb_stuck_ && arena_.full()) return false;
    return !wb_q_.empty() && (host_on() || ckpt_on());
}

bool IdleTiers::disk_waiting() const {
    std::lock_guard lk(mu_);
    return !dk_q_.empty();
}

int64_t IdleTiers::make_room(int64_t n) {
    std::lock_guard lk(mu_);
    return make_room_locked(n);
}

/* A READ NO REQUEST CLAIMS IS HELD THIS LONG. A request is admitted on the step after its read lands,
 * which is milliseconds; this only bounds what one that went away -- cancelled while it waited --
 * keeps out of reach of everybody else. */
static constexpr uint64_t kFetchHoldNs = 10ull * 1000000000ull;

bool IdleTiers::held_by_fetch(const Rec& r, uint64_t now_ns) const {
    return r.keep || (r.fetched_ns != 0 && now_ns - r.fetched_ns < kFetchHoldNs);
}

int64_t IdleTiers::make_room_locked(int64_t n) {
    if (!disk_on() || n <= 0) return 0;
    const uint64_t now = rad_mono_ns();
    int64_t freed = 0;
    /* HOST RECORDS FIRST, oldest first: nothing is in VRAM to use them, and they become disk
     * records. The walk skips the ones whose disk copy has not landed, which are the newest --
     * the writes go out in the order the copies landed -- so it stops close to the front. */
    for (auto it = host_lru_.begin(); it != host_lru_.end() && freed < n;) {
        const BlockHash key = *it;
        ++it;
        auto ri = recs_.find(key);
        if (ri == recs_.end()) continue;
        Rec& r = ri->second;
        if (r.pending || !r.on_disk || r.host_slot < 0 || r.host_slot == r.disk_src ||
            held_by_fetch(r, now))
            continue;
        unlist(r);
        free_host(r);
        r.tier = KVTier::Disk;
        relist(key, r);
        ++freed;
    }
    /* THEN THE HOST COPIES OF CLEAN VRAM ENTRIES, oldest first. The VRAM copy stays and stays
     * clean -- its disk copy is what lets it go now. These are on no list (a device record is
     * ordered by the cache), so they are found by a walk, which only a full arena of them pays. */
    if (freed < n) {
        std::vector<std::pair<uint64_t, Rec*>> c;
        for (auto& kv : recs_) {
            Rec& r = kv.second;
            if (r.tier == KVTier::Device && !r.pending && r.on_disk && r.host_slot >= 0 &&
                r.host_slot != r.disk_src && !held_by_fetch(r, now))
                c.emplace_back(r.used_ns, &r);
        }
        const size_t k = std::min(c.size(), (size_t)(n - freed));
        std::partial_sort(c.begin(), c.begin() + (long)k, c.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < k; ++i) { free_host(*c[i].second); ++freed; }
    }
    stats_.host_freed += freed;
    return freed;
}

bool IdleTiers::starved() const {
    std::lock_guard lk(mu_);
    return starved_ > 0;
}

void IdleTiers::plan_writeback(int64_t max_entries, std::vector<TierJob>* out) {
    if (!host_on() || max_entries <= 0) return;
    std::lock_guard lk(mu_);
    /* A SEPARATE, MUCH SMALLER BUDGET FOR SNAPSHOTS, because they are not the same size of thing.
     * A block entry is tens of kilobytes; a snapshot on a production hybrid is a hundred megabytes
     * or more. Copying a session's whole retained set in one pass would queue most of a second of
     * link time ahead of whatever the transfer stream is asked for next. */
    int ck_budget = 2;
    std::vector<BlockHash> later;           /* wanted only a snapshot the budget could not take */
    std::vector<size_t> ck_jobs;            /* jobs whose record wants its snapshot copied too */
    size_t first = out->size();
    while (!wb_q_.empty() && (int64_t)(out->size() - first) < max_entries) {
        const BlockHash key = wb_q_.front();
        wb_q_.pop_front();
        auto it = recs_.find(key);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        r.queued = false;
        if (!wants_copy(r)) continue;
        TierJob j;
        j.kind = TierJob::Kind::WriteBack;
        j.key = key;
        j.bytes = cfg_.entry_bytes;
        if (r.tier == KVTier::Device && r.host_slot < 0 && !r.on_disk) j.blocks = r.blocks;
        if (wants_ck_copy(r)) {
            j.ck_pos = r.ck_pos;
            ck_jobs.push_back(out->size());
        }
        out->push_back(std::move(j));
    }

    /* THE SNAPSHOT BUDGET GOES TO THE DEEPEST FIRST. A session commits its checkpoints in order,
     * so the queue holds its shallow ones ahead of its tip -- and the tip is the one its next turn
     * resumes from. Taken in queue order, the shallow ones would fill a snapshot arena a few slots
     * deep and the tip would be refused for good. Across sessions a position says nothing, and the
     * order among them is the queue's, which is as arbitrary as any. */
    std::stable_sort(ck_jobs.begin(), ck_jobs.end(),
                     [out](size_t a, size_t b) { return (*out)[a].ck_pos > (*out)[b].ck_pos; });
    for (size_t idx : ck_jobs) {
        TierJob& j = (*out)[idx];
        if (ck_budget <= 0) continue;           /* a later pass; the record is queued again */
        Rec& r = recs_[j.key];
        int32_t cs = ck_arena_.alloc();
        if (cs < 0) cs = evict_snapshot_slot(/*lossless=*/false);
        if (cs >= 0) {
            j.ck_dev   = r.ck_dev;
            j.ck_host  = cs;
            j.ck_bytes = cfg_.ckpt_bytes;
            --ck_budget;
        } else {
            /* Nowhere to put it, and nothing to evict: this snapshot goes without a copy until a
             * new commit replaces it. Waiting would retry it every pass for the same answer. */
            ++stats_.ck_starved;
            r.ck_skip = true;
        }
    }
    for (size_t i = first; i < out->size();) {
        const TierJob& j = (*out)[i];
        if (j.blocks.empty() && j.ck_host < 0) {
            later.push_back(j.key);
            out->erase(out->begin() + (long)i);
            continue;
        }
        ++i;
    }

    /* THE HOST SLOTS, AS RUNS. Taken for the whole batch at once so that entries planned together
     * sit together in the arena and each run crosses the link as one copy. A batch the arena
     * cannot hold is cut where the arena ran out: the entries past that point keep only their
     * snapshot half, or go back to the front of the queue, and the maintenance plan is told how
     * many copies were refused so it can make that much room. */
    int64_t want = 0;
    for (size_t i = first; i < out->size(); ++i) if (!(*out)[i].blocks.empty()) ++want;
    if (want > arena_.free_slots()) make_room_locked(want - arena_.free_slots());
    std::vector<int32_t> slots;
    slots.reserve((size_t)want);
    while ((int64_t)slots.size() < want) {
        int32_t got = 0;
        const int32_t s = arena_.alloc_run((int32_t)(want - (int64_t)slots.size()), &got);
        if (s < 0) break;
        for (int32_t k = 0; k < got; ++k) slots.push_back(s + k);
    }
    size_t next = 0;
    std::vector<BlockHash> refused;
    for (size_t i = first; i < out->size();) {
        TierJob& j = (*out)[i];
        if (!j.blocks.empty()) {
            if (next < slots.size()) {
                j.host_slot = slots[next++];
            } else {
                ++starved_;
                j.blocks.clear();
                if (j.ck_host < 0) {
                    refused.push_back(j.key);
                    out->erase(out->begin() + (long)i);
                    continue;
                }
            }
        }
        ++i;
    }
    if (want > 0 && slots.empty()) wb_stuck_ = true;
    for (size_t i = first; i < out->size(); ++i) {
        const TierJob& j = (*out)[i];
        Rec& r = recs_[j.key];
        unlist(r);
        r.pending = true;
        r.ck_reading = j.ck_host >= 0;
    }
    /* Back where they were: the refused ones at the front, in order, so the oldest conversation is
     * still the first copied once there is room; the snapshot-only ones at the back, since the
     * budget and not the arena is what held them. */
    for (auto it = refused.rbegin(); it != refused.rend(); ++it) {
        auto ri = recs_.find(*it);
        if (ri == recs_.end() || ri->second.queued) continue;
        wb_q_.push_front(*it);
        ri->second.queued = true;
    }
    for (const BlockHash& k : later) {
        auto ri = recs_.find(k);
        if (ri != recs_.end()) queue_copy(k, ri->second);
    }
}

void IdleTiers::take_clean(std::vector<BlockHash>* out) {
    std::lock_guard lk(mu_);
    for (const BlockHash& k : clean_q_) {
        auto it = recs_.find(k);
        if (it == recs_.end()) continue;
        it->second.clean_q = false;
        out->push_back(k);
    }
    clean_q_.clear();
}

bool IdleTiers::droppable(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    const Rec& r = it->second;
    return !r.pending && !r.doomed && r.tier == KVTier::Device && (r.host_slot >= 0 || r.on_disk);
}

void IdleTiers::drop_commit(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.tier != KVTier::Device) return;
    r.tier = r.host_slot >= 0 ? KVTier::Host : KVTier::Disk;
    ++stats_.dropped;
    wb_stuck_ = false;       /* with no disk tier, a host record is what a capacity drop takes */
    relist(key, r);
}

bool IdleTiers::snapshot_backed(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    const Rec& r = it->second;
    return !r.pending && !r.doomed && !r.ck_reading && r.ck_dev >= 0 &&
           (r.ck_host >= 0 || r.ck_on_disk);
}

void IdleTiers::snapshot_detached(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    it->second.ck_dev = -1;
}

/* ------------------------------------------------------------------ the slow half */

void IdleTiers::plan(int budget, std::vector<TierJob>* out) {
    if (!any_on() || budget <= 0) return;
    std::lock_guard lk(mu_);
    /* ---- disk copies, in the order the host copies landed ---------------------------------- */
    /* ONE SNAPSHOT BUDGET FOR THE WAY TO DISK, apart from the blocks'. A snapshot is a hundred
     * megabytes or more and the disk store copies it twice on the way out (an aligned bounce, then
     * the write), so a plan that sent a session's whole retained set at once would hold the IO
     * thread for most of a second. A record whose snapshot the budget left behind still sends its
     * blocks, and goes to the back of the queue for the rest. */
    int ck_disk_budget = 2;
    int n = 0;
    std::vector<BlockHash> later;
    while (disk_on() && !dk_q_.empty() && n < budget) {
        const BlockHash key = dk_q_.front();
        dk_q_.pop_front();
        auto ri = recs_.find(key);
        if (ri == recs_.end()) continue;
        Rec& r = ri->second;
        if (!r.disk_q) continue;
        r.disk_q = false;
        if (!wants_disk(r)) continue;
        TierJob j;
        j.kind = TierJob::Kind::ToDisk;
        j.key = key;
        j.bytes = cfg_.entry_bytes;
        if (r.host_slot >= 0 && !r.on_disk && r.disk_src < 0) {
            j.host_slot = r.host_slot;
            r.disk_src = r.host_slot;
        }
        if (cfg_.ckpt_disk_slots > 0 && r.ck_host >= 0 && !r.ck_on_disk && r.ck_disk_src < 0) {
            if (ck_disk_budget > 0) {
                j.ck_host  = r.ck_host;
                j.ck_pos   = r.ck_pos;
                j.ck_bytes = cfg_.ckpt_bytes;
                r.ck_disk_src = r.ck_host;
                --ck_disk_budget;
            } else {
                later.push_back(key);
            }
        }
        if (j.host_slot < 0 && j.ck_host < 0) continue;
        out->push_back(std::move(j));
        ++n;
    }
    for (const BlockHash& k : later) {
        auto ri = recs_.find(k);
        if (ri != recs_.end()) queue_disk(k, ri->second);
    }

    /* ---- capacity, with no disk tier: oldest first ----------------------------------------- */
    /* HOW MANY, AND NOT "UNTIL THERE IS ROOM". A queued drop does not free its slot until the
     * engine reports it, so a loop that tested the arena would queue a drop for every entry on the
     * list and empty the whole tier to make room for one. The copies refused since the last plan
     * say how much room was wanted; this frees exactly that much. With a disk tier the room comes
     * from make_room instead, for nothing, once the disk copies above have landed. */
    const int64_t want = starved_;
    starved_ = 0;
    if (disk_on()) return;
    int n_cap = 0;
    for (int64_t k = 0; k < want && n_cap < budget; ) {
        if (host_lru_.empty()) break;
        auto ri = recs_.find(host_lru_.front());
        if (ri == recs_.end()) { host_lru_.pop_front(); continue; }
        Rec& r = ri->second;
        TierJob j;
        j.kind = TierJob::Kind::DropHost;
        j.key = ri->first;
        j.host_slot = r.host_slot;
        j.bytes = cfg_.entry_bytes;
        j.ck_host = r.ck_host;
        unlist(r);
        r.pending = true;
        out->push_back(std::move(j));
        ++k;
        ++n_cap;
    }
}

bool IdleTiers::plan_promote(const BlockHash& key, TierJob* out) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    Rec& r = it->second;
    if (r.pending || r.doomed || r.tier != KVTier::Host) return false;

    *out = TierJob{};
    out->kind = TierJob::Kind::ToDevice;
    out->key = key;
    out->host_slot = r.host_slot;
    out->bytes = cfg_.entry_bytes;
    /* THE STATE COMES BACK WITH THE BLOCKS OR THE PROMOTION IS HALF A PROMOTION. `ck_dev` is an
     * output: the engine allocates the device slot, because only it holds a KVManager, and fills
     * it before this record is told the move succeeded. A snapshot still in its device slot needs
     * nothing -- the cache's index entry for it was never detached. */
    out->ck_dev  = -1;
    out->ck_host = r.ck_host;
    out->ck_pos  = r.ck_pos;
    out->ck_want = r.ck_dev < 0 && (r.ck_host >= 0 || r.ck_on_disk);
    out->ck_bytes = cfg_.ckpt_bytes;
    if (r.fetched_ns) ++stats_.disk_hits; else ++stats_.host_hits;
    unlist(r);
    r.pending = true;
    return true;
}

bool IdleTiers::plan_fetch(const std::vector<BlockHash>& chain, std::vector<TierJob>* out,
                           uint64_t now_ns) {
    if (!disk_on() || chain.empty()) return false;
    std::lock_guard lk(mu_);
    bool wait = false;
    std::vector<Rec*> recs(chain.size(), nullptr);
    std::vector<size_t> need;          /* chain positions whose blocks are only on disk */
    size_t deep = chain.size();        /* the deepest record a restore wants a snapshot for */
    for (size_t i = 0; i < chain.size(); ++i) {
        auto it = recs_.find(chain[i]);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        if (r.fetching) { wait = true; continue; }
        if (r.pending || r.doomed || r.tier == KVTier::Device) continue;
        recs[i] = &r;
        if (r.tier == KVTier::Disk) need.push_back(i);
        if (r.ck_dev < 0 && (r.ck_host >= 0 || r.ck_on_disk)) deep = i;
    }
    const bool ck_read = deep < chain.size() && recs[deep]->ck_host < 0 &&
                         cfg_.ckpt_disk_slots > 0 && ck_arena_.enabled();
    if (need.empty() && !ck_read) return wait;

    /* THE ROOM, OUT OF EVERYTHING BUT THIS CHAIN. What the disk also holds goes for nothing
     * (make_room); the chain's own host copies are what the restore is about to read, and giving
     * one up to read another would only move the gap. */
    for (Rec* r : recs) if (r) r->keep = true;
    std::vector<int32_t> slots;
    if (!need.empty()) {
        const int64_t want = (int64_t)need.size();
        if (want > arena_.free_slots()) make_room_locked(want - arena_.free_slots());
        slots.reserve((size_t)want);
        while ((int64_t)slots.size() < want) {
            int32_t got = 0;
            const int32_t s = arena_.alloc_run((int32_t)(want - (int64_t)slots.size()), &got);
            if (s < 0) break;
            for (int32_t k = 0; k < got; ++k) slots.push_back(s + k);
        }
    }
    int32_t ck_slot = -1;
    if (ck_read) {
        ck_slot = ck_arena_.alloc();
        if (ck_slot < 0) ck_slot = evict_snapshot_slot(/*lossless=*/true);
    }
    for (Rec* r : recs) if (r) r->keep = false;

    /* IN CHAIN ORDER, AND CUT WHERE THE SLOTS RAN OUT: a hit needs a contiguous prefix, so an entry
     * read past a gap would be read for nothing. */
    const size_t first = out->size();
    for (size_t k = 0; k < slots.size(); ++k) {
        TierJob j;
        j.kind = TierJob::Kind::ToHost;
        j.key = chain[need[k]];
        j.host_slot = slots[k];
        j.bytes = cfg_.entry_bytes;
        out->push_back(std::move(j));
    }
    if (ck_slot >= 0) {
        /* Only a snapshot the restore will reach: every disk entry at or before it has a slot. */
        const bool reached = slots.size() == need.size() || need[slots.size()] > deep;
        if (!reached) {
            ck_arena_.free(ck_slot);
        } else {
            TierJob* j = nullptr;
            for (size_t k = first; k < out->size(); ++k)
                if ((*out)[k].key == chain[deep]) { j = &(*out)[k]; break; }
            if (!j) {
                out->push_back(TierJob{});
                j = &out->back();
                j->kind = TierJob::Kind::ToHost;
                j->key = chain[deep];
                j->bytes = cfg_.entry_bytes;
            }
            j->ck_host  = ck_slot;
            j->ck_pos   = recs[deep]->ck_pos;
            j->ck_bytes = cfg_.ckpt_bytes;
        }
    }
    for (size_t k = first; k < out->size(); ++k) {
        Rec& r = recs_[(*out)[k].key];
        unlist(r);
        r.pending = true;
        r.fetching = true;
        r.used_ns = now_ns;
    }
    return wait || out->size() > first;
}

void IdleTiers::report(const TierJob& job, bool ok, uint64_t now_ns) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(job.key);

    /* A DISK WRITE IS NOT A PENDING JOB, so it is settled before anything below touches the
     * record's pending state -- which may belong to a restore running beside it. The slots it
     * read are the record's, unless the record gave them up (or went) while the write was out:
     * then the write's report is what frees them. */
    if (job.kind == TierJob::Kind::ToDisk) {
        if (!ok) ++stats_.failures;
        Rec* r = it == recs_.end() ? nullptr : &it->second;
        if (job.host_slot >= 0) {
            if (r && r->disk_src == job.host_slot) r->disk_src = -1;
            if (!r || r->host_slot != job.host_slot) give_slot(job.host_slot);
            else if (ok) { set_on_disk(*r, true); ++stats_.to_disk; }
        }
        if (job.ck_host >= 0) {
            if (r && r->ck_disk_src == job.ck_host) r->ck_disk_src = -1;
            if (!r || r->ck_host != job.ck_host) ck_arena_.free(job.ck_host);
            else if (ok && job.ck_ok) { r->ck_on_disk = true; ++stats_.ck_to_disk; }
        }
        /* A write that failed is not tried again: a store that refused it once will refuse the
         * next, and the entry stays host-only, which is what it was before. */
        return;
    }

    if (it == recs_.end()) {
        /* Nothing names the slots a copy was handed but the job itself. */
        if (job.kind == TierJob::Kind::WriteBack) {
            give_slot(job.host_slot);
            ck_arena_.free(job.ck_host);
        }
        return;
    }
    Rec& r = it->second;
    r.pending = false;
    r.fetching = false;
    const bool was_reading = r.ck_reading;
    r.ck_reading = false;
    /* A promotion the chain was cut before, or a read that never ran, is not a failure: nothing
     * was tried. */
    const bool may_skip = job.kind == TierJob::Kind::ToDevice || job.kind == TierJob::Kind::ToHost;
    if (!ok && (!may_skip || job.started)) ++stats_.failures;

    /* A device slot the cache let go of while this copy read it is free to go now. */
    if (was_reading && r.ck_free_after >= 0) {
        ck_frees_.push_back(r.ck_free_after);
        r.ck_free_after = -1;
    }

    /* A record that was forgotten mid-transfer is finished off here and nowhere else. Doing it at
     * the top, before the per-kind handling, means no branch below has to remember to check. */
    if (r.doomed) {
        /* THE JOB'S SLOTS AND THE RECORD'S ARE NOT THE SAME SLOTS on every path. A copy's host
         * slots were the job's until it reported, and a promotion out of disk ran through a join
         * slot the record never owned; freeing only the record's would leak those, and freeing
         * the job's as well where they ARE the record's would return one slot twice. */
        if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
        if (job.ck_host   >= 0 && job.ck_host   != r.ck_host)   ck_arena_.free(job.ck_host);
        release_storage(job.key, r);
        recs_.erase(it);
        return;
    }

    switch (job.kind) {
        case TierJob::Kind::WriteBack: {
            if (job.host_slot >= 0) {
                if (ok && r.tier == KVTier::Device && r.host_slot < 0) {
                    r.host_slot = job.host_slot;
                    ++stats_.to_host;
                } else {
                    give_slot(job.host_slot);
                }
            }
            if (job.ck_host >= 0) {
                /* THE SNAPSHOT IS KEPT ONLY IF IT IS STILL THE ONE THAT WAS COPIED. A new commit
                 * at this key, or a release by the cache, moved the record's device slot while the
                 * copy ran, and the bytes that landed describe a state nobody holds any more. */
                if (ok && job.ck_ok && r.ck_dev == job.ck_dev && r.ck_host < 0) {
                    r.ck_host = job.ck_host;
                    ++stats_.ck_to_host;
                } else {
                    ck_arena_.free(job.ck_host);
                }
            }
            relist(job.key, r);
            /* Clean now: the cache is asked whether the VRAM copy can go. */
            if (r.tier == KVTier::Device && r.host_slot >= 0) queue_clean(job.key, r);
            /* A copy that failed, or a snapshot the budget left behind, is tried again. */
            queue_copy(job.key, r);
            /* And what landed goes on to disk. */
            queue_disk(job.key, r);
            return;
        }

        case TierJob::Kind::ToDevice:
            if (ok) {
                r.tier = KVTier::Device;
                r.blocks = job.blocks;
                /* THE HOST COPY IS KEPT, which is what makes the entry clean the moment it lands:
                 * its VRAM copy can go again at no cost, and the next copy this conversation needs
                 * is only of the tokens its next turn adds. A disk entry keeps the slot it was
                 * read into for the same reason -- it holds exactly the entry's bytes -- and keeps
                 * its disk copy, which is what lets that slot go again for nothing. */
                r.host_slot = job.host_slot;
                /* THE CLOCK RESTARTS: the entry came back because a request is about to hit it. */
                if (now_ns) r.used_ns = now_ns;
                r.fetched_ns = 0;
                if (job.ck_dev >= 0) {
                    r.ck_dev = job.ck_dev;
                    r.ck_host = job.ck_host;
                    ++stats_.ck_promoted;
                } else if (job.ck_want && job.ck_host >= 0 && job.ck_ok) {
                    /* THE BLOCKS CAME BACK AND THE STATE DID NOT -- no device slot was free for it.
                     * This turn replays its linear layers from an older checkpoint or from zero,
                     * but the host copy is intact and is KEPT: the cache's entry for it stays
                     * detached, and the next time the blocks leave VRAM and come back the snapshot
                     * comes with them. Dropping it here would make one full slot pool cost the
                     * session its state for good. */
                    r.ck_host = job.ck_host;
                } else if (job.ck_want && job.ck_ok && r.ck_on_disk) {
                    /* No host slot to read the state through, and every one the arena holds is
                     * some other session's only copy. It stays on disk, reachable next time. */
                } else if (job.ck_want) {
                    /* The host copy itself could not be read -- from disk, or it was never there.
                     * There is nothing left to offer and the cache has to hear so. */
                    if (job.ck_host >= 0 && job.ck_host != r.ck_host) ck_arena_.free(job.ck_host);
                    free_ck_host(r);
                    r.ck_on_disk = false;
                    ++stats_.ck_drops;
                    if (on_ck_drop_) on_ck_drop_(job.key);
                }
                ++stats_.promoted;
                queue_clean(job.key, r);
                queue_disk(job.key, r);
                return;
            }
            if (!job.started) {
                /* NEVER STARTED: the reservation ran out before this entry, and nothing about it
                 * was touched. It stays exactly where it was. */
                relist(job.key, r);
                return;
            }
            /* A FAILED PROMOTION DROPS THE ENTRY, AND THE CACHE MUST HEAR ABOUT IT. Every other
             * failure here can leave the record where it was, because nothing was consumed. This
             * one cannot: the scatter may have written some layers and not others, so the blocks
             * hold a mixture and the cached tensors are no longer a function of the tokens. A miss
             * is always safe and this is the only way to guarantee one -- but the prefix cache
             * still holds an entry naming blocks that are not ours any more, and left alone it
             * would serve them. */
            if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
            if (job.ck_host >= 0 && job.ck_host != r.ck_host) ck_arena_.free(job.ck_host);
            {
                const bool had_ck = r.ck_on_disk || r.ck_host >= 0 || job.ck_want;
                free_host(r);
                set_on_disk(r, false);
                free_ck_host(r);
                recs_.erase(it);
                if (had_ck) { ++stats_.ck_drops; if (on_ck_drop_) on_ck_drop_(job.key); }
            }
            if (on_drop_) on_drop_(job.key);
            return;

        case TierJob::Kind::DropHost:
            give_slot(job.host_slot);
            ++stats_.host_drops;
            if (job.ck_host >= 0) { ck_arena_.free(job.ck_host); ++stats_.ck_drops; }
            recs_.erase(it);
            if (on_drop_) on_drop_(job.key);
            return;

        case TierJob::Kind::ToHost: {
            if (!job.started) {
                /* NEVER READ: the slots go back and the record is where it was, on disk. */
                give_slot(job.host_slot);
                if (job.ck_host >= 0) ck_arena_.free(job.ck_host);
                relist(job.key, r);
                return;
            }
            if (!ok) {
                /* THE STORE NO LONGER HOLDS WHAT THE RECORD SAYS IT DOES -- its own eviction does
                 * not tell the tiers, and a slot that fails its verify is dropped by the read. The
                 * record claims bytes nobody has, so it goes, and the cache has to hear so. */
                give_slot(job.host_slot);
                if (job.ck_host >= 0) ck_arena_.free(job.ck_host);
                const bool had_ck = r.ck_dev < 0 && (r.ck_on_disk || r.ck_host >= 0);
                free_host(r);
                set_on_disk(r, false);
                free_ck_host(r);
                recs_.erase(it);
                if (had_ck) { ++stats_.ck_drops; if (on_ck_drop_) on_ck_drop_(job.key); }
                if (on_drop_) on_drop_(job.key);
                return;
            }
            if (job.host_slot >= 0) {
                r.host_slot = job.host_slot;
                r.tier = KVTier::Host;
                ++stats_.fetched;
            }
            if (job.ck_host >= 0) {
                if (job.ck_ok && r.ck_host < 0 && r.ck_dev < 0) {
                    r.ck_host = job.ck_host;
                    ++stats_.ck_fetched;
                } else {
                    ck_arena_.free(job.ck_host);
                    if (!job.ck_ok && r.ck_dev < 0 && r.ck_host < 0 && r.ck_on_disk) {
                        /* The snapshot store lost it; the blocks are still good. */
                        r.ck_on_disk = false;
                        ++stats_.ck_drops;
                        if (on_ck_drop_) on_ck_drop_(job.key);
                    }
                }
            }
            r.fetched_ns = now_ns ? now_ns : rad_mono_ns();
            relist(job.key, r);
            return;
        }

        case TierJob::Kind::ToDisk:
            return;     /* settled above */
    }
}


/* ------------------------------------------------------------------ the session view */

/* A PREFIX RELATION IS WHAT MAKES TWO CHAINS THE SAME CONVERSATION. The blocks are content
 * addressed, so turn N's chain is byte for byte turn N-1's with more on the end; anything that
 * diverges anywhere is a different transcript from that point on, however much of a system prompt
 * it shares. */
static bool chain_extends(const std::vector<BlockHash>& shorter,
                          const std::vector<BlockHash>& longer) {
    if (shorter.size() > longer.size()) return false;
    for (size_t i = 0; i < shorter.size(); ++i) if (shorter[i] != longer[i]) return false;
    return true;
}

void IdleTiers::note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                             uint64_t now_ns) {
    if (chain.empty()) return;
    std::lock_guard lk(mu_);

    /* A SESSION IS A LEAF OF THE PREFIX TRIE, and that is what decides every case here. The turns
     * of one conversation arrive under different ids -- the id is the request's, and a
     * conversation is a new request every turn -- so without this the table grows a row a turn,
     * each holding its own copy of a chain that is a prefix of the next one's and none of them
     * ever revisited. Candidates are the sessions sharing this root, which is a handful at most.
     *
     * A CHAIN THIS ONE EXTENDS HAS STOPPED BEING A LEAF, so it is absorbed: its blocks are the
     * first blocks of this one, byte for byte, because the chain is content addressed.
     *
     * A CHAIN THAT EXTENDS THIS ONE MEANS THIS ONE IS NOT A LEAF EITHER. That happens when a
     * preempted request re-prefills and republishes SHORTER than the conversation it belongs to;
     * recording it would shrink the row to less than the pool is holding, and there may be
     * several conversations below it -- two clients past the same system prompt -- with nothing
     * to say which of them republished. So the most recent of them is touched and no row is
     * added, which is the one answer that is wrong about nothing. */
    uint64_t first_seen = 0;
    bool absorbed = false;
    std::vector<uint64_t>& at_root = roots_[chain[0]];
    for (size_t i = 0; i < at_root.size();) {
        auto it = sessions_.find(at_root[i]);
        if (it == sessions_.end()) { at_root.erase(at_root.begin() + (long)i); continue; }
        if (!chain_extends(it->second.chain, chain)) { ++i; continue; }
        if (first_seen == 0 || it->second.first_seen_ns < first_seen)
            first_seen = it->second.first_seen_ns;
        sessions_.erase(it);
        at_root.erase(at_root.begin() + (long)i);
        absorbed = true;
    }

    if (!absorbed) {
        Session* newest = nullptr;
        for (uint64_t sid : at_root) {
            auto it = sessions_.find(sid);
            if (it == sessions_.end() || !chain_extends(chain, it->second.chain)) continue;
            if (!newest || it->second.last_used_ns > newest->last_used_ns) newest = &it->second;
        }
        if (newest) { newest->last_used_ns = now_ns; return; }
    }

    Session& s = sessions_[id];
    s.last_used_ns = now_ns;
    s.n_tokens = n_tokens;
    s.first_seen_ns = first_seen ? first_seen : now_ns;
    s.chain = chain;
    at_root.push_back(id);
}

void IdleTiers::drop_session(uint64_t id) {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) return;
    if (!it->second.chain.empty()) {
        auto ri = roots_.find(it->second.chain[0]);
        if (ri != roots_.end()) {
            std::vector<uint64_t>& v = ri->second;
            v.erase(std::remove(v.begin(), v.end(), id), v.end());
            if (v.empty()) roots_.erase(ri);
        }
    }
    sessions_.erase(it);
}

void IdleTiers::forget_session(uint64_t id) {
    std::lock_guard lk(mu_);
    drop_session(id);
}

std::vector<IdleTiers::SessionInfo> IdleTiers::sessions(Occupancy* occ) const {
    /* The view's hold on the lock, a slice at a time (YieldingMutex). */
    struct Slice {
        YieldingMutex& m;
        explicit Slice(YieldingMutex& mu) : m(mu) { m.lock_after_others(); }
        ~Slice() { m.unlock(); }
    };
    /* Records a slice visits: a few microseconds of hash lookups, the most an engine-side taker
     * arriving mid-slice waits. */
    constexpr size_t kSliceRecords = 64;

    std::vector<SessionInfo> out;
    {
        Slice lk(mu_);
        out.reserve(sessions_.size());
        for (const auto& kv : sessions_) {
            SessionInfo si;
            si.id = kv.first;
            si.model = model_;
            si.n_tokens = kv.second.n_tokens;
            si.first_seen_ns = kv.second.first_seen_ns;
            si.last_used_ns = kv.second.last_used_ns;
            si.bytes = (int64_t)kv.second.chain.size() * cfg_.entry_bytes;
            out.push_back(std::move(si));
        }
    }
    /* SNAPSHOTS ON DISK, each once. Nothing keeps a running count of them -- a host snapshot has
     * an arena slot to count and a disk one has only a flag on its record -- and this walk visits
     * every record a stored conversation reaches, so it counts them here. They are sparse, one
     * per checkpoint interval, so the set stays small beside the walk that fills it. */
    std::unordered_set<BlockHash, BlockHashHash> disk_snaps;
    size_t kept = 0;
    for (size_t k = 0; k < out.size(); ++k) {
        SessionInfo& si = out[k];
        /* The histogram is computed here and stored nowhere. A session's blocks move one tier at
         * a time under a bounded plan, so a second copy of this would be stale between the pass
         * that moves the first block and the pass that moves the last -- which is exactly the
         * window an operator is looking at the page during. The session is found again at every
         * slice: between slices it may have been absorbed into a longer one or dropped, and a
         * row that is gone is left out. */
        bool gone = false;
        for (size_t at = 0;;) {
            Slice lk(mu_);
            const auto it = sessions_.find(si.id);
            if (it == sessions_.end()) { gone = true; break; }
            const std::vector<BlockHash>& chain = it->second.chain;
            const size_t end = std::min(chain.size(), at + kSliceRecords);
            for (; at < end; ++at) {
                const BlockHash& h = chain[at];
                auto ri = recs_.find(h);
                if (ri == recs_.end()) { ++si.on_device; continue; }
                const Rec& rr = ri->second;
                if (rr.pending) si.moving = true;
                switch (rr.tier) {
                    case KVTier::Device: ++si.on_device; break;
                    case KVTier::Host:   ++si.on_host;   break;
                    case KVTier::Disk:   ++si.on_disk;   break;
                }
                /* THE LINEAR HALF, COUNTED WHERE ITS BYTES ACTUALLY ARE and not where the blocks
                 * are. They normally agree, because a snapshot moves with the record it belongs
                 * to; when they do not -- a snapshot that could not be placed and stayed on the
                 * device -- the page says so rather than averaging the two into something neither
                 * is. */
                const bool ck = rr.ck_dev >= 0 || rr.ck_host >= 0 || rr.ck_on_disk;
                if (rr.ck_dev >= 0)        { ++si.snapshots; ++si.snap_device; }
                else if (rr.ck_host >= 0)  { ++si.snapshots; ++si.snap_host; }
                else if (rr.ck_on_disk)    { ++si.snapshots; ++si.snap_disk; }
                if (rr.ck_on_disk && occ) disk_snaps.insert(h);
                if (ck && rr.ck_pos > si.resume_tokens) si.resume_tokens = rr.ck_pos;
            }
            if (at >= chain.size()) break;
        }
        if (gone) continue;
        si.snap_bytes = si.snapshots * cfg_.ckpt_bytes;
        if (kept != k) out[kept] = std::move(si);
        ++kept;
    }
    out.resize(kept);
    std::sort(out.begin(), out.end(),
              [](const SessionInfo& a, const SessionInfo& b) {
                  return a.last_used_ns > b.last_used_ns;
              });
    if (occ) {
        Slice lk(mu_);
        occ->host_bytes = arena_.used_slots() * cfg_.entry_bytes +
                          ck_arena_.used_slots() * cfg_.ckpt_bytes;
        occ->disk_bytes = stats_.disk_bytes + (int64_t)disk_snaps.size() * cfg_.ckpt_bytes;
        /* The cap the operator gave, which the snapshot store was carved out of -- the same scope
         * as `host_bytes` above, which counts both arenas. */
        occ->host_cap_bytes = host_on() ? cfg_.host_cap_bytes + ck_arena_.cap_bytes() : 0;
        occ->disk_cap_bytes = disk_on() ? cfg_.disk_cap_bytes : 0;
    }
    return out;
}
}  /* namespace rad */
