/* sstier.cpp -- see sstier.h. IT HOLDS CONVERSATION CONTENT ON DISK, UNENCRYPTED. */
#include "mem/sstier.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rad {

/* The device layer's alignment, not a second opinion about it: every slot, every header region and
 * every staging buffer here is a multiple of it, which is what keeps every transfer on
 * DirectFile's aligned fast path and its bounce buffer unallocated. */
static constexpr int64_t kAlign = DirectFile::alignment();

static void* alloc_aligned(int64_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, (size_t)kAlign, (size_t)align_up(bytes, kAlign)) != 0) return nullptr;
    return p;
}
static constexpr uint32_t kMagicBlock = 0x4b4c4252u;   /* "RBLK" */
static constexpr uint32_t kMagicCkpt  = 0x504b4352u;   /* "RCKP" */

struct SSSlotHeader {
    uint32_t magic;
    uint32_t n_tokens;
    uint64_t fingerprint;
    uint64_t payload_bytes;
    uint64_t key_hi, key_lo;
    int64_t  pos;            /* checkpoint: the token position it restores to; 0 for a block */
    uint64_t reserved[3];
};

/* ------------------------------------------------------------------ TokenBucket */

const char* ssd_consumer_name(SSDConsumer c) {
    switch (c) {
        case SSDConsumer::PrefixCache: return "prefix-cache";
        case SSDConsumer::WeightTier:  return "weight-tier";
        default: return "?";
    }
}

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void TokenBucket::configure(int64_t bytes_per_sec, int64_t burst_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    rate_ = bytes_per_sec > 0 ? bytes_per_sec : 0;
    burst_ = burst_bytes > 0 ? burst_bytes : rate_;
    tokens_ = (double)burst_;
    last_ns_ = now_ns();
}

void TokenBucket::refill_locked() {
    int64_t t = now_ns();
    double dt = (double)(t - last_ns_) * 1e-9;
    last_ns_ = t;
    tokens_ = std::min((double)burst_, tokens_ + dt * (double)rate_);
}

bool TokenBucket::try_take(int64_t bytes, SSDConsumer who) {
    std::lock_guard<std::mutex> lk(mu_);
    Use& u = use_[(int)who];
    if (rate_ <= 0) { u.bytes += bytes; ++u.grants; return true; }
    refill_locked();
    if (tokens_ < (double)bytes) return false;
    tokens_ -= (double)bytes;
    u.bytes += bytes;
    ++u.grants;
    return true;
}

void TokenBucket::take(int64_t bytes, SSDConsumer who) {
    int64_t t0 = now_ns();
    bool waited = false;
    for (;;) {
        if (try_take(bytes, who)) break;
        waited = true;
        /* Sleep for the deficit rather than spinning. Coarse on purpose: this is a placeholder
         * and a precise scheduler for a policy nobody has measured would be effort spent on the
         * wrong thing (spec §19.5). */
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (waited) {
        std::lock_guard<std::mutex> lk(mu_);
        ++use_[(int)who].waits;
        use_[(int)who].wait_us += (now_ns() - t0) / 1000;
    }
}

TokenBucket::Use TokenBucket::use(SSDConsumer c) const {
    std::lock_guard<std::mutex> lk(mu_);
    return use_[(int)c];
}

std::string TokenBucket::report() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::string s = rate_ > 0
        ? fmt("SSD token bucket: %s/s, burst %s  (PLACEHOLDER, spec §19.5)\n",
              humanb(rate_).c_str(), humanb(burst_).c_str())
        : std::string("SSD token bucket: off (no measured rate to enforce, spec §19.5)\n");
    for (int i = 0; i < (int)SSDConsumer::N; ++i)
        s += fmt("    %-14s %10s in %lld ops, %lld waits totalling %lld ms\n",
                 ssd_consumer_name((SSDConsumer)i), humanb(use_[i].bytes).c_str(),
                 (long long)use_[i].grants, (long long)use_[i].waits,
                 (long long)(use_[i].wait_us / 1000));
    return s;
}

TokenBucket& rad_ssd_bucket() { static TokenBucket b; return b; }

/* ------------------------------------------------------------------ SSDTier */

std::string SSDTier::disclosure(const std::string& dir) {
    return fmt("SSD prefix tier active at %s.\n"
               "  It stores CONVERSATION CONTENT ON DISK, UNENCRYPTED: the cached KV blocks are a\n"
               "  lossless function of the prompt, and the prompt's token ids are written beside\n"
               "  them so a hit can be verified. Anything that persists across restarts says so\n"
               "  (spec §7.3). Put this directory on storage you are willing to treat as holding\n"
               "  the text itself.\n", dir.c_str());
}

SSDTier::~SSDTier() { close(); }

static int64_t hdr_region_bytes(int64_t tok_bytes) {
    return align_up((int64_t)sizeof(SSSlotHeader) + tok_bytes, kAlign);
}

int SSDTier::open_store(Store& s, const std::string& path, int64_t payload, int64_t n_slots,
                        int64_t tok_bytes, bool allow_buffered) {
    s.payload    = payload;
    s.hdr_bytes  = hdr_region_bytes(tok_bytes);
    s.slot_bytes = s.hdr_bytes + align_up(payload, kAlign);
    if (n_slots <= 0) return RAD_OK;

    int flags = RAD_DIO_READ | RAD_DIO_WRITE | RAD_DIO_CREATE;
    if (allow_buffered) flags |= RAD_DIO_ALLOW_BUFFERED;
    RAD_TRY(s.file.open(path.c_str(), flags));

    /* Size the store up front. DirectFile does not take a size hint and does not need one -- a
     * write past the end extends the file -- but a store laid out in one ftruncate is one extent
     * rather than N, which is the difference between a slot read being one seek and being several
     * once the filesystem has interleaved somebody else's writes between ours. */
    if (::ftruncate(s.file.fd(), (off_t)(s.slot_bytes * n_slots)) != 0)
        RAD_WARN("ftruncate %s to %s: %s -- the store will be extended by its writes instead",
                 path.c_str(), humanb(s.slot_bytes * n_slots).c_str(), strerror(errno));

    s.slots.assign((size_t)n_slots, Slot{});
    s.freelist.resize((size_t)n_slots);
    for (int64_t i = 0; i < n_slots; ++i) s.freelist[(size_t)i] = (int32_t)(n_slots - 1 - i);
    s.stage = alloc_aligned(s.slot_bytes);
    if (!s.stage) return RAD_E_NOMEM;
    return RAD_OK;
}

int SSDTier::open(const SSDTierConfig& cfg) {
    close();
    cfg_ = cfg;
    if (cfg.dir.empty()) { enabled_ = false; return RAD_OK; }
    if (cfg.block_payload <= 0 && cfg.ckpt_payload <= 0) {
        RAD_ERR("--prefix-cache-dir is set but neither a block payload nor a checkpoint payload was "
                "given; there is nothing for the tier to store");
        return RAD_E_INVAL;
    }

    /* The engine says so at startup. Not behind a verbosity flag. */
    RAD_WARN("%s", disclosure(cfg.dir).c_str());

    if (::mkdir(cfg.dir.c_str(), 0700) != 0 && errno != EEXIST) {
        RAD_ERR("cannot create %s: %s", cfg.dir.c_str(), strerror(errno));
        return RAD_E_IO;
    }

    RAD_TRY(open_store(blocks_, cfg.dir + "/blocks.bin", cfg.block_payload, cfg.n_block_slots,
                       cfg.block_tokens * (int64_t)sizeof(uint32_t), cfg.allow_buffered));
    RAD_TRY(open_store(ckpts_, cfg.dir + "/ckpts.bin", cfg.ckpt_payload, cfg.n_ckpt_slots,
                       0, cfg.allow_buffered));

    enabled_ = true;

    if (cfg.persistent) {
        /* Rebuild the index by scanning the slot headers rather than persisting a separate index
         * file. One sequential pass at startup, and no way for an index to disagree with the data
         * it names -- which is the failure a separate index file has and which produces exactly
         * the "serves one caller another caller's context" outcome this tier verifies against. */
        const int64_t nb = scan_store(blocks_, kMagicBlock);
        if (nb < 0) return (int)nb;
        const int64_t nc = scan_store(ckpts_, kMagicCkpt);
        if (nc < 0) return (int)nc;
        RAD_INFO("SSD tier: recovered %lld block and %lld checkpoint slots from %s",
                 (long long)nb, (long long)nc, cfg.dir.c_str());
    } else {
        /* Not persistent means the store must not outlive the process, and unlinking now rather
         * than at exit means it also does not outlive a crash. The descriptors stay valid. */
        ::unlink((cfg.dir + "/blocks.bin").c_str());
        ::unlink((cfg.dir + "/ckpts.bin").c_str());
    }

    report_ = fmt("SSD tier %s: %lld block slots x %s, %lld checkpoint slots x %s, %s%s\n",
                  cfg.dir.c_str(),
                  (long long)cfg.n_block_slots, humanb(blocks_.slot_bytes).c_str(),
                  (long long)cfg.n_ckpt_slots, humanb(ckpts_.slot_bytes).c_str(),
                  blocks_.file.direct() ? "O_DIRECT" : "BUFFERED",
                  cfg.persistent ? ", persistent" : ", volatile");
    return RAD_OK;
}

void SSDTier::close() {
    for (Store* s : { &blocks_, &ckpts_ }) {
        s->file.close();
        if (s->stage) { ::free(s->stage); s->stage = nullptr; }
        s->slots.clear(); s->index.clear(); s->freelist.clear();
    }
    enabled_ = false;
}

std::string SSDTier::report() const {
    std::string s = report_;
    s += fmt("    %lld hits, %lld misses, %lld verify misses, %lld evictions; "
             "%s read, %s written\n",
             (long long)stats_.hits, (long long)stats_.misses, (long long)stats_.verify_misses,
             (long long)stats_.evictions, humanb(stats_.bytes_read).c_str(),
             humanb(stats_.bytes_written).c_str());
    s += rad_ssd_bucket().report();
    return s;
}

void SSDTier::lru_unlink(Store& s, int32_t i) {
    Slot& x = s.slots[(size_t)i];
    if (x.prev >= 0) s.slots[(size_t)x.prev].next = x.next; else s.lru_head = x.next;
    if (x.next >= 0) s.slots[(size_t)x.next].prev = x.prev; else s.lru_tail = x.prev;
    x.prev = x.next = -1;
}

void SSDTier::lru_push(Store& s, int32_t i) {
    Slot& x = s.slots[(size_t)i];
    x.prev = s.lru_tail;
    x.next = -1;
    if (s.lru_tail >= 0) s.slots[(size_t)s.lru_tail].next = i; else s.lru_head = i;
    s.lru_tail = i;
}

/* THE HEADERS ARE READ MANY AT A TIME. Each is one 4 KiB direct read at its own slot, so a single
 * reader waits one device round trip a slot -- some tens of microseconds, over the millions of
 * slots a store sized in hundreds of GiB holds, which is over a minute of every start. The device
 * serves dozens of such reads at once at little more latency than one, so the scan keeps
 * kScanDepth in flight: one thread each, over its own contiguous run of slots with its own staging
 * page. The threads only record each slot's verdict in the slot itself; the index, the order and
 * the free list are built afterwards in one descending pass, which is the order a single
 * sequential reader would produce. */
static constexpr int64_t kScanDepth = 64;

int64_t SSDTier::scan_store(Store& s, uint32_t magic) {
    if (s.slots.empty()) return 0;
    s.index.clear();
    s.freelist.clear();
    s.lru_head = s.lru_tail = -1;
    for (Slot& x : s.slots) { x.used = 0; x.prev = x.next = -1; }

    const int64_t n  = (int64_t)s.slots.size();
    const int64_t nt = std::min(kScanDepth, n);
    std::atomic<bool> nomem{false};
    std::vector<std::thread> th;
    th.reserve((size_t)nt);
    for (int64_t t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            void* stage = alloc_aligned(s.hdr_bytes);
            if (!stage) { nomem.store(true); return; }
            for (int64_t i = n * t / nt; i < n * (t + 1) / nt; ++i) {
                if (s.file.read_at(stage, i * s.slot_bytes, s.hdr_bytes) < 0) continue;
                const SSSlotHeader* h = (const SSSlotHeader*)stage;
                if (h->magic != magic || h->fingerprint != cfg_.fingerprint
                    || h->payload_bytes != (uint64_t)s.payload) continue;
                Slot& x = s.slots[(size_t)i];
                x.key.hi = h->key_hi;
                x.key.lo = h->key_lo;
                x.pos    = h->pos;
                x.used   = 1;
            }
            std::free(stage);
        });
    for (std::thread& x : th) x.join();
    if (nomem.load()) {
        for (Slot& x : s.slots) x.used = 0;
        RAD_ERR("SSD tier: no staging page for the header scan");
        return RAD_E_NOMEM;
    }

    int64_t live = 0;
    for (const Slot& x : s.slots) live += x.used;
    s.index.reserve((size_t)live);
    s.freelist.reserve((size_t)(n - live));
    for (int64_t i = n - 1; i >= 0; --i) {
        if (!s.slots[(size_t)i].used) { s.freelist.push_back((int32_t)i); continue; }
        lru_push(s, (int32_t)i);
        s.index[s.slots[(size_t)i].key] = (int32_t)i;
    }
    return live;
}

int SSDTier::claim(Store& s, const BlockHash& key, int32_t* out) {
    auto it = s.index.find(key);
    if (it != s.index.end()) { *out = it->second; return RAD_OK; }
    if (!s.freelist.empty()) { *out = s.freelist.back(); s.freelist.pop_back(); return RAD_OK; }

    /* LRU. Evicting the DEEPEST entry first would be better -- a deeper block chains more tokens,
     * is less shareable, and losing it only shortens the usable prefix, whereas losing a shallow
     * one strands everything above it -- but depth is not a property this tier is told. Straight
     * LRU until the caller carries depth in. */
    const int32_t victim = s.lru_head;
    if (victim < 0) return RAD_E_FULL;
    lru_unlink(s, victim);
    s.index.erase(s.slots[(size_t)victim].key);
    s.slots[(size_t)victim].used = 0;
    ++stats_.evictions;
    *out = victim;
    return RAD_OK;
}

int SSDTier::put(Store& s, uint32_t magic, const BlockHash& key, int64_t pos,
                 const int32_t* toks, int32_t n_toks, const void* payload, int64_t bytes) {
    if (!enabled_ || s.slots.empty()) return RAD_E_UNSUPPORTED;
    if (bytes != s.payload) return RAD_E_SHAPE;
    if ((int64_t)sizeof(SSSlotHeader) + (int64_t)n_toks * 4 > s.hdr_bytes) return RAD_E_SHAPE;

    int32_t slot = -1;
    RAD_TRY(claim(s, key, &slot));
    const int64_t base = (int64_t)slot * s.slot_bytes;

    std::memset(s.stage, 0, (size_t)s.slot_bytes);
    SSSlotHeader* h = (SSSlotHeader*)s.stage;
    h->magic = magic;
    h->n_tokens = (uint32_t)n_toks;
    h->fingerprint = cfg_.fingerprint;
    h->payload_bytes = (uint64_t)bytes;
    h->key_hi = key.hi; h->key_lo = key.lo;
    h->pos = pos;
    if (n_toks > 0) std::memcpy((char*)s.stage + sizeof(SSSlotHeader), toks, (size_t)n_toks * 4);
    std::memcpy((char*)s.stage + s.hdr_bytes, payload, (size_t)bytes);

    rad_ssd_bucket().take(s.slot_bytes, SSDConsumer::PrefixCache);

    /* Payload first, header last. Written front to back, a short write would leave the header
     * COMPLETE and the payload truncated, so the slot verifies and the engine loads whatever was
     * in the file underneath it -- which is the worst failure this tier can have.
     *
     * A WRITE THAT FAILS GIVES THE SLOT BACK. claim() took it off the free list or out of the
     * index, so returning here without that leaves a slot no list names and the store one slot
     * smaller for good. Whatever the slot held before is no longer trustworthy either -- the
     * payload may be half overwritten under a header that still verifies -- so the key comes out
     * of the index and the header is cleared on a best-effort basis, which keeps a persistent
     * store from indexing it again at the next start. */
    int wst = s.file.write_at((char*)s.stage + s.hdr_bytes, base + s.hdr_bytes,
                              s.slot_bytes - s.hdr_bytes);
    if (wst >= 0) wst = s.file.write_at(s.stage, base, s.hdr_bytes);
    if (wst < 0) {
        auto it = s.index.find(key);
        if (it != s.index.end() && it->second == slot) s.index.erase(it);
        if (s.slots[(size_t)slot].used) lru_unlink(s, slot);
        s.slots[(size_t)slot].used = 0;
        s.freelist.push_back(slot);
        std::memset(s.stage, 0, (size_t)s.hdr_bytes);
        (void)s.file.write_at(s.stage, base, s.hdr_bytes);
        return wst;
    }

    if (s.slots[(size_t)slot].used) lru_unlink(s, slot);
    s.slots[(size_t)slot].key = key;
    s.slots[(size_t)slot].used = 1;
    s.slots[(size_t)slot].pos = pos;
    lru_push(s, slot);
    s.index[key] = slot;
    ++stats_.writes;
    stats_.bytes_written += s.slot_bytes;
    return RAD_OK;
}

int SSDTier::get(Store& s, uint32_t magic, const BlockHash& key, int64_t pos,
                 const int32_t* toks, int32_t n_toks, void* payload, int64_t bytes) {
    if (!enabled_ || s.slots.empty()) return RAD_E_UNSUPPORTED;
    if (bytes != s.payload) return RAD_E_SHAPE;
    auto it = s.index.find(key);
    if (it == s.index.end()) { ++stats_.misses; return RAD_E_NOTFOUND; }
    const int32_t slot = it->second;
    const int64_t base = (int64_t)slot * s.slot_bytes;

    rad_ssd_bucket().take(s.slot_bytes, SSDConsumer::PrefixCache);
    RAD_TRY(s.file.read_at(s.stage, base, s.slot_bytes));
    ++stats_.reads;
    stats_.bytes_read += s.slot_bytes;

    if (!header_ok(s, s.stage, magic, key, pos, toks, n_toks, bytes)) {
        /* Drop it rather than leaving it to fail every future probe: without this a single bad
         * slot poisons its prefix permanently. */
        forget_slot(s, slot);
        ++stats_.verify_misses;
        return RAD_E_NOTFOUND;
    }
    std::memcpy(payload, (const char*)s.stage + s.hdr_bytes, (size_t)bytes);
    lru_unlink(s, slot);
    lru_push(s, slot);
    ++stats_.hits;
    return RAD_OK;
}

/* Belt and braces on top of the index: verify the header AND the token ids. A store that survives
 * a restart, a model change or a layout change would otherwise serve one caller another caller's
 * context, and that is not a failure anybody would ever diagnose. */
bool SSDTier::header_ok(const Store& s, const void* stage, uint32_t magic, const BlockHash& key,
                        int64_t pos, const int32_t* toks, int32_t n_toks, int64_t bytes) const {
    (void)s;
    const SSSlotHeader* h = (const SSSlotHeader*)stage;
    bool ok = h->magic == magic && h->fingerprint == cfg_.fingerprint
              && h->key_hi == key.hi && h->key_lo == key.lo
              && h->payload_bytes == (uint64_t)bytes && h->pos == pos
              && h->n_tokens == (uint32_t)n_toks;
    if (ok && n_toks > 0)
        ok = std::memcmp((const char*)stage + sizeof(SSSlotHeader), toks, (size_t)n_toks * 4) == 0;
    return ok;
}

void SSDTier::forget_slot(Store& s, int32_t slot) {
    auto it = s.index.find(s.slots[(size_t)slot].key);
    if (it != s.index.end() && it->second == slot) s.index.erase(it);
    if (s.slots[(size_t)slot].used) lru_unlink(s, slot);
    s.slots[(size_t)slot].used = 0;
    s.freelist.push_back(slot);
}

/* READS OUTSTANDING AT ONCE. An NVMe drive reading tens-of-kilobyte extents at random reaches its
 * bandwidth only with many requests in flight; a thread each is the simplest way to have them, and
 * every thread has an aligned staging buffer of its own. */
static constexpr int kReaders = 16;

int SSDTier::get_blocks(const BlockHash* keys, int64_t n, void* const* payloads, int64_t bytes,
                        char* ok) {
    std::lock_guard<std::mutex> lk(io_mu_);
    Store& s = blocks_;
    for (int64_t i = 0; i < n; ++i) ok[i] = 0;
    if (!enabled_ || s.slots.empty()) return RAD_E_UNSUPPORTED;
    if (bytes != s.payload) return RAD_E_SHAPE;

    std::vector<int32_t> slot((size_t)n, -1);
    int64_t found = 0;
    for (int64_t i = 0; i < n; ++i) {
        auto it = s.index.find(keys[i]);
        if (it == s.index.end()) { ++stats_.misses; continue; }
        slot[(size_t)i] = it->second;
        ++found;
    }
    if (found == 0) return RAD_OK;

    /* The reads, and nothing that touches the index or the order: those are settled after, on this
     * thread. 1 read and verified, 2 read and not the key asked for, negative a failed read. */
    std::vector<int> st((size_t)n, 0);
    std::atomic<int64_t> next{0};
    auto reader = [&](void* stage) {
        for (int64_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) {
            if (slot[(size_t)i] < 0) continue;
            rad_ssd_bucket().take(s.slot_bytes, SSDConsumer::PrefixCache);
            const int r = s.file.read_at(stage, (int64_t)slot[(size_t)i] * s.slot_bytes,
                                         s.slot_bytes);
            if (r < 0) { st[(size_t)i] = r; continue; }
            if (!header_ok(s, stage, kMagicBlock, keys[i], 0, nullptr, 0, bytes)) {
                st[(size_t)i] = 2;
                continue;
            }
            std::memcpy(payloads[i], (const char*)stage + s.hdr_bytes, (size_t)bytes);
            st[(size_t)i] = 1;
        }
    };
    const int nt = (int)std::min<int64_t>(kReaders, found);
    std::vector<void*> stages;
    for (int t = 1; t < nt; ++t) {
        void* p = alloc_aligned(s.slot_bytes);
        if (!p) break;
        stages.push_back(p);
    }
    std::vector<std::thread> th;
    th.reserve(stages.size());
    for (void* p : stages) th.emplace_back(reader, p);
    reader(s.stage);
    for (std::thread& t : th) t.join();
    for (void* p : stages) ::free(p);

    for (int64_t i = 0; i < n; ++i) {
        const int32_t sl = slot[(size_t)i];
        if (sl < 0 || st[(size_t)i] < 0) continue;
        ++stats_.reads;
        stats_.bytes_read += s.slot_bytes;
        if (st[(size_t)i] == 2) {
            forget_slot(s, sl);
            ++stats_.verify_misses;
            continue;
        }
        lru_unlink(s, sl);
        lru_push(s, sl);
        ++stats_.hits;
        ok[i] = 1;
    }
    return RAD_OK;
}

/* ONE CALLER AT A TIME. Every call is the KV tiers' IO thread today, but a store is one staging
 * buffer, one slot table and one index, and nothing about the interface says who may call it. */
int SSDTier::put_block(const BlockHash& key, const int32_t* toks, int32_t n_toks,
                       const void* payload, int64_t bytes) {
    std::lock_guard<std::mutex> lk(io_mu_);
    return put(blocks_, kMagicBlock, key, 0, toks, n_toks, payload, bytes);
}
int SSDTier::get_block(const BlockHash& key, const int32_t* toks, int32_t n_toks,
                       void* payload, int64_t bytes) {
    std::lock_guard<std::mutex> lk(io_mu_);
    return get(blocks_, kMagicBlock, key, 0, toks, n_toks, payload, bytes);
}
int SSDTier::put_checkpoint(const BlockHash& key, int64_t pos, const void* payload, int64_t bytes) {
    std::lock_guard<std::mutex> lk(io_mu_);
    return put(ckpts_, kMagicCkpt, key, pos, nullptr, 0, payload, bytes);
}
int SSDTier::get_checkpoint(const BlockHash& key, int64_t pos, void* payload, int64_t bytes) {
    std::lock_guard<std::mutex> lk(io_mu_);
    return get(ckpts_, kMagicCkpt, key, pos, nullptr, 0, payload, bytes);
}

}  /* namespace rad */
