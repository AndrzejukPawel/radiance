/* sstier.h -- the optional SSD tier for served attention state and linear checkpoints alike, so
 * an agent session does not re-prefill its whole transcript every turn (spec §7.3).
 *
 * ============================== IT HOLDS CONVERSATION CONTENT ON DISK, UNENCRYPTED ==============
 * A KV block IS the prompt: the token ids are stored beside it so a hit can be verified, and the
 * cached tensors are a lossless function of the text. Anything that persists it across restarts
 * says so, and this engine says so at startup whenever --prefix-cache-dir is set. There is no
 * encryption option, no obfuscation, and no claim of one; if the content is sensitive the
 * directory must be on storage the operator is willing to treat as sensitive.
 * ================================================================================================
 *
 * O_DIRECT, same as the weight tier. Going through the page cache would evict the weight tier's own
 * file mappings, which is paid for in decode and would never show up as a *cache* problem. O_DIRECT
 * wants offsets, lengths and buffers all on the logical block size, so every slot and every
 * sub-region here is a multiple of it and there is no unaligned tail to handle.
 *
 * Spec §19.5 records that prefix-cache and weight-tier contention on one SSD has NO PLAN. The
 * TokenBucket below is a placeholder for a measured policy, and it is labelled as one: it is a
 * shared byte budget between the two consumers, which at least makes the contention visible and
 * boundable instead of invisible. What the right split is, or whether a rate limit is even the
 * right shape of answer rather than a priority queue, is a measurement nobody has taken.
 */
#pragma once
#include "device/directio.h"
#include "mem/prefix.h"
#include "rad_core.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* The O_DIRECT file is the device layer's rad::DirectFile (core/device/directio.h). Every extent
 * this tier reads or writes is a whole slot or a whole aligned header region, so every transfer
 * lands on its fast path and its bounce buffer is never allocated -- which is the property worth
 * stating, since the bounce exists for callers reading a header or a file tail and one on this
 * path would mean a slot geometry bug rather than a performance problem. */

/* ------------------------------------------------------------------ the shared rate limiter */
enum class SSDConsumer { PrefixCache = 0, WeightTier = 1, N = 2 };
const char* ssd_consumer_name(SSDConsumer c);

/* PLACEHOLDER FOR A MEASURED POLICY (spec §19.5).
 *
 * Both the prefix cache and the weight tier want O_DIRECT bandwidth and under load they fight. A
 * token bucket is the simplest thing that bounds the fight: one shared byte budget, refilled at a
 * stated rate, drawn on by both. What it does NOT do is decide who should win, and that is the
 * actual open question -- a weight-tier read blocks a layer that is about to execute, while a
 * prefix-cache read only makes a prefill shorter, so a priority scheme is probably the right shape
 * and a flat rate limit certainly is not. Replace this once there is a workload to measure
 * against; the per-consumer counters exist so that measurement has something to read. */
class TokenBucket {
public:
    /* 0 bytes_per_sec disables the limiter entirely, which is the default: an unmeasured rate
     * limit that throttles a machine nobody profiled is worse than none. */
    void configure(int64_t bytes_per_sec, int64_t burst_bytes);
    bool enabled() const { return rate_ > 0; }

    bool try_take(int64_t bytes, SSDConsumer who);   /* non-blocking */
    void take(int64_t bytes, SSDConsumer who);       /* sleeps until the budget allows */

    struct Use { int64_t bytes = 0; int64_t grants = 0; int64_t waits = 0; int64_t wait_us = 0; };
    Use use(SSDConsumer c) const;
    std::string report() const;

private:
    void refill_locked();

    mutable std::mutex mu_;
    int64_t rate_ = 0;          /* bytes per second */
    int64_t burst_ = 0;
    double  tokens_ = 0;
    int64_t last_ns_ = 0;
    Use     use_[(int)SSDConsumer::N];
};

/* The one bucket both consumers share. The weight tier (core/place/) draws on the same object;
 * that is the entire point of it living here rather than inside either one. */
TokenBucket& rad_ssd_bucket();

/* ------------------------------------------------------------------ the tier */
struct SSDTierConfig {
    std::string dir;                 /* Config::prefix_cache_dir */
    int64_t block_payload = 0;       /* bytes one cached attention block occupies, all groups */
    int64_t block_tokens = 0;        /* the attention block size, for the verify */
    int64_t ckpt_payload = 0;        /* KVManager::checkpoint_bytes() */
    /* Slot counts are STATED, not derived from statvfs. A tier that sizes itself from the free
     * space it happens to see at startup is a tier whose capacity changes when somebody untars
     * something, and §6's rule is that budgets do not move on their own. */
    int64_t n_block_slots = 0;
    int64_t n_ckpt_slots = 0;
    uint64_t fingerprint = 0;        /* model + layout identity; a mismatch invalidates the store */
    bool     persistent = false;     /* keep the files across restarts */
    /* Passes RAD_DIO_ALLOW_BUFFERED. Off by default: the device layer REFUSES a filesystem
     * without O_DIRECT rather than downgrading it, because a silent buffered fallback costs the
     * weight tier its own page cache and shows up only as a decode regression nobody can
     * attribute. The flag is for tests, which run on whatever /tmp happens to be. */
    bool     allow_buffered = false;
};

class SSDTier {
public:
    ~SSDTier();
    int  open(const SSDTierConfig& cfg);
    void close();
    bool enabled() const { return enabled_; }

    /* The line the engine prints at startup when the flag is on. Not optional. */
    static std::string disclosure(const std::string& dir);
    std::string report() const;

    /* Attention blocks. `toks` is the block's token ids: they are stored in the slot header and
     * memcmp'd on the way back in, so a hash collision is a miss and never a wrong answer. */
    int put_block(const BlockHash& key, const int32_t* toks, int32_t n_toks,
                  const void* payload, int64_t bytes);
    int get_block(const BlockHash& key, const int32_t* toks, int32_t n_toks,
                  void* payload, int64_t bytes);
    /* MANY BLOCKS AT ONCE, READ IN PARALLEL. A conversation coming back off disk is thousands of
     * entries of tens of kilobytes each, and read one at a time they keep the drive at a queue
     * depth of one -- a fraction of what it delivers with dozens of reads outstanding. `ok[i]` is
     * whether block i was found, read and verified into `payloads[i]`; one that was not is a miss
     * exactly as get_block's would be. Blocks stored with no token ids only. */
    int get_blocks(const BlockHash* keys, int64_t n, void* const* payloads, int64_t bytes,
                   char* ok);

    /* Linear-state checkpoints. Keyed by the chained block hash at the checkpoint position, which
     * is exactly what PrefixCache::reserve_checkpoint keys them by in VRAM. */
    int put_checkpoint(const BlockHash& key, int64_t pos, const void* payload, int64_t bytes);
    int get_checkpoint(const BlockHash& key, int64_t pos, void* payload, int64_t bytes);

    struct Stats {
        int64_t reads = 0, writes = 0, hits = 0, misses = 0, verify_misses = 0, evictions = 0;
        int64_t bytes_read = 0, bytes_written = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    /* `prev`/`next` link the used slots from least to most recently used, so a full store finds
     * the one to give up without a walk. A walk would be over every slot of a store sized in the
     * millions, once per write, on the thread that has to keep up with the host tier's copies. */
    struct Slot { BlockHash key; int32_t used = 0; int64_t pos = 0; int32_t prev = -1, next = -1; };
    struct Store {
        DirectFile file;
        int64_t    slot_bytes = 0;
        int64_t    hdr_bytes = 0;      /* header + token ids, padded */
        int64_t    payload = 0;
        std::vector<Slot> slots;
        std::unordered_map<BlockHash, int32_t, BlockHashHash> index;
        std::vector<int32_t> freelist;
        int32_t    lru_head = -1, lru_tail = -1;   /* least and most recently used */
        void* stage = nullptr;
    };
    /* A used slot leaves the order; a slot becomes the most recently used. */
    static void lru_unlink(Store& s, int32_t i);
    static void lru_push(Store& s, int32_t i);

    int  open_store(Store& s, const std::string& path, int64_t payload, int64_t n_slots,
                    int64_t tok_bytes, bool allow_buffered);
    int64_t scan_store(Store& s, uint32_t magic);
    int  claim(Store& s, const BlockHash& key, int32_t* out);
    int  put(Store& s, uint32_t magic, const BlockHash& key, int64_t pos,
             const int32_t* toks, int32_t n_toks, const void* payload, int64_t bytes);
    int  get(Store& s, uint32_t magic, const BlockHash& key, int64_t pos,
             const int32_t* toks, int32_t n_toks, void* payload, int64_t bytes);
    /* Is the slot read into `stage` the one asked for: this store, this model, this key. */
    bool header_ok(const Store& s, const void* stage, uint32_t magic, const BlockHash& key,
                   int64_t pos, const int32_t* toks, int32_t n_toks, int64_t bytes) const;
    /* A slot whose bytes are not what the index says: out of the index and the order, and free. */
    static void forget_slot(Store& s, int32_t slot);

    SSDTierConfig cfg_;
    std::mutex io_mu_;
    Store  blocks_, ckpts_;
    bool   enabled_ = false;
    Stats  stats_;
    std::string report_;
};

}  /* namespace rad */
