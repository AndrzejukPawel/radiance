// stage_transport.cpp -- rendezvous, shared instances, all_reduce/all_gather.
#include "stage.h"

// ---------------------------------------------------------------------------
// rendezvous. One per process: every rank declares the same graph in the same
// order, so arrival order pairs instances correctly (the same argument
// libr4d's rendezvous rests on). A startup deadlock is fatal either way; the
// 60 s timeout turns it into a named error instead of a hung process.
namespace {
struct Rendezvous {
    std::mutex m;
    std::condition_variable cv;
    int arrived = 0;
    int epoch = 0;
    void* region[8] = {nullptr};
    int dev[8] = {0};
    int failed = 0;
};
Rendezvous g_rv;

int rv_barrier(int world) {
    std::unique_lock<std::mutex> lk(g_rv.m);
    const int e = g_rv.epoch;
    if (++g_rv.arrived == world) {
        g_rv.arrived = 0;
        g_rv.epoch++;
        g_rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = g_rv.cv.wait_for(lk, std::chrono::seconds(60),
                                     [&] { return g_rv.epoch != e; });
    return ok ? RAD_OK : RAD_E_STATE;
}
} // namespace

int stage_rendezvous(void* my_region, int dev, int rank, int world, void** peer_region_out) {
    if (!peer_region_out) return RAD_E_INVAL;
    *peer_region_out = nullptr;
    if (world != 2 || rank < 0 || rank >= world) return RAD_E_SHAPE;
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        g_rv.region[rank] = my_region;
        g_rv.dev[rank] = dev;
    }
    if (rv_barrier(world) != RAD_OK) return RAD_E_STATE;
    void* peer = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_rv.m);
        peer = g_rv.region[1 - rank];
    }
    if (!peer) return RAD_E_STATE;
    *peer_region_out = peer;
    // Second barrier: no rank may launch until every rank published, so the
    // device-pointer resolution below reads final values on both sides.
    return rv_barrier(world);
}

// ---------------------------------------------------------------------------
// shared instance cache: one transport per (rank, world, dtype, bound). At a
// 2048-token chunk a slot is tens of MiB and a model declares over a hundred
// collectives needing a handful of them.

namespace {
struct CacheKey {
    int rank, world, dtype;
    long long maxelems;
    bool operator==(const CacheKey& o) const {
        return rank == o.rank && world == o.world && dtype == o.dtype && maxelems == o.maxelems;
    }
};
struct CacheSlot {
    CacheKey key;
    StageInst* in = nullptr;
    int refs = 0;
};
std::mutex g_cache_mu;
std::vector<CacheSlot> g_cache;

// Region layout: [slot][ready:u32][done:u32], ready/done 16B-aligned.
int build_inst(StageInst* in, int rank, int world, int dtype, long long maxelems) {
    if (world != 2 || rank < 0 || rank >= world) return RAD_E_SHAPE;
    if (maxelems <= 0) return RAD_E_INVAL;
    const int esize = stage_dtype_esize(dtype);
    const long long slot = stage_align16(maxelems * esize);
    const long long flag_off = slot; // slot is 16B-aligned by construction
    const long long total = flag_off + 16;

    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
    void* host = nullptr;
    // Mapped (zero-copy) memory: addressable from any GPU's kernel without
    // any peer mapping. This is the whole transport.
    if (hipHostMalloc(&host, (size_t)total, hipHostMallocMapped) != hipSuccess) return RAD_E_NOMEM;
    if (hipMemset(host, 0, (size_t)total) != hipSuccess) { (void)hipHostFree(host); return RAD_E_DEVICE; }
    // The zeroes must be in memory before the pointer is handed to the peer.
    if (hipStreamSynchronize(nullptr) != hipSuccess) { (void)hipHostFree(host); return RAD_E_DEVICE; }

    void* peer_host = nullptr;
    int rc = stage_rendezvous(host, dev, rank, world, &peer_host);
    if (rc != RAD_OK) { (void)hipHostFree(host); return rc; }

    // Resolve both regions under THIS rank's device. The host pointers are
    // process-common; the device mappings are per device.
    void* my_dev = nullptr;
    void* peer_dev = nullptr;
    if (hipHostGetDevicePointer(&my_dev, host, 0) != hipSuccess) { (void)hipHostFree(host); return RAD_E_DEVICE; }
    if (hipHostGetDevicePointer(&peer_dev, peer_host, 0) != hipSuccess) { (void)hipHostFree(host); return RAD_E_DEVICE; }

    // Device-resident launch counter (replay-safe ticket source).
    unsigned* counter = nullptr;
    if (hipMalloc(&counter, sizeof(unsigned)) != hipSuccess) { (void)hipHostFree(host); return RAD_E_NOMEM; }
    if (hipMemset(counter, 0, sizeof(unsigned)) != hipSuccess) {
        (void)hipFree(counter);
        (void)hipHostFree(host);
        return RAD_E_DEVICE;
    }

    in->rank = rank;
    in->world = world;
    in->dtype = dtype;
    in->esize = esize;
    in->maxelems = maxelems;
    in->slotbytes = slot;
    in->my_host = host;
    in->peer_host = peer_host;
    in->my_dev = my_dev;
    in->peer_dev = peer_dev;
    in->flag_off = flag_off;
    in->my_counter = counter;
    return RAD_OK;
}
} // namespace

int stage_acquire(int rank, int world, int dtype, long long maxelems, StageInst** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        CacheKey k{rank, world, dtype, maxelems};
        for (CacheSlot& s : g_cache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    // Built OUTSIDE the lock: build rendezvouses with the peer and holding a
    // mutex across that barrier would deadlock against the peer's own lookup.
    StageInst* in = new (std::nothrow) StageInst();
    if (!in) return RAD_E_NOMEM;
    int rc = build_inst(in, rank, world, dtype, maxelems);
    if (rc != RAD_OK) { delete in; return rc; }
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        g_cache.push_back(CacheSlot{CacheKey{rank, world, dtype, maxelems}, in, 1});
    }
    *out = in;
    return RAD_OK;
}

void stage_release(StageInst* in) {
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (size_t i = 0; i < g_cache.size(); ++i) {
            if (g_cache[i].in != in) continue;
            if (--g_cache[i].refs > 0) return;
            g_cache.erase(g_cache.begin() + (long)i);
            break;
        }
    }
    // Only this rank's own allocations. The peer's region is the peer's to free.
    if (in->my_counter) (void)hipFree(in->my_counter);
    if (in->my_host) (void)hipHostFree(in->my_host);
    delete in;
}

// Operand dtypes: see stage_operand_code in stage.h.
static int stage_check_msg(const StageInst* in, long long n, int esize_ok) {
    if (!in) return RAD_E_STATE;
    if (n <= 0) return RAD_E_SHAPE;
    if (n > in->maxelems) return RAD_E_SHAPE; // scratch was sized for maxelems
    if (!esize_ok) return RAD_E_DTYPE;
    return RAD_OK;
}

// ---------------------------------------------------------------------------
// all_reduce. Operands: x INOUT, y OUT?. Params: world_size, numel (bound),
// dtype, exact, min_bytes?, hops?.

extern "C" int stage_ar_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    const long long ws = stage_geti(p, n_p, "world_size", world);
    if (ws != world) return RAD_E_SHAPE;
    const int dt = stage_dtype_code(rad_param_gets(p, n_p, "dtype", nullptr));
    if (dt < 0) return RAD_E_DTYPE;
    const long long numel = stage_geti(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_INVAL;
    StageInst* in = nullptr;
    int rc = stage_acquire(rank, world, dt, numel, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

extern "C" void stage_ar_fini(void* instance) { stage_release((StageInst*)instance); }

static int64_t stage_numel(const RadTensor* t) { return rad_tensor_numel(t); }

extern "C" int stage_all_reduce(const RadArgs* a, RadStream s) {
    StageInst* in = (StageInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (!a || a->n_t < 1) return RAD_E_INVAL;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* y = (a->n_t > 1) ? rad_arg_in(a, 1) : nullptr;
    if (!x) return RAD_E_INVAL;
    if (!x->data || !rad_tensor_is_contiguous(x)) return RAD_E_STRIDE;
    const long long n = stage_numel(x);
    // The payload dtype is the operand's own. f16/fp16 and f32/fp32 spellings
    // share one code (the enum resolved them), so this is exact.
    if (stage_operand_code(x->dtype) != in->dtype) return RAD_E_DTYPE;
    if (y) {
        if (!y->data || !rad_tensor_is_contiguous(y)) return RAD_E_STRIDE;
        if (stage_numel(y) != n) return RAD_E_SHAPE;
        if (y->dtype != x->dtype) return RAD_E_DTYPE;
    }
    if (stage_check_msg(in, n, 1) != RAD_OK) return stage_check_msg(in, n, 1);
    const long long nbytes = n * in->esize;
    const StageAddrs ad = stage_addrs(in);
    const long inp = (long)(uintptr_t)x->data;
    const long outp = y ? (long)(uintptr_t)y->data : inp;
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    stage_xchg(ad, inp, nbytes, (long)st);
    stage_reduce(inp, ad.peer_slot, outp, n, in->dtype, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ---------------------------------------------------------------------------
// all_gather. Operands: x IN, y OUT (both required). Params: world_size,
// numel (this rank's contribution bound), dtype, row?.

extern "C" int stage_ag_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    const long long ws = stage_geti(p, n_p, "world_size", world);
    if (ws != world) return RAD_E_SHAPE;
    // A gather moves bytes; the instance width follows the declared dtype so
    // the slot is sized right. Accept any element size the ABI names.
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    int code = stage_dtype_code(dt);
    int esize = 0;
    if (code >= 0) {
        esize = stage_dtype_esize(code);
    } else {
        // Byte movers: fall back to the ABI's width table for exotic dtypes
        // (the u32/f32 candidate pairs the sampler gathers).
        const uint32_t parsed = rad_param_getdt(p, n_p, "dtype");
        if (parsed == RAD_DT_INVALID) return RAD_E_DTYPE;
        esize = (int)rad_dtype_bytes(parsed, 1);
        if (esize <= 0) return RAD_E_DTYPE;
        code = esize == 4 ? 2 : esize == 2 ? 0 : -1;
        if (code < 0) return RAD_E_DTYPE;
    }
    const long long numel = stage_geti(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_INVAL;
    StageInst* in = nullptr;
    int rc = stage_acquire(rank, world, code, numel, &in);
    if (rc != RAD_OK) return rc;
    // The gather's slot must hold CONTRIBUTion bytes under the instance's
    // width: numel*instance-esize must cover numel*actual-esize. Since the
    // instance width came from the declared dtype, they agree by construction
    // whenever the launch's operand dtype matches the declaration (checked
    // below); otherwise refuse.
    if ((long long)in->esize != esize) { stage_release(in); return RAD_E_DTYPE; }
    *out = in;
    return RAD_OK;
}

extern "C" int stage_all_gather(const RadArgs* a, RadStream s) {
    StageInst* in = (StageInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* y = rad_arg_in(a, 1);
    if (!x || !y) return RAD_E_INVAL; // no in-place form: y is world*numel
    if (!x->data || !y->data) return RAD_E_INVAL;
    if (!rad_tensor_is_contiguous(x) || !rad_tensor_is_contiguous(y)) return RAD_E_STRIDE;
    const long long n = stage_numel(x);
    if (n <= 0 || n > in->maxelems) return RAD_E_SHAPE;
    if (stage_numel(y) != n * in->world) return RAD_E_SHAPE;
    if (rad_dtype_bytes(x->dtype, 1) != in->esize) return RAD_E_DTYPE;
    if (y->data == x->data) return RAD_E_INVAL;
    const long long row = rad_args_geti_or(a, "row", 0);
    if (row < 0 || (row > 0 && n % row)) return RAD_E_SHAPE;
    const long long nbytes = n * in->esize;
    const StageAddrs ad = stage_addrs(in);
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    stage_xchg(ad, (long)(uintptr_t)x->data, nbytes, (long)st);
    stage_gather((long)(uintptr_t)x->data, ad.peer_slot, (long)(uintptr_t)y->data, n, in->esize,
                 in->rank, row, row <= 0 ? 1 : 0, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
