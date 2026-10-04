// stage_fused.cpp -- the four fused collective launches. Each is an exchange
// (shared transport: push, handshake, reduce) followed by the op's math,
// issued as several dispatches on the launch stream. All async, no
// allocation, no synchronisation -- safe for graph-era eager submission
// (describe is null, so sequences containing these run un-captured).
#include "stage.h"

// Read a possibly-ranged bound as its high end (init time).
static long long bbound(const RadParam* p, int n_p, const char* k) {
    return stage_getdim(p, n_p, k, 0);
}

static int64_t row_pitch_elems(const RadTensor* t) {
    if (!t || t->rank == 0) return 0;
    // Strides are in elements. Rank 1 pitches by its own stride (1 when
    // contiguous); higher ranks by the second-to-last axis.
    if (t->rank == 1) return t->stride[0];
    return t->stride[t->rank - 2];
}

// The message tensor must be tight: the exchange pushes logical bytes.
static int require_tight(const RadTensor* t, long long n) {
    if (!t || !t->data || !rad_tensor_is_contiguous(t)) return 0;
    if (row_pitch_elems(t) != n) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// ar_hc_write: y INOUT, inj IN, h INOUT. y reduced in place, then the gated
// residual write h[r][c*n+i] += inj[r][c]*y[r][i].
extern "C" int stage_hc_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (stage_geti(p, n_p, "world_size", world) != world) return RAD_E_SHAPE;
    const long long M = bbound(p, n_p, "M");
    const long long n = stage_geti(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_INVAL;
    StageInst* in = nullptr;
    int rc = stage_acquire(rank, world, 0 /*bf16*/, M * n, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}
extern "C" void stage_hc_fini(void* instance) { stage_release((StageInst*)instance); }

extern "C" int stage_ar_hc_write(const RadArgs* a, RadStream s) {
    StageInst* in = (StageInst*)(a ? a->instance : nullptr);
    if (!in || !a || a->n_t < 3) return a ? RAD_E_STATE : RAD_E_INVAL;
    if (a->world_size != 2) return RAD_E_SHAPE;
    const RadTensor* y = rad_arg_in(a, 0);
    const RadTensor* inj = rad_arg_in(a, 1);
    const RadTensor* h = rad_arg_in(a, 2);
    if (!y || !inj || !h) return RAD_E_INVAL;
    const long long M = rad_args_geti_or(a, "M", 0);
    const long long n = rad_args_geti_or(a, "n", 0);
    const long long hc = rad_args_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 1) return RAD_E_SHAPE;
    if (!require_tight(y, n)) return RAD_E_STRIDE;
    if (!inj->data || !h->data) return RAD_E_INVAL;
    if (stage_operand_code(y->dtype) != 0) return RAD_E_DTYPE;
    const long long nbytes = M * n * 2;
    if (M * n > in->maxelems) return RAD_E_SHAPE;
    const StageAddrs ad = stage_addrs(in);
    const long yp = (long)(uintptr_t)y->data;
    const long peer = ad.peer_slot;
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    stage_xchg(ad, yp, nbytes, (long)st);
    stage_reduce(yp, peer, yp, M * n, 0, (long)st);
    stage_hc_write(yp, (long)(uintptr_t)inj->data, (long)(uintptr_t)h->data, M, n, hc,
                   row_pitch_elems(inj), row_pitch_elems(h), n, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ---------------------------------------------------------------------------
// ar_gather_hc_write: ye,ew,sorted,sh?,sg?,y INOUT,inj,h.
// Separate instance cache (adds the inverse table for the slot lookup).

namespace {
struct GatherInst {
    StageInst* base = nullptr;
    void* inv = nullptr;    // device u32, Tmax entries
    long long tmax = 0;
};
struct GKey {
    int rank, world, dtype;
    long long maxelems, top_k;
    bool operator==(const GKey& o) const {
        return rank == o.rank && world == o.world && dtype == o.dtype &&
               maxelems == o.maxelems && top_k == o.top_k;
    }
};
struct GSlot {
    GKey key;
    GatherInst* in = nullptr;
    int refs = 0;
};
std::mutex g_gmu;
std::vector<GSlot> g_gcache;
} // namespace

static int gather_acquire(int rank, int world, long long maxelems, long long top_k,
                          GatherInst** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_gmu);
        GKey k{rank, world, 0, maxelems, top_k};
        for (GSlot& s : g_gcache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    GatherInst* g = new (std::nothrow) GatherInst();
    if (!g) return RAD_E_NOMEM;
    StageInst* base = nullptr;
    int rc = stage_acquire(rank, world, 0, maxelems, &base);
    if (rc != RAD_OK) { delete g; return rc; }
    void* inv = nullptr;
    if (hipMalloc(&inv, (size_t)(maxelems * top_k) * sizeof(int)) != hipSuccess) {
        stage_release(base);
        delete g;
        return RAD_E_NOMEM;
    }
    g->base = base;
    g->inv = inv;
    g->tmax = maxelems * top_k;
    {
        std::lock_guard<std::mutex> lk(g_gmu);
        g_gcache.push_back(GSlot{GKey{rank, world, 0, maxelems, top_k}, g, 1});
    }
    *out = g;
    return RAD_OK;
}

static void gather_release(GatherInst* g) {
    if (!g) return;
    {
        std::lock_guard<std::mutex> lk(g_gmu);
        for (size_t i = 0; i < g_gcache.size(); ++i) {
            if (g_gcache[i].in != g) continue;
            if (--g_gcache[i].refs > 0) return;
            g_gcache.erase(g_gcache.begin() + (long)i);
            break;
        }
    }
    if (g->inv) hipFree(g->inv);
    stage_release(g->base);
    delete g;
}

extern "C" int stage_ghc_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (stage_geti(p, n_p, "world_size", world) != world) return RAD_E_SHAPE;
    const long long M = bbound(p, n_p, "M");
    const long long n = stage_geti(p, n_p, "n", 0);
    const long long top_k = stage_geti(p, n_p, "top_k", 0);
    if (M <= 0 || n <= 0 || top_k <= 0) return RAD_E_INVAL;
    GatherInst* g = nullptr;
    int rc = gather_acquire(rank, world, M * n, top_k, &g);
    if (rc != RAD_OK) return rc;
    *out = g;
    return RAD_OK;
}
extern "C" void stage_ghc_fini(void* instance) { gather_release((GatherInst*)instance); }

extern "C" int stage_ar_gather_hc_write(const RadArgs* a, RadStream s) {
    GatherInst* g = (GatherInst*)(a ? a->instance : nullptr);
    if (!g || !g->base) return RAD_E_STATE;
    StageInst* in = g->base;
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    if (a->world_size != 2) return RAD_E_SHAPE;
    const RadTensor* ye = rad_arg_in(a, 0);
    const RadTensor* ew = rad_arg_in(a, 1);
    const RadTensor* sorted = rad_arg_in(a, 2);
    const RadTensor* sh = rad_arg_in(a, 3);
    const RadTensor* sg = rad_arg_in(a, 4);
    const RadTensor* y = rad_arg_in(a, 5);
    const RadTensor* inj = rad_arg_in(a, 6);
    const RadTensor* h = rad_arg_in(a, 7);
    if (!ye || !ew || !sorted || !y || !inj || !h) return RAD_E_INVAL;
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    if (!ye->data || !ew->data || !sorted->data) return RAD_E_INVAL;
    if (!rad_tensor_is_contiguous(sorted)) return RAD_E_STRIDE;
    if ((sh && !sh->data) || (sg && !sg->data)) return RAD_E_INVAL;
    if (stage_operand_code(ye->dtype) != 0 || stage_operand_code(ew->dtype) != 0)
        return RAD_E_DTYPE;
    if (sorted->dtype != (uint32_t)RAD_I32) return RAD_E_DTYPE;
    const long long M = rad_args_geti_or(a, "M", 0);
    const long long n = rad_args_geti_or(a, "n", 0);
    const long long hc = rad_args_geti_or(a, "hc", 0);
    const long long top_k = rad_args_geti_or(a, "top_k", 0);
    if (M < 1 || n < 1 || hc < 1 || top_k < 1) return RAD_E_SHAPE;
    if (!require_tight(y, n)) return RAD_E_STRIDE;
    if (stage_operand_code(y->dtype) != 0) return RAD_E_DTYPE;
    if (M * n > in->maxelems || M * top_k > g->tmax) return RAD_E_SHAPE;
    const char* act = rad_args_gets_or(a, "act", "none");
    const int sig = act && !std::strcmp(act, "sigmoid") ? 1 : 0;
    if (sh && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    const long long T = M * top_k;
    const long yp = (long)(uintptr_t)y->data;
    // gather into y (the message), then exchange+reduce in place, then write.
    stage_gather_inv((long)(uintptr_t)g->inv, (long)(uintptr_t)sorted->data, T, (long)st);
    stage_gather_rows((long)(uintptr_t)ye->data, (long)(uintptr_t)ew->data,
                      (long)(uintptr_t)sorted->data,
                      sh ? (long)(uintptr_t)sh->data : 0, sg ? (long)(uintptr_t)sg->data : 0,
                      (long)(uintptr_t)g->inv, yp, M, n, top_k, sig,
                      row_pitch_elems(ye), row_pitch_elems(ew),
                      sh ? row_pitch_elems(sh) : 0, sg ? row_pitch_elems(sg) : 0,
                      n, 1, (long)st);
    const StageAddrs ad = stage_addrs(in);
    const long peer = ad.peer_slot;
    stage_xchg(ad, yp, M * n * 2, (long)st);
    stage_reduce(yp, peer, yp, M * n, 0, (long)st);
    stage_hc_write(yp, (long)(uintptr_t)inj->data, (long)(uintptr_t)h->data, M, n, hc,
                   row_pitch_elems(inj), row_pitch_elems(h), n, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ---------------------------------------------------------------------------
// ar_rmsnorm_quant_fp8: x INOUT (message, reduced in place), res INOUT?,
// w WEIGHT, q OUT, s OUT, ob OUT?. Params M,n,eps,group,world,dtype,wadd?,wire?
extern "C" int stage_rnq_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (stage_geti(p, n_p, "world_size", world) != world) return RAD_E_SHAPE;
    const long long M = bbound(p, n_p, "M");
    const long long n = stage_geti(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_INVAL;
    StageInst* in = nullptr;
    int rc = stage_acquire(rank, world, 0 /*bf16*/, M * n, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}
extern "C" void stage_rnq_fini(void* instance) { stage_release((StageInst*)instance); }

extern "C" int stage_ar_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    StageInst* in = (StageInst*)(a ? a->instance : nullptr);
    if (!in || !a || a->n_t < 5) return RAD_E_STATE;
    if (a->world_size != 2) return RAD_E_SHAPE;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* w = rad_arg_in(a, 2);
    const RadTensor* q = rad_arg_in(a, 3);
    const RadTensor* sc = rad_arg_in(a, 4);
    const RadTensor* ob = (a->n_t > 5) ? rad_arg_in(a, 5) : nullptr;
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (res && res->dtype != (uint32_t)RAD_BF16) return RAD_E_DTYPE;
    if (ob && ob->dtype != (uint32_t)RAD_BF16) return RAD_E_DTYPE;
    if (w->dtype != (uint32_t)RAD_BF16 && w->dtype != (uint32_t)RAD_F32) return RAD_E_DTYPE;
    const int w_f32 = w->dtype == (uint32_t)RAD_F32 ? 1 : 0;
    const long long M = rad_args_geti_or(a, "M", 0);
    const long long n = rad_args_geti_or(a, "n", 0);
    const long long group = rad_args_geti_or(a, "group", 0);
    if (M < 1 || n < 1 || group < 1) return RAD_E_SHAPE;
    if (!require_tight(x, n)) return RAD_E_STRIDE;
    if (stage_operand_code(x->dtype) != 0) return RAD_E_DTYPE;
    if (M * n > in->maxelems) return RAD_E_SHAPE;
    const float eps = (float)rad_args_getf_or(a, "eps", 1e-6);
    const float wadd = (float)rad_args_getf_or(a, "wadd", 0.0);
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    const long xp = (long)(uintptr_t)x->data;
    const StageAddrs ad = stage_addrs(in);
    const long peer = ad.peer_slot;
    stage_xchg(ad, xp, M * n * 2, (long)st);
    stage_reduce(xp, peer, xp, M * n, 0, (long)st);
    stage_rmsnorm_quant_fp8(xp, res ? (long)(uintptr_t)res->data : 0,
                            (long)(uintptr_t)w->data, (long)(uintptr_t)q->data,
                            (long)(uintptr_t)sc->data, ob ? (long)(uintptr_t)ob->data : 0,
                            M, n, eps, wadd, group, n,
                            res ? row_pitch_elems(res) : 0, row_pitch_elems(w),
                            row_pitch_elems(q), row_pitch_elems(sc),
                            ob ? row_pitch_elems(ob) : 0, w_f32, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ---------------------------------------------------------------------------
// ar_ln_had_quant_i8: x IN (delta, untouched), res INOUT?, w, q, s, ob?.
// The reduced message lands in the launch scratch (a bf16 round trip the
// schema's IN marking forbids writing into x).
static int64_t stage_ln_scratch(const RadArgs* a) {
    if (!a) return 0;
    long long M = 0, n = 0;
    rad_args_geti(a, "M", &M);
    // M may be ranged at declare; the scratch hook takes the high bound.
    const RadParam* f = rad_param_find(a->p, a->n_p, "M");
    if (f && f->kind == RAD_P_RANGE) M = f->ihi;
    rad_args_geti(a, "n", &n);
    if (M <= 0 || n <= 0) return 0;
    return stage_align16(M * n * 2);
}

extern "C" int stage_lnq_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (stage_geti(p, n_p, "world_size", world) != world) return RAD_E_SHAPE;
    const long long M = bbound(p, n_p, "M");
    const long long n = stage_geti(p, n_p, "n", 0);
    const long long group = stage_geti(p, n_p, "group", 0);
    if (M <= 0 || n <= 0) return RAD_E_INVAL;
    if (group != 128 && group != 256 && group != 512) return RAD_E_SHAPE;
    StageInst* in = nullptr;
    int rc = stage_acquire(rank, world, 0 /*bf16*/, M * n, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}
extern "C" void stage_lnq_fini(void* instance) { stage_release((StageInst*)instance); }

extern "C" int stage_ar_ln_had_quant_i8(const RadArgs* a, RadStream s) {
    StageInst* in = (StageInst*)(a ? a->instance : nullptr);
    if (!in || !a || a->n_t < 5) return RAD_E_STATE;
    if (a->world_size != 2) return RAD_E_SHAPE;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* w = rad_arg_in(a, 2);
    const RadTensor* q = rad_arg_in(a, 3);
    const RadTensor* sc = rad_arg_in(a, 4);
    const RadTensor* ob = (a->n_t > 5) ? rad_arg_in(a, 5) : nullptr;
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (res && res->dtype != (uint32_t)RAD_BF16) return RAD_E_DTYPE;
    if (ob && ob->dtype != (uint32_t)RAD_BF16) return RAD_E_DTYPE;
    if (w->dtype != (uint32_t)RAD_BF16 && w->dtype != (uint32_t)RAD_F32) return RAD_E_DTYPE;
    const int w_f32_ln = w->dtype == (uint32_t)RAD_F32 ? 1 : 0;
    const long long M = rad_args_geti_or(a, "M", 0);
    const long long n = rad_args_geti_or(a, "n", 0);
    const long long had = rad_args_geti_or(a, "group", 0);
    if (M < 1 || n < 1 || (had != 128 && had != 256 && had != 512)) return RAD_E_SHAPE;
    if (!require_tight(x, n)) return RAD_E_STRIDE;
    if (stage_operand_code(x->dtype) != 0) return RAD_E_DTYPE;
    if (M * n > in->maxelems) return RAD_E_SHAPE;
    if (!a->scratch || a->scratch_bytes < M * n * 2) return RAD_E_SCRATCH;
    const float eps = (float)rad_args_getf_or(a, "eps", 1e-6);
    const float wadd = (float)rad_args_getf_or(a, "wadd", 0.0);
    hipStream_t st = (hipStream_t)s;
    if (!st) return RAD_E_INVAL;
    const long xp = (long)(uintptr_t)x->data;
    const long red = (long)(uintptr_t)a->scratch;
    const StageAddrs ad = stage_addrs(in);
    const long peer = ad.peer_slot;
    stage_xchg(ad, xp, M * n * 2, (long)st);
    stage_reduce(xp, peer, red, M * n, 0, (long)st);
    stage_ln_had_quant_i8(xp, red, res ? (long)(uintptr_t)res->data : 0,
                          (long)(uintptr_t)w->data, (long)(uintptr_t)q->data,
                          (long)(uintptr_t)sc->data, ob ? (long)(uintptr_t)ob->data : 0,
                          M, n, eps, wadd, had, n, n,
                          res ? row_pitch_elems(res) : 0, row_pitch_elems(w),
                          row_pitch_elems(q), ob ? row_pitch_elems(ob) : 0, w_f32_ln, (long)st);
    stage_xdone(ad, (long)st);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// The scratch hook for ar_ln_had_quant_i8 (the reduced message lives here).
extern "C" int64_t stage_lnq_scratch(const RadArgs* a) { return stage_ln_scratch(a); }
