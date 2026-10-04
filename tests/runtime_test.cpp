/* runtime_test.cpp -- the run phase, against a Program built by hand and a kernel plugin that
 * records what it was called with.
 *
 * The point of building the Program by hand rather than through the builder is that this test has
 * to fail for exactly one reason. A bucket table indexed off by one at a band boundary and a
 * declare phase that built the wrong boundaries are the same symptom through a real builder, and
 * only one of them is the run phase's fault.
 *
 * The device layer is faked with WEAK definitions, so wherever core/device/ is linked in its
 * strong definitions win and this file's fakes fall away without a duplicate-symbol error; the
 * assertions are written so they hold either way. The two that matter -- that a pending weight causes a
 * STREAM wait and not a host sync -- are checked through Ctx::stream_waits(), which is the
 * runtime's own counter, and through the fakes' sync counters, which stay at zero whichever
 * backend is linked.
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/arena.h"
#include "runtime/ctx.h"
#include "runtime/rank_barrier.h"
#include "runtime/residency.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <thread>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

/* ================================================================== the fake device */
/* Weak, so a linked-in real backend's strong definitions replace these silently. */
#define WEAKDEF __attribute__((weak))

static long g_stream_waits = 0;   /* rad_event_wait  -- ordering, no host block */
static long g_event_syncs  = 0;   /* rad_event_sync  -- a HOST block */
static long g_stream_syncs = 0;   /* rad_stream_sync -- a HOST block */
static long g_copies       = 0;

extern "C" {

WEAKDEF int  rad_dev_count(void) { return 1; }
WEAKDEF int  rad_dev_set(int) { return RAD_OK; }
WEAKDEF int  rad_dev_props(int, RadDeviceProps* out) {
    if (out) { std::memset(out, 0, sizeof *out); std::strcpy(out->name, "fake");
               std::strcpy(out->arch, "host"); out->is_host_backend = 1; }
    return RAD_OK;
}
WEAKDEF int  rad_dev_enable_peer(int, int) { return RAD_OK; }

WEAKDEF void* rad_dev_alloc(int64_t bytes, int) {
    size_t n = (size_t)((bytes + 4095) / 4096 * 4096);
    void* p = std::aligned_alloc(4096, n);
    if (p) std::memset(p, 0, n);
    return p;
}
WEAKDEF void  rad_dev_free(void* p, int) { std::free(p); }
WEAKDEF void* rad_dev_host_ptr(void* p) { return p; }
WEAKDEF void* rad_dev_device_ptr(void* p) { return p; }

WEAKDEF int  rad_stream_create(RadStream* out, int) { *out = (RadStream)new int(1); return RAD_OK; }
WEAKDEF void rad_stream_destroy(RadStream s) { delete (int*)s; }
WEAKDEF int  rad_stream_sync(RadStream) { ++g_stream_syncs; return RAD_OK; }

WEAKDEF int  rad_event_create(RadEvent* out) { *out = (RadEvent)new int(0); return RAD_OK; }
WEAKDEF int  rad_event_create_local(RadEvent* out) { *out = (RadEvent)new int(0); return RAD_OK; }
WEAKDEF int  rad_event_create_as(RadEvent* out, unsigned) { *out = (RadEvent)new int(0); return RAD_OK; }
WEAKDEF void rad_event_destroy(RadEvent e) { delete (int*)e; }
WEAKDEF int  rad_event_record(RadEvent, RadStream) { return RAD_OK; }
WEAKDEF int  rad_event_wait(RadStream, RadEvent) { ++g_stream_waits; return RAD_OK; }
WEAKDEF int  rad_event_query(RadEvent) { return 1; }
WEAKDEF int  rad_event_sync(RadEvent) { ++g_event_syncs; return RAD_OK; }
WEAKDEF void rad_dev_host_wrote(void) {}
WEAKDEF int  rad_event_elapsed_ms(RadEvent, RadEvent, float* out) { if (out) *out = 0.0f; return RAD_OK; }

WEAKDEF int rad_memcpy_async(void* d, const void* s, int64_t n, RadStream) {
    ++g_copies; std::memcpy(d, s, (size_t)n); return RAD_OK;
}
WEAKDEF int rad_memcpy_2d_async(void* d, int64_t dp, const void* s, int64_t sp,
                                int64_t w, int64_t h, RadStream) {
    for (int64_t r = 0; r < h; ++r)
        std::memcpy((char*)d + r * dp, (const char*)s + r * sp, (size_t)w);
    return RAD_OK;
}
WEAKDEF int rad_memset_async(void* d, int v, int64_t n, RadStream) {
    std::memset(d, v, (size_t)n); return RAD_OK;
}
WEAKDEF const char* rad_dev_last_error(void) { return "none"; }

}  /* extern "C" */

/* ================================================================== the fake kernel plugin */
/* Records what it was called with. Its push_back allocates, which is fine: the rule is that the
 * RUNTIME does not allocate on the step path, and this stands in for a kernel launch. */
struct Rec {
    const char* kernel = nullptr;
    long long   sk = 0;   /* a tuned axis, read the way a real kernel reads one */
    RadStream   stream = nullptr;
    int         n_t = 0;
    RadTensor   t[8]{};
    long long   M = -1;
    int         rank = 0, world = 0;
    void*       scratch = nullptr;
    int64_t     scratch_bytes = 0;
    void*       instance = nullptr;
};

static std::vector<Rec> g_rec;

static int record(const char* name, const RadArgs* a, RadStream s) {
    Rec r;
    r.kernel = name;
    rad_args_geti(a, "sk", &r.sk);
    r.stream = s;
    r.n_t = a->n_t;
    for (int i = 0; i < a->n_t && i < 8; ++i) r.t[i] = a->t[i];
    rad_args_geti(a, "M", &r.M);
    r.rank = a->rank;
    r.world = a->world_size;
    r.scratch = a->scratch;
    r.scratch_bytes = a->scratch_bytes;
    r.instance = a->instance;
    g_rec.push_back(r);
    return RAD_OK;
}

static int k_m16 (const RadArgs* a, RadStream s) { return record("m16",  a, s); }
static int k_m64 (const RadArgs* a, RadStream s) { return record("m64",  a, s); }
static int k_pre (const RadArgs* a, RadStream s) { return record("prefill", a, s); }
static int k_bad (const RadArgs* a, RadStream s) { record("bad", a, s); return RAD_E_SHAPE; }
static int k_null(const RadArgs*,   RadStream)            { return RAD_OK; }   /* for the cost probe */

/* ================================================================== the fake residency table */
struct FakeResidency : rad::ResidencyTable {
    rad_weight      which = 0;
    rad::WeightSlot slot{};
    mutable int     lookups = 0;
    uint64_t        ep = 1;

    std::vector<rad_weight> changed;

    const rad::WeightSlot* lookup(rad_weight w) const override {
        ++lookups;
        return w == which ? &slot : nullptr;
    }
    uint64_t epoch() const override { return ep; }
    const std::vector<rad_weight>& changes() const override { return changed; }
    void clear_changes() override { changed.clear(); }
};

/* A slot for every weight, for the tables. */
struct TableResidency : rad::ResidencyTable {
    std::vector<rad::WeightSlot> slot;
    mutable int                  lookups = 0;
    uint64_t                     ep = 1;

    std::vector<rad_weight>      changed;

    const rad::WeightSlot* lookup(rad_weight w) const override {
        ++lookups;
        return (size_t)w < slot.size() ? &slot[(size_t)w] : nullptr;
    }
    uint64_t epoch() const override { return ep; }
    const std::vector<rad_weight>& changes() const override { return changed; }
    void clear_changes() override { changed.clear(); }
    /* What the mover does: a new pointer, a new generation, and the weight named as changed. */
    void move(rad_weight w, void* p) {
        slot[(size_t)w].ptr = p;
        ++slot[(size_t)w].generation;
        ++ep;
        changed.push_back(w);
    }
};

/* ================================================================== the fixture */
static std::function<void(Ctx&)> g_body;
static void arch_step_trampoline(RadCtx* c, const RadBatch*) {
    if (g_body) g_body(*static_cast<Ctx*>(c));
}

/* The gemm_nt schema, three positional operands. Static so the pointer outlives the Program. */
static const RadParamSpec   g_gemm_params[] = {
    { "M", RAD_P_INT, RAD_REQUIRED }, { "N", RAD_P_INT, RAD_REQUIRED },
    { "K", RAD_P_INT, RAD_REQUIRED }, { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec g_gemm_opds[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOpSchema g_gemm_schema = {
    "gemm_nt", g_gemm_params, 4, g_gemm_opds, 3, "A[M,K] x B[N,K]^T -> Y[M,N]"
};

static uint16_t g_weight_bytes[64];   /* somewhere for WeightInfo::ptr to point */
static uint16_t g_moved_bytes[64];    /* where a "relocated" weight lives */

struct Fixture {
    Program                    P;
    std::deque<RadKernelInfo>  infos;
    std::deque<rad::KernelRow> rows;
    Ctx                        ctx;
    RadBatch                   batch{};

    Fixture() {
        P.meta.n_layers = 2;
        P.meta.n_expert = 4;
        P.arena_bytes   = 8192;
        P.scratch_bytes = 256;
        P.weights.resize(1);   /* handles are 1-based; index 0 is the sentinel */
        P.buffers.resize(1);
        P.ops.resize(1);
    }

    rad_weight add_weight(const char* name, uint32_t dt, int64_t d0, int64_t d1, void* ptr) {
        rad::WeightInfo w;
        w.name = name;
        w.decl.dtype = dt;
        w.decl.rank = 2;
        w.decl.shape[0] = d0;
        w.decl.shape[1] = d1;
        w.ptr = ptr;
        P.weights.push_back(w);
        return (rad_weight)(P.weights.size() - 1);
    }

    rad_buf add_buffer(const char* name, uint32_t dt, int64_t d0, int64_t d1, int64_t off) {
        rad::BufferInfo b;
        b.name = name;
        b.decl.dtype = dt;
        b.decl.rank = 2;
        b.decl.shape[0] = d0;
        b.decl.shape[1] = d1;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rad_dtype_bytes(dt, d0 * d1);
        b.arena_offset = off;
        P.buffers.push_back(b);
        return (rad_buf)(P.buffers.size() - 1);
    }

    /* One op, one band per (hi, launch, kernel name). A null launch leaves that band unresolved,
     * which is what a hole in the bucket table looks like. */
    rad_op add_op(const char* op, const char* ranged_key,
                  const std::vector<int64_t>& hi,
                  const std::vector<RadLaunchFn>& launch,
                  const std::vector<const char*>& kname,
                  const RadOpSchema* schema,
                  const std::vector<rad_weight>& weights) {
        rad::OpInfo o;
        o.op = op;
        o.index = (int32_t)P.ops.size();
        o.ranged_key = ranged_key;
        o.range_lo = 1;
        o.range_hi = hi.empty() ? 0 : hi.back();
        o.schema = schema;
        o.weights = weights;
        o.base.set_i("N", 8);
        o.base.set_i("K", 4);
        o.base.set_s("dtype", "bf16");

        for (size_t i = 0; i < hi.size(); ++i) {
            rad::Band b;
            b.hi = hi[i];
            b.miss[RAD_DOMAIN_HOST] = "the fake plugin has no host kernels";
            if (launch[i]) {
                infos.push_back(RadKernelInfo{});
                RadKernelInfo& ki = infos.back();
                ki.name   = kname[i];
                ki.op     = op;
                ki.family = "test";
                ki.domain = RAD_DOMAIN_DEVICE;
                ki.launch = launch[i];

                rows.push_back(rad::KernelRow{});
                rad::KernelRow& kr = rows.back();
                kr.info = &ki;
                kr.plugin = "fake";
                kr.index = (int)i;

                rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
                r.row = &kr;
                r.scratch_bytes = 128;
                r.tune_source = "axis defaults";
                r.geom.set_i(ranged_key, hi[i]);
                r.geom.set_i("N", 8);
                r.geom.set_i("K", 4);
                r.geom.set_s("dtype", "bf16");
            } else {
                b.miss[RAD_DOMAIN_DEVICE] = "no kernel serves this band";
            }
            o.bands.push_back(b);
        }
        P.ops.push_back(o);
        return (rad_op)(P.ops.size() - 1);
    }

    int start(rad::ResidencyTable* res = nullptr, bool profile = false) {
        CtxDesc d;
        d.program     = &P;
        d.rank        = 1;
        d.world_size  = 2;
        d.arch_step   = arch_step_trampoline;
        d.residency   = res;
        d.profile_ops = profile;
        return ctx.init(d);
    }

    int step(int n_tok, int step_no, const std::function<void(Ctx&)>& body) {
        batch.phase = n_tok > 1 ? RAD_PHASE_PREFILL : RAD_PHASE_DECODE;
        batch.step  = step_no;
        batch.n_tok = n_tok;
        batch.n_seq = 1;
        g_body = body;
        int s = ctx.run_step(&batch);
        g_body = nullptr;
        return s;
    }
};

/* ================================================================== tests */

/* The bucket boundaries are the constraint values themselves and the band is inclusive at its
 * upper bound (spec §2.2). Off by one here is a prefill kernel running at M=1, which is exactly
 * the loss the whole mechanism exists to recover -- and it is invisible in the output. */
TEST(bucket_table_indexes_at_band_boundaries) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 16, 64, 8192 },
                         { k_m16, k_m64, k_pre }, { "m16", "m64", "prefill" },
                         &g_gemm_schema, { w });
    CHECK_OK(f.start());

    const int64_t probes[] = { 1, 15, 16, 17, 63, 64, 65, 8191, 8192 };
    const char*   want[]   = { "m16", "m16", "m16", "m64", "m64", "m64",
                               "prefill", "prefill", "prefill" };

    g_rec.clear();
    CHECK_OK(f.step(1, 0, [&](Ctx& c) {
        for (int64_t n : probes) {
            RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, n));
        }
    }));

    CHECK_EQ(g_rec.size(), sizeof(probes) / sizeof(probes[0]));
    for (size_t i = 0; i < g_rec.size() && i < sizeof(want) / sizeof(want[0]); ++i) {
        CHECK_EQ(std::string(g_rec[i].kernel), std::string(want[i]));
        /* The geometry the kernel sees is the band's, collapsed to its upper bound. */
        CHECK_EQ(g_rec[i].M, (long long)(i < 3 ? 16 : i < 6 ? 64 : 8192));
    }

    /* RAD_N_BATCH means "the batch's token count", which is what almost every op wants. */
    g_rec.clear();
    CHECK_OK(f.step(40, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(std::string(g_rec[0].kernel), std::string("m64"));

    /* And the rank's identity reaches the kernel, because collectives need it (spec §9). */
    CHECK_EQ(g_rec[0].rank, 1);
    CHECK_EQ(g_rec[0].world, 2);
    CHECK_EQ(g_rec[0].scratch_bytes, (int64_t)128);
    CHECK(g_rec[0].scratch == f.ctx.scratch());
    CHECK(g_rec[0].stream == f.ctx.stream());
}

/* rad_buf_ptr is arena_base + BufferInfo::arena_offset, and an operand offset is element
 * arithmetic on top of it. Nothing is allocated and no table is walked. */
TEST(arena_offsets_and_operand_offsets) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    char* base = (char*)f.ctx.arena_base();
    CHECK(base != nullptr);
    CHECK((char*)f.ctx.buf_ptr(x) == base + 0);
    CHECK((char*)f.ctx.buf_ptr(y) == base + 1024);
    CHECK(f.ctx.weight_ptr(w) == (void*)g_weight_bytes);

    /* The scratch region is at the tail, past the plan, page-aligned. */
    CHECK((char*)f.ctx.scratch() >= base + 8192);
    CHECK(((uintptr_t)f.ctx.scratch() & (RAD_ALIGN_UNIT - 1)) == 0);

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        /* y offset by 8 elements of bf16 = 16 bytes -- one row of the [64, 8] plane.
         *
         * AND THE ROW COUNT COMES DOWN WITH IT, to 63. An offset does not shrink the extents,
         * so `RAD_B_AT(y, 8)` on its own describes 64 rows starting one row in, which reaches
         * eight elements past the buffer -- and the runtime refuses that by name. Offsetting
         * without narrowing is the shape of a real overrun (core/runtime/issue.cpp), so the
         * test that pins offset arithmetic must not be the one place it is spelled. */
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B_N(y, 8, 63) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(g_rec[0].n_t, 3);
    CHECK(g_rec[0].t[0].data == base + 0);
    CHECK(g_rec[0].t[1].data == (void*)g_weight_bytes);
    CHECK(g_rec[0].t[2].data == base + 1024 + 16);

    /* Shapes and strides come off the declaration; dtype off the weight's stored layout. */
    CHECK_EQ(g_rec[0].t[0].shape[0], (int64_t)64);
    CHECK_EQ(g_rec[0].t[0].shape[1], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[0], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[1], (int64_t)1);
    CHECK_EQ(g_rec[0].t[1].shape[0], (int64_t)8);
    CHECK_EQ(g_rec[0].t[1].shape[1], (int64_t)4);
    CHECK_EQ((int)g_rec[0].t[1].dtype, (int)RAD_BF16);
}

/* rows > 0 narrows dim 0 and leaves every stride alone. That is how a prefill chunk of `rows`
 * tokens issues against a buffer the plan sized at max_tok, and it is only sound on the outermost
 * dimension -- which is why the ABI narrows dim 0 and nothing else. */
TEST(rows_narrows_dim0_and_keeps_strides) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    CHECK_OK(f.step(7, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B_N(x, 0, 7), RAD_W(w), RAD_B_N(y, 0, 7) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(g_rec[0].t[0].shape[0], (int64_t)7);
    CHECK_EQ(g_rec[0].t[0].shape[1], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[0], (int64_t)4);   /* unchanged: it is a product of dims 1.. */
    CHECK_EQ(g_rec[0].t[0].stride[1], (int64_t)1);
    CHECK_EQ(g_rec[0].t[2].shape[0], (int64_t)7);
    CHECK_EQ(g_rec[0].t[2].shape[1], (int64_t)8);
    CHECK_EQ(g_rec[0].t[2].stride[0], (int64_t)8);
    /* The buffer itself is still the full max_tok extent; only this view is narrow. */
    CHECK_EQ(f.P.buffers[x].decl.shape[0], (int64_t)64);
}

/* A kernel returning negative at issue is a bug in selection, not a runtime condition: the step
 * aborts, the requests in it fail, the kernel and the geometry are NAMED, and nothing falls back
 * silently (spec §17). */
TEST(negative_launch_aborts_the_step_by_name) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op good = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    rad_op bad  = f.add_op("gemm_nt", "M", { 8192 }, { k_bad }, { "wrongshape" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    int second = RAD_OK;
    int s = f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        c.issue(bad, opd, 3, RAD_N_BATCH);
        /* rad_arch_step is void and cannot know the step is lost, so it keeps going. The rest of
         * the step must not launch: the output is already wrong and the first error would be
         * buried under fifty more. */
        second = c.issue(good, opd, 3, RAD_N_BATCH);
    });

    CHECK_EQ(s, RAD_E_SHAPE);
    CHECK_EQ(second, RAD_E_SHAPE);
    CHECK_EQ(g_rec.size(), (size_t)1);              /* only the failing kernel ever ran */
    CHECK_EQ(std::string(g_rec[0].kernel), std::string("bad"));

    const char* msg = f.ctx.step_error();
    CHECK(std::strstr(msg, "wrongshape") != nullptr);          /* the kernel, by name */
    CHECK(std::strstr(msg, "fake") != nullptr);                /* and the plugin it came from */
    CHECK(std::strstr(msg, "gemm_nt") != nullptr);             /* the op */
    CHECK(std::strstr(msg, "N=8") != nullptr);                 /* the geometry */
    CHECK(std::strstr(msg, "K=4") != nullptr);
    CHECK(std::strstr(msg, "M=8192") != nullptr);

    /* The next step starts clean -- the failure is per step, not per Ctx. */
    g_rec.clear();
    CHECK_OK(f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(good, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
}

/* A weight the mover has in flight is waited on with a STREAM wait, once per generation, and the
 * host never blocks. This is the property that lets placement be dynamic without serialising the
 * step (spec §5.4). */
TEST(pending_weight_waits_on_the_stream_not_the_host) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.expert3.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });

    FakeResidency res;
    res.which = w;
    RadEvent moving = nullptr;
    CHECK_OK(rad_event_create(&moving));
    res.slot.ptr        = g_moved_bytes;    /* the mover relocated it out from under WeightInfo */
    res.slot.ready      = moving;           /* and the transfer is still in flight */
    res.slot.generation = 1;

    CHECK_OK(f.start(&res));

    const long syncs0 = g_event_syncs, ssyncs0 = g_stream_syncs;

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));   /* same generation, second time */
    }));

    /* One wait, not two: a stream wait orders everything issued after it, so repeating it per
     * issue buys nothing and would be thousands of runtime calls a step. */
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)1);
    /* No host synchronisation of any kind. Ctx::host_syncs() is the structural check and holds
     * whichever backend is linked; the fake counters catch a sync that reached the device layer by
     * some route the runtime does not count. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);
    CHECK_EQ(g_event_syncs - syncs0, 0L);
    CHECK_EQ(g_stream_syncs - ssyncs0, 0L);
    /* The residency table, not WeightInfo, decided where the bytes are. */
    CHECK_EQ(g_rec.size(), (size_t)2);
    CHECK(g_rec[0].t[1].data == (void*)g_moved_bytes);
    CHECK(g_rec[0].t[1].data != (void*)g_weight_bytes);

    /* A new transfer into the slot bumps the generation, and that is what re-arms the wait. */
    res.slot.generation = 2;
    CHECK_OK(f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)2);

    /* A settled weight -- no event -- costs no wait at all, which is the common case even under
     * expert offload because a hot expert stays put for many steps. */
    res.slot.ready = nullptr;
    res.slot.generation = 3;
    CHECK_OK(f.step(4, 2, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)2);

    f.ctx.shutdown();
    rad_event_destroy(moving);
}

/* With no residency table -- the all-in-VRAM case, where the planner put everything on tier 0 and
 * the mover never runs -- the lookup collapses to WeightInfo::ptr and there is nothing to wait
 * on. The degenerate case must not pay for the interesting one. */
TEST(all_in_vram_collapses_to_weightinfo_ptr) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start(nullptr));

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)0);
    CHECK(g_rec[0].t[1].data == (void*)g_weight_bytes);

    /* The mover is what makes a relocation visible, and it is honoured between steps: a per-issue
     * atomic load of a cold WeightInfo line costs more than the relocation it guards against. */
    f.P.weights[w].ptr = g_moved_bytes;
    f.P.weights[w].generation.fetch_add(1);
    f.ctx.note_weights_moved();
    g_rec.clear();
    CHECK_OK(f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK(g_rec[0].t[1].data == (void*)g_moved_bytes);
}

/* A WEIGHT TABLE RESOLVES ONLY THE ENTRIES THAT MOVED. A routed layer's table is its whole expert
 * set and the mover changes a few experts on most steps, so resolving every entry whenever anything
 * moved is tens of thousands of lookups a step to change a handful of pointers. The residency table
 * names what it changed; the table resolves exactly those entries, patches the device array, and
 * leaves the rest alone. A template rebinding is the one event that re-resolves every entry. */
TEST(a_weight_table_resolves_only_the_entries_that_moved) {
    static const RadOperandSpec opds[] = {
        { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WTAB, 0 }, { "y", RAD_OPD_OUT, 0 },
    };
    static const RadOpSchema schema = {
        "moe_gemm", g_gemm_params, 4, opds, 3, "A[M,K] x B[e][N,K]^T -> Y[M,N]"
    };
    static uint16_t expert_bytes[4][32], moved_expert[32];
    Fixture f;
    std::vector<rad_weight> w;
    for (int e = 0; e < 4; ++e)
        w.push_back(f.add_weight(("blk.0.e" + std::to_string(e)).c_str(), RAD_BF16, 8, 4,
                                 expert_bytes[e]));
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op  op = f.add_op("moe_gemm", "M", { 8192 }, { k_pre }, { "prefill" }, &schema, w);

    TableResidency res;
    res.slot.assign(f.P.weights.size(), rad::WeightSlot{});
    for (int e = 0; e < 4; ++e) res.slot[(size_t)w[(size_t)e]].ptr = expert_bytes[e];
    CHECK_OK(f.start(&res));

    auto issue_twice = [&](int step_no) {
        g_rec.clear();
        CHECK_OK(f.step(4, step_no, [&](Ctx& c) {
            RadOperand opd[3] = { RAD_B(x), RAD_WTAB(w[0], 4), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        }));
    };
    /* What the kernel would read: the device table, copied back through the step's stream so the
     * check holds on the fakes and on a real backend alike. */
    auto entries = [&]() {
        std::vector<void*> v(4, nullptr);
        CHECK_OK(rad_memcpy_async(v.data(), g_rec[0].t[1].data, 4 * sizeof(void*), f.ctx.stream()));
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        return v;
    };

    issue_twice(0);
    CHECK_EQ(res.lookups, 4);                      /* the first issue walked, the second did not */
    std::vector<void*> v = entries();
    for (int e = 0; e < 4; ++e) CHECK(v[(size_t)e] == (void*)expert_bytes[e]);

    issue_twice(1);                                /* nothing moved: no walk */
    CHECK_EQ(res.lookups, 4);

    /* The mover relocates expert 2 and says so: one entry is resolved, not four. */
    res.move(w[2], moved_expert);
    issue_twice(2);
    CHECK_EQ(res.lookups, 5);
    v = entries();
    CHECK(v[2] == (void*)moved_expert);
    CHECK(v[1] == (void*)expert_bytes[1]);
    CHECK(v[3] == (void*)expert_bytes[3]);

    /* Moved and moved back before the table was next issued: resolved once more, and the device
     * array ends where the residency table says. */
    res.move(w[1], moved_expert);
    res.move(w[1], expert_bytes[1]);
    issue_twice(3);
    v = entries();
    CHECK(v[1] == (void*)expert_bytes[1]);
    CHECK(v[2] == (void*)moved_expert);

    /* A template refresh is the other half: an entry the residency table has no slot for reads
     * its template, so a rebinding resolves every entry. */
    const int before = res.lookups;
    f.ctx.note_weights_moved();
    issue_twice(4);
    CHECK_EQ(res.lookups, before + 4);
}

/* Routing never blocks the compute stream: the histogram is copied back asynchronously and the
 * heat engine DRAINS it on a later step (spec §5.5). */
TEST(route_report_is_asynchronous_and_drains_oldest_first) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    CHECK_OK(f.start());

    int32_t counts_a[4] = { 7, 0, 3, 1 };
    int32_t counts_b[4] = { 1, 1, 1, 9 };
    RadRouting r{};
    r.n_expert = 4;
    r.top_k = 2;

    const long syncs0 = g_event_syncs, ssyncs0 = g_stream_syncs;

    /* Nothing reported, so the drain has nothing to hand out -- and says so rather than handing
     * back a bank full of the zeros it was allocated with. */
    CHECK_EQ(f.ctx.next_histogram(0, -1, 0), -1);

    r.expert_count = counts_a;
    CHECK_OK(f.step(4, 0, [&](Ctx& c) { CHECK_OK(c.route_report(0, &r)); }));
    r.expert_count = counts_b;
    CHECK_OK(f.step(4, 1, [&](Ctx& c) { CHECK_OK(c.route_report(0, &r)); }));

    /* BOTH STEPS SURVIVE A CONSUMER THAT DID NOT LOOK BETWEEN THEM, and they come back oldest
     * first. This is the property the ring exists for: the consumer runs at the host's pace and
     * the histograms are produced at the card's, so between two visits there can be many, and one
     * that only ever returned the newest would discard the rest.
     *
     * A BANK IS IDENTIFIED BY THE STEP IT HOLDS, not by its index, because the index is an
     * implementation detail of a ring the caller never sees the depth of. */
    /* SPIN ON THE DRAIN, because the copy is GENUINELY asynchronous: what makes a bank readable
     * is its event, not the return of run_step, and a test that assumed otherwise would be
     * asserting the opposite of the contract this test exists to prove. */
    int b0 = -1;
    while ((b0 = f.ctx.next_histogram(0, -1, 2)) < 0) std::this_thread::yield();
    CHECK_EQ(f.ctx.last_histogram_step(0, b0), 0);
    const int32_t* h0 = f.ctx.last_histogram(0, b0);
    CHECK(h0 != nullptr);
    if (h0) { CHECK_EQ(h0[0], 7); CHECK_EQ(h0[2], 3); }

    int b1 = -1;
    while ((b1 = f.ctx.next_histogram(0, 0, 2)) < 0) std::this_thread::yield();
    CHECK(b1 != b0);
    CHECK_EQ(f.ctx.last_histogram_step(0, b1), 1);
    const int32_t* h1 = f.ctx.last_histogram(0, b1);
    CHECK(h1 != nullptr);
    if (h1) CHECK_EQ(h1[3], 9);

    /* Past the newest there is nothing, which is what ends a caller's drain loop. */
    CHECK_EQ(f.ctx.next_histogram(0, 1, 2), -1);

    /* Not one host synchronisation anywhere in that. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);
    CHECK_EQ(g_event_syncs - syncs0, 0L);
    CHECK_EQ(g_stream_syncs - ssyncs0, 0L);

    /* The routing buffers themselves are recorded for the mover, not read. */
    const RadRouting* rec = f.ctx.last_routing(0);
    CHECK(rec != nullptr);
    if (rec) CHECK_EQ(rec->n_expert, (int64_t)4);

    /* A layer the model does not have is refused rather than corrupting a neighbour's bank. */
    CHECK_EQ(f.ctx.route_report(99, &r), RAD_E_INVAL);
}

/* A step that drafts reports the same layer once per pass, all into the step's one bank. The drain
 * runs again before every pass, so it must not take that bank until the step is over: taken early
 * it stands for the passes so far, and the ones after it are never read. */
TEST(a_bank_is_credited_once_for_every_pass_of_its_step) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    CHECK_OK(f.start());

    int32_t counts[4] = { 2, 0, 5, 1 };
    RadRouting r{};
    r.n_expert = 4;
    r.top_k = 2;
    r.expert_count = counts;

    const auto report = [&](Ctx& c) { CHECK_OK(c.route_report(1, &r)); };
    const auto landed = [&](int after, int before) {
        /* The copy is asynchronous; wait for it the way the drain would see it land. */
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        return f.ctx.next_histogram(1, after, before);
    };

    CHECK_OK(f.step(4, 7, report));
    CHECK_EQ(landed(-1, 7), -1);           /* step 7 is still being issued */
    CHECK_OK(f.step(1, 7, report));
    CHECK_EQ(landed(-1, 7), -1);
    CHECK_OK(f.step(1, 7, report));

    const int b = landed(-1, 8);           /* the next step's dispatch */
    CHECK(b >= 0);
    CHECK_EQ(f.ctx.last_histogram_step(1, b), 7);
    CHECK_EQ(f.ctx.last_histogram_reports(1, b), 3);
    f.ctx.mark_credited(1, b);
    CHECK_EQ(landed(7, 8), -1);

    /* Three passes of one step are not two losses. A bank recycled for a later step before anyone
     * read it is one. */
    CHECK_EQ(f.ctx.route_dropped(), (uint64_t)0);
    CHECK_OK(f.step(1, 9, report));
    CHECK_OK(f.step(1, 9 + rad::kRouteBanks, report));
    CHECK_EQ(f.ctx.route_dropped(), (uint64_t)1);
}

/* Every event the context creates at bind is one Ctx::shutdown destroys: the routing ring holds
 * one a bank, not two. */
TEST(shutdown_destroys_every_event_the_context_created) {
    const int64_t before = rad::rad_dev_live_events();
    {
        Fixture f;
        f.add_buffer("x", RAD_BF16, 64, 4, 0);
        CHECK_OK(f.start());
        CHECK(rad::rad_dev_live_events() >= before + rad::kRouteBanks);
    }
    CHECK_EQ(rad::rad_dev_live_events(), before);
}

/* A weight operand is held to the same fit as a buffer one: a slice may name fewer rows than the
 * declaration, or start further in, and never reach past its end. */
TEST(a_weight_slice_that_reaches_past_the_weight_is_refused) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    RadOperand half = RAD_W(w);
    half.offset = 4 * 4;               /* rows 4..7 */
    half.rows   = 4;
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), half, RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));

    RadOperand past = half;
    past.offset = 5 * 4;               /* rows 5..8: one row past the declaration */
    const int s = f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), past, RAD_B(y) };
        c.issue(op, opd, 3, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_SHAPE);
    CHECK(std::strstr(f.ctx.step_error(), "reaches past weight 'blk.0.w'") != nullptr);
}

/* Two kernels that use the shared scratch region, one on each lane, with no join between them: the
 * second is ordered behind the first. Each launch below QUEUES its work on the stream it is given,
 * so an unordered pair really does overlap, and a reader that ran early would see the poison. */
static const int64_t kScrBytes = 4 << 20;
static uint8_t* g_scr_src = nullptr;
static uint8_t* g_scr_dst = nullptr;

static int k_scr_write(const RadArgs* a, RadStream s) {
    rad_memset_async(a->scratch, 0xEE, kScrBytes, s);
    for (int i = 0; i < 8; ++i) rad_memcpy_async(a->scratch, g_scr_src, kScrBytes, s);
    return record("scr_write", a, s);
}
static int k_scr_read(const RadArgs* a, RadStream s) {
    rad_memcpy_async(g_scr_dst, a->scratch, kScrBytes, s);
    return record("scr_read", a, s);
}

TEST(scratch_users_on_two_lanes_are_ordered_one_behind_the_other) {
    Fixture f;
    f.P.scratch_bytes = kScrBytes;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op wr = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_write }, { "scr_write" }, nullptr, {});
    rad_op rd = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_read }, { "scr_read" }, nullptr, {});
    f.P.ops[wr].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    f.P.ops[rd].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    CHECK_OK(f.start());

    g_scr_src = (uint8_t*)rad_dev_alloc(kScrBytes, RAD_MEM_DEVICE);
    g_scr_dst = (uint8_t*)rad_dev_alloc(kScrBytes, RAD_MEM_DEVICE);
    CHECK(g_scr_src && g_scr_dst);
    if (!g_scr_src || !g_scr_dst) return;

    for (int round = 0; round < 4; ++round) {
        const int first = round & 1;   /* which lane writes; the other reads */
        for (int64_t i = 0; i < kScrBytes; ++i) g_scr_src[i] = (uint8_t)(i * 7 + round);
        std::memset(g_scr_dst, 0, (size_t)kScrBytes);
        CHECK_OK(f.step(4, round, [&](Ctx& c) {
            RadOperand opd[1] = { RAD_B(x) };
            CHECK_OK(c.set_lane(first));
            CHECK_OK(c.issue(wr, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(first ^ 1));
            CHECK_OK(c.issue(rd, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(0));
        }));
        /* run_step ends by joining lane 1 into lane 0, so lane 0 drained is both drained. */
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        CHECK_EQ(std::memcmp(g_scr_dst, g_scr_src, (size_t)kScrBytes), 0);
    }

    rad_dev_free(g_scr_src, RAD_MEM_DEVICE);
    rad_dev_free(g_scr_dst, RAD_MEM_DEVICE);
    g_scr_src = g_scr_dst = nullptr;
}

/* A PROGRAM THAT DECLARES ITS SECOND LANE GETS A SCRATCH REGION PER LANE, so two scratch users on
 * two lanes are handed disjoint bytes and neither waits for the other. The pointer each launch
 * saw is the whole of the check: a region shared would put both at the same address. */
static void* g_scr_seen[2] = { nullptr, nullptr };
static int k_scr_seen0(const RadArgs* a, RadStream s) { g_scr_seen[0] = a->scratch; return record("s0", a, s); }
static int k_scr_seen1(const RadArgs* a, RadStream s) { g_scr_seen[1] = a->scratch; return record("s1", a, s); }

TEST(a_declared_second_lane_has_its_own_scratch_region) {
    Fixture f;
    f.P.scratch_bytes   = kScrBytes;
    f.P.scratch_regions = 2;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op a0 = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_seen0 }, { "s0" }, nullptr, {});
    rad_op a1 = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_seen1 }, { "s1" }, nullptr, {});
    f.P.ops[a0].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    f.P.ops[a1].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    CHECK_OK(f.start());

    for (int first = 0; first < 2; ++first) {
        g_scr_seen[0] = g_scr_seen[1] = nullptr;
        CHECK_OK(f.step(4, first, [&](Ctx& c) {
            RadOperand opd[1] = { RAD_B(x) };
            CHECK_OK(c.set_lane(first));
            CHECK_OK(c.issue(a0, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(first ^ 1));
            CHECK_OK(c.issue(a1, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(0));
        }));
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        REQUIRE(g_scr_seen[0] && g_scr_seen[1]);
        const int64_t gap = (const char*)g_scr_seen[1] - (const char*)g_scr_seen[0];
        CHECK((gap < 0 ? -gap : gap) >= kScrBytes);
        /* The same plan entry, issued from the other lane, follows the lane and not the entry. */
        CHECK_EQ((const char*)g_scr_seen[first] == (const char*)f.ctx.scratch(), true);
    }
}

/* THE ARENA PREDICTION COUNTS EVERY REGION, so the budget resolver charges the card for what
 * Arena::init goes on to allocate. */
TEST(the_predicted_arena_counts_every_scratch_region) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 16, 16, 0);
    f.P.arena_bytes     = 512;
    f.P.scratch_bytes   = 300;
    f.P.scratch_regions = 2;
    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE, f.P.scratch_regions));
    CHECK_EQ(a.bytes(), total);
    CHECK_EQ(a.scratch_regions(), 2);
    CHECK_EQ((char*)a.scratch(1) - (char*)a.scratch(0), rad::align_up(300, RAD_ALIGN_UNIT));
    CHECK((char*)a.scratch(1) + 300 <= a.base() + a.bytes());
    a.release();
}

/* Operands are positional (spec §2.3), so a miscount is not a diagnosable failure at the kernel --
 * it is silent numerical garbage. The runtime checks the arity it was given against the schema. */
TEST(operand_arity_is_checked_against_the_schema) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    int s = f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_W(w) };
        c.issue(op, opd, 2, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_SCHEMA);
    CHECK_EQ(g_rec.size(), (size_t)0);
    CHECK(std::strstr(f.ctx.step_error(), "schema declares 3") != nullptr);
}

/* A raw operand is a pointer the core did not allocate -- a RadBatch field. It has no BufferInfo,
 * so there is no dtype and therefore no element size; an offset on one is refused by name rather
 * than guessed at. */
TEST(raw_operand_carries_the_pointer_and_refuses_an_offset) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("rope", "M", { 8192 }, { k_pre }, { "prefill" }, nullptr, {});
    CHECK_OK(f.start());

    int32_t positions[4] = { 0, 1, 2, 3 };

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_P(positions) };
        opd[1].rows = 4;
        CHECK_OK(c.issue(op, opd, 2, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK(g_rec[0].t[1].data == (void*)positions);
    CHECK_EQ(g_rec[0].t[1].shape[0], (int64_t)4);

    g_rec.clear();
    int s = f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_P(positions) };
        opd[1].offset = 2;
        c.issue(op, opd, 2, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_INVAL);
    CHECK(std::strstr(f.ctx.step_error(), "no dtype") != nullptr);
}

/* Site is a property of the weights (spec §5.1). Asking for a domain that has no kernel in some
 * band is refused, naming the op and the band, because §5.1 says the planner was told at declare
 * which sites are unavailable -- so asking anyway is a planner bug, not something to paper over. */
TEST(execution_site_refuses_a_domain_with_no_kernel) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, nullptr, { w });
    CHECK_OK(f.start());

    CHECK_EQ(f.ctx.op_domain(op), RAD_DOMAIN_DEVICE);
    CHECK_EQ(f.ctx.set_op_domain(op, RAD_DOMAIN_HOST), RAD_E_NOKERNEL);
    CHECK_EQ(f.ctx.op_domain(op), RAD_DOMAIN_DEVICE);   /* and it did not move */
    CHECK_EQ(f.ctx.set_op_domain(op, RAD_DOMAIN_DEVICE), RAD_OK);
}

/* An op whose bucket table has a hole is a declare-time failure that surfaces here only if the
 * plugin issues into the hole. Bands that DO resolve still work, which is what makes the
 * fusion-probe pattern (spec §2.3) safe. */
TEST(a_band_with_a_hole_does_not_poison_its_neighbours) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("rmsnorm", "M", { 16, 8192 }, { nullptr, k_pre },
                         { nullptr, "prefill" }, nullptr, {});
    CHECK_OK(f.start());

    g_rec.clear();
    CHECK_OK(f.step(64, 0, [&](Ctx& c) {
        RadOperand opd[1] = { RAD_B(x) };
        CHECK_OK(c.issue(op, opd, 1, 64));   /* the resolved band */
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
}

/* Per-op timing is opt-in, and enabling it restores synchronisations that change the mover's slot
 * supply -- so a profiled run is not comparable against an unprofiled one (spec §16). The caveat
 * is printed when profiling is switched on, not buried in a doc, and this checks that the
 * synchronisation it warns about is real: an unprofiled step blocks the host nowhere, a profiled
 * one blocks it at the step boundary -- once, on the compute stream, however many ops it timed.
 * Not once per op: a host-wait event per op runs a large prefill step out of interrupt signals. */
TEST(profiling_is_opt_in_and_costs_a_synchronisation) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start(nullptr, /*profile=*/true));

    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);   /* nothing yet: the cost lands at the step boundary */
    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        for (int i = 0; i < 3; ++i) CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)3);
    /* Three timed launches, one host block at the end of the step. That is the cost, it is what
     * changes the mover's slot supply, and it is why anything compared across builds runs without
     * it. Compare with every other test in this file, where host_syncs() is zero. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)1);
    f.ctx.dump_profile();
}

/* The barrier is a condvar over threads in one process, not IPC: the ranks share an address space
 * and the batch is shared by pointer (spec §1). All it has to do is order the release. */
TEST(rank_barrier_releases_all_ranks) {
    const int n = 4;
    rad::RankBarrier b(n);
    std::atomic<int> before{0}, after{0};
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i)
        th.emplace_back([&] {
            before.fetch_add(1);
            b.arrive_and_wait();
            /* Everybody arrived before anybody left: that is the whole contract. */
            if (before.load() == n) after.fetch_add(1);
            b.arrive_and_wait();
        });
    for (auto& t : th) t.join();
    CHECK_EQ(after.load(), n);
}

/* Spec §3.2 accepts roughly 1-2 ms of launch overhead per decode step at 64 layers. That number
 * has to be defensible, so the core's share of it is measured rather than asserted. What is timed
 * here is everything the runtime does per issue -- band scan, tensor materialisation, RadArgs
 * patch, indirect call -- against a launch hook that returns immediately, so what is left is ours.
 */
TEST(per_issue_cost) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 16, 64, 8192 },
                         { k_null, k_null, k_null }, { "a", "b", "c" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    const int reps = 200000;
    double    ns   = 0.0;
    CHECK_OK(f.step(1, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        /* Warm. */
        for (int i = 0; i < 10000; ++i) c.issue(op, opd, 3, RAD_N_BATCH);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) c.issue(op, opd, 3, RAD_N_BATCH);
        auto t1 = std::chrono::steady_clock::now();
        ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
    }));

    fprintf(stderr,
            "       rad_issue: %.1f ns per call (3 operands, one weight, no residency table)\n"
            "       -> ~%.0f us of core time for a 1280-issue decode step (64 layers x 20 ops),\n"
            "          against spec 3.2's 1-2 ms budget, the rest of which is the driver's\n"
            "          per-launch dispatch cost and not ours to spend.\n",
            ns, ns * 1280 / 1000.0);

    /* Loose on purpose: this is a regression tripwire, not a benchmark. A hundredfold miss means
     * an allocation or a map lookup is on the hot path. */
    CHECK(ns < 2000.0);
}


/* ================================================================== the arena's size
 *
 * TWO CALLERS NEED THE SAME ANSWER AT TWO DIFFERENT TIMES: Ctx::bind_arena, which allocates the
 * arena at the end of startup, and the VRAM budget resolver, which has to subtract it from the
 * card BEFORE the pools are allocated near the top of startup. arena_plan_bytes is the one walk
 * they share, and the reason it exists is that two copies of it would agree today and drift the
 * first time the plan grows a domain -- with the failure being a budget that overcommits the card
 * by exactly the difference.
 *
 * So the property worth asserting is not the arithmetic on its own. It is that the number the
 * resolver is handed is the number Arena::init goes on to allocate.
 */

TEST(the_predicted_arena_size_is_the_one_the_arena_allocates) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 16, 16, 0);        /* 512 B at offset 0 */
    f.add_buffer("y", RAD_BF16, 16, 16, 512);      /* 512 B after it */
    f.P.arena_bytes   = 1024;
    f.P.scratch_bytes = 256;

    int64_t dev = 0, total = 0;
    const int64_t got = rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(got, total);

    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE));
    CHECK(a.live());
    /* THE PREDICTION IS EXACT, not an upper bound: a resolver told "at least this much" cannot
     * hand the remainder to a pool. */
    CHECK_EQ(a.bytes(), total);
    CHECK_EQ(a.scratch_bytes(), 256ll);
    /* The scratch region starts on a unit boundary past the plan, so a kernel wanting an aligned
     * workspace gets one and the buffers below it are not overlapped. */
    CHECK_EQ((char*)a.scratch() - a.base(), rad::align_up(dev, RAD_ALIGN_UNIT));
    a.release();
    CHECK(!a.live());
    CHECK(a.base() == nullptr);
}

/* THE HIGH-WATER MARK IS OVER OFFSET + SIZE AND NOT OVER SIZES. The buffer plan reuses offsets
 * for buffers whose lifetimes do not overlap, so summing the sizes would reserve an arena several
 * times larger than the plan needs -- which on this engine comes out of the weight slab. */
TEST(overlapping_buffer_lifetimes_do_not_each_buy_their_own_bytes) {
    Fixture f;
    f.add_buffer("a", RAD_BF16, 64, 64, 0);        /* 8 KiB */
    f.add_buffer("b", RAD_BF16, 64, 64, 0);        /* the same 8 KiB, reused */
    f.add_buffer("c", RAD_BF16, 64, 64, 8192);     /* live alongside them */
    f.P.arena_bytes   = 16384;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 16384ll);                        /* not 24576 */
    CHECK_EQ(total, rad::align_up(16384ll, RAD_ALIGN_UNIT));
}

/* A BUFFER THE PLAN NEVER PLACED IS NOT CHARGED FOR. bind_arena refuses such a program by name,
 * and until it does, a negative offset must not be folded into a size. */
TEST(an_unplaced_buffer_is_skipped_rather_than_counted) {
    Fixture f;
    f.add_buffer("placed", RAD_BF16, 16, 16, 0);
    rad::BufferInfo unplaced;
    unplaced.name = "unplaced";
    unplaced.decl.dtype = RAD_BF16;
    unplaced.decl.rank = 2;
    unplaced.decl.shape[0] = 1024;
    unplaced.decl.shape[1] = 1024;
    unplaced.bytes = rad_dtype_bytes(RAD_BF16, 1024 * 1024);
    unplaced.arena_offset = -1;
    f.P.buffers.push_back(unplaced);
    f.P.arena_bytes   = 512;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 512ll);
    CHECK(total < 1024 * 1024);
}

/* A HOST-DOMAIN BUFFER LIVES IN THE OTHER ARENA, so its extent must not raise the device one:
 * the resolver subtracts the device figure from the CARD, and charging host bytes against VRAM
 * takes memory away from the weights that nothing will ever put there. */
TEST(a_host_domain_buffer_does_not_raise_the_device_arena) {
    Fixture f;
    f.add_buffer("dev", RAD_BF16, 16, 16, 0);      /* 512 B */
    rad::BufferInfo h;
    h.name = "host";
    h.decl.dtype = RAD_BF16;
    h.decl.rank = 2;
    h.decl.shape[0] = 256;
    h.decl.shape[1] = 256;
    h.decl.domain = RAD_DOMAIN_HOST;
    h.bytes = rad_dtype_bytes(RAD_BF16, 256 * 256);
    h.arena_offset = 0;
    f.P.buffers.push_back(h);
    f.P.arena_bytes   = 99999;                     /* the plan's own total covers BOTH domains */
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    /* With a host buffer present the device figure is the device high-water mark and not the
       plan's combined total, which is the whole reason the two are tracked apart. */
    CHECK_EQ(dev, 512ll);
    CHECK_EQ(total, rad::align_up(512ll, RAD_ALIGN_UNIT));
}

/* A ZERO-BYTE ARENA WOULD HAND EVERY BUFFER A NULL BASE and turn a plan bug into a segfault with
 * no name on it. One page instead, so a stray offset lands somewhere reportable -- and the
 * prediction has to know about that floor or it under-reports by a page. */
TEST(an_empty_program_still_buys_one_page) {
    Fixture f;
    f.P.arena_bytes   = 0;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 0ll);
    CHECK_EQ(total, (int64_t)RAD_ALIGN_UNIT);

    rad::Arena a;
    CHECK_OK(a.init(0, 0, RAD_MEM_DEVICE));
    CHECK(a.base() != nullptr);
    CHECK_EQ(a.bytes(), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(a.scratch_bytes(), 0ll);
}

TEST(the_arena_refuses_a_negative_request_rather_than_allocating_one) {
    rad::Arena a;
    CHECK_EQ(a.init(-1, 0, RAD_MEM_DEVICE), RAD_E_INVAL);
    CHECK(!a.live());
    CHECK_EQ(a.init(0, -1, RAD_MEM_DEVICE), RAD_E_INVAL);
    CHECK(!a.live());
}

/* SCRATCH IS PADDED ON ITS OWN, because it starts on a unit boundary: a plan and a scratch that
 * were summed before rounding would under-reserve by up to one unit, and the kernel that reads
 * past it is the largest one in the program. */
TEST(scratch_is_rounded_separately_from_the_plan) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 1, 1, 0);
    f.P.arena_bytes   = 1;                         /* both deliberately unaligned */
    f.P.scratch_bytes = 1;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(total, 2 * (int64_t)RAD_ALIGN_UNIT);
    CHECK(total > rad::align_up(2ll, RAD_ALIGN_UNIT));

    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE));
    CHECK_EQ(a.bytes(), total);
}

/* The out parameters are optional, and the return is the total either way. */
TEST(arena_plan_bytes_returns_the_total_with_or_without_its_out_parameters) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 8, 8, 0);
    f.P.arena_bytes   = 128;
    f.P.scratch_bytes = 64;
    int64_t dev = 0, total = 0;
    const int64_t a = rad::arena_plan_bytes(f.P, &dev, &total);
    const int64_t b = rad::arena_plan_bytes(f.P, nullptr, nullptr);
    CHECK_EQ(a, b);
    CHECK_EQ(a, total);
}

RAD_TEST_MAIN()
