/* ref_collective.cpp -- all_reduce and all_gather on the host backend.
 *
 * At world_size 1 both are the degenerate case: all_reduce is inout and already holds the answer,
 * all_gather is a copy. Above 1 they are real, and they work because of how this engine is shaped:
 * ONE PROCESS, ONE THREAD PER RANK, ONE ADDRESS SPACE (spec §1). There is no IPC to set up and no
 * serialisation to do -- a peer's buffer is a pointer -- so the whole collective is a rendezvous, a
 * read of every rank's pointer, and a second rendezvous before anyone overwrites theirs.
 *
 * That makes the host backend a FAKE MULTI-DEVICE BACKEND, which is the point: tensor-parallel code
 * paths -- the sharded declare, the collective placement, the distributed top-k -- can be exercised
 * on a machine with no accelerator at all. What it does not test is a device all-reduce's numerics,
 * and it does not pretend to.
 *
 * REDUCTION ORDER IS ASCENDING RANK, in f32, on every rank. That is libr4d's rule ("fp32
 * accumulate in ascending rank order, bit-identical across ranks") and it is a correctness
 * requirement rather than a convention: tensor-parallel ranks hold replicated state that must stay
 * bit-identical, and two ranks that summed in different orders would drift apart over a
 * generation. The `-ffp-contract=off` this plugin builds with is part of the same rule
 * (docs/OPS.md).
 *
 * `exact` is honoured in one direction only. exact=1 gets the exact sum. exact=0 -- a caller that
 * has declared it will accept a lossy quantised wire -- also gets the exact sum, because there is
 * no wire here to quantise. Serving a lossy request with an exact answer is safe; the constraint
 * exists so a caller who wanted exactness cannot be handed a lossy result by accident, not the
 * reverse. `hops` is accepted and ignored: one-shot and two-shot are the same walk over an array
 * when the peers are pointers.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>

using namespace ref;

namespace {

enum { REF_MAX_RANKS = 8 };

/* One group per process. There is exactly one world size in a radiance process and the rank
 * threads run the same declared graph in lockstep against one batch, so the sequence of collective
 * calls is identical on every rank and a single sense-reversing barrier is enough to pair them up.
 * A per-op group would need a key the ABI does not carry -- init() is called once per rank per
 * resolved instance and cannot tell two ops apart from two ranks of one op. */
struct Group {
    std::mutex              m;
    std::condition_variable cv;
    int          world   = 0;
    int          arrived = 0;
    unsigned     gen     = 0;
    const void*  slot[REF_MAX_RANKS] = {};
    int64_t      nel [REF_MAX_RANKS] = {};
    uint32_t     dt  [REF_MAX_RANKS] = {};
};

static Group& group() { static Group g; return g; }

/* Returns false on timeout. A collective IS a synchronisation, so the no-blocking rule cannot
 * apply to it -- but an unbounded wait would turn a mismatched call sequence into a wedged test
 * run with no output, and this engine exits non-zero rather than serving from a wedged queue
 * (spec §17). Thirty seconds is far longer than any correct step and far shorter than a CI
 * timeout. */
static bool barrier(Group& g) {
    std::unique_lock<std::mutex> lk(g.m);
    const unsigned mygen = g.gen;
    if (++g.arrived == g.world) {
        g.arrived = 0;
        ++g.gen;
        g.cv.notify_all();
        return true;
    }
    return g.cv.wait_for(lk, std::chrono::seconds(30), [&] { return g.gen != mygen; });
}

static int join(int rank, int world) {
    if (world <= 0 || world > REF_MAX_RANKS) return RAD_E_UNSUPPORTED;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    Group& g = group();
    std::lock_guard<std::mutex> lk(g.m);
    if (g.world == 0) g.world = world;
    else if (g.world != world) return RAD_E_INVAL;
    return RAD_OK;
}

/* Staging, so a rank does not overwrite its own buffer while a peer is still reading it. The
 * scratch the arena hands us is used when it is big enough -- that is what the scratch hook exists
 * for -- and the per-thread workspace covers a caller that passed none (a test, or a tool calling
 * the kernel directly). */
static float* staging(const RadArgs* a, int64_t nel) {
    if (a->scratch && a->scratch_bytes >= nel * (int64_t)sizeof(float))
        return (float*)a->scratch;
    return ws_get((size_t)nel);
}

}  /* namespace */

int ref_collective_init(const RadParam* p, int n_p, int rank, int world_size,
                        void** out) {
    /* The world size in the geometry wins: declare states it, and RadBuildCtx already divided the
     * model by it (spec §9). */
    int ws = world_size;
    for (int i = 0; i < n_p; ++i)
        if (p[i].key && std::strcmp(p[i].key, "world_size") == 0 && p[i].kind == RAD_P_INT)
            ws = (int)p[i].ival;
    if (out) *out = nullptr;
    if (ws <= 1) return RAD_OK;
    return join(rank, ws);
}

void ref_collective_fini(void*) {}

int64_t ref_all_reduce_scratch(const RadArgs* a) {
    const int64_t ws = p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (ws <= 1) return 0;
    const int64_t nel = p_int(a, "numel", a->n_t > 0 ? numel(&a->t[0]) : 0);
    return nel * (int64_t)sizeof(float);
}

/* ------------------------------------------------------------------ all_reduce */
/* `y` is optional and absent means IN PLACE, which is the form docs/OPS.md draws and the only one
 * the one-shot kernels need. libr4d's two-shot rows are not safe in place -- a rank gathers into
 * the output while its own input is still being read for the scatter -- so the destination has to
 * be expressible, and here it costs one branch on the store. */
int ref_all_reduce(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* y = t_in(a, 1);
    /* The message is the operand; `numel` is the band. See band_rows in ref_common.h. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    if (nel <= 0) return RAD_E_SHAPE;
    if (y && numel(y) < nel) return RAD_E_SHAPE;
    const RadTensor* dst = y ? y : x;

    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (world <= 1) {
        /* inout: the answer is already there. With a separate destination it still has to be
         * COPIED there, because a caller that passed one reads it and not x. */
        if (y) for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
        return RAD_OK;
    }
    const int rank = a->rank;
    RAD_REF_TRY(join(rank, world));

    Group& g = group();
    {
        std::lock_guard<std::mutex> lk(g.m);
        g.slot[rank] = x->data;
        g.nel [rank] = nel;
        g.dt  [rank] = x->dtype;
    }
    if (!barrier(g)) return RAD_E_DEVICE;

    for (int r = 0; r < world; ++r)
        if (!g.slot[r] || g.nel[r] != nel || g.dt[r] != x->dtype) return RAD_E_SHAPE;

    float* acc = staging(a, nel);
    for (int64_t i = 0; i < nel; ++i) {
        float s = 0.0f;
        for (int r = 0; r < world; ++r) s += ld_dt(g.dt[r], g.slot[r], i);
        acc[i] = s;
    }

    /* Everyone has finished READING before anyone WRITES. Without this the rank that finishes
     * first would overwrite an input a slower peer has not consumed yet, and the bug would show up
     * as a rank-dependent answer under load and never in a small test. */
    if (!barrier(g)) return RAD_E_DEVICE;
    for (int64_t i = 0; i < nel; ++i) stt(dst, offlin(dst, i), acc[i]);
    return RAD_OK;
}

/* ------------------------------------------------------------------ all_gather */
/* y is [world_size * numel] with rank r's contribution at offset r * numel.
 *
 * A GATHER MOVES BYTES, NOT NUMBERS, and here the difference is not academic. The vocab-parallel
 * candidate merge gathers (id, score) pairs as an opaque u32 payload -- the ids are i32 and the
 * scores are f32 bit patterns, in one buffer. Carried through float, an id of -1 (0xFFFFFFFF, the
 * empty-candidate marker) becomes 4294967296.0f, which is one past the largest u32, and storing
 * that back is undefined; it lands as zero. A float-carrying reference therefore disagrees with a
 * correct kernel on exactly the padding of every gathered candidate list, and rounds every id
 * above 2^24 as well. Copying the stored representation is also simply what the operation IS: no
 * arithmetic happens here. */
int ref_all_gather(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    /* The contribution is the operand; `numel` is the band. See band_rows in ref_common.h. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (nel <= 0 || numel(y) < nel * world) return RAD_E_SHAPE;
    /* One element has to have an address of its own for a byte copy to mean anything, which rules
     * out the sub-byte codes -- and a gather of packed weights is not a thing any caller asks
     * for, so refusing says more than a silent unpack would. */
    const int64_t esz = rad_dtype_bytes(x->dtype, 1);
    if (esz <= 0 || y->dtype != x->dtype) return RAD_E_DTYPE;

    if (world <= 1) {
        for (int64_t i = 0; i < nel; ++i)
            std::memcpy((char*)y->data + offlin(y, i) * esz,
                        (const char*)x->data + offlin(x, i) * esz, (size_t)esz);
        return RAD_OK;
    }

    const int rank = a->rank;
    RAD_REF_TRY(join(rank, world));
    Group& g = group();
    {
        std::lock_guard<std::mutex> lk(g.m);
        g.slot[rank] = x->data;
        g.nel [rank] = nel;
        g.dt  [rank] = x->dtype;
    }
    if (!barrier(g)) return RAD_E_DEVICE;

    for (int r = 0; r < world; ++r) {
        if (!g.slot[r] || g.nel[r] != nel) return RAD_E_SHAPE;
        if (g.dt[r] != x->dtype) return RAD_E_DTYPE;
    }
    /* WHERE A RANK'S ELEMENTS LAND. `row` absent (or the whole message) is the plain
     * concatenation; present, the placement is row-interleaved -- x is [rows][row] and y is
     * [rows][world][row] -- which is the form the vocab-parallel candidate merge reads, because it
     * takes one row of gathered candidates per sampled position at a single row stride. */
    const int64_t row = p_int(a, "row", 0);
    if (row < 0 || (row > 0 && nel % row)) return RAD_E_SHAPE;
    for (int r = 0; r < world; ++r)
        for (int64_t i = 0; i < nel; ++i) {
            const int64_t o = (row > 0 && row < nel)
                                  ? (i / row) * (world * row) + (int64_t)r * row + (i % row)
                                  : (int64_t)r * nel + i;
            std::memcpy((char*)y->data + offlin(y, o) * esz,
                        (const char*)g.slot[r] + i * esz, (size_t)esz);
        }

    /* y is a different buffer from x, so no second barrier is needed for correctness -- but the
     * ranks must not race ahead into the NEXT collective while a peer still reads this one's
     * input, and the pairing of barriers is what keeps the call sequences aligned. */
    if (!barrier(g)) return RAD_E_DEVICE;
    return RAD_OK;
}
