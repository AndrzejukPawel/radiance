// r4d_emit.h -- one launch, two destinations: the stream, or a description of it.
//
// ============================== WHY A SINK ==============================
//
// ABI v10 lets a kernel DESCRIBE its dispatch (`RadDescribeFn`) so the runtime can replay a
// repeatedly issued sequence as a graph instead of submitting it node by node. That gives every
// shim two entry points, and the typedef's comment names the trap they create: `launch` and
// `describe` must dispatch the SAME kernel with the SAME arguments, nothing can check that, and a
// divergence is not a crash -- it is a wrong number produced only after a capture, which reads as
// a numerics bug anywhere but here.
//
// The structural answer is to make the two paths unable to disagree: ONE body computes the
// geometry and emits into a sink, and the sink is either a stream or a descriptor array. Neither
// path can drift, because there is only one of them.
//
// ============================== THE COPY HAZARD ==============================
//
// `RadLaunchDesc::params[i]` points INTO `RadLaunchDesc::blob`. Pack into a local and copy the
// struct into place and every pointer still aims at the dead local -- garbage, and only on the
// graph path. So `r4d_emit` packs DIRECTLY into the destination and the launch path uses a
// descriptor of its own rather than a separate argument list. That is also why the launch path
// goes through `hipLaunchKernel` (which takes kernelParams) rather than the triple-chevron: the
// two paths then share not just the arithmetic but the marshalling.

#ifndef R4D_EMIT_H
#define R4D_EMIT_H

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tuple>

#include "rad_abi.h"
#include "r4d_args.h"

/* Where a described launch goes. `desc` null means PERFORM the dispatch; otherwise append to
 * `desc[0..cap)` and count it in `*n`.
 *
 * `stream` is set either way, and is not redundant when describing: a shim whose launch depends on
 * which stream it is going onto -- the split-K GEMM takes its arrival counter from a per-stream
 * pool -- must make the same choice in both modes, or the description is of a launch that never
 * happens. */
struct R4dSink;
static inline R4dSink r4d_stream_sink(RadStream s);

struct R4dSink {
    RadStream      stream = nullptr;
    RadLaunchDesc* desc   = nullptr;
    int            cap    = 0;
    int*           n      = nullptr;
};

/* A sink that only launches, for a caller that has a stream and no descriptor. */
static inline R4dSink r4d_stream_sink(RadStream s) { R4dSink k; k.stream = s; return k; }

/* Pack one argument into `d` at the next aligned offset and point `params[i]` at it. */
template <class T>
inline bool r4d_pack_one(RadLaunchDesc& d, size_t& off, int& i, const T& v) {
    static_assert(__is_trivially_copyable(T), "a kernel argument must be trivially copyable");
    off = (off + alignof(T) - 1) & ~(size_t)(alignof(T) - 1);
    if (i >= RAD_MAX_KPARAMS || off + sizeof(T) > RAD_KPARAM_BLOB) return false;
    memcpy(d.blob + off, &v, sizeof(T));
    d.param_size[i] = (uint16_t)sizeof(T);
    d.params[i++]   = d.blob + off;
    off += sizeof(T);
    return true;
}

/* THE ONE EMIT, AND IT TAKES THE KERNEL TYPED.
 *
 * `hipLaunchKernelGGL` type-checks its arguments against the kernel's signature and CONVERTS them
 * -- pass a `long long` where the kernel declares `uint32_t` and the compiler narrows it at the
 * call. `hipLaunchKernel` takes `kernelParams`, which is raw bytes: the same call would copy eight
 * bytes into a slot the kernel reads as four and corrupt every argument after it, silently, with
 * no diagnostic anywhere. Over a library's worth of shims that is a defect generator.
 *
 * So the kernel arrives as `void (*)(P...)`, which deduces its parameter types, and every argument
 * is packed as the P the kernel actually declares. The arity is checked too. What the chevron
 * launch does for free is done explicitly here, and a mismatch is a compile error rather than a
 * wrong number.
 *
 * `func` is the kernel's host stub -- for a template instantiation, its own symbol, which is the
 * point: an instantiation chosen by a parameter moves the FUNCTION and not merely an argument, and
 * the runtime has to rebuild rather than update such a node. */
template <class... P, class... A>
inline int r4d_emit(const R4dSink& sk, void (*kernel)(P...), dim3 grid, dim3 block,
                    uint32_t shared_bytes, const A&... args) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    const void* func = (const void*)kernel;
    RadLaunchDesc  local;
    RadLaunchDesc* dst = &local;
    if (sk.desc) {
        if (!sk.n || *sk.n >= sk.cap) return RAD_E_FULL;
        dst = &sk.desc[*sk.n];
    }
    RadLaunchDesc& d = *dst;
    d.func = func;
    d.grid[0] = grid.x;  d.grid[1] = grid.y;  d.grid[2] = grid.z;
    d.block[0] = block.x; d.block[1] = block.y; d.block[2] = block.z;
    d.shared_bytes = shared_bytes;
    d.n_params = 0;

    size_t off = 0;
    int    i   = 0;
    bool   ok  = true;
    /* Converted to the kernel's declared type FIRST, then packed -- see the note above. Left to
     * right, so slot i is parameter i. */
    std::tuple<P...> vals(static_cast<P>(args)...);
    std::apply([&](const P&... v) { ((ok = ok && r4d_pack_one(d, off, i, v)), ...); }, vals);
    if (!ok) return RAD_E_FULL;
    d.n_params = (uint32_t)i;

    if (sk.desc) { ++*sk.n; return RAD_OK; }
    /* THE LAST THING SUBMITTED, NAMED BEFORE IT IS SUBMITTED. A launch that faults inside the
     * driver leaves a backtrace through this frame and nothing else: the geometry is in registers
     * that -O3 has reused and the kernel is a stub address. Printing it first costs a line per
     * dispatch under a flag nobody sets in production, and it is the difference between a shape
     * to reproduce and a stack of question marks. Flushed, because the fault is the next
     * instruction. */
    /* THE LAST THING SUBMITTED, NAMED BEFORE IT IS SUBMITTED. A launch that faults inside the
     * driver leaves a backtrace through this frame and nothing else: the geometry is in registers
     * that -O3 has reused and the kernel is a stub address. Printed first, so an `emit` with no
     * `ret` after it IS the faulting dispatch.
     *
     * ONE WRITE A LINE, BECAUSE EVERY RANK IS A THREAD. A line assembled by a sequence of
     * fprintf calls interleaves with the other rank's, and the trace then shows dispatches that
     * never returned next to returns that belong to nothing -- which is worse than no trace,
     * because it reads as evidence. */
    if (r4d_trace_on()) {
        char line[1024];
        int  o = snprintf(line, sizeof line,
                          "[r4d] emit %p grid=%ux%ux%u block=%ux%ux%u lds=%u np=%u",
                          func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          shared_bytes, d.n_params);
        for (uint32_t k = 0; k < d.n_params && o > 0 && o < (int)sizeof line - 32; ++k) {
            unsigned long long v = 0;
            memcpy(&v, d.params[k], d.param_size[k] > 8 ? 8 : d.param_size[k]);
            o += snprintf(line + o, sizeof line - (size_t)o, " %u:%llx", d.param_size[k], v);
        }
        std::fprintf(stderr, "%s\n", line);
        std::fflush(stderr);
    }
    const int rc = hipLaunchKernel(func, grid, block, d.params, shared_bytes,
                                   (hipStream_t)sk.stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    if (r4d_trace_on()) { std::fprintf(stderr, "[r4d] ret %p %d\n", func, rc); std::fflush(stderr); }
    return rc;
}

/* `kernel` is named, not passed as a value, so a template instantiation is spelled at the call
 * site exactly as the triple-chevron spelled it. */
#define R4D_EMIT(sink, kernel, grid, block, lds, ...) \
    r4d_emit((sink), (kernel), (grid), (block), (uint32_t)(lds), __VA_ARGS__)

/* The two entry points, from one body. `body` is `int (*)(const RadArgs*, const R4dSink&)`. */
#define R4D_ENTRY(sym, body)                                                          \
    extern "C" int sym(const RadArgs* a, RadStream s) {                               \
        R4dSink sk; sk.stream = s;                                                    \
        return body(a, sk);                                                           \
    }                                                                                 \
    extern "C" int sym##_describe(const RadArgs* a, RadStream s, RadLaunchDesc* d,         \
                                  int cap, int* n) {                                  \
        if (!n) return RAD_E_INVAL;                                                   \
        *n = 0;                                                                       \
        R4dSink sk; sk.stream = s; sk.desc = d; sk.cap = cap; sk.n = n;               \
        return body(a, sk);                                                           \
    }

#endif  /* R4D_EMIT_H */
