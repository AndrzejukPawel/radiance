/* avx_dispatch.cpp -- the one translation unit that may not contain a vector instruction.
 *
 * ============================== THE RULE THIS FILE EXISTS TO ENFORCE ==============================
 *
 * Everything else in this plugin is compiled with -mavx2 or -mavx512f and therefore may not be
 * EXECUTED until something has established that the machine can run it. This file is what
 * establishes it, so it is compiled with no -m flags at all, and CMakeLists.txt states that as a
 * property of the target rather than leaving it to habit.
 *
 * The failure mode if this is got wrong is worth naming because it is not a wrong answer, it is a
 * SIGILL in the fallback that exists to prevent SIGILLs, and it happens before main, on whichever
 * machine the build was not tested on. A compiler is free to auto-vectorise anything it likes in a file
 * compiled with -mavx512f -- including the table-of-function-pointers copy below, which is exactly
 * the shape a vectoriser likes -- so "this function contains no intrinsics" is not the same claim
 * as "this function contains no AVX-512", and only the flags on the object give the second one.
 *
 * ============================== WHAT IS DETECTED, AND WHY __builtin_cpu_supports ==============================
 *
 * Raw CPUID is not enough and the reason is the OS, not the CPU. AVX and AVX-512 add architectural
 * state (YMM, then ZMM and the mask registers) that the kernel must agree to save across a context
 * switch; it signals that through XCR0, read with XGETBV. A CPU may enumerate AVX-512 in CPUID
 * while the OS has left the ZMM state disabled -- a VM with a restricted feature mask, an older
 * kernel, a hypervisor hiding it -- and using AVX-512 then faults on the first instruction.
 * __builtin_cpu_supports does the XGETBV check; a hand-rolled CPUID dispatch usually does not, and
 * that omission is the single most common way this kind of code is wrong.
 *
 * AVX-512 IS ALSO NOT ONE FEATURE. The check below is the conjunction of F, BW, DQ and VL because
 * the kernels use all four: BW for the byte and word shuffles the bf16 pack needs, DQ for the
 * 64-bit integer forms, VL for the 128/256-bit encodings of the AVX-512 instructions. A Knights
 * Landing has F and not the others, and a kernel that tested F alone would fault on it.
 */
#include "avx_isa.h"
#include "avx_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

/* ================================================================== the fp8 decode tables
 *
 * Built at load from the ABI's OWN converters, not typed out. A typed-out table is a second
 * definition of the format and can drift from rad_fp8e4m3_to_f32, which is exactly what
 * rad_plugin.h's header argues against at length. 256 calls at startup cost nothing and cannot
 * disagree. */
extern "C" { float avx_fp8e4m3_tab[256] = { 0 };
float avx_fp8e5m2_tab[256] = { 0 }; }

static void fp8_tables_init(void) {
    for (int i = 0; i < 256; ++i) {
        avx_fp8e4m3_tab[i] = rad_fp8e4m3_to_f32((uint8_t)i);
        avx_fp8e5m2_tab[i] = rad_fp8e5m2_to_f32((uint8_t)i);
    }
}

/* ================================================================== detection */
static int g_detected = -1;
static int g_level    = -1;

extern "C" int avx_isa_detect(void) {
    if (g_detected >= 0) return g_detected;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"))
        g_detected = AVX_LEVEL_AVX512;
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
             __builtin_cpu_supports("f16c"))
        g_detected = AVX_LEVEL_AVX2;
    else if (__builtin_cpu_supports("avx"))
        g_detected = AVX_LEVEL_AVX;
    else
        g_detected = AVX_LEVEL_SCALAR;
#else
    /* Not x86. The plugin still loads and still computes the right answers -- that is the whole
     * point of the scalar level being a real implementation rather than a stub -- it is simply
     * slow, and `rad_plugin_info().build_target` says "host" so nothing is being claimed. */
    g_detected = AVX_LEVEL_SCALAR;
#endif
    return g_detected;
}

extern "C" const char* avx_isa_name(int level) {
    switch (level) {
        case AVX_LEVEL_SCALAR: return "scalar";
        case AVX_LEVEL_AVX:    return "avx";
        case AVX_LEVEL_AVX2:   return "avx2";
        case AVX_LEVEL_AVX512: return "avx512";
        default:               return "?";
    }
}

extern "C" const AvxKernelTable* avx_table_for(int level) {
    switch (level) {
        case AVX_LEVEL_AVX512: return &avx_table_v3;
        case AVX_LEVEL_AVX2:   return &avx_table_v2;
        case AVX_LEVEL_AVX:    return &avx_table_v1;
        default:               return &avx_table_sc;
    }
}

static const AvxKernelTable* g_tab = 0;

extern "C" const AvxKernelTable* avx_table_live(void) {
    if (!g_tab) { g_level = avx_isa_detect(); g_tab = avx_table_for(g_level); }
    return g_tab;
}

extern "C" int avx_isa_level(void) {
    if (g_level < 0) (void)avx_table_live();
    return g_level;
}

extern "C" int avx_isa_force(int level) {
    const int cap = avx_isa_detect();
    /* CLAMPED, NOT HONOURED. Asking for a level the machine cannot execute is a caller error, and
     * the only honest response is to run what the machine has -- returning the level actually
     * selected so a checker can report "asked for avx512, ran avx2" rather than believing it
     * tested something it did not. Honouring it would be a SIGILL. */
    if (level < 0) level = 0;
    if (level > cap) level = cap;
    g_level = level;
    g_tab   = avx_table_for(level);
    return level;
}

/* ================================================================== the thunks
 *
 * One baseline-compiled function per op, pointed at by the registry row, doing one indirect call
 * into the selected level's table.
 *
 * WHY NOT REPOINT THE ROWS THEMSELVES. RadKernelInfo is returned by rad_kernel_at as a
 * `const RadKernelInfo*` out of a static table, and the loader is entitled to read it before
 * rad_plugin_open has run (it reads the op names to check schema agreement). A row whose `launch`
 * was null until open() had patched it is a row that is briefly wrong, and a row that is briefly
 * wrong during startup is read as a row that was never wired at all. A
 * thunk is always correct, needs no mutation of a table the ABI hands out as const, and costs one
 * predicted indirect branch per launch -- which against a kernel whose cheapest form touches a
 * 4096-element row is not measurable. */
#define AVX_THUNK(nm)                                                                             \
    extern "C" int avx_thunk_##nm(const RadArgs* a, RadStream s) {                                \
        return avx_table_live()->nm(a, s);                                                        \
    }
AVX_OP_LIST(AVX_THUNK)
#undef AVX_THUNK

/* ================================================================== scratch
 *
 * One growable arena per thread per slot. The first call at a given size allocates; every call
 * after it at or below that size does not, which is what makes the hot path allocation-free
 * without pretending a host kernel can work in zero bytes.
 *
 * ALIGNED TO 64 so that a _mm512_load_ps off it is legal -- an aligned load from a 32-byte-aligned
 * buffer is a fault at AVX-512, and `new float[]` promises only 16. The kernels mostly use the
 * unaligned loads (on every core since Nehalem the penalty is zero when the address happens to be
 * aligned, so there is nothing to win by demanding alignment), but the GEMM packing buffers use
 * aligned stores and they come from here. */
namespace {

struct Arena {
    void*  p = nullptr;
    size_t bytes = 0;
    ~Arena() { std::free(p); }
    void* get(size_t want) {
        if (want <= bytes) return p;
        /* Grow geometrically: a kernel called at a ladder of increasing sizes would otherwise
         * realloc at every rung. A request past half the address space is taken as asked, so the
         * doubling cannot wrap to zero and spin. */
        size_t nb = bytes ? bytes : 4096;
        while (nb < want) nb = nb > SIZE_MAX / 2 ? want : nb * 2;
        void* q = nb <= SIZE_MAX - 63 ? std::aligned_alloc(64, (nb + 63) & ~(size_t)63) : nullptr;
        /* NO BUFFER IS A STOP, NOT A SMALLER BUFFER. The contract is "at least the size asked
         * for" and no caller checks it: every kernel writes the full row into what comes back,
         * most of them inside an OpenMP body that cannot return a status. Handing back the old,
         * smaller buffer turns an allocation failure into a heap overflow that corrupts whatever
         * sits after it; a null would be a fault at an address that says nothing. So the process
         * stops here, with the size that could not be had. */
        if (!q) {
            std::fprintf(stderr, "libavx: could not allocate %zu bytes of kernel scratch\n", want);
            std::abort();
        }
        std::free(p);
        p = q; bytes = nb;
        return p;
    }
};

thread_local Arena g_a, g_b, g_c, g_raw;

}  /* namespace */

namespace avx {
float* scratch_f32  (size_t n) { return (float*)g_a.get(n * sizeof(float)); }
float* scratch_f32_b(size_t n) { return (float*)g_b.get(n * sizeof(float)); }
float* scratch_f32_c(size_t n) { return (float*)g_c.get(n * sizeof(float)); }
void*  scratch_raw  (size_t b) { return g_raw.get(b); }
}

/* ================================================================== plugin open/close
 * The loader resolves these by these exact names (rad_abi.h) and a plugin that spells them
 * differently is silently never initialised -- which here would mean an fp8 table of zeros and
 * every fp8 operand reading as 0.0, a wrong answer with no diagnostic. Spelled from the ABI's own
 * typedefs so a signature drift is a compile error. */
extern "C" int rad_plugin_open(void) {
    fp8_tables_init();
    (void)avx_table_live();
    return RAD_OK;
}

extern "C" void rad_plugin_close(void) {}

[[maybe_unused]] static const RadPluginOpenFn  kOpenCheck  = rad_plugin_open;
[[maybe_unused]] static const RadPluginCloseFn kCloseCheck = rad_plugin_close;

/* ================================================================== the measured FMA ceiling
 *
 * Exported for rad-avx-bench, and it has to come from HERE rather than from the bench because the
 * bench is baseline-compiled (see the top of this file) and can only emit SSE2. The per-level
 * probes live in avx_support.cpp, compiled with the same flags as every kernel, so what they
 * report is the ceiling the kernels are actually up against -- including whatever clock the part
 * holds under a sustained vector load, which on anything with an AVX-512 frequency offset is not
 * the advertised boost.
 *
 * Returns the FLOPS PERFORMED, not a rate: the caller owns the clock, because it is the caller
 * that knows to take a median over repetitions rather than trusting one. */
extern "C" double avx_peak_fma_sc(int64_t);
extern "C" double avx_peak_fma_v1(int64_t);
extern "C" double avx_peak_fma_v2(int64_t);
extern "C" double avx_peak_fma_v3(int64_t);

extern "C" double avx_peak_fma(int level, int64_t iters) {
    switch (level) {
        case AVX_LEVEL_AVX512: return avx_peak_fma_v3(iters);
        case AVX_LEVEL_AVX2:   return avx_peak_fma_v2(iters);
        case AVX_LEVEL_AVX:    return avx_peak_fma_v1(iters);
        default:               return avx_peak_fma_sc(iters);
    }
}
