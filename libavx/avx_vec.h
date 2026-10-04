/* avx_vec.h -- the vector abstraction every kernel in this plugin is written against, and the
 * dtype conversions that feed it.
 *
 * THIS HEADER IS COMPILED FOUR TIMES, once per ISA level, with AVX_LEVEL defined on the command
 * line and the matching -m flags on the object. Nothing in here is selected at runtime: by the
 * time this is preprocessed the level is a constant, so `vf` is one concrete type and every
 * operation below is one instruction or a short fixed sequence. The runtime choice happens once,
 * in avx_dispatch.cpp, between whole tables of already-compiled functions.
 *
 * ============================== WHY AN ABSTRACTION AND NOT FOUR COPIES ==============================
 *
 * Four hand-written copies of seventy-five kernels is three hundred implementations
 * to keep in agreement, and they would not stay in agreement -- the levels whatever machine you
 * develop on cannot execute are the ones that rot, and they are the levels the plugin exists for.
 * Written once against `vf`, the AVX-512 path is THE SAME ALGORITHM as the scalar path and the
 * checker's job is to prove the arithmetic agrees, not to re-derive the op.
 *
 * The cost is real and is stated rather than hidden: an abstraction cannot express the things that
 * differ in KIND between levels -- AVX-512's write masks, its 32 registers, its compress/expand.
 * So where one of those is the whole point, the kernel writes an `#if AVX_LEVEL >= 3` block and
 * says why in a comment next to it. That is a handful of places (the GEMM microkernel's register
 * blocking, top-k's compress), not the general case.
 *
 * ============================== EVERYTHING COMPUTES IN f32 ==============================
 *
 * libref's contract is f32 accumulation with IEEE RTNE narrowing (libref/ref_common.h) and this
 * plugin is checked against it, so f32 is the working type here too. Operands arrive as bf16, f16,
 * fp8 or i8; the rule is CONVERT A WHOLE ROW ONCE into an f32 scratch, compute, convert back. Not
 * per element, which is what libref does and what makes it slow: a per-element dtype switch inside
 * the inner loop is a branch the vectoriser cannot cross. The size of that effect is measured per
 * op by rad-avx-bench and recorded in each kernel's own header, not guessed at here.
 *
 * WHAT THIS PLUGIN DOES NOT PROMISE, and libref does: no reassociation and no FMA. Both are given
 * up deliberately and both are why the tolerances in avx_check.cpp are the kbench ones rather than
 * zero. A vector reduction sums in a different order by construction -- that IS the vectorisation
 * -- and refusing FMA would throw away half the machine's flops. The narrowing conversions are the
 * part that must be bit-exact, because they define what a result IS, and they are (see below).
 */
#ifndef RAD_AVX_VEC_H
#define RAD_AVX_VEC_H

#ifndef AVX_LEVEL
#error "AVX_LEVEL must be defined on the command line: 0 scalar, 1 avx, 2 avx2, 3 avx512"
#endif

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>
#include <cstring>
#include <cmath>

#if AVX_LEVEL >= 1
#include <immintrin.h>
#endif

/* The symbol suffix for this level. Every kernel defines avx_<op>_<suffix>. */
#if   AVX_LEVEL == 0
#define AVX_SUFFIX sc
#define AVX_LEVEL_NAME "scalar"
#elif AVX_LEVEL == 1
#define AVX_SUFFIX v1
#define AVX_LEVEL_NAME "avx"
#elif AVX_LEVEL == 2
#define AVX_SUFFIX v2
#define AVX_LEVEL_NAME "avx2"
#elif AVX_LEVEL == 3
#define AVX_SUFFIX v3
#define AVX_LEVEL_NAME "avx512"
#else
#error "AVX_LEVEL out of range"
#endif

#define AVX_CAT2(a, b) a##_##b
#define AVX_CAT(a, b)  AVX_CAT2(a, b)
/* avx_rmsnorm -> avx_rmsnorm_v2 at AVX_LEVEL 2. */
#define AVX_FN(name)   AVX_CAT(name, AVX_SUFFIX)

/* The header of every kernel in this plugin. `a` is the RadArgs and `stream` the RadStream, which
 * a host kernel ignores -- the house rule is that every entry point is (const RadArgs*, RadStream)
 * whatever domain it runs in (libr4d/r4d_plugin.h), and a host plugin that dropped the second
 * parameter would not be substitutable for a device one. */
#define AVX_KERNEL(nm) \
    extern "C" int AVX_FN(avx_##nm)(const RadArgs* a, [[maybe_unused]] RadStream stream)

namespace avx {
/* EVERY LEVEL GETS ITS OWN NAMESPACE, and it is an `inline` one so that a kernel writes `avx::vf`
 * without knowing which level it is being compiled at. Without this the four objects would define
 * four `avx::row_to_f32` symbols and the link would pick one of them for all four tables -- which
 * is not a link error, it is an AVX-512 helper silently running inside the scalar path, and on a
 * machine without AVX-512 that is a SIGILL from the fallback that exists to prevent SIGILLs. */
inline namespace AVX_SUFFIX {

/* ================================================================== the vector type */
#if AVX_LEVEL == 0
typedef float vf;
enum { VF_N = 1 };
static inline vf vf_zero()                    { return 0.0f; }
static inline vf vf_set1(float x)             { return x; }
static inline vf vf_load(const float* p)      { return *p; }
static inline vf vf_loadu(const float* p)     { return *p; }
static inline void vf_store(float* p, vf v)   { *p = v; }
static inline void vf_storeu(float* p, vf v)  { *p = v; }
static inline vf vf_add(vf a, vf b)           { return a + b; }
static inline vf vf_sub(vf a, vf b)           { return a - b; }
static inline vf vf_mul(vf a, vf b)           { return a * b; }
static inline vf vf_div(vf a, vf b)           { return a / b; }
static inline vf vf_max(vf a, vf b)           { return a > b ? a : b; }
static inline vf vf_min(vf a, vf b)           { return a < b ? a : b; }
static inline vf vf_fma(vf a, vf b, vf c)     { return a * b + c; }
static inline vf vf_sqrt(vf a)                { return std::sqrt(a); }
static inline vf vf_abs(vf a)                 { return std::fabs(a); }
static inline float vf_hsum(vf a)             { return a; }
static inline float vf_hmax(vf a)             { return a; }

#elif AVX_LEVEL == 1 || AVX_LEVEL == 2
typedef __m256 vf;
enum { VF_N = 8 };
static inline vf vf_zero()                    { return _mm256_setzero_ps(); }
static inline vf vf_set1(float x)             { return _mm256_set1_ps(x); }
static inline vf vf_load(const float* p)      { return _mm256_load_ps(p); }
static inline vf vf_loadu(const float* p)     { return _mm256_loadu_ps(p); }
static inline void vf_store(float* p, vf v)   { _mm256_store_ps(p, v); }
static inline void vf_storeu(float* p, vf v)  { _mm256_storeu_ps(p, v); }
static inline vf vf_add(vf a, vf b)           { return _mm256_add_ps(a, b); }
static inline vf vf_sub(vf a, vf b)           { return _mm256_sub_ps(a, b); }
static inline vf vf_mul(vf a, vf b)           { return _mm256_mul_ps(a, b); }
static inline vf vf_div(vf a, vf b)           { return _mm256_div_ps(a, b); }
static inline vf vf_max(vf a, vf b)           { return _mm256_max_ps(a, b); }
static inline vf vf_min(vf a, vf b)           { return _mm256_min_ps(a, b); }
#if AVX_LEVEL >= 2
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm256_fmadd_ps(a, b, c); }
#else
/* AVX HAS NO FMA -- it is a separate feature that arrived with Haswell, and Sandy Bridge is the
 * whole reason this level exists. Two roundings here, which makes this level numerically CLOSER
 * to libref than AVX2 is, not further. */
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif
static inline vf vf_sqrt(vf a)                { return _mm256_sqrt_ps(a); }
static inline vf vf_abs(vf a) {
    return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}
static inline float vf_hsum(vf a) {
    /* Lane-crossing first, then within the 128. Fixed order, so the answer does not depend on
     * which of two equally valid trees the compiler picks. */
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}
static inline float vf_hmax(vf a) {
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s  = _mm_max_ps(lo, hi);
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

#else  /* AVX_LEVEL == 3 */
typedef __m512 vf;
enum { VF_N = 16 };
static inline vf vf_zero()                    { return _mm512_setzero_ps(); }
static inline vf vf_set1(float x)             { return _mm512_set1_ps(x); }
static inline vf vf_load(const float* p)      { return _mm512_load_ps(p); }
static inline vf vf_loadu(const float* p)     { return _mm512_loadu_ps(p); }
static inline void vf_store(float* p, vf v)   { _mm512_store_ps(p, v); }
static inline void vf_storeu(float* p, vf v)  { _mm512_storeu_ps(p, v); }
static inline vf vf_add(vf a, vf b)           { return _mm512_add_ps(a, b); }
static inline vf vf_sub(vf a, vf b)           { return _mm512_sub_ps(a, b); }
static inline vf vf_mul(vf a, vf b)           { return _mm512_mul_ps(a, b); }
static inline vf vf_div(vf a, vf b)           { return _mm512_div_ps(a, b); }
static inline vf vf_max(vf a, vf b)           { return _mm512_max_ps(a, b); }
static inline vf vf_min(vf a, vf b)           { return _mm512_min_ps(a, b); }
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm512_fmadd_ps(a, b, c); }
static inline vf vf_sqrt(vf a)                { return _mm512_sqrt_ps(a); }
static inline vf vf_abs(vf a)                 { return _mm512_abs_ps(a); }
static inline float vf_hsum(vf a)             { return _mm512_reduce_add_ps(a); }
static inline float vf_hmax(vf a)             { return _mm512_reduce_max_ps(a); }
#endif

/* Tail handling. A masked load is the right instruction at AVX-512 and there is no such thing
 * below it, so `vf_load_n` is a mask at level 3 and a small stack copy elsewhere. Kernels use it
 * only for the RAGGED END of a row: a model width is a multiple of 128 and the tail almost never
 * runs, so this is about correctness at odd shapes, not about speed. */
static inline vf vf_load_n(const float* p, int n) {
#if AVX_LEVEL == 3
    return _mm512_maskz_loadu_ps((__mmask16)((1u << n) - 1u), p);
#elif AVX_LEVEL == 0
    return n > 0 ? *p : 0.0f;
#else
    float tmp[VF_N] = { 0 };
    for (int i = 0; i < n; ++i) tmp[i] = p[i];
    return _mm256_loadu_ps(tmp);
#endif
}

static inline void vf_store_n(float* p, vf v, int n) {
#if AVX_LEVEL == 3
    _mm512_mask_storeu_ps(p, (__mmask16)((1u << n) - 1u), v);
#elif AVX_LEVEL == 0
    if (n > 0) *p = v;
#else
    float tmp[VF_N];
    _mm256_storeu_ps(tmp, v);
    for (int i = 0; i < n; ++i) p[i] = tmp[i];
#endif
}

/* ================================================================== narrowing conversions
 *
 * THE BIT-EXACTNESS RULE. rad_plugin.h defines f32->bf16 as RTNE with the tie broken on the
 * surviving bit's parity, NaN forced quiet, and overflow allowed to carry into the bf16 infinity.
 * That definition IS the result (rad_plugin.h says so at length), and the kbench tolerance for
 * `cast` is 1e-6 -- which is to say: exact. So the vector forms below are the SAME INTEGER
 * ARITHMETIC, lane for lane, not an approximation of it, and avx_check.cpp checks `cast` over
 * every dtype pair against libref at that tolerance to hold them there.
 *
 * NOT `_mm512_cvtneps_pbh`. Zen 5 has AVX512-BF16 and its convert is also RTNE, but its NaN
 * handling is the hardware's rather than the ABI's, and the ABI's is what libref computes. One
 * instruction saved against a conversion that is a mask and two integer ops is not worth a
 * divergence the checker would have to be loosened to accept.
 */
static inline void cvt_bf16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(w));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
#elif AVX_LEVEL == 1
    /* AVX has no 256-bit integer ALU: the shift has to happen in two 128-bit halves and the
     * result is glued with vinsertf128, which is a FLOAT-domain instruction and therefore legal
     * here. This is the shape every integer helper at this level takes. */
    for (; i + 8 <= n; i += 8) {
        __m128i h  = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i lo = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
        __m128i hi = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(h, 8)), 16);
        __m256 v = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(lo)),
                                        _mm_castsi128_ps(hi), 1);
        _mm256_storeu_ps(dst + i, v);
    }
#endif
    for (; i < n; ++i) dst[i] = rad_bf16_to_f32(src[i]);
}

#if AVX_LEVEL >= 1
/* The RTNE core, on a vector of f32 bit patterns, leaving 16-bit results in the low half of each
 * lane. Shared by the three vector levels; the pack differs and the arithmetic does not. */
#if AVX_LEVEL == 3
static inline __m512i bf16_round_bits(__m512 f) {
    __m512i u    = _mm512_castps_si512(f);
    __m512i mag  = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
    __mmask16 nan = _mm512_cmpgt_epi32_mask(mag, _mm512_set1_epi32(0x7f800000));
    __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    __m512i rnd  = _mm512_srli_epi32(
                       _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)), lsb), 16);
    __m512i qnan = _mm512_or_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x0040));
    return _mm512_mask_blend_epi32(nan, rnd, qnan);
}
#elif AVX_LEVEL == 2
static inline __m256i bf16_round_bits(__m256 f) {
    __m256i u    = _mm256_castps_si256(f);
    __m256i mag  = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
    __m256i nan  = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7f800000));
    __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    __m256i rnd  = _mm256_srli_epi32(
                       _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb), 16);
    __m256i qnan = _mm256_or_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(0x0040));
    return _mm256_blendv_epi8(rnd, qnan, nan);
}
#else  /* AVX_LEVEL == 1 */
static inline __m128i bf16_round_bits128(__m128 f) {
    __m128i u    = _mm_castps_si128(f);
    __m128i mag  = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
    __m128i nan  = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x7f800000));
    __m128i lsb  = _mm_and_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(1));
    __m128i rnd  = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(u, _mm_set1_epi32(0x7fff)), lsb), 16);
    __m128i qnan = _mm_or_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(0x0040));
    return _mm_blendv_epi8(rnd, qnan, nan);
}
#endif
#endif

static inline void cvt_f32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i w = bf16_round_bits(_mm512_loadu_ps(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm512_cvtepi32_epi16(w));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i w  = bf16_round_bits(_mm256_loadu_ps(src + i));
        __m128i lo = _mm256_castsi256_si128(w);
        __m128i hi = _mm256_extracti128_si256(w, 1);
        /* Every lane is already <= 0xffff, so packus is exact rather than saturating. */
        _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(lo, hi));
    }
#elif AVX_LEVEL == 1
    for (; i + 8 <= n; i += 8) {
        __m256   v  = _mm256_loadu_ps(src + i);
        __m128i  lo = bf16_round_bits128(_mm256_castps256_ps128(v));
        __m128i  hi = bf16_round_bits128(_mm256_extractf128_ps(v, 1));
        _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(lo, hi));
    }
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_bf16(src[i]);
}

/* f16 IS HARDWARE FROM AVX2 UP AND SOFTWARE BELOW IT. F16C is its own CPUID bit and arrived one
 * generation after AVX, so the level-1 path cannot use vcvtph2ps and falls through to the ABI's
 * software convert. That is the honest thing: a level named `avx` that quietly required F16C
 * would fault on the only machines it exists for. F16C's rounding is IEEE RTNE, which is what
 * rad_f32_to_f16 computes, so the two agree bit for bit -- checked by `cast` at 1e-6. */
static inline void cvt_f16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src + i))));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(src + i))));
#endif
    for (; i < n; ++i) dst[i] = rad_f16_to_f32(src[i]);
}

static inline void cvt_f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm256_storeu_si256((__m256i*)(dst + i),
                            _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i*)(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_f16(src[i]);
}

/* ---- fp8. THE DECODE IS A 256-ENTRY TABLE AT THE SCALAR LEVEL and a bit reconstruction above it;
 * the encode is a masked fast path.
 *
 * DECODE: every possible byte is one of 256 floats, so the exact answer is a lookup and there is
 * nothing to round. The table is built by the ABI's own converter at load (avx_fp8_table_init), so
 * it cannot drift from rad_fp8e4m3_to_f32, and it is what the scalar path and every tail element
 * read. The vector e4m3 path reconstructs the bits instead, for the reason set out above
 * cvt_fp8e4m3_to_f32; e5m2 still gathers from its table.
 *
 * ENCODE: the bit-exact algorithm has three branches (subnormal, normal, saturate) and vectorising
 * all three costs more than it saves. Instead the normal case -- |x| in [2^-6, 448), which is
 * every element of a normalised activation -- is done with vector integer arithmetic, and a mask
 * of the elements that are NOT in that range drives a scalar fixup through the ABI's converter. On
 * real activations the mask is empty and the fixup loop does not run; when it does run the answer
 * is the ABI's, which is the definition. */
}  /* the tables are at GLOBAL scope with C linkage, deliberately: they are defined once in the
   baseline-compiled avx_dispatch.cpp and read by all four levels, so they must NOT pick up the
   per-level inline namespace the rest of this header is wrapped in. */
}
extern "C" {
/* NOT `const`, and that is forced: the tables are filled at load by rad_plugin_open from the
   ABI's own converters (a typed-out table would be a second definition of the format that can
   drift), so the object has to be writable. A const declaration here against a non-const
   definition would link -- C linkage does not mangle the type -- and would be a quiet type
   mismatch that no tool reports. */
extern float avx_fp8e4m3_tab[256];
extern float avx_fp8e5m2_tab[256];
}
namespace avx {
inline namespace AVX_SUFFIX {

/* THE DECODE IS A BIT RECONSTRUCTION, AND THE TABLE GATHER IS THE NULL RESULT.
 *
 * The obvious implementation is a 256-entry table -- every byte is one of 256 floats, so the answer
 * is a lookup and there is nothing to round -- gathered with vpgatherdd. `rad-avx-bench --cvt` says
 * that form LOSES TO THE SCALAR TABLE LOOKUP at both vector levels. A gather issues one cache line
 * per lane however small the table is; a 1 KiB table resident in L1 does not make that cheap, it
 * just makes it not slower than DRAM. Sixteen lanes of vpgatherdd is sixteen L1 accesses in
 * sequence.
 *
 * The reconstruction below is pure integer arithmetic and no memory at all:
 *
 *   u = (b & 0x7f) << 20  +  120 << 23    puts e in the f32 exponent field (bias 7 -> 127) and m
 *                                         in the top three mantissa bits, in one add
 *   e == 0                                is an fp8 SUBNORMAL, where the implicit 1 that f32 just
 *                                         supplied is wrong: (f - 2^-7) * 2 removes it and
 *                                         rescales, which is exact -- both operands are powers of
 *                                         two times a 3-bit mantissa
 *   mag == 0x7f                           is the format's ONLY NaN (S1111111); blended in
 *   sign                                  (b & 0x80) << 24
 *
 * The reconstruction is worth roughly 2.5x the gather at AVX2 and 3.3x at AVX-512 on that bench,
 * and the gather and the scalar table are within a few percent of each other.
 *
 * The table stays for the scalar path, where it is exact, has no branch, and is what the level is
 * for. It also stays because rad_plugin_open builds it from the ABI's own converter, so it remains
 * the definition the reconstruction above is checked against.
 *
 * WHAT IS NOT CLAIMED: that avoiding memory is obviously faster. Both forms are measured rather
 * than argued, and the checker proves the answer is identical over all 256 input bytes, NaN and
 * subnormals included, which a draw of normals never would. */
static inline void cvt_fp8e4m3_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    const __m512i c7f = _mm512_set1_epi32(0x7f);
    const __m512i cbias = _mm512_set1_epi32(120 << 23);
    const __m512  chalf = _mm512_set1_ps(0.0078125f);      /* 2^-7 */
    const __m512  cnan = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
    for (; i + 16 <= n; i += 16) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        const __m512i mag = _mm512_and_si512(b, c7f);
        const __m512i u = _mm512_add_epi32(_mm512_slli_epi32(mag, 20), cbias);
        __m512 f = _mm512_castsi512_ps(u);
        const __mmask16 sub = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(8));
        f = _mm512_mask_blend_ps(sub, f,
                                 _mm512_mul_ps(_mm512_sub_ps(f, chalf), _mm512_set1_ps(2.0f)));
        f = _mm512_mask_blend_ps(_mm512_cmpeq_epi32_mask(mag, c7f), f, cnan);
        const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(
                                      _mm512_or_si512(_mm512_castps_si512(f), sign)));
    }
#elif AVX_LEVEL == 2
    const __m256i c7f = _mm256_set1_epi32(0x7f);
    const __m256i cbias = _mm256_set1_epi32(120 << 23);
    const __m256  chalf = _mm256_set1_ps(0.0078125f);
    const __m256  cnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        const __m256i mag = _mm256_and_si256(b, c7f);
        const __m256i u = _mm256_add_epi32(_mm256_slli_epi32(mag, 20), cbias);
        __m256 f = _mm256_castsi256_ps(u);
        const __m256i sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), mag);
        f = _mm256_blendv_ps(f, _mm256_mul_ps(_mm256_sub_ps(f, chalf), _mm256_set1_ps(2.0f)),
                             _mm256_castsi256_ps(sub));
        f = _mm256_blendv_ps(f, cnan, _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, c7f)));
        const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(
                                      _mm256_or_si256(_mm256_castps_si256(f), sign)));
    }
#endif
    for (; i < n; ++i) dst[i] = avx_fp8e4m3_tab[src[i]];
}

static inline void cvt_fp8e5m2_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i i32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        _mm512_storeu_ps(dst + i, _mm512_i32gather_ps(i32, avx_fp8e5m2_tab, 4));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        _mm256_storeu_ps(dst + i, _mm256_i32gather_ps(avx_fp8e5m2_tab, i32, 4));
    }
#endif
    for (; i < n; ++i) dst[i] = avx_fp8e5m2_tab[src[i]];
}

static inline void cvt_f32_to_fp8e4m3(const float* src, uint8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    /* q = ((e+7) << 3) | rtne(mantissa >> 20), valid while 2^-6 <= |x| < 448 and x is finite.
     * Anything else takes the scalar fixup below.
     *
     * THE LOWER BOUND IS 2^-6 AND NOTHING ELSE. e4m3's smallest NORMAL is 2^-6; below it the
     * format is subnormal and the mantissa has to be renormalised with a variable shift, which is
     * the branch this fast path exists to avoid. A bound set any lower lets fp8-subnormal
     * magnitudes through the normal path, where they produce an exponent field of zero or below
     * and an answer that is wrong by orders of magnitude. The bound is written as its float bit
     * pattern with the value beside it so the next reader can check it rather than trust it. */
    const __m512i c_lo  = _mm512_set1_epi32(0x3c800000);  /* 2^-6, the smallest e4m3 normal */
    const __m512i c_hi  = _mm512_set1_epi32(0x43e00000);  /* 448,  the largest e4m3 finite  */
    for (; i + 16 <= n; i += 16) {
        __m512  f   = _mm512_loadu_ps(src + i);
        __m512i u   = _mm512_castps_si512(f);
        __m512i mag = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
        __mmask16 fast = _kand_mask16(_mm512_cmpge_epi32_mask(mag, c_lo),
                                      _mm512_cmplt_epi32_mask(mag, c_hi));
        __m512i sign = _mm512_srli_epi32(_mm512_and_si512(u, _mm512_set1_epi32(0x80000000u)), 24);
        /* RTNE on the 20 bits below the kept mantissa: add half, plus one more when the surviving
         * bit is odd -- the same tie rule as bf16, one field down. The exponent field rides along
         * in the same add, so a mantissa carry increments it for free. */
        __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(mag, 20), _mm512_set1_epi32(1));
        __m512i r    = _mm512_add_epi32(_mm512_add_epi32(mag, _mm512_set1_epi32(0x7ffff)), lsb);
        __m512i e    = _mm512_sub_epi32(_mm512_srli_epi32(r, 23), _mm512_set1_epi32(127 - 7));
        __m512i m    = _mm512_and_si512(_mm512_srli_epi32(r, 20), _mm512_set1_epi32(7));
        __m512i q    = _mm512_or_si512(sign, _mm512_or_si512(_mm512_slli_epi32(e, 3), m));
        /* A carry out of the top mantissa bit can push e past 15, and (e, m) = (15, 7) is the
         * format's only NaN encoding rather than a number. Neither can be produced by an input
         * strictly below 448, but the guard costs one compare and the alternative is a silent NaN
         * in a weight: elements that reach either drop into the fixup, which saturates. */
        __mmask16 ovf = _kor_mask16(
            _mm512_cmpgt_epi32_mask(e, _mm512_set1_epi32(15)),
            _kand_mask16(_mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(15)),
                         _mm512_cmpeq_epi32_mask(m, _mm512_set1_epi32(7))));
        fast = _kandn_mask16(ovf, fast);
        /* THE ALL-FAST CASE IS THE WHOLE POINT AND IT GETS ITS OWN EXIT. Every element of a
         * normalised activation is in [2^-6, 448), so `fast` is 0xffff on essentially every vector
         * a model produces, and the answer is one narrowing store of sixteen bytes.
         *
         * Without this exit the sixteen lanes spill to a stack array and a scalar loop tests one
         * mask bit per element -- a loop that then runs on exactly the path the vector arithmetic
         * was written to avoid, and dominates the op. A "fast path" whose epilogue is a
         * per-element branch is not a fast path. */
        if (fast == (__mmask16)0xffff) {
            _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtepi32_epi8(q));
            continue;
        }
        alignas(64) int32_t tmp[16];
        _mm512_store_si512((__m512i*)tmp, q);
        for (int k = 0; k < 16; ++k)
            dst[i + k] = (fast >> k) & 1 ? (uint8_t)tmp[k] : rad_f32_to_fp8e4m3(src[i + k]);
    }
#elif AVX_LEVEL == 2
    /* THE SAME ARITHMETIC AT 256 BITS, and it exists because without it AVX2 falls all the way
     * through to the ABI's scalar converter: an order of magnitude slower than the AVX-512 body,
     * not the 2x its width would suggest. AVX2 is the level most machines in the world run, so a
     * gap there is not a rounding error in the coverage.
     *
     * The mask is a movemask rather than a k-register, and the all-fast exit is the same: eight
     * lanes narrowed to eight bytes with two packs. */
    for (; i + 8 <= n; i += 8) {
        __m256  f   = _mm256_loadu_ps(src + i);
        __m256i u   = _mm256_castps_si256(f);
        __m256i mag = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
        const __m256i c_lo = _mm256_set1_epi32(0x3c800000);   /* 2^-6 */
        const __m256i c_hi = _mm256_set1_epi32(0x43e00000);   /* 448  */
        /* cmpgt is the only signed compare AVX2 has, so `mag >= c_lo` is written as
         * `!(c_lo > mag)` and `mag < c_hi` as `c_hi > mag`. Both operands are non-negative. */
        __m256i ge = _mm256_andnot_si256(_mm256_cmpgt_epi32(c_lo, mag),
                                         _mm256_cmpgt_epi32(c_hi, mag));
        __m256i sign = _mm256_srli_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x80000000u)), 24);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(mag, 20), _mm256_set1_epi32(1));
        __m256i r = _mm256_add_epi32(_mm256_add_epi32(mag, _mm256_set1_epi32(0x7ffff)), lsb);
        __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(r, 23), _mm256_set1_epi32(127 - 7));
        __m256i m = _mm256_and_si256(_mm256_srli_epi32(r, 20), _mm256_set1_epi32(7));
        __m256i q = _mm256_or_si256(sign, _mm256_or_si256(_mm256_slli_epi32(e, 3), m));
        __m256i ovf = _mm256_or_si256(
            _mm256_cmpgt_epi32(e, _mm256_set1_epi32(15)),
            _mm256_and_si256(_mm256_cmpeq_epi32(e, _mm256_set1_epi32(15)),
                             _mm256_cmpeq_epi32(m, _mm256_set1_epi32(7))));
        ge = _mm256_andnot_si256(ovf, ge);
        const int fast = _mm256_movemask_ps(_mm256_castsi256_ps(ge));
        if (fast == 0xff) {
            __m128i p = _mm_packus_epi32(_mm256_castsi256_si128(q),
                                         _mm256_extracti128_si256(q, 1));
            p = _mm_packus_epi16(p, p);
            _mm_storel_epi64((__m128i*)(dst + i), p);
            continue;
        }
        alignas(32) int32_t tmp[8];
        _mm256_store_si256((__m256i*)tmp, q);
        for (int k = 0; k < 8; ++k)
            dst[i + k] = (fast >> k) & 1 ? (uint8_t)tmp[k] : rad_f32_to_fp8e4m3(src[i + k]);
    }
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_fp8e4m3(src[i]);
}

static inline void cvt_f32_to_fp8e5m2(const float* src, uint8_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = rad_f32_to_fp8e5m2(src[i]);
}

/* ---- int8. `_mm*_cvtps_epi32` rounds under MXCSR, which is round-to-nearest-even unless a
 * caller has changed it -- the same rule `std::nearbyintf` follows, which is what libref uses.
 *
 * THE CLAMP IS TAKEN IN FLOAT, BEFORE THE CONVERSION, and the saturating packs are not a
 * substitute for it: a value past the int32 range and +inf both convert to 0x80000000, the
 * "integer indefinite", which the packs then saturate to -128 -- the opposite end of the range.
 * Clamping to [-128, 127] first and rounding second lands where libref's round-then-clamp does for
 * every input, the .5 ties at the ends included, so this stays bit-exact against the reference and
 * `quant_act` can be held at 1e-6. A NaN stores 0, which is also what the scalar tail does. */
#if AVX_LEVEL == 3
static inline __m512 i8_range_ps(__m512 x) {
    const __mmask16 ord = _mm512_cmp_ps_mask(x, x, _CMP_ORD_Q);
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-128.0f)), _mm512_set1_ps(127.0f));
    return _mm512_maskz_mov_ps(ord, x);
}
#elif AVX_LEVEL == 2
static inline __m256 i8_range_ps(__m256 x) {
    const __m256 ord = _mm256_cmp_ps(x, x, _CMP_ORD_Q);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-128.0f)), _mm256_set1_ps(127.0f));
    return _mm256_and_ps(x, ord);
}
#endif

static inline int8_t i8_round_sat(float v) {
    if (std::isnan(v)) return 0;
    const float r = std::nearbyintf(v);
    return (int8_t)(r < -128.f ? -128.f : (r > 127.f ? 127.f : r));
}

static inline void cvt_f32_to_i8(const float* src, int8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i q = _mm512_cvtps_epi32(i8_range_ps(_mm512_loadu_ps(src + i)));
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtsepi32_epi8(q));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i q  = _mm256_cvtps_epi32(i8_range_ps(_mm256_loadu_ps(src + i)));
        __m128i p  = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        p = _mm_packs_epi16(p, p);
        _mm_storel_epi64((__m128i*)(dst + i), p);
    }
#endif
    for (; i < n; ++i) dst[i] = i8_round_sat(src[i]);
}

/* Fused scale + int8-encode for the quant_act_i8 family: t = src*inv computed in a register
 * feeds the same cvtps/packs sequence the split form uses (MXCSR round-to-nearest-even,
 * saturating packs -- exactly libref's nearbyintf-and-clamp, bit for bit), so the bytes match
 * the scale-then-store form while a full f32 write + read of the row disappears. Kept to the
 * int8 family: the fp8 encode's bit-twiddle is a long integer chain and folding the scale into
 * it measures slower, where this one measures a few percent faster -- the same traffic saved,
 * the opposite instruction-level trade. */
static inline void cvt_scaled_f32_to_i8(const float* src, float inv, int8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    const __m512 vinv = _mm512_set1_ps(inv);
    for (; i + 16 <= n; i += 16) {
        __m512 t = _mm512_mul_ps(_mm512_loadu_ps(src + i), vinv);
        __m512i q = _mm512_cvtps_epi32(i8_range_ps(t));
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtsepi32_epi8(q));
    }
#elif AVX_LEVEL == 2
    const __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8) {
        __m256 t = _mm256_mul_ps(_mm256_loadu_ps(src + i), vinv);
        __m256i q  = _mm256_cvtps_epi32(i8_range_ps(t));
        __m128i p  = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        p = _mm_packs_epi16(p, p);
        _mm_storel_epi64((__m128i*)(dst + i), p);
    }
#endif
    for (; i < n; ++i) dst[i] = i8_round_sat(src[i] * inv);
}

static inline void cvt_i8_to_f32(const int8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i,
                         _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                             _mm_loadu_si128((const __m128i*)(src + i)))));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                             _mm_loadl_epi64((const __m128i*)(src + i)))));
#endif
    for (; i < n; ++i) dst[i] = (float)src[i];
}

/* ================================================================== the dtype-generic row
 *
 * ONE SWITCH PER ROW, NOT PER ELEMENT. This is the single most valuable thing in this header:
 * libref's inner loops call ld_dt(), which is a switch on every element, and hoisting it to the
 * row is most of the speedup on every memory-bound op before a single vector instruction is
 * issued. The f32 case is a memcpy the caller can skip entirely -- see row_f32_in below.
 *
 * W4 / W2 / MXFP4 are the packed WEIGHT codes and are decoded here too, because `dequant` and the
 * quantised GEMMs both need them and the unpack is the same nibble arithmetic in both. They are
 * raw CODES: the group scale belongs to the caller, which is the only component that knows where
 * the scale stream is (libref/ref_common.h makes the same split). */
void row_to_f32  (uint32_t dt, const void* src, float* dst, int64_t n);
void row_from_f32(uint32_t dt, const float* src, void* dst, int64_t n);

/* The strided form: element i of the row is at src[i * stride]. A separate function because the
 * contiguous one is the case that matters and a stride parameter in it would cost a multiply per
 * element on the path that does not need one. */
void row_to_f32_strided(uint32_t dt, const void* src, int64_t stride, float* dst, int64_t n);

/* When the source is already f32, hand back the pointer instead of copying. Returns `src` for
 * RAD_F32 and `scratch` (filled) otherwise, so a kernel reads:
 *     const float* x = avx::row_f32_in(dt, p, scratch, n);
 * and pays nothing at all on the f32 path. */
static inline const float* row_f32_in(uint32_t dt, const void* src, float* scratch, int64_t n) {
    if (dt == RAD_F32) return (const float*)src;
    row_to_f32(dt, src, scratch, n);
    return scratch;
}

/* ================================================================== math
 *
 * exp AND THE SIGMOID IT FEEDS. Every softmax, every gate and every attention row goes through
 * this, so a call to libm's expf per element is most of the cost of those ops. How much is
 * measured per op by rad-avx-bench.
 *
 * THE POLYNOMIAL IS MINIMAX DEGREE 5 ON [-ln2/2, ln2/2] after range reduction, which is ~0.6 ulp
 * and comfortably inside every tolerance in this project -- softmax's 1e-6 included, because the
 * softmax NORMALISES: a relative error common to every term cancels in the quotient, and what
 * survives is the DIFFERENCE between lanes' errors, an order smaller again.
 *
 * NULL RESULT, on the two ways of not writing this: the _mm512_exp_ps SVML form is not available
 * without Intel's runtime, and GCC's libmvec vectorises expf only when it can prove the loop is
 * safe to call `_ZGVeN16v_expf` from, which it cannot do inside a kernel that also does integer
 * work. Hand-writing the polynomial is not a micro-optimisation here, it is the only way to get a
 * vector exp at all. */
static inline vf vf_exp(vf x) {
#if AVX_LEVEL == 0
    return std::exp(x);
#else
    /* Clamp first: the reduction below overflows the integer exponent construction outside this
     * range. The TOP saturates, which every consumer here wants rather than an infinity. The
     * BOTTOM and a NaN are put right after the polynomial, because the clamp alone answers
     * wrongly for both: e^-inf would come out as e^lo, a tiny positive number, and a NaN would be
     * clamped to lo and vanish -- so a fully masked softmax row would read 1/n instead of the NaN
     * libref gives. Below lo e^x is under the smallest normal float and is taken as zero. */
    const vf hi = vf_set1(88.3762626647949f), lo = vf_set1(-87.3365447504f);
    const vf x0 = x;
    x = vf_min(vf_max(x, lo), hi);
    const vf log2e = vf_set1(1.44269504088896341f);
    const vf c1 = vf_set1(-0.693359375f), c2 = vf_set1(2.12194440e-4f);
#if AVX_LEVEL == 3
    vf n = _mm512_roundscale_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    vf n = _mm256_round_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
    /* x - n*ln2, split so the product is exact in f32. */
    vf r = vf_fma(n, c1, x);
    r = vf_fma(n, c2, r);
    vf r2 = vf_mul(r, r);
    /* exp(r) on [-ln2/2, ln2/2]. Horner, so the dependency chain is the depth of the polynomial
     * and not of a wider evaluation tree. */
    vf p = vf_set1(1.9875691500e-4f);
    p = vf_fma(p, r, vf_set1(1.3981999507e-3f));
    p = vf_fma(p, r, vf_set1(8.3334519073e-3f));
    p = vf_fma(p, r, vf_set1(4.1665795894e-2f));
    p = vf_fma(p, r, vf_set1(1.6666665459e-1f));
    p = vf_fma(p, r, vf_set1(5.0000001201e-1f));
    p = vf_fma(p, r2, vf_add(r, vf_set1(1.0f)));
    /* 2^n by building the exponent field directly. */
#if AVX_LEVEL == 3
    __m512i k = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    vf e = vf_mul(p, _mm512_castsi512_ps(k));
    e = _mm512_mask_mov_ps(e, _mm512_cmp_ps_mask(x0, lo, _CMP_LT_OQ), _mm512_setzero_ps());
    return _mm512_mask_mov_ps(e, _mm512_cmp_ps_mask(x0, x0, _CMP_UNORD_Q), x0);
#else
#if AVX_LEVEL == 2
    __m256i k = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    vf e = vf_mul(p, _mm256_castsi256_ps(k));
#else
    __m128i nlo = _mm_cvtps_epi32(_mm256_castps256_ps128(n));
    __m128i nhi = _mm_cvtps_epi32(_mm256_extractf128_ps(n, 1));
    nlo = _mm_slli_epi32(_mm_add_epi32(nlo, _mm_set1_epi32(127)), 23);
    nhi = _mm_slli_epi32(_mm_add_epi32(nhi, _mm_set1_epi32(127)), 23);
    __m256 k = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(nlo)),
                                    _mm_castsi128_ps(nhi), 1);
    vf e = vf_mul(p, k);
#endif
    e = _mm256_andnot_ps(_mm256_cmp_ps(x0, lo, _CMP_LT_OQ), e);
    return _mm256_blendv_ps(e, x0, _mm256_cmp_ps(x0, x0, _CMP_UNORD_Q));
#endif
#endif
}

static inline vf vf_sigmoid(vf x) {
    /* 1 / (1 + e^-x). Not the tanh form: this is what libref computes and the two differ in the
     * last ulp, which `sigmoid`'s 1e-5 f32 tolerance would pass but `hc_read`'s gate chain -- a
     * sigmoid of a dot product of a sigmoid -- accumulates. */
    return vf_div(vf_set1(1.0f), vf_add(vf_set1(1.0f), vf_exp(vf_sub(vf_zero(), x))));
}

static inline vf vf_silu(vf x) { return vf_mul(x, vf_sigmoid(x)); }

#if AVX_LEVEL >= 2
/* The CHEAP silu, for consumers that round to bf16 on the next store: a Taylor-4 exp head
 * (truncation ~6e-5) and a raw reciprocal seed (2^-14 avx512, 2^-12 avx2, no Newton step)
 * instead of the minimax tail and the division. ~3e-4 worst lane against the 6e-3 bf16
 * bucket: twentyfold of margin -- but NOT for f32 consumers, where 3e-4 does not fit 1e-5.
 * Declared here rather than inside one kernel so every bf16-store consumer can reach it;
 * silu_mul is the only caller today, because in a frontend-bound token loop -- the conv
 * kernels -- the difference is unmeasurable. Callers key on their OUTPUT dtype. */
static inline vf vf_exp_t4(vf x) {
    const vf hi = vf_set1(88.3762626647949f), lo = vf_set1(-87.3365447504f);
    x = vf_min(vf_max(x, lo), hi);
    const vf log2e = vf_set1(1.44269504088896341f);
    const vf c1 = vf_set1(-0.693359375f), c2 = vf_set1(2.12194440e-4f);
#if AVX_LEVEL == 3
    vf n = _mm512_roundscale_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    vf n = _mm256_round_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
    vf r = vf_fma(n, c1, x);
    r = vf_fma(n, c2, r);
    /* Taylor-4 head: 1 + r(1 + r(1/2 + r(1/6 + r/24))). Four FMAs against the exact path's
     * seven-op minimax tail -- the reduction above is unchanged, so the exponent build below
     * is too. */
    vf p = vf_fma(r, vf_set1(1.0f / 24.0f), vf_set1(1.0f / 6.0f));
    p = vf_fma(p, r, vf_set1(0.5f));
    p = vf_fma(p, r, vf_set1(1.0f));
    p = vf_fma(p, r, vf_set1(1.0f));
#if AVX_LEVEL == 3
    __m512i k = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    return vf_mul(p, _mm512_castsi512_ps(k));
#else
    __m256i k = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return vf_mul(p, _mm256_castsi256_ps(k));
#endif
}

static inline vf vf_silu_fast(vf x) {
#if AVX_LEVEL == 3
    vf e = vf_exp_t4(vf_sub(vf_zero(), x));
    vf d = vf_add(vf_set1(1.0f), e);
    vf r = _mm512_rcp14_ps(d);
#else
    vf e = vf_exp_t4(vf_sub(vf_zero(), x));
    vf d = vf_add(vf_set1(1.0f), e);
    vf r = _mm256_rcp_ps(d);
#endif
    /* No Newton step: d lies in [1, emax] so the seed is in range and nothing overflows. */
    return vf_mul(x, r);
}
#endif

/* Sum of squares of a row, vectorised, with a FIXED reduction tree so the answer does not depend
 * on the compiler's unrolling. Used by every norm. */
/* ================================================================== an operand's row, as f32
 *
 * THE TWO FUNCTIONS EVERY KERNEL OPENS WITH. `in_row` hands back a pointer to `n` floats holding
 * element (base + i * laststride) of the operand -- which is `t->data` itself when the operand is
 * already dense f32, so the common case copies nothing at all. `out_row` is the inverse.
 *
 * They are where the DENSE/STRIDED split lives, and it lives in one place on purpose: every op in
 * this vocabulary has to accept a strided view (libref/ref_ops.h: "no geometry constraints" has to
 * mean strides too), and seventy-five kernels each writing their own stride test is seventy-five
 * chances to write `t->stride[0]` where `laststride(t)` was meant. */
/* ONE STEP OF AN EXACT TOP-R SELECTION over a row of f32: descending by value and, ON A TIE, by
 * the LOWER index. That tie rule is libr4d's and it has to be stated rather than left to a sort's
 * stability -- a router that breaks ties differently from the kernel it is checked against routes a
 * token to a different expert, and the diff is a whole layer wide.
 *
 * `pv`/`pi` carry the previously selected (value, index); seed them with +infinity and -1. Returns
 * false when nothing is left to select. Shared by row_topk, router_topk and sample_topk, which is
 * why it is here rather than in one of them. */
bool topk_next_f32(const float* v, int64_t n, float* pv, int64_t* pi);

const float* in_row (const struct RadTensor* t, int64_t base, int64_t n, float* scr);
void         out_row(const struct RadTensor* t, int64_t base, int64_t n, const float* src);

float row_sumsq(const float* x, int64_t n);
float row_sum  (const float* x, int64_t n);
float row_amax (const float* x, int64_t n);
float row_max  (const float* x, int64_t n);

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */
#endif /* RAD_AVX_VEC_H */
