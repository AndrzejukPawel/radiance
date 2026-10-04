// Shared device-side machinery for R4D. Everything here is layout algebra verified on hardware
// by targeted layout probes. Do not "fix" any of it from a vendor doc.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// THE W4A8 SCALE GROUP LIVES HERE, not in either GEMM, because BOTH of them read the same packed
// artifact and a build where they disagree is silently wrong rather than a link error.  The
// plugin build passes -DR4D_GEMM_W4A8_GROUP to every W4 translation unit; this is the fallback for
// a translation unit built without it, and it has to be the same number for all of them.
#ifndef R4D_GEMM_W4A8_GROUP
#define R4D_GEMM_W4A8_GROUP 128      // K per scale
#endif

// ---- ragged query batches ----------------------------------------------------------------------
// A paged-attention batch may present a DIFFERENT query length per sequence: see R4DArgs::cu_q in
// r4d.h for why that is the ordinary case rather than an exotic one. These are the two lookups
// every kernel that indexes the dense q or out block needs, plus the inverse the split-KV combine
// needs because it walks output rows rather than sequences.
//
// The uniform case -- cu_q null -- compiles to one multiply and no load. The branch is
// wave-uniform (cu_q is a kernel argument), so it is a scalar test, and the loads it guards are
// scalar too.
__device__ __forceinline__ int r4d_q_off(const int* cu_q, int seq, int q_len) {
    return cu_q ? cu_q[seq] : seq * q_len;
}
__device__ __forceinline__ int r4d_q_len(const int* cu_q, int seq, int q_len) {
    return cu_q ? (cu_q[seq + 1] - cu_q[seq]) : q_len;
}
// num_seqs is at most a few tens, so the search is three or four scalar loads, taken once per
// block rather than per lane.
__device__ __forceinline__ int r4d_q_seq(const int* cu_q, int num_seqs, int q_len, int tok) {
    if (!cu_q) return tok / q_len;
    int lo = 0, hi = num_seqs;                       // invariant: cu_q[lo] <= tok < cu_q[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu_q[mid] <= tok) lo = mid; else hi = mid;
    }
    return lo;
}

typedef short    v8s __attribute__((ext_vector_type(8)));
typedef float    v8f __attribute__((ext_vector_type(8)));
typedef float    v2f __attribute__((ext_vector_type(2)));   // what cvt_pk_f32_fp8 actually returns

// exp2 with the hardware transcendental.

__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

// f32 -> bf16, round-to-nearest-even. gfx1201 has NO bf16 convert instruction, so this is the
// software form (verified absent: v_cvt_pk_bf16_f32).
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// ...and back, which is a shift and not a conversion: every bf16 is exactly a float.
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

// `lane ^ N` WITHOUT THE LDS PIPE.  `__shfl_xor` lowers to `ds_bpermute_b32` plus an
// `s_wait_dscnt` -- an LDS round trip, and when the exchange sits on a dependency chain (a dot
// product closed inside a token loop, say) that latency is the loop.  gfx10 and up carry ROW_XMASK
// (0x160 | N) in DPP16, so for N < 16 the partner read is a MODIFIER on the VALU op that consumes
// it and costs nothing at all: the compiler emits `v_add_f32_dpp ... row_xmask:2`.  N = 16 crosses
// the two 16-lane rows of a wave32 and takes v_permlanex16, still one VALU op.
// Identical arithmetic to the shuffle it replaces -- same lane mapping, same value -- on a wave
// whose lanes are all active, which is every caller's case: a butterfly is a whole-wave step.
template <int N>
__device__ __forceinline__ uint32_t r4d_xor_lane_u32(uint32_t v) {
    static_assert(N >= 1 && N <= 16 && (N & (N - 1)) == 0, "a lane xor within the wave");
    if constexpr (N < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | N, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u,
                                                      false, false);
}

template <int N>
__device__ __forceinline__ float r4d_xor_lane(float v) {
    return __uint_as_float(r4d_xor_lane_u32<N>(__float_as_uint(v)));
}

// A 64-bit value's partner: the two halves exchanged separately, which is what a 64-bit
// `__shfl_xor` does too.
template <int N>
__device__ __forceinline__ unsigned long long r4d_xor_lane_u64(unsigned long long v) {
    const uint32_t lo = r4d_xor_lane_u32<N>((uint32_t)v);
    const uint32_t hi = r4d_xor_lane_u32<N>((uint32_t)(v >> 32));
    return ((unsigned long long)hi << 32) | lo;
}

// Exchange a value between lane L and lane L^16 (the two halves a wave32 WMMA fragment splits a
// row across). v_permlanex16_b32 needs IDENTITY selectors to act as a swap: each 4-bit field names
// the source lane in the other half. sel0 covers lanes 0-7, sel1 lanes 8-15. Zero selectors would
// broadcast lane 0, which is silently wrong rather than obviously wrong.
__device__ __forceinline__ float swap16(float x) { return r4d_xor_lane<16>(x); }

// A WAVE32 SUM, WITHOUT THE LDS CROSSBAR. `__shfl_xor` lowers to `ds_bpermute_b32` on gfx1201
// whatever the offset is, and a bpermute is an LDS-pipe instruction: it queues behind every other
// LDS access and its latency is LDS latency. On a kernel whose arithmetic between reductions is
// small, a five-stage shuffle butterfly is most of the run -- the reductions cost more than the
// work between them.
//
// DPP does the exchange as a MODIFIER ON THE ADD, in the VALU, and gfx10 and later carry
// `row_xmask` (dpp_ctrl 0x160 | N) which is an XOR within each row of sixteen -- so four of the
// five butterfly offsets are expressible and only 16 crosses rows, where `v_permlanex16_b32` with
// IDENTITY selectors is the exchange (see swap16 above on why the selectors must be identity).
//
// THE LANES EXCHANGED AND THE ORDER OF THE ADDS ARE THE SAME AS THE SHUFFLE LOOP'S, so this is
// bit-identical and not merely equivalent: a kernel may be moved onto it without its generated
// text changing at any context length.
//
// Row and bank masks are all-ones because every lane participates; `bound_ctrl` only decides what
// a lane with no partner reads, and inside a full wave every lane has one.
template <int CTRL>
__device__ __forceinline__ float r4d_dpp_xor_add(float v) {
    return v + __builtin_bit_cast(float,
        __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true));
}

__device__ __forceinline__ float r4d_wave_sum(float v) {
    v += swap16(v);                        /* xor 16 -- crosses the two rows of sixteen */
    v = r4d_dpp_xor_add<0x168>(v);         /* xor  8 -- row_xmask from here down */
    v = r4d_dpp_xor_add<0x164>(v);         /* xor  4 */
    v = r4d_dpp_xor_add<0x162>(v);         /* xor  2 */
    v = r4d_dpp_xor_add<0x161>(v);         /* xor  1 */
    return v;
}

// THE SAME BUTTERFLY OVER A GROUP OF W LANES, W a power of two up to the wave, for a reduction that
// closes over fewer lanes than the wave as well as over all of it: the exchanges the loop
// `for (o = W / 2; o; o >>= 1) v = op(v, __shfl_xor(v, o))` makes, in its order, so bit-identical
// to that loop for the reason r4d_wave_sum gives.
template <int W>
__device__ __forceinline__ float r4d_xor_sum(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v += r4d_xor_lane<16>(v);
    if constexpr (W >= 16) v += r4d_xor_lane<8>(v);
    if constexpr (W >= 8)  v += r4d_xor_lane<4>(v);
    if constexpr (W >= 4)  v += r4d_xor_lane<2>(v);
    if constexpr (W >= 2)  v += r4d_xor_lane<1>(v);
    return v;
}

template <int W>
__device__ __forceinline__ float r4d_xor_max(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v = fmaxf(v, r4d_xor_lane<16>(v));
    if constexpr (W >= 16) v = fmaxf(v, r4d_xor_lane<8>(v));
    if constexpr (W >= 8)  v = fmaxf(v, r4d_xor_lane<4>(v));
    if constexpr (W >= 4)  v = fmaxf(v, r4d_xor_lane<2>(v));
    if constexpr (W >= 2)  v = fmaxf(v, r4d_xor_lane<1>(v));
    return v;
}

__device__ __forceinline__ float r4d_wave_max(float v) { return r4d_xor_max<32>(v); }

/* A WAVE'S INCLUSIVE PREFIX AND SUFFIX SUMS, in registers: row_shr (or row_shl) 1, 2, 4 and 8
 * within each row of sixteen, then the other row's total from one readlane. Lane L ends holding
 * the sum over lanes 0..L (prefix) or L..31 (suffix). For the scans a workgroup would otherwise run
 * as step-doubling rounds through LDS, two barriers a round; integer sums, so the order the lanes
 * combine in does not matter. Every lane of the wave must be active. */
__device__ __forceinline__ uint32_t r4d_wave_prefix_u32(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);   /* row_shr:1 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);   /* row_shr:2 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);   /* row_shr:4 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);   /* row_shr:8 */
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readlane((int)x, 15);
    return lane >= 16 ? x + lo : x;
}

__device__ __forceinline__ uint32_t r4d_wave_suffix_u32(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x101, 0xF, 0xF, true);   /* row_shl:1 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x102, 0xF, 0xF, true);   /* row_shl:2 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x104, 0xF, 0xF, true);   /* row_shl:4 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x108, 0xF, 0xF, true);   /* row_shl:8 */
    const uint32_t hi = (uint32_t)__builtin_amdgcn_readlane((int)x, 16);
    return lane < 16 ? x + hi : x;
}

// 8x8 transpose of 16-bit elements across each group of 8 lanes. Lane j receives element j from
// each of the 8 lanes in its group; the 8 addresses may be arbitrarily strided.
__device__ __forceinline__ v8s load_tr_b128(const void* p) {
    // This builtin requires a GLOBAL (address_space(1)) pointer from ROCm 7.2 / LLVM 22 on: a
    // generic pointer does not convert implicitly and the library does not compile without the
    // cast. The cast is sound here: every caller passes a global address, which is what
    // "global_load" means.
    return __builtin_amdgcn_global_load_tr_b128_v8i16(
        (__attribute__((address_space(1))) v8s*)const_cast<void*>(p));
}

// Gather the low (sel=0) or high (sel=1) byte of each of four 16-bit lanes of {w0,w1} into one
// dword. Written so the compiler can contract each to a single v_perm_b32.
__device__ __forceinline__ uint32_t byte_gather(uint32_t w0, uint32_t w1, int sel) {
    uint32_t s = (uint32_t)sel * 8u;
    return ((w0 >> s) & 0xffu)
         | (((w0 >> (16 + s)) & 0xffu) << 8)
         | (((w1 >> s) & 0xffu) << 16)
         | (((w1 >> (16 + s)) & 0xffu) << 24);
}

__device__ __forceinline__ void sched_barrier() { __builtin_amdgcn_sched_barrier(0); }

// __syncthreads() is a workgroup-scope acquire-release fence over ALL address spaces, so on gfx12
// it emits `global_inv scope:SCOPE_SE` on both sides of the barrier -- a vector-cache invalidate.
// Every one of those throws away the L0/L1 lines the next tile's KV loads would have hit. These
// barriers only ever order LDS traffic, so scoping the fence to the local address space makes
// the invalidate disappear.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Two-dword form of swap16.
__device__ __forceinline__ uint2 swap16_u2p(uint2 v) {
    return make_uint2(__builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.x))),
                      __builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.y))));
}

// ---------------------------------------------------------------------------------------------
// WORKGROUP WIDTH FOR THE ACTIVATION-PATH KERNELS
//
// All four of them (had_quant_act_i8, rmsnorm_had_quant_i8, gated_had_quant_i8,
// gdn_gated_norm_had_quant_i8) launch ONE workgroup per row, so at a decode batch the grid is a few
// tens of workgroups on a 64-CU part and they are latency bound, not bandwidth bound: a row moves
// tens of kilobytes and costs a small multiple of the launch floor.  Widening the workgroup is the
// only knob that adds parallelism without changing the algorithm, and it is the large win at small
// M.
//
// It stops paying, and then reverses hard, once the grid alone fills the machine.  At 1024 threads
// there is a RESIDENCY CLIFF just past M=32 -- the wider workgroup stops fitting and the cost
// roughly doubles.  It is a residency effect and not an address effect; the buffer's placement does
// not move it.  So 1024 is taken only where there is clearance below the cliff, because a batch
// that lands just above it would pay multiples of what the narrow width costs.
//
// WHERE 512 STOPS PAYING MOVES WITH K, because the LDS a row needs (K floats) is what caps how
// many of these workgroups a CU can hold.  At K=8704 only one fits at either width, so the wider
// workgroup is simply more waves per SIMD and it wins out to large M.  At K=3072 five fit, the grid
// fills the part on its own, and past M~72 the narrow one is even.  The thresholds below are those
// crossovers.
__host__ __device__ __forceinline__ int r4d_act_threads(int M, int K) {
    if (M <= 32) return 1024;
    const int lim = K >= 8192 ? (1 << 24) : (K >= 4096 ? 120 : 72);
    return M <= lim ? 512 : 256;
}
