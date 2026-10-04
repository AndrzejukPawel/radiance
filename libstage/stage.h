// stage.h -- libstage: tensor-parallel collectives staged through pinned host
// memory, for machines whose GPUs have no PCIe P2P path between them.
//
// DESIGN (see docs/NOP2P.md §4b). Radiance runs one process with one thread per
// rank in a shared address space. libr4d's collectives push directly into the
// peer's VRAM, which needs hipDeviceEnablePeerAccess -- absent here. This
// library stages through hipHostMallocMapped memory instead, which every GPU
// in the process can read and write without any peer mapping: each rank
// publishes its contribution into its own mapped region, and a device kernel
// pulls the peer's region over PCIe. All traffic is GPU<->host; no peer
// pointer is ever dereferenced.
//
// PROTOCOL (per resolved instance, world size 2). Each rank owns one mapped
// region holding two message slots plus two flag words (ready). A launch with
// index i (a host-side per-instance counter -- launches are eager, never
// graphed, so it always advances) uses slot i&1:
//
//   xfer kernel (1 block): thread 0 spins until the peer finished launch i-2
//     (slot reuse), all threads push my input bytes into my slot, a
//     system fence, thread 0 publishes ready=i+1, spins until the peer
//     published i+1, then rejoins;
//   math kernel: out[e] = narrow(in[e] + peer[e]) in fp32 ascending order,
//     or the fused op's math over the same inputs.
//
// Slot reuse is safe by distance: launch i+2's push is ordered after this
// rank's ready(i+1) wait, which the peer publishes only after its launch i+1
// started, which is after its launch i -- the reader of this slot -- finished.
// The ready words are monotonic, so there is no ABA. A handshake that cannot
// complete is a lockstep violation and traps, as libr4d's does.
//
// NUMERICS. The exchange is exact -- full bytes, never quantised -- so an
// `exact=0` declaration served here gets the exact sum, the safe direction
// (a caller that accepts lossy and receives exact is correct; the reverse
// would not be). Reductions accumulate in fp32 in ascending rank order and
// narrow exactly as the unfused chain they replace does: hardware RNE for the
// all-reduce store, round-half-up (+0x8000) for hc_write, RNE for gather
// narrows, RTNE fp8/i8 quantisers per the ABI -- so tp2-through-libstage
// reproduces the tp1 unfused kernels bit for bit. All device code compiles
// -ffp-contract=off (via rad_hipify): no FMA anywhere a sum is folded.
#pragma once

#include "rad_abi.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <stdexcept>

// ---------------------------------------------------------------------------
// errors
#define ST_HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) return RAD_E_DEVICE; } while (0)

// ---------------------------------------------------------------------------
// dtypes. 0 bf16, 1 fp16/f16, 2 fp32. The exchange moves bytes; only the
// reduction math reads this.
static inline int stage_dtype_code(const char* dt) {
    if (!dt) return -1;
    if (!std::strcmp(dt, "bf16")) return 0;
    if (!std::strcmp(dt, "fp16") || !std::strcmp(dt, "f16")) return 1;
    if (!std::strcmp(dt, "fp32") || !std::strcmp(dt, "f32")) return 2;
    return -1;
}
static inline int stage_dtype_esize(int code) { return code == 2 ? 4 : 2; }

static inline long long stage_align16(long long v) { return (v + 15) & ~15LL; }

// Operand dtype codes: the enum already resolved the f16/fp16 and f32/fp32
// spelling synonyms, so this is a plain comparison.
static inline int stage_operand_code(uint32_t dt) {
    switch (dt) {
        case RAD_BF16: return 0;
        case RAD_F16:  return 1;
        case RAD_F32:  return 2;
        default:       return -1;
    }
}

// An int parameter, or dflt. Ranges are refused (a surviving range at launch
// is a selection bug).
static inline long long stage_geti(const RadParam* p, int n_p, const char* k, long long dflt) {
    return rad_param_geti(p, n_p, k, dflt);
}
// The same for a shape hook, where a range reads as its high bound.
static inline long long stage_getdim(const RadParam* p, int n_p, const char* k, long long dflt) {
    return rad_param_getdim(p, n_p, k, dflt);
}

// ---------------------------------------------------------------------------
// transport instance: one per (rank, world, dtype, message bound), shared by
// every op that resolves to the same key (cf. libr4d's ArInst cache: at a
// 2048-token chunk a slot is tens of MiB, and a 64-layer model declares over
// a hundred collectives that need a few of them).
struct StageInst {
    int rank = 0, world = 0;
    int dtype = 0;          // 0 bf16, 1 fp16, 2 fp32
    int esize = 2;
    long long maxelems = 0; // bound the slot was sized for, in elements
    long long slotbytes = 0;
    // My mapped region and the peer's. Layout: [slot][ready:u32][done:u32],
    // 16B-aligned. Device pointers are this rank's own mappings (resolved
    // locally after the rendezvous); host pointers are process-common.
    void* my_host = nullptr;
    void* peer_host = nullptr;
    void* my_dev = nullptr;
    void* peer_dev = nullptr;
    long long flag_off = 0; // byte offset of the ready word inside the region
    // Device-resident launch counter (replay-safe: the ticket is drawn on the
    // device, so the launch arguments never change between a recording and
    // its replay -- the same reason libr4d keeps its sequence counters in
    // device memory).
    unsigned* my_counter = nullptr;
};

// The rendezvous: publish my region, wait for the peer's. One per process
// (all ranks declare the same graph in the same order, so arrival order
// pairs correctly -- the same argument libr4d's rendezvous rests on).
int stage_rendezvous(void* my_region, int dev, int rank, int world,
                     void** peer_region_out);

// Exchange addresses for a launch: base region pointers plus flag words.
// All stable across launches (replay-safe); the slot is singular.
struct StageAddrs {
    long my_slot, peer_slot, my_ready, peer_ready, peer_done, my_done, counter;
};
static inline StageAddrs stage_addrs(const StageInst* in) {
    const long my = (long)(uintptr_t)in->my_dev;
    const long peer = (long)(uintptr_t)in->peer_dev;
    StageAddrs a;
    a.my_slot = my;
    a.peer_slot = peer;
    a.my_ready = my + in->flag_off;
    a.peer_ready = peer + in->flag_off;
    a.peer_done = peer + in->flag_off + 8;
    a.my_done = my + in->flag_off + 8;
    a.counter = (long)(uintptr_t)in->my_counter;
    return a;
}

// Shared instance cache.
int stage_acquire(int rank, int world, int dtype, long long maxelems, StageInst** out);
void stage_release(StageInst* in);

// Device entry points (defined in stage_kernels.hip).
void stage_xfer(long my_slot, long peer_slot,
                long my_ready, long peer_ready, long peer_done,
                long counter, long in_ptr, long nbytes, long stream);
void stage_mark_done(long my_done_ptr, long counter, long stream);

// One exchange: transfer, then the caller issues its math, then completion.
static inline void stage_xchg(const StageAddrs& ad, long in_ptr, long nbytes, long stream) {
    stage_xfer(ad.my_slot, ad.peer_slot, ad.my_ready, ad.peer_ready, ad.peer_done,
               ad.counter, in_ptr, nbytes, stream);
}
static inline void stage_xdone(const StageAddrs& ad, long stream) {
    stage_mark_done(ad.my_done, ad.counter, stream);
}

// ---------------------------------------------------------------------------
// device kernels (stage_kernels.hip). All take raw addresses; streams are
// hipStream_t passed as long.

// The transfer + handshake. Single block. The launch ticket comes from the
// instance's device-resident counter (replay-safe: no per-launch values in
// the arguments). Single slot: launch i's push waits until the peer
// finished launch i-1 (its read), publishes ready=i+1, waits for the peer's.
void stage_xfer(long my_slot, long peer_slot,
                long my_ready, long peer_ready, long peer_done,
                long counter, long in_ptr, long nbytes, long stream);

// Completion marker, issued after the math on the same stream: my_done =
// launch count. Stable arguments (pointers only), replay-safe.
void stage_mark_done(long my_done_ptr, long counter, long stream);

// Standalone reduction: out[e] = narrow(in[e] + peer[e]), fp32 ascending.
// in==out allowed (in place).
void stage_reduce(long in_ptr, long peer_ptr, long out_ptr, long long nelem,
                  int dtype, long stream);

// all_gather. plain=1: y = [mine, peer] in rank order (rank r writes both
// halves identically on both ranks). plain=0: row-interleaved placement with
// the given row width: y is [rows][2][row].
void stage_gather(long my_x, long peer_x, long y_ptr, long long nelem, int esize,
                  int rank, long long row, int plain, long stream);

// moe gather word kernel + inverse-table build for ar_gather_hc_write.
void stage_gather_inv(long inv_ptr, long sorted_ptr, long long T, long stream);
void stage_gather_rows(long ye_ptr, long ew_ptr, long sorted_ptr, long sh_ptr, long sg_ptr,
                       long inv_ptr, long y_ptr, long M, long n, long top_k, int sig,
                       long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                       long y_ld, int use_inv, long stream);

// hc write: h[r][c*n+i] += inj[r][c] * y[r][i], round-half-up narrow.
void stage_hc_write(long y_ptr, long inj_ptr, long h_ptr, long M, long n, long hc,
                    long irow, long hrow, long yrow, long stream);

// rmsnorm + block-fp8 quant: reads reduced bf16 in x (or scratch), optional
// residual add stored back, norm, per-group fp8 with f32 scales.
void stage_rmsnorm_quant_fp8(long x_ptr, long res_ptr, long w_ptr, long q_ptr, long s_ptr,
                             long ob_ptr, long M, long n, float eps, float wadd,
                             long group, long xpitch, long rpitch, long wpitch,
                             long qpitch, long spitch, long obpitch, int w_f32, long stream);

// layernorm(form RMS, see below) + Hadamard + int8 quant. Reads reduced bf16
// from red_ptr (scratch), x/res as the addends.
void stage_ln_had_quant_i8(long x_ptr, long red_ptr, long res_ptr, long w_ptr, long q_ptr,
                           long s_ptr, long ob_ptr, long M, long n, float eps, float wadd,
                           long had, long xpitch, long redpitch, long rpitch, long wpitch,
                           long qpitch, long obpitch, int w_f32, long stream);
