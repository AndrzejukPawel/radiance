// r4d_hc_fp8.h -- the STORED ROW GEOMETRY of hc_read's two mixing matrices at E4M3, and nothing
// else. Pure C++: no HIP, no r4d.h, so the host layout hook (r4d_fp8_layout.cpp), the kernels
// (r4d_hc_bf16.hip) and the selftest derive a row from the same lines.
//
// A stored row is [K E4M3 codes][K/G f32 scales][pad to 16 bytes], one row after another, for the
// reason r4d_rad_layout_fp8_logits gives: the scales ride in the row that uses them, so a row is
// self-contained and a kernel finds its scales at a fixed offset from its codes.
//
// THE GROUP IS A PROPERTY OF THE KERNEL THAT READS THE ROW, not a tuning choice, and the two
// matrices get different ones for that reason:
//
//   mix_down  [lowrank rows, K = hc*n]   G = 128. A wave walks the row in 256-element steps with
//             eight codes a lane, so every lane's eight sit inside one group and a step needs two
//             scales a wave.
//   mix_up    [hc*n rows, K = lowrank]   G = K/4. Four threads share a row, each taking one
//             contiguous quarter of it -- 80 codes at lowrank 320 -- so a thread's quarter IS its
//             group and it multiplies by one scale after its whole partial sum. lowrank must be a
//             multiple of 64 for the quarter to be sixteen-code loads.
//
// Both are FINER than the 128 x 128 blocks the GEMMs use, which is deliberate: these rows are
// read against a normalised stream whose every element matters, and a per-row group costs one f32
// per G codes -- 3.1% of mix_down's bytes and 5% of mix_up's.
#pragma once
#include <cstdint>

namespace r4d_hc8 {

constexpr int64_t kDownGroup = 128;

// constexpr and not merely inline, which is what makes them callable from device code too: the
// compiler treats a constexpr function as both host and device.
constexpr int64_t down_group(int64_t /*K*/) { return kDownGroup; }
constexpr int64_t up_group(int64_t K) { return K / 4; }

// Whether a K can be stored with the given group at all. The kernels' loads are what decide it:
// sixteen codes at a time, a group a whole number of those.
constexpr bool group_ok(int64_t K, int64_t G) { return G > 0 && G % 16 == 0 && K % G == 0; }

// The BYTES one stored row occupies, and where its pad starts. The row starts 16-byte aligned so
// every code load is a uint4 or a uint2 at a 16- or 8-byte boundary.
constexpr int64_t row_used(int64_t K, int64_t G) { return K + 4 * (K / G); }
constexpr int64_t row_bytes(int64_t K, int64_t G) { return (row_used(K, G) + 15) & ~(int64_t)15; }

}  // namespace r4d_hc8
