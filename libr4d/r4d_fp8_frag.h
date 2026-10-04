// r4d_fp8_frag.h -- where a byte of a block-scaled FP8 WEIGHT lives once it is stored in WMMA
// FRAGMENT ORDER, and nothing else. Pure C++: no HIP, no r4d.h, so the host layout hook and the
// device kernels can include the same arithmetic instead of writing it twice and drifting.
//
// ---- why the weight is permuted at all -----------------------------------------------------
//
// The fp8a8 GEMMs read both operands through r4d_fp8::frag(), which wants lane `l` to hold ROW
// `l & 15` of a 16-row tile at k `(l >> 4) * 8`, eight bytes. A row-major weight cannot deliver
// that from a coalesced load: a wave's sixteen rows are `K` bytes apart, so asking for them
// together is sixteen separate transactions off a 256-byte cacheline, and reaches well under the
// rate a wide contiguous read does. That is why the kernels stage through LDS -- the LDS round
// trip IS the 16-row transpose -- and the staging costs a barrier, 36 KiB a block, and a second
// pass over every weight byte.
//
// Stored in fragment order the same bytes are already in lane order and none of it is needed.
//
// ---- the shape of it, and why it is 64 k and not 16 ------------------------------------------
//
// The obvious layout is one fragment: 16 rows x 16 k = 256 bytes, lane `l` taking bytes
// `l * 8`. A wave then asks for 256 contiguous bytes, which LOSES even to the staged path: a
// 256-byte request is a single cacheline and the memory system wants more in flight than one
// cacheline a wave. FOUR fragments together give each lane 32 contiguous bytes and the wave a
// 1 KiB request, which beats both the 8-byte form and LDS staging at every N and K in this family.
//
// So the unit is a (16 row, 64 k) TILE of 1024 bytes, laid out as 32 lanes x 32 bytes, and lane
// `l`'s 32 bytes are its four fragments back to back: bytes `8t` are fragment `t`, which covers
// k `t*16 + (l>>4)*8` of row `l & 15`.
//
// K must be a multiple of 64 and N a multiple of 16. Every fp8 weight in this family already
// satisfies both by a wide margin -- the constraint rows demand K and N divisible by 128, because
// that is the scale block -- so this adds no restriction anyone can hit.
#pragma once
#include <cstdint>

namespace r4d_fp8 {

enum { kFragRows = 16, kFragK = 16, kSuperK = 64, kTileBytes = 1024, kLaneBytes = 32 };

// Bytes one [N][K] weight occupies in fragment order: exactly N*K, since this is a permutation.
inline int64_t frag_bytes(int64_t N, int64_t K) { return N * K; }

// The 1 KiB tile holding row tile `rt` and k supertile `ks`, in bytes from the plane's base.
// `ksuper` is K / 64 -- the number of supertiles a row tile spans.
inline int64_t frag_tile_off(int64_t rt, int64_t ks, int64_t ksuper) {
    return (rt * ksuper + ks) * (int64_t)kTileBytes;
}

// ...and within that tile, the 32 bytes lane `l` owns. Its four fragments are at +0, +8, +16, +24.
inline int64_t frag_lane_off(int64_t lane) { return lane * (int64_t)kLaneBytes; }

// The source element a given (tile, lane, fragment, byte) came from, as a row and a k. This is the
// ONE definition of the permutation; the host transform walks it forwards and the device kernels
// index it backwards, and both are wrong together or right together.
inline void frag_src(int64_t rt, int64_t ks, int64_t lane, int64_t t, int64_t j,
                     int64_t* row, int64_t* k) {
    *row = rt * (int64_t)kFragRows + (lane & 15);
    *k   = ks * (int64_t)kSuperK + t * (int64_t)kFragK + (lane >> 4) * 8 + j;
}

}  // namespace r4d_fp8
