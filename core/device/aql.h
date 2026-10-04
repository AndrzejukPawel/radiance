/* aql.h -- what the AQL backend's translation units share.
 *
 * Three files: aql_code.cpp reads code objects (the offload bundle, the ELF note, the msgpack
 * metadata inside it), aql.cpp owns the queues and everything submitted to them, and
 * aql_blit.hip holds the copy and fill kernels the backend dispatches for itself. The last one is
 * compiled as HIP, so nothing in this header may pull in HSA or anything HIP would object to.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace aql {

/* ------------------------------------------------------------------ code objects */

/* One kernel argument as the code object's metadata lays it out. `hidden` names the implicit
 * arguments the compiler appends after the declared ones; every other kind is an explicit
 * parameter, and those map one to one, in order, onto hipLaunchKernel's `kernelParams`. */
enum class Hidden : uint8_t {
    None,           /* an explicit parameter */
    Pad,            /* hidden_none: bytes the compiler reserved and reads nothing from */
    BlockCountX, BlockCountY, BlockCountZ,
    GroupSizeX, GroupSizeY, GroupSizeZ,
    RemainderX, RemainderY, RemainderZ,
    GlobalOffsetX, GlobalOffsetY, GlobalOffsetZ,
    GridDims,
    DynamicLds,
    Unsupported,    /* anything else: a kernel that wants it is refused by name */
};

struct KernelArg {
    uint32_t offset = 0;
    uint32_t size   = 0;
    Hidden   hidden = Hidden::None;
    std::string kind;              /* the metadata's value_kind, for the refusal message */
};

struct KernelMeta {
    std::string symbol;            /* "<name>.kd" */
    uint32_t kernarg_size = 0;
    uint32_t group_size   = 0;     /* static LDS */
    uint32_t private_size = 0;     /* static scratch */
    bool     dynamic_stack = false;
    std::vector<KernelArg> args;
};

/* The code object a card runs, out of a clang offload bundle, or an empty span with `*why` naming
 * what the bundle holds and what the card accepts. `isas` is the card's own list of the targets it
 * accepts, in the runtime's order of preference: "amdgcn-amd-amdhsa--gfx1036", then the family's
 * "amdgcn-amd-amdhsa--gfx10-3-generic", each carrying the modes the card runs in
 * ("gfx942:sramecc+:xnack-"). The bundle carries no total length, so it is read entry by entry from
 * its own header; a compressed bundle is refused rather than decompressed, because nothing in this
 * tree builds one. */
bool bundle_code_object(const void* bundle, const std::vector<std::string>& isas, const void** out,
                        size_t* out_size, std::string* why);

/* Every kernel described in an AMDGPU code object's metadata note. */
bool code_object_kernels(const void* elf, size_t size, std::vector<KernelMeta>* out,
                         std::string* why);

/* ------------------------------------------------------------------ fence policy
 *
 * What the next dispatch on this thread has to be fenced at, beyond the agent scope every kernel
 * gets. Set by the backend around its own blits and cleared by the dispatch that consumes it. */
enum : uint8_t { kFenceAcquireSystem = 1, kFenceReleaseSystem = 2 };
void set_next_fence(uint8_t f);

}  // namespace aql
}  // namespace rad

struct RadCopyRange;

/* The blit kernels' launchers, defined in aql_blit.hip. Each issues exactly one kernel onto
 * `stream` through hipLaunchKernel, which the backend intercepts. Alignment is the caller's: the
 * 16-byte forms need both addresses and the length 16-byte aligned. */
extern "C" {
/* `link` when either side is across the link (pinned host memory): one narrow workgroup instead
 * of a grid that fills the card; aql_blit.hip says why. */
int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream, bool link = false);
int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                        uint64_t width, uint64_t height, void* stream);
int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream);
/* rad_memcpy_ranges_async's kernel: `prefix` a multiple of sixteen and both bases 16-aligned. */
int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix, const RadCopyRange* ranges,
                        int n, void* stream);
}
