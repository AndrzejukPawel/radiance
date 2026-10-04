/* wtab.h -- writing entries of a device weight table in stream order.
 *
 * A routed layer's weight table is a device array of expert pointers, and the mover changes a few
 * of its entries on most steps. The new values travel IN THE KERNEL'S ARGUMENTS, which the backend
 * copies when the dispatch is enqueued. A copy out of a host array would read that array when the
 * card reaches the copy instead -- and the host runs ahead of the card, so by then it may hold the
 * NEXT step's pointers, including one whose transfer that earlier step was never ordered behind.
 */
#pragma once
#include <cstdint>

#include "rad_device.h"

namespace rad {

/* One dispatch's worth of writes: dst[i] = val[i] for i < n. Sized so the argument block stays
 * inside the backend's per-dispatch argument slot with the implicit arguments beside it. */
constexpr int kWtabPatch = 24;

struct WtabPatch {
    uint32_t n   = 0;
    uint32_t pad = 0;
    void**   dst[kWtabPatch] = {};
    void*    val[kWtabPatch] = {};
};

/* Enqueue `p` on `s`. Under the host backend, whose streams run their work as it is issued and
 * whose device memory is host memory, the writes happen here. */
int wtab_patch(const WtabPatch& p, RadStream s);

}  /* namespace rad */

/* The launcher in wtab.hip: one kernel onto `stream` through hipLaunchKernel. */
extern "C" int rad_wtab_patch_launch(const rad::WtabPatch* p, void* stream);
