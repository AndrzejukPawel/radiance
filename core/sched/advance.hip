/* advance.hip -- the advance kernel: one thread a row of the batch it patches. See advance.h.
 *
 * Launched through hipLaunchKernel like any kernel library's, which the AQL backend intercepts, so
 * it is one dispatch on the rank's own queue in stream order with the pass it belongs to.
 */
#include <hip/hip_runtime.h>

#include "advance_impl.h"

namespace {

constexpr int kThreads = 64;

__global__ __launch_bounds__(kThreads) void rad_step_advance(rad::AdvanceArgs a) {
    const int i = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (i < a.n_seq) rad::adv_row(a, i);
}

}  // namespace

extern "C" int rad_step_advance_launch(const rad::AdvanceArgs* args, void* stream) {
    if (!args || args->n_seq <= 0) return 0;
    const unsigned blocks = (unsigned)((args->n_seq + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(rad_step_advance, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                       *args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
