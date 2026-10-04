/* advance.cpp -- issuing an advance. See advance.h. */
#include "advance_impl.h"

#include "device/device.h"

namespace rad {

int step_advance(const AdvanceArgs& a, RadStream s) {
    if (a.n_seq <= 0) return RAD_OK;
    /* THE HOST BACKEND RUNS A STREAM'S WORK AS IT IS ISSUED, and its device memory is host memory,
     * so the rows run here, now, in the order the stream would have run them. */
    if (device_is_host()) {
        for (int i = 0; i < a.n_seq; ++i) adv_row(a, i);
        return RAD_OK;
    }
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_step_advance_launch(&a, (void*)s) == 0 ? RAD_OK : RAD_E_DEVICE;
#else
    (void)s;
    return RAD_E_UNSUPPORTED;
#endif
}

}  /* namespace rad */
