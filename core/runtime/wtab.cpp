/* wtab.cpp -- issuing a weight-table patch. See wtab.h. */
#include "wtab.h"

#include "device/device.h"

namespace rad {

int wtab_patch(const WtabPatch& p, RadStream s) {
    if (p.n == 0) return RAD_OK;
    if (device_is_host()) {
        for (uint32_t i = 0; i < p.n; ++i) *p.dst[i] = p.val[i];
        return RAD_OK;
    }
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_wtab_patch_launch(&p, (void*)s) == 0 ? RAD_OK : RAD_E_DEVICE;
#else
    (void)s;
    return RAD_E_UNSUPPORTED;
#endif
}

}  /* namespace rad */
