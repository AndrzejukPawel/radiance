/* machine_key.cpp -- the machine half of the tuning cache key.
 *
 * Separate from tunecache.cpp on purpose: the cache is pure file I/O and must stay linkable and
 * testable without a device layer, while this needs one. It is also the only part of the key that
 * can be wrong in a way that is invisible -- a cache measured on another card is not a slower
 * answer, it is a wrong one -- so it names everything that changes the answer and nothing that
 * does not.
 */
#include "tunecache.h"

#include <cstring>

namespace rad {

std::string rad_machine_key() {
    RadDeviceProps p{};
    const int n = rad_dev_count();

    /* The host backend is a supported configuration and not a degraded one, so it gets a real key
     * rather than a placeholder: host kernels tune too, and their answer depends on the CPU. */
    if (n <= 0 || rad_dev_props(0, &p) < 0) {
#if defined(__x86_64__)
        return "host-x86_64";
#elif defined(__aarch64__)
        return "host-aarch64";
#else
        return "host-unknown";
#endif
    }

    /* Device count matters because a collective's variant depends on how many peers there are,
     * and those rows share this file. Card name is included, arch is not enough: two boards of the
     * same architecture can differ in clocks and memory width. */
    std::string name = p.is_host_backend ? "host" : p.name;
    for (char& c : name)
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
            c = '_';
    while (name.size() > 1 && name.back() == '_') name.pop_back();
    if (name.empty()) name = "unknown";

    const char* arch = p.arch[0] ? p.arch : "host";
    return fmt("%s-x%d-%s", arch, n, name.c_str());
}

}  /* namespace rad */
