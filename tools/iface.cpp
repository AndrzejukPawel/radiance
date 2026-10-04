/* iface.cpp -- the CLI plumbing every tool agrees about. Each tool parses its own flags: five
 * hand-rolled loops is less machinery than one options library, and each usage text is then
 * written for its own tool.
 *
 * The plugin loader and the declare phase are not here: they live in core/build/startup.h, because
 * they ARE the engine's startup -- rad-convert, rad-kbench and rad-tune are that startup with a
 * different last step. One copy only; two would eventually disagree about hierarchy order, which
 * is what decides the layout baked into a container.
 */
#include "iface.h"

#include <cstdlib>

namespace rad {

/* THE SAME DEFAULT THE ENGINE USES, spelled with the same macro.
 *
 * RAD_DEFAULT_HOME is the one definition of the fallback home, shared with core/config.cpp. It
 * must not be respelled here in terms of the install prefix: the engine's home is a directory
 * below that prefix, so a second spelling puts the tools and the engine one directory apart and
 * they find different plugin trees. The divergence is invisible while every run passes --home or
 * $RADIANCE_HOME explicitly, and appears the moment an installed prefix becomes a working home. */
std::string rad_home() {
    if (const char* e = std::getenv("RADIANCE_HOME")) if (*e) return e;
#ifdef RAD_DEFAULT_HOME
    return RAD_DEFAULT_HOME;
#else
    return ".";
#endif
}

std::vector<std::string> rad_split_list(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ':' || c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> rad_hierarchy_from_env() {
    const char* e = std::getenv("RADIANCE_KERNELS");
    return (e && *e) ? rad_split_list(e) : std::vector<std::string>{};
}

std::string rad_humanb_w(int64_t bytes, int w) {
    std::string s = humanb(bytes);
    if ((int)s.size() >= w) return s;
    return std::string((size_t)(w - s.size()), ' ') + s;
}

}  /* namespace rad */
