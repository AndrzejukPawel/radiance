/* tunecache.h -- the ONE tuning cache. rad-tune writes it, the selector reads it at declare.
 *
 * Constraint matching finds a kernel that is CORRECT for a geometry. Within one kernel there are
 * usually many tile configurations and the right one is a property of the shape and the machine
 * (spec §15). This file is the whole of the mechanism, and it is deliberately one file and one
 * format: the alternative is every plugin inventing its own tuning tool and its own file, and the
 * engine being unable to say why a shape is slow.
 *
 * ================================================================= THE FORMAT
 *
 *   $RADIANCE_HOME/tune/<machine>.tune
 *
 * One file per machine, because "which machine" is the coarsest key and a cache measured on
 * another card is not a slower answer, it is a wrong one. UTF-8 text, LF-terminated, because a
 * human has to be able to read it, diff it, delete one line of it, and check it into a repo
 * beside a deployment. It is small: one row per (kernel, shape) the declared graph asks for.
 *
 *   line 1        `radtune 2`                        magic and format version
 *   line 2        `machine <machine-key>`            must equal the reader's own key
 *   `#...`        comment, ignored
 *   otherwise     `<kernel> <plugin> <choice> <us> <measured> <geometry...>`
 *
 *     kernel     RadKernelInfo.name, the entry point. No spaces.
 *     plugin     RadPluginInfo.name that supplied it. No spaces.
 *     choice     the TUNED AXES that won, `key=value` comma separated and sorted by key -- for
 *                example `mb=3,npw=4,sk=8`. `-` for a kernel with nothing to tune. It is a point
 *                in the space RadTunable declares, not a name out of a list: a name would tie the
 *                cache to a spelling the library happens to use this release, and a library is
 *                free to add an axis or widen one between builds. An axis the reader no longer
 *                declares is ignored on lookup; one it declares and the row omits falls back to
 *                that axis`s default, so a cache written before an axis existed still helps.
 *     us         median microseconds per launch, `%.6f`.
 *     measured   ISO-8601 UTC, e.g. `2026-09-04T09:11:02Z`. --debug-graph prints it, because the
 *                age of a measurement -- taken before a driver or library changed underneath it --
 *                answers a large share of the questions a slow shape produces.
 *     geometry   the resolved parameters, `key=value` sorted by key, space separated. Ranges are
 *                ALREADY COLLAPSED to the band's upper bound -- a bucket table row is what was
 *                measured, so a bucket table row is what is keyed.
 *
 * The lookup key is (kernel, plugin, geometry). Not the op: two kernels can serve one op and they
 * tune independently. A row whose kernel no longer exists is kept on rewrite and
 * ignored on lookup, so downgrading a plugin does not throw away the measurements.
 *
 * A missing file is not an error. Every RadTunable carries its own `deflt`, so an untuned install
 * is fast rather than broken (spec §15), and this cache only ever moves a shape off those defaults.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace rad {

struct TuneRow {
    std::string kernel, plugin, choice, geom;
    double      us = 0.0;
    std::string measured;     /* ISO-8601 UTC, exactly as written */
};

class TuneCache {
public:
    /* Missing file -> RAD_OK with an empty cache. Malformed line -> skipped with a Warn naming
     * the line number; a cache is an optimisation and a typo in it must not stop a server. */
    int load(const std::string& radiance_home, const std::string& machine);
    int save(const std::string& radiance_home) const;

    const std::string& machine() const { return machine_; }
    void set_machine(std::string m) { machine_ = std::move(m); }

    const TuneRow* find(std::string_view kernel, std::string_view plugin,
                        std::string_view geom) const;
    void put(const TuneRow& r);

    const std::vector<TuneRow>& rows() const { return rows_; }
    bool empty() const { return rows_.empty(); }

    /* `key=value` sorted by key, space separated. The canonical spelling of a shape; every
     * component that keys on one must go through here or two of them will disagree. */
    static std::string canon_geometry(const Geometry& g);

    /* THE `choice` COLUMN, both directions. One spelling, in one place, because the tuner writes
     * it and the selector reads it and a disagreement between them is a cache that silently never
     * hits. `key=value` comma separated, sorted by key. */
    static std::string canon_choice(const std::vector<std::pair<std::string, int64_t>>& c);
    /* Unparseable pairs are DROPPED, not an error: an axis this reader does not understand is the
     * forward-compatibility case tunecache.h's header promises, and the caller filters by the keys
     * its kernel actually declares anyway. */
    static std::vector<std::pair<std::string, int64_t>> parse_choice(std::string_view s);

    static std::string path_for(const std::string& radiance_home, const std::string& machine);
    /* The same path relative to a home, for finding the first home on a search path that has one. */
    static std::string rel_path(const std::string& machine);
    static std::string now_iso8601();

private:
    static std::string index_key(std::string_view kernel, std::string_view plugin,
                                 std::string_view geom);

    std::string                                    machine_;
    std::vector<TuneRow>                           rows_;
    std::unordered_map<std::string, size_t>        index_;
};

/* The machine half of the key. Derived from the device layer, so it changes when the card does.
 * Lives beside the cache and not inside it, because the cache is pure file I/O and testable
 * without a device. */
std::string rad_machine_key();

}  /* namespace rad */
