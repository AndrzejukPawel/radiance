/* bands.cpp -- the bucket table. spec.md §2.2.
 *
 * A parameter may be declared as a range rather than a value, because some of them are properties
 * of the batch and not of the model. M is the obvious one: a GEMM sees one row at decode,
 * sixty-four at a speculative verify and thousands in a prefill chunk, and libr4d has a different
 * kernel for each band. Resolving once at M = max_tok would pick the prefill kernel and then run
 * it at M = 1, which is exactly the loss that skinny-GEMM specialisation exists to recover.
 *
 * THE BUCKET BOUNDARIES ARE THE CONSTRAINT VALUES THEMSELVES. Nothing here picks a policy, a
 * power-of-two ladder or a bucket count: the kernels already said where their bands end, and this
 * file only collects what they said. A kernel whose fast band sits at an odd boundary gets that
 * boundary, because it said so.
 */
#include "registry.h"

#include <algorithm>

namespace rad {

namespace {

/* "M <= 64" for the first band, "M in (16, 64]" for a later one -- the half-open form is what a
 * band actually is, and a reader who sees only "<= 64" on the second row will read the table as
 * overlapping. */
std::string band_span(std::string_view key, int64_t lo_excl, int64_t hi, bool first) {
    if (hi == INT64_MAX) return std::string("any shape");   /* the op has no ranged parameter */
    if (first) return fmt("%.*s <= %lld", (int)key.size(), key.data(), (long long)hi);
    return fmt("%.*s in (%lld, %lld]", (int)key.size(), key.data(), (long long)lo_excl,
               (long long)hi);
}

}  /* namespace */

std::vector<Band> Registry::build_bands(std::string_view op, std::string_view ranged_key,
                                        int64_t lo, int64_t hi, const Geometry& base,
                                        const RowFilter* accept) {
    std::vector<Band> out;

    /* No ranged parameter: one band covering everything. hi is INT64_MAX rather than the declared
     * value so the issue path's band scan finds it for any issue count -- an op with no range has no
     * value to index by, and a table that can miss on an op that cannot miss is a trap. */
    if (ranged_key.empty()) {
        lo = 0;
        hi = INT64_MAX;
    } else if (hi < lo) {
        RAD_WARN("op '%s' declares the empty range %s in [%lld, %lld]; resolving the single value "
                 "%lld", std::string(op).c_str(), std::string(ranged_key).c_str(),
                 (long long)lo, (long long)hi, (long long)lo);
        hi = lo;
    }

    /* The union of the LE, GE and EQ values candidate kernels place on the ranged parameter, from
     * EVERY plugin and BOTH domains -- the boundaries are a property of the op's candidate set,
     * not of whichever kernel happens to win a given band. DIV and IN cut nothing: a kernel that
     * wants K divisible by 128 wants it at every value in the range, and set membership is not an
     * interval, so neither can be a band edge. */
    std::vector<int64_t> bounds;
    auto add = [&](int64_t b) {
        if (b >= lo && b < hi) bounds.push_back(b);   /* clamped to the declared range; hi closes
                                                       * the last band and is appended below */
    };
    if (!ranged_key.empty()) {
        if (const std::vector<const KernelRow*>* rows = rows_for(op)) {
            for (const KernelRow* r : *rows) {
                for (int i = 0; i < r->info->n_constraints; ++i) {
                    const RadConstraint& c = r->info->constraints[i];
                    if (!c.key || ranged_key != c.key) continue;
                    switch (c.op) {
                        case RAD_C_LE: add(c.ival);     break;
                        /* A kernel that starts at v ends the band below it at v-1, so the band it
                         * serves begins exactly where it said it does. */
                        case RAD_C_GE: add(c.ival - 1); break;
                        /* An EQ needs both edges, or the single value it serves would share a band
                         * with values it does not. */
                        case RAD_C_EQ: add(c.ival - 1); add(c.ival); break;
                        default: break;
                    }
                }
            }
        }
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    bounds.push_back(hi);

    /* Each band is resolved AT ITS UPPER BOUND, in both domains. The upper bound is the worst case
     * in the band, so a kernel that serves it serves the whole band; resolving at the lower bound
     * would hand a band to a kernel that cannot run its largest value. */
    int64_t prev = lo - 1;
    for (size_t i = 0; i < bounds.size(); ++i) {
        Band b;
        b.hi = bounds[i];

        Geometry g = base;
        if (!ranged_key.empty()) g.set_i(ranged_key, b.hi);

        SelectionTrace t;
        for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
            b.dom[dom] = resolve(op, g, dom, &t, accept);
            if (!b.dom[dom]) b.miss[dom] = t.reason;
        }

        b.span = band_span(ranged_key, prev, b.hi, i == 0);
        if (!b.dom[RAD_DOMAIN_DEVICE]) {
            /* Fatal, but not here: declare always completes, and the caller reports every miss at
             * the end rather than the first one at the top (§3.1). A range with a hole in it is
             * reported as that band, not as the whole op. */
            RAD_DEBUG("op '%s' %s: no device kernel -- %s", std::string(op).c_str(),
                      b.span.c_str(), b.miss[RAD_DOMAIN_DEVICE].c_str());
        }
        if (!b.dom[RAD_DOMAIN_HOST])
            b.miss[RAD_DOMAIN_HOST] += "; site 'host' is unavailable for this band";
        out.push_back(std::move(b));
        prev = bounds[i];
    }

    /* Absence is asymmetric. No device kernel is fatal and the caller says so; no host kernel
     * removes one placement option -- host-site execution needs a RAD_DOMAIN_HOST kernel, and
     * where none resolves the planner may not put this op's arithmetic on the host. The host
     * and SSD TIERS are untouched: a weight can still live there and stage to the device (§5.1).
     *
     * NEITHER IS REPORTED HERE. This function runs once per DECLARED OP, so a 64-layer model
     * calls it 64 times for the same op name, and a warning emitted here would print one line
     * per (layer, band) -- hundreds of identical sentences at startup. The band's miss string is
     * the record instead; Builder collects them and reports once, over the whole graph. */
    return out;
}

}  /* namespace rad */
