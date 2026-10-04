/* select.cpp -- the constraint selector, the selection table, and the per-instance init/fini.
 * spec.md §2.1 and §16.
 *
 * The selector answers one question: is there a kernel compiled for this geometry, in this domain,
 * in the highest plugin that has one. It is a NECESSARY condition and not a sufficient one --
 * strides, contiguity, alignment and buffer sizes are properties of a call rather than of a model
 * and stay in the entry point, which still rejects what it cannot run. libr4d's registry makes the
 * same point at length.
 */
#include "registry.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <cstring>

namespace rad {

namespace {

/* True when every constraint holds. Otherwise `why` names the FIRST one that did not and what it
 * wanted: "no kernel" on its own is not an answer anyone can act on, and the first failure is the
 * one a reader can fix -- listing all four of a row's unmet constraints buries it. */
bool row_matches(const KernelRow& r, const Geometry& g, const std::vector<std::string>& skip,
                 std::string* why) {
    for (int i = 0; i < r.info->n_constraints; ++i) {
        const RadConstraint& c = r.info->constraints[i];
        /* A constraint on a RAD_DERIVED parameter the caller did not name is not a predicate on
         * the question -- it IS the answer, read back afterwards. Matching on it would make the
         * kernel unreachable by the only query anyone can ask (rad_abi.h, spec §7.2). */
        if (c.key && std::find(skip.begin(), skip.end(), c.key) != skip.end()) continue;
        if (!constraint_holds(c, g)) {
            if (why) *why = constraint_why(c, g);
            return false;
        }
    }
    if (why) *why = "ok";
    return true;
}

/* The question, normalised so two spellings of the same one fold together in the table. The
 * geometry is displayed as asked, because a plugin author writes the parameters in the order that
 * reads best; only the fold key is sorted. */
std::string fold_key(std::string_view op, int domain, const Geometry& g) {
    const std::string q = g.str();     /* named: split_ws returns views into it */
    std::vector<std::string> kv;
    for (auto tok : split_ws(q)) kv.emplace_back(tok);
    std::sort(kv.begin(), kv.end());
    std::string k = std::string(op) + "/" + domain_name(domain) + "/";
    for (const std::string& s : kv) { k += s; k += ';'; }
    return k;
}

}  /* namespace */

bool kernel_constraint(const KernelRow& r, std::string_view key, int c_op, long long* out) {
    if (!r.info) return false;
    for (int i = 0; i < r.info->n_constraints; ++i) {
        const RadConstraint& c = r.info->constraints[i];
        if (c.op != c_op || !c.key || key != c.key) continue;
        if (out) *out = c.ival;
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ selection */
const KernelRow* Registry::select(std::string_view op, const Geometry& g, int domain,
                                  SelectionTrace* why, bool record, const RowFilter* accept) {
    SelectionTrace local;
    SelectionTrace& t = why ? *why : local;
    t = SelectionTrace{};
    t.op     = std::string(op);
    t.query  = g.str();
    t.domain = domain;

    /* The derived keys of this op that the caller did NOT supply. Derivedness is a property of the
     * op's schema, so it is known here and nowhere else; a caller that supplied one anyway has
     * pinned it, and it drops out of this list and matches like any other parameter. */
    std::vector<std::string> derived_keys;
    if (const RadOpSchema* sch = schema(op)) {
        for (int i = 0; i < sch->n_params; ++i) {
            if (sch->params[i].required != RAD_DERIVED || !sch->params[i].key) continue;
            if (!g.has(sch->params[i].key)) derived_keys.emplace_back(sch->params[i].key);
        }
    }

    const std::vector<const KernelRow*>* rows = rows_for(op);
    const KernelRow* best = nullptr;
    std::vector<std::string> misses;

    if (rows) {
        bool in_group = false;
        int  cur_order = 0;
        for (const KernelRow* r : *rows) {
            /* A kernel in the other domain answers a different question. It is not a candidate and
             * not a refusal -- but the fact that it exists is worth carrying, because "there is a
             * device kernel and no host one" is the sentence the placement planner needs (§5.1). */
            if (r->info->domain != domain) { t.other_domain_has_rows = true; continue; }

            if (!in_group || r->plugin_order != cur_order) {
                /* THE HIERARCHY IS ABSOLUTE. The first plugin with any matching kernel supplies
                 * it; the moment we would step down to the next plugin with a winner in hand, we
                 * stop. A lower-priority plugin never outbids a higher one no matter how
                 * specialised its kernel -- that is what makes a user override an override. */
                if (best) break;
                cur_order = r->plugin_order;
                in_group  = true;
            }

            std::string whynot;
            bool ok = row_matches(*r, g, derived_keys, &whynot);
            if (ok && accept && *accept) {
                whynot = (*accept)(*r, g);
                ok = whynot.empty();
            }
            if (ok) {
                t.candidates.push_back({ r->plugin, r->info->name, r->info->priority, true, "ok" });
                /* SPECIFICITY IS DECLARED, NOT DERIVED: the highest priority wins, and a tie goes
                 * to the row declared first. Counting matched constraints would get it backwards. */
                if (!best || r->info->priority > best->info->priority) best = r;
            } else {
                t.candidates.push_back({ r->plugin, r->info->name, r->info->priority, false, whynot });
                if (std::find(misses.begin(), misses.end(), whynot) == misses.end())
                    misses.push_back(whynot);
            }
        }
    }

    if (best) {
        t.kernel   = best->info->name;
        t.plugin   = best->plugin;
        t.priority = best->info->priority;
        /* The answer to the question that could not be asked: whatever the winner's RAD_C_EQ says
         * about each skipped derived key. A kernel with no EQ on one -- ref's `block_size >= 1`,
         * which admits every value -- leaves it UNKNOWN rather than zero, because "any" is not a
         * number and the caller has to decide rather than be handed a plausible one. */
        for (const std::string& key : derived_keys) {
            SelectionTrace::DerivedParam d;
            d.key   = key;
            d.known = kernel_constraint(*best, key, RAD_C_EQ, &d.value);
            t.derived.push_back(std::move(d));
        }
    } else if (!rows) {
        const std::string did = near_miss(op);
        t.reason = fmt("no loaded kernel plugin implements op '%s'%s", t.op.c_str(),
                       did.empty() ? "" : fmt("; did you mean '%s'?", did.c_str()).c_str());
    } else if (misses.empty()) {
        t.reason = fmt("no %s kernel for op '%s' in any plugin", domain_name(domain), t.op.c_str());
    } else {
        for (size_t i = 0; i < misses.size(); ++i) t.reason += (i ? "; " : "") + misses[i];
    }

    if (record) record_(t.op, domain, g, t);
    return best;
}

/* "gemm_nt:sk=8,mb=1;all_reduce:drain=3" -> the axes asked for `op`, or empty. Parsed on every
 * resolve rather than once: resolve runs a few thousand times at startup and never again, and a
 * parsed-once cache would need a mutex for a string that does not change. */
static std::string tune_pin_for(const char* spec, std::string_view op) {
    for (const char* p = spec; *p;) {
        const char* semi = std::strchr(p, ';');
        const char* end  = semi ? semi : p + std::strlen(p);
        if (const char* colon = (const char*)std::memchr(p, ':', (size_t)(end - p))) {
            if (op == std::string_view(p, (size_t)(colon - p)))
                return std::string(colon + 1, (size_t)(end - colon - 1));
        }
        p = semi ? semi + 1 : end;
    }
    return {};
}

/* For an error message, which is the whole value of refusing a bad pin rather than defaulting. */
static std::string tunable_names(const RadKernelInfo* k) {
    std::string s;
    for (int i = 0; i < k->n_tunables; ++i) {
        if (i) s += ", ";
        s += k->tunables[i].key;
        s += '=';
        for (int j = 0; j < k->tunables[i].n_values; ++j) {
            if (j) s += '|';
            s += std::to_string((long long)k->tunables[i].values[j]);
        }
    }
    return s.empty() ? "(nothing tunable)" : s;
}

/* Is `v` one of the values this axis declares? A cached or pinned value outside the list is not
 * applied: the axis list is the kernel's statement of what it can run, and honouring a number it
 * never offered is how a stale cache launches a shape the kernel cannot serve. */
static bool axis_allows(const RadTunable& t, int64_t v) {
    for (int i = 0; i < t.n_values; ++i) if (t.values[i] == v) return true;
    return false;
}

/* Does this combination survive the kernel's own cross-axis test? Null means yes. */
static bool choice_valid(const RadKernelInfo* k, const Geometry& g,
                         const std::vector<std::pair<std::string, int64_t>>& c) {
    if (!k->tune_valid) return true;
    std::vector<RadParam> cp;
    cp.reserve(c.size());
    for (const auto& e : c) {
        RadParam p{};
        p.key = e.first.c_str();
        p.kind = RAD_P_INT;
        p.ival = e.second;
        cp.push_back(p);
    }
    return k->tune_valid(g.params(), g.n_params(), cp.data(), (int)cp.size()) == 1;
}

Resolved Registry::resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why,
                           const RowFilter* accept) {
    SelectionTrace local;
    SelectionTrace& t = why ? *why : local;

    Resolved r;
    r.row = select(op, g, domain, &t, true, accept);
    if (!r.row) return r;

    /* The frozen geometry is the query PLUS what the kernel supplied for it. Writing the derived
     * value here rather than leaving it to the caller is what makes rad_kv_block_size() and
     * RadArgs.p agree: the kernel is launched with the block size it was compiled for, and the
     * block manager sizes itself from the same number. */
    r.geom = g;
    for (const SelectionTrace::DerivedParam& d : t.derived)
        if (d.known) r.geom.set_i(d.key, d.value);
    /* THE TUNED AXES. A kernel declares a SPACE (RadTunable) and this is where a point in it is
     * chosen: the cache when this shape has been measured on this machine, each axis's own
     * `deflt` otherwise -- so an untuned install is fast rather than broken (spec §15). The core
     * owns the cache and the policy, so no plugin has to.
     *
     * THE KEY IS THE FROZEN GEOMETRY BEFORE THE CHOICES GO IN, which is why `geom_key` is captured
     * here and the choices are folded into `geom` afterwards. rad-tune keys its rows on the
     * geometry the builder recorded; a key that included its own answer could never match. It also
     * has to sit after the derived parameters are written in, or the two spellings of one shape
     * would never meet.
     *
     * An axis the cached row does not mention keeps its default, and a key the row carries that
     * this kernel no longer declares is ignored. That is exactly what tunecache.h promises: a
     * library may add or widen an axis between builds without throwing away the measurements that
     * still apply, and a cache is an optimisation that must never stop a server. */
    r.geom_key = TuneCache::canon_geometry(r.geom);

    const RadKernelInfo* ki = r.row->info;
    if (ki->n_tunables > 0) {
        std::vector<std::pair<std::string, int64_t>> chosen;
        chosen.reserve((size_t)ki->n_tunables);
        for (int i = 0; i < ki->n_tunables; ++i)
            chosen.emplace_back(ki->tunables[i].key, ki->tunables[i].deflt);
        r.tune_source = "axis defaults";

        if (!tune_.empty()) {
            if (const TuneRow* tr = tune_.find(ki->name, r.row->plugin, r.geom_key)) {
                bool took = false;
                for (const auto& kv : TuneCache::parse_choice(tr->choice))
                    for (int i = 0; i < ki->n_tunables; ++i)
                        if (kv.first == ki->tunables[i].key &&
                            axis_allows(ki->tunables[i], kv.second)) {
                            chosen[(size_t)i].second = kv.second;
                            took = true;
                        }
                if (took) r.tune_source = "tuned " + tr->measured;
            }
        }

        /* RADIANCE_TUNE=op:key=value[,key=value][;op:...] pins axes for every kernel serving that
         * op. It is the debug half of the cache, and it exists because an axis is where a kernel's
         * fences and launch geometry live: the all-reduce's drain depth and acquire ordering are
         * axes, so questions like "is the missing acquire fence why this deployment is not
         * reproducible" are answerable without a rebuild.
         *
         * An unknown key, or a value the axis does not offer, is a HARD ERROR rather than a silent
         * default: a pin that did not take is worse than no pin, because the run looks like it
         * tested something. A kernel with nothing tunable is not a failed pin, though -- the same
         * op is served by libr4d on the device and by libref on the host, and libref declares no
         * axes for anything, so refusing here would take the host band out over a device pin. */
        if (const char* pin = std::getenv("RADIANCE_TUNE")) {
            const std::string want = tune_pin_for(pin, op);
            for (const auto& kv : TuneCache::parse_choice(want)) {
                int idx = -1;
                for (int i = 0; i < ki->n_tunables; ++i)
                    if (kv.first == ki->tunables[i].key) { idx = i; break; }
                if (idx < 0 || !axis_allows(ki->tunables[idx], kv.second)) {
                    RAD_ERR("RADIANCE_TUNE asks %s=%lld of op '%s', which kernel '%s' does not "
                            "offer. It has: %s", kv.first.c_str(), (long long)kv.second,
                            std::string(op).c_str(), ki->name, tunable_names(ki).c_str());
                    r.row = nullptr;
                    return r;
                }
                chosen[(size_t)idx].second = kv.second;
                r.tune_source = "RADIANCE_TUNE";
            }
        }

        /* A combination the kernel's own cross-axis test refuses is not launched. Falling back to
         * every axis's default is the one answer that cannot be wrong here: the defaults are what
         * an untuned install runs, so they are the kernel's own statement of what it can always
         * do. A cache written for a different build, or a pin that is individually legal on each
         * axis but not together, lands here. */
        if (!choice_valid(ki, r.geom, chosen)) {
            std::vector<std::pair<std::string, int64_t>> dflt;
            for (int i = 0; i < ki->n_tunables; ++i)
                dflt.emplace_back(ki->tunables[i].key, ki->tunables[i].deflt);
            if (r.tune_source != "axis defaults")
                RAD_WARN("kernel '%s' refuses the %s axes for %s; falling back to its defaults",
                         ki->name, r.tune_source.c_str(), r.geom_key.c_str());
            chosen = std::move(dflt);
            r.tune_source = "axis defaults";
        }

        for (const auto& kv : chosen) r.geom.set_i(kv.first, kv.second);
        r.choice = TuneCache::canon_choice(chosen);
    }

    /* scratch_bytes stays 0: the scratch hook takes a RadArgs, which needs the operand tensors,
     * and those belong to the builder. It sizes the arena; this layer only says which kernel. */
    return r;
}

/* ------------------------------------------------------------------ the selection table (§16) */
void Registry::record_(const std::string& op, int domain, const Geometry& g,
                       const SelectionTrace& t) {
    const std::string key = fold_key(op, domain, g);
    std::lock_guard<std::mutex> lk(mu_);
    for (SelectionRecord& s : log_) {
        if (s.key == key) { ++s.count; return; }   /* same question, therefore the same answer */
    }
    SelectionRecord s;
    s.op       = op;
    s.domain   = domain;
    s.query    = g.str();
    s.key      = key;
    s.kernel   = t.kernel;
    s.plugin   = t.plugin;
    s.priority = t.priority;
    s.reason   = t.reason;
    /* Kept apart from the query. A derived key was not part of the question -- printing it inside
     * the geometry would read as something the caller asked for, and the next reader would go
     * looking for the code that passed it. ":=" is the kernel answering, "?" is a winner with no
     * EQ on the key, which is a hole the caller has to fill and not a value. */
    for (const SelectionTrace::DerivedParam& d : t.derived) {
        if (!s.derived.empty()) s.derived += ' ';
        s.derived += d.known ? fmt("%s:=%lld", d.key.c_str(), d.value)
                             : fmt("%s:=?", d.key.c_str());
    }
    log_.push_back(std::move(s));
}

size_t Registry::n_selections() const {
    std::lock_guard<std::mutex> lk(mu_);
    return log_.size();
}

void Registry::clear_selections() {
    std::lock_guard<std::mutex> lk(mu_);
    log_.clear();
}

std::string Registry::selection_table() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (log_.empty()) return "selection table: the selector was never asked anything\n";

    size_t wop = 2, wdom = 3, wq = 8;
    long long total = 0;
    for (const SelectionRecord& s : log_) {
        wop = std::max(wop, s.op.size());
        wdom = std::max(wdom, std::strlen(domain_name(s.domain)));
        wq = std::max(wq, s.query.size());
        total += s.count;
    }

    std::string out = fmt("selection table -- %zu distinct questions, %lld asked\n",
                          log_.size(), total);
    out += fmt("  %-*s  %-*s  %5s  %-*s  %s\n", (int)wop, "op", (int)wdom, "domain", "asked",
               (int)wq, "geometry", "resolved");
    for (const SelectionRecord& s : log_) {
        std::string got = s.kernel.empty()
            ? fmt("-- none: %s", s.reason.c_str())
            : fmt("%s (%s, prio %d)%s%s", s.kernel.c_str(), s.plugin.c_str(), s.priority,
                  s.derived.empty() ? "" : "  derived ", s.derived.c_str());
        out += fmt("  %-*s  %-*s  %5lld  %-*s  %s\n", (int)wop, s.op.c_str(),
                   (int)wdom, domain_name(s.domain), s.count, (int)wq, s.query.c_str(),
                   got.c_str());
    }
    return out;
}

/* ------------------------------------------------------------------ instances */
/* init() once per resolved instance -- one op, one band, one domain -- with the frozen geometry,
 * the rank and the world size. Some kernels have per-instance state that must be established once
 * and not per launch: libr4d's all-reduce needs its peer signal buffers and IPC handles set up
 * across ranks before the first call, and a kernel that JITs or loads a config table wants
 * somewhere to do it that is not the hot path. */
int Registry::init_instance(Resolved& r, int rank, int world_size) {
    if (!r.row || !r.row->info) return RAD_E_INVAL;
    const RadKernelInfo* k = r.row->info;
    if (!k->init) { r.instance = nullptr; return RAD_OK; }

    void* inst = nullptr;
    const int st = k->init(r.geom.params(), r.geom.n_params(), rank, world_size, &inst);
    if (st < 0) {
        /* A negative init aborts startup, and the caller names the kernel -- which it can, because
         * the message below already did. Continuing would mean launching a kernel whose one-time
         * setup failed, and that is a fault three frames later instead of here. */
        RAD_ERR("kernel '%s' (%s) failed to initialise for %s: %s", k->name, r.row->plugin.c_str(),
                r.geom.str().c_str(), rad_strerror(st));
        return st;
    }
    r.instance = inst;
    /* Registered even when the kernel returned null: fini pairs 1:1 with a successful init, and a
     * kernel with global setup and no handle still needs its fini call. */
    std::lock_guard<std::mutex> lk(mu_);
    instances_.emplace_back(k, inst);
    return RAD_OK;
}

void Registry::fini_instances() {
    std::vector<std::pair<const RadKernelInfo*, void*>> take;
    {
        std::lock_guard<std::mutex> lk(mu_);
        take.swap(instances_);
    }
    for (auto it = take.rbegin(); it != take.rend(); ++it)
        if (it->first->fini) it->first->fini(it->second);
}

size_t Registry::n_instances() const {
    std::lock_guard<std::mutex> lk(mu_);
    return instances_.size();
}

}  /* namespace rad */
