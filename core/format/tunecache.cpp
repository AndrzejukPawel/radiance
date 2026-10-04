/* tunecache.cpp -- read and write the tuning cache. The format is documented in tunecache.h and
 * that comment is the specification; this file must not drift from it.
 */
#include "tunecache.h"

#include <cstdlib>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rad {

namespace {
/* Whitespace only. See the note at its caller: rad_util.cpp's split_ws also splits on commas,
 * which is right for a constraint set and wrong for every line of this file. */
std::vector<std::string_view> split_ws_only(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t') ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}
}  /* namespace */

std::string TuneCache::rel_path(const std::string& machine) {
    return "tune/" + machine + ".tune";
}

std::string TuneCache::path_for(const std::string& home, const std::string& machine) {
    return home + "/" + rel_path(machine);
}

std::string TuneCache::now_iso8601() {
    time_t t = ::time(nullptr);
    struct tm g{};
    ::gmtime_r(&t, &g);
    char buf[32];
    ::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &g);
    return buf;
}

std::string TuneCache::canon_geometry(const Geometry& g) {
    /* Sorted by key so the same shape spells the same however the plugin ordered its params.
     * Geometry::str() is for humans and preserves declaration order; this one is a key. */
    std::vector<std::string> kv;
    const RadParam* p = g.params();
    for (int i = 0; i < g.n_params(); ++i) {
        if (p[i].kind == RAD_P_STR)
            kv.push_back(fmt("%s=%s", p[i].key, p[i].sval ? p[i].sval : ""));
        else if (p[i].kind == RAD_P_F64)
            /* %.17g so the key round-trips a double exactly. A float parameter is eps or a rope
             * theta; two shapes that differ only in eps are the same shape to a tuner, but they
             * are still different keys and a lossy spelling would silently merge them. */
            kv.push_back(fmt("%s=%.17g", p[i].key, p[i].dval));
        else
            kv.push_back(fmt("%s=%lld", p[i].key, p[i].ival));
    }
    std::sort(kv.begin(), kv.end());
    std::string out;
    for (auto& s : kv) { if (!out.empty()) out += ' '; out += s; }
    return out;
}

std::string TuneCache::canon_choice(const std::vector<std::pair<std::string, int64_t>>& c) {
    std::vector<std::string> kv;
    kv.reserve(c.size());
    for (const auto& e : c) kv.push_back(fmt("%s=%lld", e.first.c_str(), (long long)e.second));
    std::sort(kv.begin(), kv.end());
    std::string out;
    for (auto& s : kv) { if (!out.empty()) out += ','; out += s; }
    return out;
}

std::vector<std::pair<std::string, int64_t>> TuneCache::parse_choice(std::string_view s) {
    std::vector<std::pair<std::string, int64_t>> out;
    if (s == "-") return out;
    size_t i = 0;
    while (i < s.size()) {
        size_t comma = s.find(',', i);
        if (comma == std::string_view::npos) comma = s.size();
        std::string_view tok = s.substr(i, comma - i);
        const size_t eq = tok.find('=');
        if (eq != std::string_view::npos && eq > 0) {
            const std::string key(tok.substr(0, eq));
            const std::string val(tok.substr(eq + 1));
            /* strtoll rather than stoll: a malformed value is a dropped pair, not an exception on
             * a path whose whole contract is that a bad cache cannot stop a server. */
            char* end = nullptr;
            const long long v = std::strtoll(val.c_str(), &end, 10);
            if (end && *end == '\0' && !val.empty()) out.emplace_back(key, (int64_t)v);
        }
        i = comma + 1;
    }
    return out;
}

std::string TuneCache::index_key(std::string_view k, std::string_view p, std::string_view g) {
    std::string s;
    s.reserve(k.size() + p.size() + g.size() + 2);
    s.append(k); s.push_back('\x1f');
    s.append(p); s.push_back('\x1f');
    s.append(g);
    return s;
}

const TuneRow* TuneCache::find(std::string_view k, std::string_view p, std::string_view g) const {
    auto it = index_.find(index_key(k, p, g));
    return it == index_.end() ? nullptr : &rows_[it->second];
}

void TuneCache::put(const TuneRow& r) {
    std::string key = index_key(r.kernel, r.plugin, r.geom);
    auto it = index_.find(key);
    if (it != index_.end()) { rows_[it->second] = r; return; }
    index_.emplace(std::move(key), rows_.size());
    rows_.push_back(r);
}

int TuneCache::load(const std::string& home, const std::string& machine) {
    rows_.clear();
    index_.clear();
    machine_ = machine;

    const std::string path = path_for(home, machine);
    FILE* f = ::fopen(path.c_str(), "rb");
    if (!f) {
        /* No cache is the normal state of a fresh install and not a failure: a plugin's first
         * axis carries its own default, so nothing is broken, only untuned. */
        if (errno == ENOENT) return RAD_OK;
        RAD_WARN("%s: %s", path.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }

    char line[4096];
    int lineno = 0, bad = 0;
    bool saw_magic = false;

    while (::fgets(line, sizeof line, f)) {
        ++lineno;
        std::string_view s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.remove_suffix(1);
        if (s.empty() || s[0] == '#') continue;

        /* SPLIT ON WHITESPACE ONLY, NOT ON COMMAS. `split_ws` treats a comma as a separator too --
         * it is shared with the constraint parser, where "bf16 fp16,fp32" is one set -- and a
         * tuned choice is comma-separated by construction ("mb=2,npw=4,sk=8"). Splitting on commas
         * here turns one field into three and shifts `us` onto a timestamp, which parses as zero
         * and is then dropped as malformed, so every row with more than one axis disappears. It
         * disappears SILENTLY, because a cache that cannot be read is by design survivable. */
        auto tok = split_ws_only(s);
        /* A line of blanks is a blank line, and every branch below reads tok[0]. */
        if (tok.empty()) continue;
        if (!saw_magic) {
            if (tok.size() != 2 || tok[0] != "radtune") {
                RAD_WARN("%s:%d: not a rad-tune cache (expected 'radtune 2')", path.c_str(), lineno);
                ::fclose(f);
                return RAD_E_FORMAT;
            }
            if (tok[1] != "2") {
                RAD_WARN("%s: cache format version %.*s, this build reads 2 -- ignoring the cache",
                         path.c_str(), (int)tok[1].size(), tok[1].data());
                ::fclose(f);
                return RAD_E_FORMAT;
            }
            saw_magic = true;
            continue;
        }
        if (tok[0] == "machine") {
            if (tok.size() >= 2 && std::string(tok[1]) != machine)
                RAD_WARN("%s: measured on '%.*s' but this machine is '%s' -- the rows may not "
                         "describe this card", path.c_str(), (int)tok[1].size(), tok[1].data(),
                         machine.c_str());
            continue;
        }
        if (tok.size() < 5) { ++bad; continue; }

        TuneRow r;
        r.kernel   = std::string(tok[0]);
        r.plugin   = std::string(tok[1]);
        r.choice   = std::string(tok[2]) == "-" ? std::string() : std::string(tok[2]);
        r.us       = std::strtod(std::string(tok[3]).c_str(), nullptr);
        r.measured = std::string(tok[4]);
        for (size_t i = 5; i < tok.size(); ++i) {
            if (i > 5) r.geom += ' ';
            r.geom.append(tok[i]);
        }
        if (r.us <= 0.0) { ++bad; continue; }
        put(r);
    }
    ::fclose(f);

    if (!saw_magic) {
        RAD_WARN("%s: empty or missing the 'radtune 2' line", path.c_str());
        return RAD_E_FORMAT;
    }
    if (bad)
        RAD_WARN("%s: %d malformed row(s) ignored. A cache is an optimisation; a typo in it must "
                 "not stop a server.", path.c_str(), bad);
    RAD_DEBUG("tune cache %s: %zu row(s)", path.c_str(), rows_.size());
    return RAD_OK;
}

int TuneCache::save(const std::string& home) const {
    const std::string dir  = home + "/tune";
    const std::string path = path_for(home, machine_);

    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        RAD_ERR("%s: %s", dir.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }

    /* Write-then-rename, so an interrupted rad-tune leaves the previous cache intact rather than
     * a half-file the selector will warn about on every start. */
    const std::string tmp = path + ".tmp";
    FILE* f = ::fopen(tmp.c_str(), "wb");
    if (!f) { RAD_ERR("%s: %s", tmp.c_str(), std::strerror(errno)); return RAD_E_IO; }

    ::fprintf(f, "radtune 2\n");
    ::fprintf(f, "machine %s\n", machine_.c_str());
    ::fprintf(f, "# kernel plugin choice us measured geometry\n");

    /* Sorted, so the file is stable across runs and a diff shows what actually moved. */
    std::vector<const TuneRow*> v;
    v.reserve(rows_.size());
    for (const auto& r : rows_) v.push_back(&r);
    std::sort(v.begin(), v.end(), [](const TuneRow* a, const TuneRow* b) {
        if (a->kernel != b->kernel) return a->kernel < b->kernel;
        if (a->plugin != b->plugin) return a->plugin < b->plugin;
        return a->geom < b->geom;
    });

    for (const TuneRow* r : v) {
        if (r->kernel.find(' ') != std::string::npos ||
            r->plugin.find(' ') != std::string::npos ||
            r->choice.find(' ') != std::string::npos) {
            RAD_WARN("tune cache: skipping '%s' -- a name with a space in it cannot be written "
                     "to a whitespace-delimited row", r->kernel.c_str());
            continue;
        }
        ::fprintf(f, "%s %s %s %.6f %s %s\n", r->kernel.c_str(), r->plugin.c_str(),
                  r->choice.empty() ? "-" : r->choice.c_str(), r->us,
                  r->measured.empty() ? "-" : r->measured.c_str(), r->geom.c_str());
    }

    if (::fflush(f) != 0 || ::fclose(f) != 0) {
        RAD_ERR("%s: %s", tmp.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return RAD_E_IO;
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        RAD_ERR("%s -> %s: %s", tmp.c_str(), path.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return RAD_E_IO;
    }
    return RAD_OK;
}

}  /* namespace rad */
