/* rad_util.cpp -- logging, error strings, dtype arithmetic and tensor helpers. The bedrock every
 * other translation unit links against.
 */
#include <chrono>
#include "../rad_internal.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <ctime>
#include <unistd.h>

namespace rad {

/* ------------------------------------------------------------------ logging */

/* THE LEVEL COMES FROM THE ENVIRONMENT, because nothing in the tree calls `log_set_level`. Without
 * that, the level is Info for the life of every process and every RAD_DEBUG and RAD_TRACE site in
 * the engine is unreachable code that reads like a diagnostic somebody could turn on -- the
 * sampler's mask timing, the grammar's admissible-token count, the drafter's per-position trace,
 * the step loop's batch dump. RADIANCE_LOG=error|warn|info|debug|trace reaches them.
 *
 * Read once, on first use, because this is called from the step path on every logged line. The
 * setter stays: a host embedding the engine as a library has no environment to speak through. */
static Log level_from_env() {
    const char* v = std::getenv("RADIANCE_LOG");
    if (!v || !*v) return Log::Info;
    if (!std::strcmp(v, "error")) return Log::Error;
    if (!std::strcmp(v, "warn"))  return Log::Warn;
    if (!std::strcmp(v, "info"))  return Log::Info;
    if (!std::strcmp(v, "debug")) return Log::Debug;
    if (!std::strcmp(v, "trace")) return Log::Trace;
    /* Not a level. Said at the level that is always on, rather than silently ignored -- a
     * misspelt knob that changes nothing looks exactly like a knob that does not work. */
    std::fprintf(stderr, "E RADIANCE_LOG='%s' is not error|warn|info|debug|trace; using info\n", v);
    return Log::Info;
}

static Log g_level = level_from_env();
void log_set_level(Log l) { g_level = l; }
Log  log_level()          { return g_level; }

static const char* level_tag(Log l) {
    switch (l) {
        case Log::Error: return "E";
        case Log::Warn:  return "W";
        case Log::Info:  return "I";
        case Log::Debug: return "D";
        default:         return "T";
    }
}

/* Colour only when stderr is a terminal. Piped or redirected it prints ordinary lines, which is
 * the same rule the live view follows (spec §16). */
static const char* level_colour(Log l) {
    if (!isatty(2)) return "";
    switch (l) {
        case Log::Error: return "\033[31m";
        case Log::Warn:  return "\033[33m";
        case Log::Debug: return "\033[36m";
        case Log::Trace: return "\033[90m";
        default:         return "";
    }
}

/* Thread-local because the ranks' declare threads run concurrently; a plain global would have
 * whichever rank last entered a scope tagging every other rank's lines. */
static thread_local int  g_log_rank  = -1;
static bool              g_rank_tags = false;

void log_rank_tags(bool on) { g_rank_tags = on; }
LogRank::LogRank(int rank) : prev_(g_log_rank) { g_log_rank = rank; }
LogRank::~LogRank() { g_log_rank = prev_; }

void log_emit(Log l, const char* file, int line, const char* fmt, ...) {
    const char* base = std::strrchr(file, '/');
    base = base ? base + 1 : file;

    char msg[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);

    const char* col = level_colour(l);
    const char* rst = col[0] ? "\033[0m" : "";

    char rank[16] = "";
    if (g_rank_tags && g_log_rank >= 0) snprintf(rank, sizeof rank, "[%d]", g_log_rank);

    /* Debug and Trace carry the source site; Info and above do not, because an operator reading
     * a normal startup does not care which file said it. */
    if ((int)l >= (int)Log::Debug)
        fprintf(stderr, "%s%s%s%s %s:%d  %s\n", col, level_tag(l), rank, rst, base, line, msg);
    else
        fprintf(stderr, "%s%s%s%s %s\n", col, level_tag(l), rank, rst, msg);
}

void fatal(const char* fmt, ...) {
    char msg[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%sradiance: fatal: %s%s\n", isatty(2) ? "\033[31m" : "", msg,
            isatty(2) ? "\033[0m" : "");
    fflush(stderr);
    _exit(70);
}

std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    std::string s;
    if (n > 0) {
        s.resize((size_t)n);
        vsnprintf(s.data(), (size_t)n + 1, f, ap2);
    }
    va_end(ap2);
    return s;
}

uint64_t rad_mono_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string humanb(int64_t b) {
    static const char* u[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)b;
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    return i == 0 ? fmt("%lld B", (long long)b) : fmt("%.2f %s", v, u[i]);
}

std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ',')) ++i;
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t' && s[j] != ',') ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

/* ------------------------------------------------------------------ Geometry */
/* Small, linear-scanned, and that is correct: an op has a handful of parameters and the selector
 * touches this at declare, not per step. A hash map here would be slower and less predictable. */
void Geometry::set_i(std::string_view k, long long v) {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k) { kinds_[i] = RAD_P_INT; ivals_[i] = v; dirty_ = true; return; }
    keys_.emplace_back(k);
    svals_.emplace_back();
    ivals_.push_back(v);
    dvals_.push_back(0.0);
    kinds_.push_back(RAD_P_INT);
    dirty_ = true;
}

void Geometry::set_f(std::string_view k, double v) {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k) { kinds_[i] = RAD_P_F64; dvals_[i] = v; dirty_ = true; return; }
    keys_.emplace_back(k);
    svals_.emplace_back();
    ivals_.push_back(0);
    dvals_.push_back(v);
    kinds_.push_back(RAD_P_F64);
    dirty_ = true;
}

void Geometry::set_s(std::string_view k, std::string_view v) {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k) { kinds_[i] = RAD_P_STR; svals_[i] = std::string(v); dirty_ = true; return; }
    keys_.emplace_back(k);
    svals_.emplace_back(v);
    ivals_.push_back(0);
    dvals_.push_back(0.0);
    kinds_.push_back(RAD_P_STR);
    dirty_ = true;
}

bool Geometry::get_i(std::string_view k, long long* out) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_INT) { if (out) *out = ivals_[i]; return true; }
    return false;
}

bool Geometry::get_f(std::string_view k, double* out) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_F64) { if (out) *out = dvals_[i]; return true; }
    return false;
}

const char* Geometry::get_s(std::string_view k) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_STR) return svals_[i].c_str();
    return nullptr;
}

bool Geometry::has(std::string_view k) const {
    for (auto& s : keys_) if (s == k) return true;
    return false;
}

void Geometry::rebuild() const {
    auto* self = const_cast<Geometry*>(this);
    self->view_.clear();
    self->view_.reserve(keys_.size());
    for (size_t i = 0; i < keys_.size(); ++i) {
        RadParam p{};
        p.key  = keys_[i].c_str();
        p.kind = kinds_[i];
        p.ival = ivals_[i];
        p.dval = dvals_[i];
        p.sval = kinds_[i] == RAD_P_STR ? svals_[i].c_str() : nullptr;
        self->view_.push_back(p);
    }
    dirty_ = false;
}

const RadParam* Geometry::params() const { if (dirty_) rebuild(); return view_.data(); }
int             Geometry::n_params() const { if (dirty_) rebuild(); return (int)view_.size(); }

std::string Geometry::str() const {
    std::string s;
    for (size_t i = 0; i < keys_.size(); ++i) {
        if (i) s += ' ';
        s += keys_[i];
        s += '=';
        if (kinds_[i] == RAD_P_STR)      s += svals_[i];
        else if (kinds_[i] == RAD_P_F64) s += fmt("%g", dvals_[i]);
        else                             s += fmt("%lld", ivals_[i]);
    }
    return s;
}

/* ------------------------------------------------------------------ constraint matching */
/* A constraint whose key the geometry does not carry does NOT hold. That is libr4d's rule and the
 * reason its two-shot all-reduce rows have to precede the one-shot ones: a query that omits `hops`
 * fails the row that needs it, so the more constrained row can never be reached unless it is
 * asked for. Making the absence a failure is what makes "ask for hops=2 to be handed the two-shot"
 * mean something. */
bool constraint_holds(const RadConstraint& c, const Geometry& g) {
    if (c.op == RAD_C_IN) {
        const char* v = g.get_s(c.key);
        if (!v) {
            /* An integer geometry against a string set: compare its decimal form, so
             * C_IN("causal", "0 1") works against causal=1. */
            long long iv;
            if (!g.get_i(c.key, &iv)) return false;
            std::string s = fmt("%lld", iv);
            for (auto tok : split_ws(c.sval ? c.sval : "")) if (tok == s) return true;
            return false;
        }
        for (auto tok : split_ws(c.sval ? c.sval : "")) if (tok == v) return true;
        return false;
    }

    long long v;
    if (!g.get_i(c.key, &v)) return false;
    switch (c.op) {
        case RAD_C_EQ:  return v == c.ival;
        case RAD_C_LE:  return v <= c.ival;
        case RAD_C_GE:  return v >= c.ival;
        case RAD_C_DIV: return c.ival != 0 && (v % c.ival) == 0;
        default:        return false;
    }
}

std::string constraint_why(const RadConstraint& c, const Geometry& g) {
    static const char* opname[] = { "==", "<=", ">=", "%", "in" };
    const char* on = (c.op >= 0 && c.op <= 4) ? opname[c.op] : "?";

    if (c.op == RAD_C_IN) {
        const char* v = g.get_s(c.key);
        long long iv;
        std::string have = v ? std::string(v)
                             : (g.get_i(c.key, &iv) ? fmt("%lld", iv) : std::string("<absent>"));
        return fmt("%s in {%s}, have %s", c.key, c.sval ? c.sval : "", have.c_str());
    }
    long long v;
    if (!g.get_i(c.key, &v)) return fmt("%s %s %lld, but the query did not name %s",
                                        c.key, on, c.ival, c.key);
    if (c.op == RAD_C_DIV) return fmt("%s %% %lld == 0, have %lld", c.key, c.ival, v);
    return fmt("%s %s %lld, have %lld", c.key, on, c.ival, v);
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the C surface
 *
 * EMPTY, AND THAT IS THE POINT. rad_strerror, the four dtype helpers, the three tensor helpers
 * and rad_args_geti/gets are `static inline` in the ABI headers. Defined here instead -- declared
 * in abi/, defined in rad_core -- they would be callable by everything EXCEPT the plugins that
 * want them most; see the long note above rad_strerror in rad_types.h for the failure that
 * motivates the arrangement and for what it costs.
 *
 * Nothing belongs here. A helper a plugin may need belongs in the header; a helper only the
 * core needs belongs in namespace rad above, not behind extern "C". */
