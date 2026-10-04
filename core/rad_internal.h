/* rad_internal.h -- shared by everything under core/, tools/ and tests/. NOT installed and NOT
 * visible to plugins: plugins see abi/ and nothing else. C++20 is fine here.
 */
#pragma once
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_builder.h"
#include "rad_runtime.h"
#include "rad_format.h"

#include <cstdio>
#include <cstdarg>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace rad {

/* ------------------------------------------------------------------ logging */
/* Normal operation is a few lines. Everything verbose is off by default (spec §16). */
enum class Log { Error = 0, Warn = 1, Info = 2, Debug = 3, Trace = 4 };

void log_set_level(Log l);
Log  log_level();
void log_emit(Log l, const char* file, int line, const char* fmt, ...)
     __attribute__((format(printf, 4, 5)));

/* WHICH RANK IS SPEAKING.
 *
 * Every per-rank component -- the pools, the planner, the mover, the sampler -- prints the same
 * sentence once per rank, so with nothing to distinguish them a tp2 startup prints each of those
 * lines twice with no way to tell which card it is about. The tag is a SCOPE rather than
 * an argument because the components doing the printing are three layers down, and a rank
 * threaded through every log call is a rank that will be missing from the next one.
 *
 * Off unless log_rank_tags(true) is called: at tp 1 there is nothing to disambiguate, and every
 * tool that is not the engine has one rank by construction. */
void log_rank_tags(bool on);

class LogRank {
public:
    explicit LogRank(int rank);
    ~LogRank();
    LogRank(const LogRank&) = delete;
    LogRank& operator=(const LogRank&) = delete;
private:
    int prev_;
};

#define RAD_LOG(lvl, ...)                                                        \
    do { if ((int)::rad::Log::lvl <= (int)::rad::log_level())                     \
             ::rad::log_emit(::rad::Log::lvl, __FILE__, __LINE__, __VA_ARGS__); } while (0)

#define RAD_ERR(...)   RAD_LOG(Error, __VA_ARGS__)
#define RAD_WARN(...)  RAD_LOG(Warn,  __VA_ARGS__)
#define RAD_INFO(...)  RAD_LOG(Info,  __VA_ARGS__)
#define RAD_DEBUG(...) RAD_LOG(Debug, __VA_ARGS__)
#define RAD_TRACE(...) RAD_LOG(Trace, __VA_ARGS__)

/* The same with the level as a VALUE rather than an enumerator name, for the sites where whether
 * something is a warning depends on what was found. */
#define RAD_LOG_AT(lvl, ...)                                                     \
    do { ::rad::Log _l = (lvl);                                                  \
         if ((int)_l <= (int)::rad::log_level())                                 \
             ::rad::log_emit(_l, __FILE__, __LINE__, __VA_ARGS__); } while (0)

/* Propagate a negative status, logging where it came from. */
#define RAD_TRY(expr)                                                            \
    do { int _s = (expr);                                                        \
         if (_s < 0) { RAD_DEBUG("%s -> %s", #expr, rad_strerror(_s)); return _s; } } while (0)

/* An unrecoverable condition: a device fault, a corrupt container, an invariant broken. Prints
 * and exits non-zero rather than serving from a wedged queue (spec §17). */
[[noreturn]] void fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

/* ------------------------------------------------------------------ small helpers */
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));

inline int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

/* Human-readable byte counts, for every report that prints a size. */
std::string humanb(int64_t bytes);

/* ONE MONOTONIC CLOCK, so that everything reasoning about elapsed time is reasoning about the
 * same elapsed time. The idle-session tiers write a timestamp when an entry is used and read
 * one when deciding what has gone cold; those two calls live in different translation units,
 * and two clocks there would make the thresholds mean nothing in particular. Monotonic and
 * not wall time: a tier that demoted an hour of sessions because NTP stepped the clock would
 * be correct by its own arithmetic and wrong by any other measure. */
uint64_t rad_mono_ns();

/* Split on whitespace -- what RAD_C_IN's set is encoded as. */
std::vector<std::string_view> split_ws(std::string_view s);

/* ------------------------------------------------------------------ geometry */
/* A resolved parameter set: the geometry after ranges are collapsed to a band's upper bound.
 * Owns its strings, because a RadParam's char* may point into a plugin the core outlives. */
class Geometry {
public:
    Geometry() = default;

    /* THE RadParam VIEW IS KEPT CURRENT BY EVERY WRITE, never built on first read. It holds char*
     * into this object's own strings, and a view rebuilt lazily from a const reader is a write
     * that two readers can make at once: the sizing declares of one rank all read the real
     * program's geometries side by side (Engine::probe_arena_levels), and one clearing the view
     * while another filled it is the heap corruption that crashed startup. So params() writes
     * nothing, and a Geometry nobody is writing can be read from any number of threads.
     *
     * A copy takes the storage and builds its own view, because the source's points into the
     * source. A move takes the view along with the storage: the vectors hand over their arrays,
     * so every string stays where it was and every pointer stays good. The moved-from object is
     * emptied, view included, so it is a valid empty geometry rather than one whose view names
     * strings it no longer owns. */
    Geometry(const Geometry& o)
        : keys_(o.keys_), svals_(o.svals_), ivals_(o.ivals_), dvals_(o.dvals_), kinds_(o.kinds_) {
        rebuild();
    }

    Geometry& operator=(const Geometry& o) {
        if (this != &o) {
            keys_ = o.keys_; svals_ = o.svals_; ivals_ = o.ivals_; dvals_ = o.dvals_;
            kinds_ = o.kinds_;
            rebuild();
        }
        return *this;
    }

    Geometry(Geometry&& o) noexcept
        : keys_(std::move(o.keys_)), svals_(std::move(o.svals_)),
          view_(std::move(o.view_)), ivals_(std::move(o.ivals_)),
          dvals_(std::move(o.dvals_)), kinds_(std::move(o.kinds_)) {
        o.clear();
    }

    Geometry& operator=(Geometry&& o) noexcept {
        if (this != &o) {
            keys_ = std::move(o.keys_); svals_ = std::move(o.svals_);
            view_ = std::move(o.view_);
            ivals_ = std::move(o.ivals_); dvals_ = std::move(o.dvals_);
            kinds_ = std::move(o.kinds_);
            o.clear();
        }
        return *this;
    }

    void set_i(std::string_view k, long long v);
    void set_s(std::string_view k, std::string_view v);
    /* eps, rope theta, an attention scale. Carried so they reach RadArgs.p, and NOT matched on:
     * constraint_holds() never sees a float, because a tolerance is not a predicate and a kernel
     * that discriminates on eps to that precision has a bug. */
    void set_f(std::string_view k, double v);
    bool get_i(std::string_view k, long long* out) const;
    bool get_f(std::string_view k, double* out) const;
    const char* get_s(std::string_view k) const;
    bool has(std::string_view k) const;

    /* A view suitable for RadArgs.p. Valid while this object lives and is not written. */
    const RadParam* params() const { return view_.data(); }
    int             n_params() const { return (int)view_.size(); }

    std::string str() const;      /* "M=64 N=8192 K=5120 dtype=w4a8", for diagnostics */

private:
    std::vector<std::string> keys_, svals_;
    std::vector<RadParam>    view_;
    void rebuild();
    void clear() noexcept {
        keys_.clear(); svals_.clear(); view_.clear(); ivals_.clear(); dvals_.clear();
        kinds_.clear();
    }
    std::vector<long long>   ivals_;
    std::vector<double>      dvals_;
    std::vector<int>         kinds_;
};

/* Does this geometry satisfy this constraint? The one place the matching rule lives.
 *
 * A float parameter can never satisfy one: get_i and get_s both miss a RAD_P_F64 slot, so a
 * constraint naming a float key does not hold, which is the correct answer. A tolerance is not a
 * predicate, and a kernel that discriminates on eps to that precision has a bug. */
bool constraint_holds(const RadConstraint& c, const Geometry& g);
/* Why not -- for the selection table's miss column (spec §16). */
std::string constraint_why(const RadConstraint& c, const Geometry& g);

}  /* namespace rad */
