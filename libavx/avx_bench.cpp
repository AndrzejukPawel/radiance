/* avx_bench.cpp -- rad-avx-bench: what this machine can do, and what libavx gets of it.
 *
 * ============================== A SPEEDUP AGAINST libref IS NOT A RESULT ==============================
 *
 * libref is a nest of scalar loops with a per-element dtype switch and it says so in its own
 * header. "libavx is 14x libref" is therefore a statement about libref, and this tool refuses to
 * print it as though it were a statement about libavx. What it prints beside every op is the
 * fraction of THIS MACHINE'S measured ceiling the kernel reaches -- and the ceiling is measured
 * here, by --peak, rather than quoted off a spec sheet, because a spec sheet describes a part and
 * not the machine it is in: a memory channel population, a thermal cap or a firmware setting that
 * parks a link costs real throughput and shows up only under measurement.
 *
 * The libref column stays, because the ratio is what says an op is WIRED AT ALL, and because a
 * kernel that is slower than the reference is a real and findable defect. It is reported as what
 * it is.
 *
 * ============================== THE THINGS THAT MAKE A HOST BENCH LIE ==============================
 *
 * Each of these is a way a host benchmark quietly measures something other than the kernel:
 *
 *   AN INOUT OPERAND ACCUMULATES ACROSS ITERATIONS. A twenty-iteration loop over `rope` leaves the
 *   tensor rotated twenty-one times, and a harness that does not reset between iterations reports
 *   a large error for a correct kernel. Timing is not immune -- an op whose cost depends on its input
 *   (every sampler that narrows in place) measures something different each time round. h_reset
 *   restores the drawn bytes between iterations and its cost is excluded from the timing.
 *
 *   A CACHE-WARM LOOP CANNOT RANK A BANDWIDTH-BOUND KERNEL. Running the same 40 KB row a thousand
 *   times measures L1, so a warm loop ranks kernels by how they behave in cache and not by the
 *   bandwidth they need. --cold walks a working set larger than L3 between iterations; the default
 *   is warm and the column says which it is.
 *
 *   THE FIRST CALL IS NOT THE STEADY STATE. The thread_local scratch arenas allocate on their
 *   first use at a size, so iteration one includes a malloc per thread. Warm-up iterations run
 *   before the clock starts.
 *
 *   A MEDIAN, NOT A MEAN. One preempted iteration moves a mean and does not move a median, and on
 *   any machine that is also doing something else there is always one.
 */
#include "avx_harness.h"
#include "avx_isa.h"

#include <chrono>
#include <cinttypes>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* ================================================================== the machine's ceilings
 *
 * MEASURED, NOT QUOTED. Three numbers, and each is the ceiling for a different class of op here:
 *
 *   FMA THROUGHPUT bounds the GEMMs. A dependency-free chain of FMAs on enough accumulators to
 *   cover the latency, all in registers, nothing touching memory. What comes out is what the
 *   machine can retire, including whatever clock it actually holds under a vector load -- which on
 *   a part with an AVX-512 offset is not the advertised boost.
 *
 *   L1 BANDWIDTH bounds every elementwise op at a shape that fits, which is most of a decode step.
 *
 *   DRAM BANDWIDTH bounds everything at prefill shapes and bounds `logits_gemm` always: a 248K
 *   vocabulary weight does not fit in any cache, so that op is a memory benchmark wearing a GEMM's
 *   name whatever the flops say.
 *
 * All three are single-threaded by default and scale with --threads, because a per-core ceiling is
 * what an op that runs one row per thread is actually up against.
 */
static void bench_peak(Lib& avx, int threads) {
    std::printf("\n== machine ceilings (measured here, not quoted) ==\n");

    /* ---- FMA, PER ISA LEVEL, measured inside the plugin. It has to be: this file is
     * baseline-compiled like the dispatcher (see avx_dispatch.cpp) and can only emit SSE2, so a
     * ceiling computed here would be ten times below what an AVX-512 GEMM is actually up against
     * and every kernel would look like it was doing well. `avx_peak_fma` runs the probe compiled
     * with the level's own flags and hands back the FLOPS PERFORMED; the clock stays here, because
     * it is here that the median over repetitions is taken. */
    auto peak = (double (*)(int, int64_t))dlsym(avx.h, "avx_peak_fma");
    if (peak) {
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (avx.isa_detect && lvl > avx.isa_detect()) {
                std::printf("  FMA %-18s %8s              (not executable on this CPU)\n",
                            avx.isa_name(lvl), "n/a");
                continue;
            }
            std::vector<double> rates;
            for (int rep = 0; rep < 5; ++rep) {
                const int64_t iters = 1000000;
                const double t0 = now_s();
                const double fl = peak(lvl, iters);
                const double dt = now_s() - t0;
                rates.push_back(fl / dt);
            }
            /* The BEST of five, not the median: this is a CEILING, and a repetition that was
             * preempted or ran at a lower clock is not evidence the machine cannot do better. The
             * per-op numbers below take the median instead, for the opposite reason -- there the
             * question is what a kernel does in practice. */
            std::sort(rates.begin(), rates.end());
            std::printf("  FMA %-18s %8.2f GFLOP/s  (1 thread, in-register, no memory traffic)\n",
                        avx.isa_name(lvl), rates.back() / 1e9);
        }
    }

    /* ---- memory, at three working-set sizes. A read-modify-write triad (a[i] = b[i] + s*c[i])
     * because that is the traffic shape every elementwise op here has: two streams in, one out. */
    auto triad = [&](size_t bytes_per_array, const char* label) {
        const size_t n = bytes_per_array / sizeof(float);
        std::vector<float> A(n), B(n), C(n);
        for (size_t i = 0; i < n; ++i) { B[i] = (float)i * 1e-6f; C[i] = 1.0f; }
        std::vector<double> ts;
        for (int rep = 0; rep < 7; ++rep) {
            const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int64_t i = 0; i < (int64_t)n; ++i) A[(size_t)i] = B[(size_t)i] + 0.5f * C[(size_t)i];
            ts.push_back(now_s() - t0);
        }
        const double dt = median(ts);
        /* Three arrays touched: two read, one written. The write also reads the line unless the
         * store is non-temporal, which this is not -- so 3x is the honest count and 4x would be
         * counting a read-for-ownership the compiler may or may not have elided. */
        std::printf("  triad %-15s %8.2f GB/s     (%zu threads, %.0f KiB working set)\n", label,
                    3.0 * (double)n * sizeof(float) / dt / 1e9, (size_t)threads,
                    3.0 * (double)bytes_per_array / 1024.0);
    };
    triad(16 * 1024, "L1");
    triad(512 * 1024, "L2");
    triad(256u * 1024 * 1024, "DRAM");
}

/* ================================================================== conversion rates
 * The claim avx_support.cpp's header makes and does not carry the numbers for. Timed through the
 * PLUGIN's own `cast` op at each ISA level, which is the only way to time the conversion the
 * kernels actually use rather than a copy of it compiled here with different flags. */
static void bench_cvt(Lib& avx, Lib& ref) {
    std::printf("\n== dtype conversion, through `cast` (MB/s of SOURCE bytes) ==\n");
    const char* dts[] = { "bf16", "f16", "fp8_e4m3", "i8" };
    const int64_t N = 1 << 16;
    std::printf("  %-12s %10s %10s %10s %10s %10s\n", "pair", "scalar", "avx", "avx2", "avx512",
                "libref");
    for (const char* dt : dts)
        for (int dir = 0; dir < 2; ++dir) {
            const char* from = dir ? dt : "f32";
            const char* to   = dir ? "f32" : dt;
            Case c{ "cast", { HI("M", 1), HI("n", N), HS("from", from), HS("to", to) }, 0, 0 };
            const RadKernelInfo* kr = ref.row("cast");
            const RadKernelInfo* ka = avx.row("cast");
            if (!kr || !ka) continue;
            Prepared pr = h_prepare(ref, kr, c, kr, ka);
            if (!pr.ok) continue;
            const uint32_t sdt = pr.opd[0].d.dtype;
            const double src_bytes = (double)dt_bytes(sdt, N);
            char line[512];
            int off = std::snprintf(line, sizeof line, "  %-12s", (std::string(from) + "->" + to).c_str());
            for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
                if (avx.isa_force && avx.isa_force(lvl) != lvl) {
                    off += std::snprintf(line + off, sizeof line - off, " %10s", "n/a");
                    continue;
                }
                for (int w = 0; w < 20; ++w) ka->launch(&pr.ab, nullptr);
                std::vector<double> ts;
                for (int r = 0; r < 9; ++r) {
                    const double t0 = now_s();
                    for (int it = 0; it < 50; ++it) ka->launch(&pr.ab, nullptr);
                    ts.push_back((now_s() - t0) / 50);
                }
                off += std::snprintf(line + off, sizeof line - off, " %10.0f",
                                     src_bytes / median(ts) / 1e6);
            }
            {
                for (int w = 0; w < 3; ++w) kr->launch(&pr.aa, nullptr);
                std::vector<double> ts;
                for (int r = 0; r < 5; ++r) {
                    const double t0 = now_s();
                    kr->launch(&pr.aa, nullptr);
                    ts.push_back(now_s() - t0);
                }
                off += std::snprintf(line + off, sizeof line - off, " %10.0f",
                                     src_bytes / median(ts) / 1e6);
            }
            std::printf("%s\n", line);
        }
    if (avx.isa_force) avx.isa_force(AVX_N_LEVELS - 1);
}

/* ================================================================== per-op timing */
struct Row {
    std::string op, geom;
    double t_ref = 0;
    double t[AVX_N_LEVELS] = { 0, 0, 0, 0 };
    double bytes = 0, flops = 0;
};

static double time_launch(const RadKernelInfo* k, RadArgs* args, Prepared& pr, bool is_ref,
                          int warm, int iters, int reps) {
    for (int w = 0; w < warm; ++w) { k->launch(args, nullptr); h_reset(pr); }
    std::vector<double> ts;
    for (int r = 0; r < reps; ++r) {
        double acc = 0;
        for (int it = 0; it < iters; ++it) {
            const double t0 = now_s();
            k->launch(args, nullptr);
            acc += now_s() - t0;
            /* Outside the clock: an INOUT op must start from the drawn bytes every time or it is
             * timing a different computation each round. */
            h_reset(pr);
        }
        ts.push_back(acc / iters);
    }
    (void)is_ref;
    return median(ts);
}

int main(int argc, char** argv) {
    std::string ref_path = "libref.so", avx_path = "libavx.so", only_op;
    bool do_peak = false, do_cvt = false, do_ops = true;
    int threads = 1;
    int iters = 20, reps = 7;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if      (s == "--ref")  ref_path = next();
        else if (s == "--avx")  avx_path = next();
        else if (s == "--op")   only_op = next();
        else if (s == "--peak") { do_peak = true; do_ops = false; }
        else if (s == "--cvt")  { do_cvt = true; do_ops = false; }
        else if (s == "--all")  { do_peak = do_cvt = do_ops = true; }
        else if (s == "--threads") threads = std::atoi(next());
        else if (s == "--iters") iters = std::atoi(next());
        else if (s == "-h" || s == "--help") {
            std::printf(
                "rad-avx-bench -- what this machine can do, and what libavx gets of it.\n\n"
                "  --ref PATH      the reference plugin (default libref.so)\n"
                "  --avx PATH      the plugin under test (default libavx.so)\n"
                "  --peak          measure this machine's FMA and memory ceilings and stop\n"
                "  --cvt           measure the dtype conversions per ISA level and stop\n"
                "  --all           ceilings, conversions and every op\n"
                "  --op NAME       only this op\n"
                "  --threads N     OpenMP threads for the ceiling measurement (default 1)\n"
                "  --iters N       timed iterations per rep (default 20)\n");
            return 0;
        }
    }

#ifdef _OPENMP
    /* ONE THREAD BY DEFAULT, and that is a measurement decision rather than a timid one. Every
     * geometry in the case table is small -- deliberately, so the checker is fast -- and a
     * 32-thread fork/join over a 640-element row measures the barrier. The per-core number is what
     * ranks a kernel change; --threads is there for the ceiling, where the aggregate is the point. */
    omp_set_num_threads(1);
#endif

    Lib ref, avx;
    if (!ref.open(ref_path.c_str()) || !avx.open(avx_path.c_str())) return 2;
    std::printf("oracle : %s %s\nunder  : %s %s\n", ref.info->name, ref.info->version,
                avx.info->name, avx.info->version);
    if (avx.isa_detect)
        std::printf("cpu    : highest executable level is %s\n", avx.isa_name(avx.isa_detect()));

    if (do_peak) bench_peak(avx, threads);
    if (do_cvt)  bench_cvt(avx, ref);
    if (!do_ops) return 0;

    const std::vector<Case> cases = build_cases();
    std::vector<Row> rows;

    for (const Case& c : cases) {
        if (!only_op.empty() && only_op != c.op) continue;
        const RadKernelInfo* kr = ref.row(c.op);
        const RadKernelInfo* ka = avx.row(c.op);
        if (!kr || !ka) continue;
        Prepared pr = h_prepare(ref, kr, c, kr, ka);
        if (!pr.ok) continue;
        if (ka->launch(&pr.ab, nullptr) == RAD_E_UNSUPPORTED) continue;
        h_reset(pr);

        Row r;
        r.op = c.op;
        r.geom = case_str(c);
        r.bytes = c.bytes;
        r.flops = c.flops;
        /* libref gets fewer iterations: it is one to two orders slower and the whole run would be
         * spent in the oracle. Its median over three is stable enough for a ratio. */
        r.t_ref = time_launch(kr, &pr.aa, pr, true, 2, 3, 3);
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (avx.isa_force && avx.isa_force(lvl) != lvl) { r.t[lvl] = 0; continue; }
            r.t[lvl] = time_launch(ka, &pr.ab, pr, false, 5, iters, reps);
        }
        rows.push_back(r);
    }

    std::printf("\n== per op, microseconds per launch (median of %d reps x %d iters, 1 thread, "
                "cache-warm) ==\n", reps, iters);
    std::printf("%-22s %9s %9s %9s %9s %9s %8s  %s\n", "op", "libref", "scalar", "avx", "avx2",
                "avx512", "x(ref)", "rate");
    for (const Row& r : rows) {
        char rate[64] = "";
        const double best = r.t[AVX_N_LEVELS - 1] > 0 ? r.t[AVX_N_LEVELS - 1] : r.t[0];
        if (best > 0 && r.flops > 0)
            std::snprintf(rate, sizeof rate, "%.1f GFLOP/s", r.flops / best / 1e9);
        else if (best > 0 && r.bytes > 0)
            std::snprintf(rate, sizeof rate, "%.1f GB/s", r.bytes / best / 1e9);
        std::printf("%-22s %9.2f", r.op.c_str(), r.t_ref * 1e6);
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (r.t[lvl] > 0) std::printf(" %9.2f", r.t[lvl] * 1e6);
            else              std::printf(" %9s", "n/a");
        }
        std::printf(" %8.1f  %-14s  %s\n", best > 0 ? r.t_ref / best : 0.0, rate, r.geom.c_str());
    }
    std::printf("\nThe x(ref) column is a statement about libref, which is a scalar loop with a\n"
                "per-element dtype switch and says so. The rate column against the ceilings from\n"
                "--peak is the statement about libavx.\n");
    return 0;
}
