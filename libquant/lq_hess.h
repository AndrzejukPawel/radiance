/* lq_hess.h -- the calibration Hessians every calibrated quantiser reads, and the one product they
 * all spend their time in.
 *
 * WHAT A HESSIAN IS HERE. The engine's calibration tap (arch/common/rad_arch.h, calib_write) sums
 * x x^T over the real input of a layer's projection, per rank, into `gram.r<rank>.<stem>.bin`. The
 * input is the one the KERNEL multiplies, so a file says which DOMAIN it is in: the model's own
 * activation (domain 0), or that activation rotated by the unnormalised Hadamard of width N, block
 * by block (domain N) -- what a container that stores its weights rotated multiplies. A reader asks
 * for the domain its weight is quantised in, the grid's transform, and each file is brought there
 * before the sum: rotating is H -> F H F, and back is the same divided by N^2, F being its own
 * inverse up to N. So a calibration run on a bf16 model and one on a rotated four-bit container
 * both serve a rotated recipe. A layer-local quantiser's error is then the quadratic form
 *
 *     L = sum over rows r of (w_hat_r - w_r) H (w_hat_r - w_r)^T
 *
 * which decomposes over rows: two rows of a weight share H and nothing else. GPTQ's recursion,
 * AWQ's scale search, AutoRound's rounding descent and ParoQuant's rotation fit are four ways of
 * making that number small, and the last three spend their time in D . H, which is what mat_hess
 * computes.
 *
 * THE MOST SPECIFIC HESSIAN THERE IS: a routed expert's own, when the tap recorded one -- the
 * engine records one an expert for `down`, whose input is each expert's own intermediate units --
 * and otherwise the layer's, pooled over its experts and found from the weight's name with the
 * expert index stripped (calib_stem), which a layer's experts share and the process reads once.
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lq {

/* A Hessian or a factor of one as a quantiser holds it: shared, so one the process does not cache
 * -- an expert's own -- lives exactly as long as the call that read it. */
using Gram = std::shared_ptr<const std::vector<float>>;

/* `blk.7.ffn_gate_up_exps.123.weight` -> `blk.7.ffn_gate_up_exps`: the stem a layer's pooled
 * Hessian file is named by. */
std::string calib_stem(const char* name);

/* The stem `name`'s Hessian is read from in `dir`: the weight's own -- the name less `.weight`,
 * `blk.7.ffn_down_exps.123` -- when every rank's file for it together says at least
 * kOwnRowsPerK * K rows, and calib_stem's otherwise, said on stderr. A GPTQ fit to a Gram of few
 * rows fits those rows: on Qwen3.8-Flash-Next's down projection an expert's own Gram cuts its
 * held-out output error by 20-55% against the pooled one's, except for experts routed under ~2K
 * rows, where it is worse than rounding to nearest. Those carry a fraction of a percent of the
 * routing. */
constexpr int64_t kOwnRowsPerK = 4;
std::string calib_resolve(const char* dir, const char* name, int64_t K);

/* Every rank's file for `stem` in `dir`, each brought into `domain` (0, or the Hadamard width a
 * rotated grid quantises in), summed into `acc` (K x K, double, row-major). Returns how many files
 * were read; 0 when there are none. A file that is there and unusable -- the wrong magic, the wrong
 * K, short, a domain that cannot be brought to the one asked for -- is said so on stderr and
 * skipped. */
int calib_sum(const char* dir, const std::string& stem, int64_t K, int64_t domain,
              std::vector<double>* acc);

/* The summed Gram of `name`'s input in `domain` as f32 K x K, from calib_resolve's stem, or null
 * when `dir` holds none for it (said once on stderr). A layer's is cached for the life of the
 * process by (dir, stem, K, domain); an expert's own is read for the call that asked, since each
 * is used once and a model's worth would not fit. */
Gram calib_gram(const char* dir, const char* name, int64_t K, int64_t domain, const char* who);

/* G [R x K] = D [R x K] . H [K x K], H symmetric. Every element of G is summed over k in ascending
 * order whatever the thread count and the vector width, so the result is a function of the inputs
 * alone -- a quantiser's output is a stored file, and two converts of the same weight must agree.
 * Threads over column panels of H and row blocks of D. */
void mat_hess(const float* D, int64_t R, int64_t K, const float* H, float* G);

/* out[r] = D_r . G_r, the quadratic form of each row, summed in double. */
void row_dots(const float* D, const float* G, int64_t R, int64_t K, double* out);

}  /* namespace lq */
