/* lq_gptq.h -- error-feedback quantisation on any grid, against a calibration Hessian.
 *
 * GPTQ picks the codes of one column at a time and pushes the rounding error of that column into
 * the columns it has not reached yet, weighted by the inverse Hessian of the layer's real input.
 * It is the difference between a four-bit expert that serves and one that does not: plain
 * rounding treats every column as independent, and a 4-bit grid does not have the levels to
 * absorb that.
 *
 * WHAT IS HERE IS THE HESSIAN AND THE LOOP, not the grid: the codes are chosen on whatever grid
 * the rule names (lq_grid.h), against scales fixed BEFORE the loop from the (transformed) weight --
 * the "static groups" form, which keeps a scale plane a function of the weight alone.
 *
 * THE HESSIAN IS AN EXPERT'S OWN WHERE THE TAP RECORDED ONE, the layer's otherwise (lq_hess.h,
 * calib_resolve). A layer's pooled Hessian is found from the weight's logical name with an expert
 * index stripped, so the experts reading it share one factor and pay for it once; an expert's own
 * is factored for its one call. The engine's calibration tap writes both (arch/common/rad_arch.h,
 * RADIANCE_CALIB_DIR); docs/MOE-W4.md walks the bootstrap.
 */
#pragma once
#include "lq_grid.h"
#include "lq_hess.h"

namespace lq {

/* The upper Cholesky factor U of the damped inverse Hessian, U^T U = (H + lambda I)^-1, as a K x K
 * row-major f32 plane, for the weight named `name`. Null when `dir` has no calibration for it --
 * which is not an error, and the weight is then rounded to nearest -- and also null, with a
 * message, when a file is there and unusable, so a typo'd directory is not silently a quality
 * regression. With `perm` (gptq_act_order's) it is the factor of the permuted Hessian P H P^T.
 * `domain` is the grid's transform width, 0 without one (calib_sum). A layer's is cached for the
 * life of the process by (dir, stem, K, domain, damp, act order); an expert's own is not. */
Gram gptq_factor(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm = nullptr);

/* ACT ORDER: the columns by descending Hessian diagonal -- the input energy each one carries --
 * ties in column order. The identity when `dir` holds nothing for the weight. */
std::vector<int32_t> gptq_act_order(const char* dir, const char* name, int64_t K, int64_t domain);

/* Quantise `w` [N, K] in place on grid `g` against `scales` (grid_scales'), writing codes to
 * `codes` [N, K]. On return `w` holds the decoded values. With `u` null each value is rounded to
 * nearest on its own, which is byte-identical to grid_rtn. */
void gptq_quantise(const Grid& g, float* w, int32_t* codes, const Scales& scales, int64_t N,
                   int64_t K, const float* u);

/* COORDINATE DESCENT AFTER THE LOOP: `passes` sweeps over the columns of every row, each code
 * re-chosen on the grid at its conditional optimum -- the value that minimises the row's
 * e (H + lambda I) e^T, e = decoded - x, with every other column held -- against the same damped
 * Hessian GPTQ factored. GPTQ chooses each column once, before the columns after it are known;
 * a sweep revisits that choice with all of them in place, and never raises the objective.
 * `x` [N, K] is the weight GPTQ was handed (transformed), `w` its decoded result, updated in
 * place with `codes`. `h` is the undamped Gram (calib_gram's), null for nothing to do. */
void gptq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                 int64_t N, int64_t K, const float* h, double damp, int passes);


}  /* namespace lq */
