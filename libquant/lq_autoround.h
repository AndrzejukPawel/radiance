/* lq_autoround.h -- AutoRound's rounding descent, as a routine ParoQuant's second stage reuses. */
#pragma once
#include "lq_grid.h"

#include <string>

namespace lq {

/* Can AutoRound run on this grid? Integer codes, symmetric or with a zero, one scale level, blocks
 * of one row. Empty when it can, the reason when not. */
std::string autoround_refusal(const Grid& g);

/* Rows [0, N) of `w` [N, K] -- already in the grid's domain, transformed if it has a transform --
 * descended on for `iters` steps against H (K x K) or, when H is null, the diagonal `diag` (K), at
 * learning rate `lr`; the result's scales into `planes` (the grid's) and its codes too. */
void autoround_rows(const Grid& g, const float* w, int64_t N, int64_t K, const float* H,
                    const float* diag, int iters, double lr, const Planes& planes);

}  /* namespace lq */
