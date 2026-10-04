/* lq_quant.h -- what every libquant quantiser shares: the grid's options as a recipe spells them,
 * reading a grid off a rule, and the table each algorithm file exports its quantiser through.
 *
 * lq_registry.cpp holds the plugin's exports and the quantisers that are a few lines each (rtn,
 * gptq, cast); an algorithm with a loop of its own (AWQ, AutoRound, ParoQuant) is a file, and its
 * RadQuantizerInfo comes back from the function named for it below.
 */
#pragma once
#include "lq_grid.h"

#include <cstdint>

/* The grid options, shared by every quantiser over a grid. */
#define LQ_GRID_OPTIONS                                                                          \
    { "codes", RAD_P_STR, nullptr,                                                               \
      "code dtype: i2..i8, i16, i32 (signed, symmetric), u1..u8, u16 with zero= (asymmetric), "  \
      "fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1" },                                      \
    { "table", RAD_P_STR, nullptr,                                                               \
      "a codebook the codes index: nf4, iq4nl, w4nl, w5nl, binary, ternary, or values "          \
      "separated by ;" },                                                                        \
    { "group", RAD_P_INT, nullptr, "one scale per this many columns of a row" },               \
    { "block", RAD_P_STR, "1x*", "the scale block RxC, * for a whole extent: 128x128, *x*" },    \
    { "scale", RAD_P_STR, "bf16",                                                              \
      "scale dtype: f32, bf16, f16, fp8_e4m3, e8m0; under scale2= also u4..u8, i6, i8" },        \
    { "scale2", RAD_P_STR, nullptr,                                                            \
      "a second scale level the first is relative to: f32, bf16, f16, fp8_e4m3, e8m0" },        \
    { "group2", RAD_P_INT, nullptr, "the second level's block, this many columns of a row" },  \
    { "block2", RAD_P_STR, nullptr, "the second level's block RxC (default *x*, the tensor)" }, \
    { "scale2_value", RAD_P_F64, nullptr,                                                      \
      "the second level fixed at this value, exact in scale2's dtype, for every block of it" },   \
    { "zero", RAD_P_STR, nullptr, "an integer zero point a block (asymmetric): u8, u4, ..." },  \
    { "transform", RAD_P_STR, nullptr, "fwhtN: rotate each row by the unnormalised Hadamard" }, \
    { "rule", RAD_P_STR, "absmax",                                                             \
      "absmax | mx (OCP shared exponent) | fixed | search (least squared error, imatrix-"        \
      "weighted when the converter has one)" },                                                 \
    { "scale_value", RAD_P_F64, nullptr, "the scale every block takes under rule=fixed" },     \
    { "clamp", RAD_P_STR, "full", "a signed code's range: full or sym" }

/* The calibration option every calibrated quantiser takes. */
#define LQ_CALIB_OPTION(WITHOUT)                                                                 \
    { "calib", RAD_P_STR, nullptr,                                                               \
      "the directory an engine calibration run wrote (RADIANCE_CALIB_DIR); without one, or for "  \
      "a weight it holds nothing for, " WITHOUT }

namespace lq {

/* The grid a rule's options describe, checked against the weight; a message on stderr, with the
 * weight's name, when it is refused. */
int grid_of(const RadParam* o, int n_o, const RadQuantWeight* w, Grid* g);

/* A rank-1 weight -- a norm, a bias -- is declined rather than quantised: nothing in a recipe
 * should have to spell out every vector a pattern happens to match. */
inline bool declines(const RadQuantWeight* w) { return !w || w->rank < 2; }

/* Say why a rule is refused, with the weight's name, and return RAD_E_INVAL. */
int refuse(const RadQuantWeight* w, const char* why);

/* A 64-bit hash of a string, for a seed that is the same on every machine and every run. */
uint64_t name_seed(const char* s);

/* The algorithm files' quantisers. */
const RadQuantizerInfo& awq_info();
const RadQuantizerInfo& autoround_info();
const RadQuantizerInfo& paroquant_info();

}  /* namespace lq */
