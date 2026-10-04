/* qwen35moe_fp8 -- Qwen3.5 / 3.6 MoE (35B-A3B), block-scaled FP8 weights and FP8 activations.
 *
 * IT IS arch/qwen35_fp8/qwen35_fp8.cpp WITH THE FEED-FORWARD ROUTED, and that is the whole of this
 * file: three macros and an include. Everything above the FFN -- the gated attention, the gated
 * delta net, the layer schedule, the rope, the vocabulary edges, the MTP head, every switch and
 * every bisect -- is identical between the two members of this family, and reading the two
 * config.json files side by side that is not an approximation. The six places where a routed
 * feed-forward differs are behind `#if QWEN35_MOE` in that file, each with its own comment.
 *
 * WHY A SECOND TRANSLATION UNIT AT ALL, rather than one plugin that branches on `n_expert`: the
 * architecture id and the quantisation descriptor are the SELECTION KEY (spec §2.4). A container
 * converted from `model_type: qwen3_5_moe` reports `qwen35moe`, the dense one reports `qwen35`, and
 * the loader picks by that pair. One .so cannot claim both ids, and it should not: an operator
 * looking at what is installed should see two plugins for two architectures.
 *
 * The precedent is libr4d's attention dispatches -- one set of kernel templates, three translation
 * units instantiating them at three geometries. The alternative is a second 1300-line plugin kept
 * in step by hand.
 */
#define QWEN35_MOE     1
#define QWEN35_NS      qwen35moe_fp8
#define QWEN35_ARCH_ID "qwen35moe"

#include "../qwen35_fp8/qwen35_fp8.cpp"
