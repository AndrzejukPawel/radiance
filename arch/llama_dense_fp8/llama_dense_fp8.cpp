/* llama_dense_fp8 -- arch/llama_fp8 over a checkpoint that was never quantised.
 *
 * MiniCPM5-2B is the model this exists for: OpenBMB publishes it in bf16 and nothing else, so
 * "MiniCPM5-2B FP8A8" is a statement about what this engine runs rather than about what the
 * checkpoint holds. rad-convert does the quantising, by the recipe
 * data/recipes/minicpm5-2b-dspark.recipe -- one amax per 128x128 block, scale = amax/448 rounded to
 * bf16 -- so the container it writes holds what a checkpoint that shipped fp8 would, and every
 * kernel, every layout and every placement decision downstream is the same.
 *
 * A SEPARATE .so BECAUSE THE SELECTION KEY IS THE PAIR (spec §2.4): one plugin claims one
 * (architecture, quantisation), and the descriptor a dense checkpoint carries is the empty one.
 * Everything else is arch/llama_fp8/llama_fp8.cpp; `#if LLAMA_DENSE_SRC` there changes only the
 * descriptor check and what the graph dump says.
 */
#define LLAMA_DENSE_SRC 1
#define LLAMA_NS        llama_dense_fp8
#define LLAMA_QUANT     ""

#include "../llama_fp8/llama_fp8.cpp"
