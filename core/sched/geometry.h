/* geometry.h -- the chunk geometry, COMPUTED from the resolved kernels rather than documented as
 * a minimum an operator has to know.
 *
 * vLLM makes a hybrid model's operator pass a minimum `--max-num-batched-tokens` by hand, because
 * it inflates the attention block size until it matches the mamba page and then asserts the batch
 * budget covers it. The constraint is real; the fact that a human has to carry the number is not.
 * Here declare has already resolved every kernel, so the constraints that produce it are in hand:
 * the attention block size comes off the resolved attention kernel (§7.2) and the linear-state
 * chunk length off whichever resolved kernels mark a parameter RAD_PROLE_SEQ_CHUNK, and the
 * reconciliation is arithmetic.
 *
 * If the resolved geometry needs more tokens than the operator budgeted, that is said at startup
 * with the number and the engine refuses -- rather than running something subtly wrong, which here
 * means chunks that never land on a checkpoint boundary and a linear prefix hit rate of zero.
 */
#pragma once
#include "rad_core.h"

namespace rad {

struct ChunkGeometry {
    /* The attention block size, reconciled across every paged KV group. Taken from the resolved
     * kernel; libr4d's paged attention is compiled for 16 and rejects anything else. */
    int64_t attn_block = 1;

    /* The length a linear-state scan tiles at -- libr4d's gated delta net is the one that has
     * one, at 64. Scanned off the constraints of every op whose schema marks a parameter
     * RAD_PROLE_SEQ_CHUNK, and off the geometry the architecture plugin declared them with. */
    int64_t state_chunk = 1;

    /* What a NON-FINAL prefill chunk must be a multiple of: lcm(attn_block, state_chunk). A chunk
     * that ends here ends on an attention block boundary and on a linear chunk boundary at once,
     * so neither kernel is handed a partial tile it has to special-case. */
    int64_t quantum = 1;

    /* The linear-state checkpoint interval (§7.3), rounded UP from the operator's request to a
     * multiple of `quantum`, so splitting at an interval is automatically a legal chunk split.
     * Zero when the model has no linear state and checkpointing means nothing. */
    int64_t checkpoint_interval = 0;

    /* The smallest --max-num-batched-tokens this geometry can be served with. */
    int64_t min_max_tok = 1;

    /* An upper bound some kernel placed on the chunk length, or 0. Used only to detect a
     * contradiction, which is a kernel-set bug worth naming rather than clamping around. */
    int64_t max_chunk = 0;

    /* Every contributor by name, for the startup line and for the refusal. Built once. */
    std::string why;

    bool checkpoints() const { return checkpoint_interval > 0; }
};

/* Resolve the geometry from the declared program and the operator's budgets.
 *
 * Returns RAD_OK, or RAD_E_INVAL when cfg.max_tok cannot serve the geometry -- in which case the
 * required number is in out->min_max_tok and the reason in out->why, and the caller is expected to
 * print both and stop. Returns RAD_E_SHAPE when two kernels' constraints cannot be satisfied at
 * once, which is a kernel-set bug and not an operator one. */
int chunk_geometry_resolve(const Program& prog, const Config& cfg, ChunkGeometry* out);

}  /* namespace rad */
