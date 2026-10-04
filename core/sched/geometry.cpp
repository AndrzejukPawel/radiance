/* geometry.cpp -- reconciling the linear-state chunk geometry with the attention block size.
 * See geometry.h for why this is computed and not configured.
 */
#include "geometry.h"

#include <cstring>

namespace rad {

static int64_t gcd64(int64_t a, int64_t b) {
    while (b) { int64_t t = a % b; a = b; b = t; }
    return a < 0 ? -a : a;
}
static int64_t lcm64(int64_t a, int64_t b) {
    if (a <= 0) return b;
    if (b <= 0) return a;
    return a / gcd64(a, b) * b;
}

/* WHICH PARAMETER OF THIS OP IS ITS SEQUENCE-CHUNK LENGTH, asked of the op's own schema. The
 * scheduler must cut its steps on a multiple of it or a linear-state chunk straddles a sequence
 * boundary, so this is the one parameter the core reads by MEANING rather than carrying opaquely.
 *
 * The schema names it (RAD_PROLE_SEQ_CHUNK), so the core holds no vocabulary of its own. Matching
 * a list of likely parameter spellings here instead would silently miss any op that spells it
 * differently, and a missed contributor is a quantum that does not divide the kernel's tile. */
static const char* chunk_key_of(const RadOpSchema* s) {
    if (!s) return nullptr;
    for (int i = 0; i < s->n_params; ++i)
        if (s->params[i].role == RAD_PROLE_SEQ_CHUNK) return s->params[i].key;
    return nullptr;
}

/* One contributor, appended to the explanation. Kept to a line each so the refusal prints as a
 * short list and not a wall. */
static void note(std::string& why, const char* what, int64_t v, const char* from) {
    if (!why.empty()) why += "; ";
    why += fmt("%s=%lld from %s", what, (long long)v, from);
}

int chunk_geometry_resolve(const Program& prog, const Config& cfg, ChunkGeometry* out) {
    if (!out) return RAD_E_INVAL;
    ChunkGeometry g;

    /* ---------------------------------------------------------------- attention block size */
    /* One lcm across every paged group. In practice every paged group on a given kernel set has
     * the same block size -- libr4d's is 16 for all of them -- but a hybrid model with a
     * sliding-window group served by a different kernel is exactly the case §7.2 is written for,
     * and lcm is the answer that is right for both. */
    bool has_linear = false;
    for (const KVGroupInfo& kg : prog.kv_groups) {
        if (kg.decl.kind == RAD_KV_FULL || kg.decl.kind == RAD_KV_WINDOW) {
            if (kg.block_size > 0) {
                g.attn_block = lcm64(g.attn_block, kg.block_size);
                note(g.why, "attn_block", kg.block_size, kg.name.c_str());
            }
        } else {
            has_linear = true;
        }
    }

    /* ---------------------------------------------------------------- linear-state chunk */
    /* Two sources, both already resolved: what the plugin declared the op with, and what the
     * kernel that won selection constrains. The first is the model's statement, the second is the
     * kernel's, and a chunk that is a multiple of both satisfies both. */
    for (const OpInfo& op : prog.ops) {
        const char* ckey = chunk_key_of(op.schema);
        if (!ckey) continue;
        for (const Band& b : op.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                const Resolved& r = b.dom[d];
                if (!r) continue;

                long long v = 0;
                if (r.geom.get_i(ckey, &v) && v > 0) {
                    g.state_chunk = lcm64(g.state_chunk, (int64_t)v);
                    note(g.why, ckey, v, op.op.c_str());
                }

                const RadKernelInfo* ki = r.row ? r.row->info : nullptr;
                if (!ki) continue;
                for (int i = 0; i < ki->n_constraints; ++i) {
                    const RadConstraint& c = ki->constraints[i];
                    if (!c.key || std::strcmp(c.key, ckey) != 0 || c.ival <= 0) continue;
                    switch (c.op) {
                        /* DIV and EQ are both "a legal chunk is a multiple of this". GE is folded
                         * in the same way rather than as a separate floor: a multiple of N is
                         * >= N, so one number carries both facts and the arithmetic stays lcm. */
                        case RAD_C_DIV:
                        case RAD_C_EQ:
                        case RAD_C_GE:
                            g.state_chunk = lcm64(g.state_chunk, (int64_t)c.ival);
                            note(g.why, c.key, c.ival, ki->name);
                            break;
                        case RAD_C_LE:
                            g.max_chunk = g.max_chunk ? (g.max_chunk < c.ival ? g.max_chunk
                                                                              : (int64_t)c.ival)
                                                      : (int64_t)c.ival;
                            break;
                        default: break;
                    }
                }
            }
        }
    }

    /* ---------------------------------------------------------------- reconcile */
    g.quantum = lcm64(g.attn_block, g.state_chunk);
    if (g.quantum < 1) g.quantum = 1;

    if (g.max_chunk > 0 && g.quantum > g.max_chunk) {
        /* Two kernels that cannot both be fed. Naming it is the whole value: the alternative is a
         * chunk that is legal for one of them and silently wrong for the other. */
        out->why = fmt("chunk quantum %lld exceeds the %lld a resolved kernel accepts (%s)",
                       (long long)g.quantum, (long long)g.max_chunk, g.why.c_str());
        out->quantum = g.quantum;
        out->min_max_tok = g.quantum;
        return RAD_E_SHAPE;
    }

    /* The checkpoint interval is the operator's number rounded up to a legal chunk length, so that
     * "split at the interval" is always also "split at a quantum". Rounding UP rather than down
     * because down can reach zero and because a longer interval costs retention, not correctness.
     *
     * No linear state means no checkpoints: the interval is a property of a state kind that this
     * model does not have, and pretending otherwise would split chunks for nothing. */
    if (has_linear && cfg.checkpoint_interval > 0)
        g.checkpoint_interval = align_up(cfg.checkpoint_interval, g.quantum);
    else
        g.checkpoint_interval = 0;

    /* One full quantum has to fit in one step. Below that, every chunk is short, no chunk ever
     * ends on a boundary, and the checkpoint lands wherever the step happened to end -- which is
     * precisely the vLLM failure §7.3 exists to remove. Refusing is the honest response. */
    g.min_max_tok = g.quantum;

    *out = g;

    if (cfg.max_tok < g.min_max_tok) {
        RAD_ERR("--max-num-batched-tokens is %lld but this model's chunk geometry needs at least "
                "%lld: %s", (long long)cfg.max_tok, (long long)g.min_max_tok, g.why.c_str());
        RAD_ERR("a smaller budget cannot end a chunk on a linear-state boundary, so every "
                "checkpoint would land in request-unique tokens and the linear prefix hit rate "
                "would be zero. Refusing rather than serving that.");
        return RAD_E_INVAL;
    }

    /* AND THE CHUNK AN OPERATOR ACTUALLY GETS, which is not --max-num-batched-tokens whenever the
     * checkpoint interval is smaller: `plan_chunk` splits at the interval so a checkpoint is
     * always written, and that clamp comes FIRST. Printing only the budget would hide it -- every
     * budget above the interval runs the same chunk length, so a sweep over them reads as
     * "prefill is flat against the chunk" when nothing about the chunk changed. The clamp itself
     * is correct; a budget that silently does nothing is what has to be visible. */
    const int64_t eff = (g.checkpoint_interval > 0 && g.checkpoint_interval < cfg.max_tok)
                      ? g.checkpoint_interval : cfg.max_tok;
    RAD_INFO("chunk geometry: quantum %lld (attn block %lld, state chunk %lld), checkpoint "
             "interval %lld, max_tok %lld -> prefill chunks of at most %lld%s",
             (long long)g.quantum, (long long)g.attn_block, (long long)g.state_chunk,
             (long long)g.checkpoint_interval, (long long)cfg.max_tok, (long long)eff,
             eff < cfg.max_tok ? " (the checkpoint interval, not the budget)" : "");
    return RAD_OK;
}

}  /* namespace rad */
