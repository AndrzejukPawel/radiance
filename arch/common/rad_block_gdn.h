/* rad_block_gdn.h -- one gated-delta-net (linear attention) block, bf16 operands.
 *
 * Hybrid models are the normal case, not an extension (spec §7.2): Qwen3.5-27B is 48 GDN layers to
 * 16 full-attention ones. This block is the GDN half, and it composes with AttnGatedBF16 and
 * MlpBF16 in one plugin exactly as they compose with each other.
 *
 *   rmsnorm -> in gemm ([q|k|v|a|b]) -> z gemm
 *     prefill: gdn_conv_prep -> gdn_kkt_solve -> gdn_chunk_scan -> gdn_gated_rmsnorm
 *     decode:  gdn_conv_update -> gdn_recurrent_update       (which folds the gated norm)
 *   -> out gemm -> [all_reduce] -> residual add
 *
 * THE TWO PATHS ARE A BRANCH ON PHASE, WHICH IS NOT THE BRANCH SPEC §2.4 FORBIDS. A quant branch
 * makes the graph unreadable because it hides which of two op sequences a model uses; this branch
 * picks between two sequences that are both declared, both resolved, both in the graph dump, and
 * both run by the same model on different steps. The chunked path amortises across a prefill chunk
 * and the recurrent path is O(1) per token; there is no single kernel that is right for both.
 *
 * Both paths take `num_accepted`, because a linear-attention kernel pair MUST support speculative
 * rollback: the recurrent state and the conv window have already absorbed rejected tokens, and a
 * rejection is a change of read offset rather than a recompute (spec §10, OPS.md).
 *
 * ============================== TWO CONVENTIONS TO GET RIGHT ==============================
 *
 * `A_log` IS A LOG AND THE EXPONENT HAS NOT BEEN TAKEN. The decay is
 *
 *     g = -exp(A_log) . softplus(a + dt_bias)
 *
 * and the kernels -- libr4d's and libref's alike -- take the exponent themselves. The
 * safetensors checkpoint stores `linear_attn.A_log` in exactly that form, so this block declares
 * the weight as `ssm_a_log` and maps it straight through. THE GGUF CONVERSION DOES NOT: llama.cpp
 * writes `ssm_a = -exp(A_log)` at convert time, so a container built from a GGUF must undo that or
 * the decay becomes -exp(-exp(A_log)), which is a number near -1 for every head and a model that
 * forgets everything in four tokens. The name carries the convention so the mismatch is visible in
 * the weight manifest rather than in the perplexity.
 *
 * q AND k ARE L2-NORMALISED, NOT RMS-NORMALISED. Different arithmetic -- no division by the head
 * width and no gain vector -- and it lives inside gdn_conv_prep (prefill) and
 * gdn_recurrent_update (decode) because both kernels already have the head in registers. There is
 * no l2_norm op to declare and there should not be one.
 *
 * ============================== THE FUSED IN-PROJECTION ==============================
 *
 * gdn_conv_prep's `x` is the whole projection row [q | k | v | a | b]: conv_dim convolved channels
 * followed by one decay input and one beta input per value head. So the three checkpoint tensors
 * in_proj_qkv, in_proj_a and in_proj_b are ONE declared weight and ONE GEMM, which is also one
 * movement unit and one launch instead of three. The cost is the ability to place them
 * separately, which nothing wants.
 */
#ifndef RAD_BLOCK_GDN_H
#define RAD_BLOCK_GDN_H

#include "rad_arch.h"

#include <cmath>

namespace rad {
namespace arch {

struct GdnBF16 {
    struct Config {
        int64_t n_head_k   = 0;    /* per rank -- q and k share these */
        int64_t head_k     = 0;    /* the recurrent state's key width */
        int64_t n_head_v   = 0;    /* per rank; a multiple of n_head_k, 3x it on this model */
        int64_t head_v     = 0;
        int64_t conv_width = 4;
        int64_t chunk      = 64;   /* the chunked-scan interval, and the checkpoint interval */

        int64_t k_dim()    const { return n_head_k * head_k; }
        int64_t v_dim()    const { return n_head_v * head_v; }
        int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
        int64_t in_dim()   const { return conv_dim() + 2 * n_head_v; }   /* ... + a + b */
    };

    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]           the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]           normed input, then the out projection */
        rad_buf in   = 0;   /* [max_tok, in_dim]           [q|k|v|a|b], pre-convolution */
        rad_buf z    = 0;   /* [max_tok, v_dim]            the output gate */
        rad_buf q    = 0;   /* [max_tok, n_head_k, head_k] post-convolution, l2-normed */
        rad_buf k    = 0;   /* [max_tok, n_head_k, head_k] */
        rad_buf v    = 0;   /* [max_tok, n_head_v, head_v] */
        rad_buf gdec = 0;   /* [max_tok, n_head_v]         the resolved decay, per-chunk cumsum */
        rad_buf beta = 0;   /* [max_tok, n_head_v] */
        rad_buf kkt  = 0;   /* [max_tok, n_head_v, chunk]  the within-chunk inverse */
        rad_buf o    = 0;   /* [max_tok, n_head_v, head_v] */
    };

    /* Checkpoint-side names for one layer; the plugin owns the spelling (spec §2.4). */
    struct Src {
        const char* norm     = nullptr;   /* input_layernorm.weight */
        const char* in_qkv   = nullptr;   /* linear_attn.in_proj_qkv.weight */
        const char* in_a     = nullptr;   /* linear_attn.in_proj_a.weight */
        const char* in_b     = nullptr;   /* linear_attn.in_proj_b.weight */
        const char* in_z     = nullptr;   /* linear_attn.in_proj_z.weight */
        const char* conv1d   = nullptr;   /* linear_attn.conv1d.weight */
        const char* a_log    = nullptr;   /* linear_attn.A_log  -- a LOG, see the header comment */
        const char* dt_bias  = nullptr;   /* linear_attn.dt_bias */
        const char* out_norm = nullptr;   /* linear_attn.norm.weight */
        const char* out      = nullptr;   /* linear_attn.out_proj.weight */
    };

    Geom        g{};
    Config      cfg{};
    Wire        w{};
    Src         src{};
    int         layer = 0;
    rad_kvgroup kv_state = 0, kv_conv = 0;

    rad_weight w_norm = 0, w_in = 0, w_z = 0, w_conv = 0;
    rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0, w_out = 0;

    rad_op op_norm = 0, op_in = 0, op_z = 0;
    rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;   /* prefill */
    rad_op op_conv_update = 0, op_recur = 0;                          /* decode */
    rad_op op_out = 0, op_ar = 0, op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                 rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int GdnBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                            rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s) {
    g = geom; cfg = c; w = wire; src = s; layer = l; kv_state = state; kv_conv = conv;
    if (cfg.n_head_k <= 0 || cfg.head_k <= 0 || cfg.n_head_v <= 0 || cfg.head_v <= 0 ||
        cfg.conv_width <= 0 || cfg.chunk <= 0)
        return RAD_E_INVAL;
    /* q and k are shared across a group of value heads -- 16 key heads to 48 value heads on this
     * model. The kernels take H and Hg separately for exactly that; a value head count that is not
     * a multiple of the key head count is a geometry no GDN kernel serves. */
    if (cfg.n_head_v % cfg.n_head_k != 0) {
        fprintf(stderr, "radiance: %lld value heads is not a multiple of %lld key heads\n",
                (long long)cfg.n_head_v, (long long)cfg.n_head_k);
        return RAD_E_INVAL;
    }

    /* THE QUERY SCALE, AND IT IS NOT OPTIONAL. `gdn_chunk_scan` and `gdn_recurrent_update` both
     * carry a `scale` that multiplies the output -- o = scale * e^g Q S^T + ... -- and it defaults to
     * 1. FLA's chunk_gated_delta_rule defaults it to head_k^-0.5 and vllm-radiance passes exactly
     * that; llama.cpp's fused GDN does the same.
     *
     * IT LOOKS LIKE IT CANCELS AND IT DOES NOT. The scan's output goes straight into the gated RMS
     * norm, and an RMS norm divides a constant factor back out -- so scaling `o` by c should change
     * nothing. That argument is wrong here for one reason: the scan output of this model is around
     * 1e-6, `mean(o^2)` is then around 1e-12, and the norm's eps is 1e-6. eps DOMINATES the variance,
     * the norm degenerates into a fixed division by sqrt(eps), and the factor passes straight through
     * to the residual stream.
     *
     * Leaving it at 1 is visible end to end: the scan output comes out larger than the reference
     * implementations' by exactly sqrt(head_k), and with forty-eight of this model's sixty-four
     * layers delta net the residual stream then carries a delta-net contribution that many times
     * too large -- fluent grammar with no content. */
    const double q_scale = 1.0 / std::sqrt((double)cfg.head_k);

    const int64_t v_dim    = cfg.v_dim();
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t in_dim   = cfg.in_dim();

    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.attn_norm.weight", l),  src.norm));
    /* The fusion: q, k, v, a and b in one row, because that is the row
     * gdn_conv_prep reads. A row-sharded CONCAT shards each source before concatenating -- and
     * `in_proj_qkv` is ITSELF three head groups, so the row split declared below is FIVE parts and
     * not three: q, k and v out of the fused source, then a, then b. The maps come ahead of the
     * weights they name, since a checkpoint is asked what a weight is through its map. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ssm_in.weight", l),
                            { src.in_qkv, src.in_a, src.in_b }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_z.weight", l),      src.in_z));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_conv1d.weight", l), src.conv1d));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_a_log", l),         src.a_log));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_dt.bias", l),       src.dt_bias));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_norm.weight", l),   src.out_norm));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_out.weight", l),    src.out));

    w_norm     = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),      RAD_F32,  {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_in       = decl_w(b, nm.f("blk.%d.ssm_in.weight", l),         RAD_BF16, {in_dim, g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim(), cfg.n_head_v, cfg.n_head_v});
    w_z        = decl_w(b, nm.f("blk.%d.ssm_z.weight", l),          RAD_BF16, {v_dim, g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* The depthwise convolution is per channel, so it shards with the channels it convolves. The
     * checkpoint ships [conv_dim, 1, 4]; the middle extent is the depthwise group and is squeezed
     * at convert. There is NO conv bias on this architecture -- Qwen3.5's conv1d has bias=False --
     * so none is declared and the operand is passed absent rather than as a zero vector nobody
     * loads. */
    w_conv     = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l),     RAD_BF16,
                        {conv_dim, cfg.conv_width},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()});
    w_a_log    = decl_w(b, nm.f("blk.%d.ssm_a_log", l),             RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_dt_bias  = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l),           RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* One gain vector over head_v, shared by every head: replicated, like the attention q/k
     * gains. */
    w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l),       RAD_F32,  {cfg.head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_out      = decl_w(b, nm.f("blk.%d.ssm_out.weight", l),        RAD_BF16, {g.n_embd, v_dim},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)), {w.x}, {w.h});
    op_in = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", in_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_in)), {w.h}, {w.in});
    op_z  = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", v_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_z)), {w.h}, {w.z});

    /* ---- the chunked path: a prefill chunk, where the scan amortises over `chunk` tokens.
     * `chunk` is RAD_DERIVED on gdn_kkt_solve and gdn_chunk_scan for the same reason block_size is
     * on attn_paged -- the tile length belongs to the kernel -- so it is omitted there and comes
     * off the resolution. gdn_conv_prep still takes it, because the gate cumsum it writes is per
     * chunk and has to be cut at the same boundary the scan tiles on. THAT THOSE TWO NUMBERS CAN
     * DISAGREE IS A HOLE: one is supplied and one is derived, nothing compares them, and a
     * mismatch is a decay applied over the wrong span rather than a diagnostic. Closing it needs
     * either a readback for a resolved parameter or conv_prep's chunk made derived too. */
    op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                     RAD_INT("head_v", cfg.head_v), RAD_INT("chunk", cfg.chunk),
                                     RAD_INT("conv_width", cfg.conv_width)),
                          RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias)),
                      {w.in}, {w.q, w.k, w.v, w.gdec, w.beta});
    op_kkt  = rw(b, RAD_OP(b, "gdn_kkt_solve",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k)),
                     RAD_NOWEIGHTS),
                 {w.k, w.beta, w.gdec}, {w.kkt});
    op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                RAD_INT("head_v", cfg.head_v), RAD_F64("scale", q_scale)),
                     RAD_NOWEIGHTS),
                 {w.q, w.k, w.v, w.kkt, w.gdec, w.beta}, {w.o});
    op_gnorm = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", cfg.head_v),
                                 RAD_F64("eps", g.eps), RAD_STR("act", "silu")),
                      RAD_WEIGHTS(w_out_norm)),
                  {w.o, w.z}, {w.o});

    /* ---- the recurrent path: decode and speculative verify, ranged on q_len because a verify
     * step carries up to n_spec+1 query rows per sequence. `n_head_v` is stated rather than left
     * to the value tensor's rank: 48 value heads against 16 key heads is the one geometry where
     * guessing it from a flattened view gives a plausible wrong answer. */
    op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                       RAD_INT("head_k", cfg.head_k),
                                       RAD_INT("head_v", cfg.head_v),
                                       RAD_INT("conv_width", cfg.conv_width)),
                            RAD_WEIGHTS(w_conv)),
                        {w.in}, {w.q, w.k, w.v});
    /* WHAT THE STATE CACHE MEANS, and it is not derivable from the operands. libr4d's recurrent
     * update has two state contracts. PER-CANDIDATE writes one full state per candidate token and
     * needs `1 + n_spec` distinct slots; ANCHOR keeps one committed state a sequence and replays
     * the accepted prefix onto it out of a scratch slot, which is two slots whatever the depth.
     * Same `o` either way, and a completely different `state` -- so a reference that implements one
     * of them and is handed the other computes a correct answer to the other question.
     *
     * The engine already chooses: KVGroupPlan::state_copies is 2 when the deployment speculates
     * and 1 when it does not, which is exactly this condition. Saying it here is what lets a
     * reference DECLINE the form it does not implement instead of disagreeing with it, and what
     * lets the kernel refuse a caller whose pool was sized for the other one. */
    op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                 RAD_INT("head_v", cfg.head_v),
                                 RAD_INT("n_head_v", cfg.n_head_v),
                                 RAD_F64("scale", q_scale),
                                 RAD_F64("eps", g.eps), RAD_STR("act", "silu"),
                                 RAD_STR("state_form",
                                         g.max_spec > 0 ? "anchor" : "per_candidate")),
                      RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                  {w.q, w.k, w.v, w.in, w.z}, {w.o});

    op_out = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                               RAD_INT("K", v_dim), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_out)),
                {w.o}, {w.h});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});
    return RAD_OK;
}

inline void GdnBF16::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    /* HOW MANY SLOTS THE ROW NAMES. One is the state; a linear kernel serving speculation
     * asks for scratch beside it and the manager sizes the row to suit (RadKVGroupBatch).
     * The block passes the row through and does not read it. */
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T        = batch->n_tok;
    const int64_t S        = batch->n_seq;
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t H        = cfg.n_head_v;

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    RAD_ISSUE(c, op_in,   brows(w.h, T), RAD_W(w_in),   brows(w.in, T));
    RAD_ISSUE(c, op_z,    brows(w.h, T), RAD_W(w_z),    brows(w.z, T));

    /* THE TWO PATHS, EACH OVER ITS OWN RUN OF SEQUENCES. rad_block_gdn_fp8.h's step() carries the
     * long form of why; the short one is that the step is sorted decode-first, every kernel here
     * derives both its grid and its rows from cu_seqlens, and cu holds ABSOLUTE offsets -- so the
     * prefill half is addressed by handing it `cu + D` with the token operands left at their
     * base, and the decode half by narrowing both to DT. The gated norm takes no cu and is per
     * token, so it takes a row slice instead. */
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    /* And the band the decode half selects on. max_q_len over a mixed step is the prefill chunk;
     * over a pure decode step it is already the right number, so a batch that leaves
     * max_q_len_decode zero still selects correctly there. */
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;
    const int64_t v_dim = cfg.v_dim();

    if (D > 0) {
        RAD_ISSUE_N(c, op_conv_update, qd,
                    brows(w.in, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
        /* a and b are the last 2*n_head_v columns of the projection row, read in place. The
         * recurrent kernel folds the gated norm, which is why it takes z and the gain: at one
         * token a head there is nothing to gain from a second pass over the state. */
        RAD_ISSUE_N(c, op_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    bcol(w.in, conv_dim, H, DT), bcol(w.in, conv_dim + H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.z, DT), RAD_W(w_out_norm), brows(w.o, DT),
                    /* The folded out-projection quantiser, which this block does not want: the
                     * bf16 out projection reads `o` directly. Absent is TWO NULL OPERANDS, not a
                     * shorter list -- operands are positional and the runtime refuses a count
                     * that does not match the schema's sixteen (core/runtime/issue.cpp:111). */
                    RAD_NONE, RAD_NONE);
    }

    if (P > 0) {
        /* Sixteen operands, inputs before outputs (docs/OPS.md). `a` and `b_gate` are the same
         * strided views of the projection row the recurrent path builds above; they are optional
         * in the schema -- ref falls back to the fused `x` tail -- but libr4d's conv_prep
         * dereferences them and `cache_idx` unconditionally, so omitting them does not select a
         * cheaper kernel, it silently drops the layer to the reference one.
         *
         * `has_init` is ctx_lens, and it is exactly the right array: a zero entry means the slot
         * holds no history, and a sequence's context length before this step is zero precisely
         * when it is starting fresh -- a new sequence, or one preempted and replaying, whose conv
         * slot was recycled and holds another sequence's window. Without it the first chunk of
         * every prefill convolves against whatever the slot last contained. */
        RAD_ISSUE(c, op_conv_prep,
                  brows(w.in, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log), RAD_W(w_dt_bias),
                  kv_cache(kv_conv, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  bcol(w.in, conv_dim, H, T), bcol(w.in, conv_dim + H, H, T),
                  praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                  praw(batch->ctx_lens + D, RAD_I32, P),
                  brows(w.q, T), brows(w.k, T), brows(w.v, T),
                  brows(w.gdec, T), brows(w.beta, T));
        RAD_ISSUE(c, op_kkt,
                  brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
                  praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(w.kkt, T));
        RAD_ISSUE(c, op_scan,
                  brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
                  brows(w.gdec, T), brows(w.beta, T),
                  kv_cache(kv_state, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  brows(w.o, T), kv_cache(kv_state, layer),
                  praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P, st_w));
        /* PER TOKEN, so a row slice and not a sequence sub-array: over the decode rows the
         * recurrent kernel has already applied the gate in its own epilogue. */
        RAD_ISSUE_N(c, op_gnorm, PT,
                    brow_slice(w.o, DT, PT, v_dim), brow_slice(w.z, DT, PT, v_dim),
                    RAD_W(w_out_norm), brow_slice(w.o, DT, PT, v_dim));
    }
    RAD_ISSUE(c, op_out, brows(w.o, T), RAD_W(w_out), brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GDN_H */
