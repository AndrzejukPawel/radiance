/* rad_block_dspark.h -- the DSpark block-diffusion drafter, fp8 weights over a bf16 checkpoint.
 *
 * ============================== WHAT IT IS ==============================
 *
 * The THIRD drafter this engine serves, and the second block-diffusion one. It is
 * `openbmb/MiniCPM5-2B-DSpark`: five plain Qwen3 attention layers that propose a whole block of
 * tokens in ONE forward pass, conditioned on the target trunk's residual stream at five
 * intermediate layers. Published acceptance length 5.52 at T = 0 (math 6.05 / code 6.11 /
 * general 4.16) against MTP's 3.74-5.02.
 *
 * IT IS NOT rad_block_dflash2.h, and the name is the trap. DSpark subclasses SpecForge's
 * `DFlashDraftModel`, which is PLAIN QWEN3 ATTENTION LAYERS; this library's `Dflash2` is the
 * conv-based drafter (`dflash_conv`, `attn_conv_base`, `sel_rank`/`sel_topk`) from a different
 * checkpoint of the same family. Same lineage, same word in the name, different architecture --
 * and the checkpoint settles it: 5 layers of q/k/v/o + q_norm/k_norm + gate/up/down and not one
 * conv or selection tensor. What IS taken from that file is its shape (Config / Wire / declare /
 * context / step), its two rules, and -- see THE MARKOV HEAD below -- its selector.
 *
 * ============================== THE FORWARD PASS, PINNED ==============================
 *
 * The reference is NOT in the published repo: `config.json` says `auto_map: dspark.DSparkDraftModel`
 * and the repo holds only config / README / safetensors. `RadixArk/Qwen3.8-27B-DSpark` ships
 * `dspark.py` and `dflash.py`, and that pair is the spec this block implements. The composition is
 * not recoverable from the tensor list alone, so it is written out here:
 *
 *     target_hidden = hidden_norm(fc(concat of the trunk's residual stream after layers
 *                                    1, 10, 20, 30, 39))          [n_taps * n_embd] -> [n_embd]
 *     h = target.embed_tokens([anchor, MASK, MASK, ...])            `block` noise rows
 *     per layer (x5):
 *         h_n = input_layernorm(h)
 *         q   = q_norm(q_proj(h_n))                                 Q FROM THE NOISE ROWS ONLY
 *         k   = k_norm(concat(k_proj(target_hidden), k_proj(h_n)))  context first, then noise
 *         v   =        concat(v_proj(target_hidden), v_proj(h_n))
 *         rope(q, k);  attention is NON-CAUSAL, mask None at inference
 *         h  += o_proj(attn);  h += mlp(post_attention_layernorm(h))
 *     logits = target.lm_head(norm(h)) + markov_w2 @ markov_w1[prev_token]
 *
 * `fc` THEN `hidden_norm`, in that order, and `target_hidden` is computed ONCE and fed to EVERY
 * layer, which projects it with its OWN k_proj and v_proj. The drafter ships no embedding table
 * and no lm_head; it reads the target's, which is why a vocabulary that disagrees is not a drafter
 * for this target at all.
 *
 * ============================== THE DUAL-SOURCE K/V IS AN ORDINARY KV CACHE ==============================
 *
 * `concat(proj(target_hidden), proj(noise))` needs no new kernel and no fused-operand games. The
 * reference's own cache says why: `past_key_values.update` appends the context rows and the noise
 * rows, the block attends over the whole thing, and then `crop(start)` throws the noise away. That
 * is a paged KV group with the context written per COMMITTED TOKEN and the noise rows written into
 * slots past `seqused` -- which the next context pass overwrites, so the crop is free.
 *
 * So the split is the same two passes Dflash2 has, for the same reason:
 *
 *   THE CONTEXT PASS, one row a committed token. It runs the five k/v projections over the
 *   projected trunk taps and stores them. The drafter's own layers never run over the context.
 *
 *   THE QUERY PASS, `block` rows a sequence, over the drafter's five layers, attending to the
 *   context rows AND to each other.
 *
 * AND THE PER-HEAD NORM OVER THE CONCAT IS THE SAME NUMBER EITHER WAY. `k_norm` is an RMS norm
 * over each head's own `head_dim` row, so norming the context half in one pass and the noise half
 * in the other is bit-for-bit what norming the concatenation would have been. Nothing about the
 * split is an approximation.
 *
 * `attn_paged` ALREADY TAKES `causal`, so `causal = 0` serves the non-causal block: every noise
 * row sees every other row of its own block as well as the whole context. This drafter is
 * 16 head / 2 kv / head_dim 128 = GQA 8, the same shape as its target, so the attention and the
 * norm/rope rows the trunk resolves serve it unchanged.
 *
 * ============================== THE ROW ALIGNMENT, WHICH IS THE WHOLE GAME ==============================
 *
 * `block_size` IS THE NUMBER OF NOISE ROWS AND ALSO THE NUMBER OF DRAFTS. Seven rows in, seven
 * tokens out, verify width eight. The noise stream is [anchor, MASK x (block-1)] and ROW j
 * PREDICTS THE TOKEN AT POSITION start + j + 1 -- the ordinary next-token relation, anchor row
 * included, which is why `steps() == block` here and `steps() == block - 1` in Dflash2.
 *
 * THAT IS THE OPPOSITE CONVENTION TO Dflash2 AND IT IS INVISIBLE IF YOU GET IT WRONG. Dflash2's
 * anchor row predicts nothing and each MASK row predicts the token at ITS OWN position; DSpark's
 * anchor row is a real prediction and no row is wasted. Both run, both produce `depth` tokens, and
 * the model's own output stays exactly correct either way because every draft is verified against
 * the trunk -- what happens instead is that the proposals are shifted one position and acceptance
 * falls to roughly chance. The core stages the block against `RadDrafterDecl::block` and runs the
 * head over the LAST `depth` rows, which is the one rule both conventions reduce to; rad_builder.h
 * argues it, and this file is the reason that field exists.
 *
 * ============================== THE MARKOV HEAD IS Dflash2'S SELECTOR ==============================
 *
 * DSpark adds a rank-256 learned bigram bias to the draft logits:
 *
 *     logits[l] += markov_w2 @ markov_w1[prev(l)]      prev(0) = the anchor,
 *                                                      prev(l) = the token chosen at l - 1
 *
 * which is SERIAL in l for exactly the reason a block-diffusion drafter needs a selector at all:
 * the positions were computed together and the bias at l depends on the token chosen at l - 1.
 * Written literally it is seven full-vocabulary [rank, n_vocab] GEMMs with an argmax between each
 * -- 66 MiB of `markov_w2` read seven times a step to use seven of its rows.
 *
 * `dflash_select` ALREADY COMPUTES THIS. Its edge score is
 *
 *     unary[l][c] + sum_r pred[prev][r] * hp[l][r] * succ[cand[l][c]][r]
 *
 * over the base head's top K candidates, with the predecessor read after the previous step's
 * argmax inside one workgroup and no launch between steps. Set `hp` ABSENT -- it is optional, and
 * absent it is ones -- and the trilinear form drops to the plain bigram
 * `<pred[prev], succ[cand]>`; pass `markov_w1` as `pred` and `markov_w2` as `succ` and the walk IS
 * the Markov head, greedily decoded. The codebook gather is K * rank * 2 bytes a position, 8 KiB
 * at K = 16, against 466 MiB for the literal form.
 *
 * WHAT IS APPROXIMATED IS THE CANDIDATE SET AND NOTHING ELSE: the bias can only re-rank the base
 * head's top K, where the reference could promote any of 130560. That is the same trade the 2-bit
 * draft head makes and it is sound for the same reason -- a draft is a proposal, the target
 * verifies every one of them with its own untouched head, and speculative decoding is
 * distribution-preserving, so a coarser draft costs acceptance and CANNOT change an emitted token.
 * `RADIANCE_DRAFT_RAW=dspark_cand` proposes column 0 of the candidate plane instead, which is the
 * base head's own argmax with the Markov walk thrown away: it isolates what the walk contributes,
 * and it needs no rebuild.
 *
 * THE CONFIDENCE HEAD IS NOT IMPLEMENTED AND THAT IS NOT A GAP. `AcceptRatePredictor` predicts a
 * per-position acceptance probability so a serving loop can shorten the block adaptively; this
 * engine's depth is fixed at declare time and only 1, 3 and 7 are legal. It changes nothing the
 * drafter proposes, so ignoring it is exactly correct -- unlike the input/output scalars Dflash2
 * refuses, which would silently move every logit.
 *
 * ============================== THE NORMS ARE PLAIN ==============================
 *
 * `w`, not `1 + w`: the reference norms with `Qwen3RMSNorm` and the config says `model_type:
 * qwen3`. `Geom::wadd` and `wadd_qk` are therefore ZERO for this block, which happens to agree
 * with its MiniCPM5 target -- but it is the drafter's own answer and not inherited, and the two
 * geometries are separate structs partly for this.
 *
 * ============================== WHAT IS FP8 AND WHAT IS NOT ==============================
 *
 *   E4M3 + a bf16 [N/128][K/128] plane   fc, q/k/v/o, gate_up, down -- every linear
 *   BF16, no plane                       every norm, and both Markov codebooks
 *
 * The checkpoint is dense bf16 throughout, so the first row is rad_fp8.h's DenseFP8 over a weight
 * the recipe quantises to block fp8. The reason is bytes -- the drafter is read WHOLE on every
 * decode step, so 324M parameters at bf16 is 648 MB a step and at fp8 is 324 MB. Nothing it
 * computes can change an emitted token; the only thing quantising it can move is acceptance.
 *
 * The codebooks stay bf16 and REPLICATED because they are gathered by GLOBAL token id out of a
 * merged candidate set: a vocabulary shard would let a rank score only the candidates that
 * happened to land in its own slice. 66.8 MiB each, and the gather is K rows a position, so what
 * costs is the residency and not the reads.
 *
 * ============================== WHAT IS LEFT ON THE TABLE ==============================
 *
 * Two things this block does not do, and what each would be worth:
 *
 *   THE Q/K NORM AND ROPE ARE FOUR LAUNCHES A LAYER where `qk_norm_rope` would be one. That op
 *   takes a FUSED [q|k] row and this block keeps q, k and v in three buffers -- which the context
 *   pass forces for k and v (it computes them with no q at all, and a block-scaled weight's layout
 *   follows its op's N) but does not force for the query pass. Fusing would want a [q_dim + kv_dim]
 *   buffer, a column-slice store, and a contiguous q lift for the attention, which is the trunk's
 *   own arrangement (rad_block_attn_fp8.h). It is worth twenty launches a step.
 *
 *   THE ALL-REDUCE IS NOT FOLDED INTO THE FOLLOWING NORM. NormQuantFP8 has the fused form and this
 *   block passes kArNone, so a tp2 query pass issues ten collectives of its own. At `block` rows a
 *   sequence the message is 28 KiB at one sequence -- under the lossy wire's floor and served
 *   exactly -- so what a fold saves is launches, not bytes.
 */
#ifndef RAD_BLOCK_DSPARK_H
#define RAD_BLOCK_DSPARK_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

struct Dspark {
    enum { MAX_LAYERS = 8 };
    /* The 2-bit draft head's scale group, which is the w2a8 row's own and not a choice. */
    enum { DS_W2_GROUP = 128 };

    /* Everything the drafter's own config.json says that its geometry does not. Read by the plugin
     * from `draft.*` metadata keys and handed in whole, so this file reads no environment and no
     * container. */
    struct Config {
        int64_t n_taps = 0;     /* how many trunk layers `fc` concatenates */
        int64_t block  = 0;     /* NOISE ROWS a sequence -- and drafts, one per row */
        int64_t rank   = 0;     /* markov_rank: the codebooks' width */
        int64_t top_k  = 0;     /* candidates the Markov walk re-ranks */
        int64_t window = 0;     /* sliding attention width, 0 for none */
        int     causal = 0;
        /* EVERY ROW PREDICTS, so this is `block` and not `block - 1`. See THE ROW ALIGNMENT. */
        int64_t steps() const { return block; }
    };

    struct Wire {
        /* ---- the context pass, one row a COMMITTED TOKEN */
        rad_buf src = 0;   /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        rad_buf c   = 0;   /* [max_tok, n_embd]           fc's output, then normed in place */
        /* K AND V ARE TWO BUFFERS, not one [max_tok, 2*kv_dim] with K in its left half. Both
         * halves are written by one projection and read by one kv_store, so fusing them is the
         * obvious thing -- and it is wrong, because of the PER-HEAD NORM in between. `rmsnorm`
         * takes ONE row pitch. K inside a fused row is `head_dim` apart within a token and
         * `2*kv_dim` apart between tokens, and no single pitch is both; answering with the
         * between-tokens stride normalises every eighth head against the wrong numbers, which is
         * a drafter that runs and barely proposes anything the trunk accepts. libr4d refuses that
         * operand (libr4d/r4d_model.cpp, rows_pitch). */
        rad_buf ck  = 0;   /* [max_tok, kv_dim]           ONE layer's K, reused across layers */
        rad_buf cv  = 0;   /* [max_tok, kv_dim]           ONE layer's V, reused across layers */
        rad_buf srcq = 0, srcs = 0;   /* the fp8 twins: e4m3 | f32 [.., /128] */
        rad_buf cq   = 0, cs   = 0;

        /* ---- the query pass, `block` rows a sequence */
        rad_buf x = 0;     /* [qrows, n_embd]  the drafter's residual stream */
        /* THE NORMED INPUT AND THE BLOCK'S DELTA ARE ONE ACTIVATION, which is the trunk's own
         * arrangement (rad_block_attn_fp8.h): o_proj and down_proj write `h.x`, and the NEXT
         * norm reads it as the delta to fold into the stream before norming. One buffer, and the
         * op that reads it also writes it. */
        ActFP8  h{};       /* [qrows, n_embd] */
        rad_buf q = 0;     /* [qrows, q_dim] */
        rad_buf k = 0;     /* [qrows, kv_dim]   -- two buffers, and `ck` above says why */
        rad_buf v = 0;     /* [qrows, kv_dim] */
        ActFP8  attn{};    /* [qrows, q_dim] */
        rad_buf gate_up = 0; /* [qrows, 2 * n_ff] */
        ActFP8  ffn{};     /* [qrows, n_ff] */

        /* ---- the head and the Markov walk, one row a DRAFT POSITION */
        rad_buf sel_h = 0; /* [srows, n_embd]  the gathered final hidden states */
        rad_buf dq    = 0; /* [srows, n_embd]  i8   the 2-bit head's quantised activation */
        rad_buf ds    = 0; /* [srows, 1]       f32  one scale a row */
        rad_buf dsum  = 0; /* [n_embd/128, pad] f32 GROUP-major, quant_act_i8's asum plane */
        rad_buf dlog  = 0; /* [srows, head_rows(g)] bf16  the coarse logits */
        rad_buf cand  = 0; /* [srows, top_k] i32  candidate token ids, GLOBAL */
        rad_buf unary = 0; /* [srows, top_k] f32  and their logits */
        /* The vocab-parallel pair, declared only when the head is sharded. Both are (int32 id,
         * float value) pairs read as f32: this rank's top-R, and the all-gathered world * R. */
        rad_buf dpair = 0; /* [srows, top_k, 2]         f32 */
        rad_buf dgath = 0; /* [srows, world * top_k, 2] f32 */
        rad_buf tokens = 0;/* [max_seqs, steps] i32  THE DRAFT -- the engine reads this by name */
    };

    /* rad_arch.h's, asked through the block so whoever sizes `dlog` and the pair planes does not
     * re-derive the rule. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    Geom    g{};      /* the DRAFTER's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, so the KV group's layer_slot lookup finds these five */
    int64_t qrows = 0, crows = 0, srows = 0;
    int64_t gated_rows = 0;    /* the band op_gug was declared over; 0 when there is no fold */
    rad_kvgroup kv = 0;

    /* ---- model-scope weights */
    DenseFP8   w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_mk1 = 0, w_mk2 = 0;         /* markov_w1 (pred), markov_w2 (succ) */
    rad_weight w_tok = 0;                    /* the TARGET's embedding table */
    rad_weight w_dh = 0, w_dhs = 0;          /* the 2-bit copy of the TARGET's lm_head */

    /* ---- per-layer weights. Q, K AND V ARE THREE WEIGHTS and not the trunk's fused one: the
     * context pass wants k and v over the projected taps and has no use for q at all, so a fusion
     * would make it compute and discard a q_dim-wide GEMM per layer per context token. K and V are
     * apart from each other because a block-scaled weight's layout is a function of its OP's N and
     * both passes take one of them at a time. */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    DenseFP8   w_q[MAX_LAYERS]{}, w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};

    /* ---- context-pass ops */
    rad_op op_qsrc = 0, op_cfc = 0, op_cnorm = 0, op_qc = 0, op_crope = 0, op_cstore = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    /* ---- query-pass ops. The weightless ones are declared once and issued per layer: an op's
     * declared read and write sets are what the buffer planner reads, and they are the same every
     * layer. The per-layer ones are per-layer because a weight's prefetch interval and residency
     * class come from the op that declares it. */
    rad_op op_embed = 0, op_embed_ar = 0;
    /* THE FUSED NORM / RESIDUAL-ADD / QUANTISE, declared by hand rather than through rad_fp8.h's
     * NormQuantFP8. The wrapper's extra machinery is its FUSED-COLLECTIVE forms, which this block
     * does not take (see WHAT IS LEFT ON THE TABLE), so what is left of it is one op -- and an
     * ARRAY of the wrapper is not compilable: it nests a Geom, and g++ 16.1 ICEs in build_vec_init
     * value-initialising an array of that inside an enclosing aggregate assignment (`m = Model{}`
     * in the plugin). Handles are what every other member here is anyway. */
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {}, op_outnorm = 0;
    rad_op op_qat = 0;
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_gug[MAX_LAYERS] = {};   /* the gate_up|silu|quant fold; 0 when nothing serves it */
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    rad_op op_act = 0, op_qff = 0, op_ar = 0;

    /* ---- the head and the Markov walk */
    rad_op op_gather = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0;
    rad_op op_dgather = 0, op_dmerge = 0;   /* the vocab-parallel pair; 0 when the head is whole */
    rad_op op_sel = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck_prefix,
                 const char* ck_lm, const char* ck_lm_alt, const Wire& wire);
    /* The context pass, one row a committed token. `batch` carries those rows and the draft KV
     * group's slot mapping; nothing about the query block is read. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass, the head and the Markov walk: one call, `block` rows a sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dspark::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                           int base_layer, rad_kvgroup group, rad_weight tok_embd,
                           const char* ck, const char* ck_lm, const char* ck_lm_alt,
                           const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_lm || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: dspark has %lld layers; this block carries at most %d\n",
                (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0 || cfg.rank <= 0 || cfg.top_k <= 0) return RAD_E_INVAL;

    const int64_t qd = q_dim(), kvd = kv_dim();

    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    /* EVERY QUERY ROW PREDICTS, so the head runs over as many rows as the block has. They are the
     * same number and they are kept apart anyway: one is a pass's rows and the other is what the
     * head was declared over, and Dflash2 -- the same struct shape over the other convention --
     * has them differ by one sequence's worth. */
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dspark.fc"), g.n_embd, cfg.n_taps * g.n_embd,
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(),
                              { nm.ckpt("%s.fc.weight", ck) }));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.hidden_norm.weight"),
                          nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.norm.weight"), nm.ckpt("%s.norm.weight", ck)));

    w_hnorm = decl_w(b, nm.f("dspark.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dspark.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* THE TWO MARKOV CODEBOOKS, replicated at the WHOLE vocabulary and bf16. `markov_w1` is an
     * nn.Embedding and `markov_w2` an nn.Linear, and both store [vocab, rank] -- so the bigram
     * bias `w2 @ w1[prev]` is `<w1[prev], w2[cand]>` and the two go straight into the selector's
     * `pred` and `succ` slots in that order. RAD_ACCESS_VOCAB because the reads are a gather of K
     * rows a position and not a pass over the plane. */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w1"),
                          nm.ckpt("%s.markov_head.markov_w1.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w2"),
                          nm.ckpt("%s.markov_head.markov_w2.weight", ck)));

    w_mk1 = decl_w(b, nm.f("dspark.markov_w1"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    w_mk2 = decl_w(b, nm.f("dspark.markov_w2"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

    /* THE 2-BIT DRAFT HEAD, over the TARGET's lm_head. `dspark.draft_head.weight` is that tensor
     * under a name of its own, which the recipe quantises to the 2-bit grid (u2 codes, an f16
     * scale and a u8 zero a group of 128): one checkpoint tensor, two logical weights. The
     * argument is rad_block_mtp_fp8.h's and it is the same one the Markov walk's top-K rests on: a
     * draft is verified, so a coarser head costs acceptance and cannot move an emitted token.
     *
     * A TIED HEAD IS THE ALTERNATIVE SOURCE. Small LLaMA derivatives ship no `lm_head.weight` and
     * read the embedding table as both; the trunk's own head is declared that way and the
     * drafter's copy has to agree with it or the two would disagree about what the model's
     * vocabulary projection IS. */
    const bool dh_shard = head_sharded(g);
    const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const char* dhn = nm.f("dspark.draft_head.weight");
    if (ck_lm_alt && *ck_lm_alt) RAD_ARCH_TRY(map_copy_alt(b, dhn, ck_lm, ck_lm_alt));
    else                         RAD_ARCH_TRY(map_copy(b, dhn, ck_lm));
    w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {head_rows(g), g.n_embd},
                      RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    w_dhs = decl_view(b, nm.f("dspark.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                      {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    if (!w_dh || !w_dhs) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup grp = grp_layer(first_layer + i);
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));

        w_anorm[i] = decl_w(b, nm.f("dspark.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        w_fnorm[i] = decl_w(b, nm.f("dspark.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dspark.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dspark.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dspark.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        w_qn[i] = decl_w(b, nm.f("dspark.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        w_kn[i] = decl_w(b, nm.f("dspark.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dspark.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dspark.blk.%d.ffn_gate_up", i),
                                     2 * g.n_ff, g.n_embd,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dspark.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp,
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
    }

    /* NON-CAUSAL. The block is denoised as a whole, so every query row sees every other row of its
     * own block as well as the context -- `causal` 0 is the drafter's own `is_causal: false` and
     * not a default.
     *
     * DECLARED FIRST, before either kv_store, even though it runs after both. `block_size` is
     * RAD_DERIVED -- the KV block is a property of the resolved attention kernel and not a number
     * anyone gets to choose -- so it is omitted here and READ BACK with rad_kv_block_size() for
     * the two stores, which need it as a parameter. Ask before the attention op is declared and
     * the answer is zero. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {w.q}, {w.attn.x});

    /* ---------------------------------------------------------------- the context pass
     *
     * EVERY LINEAR HERE IS `fp8a8` and every activation that feeds one gets a `quant_act_fp8` in
     * front of it. There is no `fp8a16` band anywhere in this block: that kernel is M <= 1 and
     * neither pass ever runs at one row -- a context pass is a committed chunk and a query pass is
     * `block` rows by construction. */
    op_qsrc = rw(b, RAD_OP(b, "quant_act_fp8",
                    RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.src}, {w.srcq, w.srcs});
    op_cfc = rw(b, RAD_OP(b, "gemm_nt_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", g.n_embd),
                              RAD_INT("K", cfg.n_taps * g.n_embd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                   RAD_WEIGHTS(w_fc.w, w_fc.s)),
               {w.srcq, w.srcs}, {w.c});
    /* IN PLACE. Safe for any implementation that computes a row's scale before writing the row,
     * which none can avoid doing; the per-head q/k norms below run in place for the same reason. */
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    /* ONE QUANTISE FOR ALL FIVE LAYERS. `c` is the normed projection of the trunk's taps and every
     * layer's k|v reads that same buffer, so this is one launch a context pass and not five. */
    op_qc = rw(b, RAD_OP(b, "quant_act_fp8",
                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.c}, {w.cq, w.cs});
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* TWO OPS OF THE SAME SHAPE, one per weight: an op's declared weight list is what gives a
         * weight its prefetch interval and its residency class. */
        op_ck[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_k[i].w, w_k[i].s)),
                     {w.cq, w.cs}, {w.ck});
        op_cv[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_v[i].w, w_v[i].s)),
                     {w.cq, w.cs}, {w.cv});
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    /* K-ONLY ROPE. `rope` rotates the first n_head + n_head_kv heads of its operand and leaves the
     * V heads that follow alone, so a K-only tensor is described by n_head = n_head_kv and
     * n_head_kv = 0: all the heads there are rotate, none follow them. Weightless, so one op
     * serves every layer.
     *
     * THE POSITION IS THE COMMITTED TOKEN'S OWN, which is what makes the split passes equal the
     * reference's one concatenated rope: it hands `k` the full position range and `q` only the
     * last `q_len` of it, so a context row at absolute position p is rotated by p either way. */
    op_crope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.x});
    /* The target's embedding is vocab-sharded, so this gather is partial and takes the same EXACT
     * reduction the trunk's does: each token's row comes from one rank and every other contributes
     * a zero, so a lossy wire would be pure loss on a value that is already the answer. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});

    /* THE NORM, THE RESIDUAL ADD AND THE QUANTISER ARE ONE OP -- `rmsnorm_quant_fp8`, which
     * rad_fp8.h's NormQuantFP8 wraps. Layer 0 has no delta to fold (its input is the embedding
     * itself), every layer after it folds the previous feed-forward's, and `nq_out` folds the last
     * one into the FINAL norm. So there is no `add` op in this block at all and the final norm is
     * not a separate launch either.
     *
     * `nq_out` QUANTISES SOMETHING NOTHING READS. The head after it wants bf16 (it runs its own
     * i8 quantiser at a different group width), and this op writes the bf16 normed row as well as
     * the fp8 twin -- so the twin is 14 KiB of waste at a seven-row block, against a launch. The
     * alternative is `add` + `rmsnorm`, which is two, because libr4d has no plain `rmsnorm_add`
     * row: `rmsnorm_quant_fp8` IS this library's fused norm-add. */
    /* `x` IS THE DELTA AND `residual` IS THE STREAM, and the op adds the first into the second
     * before norming: residual_out = x + residual, y = rmsnorm(residual_out) * w. So a FOLDED
     * site passes the previous block's output buffer as `x` and the stream as `residual`, and the
     * UNFOLDED one (layer 0, whose input is the embedding) passes the stream as `x` and no
     * residual at all. An absent optional operand is a null tensor in a full-length list.
     *
     * `residual` is INOUT, so it is in both sets: the buffer planner reads these for liveness and
     * an INOUT that claimed only one of them is a plan free to reorder a reader in front of it. */
    auto decl_norm = [&](rad_weight gain, bool fold) -> rad_op {
        rad_op h = RAD_OP(b, "rmsnorm_quant_fp8",
                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                     RAD_F64("eps", g.eps), RAD_INT("group", RAD_FP8_BLOCK),
                                     RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                          RAD_WEIGHTS(gain));
        if (!h) return 0;
        rad_buf rd[2] = { fold ? w.h.x : w.x, w.x };
        rad_buf wr[4] = { w.h.q, w.h.s, w.h.x, w.x };
        rad_op_reads (b, h, rd, fold ? 2 : 1);
        rad_op_writes(b, h, wr, fold ? 4 : 3);
        return h;
    };
    op_qat = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.attn.x}, {w.attn.q, w.attn.s});
    if (!op_qat) return RAD_E_INVAL;

    /* THE NORMS ARE DECLARED WHERE step() ISSUES THEM, because declaration order is the buffer
     * planner's liveness: a transient lives over the op indices that touch it, and two whose spans
     * are disjoint may share bytes (core/build/rad_bufplan.cpp). The stream `x` and the norm
     * outputs `h` are live across every layer, and the norms are the only ops that touch the
     * stream after the embedding, so their positions are the stream's lifetime. Each layer's two
     * norms sit inside the loop and the final one after every op the layers issue -- the shared
     * rope, store, silu_mul and quantiser handles below included.
     *
     * Declared together ahead of the layers, the stream's lifetime ended before the first layer's
     * temporaries began, and the packer put a feed-forward temporary on top of it. The fused gated
     * GEMM never writes those, so the overlap stayed silent up to 64 rows a pass; past 64 the
     * unfused pair writes `gate_up` and `ffn.x`, and every draft was rejected from the tenth
     * sequence on. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = decl_norm(w_anorm[i], i != 0);
        if (!op_anorm[i]) return RAD_E_INVAL;
        op_qp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", qd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_q[i].w, w_q[i].s)),
                     {w.h.q, w.h.s}, {w.q});
        op_kp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_k[i].w, w_k[i].s)),
                     {w.h.q, w.h.s}, {w.k});
        op_vp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_v[i].w, w_v[i].s)),
                     {w.h.q, w.h.s}, {w.v});
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv),
                                    RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", g.n_embd),
                                    RAD_INT("K", qd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_o[i].w, w_o[i].s)),
                     {w.attn.q, w.attn.s}, {w.h.x});
        op_fnorm[i] = decl_norm(w_fnorm[i], true);
        if (!op_fnorm[i]) return RAD_E_INVAL;
        op_gu[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", 2 * g.n_ff),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                     {w.h.q, w.h.s}, {w.gate_up});
        /* THE GATED FOLD: gate_up, silu_mul and the quantiser in one GEMM epilogue, over the same
         * weights and into the same activation. NOT RAD_ARCH_TRY -- a kernel set without the row
         * is the ordinary case and the pair below still runs -- and `gated_rows` is the band it
         * was declared over, checked at the issue site, because M past the band is a REFUSAL in
         * the kernel and not a slower path. */
        {
            const int64_t hi = qrows < 64 ? qrows : 64;
            if (hi > 0) {
                op_gug[i] = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * g.n_ff),
                                              RAD_INT("K", g.n_embd),
                                              RAD_INT("group", RAD_FP8_BLOCK),
                                              RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                               {w.h.q, w.h.s}, {w.ffn.q, w.ffn.s});
                if (op_gug[i]) gated_rows = hi;
            }
        }
        op_dn[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", g.n_embd),
                                    RAD_INT("K", g.n_ff), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_dn[i].w, w_dn[i].s)),
                     {w.ffn.q, w.ffn.s}, {w.h.x});
    }

    op_qrope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn.x});
    op_qff = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.ffn.x}, {w.ffn.q, w.ffn.s});
    /* ONE COLLECTIVE DECLARATION, issued after the out projection and after the down projection:
     * the two have the same shape over the same buffer. It is EXACT or lossy by the deployment's
     * own setting, exactly as the trunk's is -- the drafter's message is the same residual stream
     * and there is no reason for it to be held to a different standard than the thing it drafts
     * for. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h.x}, {w.h.x});
    op_outnorm = decl_norm(w_norm, true);
    if (!op_outnorm) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- the head and the walk
     *
     * EVERY ROW OF THE BLOCK PREDICTS, so the head runs over all `steps` of them and the anchor is
     * not skipped -- THE OPPOSITE of Dflash2, and the paragraph at the top of this file is why.
     * The gather is what makes those rows contiguous, and it is also what fixes the head's row
     * numbering at s * steps + l, which is the numbering the walk reads. It is the identity map
     * for this convention as the scheduler emits it; it is issued anyway, because `out_ids` is the
     * contract and RADIANCE_DFLASH_ANCHOR is allowed to move it. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h.x}, {w.sel_h});
    op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_INT("group", DS_W2_GROUP), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.sel_h}, {w.dq, w.ds, w.dsum});
    op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                                  RAD_INT("K", g.n_embd), RAD_INT("group", DS_W2_GROUP),
                                  RAD_STR("dtype", "w2a8")),
                       RAD_WEIGHTS(w_dh, w_dhs)),
                   {w.dq, w.ds, w.dsum}, {w.dlog});
    /* Sharded, the top-R runs over THIS RANK'S plane and reports GLOBAL ids, and it also writes
     * the pair plane the gather moves. `vocab_off` is omitted entirely when the head is
     * replicated: an optional parameter present with the value 0 is not the same declaration as
     * an absent one. */
    op_dtopk = dh_shard
        ? rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.top_k), RAD_INT("vocab_off", g.vocab_off),
                           RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary, w.dpair})
        : rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.top_k), RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary});
    if (dh_shard) {
        /* ROW-INTERLEAVED, so one draft position's pairs from every rank land side by side --
         * `row_topk_merge` reads one row at one row stride. The merge reduces them to the same
         * global top-R an unsharded head would have found, exactly, ties included, which is what
         * lets the codebooks stay replicated and indexed by global id. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * cfg.top_k * 2),
                                      RAD_STR("dtype", "f32"),
                                      RAD_INT("row", cfg.top_k * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                          RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", cfg.top_k),
                                     RAD_INT("world_size", g.world),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.dgath}, {w.cand, w.unary});
    }
    /* THE MARKOV WALK. `dflash_select` with the hidden projection ABSENT is the plain bigram
     * `unary + <markov_w1[prev], markov_w2[cand]>`, decoded greedily with the predecessor read
     * after the previous step's argmax -- which is the Markov head, and the top of this file
     * argues both halves of that claim. Operands are POSITIONAL, so the absent `hp` is still
     * passed, as RAD_NONE, at the issue site.
     *
     * `anchor_stride` is `block`, because the operand is the step's own token_ids: a draft pass
     * carries `block` ids a sequence and the anchor is the first of them. */
    op_sel = rw(b, RAD_OP(b, "dflash_select",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.top_k), RAD_INT("rank", cfg.rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_mk1, w_mk2)),
               {w.cand, w.unary}, {w.tokens});

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dspark::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    RAD_ISSUE_N(c, op_qsrc, T, brows(w.src, T), brows(w.srcq, T), brows(w.srcs, T));
    RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                brows(w.c, T), RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv,
                    brows(w.ck, T), RAD_W(w_kn[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dspark::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    /* The fused norm at its three sites. Folded, `x` is the delta the previous block left in
     * `h.x` and the stream is added into; unfolded (layer 0), `x` IS the stream and there is no
     * residual operand -- an absent optional operand is still passed, as RAD_NONE. */
    auto issue_norm = [&](rad_op op, bool fold, rad_weight gain, int64_t rows) {
        RAD_ISSUE_N(c, op, rows, brows(fold ? w.h.x : w.x, rows),
                    fold ? brows(w.x, rows) : RAD_NONE, RAD_W(gain),
                    brows(w.h.q, rows), brows(w.h.s, rows), brows(w.h.x, rows));
    };

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok),
                brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        /* ---- attention. The norm folds the previous layer's feed-forward delta into the stream
         * and quantises, all in one launch; at layer 0 there is no delta and it norms the
         * embedding itself. */
        issue_norm(op_anorm[i], i != 0, w_anorm[i], T);
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_q[i].w), RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
        /* THE NOISE ROWS GO INTO THE CACHE, past this sequence's `seqused`, and the next context
         * pass overwrites them from the trunk's real hidden states. That is the reference's
         * `crop(start)` and it is why nothing here has to undo anything. */
        RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn.x, T));
        RAD_ISSUE_N(c, op_qat, T, brows(w.attn.x, T), brows(w.attn.q, T), brows(w.attn.s, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.attn.q, T), brows(w.attn.s, T),
                    RAD_W(w_o[i].w), RAD_W(w_o[i].s), brows(w.h.x, T), RAD_NONE, RAD_NONE);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);

        /* ---- the feed-forward, behind the same fused norm-add-quantise */
        issue_norm(op_fnorm[i], true, w_fnorm[i], T);
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.h.q, T), brows(w.h.s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffn.q, T), brows(w.ffn.s, T),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.h.q, T), brows(w.h.s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE,
                        RAD_NONE);
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn.x, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ffn.x, T), brows(w.ffn.q, T), brows(w.ffn.s, T));
        }
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffn.q, T), brows(w.ffn.s, T),
                    RAD_W(w_dn[i].w), RAD_W(w_dn[i].s), brows(w.h.x, T), RAD_NONE, RAD_NONE);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    }

    /* The last layer's delta folds into the FINAL norm rather than into a norm of its own. */
    issue_norm(op_outnorm, true, w_norm, T);

    /* The rows that predict something, which here is all of them. The scheduler names them
     * through out_ids, the same field the trunk's own logits gather reads. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h.x), praw(batch->out_ids, RAD_I32, S),
                brows(w.sel_h, S));

    /* `dsum` is quant_act_i8's GROUP-major plane, [n_embd/group][ceil(M/16)*16], so it is passed
     * whole: there is no row slice of it that means anything. */
    RAD_ISSUE_N(c, op_dquant, S, brows(w.sel_h, S), brows(w.dq, S), brows(w.ds, S),
                RAD_B(w.dsum));
    /* The trailing operand is mxfp4's optional reference and is absent here; an absent optional
     * operand is a null tensor in a FULL-LENGTH list, not a shorter list. */
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.dq, S), brows(w.ds, S), RAD_W(w_dh), RAD_W(w_dhs),
                brows(w.dlog, S), RAD_B(w.dsum), RAD_NONE);
    if (op_dmerge) {
        /* Sharded, this rank's top-R is not the answer -- it is this rank's CANDIDATES, and
         * `cand`/`unary` are overwritten by the merge two ops below. Writing them here anyway
         * costs nothing and keeps one issue site for both paths. */
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_dmerge, S, brows(w.dgath, S), brows(w.cand, S), brows(w.unary, S));
    } else {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    RAD_NONE);
    }
    /* RAD_NONE in the third slot is the ABSENT hidden projection -- the bigram form. */
    RAD_ISSUE_N(c, op_sel, batch->n_seq, brows(w.cand, S), brows(w.unary, S), RAD_NONE,
                praw(batch->token_ids, RAD_I32, T), RAD_W(w_mk1), RAD_W(w_mk2),
                brows(w.tokens, batch->n_seq));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DSPARK_H */
