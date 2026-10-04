/* llama_fp8 -- the plain LLaMA graph with block-scaled FP8 weights and FP8 activations.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * `model_type: llama` with `LlamaForCausalLM`, which is the most widely shared decoder shape there
 * is and is what MiniCPM5-2B reports: 42 layers, hidden 2048, 16 query heads over 2 KV heads at
 * head_dim 128, SwiGLU at 6144, RMS norms, full rotary at theta 5e6, an untied 130560-row head,
 * and no biases anywhere. Every layer is full attention; there is no gated delta net, no layer
 * schedule, no output gate and no per-head q/k norm, which is precisely what separates this file
 * from arch/qwen35_fp8.
 *
 * ============================== TWO CHECKPOINT FORMATS, ONE GRAPH ==============================
 *
 * The selection key is the PAIR (architecture, quantisation) -- spec §2.4 -- and one .so claims one
 * pair, so this source builds twice:
 *
 *   llama_fp8        quant "fp8_e4m3"    the checkpoint already holds E4M3 codes and a
 *                                        `weight_scale_inv` plane, which are the two planes of
 *                                        each linear as the engine reads them.
 *   llama_dense_fp8  quant ""            the checkpoint is one dense bf16 tensor per linear, and
 *                                        a RECIPE quantises each to those two planes at convert
 *                                        time (data/recipes/minicpm5-2b-dspark.recipe). The
 *                                        container then holds what the first one's checkpoint
 *                                        does.
 *
 * THE SECOND IS THE ONE MiniCPM5-2B NEEDS, because OpenBMB publishes it in bf16 and nothing else.
 * "MiniCPM5-2B FP8A8" is therefore a statement about what this engine RUNS and not about what the
 * checkpoint holds, and the conversion is where the two meet: one pass per 128x128 block, amax,
 * scale = amax/448 rounded to bf16, divide and encode. Both builds declare the same weights and
 * the same ops, so nothing downstream -- the kernels, the container, the placement plan -- can
 * tell which one ran; the descriptor is the only reason there are two.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 *   E4M3 + a bf16 [N/128][K/128] plane   self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj
 *   BF16, no plane                       every norm and the embedding
 *
 * The lm_head is declared at whatever the model holds it as: bf16 as every llama checkpoint ships
 * it, or block fp8 where a recipe quantised it -- the widest GEMM in the model at half the bytes.
 * The logits kernel that reads that encoding is the one selected. Both vocabulary edges are
 * ROW-SHARDED with the vocabulary.
 */
#ifndef LLAMA_DENSE_SRC
#define LLAMA_DENSE_SRC 0
#endif
#ifndef LLAMA_NS
#define LLAMA_NS llama_fp8
#endif
#ifndef LLAMA_QUANT
#define LLAMA_QUANT "fp8_e4m3"
#endif

#include <cstdlib>
#include <cstring>
#include <vector>

#include <arch/rad_arch.h>
#include <arch/rad_block_attn_fp8.h>
#include <arch/rad_block_dspark.h>
#include <arch/rad_block_mlp_fp8.h>

namespace LLAMA_NS {

using namespace rad::arch;

struct Layer {
    AttnFP8 attn;
    MlpFP8  mlp;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t n_rot = 0;
    int64_t n_main = 0;

    rad_kvgroup kv_full = 0;

    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActFP8  a_h{};
    rad_buf b_qkv = 0;
    rad_buf b_q   = 0;
    ActFP8  a_attn{};
    rad_buf b_gate_up = 0;
    ActFP8  a_ffn{};

    /* THE AUXILIARY HIDDEN-STATE TAPS, declared only when the container carries a drafter that
     * asks for them. A DSpark or DFlash drafter does not read the trunk's FINAL hidden state --
     * it reads the RESIDUAL STREAM at several intermediate layers and projects the concatenation
     * down, so its `fc` is [n_embd, n_taps * n_embd] and the input has to be one contiguous row.
     * The trunk keeps one residual buffer that every layer overwrites, so the rows are copied out
     * as they pass, into the column slice that tap owns. This is arch/qwen35_fp8's mechanism and
     * the comments there carry the reasoning; what is here is the same thing over this graph. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    /* THE DRAFTER, declared only when the container carries one and this serve intends to
     * speculate. Its layers get absolute layer indices AFTER the trunk's, so the KV group's
     * layer_slot lookup finds them and its weights land in layer groups of their own. */
    Dspark       ds{};
    Dspark::Wire dsw{};
    rad_kvgroup  kv_draft = 0;
    bool         have_ds = false;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;

    std::vector<Layer> layers;
};

/* "1 10 20 30 39" -> the layers whose output a drafter wants. rad-convert flattens a config.json
 * array into a space-separated string (tools/rad_convert.cpp, flatten_json), so this is the shape
 * every list-valued key arrives in. Out-of-range entries are DROPPED rather than clamped: a
 * drafter naming a layer this model does not have is paired with the wrong target, and silently
 * reading layer 41 instead of 49 would turn that into a quality mystery. */
static int parse_tap_layers(const char* s, int64_t n_layer, int* out, int max) {
    if (!s) return 0;
    int n = 0;
    for (const char* p = s; *p && n < max;) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        p = e;
        if (v >= 0 && v < n_layer) out[n++] = (int)v;
    }
    return n;
}

static Model g_model[MAX_RANKS];

/* THE CONTAINER'S QUANTISATION DESCRIPTOR IS THE SELECTION KEY AND IT IS CHECKED ANYWAY, because
 * `--arch` can override the architecture id at convert time and a plugin that is handed the wrong
 * checkpoint should say which half disagrees rather than fail at a name map. */
static int check_format(const RadModelMeta* meta) {
    const char* q = meta->quant ? meta->quant : "";
#if LLAMA_DENSE_SRC
    if (*q) {
        fprintf(stderr, "radiance: llama_dense_fp8 serves an UNQUANTISED checkpoint and this one "
                        "declares '%s'. The fp8 sibling (llama_fp8) reads that one.\n", q);
        return RAD_E_UNSUPPORTED;
    }
#else
    if (std::strncmp(q, "fp8", 3) != 0) {
        fprintf(stderr, "radiance: llama_fp8 serves a block-scaled fp8 checkpoint and this one "
                        "declares '%s'. A dense bf16 one is served by llama_dense_fp8, after a "
                        "recipe has quantised it.\n", *q ? q : "(unquantised)");
        return RAD_E_UNSUPPORTED;
    }
#endif
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: llama has no routed feed-forward and this checkpoint declares "
                        "%lld experts\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* THE DSPARK DRAFTER. The Geom is the TRUNK'S with the drafter's own geometry substituted in: the
 * vocabulary, the deployment bounds, the rank and the wire stay the trunk's (they are properties
 * of this serve, not of the checkpoint) and everything about the layers comes from `draft.*`.
 *
 * Nothing here is defaulted quietly. A container whose `draft.` config does not describe this
 * drafter, or describes one this file does not implement, is refused BY NAME: a drafter that
 * silently ignored a scale factor or a head type would simply accept less, which reads as a
 * quality mystery and not as a missing feature (spec §17).
 *
 * THE ONE THING THAT IS DELIBERATELY IGNORED is the confidence head, and rad_block_dspark.h argues
 * it: `AcceptRatePredictor` exists so a serving loop can shorten the block adaptively, this
 * engine's depth is fixed at declare time, and the head changes nothing the drafter proposes. */
static int declare_dspark(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg = m.g;   /* n_embd, the vocabulary, the deployment bounds, the rank, the wire */

    dg.n_layer       = rad_meta_geti(meta, "draft.num_hidden_layers", 0);
    dg.head_dim      = rad_meta_geti(meta, "draft.head_dim", 0);
    dg.n_head_all    = rad_meta_geti(meta, "draft.num_attention_heads", 0);
    dg.n_head_kv_all = rad_meta_geti(meta, "draft.num_key_value_heads", 0);
    dg.n_ff_all      = rad_meta_geti(meta, "draft.intermediate_size", 0);
    dg.eps           = (float)rad_meta_getf(meta, "draft.rms_norm_eps", 0.0);
    dg.theta         = (float)rad_meta_getf(meta, "draft.rope_parameters.rope_theta",
                                            rad_meta_getf(meta, "draft.rope_theta", 0.0));
    dg.rope_scale    = 1.0f;
    /* PLAIN RMS NORM: the reference norms with Qwen3RMSNorm and the config says `model_type:
     * qwen3`. It happens to agree with this MiniCPM5 target, and it is asked of the DRAFTER's
     * config anyway -- two conventions in one process is exactly why this is a second Geom. */
    dg.wadd    = 0.0f;
    dg.wadd_qk = 0.0f;

    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: the container's `draft.` config does not describe a DSpark "
                        "drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, ffn %lld, "
                        "rms_eps %g, rope_theta %g\n",
                (long long)dg.n_layer, (long long)dg.n_head_all, (long long)dg.n_head_kv_all,
                (long long)dg.head_dim, (long long)dg.n_ff_all, (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    int s;
    if ((s = divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads", &dg.n_head))    < 0) return s;
    if ((s = divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads", &dg.n_head_kv)) < 0) return s;
    if ((s = divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",   &dg.n_ff))      < 0) return s;

    /* THE DRAFTER BORROWS THE TARGET'S EMBEDDING TABLE AND lm_head -- it ships neither -- so a
     * vocabulary that disagrees is not a drafter for this target at all. */
    const long long dvocab = rad_meta_geti(meta, "draft.vocab_size", dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: the DSpark drafter claims a %lld-token vocabulary and the "
                        "target has %lld. The drafter ships no embedding table and no lm_head; it "
                        "reads the target's, so the two have to be the same vocabulary.\n",
                dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }

    Dspark::Config dc{};
    dc.n_taps = m.n_taps;
    dc.rank   = rad_meta_geti(meta, "draft.markov_rank", 0);
    dc.causal = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.sliding_window", 0);
    /* HOW MANY CANDIDATES THE MARKOV WALK RE-RANKS. The reference biases all 130560 logits and
     * takes the argmax; this re-ranks the base head's top K, which is the only approximation in
     * the head and it can cost acceptance and nothing else (rad_block_dspark.h). K is bounded by
     * the selector at 32; 16 is 8 KiB of codebook a position, which is free, and it is read from
     * the container so a checkpoint can ask for more. */
    dc.top_k  = rad_meta_geti(meta, "draft.markov_top_k", 16);
    if (dc.top_k < 1 || dc.top_k > 32) {
        fprintf(stderr, "radiance: draft.markov_top_k is %lld; the Markov walk scores at most 32 "
                        "candidates a position\n", (long long)dc.top_k);
        return RAD_E_UNSUPPORTED;
    }

    /* THE BLOCK IS THE NUMBER OF NOISE ROWS AND ALSO THE NUMBER OF DRAFTS -- every row predicts
     * the token after it, the anchor row included. So the engine's depth IS `block_size`, where a
     * Dflash2 container's is `block_size - 1`, and that off-by-one is the whole reason
     * RadDrafterDecl carries a `block` field. Getting it wrong runs, drafts, and accepts nothing.
     *
     * A BLOCK-DIFFUSION DRAFTER WAS TRAINED FOR ONE BLOCK LENGTH, so a serve that asks for another
     * depth is refused here rather than run: the reference denoises exactly `block_size` masked
     * positions and there is no partial block. The probe reports `draft_depth_fixed`, so the
     * engine will normally have picked this already. */
    const long long trained = rad_meta_geti(meta, "draft.block_size", 0);
    if (trained < 2) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no draft.block_size; the "
                        "number of masked positions it denoises is not a number this engine may "
                        "choose.\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec != trained) {
        fprintf(stderr, "radiance: the DSpark drafter was trained at block_size %lld, which is "
                        "%lld draft tokens, and this serve asks for %d. A block-diffusion drafter "
                        "denoises a fixed number of mask positions; run it at %lld.\n",
                trained, trained, ctx->max_spec, trained);
        return RAD_E_UNSUPPORTED;
    }
    dc.block = trained;

    /* Two things this drafter's family can carry that this implementation does not apply. Each
     * changes what the drafter proposes, which is why neither is ignored quietly. */
    const char* mh = rad_meta_gets(meta, "draft.markov_head_type", "vanilla");
    if (dc.rank <= 0 || !mh || std::strcmp(mh, "vanilla") != 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint has markov_rank %lld and "
                        "markov_head_type '%s'. Only the rank-N 'vanilla' bigram head is "
                        "implemented, and it is what the published checkpoints ship.\n",
                (long long)dc.rank, mh ? mh : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    if (dc.rank % 32 || dc.rank > 512) {
        fprintf(stderr, "radiance: markov_rank %lld; the walk that applies the bigram bias takes "
                        "a multiple of 32 up to 512\n", (long long)dc.rank);
        return RAD_E_UNSUPPORTED;
    }

    /* THE DRAFTER'S OWN KV CACHE, a group of its own. It cannot share the trunk's: five layers
     * against forty-two, and -- when the checkpoint asks for one -- a WINDOW. RAD_KV_WINDOW
     * recycles the blocks that fall below it; DSpark's published config sets `sliding_window:
     * null`, so this is normally a full cache of five layers. Its width is the trunk's,
     * --kv-cache-dtype: one flag says what every attention cache of the deployment is made of. */
    RadKVGroupDecl dd{};
    dd.kind      = dc.window > 0 ? RAD_KV_WINDOW : RAD_KV_FULL;
    dd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    dd.window    = dc.window;
    dd.n_head_kv = dg.n_head_kv;
    dd.head_dim  = dg.head_dim;
    m.kv_draft = rad_decl_kv_group(b, nm.f("kv_draft"), &dd);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, m.kv_draft));
    dg.kv_dtype = rad_dtype_name(dd.dtype);

    /* Activation buffers. The context pass is one row a COMMITTED TOKEN and runs over a whole
     * prefill chunk; everything after it is one row a DRAFT POSITION, `block` of them a sequence,
     * so the two halves are sized apart. None of it is large -- the drafter's cost is its
     * weights, not its activations. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dspark::Wire& dw = m.dsw;
    dw.src  = m.b_dfsrc;
    dw.c    = decl_b(b, nm.f("ds_c"),  dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck   = decl_b(b, nm.f("ds_ck"), dg.act_dtype, {dg.max_tok, kvd});
    dw.cv   = decl_b(b, nm.f("ds_cv"), dg.act_dtype, {dg.max_tok, kvd});
    /* THE FP8 TWINS OF THE CONTEXT PASS. `srcq` is the largest of them by a distance -- 20 MiB at
     * a 2048-token chunk, five taps and hidden 2048, against the 40 MiB `draft_src` it shadows --
     * and it buys half the bytes of the one GEMM the context pass spends its time in. */
    dw.srcq = decl_b(b, nm.f("ds_srcq"), RAD_F8E4M3, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs = decl_b(b, nm.f("ds_srcs"), RAD_F32,
                     {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq   = decl_b(b, nm.f("ds_cq"), RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs   = decl_b(b, nm.f("ds_cs"), RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});

    dw.x      = decl_b(b, nm.f("ds_x"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.x    = decl_b(b, nm.f("ds_h"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.q    = decl_b(b, nm.f("ds_h.q"), RAD_F8E4M3, {qr, dg.n_embd});
    dw.h.s    = decl_b(b, nm.f("ds_h.s"), RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.q      = decl_b(b, nm.f("ds_q"), dg.act_dtype, {qr, qd});
    dw.k      = decl_b(b, nm.f("ds_k"), dg.act_dtype, {qr, kvd});
    dw.v      = decl_b(b, nm.f("ds_v"), dg.act_dtype, {qr, kvd});
    dw.attn.x = decl_b(b, nm.f("ds_attn"), dg.act_dtype, {qr, qd});
    dw.attn.q = decl_b(b, nm.f("ds_attn.q"), RAD_F8E4M3, {qr, qd});
    dw.attn.s = decl_b(b, nm.f("ds_attn.s"), RAD_F32,    {qr, fp8_blocks(qd)});
    dw.gate_up = decl_b(b, nm.f("ds_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn.x  = decl_b(b, nm.f("ds_ffn"), dg.act_dtype, {qr, dg.n_ff});
    dw.ffn.q  = decl_b(b, nm.f("ds_ffn.q"), RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffn.s  = decl_b(b, nm.f("ds_ffn.s"), RAD_F32,    {qr, fp8_blocks(dg.n_ff)});

    dw.sel_h  = decl_b(b, nm.f("ds_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.dq     = decl_b(b, nm.f("ds_dq"), RAD_I8,  {sr, dg.n_embd});
    dw.ds     = decl_b(b, nm.f("ds_ds"), RAD_F32, {sr, 1});
    /* GROUP-major, [n_embd/group][ceil(M/16)*16] -- quant_act_i8's asum plane, whose first
     * dimension is the K group and not the token. */
    dw.dsum   = decl_b(b, nm.f("ds_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* THIS RANK'S plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded, asked here rather than re-derived, because a buffer that disagreed with the
     * op declared over it is a read past its end. */
    dw.dlog   = decl_b(b, nm.f("ds_dlog"), dg.act_dtype, {sr, Dspark::head_rows(dg)});
    dw.cand   = decl_b(b, nm.f("dspark_cand"), RAD_I32, {sr, dc.top_k});
    dw.unary  = decl_b(b, nm.f("ds_unary"), RAD_F32, {sr, dc.top_k});
    if (Dspark::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("ds_dpair"), RAD_F32, {sr, dc.top_k, 2});
        dw.dgath = decl_b(b, nm.f("ds_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.top_k, 2});
    }
    /* NAMED, because the engine reads the draft out of it by name: a block pass proposes through
     * the drafter's own walk and never through the sampler's token buffer. */
    dw.tokens = decl_b(b, nm.f("dspark_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    /* A TIED lm_head IS THE ALTERNATIVE SOURCE, and it has to be the same pair the trunk's own
     * head reads or the drafter would be projecting onto a different vocabulary. */
    RAD_ARCH_TRY(m.ds.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), nm.ckpt("model.embed_tokens.weight"),
                              dw));
    m.have_ds = true;

    /* WHAT THE ENGINE DRIVES. Block: a context pass that mirrors the trunk step, then ONE pass
     * that fills the whole block. `block` is `depth` and not `depth + 1`, which is this drafter's
     * trained convention and the field's reason for existing. `window` must be the number this
     * plugin gated its OWN attention with, because the scheduler stages the block pass against
     * it: two readings of "how far back can the drafter see" is a drafter attending to positions
     * the scheduler did not stage. */
    RadDrafterDecl drf{};
    drf.name           = "dspark";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.mask_token_id", -1);
    if (drf.mask_token < 0)
        drf.mask_token = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no mask_token_id, and every "
                        "position the block has not filled embeds it -- there is nothing to "
                        "run.\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    rad_note(b, "the DSpark drafter is declared: %lld layers of its own at absolute %d..%d, "
                "%lld heads / %lld kv x %lld, ffn %lld, %s attention with a %lld-token window, "
                "reading the trunk at %d tap(s). ONE forward over %lld noise row(s) proposes "
                "%lld tokens -- there are no rounds -- and EVERY row predicts, the anchor "
                "included, so the block is `depth` rows and not `depth + 1`",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.block, (long long)dc.steps());
    rad_note(b, "the rank-%lld Markov bigram head runs as the candidate walk with no hidden "
                "projection: `unary + <markov_w1[prev], markov_w2[cand]>` over the draft head's "
                "top %lld, decoded greedily. The confidence head is present in the checkpoint and "
                "deliberately not applied -- it drives adaptive block length, which this engine "
                "does not have, and it changes nothing the drafter proposes",
             (long long)dc.rank, (long long)dc.top_k);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the ACTIVATION dtype: everything but the two GEMM operands runs at it. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    Geom&  g  = m.g;
    Names& nm = m.nm;
    m.n_main  = g.n_layer;

    /* THE ROTARY WIDTH IS THE WHOLE HEAD unless the config says otherwise. LLaMA rotates every
     * dimension; `partial_rotary_factor` exists in this config schema and Phi-style models set it,
     * so it is read rather than assumed -- but the DEFAULT is the family's answer and not a
     * refusal, because a checkpoint that states nothing is stating "all of it". */
    {
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        m.n_rot = prf > 0.0 ? (int64_t)(g.head_dim * prf) : g.head_dim;
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: llama: rotary width %lld is not an even width inside head_dim "
                        "%lld\n", (long long)m.n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* ---- the KV cache. One group: every layer is full attention. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* ---- activations. */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    m.b_x      = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd));
    m.b_qkv    = decl_b(b, nm.f("attn_qkv"), g.act_dtype, {g.max_tok, q_dim + 2 * kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim));
    m.b_gate_up = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff));
    m.b_hout   = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* ---- the drafter's taps. Gated on BOTH halves of the question: the container has to carry a
     * drafter that reads them, and the deployment has to intend to speculate. The buffer is one
     * residual stream per tap -- 40 MiB at a 2048-token chunk, five taps and hidden 2048 -- which
     * a server that never drafts should not be carrying.
     *
     * TWO KEY SPELLINGS, because the two checkpoints spell it differently and both are real. A
     * SpecForge DFlash config nests the drafter's wiring under `dflash_config`; the published
     * DSpark configs (openbmb/MiniCPM5-2B-DSpark and its siblings) put the same fields at the top
     * level. Reading the flat one first and falling back keeps one plugin serving both rather
     * than making the spelling an architecture. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
            const char* ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
            if (!ids) ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
            m.n_taps = parse_tap_layers(ids, m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("draft_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            } else {
                fprintf(stderr, "radiance: llama: the container carries a '%s' drafter but names "
                                "no trunk layer this model has; it is paired with a different "
                                "target\n", da);
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    m.b_logits = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both ROW-SHARDED with the vocabulary. A tied head is expressed by
     * the alternative source: a model that ships no `lm_head.weight` reads the embedding table as
     * both, which is what every small LLaMA derivative does. The head is declared at its codes'
     * dtype -- bf16, or E4M3 where a recipe made it block fp8 -- after its map, which is how the
     * model is asked. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* EXACT, unconditionally: each token's embedding comes from exactly one rank and every other
     * contributes a zero, so a lossy wire would be pure loss on a value that is already the
     * answer. The same argument arch/qwen35_fp8 makes about its own. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    AttnFP8::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qkv = m.b_qkv; aw.q = m.b_q; aw.attn = m.a_attn;
    MlpFP8::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;

    auto is_tap = [&](int64_t l) {
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l) return true;
        return false;
    };

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        /* THE RESIDUAL FOLD, and the rule is arch/qwen35_fp8's with exactly one of its exceptions:
         * there are no ablation switches here, so every block folds its predecessor's delta into
         * its own norm and the LAST layer's feed-forward keeps its own add -- for the final norm,
         * which is a plain `rmsnorm` with no residual operand.
         *
         * A TAP IS THE OTHER READER OF `x` BETWEEN BLOCKS, and it forces the add to stay where it
         * was. What the drafter conditions on has to be that layer's REAL output, so a tap layer's
         * feed-forward keeps its own add and the block after it has nothing left to fold. Get this
         * wrong and the taps carry a residual stream missing its last delta -- a drafter that
         * still runs, still proposes, and is quietly conditioned on the wrong thing.
         *
         * `ar_in`/`ar_out` move the collective with the add, one boundary at a time, at two ranks
         * only: at tp1 there is none and at four the fused kernel does not exist. They need no
         * change for the taps -- a tap layer's MLP has `add`, so it does not defer its all-reduce,
         * and the next mixer does not fold, so it looks for none. The two stay consistent because
         * both are derived from the same two flags. */
        const bool tp2      = g.world == 2;
        const bool mix_fold = l > 0 && !is_tap(l - 1);
        const bool mlp_add  = is_tap(l) || (l + 1 >= m.n_main);
        const int  mix_ar_in  = !tp2 ? kArNone
                              : mix_fold ? kArAny
                              : (l == 0 && m.op_embed_ar) ? kArExact : kArNone;

        const char* pre = "model.layers";
        AttnFP8::Src as{};
        as.norm = nm.ckpt("%s.%lld.input_layernorm.weight",  pre, (long long)l);
        as.q    = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
        as.k    = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
        as.v    = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
        as.o    = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                      mix_fold, /*add_out=*/false, mix_ar_in, tp2));

        MlpFP8::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, /*fold=*/true, mlp_add,
                                     tp2 ? kArAny : kArNone, !mlp_add && tp2));
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.a_h.x});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* THE DRAFTER IS DECLARED LAST, AFTER `op_logits`, AND THE ORDER IS LOAD-BEARING.
     *
     * DECLARATION ORDER IS WHAT THE BUFFER PLANNER'S LIVENESS IS: a transient's range is
     * [first_def, last_use] over OP INDICES, and two buffers whose ranges are disjoint may share
     * bytes (core/build/rad_bufplan.cpp). The drafter's passes run AFTER the whole trunk pass, so
     * declaring them here is the order that says so. Declared in the MIDDLE -- before
     * `op_final_norm`, say -- the trunk's own tail ops get indices past the drafter's, the planner
     * reads that as "these live after the drafter is done", and it shares bytes between a trunk
     * buffer and a drafter buffer that are both live inside one engine step.
     *
     * THE SYMPTOM OF GETTING IT WRONG IS NOT A CRASH: the trunk's output stays fluent, because row
     * 0 of a verify step is computed before the clobber lands, and EVERY DRAFT IS REJECTED, because
     * rows 1..n_spec are not. Speculation looks broken and the drafter looks wrong while it is
     * proposing exactly the right tokens. It is the one class of fault the op oracle cannot see: a
     * buffer clobbered before an op runs makes the device and the host compute the same wrong
     * answer and agree.
     *
     * The same declaration can be correct at one world size and wrong at another: the shard
     * halves every sharded buffer and the greedy packer's offsets are a function of the sizes.
     * arch/qwen35_fp8 declares its drafter here for the same reason; this is not a stylistic
     * match. */
    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dspark(b, meta, ctx, m, (int)m.n_main));

    rad_note(b, "rank %d/%d: %lld layers, all full attention; n_head=%lld n_head_kv=%lld "
                "head_dim=%lld rot=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)m.n_rot, (long long)g.n_ff,
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
#if LLAMA_DENSE_SRC
    rad_note(b, "the checkpoint is DENSE bf16 and its recipe quantised every linear to E4M3 with "
                "a bf16 [N/%d,K/%d] scale plane -- one amax per block, scale = amax/448 rounded "
                "to bf16. The container is half the checkpoint's size and the graph is the one "
                "llama_fp8 declares over a checkpoint that shipped that way",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
#else
    rad_note(b, "weights are block-scaled fp8: E4M3 [N,K] with a bf16 [N/%d,K/%d] scale plane, "
                "one stored weight taken by gemm_nt_q as two operands",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
#endif
    rad_note(b, "NOT fp8 in the checkpoint: every norm, the embedding and the lm_head. The head "
                "is declared at what the model holds -- block fp8 where a recipe made it so");
    if (m.n_taps > 0) {
        char ids[64] = {};
        int  off = 0;
        for (int t = 0; t < m.n_taps && off < (int)sizeof(ids) - 8; ++t)
            off += snprintf(ids + off, sizeof(ids) - (size_t)off, t ? " %d" : "%d",
                            m.tap_layer[t]);
        rad_note(b, "the drafter reads the residual stream at layer(s) %s into a [%lld, %lld] "
                    "buffer -- %lld tap(s) x %lld. Those layers keep their own residual add and "
                    "the layer after each one has nothing to fold, so the tap sees the real "
                    "output rather than a stream still missing its last delta",
                 ids, (long long)g.max_tok, (long long)((int64_t)m.n_taps * g.n_embd),
                 (long long)m.n_taps, (long long)g.n_embd);
    }
    rad_note(b, "q, k and v are ONE fused [%lld, %lld] projection and `rope` rotates the fused "
                "row in place -- q heads then k heads, v untouched. There is no per-head q/k norm "
                "and no output gate in this family; qwen35 has both and that is the whole "
                "difference between the two attention blocks",
             (long long)(q_dim + 2 * kv_dim), (long long)g.n_embd);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    /* A DRAFTER PASS, AND THE SIGN SAYS WHICH KIND. For a block-diffusion drafter the two kinds
     * are the two passes: POSITIVE is the QUERY pass, which proposes the whole block in one go,
     * and NEGATIVE is the CONTEXT pass, which turns the tokens the trunk just committed into the
     * drafter's own K and V. There is no third kind and no round counter -- this drafter runs its
     * layers once per step. Zero is a trunk step, so a zeroed RadBatch runs the model. */
    if (batch->draft_pass != 0) {
        if (!m.have_ds) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` block, or declare ran at max_spec 0, "
                             "which is the engine saying it would not draft");
            return;
        }
        if (batch->draft_pass > 0) m.ds.step(c, batch);
        else                       m.ds.context(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar && !ar_wire_is_exact(m.g, batch->n_tok))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    for (size_t li = 0; li < m.layers.size(); ++li) {
        m.layers[li].attn.step(c, batch);
        m.layers[li].mlp.step(c, batch);
        /* The tap, into the column slice this one owns. It runs on the PREFILL and DECODE passes
         * alike, because the drafter conditions on every committed token's residual stream and
         * the rows it wants are whatever this batch just computed. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.a_h.x, batch->n_tok));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* WHAT THIS CONTAINER CAN SPECULATE TO. A plain llama container answers zero -- there is no MTP
 * head in this family and no draft block -- and one converted with `--draft-model` against a
 * DSpark checkpoint answers its trained block length.
 *
 * `draft_depth` IS `block_size` AND NOT `block_size - 1`, which is the one number a reader coming
 * from arch/qwen35_fp8 will expect to be different: DSpark's anchor row predicts the token after
 * it like every other row, so seven noise rows are seven drafts. `draft_depth_fixed` says the
 * engine may not pick another -- the drafter denoises a fixed number of masked positions and was
 * trained for exactly one block length. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
        const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
        if (bs > 1) {
            out->draft_depth       = (int)bs;
            out->draft_depth_fixed = 1;
        }
    }
    return RAD_OK;
}

}  /* namespace LLAMA_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* One more level of expansion so `#NS` sees the expanded namespace; arch/qwen35_fp8 explains it. */
#define LLAMA_PROBE_X(NS)        RAD_ARCH_PROBE(NS)
#define LLAMA_PLUGIN_X(NS, ...)  RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
LLAMA_PROBE_X(LLAMA_NS)
#if LLAMA_DENSE_SRC
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama) from a DENSE bf16 checkpoint, made block-scaled fp8 by a recipe "
               "at convert time and served fp8 weights / fp8 activations")
#else
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama), block-scaled fp8 weights and fp8 activations")
#endif
#endif
