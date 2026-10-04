/* r4d_fp8_args.h -- the block-scaled fp8 family's operand list, read once for five kernels.
 *
 * Every kernel that reads `<linear>.weight` as the checkpoint stores it -- F8_E4M3 [N][K] with a
 * BF16 scale on a [ceil(N/128)][ceil(K/128)] grid -- takes the same five operands in the same
 * order, because they implement the same op at different M. The reader belongs to the family, so
 * it is a header beside them rather than an adapter between them and the core. See
 * libr4d/r4d_args.h.
 */
#ifndef R4D_FP8_ARGS_H
#define R4D_FP8_ARGS_H

#include "r4d.h"
#include "r4d_args.h"

enum { F_A = 0, F_ASCALE = 1, F_B = 2, F_BSCALE = 3, F_Y = 4 };
/* ...and for `quant_act_fp8` and `sample_chain`. */
enum { QF_X = 0, QF_Q = 1, QF_S = 2 };
/* The two fused forms. Shaped on their int8 siblings (`rmsnorm_had_quant_i8`,
 * `gated_had_quant_i8`) minus the operands the rotation needs, so an architecture that switches
 * activation format changes op names and not operand order. */
enum { RQ_X = 0, RQ_RES = 1, RQ_W = 2, RQ_Q = 3, RQ_S = 4, RQ_Y = 5 };
enum { GQ_GU = 0, GQ_Q = 1, GQ_S = 2, GQ_Y = 3 };
enum { TQ_G = 0, TQ_X = 1, TQ_Q = 2, TQ_S = 3, TQ_Y = 4 };
enum { SC_LOGITS = 0, SC_QUERY = 1, SC_U = 2, SC_DRAW = 3 };

/* A required integer parameter, or 0. Distinct from r4d_geti_or's defaulting because these three
 * have no sensible default: a geometry without M, N and K is not a geometry. */
static inline long long aff_or(const RadArgs* a, const char* k) { return r4d_geti_or(a, k, 0); }

struct Fp8Ops {
    const RadTensor *a, *ascale, *b, *bscale, *y;
    long long M, N, K, nkb;
    long a_ld, sa_ld, w_ld, sw_ld;    /* a_ld / w_ld in ELEMENTS here; see the note at each use */
};

/* Pitches, and the one place this family's two operand pitches are read.
 *
 * libr4d takes the WEIGHT pitch in BYTES and the SCALE pitch in ELEMENTS, which is not an
 * inconsistency: an E4M3 row is bytes and a bf16 scale row is not, and giving both the same unit
 * would make one of the two conversions invisible at the call site. Restated here so the shim
 * cannot get it wrong quietly. */
/* The weight's scale plane: per 128x128 tile (block E4M3), or per ROW per 128 K (the int8 twins). */
enum { FP8_OPS_BLOCK_SCALE = 0, FP8_OPS_ROW_SCALE = 1 };

static inline int fp8_ops(const RadArgs* a, Fp8Ops* g, int want_ascale,
                          int wscale = FP8_OPS_BLOCK_SCALE) {
    std::memset(g, 0, sizeof(*g));
    g->a      = r4d_opt(a, F_A);
    g->ascale = r4d_opt(a, F_ASCALE);
    g->b      = r4d_opt(a, F_B);
    g->bscale = r4d_opt(a, F_BSCALE);
    g->y      = r4d_opt(a, F_Y);
    if (!g->a || !g->b || !g->bscale || !g->y) return R4D_NO(RAD_E_INVAL);
    if (want_ascale && !g->ascale) return R4D_NO(RAD_E_INVAL);

    if (g->b->rank != 2 || g->bscale->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* ROWS ARE DIM 0 AND EVERYTHING AFTER IT IS THE ROW -- the same reading r4d_rad_quant_act_fp8
     * takes of its input, and for the same reason. A gated attention block's q|gate projection is
     * declared [tokens, heads, 2*head_dim] because the de-interleaving that follows needs the head
     * axis; its N is heads*2*head_dim and the GEMM neither knows nor cares. Demanding rank 2 here
     * would refuse that projection, which is the first thing a gated attention layer does. */
    if (g->y->rank < 2) return R4D_NO(RAD_E_SHAPE);
    g->M = g->y->shape[0];
    if (g->M < 1 || r4d_numel(g->y) % g->M) return R4D_NO(RAD_E_SHAPE);
    g->N = r4d_numel(g->y) / g->M;
    g->K = g->b->shape[1];
    if (g->b->shape[0] != g->N) return R4D_NO(RAD_E_SHAPE);
    if (g->M < 1 || g->N < 1 || g->K < 1) return R4D_NO(RAD_E_SHAPE);

    /* The layout hook and the .rad it wrote are built against R4D_FP8_BLOCK; libr4d compiles its
     * own copy in. A scalar compare against a constant costs nothing and the alternative is a
     * silently wrong scale grid. */
    const long long blk = r4d_gemm_fp8a8_block();
    if (blk != R4D_FP8_BLOCK) return R4D_NO(RAD_E_SHAPE);
    g->nkb = (g->K + blk - 1) / blk;
    /* The scale plane's own extents, checked rather than assumed: this is the operand a converter
     * is most likely to hand over at the wrong grid, and a wrong grid reads a plausible number. */
    if (g->bscale->shape[0] != (wscale == FP8_OPS_ROW_SCALE ? g->N : (g->N + blk - 1) / blk))
        return R4D_NO(RAD_E_SHAPE);
    if (g->bscale->shape[1] != g->nkb) return R4D_NO(RAD_E_SHAPE);

    const long long wp = r4d_row_pitch(g->b), sp = r4d_row_pitch(g->bscale);
    const long long yp = r4d_row_pitch(g->y);
    if (wp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The output has no pitch argument -- libr4d writes C[r*N + c]. */
    if (yp != g->N) return R4D_NO(RAD_E_STRIDE);
    g->w_ld = wp;      /* elements == bytes for E4M3 */
    g->sw_ld = sp;

    if (g->a->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (g->a->shape[0] != g->M || r4d_numel(g->a) != g->M * g->K) return R4D_NO(RAD_E_SHAPE);
    const long long ap = r4d_row_pitch(g->a);
    if (ap < 0) return R4D_NO(RAD_E_STRIDE);
    g->a_ld = ap;
    if (g->ascale) {
        if (g->ascale->rank != 2) return R4D_NO(RAD_E_SHAPE);
        if (g->ascale->shape[0] != g->M || g->ascale->shape[1] != g->nkb) return R4D_NO(RAD_E_SHAPE);
        const long long asp = r4d_row_pitch(g->ascale);
        if (asp < 0) return R4D_NO(RAD_E_STRIDE);
        g->sa_ld = asp;
    }
    /* 16, not 128. The widest weight load in the family is a uint4 in the matvec and a 32-byte
     * staged pair in the GEMM, both at a 16-byte offset from the row base. A 128-byte demand would
     * be a claim about the cacheline, which is 256 on this part and is not an alignment rule of any
     * load here. */
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 16)) return R4D_NO(RAD_E_ALIGN);
    return RAD_OK;
}


#endif  /* R4D_FP8_ARGS_H */
