/* r4d_gdn_args.h -- what the gated-delta-net kernels share when they read their arguments.
 *
 * The GDN family is seven kernels across six translation units, and they share two things: a
 * dtype check that has to be by name, and `num_seqs` read off the cu_seqlens tensor. Both belong
 * to the family rather than to any one kernel, so they live here -- a header beside them, not an
 * adapter directory between them and the core. See libr4d/r4d_args.h.
 */
#ifndef R4D_GDN_ARGS_H
#define R4D_GDN_ARGS_H

#include "r4d.h"
#include "r4d_args.h"


/* EVERY POINTER libr4d TAKES AS `const void*` HAS ONE ELEMENT TYPE, AND THE CAST IS INSIDE THE
 * KERNEL. `r4d_gdn_conv_prep_w4_h128_bf16` casts `wgt` to `const unsigned short*` and `A_log` to
 * `const float*`; nothing in the signature says so.
 *
 * The hazard is concrete, not theoretical. A plugin that declares `ssm_conv1d.weight` f32 or the
 * `gdn_kkt` buffer f32 hands libr4d planes it reads as bf16: the convolution takes two bf16
 * elements out of every float, and the chunk scan's inverse is written as bf16 into a plane sized
 * and read as f32. Neither shows up on the host path, because libref reads every operand through
 * its OWN dtype and is therefore correct with either -- so the two implementations disagree and
 * only the device one is wrong.
 *
 * A shim exists to reject what the entry point would reject if it could see the argument (spec
 * §2.1, and the file header above says the same about pitches). A dtype the kernel silently
 * reinterprets is the strongest case of that there is, so it is checked here, by name. */
static inline int r4d_dt(const RadTensor* t, uint32_t want) {
    if (!t) return RAD_OK;                       /* an absent optional operand has no dtype */
    return t->dtype == want ? RAD_OK : RAD_E_DTYPE;
}
#define R4D_DT(t, want) do { const int _d = r4d_dt((t), (want)); if (_d != RAD_OK) return _d; } while (0)

/* num_seqs from the cu_seqlens tensor: it is [N+1] and nothing else in the argument list carries
 * N. Refusing a cu of rank != 1 here is what stops an off-by-one N reaching a kernel that indexes
 * cu[N] unconditionally. */
static inline int seqs_of(const RadTensor* cu, int* out) {
    if (!cu || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    *out = (int)(cu->shape[0] - 1);
    return RAD_OK;
}

/* libr4d's compiled-in geometry, so a rebuild at a different head size cannot silently change what
 * the rows mean. */
static inline int gdn_dims_ok(void) {
    int hk = 0, hv = 0, ch = 0;
    r4d_gdn_dims(&hk, &hv, &ch);
    return (hk == 128 && hv == 128 && ch == 64) ? RAD_OK : RAD_E_SHAPE;
}


/* Elements between consecutive `width`-wide rows, for an operand that spells them either way: a
 * [.., heads, width] tensor steps a row with the stride of the axis above the last, and a flattened
 * [tokens, heads*width] one steps it with `width`. Negative when neither holds densely. */
static inline long long gn_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] == width) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : width;
    }
    return r4d_contig(t) ? width : -1;
}

#endif  /* R4D_GDN_ARGS_H */
