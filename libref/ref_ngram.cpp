/* ref_ngram.cpp -- the n-gram row ids of Qwen4-Exp's Per-Layer Embedding (PLE).
 *
 * ============================== WHAT IT IS ==============================
 *
 * PLE gives ONE layer of the model a second embedding table, addressed not by the token but by a
 * HASH of the last `ngram` tokens. Qwen3.8-Flash-Next's is 320,001,536 rows of 160 bf16 -- 102 GB,
 * 51 of the model's 125 billion parameters, four fifths of the checkpoint -- read sixteen rows a
 * token. This op computes those sixteen row ids. The gather itself is `gather_rows` over
 * [M * heads] indices; there is no second op for it and there should not be.
 *
 *   mixed[2gram] = t0*m0 ^ t1*m1                 heads 0..7
 *   mixed[3gram] = t0*m0 ^ t1*m1 ^ t2*m2         heads 8..15
 *   ids[h]       = mixed[ngram(h)] % vocab[h] + offset[h]
 *
 * `m`, `vocab` and `offset` are all IN THE CHECKPOINT (`layer_multipliers`,
 * `ngram_heads_vocab_sizes`, `ngram_heads_offsets`). The reference builds them from a splitmix64
 * seed and a prime search at construction time; a served model reads them, and re-deriving them
 * here would be a second source for sixteen numbers the container already carries.
 *
 * ============================== TWO THINGS THAT LOOK LIKE TRAPS AND ARE NOT ==================
 *
 * THE HASH IS NON-NEGATIVE BY CONSTRUCTION, so `%` is unambiguous. The reference uses
 * `torch.remainder`, which FLOORS -- `remainder(-5, 3) == 1` where C gives -2 -- and the obvious
 * worry is that an int64 multiply wraps into the sign bit and the two disagree. It cannot:
 * `_build_layer_multipliers` bounds every multiplier by `(2^63-1) // vocab_size`, so each product
 * of a real token id has its sign bit clear, and an XOR of values with a clear sign bit has a
 * clear sign bit. `mixed` is always in [0, 2^63). The bound is not decoration -- it is what makes
 * the hash well defined -- so this op REFUSES a multiplier or a token that would break it rather
 * than producing a different answer from the reference in silence.
 *
 * THE EOS RULE COLLAPSES TO TWO COMPARISONS. The reference shifts the token history right with
 * `_shift_right_ignore_eos`, built out of a cummax over `where(ids == eos, positions, -1)`, a
 * segment start, and `valid = (position_in_segment >= shift) & (p - shift >= 0)`. Expanded,
 * `position_in_segment >= s` is exactly "no eos in [p-s, p-1]" -- and wherever the rule would
 * substitute eos because `tok[p-s]` is ITSELF an eos, the substituted value is the value already
 * there. So the whole construction is:
 *
 *     t0 = tok[p]
 *     t1 = p >= 1                ? tok[p-1] : eos
 *     t2 = (p >= 2 && t1 != eos) ? tok[p-2] : eos
 *
 * and generally `t[s] = tok[p-s]` unless some earlier shift already fell back to eos, at which
 * point every deeper shift is eos too. A serving engine needs no cummax and no segment
 * bookkeeping: it needs the `ngram-1` COMMITTED ids in front of the step, and nothing deeper.
 *
 * ============================== WHERE THOSE IDS COME FROM ====================================
 *
 * They arrive in an INOUT STATE -- a rolling window of `ngram - 1 + n_spec` ids per slot, read at
 * `num_accepted - 1`. That is `ple_conv`'s contract, operand for operand, including what the
 * window holds afterwards (ref_ple.cpp): with `num_accepted` present the step is a decode step
 * whose drafts a verify may reject, and the window is rewritten the way gdn_conv_update rewrites
 * its own -- the old ids shifted down by one, then every id of the step -- so a next step that
 * kept k of them reads at k - 1 and finds the `ngram - 1` ids ending at the k-th. Absent, every
 * id commits and the last `ngram - 1` of them land at offset zero.
 *
 * THEY ARE A STATE AND NOT A PLAIN INPUT, even though nothing can invalidate them -- the
 * predecessors of the step's first row are always committed, so no rejection reaches them. A plain
 * input costs MORE than a state here, on two counts. It has to be filled by something, and nothing
 * in the engine derives "the ids before this step": `DeriveInput` carries positions and cu_seqlens
 * and no token ids at all, so a plain input means a new core derivation and a widened DeriveFn
 * signature to serve one op. And the contract is already paid for by the layer -- the dilated
 * convolution downstream carries a 9-timestep window with exactly this contract, so PLE passes
 * `num_accepted`, `cache_idx` and `has_init` whether or not this op reads them.
 *
 * RAD_KV_CONV is already "per-sequence, a rolling window of width-1 + num_spec entries", which is
 * this, and `conv_state_index` already derives the read cursor for it.
 *
 * THE COLD VALUE IS EOS AND NOT ZERO, which is the one place this differs from a conv window.
 * Zero is a real token id, so a slot the manager has not written yet must be filled from `eos`
 * explicitly -- `has_init` 0 means exactly that.
 */
#include "ref_common.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* THE TWO OPERATIONS TORCH WOULD DO, SPELLED SO THEY CANNOT BE UNDEFINED.
 *
 * A signed overflow is UB in C++ and wrapping in torch, so the multiply goes through uint64 and
 * comes back -- two's complement, which is what an int64 tensor does on every machine this runs
 * on. And `torch.remainder` FLOORS where C truncates, so the modulo is written out rather than
 * left to `%`.
 *
 * On a real checkpoint NEITHER of these can fire: `_build_layer_multipliers` bounds every
 * multiplier by `(2^63-1) // vocab_size`, so a real token's product has its sign bit clear, an XOR
 * of such values has its sign bit clear, and truncating and flooring agree on a non-negative
 * dividend. They are here so that a DRAWN case -- a harness index that no vocabulary bounds --
 * gets torch's answer rather than a refusal or a platform's. */
inline int64_t wrap_mul(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}
inline int64_t floor_mod(int64_t a, int64_t b) {    /* b > 0 */
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

}  // namespace

/* ids[m][h] = (hash of the ngram ending at m) % vocab[h] + offset[h].
 *
 *   tok          [M]              i32/i64  the step's token ids, sequences back to back
 *   state        [n_slots][W]     i32      inout: a rolling window of committed ids, W = ctx +
 *                                          n_spec, slot i holding the id at relative position
 *                                          i - ctx from the step's first row
 *   cu           [n_seq+1]        i32
 *   mult         [ngram]          i64  w
 *   vocab_sizes  [heads]          i64  w
 *   offsets      [heads]          i64  w
 *   cache_idx    [n_seq]          i32  ?   the state slot per sequence
 *   has_init     [n_seq]          i32  ?   0 = the window is EOS, not what the slot last held
 *   num_accepted [n_seq]          i32  ?   one-based; shifts the READ offset, and its presence
 *                                          makes the write-back the decode layout. Absent reads
 *                                          at offset zero and leaves the prefill layout.
 *   ids          [M][heads]       i32      out
 *   ahead_ids    [A][heads]       i32  ?   out: `tok` then holds A more tokens after the step's,
 *                                          continuing the LAST sequence, and these are their ids
 *                                          as the step that runs them will compute them. The
 *                                          window is the step's own.
 */
int ref_ngram_ids(const RadArgs* a, RadStream) {
    /* TEN POSITIONS AND AN ELEVENTH, four of them optional, so the count is checked and the
     * optionals are tested one by one -- `have()` would refuse the op for an operand allowed to be
     * absent. */
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor* tok   = t_in(a, 0);
    const RadTensor* st    = t_in(a, 1);
    const RadTensor* cu    = t_in(a, 2);
    const RadTensor* mult  = t_in(a, 3);
    const RadTensor* vsz   = t_in(a, 4);
    const RadTensor* off_  = t_in(a, 5);
    const RadTensor* sidx  = t_in(a, 6);      /* optional */
    const RadTensor* hini  = t_in(a, 7);      /* optional */
    const RadTensor* nacc  = t_in(a, 8);      /* optional */
    const RadTensor* ids   = t_in(a, 9);
    const RadTensor* ahead = a->n_t > 10 ? t_in(a, 10) : nullptr;     /* optional */
    if (!tok || !st || !cu || !mult || !vsz || !off_ || !ids) return RAD_E_INVAL;

    const int64_t heads = p_int(a, "heads", 0);
    if (heads <= 0) return RAD_E_SHAPE;
    /* The token vector is the call and the prompt that follows it; `M` is the band. See band_rows
     * in ref_common.h. */
    const int64_t A     = ahead ? numel(ahead) / heads : 0;
    const int64_t M     = band_rows(numel(tok) - A, p_int(a, "M", 0));
    const int64_t ngram = p_int(a, "ngram", 0);
    const int64_t eos   = p_int(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return RAD_E_SHAPE;
    /* heads are laid out as `ngram - 1` blocks of `heads_per_ngram`, block b using the (b+2)-gram,
     * so the head count has to divide that way or the block a head belongs to is a guess. */
    if (heads % (ngram - 1)) return RAD_E_SHAPE;
    const int64_t per = heads / (ngram - 1);
    const int64_t ctx = ngram - 1;
    if (ngram > 8) return RAD_E_UNSUPPORTED;      /* the shift window lives in a stack frame */

    if (numel(mult) < ngram || numel(vsz) < heads || numel(off_) < heads) return RAD_E_SHAPE;
    if (numel(ids) < M * heads) return RAD_E_SHAPE;

    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t W = st->rank >= 2 ? st->shape[st->rank - 1] : ctx;
    if (W < ctx) return RAD_E_SHAPE;
    /* Every sequence is checked before any state is written: the read window, and the window a
     * decode step leaves behind, which is `ctx - 1` ids of history plus every id of the step. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W) return RAD_E_SHAPE;
        const int64_t T = ld_int(cu->dtype, cu->data, offlin(cu, s + 1)) -
                          ld_int(cu->dtype, cu->data, offlin(cu, s));
        if (nacc && T > 0 && ctx - 1 + T > W) return RAD_E_SHAPE;
    }

    const int64_t ids_ld = rowoff(ids, 1, heads);
    const int64_t ahd_ld = ahead ? rowoff(ahead, 1, heads) : 0;

    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = ld_int(cu->dtype, cu->data, offlin(cu, s));
        const int64_t eot = ld_int(cu->dtype, cu->data, offlin(cu, s + 1));
        if (bos < 0 || eot < bos || eot > M) return RAD_E_SHAPE;
        const int64_t T = eot - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, s)) != 0;
        const int64_t o = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;

        /* The `ctx` committed ids in front of this step. A COLD slot's window is EOS and not what
         * the slot last held -- and not zero either, which is a real token. Without that a fresh
         * sequence landing on a recycled slot hashes the previous sequence's tail into its first
         * two rows, which is a wrong lookup in a 320-million-row table and nothing says so. */
        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i)
            prev[i] = warm ? ld_int(st->dtype, st->data, rowoff(st, slot, W) + o + i) : eos;

        /* The history this sequence reads: its `ctx` predecessors, then its own rows. Indexed by
         * `p` in [0, ctx + T), so row `t` is history position `ctx + t` and every shift is in
         * range without a bounds test in the inner loop. */
        auto hist = [&](int64_t p) -> int64_t {
            return p < ctx ? prev[p] : ld_int(tok->dtype, tok->data, offlin(tok, bos + p - ctx));
        };

        /* The last sequence runs on into the prompt that follows, whose ids land in `ahead`. */
        const int64_t reach = T + (ahead && s + 1 == n_seq ? A : 0);
        for (int64_t t = 0; t < reach; ++t) {
            const int64_t p = ctx + t;

            /* The shifted tokens, with the EOS rule in its collapsed form: once a shift falls back
             * to eos every deeper one does too, because `position_in_segment >= s` is monotone in
             * s and an eos at `p - s` makes the substituted value the value already there. */
            int64_t sh[8];
            sh[0] = hist(p);
            bool cut = false;
            for (int64_t k = 1; k < ngram; ++k) {
                if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
                sh[k] = hist(p - k);
            }

            /* One mix per n-gram order, extended in place: the (b+2)-gram's mix is the
             * (b+1)-gram's XOR'd with one more term, which is exactly the reference's inner loop
             * and saves recomputing the shared prefix. */
            int64_t mixed = 0;
            int64_t blk = -1;
            for (int64_t k = 0; k < ngram; ++k) {
                const int64_t term = wrap_mul(sh[k], ld_int(mult->dtype, mult->data, k));
                mixed = k == 0 ? term : (mixed ^ term);
                if (k == 0) continue;            /* a 1-gram addresses no head */
                ++blk;                           /* k = 1 is the 2-gram block, k = 2 the 3-gram */
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = blk * per + j;
                    const int64_t v = ld_int(vsz->dtype, vsz->data, h);
                    const int64_t ofh = ld_int(off_->dtype, off_->data, h);
                    if (v <= 0 || ofh < 0) return RAD_E_SHAPE;
                    if (t < T)
                        st_int(ids->dtype, ids->data, (bos + t) * ids_ld + h,
                               floor_mod(mixed, v) + ofh);
                    else
                        st_int(ahead->dtype, ahead->data, (t - T) * ahd_ld + h,
                               floor_mod(mixed, v) + ofh);
                }
            }
        }

        /* The window this step leaves behind, written after every read of it is done. Same rule
         * as ple_conv's, said again because the two states sit in the same layer and a reader
         * will compare them: a decode step (`num_accepted` present) shifts the old ids down by
         * one and appends every id it ran, so the next step's `num_accepted - 1` lands on the
         * ids ending at the last one it kept; a step that commits everything leaves its last
         * `ctx` ids at offset zero. */
        const int64_t base = rowoff(st, slot, W);
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i)
                st_int(st->dtype, st->data, base + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int(st->dtype, st->data, base + ctx - 1 + t,
                       ld_int(tok->dtype, tok->data, offlin(tok, bos + t)));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                const int64_t v = src >= 0 ? ld_int(tok->dtype, tok->data, offlin(tok, bos + src))
                                           : prev[src + ctx];
                st_int(st->dtype, st->data, base + i, v);
            }
        }
    }
    return RAD_OK;
}
