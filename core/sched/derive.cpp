/* derive.cpp -- the derivation table. One row per name; adding one is adding a row.
 */
#include "derive.h"

#include <cstring>

namespace rad {

/* ------------------------------------------------------------------ seq_start_index */
/* [n_seq+1] -- where each sequence's tokens begin in this step's flat token array, and the total.
 * This is cu_seqlens, exposed as a declared buffer so a plugin can name it as an operand instead
 * of reaching into RadBatch. It costs a copy of n_seq+1 int32 and it means an op's operand list
 * says what it reads. */
static int d_seq_start_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    int64_t n = in.n_seq + 1;
    if (n > cap) return RAD_E_FULL;
    std::memcpy(out, in.cu_seqlens, (size_t)n * sizeof(int32_t));
    *n_out = n;
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv_state_index */
/* [n_seq] -- the entry of the conv state cache this sequence's update reads from.
 *
 * libr4d treats the conv state as a rolling window of `width-1 + num_spec` entries and reads at
 * the slot the last accepted token left, so a speculative rejection is a change of READ OFFSET
 * rather than a recompute (spec §10). The window base is the sequence's state slot; the offset is
 * the cursor the scheduler advances by 1 + num_accepted per verify. Both numbers live in the
 * request's bookkeeping, which is why this is a scheduler-side derivation and not a kernel one. */
static int d_conv_state_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    if (in.n_seq > cap) return RAD_E_FULL;
    if (!in.conv_slot) return RAD_E_STATE;
    const int32_t W = in.conv_window > 0 ? in.conv_window : 1;
    for (int64_t i = 0; i < in.n_seq; ++i) {
        const StepEntry& e = in.plan->e[(size_t)i];
        int32_t slot = in.conv_slot[i];
        out[i] = slot < 0 ? -1 : (int32_t)(slot * W + (e.conv_cursor % W));
    }
    *n_out = in.n_seq;
    return RAD_OK;
}

/* ------------------------------------------------------- the linear-state chunk walk */
/* The chunked linear-attention kernels tile a sequence in fixed chunks measured from the
 * sequence's own origin, not from the start of the step's token array. In a varlen batch those two
 * disagree, so the boundaries have to be derived: a chunk may never cross a sequence boundary, and
 * a chunk boundary must land on an absolute multiple of the chunk length or the recurrent state
 * carried into the next chunk is the state of a partial tile.
 *
 * The scheduler already guarantees a prefill chunk ends on a multiple of the quantum (geometry.h),
 * so in a pure prefill step the interior boundaries are exactly the multiples; the walk still runs
 * because a mixed step's decode rows start at arbitrary positions. */
template <bool WantSeq>
static int chunk_walk(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    const int64_t C = in.state_chunk > 0 ? in.state_chunk : 1;
    int64_t k = 0;

    if (!WantSeq) {
        if (cap < 1) return RAD_E_FULL;
        out[k++] = 0;
    }
    for (int64_t i = 0; i < in.n_seq; ++i) {
        int64_t b = in.cu_seqlens[i], e = in.cu_seqlens[i + 1];
        if (e <= b) continue;
        int64_t p0 = in.positions[b];
        /* First interior boundary: the next absolute multiple of C strictly after p0. */
        int64_t next = ((p0 / C) + 1) * C;
        for (int64_t abs = next; abs < p0 + (e - b); abs += C) {
            if (k >= cap) return RAD_E_FULL;
            out[k++] = WantSeq ? (int32_t)i : (int32_t)(b + (abs - p0));
        }
        if (k >= cap) return RAD_E_FULL;
        out[k++] = WantSeq ? (int32_t)i : (int32_t)e;
    }
    *n_out = k;
    return RAD_OK;
}

/* [n_chunks+1] token indices: chunk c covers [out[c], out[c+1]). out[0] is 0 and the last entry is
 * n_tok, so the array reads exactly like cu_seqlens does. */
static int d_state_chunk_bounds(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<false>(in, out, cap, n_out);
}

/* [n_chunks] -- which sequence each chunk belongs to. Recoverable from the bounds by a search
 * against cu_seqlens, but the search is per chunk in every layer that wants it, and this is one
 * pass shared by all of them. */
static int d_state_chunk_seq(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<true>(in, out, cap, n_out);
}

/* ------------------------------------------------------------------ the table */
static const DeriveRow g_rows[] = {
    { "seq_start_index",  d_seq_start_index,
      "[n_seq+1] first token index of each sequence in this step, and the total" },
    { "conv_state_index", d_conv_state_index,
      "[n_seq] conv rolling-window entry to read, base slot * window + accepted cursor" },
    { "state_chunk_bounds", d_state_chunk_bounds,
      "[n_chunks+1] token indices bounding each linear-state chunk; never crosses a sequence" },
    { "state_chunk_seq",    d_state_chunk_seq,
      "[n_chunks] the sequence each chunk belongs to" },
};

int derive_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < (int)(sizeof g_rows / sizeof g_rows[0]); ++i)
        if (std::strcmp(g_rows[i].name, name) == 0) return i;
    return -1;
}

const DeriveRow* derive_row(int i) {
    return (i >= 0 && i < (int)(sizeof g_rows / sizeof g_rows[0])) ? &g_rows[i] : nullptr;
}

int derive_count() { return (int)(sizeof g_rows / sizeof g_rows[0]); }

}  /* namespace rad */
