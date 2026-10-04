/* derive.h -- per-step metadata the core computes ONCE, by name, into the arena.
 *
 * An architecture plugin that needs linear-state chunk boundaries or conv-state indices declares
 * them at build time as a RAD_BUF_DERIVED buffer and names the derivation. The core computes it
 * once per step; it is NOT recomputed per layer, and layers that share a geometry share the buffer.
 *
 * Sharing like this is what vLLM has no natural place for: it has to establish "same step" through
 * a module-level counter, because nothing there is an object that belongs to a step, and a marker
 * held per metadata-builder instead removes nothing -- every KV-cache group is still the first one
 * its own builder sees. Here the step batch IS that place.
 *
 * Adding a derivation is a row in the table in derive.cpp. The core is not the thing you edit to
 * add a kernel; it is the thing you edit to add a shape of per-step metadata, and that list is
 * short and shared.
 */
#pragma once
#include "batch.h"

namespace rad {

/* Everything a derivation may read. All host memory, already staged for this step: a derivation
 * that needed a device read would be a synchronisation on the step path. */
struct DeriveInput {
    const StepPlan* plan = nullptr;
    const int32_t*  cu_seqlens = nullptr;   /* [n_seq+1] */
    const int32_t*  positions = nullptr;    /* [n_tok] absolute position per token */
    const int32_t*  conv_slot = nullptr;    /* [n_seq] the conv group's state slot, -1 if none */
    int64_t         n_tok = 0;
    int64_t         n_seq = 0;
    int64_t         state_chunk = 1;        /* the resolved linear-state chunk length */
    int32_t         conv_window = 1;        /* conv_width - 1 + max_spec */
};

/* Writes at most `cap` int32 elements to `out` and reports how many. Returns RAD_OK or negative;
 * negative aborts the step, because a plugin issuing an op against a half-filled derived buffer is
 * the silent-wrong-output failure this engine exists to remove. */
typedef int (*DeriveFn)(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out);

struct DeriveRow {
    const char* name;
    DeriveFn    fn;
    const char* doc;
};

/* -1 if the name is not a derivation the core knows. Reported at declare, by name, with the known
 * list -- the same treatment a misspelled op gets (spec §2.3). */
int              derive_index(const char* name);
const DeriveRow* derive_row(int index);
int              derive_count();

}  /* namespace rad */
