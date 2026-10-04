/* gen_config.h -- the sampling a checkpoint states for itself.
 *
 * A checkpoint ships a generation_config.json saying how it expects to be sampled, and serving it
 * at anything else is serving a different model. This is the reader for that file, in the three
 * forms it reaches a running engine: as a file named on the command line, as a file beside the
 * container, and as the `generation.*` metadata rad-convert flattened into the container itself.
 *
 * WHY IT IS ITS OWN COMPONENT. The values decide what every request that sends no sampler fields
 * is served with, and the failure they prevent is silent: SamplingParams' defaults are an
 * untruncated sampler -- temperature 1, top_k off, top_p 1 -- and a model tuned against a
 * truncated tail, sampled with its whole vocabulary live, will now and then draw a token from far
 * down its distribution. End-of-turn is one of those tokens, so the symptom is not prose that
 * reads badly, it is a reply that stops early: an agent turn that carries text and no tool call,
 * which reads from the outside as the server having dropped the request. A defect with no
 * signature of its own is one to hold a test against, and a file-static helper cannot have one.
 *
 * NOTHING HERE REFUSES. These values arrive with the model rather than from a person, and a
 * checkpoint with one implausible entry is still a checkpoint worth serving: a field outside its
 * legal range is warned about and dropped, leaving whatever was underneath it. The command-line
 * overrides are checked in core/config.cpp instead, where the answer can still be "that is not a
 * temperature" before anything starts.
 */
#pragma once
#include "rad_core.h"

#include <functional>
#include <string>

namespace rad {

/* One config's sampler fields. -1 IS "THIS SOURCE DID NOT SAY" RATHER THAN A VALUE, because every
 * field here has a legal setting that would otherwise be indistinguishable from silence: 0 is a
 * real temperature (greedy), 0 is a real top_k (off) and 0 is a real min_p. A source that cannot
 * say "unset" cannot have another resolved underneath it. */
struct GenSampling {
    float temp = -1.0f, top_p = -1.0f, min_p = -1.0f;
    int   top_k = -1;

    bool any() const { return temp >= 0.0f || top_p >= 0.0f || min_p >= 0.0f || top_k >= 0; }

    /* Overlay onto `base`, field by field. A field this did not state leaves base's alone, which
     * is what lets sources stack: the container underneath, a file over it, a flag over that. */
    SamplingParams onto(SamplingParams base) const;

    /* "temperature 0.70, top_k 20 (off), top_p 0.95, min_p 0.00" -- for the line the server logs
     * about what it settled on. Which sampler a deployment serves decides what its output looks
     * like, and the untruncated fallback is the one a reader most needs told: it is
     * indistinguishable from a configured sampler in every other way. */
    static std::string describe(const SamplingParams& sp);
};

/* Parse the sampler fields out of generation_config.json TEXT. `src` names the origin in any
 * warning. Returns whether anything was taken. Unparseable text warns and takes nothing. */
bool gen_sampling_parse(const std::string& json_text, const char* src, GenSampling* out);

/* The same from a file. Absent is not an error unless `required` -- a checkpoint that ships no
 * generation config is served from whatever lies underneath it, not refused. */
bool gen_sampling_from_file(const std::string& path, bool required, GenSampling* out);

/* The same out of a flat key/value table: the container's `generation.*` metadata, whose values
 * are all strings because that is how rad-convert writes free-form metadata. `get` is handed a
 * bare field name ("top_k") and returns null for a key the table does not hold. */
bool gen_sampling_from_kv(const std::function<const char*(const char*)>& get, GenSampling* out);

/* `<model>.rad` -> `<model>.generation.json`.
 *
 * THE SIDECAR EXISTS BECAUSE A CONTAINER CANNOT BE EDITED. Metadata lives ahead of the weight
 * data, so adding a key to a converted model means rewriting every byte of it, and a container
 * that carries no `generation.*` metadata reads its sampling from nowhere. A file beside it is
 * the one way to correct such a container without converting the weights a second time. */
std::string gen_sidecar_path(const std::string& model_path);

}  /* namespace rad */
