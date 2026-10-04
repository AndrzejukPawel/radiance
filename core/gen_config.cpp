/* gen_config.cpp -- reading a checkpoint's own sampling. The header carries the argument. */
#include "gen_config.h"

#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace rad {
namespace {

/* A numeric field, taken only if it lands in range. Out of range is warned about and dropped:
 * the value came with the model, and one implausible entry is not a reason to refuse the rest. */
void take(const char* what, const char* src, double v, double lo, double hi, bool lo_open,
          float* dst) {
    const bool ok = (lo_open ? v > lo : v >= lo) && v <= hi;
    if (ok) { *dst = (float)v; return; }
    RAD_WARN("%s states %s %g, which is outside %s%g, %g]; ignoring it",
             src, what, v, lo_open ? "(" : "[", lo, hi);
}

}  /* namespace */

SamplingParams GenSampling::onto(SamplingParams base) const {
    if (temp  >= 0.0f) base.temp  = temp;
    if (top_p >= 0.0f) base.top_p = top_p;
    if (min_p >= 0.0f) base.min_p = min_p;
    if (top_k >= 0)    base.top_k = top_k;
    return base;
}

std::string GenSampling::describe(const SamplingParams& sp) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "temperature %.2f, top_k %d%s, top_p %.2f, min_p %.2f",
                  (double)sp.temp, sp.top_k, sp.top_k == 0 ? " (off)" : "",
                  (double)sp.top_p, (double)sp.min_p);
    return buf;
}

bool gen_sampling_parse(const std::string& json_text, const char* src, GenSampling* out) {
    nlohmann::json j;
    try { j = nlohmann::json::parse(json_text); }
    catch (const std::exception& e) { RAD_WARN("%s: %s; ignoring it", src, e.what()); return false; }
    if (!j.is_object()) { RAD_WARN("%s: not a JSON object; ignoring it", src); return false; }

    /* `do_sample: false` IS THE WHOLE ANSWER when it is there. transformers ignores every other
     * sampler field under it, and greedy is the one thing this sampler spells as a temperature --
     * so taking top_k or top_p as well would describe a sampler the checkpoint did not ask for. */
    auto it = j.find("do_sample");
    if (it != j.end() && it->is_boolean() && !it->get<bool>()) { out->temp = 0.0f; return true; }

    it = j.find("temperature");
    if (it != j.end() && it->is_number()) take("temperature", src, it->get<double>(), 0.0, 2.0, false, &out->temp);
    it = j.find("top_p");
    if (it != j.end() && it->is_number()) take("top_p", src, it->get<double>(), 0.0, 1.0, true, &out->top_p);
    it = j.find("min_p");
    if (it != j.end() && it->is_number()) take("min_p", src, it->get<double>(), 0.0, 1.0, false, &out->min_p);
    it = j.find("top_k");
    if (it != j.end() && it->is_number_integer()) {
        const int64_t v = it->get<int64_t>();
        /* -1 is transformers' spelling of "no top_k" and this sampler's is 0; anything below that
         * is not a count at all. */
        if (v >= -1) out->top_k = (int)(v < 0 ? 0 : v);
        else RAD_WARN("%s states top_k %lld, which is not a count; ignoring it", src, (long long)v);
    }
    return out->any();
}

bool gen_sampling_from_file(const std::string& path, bool required, GenSampling* out) {
    std::ifstream f(path);
    if (!f) {
        if (required) RAD_ERR("%s: cannot be read", path.c_str());
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return gen_sampling_parse(text, path.c_str(), out);
}

bool gen_sampling_from_kv(const std::function<const char*(const char*)>& get, GenSampling* out) {
    /* REBUILT AS JSON RATHER THAN PARSED FIELD BY FIELD, so that a value written by rad-convert
     * and a value read out of the original file go through exactly one set of range checks and
     * one reading of `do_sample`. Two parsers for one schema is how the two come to disagree. */
    nlohmann::json j = nlohmann::json::object();
    static const char* const kFields[] = { "do_sample", "temperature", "top_p", "min_p", "top_k" };
    for (const char* k : kFields) {
        const char* v = get(k);
        if (!v || !*v) continue;
        if (std::strcmp(k, "do_sample") == 0) {
            /* rad-convert writes a JSON boolean as "1" or "0". */
            j[k] = (std::strcmp(v, "0") != 0);
            continue;
        }
        char* end = nullptr;
        const double d = std::strtod(v, &end);
        if (end == v) {
            RAD_WARN("the container states generation.%s as \"%s\", which is not a number; "
                     "ignoring it", k, v);
            continue;
        }
        if (std::strcmp(k, "top_k") == 0) j[k] = (int64_t)d;
        else                              j[k] = d;
    }
    if (j.empty()) return false;
    return gen_sampling_parse(j.dump(), "the container's generation metadata", out);
}

std::string gen_sidecar_path(const std::string& model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot   = model_path.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? model_path.substr(0, dot) : model_path) + ".generation.json";
}

}  /* namespace rad */
