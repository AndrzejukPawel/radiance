/* startup.h -- steps 1 through 4 of spec §1's startup sequence, shared by the engine and by every
 * tool that needs them.
 *
 *   1. Load kernel plugins from $RADIANCE_HOME/kernels/, in the configured hierarchy order.
 *   3. Look up the architecture id in $RADIANCE_HOME/architectures/.
 *   4. Declare.
 *
 * These live in core rather than in tools/ because rad-convert, rad-check and rad-tune are all
 * the engine's startup with a different last step (spec §4.2, §15, §17). Two implementations of
 * "load the plugins and run declare" would eventually disagree about hierarchy order, and the
 * hierarchy order is what decides which layout got baked into the container.
 */
#pragma once
#include "rad_core.h"
#include "format/radfile.h"
#include "format/checkpoint.h"
#include "plugin/registry.h"
#include "build/rad_build.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rad {

/* A loaded plugin, as a tool names it. */
struct PluginId { std::string name, version; };

struct LoadedPlugins {
    /* Kernel plugins in HIERARCHY ORDER, which is what decides selection, so a report of what ran
     * names them in it. */
    std::vector<PluginId>       kernels;
    std::string                 arch_plugin;
    std::string                 arch_version;
    std::string                 arch_id;
    /* The other half of the key that selected it, "" for an unquantised container. */
    std::string                 arch_quant;

    /* `name@version`, space separated, in hierarchy order: what a benchmark records as the
     * libraries it measured. */
    std::string libraries() const {
        std::string s;
        for (const auto& p : kernels) {
            if (!s.empty()) s += ' ';
            s += p.name + "@" + (p.version.empty() ? "0" : p.version);
        }
        return s;
    }
};

/* $RADIANCE_HOME AS A SEARCH PATH: one or more homes separated by ':', the first taking precedence
 * -- a directory of one's own plugins ahead of an installation's, without copying the installation.
 * Each home holds kernels/, architectures/ and quantizers/, any of which it may lack; a plugin a
 * home earlier on the path supplies shadows one of the same file name later on it. Files that are
 * not plugins -- the tuning cache, rad-kbench's fixtures -- are read from the first home that has
 * them and written to the first home. */
std::vector<std::string> rad_home_dirs(const std::string& home);
/* `<home>/<rel>` for the first home on the path that has it; the first home's when none does. */
std::string rad_home_file(const std::string& home, const std::string& rel);

/* The Registry loaded into is process-lifetime, because a Resolved, a Band and a KernelRow all
 * point into a plugin's static tables and the Program outlives this call. */
/* `arch_id` and `quant` are BOTH halves of the selection key (spec §2.4, rad_builder.h): a family
 * has one plugin per weight format and they all answer the same architecture id. `quant` is the
 * container's own descriptor -- RadModelMeta::quant, RadFileHeader::quant -- and "" is the
 * unquantised case, which matches a plugin that says "" and nothing else. */
int rad_tools_load_plugins(const std::string& radiance_home,
                           const std::vector<std::string>& hierarchy,
                           const std::string& arch_id,
                           const std::string& quant,
                           LoadedPlugins* out);

/* `also` declares into the SAME builder after the architecture plugin has. That is what puts the
 * sampler chain in the same Program: its ops resolve through the same hierarchy at the same
 * time as the plugin's, so a missing sampler kernel is in the same miss list rather than
 * discovered at the first request that asks for it (spec §3.1, §13). Empty for the tools --
 * rad-convert cares about the weight layout and has no sampler. */
int rad_tools_probe(const LoadedPlugins& plugins, const RadModelMeta& meta, RadArchProbe* out);

/* The architecture plugin's declared reply format (rad_arch_chat_format), or null when it
 * declares none and the format is to be derived from the chat template. */
const RadChatFormat* rad_tools_chat_format(const LoadedPlugins& plugins, const RadModelMeta& meta);

/* `ref`, when given, makes this a SIZING declare (RadBuildCtx::shape_probe, Builder::
 * set_reference): the plugin declares against the bands `ref` resolved and nothing is selected
 * or initialised. The Program that comes back is for its buffers and nothing else. */
/* `sources` answers rad_weight_encoding (Builder::set_sources): a container's entries, a
 * checkpoint's tensors, or a recipe. Empty, every weight is read as plain in its declared dtype. */
/* `instances` false selects every kernel and initialises none (Builder::set_instances): for a
 * caller that launches nothing, such as rad-convert planning a container's stored form. */
int rad_tools_declare(const LoadedPlugins& plugins, const RadModelMeta& meta,
                      const RadBuildCtx& ctx, Program* out,
                      const std::function<int(RadBuilder*, int64_t logits_width)>& also = {},
                      const Program* ref = nullptr, const WeightSourceFn& sources = {},
                      bool instances = true);

/* A container's entries as the source of every declaration's encoding. `f` must outlive the
 * declare. */
WeightSourceFn rad_container_sources(const RadFile& f);

/* A CHECKPOINT AS THE SOURCE, served directly (spec §4.3): each declared weight resolved through
 * the name map once, and kept for the loader, which reads the same tensors the answer was made
 * from. Every rank declares at once, so it is locked.
 *
 * Only a TRIVIAL encoding is answered -- bf16, f16, f32, block-scaled FP8 -- because nothing is
 * quantised at load. A weight whose checkpoint form is anything else is recorded with the reason
 * and answered RAD_E_UNSUPPORTED; refused() names them after declare, and the remedy is a
 * container. */
class CheckpointSources {
public:
    explicit CheckpointSources(Checkpoint& ck) : ck_(ck) {}
    WeightSourceFn fn();
    /* After declare: what `name` resolved to, or null. */
    const CkptWeight* find(const std::string& name) const;
    /* Why `name` resolved to nothing, as the name map last answered it; empty if it did. */
    std::string missing(const std::string& name) const;
    /* Every weight that is not served as it stands, "name: why", sorted. */
    std::vector<std::string> refused() const;

private:
    int answer(const std::string& name, const Program& prog, WeightSource* out);
    Checkpoint& ck_;
    mutable std::mutex mu_;
    std::map<std::string, CkptWeight> got_;
    std::map<std::string, std::string> refused_;
    std::map<std::string, std::string> missing_;
};


/* ------------------------------------------------------------------ the reference-kernel gate */
/* A reference library -- one that declares itself so, RadPluginReferenceFn -- is BOTH the fallback
 * that makes an unsupported model run at reduced speed and the oracle rad-kbench measures against
 * (spec §17). The first of those is a hazard in production and
 * the second is not, so which one is happening has to be a decision and not a discovery: a model
 * that silently fell back to the reference implementation for its GEMMs is a model running orders
 * of magnitude below what the machine can do, and nothing in the output says so.
 *
 * So: resolving to a reference library is FATAL unless --debug-accept-reference-kernels was
 * passed. The flag is spelled with `debug` in it because that is what it is for. */

int rad_reference_ops(const Program& p, std::vector<std::string>* out);

/* Report and refuse, or report and continue when `allow`. `what` names the caller ("serve",
 * "the container's layout") so the message says what the consequence actually is. */
int rad_gate_reference_kernels(const Program& p, bool allow, const char* what);

Registry& rad_tools_registry();

}  /* namespace rad */
