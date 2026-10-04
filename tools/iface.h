/* iface.h -- what the tools need from components outside tools/.
 *
 * rad-info, rad-convert, rad-kbench and rad-tune sit on top of three things the
 * format component does not own: the GGUF reader and the tokenizer.json parser (core/text), and
 * the plugin registry and the declare phase (core/plugin, core/build). This header is the one
 * place those expectations are written down, so they are one file to check rather than one per
 * tool to grep, and so the tools have a single seam if any of them moves.
 *
 * rad-convert, rad-kbench and rad-tune all need exactly the same thing -- the plugins loaded in
 * hierarchy order and the declare phase run against a model's metadata -- because all three are
 * the engine's startup with a different last step (spec §4.2, §15, §17).
 */
#pragma once
#include "rad_core.h"
#include "format/radfile.h"

#include "text/gguf.h"
#include "build/startup.h"

#include <string>
#include <vector>

namespace rad {

/* The vocab pipeline lives in core/text: VocabBuild, parse_tokenizer_json, vocab_from_gguf and
 * vocab_serialize. A tool that needs it includes text/tokenizer.h directly -- there is nothing to
 * restate here, and restating it is how two definitions of VocabBuild come to exist. The
 * container's half of that contract is RadWriter::intern_string() and set_vocab_section(). */

/* Plugin loading and the declare phase are core/build/startup.h -- they are the engine's startup,
 * and rad-convert, rad-kbench and rad-tune are that startup with a different last step. */

/* ================================================================== shared CLI plumbing */
/* Every tool parses its own flags -- five hand-rolled loops is less machinery than one options
 * library, and each usage text is then written for its own tool. What IS shared is the small set
 * of things they must agree about. */

std::string rad_home();                              /* $RADIANCE_HOME, else the install prefix */
std::vector<std::string> rad_hierarchy_from_env();   /* $RADIANCE_KERNELS, colon separated */
std::vector<std::string> rad_split_list(const std::string& s);   /* ':' or ',' separated */

/* Bytes in human units, right-aligned in `w` columns. Every report that prints a size uses it. */
std::string rad_humanb_w(int64_t bytes, int w);

}  /* namespace rad */
