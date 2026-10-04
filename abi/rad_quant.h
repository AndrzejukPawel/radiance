/* rad_quant.h -- the quantiser plugin ABI. spec.md §4.5.
 *
 * A quantiser turns a weight, handed to it in f32, into the planes of an ENCODING (rad_encoding.h).
 * It is the third plugin kind beside kernels and architectures, loaded from
 * $RADIANCE_HOME/quantizers, and it is the only place a weight's numbers are chosen: a kernel
 * rearranges planes it is given (rad_abi.h, RadRelayoutFn) and never quantises.
 *
 * rad-convert drives quantisers through a RECIPE -- ordered rules from a logical weight name to a
 * quantiser and its options (spec §4.4). For each weight a rule picks, it asks the quantiser which
 * encoding it would write (`encoding`), lets the architecture plugin declare against that, and
 * then hands the weight over in row blocks (`row_block`, `quantize`), so a 51 GB table streams.
 *
 * WHAT A QUANTISER IS GIVEN. The weight's logical name, shape and movement group, and when the
 * converter has an imatrix, the weight's column importances. Anything else it calibrates on -- a
 * GPTQ Hessian, AutoRound's activations -- is an option it reads itself (`calib=DIR`): what a
 * Hessian is and how it is stored is the quantiser's business, exactly as a layout is a kernel's.
 *
 * WHAT IT HANDS BACK is the canonical planes and nothing else, so its output is readable by every
 * kernel library that accepts the encoding and by the core's own decoder -- which is how rad-convert
 * measures its error without knowing what it did.
 *
 * Pure C, like the rest of abi/.
 */
#ifndef RAD_QUANT_H
#define RAD_QUANT_H

#include "rad_abi.h"
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What a quantiser is told about the weight it is quantising. */
typedef struct RadQuantWeight {
    const char* name;                 /* the logical (declared) name: "blk.7.ffn_down_exps.3.weight" */
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];  /* the logical shape */
    int64_t     rows, cols;           /* the [rows, cols] view every plane is laid over */
    /* The movement group; -1 for none. -1 also when `encoding` is asked at declare, which runs
     * before the weight's group exists -- so an encoding may depend on the name and the shape and
     * not on these, and rad-convert refuses a quantiser whose answer moves once it is told. */
    int32_t     layer, expert;
    /* The imatrix's column importances for this weight -- a second moment of its input, one a
     * column -- or null when the converter has none. Per expert where the imatrix has the form. */
    const float* importance;
} RadQuantWeight;

/* One option a quantiser takes, for `rad-convert --list-quantizers` and for typing a recipe's
 * `key=value` text before the quantiser sees it: an option declared RAD_P_INT arrives as an int,
 * RAD_P_F64 as a double, RAD_P_STR as the text. An option a recipe gives and the table does not
 * name is refused by the converter, by name, before anything is quantised. */
typedef struct RadQuantOption {
    const char* key;
    int         type;                 /* RAD_P_INT | RAD_P_F64 | RAD_P_STR */
    const char* deflt;                /* the default, as text; null for none */
    const char* doc;
} RadQuantOption;

/* The encoding this quantiser writes for `w` under options `o`. RAD_E_UNSUPPORTED declines -- a
 * weight it does not quantise, like a rank-1 norm -- and the converter moves on to the next rule;
 * any other failure is an error in the recipe, said with the weight's name. */
typedef int (*RadQuantEncodingFn)(const RadParam* o, int n_o, const RadQuantWeight* w,
                                  RadEncoding* out);

/* The rows one `quantize` call may take: every call but the last gets a multiple of this, starting
 * on a multiple of it. 0 means the whole weight in one call -- what anything with a cross-row
 * decision needs (a per-tensor scale it derives, a rotation it learns, a Hessian it shares). It must
 * be a multiple of every TILED plane's block rows. */
typedef int64_t (*RadQuantRowBlockFn)(const RadParam* o, int n_o, const RadQuantWeight* w);

/* Quantise logical rows [row0, row0 + rows) of `w`, given as f32 [rows, cols] in `src`, into the
 * planes. `planes[i]` points at the element row of plane i that covers logical row `row0` -- for a
 * TILED plane of block rows br that is row row0 / br, for a plane covering the whole extent and for
 * a TABLE plane it is the plane's start -- and the call writes exactly the rows its block covers.
 * The buffers arrive zeroed. `src` may be modified. */
typedef int (*RadQuantizeFn)(const RadParam* o, int n_o, const RadQuantWeight* w,
                             float* src, int64_t row0, int64_t rows, void* const* planes);

/* Decode rows [row0, row0 + rows) back to f32 [rows, cols], for an encoding whose scheme the core
 * does not decode. Null for a quantiser that writes only "plain" and "affine", which the core
 * decodes itself. `planes[i]` points at plane i's start.
 *
 * It is how a scheme of a quantiser's own is READ: the oracle checking a kernel against the
 * reference, rad-kbench drawing a weight, rad-convert measuring the quantiser's error all decode
 * through it. The core asks every loaded quantiser that has one, in load order, and takes the
 * first answer that is not RAD_E_UNSUPPORTED -- so a decoder answers RAD_E_UNSUPPORTED for an
 * encoding it did not write. `w` names the weight where the caller knows it; a decoder must not
 * need more than the encoding, `rows` and `cols`, because the oracle knows no more. */
typedef int (*RadQuantDecodeFn)(const RadEncoding* e, const RadQuantWeight* w,
                                const void* const* planes, int64_t row0, int64_t rows,
                                float* out);

typedef struct RadQuantizerInfo {
    const char*           name;      /* what a recipe names: "fp8-block", "int", "gptq" */
    const char*           doc;
    const RadQuantOption* options;
    int                   n_options;
    RadQuantEncodingFn    encoding;
    RadQuantRowBlockFn    row_block;
    RadQuantizeFn         quantize;
    RadQuantDecodeFn      decode;    /* optional */
} RadQuantizerInfo;

/* Every quantiser plugin exports these, beside rad_plugin_abi_version and rad_plugin_info (whose
 * kind is RAD_PLUGIN_QUANT). Two plugins naming the same quantiser is refused at load, both named,
 * for the reason two kernel plugins may not disagree about an op's schema: a recipe that names it
 * would mean different numbers depending on load order. */
int                     rad_quant_count(void);
const RadQuantizerInfo* rad_quant_at(int i);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RAD_QUANT_H */
