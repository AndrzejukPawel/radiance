/* avx_table.cpp -- one ISA level's table of kernel entry points, compiled once per level.
 *
 * There is nothing to read here and that is the point. The table is generated from AVX_OP_LIST in
 * avx_isa.h, so the member order and the initialiser order are the SAME MACRO EXPANSION and cannot
 * drift: a table whose members were typed out twice is a table where `silu` and `sigmoid` end up
 * swapped one day, and the symptom of that is a model that produces fluent wrong text rather than
 * a failure anything reports. The declarations come from the same list too, so an op added to
 * AVX_OP_LIST without an implementation at every level is a LINK error naming the missing symbol
 * and the level it is missing from -- which is the diagnostic you want, four times over.
 */
#include "avx_vec.h"
#include "avx_isa.h"

/* Every kernel at this level, declared. AVX_FN pastes the level's suffix on. */
#define AVX_DECL(nm) extern "C" int AVX_FN(avx_##nm)(const RadArgs*, RadStream);
AVX_OP_LIST(AVX_DECL)
#undef AVX_DECL

#define AVX_ENTRY(nm) AVX_FN(avx_##nm),

extern "C" const AvxKernelTable AVX_FN(avx_table) = {
    AVX_OP_LIST(AVX_ENTRY)
};

#undef AVX_ENTRY
