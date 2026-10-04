// stage_rows.cpp -- libstage schemas, constraints, rows and exports.
//
// Schemas are transcribed from libr4d's row table EXACTLY: the first plugin in
// hierarchy order to declare an op fixes its schema, and this plugin is meant
// to lead (--kernels libstage,...), so any disagreement would refuse both
// plugins at load. Constraints are this plugin's own (world 2, exact served
// always, no wire/width limits the transport does not have).
#include "stage.h"

#define ST_ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define ST_NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

#define ST_P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define ST_P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define ST_P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define ST_P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
#define ST_P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define ST_P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }

#define ST_C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define ST_C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define ST_C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define ST_C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define ST_C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }

// ---------------------------------------------------------------- schemas
static const RadParamSpec st_pAllReduce[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "exact",      RAD_P_INT, RAD_REQUIRED },
    { "min_bytes",  RAD_P_INT, RAD_OPTIONAL },
    { "hops",       RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec st_oAllReduce[] = {
    { "x", RAD_OPD_INOUT, 0 },
    { "y", RAD_OPD_OUT,   1 },
};
static const RadParamSpec st_pAllGather[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "row",        RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec st_oAllGather[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec st_pArHcWrite[] = {
    ST_P_INT("M"), ST_P_INT("n"), ST_P_INT("hc"), ST_P_INT("world_size"), ST_P_STR("dtype"),
    ST_P_INTO("wire"),
};
static const RadOperandSpec st_oArHcWrite[] = {
    { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 }, { "h", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec st_pArGatherHcWrite[] = {
    ST_P_INT("M"), ST_P_INT("n"), ST_P_INT("hc"), ST_P_INT("top_k"), ST_P_INT("world_size"),
    ST_P_STR("dtype"), ST_P_STRO("act"), ST_P_INTO("wire"),
};
static const RadOperandSpec st_oArGatherHcWrite[] = {
    { "y_expert", RAD_OPD_IN, 0 }, { "expert_w", RAD_OPD_IN, 0 }, { "sorted_tok", RAD_OPD_IN, 0 },
    { "shared", RAD_OPD_IN, 1 },   { "shared_gate", RAD_OPD_IN, 1 },
    { "y", RAD_OPD_INOUT, 0 },     { "inj", RAD_OPD_IN, 0 },       { "h", RAD_OPD_INOUT, 0 },
};
static const RadParamSpec st_pArRmsNormQuantFp8[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    ST_P_F64("eps"),
    { "group",      RAD_P_INT, RAD_REQUIRED },
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    ST_P_F64O("wadd"),
    { "wire",       RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec st_oArRmsNormQuantFp8[] = {
    { "x",        RAD_OPD_INOUT,  0 },
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
};
static const RadParamSpec st_pArLnHadQuant[] = {
    { "M",          RAD_P_INT, RAD_REQUIRED },
    { "n",          RAD_P_INT, RAD_REQUIRED },
    ST_P_F64("eps"),
    { "group",      RAD_P_INT, RAD_REQUIRED },
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    ST_P_F64O("wadd"),
    { "wire",       RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec st_oArLnHadQuant[] = {
    { "x",        RAD_OPD_IN,     0 },
    { "residual", RAD_OPD_INOUT,  1 },
    { "w",        RAD_OPD_WEIGHT, 0 },
    { "q",        RAD_OPD_OUT,    0 },
    { "scale",    RAD_OPD_OUT,    0 },
    { "out_bf16", RAD_OPD_OUT,    1 },
};

#define ST_SCHEMA(op, p, o, doc) { op, p, ST_NELEM(p), o, ST_NELEM(o), doc }
static const RadOpSchema st_schemas[] = {
    ST_SCHEMA("all_reduce", st_pAllReduce, st_oAllReduce,
        "sum across ranks, f32, ascending rank order."),
    ST_SCHEMA("all_gather", st_pAllGather, st_oAllGather,
        "gather every rank's shard."),
    ST_SCHEMA("ar_hc_write", st_pArHcWrite, st_oArHcWrite,
        "the two-rank all-reduce and the gated residual's write that follows it, in one "
        "kernel. Both wires."),
    ST_SCHEMA("ar_gather_hc_write", st_pArGatherHcWrite, st_oArGatherHcWrite,
        "moe_gather, then ar_hc_write, in one kernel: the routed experts' rows summed back to "
        "token order (with the shared arm folded when present) ARE the all-reduce's message, "
        "and the gated residual's write follows the reduction. Both wires."),
    ST_SCHEMA("ar_rmsnorm_quant_fp8", st_pArRmsNormQuantFp8, st_oArRmsNormQuantFp8,
        "the two-rank all-reduce and the residual add, RMS norm and block-scaled fp8 "
        "quantisation that always follow it, in one kernel. Both wires."),
    ST_SCHEMA("ar_ln_had_quant_i8", st_pArLnHadQuant, st_oArLnHadQuant,
        "the two-rank all-reduce and the residual add, RMS norm, rotation and quantisation "
        "that always follow it, in one kernel."),
};

// ---------------------------------------------------------------- shapes
// Buffer extents for tools calling these kernels cold. numel/M read as high
// bounds (a ranged declaration describes the whole band).
static int st_shape_ar(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out) return RAD_E_INVAL;
    if (operand != 0 && operand != 1) return RAD_E_UNSUPPORTED;
    const long long numel = stage_getdim(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_SHAPE;
    out->dtype = RAD_BF16;
    out->rank = 1;
    out->shape[0] = numel;
    for (int i = 1; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}
static int st_shape_ag(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out) return RAD_E_INVAL;
    if (operand != 0 && operand != 1) return RAD_E_UNSUPPORTED;
    const long long numel = stage_getdim(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_SHAPE;
    out->dtype = RAD_BF16;
    out->rank = 1;
    out->shape[0] = operand == 0 ? numel : 2 * numel;
    for (int i = 1; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}
static int st_shape_hc(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    // y [M,n], inj [M,hc], h [M,hc*n].
    if (!out) return RAD_E_INVAL;
    if (operand < 0 || operand > 2) return RAD_E_UNSUPPORTED;
    const long long M = stage_getdim(p, n_p, "M", 0);
    const long long n = stage_getdim(p, n_p, "n", 0);
    const long long hc = stage_getdim(p, n_p, "hc", 0);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_SHAPE;
    out->dtype = RAD_BF16;
    out->rank = 2;
    out->shape[0] = M;
    out->shape[1] = operand == 1 ? hc : (operand == 2 ? hc * n : n);
    for (int i = 2; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}
static int st_shape_rnq(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    // x/res/ob [M,n], w [n], q [M,n] u8, s [M,ngrp] f32.
    if (!out) return RAD_E_INVAL;
    if (operand < 0 || operand > 5) return RAD_E_UNSUPPORTED;
    const long long M = stage_getdim(p, n_p, "M", 0);
    const long long n = stage_getdim(p, n_p, "n", 0);
    long long group = stage_getdim(p, n_p, "group", 128);
    if (M <= 0 || n <= 0 || group <= 0) return RAD_E_SHAPE;
    const long long ngrp = (n + group - 1) / group;
    out->rank = (operand == 2) ? 1 : 2;
    if (operand == 2) {
        out->dtype = RAD_BF16;
        out->shape[0] = n;
    } else if (operand == 3) {
        out->dtype = RAD_F8E4M3;
        out->shape[0] = M;
        out->shape[1] = n;
    } else if (operand == 4) {
        out->dtype = RAD_F32;
        out->shape[0] = M;
        out->shape[1] = ngrp;
    } else {
        out->dtype = RAD_BF16;
        out->shape[0] = M;
        out->shape[1] = n;
    }
    for (int i = 2; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}
static int st_shape_lnq(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    // x/res/ob [M,n], w [n], q [M,n] i8, s [M] f32.
    if (!out) return RAD_E_INVAL;
    if (operand < 0 || operand > 5) return RAD_E_UNSUPPORTED;
    const long long M = stage_getdim(p, n_p, "M", 0);
    const long long n = stage_getdim(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_SHAPE;
    out->rank = (operand == 2) ? 1 : (operand == 4 ? 1 : 2);
    if (operand == 2) {
        out->dtype = RAD_BF16;
        out->shape[0] = n;
    } else if (operand == 3) {
        out->dtype = RAD_I8;
        out->shape[0] = M;
        out->shape[1] = n;
    } else if (operand == 4) {
        out->dtype = RAD_F32;
        out->shape[0] = M;
    } else {
        out->dtype = RAD_BF16;
        out->shape[0] = M;
        out->shape[1] = n;
    }
    for (int i = 2; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}
static int st_shape_ghc(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    // ye [T,ye unknown] -> describe as [T,n]; ew [M,top_k]; sorted [T];
    // sh [M,n]; sg [M,1]; y [M,n]; inj [M,hc]; h [M,hc*n].
    if (!out) return RAD_E_INVAL;
    if (operand < 0 || operand > 7) return RAD_E_UNSUPPORTED;
    const long long M = stage_getdim(p, n_p, "M", 0);
    const long long n = stage_getdim(p, n_p, "n", 0);
    const long long hc = stage_getdim(p, n_p, "hc", 0);
    const long long top_k = stage_getdim(p, n_p, "top_k", 0);
    if (M <= 0 || n <= 0 || hc <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const long long T = M * top_k;
    out->dtype = RAD_BF16;
    out->rank = 2;
    switch (operand) {
        case 0: out->shape[0] = T; out->shape[1] = n; break;
        case 1: out->shape[0] = M; out->shape[1] = top_k; break;
        case 2: out->dtype = RAD_I32; out->shape[0] = T; out->shape[1] = 1; break;
        case 3: out->shape[0] = M; out->shape[1] = n; break;
        case 4: out->shape[0] = M; out->shape[1] = 1; break;
        case 5: out->shape[0] = M; out->shape[1] = n; break;
        case 6: out->shape[0] = M; out->shape[1] = hc; break;
        default: out->shape[0] = M; out->shape[1] = hc * n; break;
    }
    for (int i = 2; i < RAD_MAX_RANK; ++i) out->shape[i] = 0;
    return RAD_OK;
}

// ---------------------------------------------------------------- constraints
// world 2, exact served always (serving a lossy request exactly is safe),
// dtypes this transport sums. No width/wire limits: the exchange moves bytes.
static const RadConstraint st_cAr[] = {
    ST_C_EQ("world_size", 2), ST_C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
};
static const RadConstraint st_cAg[] = {
    ST_C_EQ("world_size", 2), ST_C_IN("dtype", "bf16 fp16 f16 fp32 f32"),
};
static const RadConstraint st_cHc[] = {
    ST_C_GE("M", 1), ST_C_GE("hc", 2), ST_C_LE("hc", 8), ST_C_IN("dtype", "bf16"),
    ST_C_EQ("world_size", 2),
};
static const RadConstraint st_cGhc[] = {
    ST_C_GE("M", 1), ST_C_GE("hc", 2), ST_C_LE("hc", 8), ST_C_IN("dtype", "bf16"),
    ST_C_EQ("world_size", 2), ST_C_GE("top_k", 1), ST_C_LE("top_k", 64),
};
static const RadConstraint st_cRnq[] = {
    ST_C_EQ("group", 128), ST_C_IN("dtype", "bf16"), ST_C_EQ("world_size", 2),
};
static const RadConstraint st_cLnq[] = {
    ST_C_IN("dtype", "bf16"), ST_C_EQ("world_size", 2),
};

// Tunables: the same axes libr4d's AR family tunes, so RADIANCE_TUNE pins and
// arch-declared tuned values resolve. nblocks/nthreads size this plugin's
// grids; drain/acq are accepted and ignored (this transport's fences are
// fixed correct).
static const int64_t st_kNb[] = {0, 8, 16, 24};
static const int64_t st_kNt[] = {256, 512, 1024};
static const int64_t st_kDr[] = {1, 3};
static const int64_t st_kAc[] = {0, 1};
static const RadTunable st_tAr[] = {
    {"nblocks", st_kNb, ST_NELEM(st_kNb), 0, "blocks in the grid. 0 = scale with the message"},
    {"nthreads", st_kNt, ST_NELEM(st_kNt), 1024, "threads per block"},
    {"drain", st_kDr, ST_NELEM(st_kDr), 3, "accepted, ignored: fixed system fence"},
    {"acq", st_kAc, ST_NELEM(st_kAc), 0, "accepted, ignored: fixed acquire"},
};

// ---------------------------------------------------------------- launches
extern "C" int stage_ar_init(const RadParam*, int, int, int, void**);
extern "C" void stage_ar_fini(void*);
extern "C" int stage_all_reduce(const RadArgs*, RadStream);
extern "C" int stage_ag_init(const RadParam*, int, int, int, void**);
extern "C" int stage_all_gather(const RadArgs*, RadStream);
extern "C" int stage_hc_init(const RadParam*, int, int, int, void**);
extern "C" void stage_hc_fini(void*);
extern "C" int stage_ar_hc_write(const RadArgs*, RadStream);
extern "C" int stage_ghc_init(const RadParam*, int, int, int, void**);
extern "C" void stage_ghc_fini(void*);
extern "C" int stage_ar_gather_hc_write(const RadArgs*, RadStream);
extern "C" int stage_rnq_init(const RadParam*, int, int, int, void**);
extern "C" void stage_rnq_fini(void*);
extern "C" int stage_ar_rmsnorm_quant_fp8(const RadArgs*, RadStream);
extern "C" int stage_lnq_init(const RadParam*, int, int, int, void**);
extern "C" void stage_lnq_fini(void*);
extern "C" int stage_ar_ln_had_quant_i8(const RadArgs*, RadStream);
extern "C" int64_t stage_lnq_scratch(const RadArgs*);

// ---------------------------------------------------------------- rows
static const RadKernelInfo st_kernels[] = {
    {"stage_all_reduce", "all_reduce", "ar",
     "all-reduce (sum) staged through pinned host memory: each rank publishes "
     "to its own mapped region and a kernel pulls the peer's over PCIe",
     "exactly 2 ranks, no P2P mapping; message in one mapped slot; cudagraph-unsafe (eager)",
     "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cAr), ST_ROW(st_tAr), nullptr,
     stage_ar_init, stage_ar_fini, stage_all_reduce,
     nullptr, nullptr, nullptr, st_shape_ar, nullptr, nullptr, nullptr},
    {"stage_all_gather", "all_gather", "ar",
     "all-gather staged through pinned host memory, plain or row-interleaved",
     "exactly 2 ranks, no P2P mapping; y is [2*numel], rank r at r*numel",
     "any payload -- bytes moved, never summed",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cAg), ST_ROW(st_tAr), nullptr,
     stage_ag_init, stage_ar_fini, stage_all_gather,
     nullptr, nullptr, nullptr, st_shape_ag, nullptr, nullptr, nullptr},
    {"stage_ar_hc_write", "ar_hc_write", "norm",
     "the two-rank all-reduce (staged) and the gated residual's write, in sequence: "
     "h[s] += inj[s] * y over the reduced y",
     "exactly 2 ranks, no P2P mapping; y tight; exact wire only",
     "bf16 throughout; sum rounded to bf16 as the standalone all-reduce stores it",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cHc), ST_ROW(st_tAr), nullptr,
     stage_hc_init, stage_hc_fini, stage_ar_hc_write,
     nullptr, nullptr, nullptr, st_shape_hc, nullptr, nullptr, nullptr},
    {"stage_ar_gather_hc_write", "ar_gather_hc_write", "norm",
     "moe_gather folded in front of a staged ar_hc_write",
     "exactly 2 ranks, no P2P mapping; y tight; exact wire only",
     "bf16 throughout; gather narrows RNE, sum as the standalone all-reduce",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cGhc), ST_ROW(st_tAr), nullptr,
     stage_ghc_init, stage_ghc_fini, stage_ar_gather_hc_write,
     nullptr, nullptr, nullptr, st_shape_ghc, nullptr, nullptr, nullptr},
    {"stage_ar_rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8", "quant",
     "the two-rank all-reduce (staged) and the residual add, RMS norm and "
     "block-scaled fp8 quantisation, in sequence",
     "exactly 2 ranks, no P2P mapping; x tight; group 128; exact wire only",
     "bf16 in, E4M3 out with f32 scale per (row, 128); codes over the BF16-rounded norm",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cRnq), ST_ROW(st_tAr), nullptr,
     stage_rnq_init, stage_rnq_fini, stage_ar_rmsnorm_quant_fp8,
     nullptr, nullptr, nullptr, st_shape_rnq, nullptr, nullptr, nullptr},
    {"stage_ar_ln_had_quant_i8", "ar_ln_had_quant_i8", "quant",
     "the two-rank all-reduce (staged) and the residual add, RMS norm, block "
     "Hadamard and per-row int8 quantisation, in sequence; the reduced message "
     "lives in the launch scratch",
     "exactly 2 ranks, no P2P mapping; x tight; rotation 128/256/512; exact wire only",
     "bf16 in, int8 out with an f32 per-row scale carrying 1/sqrt(had)",
     RAD_DOMAIN_DEVICE, 0, ST_ROW(st_cLnq), ST_ROW(st_tAr), nullptr,
     stage_lnq_init, stage_lnq_fini, stage_ar_ln_had_quant_i8, stage_lnq_scratch,
     nullptr, nullptr, st_shape_lnq, nullptr, nullptr, nullptr},
};

// ---------------------------------------------------------------- exports
static const RadPluginInfo st_info = {
    RAD_PLUGIN_KERNEL,
    "libstage",
    "0.1.0",
    "tensor-parallel collectives staged through pinned host memory for machines "
    "whose GPUs have no PCIe P2P path: all_reduce, all_gather and the four fused "
    "forms, exact wire only, world size 2",
    "",
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &st_info; }
extern "C" int rad_kernel_schema_count(void) { return ST_NELEM(st_schemas); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &st_schemas[i];
}
extern "C" int rad_kernel_count(void) { return ST_NELEM(st_kernels); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    return &st_kernels[i];
}
