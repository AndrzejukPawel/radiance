/* lq_grid.h -- a quantisation GRID: what codes a weight may take and how its scales sit over it.
 *
 * libquant separates three things a recipe names independently (spec.md §4.5): the GRID, which is
 * the code type and the scale, zero and transform structure around it; the ALGORITHM that picks
 * codes on that grid (round-to-nearest, GPTQ, AWQ, ...); and the encoding the result is stored as,
 * which follows from the grid alone. Every algorithm takes the same grid options, so a grid
 * written once is available to all of them.
 *
 * A grid is parsed from a rule's options:
 *
 *   codes=DT        the code dtype: i2..i8, i16, i32 (symmetric), u1..u8, u16 with zero=
 *                   (asymmetric), or a float code: fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2,
 *                   fp4_e2m1. Required, except beside table=, which implies it.
 *   table=T         a SCALAR CODEBOOK: each code indexes a table of values, which a block's scale
 *                   multiplies. nf4 (QLoRA's NormalFloat), iq4nl (llama.cpp's IQ4_NL values),
 *                   w4nl (the rotated expert grid libr4d's w4nla8h kernel decodes), w5nl (its
 *                   32-level form, libr4d's w5nl64a8h),
 *                   binary {-1, +1}, ternary {-1, 0, +1}, or the values themselves, `;`-separated.
 *                   The codes are the narrowest unsigned width that indexes it.
 *   group=N         one scale per N columns of a row -- block [1 x N]
 *   block=RxC       the scale block in general; `*` is the whole extent: 128x128, 1x*, *x*
 *                   (default: one scale a row, 1x*)
 *   scale=DT        the scale dtype: f32, bf16, f16, fp8_e4m3, e8m0 (default bf16); under scale2=
 *                   also an integer, u4..u8, i6, i8 -- a K-quant's sub-block scale
 *   scale2=DT       a SECOND SCALE LEVEL the first is relative to: f32, bf16, f16, fp8_e4m3, e8m0.
 *                   NVFP4 is fp4 codes, an fp8_e4m3 scale a [1 x 16] and an f32 scale2 a tensor.
 *   scale2_value=X  the second level FIXED at X for every one of its blocks, rather than taken
 *                   from the largest first-level scale under it: exactly representable in scale2=.
 *                   It is a constant a reader can know -- a first level of E4M3 sub-scales over a
 *                   model whose scales sit inside E4M3's range needs no per-row level, only an
 *                   offset (libr4d's w4nl64a8h experts) -- and a first level it would push past its
 *                   dtype's top is held there, said once on stderr.
 *   group2=N        the second level's block, [1 x N]; or
 *   block2=RxC      in general (default: the whole tensor, *x*). A multiple of the first level's
 *                   block in each extent.
 *   zero=DT         an integer zero point a block, making the grid asymmetric: u8, u4 ...
 *   transform=fwhtN rotate each row by the unnormalised Hadamard before quantising
 *   rule=R          how a block's scale is chosen: absmax (the default), mx (the OCP shared
 *                   exponent, the default for an e8m0 scale), fixed (scale_value=, every block),
 *                   search (the scale, or the asymmetric range, of least squared error over a
 *                   sweep of candidates and their least-squares refits, weighted by the imatrix's
 *                   column importances when the converter has them)
 *   scale_value=X   the fixed scale for rule=fixed
 *   clamp=full|sym  a signed code's range: [-2^(b-1), 2^(b-1)-1], or symmetric about zero
 *
 * THE RULES ARE THE ONES THE CONTAINERS IN SERVICE WERE MADE WITH, exactly -- the order of a
 * division, which number is rounded to the scale dtype before the codes are chosen against it, and
 * what a block of zeros gets -- so a recipe reproduces those containers byte for byte. Each is
 * stated where it is implemented. The forms no container in service uses -- a codebook, a second
 * level, the searched rule and the 16- and 32-bit codes -- go through a general path of their own,
 * which leaves those rules where they are.
 */
#pragma once
#include "rad_quant.h"

#include <string>
#include <vector>

namespace lq {

struct Grid {
    uint32_t codes = RAD_DT_INVALID;
    bool     is_float = false;     /* a float code: fp8, fp6, fp4 */
    bool     asym = false;         /* a zero plane */
    bool     wide = false;         /* a 16- or 32-bit integer code, chosen in double */
    uint32_t scale_dt = RAD_BF16;
    uint32_t zero_dt = RAD_DT_INVALID;
    int64_t  br = 1, bc = 0;       /* the scale block; 0 = the whole extent */
    int64_t  fwht = 0;             /* the transform's order, or 0 */
    int      qmin = 0, qmax = 0;   /* an integer code's range */
    float    fmax = 0.0f;          /* a float code's largest magnitude */
    int      emax = 0;             /* a float code's largest normal exponent, for rule=mx */
    enum Rule { ABSMAX, MX, FIXED, SEARCH } rule = ABSMAX;
    double   fixed = 0.0;

    /* A scalar codebook: the value of each code, in code order. */
    std::vector<float> table;
    uint32_t table_dt = RAD_DT_INVALID;
    std::vector<int>   table_order;  /* the codes, by ascending value */
    float    tpeak = 0.0f;           /* the entry of largest magnitude, signed */
    /* The extremes are not mirror images (iq4nl: -127 and +113), so a block's scale takes the
     * sign of its largest-magnitude value and that value lands on `tpeak` exactly. */
    bool     tsigned = false;

    /* A second scale level, which the first is relative to. */
    bool     lv2 = false;
    uint32_t scale2_dt = RAD_DT_INVALID;
    int64_t  br2 = 0, bc2 = 0;       /* its block; 0 = the whole extent */
    double   fixed2 = 0.0;           /* scale2_value: every block's stored value, or 0 */
    bool     l1_int = false;         /* the first level is an integer sub-scale */
    int      l1_lo = 0, l1_hi = 0;   /* its range */
    float    l1max = 0.0f;           /* the largest first-level value its dtype holds */

    /* The planes of the encoding, by index; -1 when the grid has none. */
    int plane_zero() const { return asym ? 2 : -1; }
    int plane_scale2() const { return lv2 ? (asym ? 3 : 2) : -1; }
    int plane_table() const {
        return table.empty() ? -1 : 2 + (asym ? 1 : 0) + (lv2 ? 1 : 0);
    }
    /* Does this grid take the general path (see the head of the file)? */
    bool general() const { return lv2 || rule == SEARCH || !table.empty() || wide; }
};

/* Parse a grid from options. RAD_OK, or RAD_E_INVAL with `why` naming the option. */
int grid_parse(const RadParam* o, int n_o, Grid* g, std::string* why);

/* The encoding a grid writes. */
RadEncoding grid_encoding(const Grid& g);

/* The rows one quantize call may take for this grid: a multiple of every level's block rows, or 0
 * when a scale covers every row and has to see them all. */
int64_t grid_row_block(const Grid& g, int64_t rows);

/* One block's quantiser state: the scale the codes are chosen against, its stored form, the
 * reciprocal, and for an asymmetric grid the zero. */
struct BlockScale {
    float   s = 1.0f;        /* the value codes are chosen against: the stored scale(s), widened */
    float   inv = 1.0f;
    double  inv_d = 1.0;     /* rule=fixed reads its reciprocal in double; see grid.cpp */
    int     zero = 0;
    int     e8 = 0;          /* rule=mx: the shared exponent's E8M0 code */
    float   stored = 1.0f;   /* what the scale plane holds: s, divided by a transform's order --
                                or, under a second level, the first level's own value */
    float   lo = 0.0f;       /* an asymmetric block's low end, which the zero is taken against */
};

/* The scale of the block covering logical rows [r0, r1) and columns [c0, c1) of `w`, f32 [*, cols],
 * already transformed. `imp` is a column's importance for rule=search, or null. Under a second
 * level this is the first level's ideal scale, unrounded; grid_scales finishes it. */
BlockScale grid_block_scale(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1,
                            int64_t c0, int64_t c1, const float* imp = nullptr);

/* One value onto the grid: its code, and the value the code decodes to (for error feedback). */
int   grid_code(const Grid& g, const BlockScale& b, float v);
float grid_decode(const Grid& g, const BlockScale& b, int code);

/* Write a scale (and a zero) into element `i` of its plane row. */
void grid_store_scale(const Grid& g, const BlockScale& b, void* scale_row, void* zero_row,
                      int64_t i);

/* The geometry of a quantize call: where each plane's rows for this call start, as the ABI hands
 * them over, and the plane row widths in bytes. */
struct Planes {
    uint8_t* codes = nullptr;
    uint8_t* scale = nullptr;
    uint8_t* zero = nullptr;
    uint8_t* scale2 = nullptr;
    uint8_t* table = nullptr;
    int64_t  codes_row = 0, scale_row = 0, zero_row = 0, scale2_row = 0;   /* bytes */
    int64_t  scale_cols = 0, scale2_cols = 0;                             /* elements a row */
};
Planes grid_planes(const Grid& g, int64_t cols, void* const* planes);

/* Rotate rows in place when the grid carries a transform. */
void grid_transform_rows(const Grid& g, float* w, int64_t rows, int64_t cols);

/* Every scale of a call's rows, both levels: the first level block by block in plane order, the
 * second level's stored values in theirs. */
struct Scales {
    std::vector<BlockScale> b;
    int64_t nr = 0, nc = 0;            /* first-level blocks */
    int64_t br = 1, bc = 1;            /* their extents, resolved against the call */
    std::vector<float> s2;             /* the second level as stored */
    int64_t nr2 = 0, nc2 = 0;
    const BlockScale& at(int64_t r, int64_t c) const {
        return b[(size_t)((r / br) * nc + c / bc)];
    }
};

/* The whole scale grid of rows [0, rows) of `w`, without choosing a code -- what round-to-nearest
 * codes against and the static-groups form an error-feedback algorithm quantises against. `imp`
 * is the columns' importances for rule=search, or null. */
Scales grid_scales(const Grid& g, const float* w, int64_t rows, int64_t cols,
                   const float* imp = nullptr);

/* Write every scale plane, the zeros and the codebook of a call. */
void grid_store_scales(const Grid& g, const Scales& s, const Planes& p);

/* Store one code into element `i` of a code row, any width this grid writes. */
void grid_store_code(const Grid& g, uint8_t* row, int64_t i, int code);

/* Rows [0, rows) of `w` quantised and decoded again, into `out` [rows, cols]: the weight a reader of
 * this grid would see, for an algorithm that measures a candidate. `rows` covers whole blocks. */
void grid_fake(const Grid& g, const float* w, int64_t rows, int64_t cols, float* out,
               const float* imp = nullptr);

/* A block's state from its scale -- `s` unrounded, `lo` the asymmetric low end the zero is taken
 * against -- rounded as the general path rounds one. For an algorithm that chose the scale itself;
 * a grid with a second level has its own pass and does not take this. */
BlockScale grid_finish(const Grid& g, double s, float lo);

/* Round-to-nearest over rows [0, rows) of `w` (logical rows [row0, row0+rows)). The block scales
 * are taken over each block's own rows, so `rows` must cover whole blocks. */
int grid_rtn(const Grid& g, const float* w, int64_t row0, int64_t rows, int64_t cols,
             const Planes& p, const float* imp = nullptr);

}  /* namespace lq */
