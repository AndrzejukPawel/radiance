/* kfixture.h -- the recorded reference a kernel library is checked against, with no model in the
 * room.
 *
 * WHY THIS EXISTS. Checking against a real container -- declare the model's graph, take the
 * weights out of the .rad in the layout the resolved kernel asked for, run the oracle beside the
 * kernel and compare -- is the right test of a CONTAINER and the wrong test of a LIBRARY. It needs
 * a 27B file nobody has on a build machine, and it spends its whole runtime inside libref, which
 * is a straight-loop host implementation: at model shapes a single `logits_gemm` over a 248K
 * vocabulary costs seconds of wall clock and a single fp8 GEMM tens of seconds, so a full pass
 * runs for tens of minutes to produce numbers that do not change between runs.
 *
 * So the oracle runs ONCE, when a fixture is recorded, and what it produced is checked in. A
 * library is then checked against a fixed artifact in seconds, on any machine, with no model
 * present -- and the artifact is a regression test on libref itself, which a live oracle can never
 * be: if ref's arithmetic drifts, a live comparison moves both sides together and reports
 * agreement.
 *
 * WHAT IS RECORDED, AND WHY IT IS NOT THE WHOLE OUTPUT. A recorded reference for every case at
 * real model shapes is tens of gigabytes -- one `logits_gemm` output alone is 31 MB -- and a
 * repository is not the place for it. What is stored instead, per output operand:
 *
 *   - the sum of squares over EVERY element, and the element count. This is the rel_l2
 *     denominator and it is also the check a sample cannot make: a kernel that writes a correct
 *     first tile and zeros the rest has the right values everywhere the sample looks and the
 *     wrong norm.
 *   - how many elements were non-finite. Not a tolerance question (spec: a NaN on one side and a
 *     number on the other is a failure, checked first).
 *   - the value at N sampled positions, the positions derived from a seed so only the values are
 *     stored. 1024 of them: a kernel wrong in one element per thousand is caught with probability
 *     1 - (1 - 1e-3)^1024 = 64%, one wrong in a hundred with probability better than 1 - 3e-5, and
 *     a kernel wrong in one element out of seventeen million is beneath what an rel_l2 over
 *     seventeen million elements would have reported anyway.
 *
 * INPUTS ARE NOT STORED AT ALL. They are drawn from a seed that IS stored, by the same code on
 * both sides, so the bytes are identical without anyone shipping them. That is also what keeps
 * the file small enough to read at every ctest run.
 */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* How many positions of each output operand are sampled. Fixed rather than a parameter: it is
 * baked into every recorded file, and a file whose sample count differs from the reader's is a
 * file the reader would silently misinterpret. */
constexpr int KF_SAMPLES = 1024;

/* The block the full-tensor reduction is threaded over. Part of the contract for the same reason
 * RAD_FILL_BLOCK is: the partial sums are added back in block order, so the answer is the same
 * however many threads ran, and a reference recorded on one machine reads on another. */
constexpr int64_t KF_REDUCE_BLOCK = 1 << 18;

/* The fill laws, mirroring OpdShape::Fill. Named here rather than shared with it because the
 * fixture is the on-disk contract: OpdShape is free to renumber, this is not. */
enum {
    KF_FILL_NORMAL = 0,
    KF_FILL_SIGMOID = 1,
    KF_FILL_GATE_CUMSUM = 2,
    KF_FILL_TRI_LOWER = 3,
    KF_FILL_UNIT_ROW = 4,
};

/* A geometry key/value, stored TYPED. The alternative is to store `Geometry::str()` and parse it
 * back, and there is no parser -- writing one means deciding whether `1e+07` is a rope theta or a
 * string and whether `f32` is a float, on every load, forever. */
struct KfParam {
    std::string key;
    int         kind = 0;        /* RAD_P_INT | RAD_P_F64 | RAD_P_STR */
    long long   i = 0;
    double      d = 0;
    std::string s;
};

struct KfOperand {
    int      role = 0;
    uint32_t dtype = 0;
    std::vector<int64_t> shape;
    bool     absent = false;

    /* HOW THE INPUT WAS DRAWN. Replay has to reproduce the exact bytes the oracle saw, and the
     * draw is not uniform: an index operand is drawn over a range, a cumulative-length operand is
     * the batch's shape rather than a draw at all, and a gated-delta-net decay is a cumulative sum
     * of log-decays because a normal draw makes the reference overflow. All of that lives in
     * OpdShape at record time and none of it is recoverable from the shape alone. */
    int      fill = 0;
    bool     is_index = false, idx_unique = false, idx_cu = false;
    int64_t  idx_max = 0, idx_const = -1, fill_chunk = 0;

    /* A WEIGHT OPERAND IS PLANES OF AN ENCODING (spec §4.3), and a kernel lays it out from those
     * planes rather than from the tensor the reference reads. So the recording carries the
     * encoding, which planes this operand selects (indices into it) and each one's extents in its
     * own elements; the planes are drawn from the operand's seed and the reference reads what
     * they decode to (kf_planes_logical). Absent for every other operand, and for a weight
     * written by hand in a test, which is then drawn as the tensor it is. */
    bool        has_enc = false;
    RadEncoding enc{};
    std::vector<int>     sel;
    std::vector<int64_t> sel_rows, sel_cols;

    /* The reference, for OUT and INOUT operands of a case that has one. */
    bool     have_ref = false;
    int64_t  ref_n = 0;
    double   ref_sumsq = 0;
    int64_t  ref_nonfinite = 0;
    std::vector<float> sample;   /* KF_SAMPLES values at seed-derived positions */
};

struct KfCase {
    std::string op;
    std::vector<KfParam>   params;
    std::vector<KfOperand> opd;
    uint64_t    seed = 0;
    /* THE RANGED PARAMETER'S VALUE, recorded rather than re-derived. The draw needs it as a
     * scalar -- a cumulative-length operand divides a token count across sequences -- and the key
     * it lives under is the op's own ("M", "q_len", ...). A reader guessing which key to read
     * would guess wrong for exactly the ops whose ranged key is unusual. */
    int64_t     m = 1;
    /* The band this point came from, carried for the report only: "M=65 is the bottom of the
     * band M in (64, 512]" is most of what makes a tail failure legible. */
    std::string band;
    /* A case with no reference: recorded because it is worth TIMING at a real model shape and its
     * output is too large to carry. Speed is measurable without an oracle; correctness is not. */
    bool        speed_only = false;
};

struct Kfixture {
    /* 4 carries each weight operand's encoding and selected planes (KfOperand::enc); an earlier
     * file has neither and is refused -- record it again. */
    uint32_t    version = 4;
    std::string source;        /* the container it was recorded from */
    std::string oracle;        /* the plugin and version that produced the references */
    std::string recorded;      /* an ISO date, so a stale fixture is visible */
    std::vector<KfCase> cases;

    int load(const std::string& path);
    int save(const std::string& path) const;
};

/* The positions sampled in an operand of `n` elements, derived rather than stored. Both the
 * recorder and the reader call this, which is the only reason storing just the values works. */
void kf_sample_positions(uint64_t seed, int64_t n, std::vector<int64_t>* out);

/* The seed for operand `k` of a case. One function so record and replay cannot drift. */
uint64_t kf_operand_seed(uint64_t case_seed, int k);


/* ================================================================== drawing an input
 *
 * ONE IMPLEMENTATION, CALLED BY BOTH SIDES. The recorder and the reader must produce byte-
 * identical inputs or every recorded reference is noise, and the draw is not uniform: an index
 * operand is drawn over a range that depends on the operand before it, a cumulative-length
 * operand is the batch's shape rather than a draw at all, and a gated-delta-net decay is a
 * cumulative sum of log-decays because a normal draw makes the reference overflow to 2e38. Two
 * copies of that, one in the recorder and one in the reader, is two copies that will differ.
 *
 * The operand before this one -- `opd[k - 1]` -- supplies an index operand's default range in its
 * leading extent; for the first operand there is none. `vocab_rows` is the fallback range in that
 * case: a token id has to come from somewhere and the model's vocabulary is where.
 */
/* ================================================================== the weight a kernel reads
 *
 * A LIBRARY MAY WANT ITS WEIGHTS ARRANGED, AND A FIXTURE CANNOT CARRY EVERY ARRANGEMENT.
 *
 * libr4d's fp8 GEMMs read their weight in WMMA fragment order and its int4 experts with their
 * nibbles reordered; libref reads rows. A container holds neither -- it holds the encoding's
 * canonical planes, and each kernel relayouts them at load. The fixture does the same: it carries
 * the encoding, draws the planes, and each library runs its own layout hooks over them to get the
 * bytes it actually reads, while the reference reads what the planes decode to. Same planes in,
 * each kernel's own bytes out, which is what makes one recording serve libraries that disagree.
 */
/* The operand's encoding and selection, from the declaration it was recorded against. */
void kf_operand_enc(const WeightInfo& w, KfOperand* d);

/* A weight operand's selected planes, drawn from `seed`: each dense in its own elements, a row
 * starting on a byte, and every value one its role can hold -- a code inside its codebook, a
 * permutation that is one, an exponent scale that neither overflows nor vanishes. */
void kf_draw_planes(const KfOperand& d, uint64_t seed, std::vector<std::vector<uint8_t>>* out);

/* What a reference reads for those planes: the operand's dtype and shape (enc_selection_logical).
 * RAD_OK, or the refusal with `why`. */
int kf_planes_logical(const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                      std::vector<uint8_t>* out, std::string* why);

/* The bytes kernel `info` reads for operand `operand`, laid out from the planes. RAD_OK with
 * `out` holding them and `lay` describing them; RAD_E_UNSUPPORTED when the kernel reads the
 * selected plane as it is -- the bytes kf_planes_logical gives, widened if the operand is wider;
 * RAD_E_DTYPE when the kernel does not read this encoding at all. Anything else is a decline the
 * caller must name rather than absorb. */
int kf_lay_weight(const RadKernelInfo* info, const RadParam* p, int n_p, int operand,
                  const KfOperand& d, const std::vector<std::vector<uint8_t>>& planes,
                  RadLayout* lay, std::vector<uint8_t>* out);

/* How the engine describes a laid-out weight operand: the STORED dtype with the DECLARED shape.
 * Not the RadLayout's shape -- see the comment on the definition. */
void kf_describe_laid(const KfOperand& d, const RadLayout& lay, void* data, RadTensor* out);

/* The seed a WEIGHT operand is drawn from: its own identity, not the case's. Public because it
 * is also the key a caller caches the drawn bytes under -- see kf_draw. */
uint64_t kf_weight_seed(const KfOperand& d, size_t k);

/* THE FALLBACK RANGE FOR AN INDEX OPERAND WITH NOTHING TO INDEX, and it exists so that the
 * recorder and the reader cannot disagree about it. The recorder knows the model's vocabulary and
 * the reader does not, so a rule that reached for `meta.n_vocab` on one side and something else on
 * the other would draw different token ids from the same seed -- a divergence that surfaces as a
 * wrong answer in a kernel that is right. Derived from the case itself, identically, on both. */
int64_t kf_vocab_rows(const std::vector<KfOperand>& opd);

void kf_draw(const std::vector<KfOperand>& opd, size_t k, int64_t M, int64_t vocab_rows,
             uint64_t seed, std::vector<uint8_t>* out);

/* The three numbers a reference is reduced to, computed over a widened output. Called by the
 * recorder on the ORACLE's output and by the reader on the KERNEL's, so the two are reduced the
 * same way. */
void kf_reduce(const float* v, int64_t n, uint64_t seed, double* sumsq, int64_t* nonfinite,
               std::vector<float>* sample);

}  /* namespace rad */
