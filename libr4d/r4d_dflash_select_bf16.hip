// DFlash2's candidate selector: the edge scores and the path walk, one workgroup a sequence.
//
// A block-diffusion drafter proposes every position of a block in ONE backbone pass, so it has no
// way to condition position l+1 on what it chose at position l -- the two were computed together.
// The selector is what puts that dependence back. Each position keeps the draft head's top K
// candidates and a low-rank bilinear form scores every (predecessor, candidate) EDGE between
// consecutive positions; a greedy walk over those edges picks one token per position.
//
//   score[s, l, c] = unary[s, l, c]
//                  + sum_r  pred[pid(s,l)][r] * hp[s,l][r] * succ[cand[s,l,c]][r]
//
//   pid(s, 0) = anchor[s * anchor_stride]           the token the target just committed
//   pid(s, l) = cand[s, l-1, chosen(s, l-1)]       the walk's own previous pick
//
// `hp` IS OPTIONAL, AND ABSENT IT IS ONES -- which drops the trilinear form to a plain BIGRAM one,
// score = unary + <pred[pid], succ[cand]>, with no dependence on the hidden state at all. That is
// not a degenerate case: it is exactly DSpark's Markov head, a rank-256 learned
// bigram bias `markov_w2 @ markov_w1[prev]` added to the draft logits, whose previous token is the
// anchor at position 0 and the drafter's own last pick after it. The head is a full-vocabulary
// [rank, n_vocab] GEMM per position in the reference and it is SERIAL in exactly the way this walk
// already is, so running it here costs the candidate rows of two codebooks instead of seven passes
// over 66 MiB -- and the greedy argmax over the base head's top K is what makes that exact enough
// to matter only to acceptance (a draft is verified; see the 2-bit head's argument).
//
// THE WALK IS SERIAL IN l and that is the whole reason this is a kernel and not a batched GEMM
// followed by an argmax. The reference materialises the full [steps, K, K] score tensor -- vLLM's
// `_score_edges` einsum -- of which exactly one row per step is ever read, because the predecessor
// index is not known until the previous step has been decided. Scoring K*K edges to use K of them
// is 16x the work and a round trip per step; here the predecessor's codebook row is fetched after
// the previous step's argmax, inside one workgroup, with no launch between.
//
// The arithmetic is tiny -- steps * K * R multiply-adds a sequence, 28K at the shipped 7/16/256 --
// so this is a latency kernel, not a throughput one. It is written for one workgroup a sequence
// with no global synchronisation at all.
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"   // r4d_bf2f, r4d_u16
#include "r4d_common.h"   // r4d_wave_sum

#define DSL_THREADS 256
#define DSL_WAVE    32
#define DSL_WAVES   (DSL_THREADS / DSL_WAVE)
#define DSL_MAX_K   32              // the entry point rejects more; ssc is sized for it
#define DSL_MAX_R   512             // and sa for this

// cand   [rows, cand_ld] i32   global token ids, row = s * steps + l
// unary  [rows, cand_ld] f32   the draft head's logit for each candidate (row_topk's value plane)
// hp     [rows, hp_ld]   bf16  hidden_projection(x) for the same row
// anchor [M * anchor_stride] i32  the last committed token of each sequence, at stride s
// pred   [vocab, R]      bf16  predecessor codebook
// succ   [vocab, R]      bf16  successor codebook
// out    [M, out_ld]     i32   the chosen token per step
//
// AN ID OUTSIDE [0, V) IS NOT A CODEBOOK ROW, and one reaches this kernel: row_topk_merge reports a
// slot no rank could fill as id -1, and step 0's predecessor is whatever the anchor plane holds. So
// the codebook reads are guarded exactly as ref_dflash_select guards them -- a predecessor out of
// range contributes nothing, and a candidate out of range scores its unary term alone -- rather
// than gathering a row from before or past the codebook.
__global__ __launch_bounds__(DSL_THREADS) void r4d_dflash_select_kernel(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ out, int steps, int K, int R, int V,
    int cand_ld, int hp_ld, int out_ld, int anchor_stride)
{
  const int s    = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & (DSL_WAVE - 1);
  const int wave = tid / DSL_WAVE;

  // a[r] = pred[pid][r] * hp[r] -- the half of the bilinear form that does not depend on the
  // candidate, so it is computed once per step and read K times out of LDS.
  __shared__ float sa[DSL_MAX_R];
  __shared__ float ssc[DSL_MAX_K];
  __shared__ int   schosen;         // the previous step's INDEX into its candidate list

  for (int l = 0; l < steps; ++l) {
    const long row = (long)s * steps + l;
    // Step 0's predecessor is the committed anchor; every later step's is the token this walk
    // picked last time round, which is why the read has to happen after the barrier below.
    const int pid = (l == 0) ? anchor[(long)s * anchor_stride]
                             : cand[(row - 1) * cand_ld + schosen];
    const bool pid_ok = pid >= 0 && pid < V;

    const r4d_u16* __restrict__ prow = pred + (long)(pid_ok ? pid : 0) * R;
    /* hp absent is hp == 1: the predecessor's codebook row goes in untouched. Branching outside
     * the loop rather than multiplying by a materialised plane of ones keeps the bigram case from
     * reading R bf16 a step for a number it already knows. */
    if (!pid_ok) {
      for (int r = tid; r < R; r += DSL_THREADS) sa[r] = 0.0f;
    } else if (hp) {
      const r4d_u16* __restrict__ hrow = hp + row * hp_ld;
      for (int r = tid; r < R; r += DSL_THREADS)
        sa[r] = r4d_bf2f(prow[r]) * r4d_bf2f(hrow[r]);
    } else {
      for (int r = tid; r < R; r += DSL_THREADS)
        sa[r] = r4d_bf2f(prow[r]);
    }
    __syncthreads();

    // One wave a candidate. R is a multiple of the wave, so the strided loop is balanced and the
    // reduction is five xor-butterfly steps with no barrier inside it.
    for (int c = wave; c < K; c += DSL_WAVES) {
      const int cid = cand[row * cand_ld + c];
      float acc = 0.0f;
      if (cid >= 0 && cid < V) {
        const r4d_u16* __restrict__ srow = succ + (long)cid * R;
        for (int r = lane; r < R; r += DSL_WAVE) acc += sa[r] * r4d_bf2f(srow[r]);
      }
      acc = r4d_wave_sum(acc);
      if (lane == 0) ssc[c] = acc + unary[row * cand_ld + c];
    }
    __syncthreads();

    // K is at most 32 and the walk is serial anyway, so the argmax is one thread. Ties go to the
    // lower index, which is `row_topk`'s own tie rule and therefore the more likely candidate.
    // Seeded below every score, as ref seeds it, so a NaN in the first slot cannot win by position.
    if (tid == 0) {
      int   best = 0;
      float bv   = -INFINITY;
      for (int c = 0; c < K; ++c)
        if (ssc[c] > bv) { bv = ssc[c]; best = c; }
      schosen = best;
      out[(long)s * out_ld + l] = cand[row * cand_ld + best];
    }
    // Both directions: the next step reads `schosen` and overwrites `sa`, and every other thread
    // is still reading this step's.
    __syncthreads();
  }
}

extern "C" int r4d_dflash_select_bf16(
    const void* cand, const void* unary, const void* hp, const void* anchor,
    const void* pred, const void* succ, void* out,
    int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
    int anchor_stride, void* stream)
{
  if (M <= 0 || steps <= 0) return 0;
  if (K < 1 || K > DSL_MAX_K) return -1;
  if (R < DSL_WAVE || R > DSL_MAX_R || (R % DSL_WAVE)) return -1;
  if (V < 1) return -1;
  if (cand_ld < K || out_ld < steps || anchor_stride < 1) return -1;
  /* `hp` may be null -- the bigram form -- and then there is no plane to have a pitch. */
  if (hp && hp_ld < R) return -1;

  hipLaunchKernelGGL(r4d_dflash_select_kernel, dim3(M), dim3(DSL_THREADS), 0, (hipStream_t)stream,
                     (const int*)cand, (const float*)unary, (const r4d_u16*)hp,
                     (const int*)anchor, (const r4d_u16*)pred, (const r4d_u16*)succ,
                     (int*)out, steps, K, R, V, cand_ld, hp_ld, out_ld, anchor_stride);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
//
// The shared operand helpers are r4d_gdn_args.h. See libr4d/r4d_args.h for why there is no
// adapter directory between the core and these launches.
/* ------------------------------------------------------------------ dflash select */

enum { DS_CAND = 0, DS_UNARY, DS_HP, DS_ANCHOR, DS_PRED, DS_SUCC, DS_OUT };

/* A rank-2 [rows, width] operand whose row pitch libr4d takes explicitly. `width` is checked
 * rather than derived: this op reads three planes at the SAME row index and a plane whose width is
 * not what the parameters say would still have a plausible number of rows. */
static long long ds_plane(const RadTensor* t, uint32_t dtype, long long width, long long rows) {
    if (!t || !t->data || t->rank != 2) return -1;
    if (t->dtype != dtype) return -1;
    if (t->shape[1] < width || t->shape[0] < rows) return -1;
    const long long pitch = r4d_row_pitch(t);
    return (pitch >= width) ? pitch : -1;
}

extern "C" int r4d_rad_dflash_select(const RadArgs* a, RadStream s) {
    const RadTensor* cand = r4d_opt(a, DS_CAND);
    const RadTensor* un   = r4d_opt(a, DS_UNARY);
    const RadTensor* hp   = r4d_opt(a, DS_HP);
    const RadTensor* anc  = r4d_opt(a, DS_ANCHOR);
    const RadTensor* pr   = r4d_opt(a, DS_PRED);
    const RadTensor* sc   = r4d_opt(a, DS_SUCC);
    const RadTensor* out  = r4d_opt(a, DS_OUT);
    /* `hp` is the one that may be absent; every other operand is load-bearing. */
    if (!cand || !un || !anc || !pr || !sc || !out) return R4D_NO(RAD_E_INVAL);

    const long long steps = r4d_geti_or(a, "steps", 0);
    const long long K     = r4d_geti_or(a, "top_k", 0);
    const long long R     = r4d_geti_or(a, "rank", 0);
    const long long V     = r4d_geti_or(a, "n_vocab", 0);
    /* M IS THE OPERAND'S ROW COUNT AND NOT THE PARAMETER'S. RadArgs carries the geometry with its
     * RANGES COLLAPSED -- `M` here is the band's bound, max_seqs, and not the sequence count this
     * step issued (the same trap fill_paged spells out for q_len). Reading the parameter would
     * make the selector demand max_seqs * steps rows of candidates on a one-sequence step and
     * refuse every launch. The output plane is one row a sequence, so it is the count. */
    if (out->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = out->shape[0];
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || V <= 0) return R4D_NO(RAD_E_INVAL);

    /* The three input planes are read at row s*steps+l, so between them they must hold M*steps
     * rows -- a short one is a walk that reads another sequence's candidates and proposes a
     * perfectly plausible draft belonging to the wrong request. */
    const long long rows    = M * steps;
    const long long cand_ld = ds_plane(cand, RAD_I32,  K, rows);
    const long long un_ld   = ds_plane(un,   RAD_F32,  K, rows);
    const long long out_ld  = ds_plane(out,  RAD_I32,  steps, M);
    /* ABSENT IS A NULL `data`, NOT A MISSING OPERAND: operands are positional, so a graph that
     * does not use the hidden projection still passes it, empty. Both spellings arrive here and
     * both mean the bigram form. */
    const bool     has_hp = hp && hp->data;
    const long long hp_ld = has_hp ? ds_plane(hp, RAD_BF16, R, rows) : 0;
    if (cand_ld < 0 || un_ld < 0 || hp_ld < 0 || out_ld < 0) return R4D_NO(RAD_E_SHAPE);
    /* One pitch for both, because the kernel indexes them with the same expression. */
    if (cand_ld != un_ld) return R4D_NO(RAD_E_STRIDE);

    /* The codebooks are gathered by GLOBAL token id, so a row-sharded one would let a rank score
     * only the candidates that happen to land in its own vocabulary slice -- and the candidates
     * arrive already merged across ranks. Both are declared replicated at the whole vocabulary;
     * this is where that would otherwise fail silently. */
    if (pr->dtype != RAD_BF16 || sc->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(pr) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(pr) < V * R || r4d_numel(sc) < V * R) return R4D_NO(RAD_E_SHAPE);

    /* `anchor` is usually the step's own token_ids, which carries `block` ids a sequence with the
     * anchor first -- so the stride is a parameter and a packed [M] array is the stride-1 case. */
    const long long astr = r4d_geti_or(a, "anchor_stride", 1);
    if (anc->dtype != RAD_I32 && anc->dtype != RAD_U32) return R4D_NO(RAD_E_DTYPE);
    if (astr < 1 || !r4d_contig(anc)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(anc) < (M - 1) * astr + 1) return R4D_NO(RAD_E_SHAPE);


    return r4d_dflash_select_bf16(cand->data, un->data, has_hp ? hp->data : nullptr,
                                  anc->data, pr->data, sc->data,
                                  out->data, (int)M, (int)steps, (int)K, (int)R, (int)V,
                                  (int)cand_ld, (int)hp_ld, (int)out_ld, (int)astr, s) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
