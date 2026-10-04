/* opshapes.cpp -- the non-inline half of opshapes.h: dtype widening and the seeded fill.
 *
 * Shared by rad-kbench and rad-tune. Two copies of a Box-Muller and a bf16 rounding rule is two
 * places for them to differ, and a tuner measuring different bytes from the checker is a tuner
 * whose numbers describe a program nobody ran.
 */
#include "opshapes.h"

#include <thread>

#include <cmath>
#include <cstring>

namespace rad {

void rad_widen(const void* src, uint32_t dtype, int64_t n, std::vector<float>* out) {
    out->assign((size_t)n, 0.0f);
    switch (dtype) {
        case RAD_F32:
            std::memcpy(out->data(), src, (size_t)n * 4);
            break;
        case RAD_BF16:
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t b = (uint32_t)((const uint16_t*)src)[i] << 16;
                float f;
                std::memcpy(&f, &b, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_F16:
            for (int64_t i = 0; i < n; ++i) {
                const uint16_t h = ((const uint16_t*)src)[i];
                const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) {
                    /* Subnormal. Renormalise rather than flush: a flushed small weight is a wrong
                     * one, and it is wrong in a way that never trips a norm-based tolerance. */
                    if (m == 0) bits = sign;
                    else {
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x400) == 0);
                        bits = sign | ((uint32_t)(112 - sh) << 23) | ((mm & 0x3FF) << 13);
                    }
                } else if (e == 0x1F) {
                    bits = sign | 0x7F800000u | (m << 13);
                } else {
                    bits = sign | ((e + 112) << 23) | (m << 13);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        /* OCP fp8. These two are what make the block-scaled fp8 family checkable at all, and
         * without them it fails VACUOUSLY rather than visibly: rad_narrow would leave the E4M3
         * activation at its zero fill, so kernel and oracle both compute zero, rad_widen reads the
         * E4M3 output back as zero, and every case reports rel_l2 = 0 and "ok". A vacuous pass is
         * worse than a skip, which is what the `default` below is careful to produce for a dtype
         * whose bytes only its own kernel understands. e4m3 has no infinity and S1111111 is its
         * only NaN; e5m2 is IEEE-shaped. */
        case RAD_F8E4M3:
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t b = ((const uint8_t*)src)[i];
                const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
                const uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
                uint32_t bits;
                if (e == 0) {
                    if (m == 0) bits = sign;
                    else {                                  /* subnormal: m * 2^-9 */
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x8u) == 0);
                        bits = sign | ((uint32_t)(127 - 6 - sh) << 23) | ((mm & 0x7u) << 20);
                    }
                } else if (e == 0xFu && m == 0x7u) {
                    bits = sign | 0x7FC00000u;              /* the one NaN encoding */
                } else {
                    bits = sign | ((e + 127 - 7) << 23) | (m << 20);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_F8E5M2:
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t b = ((const uint8_t*)src)[i];
                const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
                const uint32_t e = (b >> 2) & 0x1Fu, m = b & 0x3u;
                uint32_t bits;
                if (e == 0) {
                    if (m == 0) bits = sign;
                    else {
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x4u) == 0);
                        bits = sign | ((uint32_t)(127 - 14 - sh) << 23) | ((mm & 0x3u) << 21);
                    }
                } else if (e == 0x1Fu) {
                    bits = sign | 0x7F800000u | (m << 21);
                } else {
                    bits = sign | ((e + 127 - 15) << 23) | (m << 21);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_I8:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int8_t*)src)[i];
            break;
        case RAD_U8:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const uint8_t*)src)[i];
            break;
        case RAD_I32:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int32_t*)src)[i];
            break;
        /* THE INTEGER TYPES MUST BE READABLE HERE OR A COMPARISON AGAINST THEM IS VACUOUS.
         *
         * An output operand is widened on the recorder's side and again on the reader's, so a
         * dtype that falls through to `default` is reduced to a norm of zero and a sample of
         * zeros on BOTH sides -- which agrees perfectly and reports as a pass. A kernel writing
         * arbitrary bytes into such an operand is indistinguishable from a correct one.
         *
         * The set is the one libref's own element accessor reads, because libref is what the
         * recording is of: a dtype the oracle can produce and this cannot is an output nothing
         * checks. i64 past 2^24 does not survive the float this widens into, which is the same
         * bound rad_narrow documents and is fine for what these operands carry (sizes, offsets,
         * token ids below a quarter-million vocabulary). */
        case RAD_I16:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int16_t*)src)[i];
            break;
        case RAD_I64:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int64_t*)src)[i];
            break;
        case RAD_U32:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const uint32_t*)src)[i];
            break;
        case RAD_BOOL:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = ((const uint8_t*)src)[i] ? 1.0f : 0.0f;
            break;
        default:
            /* A plugin-private dtype: only its own kernel knows what the bytes mean. Zeroed, and
             * the caller sees a zero reference and reports it as such rather than as agreement. */
            break;
    }
}

void rad_narrow(const float* src, uint32_t dtype, int64_t n, void* dst) {
    switch (dtype) {
        case RAD_F32:
            std::memcpy(dst, src, (size_t)n * 4);
            break;
        case RAD_BF16:
            /* Round-to-nearest-even, the same rule rad-convert uses. Truncating here would bias
             * every generated input toward zero and quietly flatter every kernel. */
            for (int64_t i = 0; i < n; ++i) {
                uint32_t b;
                std::memcpy(&b, &src[i], 4);
                ((uint16_t*)dst)[i] = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
            }
            break;
        case RAD_F16:
            for (int64_t i = 0; i < n; ++i) {
                uint32_t x;
                std::memcpy(&x, &src[i], 4);
                const uint32_t sign = (x >> 16) & 0x8000u;
                int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
                const uint32_t m = x & 0x7FFFFF;
                uint16_t h;
                if (e >= 0x1F)      h = (uint16_t)(sign | 0x7C00u);
                else if (e <= 0)    h = (uint16_t)sign;
                else                h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
                ((uint16_t*)dst)[i] = h;
            }
            break;
        /* The encode side of the pair above. `rad_fill_normal` draws unit normals and E4M3 holds
         * 448, so no scaling is applied -- the values land in the middle of the format's range,
         * which is where a quantised GEMM actually operates. Round-to-nearest-even on the 3-bit
         * mantissa with a real subnormal path, matching libref/ref_common.h; truncating would
         * bias every generated input toward zero and flatter every kernel. */
        case RAD_F8E4M3:
            for (int64_t i = 0; i < n; ++i) {
                float a = src[i] < 0 ? -src[i] : src[i];
                uint32_t sb;
                std::memcpy(&sb, &src[i], 4);
                const uint8_t sign = (uint8_t)((sb >> 24) & 0x80u);
                if (!(a > 0.0f)) { ((uint8_t*)dst)[i] = sign; continue; }
                if (a > 448.0f) a = 448.0f;
                if (a >= 0x1p-6f) {
                    uint32_t u;
                    std::memcpy(&u, &a, 4);
                    int e = (int)((u >> 23) & 0xFFu) - 127;
                    uint32_t mm = (u & 0x7FFFFFu) >> 20;
                    const uint32_t rem = u & 0xFFFFFu, half = 1u << 19;
                    if (rem > half || (rem == half && (mm & 1u))) { if (++mm == 8u) { mm = 0; ++e; } }
                    if (e > 8) { e = 8; mm = 7; }
                    ((uint8_t*)dst)[i] = (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
                } else {
                    const float scaled = a * 512.0f;
                    uint32_t m = (uint32_t)(scaled + 0.5f);
                    if ((float)m - scaled == 0.5f && (m & 1u)) --m;
                    ((uint8_t*)dst)[i] = m > 7u ? (uint8_t)(sign | (1u << 3)) : (uint8_t)(sign | m);
                }
            }
            break;
        case RAD_F8E5M2:
            for (int64_t i = 0; i < n; ++i) {
                uint32_t x;
                std::memcpy(&x, &src[i], 4);
                const uint32_t sign = (x >> 24) & 0x80u;
                int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
                const uint32_t m = x & 0x7FFFFF;
                uint8_t h;
                if (e >= 0x1F)      h = (uint8_t)(sign | 0x7Cu);
                else if (e <= 0)    h = (uint8_t)sign;
                else                h = (uint8_t)(sign | ((uint32_t)e << 2) | (m >> 21));
                ((uint8_t*)dst)[i] = h;
            }
            break;
        case RAD_I8:
            for (int64_t i = 0; i < n; ++i) {
                float v = src[i] * 32.0f;
                v = v < -127.f ? -127.f : (v > 127.f ? 127.f : v);
                ((int8_t*)dst)[i] = (int8_t)std::lrintf(v);
            }
            break;
        case RAD_U8:
            for (int64_t i = 0; i < n; ++i) {
                float v = src[i];
                v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
                ((uint8_t*)dst)[i] = (uint8_t)std::lrintf(v);
            }
            break;
        case RAD_I32:
            for (int64_t i = 0; i < n; ++i) ((int32_t*)dst)[i] = (int32_t)std::lrintf(src[i]);
            break;
        /* i64 MUST BE HANDLED HERE, because the `default` below zeroes: an unhandled i64 operand
         * is drawn as a buffer of zeros, silently, since zero is a legal value for most of them.
         * The ops that do check read them as sizes -- ngram_ids refuses `vocab_size <= 0` by name,
         * so with zeroed sizes it would be refused at every geometry and never checked on any
         * plugin, and that refusal reads exactly like a shape the reference does not serve.
         *
         * `llrintf` and not `lrintf`: long is 32-bit on some targets and this type is not.
         * A float carries 24 bits, so a drawn i64 past 2^24 does not round-trip -- which is fine
         * for what i64 operands in this vocabulary actually are (sizes, offsets, multipliers) and
         * is why the shapes that need a large exact value pass it as idx_const rather than a draw. */
        case RAD_I64:
            for (int64_t i = 0; i < n; ++i) ((int64_t*)dst)[i] = (int64_t)std::llrintf(src[i]);
            break;
        /* The rest of the integer vocabulary, for the reason the i64 note above gives and with
         * the same consequence: zeroed by `default`, an operand of one of these types is handed
         * to the kernel as a buffer of zeros, which is a legal value for most of them and so
         * never reports as anything. Kept in step with rad_widen -- a type this can write and
         * that cannot read back is an output whose comparison is vacuous. */
        case RAD_I16:
            for (int64_t i = 0; i < n; ++i) ((int16_t*)dst)[i] = (int16_t)std::lrintf(src[i]);
            break;
        case RAD_U32:
            for (int64_t i = 0; i < n; ++i) {
                const float v = src[i] < 0.f ? 0.f : src[i];
                ((uint32_t*)dst)[i] = (uint32_t)std::llrintf(v);
            }
            break;
        case RAD_BOOL:
            for (int64_t i = 0; i < n; ++i) ((uint8_t*)dst)[i] = src[i] != 0.0f ? 1u : 0u;
            break;
        default:
            std::memset(dst, 0, (size_t)rad_dtype_bytes(dtype, n));
            break;
    }
}

uint64_t rad_case_seed(uint64_t base, std::string_view op, int band, int domain, int operand) {
    uint64_t h = base ^ 0xCBF29CE484222325ull;
    for (char c : op) { h ^= (unsigned char)c; h *= 0x100000001B3ull; }
    h ^= (uint64_t)band * 0x9E37u;      h *= 0x100000001B3ull;
    h ^= (uint64_t)domain * 0x85EBu;    h *= 0x100000001B3ull;
    h ^= (uint64_t)operand * 0xC2B2u;   h *= 0x100000001B3ull;
    return h ? h : 1;
}

/* Standard normal by Box-Muller. Activations in a transformer are roughly Gaussian, and the tails
 * are the point: an outlier is what breaks a per-row scale, and a uniform fill has none. */
/* BLOCK-SEEDED AND THREADED, and the block seeding is what makes the threading legal.
 *
 * This is the hot loop of the whole tool: a replay draws every operand of every case, and an fp8
 * GEMM weight at N=34816 K=5120 is 178 million elements of Box-Muller -- four transcendentals per
 * pair, chained through one splitmix state. Single threaded that is most of a replay's wall clock,
 * and a benchmark that spends its time generating its own inputs is a benchmark nobody runs.
 *
 * The chain is the problem: element i depends on element i-2, so nothing can be split. So the
 * stream is cut into fixed blocks and each block derives its OWN seed from (seed, block index).
 * The values then do not depend on how many threads ran, or on whether any did -- which is not a
 * nicety here but the contract: a fixture recorded on one machine is replayed on another, and a
 * draw that varied with core count would make every recorded reference wrong somewhere else.
 */
const int64_t RAD_FILL_BLOCK = 1 << 16;

static void fill_block(float* p, int64_t n, uint64_t seed, float sigma) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; i += 2) {
        const double u1 = ((double)(rad_splitmix(s) >> 11) + 1.0) * (1.0 / 9007199254740992.0);
        const double u2 = ((double)(rad_splitmix(s) >> 11)) * (1.0 / 9007199254740992.0);
        const double r = std::sqrt(-2.0 * std::log(u1)), t = 6.283185307179586 * u2;
        p[i] = (float)((double)sigma * r * std::cos(t));
        if (i + 1 < n) p[i + 1] = (float)((double)sigma * r * std::sin(t));
    }
}

static uint64_t block_seed(uint64_t seed, int64_t b) {
    uint64_t s = seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(b + 1));
    return rad_splitmix(s);
}

/* ONE BLOCK OF THAT STREAM, for a caller drawing a large operand a block at a time so the float
 * intermediate never gets as big as the tensor. Same seed, same block index, same bytes. */
void rad_fill_normal_block(float* p, int64_t n, uint64_t seed, int64_t block, float sigma) {
    fill_block(p, n, block_seed(seed, block), sigma);
}


void rad_fill_normal(float* p, int64_t n, uint64_t seed, float sigma) {
    if (n <= 0) return;
    const int64_t nb = (n + RAD_FILL_BLOCK - 1) / RAD_FILL_BLOCK;
    if (nb == 1) { fill_block(p, n, block_seed(seed, 0), sigma); return; }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    const int64_t nthreads = std::min<int64_t>(nb, (int64_t)hw);
    std::vector<std::thread> th;
    th.reserve((size_t)nthreads);
    for (int64_t t = 0; t < nthreads; ++t) {
        th.emplace_back([&, t]() {
            for (int64_t b = t; b < nb; b += nthreads) {
                const int64_t lo = b * RAD_FILL_BLOCK;
                fill_block(p + lo, std::min<int64_t>(RAD_FILL_BLOCK, n - lo),
                           block_seed(seed, b), sigma);
            }
        });
    }
    for (auto& x : th) x.join();
}


}  /* namespace rad */
