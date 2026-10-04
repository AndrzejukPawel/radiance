/* lq_common.h -- the number conversions every libquant quantiser shares, written out once.
 *
 * WHY THESE ARE NOT rad_plugin.h's. They are the rules the containers in service were quantised by,
 * and a converter that wants to reproduce those containers byte for byte has to round exactly as
 * they were rounded. Where a rule here and rad_plugin.h's agree on every finite input the
 * difference is only in what a NaN becomes; where they could disagree -- the integer round, which
 * is half AWAY from zero, and the fp4 encoder, which breaks a tie toward the even CODE -- the
 * comment says so. Each is a pure function of its input.
 */
#pragma once
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace lq {

/* bf16, round to nearest even; a NaN stays a NaN. */
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu))
        return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* f16, round to nearest even, subnormals kept, overflow to infinity. */
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        const uint32_t v = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t out = v;
        if (rem > half || (rem == half && (v & 1u))) ++out;
        return (uint16_t)(sign | out);
    }
    uint32_t v = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    return (uint16_t)(sign | v);
}
inline float f16_to_f32(uint16_t h) { return rad_f16_to_f32(h); }

/* E4M3-fn: clamp at 448, round to nearest even on the 3-bit mantissa, a real subnormal rung below
 * 2^-6, and a NaN encoded as a signed zero -- a weight is never NaN, and the one place a NaN could
 * come from is a scale of zero, which the scale rules below never produce. */
inline uint8_t f32_to_e4m3(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 24) & 0x80u;
    float a = std::fabs(f);
    if (!(a > 0.0f)) return (uint8_t)sign;
    if (a > 448.0f) a = 448.0f;
    if (a >= 0x1p-6f) {
        uint32_t u;
        std::memcpy(&u, &a, 4);
        int e = (int)((u >> 23) & 0xFFu) - 127;
        uint32_t mm = (u & 0x7FFFFFu) >> 20;
        const uint32_t rem = u & 0xFFFFFu, half = 1u << 19;
        if (rem > half || (rem == half && (mm & 1u))) { if (++mm == 8u) { mm = 0; ++e; } }
        if (e > 8) { e = 8; mm = 7; }
        return (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
    }
    const float scaled = a * 512.0f;
    uint32_t m = (uint32_t)(scaled + 0.5f);
    if ((float)m - scaled == 0.5f && (m & 1u)) --m;
    if (m > 7u) return (uint8_t)(sign | (1u << 3));
    return (uint8_t)(sign | m);
}

/* FP4 e2m1 by nearest magnitude, a tie going to the EVEN CODE -- which on this grid is the
 * smaller magnitude except at 2.5, where it is 2.0 (code 4) over 3.0 (code 5), the hardware's
 * own round-to-nearest-even. */
inline uint8_t f32_to_e2m1(float v) {
    static const float k[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    const float a = v < 0.0f ? -v : v;
    int best = 0;
    float bd = -1.0f;
    for (int i = 0; i < 8; ++i) {
        const float d = a > k[i] ? a - k[i] : k[i] - a;
        if (bd < 0.0f || d < bd || (d == bd && (i & 1) == 0)) { bd = d; best = i; }
    }
    return (uint8_t)(sign | (uint8_t)best);
}

/* Any small float grid by exhaustive nearest search over its codes, ties to the even code. For
 * fp6 and e5m2, which nothing here needs to be fast. */
inline uint32_t f32_to_small_float(float v, uint32_t dt) {
    const int bits = rad_dtype_bits(dt);
    const uint32_t n = 1u << bits;
    uint32_t best = 0;
    float bd = -1.0f;
    /* Eight bytes, not one: rad_load_f32 reads any dtype up to 64 bits, and the compiler cannot
     * see that `dt` is a small one. */
    uint8_t buf[8] = {};
    for (uint32_t c = 0; c < n; ++c) {
        buf[0] = (uint8_t)c;
        const float x = rad_load_f32(buf, dt, 0);
        if (std::isnan(x) || std::isinf(x)) continue;
        const float d = std::fabs(x - v);
        if (bd < 0.0f || d < bd || (d == bd && (c & 1u) == 0)) { bd = d; best = c; }
    }
    return best;
}

/* The integer round every integer grid here uses: HALF AWAY FROM ZERO, by truncation of v +- 0.5.
 * It is what the containers in service were packed with, and lround differs from it only at
 * values no representable f32 product reaches -- but "only" is a claim, and the byte-for-byte
 * reproduction of those containers is a fact. */
inline int round_away(float v) { return (int)(v < 0.0f ? v - 0.5f : v + 0.5f); }

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* The unnormalised Sylvester Walsh-Hadamard transform in place, the butterfly keeping a+b low and
 * a-b high: the transform an "fwhtN" encoding names and libr4d's r4d_fwht_wave computes. */
inline void fwht_inplace(float* v, int64_t n) {
    for (int64_t h = 1; h < n; h <<= 1)
        for (int64_t i = 0; i < n; i += h << 1)
            for (int64_t j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

}  /* namespace lq */
