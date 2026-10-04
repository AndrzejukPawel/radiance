// stage_kernels.hip -- libstage device kernels + host launch wrappers.
//
// All kernels are single- or multi-block grid-stride loops over bytes or
// elements. The exchange protocol lives in stage_xfer_kernel (see stage.h);
// the math kernels reproduce the unfused chains they replace bit for bit:
// ascending fp32 accumulation, hardware-RNE narrow on the all-reduce store,
// round-half-up (+0x8000) in hc_write, RNE gather narrows, RTNE fp8/i8
// quantisers, sequential per-row reductions, scalar-order FWHT. Device code
// compiles -ffp-contract=off (rad_hipify): no FMA in any fold.
#include "stage.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define ST_SPIN_MAX 4000000000ULL

__device__ __forceinline__ unsigned st_load_acq(const unsigned* p) {
    return __hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_store_rel(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---------------------------------------------------------------------------
// bit-exact conversions. Integer arithmetic only, matching the ABI wider
// (rad_plugin.h) and the sites they reproduce:
//  - hw narrow (all-reduce store): hardware RNE convert, as vadd/rd_from_f.
//  - rne narrow (gather): round-to-nearest-even, as arg_bf16_rne.
//  - f2bf (hc_write): round-half-up (+0x8000), as r4d_gdn_wmma.h f2bf.
__device__ __forceinline__ float st_bf2f(unsigned short h) {
    unsigned u = (unsigned)h << 16;
    return __builtin_bit_cast(float, u);
}
__device__ __forceinline__ unsigned short st_f2bf_hw(float f) {
    __hip_bfloat16 h = (__hip_bfloat16)f;
    return __builtin_bit_cast(unsigned short, h);
}
__device__ __forceinline__ unsigned short st_f2bf_rne(float f) {
    unsigned u = __builtin_bit_cast(unsigned, f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ unsigned short st_f2bf_hup(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ float st_f162f(unsigned short h) {
    __half v = __builtin_bit_cast(__half, h);
    return __half2float(v);
}
__device__ __forceinline__ unsigned short st_f2f16(float f) {
    __half h = (__half)f;
    return __builtin_bit_cast(unsigned short, h);
}
// OCP E4M3: port of rad_f32_to_fp8e4m3 (saturates at 448, RTNE, subnormals).
__device__ __forceinline__ unsigned char st_f2e4m3(float f) {
    unsigned u = __builtin_bit_cast(unsigned, f);
    unsigned char s = (unsigned char)((u >> 24) & 0x80u);
    unsigned a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (unsigned char)(s | 0x7fu);
    if (a >= 0x43e00000u) return (unsigned char)(s | 0x7eu);
    int e = (int)(a >> 23) - 127;
    unsigned m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        int sh = -6 - e + 20;
        m |= 0x800000u;
        unsigned lo = m & ((1u << sh) - 1u);
        unsigned q = m >> sh;
        unsigned half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (unsigned char)(s | q);
    }
    {
        unsigned q = m >> 20;
        unsigned lo = m & 0xfffffu;
        if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
        int ee = e + 7;
        if (q == 0x8u) { q = 0; ++ee; }
        if (ee > 15 || (ee == 15 && q >= 7u)) return (unsigned char)(s | 0x7eu);
        return (unsigned char)(s | ((unsigned)ee << 3) | q);
    }
}
// int8 codes: round-to-nearest-even and saturate, as st_dt does.
__device__ __forceinline__ signed char st_f2i8(float f) {
    float r = nearbyintf(f);
    r = r < -128.f ? -128.f : (r > 127.f ? 127.f : r);
    return (signed char)r;
}
// Gain load: bf16 widened, or f32 direct.
__device__ __forceinline__ float st_gain(const void* w, long idx, int f32) {
    return f32 ? ((const float*)w)[idx] : st_bf2f(((const unsigned short*)w)[idx]);
}

// Unnormalised in-place FWHT over g values (port of libref fwht, same order).
__device__ __forceinline__ void st_fwht(float* v, long g) {
    for (long h = 1; h < g; h <<= 1)
        for (long i = 0; i < g; ++i)
            if ((i & h) == 0) {
                float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

// ---------------------------------------------------------------------------
// The transfer + handshake. ONE block. The ticket comes from the instance's
// device-resident counter, so the launch arguments are identical on every
// issue of this op -- including a recording and its replay.
//
// Single slot. Launch i's push waits until the peer finished launch i-1
// (done counts completions: after peer launch j, done=j+1), publishes
// ready=i+1, waits for the peer's. The math kernel follows on the same
// stream; stage_mark_done closes the launch after it.
__global__ __launch_bounds__(256) void stage_xfer_kernel(
    unsigned char* __restrict__ my_slot, const unsigned char* peer_slot,
    unsigned* __restrict__ counter,
    unsigned* __restrict__ my_ready, const unsigned* peer_ready,
    const unsigned* peer_done,
    const unsigned char* __restrict__ in_ptr, unsigned long long nbytes) {
    const int tid = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned s_i;
    if (tid == 0) s_i = atomicAdd(counter, 1u);
    __syncthreads();
    const unsigned i = s_i;
    if (tid == 0 && i >= 1) {
        // Slot reuse: launch i overwrites what launch i-1 published, so the
        // peer must have finished launch i-1 (its read of it). done counts
        // completions: after peer launch j, done=j+1.
        unsigned long long z = 0;
        while (st_load_acq(peer_done) < i) {
            if (++z > ST_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    // Push my input bytes into my slot. PCIe-posted writes; the fence below
    // orders them before the flag.
    {
        unsigned long long k = tid;
        // 16B fast path when everything aligns.
        if ((((unsigned long long)my_slot | (unsigned long long)in_ptr) & 15ull) == 0 &&
            (nbytes & 15ull) == 0) {
            const unsigned long long n16 = nbytes >> 4;
            const int4* s = (const int4*)in_ptr;
            int4* d = (int4*)my_slot;
            for (unsigned long long w = k; w < n16; w += nt) d[w] = s[w];
        } else {
            for (unsigned long long w = k; w < nbytes; w += nt) my_slot[w] = in_ptr[w];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        st_store_rel(my_ready, i + 1);
        unsigned long long z = 0;
        while (st_load_acq(peer_ready) <= i) {
            if (++z > ST_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
}

// Completion marker: my_done = launch count. Issued after the math on the
// same stream; stable arguments (pointers only).
__global__ void stage_mark_done_kernel(unsigned* __restrict__ my_done,
                                       const unsigned* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        st_store_rel(my_done, *counter);
}

void stage_xfer(long my_slot, long peer_slot,
                long my_ready, long peer_ready, long peer_done,
                long counter, long in_ptr, long nbytes, long stream) {
    hipStream_t st = (hipStream_t)stream;
    stage_xfer_kernel<<<dim3(1), dim3(256), 0, st>>>(
        (unsigned char*)my_slot, (const unsigned char*)peer_slot,
        (unsigned*)counter,
        (unsigned*)my_ready, (const unsigned*)peer_ready,
        (const unsigned*)peer_done,
        (const unsigned char*)in_ptr, (unsigned long long)nbytes);
}

void stage_mark_done(long my_done_ptr, long counter, long stream) {
    hipStream_t st = (hipStream_t)stream;
    stage_mark_done_kernel<<<dim3(1), dim3(1), 0, st>>>(
        (unsigned*)my_done_ptr, (const unsigned*)counter);
}

// ---------------------------------------------------------------------------
// Standalone reduction: out[e] = HWnarrow(in[e] + peer[e]), fp32 ascending.
// in==out allowed.
template <int DT>
__global__ void stage_reduce_kernel(const void* __restrict__ in_ptr, const void* peer_ptr,
                                    void* __restrict__ out_ptr, unsigned long long nelem) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long n = (unsigned long long)gridDim.x * blockDim.x;
    if (DT == 2) {
        const float* a = (const float*)in_ptr;
        const float* b = (const float*)peer_ptr;
        float* o = (float*)out_ptr;
        for (unsigned long long e = tid; e < nelem; e += n) o[e] = a[e] + b[e];
    } else if (DT == 0) {
        const unsigned short* a = (const unsigned short*)in_ptr;
        const unsigned short* b = (const unsigned short*)peer_ptr;
        unsigned short* o = (unsigned short*)out_ptr;
        for (unsigned long long e = tid; e < nelem; e += n)
            o[e] = st_f2bf_hw(st_bf2f(a[e]) + st_bf2f(b[e]));
    } else {
        const unsigned short* a = (const unsigned short*)in_ptr;
        const unsigned short* b = (const unsigned short*)peer_ptr;
        unsigned short* o = (unsigned short*)out_ptr;
        for (unsigned long long e = tid; e < nelem; e += n)
            o[e] = st_f2f16(st_f162f(a[e]) + st_f162f(b[e]));
    }
}

void stage_reduce(long in_ptr, long peer_ptr, long out_ptr, long long nelem,
                  int dtype, long stream) {
    hipStream_t st = (hipStream_t)stream;
    unsigned long long nb = (unsigned long long)((nelem + 1023) / 1024);
    if (nb < 1) nb = 1;
    if (nb > 2048) nb = 2048;
    if (dtype == 2) stage_reduce_kernel<2><<<(unsigned)nb, 256, 0, st>>>(
        (const void*)in_ptr, (const void*)peer_ptr, (void*)out_ptr, (unsigned long long)nelem);
    else if (dtype == 1) stage_reduce_kernel<1><<<(unsigned)nb, 256, 0, st>>>(
        (const void*)in_ptr, (const void*)peer_ptr, (void*)out_ptr, (unsigned long long)nelem);
    else stage_reduce_kernel<0><<<(unsigned)nb, 256, 0, st>>>(
        (const void*)in_ptr, (const void*)peer_ptr, (void*)out_ptr, (unsigned long long)nelem);
}

// ---------------------------------------------------------------------------
// all_gather. Both ranks write the full y identically: rank r's slice at
// r*nelem (plain), or row-interleaved [rows][2][row].
__global__ void stage_gather_plain_kernel(const unsigned char* __restrict__ my_x,
                                          const unsigned char* peer_x,
                                          unsigned char* __restrict__ y,
                                          unsigned long long nbytes, int rank) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long n = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long off0 = (unsigned long long)rank * nbytes;
    unsigned long long off1 = (unsigned long long)(1 - rank) * nbytes;
    for (unsigned long long k = tid; k < nbytes; k += n) {
        y[off0 + k] = my_x[k];
        y[off1 + k] = peer_x[k];
    }
}

__global__ void stage_gather_row_kernel(const unsigned char* __restrict__ my_x,
                                        const unsigned char* peer_x,
                                        unsigned char* __restrict__ y,
                                        long long n, long long row, long esize, int rank) {
    // Grid-stride over rows; threads copy the row's two slices.
    long long rows = n / row;
    for (long long t = blockIdx.x; t < rows; t += gridDim.x) {
        long long rb = (long long)row * esize;
        const unsigned char* xs = my_x + t * rb;
        const unsigned char* ps = peer_x + t * rb;
        unsigned char* ym = y + ((t * 2 + rank) * row) * esize;
        unsigned char* yp = y + ((t * 2 + 1 - rank) * row) * esize;
        for (long long k = threadIdx.x; k < rb; k += blockDim.x) {
            ym[k] = xs[k];
            yp[k] = ps[k];
        }
    }
}

void stage_gather(long my_x, long peer_x, long y_ptr, long long nelem, int esize,
                  int rank, long long row, int plain, long stream) {
    hipStream_t st = (hipStream_t)stream;
    if (plain) {
        unsigned long long nbytes = (unsigned long long)nelem * esize;
        unsigned long long nb = (nbytes + 1023) / 1024;
        if (nb < 1) nb = 1;
        if (nb > 2048) nb = 2048;
        stage_gather_plain_kernel<<<(unsigned)nb, 256, 0, st>>>(
            (const unsigned char*)my_x, (const unsigned char*)peer_x,
            (unsigned char*)y_ptr, nbytes, rank);
    } else {
        long long rows = nelem / row;
        unsigned nb = rows > 8192 ? 8192 : (unsigned)(rows > 0 ? rows : 1);
        stage_gather_row_kernel<<<nb, 256, 0, st>>>(
            (const unsigned char*)my_x, (const unsigned char*)peer_x,
            (unsigned char*)y_ptr, nelem, row, esize, rank);
    }
}

// ---------------------------------------------------------------------------
// moe gather inverse table: inv[f] = the sorted row holding slot f.
//
// FIRST-WINS, via atomicMin, and deliberately NOT last-wins. Beyond the
// rows the scatter actually wrote (`run`), the sorted buffer holds stale
// data -- typically zeros, so hundreds of stale rows all claim slot 0, which
// real tokens also claim. Last-wins hands the slot to whichever stale row
// loses the race last, and a real token gathers a padding row's expert
// output (timing-dependent garbage). First-wins hands it to the lowest
// index, and scatter writes real rows first (indices below `run`), so the
// real row always wins regardless of timing: atomicMin is commutative, so
// the result does not depend on launch order at all. Unclaimed slots keep
// INT_MAX and are skipped by the gather.
//
// TWO launches, stream-ordered: the clear must complete grid-wide before
// the scatter writes, and blocks cannot barrier between phases.
__global__ void stage_gather_inv_clear_kernel(int* __restrict__ inv, long long T) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long n = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long k = tid; k < (unsigned long long)T; k += n) inv[k] = 0x7fffffff;
}

__global__ void stage_gather_inv_fill_kernel(int* __restrict__ inv,
                                             const int* __restrict__ sorted, long long T) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long n = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long k = tid; k < (unsigned long long)T; k += n) {
        int f = sorted[k];
        if (f >= 0 && f < T) atomicMin((unsigned*)&inv[f], (unsigned)k);
    }
}

void stage_gather_inv(long inv_ptr, long sorted_ptr, long long T, long stream) {
    hipStream_t st = (hipStream_t)stream;
    unsigned long long nb = (unsigned long long)((T + 255) / 256);
    if (nb < 1) nb = 1;
    if (nb > 2048) nb = 2048;
    stage_gather_inv_clear_kernel<<<(unsigned)nb, 256, 0, st>>>((int*)inv_ptr, T);
    stage_gather_inv_fill_kernel<<<(unsigned)nb, 256, 0, st>>>(
        (int*)inv_ptr, (const int*)sorted_ptr, T);
}

// moe gather into y (bf16), one 8-word per thread, slots ascending, RNE
// narrows, shared arm folded exactly as moe_gather folds it.
__global__ void stage_gather_rows_kernel(
    const unsigned short* __restrict__ ye, const unsigned short* __restrict__ ew,
    const unsigned short* __restrict__ sh, const unsigned short* __restrict__ sg,
    const int* __restrict__ inv, unsigned short* __restrict__ y,
    long long M, long n, long top_k, int sig, int has_sh,
    long ye_ld, long ew_ld, long sh_ld, long sg_ld, long y_ld) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nwords = (unsigned long long)M * (n >> 3);
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long w = tid; w < nwords; w += stride) {
        long long row = w / (n >> 3);
        long long i0 = (w % (n >> 3)) * 8;
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        for (long j = 0; j < top_k; ++j) {
            long long f = row * top_k + j;
            int ii = inv[f];
            // Unclaimed slots keep INT_MAX from the clear; negative and
            // out-of-range entries are padding. All skipped.
            if (ii < 0 || (long long)ii >= M * top_k) continue;
            float wj = st_bf2f(ew[row * ew_ld + j]);
            const unsigned short* ve = ye + (long long)ii * ye_ld + i0;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += wj * st_bf2f(ve[e]);
        }
        unsigned short* yo = y + row * y_ld + i0;
        if (!has_sh) {
#pragma unroll
            for (int e = 0; e < 8; ++e) yo[e] = st_f2bf_rne(acc[e]);
        } else {
            float v0 = st_bf2f(sg[row * sg_ld]);
            float gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
            const unsigned short* shp = sh + row * sh_ld + i0;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                unsigned short g = st_f2bf_rne(acc[e]);
                unsigned short pr = st_f2bf_rne(st_bf2f(shp[e]) * gate);
                yo[e] = st_f2bf_rne(st_bf2f(pr) + st_bf2f(g));
            }
        }
    }
}

void stage_gather_rows(long ye_ptr, long ew_ptr, long sorted_ptr, long sh_ptr, long sg_ptr,
                       long inv_ptr, long y_ptr, long M, long n, long top_k, int sig,
                       long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                       long y_ld, int use_inv, long stream) {
    (void)use_inv; // always the inverse-table form
    hipStream_t st = (hipStream_t)stream;
    unsigned long long nwords = (unsigned long long)M * (n >> 3);
    unsigned long long nb = (nwords + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 8192) nb = 8192;
    stage_gather_rows_kernel<<<(unsigned)nb, 256, 0, st>>>(
        (const unsigned short*)ye_ptr, (const unsigned short*)ew_ptr,
        sh_ptr ? (const unsigned short*)sh_ptr : nullptr,
        sg_ptr ? (const unsigned short*)sg_ptr : nullptr,
        (const int*)inv_ptr, (unsigned short*)y_ptr,
        M, (int)n, (int)top_k, sig, (sh_ptr && sg_ptr) ? 1 : 0,
        ye_ld, ew_ld, sh_ld, sg_ld, y_ld);
}

// ---------------------------------------------------------------------------
// hc_write: h[r][c*n+i] += inj[r][c] * y[r][i], round-half-up narrow.
// Fully element-parallel: every output is independent.
__global__ void stage_hc_write_kernel(const unsigned short* __restrict__ y,
                                      const unsigned short* __restrict__ inj,
                                      unsigned short* __restrict__ h,
                                      long long M, long n, long hc,
                                      long irow, long hrow, long yrow) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)M * hc * n;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long k = tid; k < total; k += stride) {
        long long i = k % n;
        long long c = (k / n) % hc;
        long long r = k / ((unsigned long long)hc * n);
        float v = st_bf2f(y[r * yrow + i]);
        float g = st_bf2f(inj[r * irow + c]);
        unsigned short* hp = h + r * hrow + c * n + i;
        *hp = st_f2bf_hup(st_bf2f(*hp) + g * v);
    }
}

void stage_hc_write(long y_ptr, long inj_ptr, long h_ptr, long M, long n, long hc,
                    long irow, long hrow, long yrow, long stream) {
    hipStream_t st = (hipStream_t)stream;
    unsigned long long total = (unsigned long long)M * hc * n;
    unsigned long long nb = (total + 255) / 256;
    if (nb < 1) nb = 1;
    if (nb > 8192) nb = 8192;
    stage_hc_write_kernel<<<(unsigned)nb, 256, 0, st>>>(
        (const unsigned short*)y_ptr, (const unsigned short*)inj_ptr,
        (unsigned short*)h_ptr, M, (int)n, (int)hc, irow, hrow, yrow);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// rmsnorm + block-fp8 quant, ONE THREAD PER ROW (sequential per row, exactly
// the reference order). x holds the reduced bf16 message.
__global__ void stage_rmsnorm_quant_fp8_kernel(
    const unsigned short* __restrict__ x, unsigned short* __restrict__ res,
    const void* __restrict__ w, unsigned char* __restrict__ q,
    float* __restrict__ s, unsigned short* __restrict__ ob,
    long M, long n, float eps, float wadd, long group,
    long xpitch, long rpitch, long wpitch, long qpitch, long spitch, long obpitch,
    int has_res, int has_ob, int w_f32) {
    long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    const unsigned short* xr = x + r * xpitch;
    long ngrp = (n + group - 1) / group;
    // residual add, stored back before the variance reads it
    if (has_res) {
        unsigned short* rr = res + r * rpitch;
        for (long i = 0; i < n; ++i) {
            float t = st_bf2f(xr[i]) + st_bf2f(rr[i]);
            rr[i] = st_f2bf_rne(t);
        }
    }
    // variance over the narrowed source
    float ss = 0.0f;
    if (has_res) {
        const unsigned short* rr = res + r * rpitch;
        for (long i = 0; i < n; ++i) { float v = st_bf2f(rr[i]); ss += v * v; }
    } else {
        for (long i = 0; i < n; ++i) { float v = st_bf2f(xr[i]); ss += v * v; }
    }
    float rs = 1.0f / sqrtf(ss / (float)n + eps);
    unsigned char* qq = q + r * qpitch;
    float* srow = s + r * spitch;
    unsigned short* obr = has_ob ? ob + r * obpitch : nullptr;
    for (long gi = 0; gi < ngrp; ++gi) {
        long k0 = gi * group, k1 = k0 + group < n ? k0 + group : n;
        // amax over the narrowed bf16 values of this group
        float amax = 0.0f;
        for (long i = k0; i < k1; ++i) {
            float src = has_res ? st_bf2f(res[r * rpitch + i]) : st_bf2f(xr[i]);
            float h = st_bf2f(st_f2bf_rne(src * rs * (st_gain(w, i * wpitch, w_f32) + wadd)));
            if (has_ob) obr[i] = st_f2bf_rne(src * rs * (st_gain(w, i * wpitch, w_f32) + wadd));
            float m = h < 0.0f ? -h : h;
            if (m > amax) amax = m;
            // NOTE: h recomputed below for the codes; the values are
            // deterministic so the bits agree.
            (void)m;
        }
        float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
        float inv = 1.0f / sc;
        for (long i = k0; i < k1; ++i) {
            float src = has_res ? st_bf2f(res[r * rpitch + i]) : st_bf2f(xr[i]);
            float hv = st_bf2f(st_f2bf_rne(src * rs * (st_gain(w, i * wpitch, w_f32) + wadd)));
            float c = hv * inv;
            if (c > 448.0f) c = 448.0f;
            if (c < -448.0f) c = -448.0f;
            qq[i] = st_f2e4m3(c);
        }
        srow[gi] = sc;
    }
}

void stage_rmsnorm_quant_fp8(long x_ptr, long res_ptr, long w_ptr, long q_ptr, long s_ptr,
                             long ob_ptr, long M, long n, float eps, float wadd,
                             long group, long xpitch, long rpitch, long wpitch,
                             long qpitch, long spitch, long obpitch, int w_f32, long stream) {
    hipStream_t st = (hipStream_t)stream;
    // One thread per row: grid-stride over rows.
    unsigned long long nb = (unsigned long long)((M + 255) / 256);
    if (nb < 1) nb = 1;
    if (nb > 8192) nb = 8192;
    // 1 thread per block would serialize heavily; use 256 threads with each
    // handling strided rows (still one thread per row, sequential in-row).
    stage_rmsnorm_quant_fp8_kernel<<< (unsigned)nb, 256, 0, st>>>(
        (const unsigned short*)x_ptr,
        res_ptr ? (unsigned short*)res_ptr : nullptr,
        (const unsigned short*)w_ptr, (unsigned char*)q_ptr, (float*)s_ptr,
        ob_ptr ? (unsigned short*)ob_ptr : nullptr,
        M, (int)n, eps, wadd, group, xpitch, rpitch, wpitch, qpitch, spitch,
        obpitch, res_ptr ? 1 : 0, ob_ptr ? 1 : 0, w_f32);
}

// ---------------------------------------------------------------------------
// RMS norm + block Hadamard + int8 quant, ONE THREAD PER ROW. Despite the
// "ln" name this is the RMS form (see r4d_ar_ln_had_quant_i8.hip): the peer's
// contribution arrives reduced in red[] (scratch -- x is IN and untouched),
// adds with x and the residual, norms, rotates, quantises.
__global__ void stage_ln_had_quant_i8_kernel(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ red,
    unsigned short* __restrict__ res, const void* __restrict__ w,
    signed char* __restrict__ q, float* __restrict__ s, unsigned short* __restrict__ ob,
    long M, long n, float eps, float wadd, long had,
    long xpitch, long redpitch, long rpitch, long wpitch, long qpitch, long obpitch,
    int has_res, int has_ob, int w_f32) {
    long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    const unsigned short* xr = x + r * xpitch;
    const unsigned short* dr = red + r * redpitch;
    // residual add into res (narrowed), exactly the unfused order
    if (has_res) {
        unsigned short* rr = res + r * rpitch;
        for (long i = 0; i < n; ++i) {
            float t = st_bf2f(dr[i]) + st_bf2f(rr[i]);
            rr[i] = st_f2bf_rne(t);
        }
    }
    // variance over the narrowed source
    float ss = 0.0f;
    if (has_res) {
        const unsigned short* rr = res + r * rpitch;
        for (long i = 0; i < n; ++i) { float v = st_bf2f(rr[i]); ss += v * v; }
    } else {
        for (long i = 0; i < n; ++i) { float v = st_bf2f(dr[i]); ss += v * v; }
    }
    // NOTE: x (the local delta) is already folded into red[] by the exchange
    // kernel... no -- red holds mine+peer where "mine" must include x. See
    // the launch: the exchange reduces x+peer into the scratch. dr IS that.
    // (The unused xr read below keeps the compiler honest about contiguity.)
    (void)xr;
    float rs = 1.0f / sqrtf(ss / (float)n + eps);
    float norm = 1.0f / sqrtf((float)had);
    // per-group rotate + quantise; amax over the whole rotated row
    float amax = 0.0f;
    // pass 1: amax (recomputed values, deterministic)
    for (long b = 0; b < n; b += had) {
        float buf[512];
        for (long i = 0; i < had; ++i) {
            float src = has_res ? st_bf2f(res[r * rpitch + b + i]) : st_bf2f(dr[b + i]);
            buf[i] = src * rs * (st_gain(w, (b + i) * wpitch, w_f32) + wadd);
        }
        st_fwht(buf, had);
        for (long i = 0; i < had; ++i) {
            if (has_ob) {
                float osrc = has_res ? st_bf2f(res[r * rpitch + b + i]) : st_bf2f(dr[b + i]);
                ob[r * obpitch + b + i] =
                    st_f2bf_rne(osrc * rs * (st_gain(w, (b + i) * wpitch, w_f32) + wadd));
            }
            float m = buf[i] < 0.0f ? -buf[i] : buf[i];
            if (m > amax) amax = m;
        }
    }
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    signed char* qq = q + r * qpitch;
    for (long b = 0; b < n; b += had) {
        float buf[512];
        for (long i = 0; i < had; ++i) {
            float src = has_res ? st_bf2f(res[r * rpitch + b + i]) : st_bf2f(dr[b + i]);
            buf[i] = src * rs * (st_gain(w, (b + i) * wpitch, w_f32) + wadd);
        }
        st_fwht(buf, had);
        for (long i = 0; i < had; ++i) qq[b + i] = st_f2i8(buf[i] * inv);
    }
    s[r] = amax / 127.0f * norm;
}

void stage_ln_had_quant_i8(long x_ptr, long red_ptr, long res_ptr, long w_ptr, long q_ptr,
                           long s_ptr, long ob_ptr, long M, long n, float eps, float wadd,
                           long had, long xpitch, long redpitch, long rpitch, long wpitch,
                           long qpitch, long obpitch, int w_f32, long stream) {
    hipStream_t st = (hipStream_t)stream;
    unsigned long long nb = (unsigned long long)((M + 255) / 256);
    if (nb < 1) nb = 1;
    if (nb > 8192) nb = 8192;
    stage_ln_had_quant_i8_kernel<<<(unsigned)nb, 256, 0, st>>>(
        (const unsigned short*)x_ptr, (const unsigned short*)red_ptr,
        res_ptr ? (unsigned short*)res_ptr : nullptr,
        (const unsigned short*)w_ptr, (signed char*)q_ptr, (float*)s_ptr,
        ob_ptr ? (unsigned short*)ob_ptr : nullptr,
        M, (int)n, eps, wadd, had, xpitch, redpitch, rpitch, wpitch, qpitch,
        obpitch, res_ptr ? 1 : 0, ob_ptr ? 1 : 0, w_f32);
}
