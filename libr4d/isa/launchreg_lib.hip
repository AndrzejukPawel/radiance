// THE LIBRARY HALF of the registered-kernel test. Built twice, at -DNK=2 and -DNK=512, and
// dlopen'd -- because that is libr4d's actual shape: a shared object holding every template
// instantiation of every kernel, entered through a function pointer.
//
// Both builds launch THE SAME kernel, k<0>, with the same arguments. The only difference is how
// many other instantiations sit in the module beside it.
#include <hip/hip_runtime.h>

template <int I>
__global__ void k(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = (float)I; }

/* The table exists to keep the instantiations alive. Without a use, the compiler drops every k<I>
 * that is never launched and both builds register one kernel. */
static const void* g_tab[NK];

template <int I>
struct Reg {
    static void go() { g_tab[I] = (const void*)k<I>; Reg<I - 1>::go(); }
};
template <>
struct Reg<0> {
    static void go() { g_tab[0] = (const void*)k<0>; }
};

/* A THUNK PER INSTANTIATION. The arms above launch one kernel over and over, which is the one
 * thing a decode step never does: it issues forty-five DIFFERENT kernels, each through its own
 * shim, each with its own argument buffer and its own host stub. Everything the tight loop keeps
 * in L1 -- the code, the stub, the args -- is cold on every one of those. This makes that
 * difference, and only that difference, measurable. */
typedef void (*Thunk)(float*, hipStream_t);
template <int I>
static void thunk(float* p, hipStream_t st) {
    hipLaunchKernelGGL(k<I>, dim3(1), dim3(64), 0, st, p);
}
static Thunk g_thunk[NK];
template <int I>
struct RegT {
    static void go() { g_thunk[I] = &thunk<I>; RegT<I - 1>::go(); }
};
template <>
struct RegT<0> {
    static void go() { g_thunk[0] = &thunk<0>; }
};

extern "C" int lr_launch_distinct(void* p, void* stream, int n) {
    for (int i = 0; i < n; ++i) g_thunk[i % NK]((float*)p, (hipStream_t)stream);
    return 0;
}

extern "C" int lr_registered(void) {
    Reg<NK - 1>::go();
    RegT<NK - 1>::go();
    int live = 0;
    for (int i = 0; i < NK; ++i) live += g_tab[i] != 0;
    return live;
}

extern "C" int lr_launch(void* p, void* stream) {
    hipLaunchKernelGGL(k<0>, dim3(1), dim3(64), 0, (hipStream_t)stream, (float*)p);
    return 0;
}
