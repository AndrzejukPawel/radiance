# libavx -- the conventional op vocabulary in hand-written x86 SIMD, dispatched by CPU feature.
#
# ============================ THE BUILD IS THE DISPATCH ============================
#
# This file is not a list of sources. It is the mechanism that makes runtime dispatch SAFE, and
# every decision in it is load-bearing:
#
#   THE KERNELS ARE COMPILED FOUR TIMES, once per ISA level, into four OBJECT libraries with
#   different -m flags and AVX_LEVEL defined to 0/1/2/3. Same sources, four instruction streams.
#   avx_vec.h wraps each level in its own inline namespace so the four sets of symbols do not
#   collide -- which is not a nicety: without it the linker would pick one definition of
#   avx::row_to_f32 for all four tables, and on a machine without AVX-512 the SCALAR path would
#   execute AVX-512 and fault.
#
#   THE DISPATCHER AND THE REGISTRY ARE COMPILED BASELINE, with no -m flags at all. They are what
#   decides whether the other levels may run, so an instruction above the baseline in either of
#   them faults before the decision is made. Note that this is a property of the FLAGS and not of
#   the source: a compiler given -mavx512f may auto-vectorise anything, so "contains no intrinsics"
#   is not the same claim.
#
#   -march=native APPEARS NOWHERE. It is the obvious thing and it is wrong for a plugin: the .so is
#   built on one machine, copied to others, and dlopened by an engine that has no idea what it was
#   compiled for. A binary built with -march=native on a Zen 5 host SIGILLs on the first
#   AVX-512 instruction anywhere older, before main, with no diagnostic.
#
# ============================ IN THE TREE, AND ALONE ============================
#
# The root CMakeLists adds this directory on every x86-64 build: these are the production host
# kernels, and a model whose placement puts an op on the host -- the n-gram embedding's gather,
# every step -- has no other kernel for it that the engine will serve on.
#
# It also builds ON ITS OWN, as any third party's plugin does: against an installed radiance,
# through find_package and the same rad_add_plugin -- `cmake -S libavx -B build-avx
# -DCMAKE_PREFIX_PATH=<radiance prefix>`. A box with no ROCm builds and checks it that way without
# the HIP probe and the whole core; the oracle the checker compares against is the installed
# reference library.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libavx LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# The harnesses land beside the plugin, in the build's $RADIANCE_HOME, where the loader ignores
# anything that is not a .so.
set(AVX_OUT_DIR ${RAD_PLUGIN_OUT}/kernels)

# ============================ the four ISA levels ============================
# Every kernel translation unit. avx_dispatch.cpp and avx_registry.cpp are NOT here -- they are
# baseline and belong to the plugin target itself.
set(AVX_KERNEL_SRC
  avx_support.cpp
  avx_table.cpp
  avx_elementwise.cpp
  avx_norm.cpp
  avx_hc.cpp
  avx_quant.cpp
  avx_gemm.cpp
  avx_attn.cpp
  avx_moe.cpp
  avx_gdn.cpp
  avx_sample.cpp
  avx_misc.cpp
)

# The flag sets, one per level. Written out rather than derived, because each one is a claim about
# what instructions the object may contain and a derived list hides that claim.
#
#   0 scalar  nothing. x86-64 baseline, which is SSE2 -- the compiler will use it for scalar float
#             and that is fine: SSE2 is in the base ISA and exists on every x86-64 that boots.
#   1 avx     -mavx alone. NOT -mf16c and NOT -mfma: both are separate CPUID bits that arrived
#             after AVX (Ivy Bridge and Haswell), and a level called `avx` that required them would
#             fault on exactly the Sandy Bridge machines it exists for.
#   2 avx2    -mavx2 -mfma -mf16c. These three ship together on every part that has any of them.
#   3 avx512  F + BW + DQ + VL, which is what avx_dispatch.cpp checks for as a conjunction, plus
#             the AVX2 set beneath it. Deliberately NOT avx512bf16 / avx512vnni: many AVX-512
#             parts have neither, and neither is used (see avx_vec.h on why the bf16 convert is
#             done with integer ops rather than vcvtneps2bf16).
set(AVX_FLAGS_0 "")
set(AVX_FLAGS_1 -mavx)
set(AVX_FLAGS_2 -mavx2 -mfma -mf16c)
set(AVX_FLAGS_3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma -mf16c)

find_package(OpenMP QUIET)

foreach(lvl RANGE 0 3)
  add_library(avx_lvl${lvl} OBJECT ${AVX_KERNEL_SRC})
  target_compile_definitions(avx_lvl${lvl} PRIVATE AVX_LEVEL=${lvl})
  target_link_libraries(avx_lvl${lvl} PRIVATE radiance::abi)
  target_include_directories(avx_lvl${lvl} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(avx_lvl${lvl} PRIVATE ${AVX_FLAGS_${lvl}}
    # CONTRACTION IS ALLOWED HERE AND THAT IS THE OPPOSITE OF libref'S RULE, stated so the
    # difference is deliberate rather than inherited. libref passes -ffp-contract=off because it is
    # the ORACLE: a reference that contracts disagrees with itself between compilers. libavx is a
    # fast kernel checked against that oracle to a tolerance, and refusing FMA would throw away
    # half the machine's floating-point throughput to chase a bit-exactness it already gives up by
    # vectorising the reduction at all.
    -ffp-contract=fast
    # NO -ffast-math, and it is worth saying why given the above. Contraction changes the last ulp;
    # -ffast-math changes what the code MEANS -- it lets the compiler assume no NaN and no signed
    # zero, and this vocabulary is full of both (a masked logit is -inf, an attention row with no
    # valid keys is a NaN the online softmax has to carry). The checker compares the finiteness
    # pattern before it compares any number, exactly so that this cannot be smuggled in.
    -fno-math-errno)
  set_property(TARGET avx_lvl${lvl} PROPERTY POSITION_INDEPENDENT_CODE ON)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(avx_lvl${lvl} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

# ============================ the plugin ============================
# avx_ngram.cpp is baseline for the same reason as the other two, and for one more: it holds the
# embedding gather's row cache and its reader thread, and there must be ONE of each in the process
# whatever level runs -- compiled per level, each level would keep its own.
set(AVX_BASELINE_SRC avx_dispatch.cpp avx_registry.cpp avx_ngram.cpp)

rad_add_plugin(libavx kernels SOURCES ${AVX_BASELINE_SRC} INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

foreach(lvl RANGE 0 3)
  target_link_libraries(libavx PRIVATE avx_lvl${lvl})
endforeach()
find_package(Threads REQUIRED)
target_link_libraries(libavx PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libavx PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libavx: no OpenMP -- the plugin will run single-threaded")
endif()

# ============================ the harnesses ============================
# They dlopen both plugins by path and link NEITHER. That is deliberate and it is what makes them
# usable in both builds and against a plugin built somewhere else entirely: the checker's whole
# claim is that it exercises the .so the engine would load, and a checker that statically linked
# the kernels would be exercising a different binary.
add_executable(rad-avx-check avx_check.cpp avx_check_rows.cpp avx_cases.cpp)
target_include_directories(rad-avx-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad-avx-check PRIVATE radiance::abi ${CMAKE_DL_LIBS} Threads::Threads)
set_target_properties(rad-avx-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${AVX_OUT_DIR})

add_executable(rad-avx-bench avx_bench.cpp avx_cases.cpp)
target_include_directories(rad-avx-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad-avx-bench PRIVATE radiance::abi ${CMAKE_DL_LIBS})
set_target_properties(rad-avx-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${AVX_OUT_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(rad-avx-bench PRIVATE OpenMP::OpenMP_CXX)
endif()
