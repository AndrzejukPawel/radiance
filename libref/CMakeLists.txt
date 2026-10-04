# libref -- every op in the conventional vocabulary, naively and generically (spec §17).
#
# It links radiance::abi AND NOTHING ELSE. A kernel plugin does not link the core: abi/ is the
# whole contract, and a plugin that reached into core/ would stop being substitutable, which is the
# one property the plugin boundary exists to have. What it shares with the engine -- the dtype
# conversions, the parameter readers -- it takes from the ABI headers, which define them inline.
#
# The glob is CONFIGURE_DEPENDS, so a new translation unit in this directory is picked up by the
# next build with no edit here.
# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S libref -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`) -- the same
# rad_add_plugin either way (cmake/RadAddPlugin.cmake).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libref LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

file(GLOB REF_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

rad_add_plugin(libref kernels SOURCES ${REF_SRC})

# -ffp-contract=off IS A CORRECTNESS FLAG HERE, not a tuning one. rad-kbench compares resolved
# kernels against this plugin to a tolerance, so `a*b + c` has to round twice: a reference that
# contracts into an FMA disagrees with itself between two compilers, and one that reassociates
# cannot be trusted to a tolerance at all (libref/ref_common.h).
target_compile_options(libref PRIVATE -ffp-contract=off)

# OpenMP over the outer dimension only -- one thread owns whole output elements, and there is not
# one reduction() clause in the plugin, so two runs at different thread counts are bit-identical.
# Optional: without it the pragmas are ignored and the plugin is correct and slower, which is the
# right failure mode for a fallback.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libref PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libref: no OpenMP -- the reference will run single-threaded")
endif()
