# libquant -- the built-in quantisers (spec §4.5). Host C++ only.
#
# It links rad_abi AND NOTHING ELSE, like every plugin: abi/rad_quant.h is the whole contract, and a
# quantiser that reached into core/ would stop being substitutable -- which is the property that
# lets a user ship a quantiser of their own beside it.
# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S libquant -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`) -- the same
# rad_add_plugin either way (cmake/RadAddPlugin.cmake).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libquant LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

file(GLOB LQ_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

rad_add_plugin(libquant quantizers SOURCES ${LQ_SRC} INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

# -ffp-contract=off because a quantiser's output is a stored file: the same weights converted by two
# builds have to give the same bytes, and an FMA the compiler chose in one and not the other moves a
# value on a rounding boundary into the next code.
target_compile_options(libquant PRIVATE -ffp-contract=off)

# OpenMP over rows: GPTQ over a 512-expert layer is N*K^2/2 flops an expert, and round-to-nearest
# over a 51 GB table is a long single-threaded loop. Rows are independent in both, so the result
# does not depend on the thread count. Optional: without it everything is correct and serial.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libquant PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libquant: no OpenMP -- quantisers will run single-threaded")
endif()
