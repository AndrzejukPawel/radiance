# libstage -- tensor-parallel collectives staged through pinned host memory,
# for machines whose GPUs have no PCIe P2P path between them.
#
# A plugin like any other: built here inside radiance's tree, or on its own
# against an installed radiance (`cmake -S libstage -B build-libstage
# -DCMAKE_PREFIX_PATH=<prefix>`, with the HIP compiler named as for the tree)
# -- the same rad_add_plugin either way (cmake/RadAddPlugin.cmake).
#
# Device code is plain loads, stores and system-scope flag handshakes over
# hipHostMallocMapped memory -- no architecture-specific instructions -- so
# this library serves whatever RAD_GPU_TARGETS names. Without HIP there is no
# device side at all (the exchange kernels ARE the transport), so unlike a
# library with a host fallback this one is skipped whole.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libstage LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

if(NOT RAD_HIP_FOUND)
  message(STATUS "libstage: needs the HIP backend (its transport is device kernels) -- not built")
  return()
endif()

rad_add_plugin(libstage kernels
  SOURCES     stage_transport.cpp stage_fused.cpp stage_rows.cpp
  HIP_SOURCES stage_kernels.hip
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})
