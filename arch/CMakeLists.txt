# Architecture plugins. One directory each, one .so each, linking only rad_abi -- an architecture
# plugin does not link the core.
#
# qwen35 covers the Qwen3.5, Qwen3.6 and Qwen3.8 dense models, which all report that id, and
# qwen35moe is the same graph with a routed FFN. See docs/ARCHITECTURES.md for the geometry.
# One plugin per (architecture, quantisation) -- spec §2.4, and the pair is the selection key, so
# qwen35_bf16 and qwen35_fp8 both claim "qwen35" and are told apart by the container's quantisation
# descriptor. See rad_arch_quant in abi/rad_builder.h.
#
# The plugins are discovered, not listed. A hand-written list can name a plugin that has no
# directory, and a name in a build file that resolves to nothing is worse than no name: it is a
# plugin an operator will look for and not find, with no error anywhere to say it was never built.
# With discovery, what is in the tree is what is built, and what is not in the tree does not appear
# in a list.
#
# Globbed the way libref/CMakeLists.txt globs its translation units, and SORTED, because
# file(GLOB) returns directory order and an unsorted list makes the link order -- and therefore the
# plugin load order, which is the hierarchy that decides which architecture wins a tie -- depend on
# the filesystem.
file(GLOB RAD_ARCH_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT RAD_ARCH_DIRS)

# THE INCLUDE SPELLING IS THE INSTALLED ONE. arch/common/ installs as <include>/radiance/arch/, so a
# plugin built against an installed radiance writes `#include <arch/rad_arch.h>`; the plugins in
# this tree write the same, through a directory in the build tree whose arch/ is this common/.
# A plugin here that could only be built here would not be a plugin like any other.
set(RAD_ARCH_INCLUDE_ROOT ${CMAKE_BINARY_DIR}/rad_include)
file(MAKE_DIRECTORY ${RAD_ARCH_INCLUDE_ROOT})
if(NOT EXISTS ${RAD_ARCH_INCLUDE_ROOT}/arch)
  file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/common ${RAD_ARCH_INCLUDE_ROOT}/arch SYMBOLIC)
endif()
target_include_directories(rad_abi INTERFACE $<BUILD_INTERFACE:${RAD_ARCH_INCLUDE_ROOT}>)

foreach(d ${RAD_ARCH_DIRS})
  if(NOT IS_DIRECTORY ${d})
    continue()
  endif()
  get_filename_component(a ${d} NAME)
  # common/ is the shared header library every plugin includes, not a plugin. It is excluded by
  # name, and that is the only directory with special treatment.
  if(a STREQUAL "common")
    continue()
  endif()
  if(EXISTS ${d}/CMakeLists.txt)
    # Every plugin carries its own build rules, the ones it would be built with out of this tree.
    add_subdirectory(${a})
  else()
    # Named rather than skipped in silence. A directory under arch/ with no CMakeLists.txt is
    # either a plugin missing its build file or a directory that does not belong here, and both of
    # those are things somebody wants to be told about at configure time rather than discovering
    # from an architecture the engine cannot find.
    message(STATUS
      "arch: '${a}' has no CMakeLists.txt -- not an architecture plugin, skipped")
  endif()
endforeach()

# arch/common/ ships. abi/rad_builder.h tells a plugin author, in the ABI header itself, to use the
# grp_layer()/grp_expert()/grp_model() inline functions in arch/common/rad_arch.h rather than the
# rad_group_*() macros beneath that note, because those macros are C compound literals and C++
# accepts them only as a GNU extension -- and an architecture plugin in C++ is the norm. Without
# this install the shipped header would point at a file that is not shipped, and the one documented
# way to write a conforming plugin would be available only inside this tree.
#
# Installed FLAT into arch/, which is what makes it work without touching a single #include:
# these headers include each other by bare basename ("rad_arch.h", "rad_fp8.h"), which resolves
# relative to the including file in both trees, and rad_arch.h's own "rad_builder.h" resolves
# against the include root that radiance::abi puts on the path. Every plugin, in this tree or out of
# it, writes `#include <arch/rad_arch.h>` -- see RAD_ARCH_INCLUDE_ROOT above for how that resolves
# here.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/radiance/arch
        FILES_MATCHING PATTERN "*.h")
