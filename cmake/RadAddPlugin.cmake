# rad_add_plugin / rad_hipify -- the ONE definition of how a source file becomes a loadable
# radiance plugin, shared by the in-tree build and by an out-of-tree one.
#
# This file is included from two different worlds. The top-level CMakeLists.txt includes it while
# the ABI target is being built in this very tree; the installed radiance-config.cmake includes it
# after pulling the same target in as an import from radianceTargets.cmake. It is a file of its own
# because a third party who runs find_package(radiance) needs it as much as the tree does: the
# single call that decides what a plugin links, where it lands, and how its device sources reach
# hipcc cannot be a private detail of radiance's own build, or every out-of-tree plugin has to
# reinvent it by reading our CMakeLists.
#
# Copying the function into the package config instead is the worse option. Two copies of the rule
# that fixes the plugin layout drift, and the drift does not present as a build error: it presents
# as a plugin that loads in the tree it was written in and is silently absent everywhere else,
# because it was emitted one directory level away from where the loader scans. So there is one
# file, and both worlds include it.
#
# What an includer must set first:
#   radiance::abi      the ABI interface target -- an in-tree ALIAS, or the imported target
#   RAD_PLUGIN_OUT     the build-tree $RADIANCE_HOME; a plugin lands in ${RAD_PLUGIN_OUT}/<kind>
#   RAD_HIP_FOUND      whether device sources can be compiled at all
#   RAD_GPU_TARGETS    the offload architectures this build targets. Any gfx name; a library that
#                      serves only some of them narrows the list with rad_plugin_gpu_targets()
#   RAD_HIP_RUNTIME    the amdhip64 library a device plugin links
# The top-level file sets all five as it detects them. radiance-config.cmake sets all five from what
# the install recorded, so that an out-of-tree plugin is compiled against the same runtime as the
# engine that will dlopen it; its GPU targets are the consumer's to name (-DRAD_GPU_TARGETS) and
# default to the ones the engine was built for.
#
# And one an includer MAY set:
#   RAD_PLUGIN_INSTALL_HOME  the $RADIANCE_HOME `cmake --install` puts a plugin under. Unset in
#                      the tree, where a plugin installs beside the engine under the same prefix;
#                      radiance-config.cmake sets it to the found installation's home, so a plugin
#                      built out of tree installs where that engine's loader scans.

# For CMAKE_INSTALL_DATADIR, below. Idempotent, and included here rather than left to the includer
# because this file is what needs it: a consumer project has no reason to have included it, and the
# install destination would silently become the relative path "radiance/kernels" -- interpreted
# against the prefix, so it lands in <prefix>/radiance/kernels and the loader never looks there.
include(GNUInstallDirs)

# What radiance's own tree sets for every plugin it builds, for a plugin built on its own against
# an installed radiance: C++20, and a build type that optimises when none was asked for. Every
# plugin directory in radiance's tree opens with the same few lines a third party's does --
#
#   if(NOT COMMAND rad_add_plugin)          # on its own, not inside radiance's tree
#     cmake_minimum_required(VERSION 3.21)
#     project(mylib LANGUAGES CXX)          # CXX HIP for device code
#     find_package(radiance REQUIRED)
#     rad_plugin_defaults()
#   endif()
#
# -- so the libraries radiance ships are built and installed exactly as anyone else's are.
macro(rad_plugin_defaults)
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
  endif()
endmacro()

# A kernel, architecture or quantiser plugin: one call, so a new plugin is a directory and a line.
# `kind` is the $RADIANCE_HOME directory the engine scans for it. An installed plugin must name one
# of the three the engine scans -- a misspelt kind would build, install, and never be loaded; the
# test plugins, which are never installed (RAD_PLUGIN_INSTALL below), use directories of their own.
function(rad_add_plugin name kind)
  cmake_parse_arguments(P "" "" "SOURCES;HIP_SOURCES;LIBS;INCLUDES;GPU_TARGETS" ${ARGN})
  if((NOT DEFINED RAD_PLUGIN_INSTALL OR RAD_PLUGIN_INSTALL) AND
     NOT kind MATCHES "^(kernels|architectures|quantizers)$")
    message(FATAL_ERROR
      "rad_add_plugin(${name} ${kind}): the kind is the directory the engine loads the plugin "
      "from, and it is one of kernels, architectures or quantizers.")
  endif()
  set(srcs ${P_SOURCES})
  if(P_HIP_SOURCES AND NOT RAD_HIP_FOUND)
    # Said rather than skipped in silence: a kernel library whose device sources were dropped
    # builds, installs, and has rows with no kernels behind them.
    message(WARNING
      "rad_add_plugin(${name}) was given HIP_SOURCES but this radiance has no device build "
      "(RAD_HIP_FOUND is OFF), so they are left out. A radiance configured without ROCm, or on a "
      "machine with no card and no -DRAD_GPU_TARGETS, has none.")
  endif()
  if(RAD_HIP_FOUND AND P_HIP_SOURCES)
    # The HIP language has to be enabled by the project that owns the target, not by us:
    # enable_language() is a project-scope decision and a library file that reaches out and enables
    # a language behind the consumer's back changes their compiler set. In-tree the top-level file
    # does it. Out-of-tree it is one line in the consumer's CMakeLists, and the failure without it
    # is "Cannot find source file" or a source compiled as C++ with --offload-arch on the command
    # line, neither of which names the missing enable_language(HIP).
    get_property(_rad_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "HIP" IN_LIST _rad_langs)
      message(FATAL_ERROR
        "rad_add_plugin(${name}) was given HIP_SOURCES but the HIP language is not enabled in this "
        "project. Add `enable_language(HIP)` after project() -- radiance cannot enable it for you, "
        "because the language a project compiles in is the project's own decision.")
    endif()
    set_source_files_properties(${P_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    list(APPEND srcs ${P_HIP_SOURCES})
  endif()
  add_library(${name} MODULE ${srcs})
  target_link_libraries(${name} PRIVATE radiance::abi ${P_LIBS})
  target_include_directories(${name} PRIVATE ${P_INCLUDES})
  # The build tree is a working $RADIANCE_HOME, and it is created here rather than assumed: a
  # LIBRARY_OUTPUT_DIRECTORY that does not exist is created by the build, but the engine scans the
  # directory at startup and an out-of-tree build with one kernel plugin and no architectures would
  # otherwise have no architectures/ for the scan to find and fail at the scan rather than at the
  # lookup.
  file(MAKE_DIRECTORY ${RAD_PLUGIN_OUT}/${kind})
  set_target_properties(${name} PROPERTIES
    PREFIX "" OUTPUT_NAME "${name}"
    LIBRARY_OUTPUT_DIRECTORY ${RAD_PLUGIN_OUT}/${kind})
  if(RAD_HIP_FOUND AND P_HIP_SOURCES)
    rad_hipify(${name} ${P_GPU_TARGETS})
  endif()

  # ...and so is the install tree, in the same shape. RAD_DEFAULT_HOME points at
  # <prefix>/share/radiance, and without this rule nothing is installed into it, so the only
  # $RADIANCE_HOME that works is somebody's build directory. A plugin built and not installed is the
  # same class of defect as a plugin built and never loaded.
  #
  # RAD_PLUGIN_INSTALL is the opt-out, and it has exactly one in-tree user: core/plugin/testplugin
  # (set in core/CMakeLists.txt, where that subtree is reached), whose .so files exist to be
  # refused, to conflict, to report a stale ABI and to stand in for a card. Shipping those
  # into a prefix would put rows into a real installation that implement nothing -- see the same
  # argument, made about the build tree, in core/plugin/testplugin/CMakeLists.txt.
  if(NOT DEFINED RAD_PLUGIN_INSTALL)
    set(RAD_PLUGIN_INSTALL ON)
  endif()
  if(RAD_PLUGIN_INSTALL)
    if(RAD_PLUGIN_INSTALL_HOME)
      set(_rad_dest "${RAD_PLUGIN_INSTALL_HOME}/${kind}")
    else()
      set(_rad_dest "${CMAKE_INSTALL_DATADIR}/radiance/${kind}")
    endif()
    install(TARGETS ${name} LIBRARY DESTINATION "${_rad_dest}")
  endif()
endfunction()

# Mark a target's device sources as HIP and pin the architecture. -ffp-contract=off is mandatory
# for any quantised cross-rank op: two ranks see the two dequantised products in the opposite order
# and contracting either into an FMA makes them disagree by ~1 ULP, which breaks the
# replicated-state invariant.
function(rad_hipify tgt)
  # Any extra arguments are the architectures THIS target is compiled for, which is how a library
  # that serves one generation stays out of the build for the others. With none given the target
  # takes the whole build's list.
  set(arches "${ARGN}")
  if(NOT arches)
    set(arches "${RAD_GPU_TARGETS}")
  endif()
  set_target_properties(${tgt} PROPERTIES HIP_ARCHITECTURES "${arches}")
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
  # THE DEVICE OPTIMISATION LEVEL, ON THE TARGET, because CMake supplies no usable one for HIP:
  # depending on its version CMAKE_HIP_FLAGS_<CONFIG> is empty -- the kernels compile at -O0 with
  # asserts live -- or the host-style '-O2 -g'. The top-level CMakeLists.txt sets the variables for
  # the tree; a plugin built out of tree gets the same levels here, after the toolchain's flags on
  # the command line, so they win. A later -O replaces an earlier one but a later -O does not undo
  # -g, so the levels outside Debug end in -g0: device debug info costs compile time, inhibits
  # scheduling, and makes an out-of-tree build's code objects differ from the tree's. NDEBUG is not
  # decoration on the device: an assert is a hostcall, and the engine's queues provide no hostcall
  # buffer (docs/PLUGIN.md). RAD_HIP_FLAGS_<CONFIG> overrides, as it does in the tree.
  foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    if(RAD_HIP_FLAGS_${cfg})
      set(lvl "${RAD_HIP_FLAGS_${cfg}}")
    elseif(cfg STREQUAL "DEBUG")
      set(lvl "-O0 -g")
    elseif(cfg STREQUAL "MINSIZEREL")
      set(lvl "-Os -DNDEBUG -g0")
    else()
      set(lvl "-O3 -DNDEBUG -g0")
    endif()
    separate_arguments(lvl)
    target_compile_options(${tgt} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:${cfg}>>:${lvl}>")
  endforeach()
  # AN UNCOMPRESSED FAT BINARY, because the engine reads the code object out of it itself
  # (core/device/aql_code.cpp) and refuses a compressed one -- at the first launch, as a kernel
  # failure. A toolchain or a CMAKE_HIP_FLAGS that turns compression on is overridden here.
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--no-offload-compress>)
  target_link_options(${tgt} PRIVATE $<$<LINK_LANGUAGE:HIP>:--no-offload-compress>)
  target_link_libraries(${tgt} PRIVATE ${RAD_HIP_RUNTIME})
endfunction()

# Narrow RAD_GPU_TARGETS to the architectures ONE kernel library actually serves. A library states
# its coverage as regular expressions -- "^gfx12" for a generation, "^gfx1201$" for a single chip --
# and gets back the intersection with what this build is targeting, which may be empty.
#
# AN EMPTY RESULT IS NOT AN ERROR. It means this build's cards are served by some other kernel
# library, and the right response is for the plugin to skip itself and say so, leaving the rest of
# the tree to configure normally. A kernel library that failed the configure instead would make
# every card it does not cover an unbuildable project rather than an unserved one -- which is the
# opposite of what a plugin boundary is for.
function(rad_plugin_gpu_targets out)
  set(keep "")
  foreach(t ${RAD_GPU_TARGETS})
    foreach(pat ${ARGN})
      if(t MATCHES "${pat}")
        list(APPEND keep ${t})
        break()
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES keep)
  set(${out} "${keep}" PARENT_SCOPE)
endfunction()
