# Test plugins from one source, so that plugin_test has a hierarchy to load, a schema conflict to
# refuse and a stale ABI to reject. They go under $RADIANCE_HOME/testkernels rather than kernels/,
# because the engine scans kernels/ at startup and these rows implement nothing: a test plugin that
# a real run could select is a fallback nobody knows about (spec.md §17).
#
# The two that must be refused live one directory down, so that loading testkernels/ is a clean
# load and the refusals are asked for explicitly by path.
rad_add_plugin(radtest_base testkernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_base PRIVATE RADTEST_BASE=1)

rad_add_plugin(radtest_over testkernels SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_over PRIVATE RADTEST_OVER=1)

rad_add_plugin(radtest_conflict testkernels/bad SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_conflict PRIVATE RADTEST_CONFLICT=1)

rad_add_plugin(radtest_badabi testkernels/bad SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_badabi PRIVATE RADTEST_BADABI=1)

# Two architecture plugins claiming one architecture id, in their own directory so that loading it
# is the whole test: the first is identified, the second is refused.
rad_add_plugin(radtest_arch testarch SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_arch PRIVATE RADTEST_ARCH=1)

rad_add_plugin(radtest_arch2 testarch SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_arch2 PRIVATE RADTEST_ARCH2=1)

# Two more from the same source, in a directory of their own, so that a load with NO hierarchy has
# something to be observed on. One of them declares itself a reference implementation.
#
# THE FILE NAMES ARE THE POINT. load_dir reads a directory in sorted order, so `..._a` (the
# reference) is staged first; if the loader did not pin a reference library below every other
# kernel plugin, it would keep that position and answer for every op, and the library that actually
# serves the card would never be selected. Naming them the other way round would make the test pass
# on alphabetical order alone. Its name is nobody's in particular: the rule is the declaration's.
rad_add_plugin(radtest_ord_a testkernels/order SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_ord_a PRIVATE RADTEST_BASE=1 RADTEST_REFERENCE=1
                                                 RADTEST_BASE_NAME=\"radtest_reference\")

rad_add_plugin(radtest_ord_b testkernels/order SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)
target_compile_definitions(radtest_ord_b PRIVATE RADTEST_BASE=1)

# The KV tiers' transfer test. Unlike everything above it COMPUTES -- byte-row gather and scatter
# registered in the device domain, running on the host backend's memory -- so it lives in a
# directory of its own that only tests/mem_test.cpp loads, by path. It calls back into the device
# API for stream order, which the test executable exports.
rad_add_plugin(radtest_rows testrows SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tierrows.cc)

# A quantiser with a scheme of its own, which only its decode hook reads: tests/quant_test.cpp
# loads it by path to show a weight it wrote decodes through the core's decoder all the same.
rad_add_plugin(radtest_quant testquant SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testquant.cc)
