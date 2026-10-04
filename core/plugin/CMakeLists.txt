# The plugin layer's own sources (registry.h, loader.cpp, schema.cpp, select.cpp, bands.cpp) are
# picked up by core/CMakeLists.txt's per-directory glob and compiled into rad_core. This file
# exists only for the test plugins below it, which must be .so files rather than part of the
# engine -- the plugin loader cannot be tested by anything it does not dlopen.
add_subdirectory(testplugin)
