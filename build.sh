#!/bin/sh
# build.sh -- configure and build.
#
# With no arguments the build configures itself: if hipcc is present and a card can be enumerated,
# device kernels are compiled for that card's architecture, and otherwise the host backend is
# built. The host backend is a real implementation, not a stub -- everything except device kernels
# is exercised through it -- so a machine with no ROCm is a supported development configuration.
#
# Pass -DRAD_GPU_TARGETS=<arch>[;<arch>...] to cmake to compile for cards this machine does not
# have. Which of them actually get kernels is decided by the kernel libraries present; each one
# declares the architectures it serves.
#
#   ./build.sh              configure + build + test
#   ./build.sh -c           wipe the build directory first
#   ./build.sh --no-hip     force the host backend even where hipcc exists
set -e

BUILD=build
CMAKE_ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--clean) rm -rf "$BUILD" ;;
        --no-hip)   CMAKE_ARGS="$CMAKE_ARGS -DRAD_WITH_HIP=OFF" ;;
        --asan)     CMAKE_ARGS="$CMAKE_ARGS -DRAD_ASAN=ON" ;;
        --debug)    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug" ;;
        -b)         shift; BUILD="$1" ;;
        *)          echo "build.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

cmake -S . -B "$BUILD" -G Ninja $CMAKE_ARGS
cmake --build "$BUILD" -j"$(nproc)"
ctest --test-dir "$BUILD" --output-on-failure
