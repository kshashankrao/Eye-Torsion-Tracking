#!/bin/bash
# build.sh - Build script for C++ executable (Release or Debug mode)

set -e

# Default build type is Release
BUILD_TYPE="Release"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --debug|-d)
        BUILD_TYPE="Debug"
        shift
        ;;
        --release|-r)
        BUILD_TYPE="Release"
        shift
        ;;
    esac
done

echo "=== Building C++ project in ${BUILD_TYPE} mode ==="

# Create a build directory corresponding to the build type
BUILD_DIR="build_${BUILD_TYPE,,}" # lowercase build type
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and compile
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j$(nproc)

echo "=== Build Complete! Binary is located in $BUILD_DIR/ ==="
cd ..
