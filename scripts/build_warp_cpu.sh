#!/bin/bash
set -euo pipefail

# build_warp_cpu.sh:
# Builds an optimized (-march=native + Intel oneAPI + OpenMP) CPU-only NVIDIA Warp (warp-lang).
# Eliminates the "Warp CUDA error: Warp requires CUDA driver 12.0 or higher..." error on systems
# running legacy NVIDIA drivers (such as Driver 470 / CUDA 11.4).

BUILD_DIR="/tmp/warp_build"
VENV_PYTHON="${VENV_PYTHON:-/opt/venv312/bin/python}"

echo "=== NVIDIA Warp CPU-only Build (-march=native + Intel oneAPI) ==="

# Source Intel oneAPI environment if available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Sourcing Intel oneAPI environment..."
    set +u
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

# Locate warp package native directory
WARP_DIR="$("$VENV_PYTHON" -c "import warp, os; print(os.path.dirname(warp.__file__))")"
WARP_NATIVE_DIR="${WARP_DIR}/native"

if [ ! -d "$WARP_NATIVE_DIR" ]; then
    echo "Error: Warp native source directory not found at $WARP_NATIVE_DIR" >&2
    exit 1
fi

echo "Found Warp package directory: $WARP_DIR"
echo "Found Warp native source directory: $WARP_NATIVE_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# NanoVDB headers ship with warp-lang (warp/native/nanovdb); NANOVDB_DIR may override.
NANOVDB_DIR="${NANOVDB_DIR:-$WARP_NATIVE_DIR}"
if [ ! -d "$NANOVDB_DIR/nanovdb" ]; then
    echo "Error: NanoVDB headers not found at $NANOVDB_DIR" >&2
    exit 1
fi

# Intel oneAPI include paths (if present)
MKL_INC=""
TBB_INC=""
[ -d "/opt/intel/oneapi/mkl/latest/include" ] && MKL_INC="-I/opt/intel/oneapi/mkl/latest/include"
[ -d "/opt/intel/oneapi/tbb/latest/include" ] && TBB_INC="-I/opt/intel/oneapi/tbb/latest/include"

# Create version script to export only warp public API symbols
cat << 'MAPEOF' > "$BUILD_DIR/warp.map"
{
    global:
        init;
        shutdown;
        cuda_driver_version;
        cuda_toolkit_version;
    local:
        *;
};
MAPEOF

CXX_FLAGS=(
    -O3
    -march=native
    -mtune=native
    --std=c++17
    -fno-rtti
    -fPIC
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -DWP_ENABLE_CUDA=0
    -DWP_ENABLE_MATHDX=0
    -DWP_ENABLE_CUDA_COMPATIBILITY=0
    -DWP_DISABLE_CUBQL=1
    -DWP_ENABLE_DEBUG=0
    -DNDEBUG
    -D_GLIBCXX_USE_CXX11_ABI=0
    -fopenmp
    -I"$WARP_NATIVE_DIR"
    -I"$NANOVDB_DIR"
)

[ -n "$MKL_INC" ] && CXX_FLAGS+=("$MKL_INC")
[ -n "$TBB_INC" ] && CXX_FLAGS+=("$TBB_INC")

SOURCES=(
    warp.cpp
    bvh.cpp
    bvh_cubql.cpp
    scan.cpp
    apic.cpp
    alloc_tracker.cpp
    crt.cpp
    error.cpp
    cuda_util.cpp
    mesh.cpp
    hashgrid.cpp
    reduce.cpp
    runlength_encode.cpp
    sort.cpp
    sparse.cpp
    volume.cpp
    volume_builder.cpp
    texture.cpp
    mathdx.cpp
    coloring.cpp
    deterministic.cpp
)

echo "Compiling ${#SOURCES[@]} source files with -march=native and OpenMP..."
OBJS=()
for src in "${SOURCES[@]}"; do
    obj="${src}.o"
    OBJS+=("$obj")
    echo "  [CXX] $src -> $obj"
    g++ "${CXX_FLAGS[@]}" -c "$WARP_NATIVE_DIR/$src" -o "$obj"
done

echo "Linking warp.so..."
g++ -shared -O3 -march=native -fopenmp \
    -static-libstdc++ -static-libgcc \
    -Wl,--version-script="$BUILD_DIR/warp.map" \
    -Wl,-z,lazy \
    -Wl,--exclude-libs,ALL \
    -o "$BUILD_DIR/warp.so" \
    "${OBJS[@]}"

strip "$BUILD_DIR/warp.so"
echo "Built $BUILD_DIR/warp.so ($(stat -c %s "$BUILD_DIR/warp.so") bytes)"

# Install into the active python environment
TARGET_BIN="${WARP_DIR}/bin/warp.so"
echo "Installing to $TARGET_BIN..."
cp -f "$BUILD_DIR/warp.so" "$TARGET_BIN"

echo "=== Warp CPU-only build and installation successful! ==="
"$VENV_PYTHON" -c "import warp as wp; wp.init(); print('Warp verification: SUCCESS (device=cpu)')"
