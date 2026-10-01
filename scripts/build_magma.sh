#!/usr/bin/env bash
# Build and install MAGMA (unpatched upstream release) for sm_35 + sm_50 against Intel oneMKL.
# Usage: build_magma.sh <install-prefix>
set -euo pipefail

PREFIX="${1:?install prefix}"
JOBS="${JOBS:-4}"
MAGMA_VERSION="${MAGMA_VERSION:-2.10.0}"
MAGMA_SHA256="${MAGMA_SHA256:-ea0c57fcb64ac2fd7ffe8f02d8fe18f07055c5b7fba0164f565d1e3a85148fb5}"
WORK="${WORK:-/tmp/magma}"

mkdir -p "$WORK" && cd "$WORK"
curl -fsSLO "https://icl.utk.edu/projectsfiles/magma/downloads/magma-${MAGMA_VERSION}.tar.gz"
echo "${MAGMA_SHA256}  magma-${MAGMA_VERSION}.tar.gz" | sha256sum -c -
tar -xzf "magma-${MAGMA_VERSION}.tar.gz"

set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
set -u

# Intel10_64lp = mkl_intel_lp64 + mkl_intel_thread + libiomp5 (same as the previous prebuilt MAGMA).
# magma_sparse is built too so that the stock install rules work without patching.
cmake -S "magma-${MAGMA_VERSION}" -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DGPU_TARGET="sm_35 sm_50" \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_FORTRAN=OFF \
    -DBLA_VENDOR=Intel10_64lp
cmake --build build --target magma magma_sparse -j"$JOBS"
cmake --install build

cd / && rm -rf "$WORK"
ldd "$PREFIX/lib/libmagma.so" | grep -E 'mkl|omp'
