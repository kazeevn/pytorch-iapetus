#!/usr/bin/env bash
# Build and install NCCL from the third_party/nccl submodule (kazeevn/nccl, branch v2.23.4-kepler).
# Usage: build_nccl.sh <nccl-source-dir> <install-prefix>
# Host code is compiled with $CXX (GCC 12); nvcc uses $CUDAHOSTCXX (GCC 11).
set -euo pipefail

SRC="${1:?nccl source dir}"
PREFIX="${2:?install prefix}"
JOBS="${JOBS:-4}"
BUILDDIR="${BUILDDIR:-/tmp/nccl-build}"

# src.install depends on src.build, so a single invocation builds and installs.
make -C "$SRC" -j"$JOBS" src.install \
    BUILDDIR="$BUILDDIR" \
    PREFIX="$PREFIX" \
    CUDA_HOME=/usr/local/cuda \
    CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}" \
    NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50"
rm -rf "$BUILDDIR"

ls -l "$PREFIX/lib"
