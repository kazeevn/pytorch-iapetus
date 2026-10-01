# Builder image: CUDA 11.8 toolchain for Kepler (sm_35) + Maxwell (sm_50) on driver 470.
#
# Follows CONVENTIONS.md:
#   (c) latest Ubuntu 22.04 packages (dist-upgrade)
#   (d) GCC 12 for host code (C++20 allowed); nvcc uses GCC 11 (C++17 max), because
#       CUDA 11.8's front end cannot parse GCC 12's libstdc++ / x86 intrinsics headers
#   (e) -march=native, sm_35 + sm_50 only, Intel oneMKL + oneDNN, OpenMPI
#
# Build from the repository root:  scripts/build_images.sh builder

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS toolchain

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# NVIDIA's apt-mark holds (cuBLAS, NCCL) keep CUDA 11.8 builds during dist-upgrade.
# The packaged NCCL is replaced by our patched build (third_party/nccl), and
# cuda-compat is removed: forward compatibility is unsupported on Kepler/Maxwell.
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential gcc-11 g++-11 gcc-12 g++-12 \
        git curl ca-certificates gnupg pkg-config ninja-build \
        libopenblas-dev liblapack-dev \
        libopenmpi-dev openmpi-bin && \
    apt-get purge -y --allow-change-held-packages libnccl2 libnccl-dev cuda-compat-11-8 && \
    rm -rf /usr/local/cuda/compat /etc/ld.so.conf.d/*cuda-compat* /var/lib/apt/lists/* && \
    ldconfig && \
    test -x /usr/local/cuda/bin/nvcc

# GCC 12 is the default gcc/g++/cc/c++; GCC 11 stays installed for nvcc.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11 && \
    gcc --version | head -n1 | grep -q ' 12\.' && \
    c++ --version | head -n1 | grep -q ' 12\.'

# CUDAHOSTCXX: nvcc host compiler for every CMake project (incl. find_package(Torch) consumers).
# NVCC_APPEND_FLAGS: applies -march=native to the host side of every nvcc invocation.
# CUDAARCHS / TORCH_CUDA_ARCH_LIST: default CUDA targets for CMake / torch.utils.cpp_extension.
ENV CUDAHOSTCXX=/usr/bin/g++-11 \
    CFLAGS="-march=native" \
    CXXFLAGS="-march=native" \
    NVCC_APPEND_FLAGS="-Xcompiler=-march=native" \
    CUDAARCHS="35;50" \
    TORCH_CUDA_ARCH_LIST="3.5;5.0"

# Intel oneAPI: oneMKL (BLAS/LAPACK) and oneDNN
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends intel-oneapi-mkl-devel intel-oneapi-dnnl-devel && \
    rm -rf /var/lib/apt/lists/* && \
    printf '%s\n' \
        /opt/intel/oneapi/mkl/latest/lib/intel64 \
        /opt/intel/oneapi/compiler/latest/lib/intel64 \
        /opt/intel/oneapi/compiler/latest/lib \
        /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin \
        /opt/intel/oneapi/dnnl/latest/lib \
        /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8 \
        > /etc/ld.so.conf.d/intel-oneapi.conf && \
    ldconfig

ENV MKLROOT=/opt/intel/oneapi/mkl/latest \
    ONEAPI_ROOT=/opt/intel/oneapi

# Python 3.12 virtual environment managed by uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.12 && \
    uv venv --seed /opt/venv312 --python 3.12 && \
    uv pip install --python /opt/venv312 \
        numpy pyyaml setuptools wheel build scikit-build-core packaging six \
        typing-extensions sympy jinja2 filelock networkx cmake ninja
ENV PATH="/opt/venv312/bin:${PATH}"

RUN git config --system --add safe.directory '*'

ARG JOBS=4

# ---- NCCL 2.23.4 with the Kepler/Maxwell memory-pool fallback (kazeevn/nccl) ----
FROM toolchain AS nccl
ARG JOBS
COPY third_party/nccl /src/nccl
COPY scripts/build_nccl.sh /tmp/build_nccl.sh
RUN JOBS=${JOBS} /tmp/build_nccl.sh /src/nccl /opt/nccl

# ---- MAGMA 2.10.0 (unpatched upstream release) against oneMKL ----
FROM toolchain AS magma
ARG JOBS
COPY scripts/build_magma.sh /tmp/build_magma.sh
RUN JOBS=${JOBS} /tmp/build_magma.sh /opt/magma

# ---- Final builder image ----
FROM toolchain AS builder
COPY --from=nccl /opt/nccl /opt/nccl
COPY --from=magma /opt/magma /opt/magma
RUN echo /opt/nccl/lib > /etc/ld.so.conf.d/nccl.conf && \
    echo /opt/magma/lib > /etc/ld.so.conf.d/magma.conf && \
    ln -sf /opt/magma /usr/local/magma && \
    ldconfig
ENV NCCL_ROOT=/opt/nccl \
    MAGMA_HOME=/opt/magma

WORKDIR /workspace
