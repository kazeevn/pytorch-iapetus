#!/usr/bin/env bash
# Build the PyTorch wheel from third_party/pytorch (kazeevn/pytorch_kepler) into dist/.
#
# Run from the host: re-executes itself inside the builder image with the repository
# mounted at /workspace, as the invoking user (so build products are not root-owned).
set -euo pipefail

BUILDER_TAG="${BUILDER_TAG:-iapetus/builder:cuda11.8-py312}"

if [ ! -f /.dockerenv ]; then
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    exec docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$REPO:/workspace" -w /workspace \
        "$BUILDER_TAG" scripts/build_pytorch.sh "$@"
fi

echo "=== Environment Info ==="
python --version
gcc --version | head -n 1
"$CUDAHOSTCXX" --version | head -n 1
nvcc --version | tail -n 2

echo "=== CUDA Driver Test ==="
python -c "import ctypes; cuda = ctypes.CDLL('libcuda.so'); v = ctypes.c_int(); res = cuda.cuDriverGetVersion(ctypes.byref(v)); print('Driver CUDA version:', v.value if res == 0 else 'Error')" || true

echo "=== Starting PyTorch Build ==="
cd /workspace/third_party/pytorch

set +u
source /opt/intel/oneapi/setvars.sh 2>/dev/null || true
set -u

# CPU backends: Intel oneMKL (BLAS/LAPACK), oneDNN, OpenMP
export ONEAPI_ROOT="/opt/intel/oneapi"
export MKLROOT="/opt/intel/oneapi/mkl/latest"
export INTEL_MKL_DIR="/opt/intel/oneapi/mkl/latest"
export BLAS="MKL"
export USE_LAPACK=1
export MKL_THREADING="OMP"
export CMAKE_PREFIX_PATH="/usr/local/cuda;/opt/intel/oneapi/mkl/latest;/opt/intel/oneapi/dnnl/latest;/opt/magma;/opt/nccl;${CMAKE_PREFIX_PATH:-}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/opt/venv312/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/compiler/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/dnnl/latest/lib:/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8:/opt/magma/lib:/opt/nccl/lib:${LD_LIBRARY_PATH:-}"

# Host code: GCC 12, C++20 (PyTorch default); CUDA code: nvcc + GCC 11 ($CUDAHOSTCXX), C++17
export CMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX"
export CFLAGS="-march=native"
export CXXFLAGS="-march=native"

# Kepler sm_35 + Maxwell sm_50 only
export TORCH_CUDA_ARCH_LIST="3.5;5.0"
export PYTORCH_BUILD_VERSION="2.14.0"
export PYTORCH_BUILD_NUMBER="2"

export MAX_JOBS="${MAX_JOBS:-4}"
export USE_CUDA=1
export USE_CUDNN=0
export BUILD_TEST=0
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_C10D_GLOO=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export NCCL_ROOT="/opt/nccl"
export USE_C10D_NCCL=1
export USE_MPI=1
export USE_C10D_MPI=1
export USE_UCC=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_CUSPARSELT=0
export USE_CUDSS=0
export USE_CUFILE=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_TENSORRT=0
export USE_ROCM=0
export USE_XPU=0
export USE_AITER=0
export USE_MSLK=0
export USE_KLEIDIAI=0
export USE_MKLDNN=1
export USE_ITT=1
export USE_NVFUSER=0
export USE_LLVM=0
export USE_KINETO=0
export USE_MAGMA=1
export MAGMA_HOME="/opt/magma"
export USE_TENSORPIPE=0
export CMAKE_BUILD_TYPE=Release
export REL_WITH_DEB_INFO=0

mkdir -p /workspace/dist
pip wheel --no-build-isolation --no-deps -w /workspace/dist/ -v .

echo "=== Build Complete ==="
ls -lh /workspace/dist/
