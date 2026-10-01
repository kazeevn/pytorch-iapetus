#!/usr/bin/env bash
set -e

echo "=== Environment Info ==="
python3 --version
gcc --version | head -n 1
nvcc --version | tail -n 2

echo "=== CUDA Driver Test ==="
python3 -c "import ctypes; cuda = ctypes.CDLL('libcuda.so'); v = ctypes.c_int(); res = cuda.cuDriverGetVersion(ctypes.byref(v)); print('Driver CUDA version:', v.value if res == 0 else 'Error')" || true

echo "=== Starting PyTorch Build ==="
cd /workspace/pytorch

# Configuration flags
export CMAKE_PREFIX_PATH="/usr/local/cuda"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Native host CPU optimizations
export CFLAGS="-march=native ${CFLAGS:-}"
export CXXFLAGS="-march=native ${CXXFLAGS:-}"

git config --global --add safe.directory '*' || true

# Target sm_35 for Kepler GK110 Tesla K20c
export TORCH_CUDA_ARCH_LIST="3.5"
export PYTORCH_BUILD_VERSION="2.8.0"
export PYTORCH_BUILD_NUMBER="1"

# Build optimization flags
export MAX_JOBS=10
export USE_CUDA=1
export USE_CUDNN=1
export BUILD_TEST=0
export USE_DISTRIBUTED=0
export USE_NCCL=0
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
export USE_MKLDNN=0
export USE_KINETO=0
export USE_MAGMA=0
export USE_TENSORPIPE=0
export CMAKE_BUILD_TYPE=Release
export REL_WITH_DEB_INFO=0

# Build wheel
python3 setup.py bdist_wheel

echo "=== Build Complete ==="
ls -lh dist/
