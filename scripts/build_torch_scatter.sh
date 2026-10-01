#!/bin/bash
# Build and install torch_scatter from source against the installed PyTorch.
# Usage: build_torch_scatter.sh [source-dir]
set -euo pipefail

SRC="${1:-/workspace/third_party/pytorch_scatter}"
VENV_PYTHON="${VENV_PYTHON:-/opt/venv312/bin/python}"
VENV_PIP="${VENV_PIP:-/opt/venv312/bin/pip}"

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Building torch_scatter from $SRC ==="
"$VENV_PIP" install --no-cache-dir --no-deps --no-build-isolation "$SRC"

echo "=== Verifying torch_scatter installation ==="
"$VENV_PYTHON" -c "
import torch
import torch_scatter
print('PyTorch Version       :', torch.__version__)
print('torch_scatter Version :', torch_scatter.__version__)
"
