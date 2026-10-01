#!/bin/bash
# Build and install OpenEquivariance from source against the installed PyTorch.
# Usage: build_openequivariance.sh [source-dir]
set -euo pipefail

SRC="${1:-/workspace/third_party/openequivariance}"
VENV_PYTHON="${VENV_PYTHON:-/opt/venv312/bin/python}"
VENV_PIP="${VENV_PIP:-/opt/venv312/bin/pip}"

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [ -f "$SRC/openequivariance/pyproject.toml" ]; then
    SRC_DIR="$SRC/openequivariance"
else
    SRC_DIR="$SRC"
fi

echo "=== Building and installing OpenEquivariance from $SRC_DIR ==="
"$VENV_PIP" install --no-cache-dir --no-deps --no-build-isolation "$SRC_DIR"

echo "=== Verifying OpenEquivariance installation ==="
"$VENV_PYTHON" -c "
import torch
import openequivariance as oeq
print('PyTorch Version            :', torch.__version__)
print('OpenEquivariance Version   :', oeq.__version__)
print('Stable extension SO path   :', oeq.torch_ext_so_path())
assert oeq.torch_ext_so_path() is not None, 'Stable extension SO not found!'
"
