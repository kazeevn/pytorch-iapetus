#!/bin/bash
set -euo pipefail

# build_metatomic_torch.sh:
# Compiles metatensor-torch and metatomic-torch from source against the installed PyTorch
# and installs the metatomic C++ library (/usr/local/lib/libmetatomic_torch.so,
# /usr/local/include/metatomic/) plus the Python packages into /opt/venv312.
# Both link libtorch, so they are built here (C++20 host code, GCC 12, -march=native)
# rather than taken from PyPI wheels built against upstream torch.

VENV_PYTHON="${VENV_PYTHON:-/opt/venv312/bin/python}"
VENV_PIP="${VENV_PIP:-/opt/venv312/bin/pip}"
BUILD_DIR="${BUILD_DIR:-/tmp/metatomic_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

METATENSOR_TORCH_VERSION="0.10.6"
METATOMIC_TORCH_VERSION="0.1.18"
METATOMIC_TORCH_SHA256="bcd66d286239c9cf5da3e69e8f36fec41a80770708e3590e18b83e7a4461faa3"
METATOMIC_TORCH_URL="https://files.pythonhosted.org/packages/84/a6/7be10aeea955856ad91860549c42241c1bc51e951cb908c88bcee8590e14/metatomic_torch-${METATOMIC_TORCH_VERSION}.tar.gz"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

echo "=== Building metatensor-torch ${METATENSOR_TORCH_VERSION} and metatomic-torch ${METATOMIC_TORCH_VERSION} ==="

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

# 1. metatensor-torch from the PyPI sdist (dependencies come from docker/requirements.txt)
"$VENV_PIP" install --no-cache-dir --no-deps --no-build-isolation \
    --no-binary metatensor-torch "metatensor-torch==${METATENSOR_TORCH_VERSION}"

# 2. metatomic-torch sdist (it embeds the C++ library tarball)
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
curl -fsSLO "$METATOMIC_TORCH_URL"
echo "${METATOMIC_TORCH_SHA256}  metatomic_torch-${METATOMIC_TORCH_VERSION}.tar.gz" | sha256sum -c -
tar -xzf "metatomic_torch-${METATOMIC_TORCH_VERSION}.tar.gz"
PY_SRC="$BUILD_DIR/metatomic_torch-${METATOMIC_TORCH_VERSION}"
tar -xzf "$PY_SRC/metatomic-torch-cxx-${METATOMIC_TORCH_VERSION}.tar.gz"
CXX_SRC="$BUILD_DIR/metatomic-torch-cxx-${METATOMIC_TORCH_VERSION}"

PREFIX_PATH="$("$VENV_PYTHON" -c 'import torch, metatensor, metatensor.torch; print(f"{torch.utils.cmake_prefix_path};{metatensor.utils.cmake_prefix_path};{metatensor.torch.utils.cmake_prefix_path}")')"
echo "CMake prefix paths: $PREFIX_PATH"

# 3. C++ library + tests
cmake -S "$CXX_SRC" -B "$BUILD_DIR/cxx" \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMETATOMIC_TORCH_TESTS=ON
cmake --build "$BUILD_DIR/cxx" --target install
(cd "$BUILD_DIR/cxx" && ctest --output-on-failure)
ldconfig

# 4. Python bindings linked to the C++ library installed above
cd "$PY_SRC"
METATOMIC_TORCH_PYTHON_USE_EXTERNAL_LIB=ON \
CMAKE_PREFIX_PATH="$INSTALL_PREFIX;$PREFIX_PATH" \
"$VENV_PIP" install . --no-build-isolation --no-deps --no-cache-dir

cd / && rm -rf "$BUILD_DIR"

echo "=== Metatomic-torch build and installation complete ==="
"$VENV_PYTHON" -c "
import torch
import metatensor.torch
import metatomic.torch
print('PyTorch Version          :', torch.__version__)
print('Metatensor Torch Version :', metatensor.torch.__version__)
print('Metatomic Torch Version  :', metatomic.torch.__version__)
print('Metatomic C++ Op Version :', torch.ops.metatomic.version())
"
