# PyTorch 2.14 (CUDA 11.8 + Python 3.12) Usage Guide

This guide documents how to use the custom-compiled **PyTorch 2.14** wheel and its accompanying Docker image. The reference machine has two Tesla K20c (Kepler `sm_35`) and one GeForce GTX 750 Ti (Maxwell `sm_50`); hardware requirements are listed in the [README](../README.md#will-it-run-on-my-machine).

The examples use the published image `ghcr.io/kazeevn/pytorch-iapetus:2.14.0`, which follows the newest release for PyTorch 2.14.0; pin a release such as `2.14.0-r1` for reproducible work (see [Releases](https://github.com/kazeevn/pytorch-iapetus/releases)). If you built the image yourself, substitute the local tag `iapetus/pytorch:2.14.0-cuda11.8-py312`.

---

## 1. Artifacts & Environment Overview

* **Wheel File:**
  `torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl` (271 MB; attached to each GitHub Release, or built into `dist/` by `scripts/build_pytorch.sh`). It needs the builder image's CUDA, NCCL, MAGMA, oneMKL and OpenMPI libraries
* **Docker Image:**
  * `ghcr.io/kazeevn/pytorch-iapetus:2.14.0` *(published copy of `iapetus/pytorch:2.14.0-cuda11.8-py312`, built from `docker/runtime.Dockerfile`; auto-detects host user UID/GID & GPU permissions)*
* **Python Runtime:** Python 3.12 (managed via `uv` in `/opt/venv312`)
* **CUDA Version:** CUDA 11.8 (Compatible with NVIDIA driver `470.256.02`)
* **Target GPU Architectures:** Dual `sm_35` (Tesla K20c) + `sm_50` (GeForce GTX 750 Ti)
* **CPU Linear Algebra:** **Intel oneAPI MKL (oneMKL 2026.1)** + **oneDNN (v3.12.0)** with LAPACK and OpenMP multi-threading enabled
* **GPU Dense Linear Algebra (MAGMA 2.10.0):** Built from source with Intel oneAPI MKL support for `sm_35` + `sm_50`. Fully integrated (`torch.cuda.has_magma == True`), enabling GPU-accelerated general non-symmetric eigendecomposition (`torch.linalg.eig`)
* **CPU Tuning:** `-march=native` on a Haswell-E (AVX2, FMA, BMI2); the image needs a Haswell-or-newer CPU
* **Compilers:** GCC 12 for host code (C++20), nvcc 11.8 with GCC 11 for CUDA code (C++17); see [`CONVENTIONS.md`](CONVENTIONS.md)
* **NVRTC / JIT:** Custom patch dynamically using `--std=c++17` on CUDA < 12.0 drivers, allowing runtime JIT and TorchScript fusers to execute on Driver 470
* **Small Matrix Determinants:** Fast-path analytic closed-form determinants for 2x2 and 3x3 matrices with full autograd differentiability
* **cuDNN:** Disabled (`USE_CUDNN=0` — cuDNN 8 drops Kepler/Maxwell support)
* **Distributed Backends Enabled:**
  * **NCCL 2.23.4** (custom-built with stream-ordered memory pool fallback for legacy GPUs)
  * **Gloo** (CPU and multi-GPU CUDA)
  * **OpenMPI 4.1.2** (CPU multi-rank collectives)
* **Container User:** the entrypoint runs commands as the owner of the mounted `/workspace` (or `HOST_UID`/`HOST_GID`), with `video` and `render` GPU group membership, so files written to mounted volumes belong to the host user. Pass `-e HOST_UID=0` for a root shell. See [`MULTIUSER.md`](MULTIUSER.md).

---

## 2. Mandatory Docker Run Flags

When running containers with this build, the following flags are required:

| Docker Flag | Reason / Impact |
| :--- | :--- |
| `--runtime=nvidia` | Injects NVIDIA GPU character devices and UVM (`/dev/nvidia-uvm`) into the container. |
| `-e NVIDIA_VISIBLE_DEVICES=all` | Exposes all physical GPUs (on the reference machine GPU 0: K20c, GPU 1: K20c, GPU 2: GTX 750 Ti). |
| `--ipc=host` *(or `--shm-size=8g`)* | **CRITICAL for Multi-GPU / NCCL:** The heterogeneous setup (Kepler $\leftrightarrow$ Maxwell) lacks direct PCIe peer-to-peer (P2P) hardware access. NCCL falls back to shared memory (`SHM/direct/direct`). Docker's default 64 MB shm triggers `SIGBUS (exit code 135)` without this flag. |

---

## 3. Method 1: Using the Ready-to-Run Docker Image (Recommended)

The image `ghcr.io/kazeevn/pytorch-iapetus:2.14.0` has PyTorch 2.14, Python 3.12, NCCL 2.23.4, and OpenMPI installed and configured in `PATH`.

### Quick Health Check
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  python -c "
import torch, torch.distributed as dist
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
print('MKL Available   :', torch.backends.mkl.is_available())
print('oneDNN Available:', torch.backends.mkldnn.is_available())
print('MAGMA Available :', torch.cuda.has_magma)
print('Device Count    :', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  [{i}] {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_capability(i)})')
print('Backends (Gloo, NCCL, MPI):', dist.is_gloo_available(), dist.is_nccl_available(), dist.is_mpi_available())
"
```

### Interactive Bash Session with Workspace Mounted
To mount the current directory into `/workspace` and open an interactive bash shell:
```bash
docker run --rm -it \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  bash
```

Inside the container, `python` points directly to Python 3.12 with PyTorch 2.14 ready to use:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Running a Python Script
```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  python my_script.py
```

### Running Multi-GPU Distributed Training (`torchrun`)
To run 3-rank distributed training across both Tesla K20c cards and the GTX 750 Ti:
```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  torchrun --nproc_per_node=3 train.py
```

### Running Multi-Process MPI Jobs
To launch 3 MPI processes (1 per GPU):
```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  mpirun --allow-run-as-root -n 3 python mpi_train.py
```

---

## 4. Method 2: Installing the Wheel into Any CUDA 11.8 / Python 3.12 Container

The wheel is not published; build it first with `scripts/build_pytorch.sh` (which also needs the locally built
`iapetus/builder` image). Then, to use a different base image or custom Docker container:

1. **Volume Mount the Wheel Directory:** Mount the repository's `dist/` into the container.
2. **Install the Wheel:**
   ```bash
   pip install /dist/torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl
   ```

### Complete Docker Command Example:
```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -v "$(pwd)/dist:/dist" \
  -v "$(pwd):/workspace" \
  iapetus/builder:cuda11.8-py312 \
  bash -c '
    /opt/venv312/bin/pip install --no-deps -q /dist/torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl
    /opt/venv312/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  '
```

---

## 5. Verification Test Suites

The test scripts in this repository's `tests/` directory (run from a clone of the repository; the runner scripts honour `IMAGE=...`) to verify hardware, single-device ops, multi-GPU distributed collectives, and MAGMA linear algebra:

### 1. MAGMA GPU Linear Algebra & Precision Suite
Validates `torch.cuda.has_magma`, non-symmetric eigendecomposition (`torch.linalg.eig`), matrix inverse, linear solve, Cholesky, QR, and SVD against Intel oneMKL with $\le 10^{-14}$ error:
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  python tests/test_magma.py
```
*(Result: All MAGMA tests passed successfully)*

### 2. Single-Device Unit Tests (All 3 GPUs)
Executes `test_cuda`, `test_cuda_multigpu`, `test_autograd`, and `test_nn` across all three cards:
```bash
tests/run_unit_tests.sh
```
*(Result: 36 Passed, 0 Failed)*

### 3. Distributed Collectives & Official PyTorch Distributed Tests
Executes multi-GPU collective tests (Gloo, NCCL, OpenMPI) as well as official PyTorch test suites from `test/distributed/`:
```bash
tests/run_distributed_tests.sh
```
### 4. Metatomic-Torch Full Verification Suite
Executes the comprehensive 7-stage test suite covering versions, C++ headers, shared libraries, CPU atomistic structures, GPU device detection, model capabilities, unit conversions, ASE calculator & serialization, and end-to-end C++ compilation with `find_package(metatomic_torch)`:
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" \
  ghcr.io/kazeevn/pytorch-iapetus:2.14.0 \
  python tests/test_metatomic.py
```
*(Result: 7/7 test stages passed successfully)*

---

## 6. Metatomic-Torch Atomistic ML Guide

The container includes custom-built **`metatomic-torch 0.1.18`** and **`metatensor-torch 0.10.6`** compiled directly against PyTorch 2.14.0.post2.

### Python Quickstart
```python
import torch
import metatensor.torch
import metatomic.torch
from metatomic.torch import System, ModelCapabilities, ModelOutput, unit_conversion_factor
from ase import Atoms
from metatomic.torch import systems_to_torch

# 1. Verify custom C++ TorchScript operator
print("Metatomic version op:", torch.ops.metatomic.version())

# 2. Convert ASE Atoms directly to Metatomic System
atoms = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
system = systems_to_torch(atoms)
print("Torch System:", system)

# 3. Unit conversions
bohr_to_angstrom = unit_conversion_factor("bohr", "Angstrom")
print("1 bohr in Angstrom:", bohr_to_angstrom)
```

### C++ Downstream Integration (CMake)
The C++ library `libmetatomic_torch.so`, C++ headers, and CMake config files are installed in `/usr/local`.

In your project's `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.22)
project(my_atomistic_sim CXX)

find_package(Torch REQUIRED)
find_package(metatensor_torch REQUIRED)
find_package(metatomic_torch REQUIRED)

add_executable(my_sim main.cpp)
target_link_libraries(my_sim PRIVATE metatomic_torch torch)
target_compile_features(my_sim PRIVATE cxx_std_17)
```

The container automatically exports the required `CMAKE_PREFIX_PATH`:
```text
/usr/local:/opt/venv312/lib/python3.12/site-packages/torch/share/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor/lib/cmake:/opt/venv312/lib/python3.12/site-packages/metatensor_torch/torch-2.14/lib/cmake
```

---

## 7. Helpful Environment Flags Reference

| Variable | Recommended Setting | Purpose |
| :--- | :--- | :--- |
| `MAGMA_HOME` | `/opt/magma` | Root path to custom MAGMA 2.10.0 headers and libraries. Preconfigured in Docker images. |
| `NCCL_IGNORE_DISABLED_P2P` | `1` | Suppresses benign warnings when NCCL detects that GPU 0/1 (Kepler) cannot P2P to GPU 2 (Maxwell) and automatically falls back to shared memory rings. |
| `NCCL_DEBUG` | `INFO` (or `WARN`) | Displays NCCL topology, channel setup, and ring transport diagnostics. |
| `OMPI_ALLOW_RUN_AS_ROOT` | `1` | Allows OpenMPI to run inside containers where root is the default user. |
| `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM` | `1` | Confirms OpenMPI execution under root. |
| `GLOO_SOCKET_IFNAME` | `lo` | Binds Gloo TCP communications to the local loopback interface. |
