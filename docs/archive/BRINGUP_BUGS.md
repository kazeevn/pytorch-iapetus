# PyTorch 2.14 Build Issues & Wheel Rebuild Guide

This document details the issues encountered when running deep learning and scientific workloads (such as PyTorch + ASE + ORB + PyXtal) with custom-built **PyTorch 2.14.0 (CUDA 11.8 + Python 3.12)** on Kepler (`sm_35`, Tesla K20c) and Maxwell (`sm_50`, GTX 750 Ti) under NVIDIA Driver `470.256.02` (CUDA 11.4), along with their resolutions.

---

## Status Summary

| Issue | Description | Status in `pytorch:2.14.0-cuda11.8-py312` |
| :--- | :--- | :--- |
| **1** | Missing LAPACK in CPU build | **RESOLVED** — Built with Intel oneAPI MKL (`BLAS=MKL`, `USE_LAPACK=1`, `USE_MKLDNN=1`, OpenMP) |
| **2** | NVRTC JIT failure (`invalid value for --std (-std)`) | **RESOLVED** — Dynamic C++ NVRTC patch selecting `--std=c++17` on CUDA < 12 + `USE_NVFUSER=0` |
| **3** | `torch.linalg.det` execution on Kepler (`sm_35`) | **RESOLVED** — Native C++ analytic closed-form fast path for 2x2 and 3x3 matrices with full autograd |
| **4** | NVIDIA Warp driver 12 requirement | **DOCUMENTED WORKAROUND** — CPU graph construction + GPU forward/backward evaluation |

---

## 1. Missing LAPACK in PyTorch CPU Build

### Status: RESOLVED
Fixed in `torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl` and Docker image `pytorch:2.14.0-cuda11.8-py312`.

### Symptom
When calling linear algebra routines on CPU tensors (such as `torch.linalg.solve`, `torch.linalg.lu_factor`, or `torch.linalg.inv` during crystal coordinate conversion or ASE optimizations), PyTorch crashed with:
```
RuntimeError: LinAlgError: torch.linalg.solve: LAPACK library not found in compilation
```
or
```
RuntimeError: Calling torch.linalg.lu_factor on a CPU tensor requires compiling PyTorch with LAPACK. Please use PyTorch built with LAPACK support.
```

### Root Cause
In previous build images, no BLAS or LAPACK development packages were installed. CMake configured PyTorch without an active LAPACK provider (`USE_LAPACK=0` / `BLAS=None`), generating placeholder stubs that raise runtime exceptions whenever CPU LAPACK routines (`dgesv`, `dgetrf`, etc.) are invoked.

### Solution & Fix Applied
1. **Intel oneAPI MKL & oneDNN via APT:**
   In [`Dockerfile.builder.py312`](Dockerfile.builder.py312), configured the official Intel APT repository keyring and installed Intel oneAPI development packages:
   ```dockerfile
   RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
       | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
       echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
       | tee /etc/apt/sources.list.d/oneAPI.list && \
       apt-get update && apt-get install -y --no-install-recommends \
           intel-oneapi-mkl-devel \
           intel-oneapi-dnnl-devel \
           libopenblas-dev \
           liblapack-dev \
           libopenmpi-dev \
           openmpi-bin \
       && rm -rf /var/lib/apt/lists/*
   ```
2. **Dynamic Linker Configuration:**
   Added `/opt/intel/oneapi/mkl/latest/lib/intel64` and `/opt/intel/oneapi/dnnl/latest/lib` to `/etc/ld.so.conf.d/intel-oneapi.conf` and ran `ldconfig`.
3. **Build Flags in [`build_pytorch_2.14.sh`](build_pytorch_2.14.sh):**
   ```bash
   source /opt/intel/oneapi/setvars.sh
   export BLAS="MKL"
   export USE_LAPACK=1
   export MKL_THREADING="OMP"
   export USE_MKLDNN=1
   export USE_ITT=1
   export MKL_ROOT="/opt/intel/oneapi/mkl/latest"
   export DNNL_ROOT="/opt/intel/oneapi/dnnl/latest"
   ```

### Verification
- `torch.backends.mkl.is_available() == True`
- `torch.backends.mkldnn.is_available() == True`
- CPU LAPACK operations (`torch.linalg.solve`, `torch.linalg.lu_factor`, `torch.linalg.inv`, `cholesky`, `eigh`) verified with machine precision error (~$10^{-16}$).

### Fallback Runtime Workaround (for unpatched wheels only)
Monkey-patch `torch.linalg.solve` and `torch.linalg.inv` to delegate CPU tensors to NumPy LAPACK:
```python
import numpy as np
import torch

_orig_solve = torch.linalg.solve
def safe_solve(A, B, *args, **kwargs):
    if not A.is_cuda:
        res = np.linalg.solve(A.detach().numpy(), B.detach().numpy())
        return torch.from_numpy(res).to(dtype=B.dtype)
    return _orig_solve(A, B, *args, **kwargs)
torch.linalg.solve = safe_solve

_orig_inv = torch.linalg.inv
def safe_inv(A, *args, **kwargs):
    if not A.is_cuda:
        res = np.linalg.inv(A.detach().numpy())
        return torch.from_numpy(res).to(dtype=A.dtype)
    return _orig_inv(A, *args, **kwargs)
torch.linalg.inv = safe_inv
```

---

## 2. CUDA NVRTC JIT Compilation Failure on Driver 470 (CUDA 11.4)

### Status: RESOLVED
Fixed in `torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl` and Docker image `pytorch:2.14.0-cuda11.8-py312`.

### Symptom
When PyTorch executed operations triggering runtime JIT compilation on GPU (e.g. `TorchScript`, Tensor Expression fusers, `nvfuser`, or CUDA pointwise fusers), execution crashed with:
```
nvrtc: error: invalid value for --std (-std)
```

### Root Cause
The wheel was compiled with CUDA Toolkit 11.8 (`nvcc 11.8`), but the host system operates on NVIDIA Driver `470.256.02` (providing CUDA Driver 11.4 / NVRTC 11.4). Upstream PyTorch hardcoded `--std=c++20` into NVRTC compilation options. However, NVRTC only introduced `--std=c++20` support in CUDA 12.0; CUDA 11.x NVRTC rejects `--std=c++20` as an invalid argument.

### Solution & Fix Applied
1. **Dynamic NVRTC Version Checks in PyTorch Source:**
   - [`aten/src/ATen/native/cuda/jit_utils.cpp`](pytorch-v2.14.0/aten/src/ATen/native/cuda/jit_utils.cpp):
     Query NVRTC version at runtime and use `--std=c++17` for NVRTC < 12:
     ```cpp
     int nvrtc_major = 0, nvrtc_minor = 0;
     AT_CUDA_NVRTC_CHECK(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
     const char* std_flag = (nvrtc_major >= 12) ? "--std=c++20" : "--std=c++17";
     ```
   - [`torch/csrc/jit/tensorexpr/cuda_codegen.cpp`](pytorch-v2.14.0/torch/csrc/jit/tensorexpr/cuda_codegen.cpp):
     Dynamically select `--std=c++17` when `nvrtc_major < 12`.
   - [`torch/csrc/jit/codegen/fuser/cuda/fused_kernel.cpp`](pytorch-v2.14.0/torch/csrc/jit/codegen/fuser/cuda/fused_kernel.cpp):
     Dynamically select `--std=c++17` when `nvrtc_major < 12`.
2. **Build Configuration:**
   Disabled `USE_NVFUSER=0` and `USE_LLVM=0` in `build_pytorch_2.14.sh` since `nvfuser` is designed for newer architectures (`sm_70`+) and requires newer CUDA toolchains.

### Verification
Runtime TorchScript and TensorExpr JIT kernels compile and execute without error on `cuda:0` (Kepler `sm_35`), `cuda:1` (Kepler `sm_35`), and `cuda:2` (Maxwell `sm_50`).

### Fallback Runtime Workaround (for unpatched wheels only)
Disable runtime JIT fusers before executing GPU operations:
```python
import torch

torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
```

---

## 3. `torch.linalg.det` JIT Execution on Kepler (`sm_35`)

### Status: RESOLVED
Fixed in `torch-2.14.0.post2-cp312-cp312-linux_x86_64.whl` and Docker image `pytorch:2.14.0-cuda11.8-py312`.

### Symptom
Calling `torch.linalg.det` on 3x3 matrices (such as crystal unit cell metric tensors or strain tensors) on Kepler GPUs failed due to NVRTC compilation paths or missing batched LU cuSOLVER kernels on legacy architectures.

### Solution & Fix Applied
Added native closed-form analytic fast-paths for 2x2 and 3x3 matrices in [`aten/src/ATen/native/LinearAlgebra.cpp`](pytorch-v2.14.0/aten/src/ATen/native/LinearAlgebra.cpp):
```cpp
// Fast path for 2x2 and 3x3 matrices (avoids cuSOLVER/NVRTC failures on Kepler sm_35)
if (A.size(-1) == 2 && A.size(-2) == 2) {
    auto a00 = A.select(-2, 0).select(-1, 0);
    auto a01 = A.select(-2, 0).select(-1, 1);
    auto a10 = A.select(-2, 1).select(-1, 0);
    auto a11 = A.select(-2, 1).select(-1, 1);
    return a00 * a11 - a01 * a10;
}
if (A.size(-1) == 3 && A.size(-2) == 3) {
    auto a00 = A.select(-2, 0).select(-1, 0);
    auto a01 = A.select(-2, 0).select(-1, 1);
    auto a02 = A.select(-2, 0).select(-1, 2);
    auto a10 = A.select(-2, 1).select(-1, 0);
    auto a11 = A.select(-2, 1).select(-1, 1);
    auto a12 = A.select(-2, 1).select(-1, 2);
    auto a20 = A.select(-2, 2).select(-1, 0);
    auto a21 = A.select(-2, 2).select(-1, 1);
    auto a22 = A.select(-2, 2).select(-1, 2);
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}
```
This closed-form analytic implementation:
- Executes entirely via basic elementwise tensor arithmetic, bypassing cuSOLVER LU factorization entirely.
- Fully supports autograd backward differentiation through standard PyTorch operator graph tracking.
- Works identically on CPU, Kepler (`sm_35`), and Maxwell (`sm_50`).

### Verification
- Evaluated on CPU, `cuda:0` (Kepler), `cuda:1` (Kepler), and `cuda:2` (Maxwell).
- Forward results match NumPy `np.linalg.det` within $10^{-6}$ (FP32) and $10^{-15}$ (FP64).
- Autograd backward gradients (`loss.backward()`) verified exact against analytic cofactor matrices.

### Fallback Runtime Workaround (for unpatched wheels only)
Patch `torch.linalg.det` in Python:
```python
import torch

_orig_det = torch.linalg.det
def safe_det(A, *args, **kwargs):
    if A.shape[-2:] == (3, 3):
        return (
            A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
            - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
            + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
        )
    return _orig_det(A, *args, **kwargs)
torch.linalg.det = safe_det
```

---

## 4. NVIDIA Warp (`warp-lang`) CUDA Driver 12 Requirement

### Status: RESOLVED (Native CPU-Only Build with Intel oneAPI & `-march=native`)
Eliminated CUDA 12 driver dependency and startup errors on legacy NVIDIA Driver 470 (CUDA 11.4).

### Symptom
Stock PyPI wheels of `warp-lang` (1.17.0) are compiled with `WP_ENABLE_CUDA=1` against CUDA Toolkit 12.9, hardcoding `#define WP_CUDA_DRIVER_VERSION 12000`. When imported or initialized (`wp.init()`) by packages like `nvalchemiops` (used by `orb-models` for periodic boundary condition neighbor finding), Warp emits on `stderr`:
```
Warp CUDA error: Warp requires CUDA driver 12.0 or higher, but the current driver only supports CUDA 11.4
```
Even when user code explicitly targets `device="cpu"`, this error was previously printed during initialization.

### Resolution
1. **Source Compilation of CPU-Only `warp.so`**:
   - Recompiled Warp's native C++ source files with `-DWP_ENABLE_CUDA=0`, `-DWP_ENABLE_MATHDX=0`, `-DWP_DISABLE_CUBQL=1`, `-DWP_ENABLE_DEBUG=0`, and `-D_GLIBCXX_USE_CXX11_ABI=0`.
   - Optimized for the host architecture with `-O3 -march=native -mtune=native` and OpenMP multi-threading (`-fopenmp`), linking against Intel oneAPI (MKL / TBB) headers.
   - Reduced `warp.so` library size from 356 MB to 2.06 MB (a ~99.4% reduction).
2. **Repackaged Custom Wheel**:
   - Saved at `archive/dist-warp/warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl` (33.6 MB down from 164 MB).
   - Fully valid wheel satisfying downstream dependencies requiring `warp-lang`.
3. **Reproducible Build Script & Assets**:
   - Build script: [`build_warp_cpu.sh`](file:///home/kna/pytorch-research/build_warp_cpu.sh).
   - Vendored NanoVDB headers: [`archive/nanovdb_include`](file:///home/kna/pytorch-research/archive/nanovdb_include).
4. **Integration**:
   - Committed directly into the Docker runtime image `wyckoff-study:universal-oneapi`.
   - Verified clean initialization (`CUDA not enabled in this build; Devices: "cpu" : "x86_64"`) with zero CUDA errors or warnings.
   - Tested and verified end-to-end atomic graph generation and neighbor search via `orb-models` / `nvalchemiops` using `device="cpu"`.

