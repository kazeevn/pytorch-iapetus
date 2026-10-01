# Known Issues and Workarounds: PyTorch 2.14 (CUDA 11.8 / sm_35 / sm_50)

This document tracks known runtime limitations, build constraints, and recommended workarounds for the custom PyTorch 2.14.0 distribution targeting legacy NVIDIA architectures (Kepler `sm_35` / Maxwell `sm_50`) on Driver 470.256.02.

---

## 1. [RESOLVED] GPU Linear Algebra via MAGMA (`USE_MAGMA=1`)

### Status: RESOLVED (Build Post-2)

MAGMA 2.10.0 was successfully compiled from source with Intel oneAPI MKL (oneMKL 2026.1) support for dual architectures `sm_35` (Kepler) and `sm_50` (Maxwell), and integrated directly into the PyTorch 2.14.0 distribution.

- `torch.cuda.has_magma` evaluates to `True`.
- General non-symmetric eigendecomposition `torch.linalg.eig` works directly on CUDA tensors across both single and double precision (`float32`, `float64`) and batched dimensions `(B, N, N)`.
- Verification script [`tests/test_magma.py`](../../tests/test_magma.py) validates GPU eigendecomposition, matrix inversion, linear solve, Cholesky, QR, and SVD against CPU Intel oneMKL with machine precision accuracy ($\le 10^{-14}$ on `float64`).

---

### Implementation Details

1. **Custom Build of MAGMA 2.10.0** (now reproduced by [`scripts/build_magma.sh`](../../scripts/build_magma.sh) in the builder image):
   - Source: MAGMA 2.10.0 (released February 2026).
   - Build system: CMake configured with `-DGPU_TARGET="sm_35 sm_50" -DMAGMA_WITH_MKL=ON -DBUILD_SHARED_LIBS=ON`.
   - Linked against Intel oneMKL 2026.1 (`libmkl_intel_lp64.so`, `libmkl_gnu_thread.so`, `libmkl_core.so`) and OpenMP threading (`libiomp5.so`).
   - Installed to `/opt/magma` with runtime library paths registered in `/etc/ld.so.conf.d/magma.conf` and `/etc/ld.so.conf.d/intel-oneapi.conf`.
2. **FindMAGMA.cmake Fix**:
   - Added `CMAKE_REQUIRED_INCLUDES` in `cmake/Modules/FindMAGMA.cmake` pointing to `${MAGMA_INCLUDE_DIR}` and CUDA include paths so `check_prototype_definition(magma_get_sgeqrf_nb ...)` properly detects MAGMA V2 API during CMake configuration.
3. **PyTorch Build Flags**:
   - `USE_MAGMA=1`, `MAGMA_HOME=/opt/magma`, and `-DMAGMA_V2` enabled.

---

### Verification Summary

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" ghcr.io/kazeevn/pytorch-iapetus:2.14.0-cuda11.8-py312 python tests/test_magma.py
```
Output:
- `MAGMA Available : True`
- `Float32 non-symmetric eigendecomposition: error < 1.2e-06`
- `Float64 non-symmetric eigendecomposition: error < 3.2e-15`
- `Batched eigendecomposition [10, 5, 5]: error < 3.6e-15`
- `GPU MAGMA vs CPU MKL consistency: diff < 5.4e-15`

---

## 2. [RESOLVED] Missing libuv in Distributed TCPStore (`torchrun` & `tcp://` Rendezvous)

### Status: RESOLVED

### Overview & Symptoms

When running multi-GPU distributed training using `torchrun` or initializing process groups via `dist.init_process_group("...", init_method="tcp://...")`, PyTorch previously raised:
```text
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support, run with USE_LIBUV=0 to disable it.
```

### Resolution

This issue is resolved in the runtime image via a multi-layered fix:
1. **Container Environment**:
   `ENV USE_LIBUV=0` is permanently set in [`docker/runtime.Dockerfile`](../../docker/runtime.Dockerfile), `/etc/environment`, `/etc/bash.bashrc`, and [`docker/entrypoint.sh`](../../docker/entrypoint.sh). PyTorch's Python modules ([`rendezvous.py`](torch/distributed/rendezvous.py) and [`elastic`](torch/distributed/elastic/utils/distributed.py)) explicitly read `os.environ.get("USE_LIBUV")` and set `use_libuv=False` automatically.
2. **Persistent Python Hook**:
   [`docker/sitecustomize.py`](../../docker/sitecustomize.py) (installed as `/opt/venv312/lib/python3.12/site-packages/sitecustomize.py`) was updated to unconditionally ensure `kwargs["use_libuv"] = False`:
   ```python
   try:
       import torch
       if hasattr(torch, "_C") and hasattr(torch._C, "_distributed_c10d"):
           c10d = torch._C._distributed_c10d
           if hasattr(c10d, "TCPStore"):
               _orig_init = c10d.TCPStore.__init__
               def _safe_init(self, *args, **kwargs):
                   kwargs["use_libuv"] = False
                   _orig_init(self, *args, **kwargs)
               c10d.TCPStore.__init__ = _safe_init
   except Exception:
       pass
   ```
   This ensures that direct instantiations of `TCPStore` in C++/PyBind and Python modules are safely redirected to the standard legacy socket daemon.



---

## 3. [RESOLVED] NCCL communicator init fails on Kepler (`cudaMemPoolCreate`)

### Status: RESOLVED (patched NCCL 2.23.4)

NCCL 2.23 unconditionally creates a stream-ordered memory pool per communicator. Kepler `sm_35` devices
report `cudaDevAttrMemoryPoolsSupported == 0` under CUDA 11.8 / driver 470, so communicator init fails.
The fix lives in [kazeevn/nccl@v2.23.4-kepler](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler)
(submodule `third_party/nccl`):

- create `comm->memPool` only when the device supports memory pools;
- otherwise allocate persistent work buffers with `cudaMalloc`; they are released with `cudaFree`,
  which is valid for both allocation paths;
- skip `cudaMemPoolDestroy` when no pool exists.

The builder image compiles it with [`scripts/build_nccl.sh`](../../scripts/build_nccl.sh) and installs it to `/opt/nccl`.

---

## 4. [RESOLVED] CUDA 11.8 nvcc host compiler and C++ standard in `torch.utils.cpp_extension`

### Status: RESOLVED (patched `kazeevn/pytorch_kepler`)

nvcc 11.8's front end rejects GCC 12 system headers (`bits/random.h`: `__extension__ using`;
`avx512fp16intrin.h`: `_Float16`), even with `-allow-unsupported-compiler`. The builder therefore uses
GCC 12 for host code and GCC 11 for nvcc (`CUDAHOSTCXX`). `NVCC_CCBIN` is not honoured by CUDA 11.8.
Plain `nvcc` invocations without `-ccbin` use `g++` (GCC 12) and fail; pass `-ccbin "$CUDAHOSTCXX"`.

In `kazeevn/pytorch_kepler` (commit `6fb679f`):
1. `torch.utils.cpp_extension` defaults nvcc CUDA flags to `-std=c++17` on CUDA < 12.0 (CUDA 11.8 nvcc rejects `-std=c++20`).
2. `torch.utils.cpp_extension` automatically passes `-ccbin "$CUDAHOSTCXX"` if `CUDAHOSTCXX` is set.
3. `torch/csrc/autograd/edge.h` and `c10/util/intrusive_ptr.h` provide C++17 nullptr comparison compatibility for `intrusive_ptr`.
4. CUDA extensions including `torch_scatter` compile and run cleanly for both `sm_35` and `sm_50`.
