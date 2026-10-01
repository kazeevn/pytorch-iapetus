# Project Conventions

Rules for building the PyTorch ecosystem from source for this machine:
Tesla K20c ×2 (Kepler `sm_35`) + GTX 750 Ti (Maxwell `sm_50`), NVIDIA driver 470 (CUDA 11.4 driver API),
Intel Core i7-5930K (Haswell-E, AVX2). Every build script and Dockerfile in this repository follows them.

---

## (a) Repository

The project root is **[kazeevn/pytorch-iapetus](https://github.com/kazeevn/pytorch-iapetus)**.
Everything needed to reproduce the images lives here, or is referenced from here by a pinned
submodule or a pinned URL plus sha256.

| Path | Contents |
| :--- | :--- |
| `docker/` | Dockerfiles (`builder`, `runtime`), entrypoint, pinned runtime `requirements.txt` |
| `scripts/` | Build scripts (`build_images.sh`, `build_pytorch.sh`, `build_nccl.sh`, `build_magma.sh`, ...) |
| `tests/` | Verification suites and their Docker runners |
| `docs/` | Usage, multi-user setup, known issues |
| `third_party/` | Git submodules: our forks and pinned upstream sources |
| `dist/` | Built wheels (git-ignored; not published, shipped inside the runtime image) |
| `archive/` | Historical notes (tracked) and large legacy artifacts (git-ignored, local only) |

Do not commit build products, wheels, tarballs or vendored source trees.

The runtime image is published as `ghcr.io/kazeevn/pytorch-iapetus:<torch>-cuda11.8-py312` (plus `:latest`)
with `scripts/build_images.sh push`. Local builds keep the `iapetus/*` tags. Publish only images built from
a clean, pushed commit of this repository, so the image can be reproduced from its sources.

## (b) Third-party sources

**Patched libraries** are forked to `github.com/kazeevn` and added as submodules under `third_party/`:

1. Fork upstream (`gh repo fork <owner>/<repo> --clone=false`).
2. Branch `<upstream-tag>-kepler` from the upstream release tag being built.
3. Make focused commits, each explaining *why* the change is needed on this hardware or toolchain.
4. Push the branch to the fork, then
   `git submodule add -b <branch> https://github.com/kazeevn/<repo>.git third_party/<name>`.
5. In the submodule checkout, `origin` is the fork and `upstream` is the original repository.

**Unpatched libraries** are pinned without forking: either an upstream submodule at a fixed
commit, or a release URL plus sha256 checked in the build script.

| Component | Source | Pin | Patched? |
| :--- | :--- | :--- | :--- |
| PyTorch 2.14.0 | [kazeevn/pytorch_kepler](https://github.com/kazeevn/pytorch_kepler/tree/v2.14.0-kepler) → `third_party/pytorch` | branch `v2.14.0-kepler` | yes |
| NCCL 2.23.4 | [kazeevn/nccl](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler) → `third_party/nccl` | branch `v2.23.4-kepler` | yes |
| torch_scatter 2.1.2 | upstream `rusty1s/pytorch_scatter` → `third_party/pytorch_scatter` (`scripts/build_torch_scatter.sh`) | commit `f514c10` | no (unpatched) |
| OpenEquivariance 0.7.0 | [kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler) → `third_party/openequivariance` (`scripts/build_openequivariance.sh`) | branch `v0.7.0-kepler` | yes |
| MAGMA 2.10.0 | ICL release tarball (`scripts/build_magma.sh`) | sha256 | no |
| metatomic-torch 0.1.18 | PyPI sdist (`scripts/build_metatomic_torch.sh`) | sha256 | no |
| metatensor-torch 0.10.6 | PyPI sdist (`scripts/build_metatomic_torch.sh`) | version | no |
| Other Python packages | PyPI wheels (`docker/requirements.txt`) | exact versions | no |

## (c) Base image and system packages

- Base image: `nvidia/cuda:11.8.0-devel-ubuntu22.04`. CUDA 11.8 is the last toolkit that supports `sm_35`.
- Every image runs `apt-get update && apt-get dist-upgrade -y`, so it uses the latest Ubuntu 22.04
  packages. NVIDIA's `apt-mark hold`s keep the CUDA 11.8 builds of cuBLAS etc.
- `cuda-compat` is purged: forward compatibility is unsupported on Kepler/Maxwell and breaks CUDA
  initialization. The packaged NCCL is purged as well and replaced by `third_party/nccl`.

## (d) Compilers and C++ standards

| Code | Compiler | Max standard |
| :--- | :--- | :--- |
| Host C/C++ | **GCC 12** (default `gcc`/`g++`/`cc`/`c++`) | C++20 |
| CUDA (`.cu`, and every header a `.cu` includes) | nvcc 11.8 with **GCC 11** as host compiler | **C++17** |

Why nvcc needs GCC 11: CUDA 11.8 officially supports only GCC ≤ 11. Its front end can't parse GCC 12
system headers even with `-allow-unsupported-compiler`. Verified errors:

- `bits/random.h(104): error: expected a declaration` (from `__extension__ using`)
- `avx512fp16intrin.h: identifier "_Float16" is undefined` (pulled in through `<immintrin.h>`)

GCC 11 and GCC 12 share the C++ ABI, and the final link uses GCC 12's libstdc++.

How the builder image enforces this:

- `update-alternatives` makes GCC 12 the default compiler.
- `CUDAHOSTCXX=/usr/bin/g++-11` sets nvcc's host compiler for every CMake project, including
  `find_package(Torch)` consumers. PyTorch's CUDA language probe fails without it.
- NCCL's Makefile honours `CUDAHOSTCXX` (patch in `kazeevn/nccl`).
- Do **not** set `CC`/`CXX` to GCC 12 in the environment. `torch.utils.cpp_extension` forwards `$CC`
  to nvcc as `-ccbin`.

If a package's CUDA code (or a header it includes from `.cu` files) needs C++20, patch it to
C++17 and fork it under rule (b). Host-only C++20 is fine.

## (e) Optimization targets

- **CPU:** compile with `-march=native`. The builder sets `CFLAGS`, `CXXFLAGS` and
  `NVCC_APPEND_FLAGS=-Xcompiler=-march=native`. Images are therefore specific to this CPU.
- **GPU:** target **only** `sm_35` and `sm_50`, using whichever knob the build system reads:
  `TORCH_CUDA_ARCH_LIST="3.5;5.0"`, `CUDAARCHS="35;50"`, NCCL `NVCC_GENCODE`, MAGMA `GPU_TARGET="sm_35 sm_50"`.
- **CPU backends:** if a package supports them, use Intel oneMKL (BLAS/LAPACK), oneDNN, OpenMPI and
  OpenMP. Examples: PyTorch (`BLAS=MKL`, `USE_MKLDNN=1`, `USE_MPI=1`) and MAGMA (`BLA_VENDOR=Intel10_64lp`).
- Anything that links libtorch (torch extensions) is compiled from source against our wheel.
  Pure-CPU binary wheels from PyPI (numpy, scipy, metatensor-core, vesin, ...) are accepted as-is.

---

## Compliance status

| Item | Status |
| :--- | :--- |
| `dist/torch-2.14.0.post2` wheel | Built **before** these conventions (GCC 11 host, old builder). Rebuild with `scripts/build_pytorch.sh` |
| `torch.utils.cpp_extension` | Patched in `kazeevn/pytorch_kepler` (commit `6fb679f`): defaults nvcc to `-std=c++17` on CUDA < 12 and uses `$CUDAHOSTCXX` for `-ccbin` |
| `torch/csrc/autograd/edge.h` | Patched in `kazeevn/pytorch_kepler` (commit `6fb679f`): uses `static_cast<bool>(function)` and adds `operator!=`/`==` with `nullptr` in `intrusive_ptr.h` for C++17 compatibility |
| `torch_scatter` 2.1.2 | Compiled from submodule `third_party/pytorch_scatter` for `sm_35` + `sm_50` against our PyTorch wheel (`scripts/build_torch_scatter.sh`) and included in the runtime image |
| OpenEquivariance 0.7.0 | Patched in [kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler) (branch `v0.7.0-kepler`): CUDA 11 Driver API fallback (`CUmodule`), NVRTC 11.8 flag compatibility, multi-device/multi-architecture kernel caching, and installed LibTorch C ABI detection. Compiled from `third_party/openequivariance` and included in the runtime image |
| Images (`iapetus/builder`, `iapetus/pytorch`) | Built from this repository on 2026-10-01. Toolchain, NCCL/MAGMA, PyTorch, metatomic stack, torch_scatter, and OpenEquivariance verified on CPU and Kepler `sm_35` + Maxwell `sm_50` GPUs |
| Published image (`ghcr.io/kazeevn/pytorch-iapetus`) | `iapetus/pytorch` of 2026-10-01 plus OCI labels from `docker/runtime.Dockerfile`. It contains the pre-conventions torch wheel listed above |
