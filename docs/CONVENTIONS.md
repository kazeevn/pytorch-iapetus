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
| `docs/` | Documentation of the current code: conventions, usage, multi-user setup |
| `docs/archive/` | Historical documents: bring-up notes, resolved bugs, research |
| `third_party/` | Git submodules: our forks and pinned upstream sources |
| `dist/` | Built wheels (git-ignored; published as GitHub Release assets) |
| `archive/` | Legacy Dockerfiles and experiment scripts (tracked) and large artifacts (git-ignored, local only) |

Do not commit build products, wheels, tarballs or vendored source trees.

**Documentation.** `docs/*.md` (and `README.md`) describe the code as it is now. A change that makes a document
wrong updates it in the same commit. Anything that only describes the past (how a bug was found, superseded
setups, research that led to a decision) moves to `docs/archive/`. Archived documents are not kept in sync with
the code, and their links and paths may be stale.

**Releases.** Versions mirror the official `pytorch/pytorch` tags with `iapetus` as the variant:
`<PyTorch>-cuda<CUDA>-cudnn<cuDNN>-iapetus-r<revision>` (for example,
`2.14.0-cuda11.8-cudnn8.7-iapetus-r3`). Bump the revision for any change to the published image;
restart at `r1` with a new PyTorch version. `scripts/release.sh <revision>` pushes the runtime image
to `ghcr.io/kazeevn/pytorch` as `:<version>` (immutable),
`:<PyTorch>-cuda<CUDA>-cudnn<cuDNN>-iapetus` and `:latest`, tags the commit `v<version>`, and creates a GitHub Release with the image digest, component
versions and the torch wheel. Release only images built by `scripts/build_images.sh runtime` from a clean,
pushed commit (the image's `org.opencontainers.image.revision` label must match `HEAD`), so every release can be
reproduced from its tag. Local builds keep the `iapetus/*` tags.

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
| PyTorch 2.14.1 | [kazeevn/pytorch_kepler](https://github.com/kazeevn/pytorch_kepler/tree/v2.14.1-kepler) → `third_party/pytorch` | branch `v2.14.1-kepler` | yes |
| cuDNN frontend 0.9.2 | NVIDIA upstream → `third_party/pytorch/third_party/cudnn_frontend` | tag `v0.9.2` (commit `12f35fa2`) | no (upstream submodule pin in PyTorch fork) |
| NCCL 2.23.4 | [kazeevn/nccl](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler) → `third_party/nccl` | branch `v2.23.4-kepler` | yes |
| torch_scatter 2.1.2 | upstream `rusty1s/pytorch_scatter` → `third_party/pytorch_scatter` (`scripts/build_torch_scatter.sh`) | commit `f514c10` | no (unpatched) |
| torch_sparse 0.6.18 | upstream `rusty1s/pytorch_sparse` → `third_party/pytorch_sparse` (`scripts/build_torch_sparse.sh`) | tag `0.6.18` (commit `7d22892`) | no (unpatched) |
| OpenEquivariance 0.7.0 | [kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler) → `third_party/openequivariance` (`scripts/build_openequivariance.sh`) | branch `v0.7.0-kepler` | yes |
| MAGMA 2.10.0 | ICL release tarball (`scripts/build_magma.sh`) | sha256 | no |
| metatomic-torch 0.1.18 | PyPI sdist (`scripts/build_metatomic_torch.sh`) | sha256 | no |
| metatensor-torch 0.10.6 | PyPI sdist (`scripts/build_metatomic_torch.sh`) | version | no |
| Other Python packages | PyPI wheels (`docker/requirements.txt`) | exact versions | no |

## (c) Base image and system packages

- Base image: `nvidia/cuda:11.8.0-devel-ubuntu22.04`. CUDA 11.8 is the last toolkit that supports `sm_35`.
- Install and hold `libcudnn8` and `libcudnn8-dev` at `8.7.0.84-1+cuda11.8` from NVIDIA's apt
  repository. [NVIDIA's 8.7 support matrix](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/support-matrix/index.html)
  includes `sm_35`; [the 8.8 matrix](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-880/support-matrix/index.html)
  requires `sm_50` or later.
- Pin PyTorch's nested `cudnn_frontend` submodule to upstream `v0.9.2`: its headers compile with CUDA 11.8
  and cuDNN 8.7. The newer frontend bundled by PyTorch 2.14 references CUDA 12 driver types and later cuDNN APIs.
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
| `dist/torch-2.14.0.post2` wheel | Built **before** these conventions (GCC 11 host, old builder, no cuDNN). Retained as the existing artifact |
| `dist/torch-2.14.0.post3` wheel | Built on 2026-10-04 with CUDA 11.8, cuDNN 8.7, and frontend 0.9.2; `libtorch_cuda.so` links to `libcudnn.so.8` |
| `dist/torch-2.14.1` wheel | `scripts/build_pytorch.sh` builds it from branch `v2.14.1-kepler`: the three `v2.14.0-kepler` patches rebased unchanged onto upstream `v2.14.1` (whose changes are MPS fixes, CI and the version bump) |
| cuDNN 8.7 | Builder installs and holds `libcudnn8` and `libcudnn8-dev` at `8.7.0.84-1+cuda11.8`; PyTorch has `USE_CUDNN=1`, and the runtime image reports version 8700. Forward and backward convolution passed on both Kepler K20c GPUs and the Maxwell GTX 750 Ti |
| cuDNN frontend 0.9.2 | PyTorch fork commit `e9fe9928` pins upstream frontend `12f35fa2`; frontend header and PyTorch's convolution, MHA, quantized convolution/linear, and CUDA hooks translation units compile with CUDA 11.8 and cuDNN 8.7 |
| `torch.utils.cpp_extension` | Patched in `kazeevn/pytorch_kepler` (commit `55006b9`): defaults nvcc to `-std=c++17` on CUDA < 12 and uses `$CUDAHOSTCXX` for `-ccbin` |
| `torch/csrc/autograd/edge.h` | Patched in `kazeevn/pytorch_kepler` (commit `55006b9`): uses `static_cast<bool>(function)` and adds `operator!=`/`==` with `nullptr` in `intrusive_ptr.h` for C++17 compatibility |
| `torch_scatter` 2.1.2 | Compiled from submodule `third_party/pytorch_scatter` for `sm_35` + `sm_50` against our PyTorch wheel (`scripts/build_torch_scatter.sh`) and included in the runtime image |
| `torch_sparse` 0.6.18 | Compiled from submodule `third_party/pytorch_sparse` for `sm_35` + `sm_50` against our PyTorch wheel (`scripts/build_torch_sparse.sh`) and included in the runtime image |
| OpenEquivariance 0.7.0 | Patched in [kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler) (branch `v0.7.0-kepler`): CUDA 11 Driver API fallback (`CUmodule`), NVRTC 11.8 flag compatibility, multi-device/multi-architecture kernel caching, and installed LibTorch C ABI detection. Compiled from `third_party/openequivariance` and included in the runtime image |
| Images (`iapetus/builder`, `iapetus/pytorch`) | Current `cuda11.8-cudnn8.7-py312` builder and runtime images built on 2026-10-04; runtime imports all source extensions and passes cuDNN convolution on both K20c GPUs and the GTX 750 Ti |
| Release `2.14.0-cuda11.8-iapetus-r1` (`ghcr.io/kazeevn/pytorch`) | `iapetus/pytorch` of 2026-10-01 plus source/description/title labels, released with `FORCE=1`: it predates the `revision` label, so it is not tied to a commit by label. It contains the pre-conventions torch wheel listed above |
| Release `2.14.0-cuda11.8-iapetus-r3` (`ghcr.io/kazeevn/pytorch`) | cuDNN 8.7 runtime with `post3` wheel, built from commit `e4d54db` and retained under its original tag |
| Release `2.14.0-cuda11.8-cudnn8.7-iapetus-r3` (`ghcr.io/kazeevn/pytorch`) | Current cuDNN 8.7 release with the official-style tag, `post3` wheel, and a revision label matching its Git tag |
