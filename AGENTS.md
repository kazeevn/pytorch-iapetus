# Agent instructions

This repository builds PyTorch 2.14 and its ecosystem from source for Kepler `sm_35` + Maxwell `sm_50`
GPUs on driver 470 (CUDA 11.8). **Follow [`CONVENTIONS.md`](CONVENTIONS.md).** In short:

- **Sources:** if you patch a library, fork it to `github.com/kazeevn`, commit on a `<upstream-tag>-kepler`
  branch, and add it as a submodule in `third_party/`. Never vendor source trees, tarballs or wheels into this repo.
  Pin unpatched dependencies (upstream submodule commit, or URL + sha256).
- **Images:** base `nvidia/cuda:11.8.0-devel-ubuntu22.04` + `apt-get update && apt-get dist-upgrade -y`.
- **Compilers:** GCC 12 for host code (C++20 allowed). nvcc 11.8 uses GCC 11 (`CUDAHOSTCXX=/usr/bin/g++-11`)
  and C++17. If CUDA code needs C++20, patch it to C++17 (fork + submodule). Don't export `CC`/`CXX`:
  `torch.utils.cpp_extension` forwards `$CC` to nvcc.
- **Targets:** `-march=native`; CUDA `sm_35` + `sm_50` only; use oneMKL / oneDNN / OpenMPI / OpenMP when
  the package supports them.

Practicalities:

- Build inside the builder image (`scripts/build_images.sh`, `scripts/build_pytorch.sh`). Run containers
  as the invoking user (`--user "$(id -u):$(id -g)"` or the runtime image's entrypoint). Root-owned files
  in this checkout break later work.
- The host has 6 physical cores; keep build parallelism at about 4–6 jobs.
- Tag new images under `iapetus/*`. Don't retag or remove `iapetus/pytorch:2.14.0-cuda11.8-py312`
  without asking: other users run it.
- The public copy is `ghcr.io/kazeevn/pytorch-iapetus` (`scripts/build_images.sh push`); push only when asked.
- Update the compliance table in `CONVENTIONS.md` when you fix or introduce a deviation.
