# pytorch-iapetus

PyTorch 2.14 and its ecosystem, compiled from source for legacy NVIDIA GPUs:
**Tesla K20c (Kepler `sm_35`) ×2 + GeForce GTX 750 Ti (Maxwell `sm_50`)** on NVIDIA driver `470.256.02`
(CUDA driver API 11.4), with CUDA 11.8 and Python 3.12.

**Read [`CONVENTIONS.md`](CONVENTIONS.md) before changing anything.** It defines where sources come
from (forks vs. pinned upstream), the compiler and C++-standard rules (GCC 12 / C++20 host, nvcc + GCC 11 /
C++17 device), and the optimization targets (`-march=native`, `sm_35` + `sm_50` only, oneMKL / oneDNN / OpenMPI).

---

## What's inside

- **PyTorch 2.14.0** ([kazeevn/pytorch_kepler](https://github.com/kazeevn/pytorch_kepler/tree/v2.14.0-kepler)):
  CUDA 11.8 support restored, C++17 fallbacks in CUDA-visible code, NVRTC `--std=c++17` on CUDA < 12,
  analytic 2×2/3×3 determinants.
- **CPU:** Intel oneMKL (BLAS/LAPACK, OpenMP threading), oneDNN, `-march=native` (AVX2/FMA).
- **GPU linear algebra:** MAGMA 2.10.0 against oneMKL (`torch.cuda.has_magma == True`, GPU `torch.linalg.eig`).
- **Distributed:** NCCL 2.23.4 ([kazeevn/nccl](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler),
  memory-pool fallback for Kepler/Maxwell), Gloo, OpenMPI 4.1.2.
- **Atomistic ML:** metatensor-torch 0.10.6 and metatomic-torch 0.1.18 (C++ library + Python), compiled
  against this PyTorch. torch_scatter 2.1.2 compiled from source for Kepler + Maxwell. OpenEquivariance 0.7.0
  ([kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler)) with CUDA 11 Driver API
  fallback and multi-device kernel caching. ASE, vesin, warp-lang, orb-models, pymatgen and others are pinned in [`docker/requirements.txt`](docker/requirements.txt).
- **Multi-user entrypoint:** the container runs as the owner of the mounted `/workspace`
  (see [`docs/MULTIUSER.md`](docs/MULTIUSER.md)).

## Hardware target

| Index | Device | Architecture | Compute capability | VRAM |
| :--- | :--- | :--- | :--- | :--- |
| `cuda:0` | Tesla K20c | Kepler | `sm_35` | 5 GB |
| `cuda:1` | Tesla K20c | Kepler | `sm_35` | 5 GB |
| `cuda:2` | GeForce GTX 750 Ti | Maxwell | `sm_50` | 2 GB |

## Repository layout

```text
pytorch-iapetus/
├── CONVENTIONS.md            # project rules (read first)
├── docker/
│   ├── builder.Dockerfile    # CUDA 11.8 toolchain + GCC 12/11 + oneMKL/oneDNN + OpenMPI + NCCL + MAGMA
│   ├── runtime.Dockerfile    # builder + torch wheel + atomistic-ML stack + multi-user entrypoint
│   ├── requirements.txt      # pinned runtime Python packages
│   ├── entrypoint.sh         # dynamic UID/GID + privilege drop
│   └── sitecustomize.py      # forces TCPStore(use_libuv=False)
├── scripts/
│   ├── build_images.sh       # docker build for builder / runtime
│   ├── build_pytorch.sh      # PyTorch wheel → dist/ (runs inside the builder image)
│   ├── build_nccl.sh         # NCCL from third_party/nccl (used by builder.Dockerfile)
│   ├── build_magma.sh        # MAGMA from the pinned ICL tarball (used by builder.Dockerfile)
│   ├── build_torch_scatter.sh# torch_scatter from third_party/pytorch_scatter
│   ├── build_openequivariance.sh # OpenEquivariance from third_party/openequivariance
│   ├── build_metatomic_torch.sh
│   └── build_warp_cpu.sh     # optional CPU-only warp rebuild for driver 470
├── tests/                    # test_magma / test_metatomic / test_torch_scatter / test_openequivariance / distributed runners
├── docs/                     # USAGE.md, MULTIUSER.md, BUGS.md
├── third_party/              # submodules: pytorch (fork), nccl (fork), pytorch_scatter (upstream), openequivariance (fork)
├── dist/                     # wheels (git-ignored)
└── archive/                  # history; large legacy artifacts are git-ignored
```

## Getting started

```bash
git clone --recurse-submodules=third_party/nccl https://github.com/kazeevn/pytorch-iapetus.git
cd pytorch-iapetus
git submodule update --init third_party/pytorch   # large; only needed to rebuild PyTorch
```

The PyTorch submodule's own submodules are needed only for a full PyTorch rebuild:
`git -C third_party/pytorch submodule update --init --recursive`.

### Build

```bash
scripts/build_images.sh builder    # iapetus/builder:cuda11.8-py312 (compiles NCCL + MAGMA)
scripts/build_pytorch.sh           # optional: rebuild dist/torch-*.whl inside the builder
scripts/build_images.sh runtime    # iapetus/pytorch:2.14.0-cuda11.8-py312 (needs dist/*.whl)
```

### Run

```bash
docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" iapetus/pytorch:2.14.0-cuda11.8-py312
```

> [!IMPORTANT]
> `--runtime=nvidia` and `-e NVIDIA_VISIBLE_DEVICES=all` expose the GPUs. `--ipc=host` (or `--shm-size=8g`)
> is required for NCCL: Kepler ↔ Maxwell has no P2P, and NCCL's shared-memory fallback fails with
> `SIGBUS` under Docker's default 64 MB `/dev/shm`.

### Test

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host -v "$(pwd):/workspace" \
  iapetus/pytorch:2.14.0-cuda11.8-py312 python tests/test_magma.py
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host -v "$(pwd):/workspace" \
  iapetus/pytorch:2.14.0-cuda11.8-py312 python tests/test_metatomic.py
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host -v "$(pwd):/workspace" \
  iapetus/pytorch:2.14.0-cuda11.8-py312 python tests/test_openequivariance.py
tests/run_unit_tests.sh
tests/run_distributed_tests.sh
```

## Images

| Tag | Built from |
| :--- | :--- |
| `iapetus/builder:cuda11.8-py312` | `docker/builder.Dockerfile` |
| `iapetus/pytorch:2.14.0-cuda11.8-py312` | `docker/runtime.Dockerfile` |

## Further documentation

- [`CONVENTIONS.md`](CONVENTIONS.md): project rules and current compliance status
- [`docs/USAGE.md`](docs/USAGE.md): running workloads, test suites, environment flags
- [`docs/MULTIUSER.md`](docs/MULTIUSER.md): dynamic UID/GID entrypoint design
- [`docs/BUGS.md`](docs/BUGS.md): known issues and resolved problems
- [`archive/README.md`](archive/README.md): history of the bring-up
