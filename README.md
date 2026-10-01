# pytorch-iapetus

**PyTorch 2.14 for GPUs that everyone else gave up on.** This is a Docker image and the recipe behind it.
It runs PyTorch 2.14, Python 3.12 and an atomistic-ML stack on NVIDIA Kepler (`sm_35`) and Maxwell (`sm_50`)
GPUs, using CUDA 11.8 on the last driver that still supports Kepler (470).

Our machine is called *iapetus*. It has an Intel Core i7-5930K on an ASUS X99-E WS (BIOS dated November 2014),
**two Tesla K20c** and **one GeForce GTX 750 Ti**. If you own something similar, this image should work for you.

## Quick start

**1. NVIDIA driver 470.** It is the last driver branch that supports Kepler. Check what you have:

```bash
nvidia-smi    # the header should say "Driver Version: 470.xx"
```

If your distribution still packages it, install it from there (e.g. `sudo apt install nvidia-driver-470` on
Ubuntu). On newer kernels, where the official 470 driver no longer builds, use the community patches from
[joanbm/nvidia-470xx-linux-mainline](https://github.com/joanbm/nvidia-470xx-linux-mainline) (iapetus runs
Linux 7.0 this way). Maxwell-only machines can probably use a newer driver, but we haven't tested that.

**2. Docker with the NVIDIA Container Toolkit.** Install the toolkit following
[NVIDIA's guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then register its runtime with Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**3. A CPU with AVX2** (Intel Haswell or newer, AMD Zen). The image is compiled for Haswell, and older CPUs
crash with `Illegal instruction`:

```bash
grep -qw avx2 /proc/cpuinfo && echo "AVX2: ok"
```

**4. Pull and run:**

```bash
docker pull ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus \
  python -c "import torch; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.has_magma)"
```

> [!IMPORTANT]
> `--runtime=nvidia` and `-e NVIDIA_VISIBLE_DEVICES=all` expose the GPUs. `--ipc=host` (or `--shm-size=8g`) is
> required for NCCL: Kepler ↔ Maxwell has no P2P, and NCCL's shared-memory fallback fails with `SIGBUS` under
> Docker's default 64 MB `/dev/shm`.

The container runs as the owner of the mounted `/workspace`, so files you create are yours. Pin a release
(`2.14.0-cuda11.8-iapetus-r1`, see [Releases](https://github.com/kazeevn/pytorch-iapetus/releases)) for
reproducible work. More examples (torchrun, MPI, C++ extensions, metatomic) are in [`docs/USAGE.md`](docs/USAGE.md),
and the full hardware list is under [Will it run on my machine?](#will-it-run-on-my-machine).

### Test it on your hardware

The test suites live in this repository. Clone it and mount it as `/workspace`:

```bash
git clone https://github.com/kazeevn/pytorch-iapetus.git && cd pytorch-iapetus
IMAGE=ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus
for t in test_magma test_torch_scatter test_metatomic test_openequivariance; do
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host -v "$(pwd):/workspace" \
    "$IMAGE" python "tests/$t.py"
done
IMAGE="$IMAGE" tests/run_unit_tests.sh
IMAGE="$IMAGE" tests/run_distributed_tests.sh
```

If it works (or fails in an interesting way) on hardware we haven't tested, please open an issue. We'd love
to hear what other museum pieces are still running.

---

## Why this exists

On the day we started, we asked an AI assistant to assess the machine: *"Realistically speaking, is the
machine mostly useless?"* It said: *"For modern AI and heavy production scientific computing, yes — it is
mostly obsolete."* That was fair. The official PyTorch binaries dropped Kepler years ago, cuDNN 8 won't run
on it, and PyTorch 2.14 hard-requires CUDA 12.6, which won't run on it either. Six weeks later the same box
runs PyTorch 2.14 on Python 3.12, built from source with oneMKL and oneDNN. It has GPU MAGMA (`torch.linalg.eig`
on a Tesla model from 2012, with errors around 1e-15), NCCL patched to work without stream-ordered memory pools, and NCCL
collectives across two Kepler cards and one Maxwell card that don't share a GPU generation or PCIe peer-to-peer access.
On top of that sit metatomic, torch_scatter and OpenEquivariance. One of us looked at PyTorch 2.8 running
on a K20c and typed *"Wow! Amazing!"*, and we still feel that way. The machine has meanwhile done real
research: thousands of crystal-structure relaxations with ORB and multi-GPU training runs, while both K20c
cards retired VRAM pages after double-bit ECC errors. Nearly all of the porting was done by AI coding
agents (Gemini and Claude) working in this repository, with a human asking pointed questions like
*"You are compiling only for sm_50, aren't you?"*. Codex now runs research code in the finished container
as if it were a normal machine. Every patch is in a public fork so the next person
doesn't have to rediscover it.

## Why "iapetus"

We adopted the machine from another NUS department and wanted to name it after something old and mythical.
Iapetus turned out to fit better than we planned:

- **A Titan.** Iapetus is one of the Titans, the older generation of Greek gods. The Olympians overthrew them
  in the Titanomachy and locked them in Tartarus. Kepler and Maxwell are an old generation of GPUs, replaced
  by Pascal through Blackwell and locked out of CUDA 12 and modern PyTorch. Our K20c cards use GK110, the same
  chip as NVIDIA's original GeForce GTX **Titan**.
- **Father of Prometheus and Atlas.** Prometheus stole fire from the gods and gave it to mortals. This project
  takes PyTorch 2.14, which is built for new GPUs, and gives it to old ones. Atlas holds up the sky. The two
  K20c cards held up weeks of relaxation and training jobs, even after both had retired VRAM pages.
- **A two-faced moon.** Saturn's moon Iapetus, discovered by Cassini in 1671, has one dark and one bright
  hemisphere. Our machine also has two halves that don't match: two Kepler Tesla compute cards
  and one Maxwell gaming card, with no peer-to-peer access between them.
- **Rehomed.** In some versions of the myth the Titans are later released from Tartarus. Our machine was
  handed over by its previous owners and now has a second career.

---

## Will it run on my machine?

| Requirement | Details |
| :--- | :--- |
| **CPU** | x86-64 with **AVX2, FMA and BMI2**: Intel Haswell (2013) or newer, or AMD Zen. Everything is compiled with `-march=native` on a Haswell-E. Sandy Bridge / Ivy Bridge / AMD FX will crash with `Illegal instruction`. In that case, rebuild from source on your machine (see below). |
| **GPU** | Compute capability **3.5 / 3.7** (Tesla K20/K20X/K40/K80, GTX 780/Titan/Titan Black) or **5.x** (GTX 750/750 Ti, GTX 9xx, Titan X Maxwell, Tesla M40/M60). The image contains `sm_35` and `sm_50` machine code only, with no PTX. CUDA cubins run on the same major architecture with an equal or higher minor version, so `sm_37` and `sm_52` should work. **Tested only on K20c and GTX 750 Ti.** Kepler `sm_30` (GTX 680, K10) and Pascal or newer are not supported. |
| **Driver** | NVIDIA **470.xx** (tested: `470.256.02`). It is the last branch that supports Kepler. On kernels newer than 470 supports, we build it with the community patches from [joanbm/nvidia-470xx-linux-mainline](https://github.com/joanbm/nvidia-470xx-linux-mainline) via DKMS (iapetus runs Linux 7.0). Maxwell-only hosts can probably use a newer driver, but we haven't tested that. |
| **Container runtime** | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (tested: 1.20.1) |
| **Disk** | About 7 GB to download and 22 GB unpacked. The image includes the full CUDA 11.8 / GCC / oneAPI toolchain, so you can compile extensions inside it. |

## What's inside

- **PyTorch 2.14.0** ([kazeevn/pytorch_kepler](https://github.com/kazeevn/pytorch_kepler/tree/v2.14.0-kepler)):
  CUDA 11.8 support restored, C++17 fallbacks in CUDA-visible code, NVRTC `--std=c++17` on CUDA < 12,
  analytic 2×2/3×3 determinants. cuDNN is disabled because cuDNN 8 doesn't support Kepler.
- **CPU:** Intel oneMKL (BLAS/LAPACK, OpenMP threading), oneDNN, `-march=native` (Haswell: AVX2/FMA).
- **GPU linear algebra:** MAGMA 2.10.0 against oneMKL (`torch.cuda.has_magma == True`, GPU `torch.linalg.eig`).
- **Distributed:** NCCL 2.23.4 ([kazeevn/nccl](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler),
  memory-pool fallback for Kepler/Maxwell), Gloo, OpenMPI 4.1.2.
- **Atomistic ML:** metatensor-torch 0.10.6 and metatomic-torch 0.1.18 (C++ library + Python), compiled
  against this PyTorch. torch_scatter 2.1.2 compiled from source for Kepler + Maxwell. OpenEquivariance 0.7.0
  ([kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler)) with CUDA 11 Driver API
  fallback and multi-device kernel caching. ASE, vesin, warp-lang, orb-models, pymatgen and others are pinned in [`docker/requirements.txt`](docker/requirements.txt).
- **Toolchain:** CUDA 11.8, GCC 12 (host) + GCC 11 (nvcc host compiler), oneAPI, CMake. `torch.utils.cpp_extension`
  builds work out of the box (see [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)).
- **Multi-user entrypoint:** the container runs as the owner of the mounted `/workspace`, so files you create
  belong to you and not root (see [`docs/MULTIUSER.md`](docs/MULTIUSER.md)).

Our reference machine:

| Index | Device | Architecture | Compute capability | VRAM |
| :--- | :--- | :--- | :--- | :--- |
| `cuda:0` | Tesla K20c | Kepler | `sm_35` | 5 GB |
| `cuda:1` | Tesla K20c | Kepler | `sm_35` | 5 GB |
| `cuda:2` | GeForce GTX 750 Ti | Maxwell | `sm_50` | 2 GB |

---

## Building from source

Build from source if your CPU lacks AVX2, if you want `-march=native` for your own CPU, or if you have
different GPUs. **Read [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) before changing anything.** It defines where sources
come from (forks vs. pinned upstream), the compiler and C++-standard rules (GCC 12 / C++20 host, nvcc + GCC 11 /
C++17 device), and the optimization targets (`-march=native`, `sm_35` + `sm_50` only, oneMKL / oneDNN / OpenMPI).
For other GPUs, change `TORCH_CUDA_ARCH_LIST` / `CUDAARCHS` in `docker/builder.Dockerfile` and the NCCL / MAGMA
build scripts.

```bash
git clone --recurse-submodules=third_party/nccl https://github.com/kazeevn/pytorch-iapetus.git
cd pytorch-iapetus
git submodule update --init third_party/pytorch   # large; only needed to rebuild PyTorch
git -C third_party/pytorch submodule update --init --recursive   # only for a full PyTorch rebuild

scripts/build_images.sh builder    # iapetus/builder:cuda11.8-py312 (compiles NCCL + MAGMA)
scripts/build_pytorch.sh           # PyTorch wheel → dist/ (runs inside the builder)
scripts/build_images.sh runtime    # iapetus/pytorch:2.14.0-cuda11.8-py312 (needs dist/*.whl)
```

The PyTorch wheel is attached to each [GitHub Release](https://github.com/kazeevn/pytorch-iapetus/releases). It is
not self-contained: it needs the CUDA 11.8, NCCL, MAGMA, oneMKL and OpenMPI libraries from the builder image.

### Images

| Tag | Built from |
| :--- | :--- |
| `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus-r<N>` | a published release of `iapetus/pytorch`; never changes |
| `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus`, `:latest` | the newest release for PyTorch 2.14.0 / overall |
| `iapetus/builder:cuda11.8-py312` | `docker/builder.Dockerfile` (local only) |
| `iapetus/pytorch:2.14.0-cuda11.8-py312` | `docker/runtime.Dockerfile` (local build tag) |

### Versions and releases

Tags follow the official `pytorch/pytorch` images (`2.14.0-cuda11.8-cudnn9-runtime` there), with `iapetus` as
the variant: `2.14.0-cuda11.8-iapetus-r1`. There is no cuDNN, since cuDNN doesn't support Kepler. The revision
`rN` goes up whenever the image changes without a new PyTorch version (new packages, patches, dependency
updates, a rebuilt wheel) and restarts at `r1` for a new PyTorch version. Each release has the same git tag
(`v2.14.0-cuda11.8-iapetus-r1`) and a
[GitHub Release](https://github.com/kazeevn/pytorch-iapetus/releases) with the changes, the image digest,
component versions and the PyTorch wheel. Pin `…-iapetus-rN` for reproducible work; `2.14.0-cuda11.8-iapetus`
follows the newest revision. Maintainers publish with `scripts/release.sh <revision>` (see the script header).

## Repository layout

```text
pytorch-iapetus/
├── docker/
│   ├── builder.Dockerfile    # CUDA 11.8 toolchain + GCC 12/11 + oneMKL/oneDNN + OpenMPI + NCCL + MAGMA
│   ├── runtime.Dockerfile    # builder + torch wheel + atomistic-ML stack + multi-user entrypoint
│   ├── requirements.txt      # pinned runtime Python packages
│   ├── entrypoint.sh         # dynamic UID/GID + privilege drop
│   └── sitecustomize.py      # forces TCPStore(use_libuv=False)
├── scripts/
│   ├── build_images.sh       # docker build for builder / runtime
│   ├── release.sh            # push the image to GHCR + GitHub Release (<pytorch>-cuda<cuda>-iapetus-r<N>)
│   ├── build_pytorch.sh      # PyTorch wheel → dist/ (runs inside the builder image)
│   ├── build_nccl.sh         # NCCL from third_party/nccl (used by builder.Dockerfile)
│   ├── build_magma.sh        # MAGMA from the pinned ICL tarball (used by builder.Dockerfile)
│   ├── build_torch_scatter.sh# torch_scatter from third_party/pytorch_scatter
│   ├── build_openequivariance.sh # OpenEquivariance from third_party/openequivariance
│   ├── build_metatomic_torch.sh
│   └── build_warp_cpu.sh     # optional CPU-only warp rebuild for driver 470
├── tests/                    # test_magma / test_metatomic / test_torch_scatter / test_openequivariance / distributed runners
├── docs/                     # CONVENTIONS.md (read first), USAGE.md, MULTIUSER.md; archive/ = history
├── third_party/              # submodules: pytorch (fork), nccl (fork), pytorch_scatter (upstream), openequivariance (fork)
├── dist/                     # wheels (git-ignored)
├── LICENSE                   # Apache 2.0
└── archive/                  # legacy Dockerfiles and experiments from the bring-up
```

## Further documentation

- [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md): project rules and current compliance status
- [`docs/USAGE.md`](docs/USAGE.md): running workloads, test suites, environment flags
- [`docs/MULTIUSER.md`](docs/MULTIUSER.md): dynamic UID/GID entrypoint design
- [`docs/archive/`](docs/archive/): history of the bring-up: problems we hit and how they were fixed ([`BUGS.md`](docs/archive/BUGS.md), [`BRINGUP_BUGS.md`](docs/archive/BRINGUP_BUGS.md)), CUDA/driver compatibility research
- [`archive/README.md`](archive/README.md): legacy build files and local-only artifacts

## License

**This repository** (build scripts, Dockerfiles, tests and documentation) is licensed under the
[Apache License 2.0](LICENSE). Our forks in `third_party/` keep their upstream licenses.

**The container image** is a derivative of NVIDIA's `nvidia/cuda` image. It is distributed under the
[NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license),
included in the image as `/NGC-DL-CONTAINER-LICENSE`, and under the licenses of the software it contains:
Intel oneAPI / oneMKL, PyTorch, NCCL, MAGMA, the Ubuntu packages and the pinned Python packages. By using the
image you accept those terms. Our own files in the image are Apache 2.0.
