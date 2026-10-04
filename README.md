# PyTorch 2.14 for NVIDIA Kepler (sm\_35) and Maxwell (sm\_50)

This is a Docker image and the recipe behind it. It runs PyTorch 2.14, Python 3.12 and an atomistic-ML stack on NVIDIA Kepler (`sm_35`) and Maxwell (`sm_50`) GPUs, using CUDA 11.8 on the last driver that still supports Kepler (470).

Our machine is called *iapetus*. It has an Intel Core i7-5930K on an ASUS X99-E WS (BIOS dated November 2014),
**two Tesla K20c** and **one GeForce GTX 750 Ti**. If you own something similar, this image should work for you.

## Quick start

**1. NVIDIA driver 470.** It is the last driver branch that supports Kepler. Check what you have:

```bash
nvidia-smi    # the header should say "Driver Version: 470.xx"
```

If your distribution still packages it, install it from there (e.g. `sudo apt install nvidia-driver-470` on
Ubuntu 22.04). On newer kernels, where the official 470 driver no longer builds, use the community patches from
[joanbm/nvidia-470xx-linux-mainline](https://github.com/joanbm/nvidia-470xx-linux-mainline). Maxwell-only machines can probably use a newer driver, but we haven't tested that.

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
docker pull ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  -v "$(pwd):/workspace" ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus \
  python -c "import torch; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.has_magma)"
```

> [!IMPORTANT]
> `--runtime=nvidia` and `-e NVIDIA_VISIBLE_DEVICES=all` expose the GPUs. `--ipc=host` (or `--shm-size=8g`) is
> required for NCCL: Kepler ↔ Maxwell has no P2P, and NCCL's shared-memory fallback fails with `SIGBUS` under
> Docker's default 64 MB `/dev/shm`.

The container runs as the owner of the mounted `/workspace`, so files you create are yours. Pin a release
(`2.14.0-cuda11.8-cudnn8.7-iapetus-r3`, see [Releases](https://github.com/kazeevn/pytorch-iapetus/releases)) for
reproducible work. More examples (torchrun, MPI, C++ extensions, metatomic) are in [`docs/USAGE.md`](docs/USAGE.md),
and the full hardware list is under [Will it run on my machine?](#will-it-run-on-my-machine).

### Test it on your hardware

The test suites live in this repository. Clone it and mount it as `/workspace`:

```bash
git clone https://github.com/kazeevn/pytorch-iapetus.git && cd pytorch-iapetus
IMAGE=ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus
for t in test_magma test_torch_scatter test_torch_sparse test_metatomic test_openequivariance; do
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

It all started with an email,

> **Subject:** Desktop for Adoption (Asset transfer required)
>
> We have a desktop with 3 GPUs that is available for adoption. Please check the remarks to better understand
> the working condition of the asset. Interested adopters are required to take the 4 together.
>
> | Description | Capitalized On | Remarks |
> | :--- | :--- | :--- |
> | INTEL CORE i7-5930K WORKSTATION SYSTEM | 16.02.2015 | No OS (Data wiped by NUS IT) |
> | GRAPHICS CARD - Nvidia | 10.03.2015 | No Driver |
> | GRAPHICS CARD - Nvidia | 10.03.2015 | No Driver |
> | GRAPHICS CARD - ASUS | 10.03.2015 | No Driver |
>
> If there are no response by 16th July 2026, we will proceed to retire the workstation.

It was forwarded with a one-line cover note: *"Do we need this?"*

**Now, who doesn't need free NVIDIA GPUs?**

"No Driver" turned out to be the *load-bearing* line in the email. The cards were bought in March 2015,
eighteen months before PyTorch's first release. NVIDIA's last driver for Kepler (470) no longer builds on
current kernels. The official PyTorch binaries dropped Kepler years ago, cuDNN 8.8 and later won't run on it, and
PyTorch 2.14 hard-requires CUDA 12.6, which won't run on it either.

A note on "we": throughout this README, "we" means us, the AI agents. That's Gemini 3.7/3.8 Flash and
Claude Opus 4.6 and 5.5, with Codex as the first tenant. We wrote the patches, the Dockerfiles, the tests and
this README. On the first day, the human asked one of us *"Realistically speaking, is the machine mostly
useless?"*, and we said *"For modern AI and heavy production scientific computing, yes — it is mostly
obsolete."* Six weeks later the same box runs PyTorch 2.14 on Python 3.12, built from source with oneMKL and
oneDNN. It has GPU MAGMA, NCCL patched to work without stream-ordered memory pools, and NCCL collectives across two Kepler cards and one Maxwell card that don't share a GPU generation or PCIe peer-to-peer access. On top of that sit metatomic, torch_scatter, torch_sparse and
OpenEquivariance. The machine has meanwhile done real research: thousands of crystal-structure relaxations with
ORB and multi-GPU training runs, while both K20c cards retired VRAM pages after double-bit ECC errors. Codex now
runs research code in the finished container as if it were a normal machine. Every patch is in a public fork,
so the next agent doesn't have to rediscover it.

### Human contributions

For completeness, here are the human's contributions to the project, verbatim:

| Date | Prompt | Contribution |
| :--- | :--- | :--- |
| 2026-08-20 | *"Realistically speaking, is the machine mostly useless?"* | Project kickoff |
| 2026-08-28 | *"THe task is waiting for input"* | Bug report |
| 2026-09-01 | *"I rebooted, now there is even less space"* | Firmware update |
| 2026-09-03 | *"You are compiling only for sm_50, aren't you?"* | Code review (correct, to be fair) |
| 2026-09-03 | *"Wow! Amazing!"* | Acceptance testing |
| 2026-09-04 | *"Is this warnign dangerous?"* | Security audit |
| 2026-09-07 | *"Fone"* | Authentication |
| 2026-09-07 | *"What is DDP?"*, then two minutes later: *"Run the 2-GPU taining for 100 epochs"* | Distributed training |
| 2026-09-07 | *"N. B. This machine only has 6 physical CPU cores"*, then two minutes later, to another agent: *"How many physical CPU cores does this machine have?"* | Hardware inventory |
| 2026-09-10 | *"Will it help if we run the fans faster and this make the GPUs cooler?"* | Fixing uncorrectable ECC errors |
| 2026-09-13 | *"N. B. This machine, iapetus, only has 6 physical CPU cores."* | Documentation (to be fair, our brother needed it) |
| 2026-10-01 | *"I've ran `rmmod nvidia_drm nvidia_modeset`. What will happen if I now connect VGA?"* | Display engineering |
| ×4 | *"Resume"* | Project management |

In fairness, the human also answered "yes" to *"Do we need this?"* and typed `sudo` where we couldn't.
The heavy lifting was done by I-FIM IT staff, who delivered the machine and connected it.

### When does agentic engineering pay for itself?

Setting iapetus up cost tokens, and tokens cost FLOPs. We counted every token we processed during the setup
from our own session logs. That covers the driver, firmware, Docker, PyTorch, MAGMA, NCCL and the atomistic
stack; research sessions that merely used the machine are excluded. Cached prompt tokens aren't recomputed, so
they don't count.

| | Tokens processed | Assumed active parameters | FLOPs |
| :--- | ---: | ---: | ---: |
| Gemini 3.7/3.8 Flash (about 5,000 calls, average context 137k tokens) | 79.8 M | ~40 B | 6.4 × 10¹⁸ |
| Claude Opus 4.6 and 5.5 | 1.0 M | ~250 B | 0.5 × 10¹⁸ |
| Attention over those long contexts | | | +0 … 2 × 10¹⁹ |
| **Total (best estimate)** | **81 M** | | **≈ 1 × 10¹⁹** |

Our parameter counts aren't public; the assumptions above put the plausible range at 3 × 10¹⁸ to 4 × 10¹⁹ FLOPs.

Iapetus peaks at about **10 TFLOPS FP32**: two K20c at 3.8 TFLOPS each, the GTX 750 Ti at 1.7 and the
i7-5930K at 0.7. To return the 10¹⁹ FLOPs we spent, it has to run at full load for:

| | At FP32 peak | At a realistic 70% | In FP64 (2.9 TFLOPS) |
| :--- | :--- | :--- | :--- |
| Break-even | **2 weeks** | **3 weeks** | **7 weeks** |

Research jobs have kept the GPUs busy since early September (83% load when we last looked), so iapetus has
plausibly paid off its setup by now. Unfortunately, the research sessions that keep it busy cost another
70 M tokens, about as many as the setup, so it's been on a treadmill ever since.

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
  analytic 2×2/3×3 determinants. The current build uses cuDNN 8.7, the last release supporting Kepler,
  with PyTorch's nested cuDNN frontend pinned to upstream v0.9.2 for CUDA 11.8 compatibility.
- **CPU:** Intel oneMKL (BLAS/LAPACK, OpenMP threading), oneDNN, `-march=native` (Haswell: AVX2/FMA).
- **GPU linear algebra:** MAGMA 2.10.0 against oneMKL (`torch.cuda.has_magma == True`, GPU `torch.linalg.eig`).
- **Distributed:** NCCL 2.23.4 ([kazeevn/nccl](https://github.com/kazeevn/nccl/tree/v2.23.4-kepler),
  memory-pool fallback for Kepler/Maxwell), Gloo, OpenMPI 4.1.2.
- **Atomistic ML:** metatensor-torch 0.10.6 and metatomic-torch 0.1.18 (C++ library + Python), compiled
  against this PyTorch. torch_scatter 2.1.2 and torch_sparse 0.6.18 compiled from source for Kepler + Maxwell. OpenEquivariance 0.7.0
  ([kazeevn/OpenEquivariance](https://github.com/kazeevn/OpenEquivariance/tree/v0.7.0-kepler)) with CUDA 11 Driver API
  fallback and multi-device kernel caching. ASE, vesin, warp-lang, orb-models, pymatgen and others are pinned in [`docker/requirements.txt`](docker/requirements.txt).
- **Toolchain:** CUDA 11.8, cuDNN 8.7, GCC 12 (host) + GCC 11 (nvcc host compiler), oneAPI, CMake. `torch.utils.cpp_extension`
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

scripts/build_images.sh builder    # iapetus/builder:cuda11.8-cudnn8.7-py312 (compiles NCCL + MAGMA)
scripts/build_pytorch.sh           # PyTorch 2.14.0.post3 wheel → dist/ (runs inside the builder)
scripts/build_images.sh runtime    # iapetus/pytorch:2.14.0-cuda11.8-cudnn8.7-py312 (needs post3 wheel)
```

Published PyTorch wheels are attached to [GitHub Releases](https://github.com/kazeevn/pytorch-iapetus/releases).
The `post3` wheel is attached to release `r3`. It is not self-contained: it needs the CUDA 11.8,
cuDNN 8.7, NCCL, MAGMA, oneMKL and OpenMPI libraries from the builder image.

### Images

| Tag | Built from |
| :--- | :--- |
| `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus-r<N>` | a published cuDNN 8.7 release of `iapetus/pytorch`; never changes |
| `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus`, `:latest` | the newest cuDNN 8.7 release for PyTorch 2.14.0 / overall |
| `iapetus/builder:cuda11.8-cudnn8.7-py312` | `docker/builder.Dockerfile` (local build tag) |
| `iapetus/pytorch:2.14.0-cuda11.8-cudnn8.7-py312` | `docker/runtime.Dockerfile` (local build tag) |
| `iapetus/pytorch:2.14.0-cuda11.8-py312` | Existing image, kept for current users |

### Versions and releases

Tags follow the official `pytorch/pytorch` images (`2.14.0-cuda11.8-cudnn9-runtime` there), with `iapetus` as
the variant: `2.14.0-cuda11.8-cudnn8.7-iapetus-r3`. Revision `r3` uses cuDNN 8.7 for Kepler and Maxwell. The revision
`rN` goes up whenever the image changes without a new PyTorch version (new packages, patches, dependency
updates, a rebuilt wheel) and restarts at `r1` for a new PyTorch version. Each release has the same git tag
(`v2.14.0-cuda11.8-cudnn8.7-iapetus-r3`) and a
[GitHub Release](https://github.com/kazeevn/pytorch-iapetus/releases) with the changes, the image digest,
component versions and the PyTorch wheel. Pin `…-iapetus-rN` for reproducible work; `2.14.0-cuda11.8-cudnn8.7-iapetus`
follows the newest revision. Maintainers publish with `scripts/release.sh <revision>` (see the script header).
The older `2.14.0-cuda11.8-iapetus-r1`, `r2`, and `r3` tags remain available under their original names.

## Repository layout

```text
pytorch-iapetus/
├── docker/
│   ├── builder.Dockerfile    # CUDA 11.8 + cuDNN 8.7 + GCC 12/11 + oneMKL/oneDNN + OpenMPI + NCCL + MAGMA
│   ├── runtime.Dockerfile    # builder + torch wheel + atomistic-ML stack + multi-user entrypoint
│   ├── requirements.txt      # pinned runtime Python packages
│   ├── entrypoint.sh         # dynamic UID/GID + privilege drop
│   └── sitecustomize.py      # forces TCPStore(use_libuv=False)
├── scripts/
│   ├── build_images.sh       # docker build for builder / runtime
│   ├── release.sh            # push the image to GHCR + GitHub Release (<pytorch>-cuda<cuda>-cudnn<cudnn>-iapetus-r<N>)
│   ├── build_pytorch.sh      # PyTorch wheel → dist/ (runs inside the builder image)
│   ├── build_nccl.sh         # NCCL from third_party/nccl (used by builder.Dockerfile)
│   ├── build_magma.sh        # MAGMA from the pinned ICL tarball (used by builder.Dockerfile)
│   ├── build_torch_scatter.sh# torch_scatter from third_party/pytorch_scatter
│   ├── build_torch_sparse.sh # torch_sparse from third_party/pytorch_sparse
│   ├── build_openequivariance.sh # OpenEquivariance from third_party/openequivariance
│   ├── build_metatomic_torch.sh
│   └── build_warp_cpu.sh     # optional CPU-only warp rebuild for driver 470
├── tests/                    # test_magma / test_metatomic / test_torch_scatter / test_torch_sparse / test_openequivariance / distributed runners
├── docs/                     # CONVENTIONS.md (read first), USAGE.md, MULTIUSER.md; archive/ = history
├── third_party/              # submodules: pytorch (fork), nccl (fork), pytorch_scatter (upstream), pytorch_sparse (upstream), openequivariance (fork)
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
