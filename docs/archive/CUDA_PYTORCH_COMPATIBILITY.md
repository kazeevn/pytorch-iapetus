# CUDA and PyTorch Compatibility Matrix for Docker

This document records the empirical findings and architecture compatibility boundaries for running CUDA and PyTorch containers on this host machine.

---

## 1. System Hardware & Driver Baseline

* **Host OS:** Ubuntu 26.04.1 LTS (`x86_64`)
* **NVIDIA Driver Version:** `470.256.02` (The final legacy driver branch supporting Kepler GK110)
* **Host CUDA API Support:** CUDA 11.4 native driver capability
* **Detected GPUs:**
  * **GPU 0:** `Tesla K20c` (Compute Capability **`sm_35`**, Kepler GK110, 4743 MiB VRAM)
  * **GPU 1:** `Tesla K20c` (Compute Capability **`sm_35`**, Kepler GK110, 4743 MiB VRAM)
  * **GPU 2:** `NVIDIA GeForce GTX 750 Ti` (Compute Capability **`sm_50`**, Maxwell GM107, 1999 MiB VRAM)

---

## 2. Executive Summary & Compatibility Matrix

| Hardware Target | Maximum Supported PyTorch Image | PyTorch Version | Bundled CUDA | Status & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **Tesla K20c** (`sm_35`) | `pytorch:2.8.0-cuda11.8-kepler` (Custom) / `pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime` (Official) | **2.8.0** (Built from source) / **1.3.0** (Official) | **CUDA 11.8** | **PyTorch 2.8.0 built from source targeting `sm_35`** with `-march=native`. Runs on CUDA 11.8 on driver `470.256.02`. Note: cuDNN 8 dropped `sm_35`, so `torch.backends.cudnn.enabled = False` must be used for CNNs to use ATen native CUDA kernels. |
| **GeForce GTX 750 Ti** (`sm_50`) | `pytorch:2.8.0-cuda11.8-maxwell` (Custom) / `pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime` (Official) | **2.8.0** (Built from source) / **2.7.1** (Official) | **CUDA 11.8** | **PyTorch 2.8.0 built from source targeting `sm_50`**. Fully verified with CUDA 11.8 on driver `470.256.02`. Official prebuilt wheels stopped at 2.7.1. |
| **Raw CUDA (C/C++ nvcc)** | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` | N/A (nvcc) | **CUDA 11.8** | Supported via CUDA Minor Version Compatibility when bypassing `cuda-compat`. CUDA 12+ requires driver >= 525 and drops `sm_35`. |

---

## 3. Critical Technical Findings & Traps

### A. Docker GPU Flag & Missing `/dev/nvidia-uvm`
* Unified Virtual Memory (`nvidia-uvm`) is required for CUDA initialization (`cuInit`).
* On this host, default Docker CDI/containerd device injection with `--gpus all` does **not** map `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` into the container, causing `cuInit` to fail with error `999 (CUDA_ERROR_UNKNOWN)`.
* **Solution:** Launch containers using `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` (which invokes the NVIDIA container runtime hook to mount UVM devices), or explicitly pass `--device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`.

### B. The `cuda-compat` Forward Compatibility Failure on Kepler/Maxwell
* NVIDIA's official `nvidia/cuda:11.8.0-...` devel images bundle the `cuda-compat-11-8` package (`/usr/local/cuda/compat/libcuda.so.520.61.05`) to allow newer CUDA toolkits on older drivers.
* However, NVIDIA CUDA Forward Compatibility is **only supported on Pascal datacenter GPUs and newer** (sm_60+ / Volta / Turing / Ampere).
* When initializing CUDA on Kepler (`sm_35`) or consumer Maxwell (`sm_50`), the forward compatibility library fails with:
  ```
  cudaMalloc status: forward compatibility was attempted on non supported HW
  ```
* **Solution for raw CUDA 11.8:** Removing `/etc/ld.so.conf.d/cuda-compat*.conf` and `/usr/local/cuda/compat` forces CUDA to use the host's native `470.256.02` driver library via CUDA Minor Version Compatibility (which permits CUDA 11.x on driver >= 450.80.02). Kernels compiled for `sm_35` and `sm_50` then execute without issue.
* Official `pytorch/pytorch` runtime images do not include `cuda-compat` and are unaffected by this issue.

### C. Driver Limitation for CUDA 12+
* CUDA 12.0+ requires NVIDIA Driver >= `525.60.13`.
* Because driver `470.256.02` cannot be upgraded further without losing Kepler GK110 GPU support, **CUDA 11.8 is the hard upper limit** for the CUDA toolkit on this system.

---

## 4. PyTorch Version Boundary Details

### 1. Tesla K20c (`sm_35`) Breakdown
* **PyTorch 1.0.1 (`pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime`):** Supported (`sm_35` kernels present).
* **PyTorch 1.3.0 (`pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime`):** Supported (**Latest working release**).
* **PyTorch 1.4.0 (`pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime`):** Fails with `CUDA error: no kernel image is available for execution on the device`.
* **PyTorch 1.6.0 (`pytorch/pytorch:1.6-cuda10.1-cudnn7-runtime`):** Fails (`The current PyTorch install supports CUDA capabilities sm_37 sm_50...`).
* **PyTorch 1.7.0 – 2.7.1 (Official):** Fails (`The minimum cuda capability supported by this library is 3.7`).
* **PyTorch 2.8.0 (Custom Build - `pytorch:2.8.0-cuda11.8-kepler`):** **Supported and verified.**
  - Compiled from source targeting `TORCH_CUDA_ARCH_LIST="3.5"` and CUDA 11.8 with `-march=native`.
  - Required patch in `c10/cuda/driver_api.cpp`: for `CUDA_VERSION < 12000`, `cudaGetDriverEntryPoint` takes 3 arguments without `cudaDriverEntryPointQueryResult`.
  - Resulting wheel package: `dist-sm35/torch-2.8.0-cp310-cp310-linux_x86_64.whl` (138 MB).
  - Tested on both Tesla K20c GPUs (0 & 1): matrix multiplication, tensor autograd, and full neural network training.
  - **Critical cuDNN Note:** NVIDIA cuDNN 8 dropped `sm_35` support and returns `CUDNN_STATUS_ARCH_MISMATCH`. When running convolution layers (`Conv2d`), disable cuDNN via `torch.backends.cudnn.enabled = False` to utilize ATen's native `sm_35` CUDA convolution kernels.

### 2. GeForce GTX 750 Ti (`sm_50`) Breakdown
* **PyTorch 1.0 through 2.7.1:** Fully supported (`sm_50` is included in `torch.cuda.get_arch_list()`).
* **PyTorch 2.7.1 (`pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime`):** Latest official release runnable without rebuilding.
* **PyTorch 2.8.0 (Official):** Dropped official CUDA 11.8 builds (all prebuilt wheels target CUDA 12+, incompatible with driver `470.256.02`).
* **PyTorch 2.8.0 (Custom Build - `pytorch:2.8.0-cuda11.8-maxwell`):** **Supported and verified.**
  - Compiled from source targeting `TORCH_CUDA_ARCH_LIST="5.0"` and CUDA 11.8.
  - Required patch in `c10/cuda/driver_api.cpp`: for `CUDA_VERSION < 12000`, `cudaGetDriverEntryPoint` takes 3 arguments without `cudaDriverEntryPointQueryResult`.
  - Resulting wheel package: `pytorch-v2.8.0/dist/torch-2.8.0-cp310-cp310-linux_x86_64.whl`.
  - Tested with tensor operations, matrix multiplication, and deep neural network training (Conv2d + Linear forward/backward autograd) on GPU 2.

---

## 5. Verification Commands

### Test Tesla K20c (PyTorch 1.3.0 on GPUs 0 & 1)
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime \
  python -c '
import torch
print("PyTorch Version:", torch.__version__, "| CUDA Version:", torch.version.cuda)
for i in [0, 1]:
    torch.cuda.set_device(i)
    x = torch.ones(10, device=f"cuda:{i}")
    y = x + x
    torch.cuda.synchronize()
    print(f"Device {i} ({torch.cuda.get_device_name(i)}): SUCCESS! y[0]={y[0].item()}")
'
```

### Test Tesla K20c (PyTorch 2.8.0 Custom Build on GPUs 0 & 1)
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  pytorch:2.8.0-cuda11.8-kepler \
  python3 -c '
import torch
print("PyTorch Version:", torch.__version__, "| CUDA Version:", torch.version.cuda)
for i in [0, 1]:
    torch.cuda.set_device(i)
    x = torch.randn(1000, 1000, device=f"cuda:{i}")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print(f"Device {i} ({torch.cuda.get_device_name(i)}): SUCCESS! y[0,0]={y[0,0].item():.4f}")
'
```

### Test GeForce GTX 750 Ti (PyTorch 2.7.1 on GPU 2)
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime \
  python -c '
import torch
print("PyTorch Version:", torch.__version__, "| CUDA Version:", torch.version.cuda)
torch.cuda.set_device(2)
x = torch.ones(10, device="cuda:2")
y = x + x
torch.cuda.synchronize()
print(f"Device 2 ({torch.cuda.get_device_name(2)}): SUCCESS! y[0]={y[0].item()}")
'
```

### Test GeForce GTX 750 Ti (PyTorch 2.8.0 Custom Build on GPU 2)
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  pytorch:2.8.0-cuda11.8-maxwell \
  python3 -c '
import torch
print("PyTorch Version:", torch.__version__, "| CUDA Version:", torch.version.cuda)
torch.cuda.set_device(2)
x = torch.randn(1000, 1000, device="cuda:2")
y = torch.matmul(x, x)
torch.cuda.synchronize()
print(f"Device 2 ({torch.cuda.get_device_name(2)}): SUCCESS! y[0,0]={y[0,0].item():.4f}")
'
```

### Test Raw CUDA 11.8 Execution (Bypassing `cuda-compat`)
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
  bash -c '
rm -rf /etc/ld.so.conf.d/cuda-compat*.conf /usr/local/cuda/compat && ldconfig
cat << "EOF" > test.cu
#include <iostream>
__global__ void add(int n, float *x, float *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) y[i] = x[i] + y[i];
}
int main() {
    for (int dev = 0; dev < 3; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        float *d_x, *d_y, h_x = 1.0f, h_y = 2.0f;
        cudaMalloc(&d_x, sizeof(float)); cudaMalloc(&d_y, sizeof(float));
        cudaMemcpy(d_x, &h_x, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, &h_y, sizeof(float), cudaMemcpyHostToDevice);
        add<<<1, 1>>>(1, d_x, d_y);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Device " << dev << " (" << p.name << "): Result = " << h_y << std::endl;
        cudaFree(d_x); cudaFree(d_y);
    }
    return 0;
}
EOF
nvcc -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 test.cu -o test
./test
'
```

---

## 6. PyTorch 2.14 & Distributed Multi-GPU Support

* **Python Version:** Python 3.12 (installed and managed in container via Astral `uv`).
* **Target Architectures:** Dual `sm_35` (Tesla K20c Kepler) + `sm_50` (GeForce GTX 750 Ti Maxwell).
* **Compiler Flags:** `-march=native` CPU instructions enabled; cuDNN disabled (`USE_CUDNN=0`).
* **Source Patches Applied:**
  1. `c10/cuda/driver_api.cpp`: 3-argument signature for `cudaGetDriverEntryPoint` on CUDA < 12.0.
  2. `aten/src/ATen/cuda/cub.cuh` & `cub_definitions.cuh`: namespace alignment and fallback for Kepler CUB.
  3. `aten/src/ATen/native/cuda/GroupMM.cu` & `ScaledGroupMM.cu`: compile guards bypassing SM >= 8.0 tensor core kernels.

### Distributed Backends Overview

| Backend | Primary Transport | Supported Tensors | Status on this Machine | Notes & Topology |
| :--- | :--- | :--- | :--- | :--- |
| **NCCL** | Direct PCIe / SHM Ring | CUDA | **WORKING & VERIFIED** | Uses custom-built NCCL 2.23.4 patched for Kepler/Maxwell GPUs (`cudaDevAttrMemoryPoolsSupported == 0` fallback from `cudaMemPoolCreate` to standard `cudaMalloc`). P2P operates across 2× K20c (`cuda:0` $\leftrightarrow$ `cuda:1`); falls back to `SHM/direct/direct` ring across K20c + 750 Ti (`cuda:2`). |
| **Gloo** | TCP Loopback (`127.0.0.1`) + SHM | CPU & CUDA | **WORKING & VERIFIED** | Universal fallback backend. Architecture-independent. Verified on both CPU and multi-GPU CUDA collective operations across all 3 ranks. |
| **MPI** | OpenMPI SHM / TCP | CPU (CUDA requires CUDA-aware build) | **WORKING & VERIFIED** | Tested with OpenMPI 4.1.2. CPU collectives (`all_reduce`, `broadcast`, `all_gather`) pass across all ranks. Note: Standard Ubuntu package is host-only (non-CUDA-aware); CUDA tensors pass through NCCL/Gloo. |

---

## 7. Container Requirements for Distributed Multi-GPU Workloads

To run PyTorch distributed workloads (DDP, NCCL collectives, Gloo, or MPI) reliably inside Docker containers on this machine, the following container configurations are required:

### 1. Mandatory Docker Run Flags
* **`--ipc=host` or `--shm-size=8g` (CRITICAL)**:
  NCCL creates shared-memory rings (`SHM/direct/direct`) when peer-to-peer (P2P) PCIe access is unavailable between differing GPU architectures (Kepler $\leftrightarrow$ Maxwell). Docker's default shared memory size is only 64 MB, which triggers **`SIGBUS (exit code 135)`** during `ncclCommInitAll` or collective communication.
* **`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`**:
  Ensures that `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` are properly injected alongside device character nodes.
* **`--gpus all`**:
  Exposes all 3 physical GPUs to the container.

### 2. Recommended Runtime Environment Variables
```bash
# Repress benign warnings about disabled P2P between Kepler and Maxwell
export NCCL_IGNORE_DISABLED_P2P=1

# Enable diagnostic logging for collective operations
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV

# For OpenMPI execution as root inside container
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Bind Gloo to local loopback interface
export GLOO_SOCKET_IFNAME=lo
```

### 3. Example DDP / Collective Execution Command
```bash
docker run --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  -e NCCL_IGNORE_DISABLED_P2P=1 \
  pytorch-builder:cuda11.8-py312 \
  python3 -c "
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)
print('NCCL ProcessGroup successfully initialized!')
dist.destroy_process_group()
"
```
