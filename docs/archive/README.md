# Documentation archive

Historical documents. They record how this project got to its current state and are **not** kept in sync with
the code: paths, image tags and links in them may be stale. Current documentation is in [`docs/`](../).

| Document | Contents |
| :--- | :--- |
| [`BUGS.md`](BUGS.md) | Problems hit after the 2.14 build worked (MAGMA, libuv/TCPStore, NCCL memory pools, `cpp_extension` + nvcc 11.8) and how each was fixed |
| [`BRINGUP_BUGS.md`](BRINGUP_BUGS.md) | Earlier issues with the first 2.14 wheels under real workloads (missing LAPACK, NVRTC `--std` on driver 470, `det` JIT on Kepler, warp-lang, ...) and their fixes; paths refer to the pre-repository layout |
| [`CUDA_PYTORCH_COMPATIBILITY.md`](CUDA_PYTORCH_COMPATIBILITY.md) | Initial research: which CUDA, driver and PyTorch versions can run on Kepler/Maxwell, and which official images work |
