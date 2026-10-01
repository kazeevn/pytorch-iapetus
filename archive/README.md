# Historical Artifacts Archive

Superseded builds and experiments from the bring-up of PyTorch on Kepler (`sm_35`) and Maxwell (`sm_50`).
Historical *documents* live in [`docs/archive/`](../docs/archive/).
Only small text files are tracked in git; large items (marked *local*) are git-ignored and exist only on
the build machine.

| Item | Description | Replaced by |
| :--- | :--- | :--- |
| `legacy-docker/` | Hand-assembled Dockerfile chain for 2.8 and 2.14 (`builder` → `builder.py312` → `runtime.py312` → `universal` → `metatomic`) | `docker/builder.Dockerfile`, `docker/runtime.Dockerfile` |
| `build_pytorch.sh` | Build script for PyTorch 2.8 | `scripts/build_pytorch.sh` |
| `dockerhub_pytorch_tags.json` | Scrape of official PyTorch Docker Hub tags | reference data |
| `mace-experiments/` | Early MACE benchmarks (`model_cache/` is *local*) | — |
| `prebuilt/` *(local)* | Prebuilt NCCL 2.23.4 and MAGMA 2.10.0 binaries copied into the legacy builder | built from source in `docker/builder.Dockerfile` |
| `sources/` *(local)* | metatomic 0.1.18 sdists (identical to PyPI) and NanoVDB headers | downloaded and sha256-checked by `scripts/build_metatomic_torch.sh`; NanoVDB ships with warp-lang |
| `pytorch-v2.8.0/` *(local)* | PyTorch 2.8 source tree | `third_party/pytorch` (2.14) |
| `dist-sm35/`, `dist-sm50/`, `dist-nodist/`, `dist-warp/` *(local)* | Intermediate wheels | `dist/` |
| `torch-2.14.0-cp312-*.whl[.bak]` *(local)* | Earlier 2.14 wheels (the non-`.bak` one is byte-identical to `dist/…post2…whl`) | `dist/` |
| `mace-repo/` *(local)* | MACE clone used for early tests | — |
