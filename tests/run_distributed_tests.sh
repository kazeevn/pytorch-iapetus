#!/usr/bin/env bash
# Multi-GPU collectives (Gloo, NCCL, MPI) plus official PyTorch distributed tests, in the runtime image.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-iapetus/pytorch:2.14.0-cuda11.8-py312}"

echo "================================================================="
echo " Starting Full Distributed & Device Unit Test Verification Suite "
echo "================================================================="

docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host -e HOST_UID=0 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e NCCL_DEBUG=INFO \
  -v "$REPO/third_party/pytorch/test:/workspace/pytorch/test" \
  -v "$REPO/tests/test_distributed_all.py:/workspace/test_distributed_all.py" \
  "$IMAGE" \
  bash -c '
set -e
/opt/venv312/bin/pip install -q pytest hypothesis expecttest optree psutil

echo "=== Running Backend & Collective Tests (Gloo, NCCL, P2P, Device Ops) ==="
/opt/venv312/bin/python /workspace/test_distributed_all.py

echo "=== Running MPI Collective Tests (OpenMPI with 3 ranks) ==="
mpirun --allow-run-as-root -n 3 /opt/venv312/bin/python /workspace/test_distributed_all.py --mpi-worker

echo "=== Running Official PyTorch Distributed Unit Tests ==="
cd /workspace/pytorch/test
export PYTHONPATH=/workspace/pytorch:/workspace/pytorch/test:$PYTHONPATH

echo "--- 1. Official PyTorch Distributed: Backend Initialization (Gloo, NCCL, MPI) ---"
/opt/venv312/bin/python -m unittest distributed.test_c10d_common.ProcessGroupWithDispatchedCollectivesTests.test_init_process_group_for_all_backends

echo "--- 2. Official PyTorch Distributed: Gloo CPU & CUDA Collectives ---"
/opt/venv312/bin/python distributed/test_c10d_gloo.py ProcessGroupGlooTest.test_allreduce_basics
/opt/venv312/bin/python distributed/test_c10d_gloo.py ProcessGroupGlooTest.test_broadcast_basics
/opt/venv312/bin/python distributed/test_c10d_gloo.py ProcessGroupGlooTest.test_allreduce_basics_cuda

echo "--- 3. Official PyTorch Distributed: NCCL Init & Collectives ---"
/opt/venv312/bin/python distributed/test_c10d_nccl.py ProcessGroupNCCLInitTest
/opt/venv312/bin/python distributed/test_c10d_nccl.py ProcessGroupNCCLOneRankTest
/opt/venv312/bin/python distributed/test_c10d_nccl.py NcclProcessGroupWithDispatchedCollectivesTests.test_collectives

echo "================================================================="
echo " All Multi-GPU & Distributed Tests Passed Successfully!          "
echo "================================================================="
'
