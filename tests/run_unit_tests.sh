#!/usr/bin/env bash
# Runs a curated subset of the official PyTorch unit tests inside the runtime image.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-iapetus/pytorch:2.14.1-cuda11.8-cudnn8.7-py312}"

echo "=== Running Official PyTorch Unit Tests ($IMAGE) ==="

docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
  --ipc=host -e HOST_UID=0 \
  -v "$REPO/third_party/pytorch/test:/workspace/test" \
  -v "$REPO/tests:/workspace/tests" \
  "$IMAGE" \
  bash -c '
set -e
/opt/venv312/bin/python /workspace/tests/test_cudnn.py
/opt/venv312/bin/python /workspace/tests/test_torch_scatter.py
/opt/venv312/bin/python /workspace/tests/test_torch_sparse.py
/opt/venv312/bin/python /workspace/tests/test_openequivariance.py
/opt/venv312/bin/pip install -q pytest hypothesis expecttest optree psutil
cd /workspace

/opt/venv312/bin/python - << "PYEOF"
import sys
import unittest
import torch

sys.path.insert(0, "/workspace/test")
sys.path.insert(0, "/workspace")

print("================================================================")
print(" PyTorch Official Unit Test Suite Runner")
print(f" PyTorch {torch.__version__} | Python {sys.version.split()[0]}")
print(f" Detected {torch.cuda.device_count()} GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)} (Compute Capability {torch.cuda.get_device_capability(i)})")
print("================================================================\n")

suites = [
    ("test_cuda", [
        ("TestCudaArchList", "test_get_arch_list_does_not_require_cuda_is_available"),
        ("TestCuda", "test_cuda_get_device_name"),
        ("TestCuda", "test_cuda_get_device_capability"),
        ("TestCuda", "test_cuda_get_device_properties"),
        ("TestCuda", "test_memory_allocation"),
        ("TestCuda", "test_memory_stats"),
        ("TestCuda", "test_pinned_memory_empty_cache"),
        ("TestCuda", "test_copy_non_blocking"),
        ("TestCuda", "test_type_conversions"),
        ("TestCuda", "test_torch_manual_seed_seeds_cuda_devices"),
        ("TestCuda", "test_cublas_workspace_explicit_allocation"),
        ("TestCuda", "test_cublas_unified_workspace"),
        ("TestCuda", "test_cublas_allow_tf32_get_set"),
        ("TestCuda", "test_float32_matmul_precision_get_set"),
        ("TestCuda", "test_streams"),
        ("TestCuda", "test_stream_event_repr"),
        ("TestCuda", "test_record_stream"),
        ("TestCuda", "test_stream_context_manager"),
        ("TestCuda", "test_multi_device_stream_context_manager"),
        ("TestCudaAllocator", "test_allocator_settings"),
    ]),
    ("test_cuda_multigpu", [
        ("TestCudaMultiGPU", "test_cuda_synchronize"),
        ("TestCudaMultiGPU", "test_autogpu"),
        ("TestCudaMultiGPU", "test_copy_device"),
        ("TestCudaMultiGPU", "test_cat_autogpu"),
        ("TestCudaMultiGPU", "test_cuda_set_device"),
        ("TestCudaMultiGPU", "test_current_stream"),
        ("TestCudaMultiGPU", "test_default_stream"),
        ("TestCudaMultiGPU", "test_streams_multi_gpu"),
        ("TestCudaMultiGPU", "test_tensor_device"),
        ("TestCudaMultiGPU", "test_events_wait"),
        ("TestCudaMultiGPU", "test_events_multi_gpu_query"),
        ("TestCudaMultiGPU", "test_mem_get_info"),
        ("TestCudaComm", "test_broadcast_gpu"),
        ("TestCudaComm", "test_reduce_add"),
    ]),
    ("test_autograd", [
        ("TestAutogradDeviceTypeCUDA", "test_min_max_aminmax_median_backprops_to_all_values_cuda"),
    ]),
    ("test_nn", [
        ("TestNNDeviceTypeCUDA", "test_grid_sample_backward_error_checking_cuda"),
    ]),
]

total_passed = 0
total_failed = 0
total_skipped = 0

for module_name, test_cases in suites:
    print(f"--- Running Suite: {module_name}.py ---")
    import importlib
    mod = importlib.import_module(module_name)

    suite = unittest.TestSuite()
    for cls_name, method_name in test_cases:
        cls = getattr(mod, cls_name)
        suite.addTest(cls(method_name))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    total_passed += passed
    total_failed += len(result.failures) + len(result.errors)
    total_skipped += len(result.skipped)

print("\n================================================================")
print(f" ALL SUITES SUMMARY: {total_passed} Passed | {total_skipped} Skipped | {total_failed} Failed")
print("================================================================")
if total_failed > 0:
    sys.exit(1)
PYEOF
'
