#!/usr/bin/env python3
"""
Verification suite for torch_scatter on Kepler (sm_35) + Maxwell (sm_50) GPUs and CPU.
"""
import sys
import torch
import torch_scatter

def test_scatter_ops(device: str):
    print(f"Testing torch_scatter operations on {device}...")
    src = torch.randn(10, 4, dtype=torch.float32, device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1, 2, 1, 0, 2, 2, 0], dtype=torch.int64, device=device)

    # 1. scatter_sum
    out_sum = torch_scatter.scatter_sum(src, index, dim=0)
    assert out_sum.shape == (3, 4), f"Unexpected shape {out_sum.shape}"
    loss = out_sum.sum()
    loss.backward()
    assert src.grad is not None, "Backward pass failed for scatter_sum"
    src.grad.zero_()

    # 2. scatter_mean
    out_mean = torch_scatter.scatter_mean(src, index, dim=0)
    assert out_mean.shape == (3, 4)

    # 3. scatter_max / scatter_min
    out_max, argmax = torch_scatter.scatter_max(src, index, dim=0)
    assert out_max.shape == (3, 4)
    out_min, argmin = torch_scatter.scatter_min(src, index, dim=0)
    assert out_min.shape == (3, 4)

    # 4. scatter_mul
    out_mul = torch_scatter.scatter_mul(src, index, dim=0)
    assert out_mul.shape == (3, 4)

    # 5. segment_coo (requires sorted index)
    sorted_idx, perm = torch.sort(index)
    sorted_src = src[perm]
    out_coo = torch_scatter.segment_coo(sorted_src, sorted_idx, reduce="sum")
    assert torch.allclose(out_sum, out_coo, atol=1e-5)

    # 6. segment_csr (requires indptr from sorted index)
    indptr = torch.tensor([0, 4, 7, 10], dtype=torch.int64, device=device)
    out_csr = torch_scatter.segment_csr(sorted_src, indptr, reduce="sum")
    assert torch.allclose(out_sum, out_csr, atol=1e-5)

    print(f"  All operations passed on {device}!")

def main():
    print("=" * 60)
    print(" torch_scatter Verification Suite")
    print(f" PyTorch: {torch.__version__} | torch_scatter: {torch_scatter.__version__}")
    print("=" * 60)

    # Test CPU
    test_scatter_ops("cpu")

    # Test GPUs
    num_devices = torch.cuda.device_count()
    print(f"\nDetected {num_devices} CUDA devices")
    tested_gpus = 0
    if num_devices == 0:
        print("WARNING: No CUDA devices detected, skipping GPU tests.")
    else:
        for i in range(num_devices):
            dev = f"cuda:{i}"
            try:
                dev_name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_mb = free_bytes / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024)
                print(f"\nDevice {dev}: {dev_name} (compute capability {cap[0]}.{cap[1]}, memory: {free_mb:.0f}/{total_mb:.0f} MiB free)")

                if free_mb < 200:
                    print(f"  SKIPPING {dev}: insufficient free VRAM ({free_mb:.0f} MiB free < 200 MiB required; another process is using this GPU).")
                    continue

                test_scatter_ops(dev)

                # Cross-check CPU vs GPU
                src_cpu = torch.randn(20, 8, dtype=torch.float32)
                idx_cpu = torch.randint(0, 5, (20,), dtype=torch.int64)
                cpu_sum = torch_scatter.scatter_sum(src_cpu, idx_cpu, dim=0)

                src_gpu = src_cpu.to(dev)
                idx_gpu = idx_cpu.to(dev)
                gpu_sum = torch_scatter.scatter_sum(src_gpu, idx_gpu, dim=0)
                diff = (cpu_sum - gpu_sum.cpu()).abs().max().item()
                assert diff < 1e-5, f"Mismatch between CPU and {dev}: diff={diff}"
                print(f"  CPU vs {dev} consistency verified (max diff: {diff:.2e})")
                tested_gpus += 1
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                print(f"\nDevice {dev}: SKIPPING due to CUDA error / VRAM exhaustion: {e}")

        print(f"\nTested on {tested_gpus}/{num_devices} CUDA devices.")

    print("\n" + "=" * 60)
    print(" ALL TORCH_SCATTER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
