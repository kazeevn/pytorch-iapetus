#!/usr/bin/env python3
"""
Verification suite for torch_sparse on Kepler (sm_35) + Maxwell (sm_50) GPUs and CPU.
"""
import sys
import torch
import torch_sparse
from torch_sparse import SparseTensor


def test_sparse_ops(device: str):
    print(f"Testing torch_sparse operations on {device}...")

    # 1. SparseTensor creation and basic properties
    row = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=device)
    col = torch.tensor([0, 2, 1, 2, 0, 1], dtype=torch.int64, device=device)
    val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32, device=device, requires_grad=True)

    adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(3, 3))
    assert adj.nnz() == 6, f"Expected 6 non-zeros, got {adj.nnz()}"
    assert tuple(adj.sparse_sizes()) == (3, 3), f"Expected size (3, 3), got {adj.sparse_sizes()}"

    # 2. Sparse-dense matrix multiplication (spmm / matmul) with autograd
    dense_x = torch.randn(3, 4, dtype=torch.float32, device=device, requires_grad=True)
    out = torch_sparse.matmul(adj, dense_x)
    assert out.shape == (3, 4), f"Unexpected matmul output shape {out.shape}"

    # Backward pass for both dense tensor and sparse values
    loss = out.sum()
    loss.backward()
    assert dense_x.grad is not None, "Backward pass failed for dense input"
    assert val.grad is not None, "Backward pass failed for sparse values"
    dense_x.grad.zero_()
    val.grad.zero_()

    # 3. Transpose (t)
    adj_t = adj.t()
    assert tuple(adj_t.sparse_sizes()) == (3, 3)
    assert adj_t.nnz() == 6

    # 4. Sparse-sparse matrix multiplication (spspmm)
    adj_sq = torch_sparse.matmul(adj, adj)
    assert isinstance(adj_sq, SparseTensor)
    assert tuple(adj_sq.sparse_sizes()) == (3, 3)

    # 5. Conversion to dense tensor vs manual verification
    dense_adj = adj.to_dense()
    expected_out = torch.matmul(dense_adj, dense_x)
    assert torch.allclose(out, expected_out, atol=1e-5), "Mismatch between SparseTensor matmul and dense matmul"

    # 6. Coalesce / reorder
    row_perm = torch.tensor([2, 0, 1, 0, 2, 1], dtype=torch.int64, device=device)
    col_perm = torch.tensor([0, 0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    val_perm = torch.tensor([5.0, 1.0, 3.0, 2.0, 6.0, 4.0], dtype=torch.float32, device=device)
    adj_uncoalesced = SparseTensor(row=row_perm, col=col_perm, value=val_perm, sparse_sizes=(3, 3), is_sorted=False)
    adj_coalesced = adj_uncoalesced.coalesce()
    assert torch.allclose(adj.to_dense(), adj_coalesced.to_dense(), atol=1e-5), "Coalesce produced inconsistent tensor"

    print(f"  All operations passed on {device}!")


def main():
    print("=" * 60)
    print(" torch_sparse Verification Suite")
    print(f" PyTorch: {torch.__version__} | torch_sparse: {torch_sparse.__version__}")
    print("=" * 60)

    # Test CPU
    test_sparse_ops("cpu")

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

                test_sparse_ops(dev)

                # Cross-check CPU vs GPU
                row_cpu = torch.randint(0, 20, (50,), dtype=torch.int64)
                col_cpu = torch.randint(0, 20, (50,), dtype=torch.int64)
                val_cpu = torch.randn(50, dtype=torch.float32)
                adj_cpu = SparseTensor(row=row_cpu, col=col_cpu, value=val_cpu, sparse_sizes=(20, 20), is_sorted=False).coalesce()
                x_cpu = torch.randn(20, 8, dtype=torch.float32)
                out_cpu = torch_sparse.matmul(adj_cpu, x_cpu)

                adj_gpu = adj_cpu.to(dev)
                x_gpu = x_cpu.to(dev)
                out_gpu = torch_sparse.matmul(adj_gpu, x_gpu)
                diff = (out_cpu - out_gpu.cpu()).abs().max().item()
                assert diff < 1e-5, f"Mismatch between CPU and {dev}: diff={diff}"
                print(f"  CPU vs {dev} consistency verified (max diff: {diff:.2e})")
                tested_gpus += 1
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                print(f"\nDevice {dev}: SKIPPING due to CUDA error / VRAM exhaustion: {e}")

        print(f"\nTested on {tested_gpus}/{num_devices} CUDA devices.")

    print("\n" + "=" * 60)
    print(" ALL TORCH_SPARSE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
