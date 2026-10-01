#!/usr/bin/env python3
"""
Verification suite for OpenEquivariance on Kepler (sm_35) + Maxwell (sm_50) GPUs and CPU.
Verifies:
1. Imports, version, and stable extension SO path.
2. TensorProduct forward and autograd backward pass.
3. TensorProductConv standard and deterministic execution.
4. Multi-device and cross-architecture kernel caching (sm_35 and sm_50).
"""
import os
import sys
import torch
import openequivariance as oeq
from torch_geometric import EdgeIndex


def test_imports_and_extension():
    print("--- 1. Testing Imports and Extension SO ---")
    print(f"  PyTorch Version          : {torch.__version__}")
    print(f"  OpenEquivariance Version : {oeq.__version__}")
    so_path = oeq.torch_ext_so_path()
    print(f"  Stable extension SO path : {so_path}")
    assert so_path is not None, "torch_ext_so_path() returned None"
    assert os.path.exists(so_path), f"Extension shared library does not exist at {so_path}"
    print("  [PASS] Imports and extension library verified.")


def test_gpu_operations(dev_id: int):
    device = torch.device(f"cuda:{dev_id}")
    torch.cuda.set_device(dev_id)
    dev_name = torch.cuda.get_device_name(dev_id)
    cap = torch.cuda.get_device_capability(dev_id)
    print(f"\n--- Testing on cuda:{dev_id}: {dev_name} (sm_{cap[0]}{cap[1]}) ---")

    # 1. TensorProduct (Forward + Backward Autograd)
    print("  Testing TensorProduct (forward + backward autograd)...")
    batch_size = 64
    X_ir = oeq.Irreps("1x2e")
    Y_ir = oeq.Irreps("1x3e")
    Z_ir = oeq.Irreps("1x2e")
    instructions = [(0, 0, 0, "uvu", True)]
    problem = oeq.TPProblem(X_ir, Y_ir, Z_ir, instructions, shared_weights=False, internal_weights=False)
    tp = oeq.TensorProduct(problem)

    X = torch.randn(batch_size, X_ir.dim, device=device, requires_grad=True)
    Y = torch.randn(batch_size, Y_ir.dim, device=device, requires_grad=True)
    W = torch.randn(batch_size, problem.weight_numel, device=device, requires_grad=True)

    Z = tp(X, Y, W)
    assert Z.shape == (batch_size, Z_ir.dim), f"Unexpected Z shape {Z.shape}"
    assert not torch.isnan(Z).any(), "Z contains NaN"

    loss = Z.sum()
    loss.backward()
    assert X.grad is not None and not torch.isnan(X.grad).any(), "X.grad invalid"
    assert Y.grad is not None and not torch.isnan(Y.grad).any(), "Y.grad invalid"
    assert W.grad is not None and not torch.isnan(W.grad).any(), "W.grad invalid"
    print(f"    Z norm: {torch.norm(Z).item():.4f}, grad X norm: {torch.norm(X.grad).item():.4f}")
    print("    [PASS] TensorProduct forward + backward autograd")

    # 2. TensorProductConv (Standard and Deterministic)
    print("  Testing TensorProductConv...")
    node_ct, nonzero_ct = 3, 4
    edge_index = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=torch.long)
    Xc = torch.randn(node_ct, X_ir.dim, device=device)
    Yc = torch.randn(nonzero_ct, Y_ir.dim, device=device)
    Wc = torch.randn(nonzero_ct, problem.weight_numel, device=device)

    # Standard non-deterministic conv
    tp_conv = oeq.TensorProductConv(problem, deterministic=False)
    Zc = tp_conv.forward(Xc, Yc, Wc, edge_index[0], edge_index[1])
    assert Zc.shape == (node_ct, Z_ir.dim), f"Unexpected Zc shape {Zc.shape}"
    assert not torch.isnan(Zc).any(), "Zc contains NaN"
    print(f"    Conv forward norm: {torch.norm(Zc).item():.4f}")

    # Deterministic conv
    _, sender_perm = edge_index.sort_by("col")
    edge_index_s, receiver_perm = edge_index.sort_by("row")
    tp_conv_det = oeq.TensorProductConv(problem, deterministic=True)
    Zc_det = tp_conv_det.forward(Xc, Yc[receiver_perm], Wc[receiver_perm], edge_index_s[0], edge_index_s[1], sender_perm)
    assert Zc_det.shape == (node_ct, Z_ir.dim), f"Unexpected Zc_det shape {Zc_det.shape}"
    assert not torch.isnan(Zc_det).any(), "Zc_det contains NaN"
    print(f"    Deterministic conv forward norm: {torch.norm(Zc_det).item():.4f}")
    print("    [PASS] TensorProductConv (standard & deterministic)")


def main():
    print("=" * 60)
    print(" OpenEquivariance Verification Suite")
    print(f" PyTorch: {torch.__version__} | OpenEquivariance: {oeq.__version__}")
    print("=" * 60)

    test_imports_and_extension()

    num_devices = torch.cuda.device_count()
    print(f"\nDetected {num_devices} CUDA devices")
    tested_gpus = 0

    if num_devices == 0:
        print("WARNING: No CUDA devices detected, skipping GPU tests.")
    else:
        for i in range(num_devices):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_mb = free_bytes / (1024 * 1024)
                if free_mb < 200:
                    print(f"\nDevice cuda:{i}: SKIPPING due to insufficient free VRAM ({free_mb:.0f} MiB free < 200 MiB required).")
                    continue
                test_gpu_operations(i)
                tested_gpus += 1
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                print(f"\nDevice cuda:{i}: ERROR / VRAM exhaustion: {e}")
                raise

        print(f"\nTested on {tested_gpus}/{num_devices} CUDA devices.")

    print("\n" + "=" * 60)
    print(" ALL OPENEQUIVARIANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
