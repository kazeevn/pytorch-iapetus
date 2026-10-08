"""
Test MAGMA GPU linear algebra acceleration in PyTorch 2.14.
Verifies that MAGMA is linked and enabled (torch.cuda.has_magma == True),
and validates non-symmetric eigendecomposition (torch.linalg.eig) and
related dense linear algebra operations on GPU.
"""

import sys
import torch

def test_magma():
    print("=" * 60)
    print(" PyTorch MAGMA Verification Test Suite")
    print(f" PyTorch Version : {torch.__version__}")
    print(f" CUDA Available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f" Device Count    : {device_count}")
    for i in range(device_count):
        print(f"   [{i}] {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_capability(i)})")

    print(f" Intel MKL       : {torch.backends.mkl.is_available()}")
    print(f" Intel oneDNN    : {torch.backends.mkldnn.is_available()}")
    print(f" MAGMA Available : {torch.cuda.has_magma}")
    print("=" * 60)

    assert torch.cuda.has_magma, "FATAL: torch.cuda.has_magma is False! PyTorch was not linked with MAGMA."
    device = torch.device("cuda:0")

    # 1. Test Float32 Non-symmetric Eigendecomposition
    print("\n[Test 1] Float32 non-symmetric eigendecomposition (torch.linalg.eig)...")
    A32 = torch.randn(6, 6, device=device, dtype=torch.float32)
    w32, v32 = torch.linalg.eig(A32)
    diff32 = torch.max(torch.abs(A32.to(torch.complex64) @ v32 - v32 @ torch.diag(w32))).item()
    print(f"  Float32 reconstruction max error ||A*v - v*w||: {diff32:.4e}")
    assert diff32 < 1e-4, f"Error too large: {diff32}"
    print("  -> PASSED")

    # 2. Test Float64 Non-symmetric Eigendecomposition
    print("\n[Test 2] Float64 non-symmetric eigendecomposition (torch.linalg.eig)...")
    A64 = torch.randn(8, 8, device=device, dtype=torch.float64)
    w64, v64 = torch.linalg.eig(A64)
    diff64 = torch.max(torch.abs(A64.to(torch.complex128) @ v64 - v64 @ torch.diag(w64))).item()
    print(f"  Float64 reconstruction max error ||A*v - v*w||: {diff64:.4e}")
    assert diff64 < 1e-11, f"Error too large: {diff64}"
    print("  -> PASSED")

    # 3. Test Batched Float32 & Float64 Eigendecomposition
    print("\n[Test 3] Batched eigendecomposition (shape [10, 5, 5])...")
    Abatch = torch.randn(10, 5, 5, device=device, dtype=torch.float64)
    wbatch, vbatch = torch.linalg.eig(Abatch)
    assert wbatch.shape == (10, 5)
    assert vbatch.shape == (10, 5, 5)
    max_batch_diff = 0.0
    for b in range(10):
        diff = torch.max(torch.abs(Abatch[b].to(torch.complex128) @ vbatch[b] - vbatch[b] @ torch.diag(wbatch[b]))).item()
        max_batch_diff = max(max_batch_diff, diff)
    print(f"  Batched Float64 max error: {max_batch_diff:.4e}")
    assert max_batch_diff < 1e-11, f"Error too large: {max_batch_diff}"
    print("  -> PASSED")

    # 4. Compare GPU eigenvalues with CPU MKL eigenvalues
    print("\n[Test 4] Numerical consistency check (GPU MAGMA vs CPU Intel MKL)...")
    A_sym = torch.randn(12, 12, dtype=torch.float64)
    A_gpu = A_sym.to(device)
    w_gpu, _ = torch.linalg.eig(A_gpu)
    w_cpu, _ = torch.linalg.eig(A_sym)
    # Sort eigenvalues by magnitude for invariant comparison
    sorted_gpu = torch.sort(torch.abs(w_gpu.cpu())).values
    sorted_cpu = torch.sort(torch.abs(w_cpu)).values
    eig_diff = torch.max(torch.abs(sorted_gpu - sorted_cpu)).item()
    print(f"  Max absolute eigenvalue difference (GPU vs CPU): {eig_diff:.4e}")
    assert eig_diff < 1e-11, f"Difference too large: {eig_diff}"
    print("  -> PASSED")

    # 5. Test Linear Solve and Inversion
    print("\n[Test 5] Linear Solve and Matrix Inversion...")
    A_inv = torch.randn(8, 8, device=device, dtype=torch.float64)
    A_inv_res = torch.linalg.inv(A_inv)
    eye_diff = torch.max(torch.abs(A_inv @ A_inv_res - torch.eye(8, device=device, dtype=torch.float64))).item()
    print(f"  Matrix inverse ||A * A^-1 - I||: {eye_diff:.4e}")
    assert eye_diff < 1e-11, f"Inverse error too large: {eye_diff}"

    b = torch.randn(8, 2, device=device, dtype=torch.float64)
    x = torch.linalg.solve(A_inv, b)
    solve_diff = torch.max(torch.abs(A_inv @ x - b)).item()
    print(f"  Linear solve ||A * x - b||: {solve_diff:.4e}")
    assert solve_diff < 1e-11, f"Solve error too large: {solve_diff}"
    print("  -> PASSED")

    # 6. Test Cholesky, QR, and SVD
    print("\n[Test 6] Cholesky, QR, and SVD factorizations...")
    M = torch.randn(10, 10, device=device, dtype=torch.float64)
    SPD = M @ M.T + torch.eye(10, device=device, dtype=torch.float64) * 0.1
    L = torch.linalg.cholesky(SPD)
    chol_diff = torch.max(torch.abs(L @ L.T - SPD)).item()
    print(f"  Cholesky ||L * L^T - SPD||: {chol_diff:.4e}")
    assert chol_diff < 1e-11, f"Cholesky error too large: {chol_diff}"

    Q, R = torch.linalg.qr(M)
    qr_diff = torch.max(torch.abs(Q @ R - M)).item()
    print(f"  QR ||Q * R - M||: {qr_diff:.4e}")
    assert qr_diff < 1e-11, f"QR error too large: {qr_diff}"

    U, S, Vh = torch.linalg.svd(M)
    svd_diff = torch.max(torch.abs(U @ torch.diag(S) @ Vh - M)).item()
    print(f"  SVD ||U * S * Vh - M||: {svd_diff:.4e}")
    assert svd_diff < 1e-11, f"SVD error too large: {svd_diff}"
    print("  -> PASSED")

    print("\n" + "=" * 60)
    print(" ALL MAGMA & LINEAR ALGEBRA TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_magma()
