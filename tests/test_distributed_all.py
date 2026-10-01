#!/usr/bin/env python3
"""
Comprehensive Distributed & Multi-GPU Test Suite for PyTorch 2.14
Validates:
- Backend availability (Gloo, NCCL, MPI)
- Single-device execution and Autograd on all 3 devices (GPU 0, 1: Tesla K20c; GPU 2: GTX 750 Ti)
- Gloo CPU & CUDA collectives (3 ranks)
- NCCL CUDA collectives:
    * 2 ranks (Kepler <-> Kepler P2P)
    * 3 ranks (Kepler <-> Kepler <-> Maxwell heterogeneous SHM)
- MPI collectives (CPU & CUDA)
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_environment_and_backends():
    print_header("1. PyTorch & Distributed Backend Availability")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    print(f"CUDA Version   : {torch.version.cuda}")
    print(f"Arch List      : {torch.cuda.get_arch_list()}")
    print(f"Device Count   : {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {prop.name} (sm_{cap[0]}{cap[1]}, Memory: {prop.total_memory / (1024**2):.1f} MiB)")
        
    print("\nBackend Check:")
    dist_avail = dist.is_available()
    gloo_avail = dist.is_gloo_available()
    nccl_avail = dist.is_nccl_available()
    mpi_avail = dist.is_mpi_available()
    
    print(f"  torch.distributed.is_available()     : {dist_avail}")
    print(f"  torch.distributed.is_gloo_available(): {gloo_avail}")
    print(f"  torch.distributed.is_nccl_available(): {nccl_avail}")
    print(f"  torch.distributed.is_mpi_available() : {mpi_avail}")
    
    assert dist_avail, "dist.is_available() must be True"
    assert gloo_avail, "dist.is_gloo_available() must be True"
    assert nccl_avail, "dist.is_nccl_available() must be True"
    assert mpi_avail, "dist.is_mpi_available() must be True"
    print(">>> [PASS] All three distributed backends are compiled and available!")

def test_all_individual_devices():
    print_header("2. Individual GPU Execution & Autograd (All Devices)")
    for dev_idx in range(torch.cuda.device_count()):
        dev = f"cuda:{dev_idx}"
        name = torch.cuda.get_device_name(dev_idx)
        print(f"Testing {dev} ({name})...")
        
        # Matrix multiplication
        a = torch.randn(512, 512, device=dev, dtype=torch.float32)
        b = torch.randn(512, 512, device=dev, dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(dev_idx)
        assert c.shape == (512, 512), f"Matmul failed on {dev}"
        
        # Autograd backward
        x = torch.randn(256, 256, device=dev, requires_grad=True)
        y = (x ** 2 + 3 * x).sum()
        y.backward()
        torch.cuda.synchronize(dev_idx)
        assert x.grad is not None, f"Autograd failed on {dev}"
        expected_grad = 2 * x.detach() + 3
        diff = torch.max(torch.abs(x.grad - expected_grad)).item()
        assert diff < 1e-5, f"Grad difference {diff} too high on {dev}"
        
        # Allocations and cleanup
        del a, b, c, x, y, expected_grad
        torch.cuda.empty_cache()
        print(f"  {dev} ({name}): PASS (Compute, Matmul, Autograd)")
        
    print("\nPeer-to-Peer access matrix:")
    n = torch.cuda.device_count()
    for i in range(n):
        for j in range(n):
            if i != j:
                p2p = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU {i} -> GPU {j} P2P access: {p2p}")
    print(">>> [PASS] All individual devices verified successfully!")

# --- Gloo Worker ---
def _gloo_cpu_worker(rank, world_size, init_file):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    
    # 1. All-reduce
    t = torch.tensor([float(rank + 1)], dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    assert t.item() == expected_sum, f"Rank {rank} expected sum {expected_sum}, got {t.item()}"
    
    # 2. Broadcast
    b = torch.tensor([42.0]) if rank == 0 else torch.tensor([0.0])
    dist.broadcast(b, src=0)
    assert b.item() == 42.0, f"Rank {rank} broadcast failed"
    
    # 3. All-gather
    tensor_list = [torch.zeros(1) for _ in range(world_size)]
    my_tensor = torch.tensor([float(rank * 10)])
    dist.all_gather(tensor_list, my_tensor)
    for i, gathered in enumerate(tensor_list):
        assert gathered.item() == float(i * 10), f"Rank {rank} all_gather mismatch at idx {i}"
        
    dist.destroy_process_group()

def _gloo_cuda_worker(rank, world_size, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    
    t = torch.tensor([float(rank + 1)], dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    assert t.item() == expected_sum, f"Gloo CUDA Rank {rank} expected sum {expected_sum}, got {t.item()}"
    
    dist.destroy_process_group()

def test_gloo():
    print_header("3. Gloo Distributed Collective Tests (CPU & CUDA)")
    world_size = min(3, torch.cuda.device_count())
    
    print(f"Testing Gloo CPU Collectives ({world_size} ranks)...")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(_gloo_cpu_worker, args=(world_size, init_file), nprocs=world_size, join=True)
        print("  Gloo CPU (all_reduce, broadcast, all_gather): PASS")
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
            
    print(f"Testing Gloo CUDA Collectives ({world_size} ranks)...")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(_gloo_cuda_worker, args=(world_size, init_file), nprocs=world_size, join=True)
        print("  Gloo CUDA (all_reduce across devices): PASS")
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
    print(">>> [PASS] Gloo backend collectives verified!")

# --- NCCL Worker ---
def _nccl_worker(rank, world_size, device_ids, init_file):
    gpu_id = device_ids[rank]
    torch.cuda.set_device(gpu_id)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    
    # 1. All-reduce
    t = torch.tensor([float(rank + 1)], dtype=torch.float32, device=f"cuda:{gpu_id}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    assert t.item() == expected_sum, f"NCCL Rank {rank} (GPU {gpu_id}) expected {expected_sum}, got {t.item()}"
    
    # 2. Broadcast
    b = torch.tensor([99.0], device=f"cuda:{gpu_id}") if rank == 0 else torch.tensor([0.0], device=f"cuda:{gpu_id}")
    dist.broadcast(b, src=0)
    assert b.item() == 99.0, f"NCCL Rank {rank} broadcast failed"
    
    # 3. All-gather
    gathered = [torch.zeros(1, device=f"cuda:{gpu_id}") for _ in range(world_size)]
    src = torch.tensor([float((rank + 1) * 7)], device=f"cuda:{gpu_id}")
    dist.all_gather(gathered, src)
    for i, g in enumerate(gathered):
        expected_val = float((i + 1) * 7)
        assert g.item() == expected_val, f"NCCL Rank {rank} gathered mismatch: expected {expected_val}, got {g.item()}"
        
    dist.destroy_process_group()

def test_nccl():
    print_header("4. NCCL Distributed Collective Tests (CUDA)")
    
    # Phase A: 2 GPUs - Kepler to Kepler (GPU 0 and 1)
    print("Phase A: NCCL 2 ranks on homogeneous Kepler GPUs (GPU 0 <-> GPU 1, P2P)")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(_nccl_worker, args=(2, [0, 1], init_file), nprocs=2, join=True)
        print("  NCCL Kepler-Kepler (all_reduce, broadcast, all_gather): PASS")
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
            
    # Phase B: 3 GPUs - Heterogeneous Kepler (GPU 0, 1) + Maxwell (GPU 2)
    print("\nPhase B: NCCL 3 ranks on heterogeneous GPUs (GPU 0 [sm_35], GPU 1 [sm_35], GPU 2 [sm_50])")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    try:
        mp.spawn(_nccl_worker, args=(3, [0, 1, 2], init_file), nprocs=3, join=True)
        print("  NCCL Kepler-Maxwell Heterogeneous (all_reduce, broadcast, all_gather): PASS")
    finally:
        if os.path.exists(init_file):
            os.remove(init_file)
    print(">>> [PASS] NCCL backend collectives verified across homogeneous and heterogeneous topologies!")

# --- MPI Worker (invoked via mpirun) ---
def run_mpi_worker():
    dist.init_process_group("mpi")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # CPU all-reduce
    t_cpu = torch.tensor([float(rank + 1)], dtype=torch.float32)
    dist.all_reduce(t_cpu, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    assert t_cpu.item() == expected_sum, f"MPI CPU Rank {rank} expected {expected_sum}, got {t_cpu.item()}"
    
    # CPU broadcast
    b_cpu = torch.tensor([123.0]) if rank == 0 else torch.tensor([0.0])
    dist.broadcast(b_cpu, src=0)
    assert b_cpu.item() == 123.0, f"MPI CPU Rank {rank} broadcast failed"
    
    # CPU all_gather
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([float(rank * 5)]))
    for i, g in enumerate(gathered):
        assert g.item() == float(i * 5)
        
    print(f"  [MPI Rank {rank}/{world_size}] CPU collectives (all_reduce, broadcast, all_gather) SUCCESS")
    
    # CUDA check: test if MPI is CUDA-aware
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        torch.cuda.set_device(rank)
        t_cuda = torch.tensor([float(rank + 1)], dtype=torch.float32, device=f"cuda:{rank}")
        try:
            dist.all_reduce(t_cuda, op=dist.ReduceOp.SUM)
            assert t_cuda.item() == expected_sum
            print(f"  [MPI Rank {rank}/{world_size}] CUDA collective SUCCESS")
        except RuntimeError as e:
            if "CUDA-aware MPI" in str(e):
                print(f"  [MPI Rank {rank}/{world_size}] OpenMPI is non-CUDA-aware (standard distro build) - CPU collectives fully functional")
            else:
                raise
        
    dist.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mpi-worker":
        run_mpi_worker()
    else:
        test_environment_and_backends()
        test_all_individual_devices()
        test_gloo()
        test_nccl()
        print_header("ALL GLOO & NCCL COLLECTIVE TESTS PASSED SUCCESSFULLY!")
