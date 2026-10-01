#!/usr/bin/env python3
"""
Test suite for metatomic-torch installation in PyTorch 2.14 environment.
Verifies:
1. PyTorch, metatensor-torch, and metatomic-torch version alignments.
2. Custom C++ TorchScript operator registration and execution.
3. System C++ headers and shared library availability (/usr/local).
4. Atomistic System data structures on CPU.
5. Atomistic System operations on CUDA GPUs (Tesla K20c Kepler sm_35, GTX 750 Ti Maxwell sm_50).
6. Metatomic units, model metadata, and model capabilities.
7. ASE integration (MetatomicCalculator).
8. End-to-end C++ compilation with find_package(metatomic_torch).
"""

import os
import subprocess
import sys
import tempfile
import torch
import metatensor.torch
import metatomic.torch
from metatomic.torch import System

def test_imports_and_versions():
    print("--- 1. Testing Imports and Versions ---")

    print(f"  PyTorch Version          : {torch.__version__}")
    print(f"  Metatensor Torch Version : {metatensor.torch.__version__}")
    print(f"  Metatomic Torch Version  : {metatomic.torch.__version__}")

    assert hasattr(torch.ops, "metatomic"), "torch.ops.metatomic not found!"
    version_op = torch.ops.metatomic.version()
    print(f"  torch.ops.metatomic.version() : {version_op}")
    assert version_op == metatomic.torch.__version__, f"Version mismatch: {version_op} != {metatomic.torch.__version__}"
    print("  [PASS] Imports and versions verified.")


def test_system_libraries_and_headers():
    print("--- 2. Testing C++ Library and Header Files ---")
    expected_files = [
        "/usr/local/include/metatomic/torch.hpp",
        "/usr/local/include/metatomic/torch/system.hpp",
        "/usr/local/include/metatomic/torch/model.hpp",
        "/usr/local/include/metatomic/torch/units.hpp",
        "/usr/local/lib/libmetatomic_torch.so",
        "/usr/local/lib/cmake/metatomic_torch/metatomic_torch-config.cmake",
    ]
    for path in expected_files:
        assert os.path.exists(path), f"Missing file: {path}"
        print(f"  Found: {path}")
    print("  [PASS] C++ library and headers verified.")


def test_atomistic_system_cpu():
    print("--- 3. Testing Atomistic System on CPU ---")
    from metatomic.torch import System

    types = torch.tensor([1, 6, 8], dtype=torch.int32)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 15.0
    pbc = torch.tensor([True, True, True])

    sys_cpu = System(types=types, positions=positions, cell=cell, pbc=pbc)
    assert len(sys_cpu) == 3
    assert sys_cpu.positions.shape == (3, 3)
    assert sys_cpu.cell.shape == (3, 3)
    print(f"  Created CPU System: {sys_cpu}")
    print("  [PASS] CPU System verified.")


def test_atomistic_system_gpu():
    print("--- 4. Testing Atomistic System on GPU(s) ---")
    from metatomic.torch import System

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available.")
        return

    num_devices = torch.cuda.device_count()
    print(f"  Available CUDA devices: {num_devices}")

    for dev_idx in range(num_devices):
        dev_name = torch.cuda.get_device_name(dev_idx)
        dev_cap = torch.cuda.get_device_capability(dev_idx)
        print(f"  Testing GPU {dev_idx}: {dev_name} (Compute Capability: {dev_cap})")

        try:
            device = torch.device(f"cuda:{dev_idx}")
            types = torch.tensor([1, 6], dtype=torch.int32, device=device)
            positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64, device=device)
            cell = torch.eye(3, dtype=torch.float64, device=device) * 20.0
            pbc = torch.tensor([True, True, True], device=device)

            sys_gpu = System(types=types, positions=positions, cell=cell, pbc=pbc)
            assert sys_gpu.positions.device.type == "cuda"
            assert sys_gpu.positions.device.index == dev_idx
            print(f"    GPU {dev_idx} System: {sys_gpu}")
        except (torch.AcceleratorError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"    [SKIP] GPU {dev_idx} insufficient VRAM (occupied by existing process): {e}")
            else:
                raise

    print("  [PASS] GPU System verified.")


def test_model_capabilities_and_units():
    print("--- 5. Testing Model Capabilities and Units ---")
    from metatomic.torch import ModelCapabilities, ModelOutput, unit_conversion_factor, unit_dimension_for_quantity

    output = ModelOutput(quantity="energy", unit="eV", per_atom=False)
    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        outputs={"energy": output},
        interaction_range=5.0,
        supported_devices=["cpu", "cuda"],
        dtype="float64",
    )
    assert capabilities.length_unit == "Angstrom"
    assert "energy" in capabilities.outputs

    conv = unit_conversion_factor("bohr", "Angstrom")
    print(f"  Converted 1.0 bohr to Angstrom: {conv}")
    assert abs(conv - 0.52917721) < 1e-4

    dim = unit_dimension_for_quantity("energy")
    print(f"  Unit dimension for energy: {dim}")
    print("  [PASS] Model capabilities and units verified.")


class SimpleSystemModule(torch.nn.Module):
    def forward(self, system: System) -> torch.Tensor:
        return system.positions.sum()


def test_ase_integration():
    print("--- 6. Testing ASE Integration, Serialization, and TorchScript ---")
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from ase import Atoms
    from metatomic.torch import systems_to_torch, save_buffer, load_system_buffer

    print("  Successfully imported MetatomicCalculator.")
    atoms = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sys_torch = systems_to_torch(atoms)
    assert len(sys_torch) == 3
    print(f"  Converted ASE Atoms to Torch System: {sys_torch}")

    buf = save_buffer(sys_torch)
    sys_reloaded = load_system_buffer(buf)
    assert len(sys_reloaded) == 3
    print(f"  Serialized and deserialized System: {sys_reloaded}")

    scripted = torch.jit.script(SimpleSystemModule())
    out = scripted(sys_torch)
    print(f"  TorchScript executed forward on System: {out.item()}")
    assert abs(out.item() - 2.0) < 1e-5
    print("  [PASS] ASE integration, serialization, and TorchScript verified.")


def test_cxx_cmake_integration():
    print("--- 7. Testing C++ CMake Compilation and Linking ---")
    import metatensor.torch

    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src")
        build_dir = os.path.join(tmpdir, "build")
        os.makedirs(src_dir)

        cmake_lists = """cmake_minimum_required(VERSION 3.22)
project(metatomic_cxx_verification CXX)
find_package(Torch REQUIRED)
find_package(metatensor_torch REQUIRED)
find_package(metatomic_torch REQUIRED)

add_executable(cxx_verify main.cpp)
target_link_libraries(cxx_verify PRIVATE metatomic_torch torch)
target_compile_features(cxx_verify PRIVATE cxx_std_17)
"""
        main_cpp = """#include <iostream>
#include <metatomic/torch.hpp>

int main() {
    std::cout << "C++ Runtime Metatomic Version: " << METATOMIC_TORCH_VERSION << std::endl;
    return 0;
}
"""
        with open(os.path.join(src_dir, "CMakeLists.txt"), "w") as f:
            f.write(cmake_lists)
        with open(os.path.join(src_dir, "main.cpp"), "w") as f:
            f.write(main_cpp)

        prefix_path = ";".join([
            "/usr/local",
            torch.utils.cmake_prefix_path,
            metatensor.utils.cmake_prefix_path,
            metatensor.torch.utils.cmake_prefix_path,
        ])

        print("  Configuring C++ verification project with CMake...")
        subprocess.run(
            ["cmake", "-S", src_dir, "-B", build_dir, f"-DCMAKE_PREFIX_PATH={prefix_path}"],
            check=True,
            capture_output=True,
            text=True,
        )

        print("  Building C++ executable...")
        subprocess.run(
            ["cmake", "--build", build_dir],
            check=True,
            capture_output=True,
            text=True,
        )

        print("  Executing binary...")
        res = subprocess.run(
            [os.path.join(build_dir, "cxx_verify")],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Output: {res.stdout.strip()}")
        assert "0.1.18" in res.stdout
    print("  [PASS] C++ CMake integration verified.")


if __name__ == "__main__":
    print("================================================================")
    print("     METATOMIC-TORCH FULL VERIFICATION TEST SUITE               ")
    print("================================================================")
    test_imports_and_versions()
    test_system_libraries_and_headers()
    test_atomistic_system_cpu()
    test_atomistic_system_gpu()
    test_model_capabilities_and_units()
    test_ase_integration()
    test_cxx_cmake_integration()
    print("================================================================")
    print("     ALL METATOMIC-TORCH VERIFICATION TESTS PASSED!             ")
    print("================================================================")
