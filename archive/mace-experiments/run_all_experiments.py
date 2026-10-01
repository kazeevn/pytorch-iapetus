import subprocess
import json
import os
import sys

TAGS = [
    "v0.1",
    "v0.2.0",
    "v0.3.0",
    "v0.3.1",
    "v0.3.2",
    "v0.3.3",
    "v0.3.4",
    "v0.3.5",
    "v0.3.6",
    "v0.3.7",
    "v0.3.7b",
    "v0.3.8",
    "v0.3.9",
    "v0.3.10",
    "v0.3.11",
    "v0.3.12",
    "v0.3.13",
    "v0.3.14",
    "v0.3.15",
    "v0.3.16",
    "develop"
]

INNER_SCRIPT = """
import sys
import json
import torch
import numpy as np
from ase.spacegroup import crystal
from ase.optimize import BFGS

result = {
    "success": False,
    "mace_version": "unknown",
    "has_mace_mp": False,
    "calculator_type": "",
    "initial_energy": None,
    "initial_fmax": None,
    "final_energy": None,
    "final_fmax": None,
    "steps": 0,
    "converged": False,
    "peak_vram_mb": 0.0,
    "error": None
}

try:
    import mace
    result["mace_version"] = str(getattr(mace, "__version__", "unknown"))
    
    # Check if mace_mp exists
    try:
        from mace.calculators import mace_mp
        result["has_mace_mp"] = True
    except (ImportError, AttributeError):
        result["has_mace_mp"] = False
        
    unit_cell = crystal(
        symbols=["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90]
    )
    atoms = unit_cell * (2, 1, 1)
    assert len(atoms) == 10
    
    rng = np.random.RandomState(42)
    atoms.positions += rng.normal(0.0, 0.05, atoms.positions.shape)
    
    model_path = "/workspace/model_cache/2023-12-10-mace-128-L0_energy_epoch-249.model"
    device = "cuda"
    
    from mace.calculators import MACECalculator
    try:
        calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
        result["calculator_type"] = "MACECalculator(model_paths)"
    except TypeError:
        # Older MACE (v0.1, v0.2.0) used singular model_path
        calc = MACECalculator(model_path=model_path, device=device)
        result["calculator_type"] = "MACECalculator(model_path)"
        
    atoms.calc = calc
    
    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    fmax0 = float(np.max(np.linalg.norm(f0, axis=1)))
    result["initial_energy"] = e0
    result["initial_fmax"] = fmax0
    
    opt = BFGS(atoms, logfile=None)
    converged = opt.run(fmax=0.05, steps=20)
    
    ef = float(atoms.get_potential_energy())
    ff = atoms.get_forces()
    fmaxf = float(np.max(np.linalg.norm(ff, axis=1)))
    
    result["final_energy"] = ef
    result["final_fmax"] = fmaxf
    result["steps"] = opt.nsteps
    result["converged"] = bool(converged)
    result["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    result["success"] = True
except Exception as e:
    import traceback
    result["error"] = f"{type(e).__name__}: {str(e)}"
    result["traceback"] = traceback.format_exc()

print("JSON_RESULT:" + json.dumps(result))
"""

def test_tag(tag):
    print(f"=== Testing {tag} ===")
    repo_dir = "/home/kna/pytorch-research/mace-repo"
    subprocess.run(["git", "checkout", tag], cwd=repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    cmd = [
        "docker", "run", "--rm",
        "--runtime=nvidia",
        "-e", "NVIDIA_VISIBLE_DEVICES=2",
        "-e", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
        "-e", "PYTHONPATH=/mace-repo",
        "-v", "/home/kna/pytorch-research/mace-experiments:/workspace",
        "-v", "/home/kna/pytorch-research/mace-repo:/mace-repo",
        "mace-test-base:2.7.1",
        "python", "-c", INNER_SCRIPT
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    res = None
    for line in out.splitlines():
        if line.startswith("JSON_RESULT:"):
            res = json.loads(line[len("JSON_RESULT:"):])
            break
            
    if res is None:
        print(f"Failed to parse output for {tag}: {proc.stderr}")
        res = {"success": False, "error": f"Process exited with code {proc.returncode}: {proc.stderr[:200]}"}
    else:
        status = "SUCCESS" if res["success"] else "FAILED"
        print(f"Result for {tag}: {status} (ver={res.get('mace_version')}, steps={res.get('steps')}, final_fmax={res.get('final_fmax')}, error={res.get('error')})")
        
    res["tag"] = tag
    return res

if __name__ == "__main__":
    all_results = []
    for tag in TAGS:
        res = test_tag(tag)
        all_results.append(res)
        
    with open("/home/kna/pytorch-research/mace-experiments/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Reset git repo to develop
    subprocess.run(["git", "checkout", "develop"], cwd="/home/kna/pytorch-research/mace-repo", check=True)
    print("ALL TESTS COMPLETE! Results saved to results.json")
