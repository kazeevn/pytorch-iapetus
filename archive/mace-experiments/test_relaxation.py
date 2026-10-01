import sys
import os
import torch
import numpy as np
from ase.spacegroup import crystal
from ase.optimize import BFGS

def create_10atom_mp_structure():
    # Materials Project mp-5229: SrTiO3 perovskite (Pm-3m, #221)
    # 5 atoms in unit cell, 2x1x1 supercell gives 10 atoms (Sr2 Ti2 O6)
    unit_cell = crystal(
        symbols=['Sr', 'Ti', 'O'],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90]
    )
    atoms = unit_cell * (2, 1, 1)
    assert len(atoms) == 10, f"Expected 10 atoms, got {len(atoms)}"
    
    # Introduce small deterministic displacement to simulate unrelaxed MP structure
    rng = np.random.RandomState(42)
    displacement = rng.normal(0.0, 0.05, atoms.positions.shape)
    atoms.positions += displacement
    return atoms

def run_relaxation(device="cuda:0", model_path=None):
    print(f"PyTorch version: {torch.__version__}")
    print(f"Testing device: {device}")
    
    atoms = create_10atom_mp_structure()
    print(f"Structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    
    try:
        import mace
        mace_ver = getattr(mace, '__version__', 'unknown')
        print(f"MACE version: {mace_ver}")
        
        # Test loading calculator
        if model_path and os.path.exists(model_path):
            from mace.calculators import MACECalculator
            calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
        else:
            from mace.calculators import mace_mp
            calc = mace_mp(model="small", device=device, default_dtype="float32")
            
        atoms.calc = calc
        
        e_init = atoms.get_potential_energy()
        forces_init = atoms.get_forces()
        fmax_init = np.max(np.linalg.norm(forces_init, axis=1))
        print(f"Initial energy: {e_init:.4f} eV, Initial Fmax: {fmax_init:.4f} eV/A")
        
        # Optimize with BFGS
        opt = BFGS(atoms, logfile="-")
        converged = opt.run(fmax=0.05, steps=20)
        
        e_final = atoms.get_potential_energy()
        forces_final = atoms.get_forces()
        fmax_final = np.max(np.linalg.norm(forces_final, axis=1))
        
        if "cuda" in device:
            peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
        else:
            peak_vram = 0.0
            
        print(f"Relaxation status: converged={converged}")
        print(f"Final energy: {e_final:.4f} eV, Final Fmax: {fmax_final:.4f} eV/A")
        print(f"Peak VRAM: {peak_vram:.2f} MB")
        print(f"EXPERIMENT_RESULT: SUCCESS (mace={mace_ver}, steps={opt.nsteps}, final_fmax={fmax_final:.4f})")
        return True
    except Exception as e:
        import traceback
        print("FAILED with exception:")
        traceback.print_exc()
        print(f"EXPERIMENT_RESULT: FAILED ({type(e).__name__}: {e})")
        return False

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_relaxation(device=device, model_path=model_path)
