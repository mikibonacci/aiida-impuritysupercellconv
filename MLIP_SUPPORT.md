# MLIP Support for aiida-impuritysupercellconv

This document describes the MLIP (Machine Learning Interatomic Potential) support added to aiida-impuritysupercellconv, following the implementation pattern from aiida-muon.

## Overview

The package now supports using ASE calculators (MLIPs like MACE, CHGNet, M3GNet, etc.) for force calculations instead of DFT, via `aiida-pythonjob`. This enables:

- **Fast supercell convergence screening** with MLIPs
- **Hybrid workflows**: MLIP pre-convergence followed by DFT refinement  
- **Large system handling** where DFT is computationally prohibitive
- **Flexible calculator choice** - any ASE-compatible calculator works

## New Features

### 1. PythonJob Integration

A new `pythonjobs` module provides force calculation support:

```
aiida_impuritysupercellconv/
├── pythonjobs/
│   ├── __init__.py
│   └── forces.py         # ASE force calculation wrapper
```

### 2. Updated IsolatedImpurityWorkChain

#### New Input Parameters

- **`ML_forces`** (Bool, default=False): Enable MLIP force calculations instead of DFT
- **`pythonjob` namespace**: Exposed inputs for PythonJob configuration

#### New Methods

- **`should_run_mlip_forces()`**: Check if MLIP calculations should be used
- **`run_ase_double_forces()`**: Submit MLIP force calculations (with/without muon)

#### Updated Methods

- **`inspect_run_get_forces()`**: Now handles both DFT and MLIP outputs
  - DFT: Uses `output_trajectory` from PwBaseWorkChain
  - MLIP: Converts PythonJob forces output to trajectory format
  - Improved error handling with calculation type detection

### 3. Enhanced Builder Protocol

The `get_builder_from_protocol()` method now supports:

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    min_length=10.0,
    conv_thr=0.0257,
    max_iter_num=4,
    
    # MLIP-specific parameters
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_calculator_function,  # Direct parameter
    additional_pythonjob_inputs={
        'pythonjob_metadata': {...},  # Optional metadata
    },
    
    # DFT parameters (not needed if ML_forces=True)
    # pw_code=pw_code,
    # pseudo_family="SSSP/1.3/PBE/efficiency",
    # kpoints_distance=0.301,
)
```

### 4. Improved Error Handling

Following best practices from `find_muon.py`:

- **Calculation type detection**: Distinguishes PwBaseWorkChain vs PythonJob
- **Clear error messages**: Reports calculation PK, type, and exit status
- **UUID tracking**: Better debugging with unique identifiers
- **Unified output handling**: Consistent force extraction from different sources

## Usage Examples

### Basic MLIP Usage (MACE)

```python
from aiida import orm
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

# Load structure and code
structure = orm.load_node(STRUCTURE_PK)
pythonjob_code = orm.load_code('pythonjob@localhost')

# Define calculator
def get_mace_calculator():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu")

# Build and submit
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    min_length=10.0,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_mace_calculator,
)

workchain = submit(builder)
```

### Workflow Logic

The workchain now has conditional execution paths:

```
Start
  ↓
Init supercell
  ↓
if ML_forces:                    if not ML_forces:
  run_ase_double_forces()          run_pw_double_scf()
  (MLIP via PythonJob)             (DFT via PwBaseWorkChain)
  ↓                                ↓
inspect_run_get_forces()  ←--------┘
  ↓
Check convergence
  ↓
if not converged:
  Get larger cell
  goto force calculation
  ↓
Output converged supercell
```

## Comparison with aiida-muon

This implementation mirrors the approach in `aiida-muon/workflows/find_muon.py`:

| Feature | find_muon.py | impuritysupercellconv.py |
|---------|--------------|--------------------------|
| PythonJob namespace | ✓ | ✓ (NEW) |
| MLIP input flag | `ML_pre_relax` | `ML_forces` |
| ASE function module | `pythonjobs/relax.py` | `pythonjobs/forces.py` |
| Output handling | Energy + structure | Energy + forces |
| Process type check | `'pythonjob' in process_type` | Same pattern |
| Error handling | Detailed w/ calc type | Improved (NEW) |

### Key Differences

1. **Purpose**:
   - `find_muon.py`: Structure relaxation (minimize forces)
   - `impuritysupercellconv.py`: Force calculation (no relaxation)

2. **Outputs**:
   - `find_muon.py`: Returns relaxed structure + energy
   - `impuritysupercellconv.py`: Returns forces + energy (structure unchanged)

3. **Force Extraction**:
   - Both use `atoms.get_forces()` from ASE
   - `impuritysupercellconv.py` converts to trajectory format for consistency

## Implementation Details

### Force Calculation Function

From `pythonjobs/forces.py`:

```python
def calculate_forces(atoms, calculator):
    """Calculate forces without modifying structure."""
    if callable(calculator) and not hasattr(calculator, 'calculate'):
        calculator = calculator()
    
    atoms.calc = calculator
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    atoms.calc = None  # Remove for pickling
    
    return {
        'structure': atoms,
        'energy': final_energy,
        'forces': final_forces,
    }
```

### Output Unification

The `inspect_run_get_forces()` method normalizes outputs:

```python
if 'pythonjob' in calculation.process_type:
    # MLIP: Convert forces array to trajectory format
    forces_array = calculation.outputs.forces.value
    traj_node = orm.ArrayData()
    traj_node.set_array("forces", np.array([forces_array]))
    self.ctx.traj_out[run] = traj_node
elif isinstance(calculation, PwBaseWorkChain):
    # DFT: Use existing trajectory
    self.ctx.traj_out[run] = calculation.outputs.output_trajectory
```

This ensures `check_if_conv_achieved()` works identically for both DFT and MLIP.

## Supported Calculators

Any ASE-compatible calculator works. Tested/recommended:

- **MACE** (`mace-torch`): Universal ML potential, highly accurate
- **CHGNet** (`chgnet`): Charge-informed neural network
- **M3GNet** (`matgl`): Materials graph network
- **NequIP** (`nequip`): Equivariant neural network
- **Custom**: Any ASE Calculator subclass

## Requirements

- `aiida-pythonjob` >= 0.1.0
- `ase` >= 3.22.0
- MLIP package of choice (e.g., `mace-torch`, `chgnet`, `matgl`)

## Migration Guide

### For Existing Users

No changes needed! Default behavior (DFT force calculations) is unchanged:

```python
# This still works exactly as before
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=structure,
    # ... existing parameters
)
```

### To Enable MLIP

Add three new parameters:

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,  # pw_code no longer required
    ML_forces=True,  # NEW: Enable MLIP
    pythonjob_code=pythonjob_code,  # NEW: PythonJob code
    callback_calculator=get_calculator_function,  # NEW: Calculator callback
    # additional_pythonjob_inputs={...},  # OPTIONAL: Extra settings
)
```

## Future Improvements

Potential enhancements (mirroring find_muon.py evolution):

1. **Pre-clustering with MLIP**: Fast initial screening before DFT
2. **Hybrid convergence**: MLIP for coarse, DFT for final iteration
3. **Automatic calculator selection**: Based on structure composition
4. **Parallel force calculations**: Exploit MLIP speed for multiple structures
5. **Uncertainty quantification**: Use ensemble models for reliability estimates

## Troubleshooting

### "pythonjob_code is required when ML_forces is True"

Ensure you've loaded/created a PythonJob code:

```python
pythonjob_code = orm.load_code('pythonjob@localhost')
```

### "Calculator not found" errors

Install the MLIP package in the PythonJob environment:

```bash
pip install mace-torch  # or chgnet, matgl, etc.
```

### Force array shape mismatch

The MLIP should return forces with shape `(n_atoms, 3)`. Check your calculator returns standard ASE format.

## Contributing

To add new calculator examples or improve the implementation:

1. Follow the pattern in `examples/example_mlip_forces.py`
2. Ensure calculator function is self-contained (all imports inside)
3. Test with various structure types (with/without muon)
4. Document any calculator-specific quirks

## References

- [aiida-muon implementation](../codes/aiida-muon/aiida_muon/workflows/find_muon.py)
- [aiida-pythonjob documentation](https://github.com/aiidateam/aiida-pythonjob)
- [ASE calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
