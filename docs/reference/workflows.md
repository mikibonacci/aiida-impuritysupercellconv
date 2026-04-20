# Reference: IsolatedImpurityWorkChain

**Entry point:** `impuritysupercellconv`

```python
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain
# or
from aiida.plugins import WorkflowFactory
IsolatedImpurityWorkChain = WorkflowFactory('impuritysupercellconv')
```

---

## Inputs

### Top-level inputs

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `structure` | `StructureData` / `HubbardStructureData` | Yes | — | Input unit cell. If `HubbardStructureData`, Hubbard U parameters are propagated to supercells automatically. |
| `min_length` | `Float` | No | `None` | Minimum lattice vector (Å) for the first generated supercell. If `None`, set to the smallest unit cell vector + 1 Å. |
| `conv_thr` | `Float` | No | `0.0257` | Force convergence threshold in eV/Å (default ≈ 10⁻³ Ry/Bohr). |
| `max_iter_num` | `Int` | No | `4` | Maximum number of supercell enlargement iterations before the workflow exits with code 702. |
| `kpoints_distance` | `Float` | No | `0.301` | Minimum k-point spacing in Å⁻¹ for the k-mesh generation. |
| `pseudo_family` | `Str` | No | `'SSSP/1.3/PBE/efficiency'` | Label of the AiiDA pseudopotential family. |
| `charge_supercell` | `Bool` | No | `True` | If `True`, apply a +1 charge to the supercell (appropriate for positively charged muon). |
| `ML_forces` | `Bool` | No | `False` | If `True`, replace DFT SCF with MLIP force calculations via `aiida-pythonjob`. **Experimental.** |
| `relax_unitcell` | `Bool` | No | `False` | If `True`, relax the defect-free unit cell with `PwRelaxWorkChain` before supercell generation. |

### Exposed namespace: `pwscf`

Exposes inputs of `PwBaseWorkChain` (excluding `pw.structure` and `kpoints`).
Used for all single-point SCF force calculations.

Key sub-inputs:

| Name | Type | Description |
|---|---|---|
| `pwscf.pw.code` | `Code` | Quantum ESPRESSO `pw.x` code |
| `pwscf.pw.parameters` | `Dict` | QE input parameters (CONTROL, SYSTEM, ELECTRONS, …) |
| `pwscf.pw.metadata.options` | dict | Scheduler options (resources, walltime, etc.) |

### Exposed namespace: `relax`

Exposes inputs of `PwRelaxWorkChain` (excluding `structure` and `base_final_scf`).
Used only when `relax_unitcell=True`.

### Exposed namespace: `pythonjob` *(experimental)*

Exposes inputs of `PythonJob` from `aiida-pythonjob`.
Used only when `ML_forces=True`. Should be set via
`get_builder_from_protocol(ML_forces=True, ...)` or populated manually by
`prepare_ase_pythonjob_forces_inputs`.

---

## Outputs

| Name | Type | Required | Description |
|---|---|---|---|
| `Converged_supercell` | `StructureData` | Yes | The converged supercell with the muon inserted as the last H atom at the Voronoi interstitial site. |
| `Converged_SCmatrix` | `ArrayData` | Yes | The 3×3 integer transformation matrix from unit cell to converged supercell (key: `'SCmat'`). |

---

## Exit codes

| Code | Label | Description |
|---|---|---|
| 402 | `ERROR_SUB_PROCESS_FAILED_SCF` | A `PwBaseWorkChain` subprocess failed. |
| 403 | `ERROR_RELAXATION_FAILED` | The `PwRelaxWorkChain` pre-relaxation failed. |
| 702 | `ERROR_NUM_CONVERGENCE_ITER_EXCEEDED` | Maximum number of supercell convergence iterations reached without satisfying both convergence criteria. |
| 704 | `ERROR_FITTING_FORCES_TO_EXPONENTIAL` | The force data could not be fitted to an exponential decay (force data may not decay monotonically with distance). |

---

## Protocol builder

```python
IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code,
    structure,
    protocol=None,
    overrides=None,
    relax_unitcell=False,
    options=None,
    min_length=None,
    conv_thr=0.0257,
    kpoints_distance=0.301,
    charge_supercell=True,
    pseudo_family='SSSP/1.3/PBE/efficiency',
    max_iter_num=4,
    ML_forces=False,
    pythonjob_code=None,
    callback_calculator=None,
    model_name=None,
    additional_pythonjob_inputs={},
)
```

---

## Workflow outline

```
if should_run_relax:
    run_relax → inspect_relax

init_supcell_gen

if should_run_mlip_forces:
    run_ase_double_forces
else:
    run_pw_double_scf

inspect_run_get_forces

while continue_iter:
    increment_n_by_one
    if iteration_num_not_exceeded:
        get_larger_cell
        if should_run_mlip_forces:
            run_ase_double_forces
        else:
            run_pw_double_scf
        inspect_run_get_forces
    else:
        exit_max_iteration_exceeded   → exit 702

set_outputs
```
