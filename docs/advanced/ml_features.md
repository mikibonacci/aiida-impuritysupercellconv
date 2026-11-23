# MLIP force calculations (experimental)

!!! warning "Experimental feature"
    Everything on this page describes an **experimental** capability. The API,
    data flow, and supported calculators may change without prior notice in future
    releases. Use MLIP forces for rapid screening only; always validate
    supercell convergence with DFT before using a result in production.

---

## Architecture

When `ML_forces=True` the workflow replaces the two `PwBaseWorkChain` SCF calls
with two `PythonJob` tasks from `aiida-pythonjob`. All other logic remains
unchanged: supercell generation, Voronoi muon placement, iterative enlargement,
and convergence checking all work identically.

```
Normal DFT path:          MLIP path:
  run_pw_double_scf  →    run_ase_double_forces
  (PwBaseWorkChain)       (PythonJob × 2)
```

The `PythonJob` executes a Python closure (`forces_function`) on the remote
compute resource. The closure:

1. Receives an ASE `Atoms` structure as input.
2. Calls the user-supplied `callback_calculator()` to instantiate the MLIP.
3. Attaches the calculator to the structure and evaluates forces.
4. Returns `{'energy': float, 'forces': list}` (plain Python types for
   pickle safety across numpy versions).

---

## The callback calculator pattern

All imports must be inside the callback function because the closure is pickled
and shipped to the remote worker. The function must be importable without side
effects.

```python
def get_mace_calculator():
    import torch
    torch.serialization.add_safe_globals([slice])   # required for MACE
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu", default_dtype="float64")
```

Pass it to `get_builder_from_protocol`:

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_mace_calculator,
    model_name="mace-mp-medium",
)
```

The `model_name` string is stored in the PythonJob for provenance labelling.

---

## Supported MLIP calculators

Any ASE calculator can be used. The table shows examples that have been tested:

| Model | Package | Notes |
|---|---|---|
| MACE-MP | `mace-torch` | Universal potential, general-purpose |
| MACE-polar | `mace-torch` (nightly) | Polar materials variant |
| MatterSim | `mattersim` | Works around `pkg_resources` import in `__version__` |
| CHGNet | `chgnet` | Materials Project universal potential |
| EMT | `ase` (built-in) | Testing only, not physically meaningful |

---

## Adding a new MLIP calculator

1. Write a callback function following the pattern above.
2. Make sure all imports are inside the function body.
3. If the model has non-picklable internal state, consider wrapping it in a
   factory function that re-creates the calculator on each call.
4. Test locally:

   ```python
   calc = get_my_calculator()
   from ase.build import bulk
   atoms = bulk("Si") * 2
   atoms.calc = calc
   print(atoms.get_forces())
   ```

---

## Charge handling with MLIPs

When `charge_supercell=True` the workflow sets:

```python
atoms.info["charge"] = 1
```

Not all MLIP backends read `atoms.info["charge"]`. Models that support
charge-aware evaluation (e.g. some versions of MACE) may handle this correctly;
others will silently ignore it. Always check the documentation of your chosen MLIP
if you need a physically meaningful charged supercell.

---

## `prepare_ase_pythonjob_forces_inputs`

**Module:** `aiida_impuritysupercellconv.pythonjobs.forces`

Helper function that prepares the complete `pythonjob` input dictionary for one
force calculation run. Called internally by `get_builder_from_protocol` when
`ML_forces=True`.

```python
from aiida_impuritysupercellconv.pythonjobs.forces import prepare_ase_pythonjob_forces_inputs

inputs = prepare_ase_pythonjob_forces_inputs(
    structure=structure,
    callback_calculator=get_mace_calculator,
    pythonjob_code=pythonjob_code,
    pythonjob_metadata={
        'options': {
            'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
            'max_wallclock_seconds': 1800,
        }
    },
    charged_supercell=True,
    model_name="mace-mp-medium",
)
```

| Parameter | Type | Description |
|---|---|---|
| `structure` | `StructureData` or `ase.Atoms` | Input structure |
| `callback_calculator` | callable | Function returning an ASE calculator |
| `pythonjob_code` | `orm.Code` | The configured PythonJob code |
| `pythonjob_metadata` | dict | Scheduler options for the PythonJob |
| `pythonjob_inputs` | dict, optional | Additional pythonjob inputs |
| `charged_supercell` | bool | If `True`, sets `atoms.info["charge"] = 1` |
| `model_name` | str, optional | Label stored in output for provenance |

Returns a dict ready to assign to `builder.pythonjob`.

---

## Known limitations

- The `pythonjob` namespace has no protocol-based setup (no `get_builder_from_protocol` for `PythonJob`); resource configuration must be done manually.
- There is no automatic restart if an MLIP `PythonJob` fails mid-iteration. The full `IsolatedImpurityWorkChain` must be resubmitted.
- Force units from ASE calculators are assumed to be eV/Å; ensure your MLIP returns forces in these units.
- Models requiring GPU execution need a compute resource with a GPU and the appropriate `pythonjob` code configured for that environment.
