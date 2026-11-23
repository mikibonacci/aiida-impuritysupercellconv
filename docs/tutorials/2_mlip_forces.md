# Tutorial 2: MLIP force calculations (experimental)

!!! warning "Experimental feature"
    MLIP-based force calculations via `aiida-pythonjob` and ASE calculators are
    **experimental**. The API may change in future releases without prior notice.
    Use for rapid screening only; always validate final results with DFT.

This tutorial shows how to run `IsolatedImpurityWorkChain` with Machine Learning
Interatomic Potential (MLIP) force calculations instead of DFT. This is useful
for quickly estimating a reasonable starting supercell size before committing to
expensive DFT calculations.

## Prerequisites

- `aiida-pythonjob` installed and a `pythonjob` code configured in AiiDA
- An ASE-compatible MLIP installed, e.g. `mace-torch`, `chgnet`, or `mattersim`

```bash
pip install aiida-pythonjob mace-torch
```

---

## How it works

When `ML_forces=True` the workflow replaces the two DFT `PwBaseWorkChain` calls
(with muon, without muon) by two `PythonJob` tasks that run the ASE calculator
on the remote compute resource. All other workflow logic — supercell generation,
convergence checks, iterative enlargement — remains identical.

The user provides a **callback function** that returns an initialised ASE calculator.
The function is pickled and shipped to the worker, so all imports must be inside it.

---

## Step 1 — Define the calculator callback

```python
def get_mace_calculator():
    """Returns a MACE-MP calculator.

    All imports must be inside this function because it is pickled
    and executed on the remote computer.
    """
    import torch
    torch.serialization.add_safe_globals([slice])

    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu", default_dtype="float64")
```

---

## Step 2 — Build the inputs

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

load_profile()

structure = orm.load_node(<structure_pk>)
pythonjob_code = orm.load_code('python3@localhost')  # adjust

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_mace_calculator,
    model_name="mace-mp-medium",          # for provenance labelling
    charge_supercell=True,
)
```

!!! note "No `pw_code` needed"
    When `ML_forces=True` you do not need to supply `pw_code`; DFT builder
    setup is skipped entirely.

---

## Step 3 — Set compute resources for the PythonJob

```python
builder.pythonjob.metadata.options.resources = {
    'num_machines': 1,
    'num_mpiprocs_per_machine': 1,
}
builder.pythonjob.metadata.options.max_wallclock_seconds = 1800
```

---

## Step 4 — Submit and inspect

```python
node = submit(builder)
print(f"Submitted <{node.pk}>")
```

Results access is identical to the DFT case:

```python
node = orm.load_node(<pk>)
supercell = node.outputs.Converged_supercell
sc_matrix = node.outputs.Converged_SCmatrix.get_array('sc_mat')
print(supercell.get_formula(), sc_matrix)
```

---

## Supported MLIP calculators

Any ASE-compatible calculator can be used. The table below shows tested options:

| MLIP | Package | Callback pattern |
|---|---|---|
| MACE-MP | `mace-torch` | `mace_mp(model="medium", ...)` |
| MACE polar | `mace-torch` (nightly) | `mace_polar(model="polar-1-m", ...)` |
| MatterSim | `mattersim` | `MatterSimCalculator(...)` |
| CHGNet | `chgnet` | `CHGNetCalculator()` |

All imports must be **inside** the callback function so they are available on the
remote worker and the closure can be pickled.

---

## Limitations and caveats

- The accuracy of supercell convergence with MLIPs depends heavily on how well the
  chosen model describes your system. Always cross-check with DFT.
- MLIP force calculations do not respect the `conv_thr` in the same physical units as
  DFT; use results as a guide for the supercell size only.
- Charged supercell correction is approximate when using MLIPs (charge is stored in
  `atoms.info["charge"]`; not all MLIP backends honour this).
- The `pythonjob` namespace currently does not support automatic protocol-based setup.
