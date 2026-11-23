# How to use the protocol builder

`IsolatedImpurityWorkChain.get_builder_from_protocol` is the recommended way to
configure a run. It pre-populates sensible defaults for DFT settings and exposes
all tunable parameters as explicit keyword arguments.

## Minimal call (DFT)

```python
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw-7.2@localhost'),
    structure=structure,
)
```

## All parameters explained

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw-7.2@localhost'),
    structure=structure,

    # ── Supercell generation ─────────────────────────────────────────────────
    min_length=None,        # float, Å. Minimum lattice vector of the first supercell.
                            #   Default: smallest lattice vector of the unit cell + 1 Å.
    max_iter_num=4,         # int. Maximum iterations before giving up (exit 702).

    # ── Convergence ──────────────────────────────────────────────────────────
    conv_thr=0.0257,        # float, eV/Å. Force convergence threshold.
                            #   Default ≈ 10⁻³ Ry/Bohr (standard DFT accuracy).

    # ── DFT settings ─────────────────────────────────────────────────────────
    pseudo_family='SSSP/1.3/PBE/efficiency',  # pseudo family label
    kpoints_distance=0.301, # float, Å⁻¹. k-point mesh density.
    charge_supercell=True,  # bool. Charge the supercell +1 for a positive muon.
    protocol=None,          # str. Protocol for PwBaseWorkChain (e.g. 'fast', 'moderate').
    overrides=None,         # dict. Manual overrides for the inner pw builder.
    options=None,           # dict. Metadata options (e.g. resources, walltime).

    # ── Optional pre-relaxation ───────────────────────────────────────────────
    relax_unitcell=False,   # bool. Relax the defect-free unit cell first.

    # ── MLIP (experimental) ───────────────────────────────────────────────────
    ML_forces=False,        # bool. Use MLIP instead of DFT for force calculations.
    pythonjob_code=None,    # orm.Code. Required when ML_forces=True.
    callback_calculator=None, # callable. Returns an ASE calculator. Required when ML_forces=True.
    model_name=None,        # str. Label for the MLIP model (stored for provenance).
    additional_pythonjob_inputs={},  # dict. Extra kwargs for pythonjob setup.
)
```

## Overriding DFT parameters

Pass an `overrides` dict that will be merged into the inner `PwBaseWorkChain` builder:

```python
overrides = {
    "base": {
        "pw": {
            "parameters": {
                "SYSTEM": {
                    "ecutwfc": 60.0,
                    "ecutrho": 480.0,
                },
                "ELECTRONS": {
                    "conv_thr": 1.0e-8,
                },
            }
        }
    }
}

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=structure,
    overrides=overrides,
)
```

## Setting compute resources

```python
resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 8}
builder.pwscf.pw.metadata.options.resources = resources
builder.pwscf.pw.metadata.options.max_wallclock_seconds = 7200
```

## Using `get_builder` directly

For full manual control, use the low-level `get_builder`:

```python
from aiida.plugins import WorkflowFactory

IsolatedImpurityWorkChain = WorkflowFactory('impuritysupercellconv')
builder = IsolatedImpurityWorkChain.get_builder()

builder.structure       = structure
builder.conv_thr        = orm.Float(0.0257)
builder.max_iter_num    = orm.Int(4)
builder.kpoints_distance = orm.Float(0.301)
builder.pseudo_family   = orm.Str('SSSP/1.3/PBE/efficiency')
builder.charge_supercell = orm.Bool(True)

# Manually configure the inner PwBaseWorkChain
builder.pwscf.pw.code       = orm.load_code('pw-7.2@localhost')
builder.pwscf.pw.parameters = orm.Dict({...})
builder.pwscf.pw.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 4}
```
