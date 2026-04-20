# How to pre-relax the unit cell

By default `IsolatedImpurityWorkChain` takes the input structure as-is and does
not relax lattice parameters or ionic positions before building supercells.
If your starting structure is not converged (e.g. it was obtained from a CIF
database without prior DFT relaxation), you should enable the optional pre-relaxation
step.

## Enabling pre-relaxation

Set `relax_unitcell=True` in `get_builder_from_protocol`:

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw-7.2@localhost'),
    structure=structure,
    relax_unitcell=True,    # <-- enables the pre-relaxation step
)
```

This runs a `PwRelaxWorkChain` with `RelaxType.POSITIONS` (ionic relaxation only,
no cell shape or volume change) before the supercell convergence loop starts. The
relaxed structure is then used as the starting point for all subsequent supercell
generations.

## Customising the relaxation

The pre-relaxation step is exposed under the `relax` namespace. You can override
any `PwRelaxWorkChain` input:

```python
# Example: use tighter electronic convergence for the relaxation
builder.relax.base.pw.parameters = orm.Dict({
    "CONTROL": {"calculation": "relax"},
    "SYSTEM": {"ecutwfc": 60.0, "ecutrho": 480.0},
    "ELECTRONS": {"conv_thr": 1e-8},
    "IONS": {"ion_dynamics": "bfgs"},
})
builder.relax.base.pw.metadata.options.resources = {
    'num_machines': 1, 'num_mpiprocs_per_machine': 8,
}
```

!!! note
    The pre-relaxation uses the same `pseudo_family` as the main SCF steps.
    Setting `base_final_scf` is not supported for the pre-relaxation step and
    is automatically removed from the builder.

## How it fits in the workflow

```
should_run_relax? ─── yes ──► run_relax ──► inspect_relax ──► init_supcell_gen ──► ...
                  └─ no ──────────────────────────────────────► init_supcell_gen ──► ...
```

`should_run_relax` returns `True` only when the `relax.base.pw.parameters` dict
is non-empty. When `relax_unitcell=False` (default), the builder sets such parameters to
an empty dict, effectively skipping the step.

## Exit code

If the pre-relaxation `PwRelaxWorkChain` fails, the workflow exits immediately with:

| Code | Label | Meaning |
|---|---|---|
| 403 | `ERROR_RELAXATION_FAILED` | The `PwRelaxWorkChain` subprocess failed |
