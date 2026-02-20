# Code Improvements Summary

This document summarizes the improvements made to `aiida-impuritysupercellconv` by adopting best practices from `aiida-muon/find_muon.py`.

## 1. PythonJob Integration (NEW)

### Added Files

```
aiida_impuritysupercellconv/
├── pythonjobs/
│   ├── __init__.py (NEW)
│   └── forces.py (NEW)
└── workflows/
    └── impuritysupercellconv.py (UPDATED)
```

### Import Addition

```python
# NEW import at top of impuritysupercellconv.py
from aiida_pythonjob import PythonJob
```

## 2. Workflow Specification Improvements

### Added PythonJob Namespace

**Before:**
```python
@classmethod
def define(cls, spec):
    super().define(spec)
    spec.input("structure", ...)
    # No pythonjob support
```

**After:**
```python
@classmethod
def define(cls, spec):
    super().define(spec)
    
    # NEW: Expose PythonJob inputs
    spec.expose_inputs(
        PythonJob, 
        namespace='pythonjob',
        namespace_options={
            'required': False, 
            'populate_defaults': False,
            'help': 'Inputs for MLIPs force calculations.',
        },
    )
    spec.input("structure", ...)
```

### Added ML_forces Input Parameter

```python
# NEW input parameter
spec.input(
    "ML_forces",
    valid_type=orm.Bool,
    default=lambda: orm.Bool(False),
    required=False,
    help="Use MLIP (via pythonjob/ASE) for force calculations instead of DFT",
)
```

### Enhanced Workflow Outline

**Before:**
```python
spec.outline(
    if_(cls.should_run_relax)(...),
    cls.init_supcell_gen,
    cls.run_pw_double_scf,  # Always DFT
    cls.inspect_run_get_forces,
    while_(cls.continue_iter)(...),
    cls.set_outputs,
)
```

**After:**
```python
spec.outline(
    if_(cls.should_run_relax)(...),
    cls.init_supcell_gen,
    # NEW: Conditional execution based on ML_forces
    if_(cls.should_run_mlip_forces)(
        cls.run_ase_double_forces,
    ).else_(
        cls.run_pw_double_scf,
    ),
    cls.inspect_run_get_forces,
    while_(cls.continue_iter)(
        cls.increment_n_by_one,
        if_(cls.iteration_num_not_exceeded)(
            cls.get_larger_cell,
            # NEW: Conditional execution in loop too
            if_(cls.should_run_mlip_forces)(
                cls.run_ase_double_forces,
            ).else_(
                cls.run_pw_double_scf,
            ),
            cls.inspect_run_get_forces
        ).else_(
            cls.exit_max_iteration_exceeded,
        ),
    ),
    cls.set_outputs,
)
```

## 3. New Workflow Methods

### should_run_mlip_forces() (NEW)

```python
def should_run_mlip_forces(self):
    """Check if we should run MLIP force calculations."""
    return self.inputs.ML_forces.value
```

**Pattern from find_muon.py:**
```python
def should_run_mlip_relaxation(self):
    """Check if we should run MLIP relaxations."""
    if self.inputs.ML_pre_relax:
        return self.inputs.ML_pre_relax.value
    return False
```

### run_ase_double_forces() (NEW)

```python
def run_ase_double_forces(self):
    """Run ASE force calculations with and without the muon using MLIP."""
    inputs = AttributeDict(self.exposed_inputs(PythonJob, namespace='pythonjob'))
    
    runs = {}
    
    # With muon
    inputs.function_inputs.atoms = self.ctx.sup_struc_mu
    inputs.metadata.call_link_label = f'forces_with_muon_iter{self.ctx.n.value:02d}'
    runs["with_muon"] = self.submit(PythonJob, **inputs)
    self.report(f"Launching PythonJob (PK={runs['with_muon'].pk}) for force calculation with muon")
    
    # Without muon
    inputs.function_inputs.atoms = self.ctx.sup_struc_without_mu
    inputs.metadata.call_link_label = f'forces_without_muon_iter{self.ctx.n.value:02d}'
    runs["without_muon"] = self.submit(PythonJob, **inputs)
    self.report(f"Launching PythonJob (PK={runs['without_muon'].pk}) for force calculation without muon")
    
    return ToContext(**runs)
```

**Pattern from find_muon.py (submit_ase_relaxations):**
```python
def submit_ase_relaxations(self):
    inputs = AttributeDict(self.exposed_inputs(PythonJob, namespace='pythonjob'))
    suffix = "_ase"
    
    for i_index in range(len(self.ctx.supc_list)):
        inputs.function_inputs.atoms = self.ctx.supc_list[i_index]
        inputs.metadata.call_link_label = f'supercell_{i_index:02d}' + suffix
        future = self.submit(PythonJob, **inputs)
        key = f"workchain_{i_index}"
        self.report(f"Launching PythonJob (PK={future.pk}) for supercell structure ...")
        self.to_context(**{key: future})
```

## 4. Improved Error Handling

### inspect_run_get_forces() Improvements

**Before:**
```python
def inspect_run_get_forces(self):
    """Inspect pw run and get forces"""
    self.ctx.traj_out = {}
    for run in ["with_muon", "without_muon"]:
        calculation = self.ctx[run]

        if not calculation.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{calculation.pk}> failed"
                "with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        else:
            self.ctx.traj_out[run] = calculation.outputs.output_trajectory
```

**After:**
```python
def inspect_run_get_forces(self):
    """Inspect calculations and get forces (both DFT and MLIP supported)."""
    self.ctx.traj_out = {}
    
    for run in ["with_muon", "without_muon"]:
        calculation = self.ctx[run]

        if not calculation.is_finished_ok:
            # IMPROVED: Detect calculation type for clearer error messages
            calc_type = "PythonJob" if 'pythonjob' in calculation.process_type else "PwBaseWorkChain"
            self.report(
                f"{calc_type}<{calculation.pk}> failed "
                f"with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        else:
            # NEW: Handle different output types
            if 'pythonjob' in calculation.process_type:
                # MLIP calculation - convert forces to trajectory-like format
                forces_array = calculation.outputs.forces.value
                traj_node = orm.ArrayData()
                traj_node.set_array("forces", np.array([forces_array]))
                self.ctx.traj_out[run] = traj_node
            elif isinstance(calculation, PwBaseWorkChain):
                # DFT calculation - use output_trajectory
                self.ctx.traj_out[run] = calculation.outputs.output_trajectory
            else:
                # IMPROVED: Better error for unknown calculation types
                raise ValueError(f"Unknown calculation type: {calculation.process_type} for uuid={calculation.uuid}.")
```

**Pattern from find_muon.py (collect_relaxed_structures):**
```python
if not workchain.is_finished_ok:
    self.report(
        f"Relaxation calculation {i_index} failed with exit status {workchain.exit_status}"
    )
    n_notf += 1
    if float(n_notf) / len(supercell_list) > 0.4:
        return self.exit_codes.ERROR_RELAX_CALC_FAILED
else:
    uuid = workchain.uuid
    if isinstance(workchain, PwRelaxWorkChain):
        energy = workchain.outputs.output_parameters.get_dict()["energy"]
        rlx_structure = workchain.outputs.output_structure.get_pymatgen_structure()
        new_supercell_list.append(workchain.outputs.output_structure)
    elif 'pythonjob' in workchain.process_type:
        energy = workchain.outputs.energy.value
        rlx_structure = workchain.outputs.structure.get_pymatgen_structure()
        new_supercell_list.append(workchain.outputs.structure)
    else:
        raise ValueError(f"Unknown workchain type: {workchain.process_type} for uuid={uuid}.")
```

## 5. Builder Protocol Enhancement

### get_builder_from_protocol() Updates

**Added Parameters:**
```python
@classmethod
def get_builder_from_protocol(
    cls,
    pw_code: orm.Code = None,  # CHANGED: Now optional
    structure: Union[StructureData, LegacyStructureData] = None,  # CHANGED: Now optional
    # ... existing parameters ...
    ML_forces: bool = False,  # NEW
    pythonjob_code: orm.Code = None,  # NEW
    callback_calculator: callable = None,  # NEW
    additional_pythonjob_inputs: dict = {},  # NEW (optional extras)
    **kwargs,
):
```

**Conditional Builder Setup:**

```python
# NEW: Conditional setup based on ML_forces
if not ML_forces:
    if pw_code is None:
        raise ValueError("pw_code is required when ML_forces is False")
    
    # Setup DFT builders (existing code)
    builder_pwscf = PwBaseWorkChain.get_builder_from_protocol(...)
    builder_relax = PwRelaxWorkChain.get_builder_from_protocol(...)
    # ... set builder attributes ...

# NEW: Setup MLIP if requested
if ML_forces:
    if pythonjob_code is None:
        raise ValueError("pythonjob_code is required when ML_forces is True")
    if callback_calculator is None:
        raise ValueError("callback_calculator is required when ML_forces is True")
    
    from aiida_impuritysupercellconv.pythonjobs.forces import prepare_ase_pythonjob_forces_inputs
    pythonjob_inputs = prepare_ase_pythonjob_forces_inputs(
        structure=structure,
        pythonjob_code=pythonjob_code,
        callback_calculator=callback_calculator,
        **additional_pythonjob_inputs,
    )
    
    builder.pythonjob = pythonjob_inputs

# NEW: Set ML_forces parameter
builder.ML_forces = orm.Bool(ML_forces)
```

**Pattern from find_muon.py:**
```python
if builder.ML_pre_relax:
    from aiida_muon.pythonjobs.relax import prepare_ase_pythonjob_relaxation_inputs
    pythonjob_inputs = prepare_ase_pythonjob_relaxation_inputs(
        structure=structure,
        callback_calculator=callback_calculator,
        pythonjob_code=pythonjob_code,
        **additional_pythonjob_inputs,
    )
    builder.pythonjob = pythonjob_inputs
```

## 6. Force Calculation vs Relaxation

### Key Difference from find_muon.py

**find_muon.py (Relaxation):**
- Purpose: Optimize atomic positions
- Output: Relaxed structure + energy
- ASE operation: `optimizer.run(fmax=...)`

**impuritysupercellconv.py (Force Calculation):**
- Purpose: Calculate forces at fixed positions
- Output: Forces + energy (structure unchanged)
- ASE operation: `atoms.get_forces()`

### Implementation Comparison

**pythonjobs/relax.py (find_muon):**
```python
def optimize_structure(atoms, calculator, fmax=1e-4, optimizer='BFGS', ...):
    atoms.calc = calculator
    dyn = optimizer_class(atoms, **opt_kwargs)
    dyn.run(fmax=fmax_ase)  # Relaxation
    
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    nsteps = dyn.get_number_of_steps()
    
    return {'structure': atoms, 'energy': final_energy, 
            'forces': final_forces, 'nsteps': nsteps}
```

**pythonjobs/forces.py (impuritysupercellconv):**
```python
def calculate_forces(atoms, calculator):
    atoms.calc = calculator
    
    # No relaxation - just calculate
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    
    return {'structure': atoms, 'energy': final_energy, 
            'forces': final_forces}
```

## 7. Output Handling Improvements

### Unified Force Extraction

**Challenge:** DFT uses `output_trajectory`, MLIP uses direct `forces` output.

**Solution:** Convert MLIP forces to trajectory format in `inspect_run_get_forces()`:

```python
if 'pythonjob' in calculation.process_type:
    # MLIP: Create trajectory-like ArrayData
    forces_array = calculation.outputs.forces.value
    traj_node = orm.ArrayData()
    traj_node.set_array("forces", np.array([forces_array]))
    self.ctx.traj_out[run] = traj_node
elif isinstance(calculation, PwBaseWorkChain):
    # DFT: Use existing trajectory
    self.ctx.traj_out[run] = calculation.outputs.output_trajectory
```

**Result:** `check_if_conv_achieved()` works identically for both:
```python
atm_forc_with_muon = traj_out.get_array("forces")[0]
atm_forc_without_muon = traj_out_no_muon.get_array("forces")[0]
# ... convergence check ...
```

## 8. Documentation and Examples

### Added Files

- **`MLIP_SUPPORT.md`**: Comprehensive documentation
- **`examples/example_mlip_forces.py`**: Usage examples for MACE, CHGNet, M3GNet
- **`pythonjobs/forces.py`**: Well-documented force calculation module

### Documentation Features

- Clear usage examples
- Comparison with find_muon.py implementation
- Migration guide for existing users
- Troubleshooting section
- Supported calculators list

## Summary of Improvements

| Category | Improvement | Status |
|----------|-------------|--------|
| MLIP Support | PythonJob integration | ✅ NEW |
| Force Calculations | ASE calculator wrapper | ✅ NEW |
| Workflow Logic | Conditional MLIP/DFT execution | ✅ NEW |
| Error Handling | Type-aware error messages | ✅ IMPROVED |
| Output Handling | Unified force extraction | ✅ IMPROVED |
| Builder Protocol | MLIP parameter support | ✅ ENHANCED |
| Documentation | Usage examples & guide | ✅ NEW |
| Code Quality | Follows find_muon.py patterns | ✅ CONSISTENT |

## Testing Recommendations

1. **DFT workflow**: Verify existing functionality unchanged
2. **MLIP workflow**: Test with MACE, CHGNet, M3GNet
3. **Error handling**: Test failed calculations (both DFT & MLIP)
4. **Mixed iteration**: Verify convergence loop works with MLIP
5. **Output format**: Confirm forces extracted correctly from both sources

## Future Enhancements

Following find_muon.py's evolution, consider:

1. **Pre-clustering**: MLIP screening before DFT
2. **Hybrid modes**: MLIP initial iterations, DFT final
3. **Batch failures**: Handle >40% failure rate gracefully
4. **UUID tracking**: Enhanced debugging like find_muon.py
5. **Offset handling**: Better iteration tracking across restarts
