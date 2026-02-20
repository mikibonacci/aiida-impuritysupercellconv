# Quick Reference: MLIP Force Calculations

## Updated API (matching aiida-muon pattern)

### Minimal MLIP Setup

```python
from aiida import orm
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

# Define calculator callback
def get_mace_calculator():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu")

# Build and submit
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    min_length=10.0,
    ML_forces=True,
    pythonjob_code=orm.load_code('pythonjob@localhost'),
    callback_calculator=get_mace_calculator,  # Direct parameter (NEW!)
)

workchain = submit(builder)
```

### With Custom Metadata

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    min_length=10.0,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_mace_calculator,  # Direct parameter
    additional_pythonjob_inputs={            # Optional extras
        'pythonjob_metadata': {
            'options': {
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
                'max_wallclock_seconds': 7200,
            }
        }
    },
)
```

## Comparison with aiida-muon

### aiida-muon (find_muon.py)

```python
builder = FindMuonWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_pre_relax=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_calculator,      # Direct parameter
    additional_pythonjob_inputs={...},       # Optional extras
)
```

### aiida-impuritysupercellconv (UPDATED)

```python
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_calculator,      # Direct parameter (same!)
    additional_pythonjob_inputs={...},       # Optional extras (same!)
)
```

## Key Changes from Initial Implementation

### Before (Old API)
```python
# callback_calculator was inside additional_pythonjob_inputs
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    additional_pythonjob_inputs={
        'callback_calculator': get_calculator,  # ❌ Nested
        'pythonjob_metadata': {...},
    },
)
```

### After (New API - matches aiida-muon)
```python
# callback_calculator is a direct parameter
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    callback_calculator=get_calculator,      # ✅ Direct parameter
    additional_pythonjob_inputs={            # ✅ Only for extras
        'pythonjob_metadata': {...},
    },
)
```

## Required vs Optional Parameters

### For MLIP Force Calculations

**Required:**
- `ML_forces=True`
- `pythonjob_code` (Code instance)
- `callback_calculator` (callable function)
- `structure` (StructureData)

**Optional:**
- `min_length` (float, default computed from structure)
- `conv_thr` (float, default 0.0257 eV/Å)
- `max_iter_num` (int, default 4)
- `charge_supercell` (bool, default True)
- `additional_pythonjob_inputs` (dict, for metadata etc.)

### For DFT Force Calculations (default)

**Required:**
- `pw_code` (Code instance)
- `structure` (StructureData)

**Optional:**
- Same as MLIP + `pseudo_family`, `kpoints_distance`
- `ML_forces=False` (default, can be omitted)

## Calculator Examples

### MACE
```python
def get_mace_calculator():
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu")
```

### CHGNet
```python
def get_chgnet_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator(model_name="0.3.0", use_device="cpu")
```

### M3GNet
```python
def get_m3gnet_calculator():
    from matgl.ext.ase import M3GNetCalculator
    import matgl
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential)
```

## Validation

The workflow validates inputs:

```python
# This will raise ValueError
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    pythonjob_code=pythonjob_code,
    # callback_calculator missing! ❌
)
# Error: "callback_calculator is required when ML_forces is True"

# This will also raise ValueError
builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    structure=structure,
    ML_forces=True,
    callback_calculator=get_calculator,
    # pythonjob_code missing! ❌
)
# Error: "pythonjob_code is required when ML_forces is True"
```

## Migration from Old API

If you used the old API (callback in additional_pythonjob_inputs), update:

```python
# OLD
additional_pythonjob_inputs = {
    'callback_calculator': get_calculator,
    'pythonjob_metadata': {...},
}
builder = ...get_builder_from_protocol(..., additional_pythonjob_inputs=additional_pythonjob_inputs)

# NEW
additional_pythonjob_inputs = {
    'pythonjob_metadata': {...},  # Only extras now
}
builder = ...get_builder_from_protocol(
    ...,
    callback_calculator=get_calculator,  # Promoted to direct parameter
    additional_pythonjob_inputs=additional_pythonjob_inputs,
)
```
