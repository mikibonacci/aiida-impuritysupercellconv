# Tutorial 1: Supercell convergence for Si and LiF

This tutorial walks you through running `IsolatedImpurityWorkChain` on two simple
test systems — silicon (Si) and lithium fluoride (LiF) — using the convenient
`get_builder_from_protocol` helper.

## Prerequisites

- aiida-impuritysupercellconv installed and AiiDA profile configured
- A Quantum ESPRESSO `pw.x` code configured in AiiDA (e.g. `pw-7.2@localhost`)
- SSSP pseudopotentials installed (`SSSP/1.3/PBE/efficiency`)

---

## Step 1 — Load a structure

You can load a structure from a CIF file using pymatgen and wrap it in an AiiDA
`StructureData` node:

```python
from aiida import load_profile, orm
from pymatgen.io.cif import CifParser

load_profile()

# Load Si or LiF from CIF
parser = CifParser("aiida-impuritysupercellconv/examples/Si.cif")              # or "LiF.cif"
py_struc = parser.get_structures(primitive=True)[0]
structure = orm.StructureData(pymatgen=py_struc)
structure.store()

print(f"Stored structure: PK={structure.pk}, formula={structure.get_formula()}")
```

CIF files for Si and LiF are provided in the `examples/` directory of the repository.

---

## Step 2 — Build the inputs

Use `get_builder_from_protocol` to obtain a pre-configured builder:

```python
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

pw_code = orm.load_code('pw-7.2@localhost')   # adjust to your code label

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=structure,
)

# Optionally override defaults:
# builder.conv_thr          = orm.Float(0.0257)   # eV/Å (default)
# builder.max_iter_num      = orm.Int(4)           # iterations (default)
# builder.min_length        = orm.Float(8.0)       # Å – first supercell minimum vector
# builder.kpoints_distance  = orm.Float(0.301)     # Å⁻¹ (default)
# builder.pseudo_family     = orm.Str('SSSP/1.3/PBE/efficiency')
# builder.charge_supercell  = orm.Bool(True)       # charge +1 for positive muon (default)
```

---

## Step 3 — Adjust metadata

Set the compute resources for the SCF calculations:

```python
resources = {
    'num_machines': 1,
    'num_mpiprocs_per_machine': 4,
}
builder.pwscf.pw.metadata.options.resources = resources
builder.pwscf.pw.metadata.options.max_wallclock_seconds = 3600
```

---

## Step 4 — Submit and monitor

```python
node = submit(builder)
print(f"Submitted IsolatedImpurityWorkChain <{node.pk}>")
```

Monitor progress from the terminal:

```bash
verdi process list -a
verdi process status <pk>
```

---

## Step 5 — Inspect results

Once the workflow finishes successfully:

```python
node = orm.load_node(<pk>)           # replace <pk>

if node.is_finished_ok:
    supercell  = node.outputs.Converged_supercell
    sc_matrix  = node.outputs.Converged_SCmatrix.get_array('sc_mat')

    print("Converged formula:", supercell.get_formula())
    print("Supercell matrix:\n", sc_matrix)

    # Export the converged supercell to CIF
    supercell.get_pymatgen_structure().to(filename="converged_supercell.cif")
else:
    print(f"Workflow failed with exit status: {node.exit_status}")
    print(node.exit_message)
```

---

## Understanding the output

| Output | Type | Description |
|---|---|---|
| `Converged_supercell` | `StructureData` | The converged supercell with the muon (H atom) placed at the Voronoi interstitial site |
| `Converged_SCmatrix` | `ArrayData` | The transformation matrix mapping the unit cell to the converged supercell (array key: `SCmat`) |

The muon is represented as a hydrogen (H) atom appended as the last site. It is placed at the first
Voronoi interstitial site of the primitive cell, slightly offset (`+0.001` in fractional coordinates)
from the exact symmetric position to break symmetry.

---

## Convergence criteria

The workflow applies two independent criteria simultaneously:

1. **Minimum force criterion** — convergence is reached when the smallest host-atom force
   (excluding the muon and symmetry-constrained zero forces) drops below `conv_thr` (default
   0.0257 eV/Å, equivalent to 10⁻³ Ry/Bohr).

2. **Exponential decay criterion** — for each chemical species, the forces are fitted to an
   exponential decay as a function of distance from the muon. Convergence is reached when
   the maximum muon–atom distance in the supercell exceeds the minimum distance required by
   the fit extrapolated to `conv_thr`.

Both criteria must be satisfied for the workflow to stop and return the converged supercell.
If neither is met after `max_iter_num` iterations the workflow exits with code 702.
