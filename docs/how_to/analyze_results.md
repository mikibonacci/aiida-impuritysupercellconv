# How to analyse results

## Checking whether the workflow succeeded

```python
from aiida import orm

node = orm.load_node(<pk>)

if node.is_finished_ok:
    print("Workflow finished successfully")
elif node.is_excepted:
    print("Workflow raised an exception")
else:
    print(f"Exit status {node.exit_status}: {node.exit_message}")
```

## Retrieving the converged supercell

```python
# AiiDA StructureData node
supercell = node.outputs.Converged_supercell

# Export to various formats
print(supercell.get_formula())

py_struc = supercell.get_pymatgen_structure()
py_struc.to(filename="converged_supercell.cif")   # CIF
py_struc.to(filename="POSCAR")                     # POSCAR / VASP

ase_struc = supercell.get_ase()
```

## Retrieving the supercell transformation matrix

```python
sc_matrix_node = node.outputs.Converged_SCmatrix
sc_matrix = sc_matrix_node.get_array('sc_mat')   # 3×3 integer numpy array

print("Supercell matrix:")
print(sc_matrix)
```

The transformation matrix `M` satisfies:

```
A_supercell = M @ A_unitcell
```

where `A` are the rows of the lattice matrix (each row is a lattice vector).

## Understanding the muon position

The muon is added to the supercell as the **last** atom, represented as a
hydrogen (H) species. Its fractional coordinates are:

```python
py_struc = supercell.get_pymatgen_structure()
muon_site = py_struc[-1]                   # last site
print(muon_site.species_string)            # 'H'
print(muon_site.frac_coords)               # fractional coordinates in the supercell
```

The muon is placed at the first Voronoi interstitial site of the unit cell,
with a small +0.001 offset in each fractional coordinate to lift degeneracy.

## Inspecting the calculation provenance

Use `verdi` to explore the full provenance graph:

```bash
verdi node show <pk>
verdi process status <pk>
verdi node graph generate <pk>   # generates a PDF provenance graph
```

From Python:

```python
# List all sub-processes
for called in node.called_descendants:
    print(called.pk, called.process_label, called.exit_status)
```

## Common failure modes

| Exit code | Label | Typical cause | Suggested action |
|---|---|---|---|
| 402 | `ERROR_SUB_PROCESS_FAILED_SCF` | SCF did not converge | Increase `electron_maxstep`, lower `mixing_beta`, check pseudos |
| 403 | `ERROR_RELAXATION_FAILED` | Unit cell relaxation failed | Check structure geometry; try a different `relax_type` |
| 702 | `ERROR_NUM_CONVERGENCE_ITER_EXCEEDED` | Supercell never converged | Increase `max_iter_num`, lower `conv_thr`, check that the structure is reasonable |
| 704 | `ERROR_FITTING_FORCES_TO_EXPONENTIAL` | Force data does not decay exponentially | Check that the SCF forces are physical; inspect forces manually |