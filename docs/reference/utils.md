# Reference: Utilities

Internal utility classes used by `IsolatedImpurityWorkChain`. You do not normally
need to interact with these directly, but they are documented here for completeness
and for developers who want to extend the plugin.

---

## `ScGenerators`

**Module:** `aiida_impuritysupercellconv.workflows.utils`

Generates nearly cubic supercells for convergence checks and inserts the muon
impurity at a Voronoi interstitial site.

### Constructor

```python
ScGenerators(py_struc)
```

| Parameter | Type | Description |
|---|---|---|
| `py_struc` | `pymatgen.Structure` | Unit cell structure |

### Static methods

#### `gen_nearcubic_supc(py_struc, min_atoms, max_atoms, min_length)`

Generates a nearly cubic supercell using `pymatgen.CubicSupercellTransformation`.

| Parameter | Type | Description |
|---|---|---|
| `py_struc` | `pymatgen.Structure` | Input structure |
| `min_atoms` | `int` | Minimum number of atoms in the supercell |
| `max_atoms` | `int` or `inf` | Maximum number of atoms allowed |
| `min_length` | `float` | Minimum length of the smallest supercell lattice vector (Å) |

Returns `(supercell_structure, transformation_matrix)`.

#### `append_muon_to_supc(py_scst, sc_mat, mu_frac_coord)`

Appends the muon as a hydrogen atom to the supercell at the given fractional coordinate.
Raises `SystemExit` if the muon is too close to an existing site.

### Instance methods

#### `initialize(min_length=None)`

First supercell generation. Constraints:

- `min_atoms = num_sites + 1`
- `max_atoms = num_sites × 8`
- `min_length` defaults to `min(lattice.abc) + 1` Å

Returns `(supercell_with_mu, sc_matrix, mu_frac_coord)`.

#### `re_initialize(py_scst_with_mu, mu_frac_coord)`

Generates a larger supercell for the next iteration. Uses the current supercell
size as the new lower bound:

- `min_atoms = current_supercell.num_sites + 1`
- `min_length = min(current_supercell.lattice.abc) + 1`

Returns `(supercell_with_mu, sc_matrix)`.

---

## `ChkConvergence`

**Module:** `aiida_impuritysupercellconv.workflows.utils`

Checks whether the supercell is converged using two independent criteria applied
to the differential forces (forces with muon minus forces without muon).

### Constructor

```python
ChkConvergence(
    ase_struc,
    atomic_forces,
    mu_num_spec=1,       # atomic number or symbol identifying the muon
    conv_thr=0.0257,     # eV/Å – convergence threshold
    max_force_thr=0.082, # eV/Å – upper cutoff for forces in the fit
    mnasf=4,             # minimum number of atoms per species needed for fitting
)
```

### Methods

#### `apply_first_crit()`

Checks whether the minimum force on any host atom (excluding the muon and
symmetry-constrained zero forces) is below `conv_thr`.

Returns `True` if converged, `False` otherwise.

#### `apply_2nd_crit()`

For each chemical species, fits the forces as a function of distance from the
muon to an exponential decay `f(r) = A × exp(−r/τ)`. Convergence is achieved
for a species when the maximum muon–atom distance in the supercell exceeds the
distance extrapolated from the fit at `conv_thr`.

Returns a list of booleans, one per species (with ≥ `mnasf` atoms in the structure).
An empty list (no species had sufficient atoms) is treated as not converged.

---

## AiiDA calcfunctions

These are AiiDA provenance-tracked calculation functions. They wrap the Python
utility classes and return AiiDA data nodes.

### `init_supcgen(aiida_struc, min_length)`

| Argument | Type | Description |
|---|---|---|
| `aiida_struc` | `StructureData` | Input unit cell |
| `min_length` | `Float` | Minimum lattice vector for the first supercell |

Returns a dict with:

| Key | Type | Description |
|---|---|---|
| `SC_struc` | `StructureData` | Supercell with muon |
| `SC_struc_without_mu` | `StructureData` | Supercell without muon |
| `SCmat` | `ArrayData` | Transformation matrix (key `'sc_mat'`) |
| `Vor_site` | `ArrayData` | Voronoi site fractional coordinate (key `'Voronoi_site'`) |

### `re_init_supcgen(aiida_struc, ad_scst, vor_site)`

Like `init_supcgen` but generates a larger supercell for the next iteration.
`vor_site` is the `Vor_site` output from `init_supcgen` keeping the muon position fixed.

Returns a dict with `SC_struc`, `SC_struc_without_mu`, `SCmat`.

### `check_if_conv_achieved(aiida_struc, traj_out, traj_out_no_muon, conv_thr)`

Computes differential forces and calls `ChkConvergence.apply_first_crit()` and
`ChkConvergence.apply_2nd_crit()`. Returns an AiiDA `Bool` node.
