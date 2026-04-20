# aiida-impuritysupercellconv

AiiDA plugin for finding the converged supercell size for an interstitial impurity calculation.

## What does it do?

When placing an interstitial impurity (e.g. a muon, hydrogen, or interstitial dopant) in a crystal using DFT, the supercell must be large enough that the impurity's influence on the atomic forces decays to the cell boundary. **aiida-impuritysupercellconv** automates this convergence test.

The main workflow, `IsolatedImpurityWorkChain`, iteratively generates increasingly large, nearly cubic supercells. For each candidate size it runs two single-point SCF calculations — one with the impurity and one without — and checks whether the residual forces (with muon minus without muon) satisfy two convergence criteria:

1. **First criterion**: the minimum force on any host atom falls below the convergence threshold.
2. **Second criterion**: the forces for each chemical species decay exponentially with distance from the impurity, and the supercell is large enough to encompass the required range.

Once both criteria are met the workflow returns the converged supercell together with its transformation matrix.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Get started in minutes**

    ---

    Install the plugin, set up an AiiDA profile and run your first convergence test.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-book-open-variant:{ .lg .middle } **Follow a tutorial**

    ---

    Step-by-step walkthroughs for Si and LiF using the `get_builder_from_protocol` helper.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-frequently-asked-questions:{ .lg .middle } **How-to guides**

    ---

    Focused recipes: Hubbard DFT+U, pre-relaxation of the unit cell, result analysis, and more.

    [:octicons-arrow-right-24: How-to guides](how_to/index.md)

-   :material-file-document-outline:{ .lg .middle } **Reference**

    ---

    Full documentation of all inputs, outputs and exit codes.

    [:octicons-arrow-right-24: Reference](reference/workflows.md)

</div>

## Quick example

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

load_profile()

# Load structure
structure = orm.load_node(<pk>)  # an AiiDA StructureData node
pw_code = orm.load_code('pw-7.2@localhost')

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=structure,
)

node = submit(builder)
print(f"Submitted IsolatedImpurityWorkChain <{node.pk}>")
```

Once finished, inspect results:

```python
node = orm.load_node(<pk>)

supercell = node.outputs.Converged_supercell
sc_matrix = node.outputs.Converged_SCmatrix.get_array('sc_mat')

print(supercell.get_formula())
print(sc_matrix)
```
