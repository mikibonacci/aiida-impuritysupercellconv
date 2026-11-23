# How to run with Hubbard DFT+U

For correlated-electron systems (e.g. transition-metal oxides) you may need to
include Hubbard U corrections. `IsolatedImpurityWorkChain` automatically propagates
Hubbard parameters from the unit cell to every generated supercell when the input
structure is a `HubbardStructureData` from `aiida-quantumespresso`.

!!! note "QE version requirement"
    The Hubbard card format used by `HubbardStructureData` requires Quantum ESPRESSO ≥ 7.1.

## Step 1 — Create a HubbardStructureData

```python
from aiida import load_profile, orm
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.hubbard import Hubbard

load_profile()

# Load base structure
from pymatgen.io.cif import CifParser
py_struc = CifParser("NiO.cif").get_structures(primitive=True)[0]
base_structure = orm.StructureData(pymatgen=py_struc)

# Create Hubbard structure and set U values
hubbard_structure = HubbardStructureData.from_structure(base_structure)
hubbard_structure.initialize_onsites_hubbard('Ni', '3d', 5.0, 'U', use_kinds=True)
hubbard_structure.hubbard = Hubbard.from_list(
    hubbard_structure.hubbard.to_list(), projectors="atomic"
)
hubbard_structure.store()
```

## Step 2 — Use the Hubbard structure as input

Pass the `HubbardStructureData` directly as the `structure` input.
`IsolatedImpurityWorkChain` detects the type automatically and populates
Hubbard parameters in every supercell it creates.

```python
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=orm.load_code('pw-7.2@localhost'),
    structure=hubbard_structure,   # <-- HubbardStructureData
)

# QE parameters should include the hp.x / Hubbard card options if needed
# The Hubbard U values are embedded in the HubbardStructureData and
# passed automatically to every PwBaseWorkChain call.
```

## How supercell Hubbard propagation works

The `init_supcgen` and `re_init_supcgen` calcfunctions detect whether the input
structure is a `HubbardStructureData`. If so, they:

1. Extract the existing Hubbard parameters with `check_get_hubbard_u_parms`.
2. Construct the supercell as a `HubbardStructureData` (using
   `HubbardStructureData.from_structure`).
3. Re-apply the same U values to the corresponding species in the supercell.
4. Re-project with `projectors="atomic"`.

Both the muon-bearing and muon-free supercell structures receive the same treatment.