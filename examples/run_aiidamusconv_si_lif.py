# -*- coding: utf-8 -*-
"""Example to run the workchain"""
from aiida import load_profile, orm
from pymatgen.io.cif import CifParser

load_profile()

# Load Si or LiF from CIF
parser = CifParser("aiida-impuritysupercellconv/examples/Si.cif")              # or "LiF.cif"
py_struc = parser.get_structures(primitive=True)[0]
structure = orm.StructureData(pymatgen=py_struc)
structure.store()

print(f"Stored structure: PK={structure.pk}, formula={structure.get_formula()}")

from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

pw_code = orm.load_code('pw-7.2@localhost')   # adjust to your code label

builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
    pw_code=pw_code,
    structure=structure,
)

resources = {
    'num_machines': 1,
    'num_mpiprocs_per_machine': 4,
}
builder.pwscf.pw.metadata.options.resources = resources
builder.pwscf.pw.metadata.options.max_wallclock_seconds = 3600

node = submit(builder)
print(f"Submitted IsolatedImpurityWorkChain <{node.pk}>")
print(f"Monitor with:  verdi process status {node.pk}")