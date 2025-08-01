# -*- coding: utf-8 -*-
"""Example to run the workchain"""
from aiida import load_profile, orm
from aiida.plugins import DataFactory
from aiida.engine import run, submit
from aiida.plugins import DataFactory
from pymatgen.io.cif import CifParser

from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

load_profile()

from aiida.orm import StructureData as LegacyStructureData

#choose the StructureData to be used in the simulation.
structuredata="old"
StructureData = LegacyStructureData
    
system = "Si" #LiF, Si

if __name__ == "__main__":
    parser = CifParser(f"./{system}.cif")
    py_struc = parser.get_structures(primitive=True)[0]
    aiida_structure = StructureData(pymatgen=py_struc)

    builder = IsolatedImpurityWorkChain.get_builder()
    structure = aiida_structure
    builder.structure = structure
    # builder.max_iter_num = orm.Int(3)                  #optional
    # builder.min_length = orm.Float(5.2)                #optional
    # builder.conv_thr = orm.Float(0.1)                #optional
    # builder.kpoints_distance = orm.Float(0.401)        #optional
    ##optional depending if the label is same
    builder.pseudo_family = orm.Str('SSSP/1.3/PBE/efficiency')

    """N:B the pseudos and kpoints are no longer inputs in pwworkchain,
       already taken care of in the musConvworkchain
    """

    codename = "pw-7.2@localhost" # edit
    code = orm.Code.get_from_string(codename)
    builder.pwscf.pw.code = code

    Dict = DataFactory("dict")
    parameters = {
        "CONTROL": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tstress": True,
            "tprnfor": True,
        },
        "SYSTEM": {
            "ecutwfc": 30.0,
            "ecutrho": 240.0,
            "tot_charge": 1.0,
            #'nspin': 2,
            "occupations": "smearing",
            "smearing": "cold",
            "degauss": 0.01,
        },
        "ELECTRONS": {
            "conv_thr": 1.0e-6,
            "electron_maxstep": 300,
            "mixing_beta": 0.3,
        },
    }

    builder.pwscf.pw.parameters = Dict(dict=parameters)
    #
    builder.pwscf.pw.metadata.description = "a PWscf  test SCF"
    builder.pwscf.pw.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }
    #
    # results, node = run.get_node(builder)

    # or

    node = submit(builder)
    # node.exit_status #to check if the calculation was successful
    """
    if node.is_finished:
        if node.is_finished_ok:
            print('Finished successfully')
            res=orm.load_node(node.pk)
            py_conv_struct=res.outputs['Converged_supercell'].get_pymatgen_structure()
            py_conv_struct.to(filename="supercell_withmu.cif".format())
            Sc_matrix=res.outputs['Converged_SCmatrix'].get_array('SCmat')
            print(Sc_matrix)
        else:
            print('Excepted')
    else:
        if node.is_excepted:
            print('Excepted')
        else:
            print ('Not Finished yet')

    # Get  converged supercell results with run
    #print(results) # from #results, node = run.get_node(builder)
    #py_conv_struct=results['Converged_supercell'].get_pymatgen_structure()
    #py_conv_struct.to(filename="supercell_withmu.cif".format())
    #Sc_matrix=results['Converged_SCmatrix'].get_array('SCmat')

    #get results  with submit
    #res=orm.load_node(node.pk)
    #py_conv_struct=res.outputs['Converged_supercell'].get_pymatgen_structure()
    #py_conv_struct.to(filename="supercell_withmu.cif".format())
    #Sc_matrix=res.outputs['Converged_SCmatrix'].get_array('SCmat')
    """
