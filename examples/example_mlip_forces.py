# -*- coding: utf-8 -*-
"""
Example: Using MLIP (ASE calculators) for force calculations in IsolatedImpurityWorkChain

This example demonstrates how to use Machine Learning Interatomic Potentials (MLIPs)
via ASE calculators and aiida-pythonjob for force calculations instead of DFT.

This is useful for:
- Fast screening of supercell convergence
- Pre-convergence with MLIPs before final DFT calculations
- Large systems where DFT is too expensive for initial supercell size determination
"""

from aiida import orm, load_profile
from aiida.engine import submit
from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain
from pymatgen.io.cif import CifParser


# Load AiiDA profile
load_profile()


def example_mlip_forces(choice):
    """Example using ASE calculators for force calculations."""
    
    # Load or create your structure
    parser = CifParser("/home/jovyan/bind_mount/codes/aiida-impuritysupercellconv/examples/LiF.cif")
    parser = CifParser("/home/jovyan/bind_mount/codes/aiida-muon/examples/data/1000043_CaF2.cif")
    parser = CifParser("/home/jovyan/bind_mount/work/MLIPs_PROJECT/La6Ni4O14.cif")
    structure_pmg = parser.get_structures(primitive=False)[0]
    structure = orm.StructureData(pymatgen=structure_pmg)

    print(structure_pmg)
    
    # This function will be executed on the remote computer
    def get_mace_calculator():
        """Returns a MACE calculator instance.
        
        This function is pickled and sent to the remote computer,
        so all imports must be inside the function.
        """
        import torch
        torch.serialization.add_safe_globals([slice])

        from mace.calculators import mace_mp
        
        # Initialize MACE calculator
        # Options: "small", "medium", "large" or path to custom model
        calculator = mace_mp(
            model="medium",
            device="cpu",  # or "cuda" if GPU available
            default_dtype="float64",
        )
        return calculator

    def get_mace_polar_calculator():
        """Create and return a MACE calculator.
    
        This function will be executed on the remote computer.
        All imports must be inside the function.
        """
        import torch
        torch.serialization.add_safe_globals([slice])
        
        from mace.calculators import mace_polar

        return mace_polar(
            model="polar-1-m",      # Options: small, medium, large
            device="cpu",        # Use 'cuda' if GPU available
            default_dtype="float64",
        )

    def get_mattersim_calculator():
        """Create and return a Mattersim calculator.
    
        This function will be executed on the remote computer.
        All imports must be inside the function.
        
        Note: Works around pkg_resources import in mattersim.__version__.
        """
        import sys
        
        # Workaround for missing pkg_resources: patch mattersim.__version__ module
        # This prevents the import error when mattersim tries to import pkg_resources
        class FakeVersion:
            __version__ = "1.0.0"  # Dummy version
        
        # Pre-create the __version__ module to avoid pkg_resources import
        import types
        version_module = types.ModuleType('mattersim.__version__')
        version_module.__version__ = "1.0.0"
        sys.modules['mattersim.__version__'] = version_module
        
        from mattersim.forcefield import MatterSimCalculator
        mattersim_calculator = MatterSimCalculator(
            load_path="/home/jovyan/bind_mount/codes/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
            #load_path="/home/bonacc_m/Codes/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
            device="cpu",
        )
        return mattersim_calculator

    def get_nequip_calculator():
        """Create and return a NEquIP calculator.
        
        This function will be executed on the remote computer.
        All imports must be inside the function.
        
        Note: Handles PyTorch 2.6 weights_only compatibility issue with e3nn.
        """
        # Fix for PyTorch 2.6: Allow slice in safe globals for e3nn's constants.pt
        import torch
        torch.serialization.add_safe_globals([slice])
        
        from nequip.ase import NequIPCalculator
        nequip_calculator = NequIPCalculator.from_compiled_model(
            compile_path="/home/jovyan/bind_mount/codes/compiled_mp_l_01.nequip.pt2",
            #compile_path="/home/bonacc_m/Codes/compiled_mp_l_01.nequip.pt2",
            device="cpu",
        )
        return nequip_calculator

    choices = {
        "1": [get_mace_calculator, 'python3_mace_p311@localhost'],
        "2": [get_mattersim_calculator, 'python3_mattersim_p311@localhost'],
        "3": [get_nequip_calculator, 'python3_nequip_p311@localhost'],
        "4": [get_mace_polar_calculator, 'python3_mace_p311@localhost'],
        "5": [get_mace_polar_calculator, 'python3_mace_p311@localhost'],
    }
    
    # Get builder with MLIP force calculations enabled
    builder = IsolatedImpurityWorkChain.get_builder_from_protocol(
        structure=structure,
        min_length=10.0,  # Minimum supercell size in Angstrom
        conv_thr=0.0257,  # Force convergence threshold in eV/Å
        max_iter_num=4,   # Maximum iterations for supercell convergence
        charge_supercell= choice == "4",  # Charged supercell (for muon)
        
        # MLIP-specific parameters
        ML_forces=True,  # Enable MLIP force calculations
        pythonjob_code=orm.load_code(choices[choice][1]),  # Code for PythonJob execution
        callback_calculator=choices[choice][0],  # Direct parameter
        
        # Optional: additional pythonjob settings
        # additional_pythonjob_inputs={'pythonjob_metadata': {...}},
        
        # No need for pw_code, pseudo_family, or kpoints_distance with ML_forces=True
    )
    
    # Submit the workchain
    workchain = submit(builder)
    print(f"Submitted IsolatedImpurityWorkChain with PK={workchain.pk} using MLIP forces")
    print(f"To check the status of the calculation: verdi process report {workchain.pk}")
    
    return workchain


if __name__ == "__main__":
    # Run one of the examples
    print("Choose an example:")
    print("1. MACE calculator")
    print("2. Mattersim calculator")
    print("3. NEquIP calculator")
    print("4. MACE Polar calculator (charged supercell)")
    print("5. MACE Polar calculator (neutral supercell)")
    
    choice = input("Enter choice (1-5): ")
    
    example_mlip_forces(choice)
    # elif choice == "2":
    #     example_mlip_forces_chgnet()
    # elif choice == "3":
    #     example_mlip_forces_m3gnet()
    # elif choice == "4":
    #     example_hybrid_mlip_then_dft()
    