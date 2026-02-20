# -*- coding: utf-8 -*-
"""Functions for force calculations using ASE calculators via aiida-pythonjob."""

from typing import Callable, Union
from ase import Atoms
from aiida import orm


def prepare_ase_pythonjob_forces_inputs(
    structure: Union[orm.StructureData, Atoms],
    callback_calculator: Callable,
    pythonjob_code: orm.Code,
    pythonjob_metadata = {'options': {'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 1800}},
    pythonjob_inputs=None,
):
    """Prepare inputs for ASE-based force calculation using aiida-pythonjob.
    
    This function creates the necessary inputs to submit a PythonJob that will
    perform single-point force calculations using ASE calculators.
    
    Parameters
    ----------
    structure : ase.Atoms or orm.StructureData
        The atomic structure for which to calculate forces.
    callback_calculator : callable or ASE calculator instance
        The ASE calculator to use (e.g., MACE, CHGNet, M3GNet).
        Can be a function that returns a calculator or a calculator instance.
    pythonjob_code : orm.Code
        The PythonJob code to use for execution.
    pythonjob_metadata : dict, optional
        Metadata for the PythonJob, including resources and walltime.
    pythonjob_inputs : dict, optional
        Additional inputs for the PythonJob.
    
    Returns
    -------
    dict
        Dictionary with prepared PythonJob inputs ready for submission.
    
    Example
    -------
    >>> from aiida import orm
    >>> from aiida_pythonjob import prepare_pythonjob_inputs
    >>> from mace.calculators import mace_mp
    >>> 
    >>> structure = orm.StructureData(...)
    >>> calculator = mace_mp(model="medium", device="cpu")
    >>> 
    >>> inputs = prepare_ase_forces_inputs(
    ...     structure=structure,
    ...     calculator=calculator,
    ...     pythonjob_inputs={'code': orm.load_code('pythonjob@localhost')}
    ... )
    >>> 
    >>> # Submit the job
    >>> from aiida_pythonjob import PythonJob
    >>> future = submit(PythonJob, **inputs)
    """
    from typing import Any
    from aiida_pythonjob import prepare_pythonjob_inputs, spec
    
    # Prepare default pythonjob inputs if not provided
    pythonjob_inputs_dict = {
        'code': pythonjob_code,
        'metadata': pythonjob_metadata,
    }
    
    # Prepare the function inputs
    function_inputs = {
        'atoms': structure,
    }

    def calculate_forces(atoms, calculator):
        """Calculate forces for an ASE Atoms structure using the specified calculator.
        
        This function works with any ASE-compatible calculator (MLIPs, EMT, GPAW, etc.)
        and returns the forces without modifying the structure.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure for which to calculate forces.
        calculator : ase.calculators.calculator.Calculator or callable
            The ASE calculator to use for forces and energy, or a function that returns one.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'forces': Forces array in eV/Å
            - 'energy': Total energy in eV
            - 'structure': The original structure (unchanged)
        """
        
        # Handle calculator (could be a callable or instance)
        if callable(calculator) and not hasattr(calculator, 'calculate'):
            calculator = calculator()
        
        # Set the calculator
        atoms.calc = calculator
        
        # Extract results
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        
        # Remove calculator from atoms to allow pickling (some calculators aren't picklable)
        atoms.calc = None
        
        # Collect results
        result = {
            'structure': atoms,
            'energy': final_energy,
            'forces': final_forces,
        }
        
        return result

    def forces_function(atoms):
        """Convenience function for ASE-based force calculation.
        
        This is the main function to be called from aiida-pythonjob for force calculations
        using any ASE calculator (MLIPs, DFT codes, empirical potentials, etc.).
        """

        result = calculate_forces(atoms, calculator=callback_calculator)
        return {
            # "structure": result['structure'], 
            "energy": result['energy'], 
            "forces": result['forces'], 
        }

    
    # Prepare the complete pythonjob inputs
    pythonjob_inputs = prepare_pythonjob_inputs(
        function=forces_function,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(energy=Any, forces=Any),
        register_pickle_by_value=True,
        **pythonjob_inputs_dict
    )
    
    return pythonjob_inputs
