# -*- coding: utf-8 -*-
"""Tests for the `IsolatedImpurityWorkChain` class."""
import pytest
from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain

try:
    from aiida_pythonjob import PythonJob
    HAS_PYTHONJOB = True
except ImportError:
    HAS_PYTHONJOB = False
    PythonJob = None


@pytest.fixture
def generate_builder(generate_structure, fixture_code):
    """Generate default inputs for `IsolatedImpurityWorkChain`"""

    def _get_builder():
        """Generate default builder for `IsolatedImpurityWorkChain`"""

        inputstructure = generate_structure("Si")
        # code = fixture_code("quantumespresso.pw")

        builder = IsolatedImpurityWorkChain.get_builder()
        builder.structure = inputstructure
        # pwscf is optional (required=False); do not set it in the minimal builder

        return builder

    return _get_builder


@pytest.fixture
def generate_workchain(generate_builder):
    """Generate an instance of IsolatedImpurityWorkChain"""

    def _generate_workchain(exit_code=None):
        builder = generate_builder()
        runner = get_manager().get_runner()
        process = instantiate_process(runner, builder)

        # if exit_code is not None:
        #    node = generate_calc_job_node(
        #    entry_point_calc_job, fixture_localhost, test_name, inputs["IsolatedImpurityWorkChain"]
        #    )
        #    node.set_process_state(ProcessState.FINISHED)
        #    node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain


def test_initialize(aiida_profile, generate_workchain):
    """
    Test `IsolatedImpurityWorkChain.initialization`.
    This checks that we can create the workchain successfully,
     and that it is initialised into the correct state.
    """
    process = generate_workchain()
    assert process.init_supcell_gen() is None
    assert process.ctx.n.value == 0
    assert isinstance(process.ctx.sup_struc_mu, orm.StructureData)
    assert isinstance(process.ctx.musite, orm.ArrayData)
    assert isinstance(process.ctx.sc_mat, orm.ArrayData)


# ---------------------------------------------------------------------------
# Spec tests (no AiiDA profile needed – just class-level introspection)
# ---------------------------------------------------------------------------

def test_spec_exit_codes():
    """All expected exit codes must be registered on the spec."""
    exit_codes = IsolatedImpurityWorkChain.spec().exit_codes
    assert exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.status == 402
    assert exit_codes.ERROR_RELAXATION_FAILED.status == 403
    assert exit_codes.ERROR_NUM_CONVERGENCE_ITER_EXCEEDED.status == 702
    assert exit_codes.ERROR_FITTING_FORCES_TO_EXPONENTIAL.status == 704


def test_spec_outputs():
    """Required outputs must be defined in the spec."""
    outputs = IsolatedImpurityWorkChain.spec().outputs
    assert "Converged_supercell" in outputs
    assert "Converged_SCmatrix" in outputs


def test_spec_required_inputs():
    """'structure' must be a required input."""
    inputs_spec = IsolatedImpurityWorkChain.spec().inputs
    assert "structure" in inputs_spec


def test_spec_default_conv_thr():
    """conv_thr default should be 0.0257 eV/Å."""
    inputs_spec = IsolatedImpurityWorkChain.spec().inputs
    default_val = inputs_spec["conv_thr"].default()
    assert abs(default_val.value - 0.0257) < 1e-6


def test_spec_default_charge_supercell():
    """charge_supercell should default to True (positive muon)."""
    inputs_spec = IsolatedImpurityWorkChain.spec().inputs
    assert inputs_spec["charge_supercell"].default().value is True

@pytest.mark.skipif(not HAS_PYTHONJOB, reason="Requires aiida-pythonjob to test ML_forces input")
def test_spec_default_ml_forces():
    """ML_forces should default to False."""
    inputs_spec = IsolatedImpurityWorkChain.spec().inputs
    assert inputs_spec["ML_forces"].default().value is False


# ---------------------------------------------------------------------------
# Workflow condition methods
# ---------------------------------------------------------------------------

def test_should_run_relax_false_without_relax_inputs(aiida_profile, generate_workchain):
    """Without relax namespace inputs, should_run_relax must return False."""
    process = generate_workchain()
    assert process.should_run_relax() is False

@pytest.mark.skipif(not HAS_PYTHONJOB, reason="Requires aiida-pythonjob to test ML_forces input")
def test_should_run_mlip_false_by_default(aiida_profile, generate_workchain):
    """ML_forces defaults to False, so should_run_mlip_forces must return False."""
    process = generate_workchain()
    assert process.should_run_mlip_forces() is False


def test_iteration_num_not_exceeded_at_zero(aiida_profile, generate_workchain):
    """At n=0, iteration_num_not_exceeded must return True."""
    process = generate_workchain()
    process.init_supcell_gen()  # sets ctx.n = 0
    assert process.iteration_num_not_exceeded().value is True


# ---------------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------------

def test_get_builder_from_protocol_requires_pw_code_for_dft():
    """get_builder_from_protocol should raise ValueError when ML_forces=False and pw_code is None."""
    from aiida.orm import StructureData

    structure = StructureData()
    structure.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols="Si")

    with pytest.raises(ValueError, match="pw_code"):
        IsolatedImpurityWorkChain.get_builder_from_protocol(
            pw_code=None,
            structure=structure,
            ML_forces=False,
        )

@pytest.mark.skipif(not HAS_PYTHONJOB, reason="Requires aiida-pythonjob to test ML_forces input")
def test_get_builder_from_protocol_requires_pythonjob_code_for_mlip():
    """get_builder_from_protocol should raise ValueError when ML_forces=True and pythonjob_code is None."""
    from aiida.orm import StructureData

    structure = StructureData()
    structure.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols="Si")

    with pytest.raises(ValueError, match="pythonjob_code"):
        IsolatedImpurityWorkChain.get_builder_from_protocol(
            structure=structure,
            ML_forces=True,
            pythonjob_code=None,
            callback_calculator=lambda: None,
        )


def test_get_builder_from_protocol_requires_callback_for_mlip():
    """get_builder_from_protocol should raise ValueError when ML_forces=True and callback_calculator is None."""
    from aiida.orm import StructureData

    structure = StructureData()
    structure.set_cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols="Si")

    # Use a dummy code object to pass the pythonjob_code check
    with pytest.raises(ValueError, match="callback_calculator"):
        IsolatedImpurityWorkChain.get_builder_from_protocol(
            structure=structure,
            ML_forces=True,
            pythonjob_code=object(),   # something truthy but not a real Code
            callback_calculator=None,
        )
