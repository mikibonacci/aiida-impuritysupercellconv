# -*- coding: utf-8 -*-
"""Unit tests for ScGenerators and ChkConvergence (no AiiDA database required)."""
import numpy as np
import pytest
from ase import Atoms
from pymatgen.core import Lattice, Structure

from aiida_impuritysupercellconv.workflows.utils import ChkConvergence, ScGenerators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def si_structure():
    """Si primitive cell (pymatgen)."""
    a = 5.43
    lattice = Lattice.cubic(a)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def lif_structure():
    """LiF rock-salt unit cell (pymatgen)."""
    a = 4.035
    lattice = Lattice.cubic(a)
    return Structure(lattice, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# Helper: synthetic ASE supercell with adjustable forces
# ---------------------------------------------------------------------------

def _ase_supercell_with_forces(conv_thr: float = 0.0257, above: bool = False):
    """Build a 3×3×3 Si supercell with a muon (H) at the centre.

    Returns (ase_struc, forces) where forces are either all below or all above
    ``conv_thr`` for host atoms (the muon force is zero).
    """
    side = 10.0
    positions = [
        [ix * (side / 3), iy * (side / 3), iz * (side / 3)]
        for ix in range(3)
        for iy in range(3)
        for iz in range(3)
    ]
    symbols = ["Si"] * 27

    # Muon at centre
    positions.append([side / 2, side / 2, side / 2])
    symbols.append("H")

    ase_struc = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[side, side, side],
        pbc=True,
    )

    factor = 10.0 if above else 0.1
    forces = np.ones((len(ase_struc), 3)) * (conv_thr * factor)
    forces[-1] = [0.0, 0.0, 0.0]  # muon gets zero force

    return ase_struc, forces


# ===========================================================================
# ScGenerators tests
# ===========================================================================

class TestScGenerators:
    """Tests for the ScGenerators supercell builder."""

    def test_initialize_returns_structure_with_muon(self, si_structure):
        """First supercell must contain one H atom as the last site."""
        scg = ScGenerators(si_structure)
        py_scst_with_mu, sc_mat, mu_frac_coord = scg.initialize()

        assert py_scst_with_mu[-1].species_string == "H"
        assert sc_mat.shape == (3, 3)
        assert len(mu_frac_coord) == 3

    def test_initialize_custom_min_length_gives_larger_cell(self, si_structure):
        """A larger min_length must produce a cell with at least as many atoms."""
        scg = ScGenerators(si_structure)
        _, sc_mat_default, _ = scg.initialize()
        _, sc_mat_large, _ = scg.initialize(min_length=15.0)

        n_default = abs(round(np.linalg.det(sc_mat_default))) * si_structure.num_sites
        n_large = abs(round(np.linalg.det(sc_mat_large))) * si_structure.num_sites
        assert n_large >= n_default

    def test_initialize_too_small_min_length_raises(self, si_structure):
        """min_length smaller than the smallest lattice vector must raise ValueError."""
        scg = ScGenerators(si_structure)
        too_small = min(si_structure.lattice.abc) - 1.0
        with pytest.raises(ValueError, match="min_length"):
            scg.initialize(min_length=too_small)

    def test_initialize_sc_mat_integer_entries(self, si_structure):
        """The transformation matrix entries should be integers (or close)."""
        scg = ScGenerators(si_structure)
        _, sc_mat, _ = scg.initialize()
        assert np.allclose(sc_mat, np.round(sc_mat))

    def test_reinitialize_produces_larger_cell(self, si_structure):
        """re_initialize must return a supercell strictly larger than initialize."""
        scg = ScGenerators(si_structure)
        py_scst_1, sc_mat_1, mu_frac_coord = scg.initialize()
        py_scst_2, sc_mat_2 = scg.re_initialize(py_scst_1, mu_frac_coord)

        n1 = abs(round(np.linalg.det(sc_mat_1))) * si_structure.num_sites
        n2 = abs(round(np.linalg.det(sc_mat_2))) * si_structure.num_sites
        assert n2 > n1

    def test_reinitialize_muon_still_last(self, si_structure):
        """After re_initialize the muon should still be the last site."""
        scg = ScGenerators(si_structure)
        py_scst_1, _, mu_frac_coord = scg.initialize()
        py_scst_2, _ = scg.re_initialize(py_scst_1, mu_frac_coord)
        assert py_scst_2[-1].species_string == "H"

    def test_lif_multispecie(self, lif_structure):
        """Multi-species structures (LiF) should also work correctly."""
        scg = ScGenerators(lif_structure)
        py_scst, sc_mat, mu_frac_coord = scg.initialize()
        assert py_scst[-1].species_string == "H"
        assert sc_mat.shape == (3, 3)
        assert len(mu_frac_coord) == 3


# ===========================================================================
# ChkConvergence tests
# ===========================================================================

class TestChkConvergence:
    """Tests for the two-criterion force-convergence checker."""

    def test_first_criterion_passes_when_forces_low(self):
        """First criterion must return True when host forces are below conv_thr."""
        ase_struc, forces = _ase_supercell_with_forces(above=False)
        chk = ChkConvergence(ase_struc=ase_struc, atomic_forces=forces, conv_thr=0.0257)
        assert chk.apply_first_crit() is True

    def test_first_criterion_fails_when_forces_high(self):
        """First criterion must return False when host forces are above conv_thr."""
        ase_struc, forces = _ase_supercell_with_forces(above=True)
        chk = ChkConvergence(ase_struc=ase_struc, atomic_forces=forces, conv_thr=0.0257)
        assert chk.apply_first_crit() is False

    def test_second_criterion_returns_list(self):
        """apply_2nd_crit must always return a list."""
        ase_struc, forces = _ase_supercell_with_forces(above=False)
        chk = ChkConvergence(ase_struc=ase_struc, atomic_forces=forces, conv_thr=0.0257)
        result = chk.apply_2nd_crit()
        assert isinstance(result, list)

    def test_wrong_muon_species_raises(self):
        """Providing a species not in the structure should raise ValueError."""
        ase_struc, forces = _ase_supercell_with_forces()
        with pytest.raises(ValueError):
            ChkConvergence(ase_struc=ase_struc, atomic_forces=forces, mu_num_spec="Xe")

    def test_forces_length_mismatch_asserts(self):
        """A forces array with wrong length should raise AssertionError."""
        ase_struc, forces = _ase_supercell_with_forces()
        with pytest.raises(AssertionError):
            ChkConvergence(ase_struc=ase_struc, atomic_forces=forces[:-2])

    def test_forces_magnitude_computed_correctly(self):
        """atm_forces_mag should equal the Euclidean norm of each force vector."""
        ase_struc, forces = _ase_supercell_with_forces(above=False)
        chk = ChkConvergence(ase_struc=ase_struc, atomic_forces=forces, conv_thr=0.0257)
        expected = [np.sqrt(np.dot(f, f)) for f in forces]
        assert np.allclose(chk.atm_forces_mag, expected)

    def test_mu_id_is_last_atom(self):
        """The muon should be identified as the last atom (H appended last)."""
        ase_struc, forces = _ase_supercell_with_forces()
        chk = ChkConvergence(ase_struc=ase_struc, atomic_forces=forces)
        assert chk.mu_id == len(ase_struc) - 1
