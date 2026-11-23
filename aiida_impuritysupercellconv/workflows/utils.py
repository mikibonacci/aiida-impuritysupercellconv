# -*- coding: utf-8 -*-
"""Generates a nearly cubic supercell (SC) for convergence checks.
 DEPENDENCIES:
 impurity generator now in a  pymatgen extension
 (i) pymatgen-analysis-defects (pymatgen>=2022.10.22)
 (ii) numpy
"""
import numpy as np
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from ase.units import Bohr, Rydberg
from scipy.optimize import curve_fit


class ScGenerators:
    """
    Generates a nearly cubic supercell (SC) for convergence checks.
    Inserts a muon in the supercell at a Voronoi interstitial site.
    One of it methods initializes the supercell generation and the other
    re-initializes generation of a larger supercell-size than the former.

    Param:
        py_struc: A pymatgen "unitcell" structure instance
                  This is used to create the supercell.
    """

    @staticmethod
    def gen_nearcubic_supc(py_struc, min_atoms, max_atoms, min_length):
        """
        Function that generates the nearly cubic supercell (SC).

        Params:
            py_struc         : The pymatgen structure instance
            min_atoms        : Integer-->Min number of atoms allowed in SC
            max_atoms        : Integer-->Max number of atoms allowed in SC
            min_length       : Integer-->Min length of the smallest SC lattice vector

        Returns:
            A nearly cubic SC structure and an array of the SC grid size
        """

        cst = CubicSupercellTransformation(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=False,
        )

        py_scst = cst.apply_transformation(py_struc)
        sc_mat = cst.transformation_matrix

        return py_scst, sc_mat

    @staticmethod
    def append_muon_to_supc(py_scst, sc_mat, mu_frac_coord):
        """
        Add the muon as a hydrogen atom to the supercell (SC).

        Params:
            py_scst    : The pymatgen supercell structure
            sc_mat          : array-->the SC grid size
            mu_frac_coord     : array-->Interstitial site scaled in units

        Returns:
            A Pymatgen supercell structure that has the muon as a H atom at a Voronoi site


        """

        mu_frac_coord_sc = (np.dot(mu_frac_coord, np.linalg.inv(sc_mat))) % 1
        py_scst_withmu = py_scst.copy()

        # what if a H specie is in the structure object?
        try:
            py_scst_withmu.append(
                species="H",
                coords=mu_frac_coord_sc,
                coords_are_cartesian=False,
                validate_proximity=True,
            )
        except ValueError:
            raise SystemExit(
                "ValueError:The muon is too close to an existing site!, "
                "change muon site. Exiting...."
            ) from None

        return py_scst_withmu

    def __init__(self, py_struc):
        self.py_struc = py_struc
        self.max_atoms = np.inf
        # self.py_scst     = None
        # self.mu_frac_coord  = None

    def initialize(self, min_length: float = None):
        """
        This func initializes the first supercell (SC) generation
        with the following conditions;

        min_atoms  : number of atoms in the given struc + 1
        max_atoms  : number of atoms in the given struc*(2**3)
                    This limits the SC generation to 8 times of the given cell.
        min_length : Min length of the smallest SC lattice vector

        Returns:
            A Pymatgen supercell structure that has the muon as a H atom at a Voronoi site
        """

        min_atoms = self.py_struc.num_sites + 1
        min_length = min_length or np.min(self.py_struc.lattice.abc) + 1

        if min_length < np.min(self.py_struc.lattice.abc):
            raise ValueError(
                " Provided supercell min_length is less than the length of the"
                " smallest input cell lattice vector"
            )

        py_scst, sc_mat = self.gen_nearcubic_supc(
            self.py_struc, min_atoms, self.max_atoms, min_length
        )

        # get a Voronoi interstitial site for the muon impurity
        vig = VoronoiInterstitialGenerator()
        mu_frac_coord = list(vig._get_candidate_sites(self.py_struc))[0][0]

        # Added 0.001 to move the impurity site from symmetric position
        mu_frac_coord = [x + 0.001 for x in mu_frac_coord]

        py_scst_with_mu = self.append_muon_to_supc(py_scst, sc_mat, mu_frac_coord)

        return py_scst_with_mu, sc_mat, mu_frac_coord

    def re_initialize(self, py_scst_with_mu, mu_frac_coord):
        """
        This function re-initializes the generation of a larger supercell-size in a loop
        when a condition is not met after the first initialization above.

        Param:
            iter_num : Integer--> iteration number in the loop

        Returns:
            A Pymatgen supercell structure that has the muon as a H atom at a Voronoi site
        """

        min_atoms = py_scst_with_mu.num_sites + 1
        min_length = np.min(py_scst_with_mu.lattice.abc) + 1

        py_scst, sc_mat = self.gen_nearcubic_supc(
            self.py_struc, min_atoms, self.max_atoms, min_length
        )

        py_scst_with_mu = self.append_muon_to_supc(py_scst, sc_mat, mu_frac_coord)

        return py_scst_with_mu, sc_mat


"""
Checks for converged supercell for interstitial impurity calculations
"""

class ChkConvergence:
    """
    Checks if a supercell (SC) size is converged for muon site calculations
    using results of atomic forces from a one shot SCF calculation.
    """

    @staticmethod
    def exp_fnc(xdata, amp, tau):
        """
        An exponential decay function with;
        """
        return amp * np.exp(-tau * xdata)

    @staticmethod
    def min_sconv_dist(y_c, amp, tau):
        """
        Inverse of the exp func
        """
        return np.log(y_c / amp) / (-tau)

    def __init__(
        self,
        ase_struc,
        atomic_forces,
        mu_num_spec: int or str = 1 or "H",
        conv_thr: float = 1e-03 * Rydberg / Bohr,  # from au to eV/A
        max_force_thr: float = 0.06 * Rydberg / Bohr,  # from au to eV/A
        mnasf: int = 4,
    ):
        """
        Params:
        ase_struc  : An ASE Atom structure object

        atomic_forces  : ndarray--> array of atomic_forces in eV/A
                         Length and order of array is same as the atomic positions
        mu_num_spec    : Integer or String --> atomic number or Specie symbol for the muon
                         Default: 1 or 'H'
        conv_thr       : Float --> Converged Force threshold.
                         Default = 1e-03*Rydberg/Bohr #from au to eV/A
        max_force_thr  : Float --> Max force considered in fitting
                         Default = 0.06*Rydberg/Bohr #from au to eV/A
        mnasf          : Int   -->  Minimum number of atoms sufficient for fit
        """

        self.ase_struc = ase_struc
        self.atomic_forces = atomic_forces
        self.mu_num_spec = mu_num_spec
        self.conv_thr = conv_thr
        self.max_force_thr = max_force_thr
        self.mnasf = mnasf

        assert len(self.ase_struc.numbers) == len(
            self.atomic_forces
        ), "No. of atoms not equal to number of forces"

        # Check and get muon index
        if (
            isinstance(self.mu_num_spec, int)
            and self.mu_num_spec in self.ase_struc.numbers
        ):
            mu_idd = [
                atom.index for atom in self.ase_struc if atom.number == self.mu_num_spec
            ]

        elif isinstance(self.mu_num_spec, str) and self.mu_num_spec in set(
            self.ase_struc.get_chemical_symbols()
        ):
            mu_idd = [
                atom.index for atom in self.ase_struc if atom.symbol == self.mu_num_spec
            ]

        else:
            raise ValueError(
                f"{mu_num_spec} is not in the specie or atomic number list"
            )

        if len(mu_idd) > 1:
            raise ValueError(
                "Provided muon specie or atomic number has more than one muon in the structure"
            )
        self.mu_id = mu_idd[0]

        # magnitude of atomic forces
        self.atm_forces_mag = [np.sqrt(x.dot(x)) for x in self.atomic_forces]

    def apply_first_crit(self):
        """
        Implements the first convergence criteria;
        Convergence is achieved if one of the forces in the supercell
        (SC) is less than the force convergence threshold
        """

        # remove forces on muon  and 0. forces at boundary due to symmetry if present
        no_mu_atm_forces_mag1 = [
            v for i, v in enumerate(self.atm_forces_mag) if i != self.mu_id and v != 0.0
        ]

        if min(no_mu_atm_forces_mag1) < self.conv_thr:
            print("First SC convergence Criteria achieved")
            return True
        else:
            print("First SC convergence Criteria  NOT achieved")
            return False

    def apply_2nd_crit(self):
        """
        Implements the second convergence criteria:
        Forces for each atomic specie are fitted to an exponential
        decay of their respective atomic position distance from the muon.
        Convergence is achieved when the max relative distance is less than
        the minimum relative distance obtained fro the fit parameters.

        """
        atm_indxes = [atom.index for atom in self.ase_struc]

        atm_dist = self.ase_struc.get_distances(
            self.mu_id, atm_indxes, mic=True, vector=False
        )

        specie_num = len(set(self.ase_struc.numbers))
        specie_set = set(self.ase_struc.get_chemical_symbols())
        mu_symb = self.ase_struc.symbols[self.mu_id]

        # Remove muon specie from specie set
        specie_set = [i for i in specie_set if i != mu_symb]

        cond = []
        for i in range(0, specie_num - 1):
            specie_index = [
                atom.index for atom in self.ase_struc if atom.symbol == specie_set[i]
            ]

            if len(specie_index) >= self.mnasf:
                specie_dist = [
                    atm_dist[i]
                    for i in specie_index
                    if self.atm_forces_mag[i] < self.max_force_thr
                    and self.atm_forces_mag[i] != 0.0
                ]
                specie_force = [
                    self.atm_forces_mag[i]
                    for i in specie_index
                    if self.atm_forces_mag[i] < self.max_force_thr
                    and self.atm_forces_mag[i] != 0.0
                ]

                try:
                    par, cov = curve_fit(self.exp_fnc, specie_dist, specie_force)
                except (ValueError, RuntimeError) as err:
                    print(
                        "Check force data, maybe the data does not decay exponentially with:",
                        err,
                    )
                    cond.append(False)
                    continue

                # fit  and data check, better conditions?
                #stder = np.sqrt(np.diag(cov))
                #if stder[0] > par[0] or stder[1] > par[1]:
                #    print(
                #        f"Check force data and fit on specie {specie_set[i]},"
                #        "maybe does not decay exponentially"
                #    )
                #    cond.append(False)
                #    continue

                # find min distance  required for convergence
                min_conv_dist = self.min_sconv_dist(self.conv_thr, par[0], par[1])
                # print(f"Max mu-atom distance in the SC is {max(specie_dist)}, Min distance"
                # \" required for conv is {min_conv_dist}")

                # TO DO: print lines to be removed
                if max(specie_dist) >= min_conv_dist:
                    print(
                        f"Second SC convergence Criteria achieved on specie--> {specie_set[i]}"
                    )
                    cond.append(True)
                else:
                    print(
                        f"For specie {specie_set[i]} the 2nd SC convergence is NOT achieved,"
                        " min dist required is {min_conv_dist} Ang "
                    )
                    cond.append(False)

            else:
                print(
                    f"The current SC size is NOT sufficient for  specie, {specie_set[i]}"
                )

        if not cond:
            cond.append(False)

        return cond
