# -*- coding: utf-8 -*-
""" AiiDa IsolatedImpurityWorkChain class """
import numpy as np
from typing import Union
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction, if_, while_
from aiida.plugins import WorkflowFactory, DataFactory

try:
    from aiida_pythonjob import PythonJob
    HAS_PYTHONJOB = True
except ImportError:
    HAS_PYTHONJOB = False
    PythonJob = None


from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.common.types import ElectronicType, RelaxType, SpinType

from .utils import ChkConvergence, ScGenerators

from aiida.orm import StructureData as LegacyStructureData
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

from aiida_muon.utils.hubbard import check_get_hubbard_u_parms
from aiida_quantumespresso.common.hubbard import Hubbard

try:
    StructureData = DataFactory("atomistic.structure")
    HAS_ATOMISTIC = True
    valid_types = (StructureData, LegacyStructureData)
except Exception:
    HAS_ATOMISTIC = False
    valid_types = (LegacyStructureData,)

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
original_PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')


def PwRelaxWorkChain_override_validator(inputs,ctx=None):
    """validate inputs for impuritysupercellconv.relax; actually, it is
    just a way to avoid defining it if we do not want it. 
    otherwise the default check is done and it will excepts. 
    """
    if "relax" in inputs:
        original_PwRelaxWorkChain.spec().inputs.validator(inputs,ctx)
    else:
        return None
    
PwRelaxWorkChain.spec().inputs.validator = PwRelaxWorkChain_override_validator

@calcfunction
def create_hubbard_structure(structure: LegacyStructureData,hubbard_dict: dict):
    hubbard_structure = HubbardStructureData.from_structure(structure)
    for kind, U in hubbard_dict.items():
        hubbard_structure.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
        
    hubbard_structure.hubbard = Hubbard.from_list(hubbard_structure.hubbard.to_list(), projectors="atomic")
    return hubbard_structure

def assign_hubbard_parameters(structure: 'StructureData', hubbard_dict):
    for kind, U in hubbard_dict.items():
        structure.hubbard.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
        
@calcfunction
def init_supcgen(aiida_struc, min_length):
    """An aiida calc function that initializes supercell generation"""
    p_st = aiida_struc.get_pymatgen_structure()

    # Calls the supercell (SC) generation class
    scg = ScGenerators(p_st)
    #p_scst_mu, sc_mat, mu_frac_coord, p_scst_without_mu = scg.initialize(min_length.value)
    p_scst_mu, sc_mat, mu_frac_coord = scg.initialize(min_length.value)
    p_scst_without_mu = p_scst_mu.copy()
    p_scst_without_mu.pop(-1)    #pop out the muon since it is the last that was appendded

    if HAS_ATOMISTIC and isinstance(aiida_struc, StructureData):
        ad_scst = StructureData(pymatgen=p_scst_mu)
        ad_scst_without_mu = StructureData(pymatgen=p_scst_without_mu)
    elif isinstance(aiida_struc,LegacyStructureData):
        ad_scst = LegacyStructureData(pymatgen=p_scst_mu)
        ad_scst_without_mu = LegacyStructureData(pymatgen=p_scst_without_mu)
        if isinstance(aiida_struc, HubbardStructureData):
            hubbard_dict = check_get_hubbard_u_parms(aiida_struc.get_pymatgen())
            ad_scst = HubbardStructureData.from_structure(ad_scst)
            for kind, U in hubbard_dict.items():
                ad_scst.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
            
            ad_scst.hubbard = Hubbard.from_list(ad_scst.hubbard.to_list(), projectors="atomic")

            ad_scst_without_mu = HubbardStructureData.from_structure(ad_scst_without_mu)
            for kind, U in hubbard_dict.items():
                ad_scst_without_mu.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
                
            ad_scst_without_mu.hubbard = Hubbard.from_list(ad_scst_without_mu.hubbard.to_list(), projectors="atomic")
            

    scmat_node = orm.ArrayData()
    scmat_node.set_array("sc_mat", sc_mat)

    vor_node = orm.ArrayData()
    vor_node.set_array("Voronoi_site", np.array(mu_frac_coord))

    return {"SC_struc": ad_scst, "SCmat": scmat_node, "Vor_site": vor_node, "SC_struc_without_mu": ad_scst_without_mu,}


@calcfunction
def re_init_supcgen(aiida_struc, ad_scst, vor_site):
    """An aiida calc function that re-initializes larger supercell generation"""

    p_st = aiida_struc.get_pymatgen_structure()
    p_scst = ad_scst.get_pymatgen_structure()

    mu_frac_coord = vor_site.get_array("Voronoi_site")

    # Calls the supercell (SC) generation class
    scg = ScGenerators(p_st)
    #p_scst_mu, sc_mat, p_scst_without_mu = scg.re_initialize(p_scst, mu_frac_coord)
    p_scst_mu, sc_mat = scg.re_initialize(p_scst, mu_frac_coord)
    p_scst_without_mu = p_scst_mu.copy()
    p_scst_without_mu.pop(-1)    #pop out the muon since it is the last that was appendded

    if HAS_ATOMISTIC and isinstance(aiida_struc, StructureData):
        ad_scst = StructureData(pymatgen=p_scst_mu)
        ad_scst_without_mu = StructureData(pymatgen=p_scst_without_mu)
    elif isinstance(aiida_struc,LegacyStructureData):
        ad_scst = LegacyStructureData(pymatgen=p_scst_mu)
        ad_scst_without_mu = LegacyStructureData(pymatgen=p_scst_without_mu)
        if isinstance(aiida_struc, HubbardStructureData):
            hubbard_dict = check_get_hubbard_u_parms(aiida_struc.get_pymatgen())
            ad_scst = HubbardStructureData.from_structure(ad_scst)
            for kind, U in hubbard_dict.items():
                ad_scst.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
            
            ad_scst.hubbard = Hubbard.from_list(ad_scst.hubbard.to_list(), projectors="atomic")

            ad_scst_without_mu = HubbardStructureData.from_structure(ad_scst_without_mu)
            for kind, U in hubbard_dict.items():
                ad_scst_without_mu.initialize_onsites_hubbard(kind, '3d', U, 'U', use_kinds=True)
                
            ad_scst_without_mu.hubbard = Hubbard.from_list(ad_scst_without_mu.hubbard.to_list(), projectors="atomic")
    
    scmat_node = orm.ArrayData()
    scmat_node.set_array("sc_mat", sc_mat)

    return {"SC_struc": ad_scst, "SCmat": scmat_node, "SC_struc_without_mu": ad_scst_without_mu,}


@calcfunction
def check_if_conv_achieved(aiida_struc, traj_out, traj_out_no_muon, conv_thr):
    """An aiida calc function that checks if a supercell is converged
    for intersitial defect calculations using SCF forces
    """

    atm_forc_with_muon = traj_out.get_array("forces")[0]
    atm_forc_without_muon = traj_out_no_muon.get_array("forces")[0]
    atm_forc_without_muon = np.append(np.array(atm_forc_without_muon),np.array([[0.0,0.0,0.0]]), axis=0)
    atm_forces = np.array(atm_forc_with_muon) - np.array(atm_forc_without_muon) 
    ase_struc = aiida_struc.get_ase()

    # Calls the check supercell convergence class
    csc = ChkConvergence(ase_struc = ase_struc,
                         atomic_forces = atm_forces,
                         conv_thr = conv_thr.value)
    cond = csc.apply_first_crit()
    cond2 = csc.apply_2nd_crit()
    #print(cond,cond2)
    if cond is True and all(cond2):
        return orm.Bool(True)
    else:
        return orm.Bool(False)


def get_pseudos(aiida_struc, pseudofamily):
    """Get pseudos"""
    family = orm.load_group(pseudofamily)
    pseudos = family.get_pseudos(structure=aiida_struc)
    return pseudos


def get_kpoints(aiida_struc, k_density):
    """Get kpoints"""
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(aiida_struc)
    kpoints.set_kpoints_mesh_from_density(k_density, force_parity=False)
    return kpoints


class IsolatedImpurityWorkChain(ProtocolMixin, WorkChain):
    """WorkChain for finding converged supercell for interstitial impurity calculation"""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        if HAS_PYTHONJOB:
            spec.expose_inputs(
                PythonJob,
                namespace='pythonjob',
                namespace_options={
                    'required': False,
                    'populate_defaults': False,
                    'help': 'Inputs for MLIPs force calculations.',
                },
            )
            spec.input(
                "ML_forces",
                valid_type=orm.Bool,
                default=lambda: orm.Bool(False),
                required=False,
                help="Use MLIP (via pythonjob/ASE) for force calculations instead of DFT",
            )
        spec.input(
            "structure",
            valid_type=valid_types,
            required=True,
            help="Input initial structure",
        )
        spec.input(
            "min_length",
            valid_type=orm.Float,
            default=lambda: None,
            required=False,
            help="The minimum length of the smallest"
            " lattice vector for the first generated supercell ",
        )
        spec.input(
            "conv_thr", 
            valid_type=orm.Float, 
            default=lambda: orm.Float(0.0257), 
            required=False, 
            help="Force convergence thresh in eV/Ang, default is 1e-3 au or 0.0257 ev/A",
        )
        spec.input(
            "max_iter_num",
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            required=False,
            help="Maximum number of iteration in the supercell convergence loop",
        )
        spec.input(
            "kpoints_distance",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.301),
            required=False,
            help="The minimum desired distance in 1/Å between k-points in reciprocal space.",
        )
       
        spec.input(
            "pseudo_family",
            valid_type=orm.Str,
            default=lambda: orm.Str("SSSP/1.3/PBE/efficiency"),
            required=False,
            help="The label of the pseudo family",
        )
        spec.input(
            "charge_supercell",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="To run charged supercell for positive muon or not (neutral supercell)",
        )
        spec.input(
            "relax_unitcell",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="To relax the forces on the input unit structure or not",
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="pwscf",
            exclude=("pw.structure", "kpoints"),
            namespace_options={
                'required': False, 'populate_defaults':False,
                'help': 'the pwscf step.',
            },
        )  # use the  pw base workflow
        
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="relax",
            exclude=("structure","base_final_scf"),
            namespace_options={
                'required': False, 'populate_defaults':False,
                'help': 'the preprocess relaxation step, if needed.',
                'dynamic':True,
            },
        )  # use the  pw relax workflow

        #spec.inputs.validator = input_validator
        
        spec.outline(
            if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax
                ),
            cls.init_supcell_gen,
            if_(cls.should_run_mlip_forces)(
                cls.run_ase_double_forces,
            ).else_(
                cls.run_pw_double_scf,
            ),
            cls.inspect_run_get_forces,
            while_(cls.continue_iter)(
                cls.increment_n_by_one,
                if_(cls.iteration_num_not_exceeded)(
                    cls.get_larger_cell,
                    if_(cls.should_run_mlip_forces)(
                        cls.run_ase_double_forces,
                    ).else_(
                        cls.run_pw_double_scf,
                    ),
                    cls.inspect_run_get_forces
                ).else_(
                    cls.exit_max_iteration_exceeded,
                ),
            ),
            cls.set_outputs,
        )

        spec.output("Converged_supercell", valid_type=valid_types, required=True)
        spec.output("Converged_SCmatrix", valid_type=orm.ArrayData, required=True)

        spec.exit_code(
            403,
            "ERROR_RELAXATION_FAILED",
            message="the PwRelaxWorkchain failed",
        )
        
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="one of the PwBaseWorkChain subprocesses failed",
        )
        spec.exit_code(
            702,
            "ERROR_NUM_CONVERGENCE_ITER_EXCEEDED",
            message="Max number of supercell convergence reached ",
        )
        spec.exit_code(
            704,
            "ERROR_FITTING_FORCES_TO_EXPONENTIAL",
            message="Error in fitting the forces to an exponential",
        )

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code: orm.Code = None,
        structure: Union[LegacyStructureData, 'StructureData'] = None,
        protocol: str = None,
        overrides: dict = None,
        relax_unitcell: bool = False, 
        options = None,
        min_length: float = None,
        conv_thr: float = 0.0257,
        kpoints_distance: float = 0.301,
        charge_supercell: bool = True,
        pseudo_family: str ="SSSP/1.3/PBE/efficiency",
        max_iter_num: int = 4,
        ML_forces: bool = False,
        pythonjob_code: orm.Code = None,
        callback_calculator: callable = None,
        model_name: str = None,
        additional_pythonjob_inputs: dict = {},
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin. Used in all the sub workchains. Optional if ML_forces is True.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param relax_unitcell: optional relaxation of the unit cell before to put the defects and proceed with the supercell relaxation.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this workchain.
        :param min_length: The minimum length of the smallest lattice vector for the first generated supercell.
        :param conv_thr: The force convergence threshold in eV/Ang, default is 1e-3 au or 0.0257 ev/A
        :param kpoints_distance: the minimum desired distance in 1/Å between k-points in reciprocal space.
        :param charge_supercell: the charge in the supercell. Default is false as here we don't care about the muon charge state.
        :param pseudo_family: the label of the pseudo family.
        :param max_iter_num: Maximum number of iteration in the supercell convergence loop.
        :param ML_forces: Use MLIP (via pythonjob/ASE) for force calculations instead of DFT.
        :param pythonjob_code: The PythonJob code to use for MLIP calculations (required if ML_forces is True).
        :param callback_calculator: The ASE calculator callback function (required if ML_forces is True).
        :param model_name: Name/label of the MLIP model, stored in the output for provenance. Optional.
        :param additional_pythonjob_inputs: Additional inputs for the PythonJob (e.g., metadata, other settings).
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        builder = cls.get_builder()
    
        if not overrides: overrides = {}
            
        #overrides_pwscf = overrides.pop('pwscf',{})
        
        #overrides_pwscf = recursive_merge(overrides_pwscf, {"CONTROL": {"tprnfor": True,},})

        overrides_all = {
            "base": {
                #"pseudo_family": pseudo_family,
                "pw": {
                    "parameters": {
                "CONTROL": {
                    "tprnfor": True,
                    },
                #"SYSTEM": {
                #    "tot_charge":1 if charge_supercell else 0,
                #},
                "ELECTRONS":{
                    'electron_maxstep': 200,
                    'mixing_beta': 0.3,
                }
                      },
                    },
            },
            #"base_final_scf": {"pseudo_family": pseudo_family,},
            }

        overrides_pwscf = recursive_merge(overrides, overrides_all)
                
        # Setup DFT builders only if not using MLIP
        if not ML_forces:
            if pw_code is None:
                raise ValueError("pw_code is required when ML_forces is False")
                
            builder_pwscf = PwBaseWorkChain.get_builder_from_protocol(
                    pw_code,
                    structure,
                    protocol=protocol,
                    overrides=overrides_pwscf.get("base",None),
                    #overrides=overrides_pwscf,
                    pseudo_family=pseudo_family,
                    **kwargs,
                    )
            
            for k,v in builder_pwscf.items():
                if k in ["structure","kpoints_distance"]: continue
                setattr(builder.pwscf,k,v)
            builder.pwscf.pw.pop('structure', None)
            builder.pwscf.pop('kpoints_distance', None)  
            builder.pwscf.pop('kpoints', None)  

            builder_relax = PwRelaxWorkChain.get_builder_from_protocol(
                    pw_code,
                    structure,
                    protocol=protocol,
                    #overrides=overrides_pwscf, #IJO, we don't ever want total charge=1.0 for the unitcell relax without muon
                    overrides=overrides_pwscf.get("pre_relax",None),
                    pseudo_family=pseudo_family,
                    relax_type=RelaxType.POSITIONS, #Infinite dilute defect
                    **kwargs,
                    )
                    
            for k,v in builder_relax.items():        
                setattr(builder.relax,k,v)   
            
            builder.relax.pop('base_final_scf', None) 
            if not relax_unitcell:
                builder.relax.base.pw.parameters = orm.Dict({})
        
        # Setup MLIP pythonjob inputs if requested
        if ML_forces:
            if pythonjob_code is None:
                raise ValueError("pythonjob_code is required when ML_forces is True")
            if callback_calculator is None:
                raise ValueError("callback_calculator is required when ML_forces is True")
            
            from aiida_impuritysupercellconv.pythonjobs.forces import prepare_ase_pythonjob_forces_inputs
            pythonjob_inputs = prepare_ase_pythonjob_forces_inputs(
                structure=structure,
                pythonjob_code=pythonjob_code,
                callback_calculator=callback_calculator,
                charged_supercell=charge_supercell,
                model_name=model_name,
                **additional_pythonjob_inputs,
            )
            
            builder.pythonjob = pythonjob_inputs
        
        #we can set this also wrt to some protocol
        if min_length is not None:
            builder.min_length = orm.Float(min_length)
        builder.conv_thr=orm.Float(conv_thr)
        builder.kpoints_distance=orm.Float(kpoints_distance)
        builder.max_iter_num=orm.Int(max_iter_num)
        
        builder.structure = structure
        builder.pseudo_family = orm.Str(pseudo_family)
        builder.charge_supercell = orm.Bool(charge_supercell)
        if HAS_PYTHONJOB: builder.ML_forces = orm.Bool(ML_forces)
        
        return builder
    

    def should_run_relax(self):
        if "relax" in self.inputs:
            return len(self.inputs.relax.base.pw.parameters.get_dict()) > 0
        return False
    
    def should_run_mlip_forces(self):
        """Check if we should run MLIP force calculations."""
        return self.inputs.ML_forces.value if hasattr(self.inputs, "ML_forces") else False
    
    def run_relax(self):
        """Run the `PwBaseWorkChain` to run a relax `PwCalculation`."""

        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.inputs.structure
        inputs.base.pw.pseudos = get_pseudos(
            inputs.structure, self.inputs.pseudo_family.value
        )
        
        inputs.metadata.call_link_label = f'relax_step'

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'Relaxation of the defect-free unit cell requested.')
        self.report(f'launching PwRelaxWorkChain<{running.pk}>')

        return ToContext(calculation_run=running)
    
    def inspect_relax(self):
        calculation = self.ctx.calculation_run
        if not calculation.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{calculation.pk}> failed"
                "with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_RELAXATION_FAILED
        else:
            self.ctx.structure = calculation.outputs.output_structure
            
        return
        
    def init_supcell_gen(self):
        """initialize supercell generation"""
        self.ctx.n = orm.Int(0)
        
        if not "structure" in self.ctx: self.ctx.structure = self.inputs.structure

        if self.inputs.min_length is None:
            m_l = min(self.ctx.structure.get_pymatgen_structure().lattice.abc) + 1
            self.inputs.min_length = orm.Float(m_l)

        result_ini = init_supcgen(self.ctx.structure, self.inputs.min_length)

        self.ctx.sup_struc_mu = result_ini["SC_struc"]
        self.ctx.sup_struc_without_mu = result_ini["SC_struc_without_mu"]
        self.ctx.musite = result_ini["Vor_site"]
        self.ctx.sc_mat = result_ini["SCmat"]

    def run_ase_double_forces(self):
        """Run ASE force calculations with and without the muon using MLIP."""
        inputs = AttributeDict(self.exposed_inputs(PythonJob, namespace='pythonjob'))
        
        runs = {}
        
        # With muon
        inputs.function_inputs.atoms = self.ctx.sup_struc_mu
        inputs.metadata.call_link_label = f'forces_with_muon_iter{self.ctx.n.value:02d}'
        runs["with_muon"] = self.submit(PythonJob, **inputs)
        self.report(
            f"Launching PythonJob (PK={runs['with_muon'].pk}) for force calculation with muon"
        )
        
        # Without muon
        inputs.function_inputs.atoms = self.ctx.sup_struc_without_mu
        inputs.metadata.call_link_label = f'forces_without_muon_iter{self.ctx.n.value:02d}'
        inputs.function_inputs.charged_supercell = orm.Bool(False)
        runs["without_muon"] = self.submit(PythonJob, **inputs)
        self.report(
            f"Launching PythonJob (PK={runs['without_muon'].pk}) for force calculation without muon"
        )
        
        return ToContext(**runs)
    
    def run_pw_double_scf(self):
        """Input Qe-pw structure and run pw with and without the muon (but the charge, if any) in the cell."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="pwscf"))

        if hasattr(self.inputs,"charge_supercell"):
            inputs.pw.parameters = update_charge(inputs.pw.parameters,self.inputs.charge_supercell)  
                  
        inputs.pw.structure = self.ctx.sup_struc_mu
        inputs.pw.pseudos = get_pseudos(
            self.ctx.sup_struc_mu, self.inputs.pseudo_family.value
        )
        inputs.kpoints = get_kpoints(
            self.ctx.sup_struc_mu, self.inputs.kpoints_distance.value
        )

        runs = {}
        runs["with_muon"] = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"running SCF calculation with muon: {runs['with_muon'].pk}")
        
        # now without muon to have a reference for the force difference.
        inputs.pw.structure = self.ctx.sup_struc_without_mu
        inputs.pw.pseudos = get_pseudos(
            self.ctx.sup_struc_without_mu, self.inputs.pseudo_family.value
        )
        runs["without_muon"] = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"running SCF calculation without muon: {runs['without_muon'].pk}")
        

        return ToContext(**runs)

    def inspect_run_get_forces(self):
        """Inspect calculations and get forces (both DFT and MLIP supported)."""
        self.ctx.traj_out = {}
        
        for run in ["with_muon", "without_muon"]:
            calculation = self.ctx[run]

            if not calculation.is_finished_ok:
                calc_type = "PythonJob" if 'pythonjob' in calculation.process_type else "PwBaseWorkChain"
                self.report(
                    f"{calc_type}<{calculation.pk}> failed "
                    f"with exit status {calculation.exit_status}"
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
            else:
                # Handle different output types: DFT (output_trajectory) vs MLIP (forces)
                if 'pythonjob' in calculation.process_type:
                    # MLIP calculation - forces stored as orm.List (plain Python list)
                    # because the pythonjob returns .tolist() to avoid numpy pickle issues.
                    forces_array = np.array(calculation.outputs.forces.get_list())
                    traj_node = orm.ArrayData()
                    traj_node.set_array("forces", np.array([forces_array]))
                    self.ctx.traj_out[run] = traj_node
                elif 'quantumespresso.pw.base' in calculation.process_type:
                    # DFT calculation - use output_trajectory
                    self.ctx.traj_out[run] = calculation.outputs.output_trajectory
                else:
                    raise ValueError(f"Unknown calculation type: {calculation.process_type} for uuid={calculation.uuid}.")

    def continue_iter(self):
        """check convergence and decide if to continue the loop"""
        try:
            if not "conv_thr" in self.ctx: self.ctx.conv_thr = self.inputs.conv_thr
            conv_res = check_if_conv_achieved(self.ctx.sup_struc_mu,
                                              self.ctx.traj_out["with_muon"],
                                              self.ctx.traj_out["without_muon"],
                                              self.ctx.conv_thr)
            return conv_res.value == False
        except Exception:
            self.report(
                f"Exiting IsolatedImpurityWorkChain,Error in fitting the forces of supercell,"
                f"iteration no. <{self.ctx.n}>) to an exponential, maybe force data not exponential"
            )
            return self.exit_codes.ERROR_FITTING_FORCES_TO_EXPONENTIAL

    def increment_n_by_one(self):
        """Increase count by 1"""
        self.ctx.n += 1

    def iteration_num_not_exceeded(self):
        """Check if iteration is exceeded"""
        return self.ctx.n <= self.inputs.max_iter_num.value

    def get_larger_cell(self):
        """Previous supercell not converged - get larger supercell"""
        result_reini = re_init_supcgen(
            self.ctx.structure, self.ctx.sup_struc_mu, self.ctx.musite
        )

        self.ctx.sup_struc_mu = result_reini["SC_struc"]
        self.ctx.sup_struc_without_mu = result_reini["SC_struc_without_mu"]
        self.ctx.sc_mat = result_reini["SCmat"]

    def exit_max_iteration_exceeded(self):
        """Exit code if max iteration number is reached"""
        self.report(
            f"Exiting IsolatedImpurityWorkChain, Coverged supercell NOT achieved, next iter num"
            f" <{self.ctx.n}> is greater than max iteration number {self.inputs.max_iter_num.value}"
        )
        return self.exit_codes.ERROR_NUM_CONVERGENCE_ITER_EXCEEDED

    def set_outputs(self):
        """Print outputs"""
        self.report("Setting Outputs")
        self.out("Converged_supercell", self.ctx.sup_struc_mu)
        self.out("Converged_SCmatrix", self.ctx.sc_mat)
        
        self.report("Converged supercell found with supercell matrix:")
        self.report(self.ctx.sc_mat.get_array("sc_mat"))
    

# Functions for the input validation.
def iterdict(d,key):
  value = None
  for k,v in d.items():
    if isinstance(v, dict):
        value = iterdict(v,key)
    else:            
        if k == key:
          return v
    if value: return value


def recursive_consistency_check(input_dict):
    import copy
    
    """Validation of the inputs provided for the IsolatedImpurityWorkChain.
    """
    
    parameters = copy.deepcopy(input_dict)
    
    keys = ["occupations","smearing"]
    
    wrong_inputs_relax = []
    wrong_inputs_pwscf = []
    
    unconsistency_sentence = ''
    
    if "relax" in parameters :
        if len(parameters["relax"]["base"]["pw"]["parameters"].get_dict()) > 0:
            if parameters["relax"]["base"]["pw"]["parameters"].get_dict()["CONTROL"]["calculation"] != 'relax':
                unconsistency_sentence+=f'Checking inputs.impuritysupercellconv.relax.base.pw.parameters.CONTROL.calculation: can be only "relax". No cell relaxation should be performed.'
            
            
            if 'base_final_scf' in parameters['relax']:
                if parameters['relax']['base_final_scf'] ==  {'metadata': {}, 'pw': {'metadata': {'options': {'stash': {}}}, 'monitors': {}, 'pseudos': {}}}:
                    pass
                elif parameters['relax']['base_final_scf'] ==  {}:
                    pass
                else:
                    unconsistency_sentence+=f'Checking inputs.impuritysupercellconv.relax.base_final_scf: should not be set, the final scf after relaxation is not supported in the MusConvWorkChain.'
    
    
    
    value_input_pwscf = iterdict(parameters["pwscf"]["pw"]["parameters"].get_dict(),'tprnfor')
    value_overrides = True
    #print(value_input_relax,value_input_pwscf,value_overrides)

    if value_input_pwscf != value_overrides:
        wrong_inputs_pwscf.append('tprnfor')
        unconsistency_sentence += f'Checking inputs.impuritysupercellconv.pwscf.pw.parameters input: "tprnfor" is not correct. You provided the value "{value_input_pwscf}", but only "{value_overrides}" is consistent with your settings.\n'

    return unconsistency_sentence

def input_validator(inputs,_,caller="IsolatedImpurityWorkChain"):
    inconsistency = recursive_consistency_check(inputs)
    if len(inconsistency) > 1:
        if caller == "FindMuonWorkchain": 
            return inconsistency
        else:
            raise ValueError('\n'+inconsistency+'\n Please check the inputs of your MusConvWorkChain instance or use "get_builder_from_protocol()" method to populate correctly the inputs.\n')
    
    return #cannot return anything otherwise it Raise an error.

@calcfunction
def update_charge(parameters,charge):
    """Update the charge in the parameters"""
    parameters = parameters.get_dict()
    parameters["SYSTEM"]["tot_charge"] = 1 if charge else 0
    return orm.Dict(dict=parameters)
