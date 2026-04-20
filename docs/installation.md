# Installation

## Prerequisites

Before installing **aiida-impuritysupercellconv** make sure you have:

- Python 3.8 or newer
- A working [AiiDA](https://aiida.net) installation (v2.0+) with a configured profile
- A running AiiDA daemon (`verdi daemon start`)
- [Quantum ESPRESSO](https://www.quantum-espresso.org) ≥ 7.1 configured as an AiiDA code (for DFT calculations)
- Pseudopotential families installed (the default protocol uses `SSSP/1.3/PBE/efficiency`)

Install pseudopotentials with `aiida-pseudo`:

```bash
pip install aiida-pseudo
aiida-pseudo install sssp -v 1.3 -x PBE -p efficiency
```

## Installing aiida-impuritysupercellconv

### From PyPI (recommended)

```bash
pip install aiida-impuritysupercellconv
```

### From source

```bash
git clone https://github.com/positivemuon/aiida-impuritysupercellconv.git
cd aiida-impuritysupercellconv
pip install -e .
```

Verify the workflow is registered:

```bash
verdi plugin list aiida.workflows | grep impuritysupercellconv
```

You should see `impuritysupercellconv` listed.

---

## Optional: MLIP force calculations (experimental)

!!! warning "Experimental feature"
    Support for Machine Learning Interatomic Potential (MLIP) force calculations via
    `aiida-pythonjob` and ASE calculators is **experimental**. The API may change in future
    releases. Use it for rapid screening only; always validate supercell convergence with DFT.

To use MLIP-based force calculations you need the following additional packages:

```bash
pip install aiida-pythonjob
pip install mace-torch        # or another ASE-compatible MLIP
```

Verify the `pythonjob` plugin is registered:

```bash
verdi plugin list aiida.calculations | grep pythonjob
```

---

## Verifying the installation

Run a quick sanity check to confirm everything is in order:

```bash
python -c "from aiida_impuritysupercellconv.workflows.impuritysupercellconv import IsolatedImpurityWorkChain; print('OK')"
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `aiida-core` | ≥ 2.0 | workflow engine |
| `aiida-quantumespresso` | ≥ 4.2 | DFT calculations |
| `aiida-pseudo` | any | pseudopotential families |
| `pymatgen` | ≥ 2022.10.22 | structure manipulation |
| `pymatgen-analysis-defects` | any | Voronoi interstitial site generation |
| `ase` | any | structure I/O and unit conversion |
| `numpy` | any | numerical arrays |
| `scipy` | any | exponential curve fitting |
| `aiida-pythonjob` *(optional)* | any | MLIP force calculations |
