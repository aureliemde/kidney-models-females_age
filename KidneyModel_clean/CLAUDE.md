# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational model of renal physiology in young female rats. Computes steady-state solute concentrations, volumes, and electropotentials across 9 sequential kidney tubule segments. Originally written in Fortran by Prof. Aurelie Edwards, translated to Python by Dr. Mohammad M. Tajdini (Boston University, Dept. of Biomedical Engineering).

**Units:** CGS throughout — lengths in cm, concentrations in mmol/cm³ (= M), time in seconds.

## Running the Simulation

```bash
python main.py
```

No build step required. Numba JIT-compiles performance-critical functions on first run (slower), subsequent runs use cached compilation.

## Key Configuration: `values.py`

All simulation parameters live here — do not scatter magic numbers elsewhere:

- `NZ = 50` — spatial grid intervals (use 200 for production runs, 50 for testing)
- `NS = 16` — number of solutes (fixed; changing requires updating all modules)
- `NC = 6` — compartments per cell (fixed)
- `sngfr0` — single nephron filtration rate (boundary condition)
- `bdiabetes`, `furo` — boolean flags for disease/drug simulations
- `ncompl`, `ntorq`, `ndiabetes` — integer model switches

Physiological boundary conditions (bath concentrations at cortex/medulla/papilla) are also in `values.py`.

## Architecture

### Simulation Pipeline (sequential, `main.py`)

Each segment takes the outlet of the previous as its inlet:

```
PT → SDL → mTAL → cTAL → DCT → CNT → CCD → OMCD → IMCD → Outlet_SS
```

For each segment, the pattern is:
1. `init{Segment}()` — set up parameters, geometry, initial guesses
2. `qnewton1{Segment}()` or `qnewton2{Segment}()` — Newton-Raphson solver
3. Write outlet file to `Data/`

### Module Naming Conventions

| Prefix | Role |
|--------|------|
| `init*.py` | Initialize segment parameters and compartment structure |
| `qnewton*.py` | Newton-Raphson nonlinear solver for a segment type |
| `qflux*.py` | Flux evaluator called by the solver (residual function) |
| `jacobi*.py` | Jacobian matrix computation for the implicit solver |
| `compute_*.py` | Transporter-specific flux calculations (SGLT, NHE3, KCC, etc.) |

Solver variants: `qnewton2PT` (proximal tubule), `qnewton2SDL` (descending limb), `qnewton2b` (TAL/DCT/IMCD), `qnewton2icb` (CNT/CCD/OMCD — 4-compartment epithelium).

### Compartment Model

Each cell cross-section has up to 6 compartments:
- **M** — lumen (tubular fluid)
- **P, A, B** — epithelial cell subtypes (varies by segment)
- **E** — lateral intercellular space
- **S** — peritubular bath/interstitium

### 16 Solutes (fixed index order, defined in `values.py`)

Na⁺, K⁺, Cl⁻, HCO₃⁻, H₂CO₃, CO₂, HPO₄²⁻, H₂PO₄⁻, Urea, NH₃, NH₄⁺, H⁺, HCO₂⁻, H₂CO₂, Glucose, Ca²⁺

### Data Files (`Data/` directory)

ASCII space-separated files, scientific notation. Each segment reads its inlet from `{Segment}outlet` and writes its outlet back. Format per outlet file: 16 solute rows (lumen + bath values), then pH, membrane potential, volume.

- `PToutlet_all` — expanded PT output with all compartmental data
- `Outlet_SS` — final simulation result (urinary output)

### Global State: `glo.py` and `defs.py`

- `glo.py` — global variables shared across modules
- `defs.py` — `Membrane` class: holds all per-cell attributes (concentrations, permeabilities, transporter expression levels, surface areas, kinetic parameters)

## Dependencies

- **NumPy** — all array/linear algebra operations
- **Numba** (`@njit`) — JIT compilation for flux and transporter calculations; required for performance

Install: `pip install numpy numba`

## Comparing Results

```bash
python compare_results.py
```

Utility script for validating simulation output against reference data.
