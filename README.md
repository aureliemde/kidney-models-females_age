# Kidney Nephron Steady-State Transport Model

Full renal nephron steady-state model for young female rats. Simulates solute and water transport across all major nephron segments at physiological steady state using Newton-Raphson iteration on a spatial grid.

**Originally written in Fortran** by Prof. Aurelie Edwards  
**Translated to Python** by Dr. Mohammad M. Tajdini  
**Refactored** by Sofia Polychroniadou  
Department of Biomedical Engineering, Boston University  
*Based on: AMW model, AJP Renal 2007*

---

## Requirements

- Python 3.6+
- `numpy`
- `numba`

Install dependencies:
```bash
pip install numpy numba
```

---

## How to Run

```bash
cd KidneyModel
python main.py
```

The simulation runs the full nephron pipeline sequentially and writes output files to the working directory. No command-line arguments are needed.

---

## Nephron Pipeline

Segments are solved in order, each passing its outlet concentrations as the inlet of the next:

```
PT  →  SDL  →  mTAL  →  cTAL  →  DCT  →  CNT  →  CCD  →  OMCD  →  IMCD  →  Final Urine
```

| Segment | Full Name | Cell Types | Solver |
|---|---|---|---|
| PT | Proximal Tubule | Lumen, P-cell, LIS | `qnewton1PT` + `qnewton2PT` |
| SDL | Straight Descending Limb | Lumen only | `qnewton2SDL` |
| mTAL | Medullary Thick Ascending Limb | Lumen, P-cell, LIS | `qnewton2b` |
| cTAL | Cortical Thick Ascending Limb | Lumen, P-cell, LIS | `qnewton2b` |
| DCT | Distal Convoluted Tubule | Lumen, P-cell, LIS | `qnewton2b` |
| CNT | Connecting Tubule | Lumen, P, α, β, LIS | `qnewton2icb` |
| CCD | Cortical Collecting Duct | Lumen, P, α, β, LIS | `qnewton2icb` |
| OMCD | Outer Medullary Collecting Duct | Lumen, P, α, β, LIS | `qnewton2icb` |
| IMCD | Inner Medullary Collecting Duct | Lumen, P-cell, LIS | `qnewton2b` |

**Compartment labels:** M = lumen, P = principal cell, A = α-intercalated cell, B = β-intercalated cell, LIS = lateral intercellular space, BATH = peritubular bath.

---

## Key Configuration Parameters

All parameters are set in **`values.py`**. The most commonly adjusted ones are:

### Grid Resolution
```python
NZ = 50       # Spatial grid points per segment (use 200 for production runs)
```

### Glomerular Filtration Rate
```python
sngfr0 = 24.0e-6 / 60.0   # Single-nephron GFR (nl/min converted to cm³/s)
sngfr  = sngfr0            # Working value — modify this for GFR perturbations
```

### Number of Nephrons (for whole-kidney scaling)
```python
Nneph = 72000   # Total nephron number in rat kidney
```

### Disease / Pharmacology Flags
```python
bdiabetes = False   # Set True to enable diabetic kidney simulation
furo      = False   # Set True to simulate furosemide (NKCC2 blocker)
```

### Tubular Compliance
```python
ncompl = 1   # 1 = compliant PT (default), 0 = rigid
ntorq  = 1   # 1 = include flow-dependent torque effects, 0 = off
```

### Solute Indices (for reference)
```python
NA=0, K=1, CL=2, HCO3=3, H2CO3=4, CO2=5,
HPO4=6, H2PO4=7, UREA=8, NH3=9, NH4=10, H=11,
HCO2=12, H2CO2=13, GLU=14, CA=15
```

---

## Output Files

All outputs are written to the working directory.

| File | Contents |
|---|---|
| `PToutlet`, `SDLoutlet`, ... `IMCoutlet` | Luminal and bath concentrations at segment outlet (16 solutes + pH, EP, volume, pressure) |
| `PToutlet_all`, `SDLoutlet_all`, ... | All compartment values at outlet (lumen, P, A, B, LIS, bath) |
| `mTALresults`, `cTALresults`, `DCTresults`, `CNTresults`, `CCDresults`, `OMCresults`, `IMCresults` | Converged cellular/LIS state at segment inlet — reused as initial guesses in subsequent runs |
| `Outlet_SS` | **Final urinary output**: 16 solute fluxes in pmol/min (whole kidney) |

The `*results` files from a completed run are automatically read as initial guesses the next time the model runs, speeding up convergence. On the first run these files will not exist and the solver uses hardcoded defaults from the `init*.py` modules.

---

## Code Structure

```
KidneyModel/
├── main.py                   # Entry point — runs the full nephron pipeline
├── values.py                 # All global parameters and constants
├── defs.py                   # Membrane data structure (concentration, volume, EP, area arrays)
├── glo.py                    # Global shared state (flux recording arrays)
│
├── init*.py                  # Segment initialization (membrane properties, transporter expressions)
├── qnewton*.py               # Newton-Raphson solvers (one per segment type)
├── fcn*.py                   # Residual functions (called by Newton solvers)
├── qflux*.py                 # Membrane flux evaluators (called by residual functions)
├── jacobi2_*.py              # Numerical Jacobian builders
│
├── compute_water_fluxes.py   # Kedem-Katchalsky osmotic water flux
├── compute_ecd_fluxes.py     # Electro-convective-diffusive paracellular flux
├── compute_nhe3_fluxes.py    # NHE3 (Na/H exchanger) kinetics
├── compute_nkcc2_flux.py     # NKCC2 (Na-K-2Cl cotransporter) kinetics
├── compute_kcc_fluxes.py     # KCC4 (K-Cl cotransporter) kinetics
├── compute_ncx_fluxes.py     # NCX (Na/Ca exchanger) kinetics
├── compute_sdl_water_fluxes.py
├── sglt.py                   # SGLT1/SGLT2 kinetics (PT glucose transport)
├── fatpase.py                # Na-K-ATPase and H-ATPase pump kinetics
├── set_intconc.py            # Cortical-medullary interstitial gradient
│
├── out_data_PT.py            # PT transport summary output
├── generate_flow_diagram.py  # Flow visualization
└── generate_report_pdf.py    # PDF report generation
```

---

## Physical Model

**Units:** CGS system — cm, mmol, mmol/cm³ (= mM), seconds  
**Species:** Female rat  
**Spatial discretization:** NZ finite-difference nodes along the tubule axis  
**Solution method:** Newton-Raphson at each spatial node, marching from inlet to outlet

### Processes modelled
- Osmotic and hydrostatic water transport (Kedem-Katchalsky)
- Electro-convective-diffusive paracellular fluxes
- Active transporters: Na-K-ATPase, H-ATPase, SGLT1/2, NKCC2, NCC, ENaC, NHE3, KCC4, NCX, pendrin
- Acid-base buffer equilibria: bicarbonate, phosphate, ammonia, carboxylate
- Electrical coupling: lumen potential and transepithelial voltage
- Cortical-medullary osmotic gradient (cortex → outer medulla → inner medulla → papilla)

---

## Notes

- The first run may be slow if `*results` files are absent — the solver uses default initial guesses and may require more Newton iterations to converge.
- Increase `NZ` from 50 to 200 for production-quality spatial resolution (expect ~4× longer runtime).
- The model is for **female rats**. Male rat parameters differ in transporter expressions and morphology.
