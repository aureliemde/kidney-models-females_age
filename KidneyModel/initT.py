"""
cTAL variable initialization module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Initializes all membrane parameters for the cTAL segment: surface areas,
water permeabilities, reflection coefficients, solute permeabilities,
NET coefficients for cotransporters/exchangers, carbonate kinetics, and
boundary conditions read from the mTAL outlet file.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2005)
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


def initT(ctal: List[Membrane]) -> None:
    """
    Initialize cTAL segment parameters.

    Args:
        ctal: List of Membrane objects for each spatial segment (0 to NZ)
    """
    _set_membrane_areas(ctal)
    _set_water_permeabilities(ctal)
    _set_reflection_coefficients(ctal)
    _set_solute_permeabilities(ctal)
    _set_net_coefficients(ctal)
    _set_carbonate_kinetics(ctal)
    _set_boundary_conditions(ctal)
    _set_metabolic_parameters(ctal)
    _check_electroneutrality(ctal)


def _set_membrane_areas(ctal: List[Membrane]) -> None:
    """
    Calculate initial membrane surface areas.

    Volumes, surfaces, and angles implicitly include the number of each
    type of cell (but the PtoIratio doesn't).

    Data source: Rat kidney (AMW model, AJP Renal 2005)

    Surface areas in cm²/cm² epithelium.
    """
    SlumPinittal = 2.0
    SbasPinittal = 2.0
    SlatPinittal = 10.0
    SlumEinittal = 0.001

    for p in ctal:
        p.sbasEinit = 0.020
        p.volPinit  = 5.0
        p.volEinit  = 0.40
        p.volAinit  = 5.0  # needed for water flux subroutine
        p.volBinit  = 5.0  # needed for water flux subroutine

        # Luminal interfaces
        p.area[LUM, P]   = SlumPinittal
        p.area[LUM, A]   = SlumPinittal
        p.area[LUM, B]   = SlumPinittal
        p.area[LUM, LIS] = SlumEinittal

        # Lateral interfaces
        p.area[P, LIS] = SlatPinittal
        p.area[A, LIS] = SlatPinittal
        p.area[B, LIS] = SlatPinittal

        # Basal interfaces
        p.area[P, BATH] = SbasPinittal
        p.area[A, BATH] = SbasPinittal
        p.area[B, BATH] = SbasPinittal

        # Enforce symmetry
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                p.area[l, k] = p.area[k, l]


def _set_water_permeabilities(ctal: List[Membrane]) -> None:
    """
    Set water permeability coefficients.

    Non-dimensional dLPV = Pf / Pfref, where Pf is the osmotic
    permeability (cm/s) already multiplied by area (cm²/cm² epith).
    """
    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = 33.0e-4 / 2.0 * 1.0e-3
    Pf[LUM, A]    = 0.0
    Pf[LUM, B]    = 0.0
    Pf[LUM, LIS]  = 0.70e-4 / 0.0010
    Pf[P,   LIS]  = 170e-4  / 10.0
    Pf[A,   LIS]  = 0.0
    Pf[B,   LIS]  = 0.0
    Pf[P,   BATH] = 33.0e-4 / 2.0
    Pf[A,   BATH] = 0.0
    Pf[B,   BATH] = 0.0
    Pf[LIS, BATH] = 8000.0e-4 / 0.020

    for p in ctal:
        p.dLPV[:] = Pf / Pfref


def _set_reflection_coefficients(ctal: List[Membrane]) -> None:
    """
    Set reflection coefficients for solute transport.

    sig = 1.0: membrane is impermeable to solute (perfect reflection)
    sig = 0.0: membrane is freely permeable (no reflection)

    All interfaces are impermeable except the basement membrane
    (LIS-BATH), which is freely permeable to all solutes.
    """
    for p in ctal:
        p.sig[:, :, :]    = 1.0
        p.sig[:, LIS, BATH] = 0.0
        p.sig[:, BATH, LIS] = 0.0


def _set_solute_permeabilities(ctal: List[Membrane]) -> None:
    """
    Set dimensional solute permeabilities h(I,K,L) at each interface.

    Initial values are in units of 10⁻⁵ cm/s; they are then
    non-dimensionalised by dividing by href.
    """
    for p in ctal:
        p.h[:] = 0.0

    # --- LUM-P (apical) interface ---
    for p in ctal:
        p.h[NA,    LUM, P] = 0.0
        p.h[K,     LUM, P] = 20.0
        p.h[CL,    LUM, P] = 0.0
        p.h[HCO3,  LUM, P] = 0.0
        p.h[H2CO3, LUM, P] = 130
        p.h[CO2,   LUM, P] = 1.50e4
        p.h[HPO4,  LUM, P] = 0.0
        p.h[H2PO4, LUM, P] = 0.0
        p.h[UREA,  LUM, P] = 0.06
        p.h[NH3,   LUM, P] = 1.5e3
        p.h[NH4,   LUM, P] = 4.0
        p.h[H,     LUM, P] = 2.0e3
        p.h[HCO2,  LUM, P] = 0.0
        p.h[H2CO2, LUM, P] = 0.0
        p.h[GLU,   LUM, P] = 0.0
        p.h[CA,    LUM, P] = 0.0001

    # --- P-LIS (lateral) interface ---
    for p in ctal:
        p.h[NA,    P, LIS] = 0.000
        p.h[K,     P, LIS] = 2.00
        p.h[CL,    P, LIS] = 0.50
        p.h[HCO3,  P, LIS] = 0.10
        p.h[H2CO3, P, LIS] = 130
        p.h[CO2,   P, LIS] = 1.50e4
        p.h[HPO4,  P, LIS] = 0.008
        p.h[H2PO4, P, LIS] = 0.008
        p.h[UREA,  P, LIS] = 0.06
        p.h[NH3,   P, LIS] = 1.0e3
        p.h[NH4,   P, LIS] = 0.40
        p.h[H,     P, LIS] = 2.0e3
        p.h[HCO2,  P, LIS] = 1.0e-4
        p.h[H2CO2, P, LIS] = 1.0e-4
        p.h[GLU,   P, LIS] = 1.0e-4
        p.h[CA,    P, LIS] = 0.0001

    # --- P-BATH (basal) interface: same as P-LIS ---
    for p in ctal:
        p.h[:, P, BATH] = p.h[:, P, LIS]

    # --- A/B cell interfaces: zero (dummy compartments in cTAL) ---
    # (no action needed — h is already zero from initialisation above)

    # --- LUM-LIS (tight junction / paracellular) interface ---
    ClTJperm = 1000.0  # = 1 / SlumEinittal, converts per-LIS-area to per-epith
    for p in ctal:
        p.h[NA,    LUM, LIS] = 2.80 * ClTJperm
        p.h[K,     LUM, LIS] = 3.00 * ClTJperm
        p.h[CL,    LUM, LIS] = 1.40 * ClTJperm
        p.h[HCO3,  LUM, LIS] = 0.50 * ClTJperm
        p.h[H2CO3, LUM, LIS] = 2.00 * ClTJperm
        p.h[CO2,   LUM, LIS] = 2.00 * ClTJperm
        p.h[HPO4,  LUM, LIS] = 0.20 * ClTJperm
        p.h[H2PO4, LUM, LIS] = 0.20 * ClTJperm
        p.h[UREA,  LUM, LIS] = 0.40 * ClTJperm
        p.h[NH3,   LUM, LIS] = 6.00 * ClTJperm
        p.h[NH4,   LUM, LIS] = 3.00 * ClTJperm
        p.h[H,     LUM, LIS] = 6.00 * ClTJperm
        p.h[HCO2,  LUM, LIS] = 0.01 * ClTJperm
        p.h[H2CO2, LUM, LIS] = 0.01 * ClTJperm
        p.h[GLU,   LUM, LIS] = 0.01 * ClTJperm
        p.h[CA,    LUM, LIS] = 8.40 * ClTJperm  # cTAL paracellular permeability to Ca²⁺

    # --- LIS-BATH (basement membrane) interface ---
    areafactor = 1 / 0.02
    for p in ctal:
        p.h[NA,    LIS, BATH] = 72.14  * areafactor
        p.h[K,     LIS, BATH] = 96.18  * areafactor
        p.h[CL,    LIS, BATH] = 96.18  * areafactor
        p.h[HCO3,  LIS, BATH] = 48.09  * areafactor
        p.h[H2CO3, LIS, BATH] = 72.14  * areafactor
        p.h[CO2,   LIS, BATH] = 72.14  * areafactor
        p.h[HPO4,  LIS, BATH] = 48.09  * areafactor
        p.h[H2PO4, LIS, BATH] = 48.09  * areafactor
        p.h[UREA,  LIS, BATH] = 48.09  * areafactor
        p.h[NH3,   LIS, BATH] = 60.11  * areafactor
        p.h[NH4,   LIS, BATH] = 96.18  * areafactor
        p.h[H,     LIS, BATH] = 480.90 * areafactor
        p.h[HCO2,  LIS, BATH] = 1.0    * areafactor
        p.h[H2CO2, LIS, BATH] = 1.0    * areafactor
        p.h[GLU,   LIS, BATH] = 1.0    * areafactor
        p.h[CA,    LIS, BATH] = p.h[NA, LIS, BATH] * (7.93 / 13.3)

    # Non-dimensionalise: divide by href (values were in 10⁻⁵ cm/s)
    for p in ctal:
        p.h *= 1.0e-5 / href


def _set_net_coefficients(ctal: List[Membrane]) -> None:
    """
    Set NET (non-equilibrium thermodynamic) coefficients for
    cotransporters and exchangers, and transporter expression scalings.
    """
    dLA = np.zeros((NS, NS, NC, NC))

    # Basolateral Cl/HCO3 exchanger
    dLA[CL,  HCO3, P, LIS]  = 3.0e-9
    dLA[CL,  HCO3, P, BATH] = dLA[CL,  HCO3, P, LIS]

    # Basolateral Na/H exchanger (NHE1)
    dLA[NA,  H,    P, LIS]  = 10.0e-9
    dLA[NA,  H,    P, BATH] = dLA[NA,  H,    P, LIS]

    # Basolateral Na²/HPO4 cotransporter
    dLA[NA,  HPO4, P, LIS]  = 0.50e-9
    dLA[NA,  HPO4, P, BATH] = dLA[NA,  HPO4, P, LIS]

    # Basolateral Na/3HCO3 exchanger
    dLA[NA,  HCO3, P, LIS]  = 0.50e-9
    dLA[NA,  HCO3, P, BATH] = dLA[NA,  HCO3, P, LIS]

    # Apical NKCC2 isoforms
    xNKCC2A = 1.3 * 15.0e-9 / 1.50
    xNKCC2B = 1.3 * 18.0e-9 / 1.50
    xNKCC2F = 1.3 * 0

    # Apical NHE3
    xNHE3 = 6.0e-9

    # Basolateral KCC4
    xKCC4 = 0.70e-9

    # Basolateral Na-K-ATPase
    ATPNaKPES = 1.3 * 1300.0e-9

    # Symmetry: dLA[j, i, k, l] = dLA[i, j, k, l]
    for i in range(NS - 1):
        for j in range(i + 1, NS):
            dLA[j, i, :, :] = dLA[i, j, :, :]

    # Non-dimensionalise and assign to membranes
    for p in ctal:
        p.dLA[:] = dLA / (href * Cref)

        p.xNKCC2A = xNKCC2A / (href * Cref)
        p.xNKCC2B = xNKCC2B / (href * Cref)
        p.xNKCC2F = xNKCC2F / (href * Cref)
        p.xNHE3   = xNHE3   / (href * Cref)
        p.xKCC4   = xKCC4   / (href * Cref)

        p.ATPNaK[P, LIS]  = ATPNaKPES / (href * Cref)
        p.ATPNaK[P, BATH] = ATPNaKPES / (href * Cref)


def _set_carbonate_kinetics(ctal: List[Membrane]) -> None:
    """
    Set CO₂ hydration/dehydration rate constants.

    dkh: hydration rate (CO₂ + H₂O → H⁺ + HCO₃⁻)
    dkd: dehydration rate (H⁺ + HCO₃⁻ → CO₂ + H₂O)

    cTAL has carbonic anhydrase in all three wet compartments
    (lumen, P cell, and LIS) — all use the catalysed rate.
    """
    dkhcat = 1.450e3
    dkdcat = 496.0e3

    for p in ctal:
        p.dkd[LUM] = dkdcat
        p.dkh[LUM] = dkhcat
        p.dkd[P]   = dkdcat
        p.dkh[P]   = dkhcat
        p.dkd[LIS] = dkdcat
        p.dkh[LIS] = dkhcat


def _set_boundary_conditions(ctal: List[Membrane]) -> None:
    """
    Set entering and peritubular conditions from the mTAL outlet file.

    The cortical position parameter (pos) decreases linearly from 1
    (medullary inlet) to 0 (cortical outlet), used by set_intconc to
    interpolate interstitial concentrations along the cTAL.
    """
    ctal[0].ep[BATH] = -0.001e-3 / EPref

    pos = np.zeros(NZ + 1)
    for jz in range(NZ + 1):
        ctal[jz].ph[BATH] = ctal[0].ph[BATH]
        pos[jz] = 1.0 * (NZ - jz) / NZ

    from set_intconc import set_intconc
    set_intconc(ctal, NZ, 1, pos)

    with open('mTALoutlet', 'r') as f:
        for i in range(NS):
            ctal[0].conc[i, LUM], ctal[0].conc[i, BATH] = map(float, f.readline().split())
        ctal[0].ph[LUM],  ctal[0].ph[BATH]  = map(float, f.readline().split())
        ctal[0].ep[LUM],  ctal[0].ep[BATH]  = map(float, f.readline().split())
        ctal[0].vol[LUM], ctal[0].pres      = map(float, f.readline().split())

    for p in ctal:
        p.volLuminit = ctal[0].vol[LUM]


def _set_metabolic_parameters(ctal: List[Membrane]) -> None:
    """
    Set metabolic parameters for O₂ consumption calculations.

    TQ = TNa-QO₂ ratio (Na⁺ transport per O₂ consumed).
    Normal: 15.0; diabetic: 12.0.
    """
    for p in ctal:
        p.TQ = 15.0 if not bdiabetes else 12.0


def _check_electroneutrality(ctal: List[Membrane]) -> None:
    """
    Verify electroneutrality in initial conditions.

    Computes net charge in lumen and bath compartments at the inlet
    (jz=0) and outlet (jz=NZ). Values are not asserted; kept for
    debugging compatibility with the original Fortran code.
    """
    elecM     = np.sum(zval * ctal[0].conc[:, LUM])
    elecS     = np.sum(zval * ctal[0].conc[:, BATH])
    elecS_out = np.sum(zval * ctal[NZ].conc[:, BATH])
