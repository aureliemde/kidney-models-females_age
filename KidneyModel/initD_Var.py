"""
DCT variable initialization module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Initializes all membrane parameters for the DCT segment: surface areas,
water permeabilities, reflection coefficients, solute permeabilities,
NET coefficients for cotransporters/exchangers, Ca²⁺ transporter
scalings, carbonate kinetics, and boundary conditions read from the
cTAL outlet file.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2005)
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


def initD_Var(dct: List[Membrane]) -> float:
    """
    Initialize DCT segment parameters.

    Args:
        dct: List of Membrane objects for each spatial segment (0 to NZ)

    Returns:
        xTRPV5_dct: Non-dimensional TRPV5 Ca²⁺ channel scaling factor
    """
    _set_membrane_areas(dct)
    _set_water_permeabilities(dct)
    _set_reflection_coefficients(dct)
    _set_solute_permeabilities(dct)
    xTRPV5_dct = _set_net_coefficients(dct)
    _set_carbonate_kinetics(dct)
    _set_boundary_conditions(dct)
    _set_metabolic_parameters(dct)
    _check_electroneutrality(dct)
    return xTRPV5_dct


def _set_membrane_areas(dct: List[Membrane]) -> None:
    """
    Calculate initial membrane surface areas.

    Volumes, surfaces, and angles implicitly include the number of each
    type of cell (but the PtoIratio doesn't).

    Data source: Rat kidney (AMW model, AJP Renal 2005)

    Surface areas in cm²/cm² epithelium.
    """
    SlumPinitdct = 4.7
    SbasPinitdct = 4.7
    SlatPinitdct = 64.3
    SlumEinitdct = 0.001
    SlumAinitdct = 1.0
    SbasAinitdct = 1.0
    SlatAinitdct = 1.0
    SlumBinitdct = 1.0
    SbasBinitdct = 1.0
    SlatBinitdct = 1.0

    # VolPinitdct = 7.5   # declared in v0 but unused (value set directly on p.volPinit)
    # VolEinitdct = 0.80  # declared in v0 but unused (value set directly on p.volEinit)
    # VolAinitdct = 1.0   # declared in v0 but unused (value set directly on p.volAinit)
    # VolBinitdct = 1.0   # declared in v0 but unused (value set directly on p.volBinit)
    for p in dct:
        p.sbasEinit = 0.020
        p.volPinit  = 7.5
        p.volEinit  = 0.80
        p.volAinit  = 7.5
        p.volBinit  = 7.5

        # Luminal interfaces
        p.area[LUM, P]   = SlumPinitdct
        p.area[LUM, A]   = SlumAinitdct
        p.area[LUM, B]   = SlumBinitdct
        p.area[LUM, LIS] = SlumEinitdct

        # Lateral interfaces
        p.area[P,   LIS] = SlatPinitdct
        p.area[A,   LIS] = SlatAinitdct
        p.area[B,   LIS] = SlatBinitdct

        # Basal interfaces
        p.area[P,   BATH] = SbasPinitdct
        p.area[A,   BATH] = SbasAinitdct
        p.area[B,   BATH] = SbasBinitdct

        # Enforce symmetry
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                p.area[l, k] = p.area[k, l]


def _set_water_permeabilities(dct: List[Membrane]) -> None:
    """
    Set water permeability coefficients.

    Non-dimensional dLPV = Pf / Pfref, where Pf is the osmotic
    permeability (cm/s) already multiplied by area (cm²/cm² epith).
    """
    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = 2.4 * 0.00117
    Pf[LUM, A]    = 0.0
    Pf[LUM, B]    = 0.0
    Pf[LUM, LIS]  = 2.0
    Pf[P,   LIS]  = 2.4 * 0.00835
    Pf[A,   LIS]  = 0.0
    Pf[B,   LIS]  = 0.0
    Pf[P,   BATH] = 2.4 * 0.00835
    Pf[A,   BATH] = 0.0
    Pf[B,   BATH] = 0.0
    Pf[LIS, BATH] = 35.5

    dLPV = Pf / Pfref
    for p in dct:
        p.dLPV[:] = dLPV


def _set_reflection_coefficients(dct: List[Membrane]) -> None:
    """
    Set reflection coefficients for solute transport.

    sig = 1.0: membrane is impermeable to solute (perfect reflection)
    sig = 0.0: membrane is freely permeable (no reflection)

    All interfaces are impermeable except the basement membrane
    (LIS-BATH), which is freely permeable to all solutes.
    """
    for p in dct:
        p.sig[:, :, :]    = 1.0
        p.sig[:, LIS, BATH] = 0.0
        p.sig[:, BATH, LIS] = 0.0


def _set_solute_permeabilities(dct: List[Membrane]) -> None:
    """
    Set dimensional solute permeabilities h(I,K,L) at each interface.

    Initial values are in units of 10⁻⁵ cm/s; they are then
    non-dimensionalised by dividing by href.

    ENaC and ROMK permeabilities (hNaMP) are stored separately and
    updated each Newton step by qflux2D based on local pH and [Na⁺].
    """
    for p in dct:
        p.h[:] = 0.0

    # --- LUM-P (apical) interface ---
    for p in dct:
        p.hNaMP              = 14.00 * 0.25  # max ENaC perm (×1/4 for area)
        p.h[NA,    LUM, P]   = 0.072
        p.h[K,     LUM, P]   = 0.60
        p.h[CL,    LUM, P]   = 0.0
        p.h[HCO3,  LUM, P]   = 0.0
        p.h[H2CO3, LUM, P]   = 130
        p.h[CO2,   LUM, P]   = 1.50e4
        p.h[HPO4,  LUM, P]   = 0.0
        p.h[H2PO4, LUM, P]   = 0.0
        p.h[UREA,  LUM, P]   = 0.20
        p.h[NH3,   LUM, P]   = 200
        p.h[NH4,   LUM, P]   = 0.12
        p.h[H,     LUM, P]   = 0.20
        p.h[HCO2,  LUM, P]   = 0
        p.h[H2CO2, LUM, P]   = 0
        p.h[GLU,   LUM, P]   = 0.0
        p.h[CA,    LUM, P]   = 0.0

    # --- P-LIS (lateral) interface ---
    for p in dct:
        p.h[NA,    P, LIS]   = 0.000
        p.h[K,     P, LIS]   = 0.12
        p.h[CL,    P, LIS]   = 0.04
        p.h[HCO3,  P, LIS]   = 0.02
        p.h[H2CO3, P, LIS]   = 130
        p.h[CO2,   P, LIS]   = 1.50e4
        p.h[HPO4,  P, LIS]   = 0.002
        p.h[H2PO4, P, LIS]   = 0.002
        p.h[UREA,  P, LIS]   = 0.2
        p.h[NH3,   P, LIS]   = 200
        p.h[NH4,   P, LIS]   = 0.0234
        p.h[H,     P, LIS]   = 0.20
        p.h[HCO2,  P, LIS]   = 1.0e-4
        p.h[H2CO2, P, LIS]   = 1.0e-4
        p.h[GLU,   P, LIS]   = 1.0e-4
        p.h[CA,    P, LIS]   = 0.0

    # --- P-BATH (basal) interface: same as P-LIS ---
    for p in dct:
        p.h[:, P, BATH] = p.h[:, P, LIS]

    # --- A/B cell interfaces: zero (dummy compartments in DCT) ---
    for p in dct:
        p.h[:, LUM, A]  = 0
        p.h[:, A,   LIS] = 0
        p.h[:, A,   BATH] = 0
        p.h[:, LUM, B]  = 0
        p.h[:, B,   LIS] = 0
        p.h[:, B,   BATH] = 0

    # --- LUM-LIS (tight junction / paracellular) interface ---
    ClTJperm = 1000.0  # = 1 / SlumEinitdct, converts per-LIS-area to per-epith
    for p in dct:
        p.h[NA,    LUM, LIS] = 0.80  * ClTJperm
        p.h[K,     LUM, LIS] = 0.80  * ClTJperm
        p.h[CL,    LUM, LIS] = 1.3 * 0.50 * ClTJperm
        p.h[HCO3,  LUM, LIS] = 0.50  * ClTJperm
        p.h[H2CO3, LUM, LIS] = 0.50  * ClTJperm
        p.h[CO2,   LUM, LIS] = 0.50  * ClTJperm
        p.h[HPO4,  LUM, LIS] = 0.10  * ClTJperm
        p.h[H2PO4, LUM, LIS] = 0.10  * ClTJperm
        p.h[UREA,  LUM, LIS] = 0.20  * ClTJperm
        p.h[NH3,   LUM, LIS] = 0.80  * ClTJperm
        p.h[NH4,   LUM, LIS] = 0.80  * ClTJperm
        p.h[H,     LUM, LIS] = 0.80  * ClTJperm
        p.h[HCO2,  LUM, LIS] = 0.01  * ClTJperm
        p.h[H2CO2, LUM, LIS] = 0.01  * ClTJperm
        p.h[GLU,   LUM, LIS] = 0.01  * ClTJperm
        p.h[CA,    LUM, LIS] = 0.0001 * ClTJperm  # TO BE ADJUSTED

    # --- LIS-BATH (basement membrane) interface ---
    areafactor = 1 / 0.02
    for p in dct:
        p.h[NA,    LIS, BATH] = 63.0  * areafactor
        p.h[K,     LIS, BATH] = 84.0  * areafactor
        p.h[CL,    LIS, BATH] = 84.0  * areafactor
        p.h[HCO3,  LIS, BATH] = 42.0  * areafactor
        p.h[H2CO3, LIS, BATH] = 63.0  * areafactor
        p.h[CO2,   LIS, BATH] = 63.0  * areafactor
        p.h[HPO4,  LIS, BATH] = 42.0  * areafactor
        p.h[H2PO4, LIS, BATH] = 42.0  * areafactor
        p.h[UREA,  LIS, BATH] = 42.0  * areafactor
        p.h[NH3,   LIS, BATH] = 52.0  * areafactor
        p.h[NH4,   LIS, BATH] = 84.0  * areafactor
        p.h[H,     LIS, BATH] = 419.0 * areafactor
        p.h[HCO2,  LIS, BATH] = 1.0   * areafactor
        p.h[H2CO2, LIS, BATH] = 1.0   * areafactor
        p.h[GLU,   LIS, BATH] = 1.0   * areafactor
        p.h[CA,    LIS, BATH] = p.h[NA, LIS, BATH] * (7.93 / 13.3)

    # Non-dimensionalise: divide by href (values were in 10⁻⁵ cm/s)
    for p in dct:
        p.h *= 1.0e-5 / href


def _set_net_coefficients(dct: List[Membrane]) -> float:
    """
    Set NET (non-equilibrium thermodynamic) coefficients for
    cotransporters and exchangers, and transporter expression scalings.

    Returns:
        xTRPV5_dct: Non-dimensional TRPV5 expression scaling factor
    """
    dLA = np.zeros((NS, NS, NC, NC))

    # Apical NHE3
    xNHE3 = 200.0e-9

    # Apical NCC
    xNCC = 1.8 * 15.0e-9

    # Apical and basolateral K-Cl cotransporter
    dLA[K,   CL,   LUM, P]    = 4.0e-9
    dLA[K,   CL,   P,   LIS]  = 20.0e-9
    dLA[K,   CL,   P,   BATH] = 20.0e-9

    # Basolateral Na/H exchanger (NHE1)
    # xNHE1 = 6.95e-10  # alternative NHE1 scaling from v0 — unused
    dLA[NA,  H,    P,   LIS]  = 4.0e-9
    dLA[NA,  H,    P,   BATH] = 4.0e-9

    # Basolateral Cl/HCO3 exchanger (AE)
    dLA[CL,  HCO3, P,   LIS]  = 15.0e-9
    dLA[CL,  HCO3, P,   BATH] = 15.0e-9

    # Basolateral Na/HPO4 cotransporter
    dLA[NA,  HPO4, P,   LIS]  = 0.2e-9
    dLA[NA,  HPO4, P,   BATH] = 0.2e-9

    # Na-K-ATPase
    ATPNaKPES = 1.8 * 400.0e-9

    # Basolateral NCX exchanger
    xNCX = 50.0e-9 * 1.40

    # Basolateral PMCA pump
    PMCA = 0.70e-9 * 0.60

    # Apical TRPV5
    xTRPV5 = 13.5e6

    # Symmetry: dLA[j, i, k, l] = dLA[i, j, k, l]
    for i in range(NS - 1):
        for j in range(i + 1, NS):
            dLA[j, i, :, :] = dLA[i, j, :, :]

    # Non-dimensionalise and assign to membranes
    for p in dct:
        p.dLA[:] = dLA / (href * Cref)

        p.xNHE3 = xNHE3 / (href * Cref)
        p.xNCC  = xNCC  / (href * Cref)
        p.xNCX  = xNCX  / (href * Cref)
        p.PMCA  = PMCA  / (href * Cref)

        p.ATPNaK[P, LIS]  = ATPNaKPES / (href * Cref)
        p.ATPNaK[P, BATH] = ATPNaKPES / (href * Cref)

    xTRPV5_dct = xTRPV5 / (href * Cref)
    return xTRPV5_dct


def _set_carbonate_kinetics(dct: List[Membrane]) -> None:
    """
    Set CO₂ hydration/dehydration rate constants.

    dkh: hydration rate (CO₂ + H₂O → H⁺ + HCO₃⁻)
    dkd: dehydration rate (H⁺ + HCO₃⁻ → CO₂ + H₂O)

    Uncatalysed rate applied ×10 in lumen; carbonic-anhydrase-catalysed
    rates applied in P cell and LIS.
    """
    dkhuncat = 0.145
    dkduncat = 49.6
    dkhP = 145
    dkdP = 49600
    dkhE = 145
    dkdE = 49600

    for p in dct:
        p.dkd[LUM] = dkduncat * 10.0
        p.dkh[LUM] = dkhuncat * 10.0
        p.dkd[P]   = dkdP
        p.dkh[P]   = dkhP
        p.dkd[LIS] = dkdE
        p.dkh[LIS] = dkhE


def _set_boundary_conditions(dct: List[Membrane]) -> None:
    """
    Set entering and peritubular conditions from the cTAL outlet file.

    The cTAL outlet concentrations, pH, electrical potential, and luminal
    volume become the DCT inlet boundary conditions. Interstitial
    concentrations are set via set_intconc (all cortical: pos = 0).
    """
    with open('cTALoutlet', 'r') as f:
        for i in range(NS):
            dct[0].conc[i, LUM], dct[0].conc[i, BATH] = map(float, f.readline().split())
        dct[0].ph[LUM],  dct[0].ph[BATH]  = map(float, f.readline().split())
        dct[0].ep[LUM],  dct[0].ep[BATH]  = map(float, f.readline().split())
        dct[0].vol[LUM], dct[0].pres      = map(float, f.readline().split())

    for p in dct:
        p.volLuminit = dct[0].vol[LUM]
        p.ep[BATH]   = dct[0].ep[BATH]

    # All DCT segments are cortical: position = 0
    pos = np.zeros(NZ + 1)
    from set_intconc import set_intconc
    set_intconc(dct, NZ, 1, pos)


def _set_metabolic_parameters(dct: List[Membrane]) -> None:
    """
    Set metabolic parameters for O₂ consumption calculations.

    TQ = TNa-QO₂ ratio (Na⁺ transport per O₂ consumed).
    Normal: 15.0; diabetic: 12.0.
    """
    for p in dct:
        p.TQ = 15.0 if not bdiabetes else 12.0


def _check_electroneutrality(dct: List[Membrane]) -> None:
    """
    Verify electroneutrality in initial conditions.

    Computes net charge in lumen and bath compartments at the inlet
    (jz=0) and outlet (jz=NZ). Values are not asserted; kept for
    debugging compatibility with the original Fortran code.
    """
    elecM     = np.sum(zval * dct[0].conc[:, LUM])
    elecS     = np.sum(zval * dct[0].conc[:, BATH])
    elecS_out = np.sum(zval * dct[NZ].conc[:, BATH])
