"""
mTAL (Medullary Thick Ascending Limb) initialization module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module initializes the mTAL segment parameters. The mTAL model yields
values of concentrations, volumes, and electrical potentials in the lumen
and epithelial compartments at steady-state equilibrium.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2005)
"""

import numpy as np
from typing import List
from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initA(mtal: List[Membrane]) -> None:
    """
    Initialize mTAL segment parameters including:
    - Membrane surface areas and cell volumes
    - Water permeabilities
    - Reflection coefficients
    - Solute permeabilities
    - NET (Onsager) transport coefficients and transporter activity parameters
    - Carbonic anhydrase kinetic parameters
    - Boundary conditions
    - Metabolic parameters

    Args:
        mtal: List of Membrane objects for each spatial segment (0 to NZ)

    Solute indices (0-based):
        0=Na⁺, 1=K⁺, 2=Cl⁻, 3=HCO₃⁻, 4=H₂CO₃, 5=CO₂,
        6=HPO₄²⁻, 7=H₂PO₄⁻, 8=urea, 9=NH₃, 10=NH₄⁺, 11=H⁺,
        12=HCO₂⁻, 13=H₂CO₂, 14=glucose, 15=Ca²⁺
    """
    _set_membrane_areas(mtal)
    _set_water_permeabilities(mtal)
    _set_reflection_coefficients(mtal)
    _set_solute_permeabilities(mtal)
    _set_net_coefficients(mtal)
    _set_kinetic_parameters(mtal)
    _set_boundary_conditions(mtal)
    _set_metabolic_parameters(mtal)
    _check_electroneutrality(mtal)


def _set_membrane_areas(mtal: List[Membrane]) -> None:
    """
    Set membrane surface areas and initial cell volumes.

    Data source: Rat kidney (AMW model, AJP Renal 2005)

    Surface areas in cm²/cm² epithelium:
    - SlumP: Luminal surface of principal cell
    - SbasP: Basal surface of principal cell
    - SlatP: Lateral surface of principal cell
    - SlumE: Luminal surface of lateral intercellular space (LIS)
    - SbasE: Basal surface of LIS

    In the mTAL only lumen (M), principal cell (P), LIS (E), and bath (S)
    compartments carry transport. A and B cells share the same geometry
    but their permeabilities are set to zero.
    """
    # Surface areas (cm²/cm² epithelium)
    SlumP = 2.0
    SbasP = 2.0
    SlatP = 10.0
    SlumE = 0.001
    SbasE = 0.020

    for membrane in mtal:
        membrane.sbasEinit = SbasE
        membrane.volPinit  = 5.0
        membrane.volEinit  = 0.40
        membrane.volAinit  = 5.0   # required by water flux routine
        membrane.volBinit  = 5.0

        # Luminal interfaces: lumen → cell or LIS
        membrane.area[LUM, P]   = SlumP
        membrane.area[LUM, A]   = SlumP   # same geometry, zero permeability
        membrane.area[LUM, B]   = SlumP
        membrane.area[LUM, LIS] = SlumE

        # Lateral interfaces: cell → LIS
        membrane.area[P,   LIS] = SlatP
        membrane.area[A,   LIS] = SlatP
        membrane.area[B,   LIS] = SlatP

        # Basal interfaces: cell → bath
        membrane.area[P,   BATH] = SbasP
        membrane.area[A,   BATH] = SbasP
        membrane.area[B,   BATH] = SbasP

        # Enforce symmetry: area[l, k] = area[k, l]
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                membrane.area[l, k] = membrane.area[k, l]


def _set_water_permeabilities(mtal: List[Membrane]) -> None:
    """
    Set osmotic water permeability coefficients (non-dimensionalized by Pfref).

    Pf(K, L) = osmotic permeability at the K-L interface (cm/s)
    Non-dimensional dLPV = Pf / Pfref

    Units of dimensional water flux: cm³/s/cm² epith
    """
    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = 33.0e-4 / 2.0 * 1.0e-3   # Lumen-Principal (very low in mTAL)
    Pf[LUM, A]    = 0.0
    Pf[LUM, B]    = 0.0
    Pf[LUM, LIS]  = 0.70e-4 / 0.001           # Lumen-LIS (tight junction)
    Pf[P,   LIS]  = 170.0e-4 / 10.0           # Principal-LIS (lateral)
    Pf[A,   LIS]  = 0.0
    Pf[B,   LIS]  = 0.0
    Pf[P,   BATH] = 33.0e-4 / 2.0             # Principal-Bath (basal)
    Pf[A,   BATH] = 0.0
    Pf[B,   BATH] = 0.0
    Pf[LIS, BATH] = 8000.0e-4 / 0.020         # LIS-Bath (basement membrane)

    dLPV = Pf / Pfref

    for membrane in mtal:
        membrane.dLPV[:] = dLPV


def _set_reflection_coefficients(mtal: List[Membrane]) -> None:
    """
    Set reflection coefficients sig(I, K, L).

    sig = 1.0 everywhere (default: impermeable).
    The LIS-BATH basement membrane is freely permeable (sig = 0).
    """
    for membrane in mtal:
        membrane.sig[:, :, :] = 1.0
        membrane.sig[:, LIS, BATH] = 0.0
        membrane.sig[:, BATH, LIS] = 0.0


def _set_solute_permeabilities(mtal: List[Membrane]) -> None:
    """
    Set solute permeabilities h(I, K, L) and non-dimensionalize.

    Raw values in units of 1e-5 cm/s; scaled by 1e-5/href to non-dimensionalize.
    A and B cell interfaces have zero permeability (no transport in mTAL).

    Interfaces populated:
    - LUM-P:    apical membrane of principal cell
    - P-LIS:    lateral membrane of principal cell
    - P-BATH:   basal membrane of principal cell (= P-LIS)
    - LUM-LIS:  tight junction (scaled by ClTJperm = 1000)
    - LIS-BATH: basement membrane (scaled by areafactor = 1/0.02)
    """
    ClTJperm   = 1000.0       # 1 / SlumE (tight junction area factor)
    areafactor = 1.0 / 0.020  # 1 / SbasE

    # Apical (LUM-P) permeabilities
    h_MP = np.array([
        0.0,      # Na⁺
        20.0,     # K⁺
        0.0,      # Cl⁻
        0.0,      # HCO₃⁻
        130.0,    # H₂CO₃
        1.50e4,   # CO₂
        0.0,      # HPO₄²⁻
        0.0,      # H₂PO₄⁻
        0.06,     # urea
        1.5e3,    # NH₃
        4.0,      # NH₄⁺
        2.0e3,    # H⁺
        0.0,      # HCO₂⁻
        0.0,      # H₂CO₂
        0.001,    # glucose
        0.0001,   # Ca²⁺
    ])

    # Lateral/basal (P-LIS = P-BATH) permeabilities
    h_PE = np.array([
        0.0,      # Na⁺
        2.0,      # K⁺
        0.5,      # Cl⁻
        0.1,      # HCO₃⁻
        130.0,    # H₂CO₃
        1.50e4,   # CO₂
        0.008,    # HPO₄²⁻
        0.008,    # H₂PO₄⁻
        0.06,     # urea
        1.0e3,    # NH₃
        0.40,     # NH₄⁺
        2.0e3,    # H⁺
        1.0e-4,   # HCO₂⁻
        1.0e-4,   # H₂CO₂
        0.001,    # glucose
        0.0001,   # Ca²⁺
    ])

    # Tight junction (LUM-LIS) permeabilities
    h_ME = ClTJperm * np.array([
        2.80,   # Na⁺
        3.00,   # K⁺
        1.40,   # Cl⁻
        0.50,   # HCO₃⁻
        2.00,   # H₂CO₃
        2.00,   # CO₂
        0.20,   # HPO₄²⁻
        0.20,   # H₂PO₄⁻
        0.40,   # urea
        6.00,   # NH₃
        3.00,   # NH₄⁺
        6.00,   # H⁺
        0.01,   # HCO₂⁻
        0.01,   # H₂CO₂
        0.01,   # glucose
        4.20,   # Ca²⁺ (paracellular, mTAL-specific)
    ])

    # Basement membrane (LIS-BATH) permeabilities
    h_ES_base = areafactor * np.array([
        72.14,    # Na⁺
        96.18,    # K⁺
        96.18,    # Cl⁻
        48.09,    # HCO₃⁻
        72.14,    # H₂CO₃
        72.14,    # CO₂
        48.09,    # HPO₄²⁻
        48.09,    # H₂PO₄⁻
        48.09,    # urea
        60.11,    # NH₃
        96.18,    # NH₄⁺
        480.90,   # H⁺
        1.0,      # HCO₂⁻
        1.0,      # H₂CO₂
        1.0,      # glucose
    ])
    # Ca²⁺: scaled relative to Na⁺ by NCX ratio (7.93/13.3)
    h_ES = np.append(h_ES_base, areafactor * 72.14 * (7.93 / 13.3))

    # Non-dimensionalize and assign
    scale = 1.0e-5 / href

    for membrane in mtal:
        membrane.h[:, LUM, P]    = h_MP * scale
        membrane.h[:, P,   LIS]  = h_PE * scale
        membrane.h[:, P,   BATH] = h_PE * scale   # basal = lateral
        membrane.h[:, LUM, LIS]  = h_ME * scale
        membrane.h[:, LIS, BATH] = h_ES * scale
        # A and B cells: zero permeability (already initialized to 0)


def _set_net_coefficients(mtal: List[Membrane]) -> None:
    """
    Set NET (Onsager) cross-transport coefficients and specific transporter
    activity parameters.

    dLA(I, J, K, L): coupled transport coefficient for solutes I and J at
    the K-L interface. Units: mmol²/J/s; non-dimensionalized by href*Cref.

    Basolateral transporters in principal cell (P):
    - Cl⁻/HCO₃⁻ exchanger (AE)
    - Na⁺/H⁺ exchanger (NHE1)
    - Na⁺/HPO₄²⁻ cotransporter
    - Na⁺/HCO₃⁻ cotransporter (NBC)

    Specific transporter activity parameters (apical/basolateral):
    - NKCC2 A and F isoforms (apical Na-K-2Cl cotransporter)
    - NHE3 (apical Na⁺/H⁺ exchanger isoform 3)
    - KCC4 (basolateral K⁺-Cl⁻ cotransporter isoform 4)
    - Na-K-ATPase (basolateral)
    """
    scale = 1.0 / (href * Cref)

    for membrane in mtal:
        membrane.dLA[:, :, :, :] = 0.0

        # Basolateral Cl⁻/HCO₃⁻ exchanger
        membrane.dLA[CL,   HCO3, P, LIS]  = 3.0e-9 * scale
        membrane.dLA[CL,   HCO3, P, BATH] = 3.0e-9 * scale
        membrane.dLA[HCO3, CL,   P, LIS]  = membrane.dLA[CL, HCO3, P, LIS]
        membrane.dLA[HCO3, CL,   P, BATH] = membrane.dLA[CL, HCO3, P, BATH]

        # Basolateral Na⁺/H⁺ exchanger (NHE1)
        membrane.dLA[NA, H,  P, LIS]  = 10.0e-9 * scale
        membrane.dLA[NA, H,  P, BATH] = 10.0e-9 * scale
        membrane.dLA[H,  NA, P, LIS]  = membrane.dLA[NA, H, P, LIS]
        membrane.dLA[H,  NA, P, BATH] = membrane.dLA[NA, H, P, BATH]

        # Basolateral Na⁺/HPO₄²⁻ cotransporter
        membrane.dLA[NA,   HPO4, P, LIS]  = 0.50e-9 * scale
        membrane.dLA[NA,   HPO4, P, BATH] = 0.50e-9 * scale
        membrane.dLA[HPO4, NA,   P, LIS]  = membrane.dLA[NA, HPO4, P, LIS]
        membrane.dLA[HPO4, NA,   P, BATH] = membrane.dLA[NA, HPO4, P, BATH]

        # Basolateral Na⁺/HCO₃⁻ cotransporter (NBC)
        membrane.dLA[NA,   HCO3, P, LIS]  = 0.50e-9 * scale
        membrane.dLA[NA,   HCO3, P, BATH] = 0.50e-9 * scale
        membrane.dLA[HCO3, NA,   P, LIS]  = membrane.dLA[NA, HCO3, P, LIS]
        membrane.dLA[HCO3, NA,   P, BATH] = membrane.dLA[NA, HCO3, P, BATH]

    # Specific transporter activity parameters (non-dimensionalized)
    xNKCC2A   = 1.3 * 12.0e-9   / 1.50 * scale   # NKCC2 A isoform (apical)
    xNKCC2F   = 1.3 * 75.0e-9   / 1.50 * scale   # NKCC2 F isoform (apical, dominant)
    xNHE3     = 6.0e-9                  * scale   # NHE3 (apical)
    xKCC4     = 0.70e-9                 * scale   # KCC4 (basolateral)
    ATPNaKPES = 1.3 * 1300.0e-9        * scale   # Na-K-ATPase (basolateral)

    for membrane in mtal:
        membrane.xNKCC2A         = xNKCC2A
        membrane.xNKCC2F         = xNKCC2F
        membrane.xNHE3           = xNHE3
        membrane.xKCC4           = xKCC4
        membrane.ATPNaK[P, LIS]  = ATPNaKPES
        membrane.ATPNaK[P, BATH] = ATPNaKPES


def _set_kinetic_parameters(mtal: List[Membrane]) -> None:
    """
    Set carbonic anhydrase kinetic rate constants.

    dkh = hydration rate (CO₂ + H₂O → H₂CO₃, s⁻¹)
    dkd = dehydration rate (H₂CO₃ → CO₂ + H₂O, s⁻¹)

    The mTAL uses catalyzed (high) rates in the lumen, principal cell, and LIS.
    """
    dkh_cat = 1.450e3   # catalyzed hydration rate (s⁻¹)
    dkd_cat = 496.0e3   # catalyzed dehydration rate (s⁻¹)

    for membrane in mtal:
        membrane.dkh[LUM] = dkh_cat
        membrane.dkd[LUM] = dkd_cat
        membrane.dkh[P]   = dkh_cat
        membrane.dkd[P]   = dkd_cat
        membrane.dkh[LIS] = dkh_cat
        membrane.dkd[LIS] = dkd_cat


def _set_boundary_conditions(mtal: List[Membrane]) -> None:
    """
    Set boundary conditions from the SDL outlet and interstitial gradient.

    The mTAL inlet (jz=0) is read from SDLoutlet. Interstitial concentrations
    vary along the mTAL from inner medulla (jz=0) to outer stripe (jz=NZ),
    capturing the cortico-medullary osmotic gradient.
    """
    # Bath electrical potential (non-dimensionalized)
    ep_bath = -0.001e-3 / EPref
    for membrane in mtal:
        membrane.ep[BATH] = ep_bath

    # Read mTAL inlet from SDL outlet
    # File format: NS rows of [lumen, bath] concentrations,
    #              then [pH_lum, pH_bath], [ep_lum, ep_bath], [vol_lum, pres]
    inlet = np.loadtxt('SDLoutlet', max_rows=NS + 3)

    mtal[0].conc[:, LUM]  = inlet[:NS, 0]
    mtal[0].conc[:, BATH] = inlet[:NS, 1]
    mtal[0].ph[LUM],  mtal[0].ph[BATH] = inlet[NS]
    mtal[0].ep[LUM],  mtal[0].ep[BATH] = inlet[NS + 1]
    mtal[0].vol[LUM], mtal[0].pres     = inlet[NS + 2]

    for membrane in mtal:
        membrane.volLuminit = mtal[0].vol[LUM]

    # Position array: 1.0 at inlet (inner medulla) → 0.0 at outlet (outer stripe)
    pos = np.linspace(1.0, 0.0, NZ + 1)

    set_intconc(mtal, NZ, 2, pos)


def _set_metabolic_parameters(mtal: List[Membrane]) -> None:
    """
    Set TQ: TNa-QO₂ ratio for O₂ consumption calculations.

    Normal mTAL:   TQ = 15.0
    Diabetic mTAL: TQ = 12.0
    """
    tq_value = 12.0 if bdiabetes else 15.0

    for membrane in mtal:
        membrane.TQ = tq_value


def _check_electroneutrality(mtal: List[Membrane]) -> None:
    """
    Verify electroneutrality in initial conditions at jz=0 and jz=NZ.

    Computes net charge Σ(z_i × C_i) in lumen and bath.
    Values near zero indicate proper initialization.
    """
    membrane_0  = mtal[0]
    membrane_NZ = mtal[NZ]

    elecM     = np.sum(zval * membrane_0.conc[:,  LUM])
    elecS     = np.sum(zval * membrane_0.conc[:,  BATH])
    elecS_out = np.sum(zval * membrane_NZ.conc[:, BATH])
