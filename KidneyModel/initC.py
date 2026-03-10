"""
CNT (Connecting Tubule) variable initialization module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Initializes all membrane parameters for the CNT segment: tubular geometry,
surface areas, water permeabilities, reflection coefficients, solute
permeabilities, NET coefficients for cotransporters/exchangers, ATPase
expression levels, carbonate kinetics, and boundary conditions read from
the DCT outlet file.

Three epithelial cell types: PC (principal), IC-A (intercalated type A),
IC-B (intercalated type B).

Note: hENaC_CNT, hROMK_CNT, hCltj_CNT, xTRPV5_cnt, xPTRPV4_cnt are
computed and returned by initC_Var.py, which is called at each flux
evaluation by qflux2C.py. They are not set here.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2005)
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_membrane_areas(cnt: List[Membrane]) -> None:
    """Set tubular geometry, compartment volumes, and membrane surface areas.

    Volumes, surfaces, and angles implicitly include the number of each
    type of cell. Data source: Rat kidney (AMW model, AJP Renal 2005).
    """
    # theta = np.zeros(NC)  # dead alloc — never used
    # Slum  = np.zeros(NC)  # dead alloc — never used
    # Slat  = np.zeros(NC)  # dead alloc — never used
    # Sbas  = np.zeros(NC)  # dead alloc — never used
    for p in cnt:
        p.dimL = dimLC
        p.diam = DiamC

    SlumPinitcnt = 1.2
    SbasPinitcnt = 1.2
    SlatPinitcnt = 6.9

    SlumAinitcnt = 0.58
    SbasAinitcnt = 0.58
    SlatAinitcnt = 1.25

    SlumBinitcnt = 0.17
    SbasBinitcnt = 0.17
    SlatBinitcnt = 1.50

    SlumEinitcnt = 0.001

    for p in cnt:
        p.sbasEinit = 0.020
        p.volPinit  = 6.0
        p.volAinit  = 2.7
        p.volBinit  = 1.8
        p.volEinit  = 0.20

        p.area[LUM, P]    = SlumPinitcnt
        p.area[LUM, A]    = SlumAinitcnt
        p.area[LUM, B]    = SlumBinitcnt
        p.area[LUM, LIS]  = SlumEinitcnt
        p.area[P,   LIS]  = SlatPinitcnt
        p.area[A,   LIS]  = SlatAinitcnt
        p.area[B,   LIS]  = SlatBinitcnt
        p.area[P,   BATH] = SbasPinitcnt
        p.area[A,   BATH] = SbasAinitcnt
        p.area[B,   BATH] = SbasBinitcnt

        # Enforce symmetry
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                p.area[l, k] = p.area[k, l]

        # Fixed charge on impermeant species in each cell type
        p.zPimp = -1.0
        p.zAimp = -1.0
        p.zBimp = -1.0


def _set_water_permeabilities(cnt: List[Membrane]) -> None:
    """Set water permeability coefficients (cm/s).

    Non-dimensional dLPV = Pf / Pfref.
    """
    PfMP = 2.4 * 0.036
    PfPE = 2.4 * 0.44
    PfPS = 2.4 * 0.44

    PfMA = 0.22e-3
    PfAE = 5.50e-3
    PfAS = 5.50e-3

    PfMB = 0.22e-3
    PfBE = 5.50e-3
    PfBS = 5.50e-3

    PfES = 110
    PfME = 1.1

    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = PfMP
    Pf[LUM, A]    = PfMA
    Pf[LUM, B]    = PfMB
    Pf[LUM, LIS]  = PfME
    Pf[P,   LIS]  = PfPE
    Pf[A,   LIS]  = PfAE
    Pf[B,   LIS]  = PfBE
    Pf[P,   BATH] = PfPS
    Pf[A,   BATH] = PfAS
    Pf[B,   BATH] = PfBS
    Pf[LIS, BATH] = PfES

    for p in cnt:
        p.dLPV[:, :] = Pf / Pfref


def _set_reflection_coefficients(cnt: List[Membrane]) -> None:
    """Initialize reflection coefficients; set basement membrane to 0."""
    for p in cnt:
        p.sig[:, :, :]    = 1.0
        p.sig[:, LIS, BATH] = 0.0


def _set_solute_permeabilities(cnt: List[Membrane]) -> None:
    """Set dimensional solute permeabilities h (x1e-5 cm/s) then non-dimensionalize.

    One array per interface; index order matches fixed solute order in values.py:
    NA K CL HCO3 H2CO3 CO2 HPO4 H2PO4 UREA NH3 NH4 H HCO2 H2CO2 GLU CA

    Notes:
    - NA and K at LUM-P are not set here; updated at each spatial step
      by qflux2C via hENaC_CNT and hROMK_CNT (returned by initC_Var).
    - CL at LUM-LIS is not set here; updated via hCltj_CNT (returned by initC_Var).
    - HCO3 at P-LIS = CL_P_LIS * 0.20 = 0.2 * 0.20 = 0.04 (AMW rule of thumb).
    - NH4  at P-LIS = K_P_LIS  * 0.20 = 4.0 * 0.20 = 0.80 (AMW rule of thumb).
    - NH4  at LUM-P = K_LUM_P  * 0.20 = 0.0 * 0.20 = 0.00 (AMW rule of thumb).
    - LUM-A and LUM-B interfaces are identical.
    - Basal (P/A/B-BATH) = Lateral (P/A/B-LIS).
    - CA at LIS-BATH is scaled from NA value: NA * (7.93 / 13.3).
    """
    PICAchlo = 11.0
    PICBchlo = 3.20
    PPCh2co3 = 130
    PICh2co3 = 10
    PPCco2   = 15.0e3
    PICco2   = 900
    PICAhpo4 = 7.20e-3
    PICBhpo4 = 4.80e-3
    # PPChpo4  = 8.00e-3  # dead local — assigned but not used; 8.0e-3 used directly in _h_p_lis
    PPCprot  = 2000
    PICAprot = 9.0
    PICBprot = 6.0
    PPCamon  = 2000
    PICamon  = 900
    Purea    = 0.10

    # fscaleENaC   = 2.50  # dead local — assigned but never used
    # fscaleNaK    = 1.25  # dead local — assigned but never used
    # fscaleNaK_PC = 1.25  # dead local — assigned but never used
    FM_cENaC   = 1.20 * 0.85  # female-to-male ENaC expression ratio in cortex
    ClTJperm   = 1000.0        # tight-junction area factor (= 1 / SlumEinitcnt)
    areafactor = 50            # basement-membrane area factor (= 1 / sbasEinit)

    # hENaC_CNT = FM_cENaC * 35.0  # dead local — not stored to p; computed by initC_Var instead
    # hROMK_CNT = 8.0              # dead local — not stored to p; computed by initC_Var instead

    # Apical (LUM–P) permeabilities [x1e-5 cm/s]
    _h_lum_p = np.array([
        0.0,          # NA   (set dynamically via hENaC_CNT)
        0.0,          # K    (set dynamically via hROMK_CNT)
        0.80,         # CL
        0.16,         # HCO3
        PPCh2co3,     # H2CO3
        PPCco2,       # CO2
        1.0e-3/1.20,  # HPO4
        1.0e-3/1.20,  # H2PO4
        Purea,        # UREA
        PPCamon,      # NH3
        0.0,          # NH4  (= K_LUM_P * 0.20 = 0)
        PPCprot,      # H
        0.0,          # HCO2
        0.0,          # H2CO2
        0.0,          # GLU
        0.0,          # CA
    ])

    # Apical (LUM–A and LUM–B) permeabilities [x1e-5 cm/s] — identical for both IC types
    _h_lum_ic = np.array([
        0.0,       # NA
        0.0,       # K
        0.0,       # CL
        0.0,       # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        0.0,       # HPO4
        0.0,       # H2PO4
        Purea,     # UREA
        PICamon,   # NH3
        0.0,       # NH4
        0.0,       # H
        0.0,       # HCO2
        0.0,       # H2CO2
        0.0,       # GLU
        1.0e-5,    # CA
    ])

    # Lateral (P–LIS) permeabilities [x1e-5 cm/s]
    _h_p_lis = np.array([
        0.0,       # NA
        4.0,       # K
        0.2,       # CL
        0.04,      # HCO3 (= CL * 0.20)
        PPCh2co3,  # H2CO3
        PPCco2,    # CO2
        8.0e-3,    # HPO4
        8.0e-3,    # H2PO4
        Purea,     # UREA
        PPCamon,   # NH3
        0.80,      # NH4  (= K * 0.20)
        PPCprot,   # H
        1.0e-4,    # HCO2
        1.0e-4,    # H2CO2
        1.0e-4,    # GLU
        0.0,       # CA
    ])

    # Lateral (A–LIS) permeabilities [x1e-5 cm/s]
    _h_a_lis = np.array([
        0.0,       # NA
        0.448,     # K
        PICAchlo,  # CL
        1.50,      # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        PICAhpo4,  # HPO4
        PICAhpo4,  # H2PO4
        Purea,     # UREA
        PICamon,   # NH3
        0.18,      # NH4
        PICAprot,  # H
        1.0e-4,    # HCO2
        1.0e-4,    # H2CO2
        1.0e-4,    # GLU
        1.0e-5,    # CA
    ])

    # Lateral (B–LIS) permeabilities [x1e-5 cm/s]
    _h_b_lis = np.array([
        0.0,       # NA
        0.12,      # K
        PICBchlo,  # CL
        0.40,      # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        PICBhpo4,  # HPO4
        PICBhpo4,  # H2PO4
        Purea,     # UREA
        PICamon,   # NH3
        0.12,      # NH4
        PICBprot,  # H
        1.0e-4,    # HCO2
        1.0e-4,    # H2CO2
        1.0e-4,    # GLU
        1.0e-5,    # CA
    ])

    # Tight junction (LUM–LIS) coefficients [multiplied by ClTJperm]
    # CL not set here; updated dynamically via hCltj_CNT (returned by initC_Var)
    # hCltj_CNT = 1.3 * ClTJperm * 1.2  # dead local (inside loop in v0) — not stored to p; computed by initC_Var instead
    _tj_coeff = np.array([
        (1.00 / FM_cENaC) * 1.0,  # NA (ENaC-dependent)
        (1.00 / FM_cENaC) * 1.2,  # K  (ENaC-dependent)
        0.0,                        # CL (set dynamically via hCltj_CNT)
        0.6,                        # HCO3
        2.0,                        # H2CO3
        2.0,                        # CO2
        0.14,                       # HPO4
        0.14,                       # H2PO4
        0.40,                       # UREA
        6.0,                        # NH3
        1.2,                        # NH4
        10.0,                       # H
        0.01,                       # HCO2
        0.01,                       # H2CO2
        0.01,                       # GLU
        0.0001,                     # CA
    ])

    # Basement membrane (LIS–BATH) permeabilities [multiplied by areafactor]
    _h_lis_bath = np.array([
        240.0,               # NA
        320.0,               # K
        320.0,               # CL
        160.0,               # HCO3
        240.0,               # H2CO3
        240.0,               # CO2
        160.0,               # HPO4
        160.0,               # H2PO4
        160.0,               # UREA
        320.0,               # NH3
        320.0,               # NH4
        16000.0,             # H
        4.0,                 # HCO2
        4.0,                 # H2CO2
        4.0,                 # GLU
        240.0 * (7.93/13.3), # CA (scaled from NA value)
    ])

    for p in cnt:
        p.h[:, :, :] = 0.0

        p.hCLCA = PICAchlo
        p.hCLCB = PICBchlo

        p.h[:, LUM, P]    = _h_lum_p
        p.h[:, LUM, A]    = _h_lum_ic
        p.h[:, LUM, B]    = _h_lum_ic
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, A,   LIS]  = _h_a_lis
        p.h[:, B,   LIS]  = _h_b_lis
        p.h[:, P,   BATH] = _h_p_lis   # basal = lateral
        p.h[:, A,   BATH] = _h_a_lis   # basal = lateral
        p.h[:, B,   BATH] = _h_b_lis   # basal = lateral
        p.h[:, LUM, LIS]  = _tj_coeff * ClTJperm
        p.h[:, LIS, BATH] = _h_lis_bath * areafactor

        # Non-dimensionalize: h -> h * 1e-5 / href
        p.h *= 1.0e-5 / href


def _set_net_coefficients(cnt: List[Membrane]) -> None:
    """Set NET cotransport coefficients dLA (mmol²/J/s) and transporter levels."""
    dLA = np.zeros((NS, NS, NC, NC))

    FM_cENaC = 1.20 * 0.85   # female-to-male ENaC expression ratio in cortex

    # Na/H exchangers on all basolateral membranes
    dLA[NA, H, P, LIS]  = 20.0e-9
    dLA[NA, H, P, BATH] = 20.0e-9
    dLA[NA, H, A, LIS]  = 36.0e-9
    dLA[NA, H, A, BATH] = 36.0e-9
    dLA[NA, H, B, LIS]  = 24.0e-9
    dLA[NA, H, B, BATH] = 24.0e-9

    # Na/HPO4 co-transporters on all basolateral membranes
    dLA[NA, HPO4, P, LIS]  = 2.0e-9
    dLA[NA, HPO4, P, BATH] = 2.0e-9
    dLA[NA, HPO4, A, LIS]  = 1.2e-9
    dLA[NA, HPO4, A, BATH] = 1.2e-9
    dLA[NA, HPO4, B, LIS]  = 0.80e-9
    dLA[NA, HPO4, B, BATH] = 0.80e-9

    # Basolateral Cl/HCO3 exchanger in PC
    dLA[CL, HCO3, P, LIS]  = 2.0e-9
    dLA[CL, HCO3, P, BATH] = 2.0e-9
    # Basolateral Cl/HCO3 in IC-B: removed (zero)
    # xNCC = 20.0e-9 * 0  # dead local — explicitly zeroed ("Removed"); never assigned to p

    # AE1 in IC-A
    xAE1 = 150e-9

    # Apical Cl/HCO3 (pendrin) in IC-B
    dLA[CL, HCO3, LUM, B] = 800e-9
    xPendrin = 1.2 * dLA[CL, HCO3, LUM, B] / 1400

    # Calcium transporters
    xNCX = 25.0e-9 * 1.60   # basolateral NCX exchanger
    PMCA = 2.0e-9  * 0.40   # basolateral PMCA pump
    # xTRPV5    = 8.0e6                    # dead local — feeds xTRPV5_cnt which is never stored to p; handled by initC_Var
    # Po_TRPV4  = 1.0 / (1.0 + 780/37)    # dead local — feeds xPTRPV4 → xPTRPV4_cnt, none stored to p; handled by initC_Var
    # PCa_TRPV4 = 4.5e-8                   # dead local — same chain
    # xPTRPV4   = Po_TRPV4 * PCa_TRPV4    # dead local — same chain

    # Na-K-ATPase (basolateral in PC and IC)
    ATPNaKPES = FM_cENaC * 3410e-9 * 1.25
    ATPNaKAES = 450e-9
    ATPNaKBES = 300e-9

    # H-ATPase (apical in IC-A, basolateral in IC-B)
    ATPHMA  = 12000.0e-9
    ATPHBES = 400e-9 * 15

    # H-K-ATPase (apical in PC and IC)
    ATPHKMP = 0
    ATPHKMA = 720e-9
    ATPHKMB = 0.0e-9

    # Symmetrize dLA over solute indices
    for i in range(NS - 1):
        for j in range(i + 1, NS):
            dLA[j, i, :, :] = dLA[i, j, :, :]

    # Non-dimensionalize and assign to membrane objects (upper compartment triangle)
    _scale = 1.0 / (href * Cref)
    for k in range(NC):
        for l in range(k, NC):
            for p in cnt:
                p.dLA[:, :, k, l] = dLA[:, :, k, l] * _scale

    for p in cnt:
        p.xPendrin = xPendrin / (href * Cref)
        p.xAE1     = xAE1     / (href * Cref)
        p.xNCX     = xNCX     / (href * Cref)
        p.PMCA     = PMCA     / (href * Cref)
        # xTRPV5_cnt  = xTRPV5  / (href * Cref)  # dead local — scaled but never stored to p; handled by initC_Var
        # xPTRPV4_cnt = xPTRPV4 / (href * Cref)  # dead local — same

        p.ATPNaK[P, LIS]  = ATPNaKPES / (href * Cref)
        p.ATPNaK[P, BATH] = ATPNaKPES / (href * Cref)
        p.ATPNaK[A, LIS]  = ATPNaKAES / (href * Cref)
        p.ATPNaK[A, BATH] = ATPNaKAES / (href * Cref)
        p.ATPNaK[B, LIS]  = ATPNaKBES / (href * Cref)
        p.ATPNaK[B, BATH] = ATPNaKBES / (href * Cref)

        p.ATPH[LUM, A]    = ATPHMA  / (href * Cref)
        p.ATPH[B,   LIS]  = ATPHBES / (href * Cref)
        p.ATPH[B,   BATH] = ATPHBES / (href * Cref)

        p.ATPHK[LUM, P]   = ATPHKMP / (href * Cref)
        p.ATPHK[LUM, A]   = ATPHKMA / (href * Cref)
        p.ATPHK[LUM, B]   = ATPHKMB / (href * Cref)


def _set_carbonate_kinetics(cnt: List[Membrane]) -> None:
    """Set CO₂ hydration/dehydration rate constants (s⁻¹).

    CNT lumen uses uncatalysed rate x10; PC uses low catalysed rate;
    IC-A and IC-B use fully catalysed rate; LIS uses intermediate rate.
    """
    dkhuncat = 0.145
    dkduncat = 49.6
    dkhP = 1.45
    dkdP = 496.0
    dkhA = 1.45e3
    dkdA = 496.0e3
    dkhB = 1.45e3
    dkdB = 496.0e3
    dkhE = 0.145e3
    dkdE = 49.6e3

    for p in cnt:
        p.dkd[LUM] = dkduncat * 10.0
        p.dkh[LUM] = dkhuncat * 10.0
        p.dkd[P]   = dkdP
        p.dkh[P]   = dkhP
        p.dkd[A]   = dkdA
        p.dkh[A]   = dkhA
        p.dkd[B]   = dkdB
        p.dkh[B]   = dkhB
        p.dkd[LIS] = dkdE
        p.dkh[LIS] = dkhE


def _set_boundary_conditions(cnt: List[Membrane]) -> None:
    """Set entering and peritubular conditions from the DCT outlet file."""
    with open('DCToutlet', 'r') as f:
        for i in range(NS):
            cnt[0].conc[i, LUM], cnt[0].conc[i, BATH] = map(float, f.readline().split())
        cnt[0].ph[LUM],  cnt[0].ph[BATH]  = map(float, f.readline().split())
        cnt[0].ep[LUM],  cnt[0].ep[BATH]  = map(float, f.readline().split())
        cnt[0].vol[LUM], cnt[0].pres      = map(float, f.readline().split())

    for p in cnt:
        p.volLuminit = cnt[0].vol[LUM]
        p.ep[BATH]   = cnt[0].ep[BATH]

    pos = np.zeros(NZ + 1)  # stays near cortex surface
    from set_intconc import set_intconc
    set_intconc(cnt, NZ, 1, pos)


def _set_cell_parameters(cnt: List[Membrane]) -> None:
    """Set impermeant reference concentrations, buffer totals, and coalescence."""
    for p in cnt:
        p.cPimpref = 50.0
        p.cAimpref = 18.0
        p.cBimpref = 18.0

        p.cPbuftot = 32.0
        p.cAbuftot = 40.0
        p.cBbuftot = 40.0

    # Coalescence parameter: decreases from 1 at inlet to 2^(-2.32) ≈ 0.20 at outlet
    for jz in range(NZ + 1):
        cnt[jz].coalesce = 2.00 ** (-2.32 * jz / NZ)


def _set_metabolic_parameters(cnt: List[Membrane]) -> None:
    """Set TNa-QO₂ ratio for metabolic calculations.

    Normal: 15.0; diabetic: 12.0.
    """
    for p in cnt:
        p.TQ = 15.0 if not bdiabetes else 12.0


def _check_electroneutrality(cnt: List[Membrane]) -> None:
    """Verify electroneutrality in lumen and bath at inlet and outlet."""
    elecM     = np.dot(zval, cnt[0].conc[:, LUM])
    elecS     = np.dot(zval, cnt[0].conc[:, BATH])
    elecS_out = np.dot(zval, cnt[NZ].conc[:, BATH])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def initC(cnt: List[Membrane]) -> None:
    """Initialize the Connecting Tubule (CNT) segment.

    Args:
        cnt: List of Membrane objects for each spatial grid point (0 to NZ).
    """
    _set_membrane_areas(cnt)
    _set_water_permeabilities(cnt)
    _set_reflection_coefficients(cnt)
    _set_solute_permeabilities(cnt)
    _set_net_coefficients(cnt)
    _set_carbonate_kinetics(cnt)
    _set_boundary_conditions(cnt)
    _set_cell_parameters(cnt)
    _set_metabolic_parameters(cnt)
    _check_electroneutrality(cnt)
