"""Initialization routine for the Distal Convoluted Tubule (DCT).

Sets up membrane geometry, water/solute permeabilities, NET cotransport
coefficients, reflection coefficients, transporter expression levels,
carbonate kinetics, and reads the cTAL outlet as the inlet boundary
condition.

Written originally in Fortran by Prof. Aurelie Edwards.
Translated to Python by Dr. Mohammad M. Tajdini (Boston University,
Dept. of Biomedical Engineering). Refactored for efficiency by Sofia Polychroniadou
"""

import numpy as np

from values import *
from glo import *
from defs import *


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_membrane_areas(dct):
    """Set initial compartment volumes and membrane surface areas.

    Surface area units: µm²/µm tubule length (data from AMW model,
    AJP Renal 2005). Volumes, surfaces, and angles implicitly include
    the number of each cell type.
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
    # VolAinitdct = 1.0  # dead local — never used (p.volAinit set to 7.5 directly)
    # VolBinitdct = 1.0  # dead local — never used (p.volBinit set to 7.5 directly)

    # VolPinitdct = 7.5   # dead local — not referenced; p.volPinit set directly
    # VolEinitdct = 0.80  # dead local — not referenced; p.volEinit set directly
    for p in dct:
        p.sbasEinit = 0.020
        p.volPinit  = 7.5
        p.volEinit  = 0.80
        p.volAinit  = 7.5
        p.volBinit  = 7.5

        p.area[LUM, P]    = SlumPinitdct
        p.area[LUM, A]    = SlumAinitdct
        p.area[LUM, B]    = SlumBinitdct
        p.area[LUM, LIS]  = SlumEinitdct
        p.area[P,   LIS]  = SlatPinitdct
        p.area[A,   LIS]  = SlatAinitdct
        p.area[B,   LIS]  = SlatBinitdct
        p.area[P,   BATH] = SbasPinitdct
        p.area[A,   BATH] = SbasAinitdct
        p.area[B,   BATH] = SbasBinitdct

    # Symmetrize
    for k in range(NC - 1):
        for l in range(k + 1, NC):
            for p in dct:
                p.area[l, k] = p.area[k, l]


def _set_water_permeabilities(dct):
    """Assign water permeabilities (cm/s) and compute non-dimensional dLPV.

    dLPV = Pf / Pfref  (non-dimensional hydraulic conductivity).
    """
    PfMP = 2.4 * 0.00117
    PfMA = 0.0
    PfMB = 0.0
    PfME = 2.0
    PfPE = 2.4 * 0.00835
    PfAE = 0.0
    PfBE = 0.0
    PfPS = 2.4 * 0.00835
    PfAS = 0.0
    PfBS = 0.0
    PfES = 35.5

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

    for p in dct:
        p.dLPV[:, :] = Pf / Pfref


def _set_reflection_coefficients(dct):
    """Initialize reflection coefficients to 1; set basement membrane to 0."""
    for p in dct:
        p.sig[:, :, :] = 1.0
        p.sig[:, LIS, BATH] = 0.0
        p.sig[:, BATH, LIS] = 0.0


def _set_solute_permeabilities(dct):
    """Set dimensional solute permeabilities h (x1e-5 cm/s) then non-dimensionalize.

    One array per interface; index order matches fixed solute order in values.py:
    NA K CL HCO3 H2CO3 CO2 HPO4 H2PO4 UREA NH3 NH4 H HCO2 H2CO2 GLU CA
    """
    ClTJperm   = 1000.0   # tight-junction area factor (= 1 / SlumEinitdct)
    areafactor = 1 / 0.02  # basement-membrane area factor (= 1 / sbasEinit)

    # Apical (LUM–P) permeabilities [x1e-5 cm/s]
    _h_lum_p = np.array([
        0.072,   # NA
        0.60,    # K
        0.0,     # CL
        0.0,     # HCO3
        130,     # H2CO3
        1.50e4,  # CO2
        0.0,     # HPO4
        0.0,     # H2PO4
        0.20,    # UREA
        200,     # NH3
        0.12,    # NH4
        0.20,    # H
        0,       # HCO2
        0,       # H2CO2
        0.0,     # GLU
        0.0,     # CA
    ])

    # Lateral (P–LIS) permeabilities [x1e-5 cm/s]
    _h_p_lis = np.array([
        0.000,   # NA
        0.12,    # K
        0.04,    # CL
        0.02,    # HCO3
        130,     # H2CO3
        1.50e4,  # CO2
        0.002,   # HPO4
        0.002,   # H2PO4
        0.2,     # UREA
        200,     # NH3
        0.0234,  # NH4
        0.20,    # H
        1.0e-4,  # HCO2
        1.0e-4,  # H2CO2
        1.0e-4,  # GLU
        0.0,     # CA
    ])

    # Tight-junction (LUM–LIS) coefficients [multiplied by ClTJperm]
    _tj_coeff = np.array([
        0.80,        # NA
        0.80,        # K
        1.3 * 0.50,  # CL
        0.50,        # HCO3
        0.50,        # H2CO3
        0.50,        # CO2
        0.10,        # HPO4
        0.10,        # H2PO4
        0.20,        # UREA
        0.80,        # NH3
        0.80,        # NH4
        0.80,        # H
        0.01,        # HCO2
        0.01,        # H2CO2
        0.01,        # GLU
        0.0001,      # CA  # TO BE ADJUSTED
    ])

    # Basement membrane (LIS–BATH) permeabilities [multiplied by areafactor]
    _h_lis_bath = np.array([
        63.0,               # NA
        84.0,               # K
        84.0,               # CL
        42.0,               # HCO3
        63.0,               # H2CO3
        63.0,               # CO2
        42.0,               # HPO4
        42.0,               # H2PO4
        42.0,               # UREA
        52.0,               # NH3
        84.0,               # NH4
        419.0,              # H
        1.0,                # HCO2
        1.0,                # H2CO2
        1.0,                # GLU
        63.0 * (7.93/13.3), # CA (scaled from NA value)
    ])

    for p in dct:
        p.hNaMP = 14.00 * 0.25  # max ENaC perm, factor 1/4 for surface area

        p.h[:, LUM, P]    = _h_lum_p
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, P,   BATH] = _h_p_lis        # basal same as lateral
        p.h[:, LUM, LIS]  = _tj_coeff * ClTJperm
        p.h[:, LIS, BATH] = _h_lis_bath * areafactor

        # Non-dimensionalize: h(I,K,L) -> h(I,K,L) * 1e-5 / href
        p.h *= 1.0e-5 / href


def _set_net_coefficients(dct):
    """Set NET cotransport coefficients dLA (mmol²/J/s) and transporter levels."""
    dLA = np.zeros((NS, NS, NC, NC))

    # Apical NHE2/3 in PC
    xNHE3 = 200.0e-9

    # Apical NCC in PC
    xNCC = 1.8 * 15.0e-9

    # Apical and basolateral K–Cl in PC
    dLA[K,  CL,  LUM,  P]   = 4.0e-9
    dLA[K,  CL,    P, LIS]  = 20.0e-9
    dLA[K,  CL,    P, BATH] = 20.0e-9

    # Basolateral Na/H exchanger
    dLA[NA,  H,    P, LIS]  = 4.0e-9
    dLA[NA,  H,    P, BATH] = 4.0e-9
    # xNHE1 = 6.95e-10  # dead local — assigned but never used

    # Basolateral Cl/HCO3 exchanger
    dLA[CL, HCO3,  P, LIS]  = 15.0e-9
    dLA[CL, HCO3,  P, BATH] = 15.0e-9

    # Basolateral Na/HPO4 co-transporter
    dLA[NA, HPO4,  P, LIS]  = 0.2e-9
    dLA[NA, HPO4,  P, BATH] = 0.2e-9

    # Na-K-ATPase (basolateral in PC)
    ATPNaKPES = 1.8 * 400.0e-9

    # Basolateral NCX exchanger
    xNCX = 50.0e-9 * 1.40

    # Basolateral PMCA pump
    PMCA = 0.70e-9 * 0.60
    # xTRPV5 = 13.5e6  # dead local — assigned only to compute xTRPV5_dct which was never stored to p

    # Symmetrize dLA over solute indices
    for i in range(NS - 1):
        for j in range(i + 1, NS):
            dLA[j, i, :, :] = dLA[i, j, :, :]

    # Non-dimensionalize and assign to membrane objects (upper compartment triangle)
    _scale = 1.0 / (href * Cref)
    for k in range(NC):
        for l in range(k, NC):
            for p in dct:
                p.dLA[:, :, k, l] = dLA[:, :, k, l] * _scale

    for p in dct:
        p.xNCC  = xNCC  / (href * Cref)
        p.xNHE3 = xNHE3 / (href * Cref)
        p.xNCX  = xNCX  / (href * Cref)
        p.PMCA  = PMCA  / (href * Cref)
        # xTRPV5_dct = xTRPV5 / (href * Cref)  # dead local — scaled but never stored to p
        p.ATPNaK[P, LIS]  = ATPNaKPES / (href * Cref)
        p.ATPNaK[P, BATH] = ATPNaKPES / (href * Cref)


def _set_carbonate_kinetics(dct):
    """Set carbonate hydration/dehydration rate constants (s⁻¹).

    DCT lumen uses uncatalysed rate x10; P-cell and LIS use catalysed rate.
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


def _read_inlet_conditions(dct):
    """Read entering and peritubular conditions from cTAL outlet file."""
    with open('cTALoutlet', 'r') as f:
        for i in range(NS):
            dct[0].conc[i, LUM], dct[0].conc[i, BATH] = map(float, f.readline().split())
        dct[0].ph[LUM],  dct[0].ph[BATH]  = map(float, f.readline().split())
        dct[0].ep[LUM],  dct[0].ep[BATH]  = map(float, f.readline().split())
        dct[0].vol[LUM], dct[0].pres      = map(float, f.readline().split())

    for p in dct:
        p.volLuminit = dct[0].vol[LUM]
        p.ep[BATH]   = dct[0].ep[BATH]

    pos = np.zeros(NZ + 1)  # twirls around the cortical surface
    from set_intconc import set_intconc
    set_intconc(dct, NZ, 1, pos)


def _check_electroneutrality(dct):
    """Verify electroneutrality of lumen and bath at inlet and outlet."""
    elecM     = np.dot(zval, dct[0].conc[:, LUM])
    elecS     = np.dot(zval, dct[0].conc[:, BATH])
    elecS_out = np.dot(zval, dct[NZ].conc[:, BATH])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def initD(dct):
    """Initialize the Distal Convoluted Tubule (DCT) segment.

    Args:
    dct : list of Membrane
        List of NZ+1 Membrane objects representing the DCT grid.
    """
    _set_membrane_areas(dct)
    _set_water_permeabilities(dct)
    _set_reflection_coefficients(dct)
    _set_solute_permeabilities(dct)
    _set_net_coefficients(dct)
    _set_carbonate_kinetics(dct)
    _read_inlet_conditions(dct)

    for p in dct:
        p.TQ = 15.0 if not bdiabetes else 12.0
        # TNa-QO2 ratio: 15 in normal DCT, 12 in diabetic DCT

    _check_electroneutrality(dct)
