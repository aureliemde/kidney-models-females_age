""" Written originally in Fortran by Prof. Aurelie Edwards
 Translated to Python by Dr. Mohammad M. Tajdini
 Refactored by Sofia Polychroniadou

 Department of Biomedical Engineering
 Boston University

 Initialization of IMCD (inner medullary collecting duct) variant parameters.
 Sets up geometry, membrane permeabilities, transporter coefficients,
 and reads inlet conditions from OMCoutlet.
 Called from qflux2IMC.py; returns hENaC_IMC, hROMK_IMC, hCltj_IMC.
 Only principal cells (P) are real; A and B are phantom cells."""

import numpy as np

from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initIMC_Var(imcd):
    """Initialize IMCD variant parameters: geometry, permeabilities, transporters, inlet conditions.

    Args:
    imcd : list of Membrane
        IMCD cell array (length NZIMC+1).

    Returns:
    hENaC_IMC : float
        ENaC permeability (apical, P cell).
    hROMK_IMC : float
        ROMK permeability (apical, P cell).
    hCltj_IMC : float
        Cl tight-junction permeability (LUM-LIS).
    """
    # unused geometry scalars (kept for reference)
    # theta = np.zeros(NC)
    # Slum  = np.zeros(NC)
    # Slat  = np.zeros(NC)
    # Sbas  = np.zeros(NC)

    Pf  = np.zeros((NC, NC))
    dLA = np.zeros((NS, NS, NC, NC))
    pos = np.zeros(NZIMC + 1)   # original had NZ+1 (copy-paste error; loop uses NZIMC)

    # ------------------------------------------------------------------
    # Geometry and volumes  (AMW model, AJP Renal 2001)
    # ------------------------------------------------------------------
    SlumPinitimc = 0.6
    SbasPinitimc = 0.6
    SlatPinitimc = 5.4

    SlumEinitimc = 0.001

    # A and B cells are phantom — mirror P geometry
    SlumAinitimc = SlumPinitimc
    SbasAinitimc = SbasPinitimc
    SlatAinitimc = SlatPinitimc
    SlumBinitimc = SlumPinitimc
    SbasBinitimc = SbasPinitimc
    SlatBinitimc = SlatPinitimc

    for p in imcd:
        p.sbasEinit = 0.020
        p.volPinit  = 8.0
        p.volAinit  = 8.0   # phantom: same as P
        p.volBinit  = 8.0   # phantom: same as P
        p.volEinit  = 0.80

    # ------------------------------------------------------------------
    # Membrane surface areas
    # ------------------------------------------------------------------
    for p in imcd:
        p.area[LUM, P]    = SlumPinitimc
        p.area[LUM, A]    = SlumAinitimc   # phantom: same as P
        p.area[LUM, B]    = SlumBinitimc   # phantom: same as P
        p.area[LUM, LIS]  = SlumEinitimc
        p.area[P,   LIS]  = SlatPinitimc
        p.area[A,   LIS]  = SlatAinitimc   # phantom: same as P
        p.area[B,   LIS]  = SlatBinitimc   # phantom: same as P
        p.area[P,   BATH] = SbasPinitimc
        p.area[A,   BATH] = SbasAinitimc   # phantom: same as P
        p.area[B,   BATH] = SbasBinitimc   # phantom: same as P
        p.area += p.area.T  # symmetric copy

    # ------------------------------------------------------------------
    # Water permeabilities: dLPV(K,L) = Pf(K,L) / Pfref  [cm/s/mmHg]
    # Pf values in cm/s — divided by Vwbar to convert to osmotic units
    # A and B phantom cells: Pf = 0 (from np.zeros)
    # ------------------------------------------------------------------
    PfMP = 0.750e-3 / Vwbar
    PfPE = 0.150e-3 / Vwbar
    PfPS = 0.150e-3 / Vwbar
    PfME = 36.0e-3  / Vwbar
    PfES = 600e-3   / Vwbar

    Pf[LUM, P]    = PfMP
    Pf[LUM, LIS]  = PfME
    Pf[P,   LIS]  = PfPE
    Pf[P,   BATH] = PfPS
    Pf[LIS, BATH] = PfES

    for p in imcd:
        p.dLPV = Pf / Pfref

    # ------------------------------------------------------------------
    # Reflection coefficients: sig = 1 everywhere, 0 at LIS-BATH
    # ------------------------------------------------------------------
    for p in imcd:
        p.sig[:] = 1.0
        p.sig[:, LIS, BATH] = 0.0

    # ------------------------------------------------------------------
    # Solute permeabilities  [10^-5 cm/s initially]
    # ------------------------------------------------------------------
    # h arrays already zero from Membrane init; redundant init loop removed

    # Female-to-male ENaC expression ratio in medulla (IMCD)
    FM_mENaC = 1.20 * 0.85

    # fscaleENaC   = 2.50   # dead local — not stored to p or scaling object
    # fscaleNaK    = 1.25   # dead local
    # fscaleNaK_PC = 1.25   # dead local

    hENaC_IMC = FM_mENaC * 2.50
    hROMK_IMC = 1.0

    # -- LUM-P interface -----------------------------------------------
    # NA: ENaC not set here (cf. hENaC_IMC returned)
    # K: 1.0 set directly (= hROMK_IMC)
    _h_lum_p = np.array([
        0.0,            # NA    — not set (cf. hENaC_IMC)
        1.0,            # K     — = hROMK_IMC
        2.0e-4,         # CL
        2.0e-4,         # HCO3
        130,            # H2CO3
        15.0e3/6.5,     # CO2   (see DN model, AJP 2008)
        0.20e-3,        # HPO4
        0.20e-3,        # H2PO4
        300,            # UREA
        400,            # NH3
        0.20,           # NH4
        2.0e-4,         # H
        0.0,            # HCO2
        0.0,            # H2CO2
        0.0,            # GLU
        0.00010,        # CA
    ])

    # -- P-LIS interface -----------------------------------------------
    _h_p_lis = np.array([
        2.0e-4,         # NA
        1.50,           # K
        0.050,          # CL
        0.025,          # HCO3
        130.0,          # H2CO3
        15.0e3/6.5,     # CO2   (see DN model, AJP 2008)
        8.0e-3,         # HPO4  (as in CNT)
        8.0e-3,         # H2PO4 (as in CNT)
        15.0,           # UREA
        400,            # NH3
        0.30,           # NH4
        2.0e-4,         # H
        1.0e-4,         # HCO2
        1.0e-4,         # H2CO2
        1.0e-4,         # GLU
        0.00010,        # CA
    ])

    # -- LUM-LIS interface (tight junction, ME) ------------------------
    # Assume TJ resistivity of 5 mS/cm2
    ClTJperm = 1000.0
    hCltj_IMC = 1.3 * ClTJperm * 1.60   # Cl permeability via hCltj_IMC; CL slot set to 0 below

    _h_lum_lis = np.array([
        (1.00 / FM_mENaC) * 0.90 * ClTJperm,   # NA
        (1.00 / FM_mENaC) * 0.70 * ClTJperm,   # K
        0.0,                                    # CL   — via hCltj_IMC
        0.4  * ClTJperm,                        # HCO3  (changed from 0.15 on 04/30/19)
        1.2  * ClTJperm,                        # H2CO3
        1.2  * ClTJperm,                        # CO2
        0.2  * ClTJperm,                        # HPO4
        0.2  * ClTJperm,                        # H2PO4
        0.80 * ClTJperm,                        # UREA
        1.0  * ClTJperm,                        # NH3
        0.7  * ClTJperm,                        # NH4
        6.0  * ClTJperm,                        # H
        0.01 * ClTJperm,                        # HCO2
        0.01 * ClTJperm,                        # H2CO2
        0.01 * ClTJperm,                        # GLU
        0.0140 * ClTJperm,                      # CA  (see 2015 AJP and Carney 1988)
    ])

    # -- LIS-BATH interface (paracellular, ES; all values from AWM model) -
    areafactor = 50
    _h_lis_bath = np.array([
        60    * areafactor,               # NA
        80.0  * areafactor,               # K
        80.0  * areafactor,               # CL
        40.0  * areafactor,               # HCO3
        60.0  * areafactor,               # H2CO3
        60.0  * areafactor,               # CO2
        40.0  * areafactor,               # HPO4
        40.0  * areafactor,               # H2PO4
        40.0  * areafactor,               # UREA
        50.0  * areafactor,               # NH3
        80.0  * areafactor,               # NH4
        400.0 * areafactor,               # H
        1.0   * areafactor,               # HCO2
        1.0   * areafactor,               # H2CO2
        1.0   * areafactor,               # GLU
        60    * areafactor * (7.93/13.3), # CA
    ])

    for p in imcd:
        p.h[:, LUM, P]    = _h_lum_p
        # LUM-A, LUM-B, A-LIS, A-BATH, B-LIS, B-BATH: phantom — already zero
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, P,   BATH] = _h_p_lis   # PS = PE
        p.h[:, LUM, LIS]  = _h_lum_lis
        p.h[:, LIS, BATH] = _h_lis_bath
        p.h *= 1.0e-5 / href

    # ------------------------------------------------------------------
    # NET cross-coupling coefficients dLA  [mmol^2/J/s initially]
    # ------------------------------------------------------------------
    # dLA already zero from np.zeros; redundant init loop removed
    # Only P-cell basolateral and LUM-P apical membranes

    # Na/H exchanger on P basolateral
    dLA[NA, H,    P, LIS]  = 6.0e-9
    dLA[NA, H,    P, BATH] = 6.0e-9

    # Na2/HPO4 co-transporter on P basolateral
    dLA[NA, HPO4, P, LIS]  = 2.0e-9
    dLA[NA, HPO4, P, BATH] = 2.0e-9

    # NaKCl2 cotransporter on P basolateral (Na-K coupling term)
    dLA[NA, K,    P, LIS]  = 4.0e-9
    dLA[NA, K,    P, BATH] = 4.0e-9

    # KCl cotransporter on P basolateral
    dLA[K,  CL,   P, LIS]  = 250.0e-9
    dLA[K,  CL,   P, BATH] = 250.0e-9

    # Cl/HCO3 exchanger on P basolateral
    dLA[CL, HCO3, P, LIS]  = 20.0e-9
    dLA[CL, HCO3, P, BATH] = 20.0e-9

    # Na/Cl exchanger on apical membrane (LUM-P)
    dLA[NA, CL, LUM, P] = 1200.0e-9

    # Symmetrize dLA: dLA[J,I,:,:] = dLA[I,J,:,:]
    dLA = dLA + dLA.transpose(1, 0, 2, 3)

    # ------------------------------------------------------------------
    # ATPase coefficients  [mmol/s]
    # Only P-cell transporters (A and B are phantom)
    # ------------------------------------------------------------------
    # Na-K-ATPase (P basolateral)
    ATPNaKPES = FM_mENaC * 2000e-9 * 1.25

    # H-K-ATPase (P apical) — adjusted parameter
    ATPHKMP = 2000e-9 * 0.10 * 0.50

    # ------------------------------------------------------------------
    # Non-dimensionalize and assign transporters
    # ------------------------------------------------------------------
    _hc = href * Cref
    for p in imcd:
        p.dLA = dLA / _hc

        p.ATPNaK[P,   LIS]  = ATPNaKPES / _hc
        p.ATPNaK[P,   BATH] = ATPNaKPES / _hc

        p.ATPHK[LUM, P] = ATPHKMP / _hc

    # ------------------------------------------------------------------
    # Kinetic parameters for carbonate reactions  [s^-1]
    # A and B phantom cells use same rates as P.
    # LIS and LUM use uncatalysed rates.
    # ------------------------------------------------------------------
    dkhuncat = 0.145
    dkduncat = 49.6
    dkhP = dkhuncat * 100
    dkdP = dkduncat * 100
    dkhE = dkhuncat
    dkdE = dkduncat

    for p in imcd:
        p.dkd[LUM] = dkduncat
        p.dkh[LUM] = dkhuncat
        p.dkd[P]   = dkdP
        p.dkh[P]   = dkhP
        p.dkd[A]   = dkdP      # phantom: same as P
        p.dkh[A]   = dkhP
        p.dkd[B]   = dkdP      # phantom: same as P
        p.dkh[B]   = dkhP
        p.dkd[LIS] = dkdE
        p.dkh[LIS] = dkhE

    # ------------------------------------------------------------------
    # Boundary conditions: peritubular potential, coalescence, axial position
    # coalesce is axial: 64 OMCDs converging to a single duct
    # (consistent with Hervy and Thomas, 2003)
    # ------------------------------------------------------------------
    imcd[0].ep[BATH] = -0.001e-3 / EPref

    for jz in range(NZIMC + 1):
        imcd[jz].ep[BATH]  = imcd[0].ep[BATH]
        imcd[jz].coalesce  = 0.20 * (2.00 ** (-6.00 * jz / NZIMC))
        pos[jz] = 1.0 * jz / NZIMC

    set_intconc(imcd, NZIMC, 3, pos)

    # ------------------------------------------------------------------
    # Read inlet conditions from OMCoutlet
    # ------------------------------------------------------------------
    with open('OMCoutlet', 'r') as f:
        for I in range(1, NS + 1):
            imcd[0].conc[I-1, LUM], imcd[0].conc[I-1, BATH] = map(float, f.readline().split())
        imcd[0].ph[LUM],  imcd[0].ph[BATH]  = map(float, f.readline().split())
        imcd[0].ep[LUM],  imcd[0].ep[BATH]  = map(float, f.readline().split())
        imcd[0].vol[LUM], imcd[0].pres      = map(float, f.readline().split())

    # ------------------------------------------------------------------
    # Final per-cell state and metabolic ratio
    # ------------------------------------------------------------------
    for p in imcd:
        p.volLuminit = imcd[0].vol[LUM]
        # TNa-QO2 ratio: 15.0 in normal IMCD, 12.0 in diabetic IMCD
        p.TQ = 15.0 if not bdiabetes else 12.0

    # electroneutrality check (informational only; values not used)
    # elecM     = sum(zval[I] * imcd[0].conc[I, LUM]   for I in range(NS))
    # elecS     = sum(zval[I] * imcd[0].conc[I, BATH]  for I in range(NS))
    # elecS_out = sum(zval[I] * imcd[NZIMC].conc[I, BATH] for I in range(NS))

    return hENaC_IMC, hROMK_IMC, hCltj_IMC
