# Written originally in Fortran by Prof. Aurelie Edwards
# Translated to Python by Dr. Mohammad M. Tajdini
# Refactored by Sofia Polychroniadou
#
# Department of Biomedical Engineering
# Boston University
#
# Initialization of CCD (cortical collecting duct) parameters.
# Sets up geometry, membrane permeabilities, transporter coefficients,
# and reads inlet conditions from CNToutlet.

import numpy as np

from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initCCD_Var(ccd):
    """Initialize CCD parameters: geometry, permeabilities, transporters, inlet conditions.

    Args:
    ccd : list of Membrane
        CCD cell array (length NZ+1).

    Returns:
    hENaC_CCD : float
        Basal ENaC expression level for CCD.
    hROMK_CCD : float
        Basal ROMK expression level for CCD.
    hCltj_CCD : float
        Basal Cl tight-junction permeability for CCD.
    """
    # unused geometry scalars (kept for reference)
    # theta = np.zeros(NC)
    # Slum  = np.zeros(NC)
    # Slat  = np.zeros(NC)
    # Sbas  = np.zeros(NC)

    Pf  = np.zeros((NC, NC))
    dLA = np.zeros((NS, NS, NC, NC))
    pos = np.zeros(NZ + 1)

    # ------------------------------------------------------------------
    # Geometry and volumes
    # ------------------------------------------------------------------
    SlumPinitccd = 1.2
    SbasPinitccd = 1.2
    SlatPinitccd = 6.9

    SlumAinitccd = 0.58
    SbasAinitccd = 0.58
    SlatAinitccd = 1.25

    SlumBinitccd = 0.17
    SbasBinitccd = 0.17
    SlatBinitccd = 1.50

    SlumEinitccd = 0.001

    for p in ccd:
        p.dimL      = dimLCCD
        p.diam      = DiamCCD
        p.sbasEinit = 0.020
        p.volPinit  = 4.0
        p.volAinit  = 1.8
        p.volBinit  = 1.2
        p.volEinit  = 0.20
        p.zPimp     = -1.0
        p.zAimp     = -1.0
        p.zBimp     = -1.0

    # ------------------------------------------------------------------
    # Membrane surface areas
    # ------------------------------------------------------------------
    for p in ccd:
        p.area[LUM, P]    = SlumPinitccd
        p.area[LUM, A]    = SlumAinitccd
        p.area[LUM, B]    = SlumBinitccd
        p.area[LUM, LIS]  = SlumEinitccd
        p.area[P,   LIS]  = SlatPinitccd
        p.area[A,   LIS]  = SlatAinitccd
        p.area[B,   LIS]  = SlatBinitccd
        p.area[P,   BATH] = SbasPinitccd
        p.area[A,   BATH] = SbasAinitccd
        p.area[B,   BATH] = SbasBinitccd
        p.area += p.area.T  # symmetric copy

    # ------------------------------------------------------------------
    # Water permeabilities: dLPV(K,L) = Pf(K,L) / Pfref  [cm/s/mmHg]
    # ------------------------------------------------------------------
    PfMP = 2.4 * 0.20/1.2
    PfPE = 2.4 * 0.11*3   # factor of 3 from DN model
    PfPS = 2.4 * 0.11*3   # factor of 3 from DN model

    PfMA = 0.22e-3
    PfAE = 5.50e-3
    PfAS = 5.50e-3

    PfMB = 0.22e-3
    PfBE = 5.50e-3
    PfBS = 5.50e-3

    PfME = 1.1
    PfES = 110

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

    for p in ccd:
        p.dLPV = Pf / Pfref

    # ------------------------------------------------------------------
    # Reflection coefficients: sig = 1 everywhere, 0 at LIS-BATH
    # ------------------------------------------------------------------
    for p in ccd:
        p.sig[:] = 1.0
        p.sig[:, LIS, BATH] = 0.0

    # ------------------------------------------------------------------
    # Solute permeabilities  [10^-5 cm/s initially]
    # ------------------------------------------------------------------
    PICAchlo  = 10.0
    PICBchlo  = 3.20
    PPCh2co3  = 130
    PICh2co3  = 10
    PPCco2    = 15.0e3
    PICco2    = 900
    # PPChpo4 = 8.00e-3    # unused
    # PICAhpo4 = 4.80e-3   # unused
    # PICBhpo4 = 4.80e-3   # unused
    PPCprot   = 2000
    # PICAprot = 9.0       # unused
    PICBprot  = 6.0
    PPCamon   = 2000
    PICamon   = 900
    Purea     = 0.10

    # h arrays already zero from Membrane init; redundant init loop removed

    fac4 = 1.0/4.0
    # fac6 = 1.0/1.0   # unused
    # fac9 = 1.0/1.0   # unused

    # Female-to-male ENaC expression ratio in cortex (CNT and CCD)
    FM_cENaC = 1.20 * 0.85

    # ENaC activity modified by local factors, including pH.
    # Use hENaC_CCD as basal value of expression.
    ### FEMALE ADJUSTMENT: ENaC x 1.20
    hENaC_CCD = FM_cENaC * 35.00 * fac4

    # ROMK activity modified by local factors, including pH.
    # Use hROMK_CCD as basal value of expression.
    # ccd(:)%h(2,1,2) = 8.0*fac4 — not used, replaced with hROMK
    hROMK_CCD = 8.0 * fac4

    # fscaleENaC   = 2.50   # dead local — not stored to p or scaling object
    # fscaleNaK    = 1.25   # dead local
    # fscaleNaK_PC = 1.25   # dead local

    # -- LUM-P interface (tight junction PC) ---------------------------
    # NA: ENaC via hENaC_CCD (not set here)
    # K:  ROMK via hROMK_CCD (not set here)
    # NH4 = h[K, LUM, P] * 0.20 = 0.0 (AMW rule of thumb)
    _h_lum_p = np.array([
        0.0,             # NA    — ENaC handled via hENaC_CCD
        0.0,             # K     — ROMK handled via hROMK_CCD
        0.80 * fac4,     # CL
        0.16 * fac4,     # HCO3
        PPCh2co3,        # H2CO3
        PPCco2,          # CO2
        1.0e-3/1.20,     # HPO4  (as in CNT; different in AMW model)
        1.0e-3/1.20,     # H2PO4 (as in CNT; different in AMW model)
        Purea,           # UREA
        PPCamon,         # NH3
        0.0,             # NH4   — AMW rule: 0.20 * h[K,LUM,P] = 0
        PPCprot * fac4,  # H
        0.0,             # HCO2
        0.0,             # H2CO2
        0.0,             # GLU
        0.00010,         # CA
    ])

    # -- P-LIS interface (PC lateral) ----------------------------------
    # HCO3 = CL * 0.20 (AMW rule); NH4 = K * 0.20 (AMW rule)
    _h_p_lis = np.array([
        0.000,              # NA
        4 * fac4,           # K
        0.2 * fac4,         # CL
        0.2 * fac4 * 0.20,  # HCO3  — AMW rule of thumb
        PPCh2co3,           # H2CO3
        PPCco2,             # CO2
        8.0e-3,             # HPO4  (as in CNT)
        8.0e-3,             # H2PO4 (as in CNT)
        Purea,              # UREA
        PPCamon,            # NH3
        4 * fac4 * 0.20,    # NH4   — AMW rule of thumb
        PPCprot * fac4,     # H
        1.0e-4,             # HCO2
        1.0e-4,             # H2CO2
        1.0e-4,             # GLU
        0.00010,            # CA
    ])

    # -- LUM-A and LUM-B interfaces (identical IC lumen) ---------------
    _h_lum_ic = np.array([
        0.0,       # NA
        0.00,      # K
        0.0,       # CL
        0.0,       # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        0.0,       # HPO4
        0.0,       # H2PO4
        Purea,     # UREA
        PICamon,   # NH3
        0.00,      # NH4
        0.0,       # H
        0.0,       # HCO2
        0.0,       # H2CO2
        0.0,       # GLU
        0.00010,   # CA  (TO BE ADJUSTED)
    ])

    # -- A-LIS interface (IC-A lateral) --------------------------------
    _h_a_lis = np.array([
        0.0,       # NA
        0.050,     # K
        1.20,      # CL
        0.18,      # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        0.0120,    # HPO4  (different in AMW model)
        0.0120,    # H2PO4 (different in AMW model)
        Purea,     # UREA
        PICamon,   # NH3
        0.030,     # NH4
        1.50,      # H
        1.0e-4,    # HCO2
        1.0e-4,    # H2CO2
        1.0e-4,    # GLU
        0.00010,   # CA  (TO BE ADJUSTED)
    ])

    # -- B-LIS interface (IC-B lateral) --------------------------------
    _h_b_lis = np.array([
        0.0,              # NA
        0.12 * fac4,      # K
        PICBchlo * fac4,  # CL
        0.40 * fac4,      # HCO3
        PICh2co3,         # H2CO3
        PICco2,           # CO2
        0.120,            # HPO4  (different in AMW model)
        0.120,            # H2PO4 (different in AMW model)
        Purea,            # UREA
        PICamon,          # NH3
        0.12 * fac4,      # NH4
        PICBprot * fac4,  # H
        1.0e-4,           # HCO2
        1.0e-4,           # H2CO2
        1.0e-4,           # GLU
        0.00010,          # CA
    ])

    # -- LUM-LIS interface (tight junction, ME) ------------------------
    # Account for factor 2 increase (DN model, 2008)
    ClTJperm = 1000.0
    hCltj_CCD = 1.3 * ClTJperm * 1.2
    # ccd(:)%h(CL,LUM,LIS) = ClTJperm*1.2 — not used; replaced with hCltj_CCD

    _h_lum_lis = np.array([
        (1.00 / FM_cENaC) * 1.0  * ClTJperm,   # NA
        (1.00 / FM_cENaC) * 1.2  * ClTJperm,   # K
        0.0,                                    # CL   — via hCltj_CCD
        0.3  * ClTJperm,                        # HCO3  (changed from 0.15 on 04/30/19)
        2.0  * ClTJperm,                        # H2CO3
        2.0  * ClTJperm,                        # CO2
        0.14 * ClTJperm,                        # HPO4
        0.14 * ClTJperm,                        # H2PO4
        0.40 * ClTJperm,                        # UREA
        6.0  * ClTJperm,                        # NH3
        1.2  * ClTJperm,                        # NH4
        10.0 * ClTJperm,                        # H
        0.01 * ClTJperm,                        # HCO2
        0.01 * ClTJperm,                        # H2CO2
        0.01 * ClTJperm,                        # GLU
        0.0140 * ClTJperm,                      # CA  (see 2015 AJP and Carney 1988)
    ])

    # -- LIS-BATH interface (paracellular, ES) -------------------------
    areafactor = 100
    _h_lis_bath = np.array([
        97    * areafactor,                # NA
        130.0 * areafactor,               # K
        130.0 * areafactor,               # CL
        65.0  * areafactor,               # HCO3
        97.0  * areafactor,               # H2CO3
        97.0  * areafactor,               # CO2
        65.0  * areafactor,               # HPO4
        65.0  * areafactor,               # H2PO4
        65.0  * areafactor,               # UREA
        130.0 * areafactor,               # NH3
        130.0 * areafactor,               # NH4
        6490.0 * areafactor,              # H
        4.0   * areafactor,               # HCO2
        4.0   * areafactor,               # H2CO2
        4.0   * areafactor,               # GLU
        97    * areafactor * (7.93/13.3), # CA
    ])

    for p in ccd:
        p.h[:, LUM, P]    = _h_lum_p
        p.h[:, LUM, A]    = _h_lum_ic     # IC-A lumen
        p.h[:, LUM, B]    = _h_lum_ic     # IC-B lumen (identical to A)
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, A,   LIS]  = _h_a_lis
        p.h[:, B,   LIS]  = _h_b_lis
        p.h[:, LUM, LIS]  = _h_lum_lis
        p.h[:, LIS, BATH] = _h_lis_bath
        # basal membranes (PS, AS, BS) copy from LIS-side arrays
        p.h[:, P,   BATH] = _h_p_lis
        p.h[:, A,   BATH] = _h_a_lis
        p.h[:, B,   BATH] = _h_b_lis
        p.hCLCA = PICAchlo
        p.hCLCB = PICBchlo
        p.h *= 1.0e-5 / href

    # ------------------------------------------------------------------
    # NET cross-coupling coefficients dLA  [mmol^2/J/s initially]
    # ------------------------------------------------------------------
    # dLA already zero from np.zeros; redundant init loop removed

    # Na/H exchangers on all basolateral membranes
    dLA[NA, H, P,    LIS]  = 20.0e-9 * fac4
    dLA[NA, H, P,    BATH] = 20.0e-9 * fac4
    dLA[NA, H, A,    LIS]  = 24.0e-9 * fac4
    dLA[NA, H, A,    BATH] = 24.0e-9 * fac4
    dLA[NA, H, B,    LIS]  = 24.0e-9 * fac4
    dLA[NA, H, B,    BATH] = 24.0e-9 * fac4

    # Na2/HPO4 co-transporters on all basolateral membranes
    dLA[NA, HPO4, P,    LIS]  = 2.0e-9
    dLA[NA, HPO4, P,    BATH] = 2.0e-9
    dLA[NA, HPO4, A,    LIS]  = 0.80e-9
    dLA[NA, HPO4, A,    BATH] = 0.80e-9
    dLA[NA, HPO4, B,    LIS]  = 0.80e-9
    dLA[NA, HPO4, B,    BATH] = 0.80e-9

    # Basolateral Cl/HCO3 exchanger in PC
    dLA[CL, HCO3, P,   LIS]  = 2.0e-9 * fac4
    dLA[CL, HCO3, P,   BATH] = 2.0e-9 * fac4

    # Basolateral Cl/HCO3 exchanger in IC-B — REMOVED (multiplied by 0)
    dLA[CL, HCO3, B,   LIS]  = 16.0e-9 * fac4 * 0
    dLA[CL, HCO3, B,   BATH] = 16.0e-9 * fac4 * 0

    # AE1 in IC-A only (density of AE1 times 0.20 in DN model, AMW 2008)
    xAE1 = 15e-9 * 0.20

    # Pendrin and NDBCE in IC-B only
    dLA[CL, HCO3, LUM, B] = 800e-9 * fac4
    xPendrin = 1.2 * dLA[CL, HCO3, LUM, B] / 1700
    xNDBCE   = 0.00  # NDCBE (disabled)

    # Symmetrize dLA: dLA[J,I,:,:] = dLA[I,J,:,:]
    dLA = dLA + dLA.transpose(1, 0, 2, 3)

    # ------------------------------------------------------------------
    # ATPase coefficients  [mmol/s]
    # ------------------------------------------------------------------
    # Na-K-ATPase (basolateral in PC and IC)
    ATPNaKPES = FM_cENaC * 3410e-9 * 1.25 * fac4
    ATPNaKAES = 300e-9 * fac4
    ATPNaKBES = 300e-9 * fac4

    # H-ATPase (apical in IC-A, basolateral in IC-B)
    ATPHMA  = 1000.0e-9
    ATPHBES = 400e-9 * 15 * fac4   # adjusted parameter

    # H-K-ATPase (apical in PC and IC)
    ATPHKMP = 0
    ATPHKMA = 60.0e-9
    ATPHKMB = 0.0e-9

    # ------------------------------------------------------------------
    # Non-dimensionalize and assign transporters
    # ------------------------------------------------------------------
    _hc = href * Cref
    for p in ccd:
        p.dLA = dLA / _hc

        p.xPendrin = xPendrin / _hc
        p.xAE1     = xAE1     / _hc
        p.xNDBCE   = xNDBCE   / _hc

        p.ATPNaK[P,   LIS]  = ATPNaKPES / _hc
        p.ATPNaK[P,   BATH] = ATPNaKPES / _hc
        p.ATPNaK[A,   LIS]  = ATPNaKAES / _hc
        p.ATPNaK[A,   BATH] = ATPNaKAES / _hc
        p.ATPNaK[B,   LIS]  = ATPNaKBES / _hc
        p.ATPNaK[B,   BATH] = ATPNaKBES / _hc

        p.ATPH[LUM, A]    = ATPHMA  / _hc
        p.ATPH[B,   LIS]  = ATPHBES / _hc
        p.ATPH[B,   BATH] = ATPHBES / _hc

        p.ATPHK[LUM, P] = ATPHKMP / _hc
        p.ATPHK[LUM, A] = ATPHKMA / _hc
        p.ATPHK[LUM, B] = ATPHKMB / _hc

    # ------------------------------------------------------------------
    # Kinetic parameters for carbonate reactions  [s^-1]
    # ------------------------------------------------------------------
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

    for p in ccd:
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

    # ------------------------------------------------------------------
    # Read inlet conditions from CNToutlet
    # ------------------------------------------------------------------
    with open('CNToutlet', 'r') as f:
        for I in range(1, NS + 1):
            ccd[0].conc[I-1, LUM], ccd[0].conc[I-1, BATH] = map(float, f.readline().split())
        ccd[0].ph[LUM],  ccd[0].ph[BATH]  = map(float, f.readline().split())
        ccd[0].ep[LUM],  ccd[0].ep[BATH]  = map(float, f.readline().split())
        ccd[0].vol[LUM], ccd[0].pres      = map(float, f.readline().split())

    # axial position (0 → 1 along segment)
    for jz in range(NZ + 1):
        pos[jz] = 1.0 * jz / NZ

    set_intconc(ccd, NZ, 1, pos)

    # ------------------------------------------------------------------
    # Cell parameters, inlet state, and metabolic ratio
    # ------------------------------------------------------------------
    for p in ccd:
        p.volLuminit = ccd[0].vol[LUM]
        p.ep[BATH]   = ccd[0].ep[BATH]
        # impermeant reference concentrations
        p.cPimpref = 50.0
        p.cAimpref = 18.0
        p.cBimpref = 18.0
        # total buffer concentrations
        p.cPbuftot = 32.0
        p.cAbuftot = 40.0
        p.cBbuftot = 40.0
        # coalescence parameter (constant along CCD)
        p.coalesce = 0.2
        # TNa-QO2 ratio: 15.0 in normal CCD, 12.0 in diabetic CCD
        p.TQ = 15.0 if not bdiabetes else 12.0

    # electroneutrality check (informational only; values not used)
    # elecM     = sum(zval[I] * ccd[0].conc[I, LUM]  for I in range(NS))
    # elecS     = sum(zval[I] * ccd[0].conc[I, BATH] for I in range(NS))
    # elecS_out = sum(zval[I] * ccd[NZ].conc[I, BATH] for I in range(NS))

    return hENaC_CCD, hROMK_CCD, hCltj_CCD
