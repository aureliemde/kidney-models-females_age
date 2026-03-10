""" Written originally in Fortran by Prof. Aurelie Edwards
 Translated to Python by Dr. Mohammad M. Tajdini
 Refactored by Sofia Polychroniadou

 Department of Biomedical Engineering
 Boston University

 Initialization of OMCD (outer medullary collecting duct) parameters.
 Sets up geometry, membrane permeabilities, transporter coefficients,
 and reads inlet conditions from CCDoutlet.
 Called from main.py; returns nothing (cf. initOMC_Var which returns
 hENaC_OMC, hROMK_OMC, hCltj_OMC for use by the flux function). """

import numpy as np

from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initOMC(omcd):
    """Initialize OMCD parameters: geometry, permeabilities, transporters, inlet conditions.

    Parameters
    ----------
    omcd : list of Membrane
        OMCD cell array (length NZ+1).
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
    # Geometry and volumes  (AMW model, AJP Renal 2001)
    # ------------------------------------------------------------------
    SlumPinitomc = 1.2
    SbasPinitomc = 1.2
    SlatPinitomc = 6.9

    SlumAinitomc = 2.0
    SbasAinitomc = 2.0
    SlatAinitomc = 6.0

    SlumEinitomc = 0.001

    # B cell is phantom — mirrors A geometry
    SlumBinitomc = SlumAinitomc
    SbasBinitomc = SbasAinitomc
    SlatBinitomc = SlatAinitomc

    for p in omcd:
        p.sbasEinit = 0.020
        p.volPinit  = 4.0
        p.volAinit  = 3.0
        p.volBinit  = 3.0   # phantom: mirrors A
        p.volEinit  = 0.80
        p.zPimp     = -1.0
        p.zAimp     = -1.0
        p.zBimp     = -1.0

    # ------------------------------------------------------------------
    # Membrane surface areas
    # ------------------------------------------------------------------
    for p in omcd:
        p.area[LUM, P]    = SlumPinitomc
        p.area[LUM, A]    = SlumAinitomc
        p.area[LUM, B]    = SlumBinitomc   # phantom: same as A
        p.area[LUM, LIS]  = SlumEinitomc
        p.area[P,   LIS]  = SlatPinitomc
        p.area[A,   LIS]  = SlatAinitomc
        p.area[B,   LIS]  = SlatBinitomc   # phantom: same as A
        p.area[P,   BATH] = SbasPinitomc
        p.area[A,   BATH] = SbasAinitomc
        p.area[B,   BATH] = SbasBinitomc   # phantom: same as A
        p.area += p.area.T  # symmetric copy

    # ------------------------------------------------------------------
    # Water permeabilities: dLPV(K,L) = Pf(K,L) / Pfref  [cm/s/mmHg]
    # ------------------------------------------------------------------
    PfMP = 0.20/1.2
    PfPE = 0.11*2
    PfPS = 0.11*2

    PfMA = 0.22e-3
    PfAE = 5.50e-3
    PfAS = 5.50e-3

    PfMB = 0.22e-3
    PfBE = 5.50e-3
    PfBS = 5.50e-3

    PfME = 28   # Per AMW email on 03/08/13
    PfES = 41   # check value; 41 in OMCD

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

    for p in omcd:
        p.dLPV = Pf / Pfref

    # ------------------------------------------------------------------
    # Reflection coefficients: sig = 1 everywhere, 0 at LIS-BATH
    # ------------------------------------------------------------------
    for p in omcd:
        p.sig[:] = 1.0
        p.sig[:, LIS, BATH] = 0.0

    # ------------------------------------------------------------------
    # Solute permeabilities  [10^-5 cm/s initially]
    # ------------------------------------------------------------------
    PICAchlo  = 1.2
    PPCh2co3  = 130
    PICh2co3  = 10
    PPCco2    = 2.0e3   # Per AMW email on 03/08/13
    PICco2    = 900
    # PPChpo4 = 8.0e-3    # unused
    PPCprot   = 2000
    # PICAprot = 1.5      # unused
    PPCamon   = 2000
    PICamon   = 900
    PPCurea   = 0.10
    PICurea   = 1.0

    # h arrays already zero from Membrane init; redundant init loop removed

    fac4 = 1.0/4.0

    # Female-to-male ENaC expression ratio in medulla (OMCD)
    FM_mENaC = 1.20 * 0.85

    # hENaC_OMC = FM_mENaC * 35.00 * fac4   # dead — not returned (cf. initOMC_Var)
    # hROMK_OMC = 8.0 * fac4                # dead — not returned (cf. initOMC_Var)

    # fscaleENaC   = 2.50   # dead local — not stored to p or scaling object
    # fscaleNaK    = 1.25   # dead local
    # fscaleNaK_PC = 1.25   # dead local

    # -- LUM-P interface (tight junction PC) ---------------------------
    # NA and K slots are 0 (ENaC/ROMK not assigned here; cf. initOMC_Var)
    # NH4 = h[K, LUM, P] * 0.20 = 0.0 (AMW rule of thumb)
    # GLU = 1.0e-10 — cannot be set to 0 (convergence issue)
    _h_lum_p = np.array([
        0.0,             # NA    — not set (cf. hENaC_OMC in initOMC_Var)
        0.0,             # K     — not set (cf. hROMK_OMC in initOMC_Var)
        0.80 * fac4,     # CL    (no apical Cl conductance, Jacques)
        0.16 * fac4,     # HCO3  (no apical conductance, Jacques)
        PPCh2co3,        # H2CO3
        PPCco2,          # CO2
        1.0e-3/1.20,     # HPO4  (as in CNT)
        1.0e-3/1.20,     # H2PO4 (as in CNT)
        PPCurea,         # UREA
        PPCamon,         # NH3
        0.0,             # NH4   — AMW rule: 0.20 * h[K,LUM,P] = 0
        PPCprot * fac4,  # H
        0.0,             # HCO2
        0.0,             # H2CO2
        1.0e-10,         # GLU   — no convergence if set to 0
        0.00010,         # CA
    ])

    # -- P-LIS interface (PC lateral) ----------------------------------
    # HCO3 = CL * 0.20 (AMW rule); NH4 = K * 0.20 (AMW rule)
    _h_p_lis = np.array([
        0.000,               # NA
        4.0 * fac4,          # K
        0.20 * fac4,         # CL
        0.20 * fac4 * 0.20,  # HCO3  — AMW rule of thumb
        PPCh2co3,            # H2CO3
        PPCco2,              # CO2
        8.0e-3,              # HPO4  (as in CNT)
        8.0e-3,              # H2PO4 (as in CNT)
        PPCurea,             # UREA
        PPCamon,             # NH3
        4.0 * fac4 * 0.20,  # NH4   — AMW rule of thumb
        PPCprot * fac4,      # H
        1.0e-4,              # HCO2
        1.0e-4,              # H2CO2
        1.0e-4,              # GLU
        0.00010,             # CA
    ])

    # -- LUM-A interface (IC-A lumen) ----------------------------------
    # Note: PICurea (not PPCurea) for UREA; NH4 = 0.10e-4 (not zero)
    _h_lum_a = np.array([
        0.0,       # NA
        0.00,      # K
        0.0,       # CL
        0.0,       # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        0.0,       # HPO4
        0.0,       # H2PO4
        PICurea,   # UREA  (IC uses PICurea = 1.0, not PPCurea)
        PICamon,   # NH3
        0.10e-4,   # NH4
        0.0,       # H
        0.0,       # HCO2
        0.0,       # H2CO2
        0.0,       # GLU
        0.00010,   # CA
    ])

    # -- A-LIS interface (IC-A lateral) --------------------------------
    _h_a_lis = np.array([
        0.0,       # NA
        0.12,      # K
        PICAchlo,  # CL   (= 1.2)
        0.15,      # HCO3
        PICh2co3,  # H2CO3
        PICco2,    # CO2
        1.20e-3,   # HPO4
        1.20e-3,   # H2PO4
        PICurea,   # UREA
        PICamon,   # NH3
        0.030,     # NH4
        1.50,      # H
        1.0e-4,    # HCO2
        1.0e-4,    # H2CO2
        1.0e-4,    # GLU
        0.00010,   # CA
    ])

    # -- LUM-LIS interface (tight junction, ME) ------------------------
    # Assume TJ resistivity of 5 mS/cm2
    ClTJperm = 1000.0
    # hCltj_OMC = 1.3 * ClTJperm * 1.00   # dead — not returned (cf. initOMC_Var)

    _h_lum_lis = np.array([
        (1.00 / FM_mENaC) * 0.80 * ClTJperm,   # NA
        (1.00 / FM_mENaC) * 1.20 * ClTJperm,   # K
        0.0,                                    # CL   — via hCltj_OMC in initOMC_Var
        0.30 * ClTJperm,                        # HCO3  (changed from 0.15 on 04/30/19)
        1.2  * ClTJperm,                        # H2CO3
        1.2  * ClTJperm,                        # CO2
        0.10 * ClTJperm,                        # HPO4
        0.10 * ClTJperm,                        # H2PO4
        2.0  * ClTJperm,                        # UREA  (Per AMW email on 03/08/13)
        1.0  * ClTJperm,                        # NH3
        1.5  * ClTJperm,                        # NH4
        6.0  * ClTJperm,                        # H
        0.01 * ClTJperm,                        # HCO2
        0.01 * ClTJperm,                        # H2CO2
        0.01 * ClTJperm,                        # GLU
        0.0140 * ClTJperm,                      # CA  (see 2015 AJP and Carney 1988)
    ])

    # -- LIS-BATH interface (paracellular, ES; all values from AWM model) -
    areafactor = 50
    _h_lis_bath = np.array([
        89    * areafactor,               # NA
        118   * areafactor,               # K
        118   * areafactor,               # CL
        59    * areafactor,               # HCO3
        89    * areafactor,               # H2CO3
        89    * areafactor,               # CO2
        59    * areafactor,               # HPO4
        59    * areafactor,               # H2PO4
        59    * areafactor,               # UREA
        59    * areafactor,               # NH3
        118   * areafactor,               # NH4
        590.0 * areafactor,               # H
        1.0   * areafactor,               # HCO2
        1.0   * areafactor,               # H2CO2
        1.0   * areafactor,               # GLU
        89    * areafactor * (7.93/13.3), # CA
    ])

    for p in omcd:
        p.h[:, LUM, P]    = _h_lum_p
        p.h[:, LUM, A]    = _h_lum_a
        # LUM-B, B-LIS, B-BATH: phantom cell — already zero (redundant zeroing removed)
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, A,   LIS]  = _h_a_lis
        p.h[:, LUM, LIS]  = _h_lum_lis
        p.h[:, LIS, BATH] = _h_lis_bath
        # basal membranes (PS, AS) copy from LIS-side arrays
        p.h[:, P,   BATH] = _h_p_lis
        p.h[:, A,   BATH] = _h_a_lis
        p.hCLCA = PICAchlo
        p.h *= 1.0e-5 / href

    # ------------------------------------------------------------------
    # NET cross-coupling coefficients dLA  [mmol^2/J/s initially]
    # ------------------------------------------------------------------
    # dLA already zero from np.zeros; redundant init loop removed

    # Na/H exchangers on all basolateral membranes
    dLA[NA, H, P,    LIS]  = 2.50e-9
    dLA[NA, H, P,    BATH] = 2.50e-9
    dLA[NA, H, A,    LIS]  = 6.0e-9
    dLA[NA, H, A,    BATH] = 6.0e-9

    xNHE1P = 6.95e-10 * 680 / 695 * fac4

    # Na2/HPO4 co-transporters on all basolateral membranes
    dLA[NA, HPO4, P,    LIS]  = 2.0e-9
    dLA[NA, HPO4, P,    BATH] = 2.0e-9
    dLA[NA, HPO4, A,    LIS]  = 0.20e-9
    dLA[NA, HPO4, A,    BATH] = 0.20e-9

    # Basolateral Cl/HCO3 exchanger in PC
    dLA[CL, HCO3, P,   LIS]  = 2.0e-9 * fac4
    dLA[CL, HCO3, P,   BATH] = 2.0e-9 * fac4

    # AE1 in IC-A only
    xAE1 = 12.0e-9

    # Symmetrize dLA: dLA[J,I,:,:] = dLA[I,J,:,:]
    dLA = dLA + dLA.transpose(1, 0, 2, 3)

    # ------------------------------------------------------------------
    # ATPase coefficients  [mmol/s]
    # ------------------------------------------------------------------
    # Na-K-ATPase (basolateral in PC and IC-A only; no IC-B in OMCD)
    ATPNaKPES = FM_mENaC * 3410e-9 * 1.25 * fac4
    ATPNaKAES = 75e-9

    # H-ATPase (apical in IC-A only)
    ATPHMA = 750.0e-9

    # H-K-ATPase (apical in PC and IC-A only)
    ATPHKMP = 0
    ATPHKMA = 150e-9

    # ------------------------------------------------------------------
    # Non-dimensionalize and assign transporters
    # ------------------------------------------------------------------
    _hc = href * Cref
    for p in omcd:
        p.dLA = dLA / _hc

        p.xNHE1[P] = xNHE1P / _hc
        p.xAE1     = xAE1   / _hc

        p.ATPNaK[P,   LIS]  = ATPNaKPES / _hc
        p.ATPNaK[P,   BATH] = ATPNaKPES / _hc
        p.ATPNaK[A,   LIS]  = ATPNaKAES / _hc
        p.ATPNaK[A,   BATH] = ATPNaKAES / _hc

        p.ATPH[LUM, A] = ATPHMA / _hc

        p.ATPHK[LUM, P] = ATPHKMP / _hc
        p.ATPHK[LUM, A] = ATPHKMA / _hc

    # ------------------------------------------------------------------
    # Kinetic parameters for carbonate reactions  [s^-1]
    # Note: OMCD lumen uses uncatalysed rate (no x10 multiplier unlike CCD).
    #       LIS uses slower rate (dkdE = 49.6, not 49.6e3).
    # ------------------------------------------------------------------
    dkhuncat = 0.145
    dkduncat = 49.6
    dkhP = 1.45
    dkdP = 496.0
    dkhA = 1.45e3
    dkdA = 496.0e3
    dkhB = 1.45e3
    dkdB = 496.0e3
    dkhE = 0.145
    dkdE = 49.6

    for p in omcd:
        p.dkd[LUM] = dkduncat
        p.dkh[LUM] = dkhuncat
        p.dkd[P]   = dkdP
        p.dkh[P]   = dkhP
        p.dkd[A]   = dkdA
        p.dkh[A]   = dkhA
        p.dkd[B]   = dkdB
        p.dkh[B]   = dkhB
        p.dkd[LIS] = dkdE
        p.dkh[LIS] = dkhE

    # ------------------------------------------------------------------
    # Boundary conditions: peritubular potential + axial position
    # Note: set_intconc is called before reading CCDoutlet (preserve order).
    # ------------------------------------------------------------------
    omcd[0].ep[BATH] = -0.001e-3 / EPref

    for jz in range(NZ + 1):
        omcd[jz].ep[BATH] = omcd[0].ep[BATH]
        pos[jz] = 1.0 * jz / NZ

    set_intconc(omcd, NZ, 2, pos)

    # coalescence parameter (constant along OMCD)
    for p in omcd:
        p.coalesce = 0.2

    # ------------------------------------------------------------------
    # Read inlet conditions from CCDoutlet
    # ------------------------------------------------------------------
    with open('CCDoutlet', 'r') as f:
        for I in range(1, NS + 1):
            omcd[0].conc[I-1, LUM], omcd[0].conc[I-1, BATH] = map(float, f.readline().split())
        omcd[0].ph[LUM],  omcd[0].ph[BATH]  = map(float, f.readline().split())
        omcd[0].ep[LUM],  omcd[0].ep[BATH]  = map(float, f.readline().split())
        omcd[0].vol[LUM], omcd[0].pres      = map(float, f.readline().split())

    # ------------------------------------------------------------------
    # Final per-cell state and metabolic ratio
    # ------------------------------------------------------------------
    for p in omcd:
        p.volLuminit = omcd[0].vol[LUM]
        # TNa-QO2 ratio: 15.0 in normal OMCD, 12.0 in diabetic OMCD
        p.TQ = 15.0 if not bdiabetes else 12.0

    # electroneutrality check (informational only; values not used)
    # elecM     = sum(zval[I] * omcd[0].conc[I, LUM]  for I in range(NS))
    # elecS     = sum(zval[I] * omcd[0].conc[I, BATH] for I in range(NS))
    # elecS_out = sum(zval[I] * omcd[NZ].conc[I, BATH] for I in range(NS))

    return
