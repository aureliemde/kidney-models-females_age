"""Initialization of transporter parameters for the CNT (Connecting Tubule) segment.

Sets up membrane geometry, surface areas, water permeabilities, solute
permeabilities, NET coefficients, ATPase rates, carbonate kinetics, and
inlet conditions read from the DCToutlet file.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University
"""

import numpy as np

from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initC_Var(cnt):
    """Initialize CNT segment parameters and return dynamic permeability scalars.

    Args:
        cnt: list of Membrane objects (length NZ+1) to be initialized in-place.

    Returns:
        hENaC_CNT:   basal ENaC permeability (set dynamically in qflux2C)
        hROMK_CNT:   basal ROMK permeability (set dynamically in qflux2C)
        hCltj_CNT:   basal Cl tight-junction permeability (set dynamically in qflux2C)
        xTRPV5_cnt:  non-dimensional TRPV5 Ca channel activity
        xPTRPV4_cnt: non-dimensional TRPV4 channel activity
    """
    # theta = np.zeros(NC)  # unused geometric helpers
    # Slum  = np.zeros(NC)
    # Slat  = np.zeros(NC)
    # Sbas  = np.zeros(NC)

    dLA = np.zeros((NS, NS, NC, NC))
    pos = np.zeros(NZ + 1)

    # ------------------------------------------------------------------
    # Geometry, initial volumes, and impermeable charge valences
    # ------------------------------------------------------------------
    for p in cnt:
        p.dimL = dimLC
        p.diam = DiamC
        p.sbasEinit = 0.020
        p.volPinit  = 6.0
        p.volAinit  = 2.7
        p.volBinit  = 1.8
        p.volEinit  = 0.20
        p.zPimp = -1.0
        p.zAimp = -1.0
        p.zBimp = -1.0

    # ------------------------------------------------------------------
    # Membrane surface areas (cm² epith / cm² lumen)
    # ------------------------------------------------------------------
    SlumPinitcnt = 1.2;   SbasPinitcnt = 1.2;   SlatPinitcnt = 6.9
    SlumAinitcnt = 0.58;  SbasAinitcnt = 0.58;  SlatAinitcnt = 1.25
    SlumBinitcnt = 0.17;  SbasBinitcnt = 0.17;  SlatBinitcnt = 1.50
    SlumEinitcnt = 0.001

    for p in cnt:
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
        # Symmetric copy (lower triangle = transpose of upper triangle)
        p.area += p.area.T

    # ------------------------------------------------------------------
    # Water permeabilities (cm/s → non-dimensional via dLPV = Pf / Pfref)
    # ------------------------------------------------------------------
    PfMP = 2.4 * 0.036;  PfMA = 0.22e-3;  PfMB = 0.22e-3;  PfME = 1.1
    PfPE = 2.4 * 0.44;   PfAE = 5.50e-3;  PfBE = 5.50e-3
    PfPS = 2.4 * 0.44;   PfAS = 5.50e-3;  PfBS = 5.50e-3
    PfES = 110.0

    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = PfMP;  Pf[LUM, A]    = PfMA;  Pf[LUM, B]   = PfMB
    Pf[LUM, LIS]  = PfME
    Pf[P,   LIS]  = PfPE;  Pf[A,   LIS]  = PfAE;  Pf[B,  LIS]  = PfBE
    Pf[P,   BATH] = PfPS;  Pf[A,   BATH] = PfAS;  Pf[B,  BATH] = PfBS
    Pf[LIS, BATH] = PfES

    for p in cnt:
        p.dLPV = Pf / Pfref

    # ------------------------------------------------------------------
    # Reflection coefficients: 1 everywhere except ES interface (= 0)
    # ------------------------------------------------------------------
    for p in cnt:
        p.sig[:]          = 1.0
        p.sig[:, LIS, BATH] = 0.0

    # ------------------------------------------------------------------
    # Solute permeabilities h(I,K,L) in units of 1.0e-5 cm/s
    # Defined per interface as numpy arrays; non-dimensionalized after assignment.
    # ------------------------------------------------------------------
    PICAchlo  = 11.0;    PICBchlo  = 3.20
    PPCh2co3  = 130.0;   PICh2co3  = 10.0
    PPCco2    = 15.0e3;  PICco2    = 900.0
    # PPChpo4 = 8.00e-3  # unused; LUM-P HPO4 uses hardcoded 1.0e-3/1.20
    PICAhpo4  = 7.20e-3; PICBhpo4  = 4.80e-3
    PPCprot   = 2000.0;  PICAprot  = 9.0;   PICBprot  = 6.0
    PPCamon   = 2000.0;  PICamon   = 900.0
    Purea     = 0.10

    # Female-to-male ENaC expression ratio in cortex (CNT and CCD)
    FM_cENaC = 1.20 * 0.85

    # Dynamic scalars returned to caller (used in qflux2C to update h each Newton step)
    # fscaleENaC  = 2.50   # dead: not stored to p or to global scaling object
    # fscaleNaK   = 1.25   # dead
    # fscaleNaK_PC = 1.25  # dead
    hENaC_CNT = FM_cENaC * 35.0   # basal ENaC — set dynamically via hENaC_CNT * facNaMP * facphMP
    hROMK_CNT = 8.0                # basal ROMK  — set dynamically via hROMK_CNT * facphMP

    ClTJperm  = 1000.0             # TJ resistivity ~ 5 mS/cm2
    hCltj_CNT = 1.3 * ClTJperm * 1.2  # basal Cl TJ — set dynamically via hCltj_CNT * facphTJ

    # LUM-P (apical principal cell): ENaC (NA) and ROMK (K) set dynamically → 0 at init
    _h_lum_p = np.array([
        0.0,              # NA  (ENaC — set dynamically)
        0.0,              # K   (ROMK — set dynamically)
        0.80,             # CL
        0.16,             # HCO3
        PPCh2co3,         # H2CO3
        PPCco2,           # CO2
        1.0e-3 / 1.20,    # HPO4
        1.0e-3 / 1.20,    # H2PO4
        Purea,            # UREA
        PPCamon,          # NH3
        0.0,              # NH4 (= h[K,LUM,P] * 0.20; 0 at init since ROMK not yet set)
        PPCprot,          # H
        0.0,              # HCO2
        0.0,              # H2CO2
        0.0,              # GLU
        0.0,              # CA
    ])

    # LUM-A and LUM-B (apical IC-alpha and IC-beta): identical permeabilities
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

    # LUM-LIS (tight junction): CL (hCltj_CNT) set dynamically → 0 at init
    _h_lum_lis = np.array([
        (1.00 / FM_cENaC) * 1.0 * ClTJperm,   # NA
        (1.00 / FM_cENaC) * 1.2 * ClTJperm,   # K
        0.0,                                    # CL (set dynamically via hCltj_CNT)
        0.6  * ClTJperm,                        # HCO3
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
        0.0001 * ClTJperm,                      # CA
    ])

    # P-LIS (basolateral principal cell); copied to P-BATH
    _h_p_lis = np.array([
        0.000,          # NA
        4.0,            # K
        0.2,            # CL
        0.2 * 0.20,     # HCO3 (= h[CL,P,LIS] * 0.20, AMW rule of thumb)
        PPCh2co3,       # H2CO3
        PPCco2,         # CO2
        8.0e-3,         # HPO4
        8.0e-3,         # H2PO4
        Purea,          # UREA
        PPCamon,        # NH3
        4.0 * 0.20,     # NH4 (= h[K,P,LIS] * 0.20, AMW rule of thumb)
        PPCprot,        # H
        1.0e-4,         # HCO2
        1.0e-4,         # H2CO2
        1.0e-4,         # GLU
        0.0,            # CA
    ])

    # A-LIS (basolateral IC-alpha); copied to A-BATH
    _h_a_lis = np.array([
        0.0,        # NA
        0.448,      # K
        PICAchlo,   # CL
        1.50,       # HCO3
        PICh2co3,   # H2CO3
        PICco2,     # CO2
        PICAhpo4,   # HPO4
        PICAhpo4,   # H2PO4
        Purea,      # UREA
        PICamon,    # NH3
        0.18,       # NH4
        PICAprot,   # H
        1.0e-4,     # HCO2
        1.0e-4,     # H2CO2
        1.0e-4,     # GLU
        1.0e-5,     # CA
    ])

    # B-LIS (basolateral IC-beta); copied to B-BATH
    _h_b_lis = np.array([
        0.0,        # NA
        0.12,       # K
        PICBchlo,   # CL
        0.40,       # HCO3
        PICh2co3,   # H2CO3
        PICco2,     # CO2
        PICBhpo4,   # HPO4
        PICBhpo4,   # H2PO4
        Purea,      # UREA
        PICamon,    # NH3
        0.12,       # NH4
        PICBprot,   # H
        1.0e-4,     # HCO2
        1.0e-4,     # H2CO2
        1.0e-4,     # GLU
        1.0e-5,     # CA
    ])

    # LIS-BATH (lateral intercellular space to bath); AWM model values × areafactor
    areafactor = 50
    _h_lis_bath = np.array([
        240.0   * areafactor,                  # NA
        320.0   * areafactor,                  # K
        320.0   * areafactor,                  # CL
        160.0   * areafactor,                  # HCO3
        240.0   * areafactor,                  # H2CO3
        240.0   * areafactor,                  # CO2
        160.0   * areafactor,                  # HPO4
        160.0   * areafactor,                  # H2PO4
        160.0   * areafactor,                  # UREA
        320.0   * areafactor,                  # NH3
        320.0   * areafactor,                  # NH4
        16000.0 * areafactor,                  # H
        4.0     * areafactor,                  # HCO2
        4.0     * areafactor,                  # H2CO2
        4.0     * areafactor,                  # GLU
        240.0   * areafactor * (7.93 / 13.3),  # CA (= h[NA,LIS,BATH] * 7.93/13.3)
    ])

    for p in cnt:
        # p.h[:] = 0.0  # redundant: Membrane initializes h to zeros
        p.h[:, LUM, P]    = _h_lum_p
        p.h[:, LUM, A]    = _h_lum_ic
        p.h[:, LUM, B]    = _h_lum_ic
        p.h[:, LUM, LIS]  = _h_lum_lis
        p.h[:, P,   LIS]  = _h_p_lis
        p.h[:, A,   LIS]  = _h_a_lis
        p.h[:, B,   LIS]  = _h_b_lis
        p.h[:, P,   BATH] = _h_p_lis    # copy from LIS interface
        p.h[:, A,   BATH] = _h_a_lis    # copy from LIS interface
        p.h[:, B,   BATH] = _h_b_lis    # copy from LIS interface
        p.h[:, LIS, BATH] = _h_lis_bath
        # Store CL channel permeabilities for dynamic pH modulation in qflux2C
        p.hCLCA = PICAchlo
        p.hCLCB = PICBchlo
        # Non-dimensionalize: h(I,K,L) *= 1.0e-5 cm/s / href
        p.h *= 1.0e-5 / href

    # ------------------------------------------------------------------
    # NET (coupled-flux) coefficients dLA (mmol²/J/s)
    # ------------------------------------------------------------------
    # dLA is already zeros; explicit zeroing loop omitted as redundant.

    # NaH exchangers on all basolateral membranes
    dLA[NA, H, P, LIS]  = 20.0e-9;  dLA[NA, H, P, BATH] = 20.0e-9
    dLA[NA, H, A, LIS]  = 36.0e-9;  dLA[NA, H, A, BATH] = 36.0e-9
    dLA[NA, H, B, LIS]  = 24.0e-9;  dLA[NA, H, B, BATH] = 24.0e-9

    # Na2/HPO4 cotransporters on all basolateral membranes
    dLA[NA, HPO4, P, LIS]  = 2.0e-9;   dLA[NA, HPO4, P, BATH] = 2.0e-9
    dLA[NA, HPO4, A, LIS]  = 1.2e-9;   dLA[NA, HPO4, A, BATH] = 1.2e-9
    dLA[NA, HPO4, B, LIS]  = 0.80e-9;  dLA[NA, HPO4, B, BATH] = 0.80e-9

    # Apical NCC in PC — removed (multiplied by 0)
    # xNCC = 20.0e-9 * 0  # dead; NCC disabled

    # Basolateral Cl/HCO3 exchanger in PC
    dLA[CL, HCO3, P, LIS]  = 2.0e-9
    dLA[CL, HCO3, P, BATH] = 2.0e-9

    # Basolateral Cl/HCO3 exchanger in IC-B — removed (multiplied by 0)
    dLA[CL, HCO3, B, LIS]  = 16.0e-9 * 0
    dLA[CL, HCO3, B, BATH] = 16.0e-9 * 0

    # AE1 exchanger (IC-A basolateral only)
    xAE1 = 150e-9

    # Pendrin (apical IC-B): set as fraction of dLA[CL,HCO3,LUM,B]
    dLA[CL, HCO3, LUM, B] = 800e-9
    xPendrin = 1.2 * dLA[CL, HCO3, LUM, B] / 1400   # female adjustment

    # Symmetrize dLA in the solute-pair (I,J) dimension: dLA[J,I,:,:] = dLA[I,J,:,:]
    dLA = dLA + dLA.transpose(1, 0, 2, 3)

    # ------------------------------------------------------------------
    # Calcium-specific transporters
    # ------------------------------------------------------------------
    xNCX  = 25.0e-9 * 1.60   # basolateral NCX exchanger
    PMCA  = 2.0e-9  * 0.40   # basolateral PMCA pump
    xTRPV5 = 8.0e6            # apical TRPV5 Ca channel
    Po_TRPV4  = 1.0 / (1.0 + 780 / 37)
    PCa_TRPV4 = 4.5e-8
    xPTRPV4   = Po_TRPV4 * PCa_TRPV4  # apical TRPV4 channel density × single-channel permeability

    # ------------------------------------------------------------------
    # ATPase coefficients (mmol/s → non-dimensional via href*Cref)
    # ------------------------------------------------------------------
    ATPNaKPES = FM_cENaC * 3410e-9 * 1.25   # Na-K-ATPase, P cell basolateral
    ATPNaKAES = 450e-9                        # Na-K-ATPase, A cell basolateral
    ATPNaKBES = 300e-9                        # Na-K-ATPase, B cell basolateral

    ATPHMA  = 12000.0e-9                      # H-ATPase, apical IC-A
    ATPHBES = 400e-9 * 15                     # H-ATPase, basolateral IC-B (adjusted)

    ATPHKMP = 0                               # H-K-ATPase, apical P  (disabled)
    ATPHKMA = 720e-9                          # H-K-ATPase, apical A
    ATPHKMB = 0.0e-9                          # H-K-ATPase, apical B  (disabled)

    _hc = href * Cref
    for p in cnt:
        p.dLA                = dLA / _hc
        p.xPendrin           = xPendrin / _hc
        p.xAE1               = xAE1  / _hc
        p.xNCX               = xNCX  / _hc
        p.PMCA               = PMCA  / _hc
        p.ATPNaK[P,   LIS]  = ATPNaKPES / _hc
        p.ATPNaK[P,   BATH] = ATPNaKPES / _hc
        p.ATPNaK[A,   LIS]  = ATPNaKAES / _hc
        p.ATPNaK[A,   BATH] = ATPNaKAES / _hc
        p.ATPNaK[B,   LIS]  = ATPNaKBES / _hc
        p.ATPNaK[B,   BATH] = ATPNaKBES / _hc
        p.ATPH[LUM, A]      = ATPHMA  / _hc
        p.ATPH[B,   LIS]    = ATPHBES / _hc
        p.ATPH[B,   BATH]   = ATPHBES / _hc
        p.ATPHK[LUM, P]     = ATPHKMP / _hc
        p.ATPHK[LUM, A]     = ATPHKMA / _hc
        p.ATPHK[LUM, B]     = ATPHKMB / _hc

    xTRPV5_cnt  = xTRPV5  / _hc
    xPTRPV4_cnt = xPTRPV4 / _hc

    # ------------------------------------------------------------------
    # Carbonate kinetics (s⁻¹)
    # ------------------------------------------------------------------
    dkhuncat = 0.145;   dkduncat = 49.6
    dkhP     = 1.45;    dkdP     = 496.0
    dkhA     = 1.45e3;  dkdA     = 496.0e3
    dkhB     = 1.45e3;  dkdB     = 496.0e3
    dkhE     = 0.145e3; dkdE     = 49.6e3

    for p in cnt:
        p.dkd[LUM] = dkduncat * 10.0   # lumen: uncatalysed × 10
        p.dkh[LUM] = dkhuncat * 10.0
        p.dkd[P]   = dkdP;  p.dkh[P]   = dkhP
        p.dkd[A]   = dkdA;  p.dkh[A]   = dkhA
        p.dkd[B]   = dkdB;  p.dkh[B]   = dkhB
        p.dkd[LIS] = dkdE;  p.dkh[LIS] = dkhE

    # ------------------------------------------------------------------
    # Inlet and peritubular conditions from DCToutlet
    # ------------------------------------------------------------------
    with open('DCToutlet', 'r') as f:
        for I in range(1, 5):
            cnt[0].conc[I-1, LUM], cnt[0].conc[I-1, BATH] = map(float, f.readline().split())
        for I in range(5, NS + 1):
            cnt[0].conc[I-1, LUM], cnt[0].conc[I-1, BATH] = map(float, f.readline().split())
        cnt[0].ph[LUM],  cnt[0].ph[BATH]  = map(float, f.readline().split())
        cnt[0].ep[LUM],  cnt[0].ep[BATH]  = map(float, f.readline().split())
        cnt[0].vol[LUM], cnt[0].pres      = map(float, f.readline().split())

    for p in cnt:
        p.volLuminit = cnt[0].vol[LUM]
        p.ep[BATH]   = cnt[0].ep[BATH]

    # pos[jz] = 0 for all jz: stays near cortex surface (already zeros from np.zeros)
    set_intconc(cnt, NZ, 1, pos)

    # ------------------------------------------------------------------
    # Cell parameters: impermeant concentrations and buffer totals
    # ------------------------------------------------------------------
    for p in cnt:
        p.cPimpref = 50.0;  p.cAimpref = 18.0;  p.cBimpref = 18.0
        p.cPbuftot = 32.0;  p.cAbuftot = 40.0;  p.cBbuftot = 40.0

    # Coalescence parameter (axial gradient; indexed by jz, not via for p in cnt)
    for jz in range(NZ + 1):
        cnt[jz].coalesce = 2.00 ** (-2.32 * jz / NZ)

    # Metabolic parameters
    for p in cnt:
        p.TQ = 15.0 if not bdiabetes else 12.0  # TNa-QO2 ratio (normal vs diabetic CNT)

    # Electroneutrality check (diagnostic only — values not used downstream):
    # elecM     = np.dot(zval, cnt[0].conc[:, LUM])
    # elecS     = np.dot(zval, cnt[0].conc[:, BATH])
    # elecS_out = np.dot(zval, cnt[NZ].conc[:, BATH])

    return hENaC_CNT, hROMK_CNT, hCltj_CNT, xTRPV5_cnt, xPTRPV4_cnt
