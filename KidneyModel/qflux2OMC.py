"""Flux computation for the OMCD (Outer Medullary Collecting Duct) segment.

Computes volume and solute fluxes across all membranes at a given
axial position, given the state vector x (concentrations, volumes,
potentials).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Input:
    x: state vector — NC-1 compartments × NS2 concentrations, volumes, EPs, pressure

Output:
    Jvol: volume fluxes [NC × NC]
    Jsol: solute fluxes [NS × NC × NC]
"""

import numpy as np

from values import *
from glo import *
from defs import *
from compute_water_fluxes import compute_water_fluxes
from compute_ecd_fluxes import compute_ecd_fluxes
from fatpase import fatpase


# ---------------------------------------------------------------------------
# Module-level initialization: OMCD transporter parameters.
# ---------------------------------------------------------------------------
omcd = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initOMC_Var import initOMC_Var
hENaC_OMC, hROMK_OMC, hCltj_OMC = initOMC_Var(omcd)


def qflux2OMC(x, omcd, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute volume and solute fluxes in the OMCD at axial position Lz+1.

    Args:
        x:          State vector (concentrations, volumes, EPs, pressure)
        omcd:       List of Membrane objects for the OMCD (0 to NZ)
        Lz:         Current upstream position index
        PTinitVol:  Initial PT luminal volume
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity

    Returns:
        Jvol: np.ndarray of shape (NC, NC) — volume fluxes
        Jsol: np.ndarray of shape (NS, NC, NC) — solute fluxes
    """
    # LzOMC = Lz  # dead alias; Lz used directly throughout

    _nc = NC - 1  # 5 active compartments: LUM, P, A, B, LIS (BATH is fixed boundary)

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------
    # ONC    = np.zeros(NC)        # unused
    # PRES   = np.zeros(NC)        # unused
    # dmu    = np.zeros((NS, NC))  # unused; dmu computed internally by compute_ecd_fluxes
    # delmu2 = np.zeros((NS, NC, NC))  # unused
    # theta  = np.zeros(NC)        # unused geometric helpers
    # Slum   = np.zeros(NC)
    # Slat   = np.zeros(NC)
    # Sbas   = np.zeros(NC)
    # lwork  = 10000               # unused LAPACK workspace
    # ipiv   = np.zeros(Natp)
    # work   = np.zeros(lwork)

    C      = np.zeros((NS, NC))
    ph     = np.zeros(NC)
    Vol    = np.zeros(NC)
    EP     = np.zeros(NC)
    hkconc = np.zeros(4)

    # ------------------------------------------------------------------
    # Assign concentrations, volumes, and potentials
    # ------------------------------------------------------------------
    _ox  = _nc * NS2
    _oxv = _ox  + _nc
    _oxe = _oxv + _nc

    C[:NS, BATH] = omcd[Lz+1].conc[:NS, BATH]
    EP[BATH]     = omcd[Lz+1].ep[BATH]

    C[:NS2, :_nc] = x[:_ox].reshape(NS2, _nc)
    ph[:_nc]       = -np.log10(C[H, :_nc] / 1.0e3)
    Vol[:_nc]      = x[_ox:_oxv]
    EP[:_nc]       = x[_oxv:_oxe]
    PM             = x[_oxe]

    # ------------------------------------------------------------------
    # Update LIS–BATH surface area
    # ------------------------------------------------------------------
    omcd[Lz+1].area[LIS, BATH] = omcd[Lz+1].sbasEinit * max(Vol[LIS] / omcd[Lz+1].volEinit, 1.0)
    omcd[Lz+1].area[BATH, LIS] = omcd[Lz+1].area[LIS, BATH]

    # ------------------------------------------------------------------
    # Initialize fluxes
    # ------------------------------------------------------------------
    Jvol = np.zeros((NC, NC))
    Jsol = np.zeros((NS, NC, NC))

    # ------------------------------------------------------------------
    # Water fluxes
    # ------------------------------------------------------------------
    Jvol = compute_water_fluxes(C, PM, 0, Vol, omcd[Lz+1].volLuminit,
                                omcd[Lz+1].volEinit, omcd[Lz+1].volPinit, CPimprefOMC,
                                omcd[Lz+1].volAinit, CAimprefOMC, omcd[Lz+1].volBinit,
                                CBimprefOMC, omcd[Lz+1].area, omcd[Lz+1].sig, omcd[Lz+1].dLPV,
                                complOMC, PTinitVol)

    # ------------------------------------------------------------------
    # Solute fluxes
    # ------------------------------------------------------------------
    convert = href * Cref * np.pi * DiamOMC * 60 / 10 * 1.0e9  # mmol/s/cm² → pmol/min/mm

    # pH-dependent modulation of ENaC, ROMK, and apical paracellular Cl permeability
    facphMP = 1.0 * (0.1 + 2.0 / (1 + np.exp(-6.0 * (ph[P]   - 7.50))))
    facphTJ = 2.0 / (1.0 + np.exp(10.0 * (ph[LIS] - 7.32)))
    facNaMP = (30 / (30 + C[NA, LUM])) * (50 / (50 + C[NA, P]))

    omcd[Lz+1].h[NA, LUM, P]   = hENaC_OMC * facNaMP * facphMP
    omcd[Lz+1].h[K,  LUM, P]   = hROMK_OMC * facphMP
    omcd[Lz+1].h[CL, LUM, LIS] = hCltj_OMC * facphTJ

    # ECD fluxes and electrochemical driving forces
    Jsol, delmu = compute_ecd_fluxes(C, EP, omcd[Lz+1].area, omcd[Lz+1].sig, omcd[Lz+1].h, Jvol)

    # JNa: spot-check of ENaC flux — duplicates Jsol[NA, LUM, P]; never used:
    # XI   = zval[NA] * F * EPref / RT * (EP[LUM] - EP[P])
    # dint = np.exp(-XI)
    # if abs(1.0 - dint) < 1e-6:
    #     JNa = omcd[Lz+1].area[LUM, P] * omcd[Lz+1].h[NA, LUM, P] * (C[NA, LUM] - C[NA, P])
    # else:
    #     JNa = omcd[Lz+1].area[LUM, P] * omcd[Lz+1].h[NA, LUM, P] * XI * (C[NA, LUM] - C[NA, P] * dint) / (1.0 - dint)

    # Dimensional ECD fluxes — all commented out as unused:
    # fluxENaC      = Jsol[NA,    LUM, P]   * convert
    # fluxROMK      = Jsol[K,     LUM, P]   * convert
    # fluxKchPES    = (Jsol[K,    P,   LIS] + Jsol[K,    P,   BATH]) * convert
    # fluxKchAES    = (Jsol[K,    A,   LIS] + Jsol[K,    A,   BATH]) * convert
    # fluxClchPES   = (Jsol[CL,   P,   LIS] + Jsol[CL,   P,   BATH]) * convert
    # fluxClchAES   = (Jsol[CL,   A,   LIS] + Jsol[CL,   A,   BATH]) * convert
    # fluxBichPES   = (Jsol[HCO3, P,   LIS] + Jsol[HCO3, P,   BATH]) * convert
    # fluxBichAES   = (Jsol[HCO3, A,   LIS] + Jsol[HCO3, A,   BATH]) * convert
    # fluxH2CO3MP   = Jsol[H2CO3, LUM, P]   * convert
    # fluxH2CO3MA   = Jsol[H2CO3, LUM, A]   * convert
    # fluxCO2MP     = Jsol[CO2,   LUM, P]   * convert
    # fluxCO2MA     = Jsol[CO2,   LUM, A]   * convert
    # fluxH2CO3PES  = (Jsol[H2CO3, P, LIS] + Jsol[H2CO3, P, BATH]) * convert
    # fluxH2CO3AES  = (Jsol[H2CO3, A, LIS] + Jsol[H2CO3, A, BATH]) * convert
    # fluxCO2PES    = (Jsol[CO2,   P, LIS] + Jsol[CO2,   P, BATH]) * convert
    # fluxCO2AES    = (Jsol[CO2,   A, LIS] + Jsol[CO2,   A, BATH]) * convert
    # fluxHP2mchPES = (Jsol[HPO4,  P, LIS] + Jsol[HPO4,  P, BATH]) * convert
    # fluxHP2mchAES = (Jsol[HPO4,  A, LIS] + Jsol[HPO4,  A, BATH]) * convert
    # fluxHPmPES    = (Jsol[H2PO4, P, LIS] + Jsol[H2PO4, P, BATH]) * convert
    # fluxHPmAES    = (Jsol[H2PO4, A, LIS] + Jsol[H2PO4, A, BATH]) * convert

    # ------------------------------------------------------------------
    # Cotransporters
    # ------------------------------------------------------------------

    # Na2HPO4 cotransporter at PE,PS,AE,AS interfaces (P and A only; no B in OMCD)
    for comp in (P, A):
        # sumJES = 0.0  # dead accumulator for fluxNaPatPES/AES
        for lb in (LIS, BATH):
            dJNaP = (omcd[Lz+1].area[comp, lb] * omcd[Lz+1].dLA[NA, HPO4, comp, lb]
                     * (2 * delmu[NA, comp, lb] + delmu[HPO4, comp, lb]))
            Jsol[NA,   comp, lb] += 2 * dJNaP
            Jsol[HPO4, comp, lb] +=     dJNaP
            # sumJES += 2 * dJNaP
        # if comp == P: fluxNaPatPES = sumJES * convert  # dead
        # if comp == A: fluxNaPatAES = sumJES * convert  # dead

    # ------------------------------------------------------------------
    # Exchangers
    # ------------------------------------------------------------------

    # NaH exchanger — Fuster et al. (J Gen Physiol 2009) ping-pong model: NOT USED.
    # Computes dJNaH but never updates Jsol; fluxNaHPES immediately overwritten below.
    # Full model preserved in qflux2OMCv0.py lines 234-258.
    # K = 2  # P compartment
    # sumJES = 0.0
    # for L in range(5, 6 + 1):
    #     affnao = 34.0; affnai = 102.0; affho = 0.0183e-3; affhi = 0.054e-3
    #     fnai = C[NA, P]; hi = C[H, P]; fnao = C[NA, lb]; ho = C[H, lb]
    #     Fno = (fnao/affnao) / (1.0 + fnao/affnao + ho/affho)
    #     Fni = (fnai/affnai) / (1.0 + fnai/affnai + hi/affhi)
    #     Fho = (ho/affho)  / (1.0 + fnao/affnao + ho/affho)
    #     Fhi = (hi/affhi)  / (1.0 + fnai/affnai + hi/affhi)
    #     E2mod1 = (Fni+Fhi) / (Fni+Fhi+Fno+Fho)
    #     E1mod1 = 1.0 - E2mod1
    #     E2mod2 = (Fni**2 + Fhi**2) / (Fni**2 + Fhi**2 + Fno**2 + Fho**2)
    #     E1mod2 = 1.0 - E2mod2
    #     Fmod1 = (hi**2) / (hi**2 + (0.3e-3)**2)
    #     Rnhe = 1.0e3 * (1.0*(1.0-Fmod1)*(E2mod2*Fno**2 - E1mod2*Fni**2)
    #                     + Fmod1*(E2mod1*Fno - E1mod1*Fni))
    #     dJNaH = -1 * omcd[Lz+1].area[P, lb] * omcd[Lz+1].xNHE1[P] * Rnhe
    #     sumJES += dJNaH
    # fluxNaHPES = sumJES * convert  # overwritten by linear NaH block below

    # NaH exchanger (linear) at PE,PS,AE,AS interfaces
    for comp in (P, A):
        # sumJES = 0.0  # dead accumulator for fluxNaHPES/AES
        for lb in (LIS, BATH):
            dJNaH = (omcd[Lz+1].area[comp, lb] * omcd[Lz+1].dLA[NA, H, comp, lb]
                     * (delmu[NA, comp, lb] - delmu[H, comp, lb]))
            Jsol[NA, comp, lb] += dJNaH
            Jsol[H,  comp, lb] -= dJNaH
            # sumJES += dJNaH
        # if comp == P: fluxNaHPES = sumJES * convert  # dead
        # if comp == A: fluxNaHAES = sumJES * convert  # dead

    # Cl/HCO3 exchanger at PE,PS interfaces
    # sumJES = 0.0  # dead accumulator for fluxClHCO3exPES
    for lb in (LIS, BATH):
        dJClHCO3 = (omcd[Lz+1].area[P, lb] * omcd[Lz+1].dLA[CL, HCO3, P, lb]
                    * (delmu[CL, P, lb] - delmu[HCO3, P, lb]))
        Jsol[CL,   P, lb] += dJClHCO3
        Jsol[HCO3, P, lb] -= dJClHCO3
        # sumJES += dJClHCO3
    # fluxClHCO3exPES = sumJES * convert  # dead

    # AE1 exchanger at peritubular membrane of alpha cell (AES, ABS)
    bpp    = C[HCO3, A]
    cpp    = C[CL,   A]
    betapp = bpp / dKbpp
    gampp  = cpp / dKcpp
    # sumJES = 0.0  # dead accumulator for fluxAE1
    for lb in (LIS, BATH):
        bp    = C[HCO3, lb]
        cp    = C[CL,   lb]
        betap = bp / dKbp
        gamp  = cp / dKcp
        xT    = omcd[Lz+1].xAE1 / (1 + bpp / 172.0)
        sumum = (1 + betap + gamp) * (Pbpp * betapp + Pcpp * gampp)
        sumum += (1 + betapp + gampp) * (Pbp * betap + Pcp * gamp)
        befflux = (omcd[Lz+1].area[A, lb] * xT / sumum
                   * (Pbpp * betapp * Pcp * gamp - Pbp * betap * Pcpp * gampp))
        Jsol[HCO3, A, lb] += befflux
        Jsol[CL,   A, lb] -= befflux
        # sumJES -= befflux
    # fluxAE1 = sumJES * convert  # dead

    # ------------------------------------------------------------------
    # ATPases
    # ------------------------------------------------------------------

    # Na-K-ATPase at PE,PS,AE,AS,BE,BS interfaces (B fluxes zeroed in cancel section below)
    fluxNaKase = np.zeros(NC)  # stores (dJact5 + dJact6) * convert per compartment for FNaK
    for comp in (P, A, B):
        AffNa  = 0.2 * (1.0 + C[K, comp] / 8.33)
        actNa  = C[NA, comp] / (C[NA, comp] + AffNa)
        AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
        AffNH4 = AffK / 0.20
        actK5  = C[K, LIS]  / (C[K, LIS]  + AffK)
        actK6  = C[K, BATH] / (C[K, BATH] + AffK)

        dJact5 = omcd[Lz+1].area[comp, LIS]  * omcd[Lz+1].ATPNaK[comp, LIS]  * actNa**3 * actK5**2
        dJact6 = omcd[Lz+1].area[comp, BATH] * omcd[Lz+1].ATPNaK[comp, BATH] * actNa**3 * actK6**2

        ro5 = (C[NH4, LIS]  / AffNH4) / (C[K, LIS]  / AffK)
        ro6 = (C[NH4, BATH] / AffNH4) / (C[K, BATH] / AffK)

        Jsol[NA,  comp, LIS]  += dJact5
        Jsol[NA,  comp, BATH] += dJact6
        Jsol[K,   comp, LIS]  -= 2.0/3.0 * dJact5 / (1 + ro5)
        Jsol[K,   comp, BATH] -= 2.0/3.0 * dJact6 / (1 + ro6)
        Jsol[NH4, comp, LIS]  -= 2.0/3.0 * dJact5 * ro5 / (1 + ro5)
        Jsol[NH4, comp, BATH] -= 2.0/3.0 * dJact6 * ro6 / (1 + ro6)

        fluxNaKase[comp] = (dJact5 + dJact6) * convert

    # H-ATPase at LUM–A (alpha apical) interface
    denom13 = 1.0 + np.exp(steepA * (delmu[H, LUM, A] - dmuATPH))
    dJact13 = -omcd[Lz+1].area[LUM, A] * omcd[Lz+1].ATPH[LUM, A] / denom13
    Jsol[H, LUM, A] += dJact13
    # fluxHATPaseMA = dJact13 * convert  # dead

    # H-K-ATPase at LUM–P interface
    # Note: uses += hefflux (not 2*hefflux) — asymmetry with LUM-A block; preserved from original.
    dkf5, dkb5 = 4.0e1, 2.0e2
    hkconc[0] = C[K, P];  hkconc[1] = C[K, LUM]
    hkconc[2] = C[H, P];  hkconc[3] = C[H, LUM]
    Amat = fatpase(Natp, hkconc)
    # Amat_org = Amat  # dead alias
    # Amat_n   = Amat_org  # dead alias
    if np.linalg.det(Amat) != 0:
        Amat = np.linalg.inv(Amat)
        # Amat_inv = Amat  # dead alias
        hefflux = (omcd[Lz+1].area[LUM, P] * omcd[Lz+1].ATPHK[LUM, P]
                   * (dkf5 * Amat[6, 0] - dkb5 * Amat[7, 0]))
        Jsol[K, LUM, P] += hefflux    # single hefflux at LUM-P (original behaviour)
        Jsol[H, LUM, P] -= hefflux
        # fluxHKATPaseMP = 2 * hefflux * convert  # dead

    # H-K-ATPase at LUM–A interface
    hkconc[0] = C[K, A];  hkconc[1] = C[K, LUM]
    hkconc[2] = C[H, A];  hkconc[3] = C[H, LUM]
    Amat = fatpase(Natp, hkconc)
    # Amat_org = Amat  # dead alias
    # Amat_n   = Amat_org  # dead alias
    if np.linalg.det(Amat) != 0:
        Amat = np.linalg.inv(Amat)
        # Amat_inv = Amat  # dead alias
        hefflux = (omcd[Lz+1].area[LUM, A] * omcd[Lz+1].ATPHK[LUM, A]
                   * (dkf5 * Amat[6, 0] - dkb5 * Amat[7, 0]))
        Jsol[K, LUM, A] += 2 * hefflux   # 2*hefflux at LUM-A (original behaviour)
        Jsol[H, LUM, A] -= 2 * hefflux
        # fluxHKATPaseMA = 2 * hefflux * convert  # dead

    # ------------------------------------------------------------------
    # Cancel all IC-B fluxes (OMCD has no real B cells)
    # ------------------------------------------------------------------
    Jvol[LUM, B]  = 0.0
    Jvol[B, LIS]  = 0.0
    Jvol[B, BATH] = 0.0
    Jsol[:, LUM, B]  = 0.0
    Jsol[:, B, LIS]  = 0.0
    Jsol[:, B, BATH] = 0.0

    # ------------------------------------------------------------------
    # Store local flux values for later integration
    # ------------------------------------------------------------------
    # cv  = href*Cref*np.pi*DiamOMC*60/10*1.0e9    # duplicate of convert
    # cvw = Pfref*Cref*Vwbar*np.pi*DiamOMC*60/10*1.0e6  # unused

    omcd[Lz+1].FNatrans = (Jsol[NA, LUM, P] + Jsol[NA, LUM, A]) * convert * omcd[Lz+1].coalesce
    omcd[Lz+1].FNapara  =  Jsol[NA, LUM, LIS] * convert * omcd[Lz+1].coalesce
    omcd[Lz+1].FNaK     =  fluxNaKase[P:B+1].sum() * omcd[Lz+1].coalesce
    omcd[Lz+1].FKtrans  = (Jsol[K, LUM, P] + Jsol[K, LUM, A]) * convert * omcd[Lz+1].coalesce
    omcd[Lz+1].FKpara   =  Jsol[K, LUM, LIS] * convert * omcd[Lz+1].coalesce

    return Jvol, Jsol
