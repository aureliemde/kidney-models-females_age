"""Flux computation for the CCD (Cortical Collecting Duct) segment.

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
from fatpase import fatpase


# ---------------------------------------------------------------------------
# Module-level initialization: CNT transporter parameters (shared init pattern).
# ---------------------------------------------------------------------------
cnt = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initC_Var import initC_Var
hENaC_CNT, hROMK_CNT, hCltj_CNT, xTRPV5_cnt, xPTRPV4_cnt = initC_Var(cnt)

# ---------------------------------------------------------------------------
# Module-level initialization: CCD transporter parameters.
# ---------------------------------------------------------------------------
ccd = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initCCD_Var import initCCD_Var
hENaC_CCD, hROMK_CCD, hCltj_CCD = initCCD_Var(ccd)


def qflux2CCD(x, ccd, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute volume and solute fluxes in the CCD at axial position Lz+1.

    Args:
        x:          State vector (concentrations, volumes, EPs, pressure)
        ccd:        List of Membrane objects for the CCD (0 to NZ)
        Lz:         Current upstream position index
        PTinitVol:  Initial PT luminal volume
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity

    Returns:
        Jvol: np.ndarray of shape (NC, NC) — volume fluxes
        Jsol: np.ndarray of shape (NS, NC, NC) — solute fluxes
    """
    # LzCCD = Lz  # dead alias; Lz used directly throughout

    _nc = NC - 1  # 5 active compartments: LUM, P, A, B, LIS (BATH is fixed boundary)

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------
    # ONC  = np.zeros(NC)    # unused
    # PRES = np.zeros(NC)    # unused
    C      = np.zeros((NS, NC))
    ph     = np.zeros(NC)
    Vol    = np.zeros(NC)
    EP     = np.zeros(NC)
    hkconc = np.zeros(4)
    # theta = np.zeros(NC)   # unused geometric helpers
    # Slum  = np.zeros(NC)
    # Slat  = np.zeros(NC)
    # Sbas  = np.zeros(NC)
    # lwork = 10000           # unused LAPACK workspace
    # ipiv  = np.zeros(Natp)
    # work  = np.zeros(lwork)

    # ------------------------------------------------------------------
    # Assign concentrations, volumes, and potentials
    # ------------------------------------------------------------------
    _ox  = _nc * NS2
    _oxv = _ox  + _nc
    _oxe = _oxv + _nc

    C[:NS, BATH] = ccd[Lz+1].conc[:NS, BATH]
    EP[BATH]     = ccd[Lz+1].ep[BATH]

    C[:NS2, :_nc] = x[:_ox].reshape(NS2, _nc)
    ph[:_nc]       = -np.log10(C[H, :_nc] / 1.0e3)
    Vol[:_nc]      = x[_ox:_oxv]
    EP[:_nc]       = x[_oxv:_oxe]
    PM             = x[_oxe]

    # ------------------------------------------------------------------
    # Update LIS–BATH surface area
    # ------------------------------------------------------------------
    ccd[Lz+1].area[LIS, BATH] = ccd[Lz+1].sbasEinit * max(Vol[LIS] / ccd[Lz+1].volEinit, 1.0)
    ccd[Lz+1].area[BATH, LIS] = ccd[Lz+1].area[LIS, BATH]

    # ------------------------------------------------------------------
    # Initialize fluxes
    # ------------------------------------------------------------------
    Jvol = np.zeros((NC, NC))
    Jsol = np.zeros((NS, NC, NC))

    # ------------------------------------------------------------------
    # Water fluxes
    # ------------------------------------------------------------------
    Jvol = compute_water_fluxes(C, PM, 0, Vol, ccd[Lz+1].volLuminit,
                                ccd[Lz+1].volEinit, ccd[Lz+1].volPinit, CPimprefCCD,
                                ccd[Lz+1].volAinit, CAimprefCCD, ccd[Lz+1].volBinit,
                                CBimprefCCD, ccd[Lz+1].area, ccd[Lz+1].sig, ccd[Lz+1].dLPV,
                                complCCD, PTinitVol)

    # ------------------------------------------------------------------
    # Solute fluxes
    # ------------------------------------------------------------------
    convert = href * Cref * np.pi * DiamCCD * 60 / 10 * 1.0e9  # mmol/s/cm² → pmol/min/mm
    eps = 1.0e-6  # guard for Peclet terms

    # pH-dependent modulation of ENaC, ROMK, and apical paracellular Cl permeability
    facphMP = 1.0 * (0.1 + 2 / (1 + np.exp(-6.0 * (ph[P] - 7.50))))
    facphTJ = 2.0 / (1.0 + np.exp(10.0 * (ph[LIS] - 7.32)))
    facNaMP = (30 / (30 + C[NA, LUM])) * (50 / (50 + C[NA, P]))

    ccd[Lz+1].h[NA, LUM, P]   = hENaC_CCD * facNaMP * facphMP
    ccd[Lz+1].h[K,  LUM, P]   = hROMK_CCD * facphMP
    ccd[Lz+1].h[CL, LUM, LIS] = hCltj_CCD * facphTJ

    # Electrochemical potentials (vectorized)
    dmu = RT * np.log(np.abs(C)) + zval[:, np.newaxis] * (F * EPref) * EP[np.newaxis, :]

    # Driving forces for coupled transporter fluxes (vectorized)
    # delmu[i, k, l] = dmu[i, k] - dmu[i, l]
    delmu = dmu[:, :, np.newaxis] - dmu[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # Electro-convective-diffusive fluxes
    # ------------------------------------------------------------------
    for si in range(NS):
        for c1 in range(_nc):
            for c2 in range(c1 + 1, NC):
                XI   = zval[si] * F * EPref / RT * (EP[c1] - EP[c2])
                dint = np.exp(-XI)
                if abs(1.0 - dint) < eps:
                    Jsol[si, c1, c2] = (ccd[Lz+1].area[c1, c2] * ccd[Lz+1].h[si, c1, c2]
                                        * (C[si, c1] - C[si, c2]))
                else:
                    Jsol[si, c1, c2] = (ccd[Lz+1].area[c1, c2] * ccd[Lz+1].h[si, c1, c2]
                                        * XI * (C[si, c1] - C[si, c2] * dint) / (1.0 - dint))

                concdiff = C[si, c1] - C[si, c2]
                if abs(concdiff) > eps:
                    concmean = concdiff / np.log(abs(C[si, c1] / C[si, c2]))
                    dimless  = (Pfref * Vwbar * Cref) / href
                    Jsol[si, c1, c2] += ((1.0 - ccd[Lz+1].sig[si, c1, c2])
                                         * concmean * Jvol[c1, c2] * dimless)

    # Dimensional ECD fluxes — all commented out as unused:
    # fluxENaC     = Jsol[NA,    LUM, P]   * convert
    # fluxROMK     = Jsol[K,     LUM, P]   * convert
    # fluxKchPES   = (Jsol[K,    P,   LIS] + Jsol[K,    P,   BATH]) * convert
    # fluxKchAES   = (Jsol[K,    A,   LIS] + Jsol[K,    A,   BATH]) * convert
    # fluxKchBES   = (Jsol[K,    B,   LIS] + Jsol[K,    B,   BATH]) * convert
    # fluxClchPES  = (Jsol[CL,   P,   LIS] + Jsol[CL,   P,   BATH]) * convert
    # fluxClchAES  = (Jsol[CL,   A,   LIS] + Jsol[CL,   A,   BATH]) * convert
    # fluxClchBES  = (Jsol[CL,   B,   LIS] + Jsol[CL,   B,   BATH]) * convert
    # fluxBichPES  = (Jsol[HCO3, P,   LIS] + Jsol[HCO3, P,   BATH]) * convert
    # fluxBichAES  = (Jsol[HCO3, A,   LIS] + Jsol[HCO3, A,   BATH]) * convert
    # fluxBichBES  = (Jsol[HCO3, B,   LIS] + Jsol[HCO3, B,   BATH]) * convert
    # fluxH2CO3MP  = Jsol[H2CO3, LUM, P]   * convert
    # fluxH2CO3MA  = Jsol[H2CO3, LUM, A]   * convert
    # fluxH2CO3MB  = Jsol[H2CO3, LUM, B]   * convert
    # fluxCO2MP    = Jsol[CO2,   LUM, P]   * convert
    # fluxCO2MA    = Jsol[CO2,   LUM, A]   * convert
    # fluxCO2MB    = Jsol[CO2,   LUM, B]   * convert
    # fluxH2CO3PES = (Jsol[H2CO3, P, LIS] + Jsol[H2CO3, P, BATH]) * convert
    # fluxH2CO3AES = (Jsol[H2CO3, A, LIS] + Jsol[H2CO3, A, BATH]) * convert
    # fluxH2CO3BES = (Jsol[H2CO3, B, LIS] + Jsol[H2CO3, B, BATH]) * convert
    # fluxCO2PES   = (Jsol[CO2,   P, LIS] + Jsol[CO2,   P, BATH]) * convert
    # fluxCO2AES   = (Jsol[CO2,   A, LIS] + Jsol[CO2,   A, BATH]) * convert
    # fluxCO2BES   = (Jsol[CO2,   B, LIS] + Jsol[CO2,   B, BATH]) * convert
    # fluxHP2mchPES = (Jsol[HPO4,  P, LIS] + Jsol[HPO4,  P, BATH]) * convert
    # fluxHP2mchAES = (Jsol[HPO4,  A, LIS] + Jsol[HPO4,  A, BATH]) * convert
    # fluxHP2mchBES = (Jsol[HPO4,  B, LIS] + Jsol[HPO4,  B, BATH]) * convert
    # fluxHPmPES    = (Jsol[H2PO4, P, LIS] + Jsol[H2PO4, P, BATH]) * convert
    # fluxHPmAES    = (Jsol[H2PO4, A, LIS] + Jsol[H2PO4, A, BATH]) * convert
    # fluxHPmBES    = (Jsol[H2PO4, B, LIS] + Jsol[H2PO4, B, BATH]) * convert

    # ------------------------------------------------------------------
    # Cotransporters
    # ------------------------------------------------------------------

    # Na2HPO4 cotransporter at PE,PS,AE,AS,BE,BS interfaces
    for comp in (P, A, B):
        # sumJES = 0.0  # dead accumulator for fluxNaPatPES/AES/BES
        for lb in (LIS, BATH):
            dJNaP = (ccd[Lz+1].area[comp, lb] * ccd[Lz+1].dLA[NA, HPO4, comp, lb]
                     * (2 * delmu[NA, comp, lb] + delmu[HPO4, comp, lb]))
            Jsol[NA,   comp, lb] += 2 * dJNaP
            Jsol[HPO4, comp, lb] +=     dJNaP
            # sumJES += 2 * dJNaP
        # if comp == P: fluxNaPatPES = sumJES * convert  # dead
        # if comp == A: fluxNaPatAES = sumJES * convert  # dead
        # if comp == B: fluxNaPatBES = sumJES * convert  # dead

    # AE4 = NaHCO3 cotransporter at BE,BS interfaces
    # Jsol never updated in original — entire block dead; kept for reference:
    # for lb in (LIS, BATH):
    #     dJNaBic = (ccd[Lz+1].area[B, lb] * ccd[Lz+1].dLA[NA, HCO3, B, lb]
    #                * (delmu[NA, B, lb] + nstoch * delmu[HCO3, B, lb]))
    #     # sumJES += dJNaBic  # no Jsol update in original
    # fluxAE4BES = sumJES * convert  # dead

    # ------------------------------------------------------------------
    # Exchangers
    # ------------------------------------------------------------------

    # NaH exchanger (NHE1) at PE,PS,AE,AS,BE,BS interfaces
    for comp in (P, A, B):
        # sumJES = 0.0  # dead accumulator for fluxNaHPES/AES/BES
        for lb in (LIS, BATH):
            dJNaH = (ccd[Lz+1].area[comp, lb] * ccd[Lz+1].dLA[NA, H, comp, lb]
                     * (delmu[NA, comp, lb] - delmu[H, comp, lb]))
            Jsol[NA, comp, lb] += dJNaH
            Jsol[H,  comp, lb] -= dJNaH
            # sumJES += dJNaH
        # if comp == P: fluxNaHPES = sumJES * convert  # dead
        # if comp == A: fluxNaHAES = sumJES * convert  # dead
        # if comp == B: fluxNaHBES = sumJES * convert  # dead

    # Cl/HCO3 exchanger at PE,PS interfaces
    # sumJES = 0.0  # dead accumulator for fluxClHCO3exPES
    for lb in (LIS, BATH):
        dJClHCO3 = (ccd[Lz+1].area[P, lb] * ccd[Lz+1].dLA[CL, HCO3, P, lb]
                    * (delmu[CL, P, lb] - delmu[HCO3, P, lb]))
        Jsol[CL,   P, lb] += dJClHCO3
        Jsol[HCO3, P, lb] -= dJClHCO3
        # sumJES += dJClHCO3
    # fluxClHCO3exPES = sumJES * convert  # dead

    # Cl/HCO3 exchanger at BE,BS interfaces (labelled REMOVED in original — dLA may be zero)
    # sumJES = 0.0  # dead accumulator for fluxClHCO3exBES
    for lb in (LIS, BATH):
        dJClHCO3 = (ccd[Lz+1].area[B, lb] * ccd[Lz+1].dLA[CL, HCO3, B, lb]
                    * (delmu[CL, B, lb] - delmu[HCO3, B, lb]))
        Jsol[CL,   B, lb] += dJClHCO3
        Jsol[HCO3, B, lb] -= dJClHCO3
        # sumJES += dJClHCO3
    # fluxClHCO3exBES = sumJES * convert  # dead

    # NDBCE exchanger at LUM–B interface (ping-pong model)
    alpe = C[NA,   LUM] / dKnaeBCE
    alpi = C[NA,   B]   / dKnaiBCE
    bete = C[HCO3, LUM] / dKbieBCE
    beti = C[HCO3, B]   / dKbiiBCE
    game = C[CL,   LUM] / dKcleBCE
    gami = C[CL,   B]   / dKcliBCE

    dele  = 1.0 + alpe * (1.0 + bete + bete**2) + game
    deli  = 1.0 + alpi * (1.0 + beti + beti**2) + gami
    etae  = game + alpe * (bete**2)
    etai  = gami + alpi * (beti**2)
    sigma = dele * etai + deli * etae

    xe = etai / sigma
    xi = etae / sigma
    dJNDBCE4 = (ccd[Lz+1].xNDBCE * ccd[Lz+1].area[LUM, B] * pBCE
                * (alpe * (bete**2) * xe - alpi * (beti**2) * xi))

    Jsol[NA,   LUM, B] +=     dJNDBCE4
    Jsol[CL,   LUM, B] -=     dJNDBCE4
    Jsol[HCO3, LUM, B] += 2 * dJNDBCE4
    # fluxNDBCEatMB = dJNDBCE4 * convert  # dead

    # Pendrin exchanger at the apical membrane of the beta cell (ping-pong model)
    # Old linear driving-force term (dead; replaced by ping-pong model below):
    # dJClHCO3 = ccd[Lz+1].dLA[CL, HCO3, LUM, B] * (delmu[CL, LUM, B] - delmu[HCO3, LUM, B])

    Pbiiclepd = Pbieclepd * Pcliclepd  # internal = external affinity assumption
    Pohiclepd = Poheclepd * Pcliclepd

    ohe = 1.0e-11 / C[H, LUM]   # [H+] in mM → [OH-] in M
    ohi = 1.0e-11 / C[H, B]
    alpe = ohe / dKohpd
    alpi = ohi / dKohpd
    game = C[CL,   LUM] / dKclpd
    gami = C[CL,   B]   / dKclpd
    bete = C[HCO3, LUM] / dKbipd
    beti = C[HCO3, B]   / dKbipd

    dele  = 1.0 + game + bete + alpe
    deli  = 1.0 + gami + beti + alpi
    etae  = 1.0 * game + Pbieclepd * bete + Poheclepd * alpe
    etai  = Pcliclepd * gami + Pbiiclepd * beti + Pohiclepd * alpi
    sigma = dele * etai + deli * etae

    # Stimulation by low pHi
    if ph[B] < 7.9:
        adj = np.exp((1 / 6.4189) * np.log((7.9 - ph[B]) / 0.0115))
    else:
        adj = 1.0

    factpd = ccd[Lz+1].area[LUM, B] * ccd[Lz+1].xPendrin * Pclepd * adj
    dJclpd = factpd * (1.0 * game * etai - Pcliclepd * gami * etae) / sigma
    dJbipd = factpd * (Pbieclepd * bete * etai - Pbiiclepd * beti * etae) / sigma
    # dJohpd = factpd * (Poheclepd * alpe * etai - Pohiclepd * alpi * etae) / sigma  # computed but not added to Jsol

    Jsol[CL,   LUM, B] += dJclpd
    Jsol[HCO3, LUM, B] += dJbipd
    # fluxPdMB = dJclpd * convert  # dead

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
        xT    = ccd[Lz+1].xAE1 / (1 + bpp / 172.0)
        sumum = (1 + betap + gamp) * (Pbpp * betapp + Pcpp * gampp)
        sumum = sumum + (1 + betapp + gampp) * (Pbp * betap + Pcp * gamp)
        befflux = (ccd[Lz+1].area[A, lb] * xT / sumum
                   * (Pbpp * betapp * Pcp * gamp - Pbp * betap * Pcpp * gampp))
        Jsol[HCO3, A, lb] += befflux
        Jsol[CL,   A, lb] -= befflux
        # sumJES -= befflux
    # fluxAE1 = sumJES * convert  # dead

    # ------------------------------------------------------------------
    # ATPases
    # ------------------------------------------------------------------

    # Na-K-ATPase at PE,PS,AE,AS,BE,BS interfaces
    fluxNaKase = np.zeros(NC)  # stores (dJact5 + dJact6) * convert per compartment for FNaK
    for comp in (P, A, B):
        AffNa  = 0.2 * (1.0 + C[K, comp] / 8.33)
        actNa  = C[NA, comp] / (C[NA, comp] + AffNa)
        AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
        AffNH4 = AffK / 0.20
        actK5  = C[K, LIS]  / (C[K, LIS]  + AffK)
        actK6  = C[K, BATH] / (C[K, BATH] + AffK)

        dJact5 = ccd[Lz+1].area[comp, LIS]  * ccd[Lz+1].ATPNaK[comp, LIS]  * actNa**3 * actK5**2
        dJact6 = ccd[Lz+1].area[comp, BATH] * ccd[Lz+1].ATPNaK[comp, BATH] * actNa**3 * actK6**2

        ro5 = (C[NH4, LIS]  / AffNH4) / (C[K, LIS]  / AffK)
        ro6 = (C[NH4, BATH] / AffNH4) / (C[K, BATH] / AffK)

        Jsol[NA,  comp, LIS]  += dJact5
        Jsol[NA,  comp, BATH] += dJact6
        Jsol[K,   comp, LIS]  -= 2.0/3.0 * dJact5 / (1 + ro5)
        Jsol[K,   comp, BATH] -= 2.0/3.0 * dJact6 / (1 + ro6)
        Jsol[NH4, comp, LIS]  -= 2.0/3.0 * dJact5 * ro5 / (1 + ro5)
        Jsol[NH4, comp, BATH] -= 2.0/3.0 * dJact6 * ro6 / (1 + ro6)

        fluxNaKase[comp] = (dJact5 + dJact6) * convert

    # H-ATPase at LUM–A (alpha apical) and B–LIS, B–BATH (beta basolateral) interfaces
    denom13 = 1.0 + np.exp( steepA * (dmu[H, LUM] - dmu[H, A]    - dmuATPH))
    denom45 = 1.0 + np.exp(-steepB * (dmu[H, B]   - dmu[H, LIS]  - dmuATPH))
    denom46 = 1.0 + np.exp(-steepB * (dmu[H, B]   - dmu[H, BATH] - dmuATPH))

    dJact13 = -ccd[Lz+1].area[LUM, A] * ccd[Lz+1].ATPH[LUM, A] / denom13
    dJact45 =  ccd[Lz+1].area[B, LIS]  * ccd[Lz+1].ATPH[B, LIS]  / denom45
    dJact46 =  ccd[Lz+1].area[B, BATH] * ccd[Lz+1].ATPH[B, BATH] / denom46

    Jsol[H, LUM, A]  += dJact13
    Jsol[H, B,   LIS]  += dJact45
    Jsol[H, B,   BATH] += dJact46
    # fluxHATPaseMA  = dJact13 * convert            # dead
    # fluxHATPaseBES = (dJact45 + dJact46) * convert  # dead

    # H-K-ATPase at LUM–P, LUM–A, LUM–B interfaces
    dkf5, dkb5 = 4.0e1, 2.0e2
    for comp in (P, A, B):
        hkconc[0] = C[K, comp];  hkconc[1] = C[K, LUM]
        hkconc[2] = C[H, comp];  hkconc[3] = C[H, LUM]
        Amat = fatpase(Natp, hkconc)
        # Amat_org = Amat  # dead alias
        # Amat_n   = Amat_org  # dead alias
        if np.linalg.det(Amat) != 0:
            Amat = np.linalg.inv(Amat)
            # Amat_inv = Amat  # dead alias
            hefflux = (ccd[Lz+1].area[LUM, comp] * ccd[Lz+1].ATPHK[LUM, comp]
                       * (dkf5 * Amat[6, 0] - dkb5 * Amat[7, 0]))
            Jsol[K, LUM, comp] += 2 * hefflux
            Jsol[H, LUM, comp] -= 2 * hefflux
            # if comp == P: fluxHKATPaseMP = 2 * hefflux * convert  # dead
            # if comp == A: fluxHKATPaseMA = 2 * hefflux * convert  # dead
            # if comp == B: fluxHKATPaseMB = 2 * hefflux * convert  # dead

    # ------------------------------------------------------------------
    # Store local flux values for later integration
    # ------------------------------------------------------------------
    # cv  = href*Cref*np.pi*DiamCCD*60/10*1.0e9    # duplicate of convert
    # cvw = Pfref*Cref*Vwbar*np.pi*DiamCCD*60/10*1.0e6  # unused

    ccd[Lz+1].FNatrans = (Jsol[NA, LUM, P] + Jsol[NA, LUM, A] + Jsol[NA, LUM, B]) * convert * ccd[Lz+1].coalesce
    ccd[Lz+1].FNapara  =  Jsol[NA, LUM, LIS] * convert * ccd[Lz+1].coalesce
    ccd[Lz+1].FNaK     =  fluxNaKase[P:B+1].sum() * ccd[Lz+1].coalesce
    ccd[Lz+1].FKtrans  = (Jsol[K, LUM, P] + Jsol[K, LUM, A] + Jsol[K, LUM, B]) * convert * ccd[Lz+1].coalesce
    ccd[Lz+1].FKpara   =  Jsol[K, LUM, LIS] * convert * ccd[Lz+1].coalesce

    return Jvol, Jsol
