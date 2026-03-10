"""Flux computation for the CNT (Connecting Tubule) segment.

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
from compute_ncx_fluxes import compute_ncx_fluxes
from fatpase import fatpase


# ---------------------------------------------------------------------------
# Module-level initialization: CNT transporter parameters.
# initC_Var is called once at import time; results reused at every Newton step.
# ---------------------------------------------------------------------------
cnt = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initC_Var import initC_Var
hENaC_CNT, hROMK_CNT, hCltj_CNT, xTRPV5_cnt, xPTRPV4_cnt = initC_Var(cnt)


def qflux2C(x, cnt, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute volume and solute fluxes in the CNT at axial position Lz+1.

    Args:
        x:          State vector (concentrations, volumes, EPs, pressure)
        cnt:        List of Membrane objects for the CNT (0 to NZ)
        Lz:         Current upstream position index
        PTinitVol:  Initial PT luminal volume
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity

    Returns:
        Jvol: np.ndarray of shape (NC, NC) — volume fluxes
        Jsol: np.ndarray of shape (NS, NC, NC) — solute fluxes
    """
    # LzC = Lz  # unused alias; kept for reference

    _nc = NC - 1  # 5 active compartments: LUM, P, A, B, LIS

    # x-vector offsets (match qnewton2icb packing)
    _ox  = _nc * NS2
    _oxv = _ox  + _nc
    _oxe = _oxv + _nc

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------
    # ONC  = np.zeros(NC)       # unused
    # PRES = np.zeros(NC)       # unused
    C     = np.zeros((NS, NC))
    # dmu = np.zeros((NS, NC))  # unused; delmu comes from compute_ecd_fluxes
    ph    = np.zeros(NC)
    Vol   = np.zeros(NC)
    EP    = np.zeros(NC)
    delmu = np.zeros((NS, NC, NC))
    hkconc  = np.zeros(4)
    # theta = np.zeros(NC)      # unused
    # Slum  = np.zeros(NC)      # unused
    # Slat  = np.zeros(NC)      # unused
    # Sbas  = np.zeros(NC)      # unused
    # lwork = 10000             # unused LAPACK remnant
    # ipiv  = np.zeros(Natp)   # unused LAPACK remnant
    # work  = np.zeros(lwork)  # unused LAPACK remnant
    var_ncx = np.zeros(16)

    # ------------------------------------------------------------------
    # Assign concentrations, volumes, and potentials
    # ------------------------------------------------------------------
    C[:NS, BATH] = cnt[Lz+1].conc[:NS, BATH]
    EP[BATH]     = cnt[Lz+1].ep[BATH]

    C[:NS2, :_nc] = x[:_ox].reshape(NS2, _nc)
    ph[:_nc]      = -np.log10(C[H, :_nc] / 1.0e3)
    Vol[:_nc]     = x[_ox:_oxv]
    EP[:_nc]      = x[_oxv:_oxe]
    PM            = x[_oxe]

    # ------------------------------------------------------------------
    # Update LIS–BATH surface area
    # ------------------------------------------------------------------
    cnt[Lz+1].area[LIS, BATH] = cnt[Lz+1].sbasEinit * max(Vol[LIS] / cnt[Lz+1].volEinit, 1.0)
    cnt[Lz+1].area[BATH, LIS] = cnt[Lz+1].area[LIS, BATH]

    # ------------------------------------------------------------------
    # Initialize fluxes
    # ------------------------------------------------------------------
    Jvol = np.zeros((NC, NC))
    Jsol = np.zeros((NS, NC, NC))

    # Conversion factor: nondimensional → pmol/min/mm tubule
    convert = href * Cref * np.pi * DiamC * 60 / 10 * 1.0e9

    # ------------------------------------------------------------------
    # Water fluxes
    # ------------------------------------------------------------------
    Jvol = compute_water_fluxes(C, PM, 0, Vol, cnt[Lz+1].volLuminit,
                                cnt[Lz+1].volEinit,  cnt[Lz+1].volPinit, CPimprefC,
                                cnt[Lz+1].volAinit,  CAimprefC,
                                cnt[Lz+1].volBinit,  CBimprefC,
                                cnt[Lz+1].area, cnt[Lz+1].sig,
                                cnt[Lz+1].dLPV, complC, PTinitVol)

    # ------------------------------------------------------------------
    # Electro-convective-diffusive (ECD) fluxes
    # ------------------------------------------------------------------
    # pH-dependent permeabilities: ENaC, ROMK, apical TJ Cl
    facphMP = 1.0 * (0.1 + 2 / (1 + np.exp(-6.0 * (ph[P] - 7.50))))
    facphTJ = 2.0 / (1.0 + np.exp(10.0 * (ph[LIS] - 7.32)))
    facNaMP = (30 / (30 + C[NA, LUM])) * (50 / (50 + C[NA, P]))

    cnt[Lz+1].h[NA, LUM, P]   = hENaC_CNT * facNaMP * facphMP
    cnt[Lz+1].h[K,  LUM, P]   = hROMK_CNT * facphMP
    cnt[Lz+1].h[CL, LUM, LIS] = hCltj_CNT * facphTJ

    Jsol, delmu = compute_ecd_fluxes(C, EP, cnt[Lz+1].area, cnt[Lz+1].sig,
                                     cnt[Lz+1].h, Jvol)

    # Dimensional ECD fluxes (print-out only; not stored):
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
    # fluxH2CO3PES = (Jsol[H2CO3,P,   LIS] + Jsol[H2CO3,P,   BATH]) * convert
    # fluxH2CO3AES = (Jsol[H2CO3,A,   LIS] + Jsol[H2CO3,A,   BATH]) * convert
    # fluxH2CO3BES = (Jsol[H2CO3,B,   LIS] + Jsol[H2CO3,B,   BATH]) * convert
    # fluxCO2PES   = (Jsol[CO2,  P,   LIS] + Jsol[CO2,  P,   BATH]) * convert
    # fluxCO2AES   = (Jsol[CO2,  A,   LIS] + Jsol[CO2,  A,   BATH]) * convert
    # fluxCO2BES   = (Jsol[CO2,  B,   LIS] + Jsol[CO2,  B,   BATH]) * convert
    # fluxHP2mchPES= (Jsol[HPO4, P,   LIS] + Jsol[HPO4, P,   BATH]) * convert
    # fluxHP2mchAES= (Jsol[HPO4, A,   LIS] + Jsol[HPO4, A,   BATH]) * convert
    # fluxHP2mchBES= (Jsol[HPO4, B,   LIS] + Jsol[HPO4, B,   BATH]) * convert
    # fluxHPmPES   = (Jsol[H2PO4,P,   LIS] + Jsol[H2PO4,P,   BATH]) * convert
    # fluxHPmAES   = (Jsol[H2PO4,A,   LIS] + Jsol[H2PO4,A,   BATH]) * convert
    # fluxHPmBES   = (Jsol[H2PO4,B,   LIS] + Jsol[H2PO4,B,   BATH]) * convert

    # ------------------------------------------------------------------
    # Na2HPO4 cotransporter at basolateral membranes (P, A, B × LIS, BATH)
    # ------------------------------------------------------------------
    for comp in (P, A, B):
        # sumJES = 0.0  # for dimensional flux; unused
        for lb in (LIS, BATH):
            dJNaP = (cnt[Lz+1].area[comp, lb] * cnt[Lz+1].dLA[NA, HPO4, comp, lb]
                     * (2*delmu[NA, comp, lb] + delmu[HPO4, comp, lb]))
            Jsol[NA,   comp, lb] += 2 * dJNaP
            Jsol[HPO4, comp, lb] += dJNaP
            # sumJES += 2 * dJNaP
        # fluxNaP[comp] = sumJES * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # NaH exchanger at basolateral membranes (P, A, B × LIS, BATH)
    # ------------------------------------------------------------------
    for comp in (P, A, B):
        # sumJES = 0.0  # for dimensional flux; unused
        for lb in (LIS, BATH):
            dJNaH = (cnt[Lz+1].area[comp, lb] * cnt[Lz+1].dLA[NA, H, comp, lb]
                     * (delmu[NA, comp, lb] - delmu[H, comp, lb]))
            Jsol[NA, comp, lb] += dJNaH
            Jsol[H,  comp, lb] -= dJNaH
            # sumJES += dJNaH
        # fluxNaH[comp] = sumJES * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # Cl/HCO3 exchanger at P basolateral (PE, PS)
    # ------------------------------------------------------------------
    # sumJES = 0.0  # for dimensional flux; unused
    for lb in (LIS, BATH):
        dJClHCO3 = (cnt[Lz+1].area[P, lb] * cnt[Lz+1].dLA[CL, HCO3, P, lb]
                    * (delmu[CL, P, lb] - delmu[HCO3, P, lb]))
        Jsol[CL,   P, lb] += dJClHCO3
        Jsol[HCO3, P, lb] -= dJClHCO3
        # sumJES += dJClHCO3
    # fluxClHCO3exPES = sumJES * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # Cl/HCO3 exchanger at B basolateral (BE, BS) — labelled REMOVED in original
    # ------------------------------------------------------------------
    # sumJES = 0.0  # for dimensional flux; unused
    for lb in (LIS, BATH):
        dJClHCO3 = (cnt[Lz+1].area[B, lb] * cnt[Lz+1].dLA[CL, HCO3, B, lb]
                    * (delmu[CL, B, lb] - delmu[HCO3, B, lb]))
        Jsol[CL,   B, lb] += dJClHCO3
        Jsol[HCO3, B, lb] -= dJClHCO3
        # sumJES += dJClHCO3
    # fluxClHCO3exBES = sumJES * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # Pendrin: Cl/HCO3/OH exchanger at apical membrane of B (beta) cell
    # ------------------------------------------------------------------
    # dJClHCO3 = cnt[Lz+1].dLA[CL, HCO3, LUM, B] * (delmu[CL, LUM, B] - delmu[HCO3, LUM, B])
    # unused — never added to Jsol; kept for reference

    Pbiiclepd = Pbieclepd * Pcliclepd   # internal = external affinity assumption
    Pohiclepd = Poheclepd * Pcliclepd

    ohe  = 1.0e-11 / C[H, LUM]   # OH- in lumen (H+ in mM, OH- in M)
    ohi  = 1.0e-11 / C[H, B]     # OH- in B cell
    alpe = ohe  / dKohpd
    alpi = ohi  / dKohpd
    game = C[CL,   LUM] / dKclpd
    gami = C[CL,   B]   / dKclpd
    bete = C[HCO3, LUM] / dKbipd
    beti = C[HCO3, B]   / dKbipd

    dele  = 1.0 + game + bete + alpe
    deli  = 1.0 + gami + beti + alpi
    etae  = 1.0 * game + Pbieclepd * bete + Poheclepd * alpe
    etai  = Pcliclepd * gami + Pbiiclepd * beti + Pohiclepd * alpi
    sigma = dele * etai + deli * etae

    # Stimulation by low intracellular pH
    if ph[B] < 7.9:
        adj = np.exp((1 / 6.4189) * np.log((7.9 - ph[B]) / 0.0115))
    else:
        adj = 1.0

    factpd = cnt[Lz+1].area[LUM, B] * cnt[Lz+1].xPendrin * Pclepd * adj

    dJclpd = factpd * (1.0 * game * etai - Pcliclepd * gami * etae) / sigma
    dJbipd = factpd * (Pbieclepd * bete * etai - Pbiiclepd * beti * etae) / sigma
    dJohpd = factpd * (Poheclepd * alpe * etai - Pohiclepd * alpi * etae) / sigma

    Jsol[CL,   LUM, B] += dJclpd
    Jsol[HCO3, LUM, B] += dJbipd
    # fluxPdMB = dJclpd * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # AE1: Cl/HCO3 exchanger at basolateral membrane of A (alpha) cell
    # b = HCO3, c = Cl; pp = internal (A cell), p = external (LIS/BATH)
    # ------------------------------------------------------------------
    bpp   = C[HCO3, A]
    cpp   = C[CL,   A]
    betapp = bpp / dKbpp
    gampp  = cpp / dKcpp

    # sumJES = 0.0  # for dimensional flux; unused
    for lb in (LIS, BATH):
        bp    = C[HCO3, lb]
        cp    = C[CL,   lb]
        betap = bp / dKbp
        gamp  = cp / dKcp
        xT    = cnt[Lz+1].xAE1 / (1 + bpp / 172.0)
        sumum = ((1 + betap  + gamp)  * (Pbpp * betapp + Pcpp * gampp)
               + (1 + betapp + gampp) * (Pbp  * betap  + Pcp  * gamp))
        befflux = (cnt[Lz+1].area[A, lb] * xT / sumum
                   * (Pbpp * betapp * Pcp * gamp - Pbp * betap * Pcpp * gampp))
        Jsol[HCO3, A, lb] += befflux
        Jsol[CL,   A, lb] -= befflux
        # sumJES -= befflux
    # fluxAE1 = sumJES * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # NCX: Na/Ca exchanger at basolateral membrane of P cell
    # ------------------------------------------------------------------
    var_ncx[0]  = C[NA, P]
    var_ncx[1]  = C[NA, LIS]
    var_ncx[2]  = C[NA, BATH]
    var_ncx[3]  = C[CA, P]
    var_ncx[4]  = C[CA, LIS]
    var_ncx[5]  = C[CA, BATH]
    var_ncx[6]  = EP[P]
    var_ncx[7]  = EP[LIS]
    var_ncx[8]  = EP[BATH]
    var_ncx[9]  = cnt[Lz+1].area[P, LIS]
    var_ncx[10] = cnt[Lz+1].area[P, BATH]
    var_ncx[11] = cnt[Lz+1].xNCX

    dJNCXca5, dJNCXca6 = compute_ncx_fluxes(var_ncx)

    Jsol[NA, P, LIS]  -= 3.0 * dJNCXca5
    Jsol[NA, P, BATH] -= 3.0 * dJNCXca6
    Jsol[CA, P, LIS]  += dJNCXca5
    Jsol[CA, P, BATH] += dJNCXca6
    fluxNCXca = (dJNCXca5 + dJNCXca6) * convert

    # ------------------------------------------------------------------
    # TRPV5: Ca2+ channel at apical membrane of P cell
    # ------------------------------------------------------------------
    k1v5 = 42.7
    k3v5 = 0.1684 * np.exp(0.6035 * ph[P])
    k4v5 = 58.7

    if ph[P] < 7.4:
        k2v5 = 55.9 + (173.3 - 55.9) / (7.0 - 7.4) * (ph[P] - 7.4)
    else:
        k2v5 = 55.9 + (30.4 - 55.9) / (8.4 - 7.4) * (ph[P] - 7.4)

    psv5 = 1.0 / (1.0 + k3v5/k4v5 + k2v5/k1v5)
    pcv5 = k2v5 / k1v5 * psv5
    pfv5 = k3v5 / k4v5 * psv5

    gfv5 = 59 + 59 / (59 + 29.0) * (91.0 - 58.0) / (7.4 - 5.4) * (ph[LUM] - 7.4)
    gsv5 = 29 + 29 / (59 + 29.0) * (91.0 - 58.0) / (7.4 - 5.4) * (ph[LUM] - 7.4)
    gv5  = (gfv5 * pfv5 + gsv5 * psv5) * 1.0e-12

    ECa       = RT / (2*F) * np.log(C[CA, P] / C[CA, LUM])   # Nernst potential [V]
    dfv5      = (EP[LUM] - EP[P]) * EPref - ECa
    finhib_v5 = 1.0 / (1.0 + C[CA, P] / Cinhib_v5)

    dJTRPV5 = cnt[Lz+1].area[LUM, P] * xTRPV5_cnt * finhib_v5 * gv5 * dfv5 / (2*F)
    Jsol[CA, LUM, P] += dJTRPV5
    fluxTRPV5 = dJTRPV5 * convert

    # ------------------------------------------------------------------
    # TRPV4: Ca2+ channel at apical membrane of P cell
    # ------------------------------------------------------------------
    flow_TRPV4 = 1.0

    XICa  = zval[CA] * F * EPref / RT * (EP[LUM] - EP[P])
    dint  = np.exp(-XICa)
    if np.abs(1.0 - dint) < 1.0e-6:
        Df_TRPV4 = xPTRPV4_cnt * (C[CA, LUM] - C[CA, P])
    else:
        Df_TRPV4 = xPTRPV4_cnt * XICa * (C[CA, LUM] - C[CA, P] * dint) / (1.0 - dint)

    dJTRPV4 = flow_TRPV4 * cnt[Lz+1].area[LUM, P] * Df_TRPV4
    Jsol[CA, LUM, P] += dJTRPV4
    fluxTRPV4 = dJTRPV4 * convert

    # ------------------------------------------------------------------
    # Na-K-ATPase at basolateral membranes (P, A, B × LIS, BATH)
    # ------------------------------------------------------------------
    fluxNaKase = np.zeros(NC)
    for comp in (P, A, B):
        AffNa  = 0.2 * (1.0 + C[K, comp] / 8.33)
        actNa  = C[NA, comp] / (C[NA, comp] + AffNa)
        AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
        AffNH4 = AffK / 0.20

        actK_LIS  = C[K, LIS]  / (C[K, LIS]  + AffK)
        actK_BATH = C[K, BATH] / (C[K, BATH] + AffK)

        dJact_LIS  = (cnt[Lz+1].area[comp, LIS]  * cnt[Lz+1].ATPNaK[comp, LIS]
                      * actNa**3 * actK_LIS**2)
        dJact_BATH = (cnt[Lz+1].area[comp, BATH] * cnt[Lz+1].ATPNaK[comp, BATH]
                      * actNa**3 * actK_BATH**2)

        ro_LIS  = (C[NH4, LIS]  / AffNH4) / (C[K, LIS]  / AffK)
        ro_BATH = (C[NH4, BATH] / AffNH4) / (C[K, BATH] / AffK)

        Jsol[NA,  comp, LIS]  += dJact_LIS
        Jsol[NA,  comp, BATH] += dJact_BATH

        Jsol[K,   comp, LIS]  -= 2.0/3.0 * dJact_LIS  / (1 + ro_LIS)
        Jsol[K,   comp, BATH] -= 2.0/3.0 * dJact_BATH / (1 + ro_BATH)

        Jsol[NH4, comp, LIS]  -= 2.0/3.0 * dJact_LIS  * ro_LIS  / (1 + ro_LIS)
        Jsol[NH4, comp, BATH] -= 2.0/3.0 * dJact_BATH * ro_BATH / (1 + ro_BATH)

        fluxNaKase[comp] = (dJact_LIS + dJact_BATH) * convert

    # ------------------------------------------------------------------
    # H-ATPase (apical A cell and basolateral B cell)
    # See Strieter & Weinstein, AJP 263, 1992 for sign conventions
    # ------------------------------------------------------------------
    denom_LUM_A  = 1.0 + np.exp( steepA * (delmu[H, LUM, A]   - dmuATPH))
    denom_B_LIS  = 1.0 + np.exp(-steepB * (delmu[H, B,   LIS]  - dmuATPH))
    denom_B_BATH = 1.0 + np.exp(-steepB * (delmu[H, B,   BATH] - dmuATPH))

    dJact_LUM_A  = -cnt[Lz+1].area[LUM, A]  * cnt[Lz+1].ATPH[LUM, A]  / denom_LUM_A
    dJact_B_LIS  =  cnt[Lz+1].area[B,   LIS] * cnt[Lz+1].ATPH[B,   LIS]  / denom_B_LIS
    dJact_B_BATH =  cnt[Lz+1].area[B,   BATH] * cnt[Lz+1].ATPH[B,  BATH] / denom_B_BATH

    Jsol[H, LUM, A]   += dJact_LUM_A
    Jsol[H, B,   LIS]  += dJact_B_LIS
    Jsol[H, B,   BATH] += dJact_B_BATH

    # fluxHATPaseMA  = dJact_LUM_A * convert              # dimensional, unused
    # fluxHATPaseBES = (dJact_B_LIS + dJact_B_BATH) * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # H-K-ATPase at apical membranes of P, A, B cells
    # ------------------------------------------------------------------
    dkf5, dkb5 = 4.0e1, 2.0e2
    for comp in (P, A, B):
        hkconc[0] = C[K, comp]    # [K]in
        hkconc[1] = C[K, LUM]     # [K]out
        hkconc[2] = C[H, comp]    # [H]in
        hkconc[3] = C[H, LUM]     # [H]out

        Amat = fatpase(Natp, hkconc)
        # Amat_org = Amat   # unused alias
        # Amat_n   = Amat   # unused alias

        if np.linalg.det(Amat) != 0:
            Amat = np.linalg.inv(Amat)
            # Amat_inv = Amat  # unused alias
            c7 = Amat[6, 0]
            c8 = Amat[7, 0]
            hefflux = (cnt[Lz+1].area[LUM, comp] * cnt[Lz+1].ATPHK[LUM, comp]
                       * (dkf5 * c7 - dkb5 * c8))
            Jsol[K, LUM, comp] += 2 * hefflux
            Jsol[H, LUM, comp] -= 2 * hefflux
            # fluxHKATPase[comp] = 2 * hefflux * convert  # dimensional, unused

    # ------------------------------------------------------------------
    # PMCA: Ca2+ pump at basolateral membrane of P cell
    # ------------------------------------------------------------------
    ratio      = C[CA, P] / (C[CA, P] + dKmPMCA)
    dJPMCA_LIS  = cnt[Lz+1].PMCA * cnt[Lz+1].area[P, LIS]  * ratio
    dJPMCA_BATH = cnt[Lz+1].PMCA * cnt[Lz+1].area[P, BATH] * ratio

    Jsol[CA, P, LIS]  += dJPMCA_LIS
    Jsol[CA, P, BATH] += dJPMCA_BATH
    fluxPMCA = (dJPMCA_LIS + dJPMCA_BATH) * convert

    # ------------------------------------------------------------------
    # Store local flux values for later integration
    # ------------------------------------------------------------------
    # cv  = href * Cref * np.pi * DiamC * 60 / 10 * 1.0e9  # duplicate of convert
    # cvw = Pfref * Cref * Vwbar * np.pi * DiamC * 60 / 10 * 1.0e6  # unused

    cnt[Lz+1].FNatrans = ((Jsol[NA, LUM, P] + Jsol[NA, LUM, A] + Jsol[NA, LUM, B])
                          * convert * cnt[Lz+1].coalesce)
    cnt[Lz+1].FNapara  = Jsol[NA, LUM, LIS] * convert * cnt[Lz+1].coalesce
    cnt[Lz+1].FNaK     = fluxNaKase[P:B+1].sum() * cnt[Lz+1].coalesce

    cnt[Lz+1].FKtrans  = ((Jsol[K, LUM, P] + Jsol[K, LUM, A] + Jsol[K, LUM, B])
                          * convert * cnt[Lz+1].coalesce)
    cnt[Lz+1].FKpara   = Jsol[K, LUM, LIS] * convert * cnt[Lz+1].coalesce

    fTRPV5_cnt[Lz+1] = fluxTRPV5  * cnt[Lz+1].coalesce
    fNCX_cnt[Lz+1]   = fluxNCXca  * cnt[Lz+1].coalesce
    fPMCA_cnt[Lz+1]  = fluxPMCA   * cnt[Lz+1].coalesce
    fTRPV4_cnt[Lz+1] = fluxTRPV4  * cnt[Lz+1].coalesce

    return Jvol, Jsol
