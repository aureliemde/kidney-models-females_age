"""
PT (Proximal Tubule) flux evaluator — inlet node (jz = 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module computes all membrane fluxes (water, electro-convective-diffusive,
transporter, and active transport) for the PT inlet cross-section (jz = 0).
Called by fcn1PT at each Newton iteration.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

from compute_water_fluxes import compute_water_fluxes
from compute_ecd_fluxes import compute_ecd_fluxes
from sglt import sglt
from compute_nhe3_fluxes import compute_nhe3_fluxes

# LzPT in "main" starts from -1 for "qnewton1PT"
# Thus, it is LzPT+1 as Fortran version (No need for LzPT+1-1)
LzPT = -1


def qflux1PT(x, Cext, EPext, pt, vol0, dcompl, dtorq, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute all membrane fluxes for the PT inlet node.

    Unpacks the solution vector x, computes water fluxes, electro-convective-
    diffusive fluxes, transporter fluxes (SGLT, GLUT, NaPi, NHE3, KCl, etc.),
    and active transport (Na-K-ATPase, H-ATPase, PMCA). Applies torque scaling
    to transcellular fluxes. Stores select flux values on the pt object.

    Returns
    -------
    Jvol : ndarray, shape (NC, NC)   -- volumetric water fluxes
    Jsol : ndarray, shape (NSPT, NC, NC) -- solute fluxes
    """
    C   = np.zeros((NSPT, NC))
    ph  = np.zeros(NC)
    Vol = np.zeros(NC)
    EP  = np.zeros(NC)

    nagluparam = np.zeros(7)

    # for HKATPase matrix inversion (reserved for future use)
    # lwork = 10000
    # ipiv  = np.zeros(Natp)
    # work  = np.zeros(lwork)

    # --- Assign concentrations, volumes, and potentials ---

    C[:NSPT, BATH] = Cext[:NSPT]
    EP[BATH] = EPext

    C[:NSPT, LUM] = x[0:3*NSPT:3]
    C[:NSPT, P]   = x[1:3*NSPT:3]
    C[:NSPT, LIS] = x[2:3*NSPT:3]

    base     = 3 * NSPT
    Vol[LUM] = x[base + 0]
    Vol[P]   = x[base + 1]
    Vol[LIS] = x[base + 2]
    EP[LUM]  = x[base + 3]
    EP[P]    = x[base + 4]
    EP[LIS]  = x[base + 5]
    PM       = x[base + 6]

    # Dummy values for unused A and B cell compartments
    C[:NSPT, A] = C[:NSPT, P]
    C[:NSPT, B] = C[:NSPT, P]

    ph[:] = -np.log10(C[H, :] / 1.0e3)

    # --- Update LIS-BATH surface area ---

    pt[0].area[LIS, BATH] = pt[0].sbasEinit * max(Vol[LIS] / pt[0].volEinit, 1.0)
    pt[0].area[BATH, LIS] = pt[0].area[LIS, BATH]

    # --- Initialize fluxes ---

    Jvol = np.zeros((NC, NC))
    Jsol = np.zeros((NSPT, NC, NC))

    # --- Water fluxes ---

    Jvol = compute_water_fluxes(C, PM, PbloodPT, Vol, pt[0].volLuminit,
                                pt[0].volEinit, pt[0].volPinit, CPimprefPT,
                                pt[0].volAinit, CAimprefPT, pt[0].volBinit,
                                CBimprefPT, pt[0].area, pt[0].sig, pt[0].dLPV,
                                complPT, PTinitVol)

    # --- Torque scaling ---

    # Compliant tubule radius (Eq. 38 in 2007 PT model)
    if dcompl < 0:
        RMtorq = pt[0].diam / 2.0  # non-compliant tubule
    else:
        RMtorq = torqR * (1.0 + torqvm * (PM - PbloodPT))

    # Torque (Eq. 37 in 2007 PT model)
    factor1 = 8.0 * visc * (Vol[LUM] * Vref) * torqL / (RMtorq ** 2)
    factor2 = 1.0 + (torqL + torqd) / RMtorq + 0.50 * ((torqL / RMtorq) ** 2)
    Torque  = factor1 * factor2

    # Torque scaling parameter (Eq. 39 in 2007 PT model)
    if dtorq < 0:
        Scaletorq = 1.0  # no torque effect on transporter density
    else:
        Scaletorq = 1.0 + TS * pt[0].scaleT * (Torque / pt[0].TM0 - 1.0)
        PTtorque[LzPT + 1] = Scaletorq

    Scaletorq = max(Scaletorq, 0.001)

    convert = href * Cref * np.pi * pt[0].diam * 60 / 10 * 1.0e9 * Scaletorq

    # --- Electro-convective-diffusive fluxes ---

    Jsol, delmu = compute_ecd_fluxes(C, EP, pt[0].area, pt[0].sig, pt[0].h, Jvol)

    # --- SGLT1: luminal Na/glucose cotransporter (kinetic formulation) ---

    nagluparam[0] = C[NA,  LUM]
    nagluparam[1] = C[NA,  P]
    nagluparam[2] = C[GLU, LUM]
    nagluparam[3] = C[GLU, P]
    nagluparam[4] = EP[LUM]
    nagluparam[5] = EP[P]
    nagluparam[6] = pt[0].CTsglt1

    n = 1
    fluxsglt1 = sglt(n, nagluparam, pt[0].area[LUM, P])

    Jsol[NA,  LUM, P] += fluxsglt1[0] * pt[0].xSGLT1
    Jsol[GLU, LUM, P] += fluxsglt1[1] * pt[0].xSGLT1
    fluxnasglt1  = fluxsglt1[0] * pt[0].xSGLT1 * convert
    fluxglusglt1 = fluxsglt1[1] * pt[0].xSGLT1 * convert

    # --- SGLT2: luminal Na/glucose cotransporter (NET formulation, AMW 2007) ---

    zeta   = F * EPref * (EP[LUM] - EP[P]) / RT
    affglu = 4.90
    affna  = 25.0
    snal   = C[NA,  LUM] / affna
    snac   = C[NA,  P]   / affna
    glul   = C[GLU, LUM] / affglu
    gluc   = C[GLU, P]   / affglu

    denom = ((1.0 + snal + glul + snal * glul) * (1.0 + snac * gluc) +
              (1.0 + snac + gluc + snac * gluc) * (1.0 + snal * glul * np.exp(zeta)))

    dJNaglu = (
        pt[0].area[LUM, P] * pt[0].dLA[NA, GLU, LUM, P] *
        (C[NA, LUM] * C[GLU, LUM] * np.exp(zeta) - C[NA, P] * C[GLU, P]) / denom
    )

    Jsol[NA,  LUM, P] += dJNaglu * pt[0].xSGLT2
    Jsol[GLU, LUM, P] += dJNaglu * pt[0].xSGLT2
    fluxnasglt2  = dJNaglu * pt[0].xSGLT2 * convert
    fluxglusglt2 = dJNaglu * pt[0].xSGLT2 * convert

    # --- Basolateral glucose transporters ---

    Gi  = C[GLU, P]
    Go5 = C[GLU, LIS]
    Go6 = C[GLU, BATH]

    # GLUT1
    affglut1    = 2.0
    Ro5         = affglut1 * (Gi - Go5) / (affglut1 + Gi) / (affglut1 + Go5)
    fluxglut1PE = pt[0].CTglut1 * pt[0].area[P, LIS]  * Ro5
    Ro6         = affglut1 * (Gi - Go6) / (affglut1 + Gi) / (affglut1 + Go6)
    fluxglut1PS = pt[0].CTglut1 * pt[0].area[P, BATH] * Ro6

    Jsol[GLU, P, LIS]  += fluxglut1PE * pt[0].xGLUT1
    Jsol[GLU, P, BATH] += fluxglut1PS * pt[0].xGLUT1
    fluxglut1 = (fluxglut1PE + fluxglut1PS) * pt[0].xGLUT1 * convert

    # GLUT2
    affglut2    = 17.0
    Ro5         = affglut2 * (Gi - Go5) / (affglut2 + Gi) / (affglut2 + Go5)
    fluxglut2PE = pt[0].CTglut2 * pt[0].area[P, LIS]  * Ro5
    Ro6         = affglut2 * (Gi - Go6) / (affglut2 + Gi) / (affglut2 + Go6)
    fluxglut2PS = pt[0].CTglut2 * pt[0].area[P, BATH] * Ro6

    Jsol[GLU, P, LIS]  += fluxglut2PE * pt[0].xGLUT2
    Jsol[GLU, P, BATH] += fluxglut2PS * pt[0].xGLUT2
    fluxglut2 = (fluxglut2PE + fluxglut2PS) * pt[0].xGLUT2 * convert

    # --- Luminal NaH2PO4 cotransporter (generic NET formulation) ---

    dJNaP   = pt[0].area[LUM, P] * pt[0].dLA[NA, H2PO4, LUM, P] * (delmu[NA, LUM, P] + delmu[H2PO4, LUM, P])
    fluxNaP = dJNaP * convert

    # NaPiIIa: 3 Na+ with 1 HPO4(2-)
    dJNaPiIIa = pt[0].area[LUM, P] * xNaPiIIaPT * (3 * delmu[NA, LUM, P] + delmu[HPO4, LUM, P])
    Jsol[NA,   LUM, P] += 3.0 * dJNaPiIIa
    Jsol[HPO4, LUM, P] += dJNaPiIIa
    fluxNaPiIIa = dJNaPiIIa * convert

    # NaPiIIc: 2 Na+ with 1 HPO4(2-); scaled by xSGLT2 (non-homogeneous distribution)
    dJNaPiIIc = pt[0].area[LUM, P] * xNaPiIIcPT * (2 * delmu[NA, LUM, P] + delmu[HPO4, LUM, P])
    dJNaPiIIc *= pt[0].xSGLT2
    Jsol[NA,   LUM, P] += 2.0 * dJNaPiIIc
    Jsol[HPO4, LUM, P] += dJNaPiIIc
    fluxNaPiIIc = dJNaPiIIc * convert

    # PiT-2: 2 Na+ with 1 H2PO4-; scaled by xSGLT2 (non-homogeneous distribution)
    dJPit2 = pt[0].area[LUM, P] * xPit2PT * (2 * delmu[NA, LUM, P] + delmu[H2PO4, LUM, P])
    dJPit2 *= pt[0].xSGLT2
    Jsol[NA,    LUM, P] += 2.0 * dJPit2
    Jsol[H2PO4, LUM, P] += dJPit2
    fluxPit2 = dJPit2 * convert

    # --- Luminal Cl/HCO3 exchanger ---

    dJClBic = pt[0].area[LUM, P] * pt[0].dLA[CL, HCO3, LUM, P] * (delmu[CL, LUM, P] - delmu[HCO3, LUM, P])
    Jsol[CL,   LUM, P] += dJClBic
    Jsol[HCO3, LUM, P] -= dJClBic
    fluxClBic = dJClBic * convert

    # --- Luminal Cl/HCO2 exchanger ---

    dJClHco2 = pt[0].area[LUM, P] * pt[0].dLA[CL, HCO2, LUM, P] * (delmu[CL, LUM, P] - delmu[HCO2, LUM, P])
    Jsol[CL,   LUM, P] += dJClHco2
    Jsol[HCO2, LUM, P] -= dJClHco2
    fluxClHco2 = dJClHco2 * convert

    # --- Luminal H-HCO2 cotransporter (addition to AMW model) ---

    dJH_Hco2 = pt[0].area[LUM, P] * pt[0].dLA[H, HCO2, LUM, P] * (delmu[H, LUM, P] + delmu[HCO2, LUM, P])
    Jsol[H,    LUM, P] += dJH_Hco2
    Jsol[HCO2, LUM, P] += dJH_Hco2
    fluxH_Hco2 = dJH_Hco2 * convert

    # --- Basolateral KCl cotransporter ---

    sumJES = 0.0
    for L in (LIS, BATH):
        dJKCl = pt[0].area[P, L] * pt[0].dLA[K, CL, P, L] * (delmu[K, P, L] + delmu[CL, P, L])
        Jsol[K,  P, L] += dJKCl
        Jsol[CL, P, L] += dJKCl
        sumJES += dJKCl
    fluxKCl = sumJES * convert

    # --- Basolateral Na(1)-HCO3(3) cotransporter ---

    sumJES = 0.0
    for L in (LIS, BATH):
        dJNaBic = pt[0].area[P, L] * pt[0].dLA[NA, HCO3, P, L] * (delmu[NA, P, L] + 3 * delmu[HCO3, P, L])
        Jsol[NA,   P, L] += dJNaBic
        Jsol[HCO3, P, L] += 3 * dJNaBic
        sumJES += dJNaBic
    fluxNaBic = sumJES * convert

    # --- Basolateral Na(1)-HCO3(2)/Cl(1) cotransporter (NDCBE) ---

    sumJES = 0.0
    for L in (LIS, BATH):
        dJNDCBE = pt[0].area[P, L] * pt[0].dLA[NA, CL, P, L] * (delmu[NA, P, L] - delmu[CL, P, L] + 2 * delmu[HCO3, P, L])
        Jsol[NA,   P, L] += dJNDCBE
        Jsol[CL,   P, L] -= dJNDCBE
        Jsol[HCO3, P, L] += 2 * dJNDCBE
        sumJES += dJNDCBE
    fluxNDCBE = sumJES * convert

    # --- NHE3 exchanger at luminal membrane ---
    # Rate constants divided by 8000/792 relative to other segments

    dJNHEsod, dJNHEprot, dJNHEamm = compute_nhe3_fluxes(C, pt[0].area[LUM, P], pt[0].xNHE3)

    dJNHEsod  *= 792.0 / 8000.0
    dJNHEprot *= 792.0 / 8000.0
    dJNHEamm  *= 792.0 / 8000.0

    Jsol[NA,  LUM, P] += dJNHEsod
    Jsol[H,   LUM, P] += dJNHEprot
    Jsol[NH4, LUM, P] += dJNHEamm
    fluxNHEsod  = dJNHEsod  * convert
    fluxNHEprot = dJNHEprot * convert
    fluxNHEamm  = dJNHEamm  * convert

    # --- Na-K-ATPase ---

    AffNa  = 0.2 * (1.0 + C[K,  P]    / 8.33)
    actNa  = C[NA, P] / (C[NA, P] + AffNa)

    AffK5  = 0.1 * (1.0 + C[NA, LIS]  / 18.5)
    AffNH5 = AffK5
    actK5  = (C[K, LIS]  + C[NH4, LIS])  / (C[K, LIS]  + C[NH4, LIS]  + AffK5)

    AffK6  = 0.1 * (1.0 + C[NA, BATH] / 18.5)
    AffNH6 = AffK6
    actK6  = (C[K, BATH] + C[NH4, BATH]) / (C[K, BATH] + C[NH4, BATH] + AffK6)

    ro5 = (C[NH4, LIS]  / AffNH5) / (C[K, LIS]  / AffK5)
    ro6 = (C[NH4, BATH] / AffNH6) / (C[K, BATH] / AffK6)

    dJactNa5 = pt[0].area[P, LIS]  * pt[0].ATPNaK[P, LIS]  * (actNa**3.0) * (actK5**2.0)
    dJactNa6 = pt[0].area[P, BATH] * pt[0].ATPNaK[P, BATH] * (actNa**3.0) * (actK6**2.0)

    dJactK5 = -2.0 / 3.0 * dJactNa5 / (1.0 + ro5)
    dJactK6 = -2.0 / 3.0 * dJactNa6 / (1.0 + ro6)

    Jsol[NA,  P, LIS]  += dJactNa5
    Jsol[NA,  P, BATH] += dJactNa6

    Jsol[K,   P, LIS]  += dJactK5
    Jsol[K,   P, BATH] += dJactK6

    Jsol[NH4, P, LIS]  += dJactK5 * ro5
    Jsol[NH4, P, BATH] += dJactK6 * ro6

    fluxNaKsod = (dJactNa5 + dJactNa6) * convert
    fluxNaKpot = (dJactK5  + dJactK6)  * convert
    fluxNaKamm = (dJactK5 * ro5 + dJactK6 * ro6) * convert

    # --- H-ATPase (see Strieter & Weinstein, AJP 263, 1992) ---

    DactH  = 1.0 + np.exp(steepATPH * (delmu[H, LUM, P] - dmuATPHPT))
    dJactH = -pt[0].area[LUM, P] * pt[0].ATPH[LUM, P] / DactH

    Jsol[H, LUM, P] += dJactH
    fluxHATPase = dJactH * convert

    # --- PMCA: putative basolateral Ca2+ pump ---
    # AffCa from Tsukamoto et al., Biochim Biophys Acta 1992

    AffCa   = 75.6e-6
    dJPMCA5 = pt[0].area[P, LIS]  * pt[0].PMCA * C[CA, P] / (C[CA, P] + AffCa)
    dJPMCA6 = pt[0].area[P, BATH] * pt[0].PMCA * C[CA, P] / (C[CA, P] + AffCa)

    Jsol[CA, P, LIS]  += dJPMCA5
    Jsol[CA, P, BATH] += dJPMCA6
    fluxPMCA = (dJPMCA5 + dJPMCA6) * convert

    # --- Apply torque scaling to transcellular fluxes ---
    # NOTE: Jvol scaling is intentionally inside the solute loop (applied NSPT times)
    # to match the original Fortran logic — do NOT vectorize or move outside the loop.

    for i in range(NSPT):
        Jsol[i, LUM, P]  *= Scaletorq
        Jsol[i, P, LIS]  *= Scaletorq
        Jsol[i, P, BATH] *= Scaletorq

        Jvol[LUM, P]  *= Scaletorq
        Jvol[P, LIS]  *= Scaletorq
        Jvol[P, BATH] *= Scaletorq

    # --- Store dimensional fluxes for output ---

    cv  = href * Cref * np.pi * pt[0].diam * 60 / 10 * 1.0e9   # pmol/min/mm
    cvw = Pfref * Cref * Vwbar * np.pi * pt[0].diam * 60 / 10 * 1.0e6  # nl/min/mm

    pt[0].FNatrans  = Jsol[NA,  LUM, P]   * cv
    pt[0].FNapara   = Jsol[NA,  LUM, LIS] * cv
    pt[0].FNaK      = fluxNaKsod
    pt[0].FHase     = fluxHATPase
    pt[0].FGluPara  = Jsol[GLU, LUM, LIS] * cv
    pt[0].FGluSGLT1 = fluxglusglt1
    pt[0].FGluSGLT2 = fluxglusglt2
    pt[0].FKtrans   = Jsol[K,   LUM, P]   * cv
    pt[0].FKpara    = Jsol[K,   LUM, LIS] * cv

    fCaTransPT[LzPT + 1] = Jsol[CA, LUM, P]   * cv
    fCaParaPT[LzPT + 1]  = Jsol[CA, LUM, LIS] * cv

    # nlocal = 1  # unused

    return Jvol, Jsol
