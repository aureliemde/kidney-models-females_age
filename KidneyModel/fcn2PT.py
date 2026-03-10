"""
PT (Proximal Tubule) residual function — interior nodes (jz > 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module computes the residual vector fvec of non-linear equations for
PT interior cross-sections (jz > 0). It is called by qnewton2PT at each
Newton iteration to evaluate the system at the current solution estimate.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

from qflux2PT import qflux2PT


def fcn2PT(n, x, iflag, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute the residual vector fvec for PT interior nodes (jz > 0).

    Unpacks known state at Lz and unknown state at Lz+1 from pars and x,
    calls qflux2PT for membrane fluxes, then assembles residual equations
    for each solute group (non-reacting, CO2/HCO3, phosphate, ammonia,
    formate, glucose, Ca2+), volume, electroneutrality, and pressure.

    Returns
    -------
    fvec : ndarray, shape (n,) -- residual vector
    """
    fvec = np.zeros(n)
    S    = np.zeros(NDPT)

    Ca   = np.zeros((NSPT, NC))
    Vola = np.zeros(NC)

    Cb   = np.zeros((NSPT, NC))
    Volb = np.zeros(NC)
    EPb  = np.zeros(NC)
    phb  = np.zeros(NC)

    dkd = np.zeros(NC)
    dkh = np.zeros(NC)

    # --- Known state at Lz (from pars) ---
    # Ca: concentrations at previous axial node
    # Vola/PMa: volumes and pressure at previous node

    Ca[:, LUM]  = pars[0:NSPT]
    Ca[:, P]    = pars[NSPT:2*NSPT]
    Ca[:, LIS]  = pars[2*NSPT:3*NSPT]
    Ca[:, BATH] = pars[3*NSPT:4*NSPT]

    base      = 4 * NSPT
    Vola[LUM] = pars[base + 0]
    Vola[P]   = pars[base + 1]
    Vola[LIS] = pars[base + 2]
    PMa       = pars[base + 3]

    # --- Step constants ---

    VolEinit  = pars[base + 4]
    VolPinit  = pars[base + 5]
    dimL      = pars[base + 6]
    CPimpref  = pars[base + 7]
    CPbuftot1 = pars[base + 8]

    dkd[LUM] = pars[base + 9]
    dkd[P]   = pars[base + 10]
    dkd[LIS] = pars[base + 11]
    dkh[LUM] = pars[base + 12]
    dkh[P]   = pars[base + 13]
    dkh[LIS] = pars[base + 14]

    dcompl = pars[base + 15]
    dtorq  = pars[base + 16]
    vol0   = pars[base + 17]   # Vol0[LUM]

    # --- Unknown state at Lz+1 (from x) ---

    Cb[:NSPT, LUM] = x[0:3*NSPT:3]
    Cb[:NSPT, P]   = x[1:3*NSPT:3]
    Cb[:NSPT, LIS] = x[2:3*NSPT:3]

    Cb[:, BATH] = pt[Lz+1].conc[:, BATH]
    EPb[BATH]   = pt[Lz+1].ep[BATH]

    phb[LUM] = -np.log10(Cb[H, LUM] / 1.0e3)
    phb[P]   = -np.log10(Cb[H, P]   / 1.0e3)
    phb[LIS] = -np.log10(Cb[H, LIS] / 1.0e3)

    xb        = 3 * NSPT
    Volb[LUM] = x[xb + 0]
    Volb[P]   = x[xb + 1]
    Volb[LIS] = x[xb + 2]
    EPb[LUM]  = x[xb + 3]
    EPb[P]    = x[xb + 4]
    EPb[LIS]  = x[xb + 5]
    PMb       = x[xb + 6]

    # --- Membrane fluxes ---

    Jvol, Jsol = qflux2PT(x, Cb[:, BATH], EPb[BATH], pt, vol0, dcompl, dtorq, Lz,
                           PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

    # --- Geometry ---

    Bm = np.pi * pt[Lz+1].diam
    Am = np.pi * (pt[Lz+1].diam ** 2) / 4.0
    # Bmcompl = 2 * np.pi * torqR * (1.0 + torqvm * (PMb - PbloodPT))  # unused

    # --- Source terms ---

    i_idx = np.arange(NSPT)

    # Lumen: axial solute conservation
    sumJsb        = Jsol[:NSPT, LUM, P] + Jsol[:NSPT, LUM, LIS]
    S[3*i_idx]    = Volb[LUM] * Cb[:NSPT, LUM] * Vref / href \
                  - Vola[LUM] * Ca[:NSPT, LUM] * Vref / href \
                  + Bm * dimL * sumJsb / NZ

    # P cell and LIS: membrane balance
    S[3*i_idx + 1] = Jsol[:NSPT, P, LIS]  + Jsol[:NSPT, P, BATH]  - Jsol[:NSPT, LUM, P]
    S[3*i_idx + 2] = Jsol[:NSPT, LIS, BATH] - Jsol[:NSPT, LUM, LIS] - Jsol[:NSPT, P, LIS]

    # Volume: lumen, P, LIS
    fvmult        = Pfref * Vwbar * Cref
    S[3*NSPT + 0] = Volb[LUM] * Vref - Vola[LUM] * Vref \
                  + Bm * dimL * (Jvol[LUM, P] + Jvol[LUM, LIS]) * fvmult / NZ
    S[3*NSPT + 1] = Jvol[P, LIS]  + Jvol[P, BATH]  - Jvol[LUM, P]
    S[3*NSPT + 2] = Jvol[LIS, BATH] - Jvol[LUM, LIS] - Jvol[P, LIS]

    # --- Residual equations ---

    # Non-reacting solutes: Na, K, Cl (fvec[0..8])
    fvec[:9] = S[:9]

    # Urea (must be assigned separately — not contiguous with Na/K/Cl block)
    fvec[24] = S[24]
    fvec[25] = S[25]
    fvec[26] = S[26]

    # Glucose: with consumption term linked to Na,K-ATPase activity
    Pumpactivity   = pt[Lz+1].FNaK / (href * Cref * np.pi * pt[Lz+1].diam * 60 / 10 * 1.0e9)
    Gluconsumption = Pumpactivity / (pt[Lz+1].TQ * 6)

    fvec[3*GLU]     = S[3*GLU]
    fvec[3*GLU + 1] = S[3*GLU + 1] + Gluconsumption * 0   # multiply by 0: consumption not currently active
    fvec[3*GLU + 2] = S[3*GLU + 2]

    # Ca2+
    fvec[3*CA]     = S[3*CA]
    fvec[3*CA + 1] = S[3*CA + 1]
    fvec[3*CA + 2] = S[3*CA + 2]

    # CO2/HCO3/H2CO3: combined flux + equilibrium + kinetics
    # Am/Bm = R/2 = D/4 is the dimensional factor for the lumen kinetic term
    fvec[9]  = S[9]  + S[12] + S[15]
    fvec[10] = S[10] + S[13] + S[16]
    fvec[11] = S[11] + S[14] + S[17]

    fvec[12] = phb[LUM] - pKHCO3 - np.log10(Cb[HCO3, LUM] / Cb[H2CO3, LUM])
    fvec[13] = phb[P]   - pKHCO3 - np.log10(Cb[HCO3, P]   / Cb[H2CO3, P])
    fvec[14] = phb[LIS] - pKHCO3 - np.log10(Cb[HCO3, LIS] / Cb[H2CO3, LIS])

    # fkin1 = dkh[LUM] * Ca[CO2, LUM] - dkd[LUM] * Ca[H2CO3, LUM]  # unused
    fkin2  = dkh[LUM] * Cb[CO2, LUM] - dkd[LUM] * Cb[H2CO3, LUM]

    facnd    = Vref / href
    fvec[15] = S[15] + Am * dimL * fkin2 / NZ / href
    fvec[16] = S[16] + Volb[P]   * (dkh[P]   * Cb[CO2, P]   - dkd[P]   * Cb[H2CO3, P])   * facnd
    fvec[17] = S[17] + max(Volb[LIS], VolEinit) * (dkh[LIS] * Cb[CO2, LIS] - dkd[LIS] * Cb[H2CO3, LIS]) * facnd

    # HPO4(2-)/H2PO4(-): combined flux + equilibrium
    fvec[18] = S[18] + S[21]
    fvec[19] = S[19] + S[22]
    fvec[20] = S[20] + S[23]

    fvec[21] = phb[LUM] - pKHPO4 - np.log10(Cb[HPO4, LUM] / Cb[H2PO4, LUM])
    fvec[22] = phb[P]   - pKHPO4 - np.log10(Cb[HPO4, P]   / Cb[H2PO4, P])
    fvec[23] = phb[LIS] - pKHPO4 - np.log10(Cb[HPO4, LIS] / Cb[H2PO4, LIS])

    # NH3/NH4+ with ammoniagenesis (torque-scaled)
    if dcompl < 0:
        RMtorq = pt[Lz+1].diam / 2.0  # non-compliant tubule
    else:
        RMtorq = torqR * (1.0 + torqvm * (PMb - PbloodPT))

    factor1 = 8.0 * visc * (Volb[LUM] * Vref) * torqL / (RMtorq**2)
    factor2 = 1.0 + (torqL + torqd) / RMtorq + 0.5 * ((torqL / RMtorq)**2)
    Torque  = factor1 * factor2

    if dtorq < 0:
        Scaletorq = 1.0
    else:
        Scaletorq = max(1.0 + TS * pt[Lz+1].scaleT * (Torque / pt[Lz+1].TM0 - 1.0), 0.001)

    Qnh4 = pt[Lz+1].qnh4 * Scaletorq if idid == 0 else 0.0

    fvec[27] = S[27] + S[30]
    fvec[28] = S[28] + S[31] - Qnh4
    fvec[29] = S[29] + S[32]

    fvec[30] = phb[LUM] - pKNH3 - np.log10(Cb[NH3, LUM] / Cb[NH4, LUM])
    fvec[31] = phb[P]   - pKNH3 - np.log10(Cb[NH3, P]   / Cb[NH4, P])
    fvec[32] = phb[LIS] - pKNH3 - np.log10(Cb[NH3, LIS] / Cb[NH4, LIS])

    # HCO2-/H2CO2: combined flux + equilibrium
    fvec[36] = S[36] + S[39]
    fvec[37] = S[37] + S[40]
    fvec[38] = S[38] + S[41]

    fvec[39] = phb[LUM] - pKHCO2 - np.log10(np.abs(Cb[HCO2, LUM] / Cb[H2CO2, LUM]))
    fvec[40] = phb[P]   - pKHCO2 - np.log10(np.abs(Cb[HCO2, P]   / Cb[H2CO2, P]))
    fvec[41] = phb[LIS] - pKHCO2 - np.log10(np.abs(Cb[HCO2, LIS] / Cb[H2CO2, LIS]))

    # H+: proton balance
    fvec[33] = S[33] + S[30] - S[18] - S[9]  - S[36]   # LUM
    fvec[34] = S[34] + S[31] - S[19] - S[10] - S[37]   # P
    fvec[35] = S[35] + S[32] - S[20] - S[11] - S[38]   # LIS

    # Volume
    baseV           = 3 * NSPT
    fvec[baseV + 0] = S[baseV + 0]
    fvec[baseV + 1] = S[baseV + 1]
    fvec[baseV + 2] = S[baseV + 2]

    # Zero net current in lumen
    currM = np.dot(zval[:NSPT], Jsol[:NSPT, LUM, P] + Jsol[:NSPT, LUM, LIS])
    fvec[4 + 3*NSPT - 1] = currM

    # Electroneutrality in P and LIS
    volPrat = VolPinit / Volb[P]
    facP    = np.exp(np.log(10.0) * (phb[P] - pKbuf))
    CimpP   = CPimpref * volPrat
    CbufP   = CPbuftot1 * volPrat * facP / (facP + 1)

    elecP = zPimpPT * CimpP - CbufP + np.dot(zval[:NSPT], Cb[:NSPT, P])
    elecE = np.dot(zval[:NSPT], Cb[:NSPT, LIS])

    fvec[5 + 3*NSPT - 1] = elecP
    fvec[6 + 3*NSPT - 1] = elecE

    # Pressure: Poiseuille law
    RMcompl = torqR * (1.0 + torqvm * (PMb - PbloodPT))
    Amcompl = np.pi * (RMcompl**2)

    if dcompl < 0:
        factor1 = 8.0 * np.pi * visc / (Am**2)
    else:
        factor1 = 8.0 * np.pi * visc / (Amcompl**2)

    fvec[7 + 3*NSPT - 1] = PMb - PMa + factor1 * Volb[LUM] * Vref * dimL / NZ

    return fvec
