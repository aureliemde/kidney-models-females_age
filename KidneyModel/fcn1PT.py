"""
PT (Proximal Tubule) residual function — inlet node (jz = 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module computes the residual vector fvec of non-linear equations for
the PT inlet cross-section (jz = 0). It is called by qnewton1PT at each
Newton iteration to evaluate the system at the current solution estimate.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

from qflux1PT import qflux1PT


def fcn1PT(n, x, iflag, numpar, pars, pt, idid, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute the residual vector fvec for the PT inlet node.

    Unpacks the solution vector x into concentrations, volumes, and potentials,
    calls qflux1PT to evaluate all membrane fluxes, then assembles the residual
    equations for each solute group (non-reacting, CO2/HCO3, phosphate,
    ammonia, formate, glucose, Ca2+), volume, and electroneutrality/current.

    Returns
    -------
    fvec : ndarray, shape (n,) -- residual vector
    """
    fvec = np.zeros(n)
    S    = np.zeros(NUPT)
    y    = np.zeros(NDPT)

    C   = np.zeros((NSPT, NC))
    Vol = np.zeros(NC)
    EP  = np.zeros(NC)
    ph  = np.zeros(NC)

    dkd = np.zeros(NC)
    dkh = np.zeros(NC)

    # Assign fixed concentrations, potentials from lumen and bath
    C[:, LUM]  = pt[0].conc[:, LUM]
    C[:, BATH] = pt[0].conc[:, BATH]

    ph[LUM]  = -np.log10(C[H, LUM]  / 1.0e3)
    ph[BATH] = -np.log10(C[H, BATH] / 1.0e3)

    EP[BATH] = pt[0].ep[BATH]

    # Unpack parameter vector
    VolEinit  = pars[0]
    VolPinit  = pars[1]
    CPimpref  = pars[2]
    CPbuftot1 = pars[3]

    dkd[P]   = pars[4]
    dkd[LIS] = pars[5]
    dkh[P]   = pars[6]
    dkh[LIS] = pars[7]

    dcompl = pars[8]
    dtorq  = pars[9]

    # Unpack solution vector x into C, Vol, EP for P and LIS
    base = 2 * NSPT

    C[:NSPT, P]   = x[0:2*NSPT:2]
    C[:NSPT, LIS] = x[1:2*NSPT:2]

    ph[P]   = -np.log10(C[H, P]   / 1.0e3)
    ph[LIS] = -np.log10(C[H, LIS] / 1.0e3)

    Vol[P]   = x[base + 0]
    Vol[LIS] = x[base + 1]
    EP[LUM]  = x[base + 2]
    EP[P]    = x[base + 3]
    EP[LIS]  = x[base + 4]

    # Pack flux input vector y: interleaved (LUM, P, LIS) per solute
    ybase = 3 * NSPT

    y[0:ybase:3] = pt[0].conc[:NSPT, LUM]
    y[1:ybase:3] = C[:NSPT, P]
    y[2:ybase:3] = C[:NSPT, LIS]

    Vol0         = pt[0].vol[LUM]
    y[ybase + 0] = Vol0
    y[ybase + 1] = Vol[P]
    y[ybase + 2] = Vol[LIS]
    y[ybase + 3] = EP[LUM]
    y[ybase + 4] = EP[P]
    y[ybase + 5] = EP[LIS]
    y[ybase + 6] = PMinitPT

    # Evaluate membrane fluxes
    Jvol, Jsol = qflux1PT(y, C[:, BATH], EP[BATH], pt, Vol0, dcompl, dtorq,
                          PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

    # Solute source terms: net flux into P and LIS
    S[0:2*NSPT:2] = -Jsol[:NSPT, LUM, P]   + Jsol[:NSPT, P, LIS] + Jsol[:NSPT, P,   BATH]
    S[1:2*NSPT:2] = -Jsol[:NSPT, LUM, LIS] - Jsol[:NSPT, P, LIS] + Jsol[:NSPT, LIS, BATH]

    # Volume source terms
    S[2*NSPT + 0] = -Jvol[LUM, P]   + Jvol[P, LIS]   + Jvol[P,   BATH]
    S[2*NSPT + 1] = -Jvol[LUM, LIS] - Jvol[P, LIS]   + Jvol[LIS, BATH]

    # --- Residual equations ---

    # Non-reacting solutes: Na, K, Cl (indices 0-5) and Urea (16-17)
    fvec[0:6]  = S[0:6]
    fvec[16]   = S[16]
    fvec[17]   = S[17]

    # Glucose: with consumption term linked to Na,K-ATPase activity
    Pumpactivity  = pt[0].FNaK / (href * Cref * np.pi * pt[0].diam * 60 / 10 * 1.0e9)
    Gluconsumption = Pumpactivity / (pt[0].TQ * 6)
    fvec[28] = S[28] + Gluconsumption * 0   # multiply by 0: consumption not currently active
    fvec[29] = S[29]

    # Ca2+
    fvec[30] = S[30]
    fvec[31] = S[31]

    # CO2/HCO3/H2CO3: combined flux + equilibrium + reaction rate
    facnd    = Vref / href
    fvec[6]  = S[6]  + S[8]  + S[10]
    fvec[7]  = S[7]  + S[9]  + S[11]
    fvec[8]  = ph[P]   - pKHCO3 - np.log10(abs(C[HCO3,  P]   / C[H2CO3, P]))
    fvec[9]  = ph[LIS] - pKHCO3 - np.log10(abs(C[HCO3,  LIS] / C[H2CO3, LIS]))
    fvec[10] = S[10] + Vol[P]   * (dkh[P]   * C[CO2, P]   - dkd[P]   * C[H2CO3, P])   * facnd
    fvec[11] = S[11] + max(Vol[LIS], VolEinit) * (dkh[LIS] * C[CO2, LIS] - dkd[LIS] * C[H2CO3, LIS]) * facnd

    # HPO4(2-)/H2PO4(-): combined flux + equilibrium
    fvec[12] = S[12] + S[14]
    fvec[13] = S[13] + S[15]
    fvec[14] = ph[P]   - pKHPO4 - np.log10(abs(C[HPO4, P]   / C[H2PO4, P]))
    fvec[15] = ph[LIS] - pKHPO4 - np.log10(abs(C[HPO4, LIS] / C[H2PO4, LIS]))

    # NH3/NH4+ with ammoniagenesis (torque-scaled)
    if dcompl < 0:
        RMtorq = pt[0].diam / 2.0   # non-compliant tubule
    else:
        RMtorq = torqR * (1.0 + torqvm * (PMinitPT - PbloodPT))

    factor1 = 8.0 * visc * (Vol0 * Vref) * torqL / (RMtorq ** 2)
    factor2 = 1.0 + (torqL + torqd) / RMtorq + 0.50 * ((torqL / RMtorq) ** 2)
    Torque  = factor1 * factor2

    if dtorq < 0:
        Scaletorq = 1.0   # no torque effect on transporter density
    else:
        Scaletorq = 1.0 + TS * pt[0].scaleT * (Torque / pt[0].TM0 - 1.0)

    Scaletorq = max(Scaletorq, 0.001)   # guard against negative values

    Qnh4 = pt[0].qnh4 * Scaletorq if idid == 0 else 0.0

    fvec[18] = S[18] + S[20] - Qnh4
    fvec[19] = S[19] + S[21]
    fvec[20] = ph[P]   - pKNH3 - np.log10(abs(C[NH3, P]   / C[NH4, P]))
    fvec[21] = ph[LIS] - pKNH3 - np.log10(abs(C[NH3, LIS] / C[NH4, LIS]))

    # HCO2-/H2CO2: combined flux + equilibrium
    fvec[24] = S[24] + S[26]
    fvec[25] = S[25] + S[27]
    fvec[26] = ph[P]   - pKHCO2 - np.log10(abs(C[HCO2, P]   / C[H2CO2, P]))
    fvec[27] = ph[LIS] - pKHCO2 - np.log10(abs(C[HCO2, LIS] / C[H2CO2, LIS]))

    # H+: proton balance
    fvec[22] = S[22] + S[20] - S[12] - S[6] - S[24]
    fvec[23] = S[23] + S[21] - S[13] - S[7] - S[25]

    # Volume equations
    fvec[base + 0] = S[2*NSPT + 0]
    fvec[base + 1] = S[2*NSPT + 1]

    # Electroneutrality in P and LIS
    volPrat = VolPinit / Vol[P]
    facP    = np.exp(np.log(10.0) * (ph[P] - pKbuf))
    CimpP   = CPimpref * volPrat
    CbufP   = CPbuftot1 * volPrat * facP / (facP + 1)

    elecP = zPimpPT * CimpP - CbufP + np.dot(zval[:NSPT], C[:NSPT, P])
    elecE = np.dot(zval[:NSPT], C[:NSPT, LIS])

    fvec[base + 2] = elecP
    fvec[base + 3] = elecE

    # Zero net current in lumen
    currM = np.dot(zval[:NSPT], Jsol[:NSPT, LUM, P] + Jsol[:NSPT, LUM, LIS])
    fvec[base + 4] = currM

    return fvec
