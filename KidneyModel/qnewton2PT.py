"""
PT (Proximal Tubule) Newton solver — interior nodes (jz > 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module implements the Newton-Raphson solver for all PT cross-sections
below the inlet (jz > 0). It determines steady-state concentrations, volumes,
electrical potentials, and luminal pressure at each axial position.

Differences from the AMW 2007 model:
  - pressure equation corrected; TS = 1.6 (not 2.2)
  - no apical Cl/HCO3 exchanger (no evidence in rat PT; Planelles 2004,
    Aronson 1997/2002)
  - small basolateral Cl conductance (Aronson & Giebisch 1997)
  - glucose consumption term linked to Na-K-ATPase activity
  - kinetic SGLT1/SGLT2 and GLUT1/GLUT2 descriptions
  - S3 segment (SGLT1 + GLUT1) accounted for

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

from fcn2PT import fcn2PT
from jacobi2_2PT import jacobi2_2PT


def qnewton2PT(pt, Lz, idid, Vol0, CPimpref, CPbuftot1, ND, ncompl2, ntorq2, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Newton-Raphson solver for PT interior nodes (jz > 0).

    Iterates until the relative change in concentrations, volumes, potentials,
    and pressure drops below TOL (1e-5). Tolerance is relaxed progressively
    after iteration 11 to aid convergence in stiff cases. Writes the converged
    solution into pt[Lz+1].

    Args:
    pt         : list of Membrane -- PT segment nodes
    Lz         : int  -- index of the upstream (previous) node
    idid       : int  -- solver flag (0 = normal, 1 = skip ammoniagenesis)
    Vol0       : array -- luminal volume at inlet (used as axial step reference)
    CPimpref   : float -- reference impermeant concentration in P cell
    CPbuftot1  : float -- total buffer concentration in P cell
    ND         : int   -- number of unknowns
    ncompl2    : int   -- compliance switch
    ntorq2     : int   -- torque switch
    PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT : floats -- passed through to fcn2PT
    """
    # EPzprev = np.zeros(NC)  # unused
    # EPz     = np.zeros(NC)  # unused

    Cb    = np.zeros((NSPT, NC))
    Volb  = np.zeros(NC)
    EPb   = np.zeros(NC)
    phb   = np.zeros(NC)

    AVF = np.zeros(ND)
    # A   = np.zeros((ND, ND))  # unused

    Cprev   = np.zeros((NSPT, NC))
    Volprev = np.zeros(NC)
    EPprev  = np.zeros(NC)
    # phprev  = np.zeros(NC)  # unused after init

    # osmol = np.zeros(NC)  # unused

    num    = ND
    numpar = 18 + 4 * NSPT

    x    = np.zeros(num)
    fvec = np.zeros(num)
    fjac = np.zeros((num, num))
    wa1  = np.zeros(num)
    # wa2  = np.zeros(num)  # unused

    pars = np.zeros(numpar)

    # lwork = 10000          # reserved for future use (HKATPase matrix inversion)
    # ipiv  = np.zeros(ND)
    # work  = np.zeros(lwork)

    TOL = 1.0e-5

    # --- Initial guess: copy values from previous axial node (Lz) ---

    Cb[:NSPT, :NC-1]    = pt[Lz].conc[:NSPT, :NC-1]
    Cprev[:NSPT, :NC-1] = Cb[:NSPT, :NC-1]

    phb[:NC-1]    = pt[Lz].ph[:NC-1]
    Volb[:NC-1]    = pt[Lz].vol[:NC-1]
    Volprev[:NC-1] = Volb[:NC-1]

    EPb[:NC-1]    = pt[Lz].ep[:NC-1]
    EPprev[:NC-1] = EPb[:NC-1]

    PMb = PMprev = pt[Lz].pres

    # --- Newton iteration ---

    res    = 1.0
    iterat = 0

    while res > TOL and iterat < 21:
        res    = 0.0
        iterat += 1

        # --- Pack solution vector x: [LUM, P, LIS] interleaved per solute ---

        x[0:3*NSPT:3] = Cb[:NSPT, LUM]
        x[1:3*NSPT:3] = Cb[:NSPT, P]
        x[2:3*NSPT:3] = Cb[:NSPT, LIS]

        base        = 3 * NSPT
        x[base + 0] = Volb[LUM]
        x[base + 1] = Volb[P]
        x[base + 2] = Volb[LIS]
        x[base + 3] = EPb[LUM]
        x[base + 4] = EPb[P]
        x[base + 5] = EPb[LIS]
        x[base + 6] = PMb

        # --- Pack parameter vector pars ---
        # Layout:
        #   [0 : NSPT)           concentrations at Lz, LUM
        #   [NSPT : 2*NSPT)      concentrations at Lz, P
        #   [2*NSPT : 3*NSPT)    concentrations at Lz, LIS
        #   [3*NSPT : 4*NSPT)    concentrations at Lz, BATH
        #   [4*NSPT + 0..3]      volumes (LUM, P, LIS) and pressure at Lz
        #   [4*NSPT + 4..17]     step constants (volEinit, volPinit, dimLPT,
        #                        CPimpref, CPbuftot1, dkd/dkh, ncompl2, ntorq2, Vol0)

        pars[0:NSPT]        = pt[Lz].conc[:NSPT, LUM]
        pars[NSPT:2*NSPT]   = pt[Lz].conc[:NSPT, P]
        pars[2*NSPT:3*NSPT] = pt[Lz].conc[:NSPT, LIS]
        pars[3*NSPT:4*NSPT] = pt[Lz].conc[:NSPT, BATH]

        base            = 4 * NSPT
        pars[base + 0]  = pt[Lz].vol[LUM]
        pars[base + 1]  = pt[Lz].vol[P]
        pars[base + 2]  = pt[Lz].vol[LIS]
        pars[base + 3]  = pt[Lz].pres

        pars[base + 4]  = pt[Lz + 1].volEinit
        pars[base + 5]  = pt[Lz + 1].volPinit

        pars[base + 6]  = dimLPT if idid == 0 else 0.0

        pars[base + 7]  = CPimpref
        pars[base + 8]  = CPbuftot1

        pars[base + 9]  = pt[Lz + 1].dkd[LUM]
        pars[base + 10] = pt[Lz + 1].dkd[P]
        pars[base + 11] = pt[Lz + 1].dkd[LIS]

        pars[base + 12] = pt[Lz + 1].dkh[LUM]
        pars[base + 13] = pt[Lz + 1].dkh[P]
        pars[base + 14] = pt[Lz + 1].dkh[LIS]

        pars[base + 15] = float(ncompl2)
        pars[base + 16] = float(ntorq2)
        pars[base + 17] = Vol0[LUM]

        # --- Evaluate residual and Jacobian ---

        ldfjac = num
        iflag  = 1
        # ml     = num   # unused
        # mu     = num   # unused
        epsfcn = 1.0e-5

        fvec = fcn2PT(num, x, iflag, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        fjac, wa1, fvec_J = jacobi2_2PT(num, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

        # Solve J * AVF = fvec (more stable than explicit matrix inversion)
        AVF[:] = np.linalg.solve(fjac, fvec)

        # --- Update concentrations ---

        newLum = np.abs(Cprev[:NSPT, LUM] - AVF[0:3*NSPT:3])
        newP   = np.abs(Cprev[:NSPT, P]   - AVF[1:3*NSPT:3])
        newLIS = np.abs(Cprev[:NSPT, LIS] - AVF[2:3*NSPT:3])

        Cb[:NSPT, LUM] = newLum
        Cb[:NSPT, P]   = newP
        Cb[:NSPT, LIS] = newLIS

        denLum = np.maximum(np.abs(Cprev[:NSPT, LUM]), 1e-30)
        denP   = np.maximum(np.abs(Cprev[:NSPT, P]),   1e-30)
        denLIS = np.maximum(np.abs(Cprev[:NSPT, LIS]), 1e-30)

        res = max(res,
                  np.max(np.abs(newLum / denLum - 1.0)),
                  np.max(np.abs(newP   / denP   - 1.0)),
                  np.max(np.abs(newLIS / denLIS - 1.0)))

        # --- Update pH ---

        phb[LUM] = -np.log10(Cb[H, LUM] / 1.0e3)
        phb[P]   = -np.log10(Cb[H, P]   / 1.0e3)
        phb[LIS] = -np.log10(Cb[H, LIS] / 1.0e3)

        # --- Update volumes ---

        base      = 3 * NSPT
        Volb[LUM] = Volprev[LUM] - AVF[base + 0]
        Volb[P]   = Volprev[P]   - AVF[base + 1]
        Volb[LIS] = Volprev[LIS] - AVF[base + 2]

        res = max(res,
                  abs(Volb[LUM] - Volprev[LUM]),
                  abs(Volb[P]   - Volprev[P]),
                  abs(Volb[LIS] - Volprev[LIS]))

        # --- Update electrical potentials ---

        EPb[LUM] = EPprev[LUM] - AVF[base + 3]
        EPb[P]   = EPprev[P]   - AVF[base + 4]
        EPb[LIS] = EPprev[LIS] - AVF[base + 5]

        res = max(res,
                  1e-3 * abs(EPb[LUM] - EPprev[LUM]),
                  1e-3 * abs(EPb[P]   - EPprev[P]),
                  1e-3 * abs(EPb[LIS] - EPprev[LIS]))

        # --- Update pressure ---

        PMb = PMprev - AVF[base + 6]
        res = max(res, abs(PMb - PMprev))

        # --- Advance previous values ---

        Cprev[:, [LUM, P, LIS]]  = Cb[:, [LUM, P, LIS]]
        Volprev[[LUM, P, LIS]]   = Volb[[LUM, P, LIS]]
        EPprev[[LUM, P, LIS]]    = EPb[[LUM, P, LIS]]
        PMprev = PMb

        if iterat >= 11:
            if iterat < 20:
                TOL = 1.0e-3
            else:
                TOL = res * 1.010

    # --- Write converged solution into pt[Lz+1] ---

    cols = [LUM, P, LIS]
    pt[Lz+1].conc[:NSPT, cols] = Cb[:NSPT, cols]
    pt[Lz+1].ph[cols]          = phb[cols]
    pt[Lz+1].vol[cols]         = Volb[cols]
    pt[Lz+1].ep[cols]          = EPb[cols]
    pt[Lz+1].pres              = PMb

    return
