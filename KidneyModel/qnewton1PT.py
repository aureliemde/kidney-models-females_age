"""
PT (Proximal Tubule) Newton solver — inlet node (jz = 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module implements the Newton-Raphson solver for the PT inlet cross-section
(jz = 0). It determines steady-state concentrations, volumes, and electrical
potentials in the lumen and epithelial compartments.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

from fcn1PT import fcn1PT
from jacobi2_1PT import jacobi2_1PT


def qnewton1PT(pt, CPimpref, CPbuftot1, idid, ncompl2, ntorq2, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Newton-Raphson solver for the PT segment inlet node (jz = 0).

    Iterates until the relative change in concentrations, volumes, and
    potentials drops below TOL (1e-5). Tolerance is relaxed progressively
    after iteration 11 to aid convergence in stiff cases.
    """
    # Initialize arrays
    AVF     = np.zeros(NUPT)
    Cprev   = np.zeros((NSPT, NC))
    Volprev = np.zeros(NC)
    EPprev  = np.zeros(NC)
    osmol   = np.zeros(NC)

    num    = NUPT   # number of unknowns = 37
    numpar = 10

    pars = np.zeros(numpar)
    x    = np.zeros(NUPT)
    fvec = np.zeros(num)
    fjac = np.zeros((num, num))
    wa1  = np.zeros(num)
    # wa2 = np.zeros(num)  # declared in original but never used

    TOL = 1.0e-5

    # Store initial concentrations, volumes, potentials, and pH
    Cprev   = pt[0].conc[:NSPT, :].copy()
    Volprev = pt[0].vol.copy()
    EPprev  = pt[0].ep.copy()
    phprev  = pt[0].ph.copy()

    # Newton iteration
    res    = 1.0
    iterat = 0

    while res > TOL and iterat < 21:
        res    = 0.0
        iterat += 1

        # Pack solution vector x: even slots = P concentrations, odd = LIS
        x[0:2*NSPT:2] = pt[0].conc[:NSPT, P]
        x[1:2*NSPT:2] = pt[0].conc[:NSPT, LIS]

        base        = 2 * NSPT
        x[base + 0] = pt[0].vol[P]
        x[base + 1] = pt[0].vol[LIS]
        x[base + 2] = pt[0].ep[LUM]
        x[base + 3] = pt[0].ep[P]
        x[base + 4] = pt[0].ep[LIS]

        # Settings for finite-difference Jacobian
        ldfjac = num
        iflag  = 1
        epsfcn = 1.0e-5

        pars[0] = pt[0].volEinit
        pars[1] = pt[0].volPinit
        pars[2] = CPimpref
        pars[3] = CPbuftot1
        pars[4] = pt[0].dkd[P]
        pars[5] = pt[0].dkd[LIS]
        pars[6] = pt[0].dkh[P]
        pars[7] = pt[0].dkh[LIS]
        pars[8] = float(ncompl2)
        pars[9] = float(ntorq2)

        fvec = fcn1PT(num, x, iflag, numpar, pars, pt, idid, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        fjac, wa1, fvec_J = jacobi2_1PT(num, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

        # Solve J * AVF = fvec (more stable than explicit matrix inversion)
        AVF[:] = np.linalg.solve(fjac, fvec)

        # Update concentrations (Newton correction); concentrations must be positive
        newP   = np.abs(Cprev[:NSPT, P]   - AVF[0:2*NSPT:2])
        newLIS = np.abs(Cprev[:NSPT, LIS] - AVF[1:2*NSPT:2])

        pt[0].conc[:NSPT, P]   = newP
        pt[0].conc[:NSPT, LIS] = newLIS

        denP = np.maximum(np.abs(Cprev[:NSPT, P]),   1e-30)
        denL = np.maximum(np.abs(Cprev[:NSPT, LIS]), 1e-30)
        res  = max(res,
                   np.max(np.abs(newP   / denP - 1.0)),
                   np.max(np.abs(newLIS / denL - 1.0)))

        # Update volumes
        pt[0].vol[P]   = Volprev[P]   - AVF[base + 0]
        pt[0].vol[LIS] = Volprev[LIS] - AVF[base + 1]
        res = max(res, abs(pt[0].vol[P]   - Volprev[P]))
        res = max(res, abs(pt[0].vol[LIS] - Volprev[LIS]))

        # Update electrical potentials
        pt[0].ep[LUM] = EPprev[LUM] - AVF[base + 2]
        pt[0].ep[P]   = EPprev[P]   - AVF[base + 3]
        pt[0].ep[LIS] = EPprev[LIS] - AVF[base + 4]
        res = max(res, abs(pt[0].ep[LUM] - EPprev[LUM]))
        res = max(res, abs(pt[0].ep[P]   - EPprev[P]))
        res = max(res, abs(pt[0].ep[LIS] - EPprev[LIS]))

        # Update previous values for next iteration
        Cprev[:NSPT, P]   = pt[0].conc[:NSPT, P]
        Cprev[:NSPT, LIS] = pt[0].conc[:NSPT, LIS]

        pt[0].ph[P]   = -np.log10(pt[0].conc[H, P]   / 1.0e3)
        pt[0].ph[LIS] = -np.log10(pt[0].conc[H, LIS] / 1.0e3)

        phprev[P]   = pt[0].ph[P]
        phprev[LIS] = pt[0].ph[LIS]

        Volprev[P]   = pt[0].vol[P]
        Volprev[LIS] = pt[0].vol[LIS]

        EPprev[LUM] = pt[0].ep[LUM]
        EPprev[P]   = pt[0].ep[P]
        EPprev[LIS] = pt[0].ep[LIS]

        if iterat >= 11:
            if iterat < 20:
                TOL = 1.0e-3
            else:
                TOL = res * 1.010

    # Final bookkeeping
    VolLumInit = pt[0].vol[LUM]

    osmol[LUM]  = LumImperm
    osmol[P]    = CPimpref * pt[0].volPinit / pt[0].vol[P]
    osmol[LIS]  = 0.0
    osmol[BATH] = BathImperm

    es    = pt[0].ep[BATH]
    elecM = 0.0

    return
