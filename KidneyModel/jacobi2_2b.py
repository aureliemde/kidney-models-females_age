"""
Numerical Jacobian computation for the mTAL/cTAL Newton solver.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes a dense finite-difference approximation of the Jacobian of
fcn2b. Called by qnewton2b for the mTAL (A) and cTAL (T) segments.
"""

import numpy as np

from values import *
from defs import *
from fcn2b import fcn2b


def jacobi2_2b(n, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid,
               Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """
    Compute a dense finite-difference Jacobian of fcn2b.

    Each column j is computed by perturbing x[j] by eps*|x[j]| and
    evaluating fcn2b at the perturbed point.

    Args:
        n:       Problem dimension
        x:       Current solution vector [n]
        fvec:    Residual at x (fcn2b(x))
        ldfjac:  Leading dimension of fjac (>= n)
        iflag:   Flag passed through to fcn2b
        epsfcn:  Relative step size for finite differences
        numpar, pars, pt, idid: Solver bookkeeping passed to fcn2b
        Lz:      Spatial segment index
        PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT: PT parameters

    Returns:
        fjac:    Approximate Jacobian matrix [ldfjac x n]
        wa1:     fcn2b evaluation at the last perturbed point
        fvec_J:  Alias of fvec (residual at x)
    """
    fvec_J = fvec
    fjac   = np.zeros((ldfjac, n))

    epsmch = 2.22044604926e-16
    eps    = np.sqrt(max(epsfcn, epsmch))

    # Dense finite-difference Jacobian
    for j in range(n):
        temp = x[j]
        h    = eps * np.abs(temp)
        if h == 0.0:
            h = eps

        x[j] = temp + h
        wa1  = fcn2b(n, x, iflag, numpar, pars, pt, idid,
                     Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        x[j] = temp  # restore

        fjac[:, j] = (wa1 - fvec_J) / h

    return fjac, wa1, fvec_J
