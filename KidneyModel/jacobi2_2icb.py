"""Jacobian matrix computation for the CNT and CCD Newton-Raphson solver.

Approximates the Jacobian of F(x) by forward finite differences,
perturbing each element of x in turn and re-evaluating the residual
via fcn2C.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Used for segments:
    idid=6: CNT (Connecting Tubule)
    idid=7: CCD (Cortical Collecting Duct)
"""

import numpy as np

from values import *
from defs import *
from fcn2C import fcn2C


def jacobi2_2icb(n, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute the dense finite-difference Jacobian of the CNT/CCD residual.

    Args:
        n:          Length of the solution/residual vector
        x:          Current solution vector (perturbed in-place, then restored)
        fvec:       Residual vector at x (unperturbed baseline)
        ldfjac:     Leading dimension of the Jacobian array (= n)
        iflag:      Solver flag (passed through to fcn2C)
        epsfcn:     Step-size control parameter for finite differences
        numpar:     Length of the parameter vector
        pars:       Upstream state vector (read-only)
        pt:         List of Membrane objects for the segment
        idid:       Segment identifier (6=CNT, 7=CCD)
        Lz:         Current spatial position index (upstream node)
        PTinitVol:  Initial PT luminal volume
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity

    Returns:
        fjac:   np.ndarray of shape (ldfjac, n) — approximate Jacobian
        wa1:    np.ndarray — residual at last perturbed point (reused by caller)
        fvec_J: np.ndarray — unperturbed residual (alias of fvec)
    """
    fvec_J = fvec
    fjac   = np.zeros((ldfjac, n))

    epsmch = 2.22044604926e-16
    eps    = np.sqrt(max(epsfcn, epsmch))

    # Forward finite-difference approximation of the Jacobian:
    # fjac[:, j] = (F(x + h*e_j) - F(x)) / h
    for j in range(n):
        temp   = x[j]
        h      = eps * abs(temp)
        if h == 0.0:
            h = eps

        x[j] = temp + h
        wa1  = fcn2C(n, x, iflag, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        x[j] = temp  # restore

        fjac[:, j] = (wa1 - fvec_J) / h

    return fjac, wa1, fvec_J
