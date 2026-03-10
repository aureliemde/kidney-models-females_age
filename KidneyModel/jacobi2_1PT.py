"""
PT (Proximal Tubule) Jacobian — inlet node (jz = 0).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module computes the dense approximate Jacobian for the PT inlet
cross-section (jz = 0) using forward finite differences. Called by
qnewton1PT at each Newton iteration.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *
from defs import *

from fcn1PT import fcn1PT


def jacobi2_1PT(n, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute dense approximate Jacobian for PT inlet node (jz=0) using forward differences."""

    fvec_J = fvec
    fjac = np.zeros((ldfjac, n))

    epsmch = 2.22044604926e-16
    eps = np.sqrt(max(epsfcn, epsmch))

    for j in range(n):
        temp = x[j]
        h = eps * np.abs(temp)
        if h == 0.0:
            h = eps

        x[j] = temp + h
        wa1 = fcn1PT(n, x, iflag, numpar, pars, pt, idid, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        x[j] = temp

        fjac[:, j] = (wa1 - fvec_J) / h

    return fjac, wa1, fvec_J
