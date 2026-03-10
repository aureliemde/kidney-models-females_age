# Written originally in Fortran by Prof. Aurelie Edwards
# Translated to Python by Dr. Mohammad M. Tajdini

# Department of Biomedical Engineering
# Boston University

###################################################

import numpy as np

from values import *
from defs import *

from fcn2PT import fcn2PT


def jacobi2_2PT(n, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Compute dense approximate Jacobian for PT interior nodes (jz>0) using forward differences."""

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
        wa1 = fcn2PT(n, x, iflag, numpar, pars, pt, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        x[j] = temp  # restore

        fjac[:, j] = (wa1 - fvec_J) / h

    return fjac, wa1, fvec_J
