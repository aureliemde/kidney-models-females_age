"""
SDL (Short Descending Limb) Jacobian computation.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Jacobian computation for SDL Newton solver using forward finite differences.

J[i,j] ≈ (F(x + h*e_j) - F(x)) / h

Step size h is adaptive:
  h = eps * |x_j|  if x_j ≠ 0
  h = eps          if x_j = 0

where eps = sqrt(max(epsfcn, machine_epsilon)).

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *
from defs import *
from fcn2SDL import fcn2SDL


def jacobi2_2SDL(n, x, fvec, ldfjac, iflag, epsfcn, numpar, pars, sdl, idid, Lz, PTinitVol):
    """Compute Jacobian matrix using forward finite differences.

    Args:
    n : int
        Number of unknowns.
    x : np.ndarray
        Solution vector [n] (perturbed temporarily, restored after each column).
    fvec : np.ndarray
        Residual vector F(x) [n].
    ldfjac : int
        Leading dimension of Jacobian (= n).
    iflag : int
        Function evaluation flag (passed to fcn2SDL).
    epsfcn : float
        User-specified step size parameter (typically 1e-5).
    numpar : int
        Number of parameters for function evaluation.
    pars : np.ndarray
        Parameter array.
    sdl : list of Membrane
        SDL membrane array.
    idid : int
        Segment identifier.
    Lz : int
        Current spatial position.
    PTinitVol : float
        Initial luminal volume (scalar) in cm³.

    Returns
    fjac : np.ndarray
        Jacobian matrix [n x n].
    wa1 : np.ndarray
        F(x + h*e_n) work array from last column.
    fvec_J : np.ndarray
        Copy of input residual vector.
    """
    epsmch = np.finfo(float).eps
    eps    = np.sqrt(max(epsfcn, epsmch))

    fjac   = np.zeros((ldfjac, n))
    fvec_J = fvec.copy()

    for j in range(n):
        h = _compute_step_size(x[j], eps)

        x_original = x[j]
        x[j]       = x_original + h

        wa1 = fcn2SDL(n, x, iflag, numpar, pars, sdl, idid, Lz, PTinitVol)

        x[j]         = x_original
        fjac[:, j]   = (wa1 - fvec_J) / h

    return fjac, wa1, fvec_J


def _compute_step_size(x_j, eps):
    """Compute adaptive finite-difference step size for variable x_j."""
    h = eps * abs(x_j)
    if h == 0.0:
        h = eps
    return h
