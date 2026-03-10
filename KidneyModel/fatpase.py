"""
H-K-ATPase kinetic matrix computation.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Constructs the 14×14 steady-state kinetic coefficient matrix for the
H-K-ATPase. Called by qflux2A (mTAL), qflux2T (cTAL), qflux2D (DCT),
qflux2C (CNT), qflux2CCD (CCD), and qflux2IMC (IMCD).

Kinetic units are M (molar). The 14 enzyme species are:
  1  K₂-E1       2  K₂-E1-ATP   3  K-E1-ATP    4  E1-ATP
  5  H-E1-ATP    6  H₂-E1-ATP   7  H₂-E1-P     8  H₂-E2-P
  9  H-E2-P     10  E2-P        11  K-E2-P      12  K₂-E2-P
 13  K₂-E2      14  K₂-E2-ATP
"""

import numpy as np
from numba import njit

from values import *


@njit
def fatpase(n, hkconc):
    """
    Build the H-K-ATPase steady-state kinetic matrix.

    Args:
        n:       Matrix dimension (= Natp = 14)
        hkconc:  Packed concentrations [4]:
                   [0] kin  = intracellular K⁺ (mM)
                   [1] kout = lumenal K⁺        (mM)
                   [2] hin  = intracellular H⁺  (mM)
                   [3] hout = lumenal H⁺         (mM)

    Returns:
        Amat: (n × n) kinetic coefficient matrix.
              Row 0 is the normalisation constraint (sum of all species = 1).
              Remaining rows are the steady-state ODEs for species 2–14.
              The caller inverts this matrix to obtain steady-state occupancies.
    """
    Amat = np.zeros((n, n))

    # Row 0: normalisation constraint — all species sum to 1
    for i in range(n):
        Amat[0, i] = 1.0

    # Reaction rate constants (units: M⁻¹s⁻¹ or s⁻¹)
    dkf1  = 1.30e7
    dkb1  = 6.50
    dkf2a = 8.90e3
    dkb2a = 7.30e4
    dkf2b = 8.90e3
    dkb2b = 7.30e4
    dkf3a = 5.30e9
    dkb3a = 6.60e2
    dkf3b = 5.30e9
    dkb3b = 6.60e2
    dkf4  = 5.0e1
    dkb4  = 2.50e6
    dkf5  = 4.0e1
    dkb5  = 2.0e2
    dkf6a = 5.0e7
    dkb6a = 8.0e12
    dkf6b = 5.0e7
    dkb6b = 8.0e12
    dkf7a = 2.60e10
    dkb7a = 1.50e8
    dkf7b = 2.60e10
    dkb7b = 1.50e8
    dkf8  = 5.40e1
    dkb8  = 3.20e1
    dkf9  = 1.75
    dkb9  = 3.50e1
    dkf10 = 5.0e4
    dkb10 = 5.0e1
    dkf11 = 5.0e2
    dkb11 = 5.0

    # Cellular metabolite concentrations (convert mM → M)
    catp = 2.0  * 1e-3
    cadp = 0.04 * 1e-3
    cpi  = 5.0  * 1e-3

    # Ionic concentrations (convert mM → M)
    kin  = hkconc[0] * 1e-3
    kout = hkconc[1] * 1e-3
    hin  = hkconc[2] * 1e-3
    hout = hkconc[3] * 1e-3

    # Steady-state ODE rows (species 2–14, rows 1–13)
    Amat[1,  1]  =  dkf2a
    Amat[1,  2]  = -(dkb2a * kin + dkf2b)
    Amat[1,  3]  =  dkb2b * kin
    Amat[2,  2]  =  dkf2b
    Amat[2,  3]  = -(dkb2b * kin + dkf3a * hin)
    Amat[2,  4]  =  dkb3a
    Amat[3,  3]  =  dkf3a * hin
    Amat[3,  4]  = -(dkb3a + dkf3b * hin)
    Amat[3,  5]  =  dkb3b
    Amat[4,  4]  =  dkf3b * hin
    Amat[4,  5]  = -(dkb3b + dkf4)
    Amat[4,  6]  =  dkb4 * cadp
    Amat[5,  5]  =  dkf4
    Amat[5,  6]  = -(dkb4 * cadp + dkf5)
    Amat[5,  7]  =  dkb5
    Amat[6,  6]  =  dkf5
    Amat[6,  7]  = -(dkb5 + dkf6a)
    Amat[6,  8]  =  dkb6a * hout
    Amat[7,  7]  =  dkf6a
    Amat[7,  8]  = -(dkb6a * hout + dkf6b)
    Amat[7,  9]  =  dkb6b * hout
    Amat[8,  8]  =  dkf6b
    Amat[8,  9]  = -(dkb6b * hout + dkf7a * kout)
    Amat[8,  10] =  dkb7a
    Amat[9,  9]  =  dkf7a * kout
    Amat[9,  10] = -(dkb7a + dkf7b * kout)
    Amat[9,  11] =  dkb7b
    Amat[10, 10] =  dkf7b * kout
    Amat[10, 11] = -(dkb7b + dkf8)
    Amat[10, 12] =  dkb8 * cpi
    Amat[11,  0] =  dkb9
    Amat[11, 11] =  dkf8
    Amat[11, 12] = -(dkb8 * cpi + dkf9 + dkf10 * catp)
    Amat[11, 13] =  dkb10
    Amat[12,  1] =  dkb11
    Amat[12, 12] =  dkf10 * catp
    Amat[12, 13] = -(dkb10 + dkf11)
    Amat[13,  0] =  dkf1 * catp
    Amat[13,  1] = -(dkb1 + dkf2a + dkb11)
    Amat[13,  2] =  dkb2a * kin
    Amat[13, 13] =  dkf11

    return Amat
