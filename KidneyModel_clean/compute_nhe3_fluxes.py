"""
NHE3 exchanger flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes Na⁺, H⁺, and NH₄⁺ fluxes across the apical NHE3
(Na⁺/H⁺ exchanger isoform 3). Called by qflux2A (mTAL), qflux2T
(cTAL), and qflux2D (DCT).

Kinetic scheme: ordered binding.
  - p  (prime)        = luminal compartment
  - pp (double prime) = cytosolic compartment
NH₄⁺ substitutes for H⁺ at the H⁺ binding site.
"""

import numpy as np
from numba import njit

from values import *


@njit
def compute_nhe3_fluxes(C, area, xNHE3):
    """
    Compute NHE3 exchanger fluxes (LUM→P).

    Args:
        C:     Concentration array [NS x NC]
        area:  Apical membrane area (LUM-P interface)
        xNHE3: Expression scaling factor

    Returns:
        dJNHEsod:  Na⁺ flux (into cell)
        dJNHEprot: H⁺ flux  (out of cell)
        dJNHEamm:  NH₄⁺ flux (out of cell, substituting for H⁺)
    """
    # Lumenal (p) and cytosolic (pp) concentrations
    ap  = C[NA,  LUM]   # Na⁺ lumen
    bp  = C[H,   LUM]   # H⁺  lumen
    cp  = C[NH4, LUM]   # NH₄⁺ lumen
    app = C[NA,  P]     # Na⁺ cell
    bpp = C[H,   P]     # H⁺  cell
    cpp = C[NH4, P]     # NH₄⁺ cell

    alp    = ap  / dKaNH
    alpp   = app / dKaNH
    betap  = bp  / dKbNH
    betapp = bpp / dKbNH
    gamp   = cp  / dKcNH
    gampp  = cpp / dKcNH

    fmod = fMNH * C[H, P] / (C[H, P] + dKINH)

    sum1    = (1 + alp  + betap  + gamp)  * (PaNH * alpp  + PbNH * betapp + PcNH * gampp)
    sum2    = (1 + alpp + betapp + gampp) * (PaNH * alp   + PbNH * betap  + PcNH * gamp)
    sum_tot = sum1 + sum2

    termNaH   = fmod * PaNH * PbNH * (alp * betapp - alpp * betap)
    termNaNH4 = fmod * PaNH * PcNH * (alp * gampp  - alpp * gamp)
    termHNH4  = fmod * PbNH * PcNH * (betap * gampp - betapp * gamp)

    dJNHEsod  = area * xNHE3 * ( termNaH  + termNaNH4) / sum_tot
    dJNHEprot = area * xNHE3 * (-termNaH  + termHNH4)  / sum_tot
    dJNHEamm  = area * xNHE3 * (-termNaNH4 - termHNH4) / sum_tot

    return dJNHEsod, dJNHEprot, dJNHEamm
