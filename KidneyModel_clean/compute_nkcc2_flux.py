"""
NKCC2 cotransporter flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes Na⁺, K⁺, Cl⁻, and NH₄⁺ fluxes across apical NKCC2
(Na-K-2Cl cotransporter) isoforms. Called by qflux2A (F and A
isoforms) and qflux2T (B and A isoforms).

Kinetic scheme: ordered binding with lumenal (1) and cellular (2)
half-reactions. NH₄⁺ substitutes for K⁺ at the K⁺ binding site.
"""

import numpy as np
from numba import njit

from values import *
from glo import *


@njit
def compute_nkcc2_flux(C, area, xNKCC2, bn2, bk2, bc2, bm2,
                       popnkcc, pnkccp, pnmccp, poppnkcc, pnkccpp, pnmccpp):
    """
    Compute NKCC2 cotransporter fluxes (LUM→P).

    Args:
        C:       Concentration array [NS x NC]
        area:    Apical membrane area (LUM-P interface)
        xNKCC2:  Isoform-specific expression scaling factor
        bn2:     Na⁺ binding affinity
        bk2:     K⁺ binding affinity
        bc2:     Cl⁻ binding affinity
        bm2:     NH₄⁺ binding affinity
        popnkcc, pnkccp, pnmccp:    Lumenal-side rate constants
        poppnkcc, pnkccpp, pnmccpp: Cellular-side rate constants

    Returns:
        dJnNKCC2: Na⁺ flux
        dJkNKCC2: K⁺ flux
        dJcNKCC2: Cl⁻ flux  (= 2 * dJnNKCC2, stoichiometry 1:1:2)
        dJmNKCC2: NH₄⁺ flux (substitutes for K⁺)
    """
    # Normalised occupancy factors — lumenal (1) and cellular (2) sides
    alp1 = C[NA,  LUM] / bn2
    alp2 = C[NA,  P]   / bn2
    bet1 = C[K,   LUM] / bk2
    bet2 = C[K,   P]   / bk2
    gam1 = C[CL,  LUM] / bc2
    gam2 = C[CL,  P]   / bc2
    dnu1 = C[NH4, LUM] / bm2
    dnu2 = C[NH4, P]   / bm2

    sig1 = 1.0 + alp1 + alp1 * gam1 * (1.0 + bet1 + bet1 * gam1 + dnu1 + dnu1 * gam1)
    sig2 = 1.0 + gam2 * (1.0 + bet2 + bet2 * gam2 + bet2 * gam2 * alp2 + dnu2 + dnu2 * gam2 + dnu2 * gam2 * alp2)

    rho1 = popnkcc  + pnkccp  * alp1 * bet1 * gam1**2 + pnmccp  * alp1 * dnu1 * gam1**2
    rho2 = poppnkcc + pnkccpp * alp2 * bet2 * gam2**2 + pnmccpp * alp2 * dnu2 * gam2**2

    bigsum = sig1 * rho2 + sig2 * rho1

    # Flux numerator terms
    t1 = poppnkcc * pnkccp  * alp1 * bet1 * gam1**2 - popnkcc * pnkccpp * alp2 * bet2 * gam2**2
    t2 = poppnkcc * pnmccp  * alp1 * dnu1 * gam1**2 - popnkcc * pnmccpp * alp2 * dnu2 * gam2**2
    t3 = pnmccpp  * alp2 * dnu2 * gam2**2 * pnkccp  * alp1 * bet1 * gam1**2
    t4 = pnmccp   * alp1 * dnu1 * gam1**2 * pnkccpp * alp2 * bet2 * gam2**2

    dJnNKCC2 = xNKCC2 * area * (t1 + t2)       / bigsum
    dJkNKCC2 = xNKCC2 * area * (t1 + t3 - t4)  / bigsum
    dJmNKCC2 = xNKCC2 * area * (t2 + t4 - t3)  / bigsum
    dJcNKCC2 = 2 * dJnNKCC2  # stoichiometry: 2 Cl⁻ per cycle

    return dJnNKCC2, dJkNKCC2, dJcNKCC2, dJmNKCC2
