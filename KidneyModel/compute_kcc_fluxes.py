"""
KCC4 cotransporter flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes K⁺, Cl⁻, and NH₄⁺ fluxes across basolateral KCC4
(K⁺-Cl⁻ cotransporter isoform 4). Called by qflux2A (mTAL) and
qflux2T (cTAL).

Kinetic scheme: ordered binding with cellular (P) and extracellular
(LIS or BATH) half-reactions. NH₄⁺ substitutes for K⁺ at the K⁺
binding site. Separate flux terms computed for lateral (P→LIS) and
basal (P→BATH) membranes.
"""

import numpy as np
from numba import njit

from values import *
from glo import *


@njit
def compute_kcc_fluxes(C, area5, area6, xKCC4):
    """
    Compute KCC4 cotransporter fluxes (P→LIS and P→BATH).

    Args:
        C:     Concentration array [NS x NC]
        area5: Lateral membrane area (P-LIS interface)
        area6: Basal membrane area (P-BATH interface)
        xKCC4: Expression scaling factor

    Returns:
        dJk5: K⁺ flux across lateral membrane (P→LIS)
        dJc5: Cl⁻ flux across lateral membrane (P→LIS)
        dJm5: NH₄⁺ flux across lateral membrane (P→LIS)
        dJk6: K⁺ flux across basal membrane (P→BATH)
        dJc6: Cl⁻ flux across basal membrane (P→BATH)
        dJm6: NH₄⁺ flux across basal membrane (P→BATH)
    """
    # Normalised occupancy factors for each compartment
    betP    = C[K,   P]    / bkkcc
    betLIS  = C[K,   LIS]  / bkkcc
    betBATH = C[K,   BATH] / bkkcc
    gamP    = C[CL,  P]    / bckcc
    gamLIS  = C[CL,  LIS]  / bckcc
    gamBATH = C[CL,  BATH] / bckcc
    dnuP    = C[NH4, P]    / bmkcc
    dnuLIS  = C[NH4, LIS]  / bmkcc
    dnuBATH = C[NH4, BATH] / bmkcc

    sigP    = 1.0 + gamP    * (1.0 + betP    + dnuP)
    sigLIS  = 1.0 + betLIS  * (1.0 + gamLIS)  + dnuLIS  * (1.0 + gamLIS)
    sigBATH = 1.0 + betBATH * (1.0 + gamBATH) + dnuBATH * (1.0 + gamBATH)

    rhoP    = poppkcc + pkccpp * betP    * gamP    + pmccpp * dnuP    * gamP
    rhoLIS  = popkcc  + pkccp  * betLIS  * gamLIS  + pmccp  * dnuLIS  * gamLIS
    rhoBATH = popkcc  + pkccp  * betBATH * gamBATH + pmccp  * dnuBATH * gamBATH

    bigsumLIS  = sigLIS  * rhoP + sigP * rhoLIS
    bigsumBATH = sigBATH * rhoP + sigP * rhoBATH

    # Flux across lateral membrane (P→LIS)
    t1LIS = poppkcc * pkccp  * betLIS  * gamLIS  - popkcc  * pkccpp * betP    * gamP
    t2LIS = pmccpp  * dnuP   * gamP    * pkccp   * betLIS  * gamLIS  - pmccp  * dnuLIS  * gamLIS  * pkccpp * betP * gamP
    t3LIS = poppkcc * pmccp  * dnuLIS  * gamLIS  - popkcc  * pmccpp * dnuP    * gamP
    t4LIS = pmccp   * dnuLIS * gamLIS  * pkccpp  * betP    * gamP    - pmccpp * dnuP    * gamP    * pkccp  * betLIS * gamLIS

    dJk5 = -xKCC4 * area5 * (t1LIS + t2LIS) / bigsumLIS
    dJm5 = -xKCC4 * area5 * (t3LIS + t4LIS) / bigsumLIS
    dJc5 = dJk5 + dJm5

    # Flux across basal membrane (P→BATH)
    t1BATH = poppkcc * pkccp  * betBATH * gamBATH - popkcc  * pkccpp * betP    * gamP
    t2BATH = pmccpp  * dnuP   * gamP    * pkccp   * betBATH * gamBATH - pmccp  * dnuBATH * gamBATH * pkccpp * betP * gamP
    t3BATH = poppkcc * pmccp  * dnuBATH * gamBATH - popkcc  * pmccpp * dnuP    * gamP
    t4BATH = pmccp   * dnuBATH * gamBATH * pkccpp * betP    * gamP    - pmccpp * dnuP    * gamP    * pkccp  * betBATH * gamBATH

    dJk6 = -xKCC4 * area6 * (t1BATH + t2BATH) / bigsumBATH
    dJm6 = -xKCC4 * area6 * (t3BATH + t4BATH) / bigsumBATH
    dJc6 = dJk6 + dJm6

    return dJk5, dJc5, dJm5, dJk6, dJc6, dJm6
