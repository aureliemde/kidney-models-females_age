"""
SGLT1/SGLT2 flux calculator.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes fluxes across Na-glucose cotransporters (SGLT1 or SGLT2) using the
kinetic model of Parent et al. (J Memb Biol 1992) with modifications from
Eskandari et al. (J Memb Biol 2005). Reaction rates for SGLT1 from Wright et al.
(Physiol Rev 2011); for SGLT2 from Mackenzie (J Biol Chem 1996).

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np
from numba import njit

from values import *


@njit
def sglt(n, nagluparam, area):
    """Compute Na+ and glucose fluxes across SGLT1 (n=1) or SGLT2 (n=2).

    Parameters
    ----------
    n          : int   -- transporter type: 1 = SGLT1, 2 = SGLT2
    nagluparam : array -- [nao, nai, gluo, glui, epo, epi, CT] (concentrations
                          in mM, potentials in non-dimensional units, CT = max
                          transporter density)
    area       : float -- membrane surface area (cm²)

    Returns
    -------
    fluxsglt : ndarray, shape (2,) -- [Na+ flux, glucose flux]
    """
    # fluxsglt = np.zeros(2, dtype=np.float64)  # reassigned below; kept for reference

    # --- Assign concentrations (convert mM → M) and potentials ---

    nao  = nagluparam[0] * 1e-3
    nai  = nagluparam[1] * 1e-3
    gluo = nagluparam[2] * 1e-3
    glui = nagluparam[3] * 1e-3
    epo  = nagluparam[4]
    epi  = nagluparam[5]
    CT   = nagluparam[6]

    pot = F * (epi - epo) * EPref / RT

    # --- Reaction rates ---

    if n == 1:
        # SGLT1 rates from Wright et al., Physiol Rev 2011
        delta   = 0.70
        alphap  = 0.30
        alphapp = 0.00
        dk12 = 140000.0 * np.exp(-pot * alphap) * (nao ** 2)
        dk21 = 300.0    * np.exp(pot * alphap)
        dk23 = 45000.0  * gluo
        dk32 = 20.0
        dk34 = 50.0
        dk43 = 50.0
        dk45 = 800.0
        dk54 = 190000.0 * glui
        dk56 = 5.0      * np.exp(-pot * alphapp)
        dk65 = 2250.0   * np.exp(pot * alphapp) * (nai ** 2)
        dk61 = 25.0     * np.exp(-pot * delta)
        dk16 = 600.0    * np.exp(pot * delta)
        dk25 = 0.01
        dk52 = 0.0005

        nstoich = 2

    elif n == 2:
        # SGLT2 rates from Mackenzie, J Biol Chem 1996
        delta   = 0.70
        alphap  = 0.30
        alphapp = 0.00
        dk12 = 20000.0 * np.exp(-pot * alphap  / 2.0) * nao
        dk21 = 400.0   * np.exp(pot * alphap   / 2.0)
        dk23 = 10000.0 * gluo
        dk32 = 20.0
        dk34 = 50.0
        dk43 = 50.0
        dk45 = 800.0
        dk54 = 6700000.0 * glui
        dk56 = 48.0    * np.exp(-pot * alphapp / 2.0)
        dk65 = 50.0    * np.exp(pot * alphapp  / 2.0) * nai
        dk61 = 35.0    * np.exp(-pot * delta   / 2.0)
        dk16 = 100.0   * np.exp(pot * delta    / 2.0)
        dk25 = 0
        dk52 = 0

        nstoich = 1

    # --- KAT terms: fractional occupancy of each enzyme state ---

    delC1 = (dk21*dk32*dk43*dk52*dk61 + dk21*dk32*dk45*dk52*dk61 + dk21*dk34*dk45*dk52*dk61 +
             dk21*dk32*dk43*dk54*dk61 + dk21*dk32*dk43*dk56*dk61 + dk25*dk32*dk43*dk56*dk61 +
             dk21*dk32*dk45*dk56*dk61 + dk25*dk32*dk45*dk56*dk61 + dk21*dk34*dk45*dk56*dk61 +
             dk23*dk34*dk45*dk56*dk61 + dk25*dk34*dk45*dk56*dk61 + dk21*dk32*dk43*dk52*dk65 +
             dk21*dk32*dk45*dk52*dk65 + dk21*dk34*dk45*dk52*dk65 + dk21*dk32*dk43*dk54*dk65)

    delC2 = (dk12*dk32*dk43*dk52*dk61 + dk12*dk32*dk45*dk52*dk61 + dk12*dk34*dk45*dk52*dk61 +
             dk12*dk32*dk43*dk54*dk61 + dk12*dk32*dk43*dk56*dk61 + dk12*dk32*dk45*dk56*dk61 +
             dk12*dk34*dk45*dk56*dk61 + dk12*dk32*dk43*dk52*dk65 + dk16*dk32*dk43*dk52*dk65 +
             dk12*dk32*dk45*dk52*dk65 + dk16*dk32*dk45*dk52*dk65 + dk12*dk34*dk45*dk52*dk65 +
             dk16*dk34*dk45*dk52*dk65 + dk12*dk32*dk43*dk54*dk65 + dk16*dk32*dk43*dk54*dk65)

    delC3 = (dk12*dk23*dk43*dk52*dk61 + dk12*dk23*dk45*dk52*dk61 + dk12*dk23*dk43*dk54*dk61 +
             dk12*dk25*dk43*dk54*dk61 + dk12*dk23*dk43*dk56*dk61 + dk12*dk23*dk45*dk56*dk61 +
             dk12*dk23*dk43*dk52*dk65 + dk16*dk23*dk43*dk52*dk65 + dk12*dk23*dk45*dk52*dk65 +
             dk16*dk23*dk45*dk52*dk65 + dk16*dk21*dk43*dk54*dk65 + dk12*dk23*dk43*dk54*dk65 +
             dk16*dk23*dk43*dk54*dk65 + dk12*dk25*dk43*dk54*dk65 + dk16*dk25*dk43*dk54*dk65)

    delC4 = (dk12*dk23*dk34*dk52*dk61 + dk12*dk25*dk32*dk54*dk61 + dk12*dk23*dk34*dk54*dk61 +
             dk12*dk25*dk34*dk54*dk61 + dk12*dk23*dk34*dk56*dk61 + dk12*dk23*dk34*dk52*dk65 +
             dk16*dk23*dk34*dk52*dk65 + dk16*dk21*dk32*dk54*dk65 + dk12*dk25*dk32*dk54*dk65 +
             dk16*dk25*dk32*dk54*dk65 + dk16*dk21*dk34*dk54*dk65 + dk12*dk23*dk34*dk54*dk65 +
             dk16*dk23*dk34*dk54*dk65 + dk12*dk25*dk34*dk54*dk65 + dk16*dk25*dk34*dk54*dk65)

    delC5 = (dk12*dk25*dk32*dk43*dk61 + dk12*dk25*dk32*dk45*dk61 + dk12*dk23*dk34*dk45*dk61 +
             dk12*dk25*dk34*dk45*dk61 + dk16*dk21*dk32*dk43*dk65 + dk12*dk25*dk32*dk43*dk65 +
             dk16*dk25*dk32*dk43*dk65 + dk16*dk21*dk32*dk45*dk65 + dk12*dk25*dk32*dk45*dk65 +
             dk16*dk25*dk32*dk45*dk65 + dk16*dk21*dk34*dk45*dk65 + dk12*dk23*dk34*dk45*dk65 +
             dk16*dk23*dk34*dk45*dk65 + dk12*dk25*dk34*dk45*dk65 + dk16*dk25*dk34*dk45*dk65)

    delC6 = (dk16*dk21*dk32*dk43*dk52 + dk16*dk21*dk32*dk45*dk52 + dk16*dk21*dk34*dk45*dk52 +
             dk16*dk21*dk32*dk43*dk54 + dk16*dk21*dk32*dk43*dk56 + dk12*dk25*dk32*dk43*dk56 +
             dk16*dk25*dk32*dk43*dk56 + dk16*dk21*dk32*dk45*dk56 + dk12*dk25*dk32*dk45*dk56 +
             dk16*dk25*dk32*dk45*dk56 + dk16*dk21*dk34*dk45*dk56 + dk12*dk23*dk34*dk45*dk56 +
             dk16*dk23*dk34*dk45*dk56 + dk12*dk25*dk34*dk45*dk56 + dk16*dk25*dk34*dk45*dk56)

    SumdelC = delC1 + delC2 + delC3 + delC4 + delC5 + delC6

    C1 = delC1 / SumdelC
    C2 = delC2 / SumdelC
    C3 = delC3 / SumdelC
    C4 = delC4 / SumdelC
    C5 = delC5 / SumdelC
    C6 = delC6 / SumdelC

    # --- Na+ and glucose fluxes ---

    fluxsglt = np.zeros(2)
    fluxsglt[0] = area * CT * nstoich * (dk34 * C3 - dk43 * C4 + dk25 * C2 - dk52 * C5)
    fluxsglt[1] = area * CT * 1.0     * (dk34 * C3 - dk43 * C4)

    return fluxsglt
