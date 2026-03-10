"""
NCX exchanger flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes Ca²⁺ fluxes across the basolateral NCX (Na⁺/Ca²⁺ exchanger)
at the P→LIS and P→BATH interfaces. Called by qflux2D (DCT) and
qflux2C (CNT).

Notation:
  - i  = intracellular (P cell)
  - o5 = extracellular LIS  (lateral membrane)
  - o6 = extracellular BATH (basal membrane)

Stoichiometry: 3 Na⁺ exchanged for 1 Ca²⁺ per cycle.
"""

import numpy as np

from values import *


def compute_ncx_fluxes(var_ncx):
    """
    Compute NCX exchanger Ca²⁺ fluxes (P→LIS and P→BATH).

    Args:
        var_ncx: Packed parameter array [16]:
            [0]  nai   = C[NA, P]       Na⁺ intracellular
            [1]  nao5  = C[NA, LIS]     Na⁺ LIS
            [2]  nao6  = C[NA, BATH]    Na⁺ BATH
            [3]  cai   = C[CA, P]       Ca²⁺ intracellular
            [4]  cao5  = C[CA, LIS]     Ca²⁺ LIS
            [5]  cao6  = C[CA, BATH]    Ca²⁺ BATH
            [6]  epi   = EP[P]          Potential intracellular
            [7]  epo5  = EP[LIS]        Potential LIS
            [8]  epo6  = EP[BATH]       Potential BATH
            [9]  area5 = area[P, LIS]   Lateral membrane area
            [10] area6 = area[P, BATH]  Basal membrane area
            [11] xNCX                   Expression scaling factor
            [12-15]                     Unused placeholders

    Returns:
        dJNCXca5: Ca²⁺ flux across lateral membrane (P→LIS)
        dJNCXca6: Ca²⁺ flux across basal membrane   (P→BATH)
    """
    nai   = var_ncx[0]
    nao5  = var_ncx[1]
    nao6  = var_ncx[2]
    cai   = var_ncx[3]
    cao5  = var_ncx[4]
    cao6  = var_ncx[5]
    epi   = var_ncx[6]
    epo5  = var_ncx[7]
    epo6  = var_ncx[8]
    area5 = var_ncx[9]
    area6 = var_ncx[10]
    xNCX  = var_ncx[11]

    # Intracellular Ca²⁺ activation (Hill-type, n=2)
    fmod = (cai / dKm_ncx)**2 / (1.0 + (cai / dKm_ncx)**2)

    # Voltage-dependent factors (gamma_ncx is the fraction of membrane
    # voltage seen by the Ca²⁺ binding site)
    phir5 = np.exp((gamma_ncx - 1.0) * F * EPref / RT * (epi - epo5))
    phir6 = np.exp((gamma_ncx - 1.0) * F * EPref / RT * (epi - epo6))
    phif5 = np.exp( gamma_ncx        * F * EPref / RT * (epi - epo5))
    phif6 = np.exp( gamma_ncx        * F * EPref / RT * (epi - epo6))

    # Denominator terms (kinetic denominators for each interface)
    g5 = (  nao5**3 * cai
          + nai**3  * cao5
          + dKmNao**3 * cai
          + nai**3  * dKmCao
          + dKmNai**3 * cao5 * (1.0 + cai  / dKmCai)
          + nao5**3 * dKmCai * (1.0 + (nai / dKmNai)**3))

    g6 = (  nao6**3 * cai
          + nai**3  * cao6
          + dKmNao**3 * cai
          + nai**3  * dKmCao
          + dKmNai**3 * cao6 * (1.0 + cai  / dKmCai)
          + nao6**3 * dKmCai * (1.0 + (nai / dKmNai)**3))

    # Forward/reverse flux factors
    fac5 = (phir5 * nao5**3 * cai - phif5 * nai**3 * cao5) / g5 / (1 + dKsat_ncx * phir5)
    fac6 = (phir6 * nao6**3 * cai - phif6 * nai**3 * cao6) / g6 / (1 + dKsat_ncx * phir6)

    dJNCXca5 = area5 * xNCX * fmod * fac5
    dJNCXca6 = area6 * xNCX * fmod * fac6

    return dJNCXca5, dJNCXca6
