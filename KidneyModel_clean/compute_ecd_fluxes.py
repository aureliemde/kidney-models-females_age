"""
Electro-convective-diffusive (ECD) flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes electrodiffusive and convective solute fluxes across all
membrane pairs for any tubule segment. Called by qflux functions that
use the standard ECD formulation (mTAL, cTAL, IMCD, etc.).

Note: The DCT uses an inline equivalent with eps=1e-6 and no
C[i,l] > 0 guard, to match its original Fortran implementation.
"""

import numpy as np
from numba import njit

from values import *
from glo import *


@njit
def compute_ecd_fluxes(C, EP, area, sig, h, Jvol):
    """
    Compute electro-convective-diffusive solute fluxes across all membrane pairs.

    For each (i, k→l) pair the flux has two components:
      - Electrodiffusive: Goldman-Hodgkin-Katz (GHK) form, using the
        Peclet parameter XI = z*F*EPref/RT * (EP[k] - EP[l]).
        When |1 - exp(-XI)| < eps the linear (diffusion-only) limit is used.
      - Convective: log-mean concentration times water flux, weighted by
        (1 - sigma) where sigma is the reflection coefficient.

    Args:
        C:    Concentration array [NS x NC]
        EP:   Electrical potential array [NC]
        area: Membrane surface area array [NC x NC]
        sig:  Reflection coefficient array [NS x NC x NC]
        h:    Permeability array [NS x NC x NC]
        Jvol: Water flux array [NC x NC]

    Returns:
        Jsol:  Solute flux array [NS x NC x NC] (upper triangle filled)
        delmu: Electrochemical potential difference array [NS x NC x NC]
    """
    Jsol  = np.zeros((NS, NC, NC))
    delmu = np.zeros((NS, NC, NC))
    dmu   = np.zeros((NS, NC))
    eps   = 1.0e-6

    # Electrochemical potential: dmu[i,k] = RT*ln|C[i,k]| + z_i*F*EPref*EP[k]
    # Concentration is clamped to eps to avoid log(0)
    for i in range(NS):
        for k in range(NC):
            cval = np.abs(C[i, k])
            if cval < eps:
                cval = eps
            dmu[i, k] = RT * np.log(cval) + zval[i] * F * EPref * EP[k]

    dimless = (Pfref * Vwbar * Cref) / href

    for i in range(NS):
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                # Electrodiffusive component (GHK)
                XI   = zval[i] * F * EPref / RT * (EP[k] - EP[l])
                dint = np.exp(-XI)
                if np.abs(1.0 - dint) < eps:
                    flux = area[k, l] * h[i, k, l] * (C[i, k] - C[i, l])
                else:
                    flux = area[k, l] * h[i, k, l] * XI * (C[i, k] - C[i, l] * dint) / (1.0 - dint)

                # Convective component (log-mean concentration)
                concdiff = C[i, k] - C[i, l]
                if np.abs(concdiff) > eps and C[i, l] > 0:
                    concmean = concdiff / np.log(np.abs(C[i, k] / C[i, l]))
                    flux += (1.0 - sig[i, k, l]) * concmean * Jvol[k, l] * dimless

                Jsol[i, k, l]  = flux
                delmu[i, k, l] = dmu[i, k] - dmu[i, l]

    return Jsol, delmu
