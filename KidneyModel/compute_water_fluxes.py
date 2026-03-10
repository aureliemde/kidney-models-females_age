"""
Osmotic water flux computation (shared utility).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes osmotic water fluxes across all membrane pairs for any tubule
segment. Called by every qflux function (PT, SDL, mTAL, cTAL, DCT,
CNT, CCD, OMCD, IMCD).

Hydraulic and oncotic pressures are non-dimensionalised by RT*Cref.
"""

import numpy as np
from numba import njit

from values import *
from glo import *
from defs import *


@njit
def compute_water_fluxes(C, PM, PB, Vol, Vol0, VolEinit, VolPinit, CPimpref,
                         VolAinit, CAimpref, VolBinit, CBimpref, area, sig, dLPV,
                         compl, PTinitVol):
    """
    Compute osmotic water fluxes across all membrane pairs.

    Args:
        C:          Concentration array [NS x NC]
        PM:         Luminal hydrostatic pressure (dimensional)
        PB:         Bath pressure (unused — interface placeholder)
        Vol:        Volume array [NC]
        Vol0:       Reference luminal volume (unused — interface placeholder)
        VolEinit:   Initial LIS volume
        VolPinit:   Initial P-cell volume
        CPimpref:   Impermeable solute reference concentration in P cell
        VolAinit:   Initial A-cell volume (unused — ONC[A] = 0)
        CAimpref:   Impermeable solute concentration in A cell (unused)
        VolBinit:   Initial B-cell volume (unused — ONC[B] = 0)
        CBimpref:   Impermeable solute concentration in B cell (unused)
        area:       Membrane surface area array [NC x NC]
        sig:        Reflection coefficient array [NS x NC x NC]
        dLPV:       Hydraulic permeability array [NC x NC]
        compl:      LIS compliance (cm H₂O⁻¹)
        PTinitVol:  Initial PT luminal volume (for LumImperm oncotic term)

    Returns:
        Jvol: Water flux array [NC x NC] (antisymmetric)
    """
    Jvol = np.zeros((NC, NC))
    PRES = np.zeros(NC)
    ONC  = np.zeros(NC)

    # ---------- HYDRAULIC PRESSURES ----------
    # Lumen and cells share luminal pressure PM; bath uses blood pressure
    PRES[LUM]  = PM / (RTosm * Cref)
    PRES[P]    = PRES[LUM]
    PRES[A]    = PRES[LUM]
    PRES[B]    = PRES[LUM]
    PRES[BATH] = PbloodPT / (RTosm * Cref)
    # LIS pressure: bath pressure + compliance term
    PRES[LIS]  = PRES[BATH] + (Vol[LIS] / VolEinit - 1.0) / compl / (RTosm * Cref)

    # ---------- ONCOTIC PRESSURES ----------
    ONC[LUM]  = LumImperm * PTinitVol / Vol[LUM]
    ONC[P]    = CPimpref * VolPinit / Vol[P]
    ONC[A]    = 0.0
    ONC[B]    = 0.0
    ONC[LIS]  = 0.0
    ONC[BATH] = BathImperm

    # ---------- WATER FLUXES ----------
    for k in range(NC - 1):
        for l in range(k + 1, NC):
            OSM = 0.0
            for j in range(NS2):
                # Exclude formate (HCO2) and formic acid (H2CO2) from osmotic sum
                if j != HCO2 and j != H2CO2:
                    OSM += sig[j, k, l] * (C[j, k] - C[j, l])

            Jvol[k, l] = area[k, l] * dLPV[k, l] * ((PRES[k] - PRES[l]) - (ONC[k] - ONC[l]) - OSM)
            Jvol[l, k] = -Jvol[k, l]  # antisymmetry

    return Jvol
