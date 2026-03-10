"""
Set interstitial concentrations for nephron segments.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This subroutine sets bath (interstitial) concentrations along a tubule segment
based on the cortical-medullary osmotic gradient. Concentrations are interpolated
between:
    - ireg=1: Cortex (constant values)
    - ireg=2: Outer Medulla (cortex → OM-IM junction)
    - ireg=3: Inner Medulla (OM-IM junction → papillary tip)

All buffer systems maintain chemical equilibrium at pH 7.323.
"""

import numpy as np
from typing import List
from dataclasses import dataclass
from values import *
from glo import *
from defs import *


@dataclass
class BufferEquilibrium:
    """Pre-computed equilibrium factors for buffer systems at given pH."""
    facpho: float   # HPO4²⁻/H2PO4⁻ equilibrium factor
    facamm: float   # NH3/NH4⁺ equilibrium factor
    fachco2: float  # HCO2⁻/H2CO2 equilibrium factor
    facbic: float   # HCO3⁻/H2CO3 equilibrium factor
    h_mM: float     # H⁺ concentration in mM


def set_intconc(
    tube: List[Membrane],
    iN: int,
    ireg: int,
    pos: np.ndarray
) -> None:
    """
    Set interstitial (bath) concentrations for tubule segments.
    
    Args:
        tube: List of Membrane objects
        iN: Number of segments (0 to iN inclusive)
        ireg: Region identifier
              1 = Cortex (constant concentrations)
              2 = Outer Medulla (linear gradient from cortex to OM-IM boundary)
              3 = Inner Medulla (linear gradient from OM-IM to papillary tip)
        pos: Position array [0 to 1] for each segment
             0 = proximal boundary, 1 = distal boundary of region
    """
    # Ensure pos is numpy array
    pos = np.asarray(pos, dtype=float)
    
    # All regions use same pH
    pH_bath = 7.323
    
    # Compute buffer equilibrium factors once (same for all regions)
    buffers = _compute_buffer_equilibrium(pH_bath)
    
    # Set concentrations based on region
    if ireg == 1:
        _set_cortex_concentrations(tube, iN, pos, pH_bath, buffers)
    elif ireg == 2:
        _set_outer_medulla_concentrations(tube, iN, pos, pH_bath, buffers)
    elif ireg == 3:
        _set_inner_medulla_concentrations(tube, iN, pos, pH_bath, buffers)
    else:
        raise ValueError(f"Invalid region ID: {ireg}. Must be 1, 2, or 3.")


def _compute_buffer_equilibrium(pH_bath: float) -> BufferEquilibrium:
    """
    Compute equilibrium distribution factors for all buffer systems.
    
    For a weak acid HA ⇌ H⁺ + A⁻ at equilibrium:
        [A⁻]/[HA] = 10^(pH - pKa)
        
    Therefore:
        [A⁻] = Total * (10^(pH-pKa))/(1 + 10^(pH-pKa))
        [HA] = Total * 1/(1 + 10^(pH-pKa))
    
    Args:
        pH_bath: Bath pH value
        
    Returns:
        BufferEquilibrium object with pre-computed factors
    """
    log10 = np.log(10.0)
    
    return BufferEquilibrium(
        facpho=np.exp(log10 * (pH_bath - pKHPO4)),   # HPO4²⁻/H2PO4⁻
        facamm=np.exp(log10 * (pH_bath - pKNH3)),    # NH3/NH4⁺
        fachco2=np.exp(log10 * (pH_bath - pKHCO2)),  # HCO2⁻/H2CO2
        facbic=np.exp(log10 * (pH_bath - pKHCO3)),   # HCO3⁻/H2CO3
        h_mM=np.exp(-log10 * pH_bath) * 1e3          # [H⁺] in mM
    )


def _set_cortex_concentrations(
    tube: List[Membrane],
    iN: int,
    pos: np.ndarray,
    pH_bath: float,
    buffers: BufferEquilibrium
) -> None:
    """
    Set cortical interstitial concentrations (constant except ammonia).
    
    Cortex has relatively constant concentrations except for ammonia,
    which varies linearly from apex (TotAmmCT) to base (TotAmmCM).
    
    Args:
        tube: Membrane array
        iN: Last segment index
        pos: Position array for interpolation
        pH_bath: Bath pH
        buffers: Pre-computed buffer factors
    """
    # Set pH for first segment (reference)
    tube[0].ph[BATH] = pH_bath
    
    # Pre-compute constant buffer speciation
    pho_hpo4 = TotPhoCM * buffers.facpho / (1.0 + buffers.facpho)
    pho_h2po4 = TotPhoCM / (1.0 + buffers.facpho)
    
    hco2_hco2 = TotHco2CM * buffers.fachco2 / (1.0 + buffers.fachco2)
    hco2_h2co2 = TotHco2CM / (1.0 + buffers.fachco2)
    
    # Ammonia varies with position (cortical gradient)
    amm_tot = TotAmmCT + (TotAmmCM - TotAmmCT) * pos
    
    for j in range(iN + 1):
        m = tube[j]
        m.ph[BATH] = pH_bath
        
        # Get view to bath column for efficient assignment
        c = m.conc[:, BATH]
        
        # Non-reacting ions (constant)
        c[NA] = TotSodCM
        c[K] = TotPotCM
        c[CL] = TotCloCM  # Will be adjusted for electroneutrality
        c[HCO3] = TotBicCM
        c[H2CO3] = TotHcoCM
        c[CO2] = TotCo2CM
        
        # Phosphate buffer (constant)
        c[HPO4] = pho_hpo4
        c[H2PO4] = pho_h2po4
        
        # Urea (constant)
        c[UREA] = TotureaCM
        
        # Ammonia buffer (position-dependent)
        # Adjust chloride to account for ammonium production
        c[CL] += (amm_tot[j] - TotAmmCM)
        c[NH3] = amm_tot[j] * buffers.facamm / (1.0 + buffers.facamm)
        c[NH4] = amm_tot[j] / (1.0 + buffers.facamm)
        
        # Proton
        c[H] = buffers.h_mM
        
        # Formate buffer (constant)
        c[HCO2] = hco2_hco2
        c[H2CO2] = hco2_h2co2
        
        # Glucose and calcium (constant)
        c[GLU] = TotgluCM
        c[CA] = TotCaCM
        
        # Enforce electroneutrality by adjusting chloride
        charge_imbalance = np.dot(zval[:NS], c[:NS])
        c[CL] += charge_imbalance


def _set_outer_medulla_concentrations(
    tube: List[Membrane],
    iN: int,
    pos: np.ndarray,
    pH_bath: float,
    buffers: BufferEquilibrium
) -> None:
    """
    Set outer medullary interstitial concentrations.
    
    Concentrations vary linearly from cortical values (pos=0)
    to OM-IM junction values (pos=1).
    
    Args:
        tube: Membrane array
        iN: Last segment index
        pos: Position array [0=cortex, 1=OM-IM boundary]
        pH_bath: Bath pH
        buffers: Pre-computed buffer factors
    """
    tube[0].ph[BATH] = pH_bath
    
    # Vectorized interpolation (compute all positions at once)
    conc_profile = _interpolate_concentrations(
        pos,
        start_values={
            'Na': TotSodCM, 'K': TotPotCM, 'Cl': TotCloCM,
            'HCO3': TotBicCM, 'H2CO3': TotHcoCM, 'CO2': TotCo2CM,
            'Pho': TotPhoCM, 'Urea': TotureaCM, 'Amm': TotAmmCM,
            'HCO2': TotHco2CM, 'Glu': TotgluCM, 'Ca': TotCaCM
        },
        end_values={
            'Na': TotSodOI, 'K': TotPotOI, 'Cl': TotCloOI,
            'HCO3': TotBicOI, 'H2CO3': TotHcoOI, 'CO2': TotCo2OI,
            'Pho': TotPhoOI, 'Urea': TotureaOI, 'Amm': TotAmmOI,
            'HCO2': TotHco2OI, 'Glu': TotGluOI, 'Ca': TotCaOI
        }
    )
    
    # Apply to all segments
    _apply_concentration_profile(tube, iN, conc_profile, pH_bath, buffers)


def _set_inner_medulla_concentrations(
    tube: List[Membrane],
    iN: int,
    pos: np.ndarray,
    pH_bath: float,
    buffers: BufferEquilibrium
) -> None:
    """
    Set inner medullary interstitial concentrations.
    
    Concentrations vary linearly from OM-IM junction (pos=0)
    to papillary tip (pos=1).
    
    Args:
        tube: Membrane array
        iN: Last segment index
        pos: Position array [0=OM-IM junction, 1=papillary tip]
        pH_bath: Bath pH
        buffers: Pre-computed buffer factors
    """
    tube[0].ph[BATH] = pH_bath
    
    # Vectorized interpolation
    conc_profile = _interpolate_concentrations(
        pos,
        start_values={
            'Na': TotSodOI, 'K': TotPotOI, 'Cl': TotCloOI,
            'HCO3': TotBicOI, 'H2CO3': TotHcoOI, 'CO2': TotCo2OI,
            'Pho': TotPhoOI, 'Urea': TotureaOI, 'Amm': TotAmmOI,
            'HCO2': TotHco2OI, 'Glu': TotGluOI, 'Ca': TotCaOI
        },
        end_values={
            'Na': TotSodPap, 'K': TotPotPap, 'Cl': TotCloPap,
            'HCO3': TotBicPap, 'H2CO3': TotHcoPap, 'CO2': TotCo2Pap,
            'Pho': TotPhoPap, 'Urea': TotureaPap, 'Amm': TotAmmPap,
            'HCO2': TotHco2Pap, 'Glu': TotGluPap, 'Ca': TotCaPap
        }
    )
    
    # Apply to all segments
    _apply_concentration_profile(tube, iN, conc_profile, pH_bath, buffers)


def _interpolate_concentrations(
    pos: np.ndarray,
    start_values: dict,
    end_values: dict
) -> dict:
    """
    Linear interpolation of all concentrations.
    
    C(pos) = C_start + (C_end - C_start) * pos
    
    Args:
        pos: Position array [0 to 1]
        start_values: Dictionary of concentrations at pos=0
        end_values: Dictionary of concentrations at pos=1
        
    Returns:
        Dictionary of interpolated concentration arrays
    """
    profile = {}
    for key in start_values:
        profile[key] = start_values[key] + (end_values[key] - start_values[key]) * pos
    return profile


def _apply_concentration_profile(
    tube: List[Membrane],
    iN: int,
    conc_profile: dict,
    pH_bath: float,
    buffers: BufferEquilibrium
) -> None:
    """
    Apply pre-computed concentration profile to all segments.
    
    Args:
        tube: Membrane array
        iN: Last segment index
        conc_profile: Dictionary of concentration arrays
        pH_bath: Bath pH
        buffers: Pre-computed buffer factors
    """
    for j in range(iN + 1):
        m = tube[j]
        m.ph[BATH] = pH_bath
        
        c = m.conc[:, BATH]
        
        # Simple ions
        c[NA] = conc_profile['Na'][j]
        c[K] = conc_profile['K'][j]
        c[CL] = conc_profile['Cl'][j]
        c[HCO3] = conc_profile['HCO3'][j]
        c[H2CO3] = conc_profile['H2CO3'][j]
        c[CO2] = conc_profile['CO2'][j]
        
        # Phosphate buffer speciation
        pho_tot = conc_profile['Pho'][j]
        c[HPO4] = pho_tot * buffers.facpho / (1.0 + buffers.facpho)
        c[H2PO4] = pho_tot / (1.0 + buffers.facpho)
        
        # Urea
        c[UREA] = conc_profile['Urea'][j]
        
        # Ammonia buffer speciation
        amm_tot = conc_profile['Amm'][j]
        c[NH3] = amm_tot * buffers.facamm / (1.0 + buffers.facamm)
        c[NH4] = amm_tot / (1.0 + buffers.facamm)
        
        # Proton
        c[H] = buffers.h_mM
        
        # Formate buffer speciation
        hco2_tot = conc_profile['HCO2'][j]
        c[HCO2] = hco2_tot * buffers.fachco2 / (1.0 + buffers.fachco2)
        c[H2CO2] = hco2_tot / (1.0 + buffers.fachco2)
        
        # Glucose and calcium
        c[GLU] = conc_profile['Glu'][j]
        c[CA] = conc_profile['Ca'][j]
        
        # Enforce electroneutrality
        charge_imbalance = np.dot(zval[:NS], c[:NS])
        c[CL] += charge_imbalance