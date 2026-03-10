"""
SDL water flux computation.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes water fluxes across SDL membranes driven by:
    1. Hydraulic pressure differences (ΔP)
    2. Oncotic pressure from impermeants (ΔONC)
    3. Osmotic pressure from solutes (ΔOSM)

Water flux equation (Kedem-Katchalsky):
    J_v = L_p * (ΔP - σΔπ)
    
where:
    L_p = hydraulic conductivity
    σ = reflection coefficient
    Δπ = osmotic pressure difference

All pressures are non-dimensionalized by dividing by RT*Cref.
"""

import numpy as np
from numba import njit
from values import *
from glo import *
from defs import *


@njit
def compute_sdl_water_fluxes(
    C: np.ndarray,
    PM: float,
    Vol: np.ndarray,
    Vol0: float,
    area: np.ndarray,
    sig: np.ndarray,
    dLPV: np.ndarray,
    PTinitVol: float  # Scalar, not array!
) -> np.ndarray:
    """
    Compute water fluxes across SDL membranes.
    
    Physical Model:
        Water moves due to three driving forces:
        1. Hydraulic pressure gradient (ΔP)
        2. Oncotic pressure from impermeant solutes (ΔONC)
        3. Osmotic pressure from permeant solutes (ΔOSM)
        
    Flux equation:
        J_v = L_p * A * (ΔP - ΔONC - σ*ΔOSM)
        
    where:
        L_p = hydraulic permeability (dLPV)
        A = membrane area
        σ = reflection coefficient (sig)
        
    Non-dimensionalization:
        All pressures divided by RT*Cref to make dimensionless
        
    Args:
        C: Concentration array [NS x NC] in mM (already non-dimensional)
        PM: Luminal hydrostatic pressure in mmHg (dimensional)
        Vol: Volume array [NC] in cm³ (non-dimensional)
        Vol0: Initial luminal volume - SCALAR float in cm³ (not currently used)
        area: Membrane area array [NC x NC] in cm²
        sig: Reflection coefficient array [NS x NC x NC] (dimensionless)
        dLPV: Hydraulic permeability array [NC x NC] (non-dimensional)
        PTinitVol: Initial luminal volume - SCALAR float in cm³ (for oncotic pressure)
        
    Returns:
        Jvol: Water flux array [NC x NC] in cm³/s/cm² epith (non-dimensional)
              Jvol[i,j] = flux from compartment i to compartment j
              
    Note: SDL is slightly water permeable but solute impermeable.
    """
    # Initialize output
    Jvol = np.zeros((NC, NC))
    
    # Compute pressure components
    PRES = _compute_hydraulic_pressures(PM)
    ONC = _compute_oncotic_pressures(Vol, PTinitVol)
    OSM = _compute_osmotic_pressure(C, sig)
    
    # Compute water flux from lumen to bath
    Jvol[LUM, BATH] = _compute_lumen_to_bath_flux(
        area, dLPV, PRES, ONC, OSM
    )
    
    # Newton's third law: flux from bath to lumen is opposite
    Jvol[BATH, LUM] = -Jvol[LUM, BATH]
    
    return Jvol


@njit
def _compute_hydraulic_pressures(PM: float) -> np.ndarray:
    """
    Compute non-dimensional hydraulic pressures in each compartment.
    
    Hydraulic pressure drives bulk water flow through the membrane.
    
    Non-dimensionalization:
        P* = P / (RT*Cref)
        
    where:
        R = gas constant = 19300 mmHg·cm³/(mmol·K) for RTosm
        T = temperature (implicit in RTosm)
        Cref = reference concentration = 1 mM
        
    Args:
        PM: Luminal hydrostatic pressure [mmHg] (dimensional)
        
    Returns:
        PRES: Non-dimensional pressure array [NC]
              PRES[LUM] = luminal pressure
              PRES[BATH] = bath (peritubular) pressure
    """
    PRES = np.zeros(NC)
    
    # Luminal (M) hydrostatic pressure (non-dimensional)
    PRES[LUM] = PM / (RTosm * Cref)
    
    # Bath (S) hydrostatic pressure (non-dimensional)
    # PbloodPT is the peritubular capillary pressure [mmHg]
    PRES[BATH] = PbloodPT / (RTosm * Cref)
    
    return PRES


@njit
def _compute_oncotic_pressures(
    Vol: np.ndarray,
    PTinitVol: float  # Scalar!
) -> np.ndarray:
    """
    Compute non-dimensional oncotic pressures from impermeant solutes.
    
    Oncotic pressure arises from large molecules (proteins, impermeants)
    that cannot cross the membrane. It opposes hydraulic pressure.
    
    Van't Hoff equation:
        π = C * RT
        
    For impermeants with conservation of mass:
        C = n/V = (C₀*V₀)/V
        
    Therefore:
        π = C₀*V₀*RT/V
        
    Non-dimensional form:
        π* = C₀*V₀/V  (since RT*Cref cancels)
        
    Args:
        Vol: Current volumes [NC] in cm³ (non-dimensional by Vref)
        PTinitVol: Initial luminal volume - SCALAR float in cm³
        
    Returns:
        ONC: Non-dimensional oncotic pressure array [NC]
             ONC[LUM] = luminal oncotic pressure
             ONC[BATH] = bath oncotic pressure
    """
    ONC = np.zeros(NC)
    
    # Luminal oncotic pressure from impermeants
    # Impermeant concentration increases as volume decreases
    # LumImperm = initial impermeant concentration [mM]
    # Original code: ONC[1-1] = LumImperm * PTinitVol / Vol[1-1]
    # PTinitVol is SCALAR representing initial luminal volume
    ONC[LUM] = LumImperm * PTinitVol / Vol[LUM]
    
    # Bath oncotic pressure (constant, proteins in blood)
    # BathImperm = plasma protein concentration [mM equivalent]
    ONC[BATH] = BathImperm
    
    return ONC


@njit
def _compute_osmotic_pressure(C: np.ndarray, sig: np.ndarray) -> float:
    """
    Compute effective osmotic pressure difference from permeant solutes.
    
    Osmotic pressure from permeant solutes depends on:
        1. Concentration difference across membrane
        2. Reflection coefficient σ (how well membrane reflects solute)
        
    Effective osmotic pressure:
        Δπ_eff = RT * Σ σᵢ * ΔCᵢ
        
    where:
        σᵢ = reflection coefficient for solute i (0 = freely permeable, 1 = impermeable)
        ΔCᵢ = concentration difference for solute i
        
    Non-dimensional form:
        Δπ* = Σ σᵢ * (Cᵢ_lumen - Cᵢ_bath)
        
    Args:
        C: Concentration array [NS x NC] in mM (non-dimensional)
        sig: Reflection coefficient array [NS x NC x NC]
             sig[i, LUM, BATH] = reflection coefficient for solute i
             
    Returns:
        OSM: Total effective osmotic pressure difference (non-dimensional)
             Positive value indicates higher osmolarity in lumen
             
    Physical Interpretation:
        - If σ = 0: solute crosses freely, no osmotic effect
        - If σ = 1: solute is reflected, full osmotic effect
        - SDL is mostly solute-impermeable (σ ≈ 1 for most solutes)
    """
    OSM = 0.0
    
    # Sum over all solutes (first NS2 are the main transported solutes)
    for i in range(NS2):
        # Concentration difference: lumen - bath
        dC = C[i, LUM] - C[i, BATH]
        
        # Weighted by reflection coefficient at lumen-bath interface
        OSM += sig[i, LUM, BATH] * dC
    
    return OSM


@njit
def _compute_lumen_to_bath_flux(
    area: np.ndarray,
    dLPV: np.ndarray,
    PRES: np.ndarray,
    ONC: np.ndarray,
    OSM: float
) -> float:
    """
    Compute water flux from lumen to bath.
    
    Kedem-Katchalsky equation:
        J_v = L_p * A * (ΔP - Δπ)
        
    where:
        ΔP = hydraulic pressure difference
        Δπ = total osmotic pressure difference (oncotic + osmotic)
        
    Driving forces (positive = favors lumen → bath):
        + ΔP_hydraulic = P_lumen - P_bath
        - ΔP_oncotic = ONC_lumen - ONC_bath
        - ΔP_osmotic = OSM (already computed as lumen - bath)
        
    SDL has two parallel pathways:
        1. Through principal cells (M → P → S)
        2. Through lateral space (M → E → S)
        
    Total conductance = sum of parallel conductances:
        L_p_total = (A_MP * L_p_MP + A_ME * L_p_ME)
        
    Args:
        area: Membrane area array [NC x NC] in cm²
        dLPV: Hydraulic permeability array [NC x NC] (non-dimensional)
        PRES: Hydraulic pressure array [NC]
        ONC: Oncotic pressure array [NC]
        OSM: Osmotic pressure difference (non-dimensional)
        
    Returns:
        Jvol: Water flux from lumen to bath [cm³/s/cm² epith]
              Positive = flow out of lumen into bath
              
    Physical Interpretation:
        - High luminal pressure → positive flux (filtration)
        - High luminal osmolarity → negative flux (absorption)
        - High plasma protein (ONC_bath) → positive flux (filtration)
    """
    # Hydraulic conductance through parallel pathways
    # Pathway 1: Lumen (M) to Principal cell (P)
    conductance_MP = area[LUM, P] * dLPV[LUM, P]
    
    # Pathway 2: Lumen (M) to Lateral space (E)
    conductance_ME = area[LUM, LIS] * dLPV[LUM, LIS]
    
    # Total conductance (parallel pathways add)
    conductance_total = conductance_MP + conductance_ME
    
    # Net driving force = ΔP - ΔONC - ΔOSM
    # All terms are already non-dimensional
    driving_force = (
        (PRES[LUM] - PRES[BATH]) -    # Hydraulic pressure drives flow
        (ONC[LUM] - ONC[BATH]) -       # Oncotic pressure opposes flow
        OSM                             # Osmotic pressure opposes flow
    )
    
    # Water flux = conductance * driving force
    Jvol = conductance_total * driving_force
    
    return Jvol