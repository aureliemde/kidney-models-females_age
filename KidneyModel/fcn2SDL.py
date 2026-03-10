"""
SDL (Short Descending Limb) residual function.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Residual function for SDL Newton-Raphson solver.
Computes the non-linear equations to be solved at any point below the SDL
inlet, determining luminal concentrations, volumes, and electrical potential.

SDL is slightly water permeable but completely solute impermeable.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *
from glo import *
from defs import *
from compute_sdl_water_fluxes import compute_sdl_water_fluxes


def fcn2SDL(n, x, iflag, numpar, pars, sdl, idid, Lz, PTinitVol):
    """Compute residual vector for SDL transport equations.

    Args
    n : int
        Number of unknowns (2 + NS2).
    x : np.ndarray
        Solution vector [concentrations, volume, pressure].
    iflag : int
        Function evaluation flag (not used).
    numpar : int
        Number of parameters.
    pars : np.ndarray
        Parameter vector from upstream position (Lz).
    sdl : list of Membrane
        SDL membrane array.
    idid : int
        Segment identifier (2 = SDL).
    Lz : int
        Current spatial position (0-based).
    PTinitVol : float
        Initial luminal volume (scalar) in cm³.

    Returns
    fvec : np.ndarray
        Residual vector [n].
    """
    state_a, state_b, geometry = _unpack_parameters(n, x, pars, sdl, Lz)
    Jvb = _compute_water_fluxes(state_b, geometry, sdl[Lz + 1], PTinitVol)
    fvec = _compute_residuals(state_a, state_b, geometry, Jvb, n)
    return fvec


def _unpack_parameters(n, x, pars, sdl, Lz):
    """Unpack parameter array and solution vector into state dictionaries.

    Returns

    state_a : dict
        Known upstream state at position Lz.
    state_b : dict
        Unknown downstream state at position Lz+1.
    geometry : dict
        Geometric parameters.
    """
    Ca = np.zeros((NS, NC))
    Vola = np.zeros(NC)

    Ca[:, LUM]  = pars[0:NS]
    Ca[:, P]    = pars[NS:2*NS]
    Ca[:, LIS]  = pars[2*NS:3*NS]
    Vola[LUM]   = pars[3*NS]
    Vola[P]     = pars[3*NS + 1]
    Vola[LIS]   = pars[3*NS + 2]
    PMa         = pars[3*NS + 3]
    Ca[:, BATH] = pars[3*NS + 4:4*NS + 4]

    dimL    = pars[4*NS + 4]
    CPimpref = pars[4*NS + 5]
    Diam    = pars[4*NS + 6]
    Vol0    = pars[4*NS + 7]

    Cb  = np.zeros((NS, NC))
    Volb = np.zeros(NC)
    EPb = np.zeros(NC)
    phb = np.zeros(NC)

    Cb[:NS2, LUM] = x[:NS2]
    Volb[LUM]     = x[NS2]
    PMb           = x[NS2 + 1]
    phb[LUM]      = -np.log10(abs(Cb[H, LUM] / 1000.0))

    Cb[:, BATH] = sdl[Lz + 1].conc[:, BATH]
    EPb[BATH]   = sdl[Lz + 1].ep[BATH]

    state_a = {'C': Ca, 'Vol': Vola, 'PM': PMa}
    state_b = {'C': Cb, 'Vol': Volb, 'EP': EPb, 'pH': phb, 'PM': PMb}
    geometry = {
        'length':    dimL,
        'diameter':  Diam,
        'area':      np.pi * (Diam ** 2) / 4.0,
        'perimeter': np.pi * Diam,
        'Vol0':      Vol0,
    }

    return state_a, state_b, geometry


def _compute_water_fluxes(state_b, geometry, membrane, PTinitVol):
    """Compute water fluxes across SDL membranes (slightly water permeable)."""
    return compute_sdl_water_fluxes(
        state_b['C'],
        state_b['PM'],
        state_b['Vol'],
        geometry['Vol0'],
        membrane.area,
        membrane.sig,
        membrane.dLPV,
        PTinitVol,
    )


def _compute_residuals(state_a, state_b, geometry, Jvb, n):
    """Compute all residual equations and return residual vector."""
    fvec = np.zeros(n)
    S = _compute_source_terms(state_a, state_b, Jvb)

    _set_nonreacting_residuals(fvec, S)
    _set_carbonate_residuals(fvec, S, state_b, geometry)
    _set_phosphate_residuals(fvec, S, state_b)
    _set_ammonia_residuals(fvec, S, state_b)
    _set_formate_residuals(fvec, S, state_b)
    _set_proton_residual(fvec, S)
    _set_volume_residual(fvec, state_a, state_b, geometry, Jvb)
    _set_pressure_residual(fvec, state_a, state_b, geometry)

    return fvec


def _compute_source_terms(state_a, state_b, Jvb):
    """Compute source terms S[i] = outflow - inflow for each solute.

    SDL is solute-impermeable so lateral solute flux is zero.
    """
    S = np.zeros(NS2)
    flow_factor = Vref / href

    for i in range(NS2):
        inflow  = state_a['Vol'][LUM] * state_a['C'][i, LUM] * flow_factor
        outflow = state_b['Vol'][LUM] * state_b['C'][i, LUM] * flow_factor
        S[i] = outflow - inflow   # lateral flux = 0 (solute impermeable)

    return S


def _set_nonreacting_residuals(fvec, S):
    """Set residuals for non-reacting solutes (Na, K, Cl, Urea, Glu, Ca)."""
    fvec[NA]   = S[NA]
    fvec[K]    = S[K]
    fvec[CL]   = S[CL]
    fvec[UREA] = S[UREA]
    fvec[GLU]  = S[GLU]
    fvec[CA]   = S[CA]


def _set_carbonate_residuals(fvec, S, state_b, geometry):
    """Set residuals for CO2/HCO3-/H2CO3 buffer system."""
    dkhuncat = 0.145
    dkduncat = 49.60

    fvec[HCO3]  = S[HCO3] + S[H2CO3] + S[CO2]
    fvec[H2CO3] = (state_b['pH'][LUM] - pKHCO3 -
                   np.log10(abs(state_b['C'][HCO3, LUM] / state_b['C'][H2CO3, LUM])))

    reaction_rate   = dkhuncat * state_b['C'][CO2, LUM] - dkduncat * state_b['C'][H2CO3, LUM]
    reaction_volume = geometry['area'] * geometry['length'] / NZ
    fvec[CO2] = S[CO2] + reaction_volume * reaction_rate / href


def _set_phosphate_residuals(fvec, S, state_b):
    """Set residuals for HPO4²⁻/H2PO4⁻ buffer system."""
    fvec[HPO4]  = S[HPO4] + S[H2PO4]
    fvec[H2PO4] = (state_b['pH'][LUM] - pKHPO4 -
                   np.log10(abs(state_b['C'][HPO4, LUM] / state_b['C'][H2PO4, LUM])))


def _set_ammonia_residuals(fvec, S, state_b):
    """Set residuals for NH3/NH4+ buffer system."""
    fvec[NH3] = S[NH3] + S[NH4]
    fvec[NH4] = (state_b['pH'][LUM] - pKNH3 -
                 np.log10(abs(state_b['C'][NH3, LUM] / state_b['C'][NH4, LUM])))


def _set_formate_residuals(fvec, S, state_b):
    """Set residuals for HCO2⁻/H2CO2 (formate) buffer system."""
    fvec[HCO2]  = S[HCO2] + S[H2CO2]
    fvec[H2CO2] = (state_b['pH'][LUM] - pKHCO2 -
                   np.log10(abs(state_b['C'][HCO2, LUM] / state_b['C'][H2CO2, LUM])))


def _set_proton_residual(fvec, S):
    """Set residual for proton (H+) charge balance."""
    fvec[H] = S[H] + S[NH4] - S[HCO3] - S[HPO4] - S[HCO2]


def _set_volume_residual(fvec, state_a, state_b, geometry, Jvb):
    """Set residual for volume (water) balance."""
    fvmult    = Pfref * Vwbar * Cref
    sumJvb    = Jvb[LUM, BATH] * fvmult
    lateral_water = geometry['perimeter'] * geometry['length'] * sumJvb / NZ

    fvola = state_a['Vol'][LUM] * Vref
    fvolb = state_b['Vol'][LUM] * Vref
    fvec[NS2] = fvolb - fvola + lateral_water


def _set_pressure_residual(fvec, state_a, state_b, geometry):
    """Set residual for pressure balance (Poiseuille flow)."""
    ratio   = 8.0 * visc / (np.pi * ((0.5 * geometry['diameter']) ** 4))
    Q       = state_b['Vol'][LUM] * Vref
    delta_L = geometry['length'] / NZ
    fvec[NS2 + 1] = state_b['PM'] - state_a['PM'] + ratio * Q * delta_L
