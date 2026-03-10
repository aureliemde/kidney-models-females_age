"""
SDL (Short Descending Limb) initialization.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Sets up geometry, membrane permeabilities, reflection coefficients,
boundary conditions, and metabolic parameters for the SDL segment.
Called from main.py; returns nothing.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2005)
"""

import numpy as np

from values import *
from glo import *
from defs import *
from set_intconc import set_intconc


def initSDL(sdl):
    """Initialize SDL segment parameters including geometry, permeabilities,
    reflection coefficients, boundary conditions, and metabolic parameters.

    Args:
    sdl : list of Membrane
        SDL cell array (length NZ+1).
    """
    _set_membrane_areas(sdl)
    _set_water_permeabilities(sdl)
    _set_reflection_coefficients(sdl)
    _set_boundary_conditions(sdl)
    _set_metabolic_parameters(sdl)

    # electroneutrality check (informational only; values not used)
    # elecM = np.sum(zval * sdl[0].conc[:, LUM])
    # elecS = np.sum(zval * sdl[0].conc[:, BATH])


def _set_membrane_areas(sdl):
    """Set membrane surface areas for SDL.

    Surface areas in cm²/cm² epithelium (rat kidney, AMW model AJP Renal 2005).
    A and B cells use the same area as P (not differentiated in SDL).
    """
    SlumPinitsdl = 2.0
    SbasPinitsdl = 2.0
    SlatPinitsdl = 10.0
    SlumEinitsdl = 0.001
    # SbasEinitsdl = 0.020   # defined but not assigned to p.sbasEinit in SDL

    for membrane in sdl:
        membrane.area[LUM, P]    = SlumPinitsdl
        membrane.area[LUM, A]    = SlumPinitsdl   # same as P in SDL
        membrane.area[LUM, B]    = SlumPinitsdl   # same as P in SDL
        membrane.area[LUM, LIS]  = SlumEinitsdl
        membrane.area[P,   LIS]  = SlatPinitsdl
        membrane.area[A,   LIS]  = SlatPinitsdl
        membrane.area[B,   LIS]  = SlatPinitsdl
        membrane.area[P,   BATH] = SbasPinitsdl
        membrane.area[A,   BATH] = SbasPinitsdl
        membrane.area[B,   BATH] = SbasPinitsdl
        membrane.area += membrane.area.T  # symmetric copy


def _set_water_permeabilities(sdl):
    """Set water permeability coefficients (non-dimensional dLPV = Pf/Pfref).

    Units of dimensional water flux: cm³/s/cm² epith.
    Non-dimensional factor: (Pfref)*Vwbar*Cref.
    """
    # First set of Pf values (not used, kept for reference)
    # PfMP = 33.0e-4 / 2.0
    # PfME = 0.70e-4 / 0.0010
    # PfPE = 170.0e-4 / 10.0
    # PfPS = 0.1 * 33.0e-4 / 2.0
    # PfES = 8000.0e-4 / 0.020

    PfMP = 0.40 / 36.0
    PfMA = 0.0
    PfMB = 0.0
    PfME = 0.22 / 0.0010
    PfPE = 0.40 / 36.0
    PfAE = 0.0
    PfBE = 0.0
    PfPS = PfPE
    PfAS = 0.0
    PfBS = 0.0
    PfES = 6.60 / 0.020

    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = PfMP
    Pf[LUM, A]    = PfMA
    Pf[LUM, B]    = PfMB
    Pf[LUM, LIS]  = PfME
    Pf[P,   LIS]  = PfPE
    Pf[A,   LIS]  = PfAE
    Pf[B,   LIS]  = PfBE
    Pf[P,   BATH] = PfPS
    Pf[A,   BATH] = PfAS
    Pf[B,   BATH] = PfBS
    Pf[LIS, BATH] = PfES

    dLPV = Pf / Pfref
    for membrane in sdl:
        membrane.dLPV[:] = dLPV

    _apply_terminal_impermeability(sdl)


def _apply_terminal_impermeability(sdl):
    """Make terminal 54% of SDL water-impermeable (from 46% onwards).

    Reflects the physiological transition to thin ascending limb.
    """
    n_segments = len(sdl)
    terminal_start_idx = int(n_segments * 0.46)
    impermeability_factor = 0.001  # reduce permeability by 1000x

    impermeable_pairs = [
        (LUM, P),
        (LUM, LIS),
        (P,   LIS),
        (P,   BATH),
    ]

    for jz in range(terminal_start_idx, n_segments):
        for i, j in impermeable_pairs:
            sdl[jz].dLPV[i, j] *= impermeability_factor
            sdl[jz].dLPV[j, i] *= impermeability_factor


def _set_reflection_coefficients(sdl):
    """Set reflection coefficients.

    sig = 1 everywhere (impermeable), 0 at LIS-BATH (freely permeable basement membrane).
    Both LIS-BATH and BATH-LIS are set to 0 (original SDL sets both directions).
    """
    for membrane in sdl:
        membrane.sig[:, :, :] = 1.0

    for membrane in sdl:
        membrane.sig[:, LIS, BATH] = 0.0
        membrane.sig[:, BATH, LIS] = 0.0


def _set_boundary_conditions(sdl):
    """Set peritubular boundary conditions and interstitial concentrations.

    Position 0 is at the outer stripe boundary (xIS = 0.3 mm from cortex),
    position 1 is deeper medulla. Interpolates between cortical and medullary values.
    """
    ep_bath = -0.001e-3 / EPref
    for membrane in sdl:
        membrane.ep[BATH] = ep_bath

    n_segments = len(sdl)
    xIS = 0.6 / 2.0
    positions = np.array([xIS + (1 - xIS) * jz / (n_segments - 1)
                          for jz in range(n_segments)])

    set_intconc(sdl, n_segments - 1, 2, positions)


def _set_metabolic_parameters(sdl):
    """Set TNa-QO2 metabolic ratio.

    TQ = 15.0 in normal SDL, 12.0 in diabetic SDL (ndiabetes != 0).
    """
    tq_value = 12.0 if ndiabetes != 0 else 15.0
    for membrane in sdl:
        membrane.TQ = tq_value
