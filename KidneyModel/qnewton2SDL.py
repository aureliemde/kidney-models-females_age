"""
SDL (Short Descending Limb) Newton-Raphson solver.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Newton-Raphson solver for the SDL segment.
Determines luminal concentrations, volumes, and electrical potential
below the inlet. sdl[Lz] is the known upstream position;
sdl[Lz+1] is the unknown downstream position solved here.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *
from glo import *
from defs import *
from fcn2SDL import fcn2SDL
from jacobi2_2SDL import jacobi2_2SDL


def qnewton2SDL(sdl, Lz, idid, Vol0, ND, PTinitVol):
    """Solve for luminal concentrations, volume, and pressure at position Lz+1.

    Uses Newton-Raphson iteration to solve the nonlinear system governing
    solute transport, water flow, and pressure in the SDL.

    Args:
    sdl : list of Membrane
        SDL membrane array.
    Lz : int
        Current spatial position index (0-based).
    idid : int
        Segment identifier (2 = SDL).
    Vol0 : np.ndarray
        Initial volume array.
    ND : int
        Number of degrees of freedom.
    PTinitVol : float
        Initial PT luminal volume (scalar) in cm³.
    """
    TOL        = 1e-5
    MAX_ITER   = 21
    RELAX_TOL  = 1e-3

    num     = 2 + NS2
    numpar  = 8 + 4 * NS

    state = _initialize_state(sdl, Lz)

    _newton_iteration_loop(
        sdl, Lz, idid, state, num, numpar,
        Vol0, PTinitVol, TOL, MAX_ITER, RELAX_TOL,
    )

    _update_solution(sdl, Lz, state)

    if Lz == NZ - 1:
        _apply_convergence_factor(sdl)


def _initialize_state(sdl, Lz):
    """Initialize state variables from the previous spatial position (Lz)."""
    Cb   = sdl[Lz].conc.copy()
    Volb = sdl[Lz].vol.copy()
    EPb  = sdl[Lz].ep.copy()
    phb  = sdl[Lz].ph.copy()
    PMb  = sdl[Lz].pres

    return {
        'Cb':      Cb,
        'Cprev':   Cb.copy(),
        'Volb':    Volb,
        'Volprev': Volb.copy(),
        'EPb':     EPb,
        'EPprev':  EPb.copy(),
        'phb':     phb,
        'phprev':  phb.copy(),
        'PMb':     PMb,
        'PMprev':  PMb,
    }


def _newton_iteration_loop(sdl, Lz, idid, state, num, numpar,
                           Vol0, PTinitVol, TOL, MAX_ITER, RELAX_TOL):
    """Perform Newton-Raphson iteration to convergence."""
    residual    = 1.0
    iteration   = 0
    current_tol = TOL

    while residual > current_tol and iteration < 40:
        iteration += 1

        x    = _pack_unknowns(state, num)
        pars = _pack_parameters(sdl, Lz, Vol0)

        fvec          = _evaluate_residual(x, num, numpar, pars, sdl, idid, Lz, PTinitVol)
        fjac          = _evaluate_jacobian(x, fvec, num, numpar, pars, sdl, idid, Lz, PTinitVol)
        delta_x       = _compute_newton_step(fjac, fvec, num)
        residual      = _update_state(state, delta_x, num)

        if iteration >= MAX_ITER:
            current_tol = RELAX_TOL if iteration < 40 else residual * 1.01

    return residual > current_tol, iteration


def _pack_unknowns(state, num):
    """Pack state variables into solution vector x."""
    x = np.zeros(num)
    x[:NS2]    = state['Cb'][:NS2, LUM]
    x[NS2]     = state['Volb'][LUM]
    x[NS2 + 1] = state['PMb']
    return x


def _pack_parameters(sdl, Lz, Vol0):
    """Pack upstream state and geometric parameters into pars array."""
    numpar = 8 + 4 * NS
    pars   = np.zeros(numpar)

    pars[0:NS]         = sdl[Lz].conc[:, LUM]
    pars[NS:2*NS]      = sdl[Lz].conc[:, P]
    pars[2*NS:3*NS]    = sdl[Lz].conc[:, LIS]
    pars[3*NS]         = sdl[Lz].vol[LUM]
    pars[3*NS + 1]     = sdl[Lz].vol[P]
    pars[3*NS + 2]     = sdl[Lz].vol[LIS]
    pars[3*NS + 3]     = sdl[Lz].pres
    pars[3*NS + 4:4*NS + 4] = sdl[Lz].conc[:, BATH]
    pars[4*NS + 4]     = dimLSDL
    pars[4*NS + 5]     = CPimprefSDL
    pars[4*NS + 6]     = DiamSDL
    pars[4*NS + 7]     = Vol0[LUM]

    return pars


def _evaluate_residual(x, num, numpar, pars, sdl, idid, Lz, PTinitVol):
    """Evaluate residual vector fvec = F(x)."""
    iflag = 1
    return fcn2SDL(num, x, iflag, numpar, pars, sdl, idid, Lz, PTinitVol)


def _evaluate_jacobian(x, fvec, num, numpar, pars, sdl, idid, Lz, PTinitVol):
    """Evaluate Jacobian matrix J = ∂F/∂x using forward finite differences."""
    ldfjac  = num
    iflag   = 1
    epsfcn  = 1.0e-5
    fjac, wa1, fvec_J = jacobi2_2SDL(
        num, x, fvec, ldfjac, iflag, epsfcn,
        numpar, pars, sdl, idid, Lz, PTinitVol,
    )
    return fjac


def _compute_newton_step(fjac, fvec, num):
    """Compute Newton step Δx = -J⁻¹ f.

    If Jacobian determinant < 1, returns zero step (system poorly conditioned).
    """
    if np.linalg.det(fjac) >= 1.0:
        try:
            return np.linalg.inv(fjac) @ fvec
        except np.linalg.LinAlgError:
            pass
    return np.zeros(num)


def _update_state(state, delta_x, num):
    """Update state variables and return convergence residual."""
    residual = 0.0

    Cb_new = state['Cprev'][:NS2, LUM] - delta_x[:NS2]
    for i in range(NS2):
        state['Cb'][i, LUM] = Cb_new[i]
        if state['Cprev'][i, LUM] != 0:
            rel = abs(Cb_new[i] / state['Cprev'][i, LUM] - 1.0)
            if i == H2CO2:
                rel *= 0.1
            residual = max(residual, rel)

    state['phb'][LUM] = -np.log10(abs(state['Cb'][H, LUM] / 1e3))

    Volb_new = state['Volprev'][LUM] - delta_x[NS2]
    state['Volb'][LUM] = Volb_new
    residual = max(residual, abs(Volb_new - state['Volprev'][LUM]))

    PMb_new = state['PMprev'] - delta_x[NS2 + 1]
    state['PMb'] = PMb_new
    residual = max(residual, abs(PMb_new - state['PMprev']))

    # EPb update (not active in SDL — lumen EP not solved; kept for future use)
    # EPb_new = state['EPprev'][LUM] - delta_x[4 + 3*NS2 - 1]
    # state['EPb'][LUM] = EPb_new
    # residual = max(residual, 1e-3 * abs(EPb_new - state['EPprev'][LUM]))
    # state['EPprev'][LUM] = state['EPb'][LUM]

    state['Cprev'][:, LUM] = state['Cb'][:, LUM]
    state['Cprev'][:, P]   = state['Cb'][:, P]
    state['Cprev'][:, LIS] = state['Cb'][:, LIS]
    state['Volprev'][LUM]  = state['Volb'][LUM]
    state['PMprev']        = state['PMb']
    state['phprev'][LUM]   = state['phb'][LUM]

    return residual


def _update_solution(sdl, Lz, state):
    """Copy converged solution into sdl[Lz+1]."""
    sdl[Lz + 1].conc[:, LUM] = state['Cb'][:, LUM]
    sdl[Lz + 1].conc[:, P]   = state['Cb'][:, P]
    sdl[Lz + 1].conc[:, LIS] = state['Cb'][:, LIS]
    sdl[Lz + 1].vol[LUM]     = state['Volb'][LUM]
    sdl[Lz + 1].pres          = state['PMb']
    sdl[Lz + 1].ph[LUM]       = state['phb'][LUM]


def _apply_convergence_factor(sdl):
    """Apply empirical concentration factor pconv = 1.3 at SDL outlet.

    Applied when Lz = NZ-1; increases luminal concentrations by pconv
    and decreases volume by pconv (conserves solute mass).
    """
    pconv = 1.3
    for membrane in sdl:
        membrane.conc[:, LUM] *= pconv
        membrane.vol[LUM]     /= pconv
