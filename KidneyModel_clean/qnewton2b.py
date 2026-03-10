"""
Multi-segment Newton-Raphson solver module (mTAL, cTAL, DCT, IMCD).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Newton solver to determine luminal and epithelial concentrations, volumes,
and electrical potentials below the inlet. Solves the coupled nonlinear
system of equations governing transport in segments with three active
compartments: lumen (M), principal cell (P), and lateral interspace (LIS).

Used for segments:
    idid=3: mTAL (Medullary Thick Ascending Limb)
    idid=4: cTAL (Cortical Thick Ascending Limb)
    idid=5: DCT  (Distal Convoluted Tubule)
    idid=9: IMCD (Inner Medullary Collecting Duct)

Note: In this code:
    tube[Lz]   = current (upstream) position
    tube[Lz+1] = next (downstream) position being solved
"""

import numpy as np
from typing import List
from values import *
from glo import *
from defs import *
from fcn2b import fcn2b
from jacobi2_2b import jacobi2_2b


def qnewton2b(
    tube: List[Membrane],
    Vol0: np.ndarray,
    Lz: int,
    idid: int,
    ND: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> None:
    """
    Solve for concentrations, volumes, and potentials at position Lz+1.

    Uses Newton-Raphson iteration to solve the nonlinear system of equations
    governing solute transport, water flow, and electrical coupling in the
    lumen (M), principal cell (P), and lateral intercellular space (LIS).

    Args:
        tube: List of Membrane objects for the segment
        Vol0: Reference volume array [cm³]
        Lz: Current spatial position index (0-based)
        idid: Segment identifier (3=mTAL, 4=cTAL, 5=DCT, 9=IMCD)
        ND: Number of degrees of freedom
        PTinitVol: Initial PT luminal volume [cm³]
        xNaPiIIaPT: PT NaPi-IIa transporter activity (passed to flux function)
        xNaPiIIcPT: PT NaPi-IIc transporter activity (passed to flux function)
        xPit2PT: PT Pit2 transporter activity (passed to flux function)

    Solution vector layout (length = 7 + 3×NS2):
        x[3i], x[3i+1], x[3i+2]  for i=0..NS2-1 : Cb[i, LUM/P/LIS]
        x[3*NS2 .. 3*NS2+2]  : Volb[LUM/P/LIS]
        x[3*NS2+3 .. 3*NS2+5]: EPb[LUM/P/LIS]
        x[3*NS2+6]           : PMb (hydrostatic pressure)

    Updates tube[Lz+1] in place with the converged solution.
    """
    num    = 7 + 3 * NS2
    numpar = 8 + 5 * NS

    # Dead variables from v0 (kept for reference):
    # A     = np.zeros((ND, ND))  # unused
    # osmol = np.zeros(NC)        # unused
    # wa2   = np.zeros(num)       # unused
    # lwork = 10000               # reserved for future use (matrix inversion)
    # ipiv  = np.zeros(ND)        # unused
    # work  = np.zeros(lwork)     # unused

    state = _initialize_state(tube, Lz)

    _newton_iteration_loop(
        tube, Lz, idid, state, num, numpar,
        PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
    )

    _update_solution(tube, Lz, state)


def _initialize_state(tube: List[Membrane], Lz: int) -> dict:
    """
    Initialize solver state from the upstream position tube[Lz].

    Values at Lz serve as the initial guess for position Lz+1.

    Returns:
        dict with current and previous-iteration arrays for concentrations
        (Cb/Cprev), volumes (Volb/Volprev), electrical potentials
        (EPb/EPprev), pH (phb/phprev), and pressure (PMb/PMprev).
    """
    Cb   = tube[Lz].conc.copy()
    Volb = tube[Lz].vol.copy()
    EPb  = tube[Lz].ep.copy()
    phb  = tube[Lz].ph.copy()
    PMb  = tube[Lz].pres

    return {
        'Cb':     Cb,
        'Cprev':  Cb.copy(),
        'Volb':   Volb,
        'Volprev': Volb.copy(),
        'EPb':    EPb,
        'EPprev': EPb.copy(),
        'phb':    phb,
        'phprev': phb.copy(),
        'PMb':    PMb,
        'PMprev': PMb,
    }


def _newton_iteration_loop(
    tube: List[Membrane],
    Lz: int,
    idid: int,
    state: dict,
    num: int,
    numpar: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> None:
    """
    Perform Newton-Raphson iteration until convergence.

    Tolerance is relaxed progressively if convergence is slow:
        iterations  1–10: TOL = 1e-5
        iterations 11–20: TOL = 1e-4  (relaxed)
    """
    residual    = 1.0
    iteration   = 0
    current_tol = 1.0e-5

    while residual > current_tol and iteration < 21:
        iteration += 1

        x    = _pack_unknowns(state, num)
        pars = _pack_parameters(tube, Lz, idid)

        fvec = _evaluate_residual(
            x, num, numpar, pars, tube, idid, Lz,
            PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
        )
        fjac = _evaluate_jacobian(
            x, fvec, num, numpar, pars, tube, idid, Lz,
            PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
        )

        AVF = _compute_newton_step(fjac, fvec)

        residual = _update_state(state, AVF)

        # Relax tolerance if convergence is slow
        if iteration >= 11:
            current_tol = 1.0e-4
            # Further relaxation branches from v0 (unreachable — loop exits at iteration 21):
            # elif iteration < 31:
            #     current_tol = 1.0e-3
            # else:
            #     current_tol = residual * 1.010


def _pack_unknowns(state: dict, num: int) -> np.ndarray:
    """
    Pack state variables into the solution vector x.

    Concentrations are interleaved: for each solute i, the three
    compartment values (LUM, P, LIS) occupy consecutive positions.

    Vector layout:
        x[3i]   = Cb[i, LUM],  i = 0..NS2-1
        x[3i+1] = Cb[i, P],    i = 0..NS2-1
        x[3i+2] = Cb[i, LIS],  i = 0..NS2-1
        x[3*NS2]     = Volb[LUM]
        x[3*NS2 + 1] = Volb[P]
        x[3*NS2 + 2] = Volb[LIS]
        x[3*NS2 + 3] = EPb[LUM]
        x[3*NS2 + 4] = EPb[P]
        x[3*NS2 + 5] = EPb[LIS]
        x[3*NS2 + 6] = PMb
    """
    x = np.zeros(num)

    # Interleaved concentrations across LUM, P, LIS
    x[0:3*NS2:3] = state['Cb'][:NS2, LUM]
    x[1:3*NS2:3] = state['Cb'][:NS2, P]
    x[2:3*NS2:3] = state['Cb'][:NS2, LIS]

    # Volumes
    x[3*NS2]     = state['Volb'][LUM]
    x[3*NS2 + 1] = state['Volb'][P]
    x[3*NS2 + 2] = state['Volb'][LIS]

    # Electrical potentials
    x[3*NS2 + 3] = state['EPb'][LUM]
    x[3*NS2 + 4] = state['EPb'][P]
    x[3*NS2 + 5] = state['EPb'][LIS]

    # Hydrostatic pressure
    x[3*NS2 + 6] = state['PMb']

    return x


def _pack_parameters(tube: List[Membrane], Lz: int, idid: int) -> np.ndarray:
    """
    Pack fixed parameters for residual and Jacobian evaluation.

    Includes upstream concentrations/volumes/pressure (held fixed during
    Newton iteration), bath concentrations at Lz and Lz+1, and
    segment-specific geometric parameters selected by idid.

    Parameter layout (length = 8 + 5×NS):
        pars[0:NS]          : tube[Lz].conc[:, LUM]
        pars[NS:2*NS]       : tube[Lz].conc[:, P]
        pars[2*NS:3*NS]     : tube[Lz].conc[:, LIS]
        pars[3*NS]          : tube[Lz].vol[LUM]
        pars[3*NS + 1]      : tube[Lz].vol[P]
        pars[3*NS + 2]      : tube[Lz].vol[LIS]
        pars[3*NS + 3]      : tube[Lz].pres
        pars[3*NS+4:4*NS+4] : tube[Lz].conc[:, BATH]   (upstream bath)
        pars[4*NS+4:5*NS+4] : tube[Lz+1].conc[:, BATH] (downstream bath)
        pars[5*NS + 4]      : tube[Lz+1].ep[BATH]
        pars[5*NS + 5]      : segment length
        pars[5*NS + 6]      : impermeant reference concentration
        pars[5*NS + 7]      : tubule diameter
    """
    numpar = 8 + 5 * NS
    pars   = np.zeros(numpar)

    # Upstream state (held fixed during Newton iteration)
    pars[0:NS]       = tube[Lz].conc[:, LUM]
    pars[NS:2*NS]    = tube[Lz].conc[:, P]
    pars[2*NS:3*NS]  = tube[Lz].conc[:, LIS]
    pars[3*NS]       = tube[Lz].vol[LUM]
    pars[3*NS + 1]   = tube[Lz].vol[P]
    pars[3*NS + 2]   = tube[Lz].vol[LIS]
    pars[3*NS + 3]   = tube[Lz].pres

    # Bath (interstitial) boundary conditions
    pars[3*NS+4:4*NS+4] = tube[Lz].conc[:, BATH]
    pars[4*NS+4:5*NS+4] = tube[Lz+1].conc[:, BATH]
    pars[5*NS + 4]      = tube[Lz+1].ep[BATH]

    # Segment-specific geometric parameters
    segment_params = {
        3: (dimLA,   CPimprefA,   DiamA),    # mTAL
        4: (dimLT,   CPimprefT,   DiamT),    # cTAL
        5: (dimLD,   CPimprefD,   DiamD),    # DCT
        9: (dimLIMC, CPimprefIMC, DiamIMC),  # IMCD
    }
    pars[5*NS+5], pars[5*NS+6], pars[5*NS+7] = segment_params[idid]

    return pars


def _evaluate_residual(
    x: np.ndarray,
    num: int,
    numpar: int,
    pars: np.ndarray,
    tube: List[Membrane],
    idid: int,
    Lz: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> np.ndarray:
    """
    Evaluate residual vector fvec = F(x).

    The residual encodes steady-state mass balance for solutes, water,
    and charge in compartments M, P, and LIS.

    Returns:
        Residual vector of length num
    """
    iflag = 1
    return fcn2b(
        num, x, iflag, numpar, pars, tube, idid, Lz,
        PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
    )


def _evaluate_jacobian(
    x: np.ndarray,
    fvec: np.ndarray,
    num: int,
    numpar: int,
    pars: np.ndarray,
    tube: List[Membrane],
    idid: int,
    Lz: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> np.ndarray:
    """
    Evaluate Jacobian matrix J = ∂F/∂x using finite differences.

        J[i,j] ≈ (F(x + ε·eⱼ) - F(x)) / ε,   ε = 1e-5

    Returns:
        Jacobian matrix of shape [num × num]
    """
    ldfjac = num
    iflag  = 1
    epsfcn = 1.0e-5
    # ml = num  # unused (full Jacobian, not banded)
    # mu = num  # unused

    fjac, wa1, fvec_J = jacobi2_2b(
        num, x, fvec, ldfjac, iflag, epsfcn,
        numpar, pars, tube, idid, Lz,
        PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
    )

    # fjac_org = fjac      # no-op alias from v0 (kept for reference)
    # fjac_n   = fjac_org  # no-op alias from v0

    return fjac


def _compute_newton_step(fjac: np.ndarray, fvec: np.ndarray) -> np.ndarray:
    """
    Compute Newton step AVF = J⁻¹ · f.

    The full step (damping factor = 1.0) is applied at every iteration.
    Original v0 damping factors: 0.250 for mTAL/cTAL/DCT, 0.500 for IMCD.

    Returns:
        Newton step vector of length num
    """
    # Original v0 used a double loop with damping:
    # for L in range(num):
    #     AVF[L] = sum(1.0 * fjac_inv[L, M] * fvec[M] for M in range(num))
    #     # original damping was 0.250 (mTAL/cTAL/DCT) or 0.500 (IMCD)
    return np.linalg.inv(fjac) @ fvec


def _update_state(state: dict, AVF: np.ndarray) -> float:
    """
    Apply Newton step to state variables and compute convergence residual.

    Update rules:
        Cb[i, comp]  = |Cprev[i, comp] - AVF[...]|    (concentrations, abs)
        phb[comp]    = -log10(Cb[H, comp] / 1e3)      (pH from H⁺)
        Volb[comp]   =  Volprev[comp] - AVF[...]       (volumes)
        EPb[comp]    =  EPprev[comp]  - AVF[...]       (potentials)
        PMb          =  PMprev        - AVF[...]        (pressure)

    Residual is the maximum of:
        - Relative change |Cb_new/Cb_old - 1| for concentrations
          (H₂CO₂ down-weighted by 0.01 as it converges slowly)
        - Absolute change for volumes and pressure
        - 1e-3 × absolute change for electrical potentials

    Updates state dict in place.

    Returns:
        Convergence residual
    """
    residual = 0.0
    Cprev    = state['Cprev']
    Cb       = state['Cb']

    # --- Concentrations (LUM, P, LIS) ---
    Cb[:NS2, LUM] = np.abs(Cprev[:NS2, LUM] - AVF[0:3*NS2:3])
    Cb[:NS2, P]   = np.abs(Cprev[:NS2, P]   - AVF[1:3*NS2:3])
    Cb[:NS2, LIS] = np.abs(Cprev[:NS2, LIS] - AVF[2:3*NS2:3])

    for i in range(NS2):
        rel_lum = abs(Cb[i, LUM] / Cprev[i, LUM] - 1.0)
        rel_p   = abs(Cb[i, P]   / Cprev[i, P]   - 1.0)
        rel_lis = abs(Cb[i, LIS] / Cprev[i, LIS] - 1.0)

        # H₂CO₂ converges slowly; down-weight its contribution
        if i == H2CO2:
            rel_lum *= 0.01
            rel_p   *= 0.01
            rel_lis *= 0.01

        residual = max(residual, rel_lum, rel_p, rel_lis)

        if Cb[i, P] <= 0.0 or Cb[i, LIS] <= 0.0:
            print(f"Warning in qnewton2b: solute {i}, "
                  f"Cb[P]={Cb[i, P]:.3e}, Cb[LIS]={Cb[i, LIS]:.3e}")

    # --- pH from H⁺ concentration ---
    state['phb'][LUM] = -np.log10(Cb[H, LUM] / 1.0e3)
    state['phb'][P]   = -np.log10(Cb[H, P]   / 1.0e3)
    state['phb'][LIS] = -np.log10(Cb[H, LIS] / 1.0e3)

    # --- Volumes ---
    for idx, comp in enumerate((LUM, P, LIS)):
        Volb_new = state['Volprev'][comp] - AVF[3*NS2 + idx]
        state['Volb'][comp] = Volb_new
        residual = max(residual, abs(Volb_new - state['Volprev'][comp]))

    # --- Electrical potentials (scaled by 1e-3 to prevent dominance) ---
    for idx, comp in enumerate((LUM, P, LIS)):
        EPb_new = state['EPprev'][comp] - AVF[3*NS2 + 3 + idx]
        state['EPb'][comp] = EPb_new
        residual = max(residual, 1.0e-3 * abs(EPb_new - state['EPprev'][comp]))

    # --- Hydrostatic pressure ---
    PMb_new = state['PMprev'] - AVF[3*NS2 + 6]
    state['PMb'] = PMb_new
    residual = max(residual, abs(PMb_new - state['PMprev']))

    # --- Store current values as "previous" for the next iteration ---
    state['Cprev'][:, LUM] = Cb[:, LUM]
    state['Cprev'][:, P]   = Cb[:, P]
    state['Cprev'][:, LIS] = Cb[:, LIS]

    for comp in (LUM, P, LIS):
        state['phprev'][comp]  = state['phb'][comp]
        state['Volprev'][comp] = state['Volb'][comp]
        state['EPprev'][comp]  = state['EPb'][comp]

    state['PMprev'] = state['PMb']

    return residual


def _update_solution(tube: List[Membrane], Lz: int, state: dict) -> None:
    """
    Transfer the converged solution into tube[Lz+1].

    Copies concentrations (all NS solutes), pH, volume, electrical
    potential, and pressure for compartments LUM, P, and LIS.
    """
    for comp in (LUM, P, LIS):
        tube[Lz+1].conc[:, comp] = state['Cb'][:, comp]
        tube[Lz+1].ph[comp]      = state['phb'][comp]
        tube[Lz+1].vol[comp]     = state['Volb'][comp]
        tube[Lz+1].ep[comp]      = state['EPb'][comp]

    tube[Lz+1].pres = state['PMb']
