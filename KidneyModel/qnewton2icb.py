"""
Newton-Raphson solver for CNT, CCD, and OMCD segments (4-compartment epithelium).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Newton solver to determine luminal and epithelial concentrations, volumes,
and electrical potentials at each spatial grid point. Solves the coupled
nonlinear system governing transport in segments with four active epithelial
compartments: lumen (M), principal cell (P), intercalated-A cell (A),
intercalated-B cell (B), and lateral interspace (LIS).

Used for segments:
    idid=6: CNT  (Connecting Tubule)
    idid=7: CCD  (Cortical Collecting Duct)
    idid=8: OMCD (Outer Medullary Collecting Duct)

Note: In this code:
    tube[Lz]   = current (upstream) position
    tube[Lz+1] = next (downstream) position being solved
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


def qnewton2icb(
    tube: List[Membrane],
    Lz: int,
    idid: int,
    Vol0: np.ndarray,
    VolEinit: float,
    VolPinit: float,
    VolAinit: float,
    VolBinit: float,
    ND: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float,
) -> None:
    """
    Solve for concentrations, volumes, and potentials at position Lz+1.

    Uses Newton-Raphson iteration with an explicit Jacobian inverse. The
    upstream node tube[Lz] provides the initial guess and boundary values.

    The solution vector x has the structure:
        x[0 : (NC-1)*NS2]           — concentrations: (NC-1) compartments × NS2 solutes
        x[(NC-1)*NS2 : ... +NC-1]   — volumes for NC-1 active compartments
        x[... +NC-1  : ... +2*(NC-1)] — potentials for NC-1 active compartments
        x[... +2*(NC-1)]            — hydrostatic pressure

    Args:
        tube:       List of Membrane objects for the segment (0 to NZ)
        Lz:         Current spatial position index (upstream node)
        idid:       Segment identifier (6=CNT, 7=CCD, 8=OMCD)
        Vol0:       Reference volume array [cm³]
        VolEinit:   Initial LIS volume [cm³]
        VolPinit:   Initial PC volume [cm³]
        VolAinit:   Initial IC-A volume [cm³]
        VolBinit:   Initial IC-B volume [cm³]
        ND:         Number of degrees of freedom
        PTinitVol:  Initial PT luminal volume (passed through to flux functions)
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity
    """
    # NC-1 active compartments: LUM, P, A, B, LIS (BATH is the fixed boundary)
    _nc = NC - 1

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------
    numpar = 1 + _nc * (NS + 2)
    pars   = np.zeros(numpar)

    num  = 1 + _nc * (NS2 + 2)   # = 11 + 5*NS2
    x    = np.zeros(num)
    fvec = np.zeros(num)
    fjac = np.zeros((num, num))

    Cb      = np.zeros((NS, NC))
    Volb    = np.zeros(NC)
    EPb     = np.zeros(NC)
    phb     = np.zeros(NC)
    Cprev   = np.zeros((NS, NC))
    Volprev = np.zeros(NC)
    EPprev  = np.zeros(NC)
    # phprev = np.zeros(NC)          # dead alloc — phprev only assigned, never read in computation
    # osmol  = np.zeros(NC)          # dead alloc — never used
    # AVF    = np.zeros(NDC)         # dead pre-alloc — overwritten in loop; size NDC may differ from num
    # A      = np.zeros((NDC, NDC))  # dead alloc — never used
    # wa1    = np.zeros(num)         # dead pre-alloc — assigned by Jacobian return but never used after
    # wa2    = np.zeros(num)         # dead alloc — never used
    # lwork  = 10000                 # dead scalar — never used
    # ipiv   = np.zeros(NDC)         # dead alloc — never used
    # work   = np.zeros(lwork)       # dead alloc — never used

    TOL = 1e-5

    # ------------------------------------------------------------------
    # Initial guess: upstream node values
    # ------------------------------------------------------------------
    Cb[:, :_nc]    = tube[Lz].conc[:, :_nc]
    Cprev[:, :_nc] = Cb[:, :_nc]
    phb[:_nc]      = tube[Lz].ph[:_nc]
    # phprev[:_nc] = phb[:_nc]  # dead — phprev only assigned, never read
    Volb[:_nc]     = tube[Lz].vol[:_nc]
    Volprev[:_nc]  = Volb[:_nc]
    EPb[:_nc]      = tube[Lz].ep[:_nc]
    EPprev[:_nc]   = EPb[:_nc]
    PMb            = tube[Lz].pres
    PMprev         = PMb

    # ------------------------------------------------------------------
    # Pack pars from upstream node (read-only during Newton iterations)
    # Layout: [conc (col-major), vol, ep, pres]
    # ------------------------------------------------------------------
    _oc = _nc * NS          # offset: start of volume entries
    _ov = _nc * (NS + 1)    # offset: start of EP entries
    _oe = _nc * (NS + 2)    # offset: pressure entry

    pars[:_oc]           = tube[Lz].conc[:, :_nc].ravel(order='F')
    pars[_oc:_oc+_nc]    = tube[Lz].vol[:_nc]
    pars[_ov:_ov+_nc]    = tube[Lz].ep[:_nc]
    pars[_oe]            = tube[Lz].pres

    # ------------------------------------------------------------------
    # Solver constants
    # ------------------------------------------------------------------
    ldfjac = num
    iflag  = 1
    epsfcn = 1.0e-5
    # ml = num  # dead — bandwidth param for banded Jacobian, not used here
    # mu = num  # dead — bandwidth param for banded Jacobian, not used here

    # x-vector offsets (reused in every iteration)
    _ox  = _nc * NS2           # start of volume block in x
    _oxv = _nc * NS2 + _nc     # start of EP block in x
    _oxe = _nc * NS2 + 2 * _nc # pressure index in x

    # ------------------------------------------------------------------
    # Select flux and Jacobian functions based on segment
    # idid 6 (CNT) and 7 (CCD) share the same flux/Jacobian routines
    # ------------------------------------------------------------------
    if idid in (6, 7):
        from fcn2C import fcn2C
        from jacobi2_2icb import jacobi2_2icb
        _fcn    = fcn2C
        _jacobi = jacobi2_2icb
    elif idid == 8:
        from fcn2OMC import fcn2OMC
        from jacobi2_2icbOMC import jacobi2_2icbOMC
        _fcn    = fcn2OMC
        _jacobi = jacobi2_2icbOMC

    # ------------------------------------------------------------------
    # Newton-Raphson iteration
    # ------------------------------------------------------------------
    res    = 1.0
    iterat = 0
    # nwrite = 0  # dead — assigned but never used

    while res > TOL and iterat < 21:
        res     = 0.0
        iterat += 1

        # Pack solution vector x
        x[:_ox]      = Cb[:NS2, :_nc].ravel()
        x[_ox:_oxv]  = Volb[:_nc]
        x[_oxv:_oxe] = EPb[:_nc]
        x[_oxe]      = PMb

        # Evaluate residual vector and Jacobian matrix
        fvec       = _fcn(num, x, iflag, numpar, pars,
                          tube, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
        fjac, _, _ = _jacobi(num, x, fvec, ldfjac, iflag, epsfcn, numpar, pars,
                              tube, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

        # Newton step: AVF = 0.5 * J^{-1} * f
        AVF = 0.5 * (np.linalg.inv(fjac) @ fvec)

        # Update concentrations (absolute value prevents negative values)
        Cb[:NS2, :_nc] = np.abs(Cprev[:NS2, :_nc] - AVF[:_ox].reshape(NS2, _nc))

        # Warn on non-positive concentrations
        for i, k in np.argwhere(Cb[:NS2, :_nc] <= 0.0):
            print("Warning in newton2icb", i, k, Cb[i, k])

        # Concentration residual (H2CO2 weighted by 0.01 — small and stiff)
        rel_err = np.abs(Cb[:NS2, :_nc] / Cprev[:NS2, :_nc] - 1.0)
        if H2CO2 < NS2:
            rel_err[H2CO2, :] *= 0.01
        res = max(res, rel_err.max())

        # Update pH, volumes, potentials, and pressure
        phb[:_nc]  = -np.log10(Cb[H, :_nc] / 1.0e3)
        Volb[:_nc] = Volprev[:_nc] - AVF[_ox:_oxv]
        EPb[:_nc]  = EPprev[:_nc]  - AVF[_oxv:_oxe]
        PMb        = PMprev        - AVF[_oxe]

        res = max(res, np.abs(Volb[:_nc] - Volprev[:_nc]).max())
        res = max(res, 1e-3 * np.abs(EPb[:_nc] - EPprev[:_nc]).max())
        res = max(res, abs(PMb - PMprev))

        # Update "previous" values for next iteration
        Cprev[:, :_nc] = Cb[:, :_nc]
        Volprev[:_nc]  = Volb[:_nc]
        EPprev[:_nc]   = EPb[:_nc]
        # phprev[:_nc] = phb[:_nc]  # dead — phprev only assigned, never read
        PMprev         = PMb

        # Adaptive tolerance relaxation
        if iterat >= 11:
            if iterat < 21:
                TOL = 1.0e-4
            elif iterat < 31:
                TOL = 1.0e-3
            else:
                TOL = res * 1.010

    # ------------------------------------------------------------------
    # Write converged solution to downstream node
    # ------------------------------------------------------------------
    tube[Lz+1].conc[:, :_nc] = Cb[:, :_nc]
    tube[Lz+1].ph[:_nc]      = phb[:_nc]
    tube[Lz+1].vol[:_nc]     = Volb[:_nc]
    tube[Lz+1].ep[:_nc]      = EPb[:_nc]
    tube[Lz+1].pres           = PMb
