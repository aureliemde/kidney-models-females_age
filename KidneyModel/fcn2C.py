"""Residual function for the CNT and CCD Newton-Raphson solver.

Computes the residual vector F(x) = 0 at each Newton iteration.
Used for segments with four active epithelial compartments:
lumen (M), principal cell (P), intercalated-A cell (A),
intercalated-B cell (B), and lateral interspace (LIS).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Used for segments:
    idid=6: CNT (Connecting Tubule)
    idid=7: CCD (Cortical Collecting Duct)
"""

import numpy as np

from values import *
from glo import *
from defs import *


def fcn2C(n, x, iflag, numpar, pars, tube, idid, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT):
    """Evaluate the residual vector F(x) = 0 for the CNT/CCD Newton solver.

    Args:
        n:          Length of the residual/solution vector
        x:          Current solution vector (concentrations, volumes, EPs, pressure)
        iflag:      Solver flag (unused; passed through to flux functions)
        numpar:     Length of the parameter vector
        pars:       Upstream state vector (read-only; packed by qnewton2icb)
        tube:       List of Membrane objects for the segment (0 to NZ)
        idid:       Segment identifier (6=CNT, 7=CCD)
        Lz:         Current spatial position index (upstream node)
        PTinitVol:  Initial PT luminal volume
        xNaPiIIaPT: PT NaPi-IIa transporter activity
        xNaPiIIcPT: PT NaPi-IIc transporter activity
        xPit2PT:    PT Pit2 transporter activity

    Returns:
        fvec: np.ndarray of shape (n,) — residual vector
    """
    _nc = NC - 1  # 5 active compartments: LUM, P, A, B, LIS (BATH is fixed boundary)

    # ------------------------------------------------------------------
    # Working arrays
    # ------------------------------------------------------------------
    fvec = np.zeros(n)
    S    = np.zeros(n)

    Ca   = np.zeros((NS, NC))
    Vola = np.zeros(NC)
    EPa  = np.zeros(NC)
    pha  = np.zeros(NC)

    Cb   = np.zeros((NS, NC))
    Volb = np.zeros(NC)
    EPb  = np.zeros(NC)
    phb  = np.zeros(NC)
    # Jva  = np.zeros((NC, NC))      # dead pre-alloc — overwritten by flux return; Jva feeds only dead sumJva
    # Jsa  = np.zeros((NS, NC, NC))  # dead pre-alloc — overwritten by flux return; Jsa feeds only dead sumJsa
    # Jvb  = np.zeros((NC, NC))      # dead pre-alloc — overwritten by direct capture from flux return
    # Jsb  = np.zeros((NS, NC, NC))  # dead pre-alloc — overwritten by direct capture from flux return
    # CaBT = np.zeros(NC)            # dead alloc — never used

    # ------------------------------------------------------------------
    # x-vector offsets (must match qnewton2icb packing)
    # ------------------------------------------------------------------
    _ox  = _nc * NS2           # start of volume block in x
    _oxv = _ox  + _nc          # start of EP block in x
    _oxe = _oxv + _nc          # pressure index in x

    # ------------------------------------------------------------------
    # Assign peritubular (BATH) concentrations from downstream node
    # ------------------------------------------------------------------
    Ca[:NS, BATH] = tube[Lz+1].conc[:NS, BATH]
    Cb[:NS, BATH] = tube[Lz+1].conc[:NS, BATH]
    EPa[BATH]     = tube[Lz+1].ep[BATH]
    EPb[BATH]     = tube[Lz+1].ep[BATH]

    # ------------------------------------------------------------------
    # Unpack upstream state (Ca, Vola, EPa, PMa) from pars
    # pars is packed column-major (Fortran order) by qnewton2icb
    # ------------------------------------------------------------------
    _oc = _nc * NS          # start of volume entries in pars
    _ov = _nc * (NS + 1)    # start of EP entries in pars
    _oe = _nc * (NS + 2)    # pressure entry in pars

    Ca[:NS, :_nc] = pars[:_oc].reshape(NS, _nc, order='F')
    pha[:_nc]     = -np.log10(Ca[H, :_nc] / 1.0e3)
    Vola[:_nc]    = pars[_oc : _oc + _nc]
    EPa[:_nc]     = pars[_ov : _ov + _nc]
    PMa           = pars[_oe]

    # Pack upstream state vector y (same layout as x) for flux call
    y = np.empty(n)
    y[:_ox]      = Ca[:NS2, :_nc].ravel()
    y[_ox:_oxv]  = Vola[:_nc]
    y[_oxv:_oxe] = EPa[:_nc]
    y[_oxe]      = PMa

    # ------------------------------------------------------------------
    # Unpack downstream state (Cb, Volb, EPb, PMb) from x
    # ------------------------------------------------------------------
    Cb[:NS2, :_nc] = x[:_ox].reshape(NS2, _nc)
    Volb[:_nc]     = x[_ox:_oxv]
    EPb[:_nc]      = x[_oxv:_oxe]
    PMb            = x[_oxe]
    phb[:_nc]      = -np.log10(Cb[H, :_nc] / 1.0e3)

    # ------------------------------------------------------------------
    # Select and call flux function
    # ------------------------------------------------------------------
    if idid == 6:
        from qflux2C import qflux2C as _flux
    elif idid == 7:
        from qflux2CCD import qflux2CCD as _flux

    _flux(y, tube, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)   # upstream node (side effects)
    # Jva = Jvol_3  # dead — Jva feeds only dead sumJva; upstream flux called for side effects only
    # Jsa = Jsol_3  # dead — Jsa feeds only dead sumJsa
    Jvb, Jsb = _flux(x, tube, Lz, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

    # ------------------------------------------------------------------
    # Geometry at downstream node
    # ------------------------------------------------------------------
    coalesce = tube[Lz+1].coalesce
    Bm = np.pi * tube[Lz+1].diam * coalesce
    Am = np.pi * (tube[Lz+1].diam ** 2) / 4.0 * coalesce

    # ------------------------------------------------------------------
    # Compute source terms S
    # S_2d[i, k] = source for solute i in compartment k (view into S[:_ox])
    # ------------------------------------------------------------------
    S_2d = S[:_ox].reshape(NS2, _nc)

    # Lumen: solute conservation
    # sumJsa = Jsa[:NS2, LUM, P] + ... + Jsa[:NS2, LUM, LIS]  # dead — computed in v0 but not used in S (upstream flux for side effects only)
    sumJsb_lum = Jsb[:NS2, LUM, P] + Jsb[:NS2, LUM, A] + Jsb[:NS2, LUM, B] + Jsb[:NS2, LUM, LIS]
    S_2d[:, LUM] = ((Volb[LUM] * Cb[:NS2, LUM] - Vola[LUM] * Ca[:NS2, LUM]) * Vref / href
                    + Bm * tube[Lz+1].dimL * sumJsb_lum / NZ)

    # Cellular compartments
    S_2d[:, P] =  Jsb[:NS2, P, LIS] + Jsb[:NS2, P, BATH] - Jsb[:NS2, LUM, P]
    S_2d[:, A] =  Jsb[:NS2, A, LIS] + Jsb[:NS2, A, BATH] - Jsb[:NS2, LUM, A]
    S_2d[:, B] =  Jsb[:NS2, B, LIS] + Jsb[:NS2, B, BATH] - Jsb[:NS2, LUM, B]
    S_2d[:, LIS] = (-Jsb[:NS2, LUM, LIS] - Jsb[:NS2, P, LIS]
                    - Jsb[:NS2, A,   LIS] - Jsb[:NS2, B, LIS]
                    + Jsb[:NS2, LIS, BATH])

    # Volume source terms
    fvmult       = Pfref * Vwbar * Cref
    # sumJva = (Jva[LUM, P] + Jva[LUM, A] + Jva[LUM, B] + Jva[LUM, LIS]) * fvmult  # dead — not used in S
    sumJvb_lum   = (Jvb[LUM, P] + Jvb[LUM, A] + Jvb[LUM, B] + Jvb[LUM, LIS]) * fvmult
    S[_ox + LUM] = (Volb[LUM] - Vola[LUM]) * Vref + Bm * tube[Lz+1].dimL * sumJvb_lum / NZ
    S[_ox + P]   =  Jvb[P,   LIS] + Jvb[P,   BATH] - Jvb[LUM, P]
    S[_ox + A]   =  Jvb[A,   LIS] + Jvb[A,   BATH] - Jvb[LUM, A]
    S[_ox + B]   =  Jvb[B,   LIS] + Jvb[B,   BATH] - Jvb[LUM, B]
    S[_ox + LIS] = -Jvb[LUM, LIS] - Jvb[P, LIS] - Jvb[A, LIS] - Jvb[B, LIS] + Jvb[LIS, BATH]

    # ------------------------------------------------------------------
    # Build residual vector fvec
    # ------------------------------------------------------------------

    # Non-reacting solutes: fvec = S (pure conservation)
    fvec[5*NA   : 5*CL  +5] = S[5*NA   : 5*CL  +5]   # NA, K, CL (contiguous)
    fvec[5*UREA : 5*UREA+5] = S[5*UREA : 5*UREA+5]
    fvec[5*GLU  : 5*CA  +5] = S[5*GLU  : 5*CA  +5]   # GLU, CA (contiguous)

    # HCO3 / H2CO3 / CO2
    fvec[5*HCO3  : 5*HCO3 +5] = (S[5*HCO3  : 5*HCO3 +5]
                                  + S[5*H2CO3 : 5*H2CO3+5]
                                  + S[5*CO2   : 5*CO2  +5])
    fvec[5*H2CO3 : 5*H2CO3+5] = phb[:_nc] - pKHCO3 - np.log10(Cb[HCO3, :_nc] / Cb[H2CO3, :_nc])

    facnd = Vref / href
    # fkin1 = tube[Lz+1].dkh[LUM] * Ca[CO2, LUM] - tube[Lz+1].dkd[LUM] * Ca[H2CO3, LUM]  # dead — computed but never used
    fkin2 = tube[Lz+1].dkh[LUM] * Cb[CO2, LUM] - tube[Lz+1].dkd[LUM] * Cb[H2CO3, LUM]
    fvec[5*CO2 + LUM] = S[5*CO2 + LUM] + Am * tube[Lz+1].dimL * fkin2 / NZ / href

    vol_eff  = np.array([Volb[P], Volb[A], Volb[B], max(Volb[LIS], tube[Lz+1].volEinit)])
    kin_cell = (tube[Lz+1].dkh[P:LIS+1] * Cb[CO2, P:LIS+1]
                - tube[Lz+1].dkd[P:LIS+1] * Cb[H2CO3, P:LIS+1])
    fvec[5*CO2+P : 5*CO2+LIS+1] = S[5*CO2+P : 5*CO2+LIS+1] + vol_eff * kin_cell * facnd

    # HPO4 / H2PO4
    fvec[5*HPO4  : 5*HPO4 +5] = S[5*HPO4  : 5*HPO4 +5] + S[5*H2PO4 : 5*H2PO4+5]
    fvec[5*H2PO4 : 5*H2PO4+5] = phb[:_nc] - pKHPO4 - np.log10(Cb[HPO4, :_nc] / Cb[H2PO4, :_nc])

    # NH3 / NH4
    fvec[5*NH3 : 5*NH3+5] = S[5*NH3 : 5*NH3+5] + S[5*NH4 : 5*NH4+5]
    fvec[5*NH4 : 5*NH4+5] = phb[:_nc] - pKNH3 - np.log10(Cb[NH3, :_nc] / Cb[NH4, :_nc])

    # H (pH conservation)
    fvec[5*H : 5*H+5] = (S[5*H    : 5*H   +5] + S[5*NH4  : 5*NH4 +5]
                         - S[5*HCO3 : 5*HCO3+5] - S[5*HPO4 : 5*HPO4+5]
                         - S[5*HCO2 : 5*HCO2+5])

    # HCO2 / H2CO2
    fvec[5*HCO2  : 5*HCO2 +5] = S[5*HCO2  : 5*HCO2 +5] + S[5*H2CO2 : 5*H2CO2+5]
    fvec[5*H2CO2 : 5*H2CO2+5] = phb[:_nc] - pKHCO2 - np.log10(np.abs(Cb[HCO2, :_nc] / Cb[H2CO2, :_nc]))

    # Volume
    fvec[_ox : _ox+_nc] = S[_ox : _ox+_nc]

    # EP: zero net current in lumen
    fvec[_oxv] = np.dot(zval[:NS2], Jsb[:NS2, LUM, P:LIS+1].sum(axis=1))

    # EP: electroneutrality in epithelial compartments
    volPrat = tube[Lz+1].volPinit / Volb[P]
    volArat = tube[Lz+1].volAinit / Volb[A]
    volBrat = tube[Lz+1].volBinit / Volb[B]

    facP = np.exp(np.log(10.0) * (phb[P] - pKbuf))
    facA = np.exp(np.log(10.0) * (phb[A] - pKbuf))
    facB = np.exp(np.log(10.0) * (phb[B] - pKbuf))

    CimpP = tube[Lz+1].cPimpref * volPrat
    CimpA = tube[Lz+1].cAimpref * volArat
    CimpB = tube[Lz+1].cBimpref * volBrat

    CbufP = tube[Lz+1].cPbuftot * volPrat * facP / (facP + 1)
    CbufA = tube[Lz+1].cAbuftot * volArat * facA / (facA + 1)
    CbufB = tube[Lz+1].cBbuftot * volBrat * facB / (facB + 1)

    fvec[_oxv + P]   = tube[Lz+1].zPimp * CimpP - CbufP + np.dot(zval[:NS2], Cb[:NS2, P])
    fvec[_oxv + A]   = tube[Lz+1].zAimp * CimpA - CbufA + np.dot(zval[:NS2], Cb[:NS2, A])
    fvec[_oxv + B]   = tube[Lz+1].zBimp * CimpB - CbufB + np.dot(zval[:NS2], Cb[:NS2, B])
    fvec[_oxv + LIS] = np.dot(zval[:NS2], Cb[:NS2, LIS])

    # Pressure (Poiseuille flow)
    ratio = 8.0 * np.pi * visc / Am**2
    if idid == 6:
        ratio *= coalesce * 2   # correct Am; add extra resistance from merging
    elif idid == 7:
        ratio *= coalesce       # correct extra coalescence factor in Am

    fvec[_oxe] = PMb - PMa + ratio * Volb[LUM] * Vref * tube[Lz+1].dimL / NZ

    return fvec
