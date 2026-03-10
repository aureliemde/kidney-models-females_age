"""
Residual function for principal-cell segments (mTAL, cTAL, DCT, IMCD).

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This subroutine computes the non-linear equations to be solved at any
point below the inlet, in order to determine concentrations, volumes,
and electrical potentials in the lumen (M), principal cell (P), and
lateral intercellular space (E).

Used for segments with principal cells only:
    idid=3: medial thick ascending limb (mTAL)
    idid=4: cortical thick ascending limb (cTAL)
    idid=5: distal convoluted tubule (DCT)
    idid=9: inner medullary collecting duct (IMCD)

Solution vector x layout (size 3*NS2 + 7 = 55):
    x[3*i : 3*i+3]  = [LUM, P, LIS] concentrations for solute i (i=0..NS2-1)
    x[3*NS2]         = luminal volume
    x[3*NS2+1]       = P cell volume
    x[3*NS2+2]       = LIS volume
    x[3*NS2+3]       = luminal electrical potential
    x[3*NS2+4]       = P cell electrical potential
    x[3*NS2+5]       = LIS electrical potential
    x[3*NS2+6]       = luminal pressure

Source term vector S layout (size NDA = 3*NS2 + 7):
    S[3*i]           = lumen source for solute i
    S[3*i+1]         = P cell source for solute i
    S[3*i+2]         = LIS source for solute i
    S[3*NS2 + 0..2]  = volume sources for LUM, P, LIS
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


def fcn2b(
    n: int,
    x: np.ndarray,
    iflag: int,
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
    Compute residual vector for principal-cell transport equations.

    Args:
        n: Number of unknowns (3*NS2 + 7 = 55)
        x: Solution vector (unknowns at position Lz+1)
        iflag: Flag for function evaluation (unused)
        numpar: Number of parameters (5*NS + 8 = 88)
        pars: Parameter vector (known state at position Lz)
        tube: Membrane array for this segment
        idid: Segment identifier (3=mTAL, 4=cTAL, 5=DCT, 9=IMCD)
        Lz: Current spatial position (0-based)
        PTinitVol: Initial luminal volume [cm³]
        xNaPiIIaPT: NaPi-IIa transporter scaling factor
        xNaPiIIcPT: NaPi-IIc transporter scaling factor
        xPit2PT: Pit2 transporter scaling factor

    Returns:
        fvec: Residual vector [n] (zero at solution)
    """
    state_a, state_b, geometry = _unpack_parameters(x, pars, tube, Lz)
    Jvb, Jsb = _evaluate_fluxes(tube, Lz, idid, x, PTinitVol,
                                 xNaPiIIaPT, xNaPiIIcPT, xPit2PT)
    return _compute_residuals(state_a, state_b, geometry, Jvb, Jsb,
                               tube, Lz, idid, n)


def _unpack_parameters(
    x: np.ndarray,
    pars: np.ndarray,
    tube: List[Membrane],
    Lz: int
) -> tuple:
    """
    Unpack parameter vector and solution vector into state dictionaries.

    Returns:
        state_a: Known upstream state at position Lz
        state_b: Unknown downstream state at position Lz+1
        geometry: Geometric parameters
    """
    # --- Upstream state at position Lz (known) ---
    Ca = np.zeros((NS, NC))
    Vola = np.zeros(NC)
    # Jva  = np.zeros((NC, NC))   # upstream water flux array — unused
    # Jsa  = np.zeros((NS, NC, NC))  # upstream solute flux array — unused
    # CaBT = np.zeros(NC)         # upstream bath concentration — unused

    Ca[:, LUM]  = pars[0:NS]
    Ca[:, P]    = pars[NS:2*NS]
    Ca[:, LIS]  = pars[2*NS:3*NS]

    Vola[LUM] = pars[3*NS]
    Vola[P]   = pars[3*NS + 1]
    Vola[LIS] = pars[3*NS + 2]
    PMa       = pars[3*NS + 3]

    Ca[:, BATH] = pars[3*NS + 4 : 4*NS + 4]

    # --- Downstream state at position Lz+1 (unknowns) ---
    Cb   = np.zeros((NS, NC))
    Volb = np.zeros(NC)
    EPb  = np.zeros(NC)
    phb  = np.zeros(NC)

    # Interleaved concentrations: x[3*i], x[3*i+1], x[3*i+2] = LUM, P, LIS for solute i
    Cb[:NS2, LUM] = x[0:3*NS2:3]
    Cb[:NS2, P]   = x[1:3*NS2:3]
    Cb[:NS2, LIS] = x[2:3*NS2:3]

    Volb[LUM] = x[3*NS2]
    Volb[P]   = x[3*NS2 + 1]
    Volb[LIS] = x[3*NS2 + 2]
    EPb[LUM]  = x[3*NS2 + 3]
    EPb[P]    = x[3*NS2 + 4]
    EPb[LIS]  = x[3*NS2 + 5]
    PMb       = x[3*NS2 + 6]

    # pH from H+ concentration (C_H in mM, pH = -log10(C_H/1000))
    for comp in (LUM, P, LIS):
        phb[comp] = -np.log10(Cb[H, comp] / 1000.0)

    # Bath concentrations and EP at Lz+1 (from parameter vector)
    Cb[:, BATH] = pars[4*NS + 4 : 5*NS + 4]
    EPb[BATH]   = pars[5*NS + 4]

    # Geometric parameters
    dimL     = pars[5*NS + 5]
    CPimpref = pars[5*NS + 6]
    Diam     = pars[5*NS + 7]

    state_a = {'C': Ca, 'Vol': Vola, 'PM': PMa}
    state_b = {'C': Cb, 'Vol': Volb, 'EP': EPb, 'pH': phb, 'PM': PMb}
    geometry = {
        'length':    dimL,
        'diameter':  Diam,
        'area':      np.pi * Diam**2 / 4.0,
        'perimeter': np.pi * Diam,
        'CPimpref':  CPimpref,
    }

    return state_a, state_b, geometry


def _evaluate_fluxes(
    tube: List[Membrane],
    Lz: int,
    idid: int,
    x: np.ndarray,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> tuple:
    """
    Dispatch to segment-specific flux function.

    Returns:
        Jvb: Water flux array [NC x NC]
        Jsb: Solute flux array [NS x NC x NC]
    """
    bath_conc = tube[Lz + 1].conc[:, BATH]
    bath_ep   = tube[Lz + 1].ep[BATH]
    args = (x, bath_conc, bath_ep, tube, Lz, PTinitVol,
            xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

    if idid == 3:
        from qflux2A import qflux2A
        return qflux2A(*args)
    elif idid == 4:
        from qflux2T import qflux2T
        return qflux2T(*args)
    elif idid == 5:
        from qflux2D import qflux2D
        return qflux2D(*args)
    elif idid == 9:
        from qflux2IMC import qflux2IMC
        return qflux2IMC(*args)


def _compute_residuals(
    state_a: dict,
    state_b: dict,
    geometry: dict,
    Jvb: np.ndarray,
    Jsb: np.ndarray,
    tube: List[Membrane],
    Lz: int,
    idid: int,
    n: int
) -> np.ndarray:
    """
    Compute all residual equations.

    Returns:
        fvec: Residual vector [n]
    """
    fvec = np.zeros(n)

    # IMCD has coalescing tubules — scale cross-sectional geometry once
    if idid == 9:
        coalesce = tube[Lz + 1].coalesce
        geometry = {**geometry,
                    'area':      geometry['area']      * coalesce,
                    'perimeter': geometry['perimeter'] * coalesce}

    S = _compute_source_terms(state_a, state_b, geometry, Jvb, Jsb)

    _set_nonreacting_residuals(fvec, S)
    _set_carbonate_residuals(fvec, S, state_b, geometry, tube, Lz)
    _set_phosphate_residuals(fvec, S, state_b)
    _set_ammonia_residuals(fvec, S, state_b)
    _set_formate_residuals(fvec, S, state_b)
    _set_proton_residuals(fvec, S)
    _set_volume_residuals(fvec, S)
    _set_ep_residuals(fvec, Jsb, state_b, geometry, tube, Lz, idid)
    _set_pressure_residual(fvec, state_a, state_b, geometry, tube, Lz, idid)

    return fvec


def _compute_source_terms(
    state_a: dict,
    state_b: dict,
    geometry: dict,
    Jvb: np.ndarray,
    Jsb: np.ndarray
) -> np.ndarray:
    """
    Compute source terms S for each solute and volume.

    Source term layout:
        S[3*i]       = lumen source for solute i (axial flow + lateral flux)
        S[3*i+1]     = P cell source for solute i (net transepithelial flux)
        S[3*i+2]     = LIS source for solute i (net transepithelial flux)
        S[3*NS2+0]   = lumen volume source
        S[3*NS2+1]   = P cell volume source
        S[3*NS2+2]   = LIS volume source
    """
    S = np.zeros(NDA)

    Ca, Cb   = state_a['C'],   state_b['C']
    Vola, Volb = state_a['Vol'], state_b['Vol']
    Bm   = geometry['perimeter']
    Am   = geometry['area']
    dimL = geometry['length']

    flow_factor = Vref / href

    # Lumen solute sources: axial flow change + lateral flux to P and LIS
    S[0:3*NS2:3] = (Volb[LUM] * Cb[:NS2, LUM] - Vola[LUM] * Ca[:NS2, LUM]) * flow_factor \
                   + Bm * dimL * (Jsb[:NS2, LUM, P] + Jsb[:NS2, LUM, LIS]) / NZ

    # P cell solute sources: net flux into P (from P→LIS + P→BATH − LUM→P)
    S[1:3*NS2:3] = Jsb[:NS2, P, LIS] + Jsb[:NS2, P, BATH] - Jsb[:NS2, LUM, P]

    # LIS solute sources: net flux into LIS (LIS→BATH − LUM→LIS − P→LIS)
    S[2:3*NS2:3] = Jsb[:NS2, LIS, BATH] - Jsb[:NS2, LUM, LIS] - Jsb[:NS2, P, LIS]

    # Lumen volume source
    fvmult = Pfref * Vwbar * Cref
    sumJvb = (Jvb[LUM, P] + Jvb[LUM, LIS]) * fvmult
    S[3*NS2]     = (Volb[LUM] - Vola[LUM]) * Vref + Bm * dimL * sumJvb / NZ

    # Cellular volume sources: net water flux into each compartment
    S[3*NS2 + 1] = Jvb[P, LIS] + Jvb[P, BATH] - Jvb[LUM, P]
    S[3*NS2 + 2] = Jvb[LIS, BATH] - Jvb[LUM, LIS] - Jvb[P, LIS]

    return S


def _set_nonreacting_residuals(fvec: np.ndarray, S: np.ndarray) -> None:
    """
    Set residuals for non-reacting solutes (one equation per compartment).

    Non-reacting solutes: Na⁺, K⁺, Cl⁻, urea, glucose, Ca²⁺
    Residual = source term (mass balance: flux in = flux out at steady state)
    """
    for sol in (NA, K, CL, UREA, GLU, CA):
        fvec[3*sol : 3*sol+3] = S[3*sol : 3*sol+3]


def _set_carbonate_residuals(
    fvec: np.ndarray,
    S: np.ndarray,
    state_b: dict,
    geometry: dict,
    tube: List[Membrane],
    Lz: int
) -> None:
    """
    Set residuals for CO₂/HCO₃⁻/H₂CO₃ buffer system.

    Per compartment (LUM, P, LIS):
        Conservation: S[HCO3] + S[H2CO3] + S[CO2] = 0
        Equilibrium:  pH − pKHCO3 − log10([HCO3⁻]/[H2CO3]) = 0
        Kinetics:     S[CO2] + reaction_volume * (kh*[CO2] − kd*[H2CO3]) = 0
    """
    Cb  = state_b['C']
    phb = state_b['pH']
    Volb = state_b['Vol']
    membrane = tube[Lz + 1]
    facnd = Vref / href

    # Total carbonate conservation (one equation per compartment)
    fvec[3*HCO3 : 3*HCO3+3] = (S[3*HCO3 : 3*HCO3+3]
                                 + S[3*H2CO3 : 3*H2CO3+3]
                                 + S[3*CO2   : 3*CO2  +3])

    # Chemical equilibrium (one equation per compartment)
    for k, comp in enumerate((LUM, P, LIS)):
        fvec[3*H2CO3 + k] = (phb[comp] - pKHCO3
                              - np.log10(Cb[HCO3, comp] / Cb[H2CO3, comp]))

    # CO₂ hydration kinetics (one equation per compartment)
    # LUM: reaction occurs in luminal volume (Am * dimL / NZ)
    # fkin1 = membrane.dkh[LUM] * Ca[CO2, LUM] - membrane.dkd[LUM] * Ca[H2CO3, LUM]  # upstream kinetics — unused
    fkin_lum = membrane.dkh[LUM] * Cb[CO2, LUM] - membrane.dkd[LUM] * Cb[H2CO3, LUM]
    fvec[3*CO2] = S[3*CO2] + geometry['area'] * geometry['length'] * fkin_lum / NZ / href

    # P: reaction occurs in cell volume
    fkin_p = membrane.dkh[P] * Cb[CO2, P] - membrane.dkd[P] * Cb[H2CO3, P]
    fvec[3*CO2 + 1] = S[3*CO2 + 1] + Volb[P] * fkin_p * facnd

    # LIS: reaction occurs in LIS volume (use initial volume as floor)
    fkin_lis = membrane.dkh[LIS] * Cb[CO2, LIS] - membrane.dkd[LIS] * Cb[H2CO3, LIS]
    fvec[3*CO2 + 2] = S[3*CO2 + 2] + max(Volb[LIS], membrane.volEinit) * fkin_lis * facnd


def _set_phosphate_residuals(
    fvec: np.ndarray,
    S: np.ndarray,
    state_b: dict
) -> None:
    """
    Set residuals for HPO₄²⁻/H₂PO₄⁻ buffer system.

    Per compartment (LUM, P, LIS):
        Conservation: S[HPO4] + S[H2PO4] = 0
        Equilibrium:  pH − pKHPO4 − log10([HPO4²⁻]/[H2PO4⁻]) = 0
    """
    Cb  = state_b['C']
    phb = state_b['pH']

    # Total phosphate conservation
    fvec[3*HPO4 : 3*HPO4+3] = S[3*HPO4 : 3*HPO4+3] + S[3*H2PO4 : 3*H2PO4+3]

    # Chemical equilibrium
    for k, comp in enumerate((LUM, P, LIS)):
        fvec[3*H2PO4 + k] = (phb[comp] - pKHPO4
                              - np.log10(Cb[HPO4, comp] / Cb[H2PO4, comp]))


def _set_ammonia_residuals(
    fvec: np.ndarray,
    S: np.ndarray,
    state_b: dict
) -> None:
    """
    Set residuals for NH₃/NH₄⁺ buffer system.

    Per compartment (LUM, P, LIS):
        Conservation: S[NH3] + S[NH4] = 0
        Equilibrium:  pH − pKNH3 − log10([NH3]/[NH4⁺]) = 0
    """
    Cb  = state_b['C']
    phb = state_b['pH']

    # Total ammonia conservation
    fvec[3*NH3 : 3*NH3+3] = S[3*NH3 : 3*NH3+3] + S[3*NH4 : 3*NH4+3]

    # Chemical equilibrium
    for k, comp in enumerate((LUM, P, LIS)):
        fvec[3*NH4 + k] = (phb[comp] - pKNH3
                           - np.log10(Cb[NH3, comp] / Cb[NH4, comp]))


def _set_formate_residuals(
    fvec: np.ndarray,
    S: np.ndarray,
    state_b: dict
) -> None:
    """
    Set residuals for HCO₂⁻/H₂CO₂ (formate) buffer system.

    Per compartment (LUM, P, LIS):
        Conservation: S[HCO2] + S[H2CO2] = 0
        Equilibrium:  pH − pKHCO2 − log10(|[HCO2⁻]/[H2CO2]|) = 0
    """
    Cb  = state_b['C']
    phb = state_b['pH']

    # Total formate conservation
    fvec[3*HCO2 : 3*HCO2+3] = S[3*HCO2 : 3*HCO2+3] + S[3*H2CO2 : 3*H2CO2+3]

    # Chemical equilibrium
    for k, comp in enumerate((LUM, P, LIS)):
        fvec[3*H2CO2 + k] = (phb[comp] - pKHCO2
                              - np.log10(np.abs(Cb[HCO2, comp] / Cb[H2CO2, comp])))


def _set_proton_residuals(fvec: np.ndarray, S: np.ndarray) -> None:
    """
    Set residuals for H⁺ balance in each compartment.

    Net H⁺ balance accounts for all buffer reactions that produce/consume H⁺:
        NH₄⁺ → NH₃ + H⁺    (releases H⁺, add S[NH4])
        HCO₃⁻ + H⁺ → H₂CO₃ (consumes H⁺, subtract S[HCO3])
        HPO₄²⁻ + H⁺ → H₂PO₄⁻ (consumes H⁺, subtract S[HPO4])
        HCO₂⁻ + H⁺ → H₂CO₂  (consumes H⁺, subtract S[HCO2])
    """
    fvec[3*H : 3*H+3] = (S[3*H   : 3*H  +3]
                          + S[3*NH4  : 3*NH4 +3]
                          - S[3*HPO4 : 3*HPO4+3]
                          - S[3*HCO3 : 3*HCO3+3]
                          - S[3*HCO2 : 3*HCO2+3])


def _set_volume_residuals(fvec: np.ndarray, S: np.ndarray) -> None:
    """
    Set residuals for volume balance in lumen, P cell, and LIS.

    The volume source terms already encode:
        LUM: (outflow − inflow) + lateral water flux
        P, LIS: net transepithelial water flux (steady-state volume)
    """
    fvec[3*NS2 : 3*NS2+3] = S[3*NS2 : 3*NS2+3]


def _set_ep_residuals(
    fvec: np.ndarray,
    Jsb: np.ndarray,
    state_b: dict,
    geometry: dict,
    tube: List[Membrane],
    Lz: int,
    idid: int
) -> None:
    """
    Set residuals for electrical potentials.

    Lumen: zero net current condition (∑ z_i * J_i = 0)
    P cell: electroneutrality (∑ z_i * C_i + impermeant charges = 0)
    LIS: electroneutrality (∑ z_i * C_i = 0)
    """
    Cb  = state_b['C']
    phb = state_b['pH']
    Volb = state_b['Vol']
    membrane = tube[Lz + 1]
    CPimpref = geometry['CPimpref']

    # Segment-specific intracellular buffer concentration
    cp_buffer_map = {3: CPbuftotA, 4: CPbuftotT, 5: CPbuftotD, 9: CPbuftotIMC}
    CPbuffer = cp_buffer_map[idid]

    # Zero net current in lumen: ∑ z_i * (J_i(LUM→P) + J_i(LUM→LIS)) = 0
    fvec[3*NS2 + 3] = np.sum(zval * (Jsb[:, LUM, P] + Jsb[:, LUM, LIS]))

    # Electroneutrality in P cell (impermeant proteins + buffer contribute)
    volPrat = membrane.volPinit / Volb[P]
    CimpP   = CPimpref * volPrat
    facP    = np.exp(np.log(10.0) * (phb[P] - pKbuf))
    CbufP   = CPbuffer * volPrat * facP / (facP + 1)
    fvec[3*NS2 + 4] = zPimpPT * CimpP - CbufP + np.sum(zval * Cb[:, P])

    # Electroneutrality in LIS (no fixed charges)
    fvec[3*NS2 + 5] = np.sum(zval * Cb[:, LIS])


def _set_pressure_residual(
    fvec: np.ndarray,
    state_a: dict,
    state_b: dict,
    geometry: dict,
    tube: List[Membrane],
    Lz: int,
    idid: int
) -> None:
    """
    Set residual for pressure balance (Poiseuille flow).

    Pressure drop ΔP = (8μ/πr⁴) * Q * ΔL where:
        μ = dynamic viscosity
        r = tube radius
        Q = volumetric flow rate
        ΔL = segment length

    For IMCD: extra coalescence resistance is added (merging tubules).
    Note: geometry['area'] is already adjusted by coalesce for IMCD.
    """
    Am    = geometry['area']
    dimL  = geometry['length']
    ratio = 8.0 * np.pi * visc / Am**2

    if idid == 9:
        ratio *= tube[Lz + 1].coalesce * 2

    fvec[3*NS2 + 6] = (state_b['PM'] - state_a['PM']
                       + ratio * state_b['Vol'][LUM] * Vref * dimL / NZ)
