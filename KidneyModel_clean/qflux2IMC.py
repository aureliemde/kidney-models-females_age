"""
IMCD flux function.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes water and solute fluxes across all IMCD membranes at one
spatial position. Called by qnewton2b (idid=9) at each Newton step.

Transporters modelled:
    Apical (LUM→P):
        - ENaC (epithelial Na⁺ channel, pH- and [Na⁺]-dependent)
        - ROMK (K⁺ channel, pH-dependent)
        - NaCl cotransporter
        - H-K-ATPase
    Apical paracellular (LUM→LIS):
        - Cl⁻ tight-junction conductance (pH-dependent)
    Basolateral (P→LIS, P→BATH):
        - Na-K-ATPase
        - NHE1 (Na⁺/H⁺ exchanger)
        - AE (Cl⁻/HCO₃⁻ exchanger)
        - Na²⁺/HPO₄²⁻ cotransporter
        - KCl cotransporter
        - NaKCl₂ cotransporter
    Paracellular (LUM→LIS):
        - Electro-convective-diffusive (ECD) fluxes
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


# Module-level IMCD initialization (runs once at import time)
imcd = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initIMC_Var import initIMC_Var
hENaC_IMC, hROMK_IMC, hCltj_IMC = initIMC_Var(imcd)


def qflux2IMC(
    x: np.ndarray,
    Cext: np.ndarray,
    EPext: float,
    imcd: List[Membrane],
    Lz: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> tuple:
    """
    Compute water and solute fluxes in the IMCD.

    Args:
        x: Solution vector (concentrations, volumes, EP, pressure at Lz+1)
        Cext: Bath (peritubular) concentrations [NS]
        EPext: Bath electrical potential
        imcd: IMCD membrane array
        Lz: Current spatial position (0-based)
        PTinitVol: Initial PT luminal volume (passed through, unused in IMCD)
        xNaPiIIaPT, xNaPiIIcPT, xPit2PT: PT transporter scalings (unused in IMCD)

    Returns:
        Jvol: Water flux array [NC x NC]
        Jsol: Solute flux array [NS x NC x NC]
    """
    membrane = imcd[Lz + 1]
    # LzIMC = Lz  # alias from v0 — unused in refactored code

    C, Vol, EP, PM, ph = _unpack_state(x, Cext, EPext)
    _update_surface_area(membrane, Vol)
    _update_permeabilities(membrane, C, ph)

    Jvol = _compute_water_fluxes(C, PM, Vol, membrane, PTinitVol)
    Jsol, delmu = _compute_ecd_fluxes(C, EP, membrane, Jvol)

    _compute_cotransporter_fluxes(Jsol, delmu, C, membrane)
    _compute_exchanger_fluxes(Jsol, delmu, membrane)
    Jnak = _compute_active_transport_fluxes(Jsol, C, membrane)

    _store_local_fluxes(Jsol, membrane, Jnak)

    return Jvol, Jsol


def _unpack_state(
    x: np.ndarray,
    Cext: np.ndarray,
    EPext: float
) -> tuple:
    """
    Unpack solution vector and bath boundary values into state arrays.

    Returns:
        C:   Concentration array [NS x NC]
        Vol: Volume array [NC]
        EP:  Electrical potential array [NC]
        PM:  Luminal pressure
        ph:  pH array [NC]
    """
    C   = np.zeros((NS, NC))
    Vol = np.zeros(NC)
    EP  = np.zeros(NC)

    # Pre-allocated in v0 but unused in refactored code — kept for reference:
    # ONC    = np.zeros(NC)            # oncotic pressures (handled inside compute_water_fluxes)
    # PRES   = np.zeros(NC)            # hydraulic pressures (handled inside compute_water_fluxes)
    # dmu    = np.zeros((NS, NC))      # electrochemical potential (superseded by delmu in compute_ecd_fluxes)
    # hkconc = np.zeros(4)             # HKATPase concentrations — constructed inline in refactored code
    # Amat   = np.zeros((Natp, Natp))  # HKATPase matrix — constructed inline in refactored code
    # theta  = np.zeros(NC)            # unused
    # Slum   = np.zeros(NC)            # unused
    # Slat   = np.zeros(NC)            # unused
    # Sbas   = np.zeros(NC)            # unused
    # lwork  = 10000                   # LAPACK workspace size — unused
    # ipiv   = np.zeros(Natp)          # LAPACK pivot array — unused
    # work   = np.zeros(lwork)         # LAPACK work array — unused

    C[:, BATH] = Cext
    EP[BATH]   = EPext

    # Interleaved concentrations: x[3*i], x[3*i+1], x[3*i+2] = LUM, P, LIS for solute i
    C[:NS2, LUM] = x[0:3*NS2:3]
    C[:NS2, P]   = x[1:3*NS2:3]
    C[:NS2, LIS] = x[2:3*NS2:3]

    xb       = 3 * NS2
    Vol[LUM] = x[xb]
    Vol[P]   = x[xb + 1]
    Vol[LIS] = x[xb + 2]
    EP[LUM]  = x[xb + 3]
    EP[P]    = x[xb + 4]
    EP[LIS]  = x[xb + 5]
    PM       = x[xb + 6]

    # A and B cell compartments use P cell values (dummy — no transport in IMCD)
    C[:, A] = C[:, P]
    C[:, B] = C[:, P]

    # pH from H⁺ concentration (converted from model units to molar)
    ph = -np.log10(C[H, :] / 1.0e3)

    return C, Vol, EP, PM, ph


def _update_surface_area(membrane: Membrane, Vol: np.ndarray) -> None:
    """
    Update the LIS-BATH (basement membrane) area.

    The basolateral area scales with LIS volume when it expands beyond
    the initial value, reflecting stretching of the basement membrane.
    """
    area = membrane.sbasEinit * max(Vol[LIS] / membrane.volEinit, 1.0)
    membrane.area[LIS,  BATH] = area
    membrane.area[BATH, LIS]  = area


def _update_permeabilities(
    membrane: Membrane,
    C: np.ndarray,
    ph: np.ndarray
) -> None:
    """
    Update pH- and [Na⁺]-dependent channel permeabilities.

    ENaC and ROMK permeabilities depend on intracellular pH (ph[P]).
    Apical Cl⁻ tight-junction conductance depends on LIS pH (ph[LIS]).
    ENaC additionally depends on luminal and cellular [Na⁺].
    """
    facphMP = 0.1 + 2.0 / (1 + np.exp(-6.0 * (ph[P] - 7.50)))
    facphTJ = 2.0 / (1.0 + np.exp(10.0 * (ph[LIS] - 7.32)))
    facNaMP = (30 / (30 + C[NA, LUM])) * (50 / (50 + C[NA, P]))

    membrane.h[NA, LUM, P]   = hENaC_IMC * facNaMP * facphMP
    membrane.h[K,  LUM, P]   = hROMK_IMC * facphMP
    membrane.h[CL, LUM, LIS] = hCltj_IMC * facphTJ


def _compute_water_fluxes(
    C: np.ndarray,
    PM: float,
    Vol: np.ndarray,
    membrane: Membrane,
    PTinitVol: float
) -> np.ndarray:
    """Compute osmotic water fluxes across all IMCD membranes."""
    from compute_water_fluxes import compute_water_fluxes
    return compute_water_fluxes(
        C, PM, 0, Vol,
        membrane.volLuminit, membrane.volEinit, membrane.volPinit, CPimprefIMC,
        membrane.volAinit,   CAimprefIMC,        membrane.volBinit, CBimprefIMC,
        membrane.area, membrane.sig, membrane.dLPV,
        complIMC, PTinitVol
    )


def _compute_ecd_fluxes(
    C: np.ndarray,
    EP: np.ndarray,
    membrane: Membrane,
    Jvol: np.ndarray
) -> tuple:
    """
    Compute electro-convective-diffusive (ECD) solute fluxes.

    Returns:
        Jsol:  Solute flux array [NS x NC x NC]
        delmu: Electrochemical potential difference array [NS x NC x NC]
    """
    # Channel flux diagnostics from v0 — computed immediately after ECD call but unused:
    # convert = href * Cref * np.pi * DiamIMC * 60 / 10 * 1.0e9
    # fluxNachMP    = Jsol[NA,    LUM, P]   * convert
    # fluxNachPES   = (Jsol[NA,   P, LIS] + Jsol[NA,   P, BATH]) * convert
    # fluxKchMP     = Jsol[K,     LUM, P]   * convert
    # fluxKchPES    = (Jsol[K,    P, LIS] + Jsol[K,    P, BATH]) * convert
    # fluxClchMP    = Jsol[CL,    LUM, P]   * convert
    # fluxClchPES   = (Jsol[CL,   P, LIS] + Jsol[CL,   P, BATH]) * convert
    # fluxBichPES   = (Jsol[HCO3, P, LIS] + Jsol[HCO3, P, BATH]) * convert
    # fluxH2CO3MP   = Jsol[H2CO3, LUM, P]  * convert
    # fluxCO2MP     = Jsol[CO2,   LUM, P]  * convert
    # fluxH2CO3PES  = (Jsol[H2CO3,P, LIS] + Jsol[H2CO3,P, BATH]) * convert
    # fluxCO2PES    = (Jsol[CO2,  P, LIS] + Jsol[CO2,  P, BATH]) * convert
    # fluxHP2mchPES = (Jsol[HPO4, P, LIS] + Jsol[HPO4, P, BATH]) * convert
    # fluxHPmPES    = (Jsol[H2PO4,P, LIS] + Jsol[H2PO4,P, BATH]) * convert
    # fluxNH3MP     = Jsol[NH3,   LUM, P]  * convert
    # fluxNH3PES    = (Jsol[NH3,  P, LIS] + Jsol[NH3,  P, BATH]) * convert
    from compute_ecd_fluxes import compute_ecd_fluxes
    return compute_ecd_fluxes(C, EP, membrane.area, membrane.sig, membrane.h, Jvol)


def _compute_cotransporter_fluxes(
    Jsol: np.ndarray,
    delmu: np.ndarray,
    C: np.ndarray,
    membrane: Membrane
) -> None:
    """
    Add cotransporter flux contributions to Jsol (in place).

    Transporters:
        - Apical NaCl cotransporter  (LUM→P)
        - Basolateral Na²⁺/HPO₄²⁻ cotransporter  (P→LIS, P→BATH)
        - Basolateral KCl cotransporter  (P→LIS, P→BATH)
        - Basolateral NaKCl₂ cotransporter  (P→LIS, P→BATH)
    """
    # Apical NaCl cotransporter
    dJNaCl = (membrane.area[LUM, P] * membrane.dLA[NA, CL, LUM, P]
              * (delmu[NA, LUM, P] + delmu[CL, LUM, P]))
    Jsol[NA, LUM, P] += dJNaCl
    Jsol[CL, LUM, P] += dJNaCl
    # fluxNaClMP = dJNaCl * convert  # apical NaCl flux diagnostic — unused

    # Basolateral Na²⁺/HPO₄²⁻ cotransporter (factor 1/2 for AMW model consistency)
    for L in (LIS, BATH):
        dJNaP = (membrane.area[P, L] * membrane.dLA[NA, HPO4, P, L] / 2
                 * (2 * delmu[NA, P, L] + delmu[HPO4, P, L]))
        Jsol[NA,   P, L] += 2 * dJNaP
        Jsol[HPO4, P, L] += dJNaP
    # sumJES = sum of 2*dJNaP over L in (LIS, BATH) — unused diagnostic
    # fluxNaPatPES = sumJES * convert  # Na flux via Na²⁺/HPO₄²⁻ cotransporter — unused

    # Basolateral KCl cotransporter
    for L in (LIS, BATH):
        dJKCl = (membrane.area[P, L] * membrane.dLA[K, CL, P, L]
                 * (delmu[K, P, L] + delmu[CL, P, L]))
        Jsol[K,  P, L] += dJKCl
        Jsol[CL, P, L] += dJKCl
    # sumJES = sum of dJKCl over L in (LIS, BATH) — unused diagnostic
    # fluxKClPES = sumJES * convert  # KCl flux diagnostic — unused

    # Basolateral NaKCl₂ cotransporter (1 Na⁺ + 1 K⁺ + 2 Cl⁻)
    for L in (LIS, BATH):
        dJNKCl2 = (membrane.area[P, L] * membrane.dLA[NA, K, P, L]
                   * (delmu[NA, P, L] + delmu[K, P, L] + 2 * delmu[CL, P, L]))
        Jsol[NA, P, L] += dJNKCl2
        Jsol[K,  P, L] += dJNKCl2
        Jsol[CL, P, L] += 2 * dJNKCl2
    # sumJES = sum of dJNKCl2 over L in (LIS, BATH) — unused diagnostic
    # fluxNKCl2PES = sumJES * convert  # NaKCl₂ flux diagnostic — unused


def _compute_exchanger_fluxes(
    Jsol: np.ndarray,
    delmu: np.ndarray,
    membrane: Membrane
) -> None:
    """
    Add exchanger flux contributions to Jsol (in place).

    Transporters:
        - Basolateral NHE1 (Na⁺/H⁺ exchanger)
        - Basolateral AE (Cl⁻/HCO₃⁻ exchanger)
    """
    # Basolateral NHE1: Na⁺ in exchange for H⁺
    for L in (LIS, BATH):
        dJNaH = (membrane.area[P, L] * membrane.dLA[NA, H, P, L]
                 * (delmu[NA, P, L] - delmu[H, P, L]))
        Jsol[NA, P, L] += dJNaH
        Jsol[H,  P, L] -= dJNaH
    # sumJES = sum of dJNaH over L in (LIS, BATH) — unused diagnostic
    # fluxNaHPES = sumJES * convert  # NHE1 Na flux diagnostic — unused

    # Basolateral Cl⁻/HCO₃⁻ exchanger (AE): Cl⁻ in exchange for HCO₃⁻
    for L in (LIS, BATH):
        dJClHCO3 = (membrane.area[P, L] * membrane.dLA[CL, HCO3, P, L]
                    * (delmu[CL, P, L] - delmu[HCO3, P, L]))
        Jsol[CL,   P, L] += dJClHCO3
        Jsol[HCO3, P, L] -= dJClHCO3
    # sumJES = sum of dJClHCO3 over L in (LIS, BATH) — unused diagnostic
    # fluxClHCO3exPES = sumJES * convert  # AE Cl/HCO₃ flux diagnostic — unused


def _compute_active_transport_fluxes(
    Jsol: np.ndarray,
    C: np.ndarray,
    membrane: Membrane
) -> float:
    """
    Add ATPase flux contributions to Jsol (in place).

    Na-K-ATPase: 3 Na⁺ out, 2 K⁺ (or NH₄⁺) in per cycle.
    NH₄⁺ affinity is 5× weaker than K⁺ in IMCD (AffNH4 = 5*AffK).

    H-K-ATPase: luminal K⁺ uptake coupled to H⁺ secretion.
    Kinetics solved via fatpase matrix inversion.

    Returns:
        Total Na⁺ flux via NaKATPase (dJact5 + dJact6) for diagnostic storage.
    """
    # --- Na-K-ATPase ---
    # Note: AffNH4 = 5*AffK in IMCD (cf. 1*AffK in other segments)
    AffNa  = 0.2 * (1.0 + C[K,  P]    / 8.33)
    actNa  = C[NA, P] / (C[NA, P] + AffNa)
    AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
    AffNH4 = 5.0 * AffK
    actK5  = C[K, LIS]  / (C[K, LIS]  + AffK)
    actK6  = C[K, BATH] / (C[K, BATH] + AffK)
    ro5 = (C[NH4, LIS]  / AffNH4) / (C[K, LIS]  / AffK)
    ro6 = (C[NH4, BATH] / AffNH4) / (C[K, BATH] / AffK)

    dJact5 = membrane.area[P, LIS]  * membrane.ATPNaK[P, LIS]  * actNa**3 * actK5**2
    dJact6 = membrane.area[P, BATH] * membrane.ATPNaK[P, BATH] * actNa**3 * actK6**2

    Jsol[NA,  P, LIS]  += dJact5
    Jsol[NA,  P, BATH] += dJact6
    Jsol[K,   P, LIS]  -= 2.0/3.0 * dJact5 / (1 + ro5)
    Jsol[K,   P, BATH] -= 2.0/3.0 * dJact6 / (1 + ro6)
    Jsol[NH4, P, LIS]  -= 2.0/3.0 * dJact5 * ro5 / (1 + ro5)
    Jsol[NH4, P, BATH] -= 2.0/3.0 * dJact6 * ro6 / (1 + ro6)

    Jnak = dJact5 + dJact6
    # fluxNaKPESsod = Jnak * convert                                                     # NaKATPase Na  flux diagnostic — unused
    # fluxNaKPESpot = -2/3.0 * (dJact5/(1+ro5) + dJact6/(1+ro6)) * convert             # NaKATPase K   flux diagnostic — unused
    # fluxNaKPESamm = -2/3.0 * (dJact5*ro5/(1+ro5) + dJact6*ro6/(1+ro6)) * convert     # NaKATPase NH₄ flux diagnostic — unused

    # --- Luminal H-K-ATPase ---
    from fatpase import fatpase
    hkconc = np.array([C[K, P], C[K, LUM], C[H, P], C[H, LUM]])
    # Amat_org = fatpase(Natp, hkconc)  # intermediate alias from v0 — unused
    # Amat_n   = Amat_org               # second alias from v0 — unused
    Amat   = np.linalg.inv(fatpase(Natp, hkconc))
    c7 = Amat[6, 0]   # H2-E1-P state
    c8 = Amat[7, 0]   # H2-E2-P state
    dkf5 = 4.0e1
    dkb5 = 2.0e2
    hefflux = (membrane.area[LUM, P] * membrane.ATPHK[LUM, P]
               * (dkf5 * c7 - dkb5 * c8))
    Jsol[K, LUM, P] += hefflux
    Jsol[H, LUM, P] -= hefflux  # No factor 2 to be consistent with AMW model results
    # fluxHKATPaseMP = hefflux * convert  # HKATPase flux diagnostic — unused

    return Jnak


def _store_local_fluxes(
    Jsol: np.ndarray,
    membrane: Membrane,
    Jnak: float
) -> None:
    """
    Store local flux diagnostics on the membrane object for integration.

    Units: pmol/min/mm tubule
    All fluxes are scaled by membrane.coalesce (IMCD cell density factor).
    """
    cv = href * Cref * np.pi * DiamIMC * 60 / 10 * 1.0e9
    # cvw = Pfref * Cref * Vwbar * np.pi * DiamIMC * 60 / 10 * 1.0e6  # dimensional water flux factor — unused

    membrane.FNatrans = Jsol[NA, LUM, P]   * cv * membrane.coalesce
    membrane.FNapara  = Jsol[NA, LUM, LIS] * cv * membrane.coalesce
    membrane.FNaK     = Jnak * cv * cv * membrane.coalesce  # cv applied twice as in original
    membrane.FHase    = 0.0
    membrane.FKtrans  = Jsol[K,  LUM, P]   * cv * membrane.coalesce
    membrane.FKpara   = Jsol[K,  LUM, LIS] * cv * membrane.coalesce
