"""
mTAL flux function.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes water and solute fluxes across all mTAL membranes at one
spatial position. Called by qnewton2b (idid=3) at each Newton step.

Transporters modelled:
    Apical (LUM→P):
        - NKCC2 F and A isoforms (Na-K-2Cl cotransporter)
        - NHE3 (Na⁺/H⁺ exchanger isoform 3)
    Basolateral (P→LIS, P→BATH):
        - Na-K-ATPase
        - KCC4 (K⁺-Cl⁻ cotransporter isoform 4)
        - NHE1 (Na⁺/H⁺ exchanger)
        - AE (Cl⁻/HCO₃⁻ exchanger)
        - Na²⁺/HPO₄²⁻ cotransporter
        - Na⁺/HCO₃⁻ cotransporter (NBC)
    Paracellular (LUM→LIS):
        - Electro-convective-diffusive (ECD) fluxes
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


def qflux2A(
    x: np.ndarray,
    Cext: np.ndarray,
    EPext: float,
    mtal: List[Membrane],
    Lz: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> tuple:
    """
    Compute water and solute fluxes in the mTAL.

    Args:
        x: Solution vector (concentrations, volumes, EP, pressure at Lz+1)
        Cext: Bath (peritubular) concentrations [NS]
        EPext: Bath electrical potential
        mtal: mTAL membrane array
        Lz: Current spatial position (0-based)
        PTinitVol: Initial PT luminal volume (passed through, unused in mTAL)
        xNaPiIIaPT, xNaPiIIcPT, xPit2PT: PT transporter scalings (unused in mTAL)

    Returns:
        Jvol: Water flux array [NC x NC]
        Jsol: Solute flux array [NS x NC x NC]
    """
    membrane = mtal[Lz + 1]
    # LzA = Lz  # alias from v0 (unused)

    C, Vol, EP, PM = _unpack_state(x, Cext, EPext)
    _update_surface_area(membrane, Vol)

    Jvol = _compute_water_fluxes(C, PM, Vol, membrane, PTinitVol)
    Jsol, delmu = _compute_ecd_fluxes(C, EP, membrane, Jvol)

    _compute_cotransporter_fluxes(Jsol, delmu, C, membrane)
    _compute_exchanger_fluxes(Jsol, delmu, C, membrane)
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
    """
    C   = np.zeros((NS, NC))
    Vol = np.zeros(NC)
    EP  = np.zeros(NC)

    # Dead variables from v0 (kept for reference):
    # ONC   = np.zeros(NC)          # unused
    # PRES  = np.zeros(NC)          # unused
    # dmu   = np.zeros((NS, NC))    # unused
    # ph    = np.zeros(NC)          # computed but never used
    # hkconc = np.zeros(4)          # unused (HKATPase)
    # Amat  = np.zeros((Natp,Natp)) # unused (HKATPase matrix)
    # theta = np.zeros(NC)          # unused
    # Slum  = np.zeros(NC)          # unused
    # Slat  = np.zeros(NC)          # unused
    # Sbas  = np.zeros(NC)          # unused
    # lwork = 10000                 # reserved for HKATPase matrix inversion
    # ipiv  = np.zeros(Natp)        # unused
    # work  = np.zeros(lwork)       # unused
    # eps   = 1.0e-4                # for Peclet terms — unused
    # (v0 also computed ph[K] = -log10(C[H,K]/1e3) for all K, but ph was never used)

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

    # A and B cell compartments use P cell values (dummy — no transport in mTAL)
    C[:, A] = C[:, P]
    C[:, B] = C[:, P]

    return C, Vol, EP, PM


def _update_surface_area(membrane: Membrane, Vol: np.ndarray) -> None:
    """
    Update the LIS-BATH (basement membrane) area.

    The basolateral area scales with LIS volume when it expands beyond
    the initial value, reflecting stretching of the basement membrane.
    """
    area = membrane.sbasEinit * max(Vol[LIS] / membrane.volEinit, 1.0)
    membrane.area[LIS,  BATH] = area
    membrane.area[BATH, LIS]  = area


def _compute_water_fluxes(
    C: np.ndarray,
    PM: float,
    Vol: np.ndarray,
    membrane: Membrane,
    PTinitVol: float
) -> np.ndarray:
    """Compute osmotic water fluxes across all mTAL membranes."""
    from compute_water_fluxes import compute_water_fluxes
    return compute_water_fluxes(
        C, PM, 0, Vol,
        membrane.volLuminit, membrane.volEinit, membrane.volPinit, CPimprefA,
        membrane.volAinit,   CAimprefA,         membrane.volBinit, CBimprefA,
        membrane.area, membrane.sig, membrane.dLPV,
        complA, PTinitVol
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
        - Basolateral Na²⁺/HPO₄²⁻ cotransporter  (P→LIS, P→BATH)
        - Basolateral Na⁺/HCO₃⁻ cotransporter (NBC)  (P→LIS, P→BATH)
        - Apical NKCC2 F-isoform  (LUM→P)
        - Apical NKCC2 A-isoform  (LUM→P)
        - Basolateral KCC4  (P→LIS, P→BATH)
    """
    from compute_nkcc2_flux import compute_nkcc2_flux
    from compute_kcc_fluxes import compute_kcc_fluxes

    # Basolateral Na²⁺/HPO₄²⁻ cotransporter: 2 Na⁺ co-transported with 1 HPO₄²⁻
    for L in (LIS, BATH):
        dJNaP = (membrane.area[P, L] * membrane.dLA[NA, HPO4, P, L]
                 * (2 * delmu[NA, P, L] + delmu[HPO4, P, L]))
        Jsol[NA,   P, L] += 2 * dJNaP
        Jsol[HPO4, P, L] += dJNaP
    # fluxNaPatPES = sumJES * convert  # diagnostic — unused

    # Basolateral Na⁺/HCO₃⁻ cotransporter (NBC): 1 Na⁺ co-transported with 3 HCO₃⁻
    # Note: coefficient dLA[NA, CL, P, L] used per original Fortran translation
    for L in (LIS, BATH):
        dJNaBic = (membrane.area[P, L] * membrane.dLA[NA, CL, P, L]
                   * (delmu[NA, P, L] + 3 * delmu[HCO3, P, L]))
        Jsol[NA,   P, L] += dJNaBic
        Jsol[HCO3, P, L] += 3 * dJNaBic
    # fluxNaBicPES = sumJES * convert  # diagnostic — unused

    # Apical NKCC2 F-isoform (dominant isoform in mTAL)
    dJnF, dJkF, dJcF, dJmF = compute_nkcc2_flux(
        C, membrane.area[LUM, P], membrane.xNKCC2F,
        bn2F, bk2F, bc2F, bm2F, popnkccF,
        pnkccpF, pnmccpF, poppnkccF, pnkccppF, pnmccppF
    )
    Jsol[NA,  LUM, P] += dJnF
    Jsol[K,   LUM, P] += dJkF
    Jsol[CL,  LUM, P] += dJcF
    Jsol[NH4, LUM, P] += dJmF
    # fluxnNKCC2F = dJnF * convert  # diagnostic — unused
    # fluxkNKCC2F = dJkF * convert  # diagnostic — unused
    # fluxcNKCC2F = dJcF * convert  # diagnostic — unused
    # fluxmNKCC2F = dJmF * convert  # diagnostic — unused

    # Apical NKCC2 A-isoform
    dJnA, dJkA, dJcA, dJmA = compute_nkcc2_flux(
        C, membrane.area[LUM, P], membrane.xNKCC2A,
        bn2A, bk2A, bc2A, bm2A, popnkccA,
        pnkccpA, pnmccpA, poppnkccA, pnkccppA, pnmccppA
    )
    Jsol[NA,  LUM, P] += dJnA
    Jsol[K,   LUM, P] += dJkA
    Jsol[CL,  LUM, P] += dJcA
    Jsol[NH4, LUM, P] += dJmA
    # fluxnNKCC2A = dJnA * convert  # diagnostic — unused
    # fluxkNKCC2A = dJkA * convert  # diagnostic — unused
    # fluxcNKCC2A = dJcA * convert  # diagnostic — unused
    # fluxmNKCC2A = dJmA * convert  # diagnostic — unused
    # fluxnNKCC2  = fluxnNKCC2A + fluxnNKCC2F  # diagnostic — unused
    # fluxkNKCC2  = fluxkNKCC2A + fluxkNKCC2F  # diagnostic — unused
    # fluxcNKCC2  = fluxcNKCC2A + fluxcNKCC2F  # diagnostic — unused
    # fluxmNKCC2  = fluxmNKCC2A + fluxmNKCC2F  # diagnostic — unused

    # Basolateral KCC4 cotransporter (K⁺-Cl⁻, with NH₄⁺ substitution for K⁺)
    dJk5, dJc5, dJm5, dJk6, dJc6, dJm6 = compute_kcc_fluxes(
        C, membrane.area[P, LIS], membrane.area[P, BATH], membrane.xKCC4
    )
    Jsol[K,   P, LIS]  += dJk5
    Jsol[CL,  P, LIS]  += dJc5
    Jsol[NH4, P, LIS]  += dJm5
    Jsol[K,   P, BATH] += dJk6
    Jsol[CL,  P, BATH] += dJc6
    Jsol[NH4, P, BATH] += dJm6
    # fluxkKCC = (dJk5 + dJk6) * convert  # diagnostic — unused
    # fluxcKCC = (dJc5 + dJc6) * convert  # diagnostic — unused
    # fluxmKCC = (dJm5 + dJm6) * convert  # diagnostic — unused


def _compute_exchanger_fluxes(
    Jsol: np.ndarray,
    delmu: np.ndarray,
    C: np.ndarray,
    membrane: Membrane
) -> None:
    """
    Add exchanger flux contributions to Jsol (in place).

    Transporters:
        - Apical NHE3 (Na⁺/H⁺ exchanger isoform 3)
        - Basolateral NHE1 (Na⁺/H⁺ exchanger)
        - Basolateral AE (Cl⁻/HCO₃⁻ exchanger)
    """
    from compute_nhe3_fluxes import compute_nhe3_fluxes

    # Apical NHE3: Na⁺ in exchange for H⁺ (and NH₄⁺)
    dJNHEsod, dJNHEprot, dJNHEamm = compute_nhe3_fluxes(
        C, membrane.area[LUM, P], membrane.xNHE3
    )
    Jsol[NA,  LUM, P] += dJNHEsod
    Jsol[H,   LUM, P] += dJNHEprot
    Jsol[NH4, LUM, P] += dJNHEamm
    # fluxNHEsod  = dJNHEsod  * convert  # diagnostic — unused
    # fluxNHEprot = dJNHEprot * convert  # diagnostic — unused
    # fluxNHEamm  = dJNHEamm  * convert  # diagnostic — unused

    # Basolateral NHE1: Na⁺ in exchange for H⁺
    for L in (LIS, BATH):
        dJNaH = (membrane.area[P, L] * membrane.dLA[NA, H, P, L]
                 * (delmu[NA, P, L] - delmu[H, P, L]))
        Jsol[NA, P, L] += dJNaH
        Jsol[H,  P, L] -= dJNaH
    # fluxNaHPES = sumJES * convert  # diagnostic — unused

    # Basolateral Cl⁻/HCO₃⁻ exchanger (AE): Cl⁻ in exchange for HCO₃⁻
    for L in (LIS, BATH):
        dJClHCO3 = (membrane.area[P, L] * membrane.dLA[CL, HCO3, P, L]
                    * (delmu[CL, P, L] - delmu[HCO3, P, L]))
        Jsol[CL,   P, L] += dJClHCO3
        Jsol[HCO3, P, L] -= dJClHCO3
    # fluxClHCO3exPES = sumJES * convert  # diagnostic — unused


def _compute_active_transport_fluxes(
    Jsol: np.ndarray,
    C: np.ndarray,
    membrane: Membrane
) -> float:
    """
    Add Na-K-ATPase flux contributions to Jsol (in place).

    The pump transports 3 Na⁺ out and 2 K⁺ (or NH₄⁺) in per cycle.
    NH₄⁺ competes with K⁺ at the extracellular site (ratio ro).

    Returns:
        Total Na⁺ flux via NaKATPase (dJact5 + dJact6) for diagnostic storage.
    """
    # Intracellular activation: Na⁺ with Cl⁻-dependent affinity
    AffNa = 0.2 * (1.0 + C[K,  P]    / 8.33)
    actNa = C[NA, P] / (C[NA, P] + AffNa)

    # Extracellular K⁺/NH₄⁺ activation: affinity modulated by bath Na⁺
    AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
    AffNH4 = AffK

    actK5 = C[K, LIS]  / (C[K, LIS]  + AffK)
    actK6 = C[K, BATH] / (C[K, BATH] + AffK)

    # NH₄⁺/K⁺ competition ratio at extracellular site
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

    # fluxNaKPESpot = -2/3.0*(dJact5/(1+ro5)+dJact6/(1+ro6))*convert  # diagnostic — unused
    # fluxNaKPESamm = -2/3.0*(dJact5*ro5/(1+ro5)+dJact6*ro6/(1+ro6))*convert  # diagnostic — unused

    return dJact5 + dJact6


def _store_local_fluxes(
    Jsol: np.ndarray,
    membrane: Membrane,
    Jnak: float
) -> None:
    """
    Store local flux diagnostics on the membrane object for integration.

    Units: pmol/min/mm tubule
    """
    cv = href * Cref * np.pi * DiamA * 60 / 10 * 1.0e9
    # cvw = Pfref * Cref * Vwbar * np.pi * DiamA * 60 / 10 * 1.0e6  # water flux conversion — unused

    membrane.FNatrans = Jsol[NA, LUM, P]   * cv
    membrane.FNapara  = Jsol[NA, LUM, LIS] * cv
    membrane.FNaK     = Jnak * cv
    membrane.FHase    = 0.0
    membrane.FKtrans  = Jsol[K,  LUM, P]   * cv
    membrane.FKpara   = Jsol[K,  LUM, LIS] * cv
