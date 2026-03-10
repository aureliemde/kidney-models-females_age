"""
DCT flux function.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Computes water and solute fluxes across all DCT membranes at one
spatial position. Called by qnewton2b (idid=5) at each Newton step.

Transporters modelled:
    Apical (LUM→P):
        - ENaC (epithelial Na⁺ channel, DCT2 only)
        - NCC (Na⁺-Cl⁻ cotransporter, DCT1 dominant)
        - NHE3 (Na⁺/H⁺ exchanger isoform 3)
        - KCl cotransporter
        - TRPV5 (Ca²⁺ channel, DCT2 only)
    Basolateral (P→LIS, P→BATH):
        - Na-K-ATPase
        - KCl cotransporter
        - NHE1 (Na⁺/H⁺ exchanger)
        - AE (Cl⁻/HCO₃⁻ exchanger)
        - Na²⁺/HPO₄²⁻ cotransporter
        - NCX (Na⁺/Ca²⁺ exchanger, DCT2 only)
        - PMCA (Ca²⁺ pump, DCT2 only)
    Paracellular (LUM→LIS):
        - Electro-convective-diffusive (ECD) fluxes
"""

import numpy as np
from typing import List

from values import *
from glo import *
from defs import *


# Module-level DCT initialization (runs once at import time)
dct = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
from initD_Var import initD_Var
xTRPV5_dct = initD_Var(dct)


def qflux2D(
    x: np.ndarray,
    Cext: np.ndarray,
    EPext: float,
    dct: List[Membrane],
    Lz: int,
    PTinitVol: float,
    xNaPiIIaPT: float,
    xNaPiIIcPT: float,
    xPit2PT: float
) -> tuple:
    """
    Compute water and solute fluxes in the DCT.

    Args:
        x: Solution vector (concentrations, volumes, EP, pressure at Lz+1)
        Cext: Bath (peritubular) concentrations [NS]
        EPext: Bath electrical potential
        dct: DCT membrane array
        Lz: Current spatial position (0-based)
        PTinitVol: Initial PT luminal volume (passed through, unused in DCT)
        xNaPiIIaPT, xNaPiIIcPT, xPit2PT: PT transporter scalings (unused in DCT)

    Returns:
        Jvol: Water flux array [NC x NC]
        Jsol: Solute flux array [NS x NC x NC]
    """
    membrane = dct[Lz + 1]
    # LzD = Lz  # alias from v0 — unused in refactored code

    C, Vol, EP, PM, ph = _unpack_state(x, Cext, EPext)
    _update_surface_area(membrane, Vol)

    NCCexp, ENaCexp, CaTexp = _compute_segment_factors(Lz)

    Jvol = _compute_water_fluxes(C, PM, Vol, membrane, PTinitVol)
    Jsol, delmu = _compute_ecd_fluxes(C, EP, membrane, Jvol, CaTexp)

    _compute_enac_flux(Jsol, C, EP, ph, membrane, ENaCexp)
    _compute_cotransporter_fluxes(Jsol, delmu, C, membrane, NCCexp)
    Jncxca = _compute_exchanger_fluxes(Jsol, delmu, C, EP, membrane, CaTexp)
    Jnak, Jtrpv5, Jpmca = _compute_active_transport_fluxes(
        Jsol, C, EP, ph, membrane, CaTexp
    )

    _store_local_fluxes(Jsol, membrane, Lz, Jnak, Jtrpv5, Jncxca, Jpmca)

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
    # hkconc = np.zeros(4)             # HKATPase concentrations — no HKATPase in DCT
    # Amat   = np.zeros((Natp, Natp))  # HKATPase matrix — unused
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

    # A and B cell compartments use P cell values (dummy — no transport in DCT)
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


def _compute_segment_factors(Lz: int) -> tuple:
    """
    Compute position-dependent scaling factors for DCT1/DCT2 transition.

    The DCT is split: DCT1 (first 2/3) has NCC and no Ca²⁺ transport;
    DCT2 (last 1/3) has ENaC and full Ca²⁺ transport (TRPV5, NCX, PMCA).
    At the boundary, NCC and ENaC overlap linearly.

    Returns:
        NCCexp:  NCC expression factor (1 → 0 across DCT2)
        ENaCexp: ENaC expression factor (0 → 1 across DCT2)
        CaTexp:  Ca²⁺ transport factor (0.001 in DCT1, 1.0 in DCT2)
    """
    xl    = Lz + 1.0
    xn    = NZ
    xdct2 = 2.0 / 3.0
    if (xl / xn) < xdct2:
        NCCexp  = 1.0
        ENaCexp = 0.0
        CaTexp  = 0.001
    else:
        NCCexp  = 1.0 - (xl / xn - xdct2) / (1.0 - xdct2)
        ENaCexp = (xl / xn - xdct2) / (1.0 - xdct2)
        CaTexp  = 1.0
    return NCCexp, ENaCexp, CaTexp


def _compute_water_fluxes(
    C: np.ndarray,
    PM: float,
    Vol: np.ndarray,
    membrane: Membrane,
    PTinitVol: float
) -> np.ndarray:
    """Compute osmotic water fluxes across all DCT membranes."""
    from compute_water_fluxes import compute_water_fluxes
    return compute_water_fluxes(
        C, PM, 0, Vol,
        membrane.volLuminit, membrane.volEinit, membrane.volPinit, CPimprefD,
        membrane.volAinit,   CAimprefD,         membrane.volBinit, CBimprefD,
        membrane.area, membrane.sig, membrane.dLPV,
        complD, PTinitVol
    )


def _compute_ecd_fluxes(
    C: np.ndarray,
    EP: np.ndarray,
    membrane: Membrane,
    Jvol: np.ndarray,
    CaTexp: float
) -> tuple:
    """
    Compute electro-convective-diffusive (ECD) solute fluxes (inline).

    Uses eps=1e-6 for the exponential Peclet threshold.
    Ca²⁺ ECD fluxes are scaled by CaTexp (suppressed in DCT1).

    Returns:
        Jsol:  Solute flux array [NS x NC x NC]
        delmu: Electrochemical potential difference array [NS x NC x NC]
    """
    eps     = 1.0e-6
    dimless = (Pfref * Vwbar * Cref) / href

    Jsol  = np.zeros((NS, NC, NC))
    delmu = np.zeros((NS, NC, NC))

    # Electrochemical potential: dmu[i, k] = RT*ln|C[i,k]| + z_i*F*EPref*EP[k]
    dmu = (RT * np.log(np.abs(C))
           + (zval[:, np.newaxis] * F * EPref) * EP[np.newaxis, :])

    for i in range(NS2):
        for k in range(NC - 1):
            for l in range(k + 1, NC):
                # Electrodiffusive component
                XI   = zval[i] * F * EPref / RT * (EP[k] - EP[l])
                dint = np.exp(-XI)
                if abs(1.0 - dint) < eps:
                    Jsol[i, k, l] = (membrane.area[k, l] * membrane.h[i, k, l]
                                     * (C[i, k] - C[i, l]))
                else:
                    Jsol[i, k, l] = (membrane.area[k, l] * membrane.h[i, k, l]
                                     * XI * (C[i, k] - C[i, l] * dint) / (1.0 - dint))

                # Convective component
                concdiff = C[i, k] - C[i, l]
                if abs(concdiff) > eps:
                    concmean = concdiff / np.log(abs(C[i, k] / C[i, l]))
                    Jsol[i, k, l] += ((1.0 - membrane.sig[i, k, l])
                                      * concmean * Jvol[k, l] * dimless)

                delmu[i, k, l] = dmu[i, k] - dmu[i, l]

    # Scale Ca²⁺ ECD fluxes by CaTexp (DCT2 only)
    Jsol[CA, LUM, P]    *= CaTexp
    Jsol[CA, P,   LIS]  *= CaTexp
    Jsol[CA, P,   BATH] *= CaTexp

    # Channel flux diagnostics from v0 — computed immediately after ECD loop but unused:
    # convert = href * Cref * np.pi * DiamD * 60 / 10 * 1.0e9
    # fluxNachMP    = Jsol[NA,   LUM, P]   * convert
    # fluxKchMP     = Jsol[K,    LUM, P]   * convert
    # fluxKchPES    = (Jsol[K,   P, LIS] + Jsol[K,   P, BATH]) * convert
    # fluxClchPES   = (Jsol[CL,  P, LIS] + Jsol[CL,  P, BATH]) * convert
    # fluxBichPES   = (Jsol[HCO3,P, LIS] + Jsol[HCO3,P, BATH]) * convert
    # fluxH2CO3MP   = Jsol[H2CO3,LUM, P]  * convert
    # fluxCO2MP     = Jsol[CO2,  LUM, P]  * convert
    # fluxH2CO3PES  = (Jsol[H2CO3,P, LIS] + Jsol[H2CO3,P, BATH]) * convert
    # fluxCO2PES    = (Jsol[CO2,  P, LIS] + Jsol[CO2,  P, BATH]) * convert
    # fluxHP2mchPES = (Jsol[HPO4, P, LIS] + Jsol[HPO4, P, BATH]) * convert
    # fluxHPmPES    = (Jsol[H2PO4,P, LIS] + Jsol[H2PO4,P, BATH]) * convert

    return Jsol, delmu


def _compute_enac_flux(
    Jsol: np.ndarray,
    C: np.ndarray,
    EP: np.ndarray,
    ph: np.ndarray,
    membrane: Membrane,
    ENaCexp: float
) -> None:
    """
    Add ENaC (epithelial Na⁺ channel) flux to Jsol (in place).

    ENaC is expressed only in DCT2 (ENaCexp > 0).
    Its permeability depends on luminal and cellular Na⁺ and intracellular pH.
    """
    eps     = 1.0e-6
    facNaMP = (30.0 / (30.0 + C[NA, LUM])) * (50.0 / (50.0 + C[NA, P]))
    facphMP = 0.1 + 2.0 / (1.0 + np.exp(-6.0 * (ph[P] - 7.50)))
    hENaC   = ENaCexp * membrane.hNaMP * facNaMP * facphMP

    XI   = zval[NA] * F * EPref / RT * (EP[LUM] - EP[P])
    dint = np.exp(-XI)
    if abs(1.0 - dint) < eps:
        dJENaC = membrane.area[LUM, P] * hENaC * (C[NA, LUM] - C[NA, P])
    else:
        dJENaC = (membrane.area[LUM, P] * hENaC * XI
                  * (C[NA, LUM] - C[NA, P] * dint) / (1.0 - dint))

    Jsol[NA, LUM, P] += dJENaC


def _compute_cotransporter_fluxes(
    Jsol: np.ndarray,
    delmu: np.ndarray,
    C: np.ndarray,
    membrane: Membrane,
    NCCexp: float
) -> None:
    """
    Add cotransporter flux contributions to Jsol (in place).

    Transporters:
        - Basolateral Na²⁺/HPO₄²⁻ cotransporter  (P→LIS, P→BATH)
        - Apical KCl cotransporter  (LUM→P)
        - Basolateral KCl cotransporter  (P→LIS, P→BATH)
        - Apical NCC (Na⁺-Cl⁻ cotransporter)  (LUM→P)
    """
    # Basolateral Na²⁺/HPO₄²⁻ cotransporter: 2 Na⁺ co-transported with 1 HPO₄²⁻
    for L in (LIS, BATH):
        dJNaP = (membrane.area[P, L] * membrane.dLA[NA, HPO4, P, L]
                 * (2 * delmu[NA, P, L] + delmu[HPO4, P, L]))
        Jsol[NA,   P, L] += 2 * dJNaP
        Jsol[HPO4, P, L] += dJNaP
    # sumJES = sum of 2*dJNaP over L in (LIS, BATH) — unused diagnostic
    # fluxNaPatPES = sumJES * convert  # Na flux via Na²⁺/HPO₄²⁻ cotransporter — unused

    # Apical KCl cotransporter
    dJKClMP = (membrane.area[LUM, P] * membrane.dLA[K, CL, LUM, P]
               * (delmu[K, LUM, P] + delmu[CL, LUM, P]))
    Jsol[K,  LUM, P] += dJKClMP
    Jsol[CL, LUM, P] += dJKClMP
    # fluxKClMP = dJKClMP * convert  # apical KCl flux diagnostic — unused

    # Basolateral KCl cotransporter
    for L in (LIS, BATH):
        dJKCl = (membrane.area[P, L] * membrane.dLA[K, CL, P, L]
                 * (delmu[K, P, L] + delmu[CL, P, L]))
        Jsol[K,  P, L] += dJKCl
        Jsol[CL, P, L] += dJKCl
    # sumJES = sum of dJKCl over L in (LIS, BATH) — unused diagnostic
    # fluxKClPES = sumJES * convert  # basolateral KCl flux diagnostic — unused

    # Apical NCC cotransporter (Na⁺-Cl⁻, dominant in DCT1)
    alp   = C[NA, LUM] / dKnncc
    alpp  = C[NA, P]   / dKnncc
    betp  = C[CL, LUM] / dKcncc
    betpp = C[CL, P]   / dKcncc
    gamp  = C[NA, LUM] * C[CL, LUM] / (dKnncc * dKncncc)
    gampp = C[NA, P]   * C[CL, P]   / (dKnncc * dKncncc)
    rhop  = 1 + alp   + betp  + gamp
    rhopp = 1 + alpp  + betpp + gampp
    sigma = rhop * (poppncc + gampp * pnppncc) + rhopp * (popncc + gamp * pnpncc)
    dJNCC = (NCCexp * membrane.area[LUM, P] * membrane.xNCC
             * pnpncc * poppncc * (gamp - gampp) / sigma)
    Jsol[NA, LUM, P] += dJNCC
    Jsol[CL, LUM, P] += dJNCC
    # fluxNCC = dJNCC * convert  # NCC flux diagnostic — unused


def _compute_exchanger_fluxes(
    Jsol: np.ndarray,
    delmu: np.ndarray,
    C: np.ndarray,
    EP: np.ndarray,
    membrane: Membrane,
    CaTexp: float
) -> float:
    """
    Add exchanger flux contributions to Jsol (in place).

    Transporters:
        - Apical NHE3 (Na⁺/H⁺ exchanger isoform 3)
        - Basolateral NHE1 (Na⁺/H⁺ exchanger)
        - Basolateral AE (Cl⁻/HCO₃⁻ exchanger)
        - Basolateral NCX (Na⁺/Ca²⁺ exchanger, scaled by CaTexp)

    Returns:
        Total Ca²⁺ flux via NCX (dJNCXca5 + dJNCXca6) for diagnostic storage.
    """
    from compute_nhe3_fluxes import compute_nhe3_fluxes
    from compute_ncx_fluxes import compute_ncx_fluxes

    # Apical NHE3: Na⁺ in exchange for H⁺ (and NH₄⁺)
    dJNHEsod, dJNHEprot, dJNHEamm = compute_nhe3_fluxes(
        C, membrane.area[LUM, P], membrane.xNHE3
    )
    Jsol[NA,  LUM, P] += dJNHEsod
    Jsol[H,   LUM, P] += dJNHEprot
    Jsol[NH4, LUM, P] += dJNHEamm
    # fluxNHEsod  = dJNHEsod  * convert  # NHE3 Na  flux diagnostic — unused
    # fluxNHEprot = dJNHEprot * convert  # NHE3 H   flux diagnostic — unused
    # fluxNHEamm  = dJNHEamm  * convert  # NHE3 NH₄ flux diagnostic — unused

    # Basolateral NHE1: Na⁺ in exchange for H⁺
    for L in (LIS, BATH):
        dJNaH = (membrane.area[P, L] * membrane.dLA[NA, H, P, L]
                 * (delmu[NA, P, L] - delmu[H, P, L]))
        # Alternative NHE1 kinetic model (Fuster et al, J Gen Physiol 2009) —
        # computed in v0 but Rnhe was never applied to fluxes; kept for reference:
        # affnao = 34.0;   affnai = 102.0
        # affho  = 0.0183e-3; affhi = 0.054e-3
        # fnai = C[NA, P];  hi = C[H, P]
        # fnao = C[NA, L];  ho = C[H, L]
        # Fno  = (fnao/affnao) / (1.0 + fnao/affnao + ho/affho)
        # Fni  = (fnai/affnai) / (1.0 + fnai/affnai + hi/affhi)
        # Fho  = (ho/affho)   / (1.0 + fnao/affnao + ho/affho)
        # Fhi  = (hi/affhi)   / (1.0 + fnai/affnai + hi/affhi)
        # E2mod1 = (Fni + Fhi) / (Fni + Fhi + Fno + Fho)
        # E1mod1 = 1.0 - E2mod1
        # E2mod2 = (Fni**2 + Fhi**2) / (Fni**2 + Fhi**2 + Fno**2 + Fho**2)
        # E1mod2 = 1.0 - E2mod2
        # Fmod1  = (hi**2) / (hi**2 + (0.3e-3)**2)
        # Rnhe   = 1.0e3 * ((1.0 - Fmod1) * (E2mod2*Fno**2 - E1mod2*Fni**2)
        #                   + Fmod1 * (E2mod1*Fno - E1mod1*Fni))  # unused
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

    # Basolateral NCX (Na⁺/Ca²⁺ exchanger): 3 Na⁺ exchanged for 1 Ca²⁺
    var_ncx = np.array([
        C[NA, P],  C[NA, LIS],  C[NA, BATH],
        C[CA, P],  C[CA, LIS],  C[CA, BATH],
        EP[P],     EP[LIS],     EP[BATH],
        membrane.area[P, LIS], membrane.area[P, BATH],
        membrane.xNCX, 0.0, 0.0, 0.0, 0.0
    ])
    dJNCXca5, dJNCXca6 = compute_ncx_fluxes(var_ncx)
    dJNCXca5 *= CaTexp
    dJNCXca6 *= CaTexp
    Jsol[NA, P, LIS]  -= 3.0 * dJNCXca5
    Jsol[NA, P, BATH] -= 3.0 * dJNCXca6
    Jsol[CA, P, LIS]  += dJNCXca5
    Jsol[CA, P, BATH] += dJNCXca6

    # fluxNCXca = (dJNCXca5 + dJNCXca6) * convert  # NCX Ca flux diagnostic — unused
    return dJNCXca5 + dJNCXca6


def _compute_active_transport_fluxes(
    Jsol: np.ndarray,
    C: np.ndarray,
    EP: np.ndarray,
    ph: np.ndarray,
    membrane: Membrane,
    CaTexp: float
) -> tuple:
    """
    Add ATPase and Ca²⁺ channel flux contributions to Jsol (in place).

    Transporters:
        - Na-K-ATPase: 3 Na⁺ out, 2 K⁺ (or NH₄⁺) in per cycle
        - TRPV5: apical Ca²⁺ channel (DCT2 only, pH-gated)
        - PMCA: basolateral Ca²⁺ pump (DCT2 only)

    Returns:
        Jnak:   Total Na⁺ flux via NaKATPase (dJact5 + dJact6)
        Jtrpv5: Ca²⁺ flux via TRPV5
        Jpmca:  Total Ca²⁺ flux via PMCA (dJPMCA5 + dJPMCA6)
    """
    # --- Na-K-ATPase ---
    AffNa = 0.2 * (1.0 + C[K,  P]    / 8.33)
    actNa = C[NA, P] / (C[NA, P] + AffNa)
    AffK   = 0.1 * (1.0 + C[NA, BATH] / 18.5)
    AffNH4 = AffK
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

    # --- TRPV5 (apical Ca²⁺ channel, pH-gated) ---
    k1v5 = 42.7
    k3v5 = 0.1684 * np.exp(0.6035 * ph[P])
    k4v5 = 58.7
    if ph[P] < 7.4:
        k2v5 = 55.9 + (173.3 - 55.9) / (7.0 - 7.4) * (ph[P] - 7.4)
    else:
        k2v5 = 55.9 + (30.4 - 55.9) / (8.4 - 7.4) * (ph[P] - 7.4)

    psv5 = 1.0 / (1.0 + k3v5/k4v5 + k2v5/k1v5)
    # pcv5 = k2v5 / k1v5 * psv5  # closed-state probability — computed in v0 but unused
    pfv5 = k3v5 / k4v5 * psv5
    gfv5 = 59 + 59 / (59 + 29.0) * (91.0 - 58.0) / (7.4 - 5.4) * (ph[LUM] - 7.4)
    gsv5 = 29 + 29 / (59 + 29.0) * (91.0 - 58.0) / (7.4 - 5.4) * (ph[LUM] - 7.4)
    gv5  = (gfv5 * pfv5 + gsv5 * psv5) * 1.0e-12

    ECa  = RT / (2 * F) * np.log(C[CA, P] / C[CA, LUM])
    dfv5 = (EP[LUM] - EP[P]) * EPref - ECa
    finhib_v5 = 1.0 / (1.0 + C[CA, P] / Cinhib_v5)

    dJTRPV5 = (CaTexp * membrane.area[LUM, P] * xTRPV5_dct
               * finhib_v5 * gv5 * dfv5 / (2 * F))
    Jsol[CA, LUM, P] += dJTRPV5
    Jtrpv5 = dJTRPV5
    # fluxTRPV5 = dJTRPV5 * convert  # TRPV5 Ca flux diagnostic — unused

    # --- PMCA (basolateral Ca²⁺ pump) ---
    ratio   = C[CA, P] / (C[CA, P] + dKmPMCA)
    dJPMCA5 = CaTexp * membrane.PMCA * membrane.area[P, LIS]  * ratio
    dJPMCA6 = CaTexp * membrane.PMCA * membrane.area[P, BATH] * ratio
    Jsol[CA, P, LIS]  += dJPMCA5
    Jsol[CA, P, BATH] += dJPMCA6
    Jpmca = dJPMCA5 + dJPMCA6
    # fluxPMCA = Jpmca * convert  # PMCA Ca flux diagnostic — unused

    return Jnak, Jtrpv5, Jpmca


def _store_local_fluxes(
    Jsol: np.ndarray,
    membrane: Membrane,
    Lz: int,
    Jnak: float,
    Jtrpv5: float,
    Jncxca: float,
    Jpmca: float
) -> None:
    """
    Store local flux diagnostics on the membrane object and global arrays.

    Units: pmol/min/mm tubule
    """
    cv = href * Cref * np.pi * DiamD * 60 / 10 * 1.0e9
    # cvw = Pfref * Cref * Vwbar * np.pi * DiamD * 60 / 10 * 1.0e6  # dimensional water flux factor — unused

    # Net flux diagnostics from v0 — computed but unused:
    # for I in range(NS):
    #     apfluxnet  = Jsol[I, LUM, P]   + Jsol[I, LUM, A]   + Jsol[I, LUM, B]   + Jsol[I, LUM, LIS]
    #     basfluxnet = Jsol[I, P,   BATH] + Jsol[I, A,   BATH] + Jsol[I, B,   BATH] + Jsol[I, LIS, BATH]
    #     fluxlatnet = Jsol[I, LUM, LIS]  + Jsol[I, P,   LIS]  + Jsol[I, A,   LIS]  + Jsol[I, B,   LIS] - Jsol[I, LIS, BATH]

    membrane.FNatrans = Jsol[NA, LUM, P]   * cv
    membrane.FNapara  = Jsol[NA, LUM, LIS] * cv
    membrane.FNaK     = Jnak   * cv
    membrane.FHase    = 0.0
    membrane.FKtrans  = Jsol[K, LUM, P]   * cv
    membrane.FKpara   = Jsol[K, LUM, LIS] * cv

    fTRPV5_dct[Lz] = Jtrpv5 * cv
    fNCX_dct[Lz]   = Jncxca * cv
    fPMCA_dct[Lz]  = Jpmca  * cv
