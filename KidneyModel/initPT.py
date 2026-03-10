"""
PT (Proximal Tubule) initialization module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module initializes the PT segment parameters. The PT model yields
values of concentrations, volumes, and electrical potentials in the lumen
and epithelial compartments at steady-state equilibrium.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)
"""

import numpy as np

from values import *
from glo import *
from defs import *

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_membrane_areas(pt):
    """Assign compartment volumes, tubule diameter, and membrane surface areas.

    S3 region (J >= xS3*NZ): P-cell apical, lateral, and basal areas x 0.5.
    Lower triangle of area is mirrored from upper triangle.
    """
    SlumPinitpt = 36.0
    SbasPinitpt = 1.0
    SlatPinitpt = 36.0
    SlumEinitpt = 0.001

    for p in pt:
        p.sbasEinit = 0.020

        p.volPinit = 10.0
        p.volEinit = 0.7

        p.volAinit = 10.0   # needed for water flux subroutine (AURELIE)
        p.volBinit = 10.0   # needed for water flux subroutine (AURELIE)

        p.diam = DiamPT

        # Apical membranes (lumen-facing)
        p.area[LUM, P]   = SlumPinitpt
        p.area[LUM, A]   = SlumPinitpt
        p.area[LUM, B]   = SlumPinitpt
        p.area[LUM, LIS] = SlumEinitpt

        # Lateral membranes (cell-LIS)
        p.area[P, LIS] = SlatPinitpt
        p.area[A, LIS] = SlatPinitpt
        p.area[B, LIS] = SlatPinitpt

        # Basal membranes (cell-bath)
        p.area[P, BATH] = SbasPinitpt
        p.area[A, BATH] = SbasPinitpt
        p.area[B, BATH] = SbasPinitpt

    # S3 segment: P-cell apical/lateral/basal areas x 0.5
    S3_start = xS3 * NZ
    for J in range(NZ + 1):
        if J >= S3_start:
            pt[J].area[LUM, P]   *= 0.5
            pt[J].area[P,   LIS]  *= 0.5
            pt[J].area[P,   BATH] *= 0.5

    # Mirror lower triangle
    for k in range(NC):
        for l in range(k + 1, NC):
            for p in pt:
                p.area[l, k] = p.area[k, l]


def _set_water_permeabilities(pt):
    """Assign osmotic water permeabilities dLPV at each membrane interface.

    Pf values are pre-multiplied by membrane area (cm2/cm2 epith); PS = PE.
    A- and B-cell water permeabilities are zero. Non-dimensionalized by Pfref.
    """
    PfMP = 0.64 * 0.40 / 36.0
    PfME = 0.22 / 0.001
    PfPE = 0.64 * 0.40 / 36.0   # PS = PE
    PfES = 6.60 / 0.020

    Pf = np.zeros((NC, NC))
    Pf[LUM, P]    = PfMP
    Pf[LUM, LIS]  = PfME
    Pf[P,   LIS]  = PfPE
    Pf[P,   BATH] = PfPE    # PS = PE
    Pf[LIS, BATH] = PfES

    for p in pt:
        p.dLPV[:, :] = Pf / Pfref


def _set_reflection_coefficients(pt):
    """Set reflection coefficients sigma(solute, K, L).

    Default: 1 everywhere. Basement membrane (LIS-BATH): 0.
    Tight junction (LUM-LIS): solute-specific (symmetrized).
    """
    for p in pt:
        p.sig[:, :, :] = 1.0

    # Basement membrane: fully permeable to all solutes
    for p in pt:
        p.sig[:, LIS,  BATH] = 0.0
        p.sig[:, BATH, LIS]  = 0.0

    # Tight junction: solute-specific reflection
    sig_tj = np.array([
        0.750,   # NA    -- Na+
        0.600,   # K     -- K+
        0.300,   # CL    -- Cl-
        0.900,   # HCO3  -- HCO3-
        0.900,   # H2CO3
        0.900,   # CO2
        0.900,   # HPO4  -- HPO4(2-)
        0.900,   # H2PO4
        0.700,   # UREA
        0.300,   # NH3
        0.600,   # NH4
        0.200,   # H     -- H+
        0.300,   # HCO2
        0.700,   # H2CO2
        1.000,   # GLU   -- glucose
        0.890,   # CA    -- Ca2+  (see Ca PT model)
    ], dtype=float)
    assert sig_tj.size == NSPT, f"Expected {NSPT} solutes, got {sig_tj.size}"

    for p in pt:
        p.sig[:, LUM, LIS] = sig_tj
        p.sig[:, LIS, LUM] = sig_tj


def _set_solute_permeabilities(pt):
    """Set dimensional solute permeabilities h(solute, K, L), then non-dimensionalize.

    Values given in 10^-5 cm/s (scaled by 1e-5/href at the end).
    Only upper triangle (K < L) is set; lower triangle stays zero.
    PS interface (P-BATH) equals PE (P-LIS).
    CA at LIS-BATH derived from NA value (ratio 7.93/13.3).
    """
    # Lumen-cell interface (LUM-P, apical membrane)
    _h_lum_p = np.array([
        0.0,          # NA    -- passive entry removed; carried by NHE3/SGLT
        0.90/36.0,    # K
        0.0,          # CL
        0.036/36.0,   # HCO3
        2.34e3/36.0,  # H2CO3
        4.32e4/36.0,  # CO2   -- 1.20e3 in AMW 2015 = 7500
        0.0,          # HPO4  -- apical entry via NaPiII only
        0.0,          # H2PO4
        3.78/36.0,    # UREA
        3.06e3/36.0,  # NH3
        0.774/36.0,   # NH4
        3.06e4/36.0,  # H
        0.0,          # HCO2
        1.80e5/36.0,  # H2CO2
        0.0,          # GLU   -- via SGLT/GLUT transporters
        0.005,        # CA
    ], dtype=float)

    # Cell-LIS interface (P-LIS, lateral membrane; also copied to P-BATH)
    _h_p_lis = np.array([
        0.0,      # NA    -- difference with AMW 2015
        0.20,     # K
        0.01,     # CL    -- difference with AMW 2015
        0.0,      # HCO3
        65.0,     # H2CO3
        1.20e3,   # CO2   -- difference with AMW 2015 = 7500
        0.00225,  # HPO4
        0.0330,   # H2PO4
        0.100,    # UREA
        100.0,    # NH3
        0.060,    # NH4
        850.0,    # H
        0.0190,   # HCO2
        6.0e3,    # H2CO2
        0.0,      # GLU   -- replaced with GLUT transporters
        0.0,      # CA
    ], dtype=float)

    # Tight junction (LUM-LIS)
    ClTJperm = 1.0 / 0.001   # area normalization factor (1/0.001)
    _h_lum_lis = np.array([
        0.38 * 26.0,  # NA
        0.38 * 29.0,  # K
        20.0,         # CL
        8.0,          # HCO3
        8.0,          # H2CO3
        8.0,          # CO2
        4.0,          # HPO4
        4.0,          # H2PO4
        8.0,          # UREA
        50.0,         # NH3
        50.0,         # NH4
        600.0,        # H
        14.0,         # HCO2
        28.0,         # H2CO2
        0.31,         # GLU   -- Garvin, AJP Renal 1990
        20.0,         # CA
    ], dtype=float) * ClTJperm

    # LIS-bath interface (basement membrane)
    areafactor = 1.0 / 0.020
    _h_lis_bath = np.array([
        100.0,   # NA
        140.0,   # K
        120.0,   # CL
        100.0,   # HCO3
        100.0,   # H2CO3
        100.0,   # CO2
        80.0,    # HPO4
        80.0,    # H2PO4
        160.0,   # UREA
        400.0,   # NH3
        400.0,   # NH4
        6.0e4,   # H
        100.0,   # HCO2
        180.0,   # H2CO2
        60.0,    # GLU
        0.0,     # CA   -- computed below from NA value
    ], dtype=float) * areafactor
    _h_lis_bath[CA] = _h_lis_bath[NA] * (7.93 / 13.3)

    for p in pt:
        p.h[:, :, :] = 0.0
        p.h[:, LUM, P]    = _h_lum_p          # apical
        p.h[:, P,   LIS]  = _h_p_lis          # lateral
        p.h[:, P,   BATH] = _h_p_lis          # PS = PE
        p.h[:, LUM, LIS]  = _h_lum_lis        # tight junction
        p.h[:, LIS, BATH] = _h_lis_bath       # basement membrane
        # A- and B-cell interfaces remain zero (initialized above)

    # Non-dimensionalize: scale full array (lower triangle is zero)
    scale_h = 1.0e-5 / href
    for p in pt:
        p.h *= scale_h


def _set_net_coefficients(pt):
    """Set NET coupling coefficients dLA and kinetic transporter parameters.

    Returns (xNaPiIIaPT, xNaPiIIcPT, xPit2PT): non-dimensionalized rate
    constants for NaPi-IIa, NaPi-IIc, and Pit2 used downstream in qflux2PT.
    """
    S3_start = xS3 * NZ
    scale    = 1.0 / (href * Cref)

    for p in pt:
        p.dLA.fill(0.0)

    # SGLT2 (S1-S2) / SGLT1 (S3): apical Na-glucose co-transporter
    if not bdiabetes:
        CTsglt1 = 0.30e-5 / 36.0                             # SGLT1 (S3 expression)
        for p in pt:
            p.dLA[NA, GLU, LUM, P] = 1.16 * 1.300e-9         # SGLT2; AMW 2007 = 270/36 = 7.50
    else:
        CTsglt1 = 0.30e-5 / 36.0 * (1.0 - 0.33)             # SGLT1 reduced 33% in diabetes
        for p in pt:
            p.dLA[NA, GLU, LUM, P] = 1.16 * 1.300e-9 * 1.38  # SGLT2 up 38% in diabetes

    CTglut1   = 1.2 * 0.2500e-5      # basolateral GLUT1
    CTglut2   = 1.3 * 0.1250e-5      # basolateral GLUT2
    CTnapiIIa = 0.75 * 0.30e-9       # apical NaPi-IIa
    CTnapiIIb = 0.75 * 0.0           # apical NaPi-IIb -- computed but not returned
    CTnapiIIc = 0.75 * 0.250e-9      # apical NaPi-IIc
    CTpit2    = 0.10e-9               # apical Pit2

    # Apical NaPi-II (Na-H2PO4 co-transporter)
    for p in pt:
        p.dLA[NA, H2PO4, LUM, P] = 1.25e-9

    # Apical Cl/HCO3 exchanger -- DIFFERENCE WITH AMW MODEL
    for p in pt:
        p.dLA[CL, HCO3, LUM, P] = 2.0e-9 * 0.0

    # Apical Cl/HCO2 exchanger -- DIFFERENCE WITH AMW MODEL
    for p in pt:
        p.dLA[CL, HCO2, LUM, P] = 5.0e-9 * 2.0

    # Basolateral K-Cl co-transporter
    for p in pt:
        p.dLA[K, CL, P, LIS]  = 5.0e-9
        p.dLA[K, CL, P, BATH] = p.dLA[K, CL, P, LIS]

    # Basolateral Na/3HCO3 exchanger
    for p in pt:
        p.dLA[NA, HCO3, P, LIS]  = 5.0e-9
        p.dLA[NA, HCO3, P, BATH] = p.dLA[NA, HCO3, P, LIS]

    # Basolateral NDCBE (Na-2HCO3/Cl co-transporter)
    # Note: three solutes transported, only two solute indices used
    for p in pt:
        p.dLA[NA, CL, P, LIS]  = 35.0e-9
        p.dLA[NA, CL, P, BATH] = p.dLA[NA, CL, P, LIS]

    # Apical H+/HCO2- co-transporter -- REMOVED (addition to AMW model)
    for p in pt:
        p.dLA[H, HCO2, LUM, P] = 0.0e-9

    # SGLT/GLUT expression: S1-S2 uses SGLT2/GLUT2; S3 uses SGLT1/GLUT1
    for J in range(NZ + 1):
        if J < S3_start:
            pt[J].xSGLT2 = 1.0;  pt[J].xSGLT1 = 0.0
            pt[J].xGLUT2 = 1.0;  pt[J].xGLUT1 = 0.0
        else:
            pt[J].xSGLT2 = 0.0;  pt[J].xSGLT1 = 1.0
            pt[J].xGLUT2 = 0.0;  pt[J].xGLUT1 = 1.0

    # Torque scaling factor: 1.0 in S1-S2, 0.5 in S3
    for J in range(NZ + 1):
        pt[J].scaleT = 1.0 if J < S3_start else 0.5

    # Non-dimensionalize dLA
    for p in pt:
        p.dLA *= scale

    # Kinetic transporter parameters (non-dimensionalized)
    xNHE3     = 0.78 * 1000.0e-9 / 36.0   # apical NHE3
    ATPNaKPES = 0.78 * 300.0e-9            # basolateral Na,K-ATPase
    ATPHMP    = 50.0e-9                    # apical H-ATPase
    PMCA      = 0.50e-9                    # basolateral PMCA
    QNH4      = 0.250e-6                   # rate of ammoniagenesis

    for p in pt:
        p.xNHE3           = xNHE3     * scale
        p.ATPNaK[P, LIS]  = ATPNaKPES * scale
        p.ATPNaK[P, BATH] = ATPNaKPES * scale
        p.ATPH[LUM, P]    = ATPHMP    * scale
        p.qnh4            = QNH4      * scale
        p.CTsglt1         = CTsglt1   * scale
        p.CTglut1         = CTglut1   * scale
        p.CTglut2         = CTglut2   * scale
        p.PMCA            = PMCA      * scale

    xNaPiIIaPT = CTnapiIIa * scale
    xNaPiIIbPT = CTnapiIIb * scale   # computed but not returned
    xNaPiIIcPT = CTnapiIIc * scale
    xPit2PT    = CTpit2    * scale

    return xNaPiIIaPT, xNaPiIIcPT, xPit2PT


def _set_carbonate_kinetics(pt):
    """Assign CO2-HCO3 reaction rate constants (s^-1) for each node.

    S1-S2 (J < xS3*NZ): full catalysed rates.
    S3   (J >= xS3*NZ): rates divided by 100.
    """
    dkhcat   = 1.450e3
    dkdcat   = 496.0e3
    S3_start = xS3 * NZ

    for J in range(NZ + 1):
        if J < S3_start:
            pt[J].dkd[:] = dkdcat
            pt[J].dkh[:] = dkhcat
        else:
            pt[J].dkd[:] = dkdcat / 100.0
            pt[J].dkh[:] = dkhcat / 100.0


def _set_boundary_conditions(pt):
    """Set bath potential and peritubular interstitial concentration gradients.

    Calls set_intconc for the cortex/S1-S2 region (flag=1) and the
    outer-medulla/S3 region (flag=2) with pre-built position arrays.
    """
    for p in pt:
        p.ep[BATH] = 0.0   # qnewton2PT assumes bath potential is zero

    ind    = int(xS3 * NZ) + 1
    indrev = (NZ + 1) - ind

    from set_intconc import set_intconc

    # Region 1: cortex / S1-S2  (indices 0 ... ind-1)
    pos1 = np.arange(ind, dtype=float) / (xS3 * NZ)
    set_intconc(pt[0:ind], ind - 1, 1, pos1)

    # Region 2: outer medulla / S3  (indices ind ... NZ)
    j_arr = np.arange(indrev, dtype=float)
    pos2  = xIS * (ind + j_arr - xS3 * NZ) / ((1.0 - xS3) * NZ)
    set_intconc(pt[ind:NZ + 1], indrev - 1, 2, pos2)


def _set_torque_parameters(pt):
    """Compute baseline luminal torque TM0 (Eq. 37, Edwards 2007 PT model).

    Female rat radius: R_TMO = 0.002250/2. Assigns TM0 to all nodes.
    """
    R_TMO   = 0.002250 / 2.0   # female case (earlier models used 0.0020/2)
    flowref = 24.0e-6 / 60.0
    factor1 = 8.0 * visc * flowref * torqL / (R_TMO ** 2)
    factor2 = 1.0 + (torqL + torqd) / R_TMO + 0.50 * (torqL / R_TMO) ** 2
    TM0 = factor1 * factor2

    for p in pt:
        p.TM0 = TM0

    # Note: Radref = torqR*(1 + torqvm*(PMinitPT - PbloodPT))
    #        125.0 = 113.74*(1 + 0.03*(12.3 - 9.00))


def _set_metabolic_parameters(pt):
    """Set TNa-QO2 ratio TQ (15 in normal kidney, 12 in diabetes)."""
    for p in pt:
        p.TQ = 15.0 if not bdiabetes else 12.0


def _set_initial_conditions(pt):
    """Set lumen entrance conditions, initial guesses, and A/B compartment copies.

    Lumen concentrations and pH are initialized to bath values (set by
    _set_boundary_conditions). If nrestart == 1, reads LUM/P/LIS values
    from PTresults; otherwise uses hard-coded physiological guesses.
    A- and B-cell states are always copied from P-cell.
    """
    pt[0].vol[LUM] = sngfr / Vref   # inlet lumen flow (non-dim)
    pt[0].pres     = PMinitPT

    for p in pt:
        p.volLuminit = pt[0].vol[LUM]

    # Lumen initialized to bath (set_intconc must already have run)
    pt[0].conc[:, LUM] = pt[0].conc[:, BATH]
    pt[0].ph[LUM]      = pt[0].ph[BATH]
    pt[0].ep[LUM]      = -0.16e-3 / EPref   # initial guess for lumen potential

    nrestart = 1   # read initial values from restart file when 1

    if nrestart == 1:
        with open('PTresults', 'r') as f:
            # File format: NSPT lines of (conc_LUM, conc_P, conc_LIS),
            # then lines for (pH_LUM pH_P pH_LIS), (vol_LUM vol_P vol_LIS),
            # (ep_LUM ep_P ep_LIS).
            for i in range(NSPT):
                pt[0].conc[i, LUM], pt[0].conc[i, P], pt[0].conc[i, LIS] = \
                    map(float, f.readline().split())
            pt[0].ph[LUM],  pt[0].ph[P],  pt[0].ph[LIS]  = map(float, f.readline().split())
            pt[0].vol[LUM], pt[0].vol[P], pt[0].vol[LIS] = map(float, f.readline().split())
            pt[0].ep[LUM],  pt[0].ep[P],  pt[0].ep[LIS]  = map(float, f.readline().split())

    else:
        pt[0].ph[P]   = 7.329
        pt[0].ph[LIS] = 7.339

        pt[0].conc[NA,    P]   = 19.6;      pt[0].conc[NA,    LIS] = 140.3
        pt[0].conc[K,     P]   = 138.1;     pt[0].conc[K,     LIS] = 4.66
        pt[0].conc[CL,    P]   = 16.3;      pt[0].conc[CL,    LIS] = 112.0
        pt[0].conc[HCO3,  P]   = 25.0;      pt[0].conc[HCO3,  LIS] = 25.6
        pt[0].conc[H2CO3, P]   = 4.36e-3;   pt[0].conc[H2CO3, LIS] = 4.36e-3
        pt[0].conc[CO2,   P]   = 1.49;      pt[0].conc[CO2,   LIS] = 1.49
        pt[0].conc[HPO4,  P]   = 8.50;      pt[0].conc[HPO4,  LIS] = 2.98
        pt[0].conc[H2PO4, P]   = 2.52;      pt[0].conc[H2PO4, LIS] = 0.86
        pt[0].conc[UREA,  P]   = 4.96;      pt[0].conc[UREA,  LIS] = 4.91
        pt[0].conc[NH3,   P]   = 3.48e-3;   pt[0].conc[NH3,   LIS] = 2.70e-3
        pt[0].conc[NH4,   P]   = 0.23;      pt[0].conc[NH4,   LIS] = 0.18
        pt[0].conc[H,     P]   = np.exp(-np.log(10.0) * pt[0].ph[P])   * 1e3
        pt[0].conc[H,     LIS] = np.exp(-np.log(10.0) * pt[0].ph[LIS]) * 1e3
        pt[0].conc[HCO2,  P]   = 0.52;      pt[0].conc[HCO2,  LIS] = 0.77
        pt[0].conc[H2CO2, P]   = 0.91e-4;   pt[0].conc[H2CO2, LIS] = 2.04e-4
        pt[0].conc[GLU,   P]   = 15.1;      pt[0].conc[GLU,   LIS] = 7.79

        pt[0].vol[P]   = 8.72
        pt[0].vol[LIS] = -1.967

        pt[0].ep[P]   = -55.6e-3 / EPref
        pt[0].ep[LIS] = -0.01e-3 / EPref

    # A- and B-cell: copy from P-cell
    pt[0].conc[:, A] = pt[0].conc[:, P]
    pt[0].conc[:, B] = pt[0].conc[:, P]
    pt[0].ph[A]      = pt[0].ph[P]
    pt[0].ph[B]      = pt[0].ph[P]
    pt[0].vol[A]     = 0.0
    pt[0].vol[B]     = 0.0
    pt[0].ep[A]      = pt[0].ep[P]
    pt[0].ep[B]      = pt[0].ep[P]


def _check_electroneutrality(pt):
    """Compute charge balance at lumen inlet, bath inlet, and bath outlet."""
    elecM     = np.dot(zval[:NSPT], pt[0].conc[:NSPT, LUM])
    elecS     = np.dot(zval[:NSPT], pt[0].conc[:NSPT, BATH])
    elecS_out = np.dot(zval[:NSPT], pt[NZ].conc[:NSPT, BATH])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def initPT(pt):
    """Initialize all PT parameters and return transporter rate constants.

    Returns:
    PTinitVol  : float -- initial lumen volume/flow (non-dim), used in qflux2PT
    xNaPiIIaPT : float -- NaPi-IIa rate constant (non-dim)
    xNaPiIIcPT : float -- NaPi-IIc rate constant (non-dim)
    xPit2PT    : float -- Pit2 rate constant (non-dim)
    """
    _set_membrane_areas(pt)
    _set_water_permeabilities(pt)
    _set_reflection_coefficients(pt)
    _set_solute_permeabilities(pt)
    xNaPiIIaPT, xNaPiIIcPT, xPit2PT = _set_net_coefficients(pt)
    _set_carbonate_kinetics(pt)
    _set_boundary_conditions(pt)
    _set_torque_parameters(pt)
    _set_metabolic_parameters(pt)
    _set_initial_conditions(pt)
    _check_electroneutrality(pt)

    PTinitVol = pt[0].vol[LUM]

    return PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT
