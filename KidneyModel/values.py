"""
Global simulation parameters and boundary conditions.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module defines all simulation-wide constants: grid size, solute/compartment
counts, physical parameters, transporter kinetics, bath boundary conditions,
and named index constants for solutes and compartments.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

# Number of solutes (NS and NS2 are legacy aliases for NSPT; all equal 16)
NSPT = 16
NS   = NSPT
NS2  = NSPT

# Number of compartments
NC = 6

# Number of intervals in space grid
#NZ = 200  # 200
#NZIMC = 200  # 200

# Set to 50 for testing, 200 for production runs
NZ = 50
NZIMC = 50

# Number of unknowns in 2 epithelial compartments + potential in lumen
NUA = 5 + 2 * NS2
NUPT = NUA

# Number of unknowns in all 4 epithelial compartments + potential in lumen
NUC = 9 + 4 * NS2

# Number of unknowns in lumen and 2 epithelial compartments
NDA = 7 + 3 * NS2
NDPT = NDA

# Number of unknowns in lumen and 4 epithelial compartments
NDC = 11 + 5 * NS2
NDIMC = 7 + 3 * NS2

# Number of nephrons in the rat
Nneph = 72000 

# Number of variables for H-K-ATPase
Natp = 14

# Number of variables for NKCC2
Nkcc2 = 15

# Types of fluxes to record (NaK, Hase, Na-trans, Na-para)
NFR = 4

# Physical parameters
PI = 3.14159265359
visc = 6.4e-6

DiamPT = 0.0021250
dimLPT = 0.935000
complPT = 0.1
PMinitPT = 15.00
PbloodPT = 9.0
xS3 = 0.88

DiamD = 0.001275
dimLD = 0.085
SRvol = 0.185
complD = 0.1
PMinit = 10.0

DiamC = 0.001530
dimLC = 0.170
complC = 0.3

DiamCCD = 0.002125
dimLCCD = 0.170
complCCD = 0.3

DiamOMC = 0.002125
dimLOMC = 0.170
complOMC = 0.1

DiamIMC = 0.00238
dimLIMC = 0.4250
complIMC = 0.1

DiamT = 0.00170
dimLT = 0.170
complT = 0.1
DiamMD = 0.00170
dimMD = 0.170
complMD = 0.1

DiamA = 0.00170
dimLA = 0.170
complA = 0.1

DiamSDL = 0.00170
dimLSDL = 0.1190

# Buffer and impermeant properties
zPimp = -1.0

CPimprefPT = 60.0
CAimprefPT = 60.0
CBimprefPT = 60.0
zPimpPT = -1.0
CPbuftotPT = 60.0

CPimprefSDL = 200.0
CPbuftot = 55.0

CPimprefA = 200.0
CAimprefA = 200.0
CBimprefA = 200.0
zPimpA = -1.0
CPbuftotA = 55.0

CPimprefT = 100.0
CAimprefT = 100.0
CBimprefT = 100.0
zPimpT = -1.0
CPbuftotT = 55.0

CPimprefMD = 100.0
zPimpMD = -1.0
CPbuftotMD = 55.0

CPimprefD = 70.0
CAimprefD = 70.0
CBimprefD = 70.0
zPimpD = -1.0
CPbuftotD = 40.0

CPimprefC = 50.0
CAimprefC = 18.0
CBimprefC = 18.0
zPimpC = -1.0
zAimpC = -1.0
zBimpC = -1.0
CPbuftotC = 32.0
CAbuftotC = 40.0
CBbuftotC = 40.0

CPimprefCCD = 50.0
CAimprefCCD = 18.0
CBimprefCCD = 18.0
zPimpCCD = -1.0
zAimpCCD = -1.0
zBimpCCD = -1.0
CPbuftotCCD = 32.0
CAbuftotCCD = 40.0
CBbuftotCCD = 40.0

CPimprefOMC = 50.0
CAimprefOMC = 60.0
CBimprefOMC = 60.0
zPimpOMC = -1.0
zAimpOMC = -1.0
zBimpOMC = -1.0
CPbuftotOMC = 32.0
CAbuftotOMC = 40.0
CBbuftotOMC = 40.0

CPimprefIMC = 50.0
CAimprefIMC = 50.0
CBimprefIMC = 50.0
zPimpIMC = -1.0
zAimpIMC = -1.0
zBimpIMC = -1.0
CPbuftotIMC = 32.0
CAbuftotIMC = 32.0
CBbuftotIMC = 32.0

# Reference concentrations, volume, permeabilities, membrane potentials 
# for non-dimensionalization
# Since Cref = 1 mM, concentrations in mM are already non-dimensional
Cref = 1.0e-3  # Equals 1 mM = 0.001 mmol/cm3
Vref = 1.0e-4  # In cm3/cm2 epith
Pfref = 1.0    # In cm/s
href = 1.0e-5  # In cm/s
EPref = 1.0e-3  # In Volts

# Vwbar = Molar volume of water (cm3/mmole)
# Formula weight of H20 = 2*1.00797 + 15.9994 = 18.0153
# There is 1 mole H20/18.0153 gm of H20.
# Density of water at 37 C = 0.99335 gm/cm3 (CRC Handbook)
# Vwbar = (gm/millimole)/(gm/cm3)=(cm3/millimole)
Vwbar = (18.0153 / 1000.0) / 0.99335

# Other parameters
RT = 2.57
RTosm = 19300.0
F = 96.5
pKHCO3 = 3.57
pKHPO4 = 6.80
pKNH3 = 9.15
pKbuf = 7.5
pKHCO2 = 3.76

# Torque parameters for proximal tubule
torqvm = 0.030
torqL = 2.50e-4
torqd = 1.50e-5

torqR = 0.000900 
TS = 1.4 

# Transport parameters for NCC
popncc = 4295000.0
poppncc = 100000.0
pnpncc = 7692.0
pnppncc = 179.0
dKnncc = 0.293
dKcncc = 112.7
dKncncc = 0.565
dKcnncc = 0.00147

# Transporter parameters
dmuATPH = 1.450
steepA = 0.40
steepB = 0.40
nNDBCEa = 0
nNDBCEb = 1
nAE4a = 0
nAE4b = 1
nstoch = 3

# Transporter parameters for AE1
Pbp = 1247.0
Pbpp = 135.0
Pcp = 562.0
Pcpp = 61.0
dKbp = 198.0
dKbpp = 198.0
dKcp = 50.0
dKcpp = 50.0

# Transporter parameters for PENDRIN
Pclepd = 10000.0
Pcliclepd = 1.239
Pbieclepd = 10.76
Poheclepd = 0.262
dKclpd = 3.01
dKbipd = 5.94
dKohpd = 1.38e-6

# Transporter parameters for NDBCE
pBCE = 10000.0
dKnaiBCE = 20.0
dKnaeBCE = 20.0
dKcliBCE = 20.0
dKcleBCE = 20.0
dKbiiBCE = 15.0
dKbieBCE = 15.0

# Transport parameters for NHE1
dknhe1na = 15.0
dknhe1h = 1.7e-5
dknhe1l = 3.6e-3
dlo = 5952.0

# TRANSPORTER PARAMETERS FOR KCC (KCC4 isoform)
poppkcc = 39280.0
popkcc = 357700.0
pkccp = 10000.0
pkccpp = 1098.0
pmccp = 2000.0
pmccpp = 219.6
bckcc = 21.08
bkkcc = 1.45
bmkcc = 1.45

# TRANSPORTER PARAMETERS FOR NKCC2 (F isoform)
poppnkccF = 39280.0
popnkccF = 357800.0
pnkccpF = 10000.0
pnkccppF = 1098.0
pnmccpF = 2000.0
pnmccppF = 219.6
bn2F = 58.93
bc2F = 13.12
bk2F = 9.149
bm2F = 9.149

# TRANSPORTER PARAMETERS FOR NKCC2 (A isoform)
poppnkccA = 75350.0
popnkccA = 259400.0
pnkccpA = 10000.0
pnkccppA = 2904.0
pnmccpA = 2000.0
pnmccppA = 580.8
bn2A = 118.8
bc2A = 0.08834
bk2A = 18710.0
bm2A = 18710.0

# TRANSPORTER PARAMETERS FOR NKCC2 (B isoform)
poppnkccB = 251700.0
popnkccB = 259600.0
pnkccpB = 10000.0
pnkccppB = 9695.0
pnmccpB = 2000.0
pnmccppB = 1939.0
bn2B = 275.0
bc2B = 0.08157
bk2B = 5577.0
bm2B = 5577.0

# TRANSPORTER PARAMETERS FOR NHE
PaNH = 8000.0
PbNH = 8000.0 * 0.48 / 1.60
PcNH = 8000.0
dKaNH = 30.0
dKbNH = 72.0e-6
dKcNH = 27.0
fMNH = 2.0
dKINH = 1.0e-3

# TRANSPORT PARAMETERS FOR H-ATPase
dmuATPHPT = 1.45  # for proximal tubule
steepATPH = 0.40
dmuHATP = 2.1
steepHATP = 0.40

# TRANSPORTER PARAMETERS FOR NCX
dKmNao = 87.50
dKmNai = 12.29
dKmCao = 1.30
dKmCai = 3.59e-3
gamma_ncx = 0.35
dKsat_ncx = 0.27
dKm_ncx = 0.125e-3

# TRANSPORTER PARAMETERS FOR PMCA
dKmPMCA = 42.6e-6

# TRANSPORTER PARAMETERS FOR TRPV5
Cinhib_v5 = 74.0e-6

# PARAMETERS FOR CASR
EC50 = 1.25

# ---------------------------------------------------------------------72
# ---------------------------------------------------------------------72
#  Solute valence
# ---------------------------------------------------------------------72
# ---------------------------------------------------------------------72

zval = np.array([
    +1.0, +1.0, -1.0, -1.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0,
    +1.0, +1.0, -1.0, 0.0, 0.0, +2.0
    ])

#---------------------------------------------------------------------72
#---------------------------------------------------------------------72
# BOUNDARY CONDITIONS IN PERITUBULAR SOLUTION
#---------------------------------------------------------------------72
#---------------------------------------------------------------------72
# Solute concentration in compartment S (mmole/liter)
# Since reference concentration is 1 mmole/liter = 1d-3 mmol/cm3,
# there is no need to convert to non-dimensional values
    
BathImperm = 2.0  # Impermeant in bath (not present in interspace)
    
LumImperm = 0.0  # No impermeant in lumen under normal conditions

# bdiabetes # true if simulating a diabetic kidney
# furo # true in the presence of furosemide

bdiabetes = False  
ndiabetes = 0 if not bdiabetes else 1

furo = False

# Specify single nephron filtration rate (SNGFR) 
sngfr0 = 24.0e-6 / 60.0  # nl/min converted to cm3/s
sngfr = sngfr0

# Initialize metabolic counters to zero
nephronAct = 0.0  # Quantifies active sodium reabsorption
nephronTNa = 0.0  # Quantifies total sodium reabsorption
nephronQO2 = 0.0  # Quantifies total O2 consumption
nephronTK = 0.0  # Quantifies total potassium reabsorption

# Initialize variables/flags
ncompl = 1  # compliant PT (>0) or not; # Set to zero for non-compliant tubule
ntorq = 1  # accounts for torque effects on transport (>0) or not; # Set to zero in the absence of torque effects
niter = 1  # iteration number for TGF

# Total ammonia at the uppermost cortex
TotAmmCT = 0.203
    
# concentrations in cortex and at cortical medullary boundary
TotSodCM = 132.0 # 144.00d0 in male rats
TotPotCM = 4.0 # 4.9d0 in male rats
TotCloCM = 121.8   # MODIFIED BELOW FOR ELECTRONEUTRALITY
TotBicCM = 25.0
TotHcoCM = 4.41e-3
TotCo2CM = 1.50
TotPhoCM = 2.60   # Old value = 3.90d0 ! In accordance with AMW 2015
TotureaCM = 5.0
TotAmmCM = 1.0
TotHco2CM = 1.0
TotCaCM = 1.25
    
if not bdiabetes:
    TotgluCM = 5.0  # normal kidney
else:
    TotgluCM = 25.0  # diabetic kidney

# concentrations at OM-IM junction
TotSodOI = 260.0 # 284.0d0 in male rats
TotPotOI = 10.0
TotCloOI = 264.94   # MODIFIED BELOW FOR ELECTRONEUTRALITY
TotBicOI = 25.0
TotHcoOI = 4.41e-3
TotCo2OI = 1.50
TotPhoOI = 3.90
TotureaOI = 20.0
TotAmmOI = 3.90
TotHco2OI = TotHco2CM  # Interstitial gradient of HCO2-/H2CO2 species
TotCaOI = 2.50
xIS = 0.60/2.00
TotGluOI = (5.0 + 1.0 / xIS) * (TotgluCM / 5.0)  # TotGluOI is set so that TotGluIS equals 6.0

# concentrations at papillary tip
TotSodPap = 260.0 + 15.0 
TotPotPap = 20.0
TotCloPap = 279.92 + 15.0 # MODIFIED BELOW FOR ELECTRONEUTRALITY
TotBicPap = 25.0
TotHcoPap = 4.41e-3
TotCo2Pap = 1.50
TotPhoPap = 3.90
TotAmmPap = 8.95
TotureaPap = 500.0
TotHco2Pap = TotHco2CM
TotCaPap = 4.0
TotGluPap = 8.50 * (TotgluCM / 5.0)   # Based on Hervy and Thomas, AJP Renal 2003

if furo:
    TotSodOI = TotSodCM
    TotSodPap = TotSodCM
    
    TotPotOI = TotPotCM
    TotPotPap = TotPotCM
    
    TotCloOI = TotCloCM
    TotCloPap = TotCloCM
    
    TotPhoOI = TotPhoCM
    TotPhoPap = TotPhoCM
    
    TotureaOI = TotureaCM
    TotureaPap = TotureaCM
    
    TotAmmOI = TotAmmCM
    TotAmmPap = TotAmmCM
    
    TotCaOI = TotCaCM
    TotCaPap = TotCaCM
    
    TotGluOI = TotgluCM
    TotGluPap = TotgluCM
    
    
# --- Compartment indices (0-based) ---
LUM   = 0   # M
P     = 1   # main PT cell
A     = 2   # cell A
B     = 3   # cell B
LIS   = 4   # E (lateral intercellular space)
BATH  = 5   # S (peritubular / interstitium)

# Solutes (Fortran 1..NSPT) -> Python 0..NSPT-1
NA   = 0
K    = 1
CL   = 2
HCO3 = 3
H2CO3= 4
CO2  = 5
HPO4 = 6
H2PO4= 7
UREA = 8
NH3  = 9
NH4  = 10
H    = 11
HCO2 = 12
H2CO2= 13
GLU  = 14
CA   = 15