"""
Main simulation driver full renal nephron model.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

Runs the steady-state renal nephron model for young female rats.
Segments computed sequentially: PT → SDL → mTAL → cTAL → DCT → CNT → CCD → OMCD → IMCD.
Each segment writes its outlet concentrations, pH, EP, and volume to a Data file.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
Morphological parameters: rat kidney (AMW model, AJP Renal 2007)

Solute indices (0-based):
  NA=0, K=1, CL=2, HCO3=3, H2CO3=4, CO2=5,
  HPO4=6, H2PO4=7, UREA=8, NH3=9, NH4=10, H=11,
  HCO2=12, H2CO2=13, GLU=14, CA=15
"""

import time
import numpy as np

from values import *
from glo    import *
from defs   import *

from initPT      import initPT
from qnewton1PT  import qnewton1PT
from qnewton2PT  import qnewton2PT
from initSDL     import initSDL
from qnewton2SDL import qnewton2SDL
from initA       import initA
from initT       import initT
from initD       import initD
from initC       import initC
from initCCD     import initCCD
from initOMC     import initOMC
from initIMC     import initIMC
from qnewton2b   import qnewton2b
from qnewton2icb import qnewton2icb

# --- Start timer ---

tic = time.time()

# --- Allocate segment node arrays ---

# membrane_inst = Membrane(NSPT, NC, NS)  # unused — kept for reference

pt   = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
sdl  = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
mtal = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
ctal = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
dct  = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
cnt  = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
ccd  = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
omcd = [Membrane(NSPT, NC, NS) for _ in range(NZ + 1)]
imcd = [Membrane(NSPT, NC, NS) for _ in range(NZIMC + 1)]

# --- Variables used in output ---

# inlet      = np.zeros(NS)  # unused
outlet     = np.zeros(NS)
# fluxs      = np.zeros(NS)  # unused
# deliv      = np.zeros(NS)  # unused
# fracdel    = np.zeros(NS)  # unused
# totalAct   = 0.0           # unused
# totalTNa   = 0.0           # unused
# o2consum   = 0.0           # unused
# nephronAct = 0.0           # unused
# nephronTNa = 0.0           # unused
# nephronQO2 = 0.0           # unused

# Scaling factor: convert volume flows from cm³/s to ml/min for Nneph nephrons
cw = Vref * 60.0 * Nneph
# cw = Vref * 60.0 * 1e6  # If basis is one nephron

# =====================================================================
# PT
# =====================================================================

print("****************************************************")
print("                    PT RESULTS")
print("****************************************************")
print()

PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT = initPT(pt)

# Inlet node: solve for compartment values given luminal boundary condition
qnewton1PT(pt, CPimprefPT, CPbuftotPT, 0, ncompl, ntorq, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

# Interior nodes: solve for lumen + compartments
for jz in range(1, NZ + 1):
    qnewton2PT(pt, jz - 1, 0, pt[0].vol, CPimprefPT, CPbuftotPT, NDPT, ncompl, ntorq, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('PToutlet', 'w') as f:
    for i in range(4):
        f.write(f"{pt[NZ].conc[i, LUM]:12.5f} {pt[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{pt[NZ].conc[i, LUM]:12.5e} {pt[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{pt[NZ].ph[LUM]:12.5f} {pt[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{pt[NZ].ep[LUM]:12.5f} {pt[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{pt[NZ].vol[LUM]:12.5e} {pt[NZ].vol[BATH]:12.5e}\n")
    f.write(f"{pt[NZ].vol[LUM]:12.5e} {pt[NZ].pres:12.5e}\n")

with open('PToutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{pt[NZ].conc[i, P]:12.5f} {pt[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{pt[NZ].conc[i, P]:12.5e} {pt[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{pt[NZ].ph[P]:12.5f} {pt[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{pt[NZ].ep[P]:12.5f} {pt[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{pt[NZ].vol[P]:12.5e} {pt[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# SDL
# =====================================================================

print("****************************************************")
print("                    SDL RESULTS")
print("****************************************************")
print()

initSDL(sdl)

# Replace inlet luminal concentrations with PT outflow values
sdl[0].conc[:, LUM] = pt[NZ].conc[:, LUM]
sdl[0].ph[LUM]      = pt[NZ].ph[LUM]
sdl[0].ep[LUM]      = pt[NZ].ep[LUM]
sdl[0].vol[LUM]     = pt[NZ].vol[LUM]
sdl[0].pres         = pt[NZ].pres

for jz in range(1, NZ + 1):
    qnewton2SDL(sdl, jz - 1, 2, sdl[0].vol, NDA, PTinitVol)

with open('SDLoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{sdl[NZ].conc[i, LUM]:12.5f} {sdl[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{sdl[NZ].conc[i, LUM]:12.5e} {sdl[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{sdl[NZ].ph[LUM]:12.5f} {sdl[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{sdl[NZ].ep[LUM]:12.5f} {sdl[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{sdl[NZ].vol[LUM]:12.5e} {sdl[NZ].vol[BATH]:12.5e}\n")

# =====================================================================
# mTAL
# =====================================================================

print("****************************************************")
print("                    mTAL RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to SDL outflow values in initA.
# Initial guesses for cellular and LIS concentrations are in mTALresults.
initA(mtal)

with open('mTALresults', 'r') as f:
    for i in range(NS):
        mtal[0].conc[i, P], mtal[0].conc[i, LIS] = map(float, f.readline().split())
    mtal[0].ph[P],  mtal[0].ph[LIS]  = map(float, f.readline().split())
    mtal[0].vol[P], mtal[0].vol[LIS] = map(float, f.readline().split())
    mtal[0].ep[P],  mtal[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2b(mtal, mtal[0].vol, jz - 1, 3, NDA, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('mTALoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{mtal[NZ].conc[i, LUM]:12.5f} {mtal[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{mtal[NZ].conc[i, LUM]:12.5e} {mtal[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{mtal[NZ].ph[LUM]:12.5f} {mtal[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{mtal[NZ].ep[LUM]:12.5f} {mtal[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{mtal[NZ].vol[LUM]:12.5e} {mtal[NZ].vol[BATH]:12.5e}\n")

with open('mTALoutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{mtal[NZ].conc[i, P]:12.5f} {mtal[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{mtal[NZ].conc[i, P]:12.5e} {mtal[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{mtal[NZ].ph[P]:12.5f} {mtal[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{mtal[NZ].ep[P]:12.5f} {mtal[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{mtal[NZ].vol[P]:12.5e} {mtal[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# cTAL
# =====================================================================

print("****************************************************")
print("                    cTAL RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to mTAL outflow values in initT.
# Initial guesses for cellular and LIS concentrations are in cTALresults.
initT(ctal)

with open('cTALresults', 'r') as f:
    for i in range(NS):
        ctal[0].conc[i, P], ctal[0].conc[i, LIS] = map(float, f.readline().split())
    ctal[0].ph[P],  ctal[0].ph[LIS]  = map(float, f.readline().split())
    ctal[0].vol[P], ctal[0].vol[LIS] = map(float, f.readline().split())
    ctal[0].ep[P],  ctal[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2b(ctal, ctal[0].vol, jz - 1, 4, NDA, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('cTALoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{ctal[NZ].conc[i, LUM]:12.5f} {ctal[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{ctal[NZ].conc[i, LUM]:12.5e} {ctal[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{ctal[NZ].ph[LUM]:12.5f} {ctal[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{ctal[NZ].ep[LUM]:12.5f} {ctal[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{ctal[NZ].vol[LUM]:12.5e} {ctal[NZ].vol[BATH]:12.5e}\n")

with open('cTALoutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{ctal[NZ].conc[i, P]:12.5f} {ctal[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{ctal[NZ].conc[i, P]:12.5e} {ctal[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{ctal[NZ].ph[P]:12.5f} {ctal[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{ctal[NZ].ep[P]:12.5f} {ctal[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{ctal[NZ].vol[P]:12.5e} {ctal[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# DCT
# =====================================================================

print("****************************************************")
print("                    DCT RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to cTAL outflow values in initD.
# Initial guesses for cellular and LIS concentrations are in DCTresults.
initD(dct)

with open('DCTresults', 'r') as f:
    for i in range(NS):
        dct[0].conc[i, P], dct[0].conc[i, LIS] = map(float, f.readline().split())
    dct[0].ph[P],  dct[0].ph[LIS]  = map(float, f.readline().split())
    dct[0].vol[P], dct[0].vol[LIS] = map(float, f.readline().split())
    dct[0].ep[P],  dct[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2b(dct, dct[0].vol, jz - 1, 5, NDA, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('DCToutlet', 'w') as f:
    for i in range(4):
        f.write(f"{dct[NZ].conc[i, LUM]:12.5f} {dct[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{dct[NZ].conc[i, LUM]:12.5e} {dct[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{dct[NZ].ph[LUM]:12.5f} {dct[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{dct[NZ].ep[LUM]:12.5f} {dct[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{dct[NZ].vol[LUM]:12.5e} {dct[NZ].vol[BATH]:12.5e}\n")

with open('DCToutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{dct[NZ].conc[i, P]:12.5f} {dct[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{dct[NZ].conc[i, P]:12.5e} {dct[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{dct[NZ].ph[P]:12.5f} {dct[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{dct[NZ].ep[P]:12.5f} {dct[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{dct[NZ].vol[P]:12.5e} {dct[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# CNT
# =====================================================================

print("****************************************************")
print("                    CNT RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to DCT outflow values in initC.
# Initial guesses for cellular and LIS concentrations are in CNTresults.
initC(cnt)

with open('CNTresults', 'r') as f:
    for i in range(NS):
        cnt[0].conc[i, P], cnt[0].conc[i, A], cnt[0].conc[i, B], cnt[0].conc[i, LIS] = map(float, f.readline().split())
    cnt[0].ph[P],  cnt[0].ph[A],  cnt[0].ph[B],  cnt[0].ph[LIS]  = map(float, f.readline().split())
    cnt[0].vol[P], cnt[0].vol[A], cnt[0].vol[B], cnt[0].vol[LIS] = map(float, f.readline().split())
    cnt[0].ep[P],  cnt[0].ep[A],  cnt[0].ep[B],  cnt[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2icb(cnt, jz - 1, 6, cnt[0].vol, cnt[jz].volEinit, cnt[jz].volPinit, cnt[jz].volAinit, cnt[jz].volBinit, NDC, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('CNToutlet', 'w') as f:
    for i in range(4):
        f.write(f"{cnt[NZ].conc[i, LUM]:12.5f} {cnt[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{cnt[NZ].conc[i, LUM]:12.5e} {cnt[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{cnt[NZ].ph[LUM]:12.5f} {cnt[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{cnt[NZ].ep[LUM]:12.5f} {cnt[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{cnt[NZ].vol[LUM]:12.5e} {cnt[NZ].vol[BATH]:12.5e}\n")

with open('CNToutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{cnt[NZ].conc[i, P]:12.5f} {cnt[NZ].conc[i, A]:12.5f} {cnt[NZ].conc[i, B]:12.5f} {cnt[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{cnt[NZ].conc[i, P]:12.5e} {cnt[NZ].conc[i, A]:12.5e} {cnt[NZ].conc[i, B]:12.5e} {cnt[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{cnt[NZ].ph[P]:12.5f} {cnt[NZ].ph[A]:12.5f} {cnt[NZ].ph[B]:12.5f} {cnt[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{cnt[NZ].ep[P]:12.5f} {cnt[NZ].ep[A]:12.5f} {cnt[NZ].ep[B]:12.5f} {cnt[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{cnt[NZ].vol[P]:12.5e} {cnt[NZ].vol[A]:12.5e} {cnt[NZ].vol[B]:12.5e} {cnt[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# CCD
# =====================================================================

print("****************************************************")
print("                    CCD RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to CNT outflow values in initCCD.
# Initial guesses for cellular and LIS concentrations are in CCDresults.
initCCD(ccd)

with open('CCDresults', 'r') as f:
    for i in range(NS):
        ccd[0].conc[i, P], ccd[0].conc[i, A], ccd[0].conc[i, B], ccd[0].conc[i, LIS] = map(float, f.readline().split())
    ccd[0].ph[P],  ccd[0].ph[A],  ccd[0].ph[B],  ccd[0].ph[LIS]  = map(float, f.readline().split())
    ccd[0].vol[P], ccd[0].vol[A], ccd[0].vol[B], ccd[0].vol[LIS] = map(float, f.readline().split())
    ccd[0].ep[P],  ccd[0].ep[A],  ccd[0].ep[B],  ccd[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2icb(ccd, jz - 1, 7, ccd[0].vol, ccd[jz].volEinit, ccd[jz].volPinit, ccd[jz].volAinit, ccd[jz].volBinit, NDC, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('CCDoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{ccd[NZ].conc[i, LUM]:12.5f} {ccd[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{ccd[NZ].conc[i, LUM]:12.5e} {ccd[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{ccd[NZ].ph[LUM]:12.5f} {ccd[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{ccd[NZ].ep[LUM]:12.5f} {ccd[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{ccd[NZ].vol[LUM]:12.5e} {ccd[NZ].vol[BATH]:12.5e}\n")

with open('CCDoutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{ccd[NZ].conc[i, P]:12.5f} {ccd[NZ].conc[i, A]:12.5f} {ccd[NZ].conc[i, B]:12.5f} {ccd[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{ccd[NZ].conc[i, P]:12.5e} {ccd[NZ].conc[i, A]:12.5e} {ccd[NZ].conc[i, B]:12.5e} {ccd[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{ccd[NZ].ph[P]:12.5f} {ccd[NZ].ph[A]:12.5f} {ccd[NZ].ph[B]:12.5f} {ccd[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{ccd[NZ].ep[P]:12.5f} {ccd[NZ].ep[A]:12.5f} {ccd[NZ].ep[B]:12.5f} {ccd[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{ccd[NZ].vol[P]:12.5e} {ccd[NZ].vol[A]:12.5e} {ccd[NZ].vol[B]:12.5e} {ccd[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# OMCD
# =====================================================================

print("****************************************************")
print("                    OMCD RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to CCD outflow values in initOMC.
# Initial guesses for cellular and LIS concentrations are in OMCresults.
initOMC(omcd)

with open('OMCresults', 'r') as f:
    for i in range(NS):
        omcd[0].conc[i, P], omcd[0].conc[i, A], omcd[0].conc[i, B], omcd[0].conc[i, LIS] = map(float, f.readline().split())
    omcd[0].ph[P],  omcd[0].ph[A],  omcd[0].ph[B],  omcd[0].ph[LIS]  = map(float, f.readline().split())
    omcd[0].vol[P], omcd[0].vol[A], omcd[0].vol[B], omcd[0].vol[LIS] = map(float, f.readline().split())
    omcd[0].ep[P],  omcd[0].ep[A],  omcd[0].ep[B],  omcd[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZ + 1):
    qnewton2icb(omcd, jz - 1, 8, omcd[0].vol, omcd[jz].volEinit, omcd[jz].volPinit, omcd[jz].volAinit, omcd[jz].volBinit, NDC, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('OMCoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{omcd[NZ].conc[i, LUM]:12.5f} {omcd[NZ].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{omcd[NZ].conc[i, LUM]:12.5e} {omcd[NZ].conc[i, BATH]:12.5e}\n")
    f.write(f"{omcd[NZ].ph[LUM]:12.5f} {omcd[NZ].ph[BATH]:12.5f}\n")
    f.write(f"{omcd[NZ].ep[LUM]:12.5f} {omcd[NZ].ep[BATH]:12.5f}\n")
    f.write(f"{omcd[NZ].vol[LUM]:12.5e} {omcd[NZ].vol[BATH]:12.5e}\n")

with open('OMCoutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{omcd[NZ].conc[i, P]:12.5f} {omcd[NZ].conc[i, A]:12.5f} {omcd[NZ].conc[i, B]:12.5f} {omcd[NZ].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{omcd[NZ].conc[i, P]:12.5e} {omcd[NZ].conc[i, A]:12.5e} {omcd[NZ].conc[i, B]:12.5e} {omcd[NZ].conc[i, LIS]:12.5e}\n")
    f.write(f"{omcd[NZ].ph[P]:12.5f} {omcd[NZ].ph[A]:12.5f} {omcd[NZ].ph[B]:12.5f} {omcd[NZ].ph[LIS]:12.5f}\n")
    f.write(f"{omcd[NZ].ep[P]:12.5f} {omcd[NZ].ep[A]:12.5f} {omcd[NZ].ep[B]:12.5f} {omcd[NZ].ep[LIS]:12.5f}\n")
    f.write(f"{omcd[NZ].vol[P]:12.5e} {omcd[NZ].vol[A]:12.5e} {omcd[NZ].vol[B]:12.5e} {omcd[NZ].vol[LIS]:12.5e}\n")

# =====================================================================
# IMCD
# =====================================================================

print("****************************************************")
print("                    IMCD RESULTS")
print("****************************************************")
print()

# Luminal concentrations are set to OMCD outflow values in initIMC.
# Initial guesses for cellular and LIS concentrations are in IMCresults.
initIMC(imcd)

with open('IMCresults', 'r') as f:
    for i in range(NS):
        imcd[0].conc[i, P], imcd[0].conc[i, LIS] = map(float, f.readline().split())
    imcd[0].ph[P],  imcd[0].ph[LIS]  = map(float, f.readline().split())
    imcd[0].vol[P], imcd[0].vol[LIS] = map(float, f.readline().split())
    imcd[0].ep[P],  imcd[0].ep[LIS]  = map(float, f.readline().split())

for jz in range(1, NZIMC + 1):
    qnewton2b(imcd, imcd[0].vol, jz - 1, 9, NDIMC, PTinitVol, xNaPiIIaPT, xNaPiIIcPT, xPit2PT)

with open('IMCoutlet', 'w') as f:
    for i in range(4):
        f.write(f"{imcd[NZIMC].conc[i, LUM]:12.5f} {imcd[NZIMC].conc[i, BATH]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{imcd[NZIMC].conc[i, LUM]:12.5e} {imcd[NZIMC].conc[i, BATH]:12.5e}\n")
    f.write(f"{imcd[NZIMC].ph[LUM]:12.5f} {imcd[NZIMC].ph[BATH]:12.5f}\n")
    f.write(f"{imcd[NZIMC].ep[LUM]:12.5f} {imcd[NZIMC].ep[BATH]:12.5f}\n")
    f.write(f"{imcd[NZIMC].vol[LUM]:12.5e} {imcd[NZIMC].vol[BATH]:12.5e}\n")

with open('IMCoutlet_all', 'w') as f:
    for i in range(4):
        f.write(f"{imcd[NZIMC].conc[i, P]:12.5f} {imcd[NZIMC].conc[i, LIS]:12.5f}\n")
    for i in range(4, NS):
        f.write(f"{imcd[NZIMC].conc[i, P]:12.5e} {imcd[NZIMC].conc[i, LIS]:12.5e}\n")
    f.write(f"{imcd[NZIMC].ph[P]:12.5f} {imcd[NZIMC].ph[LIS]:12.5f}\n")
    f.write(f"{imcd[NZIMC].ep[P]:12.5f} {imcd[NZIMC].ep[LIS]:12.5f}\n")
    f.write(f"{imcd[NZIMC].vol[P]:12.5e} {imcd[NZIMC].vol[LIS]:12.5e}\n")

# =====================================================================
# Final urinary output
# =====================================================================

outlet[:] = imcd[NZ].vol[LUM] * imcd[NZ].conc[:, LUM] * cw

with open('Outlet_SS', 'w') as f:
    for val in outlet:
        f.write(f"{val:12.5f}\n")

# Outp_1 = imcd[NZ].vol[LUM] * cw   # luminal flow at IMCD outlet (ml/min)
# Outp_2 = outlet[NA]                # Na  outlet flux
# Outp_3 = outlet[K]                 # K   outlet flux
# Outp_4 = imcd[NZ].ph[LUM]         # luminal pH at IMCD outlet

# --- Stop timer ---

toc = time.time()
