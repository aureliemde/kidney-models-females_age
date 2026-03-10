"""Written originally in Fortran by Prof. Aurelie Edwards
 Translated to Python by Dr. Mohammad M. Tajdini
 Refactored by Sofia Polychroniadou

 Department of Biomedical Engineering
 Boston University"""


import numpy as np

from values import *
from glo import *
from defs import *

from compute_o2_consumption import compute_o2_consumption


def out_data_PT(pt):
    """Print PT transport summary and compute O2 consumption for the whole PT, PCT, and S3."""

    inlet  = np.zeros(NSPT)
    outlet = np.zeros(NSPT)
    fluxs  = np.zeros(NSPT)
    deliv  = np.zeros(NSPT)

    # Assume 36,000 nephrons coalescing to 7,200 CDs; convert to ml/min
    cw = Vref * 60.0 * 36000

    deliv[:] = pt[0].conc[:, LUM] * pt[0].vol[LUM] * cw

    print("****************************************************")
    print("\nOVERALL PT RESULTS\n")
    volreab = (pt[0].vol[LUM] - pt[NZ].vol[LUM]) * cw
    print(f" Water in, reabsorbed, out (nl/min): {pt[0].vol[LUM] * cw * 1.0e3:.3f}, {-volreab * 1.0e3:.3f}, {pt[NZ].vol[LUM] * cw * 1.0e3:.3f}")
    print(f" % Water reabsorbed: {1.0 - pt[NZ].vol[LUM] / pt[0].vol[LUM]:.3f}")
    print(f" distal pressure: {pt[NZ].pres:.4f}\n")

    print("Solute in, reabsorbed, out (pmol/min)\n")

    outlet[:] = pt[NZ].vol[LUM] * pt[NZ].conc[:, LUM] * cw
    inlet[:]  = pt[0].vol[LUM]  * pt[0].conc[:, LUM]  * cw
    fluxs[:]  = inlet - outlet
    for i in range(NSPT):
        print(f" Solute {i}: {inlet[i]:.4f}, {-fluxs[i]:.4f}, {outlet[i]:.4f}")
    print("\n")

    ReabNapt = (pt[0].vol[LUM] * pt[0].conc[NA, LUM] - pt[NZ].vol[LUM] * pt[NZ].conc[NA, LUM]) * cw
    ReabClpt = (pt[0].vol[LUM] * pt[0].conc[CL, LUM] - pt[NZ].vol[LUM] * pt[NZ].conc[CL, LUM]) * cw
    ReabKpt  = (pt[0].vol[LUM] * pt[0].conc[K,  LUM] - pt[NZ].vol[LUM] * pt[NZ].conc[K,  LUM]) * cw

    print("PT reabs: absolute vs. percentage")
    print(f" Na: {ReabNapt:.4f}, {ReabNapt / (pt[0].vol[LUM] * pt[0].conc[NA, LUM] * cw):.4f}")
    print(f" Cl: {ReabClpt:.4f}, {ReabClpt / (pt[0].vol[LUM] * pt[0].conc[CL, LUM] * cw):.4f}")
    print(f" K : {ReabKpt:.4f},  {ReabKpt  / (pt[0].vol[LUM] * pt[0].conc[K,  LUM] * cw):.4f}")
    print(f" pH: {pt[0].ph[LUM]:.4f}, {pt[NZ].ph[LUM]:.4f}")

    nephronAct, nephronTNa, nephronQO2, nephronTK = compute_o2_consumption(pt, 'PT', dimLPT, 0, NZ)

    NZS3 = int(88 * NZ / 100 - 1)
    ReabGluS3 = (pt[NZS3].vol[LUM] * pt[NZS3].conc[GLU, LUM] - pt[NZ].vol[LUM] * pt[NZ].conc[GLU, LUM]) * cw

    nephronActPCT, nephronTNaPCT, nephronQO2PCT, nephronTKPCT = compute_o2_consumption(pt, 'PCT', dimLPT * NZS3 / NZ, 0, NZS3)
    nephronActS3,  nephronTNaS3,  nephronQO2S3,  nephronTKS3  = compute_o2_consumption(pt, 'S3',  dimLPT - dimLPT * NZS3 / NZ, NZS3, NZ)

    return nephronAct, nephronTNa, nephronQO2, nephronTK
