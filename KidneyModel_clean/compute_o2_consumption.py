# Written originally in Fortran by Prof. Aurelie Edwards
# Translated to Python by Dr. Mohammad M. Tajdini

# Department of Biomedical Engineering
# Boston University 

###################################################

#---------------------------------------------------------------------72
# Sodium Transport and Oxygen Consumption
#---------------------------------------------------------------------72

import numpy as np

from numba import njit

from values import *
from defs import *

@njit  
def compute_o2_consumption(tube, tubename, dimL, ind1, ind2):
    
    # outputs:
    nephronAct = 0
    nephronTNa = 0
    nephronQO2 = 0
    nephronTK = 0
        
# nHATPase = 1 if we account for ATP consumption by H-ATPase pumps in the PT
    nHATPase = 0  

    totalAct = (0.5 * (tube[ind1].FNaK + tube[ind2].FNaK) + np.sum([tube[i].FNaK for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)
    totalTNa = (0.5 * (tube[ind1].FNatrans + tube[ind2].FNatrans) + np.sum([tube[i].FNatrans for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)
    totalTNa += (0.5 * (tube[ind1].FNapara + tube[ind2].FNapara) + np.sum([tube[i].FNapara for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)
    totalHpump = (0.5 * (tube[ind1].FHase + tube[ind2].FHase) + np.sum([tube[i].FHase for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)

    o2consum = totalAct / tube[0].TQ + np.abs(totalHpump) / 10.0 * nHATPase

    totalTK = (0.5 * (tube[ind1].FKtrans + tube[ind2].FKtrans) + np.sum([tube[i].FKtrans for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)
    totalTK += (0.5 * (tube[ind1].FKpara + tube[ind2].FKpara) + np.sum([tube[i].FKpara for i in range(ind1+1, ind2)])) * dimL / (ind2 - ind1)

    print(f"{tubename} {totalAct*10.0:.11f} {totalTNa*10.0:.11f} {o2consum*10.0:.11f}")
    print(f"{totalAct*10.0:.11f} {totalTNa*10.0:.11f} {o2consum*10.0:.11f}")

    print(f"{tubename} {totalTNa*10.0:.11f} {totalTK*10.0:.11f} {o2consum*10.0:.11f}")
    print(f"{totalTNa*10.0:.6f} {totalTK*10.0:.6f} {o2consum*10.0:.6f}")

    nephronAct += totalAct * 10.0
    nephronTNa += totalTNa * 10.0
    nephronQO2 += o2consum * 10.0
    nephronTK += totalTK * 10.0

    return nephronAct, nephronTNa, nephronQO2, nephronTK
