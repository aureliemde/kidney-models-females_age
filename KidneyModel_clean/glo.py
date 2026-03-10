"""
Global shared state module.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module defines global arrays and namespace objects that are shared
across all segment modules. The Common-based namespaces mirror Fortran
COMMON blocks. All segment init and flux modules import this file.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *

fENaC = np.zeros(NZ + 1)
fNCC = np.zeros(NZ + 1)
fNCX_dct = np.zeros(NZ + 1)
fPMCA_dct = np.zeros(NZ + 1)
fTRPV5_dct = np.zeros(NZ + 1)
fNCX_cnt = np.zeros(NZ + 1)
fPMCA_cnt = np.zeros(NZ + 1)
fTRPV5_cnt = np.zeros(NZ + 1)
fTRPV4_cnt = np.zeros(NZ + 1)
fCaTransPT = np.zeros(NZ + 1)
fCaParaPT = np.zeros(NZ + 1)
PTtorque = np.zeros(NZ + 1)

# Specify Common blocks
class Common:
    """Simple namespace struct, replacing Fortran COMMON blocks."""
    pass

# length common block
#length = Common()
#length.LzPT = 0.0
#length.LzA = 0.0
#length.LzT = 0.0
#length.LzD = 0.0
#length.LzC = 0.0
#length.LzCCD = 0.0
#length.LzOMC = 0.0
#length.LzIMC = 0.0

# Sofia: from float to integer
length = Common()
length.LzPT = 0
length.LzA = 0
length.LzT = 0
length.LzD = 0
length.LzC = 0
length.LzCCD = 0
length.LzOMC = 0
length.LzIMC = 0

# options common block
options = Common()
options.ndiabetes = 0

# scaling common block
scaling = Common()
scaling.fscaleENaC = 0.0
scaling.fscaleNaK = 0.0
scaling.fscaleNaK_PC = 0.0

# pheffects common block
pheffects = Common()
pheffects.hENaC_CNT = 0.0
pheffects.hENaC_CCD = 0.0
pheffects.hENaC_OMC = 0.0
pheffects.hENaC_IMC = 0.0
pheffects.hROMK_CNT = 0.0
pheffects.hROMK_CCD = 0.0
pheffects.hROMK_OMC = 0.0
pheffects.hROMK_IMC = 0.0
pheffects.hCltj_CNT = 0.0
pheffects.hCltj_CCD = 0.0
pheffects.hCltj_OMC = 0.0
pheffects.hCltj_IMC = 0.0

# cond1 common block
cond1 = Common()
cond1.sngfr = 0.0
cond1.dctout_ref = 0.0
cond1.dctout_flow = 0.0

# cond2 common block
cond2 = Common()
cond2.BathImperm = BathImperm
cond2.LumImperm = LumImperm

# cond3 common block
cond3 = Common()
cond3.PTinitVol = 0.0
cond3.PTtorque = PTtorque

# NaPO4 common block
NaPO4 = Common()
NaPO4.xNaPiIIaPT = 0.0
NaPO4.xNaPiIIbPT = 0.0
NaPO4.xNaPiIIcPT = 0.0
NaPO4.xPit2PT = 0.0

# NaIMCD common block
NaIMCD = Common()
NaIMCD.fENaC = fENaC
NaIMCD.fNCC = fNCC

# TRPV5 common block
TRPV5 = Common()
TRPV5.xTRPV5_dct = 0.0
TRPV5.xTRPV5_cnt = 0.0

# Calcium common block
Calcium = Common()
Calcium.fNCX_dct = fNCX_dct
Calcium.fPMCA_dct = fPMCA_dct
Calcium.fTRPV5_dct = fTRPV5_dct
Calcium.fNCX_cnt = fNCX_cnt
Calcium.fPMCA_cnt = fPMCA_cnt
Calcium.fTRPV5_cnt = fTRPV5_cnt
Calcium.xPTRPV4_cnt = 0.0
Calcium.fTRPV4_cnt = fTRPV4_cnt
Calcium.fCaTransPT = fCaTransPT
Calcium.fCaParaPT = fCaParaPT

# gradOM common block
gradOM = Common()
gradOM.TotSodOI = 0.0
gradOM.TotSodCM = 0.0
gradOM.TotPotOI = 0.0
gradOM.TotPotCM = 0.0
gradOM.TotCloOI = 0.0
gradOM.TotCloCM = 0.0
gradOM.TotBicOI = 0.0
gradOM.TotBicCM = 0.0
gradOM.TotHcoOI = 0.0
gradOM.TotHcoCM = 0.0
gradOM.TotCo2OI = 0.0
gradOM.TotCo2CM = 0.0
gradOM.TotPhoOI = 0.0
gradOM.TotPhoCM = 0.0
gradOM.TotureaOI = 0.0
gradOM.TotureaCM = 0.0
gradOM.TotAmmOI = 0.0
gradOM.TotAmmCM = 0.0
gradOM.TotHco3OI = 0.0
gradOM.TotHco3CM = 0.0
gradOM.TotGluOI = 0.0
gradOM.TotGluCM = 0.0
gradOM.TotHco2OI = 0.0
gradOM.TotHco2CM = 0.0

# gradCJ common block
gradCJ = Common()
gradCJ.TotCloCJ = 0.0
gradCJ.TotCloCT = 0.0
gradCJ.TotAmmCJ = 0.0
gradCJ.TotAmmCT = 0.0

# gradIM common block
gradIM = Common()
gradIM.TotSodPap = 0.0
gradIM.TotPotPap = 0.0
gradIM.TotCloPap = 0.0
gradIM.TotBicPap = 0.0
gradIM.TotHcoPap = 0.0
gradIM.TotCo2Pap = 0.0
gradIM.TotPhoPap = 0.0
gradIM.TotAmmPap = 0.0
gradIM.TotureaPap = 0.0
gradIM.TotHco2Pap = 0.0
gradIM.TotGluPap = 0.0
gradIM.GluInt = 0.0
gradIM.Hco2Int = 0.0
