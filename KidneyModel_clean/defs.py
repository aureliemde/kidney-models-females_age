"""
Membrane data structure definition.

Written originally in Fortran by Prof. Aurelie Edwards
Translated to Python by Dr. Mohammad M. Tajdini
Refactored for efficiency by Sofia Polychroniadou

Department of Biomedical Engineering, Boston University

This module defines the Membrane class, which holds all per-node state and
parameters for a single cross-section of a tubule segment: concentrations,
volumes, potentials, permeabilities, transporter expression levels, and
kinetic parameters.

Units: CGS system (cm, mmol, mmol/cm³ = M, s)
"""

import numpy as np

from values import *

class Membrane:
    """Per-node state and parameters for one cross-section of a tubule segment.

    Holds predicted variables (concentrations, pH, volumes, potentials),
    geometry, permeabilities, transporter expression levels, and kinetic
    parameters. One Membrane instance per spatial grid node per segment.

    Parameters
    ----------
    NSPT : int -- number of solutes
    NC   : int -- number of compartments
    NS   : int -- number of solutes (legacy alias for NSPT)
    """
    def __init__(self, NSPT, NC, NS):

        # Predicted variables
        self.conc = np.zeros((NSPT, NC))  # solute concentrations
        self.ph = np.zeros(NC)            # pH
        self.vol = np.zeros(NC)           # volume
        self.ep = np.zeros(NC)            # membrane potential
        self.pres = 0.0                   # fluid pressure

        # Initial volumes
        self.volEinit = 0.0
        self.volPinit = 0.0
        self.volAinit = 0.0
        self.volBinit = 0.0
        self.volLuminit = 0.0
        self.sbasEinit = 0.0              # initial area

        # Tubular characteristics
        self.dimL = 0.0                   # tubule length
        self.diam = 0.0                   # luminal diameter
        self.area = np.zeros((NC, NC))    # membrane surface area
        self.dLPV = np.zeros((NC, NC))    # water permeability
        self.sig = np.zeros((NS, NC, NC)) # solute reflection coefficient
        self.h = np.zeros((NS, NC, NC))   # solute permeability
        self.dLA = np.zeros((NS, NS, NC, NC)) # NET coefficient

        # Maximum ATPase flux
        self.ATPNaK = np.zeros((NC, NC))
        self.ATPH = np.zeros((NC, NC))
        self.ATPHK = np.zeros((NC, NC))

        # NHE3 and NHE1 expressions
        self.xNHE3 = 0.0
        self.xNHE1 = np.zeros(NC)

        # SGLT2/SGLT1 expression parameters
        self.xSGLT2 = 0.0
        self.xSGLT1 = 0.0
        self.CTsglt1 = 0.0
        self.CTsglt2 = 0.0

        # GLUT2/GLUT1 expressions
        self.xGLUT2 = 0.0
        self.xGLUT1 = 0.0
        self.CTglut1 = 0.0
        self.CTglut2 = 0.0

        # NKCC2 expression
        self.xNKCC2A = 0.0
        self.xNKCC2B = 0.0
        self.xNKCC2F = 0.0

        # NCC expression
        self.xNCC = 0.0

        # KCC4 expression
        self.xKCC4 = 0.0
        self.xKCC4A = 0.0
        self.xKCC4B = 0.0

        # CO2/HCO3/H2CO3 reaction rates
        self.dkd = np.zeros(NC)
        self.dkh = np.zeros(NC)

        # Rate of ammoniagenesis
        self.qnh4 = 0.0

        # AE1 exchanger at peritubular membrane of alpha cell
        self.xAE1 = 0.0

        # Pendrin exchanger at apical membrane of beta cell
        self.xPendrin = 0.0

        # NDBCE exchanger at MB interface
        self.xNDBCE = 0.0

        # NCX exchanger on basolateral membranes
        self.xNCX = 0.0

        # Maximum flux across PMCA pump on basolateral membranes
        self.PMCA = 0.0

        # Various maximum permeabilities
        self.hNaMP = 0.0
        self.hCLCA = 0.0
        self.hCLCB = 0.0

        # Total buffer concentrations
        self.cPbuftot = 0.0
        self.cAbuftot = 0.0
        self.cBbuftot = 0.0

        # Reference impermeant cellular concentrations
        self.cPimpref = 0.0
        self.cAimpref = 0.0
        self.cBimpref = 0.0

        # Impermeant valence
        self.zPimp = 0.0
        self.zAimp = 0.0
        self.zBimp = 0.0

        # Parameter to scale torque effect
        self.scaleT = 0.0
        self.TM0 = 0.0  # Reference torque

        # Parameter for coalescing tubules
        self.coalesce = 0.0

        # Record for keeping track of fluxes
        self.FNatrans = 0.0
        self.FNapara = 0.0
        self.FNaK = 0.0
        self.FHase = 0.0
        self.FGluPara = 0.0
        self.FGluSGLT1 = 0.0
        self.FGluSGLT2 = 0.0
        self.FKtrans = 0.0
        self.FKpara = 0.0

        # Calculations for metabolism
        self.TQ = 0.0
        self.nephronAct = 0.0
        self.nephronTNa = 0.0
        self.nephronQO2 = 0.0
