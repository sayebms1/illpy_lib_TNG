"""

"""

import os
import numpy as np

import cosmopy

from zcode.constants import KPC, MSOL, YR

# Get local path, and data directory
_DATA_PATH = "%s/data/" % os.path.dirname(os.path.abspath(__file__))
# Contains cosmological values for each snapshot
_TIMES_FILENAME_TOS = "illustris-snapshot-cosmology-data.npz"
_TIMES_FILENAME_TNG = "illustris-tng_snapshot-cosmology-data.npz"


class _Illustris_Cosmology(cosmopy.cosmology.Cosmology):

    Omega0 = None
    OmegaLambda = None
    OmegaBaryon = None
    HPAR = None
    FNAME = None
    H0 = None

    BOX_LENGTH = 75000                          # [ckpc/h]
    BOX_VOLUME_MPC3 = None
    BOX_VOLUME_CGS = None
    NUM_SNAPS = None
    _BAD_SNAPS = None

    _Z_GRID = [10.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.02]
    _INTERP_POINTS = 40

    def __init__(self, core=None, log=None):
        super().__init__()
        if log is None and core is not None:
            log = core.log

        if log is not None:
            log.debug("Initializing `Illustris_Cosmology`")

        # tng_flag = core.sets.TNG
        # fname = _TIMES_FILENAME_TNG if tng_flag else _TIMES_FILENAME
        fname = os.path.join(_DATA_PATH, self.FNAME)

        cosmo_data = np.load(fname)
        self.snapshot_scales = cosmo_data['scale']
        msg = "Loaded cosmology with {} snapshot scales".format(self.snapshot_scales.size)
        if log is None:
            print(msg)
        else:
            log.info(msg)

        class CONV_ILL_TO_CGS:
            """Convert from illustris units to physical [cgs] units (multiply).
            """
            MASS = 1.0e10*MSOL/self.HPAR          # Convert from e10 Msol to [g]
            MDOT = 10.22*MSOL/YR                  # Multiply by this to get [g/s]
            DENS = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
            DIST = KPC/self.HPAR                  # Convert from [ckpc/h] to [comoving cm]
            VEL  = 1.0e5                          # [km/s] to [cm/s]
            CS   = 1.0                            # ??????? FIX

        class CONV_CGS_TO_SOL:
            """Convert from cgs units to (standard) solar units, e.g. Msol, PC, etc, by multiplication
            """
            MASS  = 1.0/MSOL                       # [g] ==> Msol
            MDOT  = YR/MSOL                        # [g/s] ==> [Msol/yr]
            DENS  = np.power(KPC/1000.0, 3.0)/MSOL          # [g/cm^3] ==> [Msol/pc^3]
            NDENS = np.power(KPC/1000.0, 3.0)               # [1/cm^3] ==> [1/pc^3]
            DIST  = 1000.0/KPC                         # [cm] ==> [pc]
            VEL   = 1.0e-5                         # [cm/s] ==> [km/s]
            ENER  = 1.0e-10                        # [erg/g] ==> [(km/s)^2]

        class CONV_ILL_TO_SOL:
            """Convert from illustris units to standard solar units (e.g. Msol, pc), by multiplication
            """
            MASS = CONV_ILL_TO_CGS.MASS * CONV_CGS_TO_SOL.MASS  # e10 Msol to [Msol]
            MDOT = CONV_ILL_TO_CGS.MDOT * CONV_CGS_TO_SOL.MDOT  # to [Msol/yr]
            DENS = CONV_ILL_TO_CGS.DENS * CONV_CGS_TO_SOL.DENS  # to [Msol/pc^3]
            DIST = CONV_ILL_TO_CGS.DIST * CONV_CGS_TO_SOL.DIST  # to comoving-pc

            VEL = 1.0

        self.CONV_ILL_TO_CGS = CONV_ILL_TO_CGS
        self.CONV_CGS_TO_SOL = CONV_CGS_TO_SOL
        self.CONV_ILL_TO_SOL = CONV_ILL_TO_SOL

        self.BOX_VOLUME_MPC3 = np.power(self.BOX_LENGTH*1e-3/self.HPAR, 3.0)
        self.BOX_VOLUME_CGS = np.power(self.BOX_LENGTH*KPC/self.HPAR, 3.0)

        return

    def scales(self):
        return np.array(self.snapshot_scales)

    def GET_BAD_SNAPS(self, run):
        return self._BAD_SNAPS[run]


class Illustris_Cosmology_TOS(_Illustris_Cosmology):

    Omega0 = 0.2726
    OmegaLambda = 0.7274
    OmegaBaryon = 0.0456
    HPAR = 0.704
    H0 = HPAR * 100.0

    FNAME = _TIMES_FILENAME_TOS
    NUM_SNAPS = 136

    _BAD_SNAPS = {1: [53, 55],
                  2: [],
                  3: []}


class Illustris_Cosmology_TNG(_Illustris_Cosmology):

    Omega0 = 0.3089
    OmegaLambda = 0.6911
    OmegaBaryon = 0.0486
    HPAR = 0.6774
    H0 = HPAR * 100.0

    FNAME = _TIMES_FILENAME_TNG
    NUM_SNAPS = 100

    _BAD_SNAPS = {1: [],
                  2: [],
                  3: []}
