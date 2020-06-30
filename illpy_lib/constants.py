"""
"""
import numpy as np
#change##
import illpy_lib.illcosmo as ilc


KPC = 3.085677581467192e+21   # kpc in cm
MSOL = 1.9884754153381438e+33   # Solar-mass in grams
YR = 31557600.0   # year in seconds
BOX_LENGTH = ilc.Illustris_Cosmology_TOS.BOX_LENGTH
HPAR       = ilc.Illustris_Cosmology_TOS.HPAR
BOX_VOLUME_CGS = np.power(BOX_LENGTH*KPC/HPAR, 3.0)
NUM_SNAPS = 136
# Indices for Different Types of Particles


class CONV_ILL_TO_CGS:
    """Convert from illustris units to physical [cgs] units (multiply).
    """
    MASS = 1.0e10*MSOL/0.704              # Convert from e10 Msol to [g]
    MDOT = 10.22*MSOL/YR                  # Multiply by this to get [g/s]
    DENS = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
    DIST = KPC/0.704                      # Convert from [ckpc/h] to [comoving cm]
    VEL  = 1.0e5                          # [km/s] to [cm/s]
    CS   = 1.0      


class PARTICLE(object):
    GAS  = 0
    DM   = 1
    TRAC = 3
    STAR = 4
    BH   = 5

    # _NAMES = ["Gas", "DM", "-", "Tracer", "Star", "BH"]
    # _NUM  = 6


# Numerical Constants
class DTYPE(object):
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64


_ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                          2: "L75n910FP",
                          3: "L75n455FP"}

#_PROCESSED_DIR = "/n/home00/lkelley/hernquistfs1/illustris/data/%s/output/postprocessing/"
_PROCESSED_DIR= "/n/home09/sayebms/illpy_output"
#_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/%s/output/"
_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/hernquistfs3/IllustrisTNG/Runs/%s/output/"

_DM_MASS = {1: 4.408965e-04,
            2: 3.527172e-03,
            3: 2.821738e-02}

cdh
def GET_ILLUSTRIS_DM_MASS(run):
    return _DM_MASS[run]


# def GET_BAD_SNAPS(run):
#     return _BAD_SNAPS[run]


def GET_ILLUSTRIS_RUN_NAMES(run):
    return _ILLUSTRIS_RUN_NAMES[run]


def GET_ILLUSTRIS_OUTPUT_DIR(run):
    return _ILLUSTRIS_OUTPUT_DIR_BASE % (_ILLUSTRIS_RUN_NAMES[run])


def GET_PROCESSED_DIR(run):
    return _PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
