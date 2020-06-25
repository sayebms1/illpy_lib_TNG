"""Constants for Blackhole related functions and submodules.

Classes
-------
    MERGERS : enum-type class for BH-Merger dictionary keys.
              The list ``MERGERS_PHYSICAL_KEYS`` contains the keys which pertain to values taken
              from the BH Merger files themselves
    DETAILS : enum-type class for BH-Details entries dictionary keys.
              The list ``DETAILS_PHYSICAL_KEYS`` contains the keys corresponding to values taken
              from the BH Details files themselves
    BH_TYPE : enum-type class for tracking the two types {``IN``, ``OUT``} of Merger BHs.
              The ``OUT`` BH is the one which persists after the merger, while the ``IN`` BH
              effectively dissappears.
    BH_TIME : enum-type class for the three stored, details times {``FIRST``, ``BEFORE``, ``AFTER``}.
    BH_TREE : enum-type class for BH merger tree dictionary keys.
    BH_SNAP : enum class for BHSnapshotData dictionary keys.

Functions
---------
-   _loadLogger     - Initialize a ``logging.Logger`` object for output messages.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
from glob import glob
import numpy as np

import zcode.inout as zio

from .. Constants import GET_ILLUSTRIS_RUN_NAMES, _PROCESSED_DIR, DTYPE, NUM_SNAPS


# Illustris Parameters
# ====================

_ILLUSTRIS_MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"
_ILLUSTRIS_DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

_ILLUSTRIS_MERGERS_DIRS = {3: "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_mergers/",
                           2: "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_mergers/",
                           1: ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_mergers/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_mergers/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_mergers/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_mergers/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_mergers/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_mergers/"]
                           }

_ILLUSTRIS_DETAILS_DIRS = {3: "/n/ghernquist/Illustris/Runs/L75n455FP/output/blackhole_details/",
                           2: "/n/ghernquist/Illustris/Runs/L75n910FP/combined_output/blackhole_details/",
                           1: ["/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-curie/blackhole_details/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-supermuc/blackhole_details/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_details/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_details/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_details/",
                               "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_details/"]
                           }


# Post-Processing Parameters
# ==========================
_MAX_DETAILS_PER_SNAP           = 10   # Number of details entries to store per snapshot

_PROCESSED_MERGERS_DIR          = _PROCESSED_DIR + "blackhole_mergers/"
_PROCESSED_DETAILS_DIR          = _PROCESSED_DIR + "blackhole_details/"

_MERGERS_RAW_COMBINED_FILENAME  = "ill-%d_blackhole_mergers_combined.txt"
_MERGERS_RAW_MAPPED_FILENAME    = "ill-%d_blackhole_mergers_mapped_v%.2f.npz"
_MERGERS_FIXED_FILENAME         = "ill-%d_blackhole_mergers_fixed_v%.2f.npz"

_DETAILS_TEMP_FILENAME          = "ill-%d_blackhole_details_temp_snap-%d.txt"
_DETAILS_SAVE_FILENAME          = "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"

_MERGER_DETAILS_FILENAME        = 'ill-%d_blackhole_merger-details_persnap-%03d_v%s.npz'
_REMNANT_DETAILS_FILENAME       = 'ill-%d_blackhole_remnant-details_persnap-%03d_v%s.npz'

_DETAILS_UNIQUE_IDS_FILENAME     = 'ill-%d_blackhole_details_unique-ids_snap-%03d_v%s.npz'
_DETAILS_ALL_UNIQUE_IDS_FILENAME = 'ill-%d_blackhole_details_all-unique-ids_v%s.npz'

_BLACKHOLE_TREE_FILENAME         = "ill-%d_bh-tree_v%.2f.npz"
_BLACKHOLE_TREE_DETAILS_FILENAME = "ill-%d_fin-merger-%d_bh-tree-details_v%s.npz"


_LOG_DIR = "./logs/"


class MERGERS():
    # Meta Data
    RUN       = 'run'
    CREATED   = 'created'
    NUM       = 'num'
    VERSION   = 'version'
    FILE      = 'filename'

    # Physical Parameters
    IDS       = 'ids'
    SCALES    = 'scales'
    MASSES    = 'masses'

    # Maps
    MAP_STOM  = 's2m'
    MAP_MTOS  = 'm2s'
    MAP_ONTOP = 'ontop'

MERGERS_PHYSICAL_KEYS = [MERGERS.IDS, MERGERS.SCALES, MERGERS.MASSES]


class DETAILS():
    RUN     = 'run'
    CREATED = 'created'
    VERSION = 'version'
    NUM     = 'num'
    SNAP    = 'snap'
    FILE    = 'filename'

    IDS     = 'id'
    SCALES  = 'scales'
    MASSES  = 'masses'
    MDOTS   = 'mdots'
    RHOS    = 'rhos'
    CS      = 'cs'


DETAILS_PHYSICAL_KEYS = [DETAILS.IDS, DETAILS.SCALES, DETAILS.MASSES,
                         DETAILS.MDOTS, DETAILS.RHOS, DETAILS.CS]


class BH_TYPE():
    IN  = 0
    OUT = 1

NUM_BH_TYPES = 2


class BH_TIME():
    BEFORE  = 0                                   # Before merger time (MUST = 0!)
    AFTER   = 1                                   # After (or equal) merger time (MUST = 1!)
    FIRST   = 2                                   # First matching details entry (MUST = 2!)

NUM_BH_TIMES = 3


class BH_TREE():
    LAST         = 'last'
    NEXT         = 'next'
    LAST_TIME    = 'lastTime'
    NEXT_TIME    = 'nextTime'
    NUM_FUTURE   = 'numFuture'
    NUM_PAST     = 'numPast'
    TIME_BETWEEN = 'timeBetween'

    CREATED      = 'created'
    RUN          = 'run'
    VERSION      = 'version'


class BH_SNAP():
    RUN     = 'run'
    SNAP    = 'snap'
    VERSION = 'version'
    CREATED = 'created'
    DIR_SRC = 'directory'
    VALID   = 'valid'
    TARGET  = 'target'
    FIELDS  = 'snapshot_fields'
    DTYPES  = 'snapshot_dtypes'

SNAPSHOT_FIELDS = ['ParticleIDs', 'BH_Hsml', 'BH_Mass', 'Masses',
                   'SubfindHsml', 'BH_Mdot', 'BH_Density']
SNAPSHOT_DTYPES = [DTYPE.ID, DTYPE.SCALAR, DTYPE.SCALAR, DTYPE.SCALAR,
                   DTYPE.SCALAR, DTYPE.SCALAR, DTYPE.SCALAR]


def GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run):
    filesDir = _ILLUSTRIS_MERGERS_DIRS[run]
    files = []
    if(type(filesDir) != list): filesDir = [filesDir]

    for fdir in filesDir:
        filesNames = fdir + _ILLUSTRIS_MERGERS_FILENAME_REGEX
        someFiles = sorted(glob(filesNames))
        files += someFiles

    return files


def GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run):
    filesDir = _ILLUSTRIS_DETAILS_DIRS[run]
    files = []
    if(type(filesDir) != list): filesDir = [filesDir]

    for fdir in filesDir:
        filesNames = fdir + _ILLUSTRIS_DETAILS_FILENAME_REGEX
        someFiles = sorted(glob(filesNames))
        files += someFiles

    return files


def GET_MERGERS_RAW_COMBINED_FILENAME(run):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_COMBINED_FILENAME % (run)
    return fname


def GET_MERGERS_RAW_MAPPED_FILENAME(run, version):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_RAW_MAPPED_FILENAME % (run, version)
    return fname


def GET_MERGERS_FIXED_FILENAME(run, version):
    fname = _PROCESSED_MERGERS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGERS_FIXED_FILENAME % (run, version)
    return fname


def GET_DETAILS_TEMP_FILENAME(run, snap):
    fname = _PROCESSED_DETAILS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_TEMP_FILENAME % (run, snap)
    return fname


def GET_DETAILS_SAVE_FILENAME(run, snap, version):
    fname = _PROCESSED_DETAILS_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_SAVE_FILENAME % (run, snap, version)
    return fname


def GET_MERGER_DETAILS_FILENAME(run, version, maxPerSnap):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _MERGER_DETAILS_FILENAME % (run, maxPerSnap, version)
    return fname


def GET_REMNANT_DETAILS_FILENAME(run, version, maxPerSnap):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _REMNANT_DETAILS_FILENAME % (run, maxPerSnap, version)
    return fname


def GET_BLACKHOLE_TREE_FILENAME(run, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _BLACKHOLE_TREE_FILENAME % (run, version)
    return fname


def GET_BLACKHOLE_TREE_DETAILS_FILENAME(run, fmrg, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _BLACKHOLE_TREE_DETAILS_FILENAME % (run, fmrg, version)
    return fname


def GET_DETAILS_UNIQUE_IDS_FILENAME(run, snap, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_UNIQUE_IDS_FILENAME % (run, snap, version)
    return fname


def GET_DETAILS_ALL_UNIQUE_IDS_FILENAME(run, version):
    fname = _PROCESSED_DIR % (GET_ILLUSTRIS_RUN_NAMES(run))
    fname += _DETAILS_ALL_UNIQUE_IDS_FILENAME % (run, version)
    return fname


def _GET_STATUS_FILENAME(name, run=None, version=None):
    statFilename = os.path.splitext(os.path.basename(name)) + "_stat"
    if(run): statFilename += "_ill%d" % (run)
    if(version): statFilename += "_v%s" % (str(version))
    statFilename = os.path.join(_LOG_DIR, statFilename + ".txt")
    return statFilename


def _GET_LOG_NAMES(name, run=None, rank=None, version=None):
    """Construct name and output filename for a logger.

    `name` should be the filename of the calling file.

    Returns
    -------
    logName : str,
        Name of the logging object.
    logFilename : str,
        Name of the logging output file.

    """
    # Setup name of `logger` object
    #     Remove directories and file suffixes from `name`
    logName = os.path.splitext(os.path.basename(name))[0]
    logFilename = str(logName)
    logName += "_log"
    if(rank): logName += "_rank%04d"

    # Setup name of `logger` output file
    if(run): logFilename += "_ill%d" % (run)
    if(version): logFilename += "_v%s" % (str(version))
    if(rank): logFilename += "_%04d" % (rank)
    logFilename += ".log"
    logFilename = os.path.join(_LOG_DIR, logFilename)
    logFilename = os.path.abspath(logFilename)
    return logName, logFilename


def _loadLogger(name, verbose=True, debug=False, run=None, rank=None, version=None, tofile=True):
    """Initialize a ``logging.Logger`` object for output messages.

    All processes log to output files, and the root process also outputs to `stdout`.  Constructs
    the log name based on the `name` argument, which should be the `__file__` parameter from the
    script which calls this method.

    Arguments
    ---------
    name : str
        Base (file)name from which to construct the log's name, and log filename.
    verbose : bool
        Print 'verbose' (``logging.INFO``) output to stdout.
        Overridden if ``debug == True``.
    debug : bool
        Print extremely verbose (``logging.DEBUG``) output to stdout.
        Overrides `verbose` setting.
    run : int or `None`
        Illustris run number {1,3}.  Added to log filename if provided.
    rank : int or `None`
        Rank of the current process for MPI runs.  Added to the log filename if provided.
    version : str or `None`
        Current version of script being run.  Added to the log filename if provided.
    tofile : bool
        Whether output should also be logged to a file.

    Returns
    -------
    logger : ``logging.Logger`` object
        Object for output logging.

    """
    # Get logger and log-file names
    logName, logFilename = _GET_LOG_NAMES(name, run=run, rank=rank, version=version)
    # Make sure directory exists
    zio.checkPath(logFilename)
    # Determine verbosity level
    if debug:
        strLvl = logging.DEBUG
    elif verbose:
        strLvl = logging.INFO
    else:
        strLvl = logging.WARNING

    # Turn off file-logging
    if not tofile: logFilename = None

    fileLvl = logging.DEBUG
    # Create logger
    if rank == 0 or rank is None:
        logger = zio.getLogger(logName, tofile=logFilename, fileLevel=fileLvl, strLevel=strLvl)
    else:
        logger = zio.getLogger(logName, tofile=logFilename, fileLevel=fileLvl, tostr=False)

    return logger


def _distributeSnapshots(comm):
    """Evenly distribute snapshot numbers across multiple processors.
    """
    size = comm.size
    rank = comm.rank
    mySnaps = np.arange(NUM_SNAPS)
    if size > 1:
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]

    return mySnaps


def _checkLoadSave(fname, loadsave, log):
    """See if a file exists and can be loaded, or if it needs to be reconstructed.
    """
    log.debug("_checkLoadSave()")
    log.debug(" - Checking for file '%s'" % (fname))
    data = None
    if os.path.exists(fname):
        logStr = " - File exists..."
        if not loadsave:
            logStr += " not loading it."
            log.debug(logStr)
        else:
            logStr += " loading..."
            log.debug(logStr)
            try:
                data = zio.npzToDict(fname)
            except Exception, e:
                log.warning(" - Load Failed: %s." % (str(e)))
            else:
                log.debug(" - Loaded data.")

    else:
        log.debug(" - File does not exist.")

    return data


assert BH_TYPE.IN == 0 and BH_TYPE.OUT == 1, \
    "``BH_TYPE.{IN/OUT}`` MUST be in the proper order!"

assert BH_TIME.BEFORE == 0 and BH_TIME.AFTER == 1 and BH_TIME.FIRST == 2, \
    "``BH_TIME.{BEFORE/AFTER/FIRST}`` MUST be in the proper order!"
