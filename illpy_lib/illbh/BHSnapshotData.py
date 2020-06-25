"""Collect snapshot/particle data for merger BHs.

To load the data from the illustris simulation files into intermediate data products (required to
import data), Run with something like:
    $ mpirun -n 4 python -m illpy_lib.illbh.BHSnapshotData --verbose

Once the intermediate data files exist, they can be loaded via the python API as,
    >>> import illpy_lib.illbh.BHSnapshotData
    >>> bhData = illpy_lib.illbh.BHSnapshotData(1)      # Load for illustris-1
    >>> from illpy_lib.illbh.bh_constants import BH_SNAP
    >>> print(bhData[BH_SNAP.]

Functions
---------
-   main                     - Create master and many slave processes to extract BH snapshot data.
-   loadBHSnapshotData       - Load existing BH Snapshot data save file, or recreate it (slow).

-   _runMaster               - Distribute snapshots to individual slave tasks for loading.
-   _runSlave                - Receive tasks from master process, and load BH Snapshot data.
-   _loadSingleSnapshotBHs   - Load the data for BHs in a single snapshot, save to npz file.
-   _mergeBHSnapshotFiles    - Combine BH data from individual Snapshot save files.
-   _initStorage             - Initialize dictionary for BH Snapshot data.
-   _parseArguments          - Prepare argument parser and load command line arguments.

-   _GET_BH_SNAPSHOT_DIR
-   _GET_BH_SINGLE_SNAPSHOT_FILENAME
-   _GET_BH_SNAPSHOT_FILENAME

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from datetime import datetime
import os
import sys
import logging
import argparse

# from mpi4py import MPI

from illpy_lib.constants import (NUM_SNAPS, GET_ILLUSTRIS_OUTPUT_DIR, GET_PROCESSED_DIR,
                                 DTYPE, GET_BAD_SNAPS)
from . import mergers
from . import bh_constants
from .bh_constants import MERGERS, BH_TYPE, BH_SNAP, SNAPSHOT_FIELDS, SNAPSHOT_DTYPES

# import illpy_lib as ill

import zcode.inout as zio

MPI_TAGS = zio.MPI_TAGS

DEF_RUN = 1
_VERSION = 0.4

_BH_SINGLE_SNAPSHOT_FILENAME = "ill-{0:d}_snap{1:03d}_merger-bh_snapshot_{2:.2f}.npz"
_BH_SNAPSHOT_FILENAME = "ill-{0:d}_merger-bh_snapshot_v{1:.2f}.npz"


def main():
    """Create master and many slave processes to extract BH snapshot data.
    """

    # Initialize MPI Parameters
    # -------------------------
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if (size <= 1): raise RuntimeError("Not setup for serial runs!")

    if (rank == 0):
        NAME = sys.argv[0]
        print(("\n{:s}\n{:s}\n{:s}".format(NAME, '='*len(NAME), str(datetime.now()))))
        zio.check_path(bh_constants._LOG_DIR)

    # Make sure log-path is setup before continuing
    comm.Barrier()

    # Parse Arguments
    # ---------------
    args = _parseArguments()
    run = args.run
    verbose = args.verbose

    # Load logger
    logger = bh_constants._loadLogger(__file__, verbose=verbose, run=run,
                                     rank=rank, version=_VERSION)

    logger.info("run           = %d  " % (run))
    logger.info("version       = %.2f" % (_VERSION))
    logger.info("MPI comm size = %d  " % (size))
    logger.info("Rank          = %d  " % (rank))
    logger.info("")
    logger.info("verbose       = %s  " % (str(verbose)))
    logger.info("")

    # Master Process
    # --------------
    if (rank == 0):
        beg_all = datetime.now()
        try:
            logger.debug("Running Master")
            _runMaster(run, comm, logger)
        except Exception as err:
            zio._mpiError(comm, log=logger, err=err)

        end_all = datetime.now()
        logger.debug("Done after '%s'" % (str(end_all-beg_all)))

        logger.info("Merging snapshots")
        loadBHSnapshotData(run, logger=logger, loadsave=False)

    # Slave Processes
    # ---------------
    else:

        try:
            logger.debug("Running slave")
            _runSlave(run, comm, logger)
        except Exception as err:
            zio._mpiError(comm, log=logger, err=err)

        logger.debug("Done.")

    return


def loadBHSnapshotData(run, version=None, loadsave=True, verbose=False, logger=None):
    """Load an existing BH Snapshot data save file, or attempt to recreate it (slow!).

    If the data is recreated (using ``_mergeBHSnapshotFiles``), it will be saved to an npz file.
    The loaded parameters are stored to a dictionary with keys given by the parameters in
    ``bh_constants.BH_SNAP``.

    Arguments
    ---------
    run : int,
        Illustris run number {1, 3}.
    version : flt,
        Version number to load/save.
    loadsave : bool,
        If `True`, attempt to load an existing save.
    verbose : bool,
        Print verbose output.
    logger : ``logging.Logger`` object,
        Object to use for logging output messages.

    Returns
    -------
    data : dict,
        Dictionary of BH Snapshot data.  Keys are given by the entries to ``bh_constants.BH_SNAP``.

    """

    # Create default logger if needed
    # -------------------------------
    if (not isinstance(logger, logging.Logger)):
        logger = zio.default_logger(logger, verbose=verbose)

    logger.debug("BHSnapshotData.loadBHSnapshotData()")
    if (version is None): version = _VERSION

    oldVers = False
    # Warn if attempting to use an old version number
    if (version != _VERSION):
        oldVers = True
        logger.warning("WARNING: loading v%.2f behind current v%.2f" % (version, _VERSION))

    # Get save filename
    fname = _GET_BH_SNAPSHOT_FILENAME(run, version=version)

    # Load Existing File
    # ------------------
    if (loadsave):
        logger.info("Loading from '%s'" % (fname))
        if (os.path.exists(fname)):
            data = zio.npzToDict(fname)
        else:
            logger.warning("WARNING: '%s' does not exist!  Recreating!" % (fname))
            loadsave = False

    # Recreate data (Merge individual snapshot files)
    # -----------------------------------------------
    if (not loadsave):
        logger.info("Recreating '%s'" % (fname))

        # Dont allow old versions to be recreated
        if (oldVers): raise RuntimeError("Cannot recreate outdated version %.2f!!" % (version))

        data = _mergeBHSnapshotFiles(run, logger)

        # Add Metadata
        logger.debug("Adding metadata")
        data[BH_SNAP.RUN] = run
        data[BH_SNAP.VERSION] = _VERSION
        data[BH_SNAP.CREATED] = datetime.now().ctime()
        data[BH_SNAP.FIELDS] = SNAPSHOT_FIELDS
        data[BH_SNAP.DTYPES] = SNAPSHOT_DTYPES

        # Save
        logger.debug("Saving")
        zio.dictToNPZ(data, fname)
        logger.info("Saved to '%s'" % (fname))

    return data


def _runMaster(run, comm, logger):
    """Distribute snapshots and associated mrgs to individual slave tasks for loading.

    Loads ``mergers`` and distributes them, based on snapshot, to slave processes run by the
    ``_runSlave`` method.  A status file is created to track progress.  Once all snapshots are
    distributed, this method directs the termination of all of the slave processes.

    Arguments
    ---------
    run : int,
        Ollustris simulation number {1, 3}.
    comm : ``mpi4py.MPI.Intracomm`` object,
        MPI intracommunicator object, `COMM_WORLD`.
    logger : ``logging.Logger`` object,
        Object for logging.

    """

    from mpi4py import MPI
    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    logger.info("BHSnapshotData._runMaster()")
    logger.debug("Rank %d/%d" % (rank, size))

    # Make sure output directory exists
    fname = _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, 0)
    zio.check_path(fname)

    # Load BH Mergers
    logger.info("Loading BH Mergers")
    mrgs = mergers.load_fixed_mergers(run, loadsave=True, verbose=False)
    numMergers = mrgs[MERGERS.NUM]
    logger.debug("- Loaded %d mrgs" % (numMergers))

    # Init status file
    statFileName = bh_constants._GET_STATUS_FILENAME(__file__, run=run, version=_VERSION)
    statFile = open(statFileName, 'w')
    logger.debug("Opened status file '%s'" % (statFileName))
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    num_pos = 0
    num_neg = 0
    num_new = 0
    countDone = 0
    count = 0
    times = np.zeros(NUM_SNAPS-1)

    # Iterate Over Snapshots
    # ----------------------
    #     Go over snapshots in random order to get a better estimate of ETA/duration
    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)
    logger.info("Iterating over snapshots")
    pbar = zio.getProgressBar(NUM_SNAPS-1)
    for snapNum in snapList:
        logger.debug("- Snap %d, count %d, done %d" % (snapNum, count, countDone))

        # Get Mergers occuring just after Snapshot `snapNum`
        mrgs = mrgs[MERGERS.MAP_STOM][snapNum+1]
        nums = len(mrgs)
        targetIDs = mrgs[MERGERS.IDS][mrgs]
        logger.debug("- %d Mergers from snapshot %d" % (nums, snapNum+1))

        # Look for available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()
        logger.debug("- Received signal from %d" % (src))

        # Track number of completed profiles
        if (tag == MPI_TAGS.DONE):
            durat, pos, neg, new = data
            logger.debug("- - Done after %s, pos %d, neg %d, new %d" % (durat, pos, neg, new))

            times[countDone] = durat
            num_pos += pos
            num_neg += neg
            num_new += new
            countDone += 1

        # Distribute tasks
        logger.debug("- Sending new task to %d" % (src))
        comm.send([snapNum, mrgs, targetIDs, numMergers], dest=src, tag=MPI_TAGS.START)
        logger.debug("- New task sent")

        # Write status to file and log
        dur = (datetime.now()-beg)
        fracDone = 1.0*countDone/(NUM_SNAPS-1)
        statStr = 'Snap %3d (rank %03d)   %8d/%8d = %.4f  in %s  %8d pos  %8d neg  %3d new\n' % \
            (snapNum, src, countDone, NUM_SNAPS-1, fracDone, str(dur), num_pos, num_neg, num_new)
        statFile.write(statStr)
        statFile.flush()
        logger.debug(statStr)
        count += 1
        pbar.update(count)

    statFile.write('\n\nDone after %s' % (str(datetime.now()-beg)))
    statFile.close()
    pbar.finish()

    # Close out all Processes
    # -----------------------
    numActive = size-1
    logger.info("Exiting %d active processes" % (numActive))
    while(numActive > 0):
        # Find available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()
        logger.debug("- Received signal from %d" % (src))

        # If we're recieving exit confirmation, count it
        if (tag == MPI_TAGS.EXIT): numActive -= 1
        else:
            # If a process just completed, count it
            if (tag == MPI_TAGS.DONE):
                durat, pos, neg, new = data
                logger.debug("- - %d Done after %s, pos %d, neg %d, new %d" %
                             (src, durat, pos, neg, new))
                times[countDone] = durat
                countDone += 1
                num_pos += pos
                num_neg += neg
                num_new += new

            # Send exit command
            logger.debug("Sending exit to %d.  %d Active." % (src, numActive))
            comm.send(None, dest=src, tag=MPI_TAGS.EXIT)

    fracDone = 1.0*countDone/(NUM_SNAPS-1)
    logger.debug("%d/%d = %.4f Completed tasks!" % (countDone, NUM_SNAPS-1, fracDone))
    logger.debug("Average time %.4f +- %.4f" % (np.average(times), np.std(times)))
    logger.info("Totals: pos = %5d   neg = %5d   new = %3d" % (num_pos, num_neg, num_new))

    return


def _runSlave(run, comm, logger, loadsave=True):
    """Receive snapshots and associatd mrgs from the master process.  Loads BH Snapshot data.

    This method receives task parameters from the ``_runMaster`` process, and uses the
    ``_loadSingleSnapshotBHs`` method to load and save individual snapshot data.  Once this method
    receives the exit signal, it terminates.

    Arguments
    ---------
    run : int,
        Illustris simulation run number {1, 3}.
    comm : ``mpi4py.MPI.Intracomm`` object,
        MPI intracommunicator object, `COMM_WORLD`.
    logger : ``logging.Logger`` object,
        Object for logging.
    loadsave : bool,
        Load data for this subhalo if it already exists.

    Details
    -------
    -    Waits for ``master`` process to send subhalo numbers
    -    Returns status to ``master``

    """

    from mpi4py import MPI
    stat = MPI.Status()
    rank = comm.rank
    size = comm.size
    numReady = 0
    data = {}

    logger.info("BHSnapshotData._runSlave()")
    logger.debug("Rank %d/%d" % (rank, size))

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        logger.debug("Sending ready %d" % (numReady))
        comm.send(None, dest=0, tag=MPI_TAGS.READY)
        # Receive ``task`` ([snap, idxs, bhids, numMergers])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()
        logger.debug("- Received tag %d" % (tag))

        if (tag == MPI_TAGS.START):
            # Extract parameters
            snap, idxs, bhids, numMergers = task
            logger.debug("- Starting snapshot %d" % (snap))
            beg = datetime.now()

            # Load and save snapshot
            data, pos, neg, new = _loadSingleSnapshotBHs(run, snap, numMergers, idxs, bhids,
                                                         logger, rank=rank, loadsave=loadsave)

            end = datetime.now()
            durat = (end-beg).total_seconds()
            logger.debug("- Done after %f,  pos %d, neg %d, new %d" % (durat, pos, neg, new))
            comm.send([durat, pos, neg, new], dest=0, tag=MPI_TAGS.DONE)
        elif (tag == MPI_TAGS.EXIT):
            logger.debug("- Received Exit.")
            break

        numReady += 1

    # Finish, return done
    logger.debug("Sending Exit.")
    comm.send(None, dest=0, tag=MPI_TAGS.EXIT)

    return


def _loadSingleSnapshotBHs(run, snapNum, numMergers, idxs, bhids,
                           logger, rank=0, loadsave=True):
    """Load the data for BHs in a single snapshot, save to npz file.

    If no indices (``idxs``) or BH IDs (``bhids``) are given, or this is a 'bad' snapshot,
    then it isn't actually loaded and processed.  An NPZ file with all zero entries is still
    produced.

    Arguments
    ---------
    run : int,
        Illustris simulation number {1, 3}.
    snapNum : int,
        Illustris snapshot number {1, 135}.
    numMergers : int,
        Total number of mrgs.
    idxs : (N,) array of int,
        Merger indices for this snapshot with `N` mrgs.
    bhids : (N,2,) array of int,
        BH ID numbers, `IN` and `OUT` BH for each merger.
    logger : ``logging.Logger`` object,
        Object for logging.
    rank : int
        Rank of this processor (used for logging).
    loadsave : bool,
        Load existing save if it exists.

    Returns
    -------
    data : dict,
        Data for this snapshot.
    pos : int,
        Number of BHs successfully found.
    neg : int,
        Number of BHs failed to be found.
    new : int,
        `+1` if a new file was created, otherwise `0`.

    """
    import illpy as ill

    logger.warning("BHSnapshotData._loadSingleSnapshotBHs()")

    illdir = GET_ILLUSTRIS_OUTPUT_DIR(run)
    fname = _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snapNum)
    logger.warning("Snap %d, filename '%s'" % (snapNum, fname))

    pos = 0
    neg = 0
    new = 0

    # Load and Return existing save if desired
    # ----------------------------------------
    if (loadsave and os.path.exists(fname)):
        logger.warning("Loading existing file")
        data = zio.npzToDict(fname)
        return data, pos, neg, new

    # Initialize dictionary of results
    # --------------------------------
    logger.info("Initializing storage")
    data = _initStorage(numMergers)
    for index, tid in zip(idxs, bhids):
        for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
            data[BH_SNAP.TARGET][index, BH] = tid[BH]

    # Decide if this is a valid Snapshot
    # ----------------------------------
    process_snapshot = True
    # Some illustris-1 snapshots are bad
    if (snapNum in GET_BAD_SNAPS(run)):
        logger.warning("Skipping bad snapshot.")
        process_snapshot = False

    # Make sure there are mrgs in this snapshot
    if (len(idxs) <= 0 or len(bhids) <= 0):
        logger.warning("Skipping snap %d with no valid BHs" % (snapNum))
        process_snapshot = False

    # Load And Process Snapshot if its good
    # -------------------------------------
    if (process_snapshot):
        logger.info("Processing Snapshot")

        # Load Snapshot
        # -------------
        logger.debug("- Loading snapshot %d" % (snapNum))
        with zio.StreamCapture() as output:
            snapshot = ill.snapshot.loadSubset(illdir, snapNum, 'bh', fields=SNAPSHOT_FIELDS)

        snap_keys = list(snapshot.keys())
        if ('count' in snap_keys):
            snap_keys.remove('count')
            logger.debug("- - Loaded %d particles" % (snapshot['count']))

        # Make sure all target keys are present
        union = list(set(snap_keys) & set(SNAPSHOT_FIELDS))
        if (len(union) != len(SNAPSHOT_FIELDS)):
            logger.error("snap_keys       = '%s'" % (str(snap_keys)))
            logger.error("SNAPSHOT_FIELDS = '%s'" % (str(SNAPSHOT_FIELDS)))
            errStr = "Field mismatch at Rank %d, Snap %d!" % (rank, snapNum)
            zio._mpiError(comm, log=logger, err=errStr)

        # Match target BHs
        # ----------------
        logger.debug("- Matching %d BH Mergers" % (len(bhids)))
        for index, tid in zip(idxs, bhids):
            for BH in [BH_TYPE.IN, BH_TYPE.OUT]:
                ind = np.where(snapshot['ParticleIDs'] == tid[BH])[0]
                if (len(ind) == 1):
                    pos += 1
                    data[BH_SNAP.VALID][index, BH] = True
                    for key in SNAPSHOT_FIELDS: data[key][index, BH] = snapshot[key][ind[0]]
                else:
                    neg += 1

        logger.debug("- Processed, pos %d, neg %d" % (pos, neg))

    # Add Metadata and Save File
    #  ==========================
    logger.debug("Adding metadata to dictionary")
    data[BH_SNAP.RUN]     = run
    data[BH_SNAP.SNAP]    = snapNum
    data[BH_SNAP.VERSION] = _VERSION
    data[BH_SNAP.CREATED] = datetime.now().ctime()
    data[BH_SNAP.DIR_SRC] = illdir
    data[BH_SNAP.FIELDS] = SNAPSHOT_FIELDS
    data[BH_SNAP.DTYPES] = SNAPSHOT_DTYPES

    logger.info("Saving data to '%s'" % (fname))
    zio.dictToNPZ(data, fname)
    new = 1

    return data, pos, neg, new


def _mergeBHSnapshotFiles(run, logger):
    """Combine valid BH data from individual BH Snapshot save files into a single dictionary.

    Individual snapshot files are loaded using ``_loadSingleSnapshotBHs``.  If a file is missing,
    it will be created.  If this is happenening on a single processor, it will be very slow.

    Arguments
    ---------
        run    <int> : illustris run number {1, 3}
        logger <obj> : ``logging`` logger object

    Returns
    -------
        allData <obj> : dictionary of all BH Snapshot data

    """

    logger.info("BHSnapshotData._mergeBHSnapshotFiles()")

    # Randomize order of snapshots
    snapList = np.arange(NUM_SNAPS-1)
    np.random.shuffle(snapList)

    # Initialize variables
    count = 0
    newFiles = 0
    oldFiles = 0
    num_pos = 0
    num_neg = 0
    num_val = 0
    num_tar = 0

    # Load BH Mergers
    # ---------------
    logger.info("Loading BH Mergers")
    mrgs = mergers.load_fixed_mergers(run, loadsave=True)
    numMergers = mrgs[MERGERS.NUM]
    logger.debug("Loaded %d mrgs" % (numMergers))

    # Initialize storage
    allData = _initStorage(numMergers)

    # Load each snapshot file
    # -----------------------
    logger.warning("Iterating over snapshots")
    beg = datetime.now()
    pbar = zio.getProgressBar(NUM_SNAPS-1)
    for snap in snapList:

        # Get mrgs for just after this snapshot (hence '+1')
        mrgs = mrgs[MERGERS.MAP_STOM][snap+1]
        nums = len(mrgs)
        targetIDs = mrgs[MERGERS.IDS][mrgs]
        logger.debug("- Snap %d, Count %d.  %d Mergers" % (snap, count, nums))

        # Load Snapshot Data
        logger.debug("- Loading single snapshot")
        data, pos, neg, new = _loadSingleSnapshotBHs(run, snap, numMergers, mrgs, targetIDs, logger,
                                                     loadsave=True)
        logger.debug("- - pos %d, neg %d, new %d" % (pos, neg, new))

        # Store to dictionary
        valids     = data[BH_SNAP.VALID]
        numValid   = np.count_nonzero(valids)
        numTargets = np.count_nonzero(data[BH_SNAP.TARGET] > 0)

        # Copy valid elements
        allData[BH_SNAP.TARGET][valids] = data[BH_SNAP.TARGET][valids]
        allData[BH_SNAP.VALID][valids] = data[BH_SNAP.VALID][valids]
        for key in SNAPSHOT_FIELDS:
            allData[key][valids] = data[key][valids]

        # Collect and log data
        if (new == 1):
            newFiles += 1
            logger.debug("- - New")
            logger.debug("- - - pos %d, neg %d, expected %d" % (pos, neg, pos+neg, nums))
            logger.debug("- - - Targets %d, Valid %d" % (numTargets, numValid))
        else:
            oldFiles += 1
            pos = numValid
            neg = 2*nums - pos
            logger.debug("- - Old")
            logger.debug("- - - pos %d, expected %d, neg %d" % (pos, nums, neg))
            logger.debug("- - - Targets %d, Valid %d" % (numTargets, numValid))

        # Increment tracking data
        num_pos += pos
        num_neg += neg
        num_val += numValid
        num_tar += numTargets
        count += 1
        pbar.update(count)

    pbar.finish()
    end = datetime.now()

    logger.info("Done after %s" % (str(end-beg)))
    logger.info("%d new, %d old.  Pos %d, Neg %d" % (newFiles, oldFiles, num_pos, num_neg))
    logger.info("Targets %d, Valid %d" % (num_tar, num_val))
    numValid = np.count_nonzero(allData[BH_SNAP.VALID])
    logger.info("%d/%d = %.4f valid" % (numValid, 2*numMergers, 0.5*numValid/numMergers))

    return allData


def _initStorage(numMergers):
    """Initialize dictionary for BH Snapshot data.
    """
    data = {}
    for key, typ in zip(SNAPSHOT_FIELDS, SNAPSHOT_DTYPES):
        data[key] = np.zeros([numMergers, 2], dtype=np.dtype(typ))

    data[BH_SNAP.VALID]  = np.zeros([numMergers, 2], dtype=bool)
    data[BH_SNAP.TARGET] = np.zeros([numMergers, 2], dtype=DTYPE.ID)
    return data


def _parseArguments():
    """Prepare argument parser and load command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%s %.2f' % (sys.argv[0], _VERSION))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output', default=False)
    parser.add_argument("run", type=int, nargs='?', choices=[1, 2, 3],
                        help="illustris simulation number", default=DEF_RUN)
    args = parser.parse_args()
    return args


def _GET_BH_SNAPSHOT_DIR(run):
    return GET_PROCESSED_DIR(run) + "blackhole_particles/"


def _GET_BH_SINGLE_SNAPSHOT_FILENAME(run, snap, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SINGLE_SNAPSHOT_FILENAME.format(run, snap, version)


def _GET_BH_SNAPSHOT_FILENAME(run, version=_VERSION):
    return _GET_BH_SNAPSHOT_DIR(run) + _BH_SNAPSHOT_FILENAME.format(run, version)


if (__name__ == "__main__"):
    main()
