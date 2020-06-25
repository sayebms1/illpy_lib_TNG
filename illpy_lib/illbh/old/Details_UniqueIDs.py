"""

Create all intermediate save-files using something like:
    $ mpirun -n 64 python -m illpy_lib.illbh.Details_UniqueIDs


Objects
-------
-   Settings             - Class to contain and load both command-line and API parameters.

Functions
---------
-   main                 - Performs initialization and runs the desired internal function.
-   loadUniqueIDs        - Load the Unique BH ID numbers for a particular snapshot.
-   loadAllUniqueIDs     -

-   _calculateAllUniqueIDs - Load the Unique BH ID numbers for all snapshots; MPI enabled.
-   _mergeUnique         -
-   _saveUnique          - Create, save and return a dictionary of Unique BH Data.
-   _mergeAllUnique      -
-   _checkLog            - Create a default logging object if one is not given.

"""


import os
import numpy as np
from collections import Counter
# from mpi4py import MPI
from datetime import datetime
from argparse import ArgumentParser

import zcode.inout as zio
import zcode.math as zmath

from illpy_lib.constants import NUM_SNAPS, DTYPE
from illpy_lib.illbh import bh_constants
from illpy_lib.illbh.bh_constants import (DETAILS, _LOG_DIR, _distribute_snapshots,
                                          GET_DETAILS_UNIQUE_IDS_FILENAME, _checkLoadSave)

__version__ = '0.4'


class Settings:
    run = 1
    verbose = True
    debug = True
    func = 'loadAllUniqueIDs'

    def __init__(self, kwargs=None):
        try:
            self._ipython = __IPYTHON__
        except:
            self._ipython = False

        if kwargs is not None:
            self._parseArgs(kwargs)

    def _parseArgs(self, kwargs):
        pars = ArgumentParser()

        pars.add_argument('-r', '--run', type=int, default=self.run,
                          help="Illustris run number {1,3}.")
        pars.add_argument('-v', '--verbose', action='store_true', default=self.verbose,
                          help="Print verbose output (overidden by ``debug = True``).")
        pars.add_argument('-d', '--debug', action='store_true', default=self.debug,
                          help="Print very-verbose output (overides `verbose`).")
        pars.add_argument('-f', '--func', type=str, default=self.func,
                          help="Internal functions to execute.")

        # If coming from `ipython` session, `parse_args()` will fail.
        if not self._ipython:
            kwargs = vars(pars.parse_args())

        # Move parameters from dictionary to attributes
        if kwargs:
            for key, item in list(kwargs.items()):
                setattr(self, key, item)

        return


def main(log=None, **kwargs):
    """Standard entrypoint: performs initialization and runs the desired internal function.
    """
    sets = Settings(kwargs)

    # Initialize log
    log = _checkLog(log, run=sets.run, verbose=sets.verbose, debug=sets.debug)

    log.debug("run = %d" % (sets.run))
    log.debug("verbose = '%s'" % (sets.verbose))
    log.debug("debug = '%s'" % (sets.debug))
    log.debug("func = '%s'" % (sets.func))

    # Run Selected Functions
    # ----------------------
    log.debug(" - Running '%s'" % (sets.func))
    func = globals()[sets.func]
    func(sets=sets)

    return


def loadUniqueIDs(run, snap, rank=None, loadsave=True, log=None):
    """Load the Unique BH ID numbers for a particular snapshot.

    An appropriate filename is constructed by the `GET_DETAILS_UNIQUE_IDS_FILENAME` method.
    If ``loadsave == True``, this file is (attempted to be) loaded.  If the load fails, or if
    ``loadsave == False``, then the data is recalculated and saved to the same filename.  In either
    case a dictionary of the data is returned.

    Arguments
    ---------

    Returns
    -------

    """
    if log is None:
        # Initialize log
        log = constants._loadLogger(
            __file__, debug=True, verbose=True, run=run, rank=rank, version=__version__)
        if (rank == 0):
            print("Log filename = ", log.filename)
    log.debug("loadUniqueIDs()")

    fname = GET_DETAILS_UNIQUE_IDS_FILENAME(run, snap, __version__)
    data = _checkLoadSave(fname, loadsave, log)

    # Recalculate Unique BH IDs and Scales
    # ------------------------------------
    if data is None:
        # Load `details`
        from illpy_lib.illbh import details
        dets = details.loadBHDetails(run, snap, verbose=False)
        ndets = dets[DETAILS.NUM]
        logStr = " - Snap %d: %d Details" % (snap, ndets)
        if ndets > 0:
            ids = dets[DETAILS.IDS].astype(DTYPE.ID)
            ids_uniq = np.sort(np.array(list(set(ids))))
            scales = dets[DETAILS.SCALES].astype(DTYPE.SCALAR)
            nuniq = ids_uniq.size
            logStr += ", %d Unique" % (nuniq)
            log.debug(logStr)
            scales_uniq = np.zeros((nuniq, 2), dtype=DTYPE.SCALAR)
            # Sort by IDs, then by scales
            sind = np.lexsort((scales, ids))
            cnt = 0
            # For each Unique ID, find the first and last dets-entry scale-factor
            for ii, unid in enumerate(ids_uniq):
                #     Move through sorted IDs to the current target ID number
                while cnt < ndets-1 and ids[sind[cnt]] < unid:
                    cnt += 1
                #     If this matches, it is the first matching scale-factor
                if ids[sind[cnt]] == unid:
                    scales_uniq[ii, 0] = scales[sind[cnt]]
                #     Move to the next ID number
                #         allow `cnt` to reach `ndets` (not ``ndets-1``) in case this is last ID
                while cnt < ndets and ids[sind[cnt]] == unid:
                    cnt += 1
                #     If the previous entry matches, it is the last matching scale-factor
                if cnt > 0 and ids[sind[cnt-1]] == unid:
                    scales_uniq[ii, 1] = scales[sind[cnt-1]]
                if cnt >= ndets:
                    break
        else:
            log.debug(logStr)
            ids_uniq = np.zeros(0, dtype=DTYPE.ID)
            scales_uniq = np.zeros((0, 2), dtype=DTYPE.SCALAR)

        bads = np.where(scales_uniq == 0.0)
        if bads[0].size > 0:
            errStr = "Error: some scales still zero."
            bon = np.count_nonzero(scales_uniq)
            tot = scales_uniq.size
            frac = 1.0*bon/tot
            errStr = "%d/%d = %f Good" % (bon, tot, frac)
            errStr += "Bad index: %s, %s" % (str(bads[0]), str(bads[1]))
            log.error(errStr)
            raise RuntimeError(errStr)

        # Save data to file
        data = _saveUnique(run, snap, fname, ids_uniq, scales_uniq, log)

    return data


def loadAllUniqueIDs(run=Settings.run, loadsave=True, log=None, sets=None):
    """
    """
    if sets is not None:
        run = sets.run
    log = _checkLog(log, run=run)
    log.debug("loadAllUniqueIDs()")

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    fname = bh_constants.GET_DETAILS_ALL_UNIQUE_IDS_FILENAME(run, __version__)
    log.debug(" - Filename '%s'" % (fname))
    if os.path.exists(fname):
        log.debug(" - - File Exists.")
        if loadsave:
            log.debug(" - - Loading.")
            data = zio.npzToDict(fname)
        else:
            log.debug(" - - Not Loading.")
    else:
        log.debug(" - File does Not exist.")
        loadsave = False

    if not loadsave:
        # Calculate the unique IDs for each snapshot
        _calculateAllUniqueIDs(run, loadsave=False, log=log)
        # Merge unique IDs from each Snapshot
        if rank == 0:
            snaps, ids, scales = _mergeAllUnique(run, log)
            # Save data
            data = _saveUnique(run, snaps, fname, ids, scales, log)

    if rank == 0:
        return data
    return


def _calculateAllUniqueIDs(run=1, loadsave=True, log=None):
    """Load the Unique BH ID numbers for all snapshots; MPI enabled.

    Snapshot numbers are distributed to all of the active processes, and `loadUniqueIDs` is called
    for each snapshot.

    Arguments
    ---------

    Returns
    -------

    """
    # Load MPI Parameters
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    # Initialize log
    log = _checkLog(log, run=run)
    # Distribute snapshot numbers to different processors
    mySnaps = _distribute_snapshots(comm)

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Iterate over target snapshots
    # -----------------------------
    for snap in mySnaps:
        log.debug("snap = %03d" % (snap))
        data = loadUniqueIDs(run, snap, rank, loadsave=loadsave, log=log)
        log.debug(" - %d Unique IDs" % (data[DETAILS.NUM]))

    return


def _mergeUnique(snaps, old_ids, old_scales, new_data, log):
    """
    """
    log.debug("_mergeUnique()")

    new_snap = new_data[DETAILS.SNAP]
    new_ids = new_data[DETAILS.IDS]
    new_scales = new_data[DETAILS.SCALES]

    n_old = old_ids.size
    n_new = new_ids.size
    log.debug(" - %d so far, Snap %d with %d entries" % (n_old, new_snap, n_new))

    if np.isscalar(snaps):
        old = snaps
        snaps = n_old * [None]
        for ii in range(n_old):
            snaps[ii] = [old]

    oo = 0
    nn = 0
    for ii, (nn, ss) in enumerate(zip(new_ids, new_scales)):
        # Update iterator in old array to reach at least this ID number
        while oo < n_old-1 and old_ids[oo] < nn:
            oo += 1
        # If new ID is already in old list, add to snap-list, modify first/last scales
        if old_ids[oo] == nn:
            snaps[oo].append(new_snap)
            #     Find new extrema
            old_scales[oo] = zmath.minmax([old_scales[oo], ss])
        # If new ID not in old list, add new entry
        else:
            ins = oo
            # If we need to insert this new value as the last element,
            #    have to do the final incrementation manually
            if oo == n_old-1 and nn > old_ids[oo]:
                ins = oo+1
            old_ids = np.insert(old_ids, ins, nn, axis=0)
            old_scales = np.insert(old_scales, ins, ss, axis=0)
            snaps.insert(ins, [new_snap])
            #    Update length
            n_old += 1

    if old_ids.dtype.type is not np.uint64:
        print(("types = ", old_ids.dtype.type, new_ids.dtype.type))
        raise RuntimeError("old_ids at snap = %d is non-integral!" % (new_snap))

    if new_ids.dtype.type is not np.uint64:
        print(("types = ", old_ids.dtype.type, new_ids.dtype.type))
        raise RuntimeError("new_ids at snap = %d is non-integral!" % (new_snap))

    # Make sure things seem right
    test_ids = np.hstack([old_ids, new_ids])
    test_ids = np.array(list(set(test_ids)))
    n_test = test_ids.size
    if len(old_ids) != n_test or n_test != len(old_scales) or n_test != len(snaps):
        dups = Counter(old_ids) - Counter(test_ids)
        print("Duplicates = %s" % (str(list(dups.keys()))))
        errStr = "ERROR: num unique should be %d" % (n_test)
        errStr += "\nBut len(old_ids) = %d" % (len(old_ids))
        errStr += "\nBut len(old_scales) = %d" % (len(old_scales))
        errStr += "\nBut len(snaps) = %d" % (len(snaps))
        log.error(errStr)
        raise RuntimeError(errStr)

    return snaps, old_ids, old_scales


def _saveUnique(run, snap, fname, uids, uscales, log):
    """Create, save and return a dictionary of Unique BH Data.

    Arguments
    ---------
    run : int
        Illustris run number {1,3}
    snap : int or array_like of int
        Illustris snapshot number {1,135}
    fname : str
        Filename to save to.
    uids : (N) array_like of int
        Unique ID numbers (NOTE: `N` may be zero).
    uscales : (N,2) array_like of float
        First and last scale-factors of unique BHs (NOTE: `N` may be zero).
    log : ``logging.Logger`` object
        Logging object for output.

    Returns
    -------
    data : dict
        Input data organized into a dictionary with added metadata.

    """
    log.debug("_saveUnique()")

    data = {
        DETAILS.RUN: run,
        DETAILS.SNAP: snap,
        DETAILS.FILE: fname,
        DETAILS.VERSION: __version__,
        DETAILS.CREATED: datetime.now().ctime(),

        DETAILS.IDS: uids.astype(DTYPE.ID),
        DETAILS.SCALES: uscales.astype(DTYPE.SCALAR),
        DETAILS.NUM: uids.size,
    }

    for key, val in list(data.items()):
        data[key] = np.asarray(val)

    zio.dictToNPZ(data, fname, verbose=False, log=log)
    return data


def _mergeAllUnique(run, log):
    """
    """
    log.debug("_mergeAllUnique()")
    first = True
    newDets = None
    oldDets = None
    for ii in range(NUM_SNAPS):
        newDets = loadUniqueIDs(run, ii, None, log=log)
        if oldDets is not None:
            if first:
                ids = oldDets[DETAILS.IDS]
                scales = oldDets[DETAILS.SCALES]
                snaps = ii
                ids = np.atleast_1d(ids)
                first = False

            snaps, ids, scales = _mergeUnique(snaps, ids, scales, newDets, log)

        if newDets[DETAILS.NUM] > 0:
            oldDets = newDets

    return snaps, ids, scales


def _checkLog(log, run=None, debug=Settings.debug, verbose=Settings.verbose):
    """Create a default logging object if one is not given.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if not rank:
        zio.check_path(_LOG_DIR)
    comm.Barrier()

    if log is None:
        log = constants._loadLogger(
            __file__, debug=debug, verbose=debug, run=run, rank=rank, version=__version__)
        header = "\n%s\n%s\n%s" % (__file__, '='*len(__file__), str(datetime.now()))
        log.debug(header)
    if not rank:
        print(("Log filename = ", log.filename))

    return log


if __name__ == "__main__":
    main()
