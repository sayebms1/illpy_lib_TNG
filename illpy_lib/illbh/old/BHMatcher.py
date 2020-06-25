"""
"""

import os
import logging
import numpy as np
from datetime import datetime

from illpy_lib.illbh import bh_constants

from illpy_lib.illbh.Details_UniqueIDs import loadAllUniqueIDs
from illpy_lib.illbh.bh_constants import MERGERS, DETAILS, BH_TREE, _LOG_DIR, BH_TYPE, \
    GET_MERGER_DETAILS_FILENAME, GET_REMNANT_DETAILS_FILENAME, _MAX_DETAILS_PER_SNAP, \
    _distribute_snapshots, GET_BLACKHOLE_TREE_DETAILS_FILENAME
from illpy_lib.constants import DTYPE, NUM_SNAPS

import zcode.inout as zio
import zcode.math as zmath

__version__ = '0.25'
_GET_KEYS = [DETAILS.SCALES, DETAILS.MASSES, DETAILS.MDOTS, DETAILS.RHOS, DETAILS.CS]


def main(run=1, verbose=True, debug=True, loadsave=True, redo_mergers=False, redo_remnants=True):
    from mpi4py import MPI

    # Initialization
    # --------------
    #     MPI Parameters
    comm = MPI.COMM_WORLD
    rank = comm.rank
    name = __file__
    header = "\n%s\n%s\n%s" % (name, '='*len(name), str(datetime.now()))

    if (rank == 0):
        zio.check_path(_LOG_DIR)
    comm.Barrier()

    # Initialize log
    log = bh_constants._loadLogger(
        __file__, debug=debug, verbose=verbose, run=run, rank=rank, version=__version__)
    log.debug(header)
    if (rank == 0):
        print("Log filename = ", log.filename)

    # 336,
    # TARGET = 336
    # if rank == 0: log.warning("Running 'allDetailsForBHLineage(run, %d, log)'" % (TARGET))
    # allDetailsForBHLineage(run, TARGET, log, reload=True)
    # return

    # Check status of files, determine what operations to perform
    create_mergerDets = False
    create_remnantDets = False
    if (rank == 0):
        # Check merger dets status
        mergerDetFName = GET_MERGER_DETAILS_FILENAME(
            run, __version__, _MAX_DETAILS_PER_SNAP)
        log.info("Merger Details file: '%s'" % (mergerDetFName))
        if (not os.path.exists(mergerDetFName)):
            log.warning(" - Merger dets file does not exist.")
            create_mergerDets = True
        else:
            log.info("Merger Details file exists.")
            if (not loadsave or redo_mergers):
                log.info(" - Recreating anyway.")
                create_mergerDets = True

        # Check remnants dets status
        remnantDetFName = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
        log.info("Remnant Details file: '%s'" % (remnantDetFName))

        if (not os.path.exists(remnantDetFName)):
            log.warning(" - Remnant Details file does not exist.")
            create_remnantDets = True
        else:
            log.info("Remnant Details file exists.")
            if (not loadsave or redo_remnants):
                log.info(" - Recreating anyway.")
                create_remnantDets = True

    # Synchronize control-flow flags
    create_mergerDets = comm.bcast(create_mergerDets, root=0)
    create_remnantDets = comm.bcast(create_remnantDets, root=0)
    comm.Barrier()
    mdets = None

    # Match Merger BHs to Details entries
    # -----------------------------------
    if create_mergerDets:
        log.info("Creating Merger Details")
        beg = datetime.now()
        mdets = _matchMergerDetails(run, log)
        end = datetime.now()
        log.debug(" - Done after %s" % (str(end - beg)))

    # Extend merger-dets to account for later mrgs
    # --------------------------------------------------
    if create_remnantDets:
        comm.Barrier()
        log.info("Creating Remnant Details")
        beg = datetime.now()
        _matchRemnantDetails(run, log, mdets=mdets)
        end = datetime.now()
        log.debug(" - Done after %s" % (str(end - beg)))

    return


def loadMergerDetails(run, verbose=True, log=None):
    """Load a previously calculated merger-dets file.

    Arguments
    ---------
    run : int
        Illustris run numer {1,3}.
    verbose : bool
        Should verbose output be produced if a log is created.
        Only applies if no `log` is provided.
    log : ``logging.Logger`` object or `None`
        Logging object for output.
        If provided, the `verbose` parameter is ignored.

    Returns
    -------
    megerDets : dict
        Dictionary of merger-dets data.

    """
    if (log is None):
        log = bh_constants._loadLogger(
            __file__, verbose=verbose, debug=False, run=run, tofile=False)

    log.debug("loadMergerDetails()")
    mergerDetFName = GET_MERGER_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    logStr = "Merger Details File '%s'" % (mergerDetFName)
    mergerDets = None
    if os.path.exists(mergerDetFName):
        mergerDets = zio.npzToDict(mergerDetFName)
        logStr += " loaded."
        log.info(logStr)
    else:
        logStr += " does not exist."
        log.warning(logStr)

    return mergerDets


def loadRemnantDetails(run, verbose=True, log=None):
    """Load a previously calculated remnant-dets file.

    Arguments
    ---------
    run : int
        Illustris run numer {1,3}.
    verbose : bool
        Should verbose output be produced if a log is created.
        Only applies if no `log` is provided.
    log : ``logging.Logger`` object or `None`
        Logging object for output.
        If provided, the `verbose` parameter is ignored.

    Returns
    -------
    remnantDets : dict
        Dictionary of remnant-dets data.

    """
    if (log is None):
        log = bh_constants._loadLogger(
            __file__, verbose=verbose, debug=False, run=run, tofile=False)

    log.debug("loadRemnantDetails()")
    remnantDetFName = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    log.info("Remnant Details File '%s'" % (remnantDetFName))
    remnantDets = None
    if os.path.exists(remnantDetFName):
        remnantDets = zio.npzToDict(remnantDetFName)
        log.info(" - File Loaded.")
    else:
        log.warning(" - File '%s' Does not exist!" % (remnantDetFName))

    return remnantDets


def allDetailsForBHLineage(run, mrg, log, reload=False):
    """Load all of the dets entries for a given BH lineage (merger tree).

    Arguments
    ---------
    run : int
    mrg : int
    log : ``logging.Logger`` object
    reload : bool
        Even if the results file exists, recalculate them.

    """
    log.debug("allDetailsForBHLineage()")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    bhIDs = None
    fname = None
    log.debug(" - Rank %d/%d." % (rank, size))
    if rank == 0:
        # Get all Unique ID numbers
        log.debug(" - Loading All Unique IDs")
        unique = loadAllUniqueIDs(run, log=log)
        log.debug(" - Loaded %d unique ID numbers" % (len(unique[DETAILS.IDS])))
        # get the final merger number, the unique BH IDs, and the merger indices of this tree
        from illpy_lib.illbh import BHTree
        finMerger, bhIDs, mrgInds = BHTree.allIDsForTree(run, mrg)
        # Construct the appropriate file-name
        fname = GET_BLACKHOLE_TREE_DETAILS_FILENAME(run, finMerger, __version__)
        log.debug(" - Merger %d ==> Final merger %d, filename: '%s'" % (mrg, finMerger, fname))
        if os.path.exists(fname):
            if size == 1:
                log.debug(" - File exists, loading.")
                data = zio.npzToDict(fname)
                return data
            elif not reload:
                raise RuntimeError("WE SHOULD NOT GET HERE!")

        bhIDs = np.array(bhIDs)
        numBHs = bhIDs.size
        log.info(" - Merger {} has a tree with {} unique BHs".format(mrg, numBHs))
        if numBHs < 2:
            errStr = "ERROR: only IDs found for merger {} are : {}".format(mrg, str(bhIDs))
            log.error(errStr)
            raise RuntimeError(errStr)

    # Distribute snapshots to each processor
    log.debug(" - Barrier.")
    comm.Barrier()
    mySnaps = _distribute_snapshots(comm)
    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Get dets entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    #    Use ``maxPerSnap = None`` to save ALL dets entries
    nums, scales, masses, mdots, dens, csnds, ids = \
        _detailsForMergers_snapshots(run, mySnaps, bhIDs, None, log)

    # Collect results and organize
    # ----------------------------
    if (size > 1):
        log.debug(" - Gathering")
        beg = datetime.now()
        # Gather results from each processor into ``rank=0``
        tempScales = comm.gather(scales, root=0)
        tempMasses = comm.gather(masses, root=0)
        tempMdots = comm.gather(mdots, root=0)
        tempDens = comm.gather(dens, root=0)
        tempCsnds = comm.gather(csnds, root=0)
        tempIds = comm.gather(ids, root=0)
        tempNums = comm.gather(nums, root=0)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Gather snapshot numbers for ordering
        mySnaps = comm.gather(mySnaps, root=0)

        # Organize results appropriately
        if (rank == 0):
            log.debug(" - Stacking")
            beg = datetime.now()

            foundBH = np.zeros(numBHs, dtype=bool)

            nums = np.zeros(numBHs)
            scales = numBHs*[[None]]
            masses = numBHs*[[None]]
            mdots = numBHs*[[None]]
            dens = numBHs*[[None]]
            csnds = numBHs*[[None]]
            ids = numBHs*[[None]]

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(bhIDs):
                for jj in range(size):
                    errStr = ""
                    if tempIds[jj][ii][0] is not None:
                        foundBH[ii] = True
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the dets IDs are consistent
                        if np.any(tempIds[jj][ii] != dd):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}".format(
                                ii, jj, mm, dd)
                            errStr += " tempIds = {}".format(str(tempIds[ii]))

                        # Make sure dets IDs match expected merger ID
                        if dd != mm:
                            errStr += "\nii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)

                        # If no entries have been stored yet, replace with first entries
                        if len(ids[ii]) == 1 and ids[ii][0] is None:
                            ids[ii] = tempIds[jj][ii]
                            scales[ii] = tempScales[jj][ii]
                            masses[ii] = tempMasses[jj][ii]
                            mdots[ii] = tempMdots[jj][ii]
                            dens[ii] = tempDens[jj][ii]
                            csnds[ii] = tempCsnds[jj][ii]
                        # If entries already exist, append new ones
                        else:
                            # Double check that all existing IDs are consistent with new ones
                            #    This should be redundant, but whatevs
                            if np.any(ids[ii] != dd):
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}"
                                errStr = errStr.format(ii, jj, mm, dd, str(ids))
                            ids[ii] = np.append(ids[ii], tempIds[jj][ii])
                            scales[ii] = np.append(scales[ii], tempScales[jj][ii])
                            masses[ii] = np.append(masses[ii], tempMasses[jj][ii])
                            mdots[ii] = np.append(mdots[ii], tempMdots[jj][ii])
                            dens[ii] = np.append(dens[ii], tempDens[jj][ii])
                            csnds[ii] = np.append(csnds[ii], tempCsnds[jj][ii])

                        # Count the number of entries for each BH from each processor
                        nums[ii] += tempNums[jj][ii]

                    if len(errStr) > 0:
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

                if not foundBH[ii]:
                    ind = np.squeeze(np.where(mm == unique[DETAILS.IDS])[0])
                    if len(ind) > 0:
                        inSnaps = unique[DETAILS.SNAP][ind]
                        aveSnap = np.int(np.floor(np.mean(inSnaps)))
                        errStr = "%d, bhID %d : not found, but exists at %d in unique list."
                        errStr = errStr % (ii, mm, ind)
                        errStr += "\nShould be in Snaps: %s" % (inSnaps)
                        inProc = None
                        for jj in range(size):
                            if aveSnap in mySnaps[jj]:
                                inproc = jj
                                break

                        errStr += "\nShould have been found in Processor %s" % (str(inproc))
                        log.error(errStr)

            # Merge lists of snapshots, and look for any missing
            flatSnaps = np.hstack(mySnaps)
            log.debug("Obtained %d Snapshots" % (flatSnaps.size))
            missingSnaps = []
            for ii in range(NUM_SNAPS):
                if (ii not in flatSnaps):
                    missingSnaps.append(ii)

            if (len(missingSnaps) > 0):
                log.warning("WARNING: snaps %s not in results!" % (str(missingSnaps)))

            log.debug("Total entries stored = %d" % (np.sum([np.sum(nn) for nn in nums])))
            bon = np.count_nonzero(foundBH)
            tot = foundBH.size
            frac = bon/tot
            log.debug(" - %d/%d = %.4f BHs Found." % (bon, tot, frac))

    # Sort results by time
    if (rank == 0):

        # Sort entries for each BH by scalefactor
        log.debug(" - Sorting")
        beg = datetime.now()
        for ii in range(numBHs):
            if nums[ii] == 0: continue
            # Check ID numbers yet again
            if not np.all(ids[ii] == ids[ii][0]):
                errStr = "Error!  ii = {}, ID = {}, IDs = {}"
                errStr = errStr.format(ii, ids[ii][0], ids[ii])
                log.error(errStr)
                zio.mpiError(comm, log=log, err=errStr)

            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Save data
        data = _saveDetails(fname, run, ids, scales, masses, dens, mdots, csnds, log,
                            target_ids=bhIDs, target_mergers=mrgInds, final_merger=finMerger)

    return


def infer_merger_out_masses(run, mrgs=None, mdets=None, log=None):
    """Based on 'merger' and 'dets' information, infer the 'out' BH masses at time of mrgs.

    The Illustris 'merger' files have the incorrect values output for the 'out' BH mass.  This
    method uses the data included in 'dets' entries (via ``details``), which were matched
    to mrgs here in ``BHMatcher``, to infer the approximate mass of the 'out' BH at the
    time of merger.

    The ``mergerDetails`` entries have the dets for both 'in' and 'out' BHs both 'before' and
    'after' merger.  First the 'out' BH's entries just 'before' merger are used directly as the
    'inferred' mass.  In the few cases where these entries don't exist, the code falls back on
    calculating the difference between the total mass (given by the 'out' - 'after' entry) and the
    'in' BH mass recorded by the merger event (which should be correct); if this still doesn't
    work (which shouldn't ever happen), then the difference between the 'out'-'after' mass and the
    'in'-'before' mass is used, which should also be good --- but slightly less accurate (because
    the 'in'-'before' mass might have been recorded some small period of time before the actual
    merger event.

    Arguments
    ---------
       run     <int>  :
       mrgs <dict> :
       mdets   <dict> :
       verbose <str>  :
       debug   <str>  :

    Returns
    -------
       outMass <flt64>[N] : inferred 'out' BH masses at the time of merger

    """
    if (log is None):
        log = bh_constants._loadLogger(
            __file__, verbose=True, debug=False, run=run, tofile=False)

    log.debug("infer_merger_out_masses()")

    # Load Mergers
    if (mrgs is None):
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run, verbose=False)
    m_scales = mrgs[MERGERS.SCALES]
    m_masses = mrgs[MERGERS.MASSES]
    numMergers = mrgs[MERGERS.NUM]
    del mrgs

    # Load Merger Details
    if (mdets is None):
        mdets = loadMergerDetails(run, log=log)
    d_masses = mdets[DETAILS.MASSES]
    d_scales = mdets[DETAILS.SCALES]
    del mdets

    # Find dets entries before and after merger
    massBef = np.zeros_like(m_masses)
    massAft = np.zeros_like(m_masses)
    for ii, sc in enumerate(m_scales):
        for bh in [BH_TYPE.IN, BH_TYPE.OUT]:
            if (d_scales[ii].size == 0):
                log.warning("Merger %s with zero dets entries" % str(ii))
            else:
                # 'before' is ``sc > d_scales``
                bef = _indBefAft(sc - d_scales[ii, bh])
                if (bef is not None):
                    if (np.isfinite(d_masses[ii, bh][bef])):
                        massBef[ii, bh] = d_masses[ii, bh][bef]
                    elif (bef > 0):
                        massBef[ii, bh] = d_masses[ii, bh][bef-1]

                # 'after' is ``d_scales > sc``
                aft = _indBefAft(d_scales[ii, bh] - sc)
                if (aft):
                    massAft[ii, bh] = d_masses[ii, bh][aft]

    # Fix Mass Entries
    # ----------------
    massBef = massBef.reshape(2*numMergers)
    massAft = massAft.reshape(2*numMergers)
    masses = np.zeros_like(massBef)
    ntot = masses.size

    bads = np.where(np.isfinite(massBef) == False)[0]
    print(("Bads Before = ", bads))
    print(("\t", massBef[bads]))
    bads = np.where(np.isfinite(massAft) == False)[0]
    print(("Bads After = ", bads))
    print(("\t", massAft[bads]))

    # Use 'before' masses
    inds = np.where(massBef > 0.0)
    masses[inds] = massBef[inds]
    bads = np.where(masses == 0.0)
    nfix = np.size(inds)
    nbad = np.size(bads)
    frac = nfix/ntot
    log.info(" - Used %d/%d = %.4f after masses.  %d remain" % (nfix, ntot, frac, nbad))

    massBef = massBef.reshape(numMergers, 2)
    massAft = massAft.reshape(numMergers, 2)
    masses = masses.reshape(numMergers, 2)

    return masses


def _matchMergerDetails(run, log):
    """Find dets entries matching merger-BH ID numbers.

    Stores at most ``_MAX_DETAILS_PER_SNAP`` entries per snapshot for each BH.
    """
    log.debug("_matchMergerDetails()")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    data = None

    # Load Unique ID numbers to distribute to all tasks
    bhIDsUnique = None
    if rank == 0:
        # Load Mergers
        log.debug("Loading Mergers")
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)
        numMergers = mrgs[MERGERS.NUM]
        mergerIDs = mrgs[MERGERS.IDS]

        # Find unique BH IDs from set of merger BHs
        #    these are now 1D and sorted
        bhIDsUnique, reconInds = np.unique(mergerIDs, return_inverse=True)
        numUnique = np.size(bhIDsUnique)
        numTotal = np.size(reconInds)
        frac = numUnique/numTotal
        log.debug(" - %d/%d = %.4f Unique BH IDs" % (numUnique, numTotal, frac))

    # Send unique IDs to all processors
    bhIDsUnique = comm.bcast(bhIDsUnique, root=0)

    # Distribute snapshots to each processor
    mySnaps = _distribute_snapshots(comm)

    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, mySnaps.size, mySnaps.min(), mySnaps.max()))

    # Get dets entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    nums, scales, masses, mdots, dens, csnds, ids = \
        _detailsForMergers_snapshots(run, mySnaps, bhIDsUnique, _MAX_DETAILS_PER_SNAP, log)

    # Collect results and organize
    # ----------------------------
    if size > 1:
        log.debug(" - Gathering")
        beg = datetime.now()
        # Gather results from each processor into ``rank=0``
        tempScales = comm.gather(scales, root=0)
        tempMasses = comm.gather(masses, root=0)
        tempMdots = comm.gather(mdots, root=0)
        tempDens = comm.gather(dens, root=0)
        tempCsnds = comm.gather(csnds, root=0)
        tempIds = comm.gather(ids, root=0)
        tempNums = comm.gather(nums, root=0)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Gather snapshot numbers for ordering
        mySnaps = comm.gather(mySnaps, root=0)

        # Organize results appropriately
        if rank == 0:
            log.debug(" - Stacking")
            beg = datetime.now()

            nums = np.zeros(numUnique)
            scales = numUnique*[[None]]
            masses = numUnique*[[None]]
            mdots = numUnique*[[None]]
            dens = numUnique*[[None]]
            csnds = numUnique*[[None]]
            ids = numUnique*[[None]]

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(bhIDsUnique):
                for jj in range(size):
                    errStr = ""
                    if tempIds[jj][ii][0] is not None:
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the dets IDs are consistent
                        if np.any(tempIds[jj][ii] != dd):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}"
                            errStr = errStr.format(ii, jj, mm, dd)
                            errStr += " tempIds = {}".format(str(tempIds[ii]))

                        # Make sure dets IDs match expected merger ID
                        if dd != mm:
                            errStr += "\nii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)

                        # If no entries have been stored yet, replace with first entries
                        if ids[ii][0] is None:
                            ids[ii] = tempIds[jj][ii]
                            scales[ii] = tempScales[jj][ii]
                            masses[ii] = tempMasses[jj][ii]
                            mdots[ii] = tempMdots[jj][ii]
                            dens[ii] = tempDens[jj][ii]
                            csnds[ii] = tempCsnds[jj][ii]
                        # If entries already exist, append new ones
                        else:
                            # Double check that all existing IDs are consistent with new ones
                            #    This should be redundant, but whatevs
                            if (np.any(ids[ii] != dd)):
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}"
                                errStr = errStr.format(ii, jj, mm, dd, str(ids))
                            ids[ii] = np.append(ids[ii], tempIds[jj][ii])
                            scales[ii] = np.append(scales[ii], tempScales[jj][ii])
                            masses[ii] = np.append(masses[ii], tempMasses[jj][ii])
                            mdots[ii] = np.append(mdots[ii], tempMdots[jj][ii])
                            dens[ii] = np.append(dens[ii], tempDens[jj][ii])
                            csnds[ii] = np.append(csnds[ii], tempCsnds[jj][ii])

                        # Count the number of entries for each BH from each processor
                        nums[ii] += tempNums[jj][ii]

                    if len(errStr) > 0:
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

            # Merge lists of snapshots, and look for any missing
            mySnaps = np.hstack(mySnaps)
            log.debug("Obtained %d Snapshots" % (mySnaps.size))
            missingSnaps = []
            for ii in range(NUM_SNAPS):
                if (ii not in mySnaps):
                    missingSnaps.append(ii)

            if len(missingSnaps) > 0:
                log.warning("WARNING: snaps %s not in results!" % (str(missingSnaps)))

            log.debug("Total entries stored = %d" % (np.sum([np.sum(nn) for nn in nums])))

    # Convert from uinique BH IDs back to full mrgs list.  Sort results by time (scalefactor)
    if rank == 0:
        log.debug(" - Reconstructing")
        beg = datetime.now()
        ids = np.array([ids[ii] for ii in reconInds]).reshape(numMergers, 2)
        scales = np.array([scales[ii] for ii in reconInds]).reshape(numMergers, 2)
        masses = np.array([masses[ii] for ii in reconInds]).reshape(numMergers, 2)
        dens = np.array([dens[ii] for ii in reconInds]).reshape(numMergers, 2)
        mdots = np.array([mdots[ii] for ii in reconInds]).reshape(numMergers, 2)
        csnds = np.array([csnds[ii] for ii in reconInds]).reshape(numMergers, 2)
        nums = np.array([nums[ii] for ii in reconInds]).reshape(numMergers, 2)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        ndups = np.zeros((numMergers, 2))
        nosc = np.zeros((numMergers, 2))

        # Sort entries for each BH by scalefactor
        log.debug(" - Sorting")
        beg = datetime.now()
        for ii in range(numMergers):
            for jj in range(2):
                if nums[ii, jj] == 0:
                    continue
                # Check ID numbers yet again
                if not np.all(ids[ii, jj] == mergerIDs[ii, jj]):
                    errStr = "Error!  ii = {}, jj = {}.  Merger ID = {}, det ID = {}"
                    errStr = errStr.format(ii, jj, mergerIDs[ii, jj], ids[ii, jj])
                    errStr += "nums[ii,jj] = {:s}, shape {:s}"
                    errStr = errStr.format(str(nums[ii, jj]), str(np.shape(nums[ii, jj])))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

                # Order by scale-factor
                inds = np.argsort(scales[ii, jj])
                scales[ii, jj] = scales[ii, jj][inds]
                masses[ii, jj] = masses[ii, jj][inds]
                dens[ii, jj] = dens[ii, jj][inds]
                mdots[ii, jj] = mdots[ii, jj][inds]
                csnds[ii, jj] = csnds[ii, jj][inds]

                # Find and remove duplicates
                sameScales = np.isclose(scales[ii, jj][:-1], scales[ii, jj][1:], rtol=1e-8)
                sameMasses = np.isclose(masses[ii, jj][:-1], masses[ii, jj][1:], rtol=1e-8)
                dups = np.where(sameScales & sameMasses)[0]
                if (np.size(dups) > 0):
                    ndups[ii, jj] = np.size(dups)
                    scales[ii, jj] = np.delete(scales[ii, jj], dups)
                    masses[ii, jj] = np.delete(masses[ii, jj], dups)
                    dens[ii, jj] = np.delete(dens[ii, jj], dups)
                    mdots[ii, jj] = np.delete(mdots[ii, jj], dups)
                    csnds[ii, jj] = np.delete(csnds[ii, jj], dups)
                    ids[ii, jj] = np.delete(ids[ii, jj], dups)

                # Find and remove non-monotonic entries
                bads = np.where(np.diff(masses[ii, jj]) < 0.0)[0]
                if bads.size > 0:
                    nosc[ii, jj] = bads.size
                    scales[ii, jj] = np.delete(scales[ii, jj], bads)
                    masses[ii, jj] = np.delete(masses[ii, jj], bads)
                    dens[ii, jj] = np.delete(dens[ii, jj], bads)
                    mdots[ii, jj] = np.delete(mdots[ii, jj], bads)
                    csnds[ii, jj] = np.delete(csnds[ii, jj], bads)
                    ids[ii, jj] = np.delete(ids[ii, jj], bads)

                nums[ii, jj] = scales[ii, jj].size

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))
        # Log basic statistics
        _logStats('Number of entries', nums, log)
        _logStats('Duplicate entries', ndups, log)
        _logStats('Non-monotonic entries', nosc, log)

        # Save data
        filename = GET_MERGER_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
        data = _saveDetails(filename, run, ids, scales, masses, dens, mdots, csnds, log)

    return data


def _createRemnantDetails(run, log=None, mrgs=None, mdets=None, tree=None):
    """Create and Save Remnant Details Entries.

    Loads required data objects, calls `_matchRemnantDetails()`, corrects masses, and saves
    results to a npz file named by `GET_REMNANT_DETAILS_FILENAME`.

    Arguments
    ---------
    run : int
    log : ``logging.Logger`` or `None`
    mrgs : dict or `None`
    mdets : dict of `None`
    tree : dict of `None`

    Returns
    -------
    rdets : dict

    """

    if log is None:
        log = bh_constants._loadLogger(
            __file__, debug=True, verbose=True, run=run, version=__version__)

    log.debug("_createRemnantDetails()")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    # Only use root-processor
    if rank != 0:
        return

    if mrgs is None:
        log.debug("Loading Mergers.")
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)
    if mdets is None:
        log.debug("Loading Merger-Details.")
        mdets = loadMergerDetails(run, log=log)
    if tree is None:
        log.debug("Loading BHTree.")
        from illpy_lib.illbh import BHTree
        tree = BHTree.loadTree(run)

    # Create 'remnant' profiles ('RemnantDetails') based on tree and MergerDetails
    ids, scales, masses, dens, mdots, csnds = \
        _matchRemnantDetails(run, log=log, mrgs=mrgs, mdets=mdets, tree=tree)

    mcorrected = _unmergedMasses(scales, masses, mrgs, tree[BH_TREE.NEXT], log=log)

    savename = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    rdets = _saveDetails(savename, run, ids, scales, masses, dens, mdots, csnds, log,
                         mcorrected=mcorrected)

    return rdets


def _cleanErrDetails(ids, scales, masses, dens, mdots, csnds, log):
    log.debug("_cleanErrDetails()")
    numMergers = len(ids)
    OSC_ITER = 3
    nentsBeg = np.zeros(numMergers, dtype=np.int)
    nentsEnd = np.zeros(numMergers, dtype=np.int)
    nosc = np.zeros((numMergers, OSC_ITER), dtype=np.int)

    for ii in range(numMergers):
        nentsBeg[ii] = len(scales[ii])

        bads = np.where(np.diff(masses[ii]) < 0.0)[0]
        if bads.size > 0:

            ids[ii] = np.delete(ids[ii], bads)
            scales[ii] = np.delete(scales[ii], bads)
            masses[ii] = np.delete(masses[ii], bads)
            dens[ii] = np.delete(dens[ii], bads)
            mdots[ii] = np.delete(mdots[ii], bads)
            csnds[ii] = np.delete(csnds[ii], bads)

        nentsEnd[ii] = np.size(scales[ii])

    LVL = logging.INFO
    _logStats('Initial Entries', nentsBeg, log, lvl=LVL)
    _logStats('Final Entries', nentsEnd, log, lvl=LVL)
    for jj in range(OSC_ITER):
        _logStats('Nonmonotonic Entries, pass %d' % (jj), nosc[:, jj], log, lvl=LVL)

    return ids, scales, masses, dens, mdots, csnds


def _matchRemnantDetails(run, log=None, mrgs=None, mdets=None, tree=None):
    """Combine merger-dets entries to obtain dets for an entire merger-remnant's life.

    Each merger 'out'-BH is followed in subsequent mrgs to combine the dets entries forming
    a continuous chain of dets entries for the remnant's entire life after the initial merger.

    Loads `mergers` and `MergerDetails` files and uses that data to construct remnant dets.
    Runs on a single core (processors with ``rank > 0`` simply return at start).

    Arguments
    ---------
    run : int
        Illustris simulation run number {1,3}.
    log : ``logging.Logger``
        Logging object.
    mrgs : dict or `None`
        BH-Mergers dictionary.  Loaded from file if (`None`) not provided.
    mdets : dict or `None`
        `MergerDetails` data, loaded if not proveded.
    tree : dict or `None`
        BH-Merger-Tree dictionary.  Loaded from file if (`None`) not provided.

    Returns
    -------
    data : dict
        `RemnantDetails` data.

    """
    if log is None:
        log = bh_constants._loadLogger(
            __file__, debug=True, verbose=True, run=run, version=__version__)

    log.debug("_matchRemnantDetails()")

    # Load Mergers
    log.debug("Loading Mergers")
    if mrgs is None:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)
    m_scales = mrgs[MERGERS.SCALES]
    m_ids = mrgs[MERGERS.IDS]
    numMergers = np.int(mrgs[MERGERS.NUM])     # Convert from ``np.array(int)``

    # Load merger-dets file
    if mdets is None:
        log.debug("Loading Merger-Details")
        mdets = loadMergerDetails(run, log=log)
        if mdets is None:
            raise RuntimeError("Couldn't load `mdets`.")

    # Unpack data
    d_ids = mdets[DETAILS.IDS]
    d_scales = mdets[DETAILS.SCALES]
    d_masses = mdets[DETAILS.MASSES]
    d_dens = mdets[DETAILS.RHOS]
    d_mdots = mdets[DETAILS.MDOTS]
    d_csnds = mdets[DETAILS.CS]

    # Load BH Merger Tree
    if tree is None:
        log.debug("Loading BHTree")
        from illpy_lib.illbh import BHTree
        tree = BHTree.loadTree(run)
    nextMerger = tree[BH_TREE.NEXT]

    # Initialize data for results
    ids = np.zeros(numMergers, dtype=DTYPE.ID)
    nents = np.zeros(numMergers, dtype=np.int)

    idnums = numMergers*[None]
    mrgnums = numMergers*[None]
    scales = numMergers*[None]
    masses = numMergers*[None]
    dens = numMergers*[None]
    mdots = numMergers*[None]
    csnds = numMergers*[None]

    # Iterate over all mrgs
    # ------------------------
    log.debug("Matching data to remnants")
    for ii in range(numMergers):
        # First Merger
        #    Store dets after merger time for 'out' BH
        inds = np.where(d_scales[ii, BH_TYPE.OUT] > m_scales[ii])[0]
        if inds.size > 0:
            # Make sure values are valid
            temp_mdots = d_mdots[ii, BH_TYPE.OUT][inds]
            #    Find locations which are <=0.0 or inf
            bads = np.where((temp_mdots <= 0.0) | (~np.isfinite(temp_mdots)))[0]
            # Remove bad elements
            if bads.size > 0:
                inds = np.delete(inds, bads)

            # If valid elements remain
            if inds.size > 0:
                ids[ii] = d_ids[ii, BH_TYPE.OUT][inds[0]]
                scales[ii] = d_scales[ii, BH_TYPE.OUT][inds]
                masses[ii] = d_masses[ii, BH_TYPE.OUT][inds]
                dens[ii] = d_dens[ii, BH_TYPE.OUT][inds]
                mdots[ii] = d_mdots[ii, BH_TYPE.OUT][inds]
                csnds[ii] = d_csnds[ii, BH_TYPE.OUT][inds]

                idnums[ii] = d_ids[ii, BH_TYPE.OUT][inds]
                mrgnums[ii] = ii*np.ones(inds.size, dtype=int)

        else:
            log.warning("Merger %d without post-merger dets entries!" % (ii))
            ids[ii] = m_ids[ii, BH_TYPE.OUT]
            scales[ii] = []
            masses[ii] = []
            dens[ii] = []
            mdots[ii] = []
            csnds[ii] = []

            idnums[ii] = []
            mrgnums[ii] = []

        # Subsequent mrgs
        #    Find the next merger that this 'out' BH participates in
        next = nextMerger[ii]
        checkID = m_ids[next, BH_TYPE.OUT]
        checkScale = m_scales[next]
        if ids[ii] >= 0 and next >= 0:
            # Make sure `next` is correct, fix if not
            if ids[ii] not in m_ids[next]:
                next = _findNextMerger(ids[ii], m_scales[ii], m_ids, m_scales)
                checkID = m_ids[next, BH_TYPE.OUT]
                checkScale = m_scales[next]
                # Error if still not fixed
                if next >= 0 and ids[ii] not in m_ids[next]:
                    errStr = "ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], next, str(m_ids[next]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

        #    while a subsequent merger exists... store those entries
        while next >= 0:
            nextIDs = d_ids[next, BH_TYPE.OUT][:]
            # Make sure ID numbers match
            if checkID:
                if np.any(checkID != nextIDs):
                    errStr = "ii = %d, next = %d, IDs don't match!" % (ii, next)
                    errStr += "\nids[ii] = %d, check = %d" % (ids[ii], checkID)
                    errStr += "\nd_ids = %s" % (str(nextIDs))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            if np.size(scales[ii]) > 0:
                inds = np.where(d_scales[next, BH_TYPE.OUT] > np.max(scales[ii]))[0]
            else:
                inds = np.where(d_scales[next, BH_TYPE.OUT] > m_scales[ii])[0]

            if inds.size > 0:
                # Make sure values are valid
                temp_mdots = d_mdots[next, BH_TYPE.OUT][inds]
                #    Find locations which are <=0.0 or inf
                bads = np.where((temp_mdots <= 0.0) | (~np.isfinite(temp_mdots)))[0]
                # Remove bad elements
                if bads.size > 0:
                    inds = np.delete(inds, bads)

                if inds.size > 0:
                    scales[ii] = np.append(scales[ii], d_scales[next, BH_TYPE.OUT][inds])
                    masses[ii] = np.append(masses[ii], d_masses[next, BH_TYPE.OUT][inds])
                    dens[ii] = np.append(dens[ii], d_dens[next, BH_TYPE.OUT][inds])
                    mdots[ii] = np.append(mdots[ii], d_mdots[next, BH_TYPE.OUT][inds])
                    csnds[ii] = np.append(csnds[ii], d_csnds[next, BH_TYPE.OUT][inds])

                    idnums[ii] = np.append(idnums[ii], d_ids[next, BH_TYPE.OUT][inds])
                    mrgnums[ii] = np.append(mrgnums[ii], next*np.ones(inds.size, dtype=int))

            # Get next merger in Tree
            next = nextMerger[next]
            # Make sure `next` is correct, fix if not
            if checkID not in m_ids[next] and next >= 0:
                next = _findNextMerger(checkID, checkScale, m_ids, m_scales)
                # Error if still not fixed
                if (next >= 0 and ids[ii] not in m_ids[next]):
                    errStr = "ERROR: ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], next, str(m_ids[next]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            #    Get ID of next out-BH
            checkID = m_ids[next, BH_TYPE.OUT]
            checkScale = m_scales[next]

        # Appended entries may no longer be sorted, sort them
        if np.size(scales[ii]) > 0:
            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]

            idnums[ii] = idnums[ii][inds]
            mrgnums[ii] = mrgnums[ii][inds]
            nents[ii] = np.size(scales[ii])
        else:
            log.warning("Merger %d without ANY entries." % (ii))

        if np.any(~np.isfinite(mdots[ii])):
            errStr = "Infinite mdots at merger {}\n".format(ii)
            errStr += "\tmdots = {}".format(mdots[ii])
            log.error(errStr)
            raise RuntimeError(errStr)

    log.debug("Remnant dets collected.")
    _logStats('Number of entries', nents, log)

    return idnums, scales, masses, dens, mdots, csnds


def _unmergedMasses(allScales, allMasses, mrgs, nextMerger, log):
    """Remove mass from mrgs for the remnant in this BH lineage.

    Arguments
    ---------
    mind : int
    scales : (M,) array of arrays of scalar
    masses : (M,) array of arrays of scalar
    mrgs : dict
    tree : dict
    log : ``logging.Logger`` object

    Returns
    -------
    remmass : array_like of scalar

    """
    log.debug("_unmergedMasses()")
    numMergers = mrgs[MERGERS.NUM]
    allUnmerged = []
    errMergs_mass = []     # Mergers which cause errors trying to subtract masses
    effMergs_mass = []     # Mergers which are effected by the above errors
    errMergs_id = []     # Mergers where the IDs dont match properly
    effMergs_id = []
    errCorr = []
    numMal = 0
    numBon = 0
    for mind in range(numMergers):
        # Starting ID for this remnant
        bhID = mrgs[MERGERS.IDS][mind, BH_TYPE.OUT]
        # scales and masses for this remnant
        scales = allScales[mind]
        masses = allMasses[mind]
        # Index of next merger
        next = nextMerger[mind]
        log.debug(" - Merger %d (ID %d), Next = %d" % (mind, bhID, next))
        nmal = 0
        nbon = 0
        # loop over all following mrgs
        while next >= 0:
            # Figure out which BH the remnant is; select the other-BH's mass to subtract
            if bhID == mrgs[MERGERS.IDS][next, BH_TYPE.IN]:
                otherMass = mrgs[MERGERS.MASSES][next, BH_TYPE.OUT]
            elif bhID == mrgs[MERGERS.IDS][next, BH_TYPE.OUT]:
                otherMass = mrgs[MERGERS.MASSES][next, BH_TYPE.IN]
            else:
                errStr = "Initial Merger %d, out bh ID %d\n" % (mind, bhID)
                errStr += "\tNext merger {}, IDs dont match: {}, {}".format(
                    next, mrgs[MERGERS.IDS][next, BH_TYPE.IN],
                    mrgs[MERGERS.IDS][next, BH_TYPE.OUT])
                log.error(errStr)
                errMergs_id.append(mind)
                effMergs_id.append(next)
                break
                # raise RuntimeError(errStr)

            # Figure out which entries to subtract the mass from
            mscale = mrgs[MERGERS.SCALES][next]
            # Find scales after the merger
            allInds = np.where(scales >= mscale)[0]
            if allInds.size == 0:
                log.warning("Merger %d, Next %d, no matching scales after %f!" % (mind, next, mscale))
            else:
                inds = np.array(allInds)
                if np.isclose(scales[inds[0]], mscale) and inds.size > 1 and inds[0] > 0:
                    #    Dont need `+1` because were starting at ``inds[0]-1``
                    inds = inds[np.argmax(np.diff(masses[inds[0]-1:inds[0]+2]))]
                else:
                    inds = inds[0]

                masses[inds:] -= otherMass

                bads = np.where(masses[inds:] <= 0.0)[0]
                nbad = bads.size
                ntot = masses.size - inds
                if nbad > 0:
                    nmal += 1
                    logStr = "Merger {}, next {}: {}/{} = {:.4f} Invalid masses!".format(
                        mind, next, nbad, ntot, nbad/ntot)
                    log.debug(logStr)
                    if nbad > 4 or nbad/ntot > 0.2:
                        errStr = "Merger {}, next {}: Too many ({}/{}) invalid masses!\n".format(
                            mind, next, nbad, ntot)
                        endInd = np.min([len(allInds), 5])
                        useInds = allInds[:endInd]
                        errStr += "\tinds = {}...\n".format(str(useInds))
                        errStr += "\tmscale = {}, scales = {}...\n".format(
                            mscale, str(scales[useInds]))
                        errStr += "\tother = {}, masses = {}...\n".format(
                            otherMass, str(masses[useInds]))
                        log.warning(errStr)
                        # raise RuntimeError(errStr)
                        # Add the mass back on
                        masses[inds:] += otherMass
                        errMergs_mass.append(mind)
                        effMergs_mass.append(next)
                else:
                    nbon += 1

            # Update for next merger
            bhID = mrgs[MERGERS.IDS][next, BH_TYPE.OUT]
            next = nextMerger[next]

        allUnmerged.append(masses)
        numBon += nbon
        numMal += nmal
        if nbon > 0:
            errCorr.append(mind)

    numTot = numBon + numMal
    nerrs = len(errCorr)
    log.info(" - {}/{} = {:.4f} Remnants with correction errors".format(
        nerrs, numMergers, nerrs/numMergers))
    log.info(" - {}/{} = {:.4f} Total Errors (over all mrgs).".format(
        numMal, numTot, numMal/numTot))
    # log.info(" - Errors on remnants: {}".format(str(corrErrs)))
    errMergs_id = np.array(list(set(errMergs_id)))
    effMergs_id = np.array(list(set(effMergs_id)))
    frac_id = errMergs_id.size/numMergers
    errMergs_mass = np.array(list(set(errMergs_mass)))
    effMergs_mass = np.array(list(set(effMergs_mass)))
    frac_mass = errMergs_mass.size/numMergers
    logStr = " - ID Errors:\n - - {}/{} = {:.4f} Unique Mergers Effected, by {} unique mrgs"
    logStr = logStr.format(effMergs_id.size, numMergers, frac_id, errMergs_id.size)
    log.info(logStr)
    logStr = " - Mass Errors:\n - - {}/{} = {:.4f} Unique Mergers Effected, by {} unique mrgs"
    logStr = logStr.format(effMergs_mass.size, numMergers, frac_mass, errMergs_mass.size)
    log.info(logStr)

    return allUnmerged


def _detailsForMergers_snapshots(run, snapshots, bhIDsUnique, maxPerSnap, log):
    """Find dets entries for BH IDs in a particular snapshots.

    For each snapshot, store at most `_MAX_DETAILS_PER_SNAP` entries for each blackhole,
    interpolating values to an even spacing in scale-factor if more entries are found.  The first
    and last entries found are stored as is.

    Arguments
    ---------
    run : int
        Illustris run number {1,3}.
    snapshots : int or array_like of int
        Snapshots to search for dets entries.
    bhIDsUnique : (N,) array of long
        Sequence of all unique, merger-BH ID numbers.
    maxPerSnap : int or `None`
        The maximum number of dets entries to store for each snapshot.  If more entries than
        this are found, the values are interpolated down to a linearly-even spacing of
        scale-factors within the range of the matching dets entries.
        The maximum number of entries stored is thus ``maxPerSnap*np.size(snapshots)``
        for each BH.  When loading all merger/remnant dets, this should be
        `_MAX_DETAILS_PER_SNAP`.
    log : ``logging.Logger`` object
        Object for logging stream output.

    Returns
    -------
    numStoredTotal : (N,) array of int
        For each unique BH ID, the number of dets entries stored.
    scales : (N,) array of arrays of float
        Scale-factor of each entry.
        Each of the `N` entries corresponds to a unique merger-BH.  In that entry is an array
        including all of the matching dets entries found for that BH.  The length of each of
        these arrays can be (effectively) any value between zero and
        ``_MAX_DETAILS_PER_SNAP*np.size(snapshots)``.  This applies to all of the following
        returned values as-well.
    masses : (N,) array of arrays of float
        Mass of the BH in each entry.
    mdots : (N,) array of arrays of float
        Mass accretion rate of the BH.
    dens : (N,) array of arrays of float
        Local density around the BH.
    csnds : (N,) array of arrays of float
        Local sound-speed around the bH.
    ids : (N,) array of arrays of long
        ID number of the black-hole for each entry (for error-checking).

    """
    log.debug("_detailsForMergers_snapshot()")

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Initialize
    # ----------
    #     Make sure snapshots are iterable
    snapshots = np.atleast_1d(snapshots)
    numUniqueBHs = bhIDsUnique.size
    log.debug(" - {} Unique IDs, maxPerSnap = {}".format(numUniqueBHs, maxPerSnap))

    numMatchesTotal = np.zeros(numUniqueBHs, dtype=int)   # Number of matches in all snapshots
    numStoredTotal = np.zeros(numUniqueBHs, dtype=int)    # Number of entries stored in all

    #     Empty list for each Unique BH
    ids = numUniqueBHs * [[None]]
    scales = numUniqueBHs * [[None]]
    masses = numUniqueBHs * [[None]]
    mdots = numUniqueBHs * [[None]]
    csnds = numUniqueBHs * [[None]]
    dens = numUniqueBHs * [[None]]

    # Iterate over target snapshots
    # -----------------------------
    for snap in snapshots:
        log.debug("snap = %03d" % (snap))
        numMatchesSnap = np.zeros(numUniqueBHs, dtype=int)    # Num matches in a given snapshot
        numStoredSnap = np.zeros(numUniqueBHs, dtype=int)     # Num entries stored in a snapshot

        # Load `details`
        from illpy_lib.illbh import details
        dets = details.loadBHDetails(run, snap, verbose=False)
        numDets = dets[DETAILS.NUM]
        log.debug(" - %d Details" % (numDets))
        detIDs = dets[DETAILS.IDS]
        if (np.size(detIDs) == 0): continue

        if (not isinstance(detIDs[0], DTYPE.ID)):
            errStr = "Error: incorrect dtype = %s" % (np.dtype(detIDs[0]))
            log.error(errStr)
            raise RuntimeError(errStr)

        detScales = dets[DETAILS.SCALES]

        # Sort dets entries by IDs then by scales (times)
        detSort = np.lexsort((detScales, detIDs))
        count = 0
        # Iterate over and search for each unique BH ID
        for ii, bh in enumerate(bhIDsUnique):
            tempMatches = []
            # Increment up until reaching the target BH ID
            while count < numDets and detIDs[detSort[count]] < bh:
                count += 1

            # Iterate over all matching BH IDs, storing those dets' indices
            while count < numDets and detIDs[detSort[count]] == bh:
                tempMatches.append(detSort[count])
                count += 1

            # Store values at matching indices
            tempMatches = np.array(tempMatches)
            numMatchesSnap[ii] = tempMatches.size
            #     Only if there are some matches
            if (numMatchesSnap[ii] > 0):
                tempScales = detScales[tempMatches]
                tempMasses = dets[DETAILS.MASSES][tempMatches]
                tempMdots = dets[DETAILS.MDOTS][tempMatches]
                tempDens = dets[DETAILS.RHOS][tempMatches]
                tempCsnds = dets[DETAILS.CS][tempMatches]
                tempIDs = dets[DETAILS.IDS][tempMatches]

                # Interpolate down to only `maxPerSnap` entries
                if maxPerSnap and numMatchesSnap[ii] > maxPerSnap:
                    #    Create even spacing in scale-factor to interpolate to
                    newScales = zmath.spacing(tempScales, scale='lin', num=maxPerSnap)
                    tempMasses = np.interp(newScales, tempScales, tempMasses)
                    tempMdots = np.interp(newScales, tempScales, tempMdots)
                    tempDens = np.interp(newScales, tempScales, tempDens)
                    tempCsnds = np.interp(newScales, tempScales, tempCsnds)
                    #    Cant interpolate IDs, select random subset instead...
                    tempIDs = np.random.choice(tempIDs, size=maxPerSnap, replace=False)
                    if (not isinstance(tempIDs[0], DTYPE.ID)):
                        errStr = "Error: incorrect dtype for random tempIDs = {}"
                        errStr = errStr.format(np.dtype(tempIDs[0]))
                        log.error(errStr)
                        raise RuntimeError(errStr)

                    tempScales = newScales

                # Store matches
                #    If this is the first set of entries, replace ``[None]``
                if scales[ii][0] is None:
                    try:
                        scales[ii] = tempScales
                    except:
                        print((np.shape(scales[ii])))
                        print((scales[ii]))
                        print((np.shape(tempScales)))
                        zio.mpiError(comm)

                    masses[ii] = tempMasses
                    mdots[ii] = tempMdots
                    dens[ii] = tempDens
                    csnds[ii] = tempCsnds
                    ids[ii] = tempIDs
                #    If there are already entries, append new ones
                else:
                    if tempIDs[0] != ids[ii][-1] or tempIDs[0] != bh:
                        errStr = "Snap {}, ii {}, bh = {}, prev IDs = {}, new = {}!!"
                        errStr.format(snap, ii, bh, ids[ii][-1], tempIDs[0])
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

                    scales[ii] = np.append(scales[ii], tempScales)
                    masses[ii] = np.append(masses[ii], tempMasses)
                    mdots[ii] = np.append(mdots[ii], tempMdots)
                    dens[ii] = np.append(dens[ii], tempDens)
                    csnds[ii] = np.append(csnds[ii], tempCsnds)
                    ids[ii] = np.append(ids[ii], tempIDs)

                numStoredSnap[ii] += tempScales.size

            if count >= numDets:
                break

        numMatchesTotal += numMatchesSnap
        numStoredTotal += numStoredSnap
        snapTotal = np.sum(numMatchesSnap)
        total = np.sum(numMatchesTotal)
        snapOcc = np.count_nonzero(numMatchesSnap)
        occ = np.count_nonzero(numMatchesTotal)
        log.debug(" - %7d Matches (%7d total), %4d BHs (%4d)" % (snapTotal, total, snapOcc, occ))
        log.debug(" - Average and total number stored = {:f}, {:d}".format(
            numStoredTotal.mean(), np.sum(numStoredTotal)))

    scales = np.array(scales)
    masses = np.array(masses)
    mdots = np.array(mdots)
    dens = np.array(dens)
    csnds = np.array(csnds)
    ids = np.array(ids)

    return numStoredTotal, scales, masses, mdots, dens, csnds, ids


def _saveDetails(fname, run, ids, scales, masses, dens, mdots, csnds, log, **kwargs):
    """Package dets into dictionary and save to NPZ file.
    """
    log.debug("_saveDetails()")
    data = {DETAILS.IDS: ids,
            DETAILS.SCALES: scales,
            DETAILS.MASSES: masses,
            DETAILS.RHOS: dens,
            DETAILS.MDOTS: mdots,
            DETAILS.CS: csnds,
            DETAILS.RUN: np.array(run),
            DETAILS.CREATED: np.array(datetime.now().ctime()),
            DETAILS.VERSION: np.array(__version__),
            DETAILS.FILE: np.array(fname),
            'detailsPerSnapshot': np.array(_MAX_DETAILS_PER_SNAP),
            'detailsKeys': _GET_KEYS,
            }

    # Add any additional parameters
    #    Make sure all values are arrays
    for key, val in list(kwargs.items()):
        kwargs[key] = np.asarray(val)
    data.update(kwargs)

    log.info(" - Saving data to '%s'" % (fname))
    beg = datetime.now()
    zio.dictToNPZ(data, fname, verbose=True)
    end = datetime.now()
    log.info(" - - Saved to '%s' after %s" % (fname, str(end-beg)))

    return data


def _findNextMerger(myID, myScale, ids, scales):
    """Find the next merger index in which a particular BH participates.

    Search the full list of merger-BH ID number for matches to the target BH (`myID`) which take
    place at scale factors after the initial merger scale-factor (`myScale`).
    If no match is found, `-1` is returned.

    Arguments
    ---------
    myID : long
        ID number of the target blackhole.
    myScale : float
        Scalefactor at which the initial merger occurs (look for next merger after this time).
    ids : (N,2) array of long
        ID number of all merger BHs
    scales : (N,) array of float
        Scalefactors at which all of the mrgs occur.

    Returns
    -------
    nind : int
        Index of the next merger the `myID` blackhole takes place in.
        If no next merger is found, returns `-1`.


    Used by: `_matchRemnantDetails`
    """
    # Find where this ID matches another, but they dont have the same time (i.e. not same merger)
    search = (((myID == ids[:, 0]) | (myID == ids[:, 1])) & (myScale != scales))
    nind = np.where(search)[0]
    if (np.size(nind) > 0):
        # If multiple, find first
        if (np.size(nind) > 1):
            nind = nind[np.argmin(scales[nind])]
        else:
            nind = nind[0]

    else:
        nind = -1

    return nind


def _indBefAft(scaleDiff):
    """Retrieve the index matching the minimum of `scaleDiff` greater-than zero.

    Used by: `infer_merger_out_masses`
    """
    try:
        ind = zmath.argextrema(scaleDiff, 'min', 'g')
    except ValueError:
        ind = None

    return ind


def _logStats(name, prop, log, lvl=logging.DEBUG):
    """Log basic statistics about the given property.

    Arguments
    ---------
    name : str
        Name of the property.
    prop : array_like of scalar
        Array of measures of the target property.
    log : ``logging.Logger`` object
        Object to log to.

    """
    prop = np.asarray(prop)
    prop = prop.flatten()
    cnt = np.count_nonzero(prop)
    try:
        tot = prop.size
    except:
        tot = len(prop)
    frac = cnt/tot
    log.log(lvl, " - {}:".format(name))
    log.log(lvl, " - - {:6d}/{:6d} = {:.4f} nonzero".format(cnt, tot, frac))
    log.log(lvl, " - - Median and 68% (overall): {}".format(
        str(zmath.confidenceIntervals(prop, ci=0.68))))
    log.log(lvl, " - - Median and 68% (nonzero): {}".format(
        str(zmath.confidenceIntervals(prop, ci=0.68, filter='g'))))
    return


'''
def _detailsForBHLineage(run, mrg, log, rdets=None, tree=None, mrgs=None):
    log.debug("_detailsForBHLineage()")
    # Get all merger indices in this tree
    finMerger, bhIDs, mrgInds = illpy_lib.illbh.BHTree.allIDsForTree(
        run, mrg, tree=tree, mrgs=mrgs)
'''


if (__name__ == "__main__"):
    main()
