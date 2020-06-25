"""
Construct and/or load a Blackhole Merger tree from the illustris blackhole merger data.

Functions
---------
-   loadTree                  - Load tree data from save file if possible, or recalculate directly.
-   analyzeTree               - Analyze the merger tree data to obtain typical number of repeats...
-   allIDsForTree             - Get all ID numbers for BH in the same merger-tree.

-   _constructBHTree          - Use merger data to find and connect BHs which merge multiple times.
-   _countFutureMergers       - Count the number of future mrgs using the next-merger map.
-   _countPastMergers         - Count the number of past mrgs using the last-merger map.
-   _getPastIDs               - Get all BH IDs in past-mrgs of this BHTree.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
import numpy as np
from datetime import datetime

from illpy_lib.constants import DTYPE
from . import constants
from .constants import MERGERS, BH_TYPE, BH_TREE, NUM_BH_TYPES

import zcode.inout as zio
from zcode.constants import MYR

VERSION = 0.21


def loadTree(run, mrgs=None, loadsave=True, verbose=True):
    """Load tree data from save file if possible, or recalculate directly.

    Arguments
    ---------
        run      : <int>, Illlustris run number {1, 3}
        mrgs  : <dict>, (optional=None), BHMerger data, reloaded if not provided
        loadsave : <bool>, (optional=True), try to load tree data from previous save
        verbose  : <bool>, (optional=True), Print verbose output

    Returns
    -------
        tree     : <dict>, container for tree data - see BHTree doc

    """

    if verbose: print(" - - BHTree.loadTree()")

    fname = constants.GET_BLACKHOLE_TREE_FILENAME(run, VERSION)

    # Reload existing BH Merger Tree
    # ------------------------------
    if loadsave:
        if verbose: print((" - - - Loading save file '{:s}'".format(fname)))
        if os.path.exists(fname):
            tree = zio.npzToDict(fname)
            if verbose: print(" - - - - Tree loaded")
        else:
            loadsave = False
            warnStr = "File '%s' does not exist!" % (fname)
            warnings.warn(warnStr, RuntimeWarning)

    # Recreate BH Merger Tree
    # -----------------------
    if not loadsave:
        if verbose: print(" - - - Reconstructing BH Merger Tree")
        # Load Mergers if needed
        if mrgs is None:
            from illpy_lib.illbh import mergers
            mrgs = mergers.load_fixed_mergers(run)
            if verbose: print((" - - - - Loaded {:d} mrgs".format(mrgs[MERGERS.NUM])))

        # Construct Tree
        if verbose: print(" - - - - Constructing Tree")
        tree = _constructBHTree(run, mrgs, verbose=verbose)

        # Analyze Tree Data, store meta-data to tree dictionary
        timeBetween, numPast, numFuture = analyzeTree(tree, verbose=verbose)

        # Save Tree data
        zio.dictToNPZ(tree, fname, verbose=True)

    return tree


def analyzeTree(tree, verbose=True):
    """Analyze the merger tree data to obtain typical number of repeats, etc.

    Arguments
    ---------
        tree : <dict> container for tree data - see BHTree doc
        verbose : <bool>, Print verbose output

    Returns
    -------


    """

    if verbose: print(" - - BHTree.analyzeTree()")

    last         = tree[BH_TREE.LAST]
    next         = tree[BH_TREE.NEXT]
    timeNext     = tree[BH_TREE.NEXT_TIME]
    numMergers   = len(next)

    aveFuture    = 0.0
    avePast      = 0.0
    aveFutureNum = 0
    avePastNum   = 0
    numPast      = np.zeros(numMergers, dtype=int)
    numFuture    = np.zeros(numMergers, dtype=int)

    if verbose: print((" - - - {:d} Mergers".format(numMergers)))

    # Find number of unique merger BHs (i.e. no previous mrgs)
    inds = np.where((last[:, BH_TYPE.IN] < 0) & (last[:, BH_TYPE.OUT] < 0) & (next[:] < 0))
    numTwoIsolated = len(inds[0])
    # Find those with one or the other
    inds = np.where(((last[:, BH_TYPE.IN] < 0) ^ (last[:, BH_TYPE.OUT] < 0)) & (next[:] < 0))
    numOneIsolated = len(inds[0])

    if verbose:
        print((" - - - Mergers with neither  BH previously merged = {:d}".format(numTwoIsolated)))
        print((" - - - Mergers with only one BH previously merged = {:d}".format(numOneIsolated)))

    for ii in range(numMergers):
        # Count Forward from First Mergers #
        #    If this is a first merger
        if all(last[ii, :] < 0):
            # Count the number of mrgs that the 'out' BH  from this merger, will later be in
            numFuture[ii] = _countFutureMergers(next, ii)
            # Accumulate for averaging
            aveFuture += numFuture[ii]
            aveFutureNum += 1

        # Count Backward from Last Mergers #
        #    If this is a final merger
        if next[ii] < 0:
            # Count the number of mrgs along the longest branch of past merger tree
            numPast[ii] = _countPastMergers(last, ii)
            # Accumulate for averaging
            avePast += numPast[ii]
            avePastNum += 1

    # Calculate averages
    if avePastNum   > 0:
        avePast /= avePastNum
    if aveFutureNum > 0:
        aveFuture /= aveFutureNum

    inds = np.where(next >= 0)[0]
    numRepeats = len(inds)
    fracRepeats = 1.0*numRepeats/numMergers

    indsInt = np.where(timeNext >= 0.0)[0]
    numInts = len(indsInt)
    timeStats = np.average(timeNext[indsInt]), np.std(timeNext[indsInt])
    inds = np.where(timeNext == 0.0)[0]
    numZeroInts = len(inds)

    if verbose:
        print((" - - - Repeated mergers = {:d}/{:d} = {:.4f}".format(
            numRepeats, numMergers, fracRepeats)))
        print((" - - - Average number past, future  =  {:.3f}, {:.3f}".format(avePast, aveFuture)))
        print((" - - - Number of merger intervals    = {:d}".format(numInts)))
        print((" - - - - Time between = {:.4e} +- {:.4e} [Myr]".format(
            timeStats[0]/MYR, timeStats[1]/MYR)))
        print((" - - - Number of zero time intervals = {:d}".format(numZeroInts)))

    timeBetween = timeNext[indsInt]

    # Store data to tree dictionary
    tree[BH_TREE.NUM_PAST] = numPast
    tree[BH_TREE.NUM_FUTURE] = numFuture
    tree[BH_TREE.TIME_BETWEEN] = timeBetween

    return timeBetween, numPast, numFuture


def allIDsForTree(run, mrg, tree=None, mrgs=None):
    """Get all of the ID numbers for BH in the same merger-tree as the given merger.

    Arguments
    ---------
    run : int
        Illustris simulation run number {1,3}.
    mrg : int
        Index of the target BH merger.  Any merger number in the same tree will yield the same
        results.
    tree : dict or `None`
        BHTree object will merger-tree data.  Loaded if not provided.
    mrgs : dict or `None`
        mergers object will merger data.  Loaded if not provided.

    Returns
    -------
    fin : int
        Index of the final merger this bh-tree participates in.  Acts as a unique identifier.
    allIDs : list of int
        List of all ID numbers of BHs which participate in this merger tree.

    """
    if not tree:
        tree = loadTree(run)

    nextMerg = tree[BH_TREE.NEXT]
    lastMerg = tree[BH_TREE.LAST]

    if not mrgs:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)

    m_ids = mrgs[MERGERS.IDS]

    # Go to the last merger
    fin = mrg
    while nextMerg[fin] >= 0:
        fin = nextMerg[fin]

    # Go backwards to get all IDs
    allIDs, mrgInds = _getPastIDs(m_ids, lastMerg, fin)
    return fin, allIDs, mrgInds


def _constructBHTree(run, mrgs, verbose=True):
    """Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
        run     : <int>, Illlustris run number {1, 3}
        mrgs : <dict>, mergers dictionary
        verbose : <bool>, (optional=True), Print verbose output

    Returns
    -------
        tree : <dict>  container for tree data - see BHTree doc

    """
    from . import BuildTree
    # import illpy.illcosmo

    if verbose: print(" - - BHTree.constructBHTree()")

    # cosmo = Cosmology()
    import illpy_lib.illcosmo
    cosmo = illpy_lib.illcosmo.Illustris_Cosmology()

    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
    from . import BuildTree

    numMergers = mrgs[MERGERS.NUM]
    last     = -1*np.ones([numMergers, NUM_BH_TYPES], dtype=DTYPE.INDEX)
    next     = -1*np.ones([numMergers], dtype=DTYPE.INDEX)
    lastTime = -1.0*np.ones([numMergers, NUM_BH_TYPES], dtype=DTYPE.SCALAR)
    nextTime = -1.0*np.ones([numMergers], dtype=DTYPE.SCALAR)

    # Convert merger scale factors to ages
    scales = mrgs[MERGERS.SCALES]
    times = np.array([cosmo.age(sc) for sc in scales], dtype=DTYPE.SCALAR)

    # Construct Merger Tree from node IDs
    if verbose: print(" - - - Building BH Merger Tree")
    start = datetime.now()
    mids = mrgs[MERGERS.IDS]
    BuildTree.buildTree(mids, times, last, next, lastTime, nextTime)
    stop = datetime.now()
    if verbose: print((" - - - - Built after {:s}".format(str(stop-start))))

    inds = np.where(last < 0)[0]
    if verbose: print((" - - - {:d} Missing 'last'".format(len(inds))))

    inds = np.where(next < 0)[0]
    if verbose: print((" - - - {:d} Missing 'next'".format(len(inds))))

    # Create dictionary to store data
    tree = {BH_TREE.LAST: last,
            BH_TREE.NEXT: next,
            BH_TREE.LAST_TIME: lastTime,
            BH_TREE.NEXT_TIME: nextTime,

            BH_TREE.CREATED: datetime.now().ctime(),
            BH_TREE.RUN: run,
            BH_TREE.VERSION: VERSION
            }

    return tree


def _countFutureMergers(next, ind):
    """Use the map of `next` mrgs and a starting index to count the future number of mrgs.
    """
    count = 0
    ii = ind
    while next[ii] >= 0:
        count += 1
        ii = next[ii]
    return count


def _countPastMergers(last, ind):
    """Use the map of `last` mrgs and a starting index to count the past number of mrgs.
    """
    last_in  = last[ind, BH_TYPE.IN]
    last_out = last[ind, BH_TYPE.OUT]
    num_in   = 0
    num_out  = 0
    if last_in >= 0:
        num_in = _countPastMergers(last, last_in)
    if last_out >= 0:
        num_out = _countPastMergers(last, last_out)
    return np.max([num_in, num_out])+1


def _getPastIDs(m_ids, lastMerg, ind, idlist=[], mrglist=[]):
    """Get all BH IDs in past-mrgs of this BHTree.

    Arguments
    ---------
    m_ids : (N,2) array of int
        Merger BH ID numbers.
    last : (N,2) array of int
        For a given merger, give the index of the merger for each of the constituent BHs.
        `-1` if there was no previous merger.
    ind : int
        Index of merger to follow.
    idlist : list of int
        Existing list of merger IDs to append to.  Uses a `set` type intermediate to assure unique
        values.

    Used by: `allIDsForTree`.
    """
    ids_in = [m_ids[ind, BH_TYPE.IN]]
    ids_out = [m_ids[ind, BH_TYPE.OUT]]
    mrg_in = [ind]
    mrg_out = [ind]
    last_in  = lastMerg[ind, BH_TYPE.IN]
    last_out = lastMerg[ind, BH_TYPE.OUT]
    if last_in >= 0:
        ids_in, mrg_in = _getPastIDs(m_ids, lastMerg, last_in, ids_in, mrg_in)
    if last_out >= 0:
        ids_out, mrg_out = _getPastIDs(m_ids, lastMerg, last_out, ids_out, mrg_out)
    return list(set(ids_in + ids_out + idlist)), list(set(mrg_in + mrg_out + mrglist))
