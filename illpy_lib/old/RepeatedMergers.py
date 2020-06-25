# ==================================================================================================
# RepeatedMergers.py
# ------------------
#
#
#
# ------------------
# Luke Zoltan Kelley
# LKelley@cfa.harvard.edu
# ==================================================================================================


import numpy as np
from datetime import datetime
import os

# import illpy_lib
# from illpy_lib import AuxFuncs as aux
# from illpy_lib.constants import *
# from illpy_lib.illbh.BHConstants import *

import Basics

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import FindRepeats

import plotting as gwplot

RUN = 3                                                                                         # Which illustris simulation to target
VERBOSE = True
FILE_NAME = lambda xx: "ill-%d_repeated-mergers.npz" % (xx)

LOAD = False

REPEAT_LAST = 'last'
REPEAT_NEXT = 'next'
REPEAT_LAST_TIME = 'lastTime'
REPEAT_NEXT_TIME = 'nextTime'
REPEAT_CREATED = 'created'
REPEAT_RUN = 'run'

MYR = (1.0e6)*YEAR

#  ==============================================================  #
#  ===========================  MAIN  ===========================  #
#  ==============================================================  #


def main(run=RUN, load=LOAD, verbose=VERBOSE):

    # Initialize Log File
    print("\nRepeatedMergers.py\n")

    start_time  = datetime.now()

    # Set basic Parameters
    print(" - Loading Basics")
    start = datetime.now()
    base = Basics.Basics(run)
    stop = datetime.now()
    print((" - - Loaded after {:s}".format(str(stop-start))))

    # Load Repeated Mergers #
    repeats = getRepeats(run, base, load=False, verbose=verbose)

    # Process Repeat Data #
    lowerInter, numPast, numFuture = analyzeRepeats(interval, next, last, base)

    return interval, lowerInter, numPast, numFuture

    # Plot Repeat Data #
    gwplot.plotFig4_RepeatedMergers(interval, lowerInter, numFuture)

    end_time    = datetime.now()
    durat       = end_time - start_time

    print(("Done after {:s}\n\n".format(str(durat))))

    return


def getRepeats(run, base, load=False, verbose=VERBOSE):
    """
    Load repeat data from save file if possible, or recalculate directly.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1, 3}
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    load : <bool>, (optional=False)
        Reload repeat data directly from merger data
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc

    """

    if verbose: print(" - - RepeatedMergers.getRepeats()")

    fname = FILE_NAME(run)
    # Try to load precalculated repeat data
    if (os.path.exists(fname) and not load):
        if verbose: print((" - - - Loading Repeated Merger Data from '{:s}'".format(fname)))
        start = datetime.now()
        repeats = np.load(fname)
        stop = datetime.now()
        if verbose: print((" - - - - Loaded after {:s}".format(str(stop-start))))

    # Reload repeat data from mergers
    else:
        print(" - - - Finding Repeated Mergers from Merger Data")
        start = datetime.now()
        # Find Repeats
        repeats = calculateRepeatedMergers(run, base)
        # Save Repeats data
        aux.saveDictNPZ(repeats, fname, verbose=True)
        stop = datetime.now()
        if verbose: print((" - - - - Done after {:s}".format(str(stop-start))))

    return repeats


#  ==============================================================  #
#  =====================  PRIMARY FUNCTIONS  ====================  #
#  ==============================================================  #


def calculateRepeatedMergers(run, base, verbose=VERBOSE):
    """
    Use merger data to find and connect BHs which merge multiple times.

    Arguments
    ---------
    run : <int>
        Illlustris run number {1, 3}
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc

    """

    if verbose: print(" - - RepeatedMergers.calculateRepeatedMergers()")

    numMergers = base.mergers[MERGERS_NUM]
    last     = -1 * np.ones([numMergers, NUM_BH_TYPES], dtype=int)
    next     = -1 * np.ones([numMergers],              dtype=int)
    lastTime = -1.0*np.ones([numMergers, NUM_BH_TYPES], dtype=np.float64)
    nextTime = -1.0*np.ones([numMergers],              dtype=np.float64)

    # Convert merger scale factors to ages
    if verbose: print(" - - - Converting merger times")
    start = datetime.now()
    scales = base.mergers[MERGERS_TIMES]
    times = np.array([base.cosmo.age(sc) for sc in scales], dtype=np.float64)
    stop = datetime.now()
    if verbose: print((" - - - - Done after {:s}".format(str(stop-start))))

    # Get repeated merger information
    if verbose: print(" - - - Getting repeat statistics")
    start = datetime.now()
    mids = base.mergers[MERGERS_IDS]
    FindRepeats.findRepeats(mids, times, last, next, lastTime, nextTime)
    stop = datetime.now()
    if verbose: print((" - - - - Retrieved after {:s}".format(str(stop-start))))

    inds = np.where(last < 0)[0]
    print(("MISSING LAST = ",  len(inds)))

    inds = np.where(next < 0)[0]
    print(("MISSING NEXT = ",  len(inds)))

    # Create dictionary to store data
    repeats = {
        REPEAT_LAST: last,
        REPEAT_NEXT: next,
        REPEAT_LAST_TIME: lastTime,
        REPEAT_NEXT_TIME: nextTime,
        REPEAT_CREATED: datetime.now().ctime(),
        REPEAT_RUN: run
    }
    return repeats


def analyzeRepeats(repeats, base, verbose=VERBOSE):
    """
    Analyze the data from calculation of repeated mergers to obtain typical number of repeats, etc.

    Arguments
    ---------
    repeats : <dict>
        container for repeat data - see RepeatedMergers doc
    base : <Basics>, Basics.py:Basics object
        Contains merger information
    verbose : <bool>, (optional=VERBOSE)
        Print verbose output

    Returns
    -------


    """

    if verbose: print(" - - RepeatedMergers.analyzeRepeats()")

    numMergers   = base.numMergers

    last         = repeats[REPEAT_LAST]
    next         = repeats[REPEAT_NEXT]
    timeLast     = repeats[REPEAT_LAST_TIME]
    timeNext     = repeats[REPEAT_NEXT_TIME]

    aveFuture    = 0.0
    avePast      = 0.0
    aveFutureNum = 0
    avePastNum   = 0

    # timeBetween  = -1.0*np.ones(numMergers, dtype=float)
    numPast      = np.zeros(numMergers, dtype=int)
    numFuture    = np.zeros(numMergers, dtype=int)

    # Get age of the universe now
    # nowTime = base.cosmo.age(1.0)

    if verbose: print((" - - - {:d} Mergers".format(numMergers)))

    # Find number of unique merger BHs (i.e. no previous mergers)
    inds = np.where((last[:, IN_BH] < 0) & (last[:, OUT_BH] < 0) & (next[:] < 0))
    numTwoIsolated = len(inds[0])
    inds = np.where(((last[:, IN_BH] < 0) ^ (last[:, OUT_BH] < 0)) & (next[:] < 0))                 # 'xor' comparison
    numOneIsolated = len(inds[0])

    if verbose:
        print((" - - - Mergers with neither  BH previously merged = {:d}".format(numTwoIsolated)))
        print((" - - - Mergers with only one BH previously merged = {:d}".format(numOneIsolated)))

    # Go back through All Mergers to Count Repeats #

    for ii in range(numMergers):

        # Count Forward from First Mergers #
        # If this is a first merger
        if (all(last[ii, :] < 0)):

            # Count the number of mergers that the 'out' BH  from this merger, will later be in
            numFuture[ii] = countFutureMergers(next, ii)

            # Accumulate for averaging
            aveFuture += numFuture[ii]
            aveFutureNum += 1

        # Count Backward from Last Mergers #
        # If this is a final merger
        if (next[ii] < 0):

            # Count the number of mergers along the longest branch of past merger tree
            numPast[ii] = countPastMergers(last, ii)

            # Accumulate for averaging
            avePast += numPast[ii]
            avePastNum += 1

    # Calculate averages
    if (avePastNum   > 0): avePast   /= avePastNum
    if (aveFutureNum > 0): aveFuture /= aveFutureNum

    inds = np.where(next >= 0)[0]
    numRepeats = len(inds)
    fracRepeats = 1.0*numRepeats/numMergers

    print((" - - - Number of repeated mergers = {:d}/{:d} = {:.4f}".format(numRepeats, numMergers, fracRepeats)))
    print((" - - - Average Number of Repeated mergers  past, future  =  {:.3f}, {:.3f}".format(avePast, aveFuture)))

    indsInt = np.where(timeNext >= 0.0)[0]
    print((" - - - Number of merger intervals    = {:d}".format(len(indsInt))))
    timeStats = aux.avestd(timeNext[indsInt])
    print((" - - - - Time between = {:.4e} +- {:.4e} [Myr]".format(timeStats[0]/MYR, timeStats[1]/MYR)))

    inds = np.where(timeNext == 0.0)[0]
    print((" - - - Number of zero time intervals = {:d}".format(len(inds))))

    return timeNext[indsInt], numPast, numFuture


def countFutureMergers(next, ind):
    count = 0
    ii = ind
    while(next[ii] >= 0):
        count += 1
        ii = next[ii]

    return count


def countPastMergers(last, ind):

    last_in  = last[ind, IN_BH]
    last_out = last[ind, OUT_BH]

    num_in   = 0
    num_out  = 0

    if (last_in >= 0):
        num_in = countPastMergers(last, last_in)

    if (last_out >= 0):
        num_out = countPastMergers(last, last_out)

    return np.max([num_in, num_out])+1


if __name__ == "__main__":
    main()
