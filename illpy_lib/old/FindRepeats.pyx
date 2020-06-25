"""

"""


import numpy as np
cimport numpy as np

import illpy_lib
from illpy_lib.illbh.BHConstants import IN_BH, OUT_BH

def findRepeats(np.ndarray[long,   ndim=2] ids,      np.ndarray[double, ndim=1] times,
                np.ndarray[long,   ndim=2] last,     np.ndarray[long,   ndim=1] next,  
                np.ndarray[double, ndim=2] lastTime, np.ndarray[double, ndim=1] nextTime ):
    """
    Given the complete list of merger IDs and Times, connect Mergers with the same BHs.

    For each Merger system, take the 'out' BH and see if that BH participates in any future merger.
    The output array ``next`` provides the index for the 'next' merger which the 'out' BH
    participates in.  When such a 'repeat' is found, the second ('next') merger also has the
    previous merger stored in the ``last`` array, where `last[ii,jj]` says that BH `jj`
    ({IN_BH,OUT_BH}) from merger `ii` was previously the 'out' BH from merger `last[ii,jj]`.

    The array `lastTime[ii,jj]` gives the time from merger `ii` *since the previous merger* which
    each BH `jj` ({IN_BH,OUT_BH}) participated in.  `nextTime[ii]` gives the time until the 'out'
    BH from merger `ii` mergers mergers again.

    All array entries (mergers) without repeat matches maintain their default value --- which
    should be something like `-1`.

    Arguments
    ---------
    ids : IN, <long>[N,2]
        array of merger BH indices (for both 'in' and 'out' BHs)

    times : IN, <double>[N]
        array of merger times -- !! in units of age of the universe !!

    last : INOUT, <long>[N,2]
        Index of previous merger for each BH participating in the given merger.

    next : INOUT, <long>[N]
        Index of the next merger for the resulting 'out' BH.
    
    lastTime : INOUT, <double>[N,2]
        Time *since the last* merger event, for each BH.

    nextTime : INOUT, <double>[N]
        Time *until the next* merger event for the 'out' BH.

    """

    cdef long outid, ii, jj, next_ind, last_ind
    cdef long numMergers = ids.shape[0]
    cdef int hunth = np.int(np.floor(0.01*numMergers))

    # Get indices to sort mergers by time
    cdef np.ndarray sort_inds = np.argsort(times)

    # Iterate Over Each Merger, In Order of Merger Time #

    for ii in xrange(numMergers):

        if ( ii > 0 and ii%hunth == 0 ): print("%5d/%d".format(ii, numMergers))

        # Conver to sorted merger index
        last_ind = sort_inds[ii]

        # Get the output ID from this merger
        outid = ids[last_ind, OUT_BH]

        # Iterate over all Later Mergers #
        #  use a while loop so we can break out of it
        jj = ii+1
        while( jj < numMergers ):
            # Convert to sorted index
            next_ind = sort_inds[jj]

            # If previous merger goes into this one; save relationships
            for BH in [IN_BH, OUT_BH]:

                if ( ids[next_ind,BH] == outid ):

                    # For the 'next' Merger
                    # set index of previous merger
                    last[next_ind,BH] = last_ind
                    # set time since previous merger
                    lastTime[next_ind,BH] = times[next_ind] - times[last_ind]

                    # For the 'previous' Merger
                    # set index of next merger
                    next[last_ind] = next_ind
                    # set time until merger
                    nextTime[next_ind] = times[next_ind] - times[last_ind]


                    # Break back to highest for-loop over all mergers (ii)
                    jj = numMergers
                    break



            # Increment
            jj += 1

        # } jj

    # } ii

    return 





