#cython: language_level=3
"""

"""

import numpy as np
cimport numpy as np

from illpy_lib.illbh import BH_TYPE
BH_IN = BH_TYPE.IN
BH_OUT = BH_TYPE.OUT

ctypedef np.float64_t float64
ctypedef np.int64_t int64
ctypedef np.uint64_t uint64

def build_tree(np.ndarray[uint64,  ndim=2] ids,
               np.ndarray[float64, ndim=1] scales,     np.ndarray[float64, ndim=1] times,
               np.ndarray[int64,   ndim=2] prev,       np.ndarray[int64,   ndim=1] next,
               np.ndarray[float64, ndim=2] scale_prev, np.ndarray[float64, ndim=1] scale_next,
               np.ndarray[float64, ndim=2] time_prev,  np.ndarray[float64, ndim=1] time_next):
    """
    Given the complete list of merger IDs and Times, connect Mergers with the same BHs.

    For each Merger system, take the 'out' BH and see if that BH participates in any future merger.
    The output array ``next`` provides the index for the 'next' merger which the 'out' BH
    participates in.  When such a 'repeat' is found, the second ('next') merger also has the
    previous merger stored in the ``prev`` array, where `prev[ii,jj]` says that BH `jj`
    ({BH_IN,BH_OUT}) from merger `ii` was previously the 'out' BH from merger `prev[ii,jj]`.

    The array `time_prev[ii,jj]` gives the time from merger `ii` *since the previous merger* which
    each BH `jj` ({BH_IN,BH_OUT}) participated in.  `time_next[ii]` gives the time until the 'out'
    BH from merger `ii` mergers mergers again.

    All array entries (mergers) without repeat matches maintain their default value --- which
    should be something like `-1`.

    Arguments
    ---------
    ids : IN, <long>[N,2]
        array of merger BH indices (for both 'in' and 'out' BHs)

    times : IN, <double>[N]
        array of merger times -- !! in units of age of the universe !!

    prev : INOUT, <long>[N,2]
        Index of previous merger for each BH participating in the given merger.

    next : INOUT, <long>[N]
        Index of the next merger for the resulting 'out' BH.

    time_prev : INOUT, <double>[N,2]
        Time *since the prev* merger event, for each BH.

    time_next : INOUT, <double>[N]
        Time *until the next* merger event for the 'out' BH.

    """

    cdef uint64 outid, test_id
    cdef int64 ii, jj, next_ind, last_ind
    cdef int64 num_mergers = ids.shape[0]
    cdef int hunth = np.int(np.floor(0.01*num_mergers))
    cdef float64 dt

    # Get indices to sort mergers by time
    # cdef np.ndarray sort_inds = np.argsort(times)

    # Iterate Over Each Merger, In Order of Merger Time
    # -------------------------------------------------
    for ii in range(num_mergers):
        if (ii > 0) and (ii % hunth == 0):
            print("{:3d}/100  -  {:5d}/{:d}".format(ii//hunth, ii, num_mergers))

        # Convert to sorted merger index
        # last_ind = sort_inds[ii]
        last_ind = ii

        # Get the output ID from this merger
        outid = ids[last_ind, BH_OUT]

        # Iterate over all Later Mergers
        #    use a while loop so we can break out of it
        jj = ii + 1
        while (jj < num_mergers):
            # Convert to sorted index
            # next_ind = sort_inds[jj]
            next_ind = jj

            # If previous merger goes into this one; save relationships
            for BH in [BH_IN, BH_OUT]:

                test_id = ids[next_ind, BH]

                # if (test_id == outid):
                if np.equal(test_id, outid):

                    dt = times[next_ind] - times[last_ind]

                    # For the 'next' Merger
                    #    set index of previous merger
                    prev[next_ind, BH] = last_ind
                    scale_prev[next_ind, BH] = scales[last_ind]
                    #    set time since previous merger
                    time_prev[next_ind, BH] = dt

                    # For the 'previous' Merger
                    #    set index of next merger
                    next[last_ind] = next_ind
                    scale_next[last_ind] = scales[next_ind]
                    #    set time until merger
                    time_next[last_ind] = dt

                    # Break back to highest for-loop over all mergers (ii)
                    jj = num_mergers
                    break

            # Increment
            jj += 1

    return
