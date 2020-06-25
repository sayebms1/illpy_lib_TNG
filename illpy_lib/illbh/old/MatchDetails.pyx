"""



"""

import numpy as np
cimport numpy as np

from BHConstants import BH_TYPE, BH_TIME
FST = BH_TIME.FIRST
BEF = BH_TIME.BEFORE
AFT = BH_TIME.AFTER
BH_IN = BH_TYPE.IN
BH_OUT = BH_TYPE.OUT

# A uint64 type (works with np.uint64)
ctypedef unsigned long long ULNG



def getDetailIndicesForMergers(np.ndarray[long,   ndim=1] targets, np.ndarray[ULNG,   ndim=2] mid, 
                               np.ndarray[double, ndim=1] mtime,   np.ndarray[long,   ndim=3] lind,
                               np.ndarray[double, ndim=3] ltime,   np.ndarray[long,   ndim=3] mnew,
                               np.ndarray[ULNG,   ndim=1] detid,   np.ndarray[double, ndim=1] dtime ):
    """
    Match merger BHs to details entries based on IDs and merger times.

    This method takes as input the full list of Merger BH IDs ``mid``, Merger times ``mtime``, and
    a set of details IDs ``detid`` and times ``dtime`` (presumably corresponding to one snapshot).
    The ``targets`` array controls which merger entries (by index) are searched for matches.
    For each target merger BH the earliest entry ('first')  is searched for, as well as the entries
    closest to the given merger time 'before' and 'after'.

    The INOUT 'link' arrays ``lind`` and ``ltime`` are used to track/store matches.  Anytime a 
    target BH is sucessfully matched to a detail entry, the link arrays are checked --- if there
    was no previous match, a link is made.  If a previous link was made, the new matches is
    compared to see if it is a better match (earlier for the 'first' entry, or closer to the merger
    time for the 'before' and 'after' matches).

    This function works quite quickly, so it is perfectly feasible to 'target' *all* mergers on
    each iteration (i.e. for each snapshot's worth of details entries); however, hypothetically,
    a more focused/thought-out search would be more efficient.


    Parameters
    ----------
    targets : IN, <long>[N]
        Array of indices corresponding to 'target' merger systems to be matched.

    mid : IN, <long>[M,2]
        Array of Merger BH IDs. Columns correspond to the 'in' and 'out' BHs (`BH_IN` / `BH_OUT`)

    mtime : IN, <double>[M]
        Array of times (scalefactors) of merger events.  The 'before' and 'after' events occur
        around this time

    lind : INOUT, <long>[M,2,3]
        Array of 'link' indices, i.e. for each BH and time ('before', 'after', 'first') store the
        index corresponding to the matching 'details' entry.

    ltime : INOUT, <double>[M,2,3]
        Array of 'link' times, i.e. the time of the detail entry matched for each BH and time,
        i.e. the time of the `lind` entry.

    mnew : INOUT, <int>[M,2,3], 
        Array of flags to mark which entries have been successfully matched.  `1` = True (matched)
        otherwise false (not matched)

    detid : IN, <long>[L]
        Array of BH IDs for each 'detail' entry

    dtime : IN, <double>[L]
        Array of times (scalefactors) for each 'detail' entry


    Returns
    -------
    numMatches : OUT, <int>
        Number of successful matches (including all 'before', 'after' and 'first') for either BH

    """


    # Get the lengths of all input arrays
    cdef int numMergers = mid.shape[0]
    cdef int numDetails = detid.shape[0]
    cdef int numTargets  = targets.shape[0]
    cdef int numMatches = 0

    # Sort first by ID then by time
    cdef np.ndarray s_det = np.lexsort( (dtime, detid) )

    # Declare variables
    cdef ULNG s_ind, t_id
    cdef ULNG s_first_match, s_before_match, s_after_match
    cdef ULNG d_ind_first, d_ind_before, d_ind_after
    cdef double t_time, d_first_time, d_before_time, d_after_time

    ### Iterate over Each Target Merger Binary System ###
    
    ### Iterate over Each Type of BH ###
    for BH in [BH_IN, BH_OUT]:

        # Sort *target* mergers first by ID, then by Merger Time
        s_targets = np.lexsort( (mtime[targets], mid[targets,BH]) )


        ### Iterate over Each Entry ###
        for ii in range(numTargets):
            s_ind  = targets[s_targets[ii]]                                                         # Sorted, target-merger index
            t_id   = mid[s_ind, BH]                                                                 # Target-merger BH ID
            t_time = mtime[s_ind]                                                                   # Target-merger BH Time

            ### Find First Match ###
            #   Find index in sorted 'det' arrays with first match to target ID `t_id`
            s_first_match = np.searchsorted( detid, t_id, 'left', sorter=s_det )

            # If this is not a match, no matching entries, continue to next BH
            #   If match is past last ID, no matches
            if( s_first_match >= numDetails ): continue
            #   If IDs don't match,       no matches
            if( detid[s_det[s_first_match]] != t_id ): continue
            
            ## Store first match
            d_ind_first = s_det[s_first_match]
            d_first_time = dtime[d_ind_first]

            # if there are no previous matches or new match is *earlier*
            if( lind[s_ind, BH, FST] < 0 or d_first_time < ltime[s_ind, BH, FST] ):
                lind [s_ind, BH, FST] = d_ind_first                                                 # Set link to details index
                ltime[s_ind, BH, FST] = d_first_time                                                # Set link to details time
                numMatches += 1
                mnew[s_ind, BH, FST] = 1
                

            ### Find 'before' Match ###
            #   Find the *latest* detail ID match *before* the merger time
            s_before_match = s_first_match                                                          # Has to come after the 'first' match
            if( s_before_match < numDetails-2 ):
                # Increment if the next entry is also an ID match and next time is still *before*
                while( detid[s_det[s_before_match+1]] == t_id  and 
                       dtime[s_det[s_before_match+1]] < t_time and
                       s_before_match < numDetails-2 ):
                    s_before_match += 1

            d_ind_before  = s_det[s_before_match]
            d_before_time = dtime[d_ind_before]

            # If this is still a match
            if( detid[d_ind_before] == t_id and d_before_time < t_time ):

                ## Store Before Match 
                # if there are no previous matches or new match is *later*
                if( lind[s_ind, BH, BEF] < 0 or d_before_time > ltime[s_ind, BH, BEF] ):
                    lind [s_ind, BH, BEF] = d_ind_before                                            # Set link to details index
                    ltime[s_ind, BH, BEF] = d_before_time                                           # Set link to details time
                    numMatches += 1
                    mnew[s_ind, BH, BEF] = 1


            ## Find 'after' Match
            #  Find the *earliest* detail ID match *after* the merger time
            #  Only exists if this is the 'out' BH
            if( BH == BH_OUT ):

                s_after_match = s_before_match                                                      # Has to come after the 'before' match
                if( s_after_match < numDetails-2 ):
                    # Increment if the next entry is also an ID match, but this time is still *before*
                    while( detid[s_det[s_after_match+1]] == t_id and 
                           dtime[s_det[s_after_match]] < t_time  and 
                           s_after_match < numDetails-2 ):
                        s_after_match += 1


                d_ind_after  = s_det[s_after_match]
                d_after_time = dtime[d_ind_after]

                # If this is still a match (ID matches, and this is *after* merger time)
                if( detid[d_ind_after] == t_id and d_after_time >= t_time ):

                    ## Store After Match 

                    # if there are no previous matches or new match is *earlier*
                    if( lind[s_ind, BH, AFT] < 0 or d_after_time < ltime[s_ind, BH, AFT] ):
                        lind [s_ind, BH, AFT] = d_ind_after                                         # Set link to details index
                        ltime[s_ind, BH, AFT] = d_after_time                                        # Set link to details time
                        numMatches += 1
                        mnew[s_ind, BH, AFT] = 1


            # } if BH

        # } ii

    # } BH

    return numMatches

# getDetailIndicesForMergers()






