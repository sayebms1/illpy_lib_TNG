

import numpy as np
import illpy_lib
from illpy_lib import illcosmo
from illpy_lib.illbh import BHMergers
from illpy_lib.illbh import BHDetails


RUN = 3
VERBOSE = True


def main(run=RUN, verbose=VERBOSE):

    # COSMOLOGY #
    try:
        cosmo = illcosmo.Cosmology()
        t1 = cosmo[10]                                                                              # Test direct access to scalefactor list
        t2 = cosmo.comDist(100)/cosmo.comDist(30)                                                   # Test access by index to table values
        t3 = cosmo.lumDist(0.8)/cosmo.lumDist(0.5)                                                  # Test interpolation

    except Exception as ex:
        print("ERROR: could not calculate Cosmology test values!")
        raise ex


    # Check each test value
    for ii, tval in enumerate([t1, t2, t3]):

        if (tval <= 0.0 or tval >= 1.0):
            raise RuntimeError("ERROR: illpy_lib.illcosmo.Cosmology - Test %d" % ii)

    print("\n\nCosmology looks good.\n")




    # BHDETAILS #

    # Build Details Intermediate Files
    BHDetails.main(run=run, verbose=verbose)

    # Test BHDetails
    try:
        dets = BHDetails.loadBHDetails_NPZ(run, cosmo.num-2)                                        # Load 2nd-last snapshot's details
        t1 = dets[BHDetails.DETAIL_NUM]
        t2 = len(dets[BHDetails.DETAIL_TIMES])
        t3 = np.average(dets[BHDetails.DETAIL_TIMES])

    except Exception as ex:
        print("ERROR: could not calculate BHDetails test values!")
        raise ex


    if (t1 <= 0):
        raise RuntimeError("ERROR: illpy_lib.illbh.BHDetails - Test 1 = %d!" % t1)

    if (t2 <= 0):
        raise RuntimeError("ERROR: illpy_lib.illbh.BHDetails - Test 2 = %d!" % t2)

    if (t3 < cosmo[-2] or t3 > cosmo[-1]):
        raise RuntimeError("ERROR: illpy_lib.illbh.BHDetails - Test 3 = %f!" % t3)


    print("\n\nBHDetails looks good.\n")


    # BHMERGERS #

    # Build Mergers Intermediate Files
    BHMergers.main(run=run, verbose=verbose)

    # Test BHMergers
    try:
        mrg = BHMergers.loadMergers(run, verbose)
        t1 = mrg[BHMergers.MERGERS_NUM]
        t2 = np.count_nonzero(mrg[BHMergers.MERGERS_MAP_ONTOP] == True)
        t3 = np.average(dets[BHMergers.MERGERS_TIMES])

    except Exception as ex:
        print("ERROR: could not calculate BHMergers test values!")
        raise ex


    # Check each test value
    for ii, tval in enumerate([t1, t2, t3]):
        if (tval <= 0):
            raise RuntimeError("ERROR: illpy_lib.illbh.BHMergers - Test %d" % ii)


    print("\n\nBHMergers looks good.\n")

    return


if __name__ == "__main__": main()
