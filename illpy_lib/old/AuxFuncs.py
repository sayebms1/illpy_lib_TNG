'''
from __future__ import division
import numpy as np
import os


#  ======================================  #
#  =============  PHYSICS  ==============  #
#  ======================================  #

def aToZ(a, a0=1.0):
    """ Convert a scale-factor to a redshift """
    z = (a0/a) - 1.0
    return z

def zToA(z, a0=1.0):
    """ Convert a redshift to a scale-factor """
    a = a0/(1.0+z)
    return a



#  ===================================  #
#  =============  MATH  ==============  #
#  ===================================  #




def incrementRollingStats(avevar, count, val):
    """
    Increment a rolling average and stdev calculation with a new value

    avevar   : INOUT [ave, var]  the rolling average and variance respectively
    count    : IN    [int]       the *NEW* count for this bin (including new value)
    val      : IN    [float]     the new value to be included

    return
    ------
    avevar   : [float, float] incremented average and variation

    """
    delta      = val - avevar[0]

    avevar[0] += delta/count
    avevar[1] += delta*(val - avevar[0])

    return avevar


def finishRollingStats(avevar, count):
    """ Finish a rolling average and stdev calculation by find the stdev """

    if (count > 1): avevar[1] = np.sqrt(avevar[1]/(count-1))
    else:            avevar[1] = 0.0

    return avevar




def isApprox(v1, v2, TOL=1.0e-6):
    """
    Check if two scalars are eqeual to within some tolerance.

    Parameters
    ----------
    v1 : scalar
        first value
    v2 : scalar
        second value
    TOL : scalar
        Fractional tolerance within which to return True

    Returns
    -------
    retval : bool
        True if the (conservative) fractional difference between `v1` and `v2`
        is less than or equal to the tolerance.

    """

    # Find the lesser value to find conservative fraction
    less = np.min([v1, v2])
    # Compute fractional difference
    diff = np.fabs((v1-v2)/less)

    # Compare fractional difference to tolerance
    retval = (diff <= TOL)

    return retval


def stringArray(arr, format='%.2f'):
    out = [format % elem for elem in arr]
    out = "[" + " ".join(out) + "]"
    return out





#  ====================================  #
#  =============  FILES  ==============  #
#  ====================================  #





def getFileSize(fnames, precision=1):
    """
    Return a human-readable size of a file or set of files.

    Arguments
    ---------
    fnames : <string> or list/array of <string>, paths to target file(s)
    precisions : <int>, desired decimal precision of output

    Returns
    -------
    byteStr : <string>, human-readable size of file(s)

    """

    ftype = type(fnames)
    if (ftype is not list and ftype is not np.ndarray): fnames = [fnames]

    byteSize = 0.0
    for fil in fnames: byteSize += os.path.getsize(fil)

    byteStr = bytesString(byteSize, precision)
    return byteStr



def filesExist(files):

    # Assume all files exist
    allExist = True
    # Iterate over each, if any dont exist, break
    for fil in files:
        if (not os.path.exists(fil)):
            allExist = False
            break


    return allExist




#  ====================================  #
#  =============  OTHER  ==============  #
#  ====================================  #




def iterableNotString(args):
    """
    Check if the arguments is iterable and not a string.
    """
    import types

    # if NOT iterable, return false
    if (not np.iterable(args)): return False
    # if string, return False
    if (isinstance(args, types.StringTypes)): return False

    return True

'''
