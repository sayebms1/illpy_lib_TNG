"""
Process radial profiles of Illustris subhalos.

Functions
---------
 - subhaloRadialProfiles() : construct binned, radial density profiles for all particle types


"""
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import numpy as np

from illpy_lib.constants import GET_ILLUSTRIS_DM_MASS, PARTICLE, DTYPE  #, BOX_LENGTH

from illpy_lib.subhalos import Subhalo
from illpy_lib.subhalos.Constants import SNAPSHOT, SUBHALO

import zcode.math as zmath
import zcode.inout as zio

NUM_RAD_BINS = 100


def subhaloRadialProfiles(run, snapNum, subhalo, radBins=None, nbins=NUM_RAD_BINS,
                          mostBound=None, verbose=True):
    """
    Construct binned, radial profiles of density for each particle species.

    Profiles for the velocity dispersion and gravitational potential are also constructed for
    all particle types together.

    Arguments
    ---------
       run       <int>    : illustris simulation run number {1, 3}
       snapNum   <int>    : illustris simulation snapshot number {1, 135}
       subhalo   <int>    : subhalo index number for target snapshot
       radBins   <flt>[N] : optional, right-edges of radial bins in simulation units
       nbins     <int>    : optional, numbers of bins to create if ``radBins`` is `None`
       mostBound <int>    : optional, ID number of the most-bound particle for this subhalo
       verbose   <bool>   : optional, print verbose output

    Returns
    -------
       radBins   <flt>[N]   : coordinates of right-edges of ``N`` radial bins
       posRef    <flt>[3]   : coordinates in simulation box of most-bound particle (used as C.O.M.)
       partTypes <int>[M]   : particle type numbers for ``M`` types, (``illpy_lib.constants.PARTICLE``)
       partNames <str>[M]   : particle type strings for each type
       numsBins  <int>[M, N] : binned number of particles for ``M`` particle types, ``N`` bins each
       massBins  <flt>[M, N] : binned radial mass profile
       densBins  <flt>[M, N] : binned mass density profile
       potsBins  <flt>[N]   : binned gravitational potential energy profile for all particles
       dispBins  <flt>[N]   : binned velocity dispersion profile for all particles

    """

    if verbose: print(" - - Profiler.subhaloRadialProfiles()")

    if verbose: print(" - - - Loading subhalo partile data")
    # Redirect output during this call
    with zio.StreamCapture():
        partData, partTypes = Subhalo.importSubhaloParticles(run, snapNum, subhalo, verbose=False)

    partNums = [pd['count'] for pd in partData]
    partNames = [PARTICLE.NAMES(pt) for pt in partTypes]
    numPartTypes = len(partNums)

    # Find the most-bound Particle
    #  ----------------------------

    posRef = None

    # If no particle ID is given, find it
    if (mostBound is None):
        # Get group catalog
        mostBound = Subhalo.importGroupCatalogData(
            run, snapNum, subhalos=subhalo, fields=[SUBHALO.MOST_BOUND])

    if (mostBound is None):
        warnStr  = "Could not find mostBound particle ID Number!"
        warnStr += "Run %d, Snap %d, Subhalo %d" % (run, snapNum, subhalo)
        warnings.warn(warnStr, RuntimeWarning)
        return None

    thisStr = "Run %d, Snap %d, Subhalo %d, Bound ID %d" % (run, snapNum, subhalo, mostBound)
    if verbose: print((" - - - - {:s} : Loaded {:s} particles".format(thisStr, str(partNums))))

    # Find the most-bound particle, store its position
    for pdat, pname in zip(partData, partNames):
        # Skip, if no particles of this type
        if (pdat['count'] == 0): continue
        inds = np.where(pdat[SNAPSHOT.IDS] == mostBound)[0]
        if (len(inds) == 1):
            if verbose: print((" - - - Found Most Bound Particle in '{:s}'".format(pname)))
            posRef = pdat[SNAPSHOT.POS][inds[0]]
            break

    # } pdat, pname

    # Set warning and return ``None`` if most-bound particle is not found
    if (posRef is None):
        warnStr = "Could not find most bound particle in snapshot! %s" % (thisStr)
        warnings.warn(warnStr, RuntimeWarning)
        return None

    mass = np.zeros(numPartTypes, dtype=object)
    rads = np.zeros(numPartTypes, dtype=object)
    pots = np.zeros(numPartTypes, dtype=object)
    disp = np.zeros(numPartTypes, dtype=object)
    radExtrema = None

    # Iterate over all particle types and their data
    #  ==============================================

    if verbose: print(" - - - Extracting and processing particle properties")
    for ii, (data, ptype) in enumerate(zip(partData, partTypes)):

        # Make sure the expected number of particles are found
        if (data['count'] != partNums[ii]):
            warnStr  = "%s" % (thisStr)
            warnStr += "Type '%s' count mismatch after loading!!  " % (partNames[ii])
            warnStr += "Expecting %d, Retrieved %d" % (partNums[ii], data['count'])
            warnings.warn(warnStr, RuntimeWarning)
            return None

        # Skip if this particle type has no elements
        #    use empty lists so that call to ``np.concatenate`` below works (ignored)
        if (data['count'] == 0):
            mass[ii] = []
            rads[ii] = []
            pots[ii] = []
            disp[ii] = []
            continue

        # Extract positions from snapshot, make sure reflections are nearest most-bound particle
        posn = reflectPos(data[SNAPSHOT.POS], center=posRef)

        # DarkMatter Particles all have the same mass, store that single value
        if (ptype == PARTICLE.DM): mass[ii] = [GET_ILLUSTRIS_DM_MASS(run)]
        else:                       mass[ii] = data[SNAPSHOT.MASS]

        # Convert positions to radii from ``posRef`` (most-bound particle), and find radial extrema
        rads[ii] = zmath.dist(posn, posRef)
        pots[ii] = data[SNAPSHOT.POT]
        disp[ii] = data[SNAPSHOT.SUBF_VDISP]
        radExtrema = zmath.minmax(rads[ii], prev=radExtrema, nonzero=True)

    # Create Radial Bins
    #  ------------------

    # Create radial bin spacings, these are the upper-bound radii
    if (radBins is None):
        radExtrema[0] = radExtrema[0]*0.99
        radExtrema[1] = radExtrema[1]*1.01
        radBins = zmath.spacing(radExtrema, scale='log', num=nbins)

    # Find average bin positions, and radial bin (shell) volumes
    numBins = len(radBins)
    binVols = np.zeros(numBins)
    for ii in range(len(radBins)):
        if (ii == 0): binVols[ii] = np.power(radBins[ii], 3.0)
        else:          binVols[ii] = np.power(radBins[ii], 3.0) - np.power(radBins[ii-1], 3.0)

    # Bin Properties for all Particle Types
    # -------------------------------------
    densBins = np.zeros([numPartTypes, numBins], dtype=DTYPE.SCALAR)    # Density
    massBins = np.zeros([numPartTypes, numBins], dtype=DTYPE.SCALAR)    # Mass
    numsBins = np.zeros([numPartTypes, numBins], dtype=DTYPE.INDEX)    # Count of particles

    # second dimension to store averages [0] and standard-deviations [1]
    potsBins = np.zeros([numBins, 2], dtype=DTYPE.SCALAR)               # Grav Potential Energy
    dispBins = np.zeros([numBins, 2], dtype=DTYPE.SCALAR)               # Velocity dispersion

    # Iterate over particle types
    if verbose: print(" - - - Binning properties by radii")
    for ii, (data, ptype) in enumerate(zip(partData, partTypes)):

        # Skip if this particle type has no elements
        if (data['count'] == 0): continue

        # Get the total mass in each bin
        numsBins[ii, :], massBins[ii, :] = zmath.histogram(rads[ii], radBins, weights=mass[ii],
                                                           edges='right', func='sum', stdev=False)

        # Divide by volume to get density
        densBins[ii, :] = massBins[ii, :]/binVols

    if verbose: print((" - - - - Binned {:s} particles".format(str(np.sum(numsBins, axis=1)))))

    # Consistency check on numbers of particles
    # -----------------------------------------
    #      The total number of particles ``numTot`` shouldn't necessarily be in bins.
    #      The expected number of particles ``numExp`` are those that are within the bounds of bins

    for ii in range(numPartTypes):

        numExp = np.size(np.where(rads[ii] <= radBins[-1])[0])
        numAct = np.sum(numsBins[ii])
        numTot = np.size(rads[ii])

        # If there is a discrepancy return ``None`` for error
        if (numExp != numAct):
            warnStr  = "%s\nType '%s' count mismatch after binning!" % (thisStr, partNames[ii])
            warnStr += "\nExpecting %d, Retrieved %d" % (numExp, numAct)
            warnings.warn(warnStr, RuntimeWarning)
            return None

        # If a noticeable number of particles are not binned, warn, but still continue
        elif (numAct < numTot-10 and numAct < 0.9*numTot):
            warnStr  = "%s : Type %s" % (thisStr, partNames[ii])
            warnStr += "\nTotal = %d, Expected = %d, Binned = %d" % (numTot, numExp, numAct)
            warnStr += "\nBin Extrema = %s" % (str(zmath.minmax(radBins)))
            warnStr += "\nRads = %s" % (str(rads[ii]))
            warnings.warn(warnStr, RuntimeWarning)
            raise RuntimeError("")

    # Convert list of arrays into 1D arrays of all elements
    rads = np.concatenate(rads)
    pots = np.concatenate(pots)
    disp = np.concatenate(disp)

    # Bin Grav Potentials
    counts, aves, stds = zmath.histogram(rads, radBins, weights=pots,
                                         edges='right', func='ave', stdev=True)
    potsBins[:, 0] = aves
    potsBins[:, 1] = stds

    # Bin Velocity Dispersion
    counts, aves, stds = zmath.histogram(rads, radBins, weights=disp,
                                         edges='right', func='ave', stdev=True)
    dispBins[:, 0] = aves
    dispBins[:, 1] = stds

    return radBins, posRef, mostBound, partTypes, partNames, \
        numsBins, massBins, densBins, potsBins, dispBins


def reflectPos(pos, center=None):
    """
    Given a set of position vectors, reflect those which are on the wrong edge of the box.

    Input positions ``pos`` MUST BE GIVEN IN illustris simulation units: [ckpc/h] !
    If a particular ``center`` point is not given, the median position is used.

    Arguments
    ---------
        pos    <flt>[N, 3] : array of ``N`` vectors, MUST BE IN SIMULATION UNITS
        center <flt>[3]   : optional, center coordinates, defaults to median of ``pos``

    Returns
    -------
        fix    <flt>[N, 3] : array of 'fixed' positions with bad elements reflected

    """
    from illpy_lib.illcosmo import Illustris_Cosmology_TOS
    COSMO = Illustris_Cosmology_TOS()

    FULL = COSMO.BOX_LENGTH
    HALF = 0.5*FULL

    # Create a copy of input positions
    fix = np.array(pos)

    # Use median position as center if not provided
    if (center is None):
        center = np.median(fix, axis=0)

    # Find distances to center
    offsets = fix - center

    # Reflect positions which are more than half a box away
    fix[offsets > +HALF] -= FULL
    fix[offsets < -HALF] += FULL

    return fix
