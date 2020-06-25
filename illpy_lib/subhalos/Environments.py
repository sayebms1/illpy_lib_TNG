"""
Load Subhalo and environmental data corresponding to Merger BHs.

To load all merger environments
    `env = illpy_lib.subhalos.Environments.loadMergerEnvironments(1)`

To process all environments (i.e. for all Subhalos) into individual files, run with:
    `mpirun -n NP python -m illpy_lib.subhalos.Environments RUN`
    arguments:
        NP  <int> : num processors
        RUN <int> : illustris simulation number, {1, 3}


Classes
-------
   ENVIRON  : enumerator-like object for managing subhalo (environment) parameters dictionaries
   _TAGS    : enumerator-like object for managing MPI communication
   _ENVSTAT : enumerator-like object for status of single subhalo environment import


Functions
---------
   - get_merger_and_subhalo_indices     - get merger, snapshot and subhalo index numbers
   - loadMergerEnvironments             - primary API - load all subhalo environments as dict
   - main                               - Process all merger subhalos from scratch.

   - _runMaster                         - process manages all secondary ``slave`` processes
   - _runSlave                          - secondary process loads and saves data for each subhalo
   - _collectMergerEnvironments         - merge all subhalo environment files into single dict
   - _loadSingleMergerEnv               - load a single merger-subhalo environment and save.
   - _initStorage                       - initializes dict to store data for all subhalos
   - _parseArguments                    - parse commant line arguments
   - _mpiError                          - raise an error through MPI and exit all processes

   - _GET_MERGER_SUBHALO_FILENAME       - get filename for individual subhalo file
   - _GET_MISSING_LIST_FILENAME         - get filename for list of missing subhalos
   - _GET_FAILED_LIST_FILENAME          - get filename for list of failed  subhalos
   - _GET_ENVIRONMENTS_STATUS_FILENAME  - get filename for status of ``main`` execution.
   - _GET_MERGER_ENVIRONMENT_FILENAME   - get filename for dictionary of all subhalos

"""


import numpy as np
from datetime import datetime
import sys
import os
import argparse
import warnings

# from mpi4py import MPI

import zcode.inout as zio
import zcode.math as zmath
# from zcode.constants import PC

from illpy_lib.constants import DTYPE, GET_PROCESSED_DIR  # , CONV_ILL_TO_SOL, GET_BAD_SNAPS
from illpy_lib.illbh import MERGERS, BH_TYPE

from illpy_lib.subhalos import Subhalo, Profiler, particle_hosts
from illpy_lib.subhalos.particle_hosts import OFFTAB
from illpy_lib.subhalos.Constants import SUBHALO

# Hard Settings
_VERSION      = 1.4

# Soft (Default) Settings (Can be changed from command line)
VERBOSE      = False
CHECK_EXISTS = True
RUN          = 3

RAD_BINS     = 100
RAD_EXTREMA  = [1.0, 1.0e7]     # PC (converted to simulation units)


class ENVIRON():
    """ Keys for dictionary of subhalo environmental parameters.  See source for documentation."""

    # Meta Data
    RUN   = "run"                                 # Illustris simulation number {1, 3}
    SNAP  = "snap"                                # Illustris snapshot   number {1, 135}
    VERS  = "version"                             # ``Environments`` version number
    DATE  = "created"                             # datetime of creation
    STAT  = "status"                              # Status/validity of entry

    # For each Subhalo:
    SUBH  = "subhalo"                             # Subhalo (index) number corresponding to catalog
    BPID  = "boundid"                             # Mostbound Particle ID number for each subhalo
    CENT  = "center"                              # Center position of subhalo (most-bound particle)
    TYPE  = "types"                               # Illustris Particle type numbers present
    NAME  = "names"                               # Illustris Particlr type names

    # Radial profiles
    RADS  = "rads"                                # Positions of right-edges of radial bins
    NUMS  = "nums"                                # Number of particles by type in each bin
    DENS  = "dens"                                # Dens (ave)          by type    each bin
    MASS  = "mass"                                # Mass   of particle  by type in each bin

    POTS  = "pots"                                # Grav potential for all types   each bin
    DISP  = "disp"                                # Vel dispersion     all types   each bin

    GCAT_KEYS = "cat_keys"                        # Parameters of group-catalog entries included


class _TAGS():
    READY = 0
    START = 1
    DONE  = 2
    EXIT  = 3


class _ENVSTAT():
    FAIL = -1
    EXST = 0
    NEWF = 1


# Post-Processed Files

_MERGER_SUBHALO_FILENAME_BASE = "snap{1:03d}/ill{0:d}_snap{1:03d}_subhalo{2:06d}_v{3:.2f}.npz"


def _GET_MERGER_SUBHALO_FILENAME(run, snap, subhalo, version=_VERSION):
    pDir = GET_PROCESSED_DIR(run) + "subhalos/"
    fname = pDir + _MERGER_SUBHALO_FILENAME_BASE.format(run, snap, subhalo, version)
    return fname


_MERGER_ENVIRONMENT_FILENAME = "ill%d_merger-environments_v%.2f.npz"


def _GET_MERGER_ENVIRONMENT_FILENAME(run, version=_VERSION):
    pDir = GET_PROCESSED_DIR(run)
    fname = pDir + _MERGER_ENVIRONMENT_FILENAME % (run, version)
    return fname


# Temporary Files
_MISSING_LIST_FILENAME = "ill%d_missing_merger-subhalos_v%.2f.txt"


def _GET_MISSING_LIST_FILENAME(run, version=_VERSION):
    return _MISSING_LIST_FILENAME % (run, version)


_FAILED_LIST_FILENAME = "ill%d_failed_merger-subhalos_v%.2f.txt"


def _GET_FAILED_LIST_FILENAME(run, version=_VERSION):
    return _FAILED_LIST_FILENAME % (run, version)


_ENVIRONMENTS_STATUS_FILENAME = 'stat_Environments_ill%d_v%.2f.txt'


def _GET_ENVIRONMENTS_STATUS_FILENAME(run):
    return _ENVIRONMENTS_STATUS_FILENAME % (run, _VERSION)


def get_merger_and_subhalo_indices(run, verbose=True):
    """Get indices of mergers, snapshots and subhalos.

    Arguments
    ---------

    Returns
    -------
    merger_snaps : array(N,) of int
        Snapshot number for each merger.
    snap_mergers : (135,)
        List of merger indices for each snapshot.
    subh_ind_out : array(N,) of int
        Subhalo index number for each merger out BH.

    """
    if verbose: print(" - - Environments.get_merger_and_subhalo_indices()")

    from illpy_lib.illcosmo import Illustris_Cosmology_TOS
    COSMO = Illustris_Cosmology_TOS()

    if verbose: print(" - - - Loading Mergers")
    from illpy_lib.illbh import mergers as illbh_mergers   # noqa
    mergers = illbh_mergers.loadFixedMergers(run, verbose=verbose)
    if verbose: print((" - - - - Loaded %d mergers" % (mergers[MERGERS.NUM])))

    if verbose: print(" - - - Loading BH Hosts Catalog")
    bhHosts = particle_hosts.loadBHHosts(run, loadsave=True, verbose=verbose, bar=True)

    # Snapshot for each merger
    merger_snaps = mergers[MERGERS.MAP_MTOS]
    # Mergers for each snapshot
    snap_mergers = mergers[MERGERS.MAP_STOM]
    num_mergers = len(merger_snaps)

    # Initialize merger-subhalos array to invalid `-1`
    subh_ind_out = -1*np.ones(num_mergers, dtype=DTYPE.INDEX)
    subh_ind_in = -1*np.ones(num_mergers, dtype=DTYPE.INDEX)

    # Iterate Over Snapshots, list of mergers for each
    # ------------------------------------------------
    if verbose: print(" - - - Associating Mergers with Subhalos")
    for snap, mergs in enumerate(snap_mergers):
        # Skip if no mergers
        if len(mergs) <= 0: continue

        # Get the BH ID numbers for mergers in this snapshot
        ids_out = mergers[MERGERS.IDS][mergs, BH_TYPE.OUT]
        ids_in = mergers[MERGERS.IDS][mergs, BH_TYPE.IN]
        # Select BH-Hosts dict for this snapshot
        #    Individual snapshot dictionaries have string keys (i.e. '"054"' instead of '54')
        snap_out = OFFTAB.snapDictKey(snap)
        bh_hosts_snap_out = bhHosts[snap_out]
        #    For the 'in' BH, use the previous snapshot
        snap_in = OFFTAB.snapDictKey(snap-1)
        bh_hosts_snap_in = bhHosts[snap_in]
        #   Convert from array(dict) to just dict
        # bh_hosts_snap_out = bh_hosts_snap_out.item()

        # Out BH
        bad_flag_out = False
        if bh_hosts_snap_out[OFFTAB.BH_IDS] is None: bad_flag_out = True
        elif np.size(bh_hosts_snap_out[OFFTAB.BH_IDS]) == 1:
            if bh_hosts_snap_out[OFFTAB.BH_IDS].item() is None: bad_flag_out = True
        # Check for bad Snapshots (or other problems)
        if bad_flag_out:
            if snap in COSMO.GET_BAD_SNAPS(run):
                if verbose: print((" - - - - out BAD SNAPSHOT: Run %d, Snap %d" % (run, snap)))
            else:
                raise RuntimeError("Run %d, Snap %d: Bad BH_IDS out" % (run, snap))
        else:
            # Find the subhalo hosts for these merger BHs
            subh_ind_out[mergs] = particle_hosts.subhalosForBHIDs(
                run, snap, ids_out, bhHosts=bh_hosts_snap_out, verbose=False)

        # In BH
        bad_flag_in = False
        if bh_hosts_snap_in[OFFTAB.BH_IDS] is None: bad_flag_in = True
        elif np.size(bh_hosts_snap_in[OFFTAB.BH_IDS]) == 1:
            if bh_hosts_snap_in[OFFTAB.BH_IDS].item() is None: bad_flag_in = True
        # Check for bad Snapshots (or other problems)
        if bad_flag_in:
            if snap in COSMO.GET_BAD_SNAPS(run):
                if verbose: print((" - - - - in BAD SNAPSHOT: Run %d, Snap %d" % (run, snap)))
            else:
                # raise RuntimeError("Run {}, Snap {}: Bad BH_IDS in:{}".format(
                #     run, snap, bh_hosts_snap_in[OFFTAB.BH_IDS]))
                print(("Run {}, Snap {}: Bad BH_IDS in:{}".format(
                    run, snap, bh_hosts_snap_in[OFFTAB.BH_IDS])))

        else:
            # Find the subhalo hosts for these merger BHs ('in' use previous snapshot)
            subh_ind_in[mergs] = particle_hosts.subhalosForBHIDs(
                run, snap-1, ids_in, bhHosts=bh_hosts_snap_in, verbose=False)

    n_tot = len(subh_ind_out)
    n_good = np.count_nonzero(subh_ind_out >= 0)
    if verbose: print((" - - Out Good: {:5d}/{:d} = {:.4f}".format(n_good, n_tot, n_good/n_tot)))
    n_good = np.count_nonzero(subh_ind_in >= 0)
    if verbose: print((" - - In  Good: {:5d}/{:d} = {:.4f}".format(n_good, n_tot, n_good/n_tot)))

    return merger_snaps, snap_mergers, subh_ind_out, subh_ind_in


def _runMaster(run, comm):
    """
    Run master process which manages all of the secondary ``slave`` processes.

    Details
    -------
     - Retrieves merger, snapshot and subhalo indices
     - Iterates over snapshots and merger-subhalo pairs, distributing them to ``slave`` processes
       which load each subhalo profile and writes them to individual-subhalo files.
       - Loads most-bound particle ID numbers from group caalog for each snapshot and distributes
         this to each slave-process as-well.
       - Tracks how-many and which process (and subhalos) finish successfully

    """
    from mpi4py import MPI

    stat = MPI.Status()
    # rank = comm.rank
    size = comm.size

    print(" - Initializing")

    merger_snaps, snap_mergers, subh_ind_out, subh_ind_in = \
        get_merger_and_subhalo_indices(run, verbose=True)

    # Get all subhalos for each snapshot (including duplicates and missing)
    snapSubh     = [subh_ind_out[smrg] for smrg in snap_mergers]
    # Get unique subhalos for each snapshot, discard duplicates
    snapSubh_uni = [np.array(list(set(ssubh))) for ssubh in snapSubh]
    # Discard missing matches ('-1')
    snapSubh_uni = [ssubh[np.where(ssubh != -1)] for ssubh in snapSubh_uni]

    numUni = [len(ssubh) for ssubh in snapSubh_uni]
    numUniTot = np.sum(numUni)
    numMSnaps = np.count_nonzero(numUni)

    print((" - - %d Unique subhalos over %d Snapshots" % (numUniTot, numMSnaps)))

    # Iterate over Snapshots and Subhalos
    # ===================================
    #    distribute tasks to slave processes

    count = 0
    new   = 0
    exist = 0
    fail  = 0
    times = np.zeros(numUniTot)

    statFileName = _GET_ENVIRONMENTS_STATUS_FILENAME(run)
    statFile = open(statFileName, 'w')
    print((" - - Opened status file '%s'" % (statFileName)))
    statFile.write('%s\n' % (str(datetime.now())))
    beg = datetime.now()

    for snap, subs in zmath.renumerate(snapSubh_uni):

        if len(subs) <= 0: continue

        # Create output directory (subhalo doesn't matter since only creating dir)
        #    don't let slave processes create it - makes conflicts
        fname = _GET_MERGER_SUBHALO_FILENAME(run, snap, 0)
        zio.check_path(fname)

        # Get most bound particles for each subhalo in this snapshot
        mostBound = Subhalo.importGroupCatalogData(run, snap, subhalos=subs,
                                                   fields=[SUBHALO.MOST_BOUND], verbose=False)

        # Go over each subhalo
        for boundID, subhalo in zip(mostBound, subs):

            # Write status to file
            dur = (datetime.now()-beg)
            statStr = 'Snap %3d   %8d/%8d = %.4f   in %s   %8d new   %8d exist  %8d fail\n' % \
                (snap, count, numUniTot, 1.0*count/numUniTot, str(dur), new, exist, fail)
            statFile.write(statStr)
            statFile.flush()

            # Look for available slave process
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
            source = stat.Get_source()
            tag = stat.Get_tag()

            # Track number of completed profiles
            if tag == _TAGS.DONE:
                retStat, durat = data

                times[count] = durat
                count += 1
                if retStat == _ENVSTAT.NEWF:   new   += 1
                elif retStat == _ENVSTAT.EXST: exist += 1
                else:                          fail  += 1

            # Distribute tasks
            comm.send([snap, subhalo, boundID], dest=source, tag=_TAGS.START)

    statFile.write('\n\nDone after %s' % (str(datetime.now()-beg)))
    statFile.close()

    # Close out all Processes
    # =======================

    numActive = size-1
    print((" - Exiting %d active processes" % (numActive)))
    while(numActive > 0):

        # Find available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        source = stat.Get_source()
        tag = stat.Get_tag()

        # If we're recieving exit confirmation, count it
        if tag == _TAGS.EXIT: numActive -= 1
        else:
            # If a process just completed, count it
            if tag == _TAGS.DONE:
                times[count] = data[1]
                count += 1
                if data[0]: new += 1

            # Send exit command
            comm.send(None, dest=source, tag=_TAGS.EXIT)

    print((" - - %d/%d = %.4f Completed tasks!" % (count, numUniTot, 1.0*count/numUniTot)))
    print((" - - %d New Files" % (new)))

    return


def _runSlave(run, comm, radBins=None, loadsave=True, verbose=False):
    """
    Secondary process which continually receives subhalo numbers from ``master`` to load and save.

    Arguments
    ---------
       run      <int>       : illustris simulation run number {1, 3}
       comm     <...>       : MPI intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       radBins  <scalar>[N] : optional, positions of right-edges of radial bins
       loadsave <bool>      : optional, load data for this subhalo if it already exists

    Details
    -------
     - Waits for ``master`` process to send subhalo numbers
     - Loads existing save of subhalo data if possible (and ``loadsave``), otherwise re-imports it
     - Returns status to ``master``

    """

    from mpi4py import MPI

    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    if verbose: print((" - - Environments._runSlave() : rank %d/%d" % (rank, size)))

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        comm.send(None, dest=0, tag=_TAGS.READY)
        # Receive ``task`` ([snap, boundID, subhalo])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()

        if tag == _TAGS.START:
            # Extract parameters of environment
            snap, subhalo, boundID = task
            beg = datetime.now()
            # Load and save Merger Environment
            retEnv, retStat = _loadSingleMergerEnv(run, snap, subhalo, boundID, radBins=radBins,
                                                   loadsave=True, verbose=verbose)
            end = datetime.now()
            durat = (end-beg).total_seconds()
            comm.send([retStat, durat], dest=0, tag=_TAGS.DONE)
        elif tag == _TAGS.EXIT:
            break

    # Finish, return done
    comm.send(None, dest=0, tag=_TAGS.EXIT)

    return


def _loadSingleMergerEnv(run, snap, subhalo, boundID=None, radBins=None, loadsave=True, verbose=False):
    """
    Import and save merger-subhalo environment data.

    Arguments
    ---------
        run      <int>    : illustris simulation number {1, 3}
        snap     <int>    : illustris snapshot number {0, 135}
        subhalo  <int>    : subhalo index number for shit snapshot
        boundID  <int>    : ID of this subhalo's most-bound particle
        radBins  <flt>[N] : optional, positions of radial bins for creating profiles
        loadSave <bool>   : optional, load existing save file if possible
        verbose  <bool>   : optional, print verbose output

    Returns
    -------
        env      <dict>   : loaded dictionary of environment data
        retStat  <int>    : ``_ENVSTAT`` value for status of this environment


    """

    if verbose: print(" - - Environments._loadSingleMergerEnv()")

    fname = _GET_MERGER_SUBHALO_FILENAME(run, snap, subhalo)
    if verbose: print((" - - - Filename '%s'" % (fname)))

    # If we shouldnt or cant load existing save, reload profiles
    if not loadsave or not os.path.exists(fname):

        # Load Radial Profiles
        radProfs = Profiler.subhaloRadialProfiles(run, snap, subhalo, radBins=radBins,
                                                  mostBound=boundID, verbose=verbose)

        # Invalid profiles on failure
        if radProfs is None:
            warnStr = "INVALID PROFILES at Run %d, Snap %d, Subhalo %d, Bound ID %s" \
                % (run, snap, subhalo, str(boundID))
            warnings.warn(warnStr, RuntimeWarning)
            # Set return status to failure
            retStat = _ENVSTAT.FAIL
            env = None

        # Valid profiles
        else:
            # Unpack data
            outRadBins, posRef, retBoundID, partTypes, partNames, numsBins, \
                massBins, densBins, potsBins, dispBins = radProfs

            if boundID is not None and retBoundID != boundID:
                warnStr  = "Run %d, SNap %d, Subhalo %d" % (run, snap, subhalo)
                warnStr += "\nSent BoundID = %d, Returned = %d!" % (boundID, retBoundID)
                warnings.warn(warnStr, RuntimeWarning)

            # Build dict of data
            env = {
                ENVIRON.RUN: run,
                ENVIRON.SNAP: snap,
                ENVIRON.VERS: _VERSION,
                ENVIRON.DATE: datetime.now().ctime(),
                ENVIRON.TYPE: partTypes,
                ENVIRON.NAME: partNames,

                ENVIRON.SUBH: subhalo,
                ENVIRON.BPID: retBoundID,
                ENVIRON.CENT: posRef,

                ENVIRON.RADS: outRadBins,
                ENVIRON.NUMS: numsBins,
                ENVIRON.DENS: densBins,
                ENVIRON.MASS: massBins,
                ENVIRON.POTS: potsBins,
                ENVIRON.DISP: dispBins
            }

            # Save Data as NPZ file
            zio.dictToNPZ(env, fname, verbose=verbose)
            # Set return status to new file created
            retStat = _ENVSTAT.NEWF

    # File already exists
    else:

        # Load data from save file
        env = zio.npzToDict(fname)

        if verbose:
            print((" - - - File already exists for Run %d, Snap %d, Subhalo %d" % (run, snap, subhalo)))

        # Set return status to file already exists
        retStat = _ENVSTAT.EXST

    return env, retStat


def _loadAndCheckEnv(fname, rads, lenTypeExp, warn=False, care=True):
    """
    Load merger-subhalo environment file and perform consistency checks on its contents.

    Compares the total number of particles loaded of each type to that expected (from group-cat).
    Discrepancies are allowed (but Warning is made) - because some particles might not have fit in
    the range of bin radii.  If all the particles from a given type are missing however, it is
    assumed that there is something wrong (this happens for some reason...).

    Compares the positions of the radial bins in the loaded file with those that are expected.


    Arguments
    ---------
        fname      <str>    : filename to load from
        rads       <flt>[N] : positions of radial bins
        lenTypeExp <int>[M] : number of expected particles of each type (from group-cat)
        warn       <bool>   : optional, print optional warning messages on errors
        care       <bool>   : optional, return a failure status more easily

    Returns
    -------
        dat  <dict> : merger-subhalo environment data
        stat <bool> : success/good (``True``) or failure/bad (``False``)

    """

    # Load Merger-Subhalo Environment Data
    dat = zio.npzToDict(fname)

    # Assume good
    stat = True

    # Compare particle counts
    lenTypesAct = np.sum(dat[ENVIRON.NUMS], axis=1)
    #    Its possible that counts *shouldnt* match... b/c particles outside bins
    if not np.all(lenTypesAct == lenTypeExp):
        gcatStr = ', '.join(['{:d}'.format(np.int(num)) for num in lenTypeExp])
        datStr  = ', '.join(['{:d}'.format(num) for num in lenTypesAct])
        # Always warn for this
        warnStr  = "Numbers mismatch in '%s'" % (fname)
        warnStr += "\nFilename '%s'" % (fname)
        warnStr += "\nGcat = '%s', dat = '%s'" % (gcatStr, datStr)
        warnings.warn(warnStr, RuntimeWarning)

        if care: stat = False
        else:
            # See if all particles of any type are unexpectedly missing
            for ii in range(2):
                if lenTypesAct[ii] == 0 and lenTypeExp[ii] != 0:
                    # Set as bad
                    stat = False
                    # Send Warning
                    if warn:
                        warnStr  = "All particle of type %d are missing in data!" % (ii)
                        warnStr += "Filename '%s'" % (fname)
                        warnings.warn(warnStr, RuntimeWarning)

    # Make Sure Radii Match
    if not np.all(rads == dat[ENVIRON.RADS]):
        stat = False
        if warn:
            warnStr  = "Radii mismatch!"
            warnStr += "\nFilename '%s'" % (fname)
            warnings.warn(warnStr, RuntimeWarning)

    return dat, stat


def loadMergerEnvironments(run, loadsave=True, verbose=True, version=_VERSION):
    """
    Load all subhalo environment data as a dictionary with keys from ``ENVIRON``.

    NOTE: the 'env_in' dictionary was created using `_in_merger_environments()` (i.e. manually),
    and might not be recreated appropriately by `_collectMergerEnvironments()`.

    Arguments
    ---------
    run      <int>  : illustris simulation run number, {1, 3}
    loadsave <bool> : optional, load existing save if it exists, otherwise create new
    verbose  <bool> : optional, print verbose output
    version  <flt>  : optional, version number to load (can only create current version!)

    Returns
    -------
    env <dict> : all environment data for all subhalos, keys given by ``ENVIRON`` class

    """

    if verbose: print(" - - Environments.loadMergerEnvironments()")

    fname_out = _GET_MERGER_ENVIRONMENT_FILENAME(run, version=version)
    fname_in = zio.modify_filename(fname_out, append='_in')

    # Try to Load Existing Save File
    # ------------------------------
    if loadsave:
        if verbose: print((" - - Attempting to load saved file from '{}' and '{}'".format(fname_out, fname_in)))
        if os.path.exists(fname_out):
            env_out = zio.npzToDict(fname_out)
            env_in = zio.npzToDict(fname_in)
            if verbose: print(" - - - Loaded.")
        else:
            print((" - - - File '{}' or '{}' does not exist!".format(fname_out, fname_in)))
            loadsave = False

    # Import environment data directly, and save
    # ------------------------------------------
    if not loadsave:
        if verbose: print((" - - Importing Merger Environments, version %s" % (str(_VERSION))))
        env_out, env_in = _collectMergerEnvironments(run, verbose=verbose, version=version)
        zio.dictToNPZ(env_out, fname_out, verbose=True)
        zio.dictToNPZ(env_in, fname_in, verbose=True)

    return env_out, env_in


def _in_merger_environments(run, verbose=True, version=_VERSION):

    merger_snaps, snap_mergers, subh_ind_out, subh_ind_in = \
        get_merger_and_subhalo_indices(run, verbose=verbose)
    numMergers = len(merger_snaps)

    # Get all subhalos for each snapshot (including duplicates and missing)
    snap_subh_out = [subh_ind_out[smrg] for smrg in snap_mergers]
    snap_subh_in = [subh_ind_in[smrg] for smrg in snap_mergers]

    sampleSnap = 135
    env_in = _initStorage(run, sampleSnap, snap_subh_out[sampleSnap], numMergers,
                          verbose=verbose, version=version)

    beg = datetime.now()
    pbar = zio.getProgressBar(numMergers)
    pbar.start()
    count = 0
    numGood = 0
    numBad = 0
    # Iterate over each Snapshot
    for snap, (merg, subh_in) in zmath.renumerate(list(zip(snap_mergers, snap_subh_in))):
        # Get indices of valid subhalos
        inds_subh_in = np.where(subh_in >= 0)[0]
        # Skip this snapshot if no valid subhalos
        if inds_subh_in.size == 0 or len(merg) == 0: continue
        # Select corresponding merger indices
        inds_in = np.array(merg)[inds_subh_in]

        # Get Data from Group Catalog
        # ---------------------------
        try:
            gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=subh_in[inds_subh_in], verbose=False)
        # Count bad, and skip to next snapshot on failure
        except:
            print(("gcat import snap {} failed.  {} Mergers.".format(snap, len(merg))))
            numBad += len(merg)
            count += len(merg)
            pbar.update(count)
            continue

        # Extract desired data
        for key in env_in[ENVIRON.GCAT_KEYS]:
            env_in[key][inds_in, ...] = gcat[key][...]

        # Load Each Merger-Subhalo file contents
        # --------------------------------------
        for ind_subh, merger in zip(inds_subh_in, inds_in):
            count += 1
            subhalo = subh_in[ind_subh]
            # Store Subhalo number for each merger
            env_in[ENVIRON.SUBH][merger] = subhalo
            env_in[ENVIRON.SNAP][merger] = snap
            # Set as good merger-environment
            env_in[ENVIRON.STAT][merger] = 1
            numGood += 1

        # Update progessbar
        pbar.update(count)

    pbar.finish()
    end = datetime.now()

    if verbose:
        print((" - - - Completed after %s" % (str(end-beg))))
        print((" - - - Total   %5d/%5d = %.4f" % (count,   numMergers, count/numMergers)))
        print((" - - - Good    %5d/%5d = %.4f" % (numGood, numMergers, numGood/numMergers)))
        print((" - - - Bad     %5d/%5d = %.4f" % (numBad, numMergers, numBad/numMergers)))

    fname_out = _GET_MERGER_ENVIRONMENT_FILENAME(run, version=version)
    fname_in = zio.modify_filename(fname_out, append='_in')
    print("fname_out = '{}'".format(fname_out))
    print("fname_in = '{}'".format(fname_in))
    zio.dictToNPZ(env_in, fname_in, verbose=True)

    return env_in


def _collectMergerEnvironments(run, fixFails=True, verbose=True, version=_VERSION):
    """Load each subhalo environment file and merge into single dictionary object.

    Parameters for dictionary are given by ``ENVIRON`` class.

    Arguments
    ---------
        run      <int>  : illustris simulation number {1, 3}
        fixFails <bool> : optional, attempt to fix files with errors
        verbose  <bool> : optional, print verbose output
        version  <flt>  : optional, particular version number to load

    Returns
    -------
        env <dict> : dictionary of merger-subhalo environments for all mergers

    Notes
    -----
        - Loads a sample subhalo environment to initialize storage for all merger-subhalos
        - Iterates over each snapshots (in reverse)
        - Loads the group-catalog for each snapshot, stores this data in the output dict ``env``
        - Iterates over each merger/subhalo in the current snapshot
        - If the file is missing, it is skipped.
        - The file is loaded and checked against the expected number of particles, and standard
          radial bin positions.  If ``fixFails`` is `True`:
            + file is recreated if consistency checks fail.
            + New file is checked for consistency
            + File is skipped on failure
        - Radial profiles from good subhalo files are added to output dict ``env``.
        - The number of good, missing, failed, and fixed files are tracked and reported at the end
          (if ``verbose`` is `True`).

    """

    if verbose: print(" - - Environments._collectMergerEnvironments()")

    if version != _VERSION:
        warnStr = "WARNING: using deprecated version '%s' instead of '%s'" % (version, _VERSION)
        warnings.warn(warnStr, RuntimeWarning)

    # Whether to initially print warnings should be opposite of whether we try to fix them
    warnFlag = (not fixFails)

    # Load indices for mergers, snapshots and subhalos
    #    Indices for subhalos for 'out' and 'in' BHs, where 'in' subhalos correspond to previous
    #    snapshot (relative to that of 'merger' and 'out' BH)
    merger_snaps, snap_mergers, subh_ind_out, subh_ind_in = \
        get_merger_and_subhalo_indices(run, verbose=verbose)
    numMergers = len(merger_snaps)

    # Get all subhalos for each snapshot (including duplicates and missing)
    snap_subh_out = [subh_ind_out[smrg] for smrg in snap_mergers]
    snap_subh_in = [subh_ind_in[smrg] for smrg in snap_mergers]

    # Initialize Storage
    sampleSnap = 135
    env_out = _initStorage(run, sampleSnap, snap_subh_out[sampleSnap], numMergers,
                           verbose=verbose, version=version)
    env_in = _initStorage(run, sampleSnap, snap_subh_out[sampleSnap], numMergers,
                          verbose=verbose, version=version)

    miss_fname = _GET_MISSING_LIST_FILENAME(run, version=version)
    fail_fname = _GET_FAILED_LIST_FILENAME(run, version=version)
    formatStr = '{5} {0:3}  {1:8}  {2:4}  {3:8}  {4}\n'

    radBins = env_out[ENVIRON.RADS]

    numMiss = 0
    numFail = 0
    numGood = 0
    numFixd = 0
    count = 0

    # Initialize progressbar
    pbar = zio.getProgressBar(numMergers)
    with open(miss_fname, 'w') as missFile, open(fail_fname, 'w') as failFile:
        # Write header for output files
        for outFile, outType in zip([missFile, failFile], ['missing', 'failed']):
            if verbose: print((" - - - Opened %10s file '%s'" % (outType, outFile.name)))
            outFile.write(formatStr.format('Run', 'Merger', 'Snap', 'Subhalo', 'Filename', '#'))
            outFile.flush()

        beg = datetime.now()
        pbar.start()
        # Iterate over each Snapshot
        # ==========================
        for snap, (merg, subh_out, subh_in) in zmath.renumerate(list(zip(snap_mergers, snap_subh_out, snap_subh_in))):
            # Get indices of valid subhalos
            inds_subh_out = np.where(subh_out >= 0)[0]
            inds_subh_in = np.where(subh_in >= 0)[0]
            # Skip this snapshot if no valid subhalos
            if len(inds_subh_out) == 0: continue
            # Select corresponding merger indices
            inds_out = np.array(merg)[inds_subh_out]
            inds_in = np.array(merg)[inds_subh_in]

            # Get Data from Group Catalog
            # ---------------------------

            # Out BH
            gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=subh_out[inds_subh_out], verbose=False)
            # Extract desired data
            for key in env_out[ENVIRON.GCAT_KEYS]:
                env_out[key][inds_out, ...] = gcat[key][...]

            # Load Each Merger-Subhalo file contents
            # --------------------------------------
            for ind_subh, merger in zip(inds_subh_out, inds_out):

                count += 1
                subhalo = subh_out[ind_subh]
                thisStr = "Run %d Merger %d : Snap %d, Subhalo %d" % (run, merger, snap, subhalo)

                fname = _GET_MERGER_SUBHALO_FILENAME(run, snap, subhalo, version=version)
                # Skip if file doesn't exist
                if not os.path.exists(fname):
                    warnStr = "File missing at %s" % (thisStr)
                    warnStr += "\n'%s'" % (fname)
                    warnings.warn(warnStr, RuntimeWarning)
                    numMiss += 1
                    missFile.write(formatStr.format(run, merger, snap, subhalo, fname, ' '))
                    missFile.flush()
                    continue

                # Get expected number of particles from Group Catalog
                #   Number of each particle type (ALL 6) from group catalog
                lenTypes  = np.array(env_out[SUBHALO.NUM_PARTS_TYPE][merger])
                #   Indices of target particles (only 4) from subhalo profile files
                subhTypes = env_out[ENVIRON.TYPE]
                #   Select indices of relevant particles
                lenTypes  = lenTypes[subhTypes].astype(DTYPE.INDEX)

                # Catch errors while loading file to report where the error occured (which file)
                try:
                    dat, stat = _loadAndCheckEnv(fname, radBins, lenTypes,
                                                 warn=warnFlag, care=True)

                    # If load failed, and we want to try to fix it
                    if not stat and fixFails:
                        if verbose: print((" - - - - '%s' Failed. Trying to fix." % (fname)))

                        # Recreate data file
                        dat, retStat = _loadSingleMergerEnv(run, snap, subhalo, radBins=radBins,
                                                            loadsave=False, verbose=verbose)

                        # If recreation failed, skip
                        if retStat == _ENVSTAT.FAIL:
                            warnStr  = "Recreate failed at %s" % (thisStr)
                            warnStr += "Filename '%s'" % (fname)
                            warnings.warn(warnStr, RuntimeWarning)
                            numFail += 1
                            failFile.write(formatStr.format(run, merger, snap, subhalo, fname, ' '))
                            failFile.flush()
                            continue

                        # Re-check file
                        dat, stat = _loadAndCheckEnv(fname, radBins, lenTypes,
                                                     warn=True, care=False)

                        # If it still doesnt check out, warn and skip
                        if not stat:
                            warnStr  = "Recreation still has errors at %s" % (thisStr)
                            warnStr += "Filename '%s'" % (fname)
                            warnings.warn(warnStr, RuntimeWarning)
                            numFail += 1
                            failFile.write(formatStr.format(run, merger, snap, subhalo, fname, ' '))
                            failFile.flush()
                            continue
                        else:
                            if verbose: print((" - - - - '%s' Fixed" % (fname)))
                            numFixd += 1

                # Raise error with information about this process
                except:
                    print(("Load Error at %s" % (thisStr)))
                    print(("Filename '%s'" % (fname)))
                    raise

                # Store Subhalo number for each merger
                env_out[ENVIRON.SUBH][merger] = subhalo
                env_out[ENVIRON.SNAP][merger] = snap
                env_out[ENVIRON.BPID][merger] = dat[ENVIRON.BPID]
                env_out[ENVIRON.CENT][merger] = dat[ENVIRON.CENT]

                # Profiles Data
                env_out[ENVIRON.NUMS][merger, ...] = dat[ENVIRON.NUMS]
                env_out[ENVIRON.DENS][merger, ...] = dat[ENVIRON.DENS]
                env_out[ENVIRON.MASS][merger, ...] = dat[ENVIRON.MASS]
                env_out[ENVIRON.POTS][merger, ...] = dat[ENVIRON.POTS]
                env_out[ENVIRON.DISP][merger, ...] = dat[ENVIRON.DISP]

                # Set as good merger-environment
                env_out[ENVIRON.STAT][merger] = 1
                numGood += 1

            # In BH
            try:
                gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=subh_in[inds_subh_in],
                                                      verbose=False)
                # Extract desired data
                for key in env_in[ENVIRON.GCAT_KEYS]:
                    env_in[key][inds_in, ...] = gcat[key][...]

                # Load Each Merger-Subhalo file contents
                # --------------------------------------
                for ind_subh, merger in zip(inds_subh_in, inds_in):
                    subhalo = subh_in[ind_subh]
                    # Store Subhalo number for each merger
                    env_in[ENVIRON.SUBH][merger] = subhalo
                    env_in[ENVIRON.SNAP][merger] = snap
                    # Set as good merger-environment
                    env_in[ENVIRON.STAT][merger] = 1

            except:
                warnings.warn("gcat 'env_in' import snap {} failed.  {} Mergers.".format(
                    snap, len(merg)))

            # Update progessbar
            pbar.update(count)

        pbar.finish()
        end = datetime.now()

    if verbose:
        print((" - - - Completed after %s" % (str(end-beg))))
        print((" - - - Total   %5d/%5d = %f" % (count,   numMergers, 1.0*count/numMergers)))
        print((" - - - Good    %5d/%5d = %f" % (numGood, numMergers, 1.0*numGood/numMergers)))
        print((" - - - Missing %5d/%5d = %f" % (numMiss, numMergers, 1.0*numMiss/numMergers)))
        print((" - - - Failed  %5d/%5d = %f" % (numFail, numMergers, 1.0*numFail/numMergers)))
        print((" - - - Fixed   %5d/%5d = %f" % (numFixd, numMergers, 1.0*numFixd/numMergers)))

    return env_out, env_in


def _initStorage(run, snap, subhalos, numMergers, verbose=True, version=_VERSION):
    """Use data from a sample subhalo to shape and initialize a dictionary for storage.

    Arguments
    ---------
       run        <int>    : Illustis simulation number {1, 3}
       snap       <int>    : Illustris snapshot number {1, 135}
       subhalos   <int>[N] : List of merger subhalos for this snapshot
       numMergers <int>    : Total Number of mergers
       verbose    <bool>   : optional, print verbose output
       version    <flt>    : optional, version number to initialize with

    Returns
    -------
       env <dict> : Dictionary to store environmental data with space for radial profiles
                    and subhalo catalog data

    Notes
    -----
     - Requires that version is current (i.e. ``_VERSION``)
     - Subhalo profiles only store some particle types (used ones), while some subhalo catalog
       entries store all of them

    """

    if verbose: print(" - - Environments._initStorage()")

    env = {}

    if verbose: print(" - - - Finding sample subhalo")

    # Find Sample Halo
    inds = np.where(subhalos >= 0)[0]
    sample = np.min(inds)

    # Radial Profiles for Sample Halo
    # ------------------------------------
    if verbose: print((" - - - Loading Profiles for Sample: Snap %d, Subhalo %d" % (snap, sample)))
    fname = _GET_MERGER_SUBHALO_FILENAME(run, snap, subhalos[sample], version=version)
    subh = np.load(fname)

    # Find shape of arrays for each Subhalo
    #    [numParticles, numRadialBins]
    shape_type = np.shape(subh[ENVIRON.DENS])
    #    [numRadialBins]
    shape_all  = np.shape(subh[ENVIRON.DISP])

    # Double check number of particles and radial bins
    numTypes = len(subh[ENVIRON.NAME])
    numRBins = len(subh[ENVIRON.RADS])

    # Make sure lengths are consistent
    assert shape_type[0] == numTypes, "Number of particle types doesnt match!!"
    assert shape_type[1] == numRBins, "Number of radial bins    doesnt match!!"

    # Report particle types (numbers and names)
    if verbose:
        print(("shape_type = {}".format(shape_type)))
        print(("shape_all = {}".format(shape_all)))
        print(("numMergers = {}".format(numMergers)))
        print((" - - - Particle Types %s" % (str(['%6d' % nums for nums in subh[ENVIRON.TYPE]]))))
        print((" - - - Particle Names %s" % (str(['%6s' % nams for nams in subh[ENVIRON.NAME]]))))

    # Construct shape for all subhalos
    shape_type = np.concatenate([[numMergers], shape_type])
    shape_all  = np.concatenate([[numMergers], shape_all])
    if verbose: print((" - - - Shape of Profile Arrays = %s" % (str(shape_type))))

    # Initialize meta data
    env[ENVIRON.RUN]  = subh[ENVIRON.RUN]
    env[ENVIRON.SNAP] = np.zeros(numMergers, dtype=int)
    env[ENVIRON.VERS] = version
    env[ENVIRON.DATE] = datetime.now().ctime()
    env[ENVIRON.STAT] = np.zeros(numMergers, dtype=int)

    # For each merger/subhalo
    env[ENVIRON.SUBH] = np.zeros(numMergers, dtype=DTYPE.INDEX)
    env[ENVIRON.BPID] = np.zeros(numMergers, dtype=DTYPE.ID)
    env[ENVIRON.CENT] = np.zeros([numMergers, 3], dtype=DTYPE.SCALAR)
    env[ENVIRON.TYPE] = subh[ENVIRON.TYPE]
    env[ENVIRON.NAME] = subh[ENVIRON.NAME]

    # Initialize Profiles Storage Manually
    env[ENVIRON.RADS] = subh[ENVIRON.RADS]
    #    [mergers, part-types, rad-bins]
    env[ENVIRON.NUMS] = np.zeros(shape_type)
    env[ENVIRON.DENS] = np.zeros(shape_type)
    env[ENVIRON.MASS] = np.zeros(shape_type)

    #    [mergers, rad-bins]
    env[ENVIRON.DISP] = np.zeros(shape_all)
    env[ENVIRON.POTS] = np.zeros(shape_all)

    # Catalog for Sample Halo
    # ------------------------------------
    if verbose: print((" - - - Loading Catalog for Sample: Snap %d, Subhalo %d" % (snap, sample)))
    gcat = Subhalo.importGroupCatalogData(run, snap, subhalos=sample, verbose=True)
    if verbose: print(" - - - Loading Group-Cat Keys: '%s'" % (str(list(gcat.keys()))))

    # Initialize catalog properties automatically
    env[ENVIRON.GCAT_KEYS] = list(gcat.keys())
    for key in list(gcat.keys()):
        dat = gcat[key]
        shape = np.concatenate([[numMergers], np.shape(dat)])
        env[key] = np.zeros(shape)

    return env


def _parseArguments():
    """
    Prepare argument parser and load command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%s %.2f' % (sys.argv[0], _VERSION))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output', default=VERBOSE)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check",    dest='check',   action="store_true", default=CHECK_EXISTS)
    group.add_argument("--no-check", dest='nocheck', action="store_true", default=(not CHECK_EXISTS))

    parser.add_argument("RUN", type=int, nargs='?', choices=[1, 2, 3],
                        help="illustris simulation number", default=RUN)
    args = parser.parse_args()

    return args


def main():

    # Initialize MPI Parameters
    # -------------------------

    from mpi4py import MPI
    from illpy_lib.illcosmo import Illustris_Cosmology_TOS
    COSMO = Illustris_Cosmology_TOS()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    # name = MPI.Get_processor_name()
    # stat = MPI.Status()

    if rank == 0:
        NAME = sys.argv[0]
        print(("\n%s\n%s\n%s" % (NAME, '='*len(NAME), str(datetime.now()))))

    # Parse Arguments
    # ---------------
    args    = _parseArguments()
    RUN     = args.RUN
    VERBOSE = args.verbose
    if args.check:     CHECK_EXISTS = True
    elif args.nocheck: CHECK_EXISTS = False

    # Create Radial Bins (in simulation units)
    radExtrema = np.array(RAD_EXTREMA)/COSMO.CONV_ILL_TO_SOL.DIST.value   # [pc] ==> [ill]
    radBins = zmath.spacing(radExtrema, num=RAD_BINS)

    # Master Process
    # --------------
    if rank == 0:
        print(("RUN           = %d  " % (RUN)))
        print(("VERSION       = %.2f" % (_VERSION)))
        print(("MPI COMM SIZE = %d  " % (size)))
        print("")
        print(("VERBOSE       = %s  " % (str(VERBOSE))))
        print(("CHECK_EXISTS  = %s  " % (str(CHECK_EXISTS))))
        print("")
        print(("RAD_BINS      = %d  " % (RAD_BINS)))
        print(("RAD_EXTREMA   = [%.2e, %.2e] [pc]" % (RAD_EXTREMA[0], RAD_EXTREMA[1])))
        print(("              = [%.2e, %.2e] [sim]" % (radExtrema[0], radExtrema[1])))
        beg_all = datetime.now()

        try:
            _runMaster(RUN, comm)
        except Exception as err:
            _mpiError(comm, err)

        # Check subhalo files to see if/what is missing
        checkSubhaloFiles(RUN, verbose=VERBOSE, version=_VERSION)

        end_all = datetime.now()
        print((" - - Total Duration '%s'" % (str(end_all-beg_all))))

    # Slave Processes
    # ---------------
    else:

        try:
            _runSlave(RUN, comm, radBins, verbose=True)
        except Exception as err:
            _mpiError(comm, err)

    return


def _mpiError(comm, err="ERROR"):
    """
    Raise an error through MPI and exit all processes.

    Arguments
    ---------
       comm <...> : mpi intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       err  <str> : optional, extra error-string to print

    """

    import traceback
    rank = comm.rank
    print(("\nERROR: rank %d\n%s\n" % (rank, str(datetime.now()))))
    print((sys.exc_info()[0]))
    print((err.message))
    print((err.__doc__))
    print("\n")
    print((traceback.format_exc()))
    print("\n\n")
    comm.Abort(rank)
    return


if __name__ == "__main__":
    main()
