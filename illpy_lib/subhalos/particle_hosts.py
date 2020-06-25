"""
Manage table of particle offsets for associating particles with halos and subhalos.

The table is in the form of a dictionary with keys given by the values of the ``OFFTAB`` class.
The method ``loadOffsetTable()`` is the only necessary API - it deals with constructing, saving,
and loading the offset table.

Classes
-------
    OFFTAB : enumerator-like class for dictionary key-words

Functions
---------
    # loadOffsetTable            : load offset table for target run and snapshot
    _load_bh_hosts_snap_table            : load (sub)halo host associations for blackholes in one snapshot
    _load_bh_hosts_table                : load (sub)halo host associations for blackholes in all snapshots
    main                       :
    bh_subhalos           : find subhalos for given BH IDs

    _GET_OFFSET_TABLE_FILENAME : filename which the offset table is saved/loaded to/from

    _construct_offset_table      : construct the offset table from the group catalog
    _construct_bh_index_table     : construct mapping from BH IDs to indices in snapshot files


Notes
-----
    The structure of the table is 3 different arrays with corresponding entries.
    ``halos``     (``OFFTAB.HALOS``)    : <int>[N],   halo number
    ``subhalos``  (``OFFTAB.SUBHALOS``) : <int>[N],   subhalo number
    ``particles`` (``OFFTAB.OFFSETS``)  : <int>[N, 6], particle offsets for each halo/subhalo

    The table is ordered in the same way as the snapshots, where particles are grouped into subhalos,
    which belong to halos.  Each halo also has (or can have) a group of particles not in any subhalo.
    Finally, the last entry is for particles with no halo and no subhalo.  When there is no match for
    a subhalo or halo, the corresponding number is listed as '-1'.

    For a halo 'i', with NS_i subhalos, there are NS_i+1 entries for that halo.
    If the total number of subhalos is NS = SUM_i(NS_i), and there are
    NH halos, then the total number of entries is NS + NH + 1.

    This is what the table looks like (using made-up numbers):

                            PARTICLES {0, 5}
        HALO    SUBHALO     0     1  ...   5
      | ====================================
      |    0          0     0     0  ...   0  <-- halo-0, subhalo-0, no previous particles
      |    0          1    10     4  ...   1  <--  first part0 for this subhalo is 10th part0 overall
      |    0          2    18     7  ...   3  <--  first part1 for this subhalo is  7th part1 overall
      |                              ...
      |    0         -1   130    58  ...  33  <-- particles of halo-0, no subhalo
      |
      |    1         22   137    60  ...  35  <-- In halo-0 there were 22 subhalos and 137 part0, etc
      |    1         23              ...
      |                              ...
      |                              ...
      |   -1         -1  2020   988  ... 400
      | ====================================

    Thus, given a particle5 of index 35, we know that that particle belongs to Halo-1, Subhalo-22.
    Alternatively, a particle0 index of 134 belongs to halo-0, and has no subhalo.
    Finally, any Particle1's with index 988 or after belong to no subhalo, and no halo.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from datetime import datetime

import numpy as np
import h5py
import tqdm

from illpy_lib.constants import DTYPE, PARTICLE, GET_ILLUSTRIS_OUTPUT_DIR  # , GET_BAD_SNAPS, NUM_SNAPS
from illpy_lib.subhalos.Constants import SNAPSHOT
from illpy_lib.illcosmo import Illustris_Cosmology_TOS

import zcode.inout as zio

# Directly to place processed / metadata / derived data files
#    This replaces the value in 'illpy.Constants'
#    formerly: "/n/home00/lkelley/hernquistfs1/illustris/data/%s/output/postprocessing/"
_PROCESSED_DIR = "/n/home00/lkelley/illustris/data/{}/output/postprocessing/"

import illpy as ill

_VERSION = '1.0'


class OFFTAB():
    """Keys for offset table dictionary.
    """

    RUN         = 'run'
    SNAP        = 'snapshot'
    VERSION     = 'version'
    CREATED     = 'created'
    FILENAME    = 'filename'

    HALOS       = 'halo_numbers'
    SUBHALOS    = 'subhalo_numbers'
    OFFSETS     = 'particle_offsets'

    BH_IDS      = 'bh_ids'
    BH_INDICES  = 'bh_indices'
    BH_HALOS    = 'bh_halos'
    BH_SUBHALOS = 'bh_subhalos'

    # @staticmethod
    # def snapDictKey(snap):
    #     return "%03d" % (snap)


# _FILENAME_OFFSET_TABLE = "offsets/ill{run:d}_snap{snap:03d}_offset-table_v{version}.hdf5"
# _FILENAME_BH_HOSTS_SNAP_TABLE = "bh-hosts/ill{run:d}_snap{snap:03d}_bh-hosts_v{version}.hdf5"
# _FILENAME_BH_HOSTS_TABLE = "bh-hosts/ill{run:d}_bh-hosts_v{version}.hdf5"
_FILENAME_OFFSET_TABLE = "offsets/ill{run:d}_snap{snap:03d}_offset-table.hdf5"
_FILENAME_BH_HOSTS_SNAP_TABLE = "bh-hosts/ill{run:d}_snap{snap:03d}_bh-hosts.hdf5"
_FILENAME_BH_HOSTS_TABLE = "bh-hosts/ill{run:d}_bh-hosts.hdf5"


def bh_subhalos(run, snap, log, bh_ids, bh_hosts_snap=None):
    """Find the subhalo indices for the given BH ID numbers.

    Arguments
    ---------
    run     <int>    : illustris simulation number {1, 3}
    snap    <int>    : illustris snapshot number {0, 135}
    bh_ids   <int>[N] : target BH ID numbers
    verbose <bool>   : optional, print verbose output

    Returns
    -------
    match_subhs <int>[N] : subhalo index numbers (`-1` for invalid)

    """
    log.debug("particle_hosts.bh_subhalos()")
    beg_all = datetime.now()

    # Load (Sub)Halo Offset Table
    # ---------------------------
    if bh_hosts_snap is None:
        log.info("Loading `bh_hosts_snap` for run {}, snap {}".format(run, snap))
        beg = datetime.now()
        bh_hosts_snap = _load_bh_hosts_snap_table(run, snap, log, load_saved=True)
        log.after("Loaded {} entries.".format(len(bh_hosts_snap[OFFTAB.BH_IDS])), beg)

    table_ids  = bh_hosts_snap[OFFTAB.BH_IDS][:]
    table_inds = bh_hosts_snap[OFFTAB.BH_INDICES][:]
    table_subhs = bh_hosts_snap[OFFTAB.BH_SUBHALOS][:]

    # Convert IDs to Indices
    # ----------------------
    log.info("Matching BHs to table")
    beg = datetime.now()
    # Sort IDs for faster searching
    sort_ids = np.argsort(table_ids)
    # Find matches in sorted array
    match_sorted = np.searchsorted(table_ids, bh_ids, sorter=sort_ids)
    #    Not found matches will be set to length of array.  These will be caught as incorrect below
    unmatched = (match_sorted == len(sort_ids))
    log.frac(np.count_nonzero(unmatched), len(sort_ids), "Unmatched sources", lvl=log.DEBUG)
    match_sorted[unmatched] -= 1
    # Reverse map to find matches in original array
    found = sort_ids[match_sorted]

    match_ids   = table_ids[found]
    match_inds  = table_inds[found]
    match_subhs = table_subhs[found]

    # Check Matches
    # -------------

    # Find incorrect matches
    inds = np.where(bh_ids != match_ids)[0]
    num_ids = len(bh_ids)
    num_bad = len(inds)
    num_good = num_ids - num_bad
    # Set incorrect matches to '-1'
    if len(inds) > 0:
        match_ids[inds]  = -1
        match_inds[inds] = -1
        match_subhs[inds] = -1
    log.after("Matched {:d}/{:d} = {:.3f} Good, {:d}/{:d} = {:.3f} Bad".format(
        num_good, num_ids, num_good/num_ids, num_bad, num_ids, num_bad/num_ids), beg, beg_all)
    num_no_subh = np.count_nonzero(match_subhs < 0)
    log.frac(num_no_subh, num_ids, "No Subhalos", lvl=log.INFO)

    return match_subhs


def _load_bh_hosts_table(run, log=None, load_saved=True, version=None):
    """Merge individual snapshot's blackhole hosts files into a single file.

    Arguments
    ---------
    run      <int>  : illustris simulation number {1, 3}
    load_saved <bool> : optional, load existing save if possible
    version  <flt>  : optional, target version number

    Returns
    -------
    bh_hosts <dict> : table of hosts for all snapshots

    """
    # log = check_log(log, run=run)
    log.debug("particle_hosts._load_bh_hosts_table()")
    beg_all = datetime.now()

    fname_bh_hosts = _get_filename_bh_hosts_table(run)
    _path = zio.check_path(fname_bh_hosts)
    if not os.path.isdir(_path):
        log.raise_error("Error with path for '{}' (path: '{}')".format(fname_bh_hosts, _path))

    # Load Existing Save
    # ------------------
    if load_saved:
        log.info("Loading from save '{}'".format(fname_bh_hosts))
        # Make sure path exists
        if os.path.exists(fname_bh_hosts):
            hosts_table = h5py.File(fname_bh_hosts, 'r')
        else:
            log.warning("File '{}' does not Exist.  Reconstructing.".format(fname_bh_hosts))
            load_saved = False

    # Reconstruct Hosts Table
    # -----------------------
    if not load_saved:
        log.info("Constructing Hosts Table")
        COSMO = Illustris_Cosmology_TOS()

        if version is not None:
            log.raise_error("Can only create version '{}'".format(_VERSION))

        # Select the dict-keys for snapshot hosts to transfer
        host_keys = [OFFTAB.BH_IDS, OFFTAB.BH_INDICES, OFFTAB.BH_HALOS, OFFTAB.BH_SUBHALOS]

        # Save To HDF5
        # ------------
        log.info("Writing bh-host table to file '{}'".format(fname_bh_hosts))
        beg = datetime.now()
        with h5py.File(fname_bh_hosts, 'w') as host_table:
            # Metadata
            host_table.attrs[OFFTAB.RUN]         = run
            host_table.attrs[OFFTAB.VERSION]     = _VERSION
            host_table.attrs[OFFTAB.CREATED]     = datetime.now().ctime()
            host_table.attrs[OFFTAB.FILENAME]    = fname_bh_hosts

            for snap in tqdm.trange(COSMO.NUM_SNAPS, desc="Loading snapshots"):
                # Load Snapshot BH-Hosts
                htab_snap = _load_bh_hosts_snap_table(run, snap, log, load_saved=True)
                # Extract and store target data
                snap_str = "{:03d}".format(snap)

                # Create a group for this snapshot
                snap_group = host_table.create_group(snap_str)
                # Transfer all parameters over
                for hkey in host_keys:
                    snap_group.create_dataset(hkey, data=htab_snap[hkey][:])

        log.after("Saved to '{}', size {}".format(
            fname_bh_hosts, zio.get_file_size(fname_bh_hosts)), beg, beg_all, lvl=log.WARNING)
        host_table = h5py.File(fname_bh_hosts, 'r')

    return hosts_table


def _load_bh_hosts_snap_table(run, snap, log, version=None, load_saved=True):
    """Load pre-existing, or manage the creation of the particle offset table.

    Arguments
    ---------
    run      <int>  : illustris simulation number {1, 3}
    snap     <int>  : illustris snapshot number {1, 135}
    load_saved <bool> : optional, load existing table
    verbose  <bool> : optional, print verbose output

    Returns
    -------
    offsetTable <dict> : particle offset table, see `ParticleHosts` docs for more info.

    """
    log.debug("particle_hosts._load_bh_hosts_snap_table()")
    beg_all = datetime.now()

    fname = _get_filename_bh_hosts_snap_table(run, snap)
    _path = zio.check_path(fname)
    if not os.path.isdir(_path):
        log.raise_error("Error with path for '{}' (path: '{}')".format(fname, _path))

    # Load Existing Save
    # ------------------
    if load_saved:
        # fname = FILENAME_BH_HOSTS_SNAP_TABLE(run, snap, version)
        log.info("Loading from save '{}'".format(fname))
        # Make sure path exists
        if os.path.exists(fname):
            host_table = h5py.File(fname, 'r')
        else:
            log.warning("File '{}' does not Exist.  Reconstructing.".format(fname))
            load_saved = False

    # Reconstruct Hosts Table
    # -----------------------
    if not load_saved:
        log.info("Constructing Offset Table for Snap {}".format(snap))
        COSMO = Illustris_Cosmology_TOS()

        if version is not None:
            log.raise_error("Can only create version '{}'".format(_VERSION))

        # Construct Offset Data
        beg = datetime.now()
        halo_nums, subh_nums, offsets = _construct_offset_table(run, snap, log)
        log.after("Loaded {} entries".format(len(halo_nums)), beg, beg_all)

        # Construct BH index Data
        #     Catch errors for bad snapshots
        try:
            bh_inds, bh_ids = _construct_bh_index_table(run, snap, log)
        except:
            # If this is a known bad snapshot, set values to None
            if snap in COSMO.GET_BAD_SNAPS(run):
                log.info("bad snapshot: run {}, snap {}".format(run, snap))
                bh_inds  = None
                bh_ids   = None
                bh_halos = None
                bh_subhs = None
            # If this is not a known problem, still raise error
            else:
                log.error("this is not a known bad snapshot: run {}, snap {}".format(run, snap))
                raise

        # On success, Find BH Subhalos
        else:
            bin_inds = np.digitize(bh_inds, offsets[:, PARTICLE.BH]).astype(DTYPE.INDEX) - 1
            if any(bin_inds < 0):
                log.raise_error("Some bh_inds not matched!! '{}'".format(str(bin_inds)))

            bh_halos = halo_nums[bin_inds]
            bh_subhs = subh_nums[bin_inds]

        # Save To Dict
        # ------------
        log.info("Writing snapshot bh-host table to file '{}'".format(fname))
        beg = datetime.now()
        with h5py.File(fname, 'w') as host_table:
            # Metadata
            host_table.attrs[OFFTAB.RUN]         = run
            host_table.attrs[OFFTAB.SNAP]        = snap
            host_table.attrs[OFFTAB.VERSION]     = _VERSION
            host_table.attrs[OFFTAB.CREATED]     = datetime.now().ctime()
            host_table.attrs[OFFTAB.FILENAME]    = fname

            # BH Data
            host_table.create_dataset(OFFTAB.BH_INDICES,  data=bh_inds)
            host_table.create_dataset(OFFTAB.BH_IDS,      data=bh_ids)
            host_table.create_dataset(OFFTAB.BH_HALOS,    data=bh_halos)
            host_table.create_dataset(OFFTAB.BH_SUBHALOS, data=bh_subhs)

        log.after("Saved to '{}', size {}".format(
            fname, zio.get_file_size(fname)), beg, beg_all, lvl=log.WARNING)
        host_table = h5py.File(fname, 'r')

    return host_table


def _construct_offset_table(run, snap, log):
    """Construct offset table from halo and subhalo catalogs.

    Each 'entry' is the first particle index number for a group of particles.  Particles are
    grouped by the halos and subhalos they belong to.  The first entry is particles in the first
    subhalo of the first halo.  The last entry for the first halo is particles that dont belong to
    any subhalo (but still belong to the first halo).  The very last entry is for particles that
    dont belong to any halo or subhalo.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       snap    <int>  : illustris snapshot number {0, 135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       halo_num <int>[N]   : halo      number for each offset entry
       subh_num <int>[N]   : subhalo   number for each offset entry
       offsets <int>[N, 6] : particle offsets for each offset entry

    """
    log.debug("particle_hosts._construct_offset_table()")

    # Load (Sub)Halo Catalogs
    # -----------------------

    # Illustris Data Directory where catalogs are stored
    base_path = GET_ILLUSTRIS_OUTPUT_DIR(run)

    log.info("Loading Catalogs from '{}'".format(base_path))
    halo_cat = ill.groupcat.loadHalos(base_path, snap, fields=None)
    num_halos = halo_cat['count']
    log.debug("Halos loaded: {:7d}".format(num_halos))
    subh_cat = ill.groupcat.loadSubhalos(base_path, snap, fields=None)
    num_subhs = subh_cat['count']
    log.debug("Subhalos Loaded: {:7d}".format(num_subhs))

    # Initialize Storage
    # ------------------

    table_size = num_halos + num_subhs + 1

    # See object description; recall entries are [HALO, SUBHALO, PART0, ... PART5]
    #    (Sub)halo numbers are smaller, use signed-integers for `-1` to be no (Sub)halo
    halo_num = np.zeros(table_size, dtype=DTYPE.INDEX)
    subh_num = np.zeros(table_size, dtype=DTYPE.INDEX)
    # Offsets approach total number of particles, must be uint64
    offsets = np.zeros([table_size, PARTICLE._NUM], dtype=DTYPE.ID)

    subh = 0
    offs = 0
    cum_halo_parts = np.zeros(PARTICLE._NUM, dtype=DTYPE.ID)
    cum_subh_parts = np.zeros(PARTICLE._NUM, dtype=DTYPE.ID)

    # Iterate Over Each Halo
    # ----------------------
    for ii in tqdm.trange(num_halos, desc="Loading Halos"):

        # Add the number of particles in this halo
        cum_halo_parts[:] += halo_cat['GroupLenType'][ii, :]

        # Iterate over each Subhalo, in halo ``ii``
        # -----------------------------------------
        for jj in range(halo_cat['GroupNsubs'][ii]):

            # Consistency check: make sure subhalo number is as expected
            if (jj == 0) and (subh != halo_cat['GroupFirstSub'][ii]):
                log.error("ii = {:d}, jj = {:d}, subh = {:d}".format(ii, jj, subh))
                log.raise_error("Subhalo iterator doesn't match Halo's first subhalo!")

            # Add entry for each subhalo
            halo_num[offs] = ii
            subh_num[offs] = subh
            offsets[offs, :] = cum_subh_parts

            # Add particles in this subhalo to offset counts
            cum_subh_parts[:] += subh_cat['SubhaloLenType'][subh, :]

            # Increment subhalo and entry number
            subh += 1
            offs += 1

        # Add Entry for particles with NO subhalo
        halo_num[offs] = ii                        # Still part of halo ``ii``
        subh_num[offs] = -1                        # `-1` means no (sub)halo
        offsets[offs, :] = cum_subh_parts

        # Increment particle numbers to include this halo
        cum_subh_parts = np.copy(cum_halo_parts)

        # Increment entry number
        offs += 1

    # Add entry for particles with NO halo and NO subhalo
    halo_num[offs] = -1
    subh_num[offs] = -1
    offsets[offs, :] = cum_subh_parts

    return halo_num, subh_num, offsets


def _construct_bh_index_table(run, snap, log):
    """Load all BH ID numbers and associate them with 'index' (i.e. order) numbers.

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       snap    <int>  : illustris snapshot number {1, 135}
       verbose <bool> : optional, print verbose output

    Returns
    -------
       inds    <int>[N] : BH Index numbers
       bh_ids   <int>[N] : BH particle ID numbers

    """
    log.debug("particle_hosts._construct_bh_index_table()")
    beg = datetime.now()

    # Illustris Data Directory where catalogs are stored
    base_path = GET_ILLUSTRIS_OUTPUT_DIR(run)

    # Load all BH ID numbers from snapshot (single ``fields`` parameters loads array, not dict)
    log.info("Loading BHs from run {}, snap {} : '{}'".format(run, snap, base_path))
    bh_ids = ill.snapshot.loadSubset(base_path, snap, PARTICLE.BH, fields=SNAPSHOT.IDS)
    num_bh = len(bh_ids)
    log.after("BH Loaded: {:7d}".format(num_bh), beg)
    # Create 'indices' of BHs
    inds = np.arange(num_bh)
    return inds, bh_ids


def _get_path_processed(run):
    """

    _ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                              2: "L75n910FP",
                              3: "L75n455FP"}

    _PROCESSED_DIR = "/n/home00/lkelley/illustris/data/{}/output/postprocessing/"

    """
    from .. constants import _ILLUSTRIS_RUN_NAMES
    # Make sure path ends in '/'
    path = os.path.join(_PROCESSED_DIR.format(_ILLUSTRIS_RUN_NAMES[run]), '')
    if zio.check_path(path) is None:
        raise OSError("_get_path_processed(): Path failed '{}'".format(path))
    return path


def _processed_filname(run, snap, version, filename_base):
    if version is None:
        version = _VERSION
    fname = filename_base.format(run=run, snap=snap, version=version)
    fname = os.path.join(_get_path_processed(run), fname)
    return fname


def _get_filename_offset_table(run, snap, version=None):
    fname = _processed_filname(run, snap, version, _FILENAME_OFFSET_TABLE)
    return fname


def _get_filename_bh_hosts_snap_table(run, snap):
    # fname = _processed_filname(run, snap, version, _FILENAME_BH_HOSTS_SNAP_TABLE)
    fname = _FILENAME_BH_HOSTS_SNAP_TABLE.format(run=run, snap=snap)
    fname = os.path.join(_get_path_processed(run), fname)
    return fname


def _get_filename_bh_hosts_table(run):
    # fname = _processed_filname(run, snap, version, _FILENAME_BH_HOSTS_TABLE)
    fname = _FILENAME_BH_HOSTS_TABLE.format(run=run)
    fname = os.path.join(_get_path_processed(run), fname)
    return fname


def main():
    titleStr = "illpy.Subhalos.ParticleHosts.main()"
    print("\n%s\n%s\n" % (titleStr, "="*len(titleStr)))

    import sys

    try:
        run   = np.int(sys.argv[1])
        start = np.int(sys.argv[2])
        stop  = np.int(sys.argv[3])
        skip  = np.int(sys.argv[4])

    except:
        # Print Usage
        print("usage:  ParticleHosts RUN SNAP_START SNAP_STOP SNAP_SKIP")
        print("arguments:")
        print("    RUN        <int> : illustris simulation number {1, 3}")
        print("    SNAP_START <int> : illustris snapshot   number {0, 135} to start on")
        print("    SNAP_STOP  <int> :                                     to stop  before")
        print("    SNAP_SKIP  <int> : spacing of snapshots to work on")
        print("")
        # Raise Exception
        raise

    else:
        snaps = np.arange(start, stop, skip)
        print(snaps)

        for sn in snaps:
            sys.stdout.write('\t%3d ... ' % (sn))
            sys.stdout.flush()

            beg = datetime.now()
            _load_bh_hosts_snap_table(run, sn, convert=0.4, bar=False)
            end = datetime.now()

            sys.stdout.write(' After %s\n' % (str(end-beg)))
            sys.stdout.flush()

    return


if __name__ == "__main__":
    main()
