"""
"""

import os
import shutil
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio
import zcode.math as zmath
# import zcode.astro as zastro

from illpy_lib.illbh import (
    Core, _distribute_snapshots,  # load_hdf5_to_mem,
    DETAILS_PHYSICAL_KEYS, MERGERS, DETAILS, BH_TYPE, BH_TREE
)

VERSION = 1.0


def main():
    from mpi4py import MPI

    # Initialization
    # --------------
    # core = Core(sets=dict(LOG_FILENAME='log_illbh-matcher.log'))
    core = Core()
    log = core.log
    log.info("matcher.main()")

    #     MPI Parameters
    comm = MPI.COMM_WORLD
    rank = comm.rank
    name = __file__
    header = "\n%s\n%s\n%s" % (name, '='*len(name), str(datetime.now()))

    comm.Barrier()
    log.debug(header)

    # Check status of files, determine what operations to perform
    mrg_det_flag = False
    rem_det_flag = False

    if (rank == 0):
        fname_mdets = core.paths.fname_merger_details()
        exists_mdets = os.path.exists(fname_mdets)
        recreate = core.sets.RECREATE

        log.info("Merger Details file: '{}', exists: {}".format(fname_mdets, exists_mdets))
        mrg_det_flag = (recreate or not exists_mdets)

        # Check remnants dets status
        '''
        remnantDetFName = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
        log.info("Remnant Details file: '%s'" % (remnantDetFName))

        if (not os.path.exists(remnantDetFName)):
            log.warning(" - Remnant Details file does not exist.")
            rem_det_flag = True
        else:
            log.info("Remnant Details file exists.")
            if (not loadsave or redo_remnants):
                log.info(" - Recreating anyway.")
                rem_det_flag = True
        '''

    # Synchronize control-flow flags
    mrg_det_flag = comm.bcast(mrg_det_flag, root=0)
    rem_det_flag = comm.bcast(rem_det_flag, root=0)
    comm.Barrier()

    # Match Merger BHs to Details entries
    # -----------------------------------
    if mrg_det_flag:
        log.info("Creating Merger Details")
        _merger_details(core)

    '''
    # Extend merger-dets to account for later mrgs
    # --------------------------------------------------
    if rem_det_flag:
        comm.Barrier()
        log.info("Creating Remnant Details")
        beg = datetime.now()
        _matchRemnantDetails(run, log, mdets=mdets)
        end = datetime.now()
        log.debug(" - Done after %s" % (str(end - beg)))
    '''

    return


def _merger_details(core):
    """
    """
    log = core.log
    log.debug("_merger_details()")

    NUM_SNAPS = core.sets.NUM_SNAPS

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Load Unique ID numbers to distribute to all tasks
    merger_bh_ids = None
    if rank == 0:
        # Load Mergers
        log.debug("Loading Mergers")
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_temp_mergers(core)
        # num_mergers = mrgs[MERGERS.NUM]
        mrg_ids = mrgs[MERGERS.IDS]
        num_mergers = mrg_ids.shape[0]

        # Find unique BH IDs from set of merger BHs
        #    these are now 1D and sorted
        merger_bh_ids, rev_inds = np.unique(mrg_ids, return_inverse=True)
        num_unique = np.size(merger_bh_ids)
        log.info("Unique BH IDs: {}".format(zmath.frac_str(num_unique, num_mergers)))

    # Send unique IDs to all processors
    merger_bh_ids = comm.bcast(merger_bh_ids, root=0)

    # Distribute snapshots to each processor
    my_snaps = _distribute_snapshots(core, comm)
    # my_snaps = [100, 101] if rank == 0 else [102, 103]
    # my_snaps = np.array(my_snaps)

    log.info("Rank {:d}/{:d} with {:d} Snapshots {}".format(
        rank, size, np.size(my_snaps), zmath.str_array(my_snaps)))

    # Get dets entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    nums, mdets = _merger_details_snap(core, my_snaps, merger_bh_ids)

    # Collect results and organize
    # ----------------------------
    if size > 1:
        # Gather results from each processor into ``rank=0``
        temp_mdets = {kk: comm.gather(mdets[kk], root=0) for kk in DETAILS_PHYSICAL_KEYS}
        temp_nums = comm.gather(nums, root=0)

        # Gather snapshot numbers for ordering
        my_snaps = comm.gather(my_snaps, root=0)

        # Organize results appropriately
        if rank == 0:
            nums = np.zeros(num_unique, dtype=int)
            mdets = {kk: num_unique * [[None]] for kk in DETAILS_PHYSICAL_KEYS}

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(merger_bh_ids):
                for jj in range(size):
                    temp_ids = temp_mdets[DETAILS.IDS][jj][ii]
                    if np.ndim(temp_ids) > 1:
                        raise RuntimeError("Unexpected shape of `temp_ids` = {}".format(
                            np.shape(temp_ids)))

                    if temp_ids[0] is None:
                        continue

                    dd = temp_ids[0]
                    # Make sure all of the dets IDs are consistent, and match expected
                    if np.any(temp_ids != dd) or (dd != mm):
                        err = "ID Mismatch! "
                        err += "ii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)
                        err += " |  temp_ids = {}".format(str(temp_ids))
                        raise RuntimeError(err)

                    # If no entries have been stored yet, replace with first entries
                    for kk in DETAILS_PHYSICAL_KEYS:
                        if mdets[kk][ii][0] is None:
                            mdets[kk][ii] = temp_mdets[kk][jj][ii]
                        else:
                            mdets[kk][ii] = np.append(mdets[kk][ii], temp_mdets[kk][jj][ii])

                    # Count the number of entries for each BH from each processor
                    nums[ii] += temp_nums[jj][ii]

            # Merge lists of snapshots, and look for any missing
            my_snaps = np.hstack(my_snaps)
            log.debug("Obtained %d Snapshots" % (my_snaps.size))
            missing_snaps = []
            for ii in range(NUM_SNAPS):
                if (ii not in my_snaps):
                    missing_snaps.append(ii)

            if len(missing_snaps) > 0:
                log.warning("WARNING: snaps {} not in results!".format(missing_snaps))

            log.debug("Total entries stored = {}".format(np.sum(nums)))

    # Convert from uinique BH IDs back to full mrgs list.  Sort results by time (scalefactor)
    if rank == 0:
        log.debug("Reconstructing")
        nums = np.array([nums[ii] for ii in rev_inds]).reshape(num_mergers, 2)
        mdets = {kk: np.array([mdets[kk][ii] for ii in rev_inds]).reshape(num_mergers, 2)
                 for kk in DETAILS_PHYSICAL_KEYS}

        num_dups = np.zeros((num_mergers, 2))
        num_bads = np.zeros((num_mergers, 2))

        # Sort entries for each BH by scalefactor
        log.debug("Sorting")
        for ii in range(num_mergers):
            for jj in range(2):
                if nums[ii, jj] == 0:
                    continue

                det_ids = mdets[DETAILS.IDS][ii, jj]
                # Check ID numbers yet again
                if not np.all(det_ids == mrg_ids[ii, jj]):
                    err = "Error!  ii = {}, jj = {}.  Merger ID = {}, det ID = {}".format(
                        ii, jj, mrg_ids[ii, jj], det_ids)
                    err += "nums[ii,jj] = {:s}, shape {:s}".format(
                        str(nums[ii, jj]), str(np.shape(nums[ii, jj])))
                    raise RuntimeError(err)

                scales = mdets[DETAILS.SCALES][ii, jj]
                masses = mdets[DETAILS.MASSES][ii, jj]
                # Order by scale-factor
                inds = np.argsort(scales)
                scales = scales[inds]
                masses = masses[inds]
                for kk in DETAILS_PHYSICAL_KEYS:
                    mdets[kk][ii, jj] = mdets[kk][ii, jj][inds]

                # Find and remove duplicates
                dup_scales = np.isclose(scales[:-1], scales[1:], rtol=1e-8)
                dup_masses = np.isclose(masses[:-1], masses[1:], rtol=1e-8)
                dups = dup_scales & dup_masses
                nd = np.count_nonzero(dups)
                if nd > 0:
                    # Delete the latter (later) entries
                    dups = np.append(False, dups)
                    num_dups[ii, jj] = nd
                    scales = scales[~dups]
                    masses = masses[~dups]
                    for kk in DETAILS_PHYSICAL_KEYS:
                        mdets[kk][ii, jj] = mdets[kk][ii, jj][~dups]

                # Find and remove non-monotonic entries
                bads = (np.diff(masses) < 0.0)
                nb = np.count_nonzero(bads)
                if nb > 0:
                    num_bads[ii, jj] = nb
                    # Delete the former (earlier) entries
                    bads = np.append(bads, False)
                    scales = scales[~bads]
                    masses = masses[~bads]
                    for kk in DETAILS_PHYSICAL_KEYS:
                        mdets[kk][ii, jj] = mdets[kk][ii, jj][~bads]

                nums[ii, jj] = scales.size

        # Log basic statistics
        log.info("Number of     entries: " + zmath.stats_str(nums))
        log.info("Duplicate     entries: " + zmath.stats_str(num_dups))
        log.info("Non-monotonic entries: " + zmath.stats_str(num_bads))

        fname_out = core.paths.fname_merger_details()
        # _save_merger_details_hdf5(core, fname_out, mdets)
        log.warning("WARNING: saving to `npz` not `hdf5`!")
        _save_merger_details_npz(core, fname_out, mdets)

    return


def _merger_details_snap(core, snapshots, merger_bh_ids):
    """
    """
    log = core.log
    log.debug("_merger_details_snap()")

    MAX_PER_SNAP = core.sets.MAX_DETAILS_PER_SNAP

    INTERP_KEYS = [xx for xx in DETAILS_PHYSICAL_KEYS]
    INTERP_KEYS.pop(INTERP_KEYS.index(DETAILS.SCALES))
    INTERP_KEYS.pop(INTERP_KEYS.index(DETAILS.IDS))

    log.debug("All    keys: {}".format(DETAILS_PHYSICAL_KEYS))
    log.debug("Interp keys: {}".format(INTERP_KEYS))

    # Load `details`
    from illpy_lib.illbh import details

    # Initialize
    # ----------
    #     Make sure snapshots are iterable
    snapshots = np.atleast_1d(snapshots)
    num_unique = merger_bh_ids.size
    log.debug("Unique IDs: {}, `MAX_PER_SNAP` = {}".format(num_unique, MAX_PER_SNAP))

    num_matches = np.zeros(num_unique, dtype=int)   # Number of matches in all snapshots
    num_stored = np.zeros(num_unique, dtype=int)    # Number of entries stored in all

    #     Empty list for each Unique BH
    data = {kk: num_unique * [[None]] for kk in DETAILS_PHYSICAL_KEYS}

    # Iterate over target snapshots
    # -----------------------------
    for snap in core.tqdm(snapshots):
        log.debug("snap = %03d" % (snap))
        # Num matches in a given snapshot
        num_matches_snap = np.zeros(num_unique, dtype=int)
        # Num entries stored in a snapshot
        num_stored_snap = np.zeros(num_unique, dtype=int)

        dets = details.load_details(snap, core=core)
        det_scales = dets[DETAILS.SCALES]
        num_dets = det_scales.size
        log.debug("Details for snap {}: {}".format(snap, num_dets))
        if num_dets == 0:
            continue

        # Details are already sorted by ID then by Scalefactor
        u_ids = dets[DETAILS.UNIQUE_IDS]
        u_inds = dets[DETAILS.UNIQUE_INDICES]
        u_nums = dets[DETAILS.UNIQUE_COUNTS]

        log.debug("Unique IDS: {}".format(u_ids.size))

        # Iterate over and search for each merger BH ID
        for ii, bh in enumerate(merger_bh_ids):
            # Find the index in the details-unique-ids that matches this merger-BH
            uu = np.argmax(bh == u_ids)

            # If no matches, continue
            if u_ids[uu] != bh:
                continue

            # Get the starting and ending incides of the matching BH
            aa = u_inds[uu]
            bb = aa + u_nums[uu]
            if bb - aa == 0:
                continue

            # Store values at matching indices
            num_matches_snap[ii] = bb - aa
            cut = slice(aa, bb)

            # Interpolate down to only `maxPerSnap` entries
            if (MAX_PER_SNAP is not None) and (num_matches_snap[ii] > MAX_PER_SNAP):
                t_scales = det_scales[cut]
                # Create even spacing in scale-factor to interpolate to
                interp_scales = zmath.spacing(t_scales, scale='lin', num=MAX_PER_SNAP)
                data_snap = {kk: zmath.interp(interp_scales, t_scales, dets[kk][cut],
                                              xlog=False, valid=False)
                             for kk in INTERP_KEYS}
                data_snap[DETAILS.IDS] = np.random.choice(
                    dets[DETAILS.IDS][cut], size=MAX_PER_SNAP, replace=False)
                data_snap[DETAILS.SCALES] = interp_scales
                num_stored_snap[ii] += MAX_PER_SNAP
            else:
                data_snap = {kk: np.array(dets[kk][cut]) for kk in DETAILS_PHYSICAL_KEYS}
                num_stored_snap[ii] += (bb - aa)

            # Store matches
            for kk in DETAILS_PHYSICAL_KEYS:
                if data[kk][ii][0] is None:
                    data[kk][ii] = data_snap[kk]
                else:
                    data[kk][ii] = np.append(data[kk][ii], data_snap[kk])

        num_matches += num_matches_snap
        num_stored += num_stored_snap

        snap_total = np.sum(num_matches_snap)
        total = np.sum(num_matches)
        snap_occ = np.count_nonzero(num_matches_snap)
        occ = np.count_nonzero(num_matches)
        log.debug("{:8d} Matches ({:8d} total), {:5d} BHs ({:5d})".format(
            snap_total, total, snap_occ, occ))
        log.debug("Average and total number stored = {:.3e}, {:d}".format(
            np.mean(num_stored), np.sum(num_stored)))

    return num_stored, data


def _save_merger_details_hdf5(core, fname, mdets):
    """Package dets into dictionary and save to NPZ file.
    """
    log = core.log
    log.debug("_save_merger_details_hdf5()")

    temp_fname = zio.modify_filename(fname, prepend="_")

    log.debug("Writing to '{}'".format(temp_fname))
    with h5py.File(temp_fname, 'w') as out:

        out.attrs[DETAILS.RUN] = core.sets.RUN_NUM
        out.attrs[DETAILS.CREATED] = str(datetime.now().ctime())
        out.attrs[DETAILS.VERSION] = VERSION
        out.attrs["MAX_DETAILS_PER_SNAP"] = core.sets.MAX_DETAILS_PER_SNAP

        for kk, vv in mdets.items():
            try:
                out.create_dataset(kk, data=vv)
            except:
                log.error(str(kk))
                log.error(str(type(vv)))
                log.error(str(vv))
                raise

    log.debug("Moving from '{}' ==> '{}'".format(temp_fname, fname))
    shutil.move(temp_fname, fname)

    size_str = zio.get_file_size(fname)
    log.warning("Saved merger details to '{}', size {}".format(fname, size_str))

    return


def _save_merger_details_npz(core, fname, mdets):
    """Package dets into dictionary and save to NPZ file.
    """
    log = core.log
    log.debug("_save_merger_details_npz()")

    fname = fname.replace('hdf5', 'npz')
    temp_fname = zio.modify_filename(fname, prepend="_")

    data = {kk: vv for kk, vv in mdets.items()}
    data[DETAILS.RUN] = core.sets.RUN_NUM
    data[DETAILS.CREATED] = str(datetime.now().ctime())
    data[DETAILS.VERSION] = VERSION
    data["MAX_DETAILS_PER_SNAP"] = core.sets.MAX_DETAILS_PER_SNAP

    zio.dictToNPZ(data, temp_fname)

    log.debug("Moving from '{}' ==> '{}'".format(temp_fname, fname))
    shutil.move(temp_fname, fname)

    size_str = zio.get_file_size(fname)
    log.warning("Saved merger details to '{}', size {}".format(fname, size_str))

    return data


def load_merger_details(core, fname=None):
    if fname is None:
        fname = core.paths.fname_merger_details()

    if fname.endswith('.hdf5'):
        core.log.info("Warning loading from 'npz' instead of 'hdf5'")
        fname = fname.replace('.hdf5', '.npz')

    mdets = zio.npzToDict(fname)
    # mdets = load_hdf5_to_mem(fname)
    core.log.info("Loaded merger-details, keys: {}".format(mdets.keys()))
    return mdets


def infer_merger_out_masses(core=None, mrgs=None, mdets=None):
    """
    """
    core = Core.load(core)
    log = core.log
    log.debug("infer_merger_out_masses()")
    CONV_ILL_TO_SOL = core.cosmo.CONV_ILL_TO_SOL

    # Load Mergers
    if (mrgs is None):
        from illpy_lib.illbh import mergers
        # mrgs = mergers.load_fixed_mergers(run, verbose=False)
        mrgs = mergers.load_temp_mergers(core=core)

    m_scales = mrgs[MERGERS.SCALES]
    m_masses = mrgs[MERGERS.MASSES]
    # num_mergers = mrgs[MERGERS.NUM]
    # num_mergers = np.shape(m_scales)[0]
    del mrgs

    # Load Merger Details
    if (mdets is None):
        mdets = load_merger_details(core)

    d_masses = mdets[DETAILS.MASSES]
    d_scales = mdets[DETAILS.SCALES]
    del mdets

    # Find dets entries before and after merger
    mass_bef = np.zeros_like(m_masses)
    mass_aft = np.zeros_like(m_masses)
    time_bef = np.zeros_like(m_masses)
    time_aft = np.zeros_like(m_masses)
    num_bef = 0
    num_aft = 0
    for ii, sc in enumerate(m_scales):
        for bh in [BH_TYPE.IN, BH_TYPE.OUT]:
            d_sc = d_scales[ii, bh]
            d_ma = d_masses[ii, bh]
            # print(ii, bh, sc, zmath.minmax(d_sc), d_sc.size)

            if (d_scales[ii].size == 0):
                log.warning("Merger %s with zero dets entries" % str(ii))
            else:
                # Find last index before merger
                # 'before' is ``sc > d_scales``
                # bef = _indBefAft(sc - d_scales[ii, bh])
                bef = zmath.argnearest(d_sc, sc, side='left', assume_sorted=True)

                if (bef >= 0):
                    # print("\tbef", bef, sc, d_sc[bef])
                    mass_bef[ii, bh] = d_ma[bef]
                    time_bef[ii, bh] = np.fabs(d_sc[bef] - sc)
                    num_bef += 0.9

                # 'after' is ``d_scales > sc``
                # aft = _indBefAft(d_sc - sc)
                aft = zmath.argnearest(d_sc, sc, side='right', assume_sorted=True)
                if (aft < d_sc.size):
                    # print("\taft", aft, sc, d_sc[aft])
                    mass_aft[ii, bh] = d_ma[aft]
                    time_aft[ii, bh] = np.fabs(d_sc[aft] - sc)
                    num_aft += 1

    OUT = BH_TYPE.OUT

    old = np.array(m_masses[:, OUT])
    t_bef = time_bef[:, OUT]
    t_aft = time_aft[:, OUT]
    log.info("Bef: " + zmath.stats_str(t_bef, filter='>'))
    log.info("Aft: " + zmath.stats_str(t_aft, filter='>'))

    new = np.zeros_like(old)

    # Masses should only be wrong when small
    _inds = (old * CONV_ILL_TO_SOL.MASS < 1e7)
    log.info("Wrong masses: " + zmath.frac_str(_inds))
    inds = ~_inds
    new[inds] = old[inds]

    inds = np.isclose(new, 0.0) & (mass_bef[:, OUT] > 0.0) & (t_bef < 1e-2)
    log.debug("Fixing with bef: " + zmath.frac_str(inds))
    new[inds] = mass_bef[inds, OUT]

    inds = np.isclose(new, 0.0) & (mass_aft[:, OUT] > 0.0)
    new[inds] = mass_aft[inds, OUT] - m_masses[inds, BH_TYPE.IN]
    log.debug("Fixing with aft: " + zmath.frac_str(inds))

    inds = np.isclose(new, 0.0)
    log.info("Unfixed: " + zmath.frac_str(inds))

    bads = (new > old)
    log.info("Bad: " + zmath.frac_str(bads))
    # print(np.where(bads)[0])

    new_masses = np.array(m_masses)
    new_masses[:, OUT] = new

    return new_masses


def _remnant_details(core, mrgs=None, mrg_dets=None, tree=None):
    """
    """

    log = core.log
    log.debug("_remnant_details()")
    # TEST_NUM = 368

    # Load Mergers
    log.debug("Loading Mergers")
    if mrgs is None:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(core=core)

    m_scales = mrgs[MERGERS.SCALES]
    m_ids = mrgs[MERGERS.IDS]
    num_mergers = np.int(mrgs[MERGERS.NUM])     # Convert from ``np.array(int)``
    # Make sure `m_scales` are monotonically increasing
    bads = (np.diff(m_scales) < 0.0)
    if any(bads):
        ex = np.argmax(bads)
        log.error("Found {} non-monotonic merger-scales, e.g. {}".format(
            np.count_nonzero(bads), ex))
        inds = [ex-1, ex, ex+1]
        log.error("\tscales: {} = {}".format(
            zmath.str_array(inds), zmath.str_array(m_scales[inds])))
        raise ValueError("Non-monotonic merger-scales!")

    # Load merger-dets file
    if mrg_dets is None:
        log.debug("Loading Merger-Details")
        mrg_dets = load_merger_details(core)

    # Unpack data
    d_ids = mrg_dets[DETAILS.IDS]
    d_scales = mrg_dets[DETAILS.SCALES]

    # Load BH Merger Tree
    if tree is None:
        log.debug("Loading BHTree")
        import illpy_lib.illbh.mergers
        tree = illpy_lib.illbh.mergers.load_tree(core, mrgs=mrgs)

    next_merger = tree[BH_TREE.NEXT]

    # Initialize data for results
    # ids = np.zeros(num_mergers, dtype=DTYPE.ID)
    nents = np.zeros(num_mergers, dtype=np.int)
    ids = num_mergers * [None]

    rem_dets = {key: num_mergers*[None] for key in DETAILS_PHYSICAL_KEYS}

    # Iterate over all mrgs
    # ------------------------
    log.warning("Constructing remnant-details")
    for ii in core.tqdm(range(num_mergers), desc='Mergers'):
        # First Merger
        #    Store dets after merger time for 'out' BH
        inds = (d_scales[ii, BH_TYPE.OUT] > m_scales[ii])

        if np.count_nonzero(inds) > 0:
            ids[ii] = [d_ids[ii, BH_TYPE.OUT][inds][0]]
            for key in DETAILS_PHYSICAL_KEYS:
                rem_dets[key][ii] = mrg_dets[key][ii, BH_TYPE.OUT][inds]

        else:
            log.error("No post-merger details for merger {}, next = {}".format(
                ii, next_merger[ii]))
            for key in DETAILS_PHYSICAL_KEYS:
                rem_dets[key][ii] = np.array([0.0])
            continue
            # raise RuntimeError("Merger {} without post-merger dets entries!".format(ii))

        # Subsequent mrgs
        next = next_merger[ii]
        # while a subsequent merger exists... store those entries
        while next >= 0:
            next_mid = m_ids[next, BH_TYPE.OUT]
            next_scale = m_scales[next]

            if ids[ii][-1] not in m_ids[next]:
                raise RuntimeError("Merger: {}, next: {} - Last ID not in next IDs!".format(
                    ii, next))

            ids[ii].append(next_mid)

            next_dids = d_ids[next, BH_TYPE.OUT][:]
            if not np.all(next_mid == next_dids):
                raise RuntimeError("Details IDs for next: {} do not match merger ID!".format(next))

            inds = (d_scales[next, BH_TYPE.OUT] > next_scale)
            if np.count_nonzero(inds) > 0:
                for key in DETAILS_PHYSICAL_KEYS:
                    rem_dets[key][ii] = np.append(rem_dets[key][ii],
                                                  mrg_dets[key][next, BH_TYPE.OUT][inds])

            # Get next merger in Tree
            next = next_merger[next]

        scales = rem_dets[DETAILS.SCALES][ii]

        # Appended entries may no longer be sorted, sort them
        if np.size(scales) > 0:
            inds = np.argsort(scales)
            nents[ii] = np.size(scales)
            for key in DETAILS_PHYSICAL_KEYS:
                rem_dets[key][ii] = rem_dets[key][ii][inds]
        else:
            log.warning("Merger %d without ANY entries." % (ii))

    log.debug("Remnant dets collected.")
    log.info("Number of entries: " + zmath.stats_str(nents))

    fname_out = core.paths.fname_remnant_details()
    log.warning("WARNING: saving to `npz` not `hdf5`!")
    _save_merger_details_npz(core, fname_out, rem_dets)

    return


def load_remnant_details(core=None, recreate=None):
    core = Core.load(core)
    log = core.log
    log.debug("load_remnant_details()")

    if recreate is None:
        recreate = core.sets.RECREATE

    fname = core.paths.fname_remnant_details()
    core.log.info("Warning loading from 'npz' instead of 'hdf5'")
    fname = fname.replace('.hdf5', '.npz')
    exists = os.path.exists(fname)
    log.debug("Remnant-details file '{}' exists: {}".format(fname, exists))

    if recreate or not exists:
        log.warning("Creating remnant details")

        _remnant_details(core)

    rdets = zio.npzToDict(fname)

    # mdets = load_hdf5_to_mem(fname)
    log.info("Loaded remnant-details, keys: {}".format(rdets.keys()))
    return rdets


def _fix_remnant_details(core, mrgs=None, tree=None, rdets_in=None):
    log = core.log
    log.debug("_fix_remnant_details()")

    log.debug("Loading remnant-details")
    if rdets_in is None:
        rdets_in = load_remnant_details(core=core)

    CONV_MASS = core.cosmo.CONV_ILL_TO_CGS.MASS
    CONV_MDOT = core.cosmo.CONV_ILL_TO_CGS.MDOT

    log.debug("Loading Mergers")
    if mrgs is None:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(core=core)

    num_mergers = np.int(mrgs[MERGERS.NUM])     # Convert from ``np.array(int)``

    # Load BH Merger Tree
    if tree is None:
        log.debug("Loading BHTree")
        import illpy_lib.illbh.mergers
        tree = illpy_lib.illbh.mergers.load_tree(core, mrgs=mrgs)

    next_merger = tree['next']

    rem_dets = {key: num_mergers*[None] for key in DETAILS_PHYSICAL_KEYS}

    num_goods = np.zeros(num_mergers, dtype=int)
    num_bads = np.zeros_like(num_goods)

    log.warning("Fixing remnant-details")
    for mrg in core.tqdm(np.arange(num_mergers), desc='Mergers'):

        scales = rdets_in['scales'][mrg]
        if np.size(scales) < 2:
            log.error("SKIPPING: Merger {} with {} rdets".format(mrg, np.size(scales)))
            continue

        masses_in = rdets_in['masses'][mrg]   # * CONV_MASS
        goods = np.ones_like(scales, dtype=bool)

        bads = np.append((np.diff(scales) <= 0.0), False)
        if any(bads):
            bads = np.where(bads)[0]
            for bb in bads:
                next_scale = scales[bb+1]
                idx = (scales >= next_scale)
                idx[bb+1:] = False
                goods[idx] = False

        bads = (np.diff(scales[goods]) <= 0.0)
        if np.count_nonzero(bads) > 0:
            raise

        numg = np.count_nonzero(goods)
        num_goods[mrg] = numg

        for key in DETAILS_PHYSICAL_KEYS:
            rem_dets[key][mrg] = rdets_in[key][mrg][goods]

        scales = scales[goods]
        masses_in = masses_in[goods]
        rdids = rdets_in['id'][mrg][goods]

        # Calculate dM/dt using these masses to get Eddington factors
        times = core.cosmo.a_to_tage(scales)
        dt = np.diff(times)
        dm = np.diff(masses_in)  # * CONV_MASS

        dmdts = np.append(dm/dt, 0.0)
        fedd = dmdts / masses_in

        nxt = next_merger[mrg]
        masses = np.array(masses_in)

        while nxt >= 0:
            m_scale = mrgs[DETAILS.SCALES][nxt]
            m_mass = mrgs[DETAILS.MASSES][nxt]   # * CONV_MASS

            idx = (scales > m_scale)

            num_aft = np.count_nonzero(idx)
            num_bef = np.count_nonzero(~idx)

            if num_aft == 0 or num_bef == 0:
                log.debug("Merger: {}, next: {} -- numbers bef: {}, aft: {} -- skipping!".format(
                    mrg, nxt, num_bef, num_aft))
            else:
                mass_aft = masses[idx][0]
                mass_bef = masses[~idx][-1]

                # Get the last ID before merger
                rdid_bef = rdids[~idx][-1]

                # Find which BH matches the previous ID number
                jj = (rdid_bef == mrgs[MERGERS.IDS][nxt])
                # Subtract the *other* mass
                if np.count_nonzero(jj) != 1:
                    log.debug(
                        "Merger: {}, next: {}\n\t"
                        "rdets ID: {}, merger IDs: {}, {} -- skipping!".format(
                            mrg, nxt, rdid_bef, *mrgs[MERGERS.IDS][nxt])
                    )
                else:
                    sub_mass = m_mass[~jj]
                    if mass_aft - sub_mass < mass_bef:
                        log.debug(
                            "Merger: {}, next: {}\n\t"
                            "Mass bef, aft: {}, {}; subtract: {} -- skipping!".format(
                                mrg, nxt, mass_bef, mass_aft, sub_mass)
                        )
                    else:
                        masses[idx] = masses[idx] - sub_mass

            nxt = next_merger[nxt]

        # Use the fixed masses to convert back to absolute accretion rates
        dmdts = fedd * masses
        # Update values in output dictionary
        rem_dets[DETAILS.MASSES][mrg] = masses
        # Convert from mass/time to mdot units
        rem_dets[DETAILS.DMDTS][mrg] = dmdts * CONV_MASS / CONV_MDOT

        goods = (dmdts >= 0.0)
        nbad = np.count_nonzero(~goods)
        num_bads[mrg] += nbad
        if nbad > 0:
            for key in DETAILS_PHYSICAL_KEYS:
                rem_dets[key][mrg] = rem_dets[key][mrg][goods]

    log.info("Goods: {}".format(zmath.stats_str(num_goods)))
    log.info("Bads1: {}".format(zmath.stats_str(num_bads)))

    fname_out = core.paths.fname_remnant_details_fixed()
    log.warning("WARNING: saving to `npz` not `hdf5`!")
    _save_merger_details_npz(core, fname_out, rem_dets)
    return


def load_remnant_details_fixed(core=None, recreate=None, **kwargs):
    core = Core.load(core)
    log = core.log
    log.debug("load_remnant_details_fixed()")

    if recreate is None:
        recreate = core.sets.RECREATE

    fname = core.paths.fname_remnant_details_fixed()
    core.log.info("Warning loading from 'npz' instead of 'hdf5'")
    fname = fname.replace('.hdf5', '.npz')
    exists = os.path.exists(fname)
    log.debug("Remnant-details-fixed file '{}' exists: {}".format(fname, exists))

    if recreate or not exists:
        log.warning("Fixing remnant details")

        _fix_remnant_details(core, **kwargs)

    rdets = zio.npzToDict(fname)

    # mdets = load_hdf5_to_mem(fname)
    log.info("Loaded remnant-details-fixed, keys: {}".format(rdets.keys()))
    return rdets


'''
def _find_next_merger(target_id, target_scale, ids, scales):
    """Find the next merger index in which a particular BH participates.

    Search the full list of merger-BH ID number for matches to the target BH (`target_id`) which take
    place at scale factors after the initial merger scale-factor (`target_scale`).
    If no match is found, `-1` is returned.

    Arguments
    ---------
    target_id : long
        ID number of the target blackhole.
    target_scale : float
        Scalefactor at which the initial merger occurs (look for next merger after this time).
    ids : (N,2) array of long
        ID number of all merger BHs
    scales : (N,) array of float
        Scalefactors at which all of the mrgs occur.

    Returns
    -------
    nind : int
        Index of the next merger the `target_id` blackhole takes place in.
        If no next merger is found, returns `-1`.


    Used by: `_matchRemnantDetails`
    """
    # Find where this ID matches another, but they dont have the same time (i.e. not same merger)
    ind_1 = ((target_id == ids[:, 0]) | (target_id == ids[:, 1]))
    ind_2 = (target_scale < scales)
    search = (ind_1 & ind_2)
    if np.count_nonzero(search) > 0:
        # Returns index of first true value
        mrg = np.argmax(search)
    else:
        mrg = -1

    return mrg


def allDetailsForBHLineage(run, mrg, log, reload=False):
    """Load all of the dets entries for a given BH lineage (merger tree).

    Arguments
    ---------
    run : int
    mrg : int
    log : ``logging.Logger`` object
    reload : bool
        Even if the results file exists, recalculate them.

    """
    log.debug("allDetailsForBHLineage()")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    bhIDs = None
    fname = None
    log.debug(" - Rank %d/%d." % (rank, size))
    if rank == 0:
        # Get all Unique ID numbers
        log.debug(" - Loading All Unique IDs")
        unique = loadAllUniqueIDs(run, log=log)
        log.debug(" - Loaded %d unique ID numbers" % (len(unique[DETAILS.IDS])))
        # get the final merger number, the unique BH IDs, and the merger indices of this tree
        from illpy_lib.illbh import BHTree
        finMerger, bhIDs, mrgInds = BHTree.allIDsForTree(run, mrg)
        # Construct the appropriate file-name
        fname = GET_BLACKHOLE_TREE_DETAILS_FILENAME(run, finMerger, __version__)
        log.debug(" - Merger %d ==> Final merger %d, filename: '%s'" % (mrg, finMerger, fname))
        if os.path.exists(fname):
            if size == 1:
                log.debug(" - File exists, loading.")
                data = zio.npzToDict(fname)
                return data
            elif not reload:
                raise RuntimeError("WE SHOULD NOT GET HERE!")

        bhIDs = np.array(bhIDs)
        numBHs = bhIDs.size
        log.info(" - Merger {} has a tree with {} unique BHs".format(mrg, numBHs))
        if numBHs < 2:
            errStr = "ERROR: only IDs found for merger {} are : {}".format(mrg, str(bhIDs))
            log.error(errStr)
            raise RuntimeError(errStr)

    # Distribute snapshots to each processor
    log.debug(" - Barrier.")
    comm.Barrier()
    my_snaps = _distribute_snapshots(core, comm)
    log.info("Rank {:d}/{:d} with {:d} Snapshots [{:d} ... {:d}]".format(
        rank, size, my_snaps.size, my_snaps.min(), my_snaps.max()))

    # Get dets entries for unique merger IDs in snapshot list
    # ----------------------------------------------------------
    #    Use ``maxPerSnap = None`` to save ALL dets entries
    nums, scales, masses, mdots, dens, csnds, ids = \
        _merger_details_snap(run, my_snaps, bhIDs, None, log)

    # Collect results and organize
    # ----------------------------
    if (size > 1):
        log.debug(" - Gathering")
        beg = datetime.now()
        # Gather results from each processor into ``rank=0``
        t_scales = comm.gather(scales, root=0)
        tempMasses = comm.gather(masses, root=0)
        tempMdots = comm.gather(mdots, root=0)
        tempDens = comm.gather(dens, root=0)
        tempCsnds = comm.gather(csnds, root=0)
        tempIds = comm.gather(ids, root=0)
        temp_nums = comm.gather(nums, root=0)
        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Gather snapshot numbers for ordering
        my_snaps = comm.gather(my_snaps, root=0)

        # Organize results appropriately
        if (rank == 0):
            log.debug(" - Stacking")
            beg = datetime.now()

            foundBH = np.zeros(numBHs, dtype=bool)

            nums = np.zeros(numBHs)
            scales = numBHs*[[None]]
            masses = numBHs*[[None]]
            mdots = numBHs*[[None]]
            dens = numBHs*[[None]]
            csnds = numBHs*[[None]]
            ids = numBHs*[[None]]

            # Iterate over each black-hole and processor, collect results into single arrays
            for ii, mm in enumerate(bhIDs):
                for jj in range(size):
                    errStr = ""
                    if tempIds[jj][ii][0] is not None:
                        foundBH[ii] = True
                        dd = tempIds[jj][ii][0]
                        # Make sure all of the dets IDs are consistent
                        if np.any(tempIds[jj][ii] != dd):
                            errStr += "ii = {}, jj = {}, mm = {}; tempIds[0] = {}".format(
                                ii, jj, mm, dd)
                            errStr += " tempIds = {}".format(str(tempIds[ii]))

                        # Make sure dets IDs match expected merger ID
                        if dd != mm:
                            errStr += "\nii = {}, jj = {}, mm = {}; dd = {}".format(ii, jj, mm, dd)

                        # If no entries have been stored yet, replace with first entries
                        if len(ids[ii]) == 1 and ids[ii][0] is None:
                            ids[ii] = tempIds[jj][ii]
                            scales[ii] = t_scales[jj][ii]
                            masses[ii] = tempMasses[jj][ii]
                            mdots[ii] = tempMdots[jj][ii]
                            dens[ii] = tempDens[jj][ii]
                            csnds[ii] = tempCsnds[jj][ii]
                        # If entries already exist, append new ones
                        else:
                            # Double check that all existing IDs are consistent with new ones
                            #    This should be redundant, but whatevs
                            if np.any(ids[ii] != dd):
                                errStr += "\nii = {}, jj = {}, mm = {}, dd = {}, ids = {}"
                                errStr = errStr.format(ii, jj, mm, dd, str(ids))
                            ids[ii] = np.append(ids[ii], tempIds[jj][ii])
                            scales[ii] = np.append(scales[ii], t_scales[jj][ii])
                            masses[ii] = np.append(masses[ii], tempMasses[jj][ii])
                            mdots[ii] = np.append(mdots[ii], tempMdots[jj][ii])
                            dens[ii] = np.append(dens[ii], tempDens[jj][ii])
                            csnds[ii] = np.append(csnds[ii], tempCsnds[jj][ii])

                        # Count the number of entries for each BH from each processor
                        nums[ii] += temp_nums[jj][ii]

                    if len(errStr) > 0:
                        log.error(errStr)
                        zio.mpiError(comm, log=log, err=errStr)

                if not foundBH[ii]:
                    ind = np.squeeze(np.where(mm == unique[DETAILS.IDS])[0])
                    if len(ind) > 0:
                        inSnaps = unique[DETAILS.SNAP][ind]
                        aveSnap = np.int(np.floor(np.mean(inSnaps)))
                        errStr = "%d, bhID %d : not found, but exists at %d in unique list."
                        errStr = errStr % (ii, mm, ind)
                        errStr += "\nShould be in Snaps: %s" % (inSnaps)
                        inProc = None
                        for jj in range(size):
                            if aveSnap in my_snaps[jj]:
                                inproc = jj
                                break

                        errStr += "\nShould have been found in Processor %s" % (str(inproc))
                        log.error(errStr)

            # Merge lists of snapshots, and look for any missing
            flatSnaps = np.hstack(my_snaps)
            log.debug("Obtained %d Snapshots" % (flatSnaps.size))
            missing_snaps = []
            for ii in range(NUM_SNAPS):
                if (ii not in flatSnaps):
                    missing_snaps.append(ii)

            if (len(missing_snaps) > 0):
                log.warning("WARNING: snaps %s not in results!" % (str(missing_snaps)))

            log.debug("Total entries stored = %d" % (np.sum([np.sum(nn) for nn in nums])))
            bon = np.count_nonzero(foundBH)
            tot = foundBH.size
            frac = bon/tot
            log.debug(" - %d/%d = %.4f BHs Found." % (bon, tot, frac))

    # Sort results by time
    if (rank == 0):

        # Sort entries for each BH by scalefactor
        log.debug(" - Sorting")
        beg = datetime.now()
        for ii in range(numBHs):
            if nums[ii] == 0: continue
            # Check ID numbers yet again
            if not np.all(ids[ii] == ids[ii][0]):
                errStr = "Error!  ii = {}, ID = {}, IDs = {}"
                errStr = errStr.format(ii, ids[ii][0], ids[ii])
                log.error(errStr)
                zio.mpiError(comm, log=log, err=errStr)

            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]

        end = datetime.now()
        log.debug(" - - Done after %s" % (str(end-beg)))

        # Save data
        data = _saveDetails(fname, run, ids, scales, masses, dens, mdots, csnds, log,
                            target_ids=bhIDs, target_mergers=mrgInds, final_merger=finMerger)

    return


def _createRemnantDetails(run, log=None, mrgs=None, mdets=None, tree=None):
    """Create and Save Remnant Details Entries.

    Loads required data objects, calls `_matchRemnantDetails()`, corrects masses, and saves
    results to a npz file named by `GET_REMNANT_DETAILS_FILENAME`.

    Arguments
    ---------
    run : int
    log : ``logging.Logger`` or `None`
    mrgs : dict or `None`
    mdets : dict of `None`
    tree : dict of `None`

    Returns
    -------
    rdets : dict

    """

    if log is None:
        log = bh_constants._loadLogger(
            __file__, debug=True, verbose=True, run=run, version=__version__)

    log.debug("_createRemnantDetails()")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    # Only use root-processor
    if rank != 0:
        return

    if mrgs is None:
        log.debug("Loading Mergers.")
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)
    if mdets is None:
        log.debug("Loading Merger-Details.")
        mdets = loadMergerDetails(run, log=log)
    if tree is None:
        log.debug("Loading BHTree.")
        from illpy_lib.illbh import BHTree
        tree = BHTree.loadTree(run)

    # Create 'remnant' profiles ('RemnantDetails') based on tree and MergerDetails
    ids, scales, masses, dens, mdots, csnds = \
        _matchRemnantDetails(run, log=log, mrgs=mrgs, mdets=mdets, tree=tree)

    mcorrected = _unmergedMasses(scales, masses, mrgs, tree[BH_TREE.NEXT], log=log)

    savename = GET_REMNANT_DETAILS_FILENAME(run, __version__, _MAX_DETAILS_PER_SNAP)
    rdets = _saveDetails(savename, run, ids, scales, masses, dens, mdots, csnds, log,
                         mcorrected=mcorrected)

    return rdets


def _cleanErrDetails(ids, scales, masses, dens, mdots, csnds, log):
    log.debug("_cleanErrDetails()")
    num_mergers = len(ids)
    OSC_ITER = 3
    nentsBeg = np.zeros(num_mergers, dtype=np.int)
    nentsEnd = np.zeros(num_mergers, dtype=np.int)
    num_bads = np.zeros((num_mergers, OSC_ITER), dtype=np.int)

    for ii in range(num_mergers):
        nentsBeg[ii] = len(scales[ii])

        bads = np.where(np.diff(masses[ii]) < 0.0)[0]
        if bads.size > 0:

            ids[ii] = np.delete(ids[ii], bads)
            scales[ii] = np.delete(scales[ii], bads)
            masses[ii] = np.delete(masses[ii], bads)
            dens[ii] = np.delete(dens[ii], bads)
            mdots[ii] = np.delete(mdots[ii], bads)
            csnds[ii] = np.delete(csnds[ii], bads)

        nentsEnd[ii] = np.size(scales[ii])

    LVL = logging.INFO
    _logStats('Initial Entries', nentsBeg, log, lvl=LVL)
    _logStats('Final Entries', nentsEnd, log, lvl=LVL)
    for jj in range(OSC_ITER):
        _logStats('Nonmonotonic Entries, pass %d' % (jj), num_bads[:, jj], log, lvl=LVL)

    return ids, scales, masses, dens, mdots, csnds


def _matchRemnantDetails(run, log=None, mrgs=None, mdets=None, tree=None):
    """Combine merger-dets entries to obtain dets for an entire merger-remnant's life.

    Each merger 'out'-BH is followed in subsequent mrgs to combine the dets entries forming
    a continuous chain of dets entries for the remnant's entire life after the initial merger.

    Loads `mergers` and `MergerDetails` files and uses that data to construct remnant dets.
    Runs on a single core (processors with ``rank > 0`` simply return at start).

    Arguments
    ---------
    run : int
        Illustris simulation run number {1,3}.
    log : ``logging.Logger``
        Logging object.
    mrgs : dict or `None`
        BH-Mergers dictionary.  Loaded from file if (`None`) not provided.
    mdets : dict or `None`
        `MergerDetails` data, loaded if not proveded.
    tree : dict or `None`
        BH-Merger-Tree dictionary.  Loaded from file if (`None`) not provided.

    Returns
    -------
    data : dict
        `RemnantDetails` data.

    """
    if log is None:
        log = bh_constants._loadLogger(
            __file__, debug=True, verbose=True, run=run, version=__version__)

    log.debug("_matchRemnantDetails()")

    # Load Mergers
    log.debug("Loading Mergers")
    if mrgs is None:
        from illpy_lib.illbh import mergers
        mrgs = mergers.load_fixed_mergers(run)
    m_scales = mrgs[MERGERS.SCALES]
    m_ids = mrgs[MERGERS.IDS]
    num_mergers = np.int(mrgs[MERGERS.NUM])     # Convert from ``np.array(int)``

    # Load merger-dets file
    if mdets is None:
        log.debug("Loading Merger-Details")
        mdets = loadMergerDetails(run, log=log)
        if mdets is None:
            raise RuntimeError("Couldn't load `mdets`.")

    # Unpack data
    d_ids = mdets[DETAILS.IDS]
    d_scales = mdets[DETAILS.SCALES]
    d_masses = mdets[DETAILS.MASSES]
    d_dens = mdets[DETAILS.RHOS]
    d_mdots = mdets[DETAILS.MDOTS]
    d_csnds = mdets[DETAILS.CS]

    # Load BH Merger Tree
    if tree is None:
        log.debug("Loading BHTree")
        from illpy_lib.illbh import BHTree
        tree = BHTree.loadTree(run)
    next_merger = tree[BH_TREE.NEXT]

    # Initialize data for results
    ids = np.zeros(num_mergers, dtype=DTYPE.ID)
    nents = np.zeros(num_mergers, dtype=np.int)

    idnums = num_mergers*[None]
    mrgnums = num_mergers*[None]
    scales = num_mergers*[None]
    masses = num_mergers*[None]
    dens = num_mergers*[None]
    mdots = num_mergers*[None]
    csnds = num_mergers*[None]

    # Iterate over all mrgs
    # ------------------------
    log.debug("Matching data to remnants")
    for ii in range(num_mergers):
        # First Merger
        #    Store dets after merger time for 'out' BH
        inds = np.where(d_scales[ii, BH_TYPE.OUT] > m_scales[ii])[0]
        if inds.size > 0:
            # Make sure values are valid
            temp_mdots = d_mdots[ii, BH_TYPE.OUT][inds]
            #    Find locations which are <=0.0 or inf
            bads = np.where((temp_mdots <= 0.0) | (~np.isfinite(temp_mdots)))[0]
            # Remove bad elements
            if bads.size > 0:
                inds = np.delete(inds, bads)

            # If valid elements remain
            if inds.size > 0:
                ids[ii] = d_ids[ii, BH_TYPE.OUT][inds[0]]
                scales[ii] = d_scales[ii, BH_TYPE.OUT][inds]
                masses[ii] = d_masses[ii, BH_TYPE.OUT][inds]
                dens[ii] = d_dens[ii, BH_TYPE.OUT][inds]
                mdots[ii] = d_mdots[ii, BH_TYPE.OUT][inds]
                csnds[ii] = d_csnds[ii, BH_TYPE.OUT][inds]

                idnums[ii] = d_ids[ii, BH_TYPE.OUT][inds]
                mrgnums[ii] = ii*np.ones(inds.size, dtype=int)

        else:
            log.warning("Merger %d without post-merger dets entries!" % (ii))
            ids[ii] = m_ids[ii, BH_TYPE.OUT]
            scales[ii] = []
            masses[ii] = []
            dens[ii] = []
            mdots[ii] = []
            csnds[ii] = []

            idnums[ii] = []
            mrgnums[ii] = []

        # Subsequent mrgs
        #    Find the next merger that this 'out' BH participates in
        next = next_merger[ii]
        check_id = m_ids[next, BH_TYPE.OUT]
        check_scale = m_scales[next]
        if ids[ii] >= 0 and next >= 0:
            # Make sure `next` is correct, fix if not
            if ids[ii] not in m_ids[next]:
                next = _find_next_merger(ids[ii], m_scales[ii], m_ids, m_scales)
                check_id = m_ids[next, BH_TYPE.OUT]
                check_scale = m_scales[next]
                # Error if still not fixed
                if next >= 0 and ids[ii] not in m_ids[next]:
                    errStr = "ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], next, str(m_ids[next]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

        #    while a subsequent merger exists... store those entries
        while next >= 0:
            next_ids = d_ids[next, BH_TYPE.OUT][:]
            # Make sure ID numbers match
            if check_id:
                if np.any(check_id != next_ids):
                    errStr = "ii = %d, next = %d, IDs don't match!" % (ii, next)
                    errStr += "\nids[ii] = %d, check = %d" % (ids[ii], check_id)
                    errStr += "\nd_ids = %s" % (str(next_ids))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            if np.size(scales[ii]) > 0:
                inds = np.where(d_scales[next, BH_TYPE.OUT] > np.max(scales[ii]))[0]
            else:
                inds = np.where(d_scales[next, BH_TYPE.OUT] > m_scales[ii])[0]

            if inds.size > 0:
                # Make sure values are valid
                temp_mdots = d_mdots[next, BH_TYPE.OUT][inds]
                #    Find locations which are <=0.0 or inf
                bads = np.where((temp_mdots <= 0.0) | (~np.isfinite(temp_mdots)))[0]
                # Remove bad elements
                if bads.size > 0:
                    inds = np.delete(inds, bads)

                if inds.size > 0:
                    scales[ii] = np.append(scales[ii], d_scales[next, BH_TYPE.OUT][inds])
                    masses[ii] = np.append(masses[ii], d_masses[next, BH_TYPE.OUT][inds])
                    dens[ii] = np.append(dens[ii], d_dens[next, BH_TYPE.OUT][inds])
                    mdots[ii] = np.append(mdots[ii], d_mdots[next, BH_TYPE.OUT][inds])
                    csnds[ii] = np.append(csnds[ii], d_csnds[next, BH_TYPE.OUT][inds])

                    idnums[ii] = np.append(idnums[ii], d_ids[next, BH_TYPE.OUT][inds])
                    mrgnums[ii] = np.append(mrgnums[ii], next*np.ones(inds.size, dtype=int))

            # Get next merger in Tree
            next = next_merger[next]
            # Make sure `next` is correct, fix if not
            if check_id not in m_ids[next] and next >= 0:
                next = _find_next_merger(check_id, check_scale, m_ids, m_scales)
                # Error if still not fixed
                if (next >= 0 and ids[ii] not in m_ids[next]):
                    errStr = "ERROR: ids[{}] = {}, merger ids {} = {}"
                    errStr = errStr.format(ii, ids[ii], next, str(m_ids[next]))
                    log.error(errStr)
                    zio.mpiError(comm, log=log, err=errStr)

            #    Get ID of next out-BH
            check_id = m_ids[next, BH_TYPE.OUT]
            check_scale = m_scales[next]

        # Appended entries may no longer be sorted, sort them
        if np.size(scales[ii]) > 0:
            inds = np.argsort(scales[ii])
            scales[ii] = scales[ii][inds]
            masses[ii] = masses[ii][inds]
            dens[ii] = dens[ii][inds]
            mdots[ii] = mdots[ii][inds]
            csnds[ii] = csnds[ii][inds]

            idnums[ii] = idnums[ii][inds]
            mrgnums[ii] = mrgnums[ii][inds]
            nents[ii] = np.size(scales[ii])
        else:
            log.warning("Merger %d without ANY entries." % (ii))

        if np.any(~np.isfinite(mdots[ii])):
            errStr = "Infinite mdots at merger {}\n".format(ii)
            errStr += "\tmdots = {}".format(mdots[ii])
            log.error(errStr)
            raise RuntimeError(errStr)

    log.debug("Remnant dets collected.")
    _logStats('Number of entries', nents, log)

    return idnums, scales, masses, dens, mdots, csnds


def _unmergedMasses(allScales, allMasses, mrgs, next_merger, log):
    """Remove mass from mrgs for the remnant in this BH lineage.

    Arguments
    ---------
    mind : int
    scales : (M,) array of arrays of scalar
    masses : (M,) array of arrays of scalar
    mrgs : dict
    tree : dict
    log : ``logging.Logger`` object

    Returns
    -------
    remmass : array_like of scalar

    """
    log.debug("_unmergedMasses()")
    num_mergers = mrgs[MERGERS.NUM]
    allUnmerged = []
    errMergs_mass = []     # Mergers which cause errors trying to subtract masses
    effMergs_mass = []     # Mergers which are effected by the above errors
    errMergs_id = []     # Mergers where the IDs dont match properly
    effMergs_id = []
    errCorr = []
    numMal = 0
    numBon = 0
    for mind in range(num_mergers):
        # Starting ID for this remnant
        bhID = mrgs[MERGERS.IDS][mind, BH_TYPE.OUT]
        # scales and masses for this remnant
        scales = allScales[mind]
        masses = allMasses[mind]
        # Index of next merger
        next = next_merger[mind]
        log.debug(" - Merger %d (ID %d), Next = %d" % (mind, bhID, next))
        nmal = 0
        nbon = 0
        # loop over all following mrgs
        while next >= 0:
            # Figure out which BH the remnant is; select the other-BH's mass to subtract
            if bhID == mrgs[MERGERS.IDS][next, BH_TYPE.IN]:
                otherMass = mrgs[MERGERS.MASSES][next, BH_TYPE.OUT]
            elif bhID == mrgs[MERGERS.IDS][next, BH_TYPE.OUT]:
                otherMass = mrgs[MERGERS.MASSES][next, BH_TYPE.IN]
            else:
                errStr = "Initial Merger %d, out bh ID %d\n" % (mind, bhID)
                errStr += "\tNext merger {}, IDs dont match: {}, {}".format(
                    next, mrgs[MERGERS.IDS][next, BH_TYPE.IN],
                    mrgs[MERGERS.IDS][next, BH_TYPE.OUT])
                log.error(errStr)
                errMergs_id.append(mind)
                effMergs_id.append(next)
                break
                # raise RuntimeError(errStr)

            # Figure out which entries to subtract the mass from
            mscale = mrgs[MERGERS.SCALES][next]
            # Find scales after the merger
            allInds = np.where(scales >= mscale)[0]
            if allInds.size == 0:
                log.warning("Merger %d, Next %d, no matching scales after %f!" % (mind, next, mscale))
            else:
                inds = np.array(allInds)
                if np.isclose(scales[inds[0]], mscale) and inds.size > 1 and inds[0] > 0:
                    #    Dont need `+1` because were starting at ``inds[0]-1``
                    inds = inds[np.argmax(np.diff(masses[inds[0]-1:inds[0]+2]))]
                else:
                    inds = inds[0]

                masses[inds:] -= otherMass

                bads = np.where(masses[inds:] <= 0.0)[0]
                nbad = bads.size
                ntot = masses.size - inds
                if nbad > 0:
                    nmal += 1
                    logStr = "Merger {}, next {}: {}/{} = {:.4f} Invalid masses!".format(
                        mind, next, nbad, ntot, nbad/ntot)
                    log.debug(logStr)
                    if nbad > 4 or nbad/ntot > 0.2:
                        errStr = "Merger {}, next {}: Too many ({}/{}) invalid masses!\n".format(
                            mind, next, nbad, ntot)
                        endInd = np.min([len(allInds), 5])
                        useInds = allInds[:endInd]
                        errStr += "\tinds = {}...\n".format(str(useInds))
                        errStr += "\tmscale = {}, scales = {}...\n".format(
                            mscale, str(scales[useInds]))
                        errStr += "\tother = {}, masses = {}...\n".format(
                            otherMass, str(masses[useInds]))
                        log.warning(errStr)
                        # raise RuntimeError(errStr)
                        # Add the mass back on
                        masses[inds:] += otherMass
                        errMergs_mass.append(mind)
                        effMergs_mass.append(next)
                else:
                    nbon += 1

            # Update for next merger
            bhID = mrgs[MERGERS.IDS][next, BH_TYPE.OUT]
            next = next_merger[next]

        allUnmerged.append(masses)
        numBon += nbon
        numMal += nmal
        if nbon > 0:
            errCorr.append(mind)

    numTot = numBon + numMal
    nerrs = len(errCorr)
    log.info(" - {}/{} = {:.4f} Remnants with correction errors".format(
        nerrs, num_mergers, nerrs/num_mergers))
    log.info(" - {}/{} = {:.4f} Total Errors (over all mrgs).".format(
        numMal, numTot, numMal/numTot))
    # log.info(" - Errors on remnants: {}".format(str(corrErrs)))
    errMergs_id = np.array(list(set(errMergs_id)))
    effMergs_id = np.array(list(set(effMergs_id)))
    frac_id = errMergs_id.size/num_mergers
    errMergs_mass = np.array(list(set(errMergs_mass)))
    effMergs_mass = np.array(list(set(effMergs_mass)))
    frac_mass = errMergs_mass.size/num_mergers
    logStr = " - ID Errors:\n - - {}/{} = {:.4f} Unique Mergers Effected, by {} unique mrgs"
    logStr = logStr.format(effMergs_id.size, num_mergers, frac_id, errMergs_id.size)
    log.info(logStr)
    logStr = " - Mass Errors:\n - - {}/{} = {:.4f} Unique Mergers Effected, by {} unique mrgs"
    logStr = logStr.format(effMergs_mass.size, num_mergers, frac_mass, errMergs_mass.size)
    log.info(logStr)

    return allUnmerged


def _indBefAft(scaleDiff):
    """Retrieve the index matching the minimum of `scaleDiff` greater-than zero.

    Used by: `infer_merger_out_masses`
    """
    try:
        ind = zmath.argextrema(scaleDiff, 'min', 'g')
    except ValueError:
        ind = None

    return ind


def _logStats(name, prop, log, lvl=logging.DEBUG):
    """Log basic statistics about the given property.

    Arguments
    ---------
    name : str
        Name of the property.
    prop : array_like of scalar
        Array of measures of the target property.
    log : ``logging.Logger`` object
        Object to log to.

    """
    prop = np.asarray(prop)
    prop = prop.flatten()
    cnt = np.count_nonzero(prop)
    try:
        tot = prop.size
    except:
        tot = len(prop)
    frac = cnt/tot
    log.log(lvl, " - {}:".format(name))
    log.log(lvl, " - - {:6d}/{:6d} = {:.4f} nonzero".format(cnt, tot, frac))
    log.log(lvl, " - - Median and 68% (overall): {}".format(
        str(zmath.confidenceIntervals(prop, ci=0.68))))
    log.log(lvl, " - - Median and 68% (nonzero): {}".format(
        str(zmath.confidenceIntervals(prop, ci=0.68, filter='g'))))
    return


def _detailsForBHLineage(run, mrg, log, rdets=None, tree=None, mrgs=None):
    log.debug("_detailsForBHLineage()")
    # Get all merger indices in this tree
    finMerger, bhIDs, mrgInds = illpy_lib.illbh.BHTree.allIDsForTree(
        run, mrg, tree=tree, mrgs=mrgs)
'''


if __name__ == "__main__":
    main()
