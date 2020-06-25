"""
"""

import os
# import sys
import shutil
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio
import zcode.math as zmath

from illpy_lib.constants import DTYPE
from illpy_lib.illbh import Core, DETAILS, load_hdf5_to_mem

VERSION = 0.3                                    # Version of details

_DEF_PRECISION = -8                               # Default precision


def main(reorganize_flag=True, reformat_flag=True):

    core = Core(sets=dict(LOG_FILENAME='log_illbh-details.log'))
    log = core.log

    log.info("details.main()")
    print(log.filename)

    beg = datetime.now()

    # Organize Details by Snapshot Time; create new, temporary ASCII Files
    log.debug("`reorganize_flag` = {}".format(reorganize_flag))
    if reorganize_flag:
        reorganize(core)

    # Create Dictionary Details Files
    log.debug("`reformat_flag` = {}".format(reformat_flag))
    if reformat_flag:
        reformat(core)

    end = datetime.now()
    log.info("Done after '{}'".format(end-beg))

    return


def reorganize(core=None):
    core = Core.load(core)
    log = core.log
    log.debug("details.reorganize()")

    RUN = core.sets.RUN_NUM
    NUM_SNAPS = core.sets.NUM_SNAPS

    # temp_fnames = [constants.GET_DETAILS_TEMP_FILENAME(run, snap) for snap in range(NUM_SNAPS)]
    temp_fnames = [core.paths.fname_details_temp_snap(snap, RUN) for snap in range(NUM_SNAPS)]

    loadsave = (not core.sets.RECREATE)

    # Check if all temp files already exist
    if loadsave:
        temps_exist = [os.path.exists(tfil) for tfil in temp_fnames]
        if all(temps_exist):
            log.info("All temp files exist.")
        else:
            bad = np.argmin(temps_exist)
            bad = temp_fnames[bad]
            log.warning("Temp files do not exist e.g. '{}'".format(bad))
            loadsave = False

    # If temp files dont exist, or we WANT to redo them, then create temp files
    if not loadsave:
        log.debug("Finding Illustris BH Details files")
        # Get Illustris BH Details Filenames
        # raw_fnames = constants.GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run, verbose)
        raw_fnames = core.paths.fnames_details_input
        if len(raw_fnames) < 1:
            log.raise_error("Error no dets files found!!")

        # Reorganize into temp files
        log.warning("Reorganizing details into temporary files")
        _reorganize_files(core, raw_fnames, temp_fnames)

    # Confirm all temp files exist
    temps_exist = all([os.path.exists(tfil) for tfil in temp_fnames])

    # If files are missing, raise error
    if not temps_exist:
        print(("Temporary Files still missing!  '{:s}'".format(temp_fnames[0])))
        raise RuntimeError("Temporary Files missing!")

    return temp_fnames


def _reorganize_files(core, raw_fnames, temp_fnames):

    log = core.log
    log.debug("details._reorganize_files()")

    NUM_SNAPS = core.sets.NUM_SNAPS
    snap_scales = core.cosmo.scales()

    temps = [zio.modify_filename(tt, prepend='_') for tt in temp_fnames]

    # Open new ASCII, Temp dets files
    #    Make sure path is okay
    zio.check_path(temps[0])
    # Open each temp file
    # temp_files = [open(tfil, 'w') for tfil in temp_fnames]
    temp_files = [open(tfil, 'w') for tfil in temps]

    num_temp = len(temp_files)
    num_raw  = len(raw_fnames)
    log.info("Organizing {:d} raw files into {:d} temp files".format(num_raw, num_temp))

    prec = _DEF_PRECISION
    all_num_lines_in = 0
    all_num_lines_out = 0

    # Iterate over all Illustris Details Files
    # ----------------------------------------
    for ii, raw in enumerate(core.tqdm(raw_fnames, desc='Raw files')):
        log.debug("File {}: '{}'".format(ii, raw))
        lines = []
        scales = []
        # Load all lines and entry scale-factors from raw dets file
        for dline in open(raw):
            lines.append(dline)
            # Extract scale-factor from line
            detScale = DTYPE.SCALAR(dline.split()[1])
            scales.append(detScale)

        # Convert to array
        lines = np.array(lines)
        scales = np.array(scales)
        num_lines_in = scales.size

        # If file is empty, continue
        if num_lines_in == 0:
            log.debug("\tFile empty")
            continue

        log.debug("\tLoaded {}".format(num_lines_in))

        # Round snapshot scales to desired precision
        scales_round = np.around(snap_scales, -prec)

        # Find snapshots following each entry (right-edge) or equal (include right: 'right=True')
        #    `-1` to put binaries into the snapshot UP-TO that scalefactor
        # snap_nums = np.digitize(scales, scales_round, right=True) - 1
        snap_nums = np.digitize(scales, scales_round, right=True)

        # For each Snapshot, write appropriate lines
        num_lines_out = 0
        for snap in range(NUM_SNAPS):
            inds = (snap_nums == snap)
            num_lines_out_snap = np.count_nonzero(inds)
            if num_lines_out_snap == 0:
                continue

            temp_files[snap].writelines(lines[inds])
            # log.debug("\t\tWrote {} lines to snap {}".format(num_lines_out_snap, snap))
            num_lines_out += num_lines_out_snap

        if num_lines_out != num_lines_in:
            log.error("File {}, '{}'".format(ii, raw))
            log.raise_error("Wrote {} lines, loaded {} lines!".format(num_lines_out, num_lines_in))

        all_num_lines_in += num_lines_in
        all_num_lines_out += num_lines_out

    # Close out dets files
    tot_size = 0.0
    log.info("Closing files, checking sizes")

    for ii, newdf in enumerate(temp_files):
        newdf.close()
        tot_size += os.path.getsize(newdf.name)

    ave_size = tot_size/(1.0*len(temp_files))
    size_str = zio.bytes_string(tot_size)
    ave_size_str = zio.bytes_string(ave_size)
    log.info("Total temp size = '{}', average = '{}'".format(size_str, ave_size_str))

    log.info("Input lines = {:d}, Output lines = {:d}".format(all_num_lines_in, all_num_lines_out))
    if (all_num_lines_in != all_num_lines_out):
        log.raise_error("input lines {}, does not match output lines {}!".format(
            all_num_lines_in, all_num_lines_out))

    log.info("Renaming temporary files...")
    for ii, (aa, bb) in enumerate(zip(temps, temp_fnames)):
        if ii == 0:
            log.debug("'{}' ==> '{}'".format(aa, bb))
        shutil.move(aa, bb)

    return


def reformat(core=None):
    core = Core.load(core)
    log = core.log
    NUM_SNAPS = core.sets.NUM_SNAPS

    log.debug("details.reformat()")

    out_fnames = [core.paths.fname_details_snap(snap) for snap in range(NUM_SNAPS)]

    loadsave = (not core.sets.RECREATE)

    # Check if all save files already exist, and correct versions
    if loadsave:
        out_exist = [os.path.exists(sfil) for sfil in out_fnames]
        if all(out_exist):
            log.info("All output files exist")
        else:
            bad = np.argmin(out_exist)
            bad = out_fnames[bad]
            log.warning("Output files do not exist e.g. '{}'".format(bad))
            loadsave = False

    # Re-convert files
    if (not loadsave):
        log.warning("Processing temporary files")
        temp_fnames = [core.paths.fname_details_temp_snap(snap) for snap in range(NUM_SNAPS)]
        for snap in core.tqdm(range(NUM_SNAPS)):
            temp = temp_fnames[snap]
            out = out_fnames[snap]
            _reformat_to_hdf5(core, snap, temp, out)

    # Confirm save files exist
    out_exist = [os.path.exists(sfil) for sfil in out_fnames]

    # If files are missing, raise error
    if (not all(out_exist)):
        log.raise_error("Output files missing!")

    return out_fnames


def _reformat_to_hdf5(core, snap, temp_fname, out_fname):
    """
    """

    log = core.log
    cosmo = core.cosmo
    log.debug("details._reformat_to_hdf5()")
    log.info("Snap {}, {} ==> {}".format(snap, temp_fname, out_fname))
    CONV_ILL_TO_CGS = core.cosmo.CONV_ILL_TO_CGS

    loadsave = (not core.sets.RECREATE)

    # Make Sure Temporary Files exist, Otherwise re-create them
    if (not os.path.exists(temp_fname)):
        log.raise_error("Temp file '{}' does not exist!".format(temp_fname))

    # Try to load from existing save
    if loadsave:
        if os.path.exists(out_fname):
            log.info("\tOutput file '{}' already exists.".format(out_fname))
            return

    # Load dets from ASCII File
    vals = _load_bhdetails_ascii(temp_fname)
    ids, scales, masses, mdots, rhos, cs = vals
    # Sort by ID number, then by scale-factor
    sort = np.lexsort((scales, ids))
    vals = [vv[sort] for vv in vals]
    ids, scales, masses, mdots, rhos, cs = vals

    # Find unique ID numbers, their first occurence indices, and the number of occurences
    u_ids, u_inds, u_counts = np.unique(ids, return_index=True, return_counts=True)
    num_unique = u_ids.size
    log.info("\tunique IDs: {}".format(zmath.frac_str(num_unique, ids.size)))

    # Calculate mass-differences
    dmdts = np.zeros_like(mdots)
    for ii, nn in zip(u_inds, u_counts):
        j0 = slice(ii, ii+nn-1)
        j1 = slice(ii+1, ii+nn)
        # t0 = cosmo.scale_to_age(scales[j0])
        # t1 = cosmo.scale_to_age(scales[j1])
        z0 = cosmo._a_to_z(scales[j0])
        z1 = cosmo._a_to_z(scales[j1])
        t0 = cosmo.age(z0).cgs.value
        t1 = cosmo.age(z1).cgs.value
        m0 = masses[j0]
        m1 = masses[j1]
        dm = m1 - m0
        dt = t1 - t0

        # dmdts[j1] = (m1 - m0) / dt

        ss = np.ones_like(dm)
        neg = (dm < 0.0) | (dt < 0.0)
        ss[neg] *= -1

        inds = (dt != 0.0)
        dmdts[j1][inds] = ss[inds] * np.fabs(dm[inds] / dt[inds])

    # Convert dmdts to same units as mdots
    dmdts = dmdts * CONV_ILL_TO_CGS.MASS / CONV_ILL_TO_CGS.MDOT

    with h5py.File(out_fname, 'w') as out:
        out.attrs[DETAILS.RUN] = core.sets.RUN_NUM
        out.attrs[DETAILS.SNAP] = snap
        out.attrs[DETAILS.NUM] = len(ids)
        out.attrs[DETAILS.CREATED] = str(datetime.now().ctime())
        out.attrs[DETAILS.VERSION] = VERSION

        out.create_dataset(DETAILS.IDS, data=ids)
        out.create_dataset(DETAILS.SCALES, data=scales)
        out.create_dataset(DETAILS.MASSES, data=masses)
        out.create_dataset(DETAILS.MDOTS, data=mdots)
        out.create_dataset(DETAILS.DMDTS, data=dmdts)
        out.create_dataset(DETAILS.RHOS, data=rhos)
        out.create_dataset(DETAILS.CS, data=cs)

        out.create_dataset(DETAILS.UNIQUE_IDS, data=u_ids)
        out.create_dataset(DETAILS.UNIQUE_INDICES, data=u_inds)
        out.create_dataset(DETAILS.UNIQUE_COUNTS, data=u_counts)

    size_str = zio.get_file_size(out_fname)
    log.info("\tSaved snap {} to '{}', size {}".format(snap, out_fname, size_str))

    return


def _load_bhdetails_ascii(temp_fname):
    # Files have some blank lines in them... Clean
    with open(temp_fname, 'r') as temp:
        lines = temp.readlines()

    nums = len(lines)

    # Allocate storage
    ids    = np.zeros(nums, dtype=DTYPE.ID)
    scales = np.zeros(nums, dtype=DTYPE.SCALAR)
    masses = np.zeros(nums, dtype=DTYPE.SCALAR)
    mdots  = np.zeros(nums, dtype=DTYPE.SCALAR)
    rhos   = np.zeros(nums, dtype=DTYPE.SCALAR)
    cs     = np.zeros(nums, dtype=DTYPE.SCALAR)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if (len(lin) > 0):
            tid, tim, mas, dot, rho, tcs = _parse_bhdetails_line(lin)
            ids[count] = tid
            scales[count] = tim
            masses[count] = mas
            mdots[count] = dot
            rhos[count] = rho
            cs[count] = tcs

            count += 1

    # Trim excess (shouldn't be needed)
    if (count != nums):
        trim = np.s_[count:]

        ids    = np.delete(ids, trim)
        scales = np.delete(scales, trim)
        masses = np.delete(masses, trim)
        mdots  = np.delete(mdots, trim)
        rhos   = np.delete(rhos, trim)
        cs     = np.delete(cs, trim)

    return ids, scales, masses, mdots, rhos, cs


def _parse_bhdetails_line(instr):
    """Parse a line from an Illustris blachole_details_#.txt file

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    Arguments
    ---------

    Returns
    -------
        ID, time, mass, mdot, rho, cs

    """
    args = instr.split()
    # First element is 'BH=#', trim to just the id number
    args[0] = args[0].split("BH=")[-1]
    idn  = DTYPE.ID(args[0])
    time = DTYPE.SCALAR(args[1])
    mass = DTYPE.SCALAR(args[2])
    mdot = DTYPE.SCALAR(args[3])
    rho  = DTYPE.SCALAR(args[4])
    cs   = DTYPE.SCALAR(args[5])
    return idn, time, mass, mdot, rho, cs


def load_details(snap, core=None):
    """
    """
    core = Core.load(core)
    log = core.log

    log.debug("details.load_details()")

    fname = core.paths.fname_details_snap(snap)
    log.debug("Filename for snap {}: '{}'".format(snap, fname))

    dets = load_hdf5_to_mem(fname)
    return dets


def calc_dmdt_for_details(core=None):
    """Calculate mass-differences as estimate for accretion rates, add/overwrite in HDF5 files.
    """
    core = Core.load(core)
    log = core.log
    cosmo = core.cosmo
    NUM_SNAPS = core.sets.NUM_SNAPS
    CONV_ILL_TO_CGS = core.cosmo.CONV_ILL_TO_CGS

    # log.warning("WARNING: testing `calc_dmdt_for_details`!")
    # for snap in [135]:
    for snap in core.tqdm(range(NUM_SNAPS)):
        fname = core.paths.fname_details_snap(snap)
        log.debug("Snap {}: '{}'".format(snap, fname))
        with h5py.File(fname, 'a') as data:
            scales = data[DETAILS.SCALES]
            if scales.size == 0:
                continue

            # These are already sorted by ID and scale-factor, so contiguous and chronological
            # for each BH
            masses = data[DETAILS.MASSES][:] * CONV_ILL_TO_CGS.MASS
            mdots = data[DETAILS.MDOTS]

            u_inds = data[DETAILS.UNIQUE_INDICES]
            u_counts = data[DETAILS.UNIQUE_COUNTS]

            # Calculate mass-differences
            dmdts = np.zeros_like(masses)
            count = 0
            count_all = 0

            for ii, nn in zip(u_inds, u_counts):
                j0 = slice(ii, ii+nn-1)
                j1 = slice(ii+1, ii+nn)
                # t0 = cosmo.a_to_tage(scales[j0])
                # t1 = cosmo.a_to_tage(scales[j1])
                z0 = cosmo._a_to_z(scales[j0])
                z1 = cosmo._a_to_z(scales[j1])
                t0 = cosmo.age(z0).cgs.value
                t1 = cosmo.age(z1).cgs.value
                m0 = masses[j0]
                m1 = masses[j1]
                dm = m1 - m0
                dt = t1 - t0

                ss = np.ones_like(dm)
                neg = (dm < 0.0) | (dt < 0.0)
                ss[neg] *= -1

                inds = (dt != 0.0)
                dmdts[j1][inds] = ss[inds] * np.fabs(dm[inds] / dt[inds])

                count += np.count_nonzero(inds)
                count_all += inds.size

            dmdts = dmdts / CONV_ILL_TO_CGS.MDOT
            log.info("dM/dt nonzero : " + zmath.frac_str(np.count_nonzero(dmdts), masses.size))
            log.info("mdots : " + zmath.stats_str(mdots, filter='>'))
            log.info("dmdts : " + zmath.stats_str(dmdts, filter='>'))

    return


if __name__ == "__main__":
    main(reorganize_flag=False, reformat_flag=True)

    # calc_dmdt_for_details(core=None)
