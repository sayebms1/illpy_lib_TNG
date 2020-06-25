"""
"""

import os
import shutil
import sys
from datetime import datetime

import numpy as np
import h5py

import zcode.inout as zio


try:
    import illpy
except ImportError:
    PATH_ILLPY = "/n/home00/lkelley/illustris/redesign/illpy/"
    if PATH_ILLPY not in sys.path:
        print("Added path to `illpy`: '{}'".format(PATH_ILLPY))
        sys.path.append(PATH_ILLPY)

    import illpy


try:
    import illpy_lib
except ImportError:
    PATH_ILLPY_LIB = "/n/home00/lkelley/illustris/redesign/illpy_lib/"
    if PATH_ILLPY_LIB not in sys.path:
        print("Added path to `illpy_lib`: '{}'".format(PATH_ILLPY_LIB))
        sys.path.append(PATH_ILLPY_LIB)

    import illpy_lib  # noqa


from illpy_lib.constants import NUM_SNAPS, PARTICLE

from illpy_lib.illbh import Core

VERSION = 0.1


def main():
    core = Core(sets=dict(LOG_FILENAME="log_illbh-snapshots.log", RECREATE=True))
    log = core.log

    log.info("details.main()")
    print(log.filename)

    beg = datetime.now()

    fname = core.paths.fname_bh_particles
    exists = os.path.exists(fname)

    recreate = core.sets.RECREATE
    log.debug("File '{}' exists: {}".format(fname, exists))

    if not recreate and exists:
        log.info("Particle file exists: '{}'".format(fname))
        return

    log.warning("Loading BH particle data from snapshots")
    fname_temp = zio.modify_filename(fname, prepend='_')

    log.debug("Writing to temporary file '{}'".format(fname_temp))
    # log.error("WARNING: running in TEMPORARY append mode!")
    # with h5py.File(fname_temp, 'a') as out:
    with h5py.File(fname_temp, 'r') as out:

        all_ids = set()

        for snap in core.tqdm(range(NUM_SNAPS), desc='Loading snapshots'):

            log.debug("Loading snap {}".format(snap))
            snap_str = '{:03d}'.format(snap)
            group = out.create_group(snap_str)

            try:
                bhs = illpy.snapshot.loadSubset(core.paths.INPUT, snap, PARTICLE.BH)
            except Exception as err:
                log.error("FAILED on snap {}!!!".format(snap))
                continue

            num_bhs = bhs['count']
            log.info("Snap {} Loaded {} BHs".format(snap, num_bhs))
            group.attrs['count'] = num_bhs
            if num_bhs == 0:
                continue

            ids = bhs['ParticleIDs']
            all_ids = all_ids.union(ids)
            sort = np.argsort(ids)

            keys = list(bhs.keys())
            keys.pop(keys.index('count'))
            for kk in keys:
                group.create_dataset(kk, data=bhs[kk][:][sort])

        '''
        for snap in core.tqdm(range(NUM_SNAPS), desc='Loading snapshots'):

            log.debug("Loading snap {}".format(snap))
            snap_str = '{:03d}'.format(snap)
            group = out[snap_str]

            if 'ParticleIDs' not in group.keys():
                log.error("Skipping snap {}".format(snap))
                continue

            ids = group['ParticleIDs']
            num_bhs = ids.size
            group.attrs['num'] = num_bhs

            all_ids = all_ids.union(ids)
        '''

        all_ids = np.array(sorted(list(all_ids)))
        first = NUM_SNAPS * np.ones_like(all_ids, dtype=np.uint32)
        last = np.zeros_like(all_ids, dtype=np.uint32)

        # Find the first and last snapshot that each BH is found in
        for snap in core.tqdm(range(NUM_SNAPS), desc='Finding first/last'):
            snap_str = '{:03d}'.format(snap)
            try:
                ids = out[snap_str]['ParticleIDs'][:]
            except KeyError as err:
                lvl = log.INFO if (snap in [53, 55]) else log.ERROR
                log.log(lvl, "Failed to access `ParticleIDs` from snap {}".format(snap))
                log.log(lvl, str(err))
                continue

            slots = np.searchsorted(all_ids, ids)
            first[slots] = np.minimum(first[slots], snap)
            last[slots] = np.maximum(last[slots], snap)

        out.create_dataset('unique_ids', data=all_ids)
        out.create_dataset('unique_first_snap', data=first)
        out.create_dataset('unique_last_snap', data=last)

    log.debug("Moving temporary to final file '{}' ==> '{}'".format(fname_temp, fname))
    shutil.move(fname_temp, fname)

    size_str = zio.get_file_size(fname)
    end = datetime.now()
    log.info("Saved to '{}', size {}, after {}".format(fname, size_str, end-beg))

    return


if __name__ == "__main__":
    main()
