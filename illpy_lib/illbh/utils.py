"""
"""

import numpy as np
import h5py


def load_hdf5_to_mem(fname):
    with h5py.File(fname, 'r') as data:
        # out = {kk: data[kk][()] if np.shape(data[kk]) == () else data[kk][:] for kk in data.keys()}
        # use `[()]` instead of `[:]` to handle scalar datasets
        out = {kk: data[kk][()] for kk in data.keys()}

    return out


def _distribute_snapshots(core, comm):
    """Evenly distribute snapshot numbers across multiple processors.
    """
    size = comm.size
    rank = comm.rank
    mySnaps = np.arange(core.sets.NUM_SNAPS)
    if size > 1:
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]

    return mySnaps
