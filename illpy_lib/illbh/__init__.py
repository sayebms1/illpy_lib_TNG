"""This module handles the processing of Illustris BH files.
"""

import numpy as np
# import h5py
np.seterr(divide='ignore', invalid='ignore')

# from illpy_lib.constants import NUM_SNAPS  # noqa

from . deep_core import Core  # noqa


class MERGERS:
    # Meta Data
    RUN       = 'run'
    CREATED   = 'created'
    NUM       = 'num'
    VERSION   = 'version'
    FILE      = 'filename'

    # Physical Parameters
    IDS       = 'ids'
    SCALES    = 'scales'
    MASSES    = 'masses'

    # Maps
    # MAP_STOM  = 's2m'
    # MAP_MTOS  = 'm2s'
    # MAP_ONTOP = 'ontop'
    SNAP_NUMS = "snap_nums"
    ONTOP_NEXT = "ontop_next"
    ONTOP_PREV = "ontop_prev"


MERGERS_PHYSICAL_KEYS = [MERGERS.IDS, MERGERS.SCALES, MERGERS.MASSES]


class DETAILS:
    RUN     = 'run'
    CREATED = 'created'
    VERSION = 'version'
    NUM     = 'num'
    SNAP    = 'snap'
    FILE    = 'filename'

    IDS     = 'id'
    SCALES  = 'scales'
    MASSES  = 'masses'
    MDOTS   = 'mdots'
    DMDTS   = 'dmdts'     # differences in masses
    RHOS    = 'rhos'
    CS      = 'cs'

    UNIQUE_IDS = 'unique_ids'
    UNIQUE_INDICES = 'unique_indices'
    UNIQUE_COUNTS = 'unique_counts'


DETAILS_PHYSICAL_KEYS = [DETAILS.IDS, DETAILS.SCALES, DETAILS.MASSES,
                         DETAILS.MDOTS, DETAILS.DMDTS, DETAILS.RHOS, DETAILS.CS]


class _LenMeta(type):

    def __len__(self):
        return self.__len__()


class BH_TYPE(metaclass=_LenMeta):
    IN  = 0
    OUT = 1

    @classmethod
    def __len__(cls):
        return 2


class BH_TREE:
    PREV         = 'prev'
    NEXT         = 'next'
    SCALE_PREV   = 'scale_prev'
    SCALE_NEXT   = 'scale_next'
    TIME_PREV    = 'time_prev'
    TIME_NEXT    = 'time_next'

    NUM_BEF      = 'num_bef'
    NUM_AFT      = 'num_aft'
    TIME_BETWEEN = 'time_between'

    CREATED      = 'created'
    RUN          = 'run'
    VERSION      = 'version'
    NUM          = 'num'


from . utils import load_hdf5_to_mem, _distribute_snapshots  # noqa

# from . import bh_constants  # noqa
