"""
"""
import os
import glob

import pycore

import illpy
import illpy.snapshot

class Settings(pycore.Settings):

    VERBOSITY = 20
    LOG_FILENAME = "log_illpy-lib_bh.log"
    RUN_NUM = 1

    #INPUT = "/n/ghernquist/Illustris/Runs/L75n1820FP/"
     INPUT = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output"
    # OUTPUT = "/n/regal/hernquist_lab/lkelley/illustris-processed/"
    # OUTPUT = "/n/scratchlfs/hernquist_lab/lzkelley/illustris-processed/"
    OUTPUT = "/n/home09/sayebms/illpy_output"

    # NOTE: this is automatically reset
    TNG = None

    RECREATE = False
    BREAK_ON_FAIL = False
    MAX_DETAILS_PER_SNAP = 10

    def __init__(self, parse_cl=None, **kwargs):
        super().__init__(parse_cl=None, **kwargs)
        tng_flag = self.check_tng()
        if self.TNG is not None:
            raise ValueError("Do not set `TNG` flag manually!")

        self.TNG = tng_flag
        print("Running in TNG mode: '{}'".format(tng_flag))

        if tng_flag:
            self.setup_tng()
        else:
            self.setup_tos()

        # self._setup()

        return

    def add_arguments(argself, parser):
        '''
        parser.add_argument(
            '-s', '--sim', type=str,
            help='type of simulation being processed')
        '''
        return

    def check_tng(self):
        log = self._core.log
        log.debug("check_tng()")
        
        inpath = self.INPUT
        tng_flag = ('tng' in inpath.lower())
        inpath = os.path.join(inpath, 'output', '')

        # Double check based on header information
        try:
            header = illpy.snapshot.get_header(inpath)
        except OSError as err:
            log.error("{}".format(str(err)))
        else:
            keys = [kk.lower() for kk in header.keys()]
            check1 = ('git_commit' in keys)
            check2 = (header['HubbleParam'] < 0.7)
            checks = [tng_flag, check1, check2]
            if not all(checks) and any(checks):
                raise RuntimeError("Cannot confirm whether TNG or TOS!  {}".format(checks))

        return tng_flag

    def setup_tos(self):
        # self.HPAR = 0.704
        self.NUM_SNAPS = 136
        return

    def setup_tng(self):
        # self.HPAR = 0.6774
        self.NUM_SNAPS = 100
        return


class Paths(pycore.Paths):
    _MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"
    _DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

    FNAME_DETAILS_CLEAN = "bh_details.hdf5"
    FNAME_MERGERS_CLEAN = "bh_mergers.hdf5"

    FNAME_BH_PARTICLES = "bh_particles.hdf5"

    # "ill-%d_blackhole_details_temp_snap-%d.txt"
    FNAME_DETAILS_TEMP_SNAP = "ill-{run_num:d}_blackhole_details_temp_snap-{snap_num:03d}.txt"

    # "ill-%d_blackhole_details_save_snap-%d_v%.2f.npz"
    FNAME_DETAILS_SNAP = "ill-{run_num:d}_blackhole_details_snap-{snap_num:03d}.hdf5"

    # _MERGERS_RAW_COMBINED_FILENAME  = "ill-%d_blackhole_mergers_combined.txt"
    # _MERGERS_RAW_MAPPED_FILENAME    = "ill-%d_blackhole_mergers_mapped_v%.2f.npz"
    FNAME_MERGERS_TEMP = "ill-{run_num:d}_blackhole_mergers_temp.hdf5"

    # _MERGERS_FIXED_FILENAME         = "ill-%d_blackhole_mergers_fixed_v%.2f.npz"
    FNAME_MERGERS_FIXED = "ill-{run_num:d}_blackhole_mergers_fixed.hdf5"

    # _MERGER_DETAILS_FILENAME        = 'ill-%d_blackhole_merger-details_persnap-%03d_v%s.npz'
    FNAME_MERGER_DETAILS = "ill-{run_num:d}_blackhole_merger-details_per-snap-{per_snap:03d}.hdf5"

    # _REMNANT_DETAILS_FILENAME       = 'ill-%d_blackhole_remnant-details_persnap-%03d_v%s.npz'
    FNAME_REMNANT_DETAILS = "ill-{run_num:d}_blackhole_remnant-details_per-snap-{per_snap:03d}.hdf5"

    FNAME_REMNANT_DETAILS_FIXED = "ill-{run_num:d}_blackhole_remnant-details-fixed_per-snap-{per_snap:03d}.hdf5"

    # _BLACKHOLE_TREE_FILENAME         = "ill-%d_bh-tree_v%.2f.npz"
    FNAME_MERGER_TREE = "ill-{run_num:d}_blackhole_merger-tree.hdf5"

    # The substituted string should be either 'mergers' or 'details'
    _ILL_1_TXT_DIRS = [
        "txt-files-curie/blackhole_{}/",
        "txt-files-supermuc/blackhole_{}/",
        "txt-files-partial/Aug8/blackhole_{}/",
        "txt-files-partial/Aug14/blackhole_{}/",
        "txt-files-partial/Sep25/blackhole_{}/",
        "txt-files-partial/Oct10/blackhole_{}/"
    ]

    def __init__(self, core, **kwargs):
        super().__init__(core)
        self.OUTPUT = os.path.realpath(core.sets.OUTPUT)
        self.INPUT = os.path.realpath(core.sets.INPUT)
        return

    @property
    def mergers_input(self):
        return self._find_input_files('mergers', self._MERGERS_FILENAME_REGEX)

    @property
    def fnames_details_input(self):
        return self._find_input_files('details', self._DETAILS_FILENAME_REGEX)

    def _find_input_files(self, name, regex):
        log = self._core.log

        if ('illustris-1' in self.INPUT.lower()) or ('L75n1820FP' in self.INPUT):
            log.debug("Input looks like `illustris-1` ('{}')".format(self.INPUT))
            _path = os.path.join(self.INPUT, 'txt-files/txtfiles_new/')
            paths = [os.path.join(_path, td.format(name), '') for td in self._ILL_1_TXT_DIRS]
        elif ('illustris-2' in self.INPUT.lower()) or ('L75n910FP' in self.INPUT):
            log.debug("Input looks like `illustris-2` ('{}')".format(self.INPUT))
            # subdir = "/combined_output/blackhole_mergers/"
            subdir = "combined_output/blackhole_{}/".format(name)
            paths = [os.path.join(self.INPUT, subdir)]
        else:
            log.debug("Input looks like `illustris-3` or default ('{}')".format(
                self.INPUT))
            # subdir = "/output/blackhole_mergers/"
            subdir = "output/blackhole_{}/".format(name)
            paths = [os.path.join(self.INPUT, subdir)]
            # print(self.INPUT, subdir, os.path.join(self.INPUT, subdir))
            # print("Paths = '{}'".format(paths))

        files = []
        log.debug("Checking {} directories for {} files".format(len(paths), name))
        for pp in paths:
            if not os.path.exists(pp):
                raise RuntimeError("Expected path '{}' does not exist!".format(pp))
            pattern = os.path.join(pp, regex)
            log.debug("  Getting {} files from '{}'".format(name, pp))
            _fils = sorted(glob.glob(pattern))
            num_fils = len(_fils)
            if num_fils == 0:
                raise RuntimeError("No {} files found matching '{}'".format(name, pattern))
            log.debug("    Found '{}' files, e.g. '{}'".format(
                num_fils, os.path.basename(_fils[0])))
            files += _fils

        log.debug("Found {} {} files".format(len(files), name))
        return files

    @property
    def details_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_DETAILS_CLEAN)

    @property
    def output(self):
        path = self.OUTPUT
        if not os.path.exists(path):
            self._core.log.warning("Output path '{}' does not exist, creating".format(path))
            os.makedirs(path)

        return path

    @property
    def output_details(self):
        return os.path.join(self.OUTPUT, "details", "")

    @property
    def mergers_clean(self):
        return os.path.join(self.OUTPUT, self.FNAME_MERGERS_CLEAN)

    @property
    def output_figs(self):
        # path = os.path.join('.', self._DNAME_PLOTS, "")
        # path = os.path.realpath(path)
        path = os.path.join('.', self._core.sets.NAME.lower(), "")
        path = self.check_path(path)
        return path

    def fname_details_temp_snap(self, snap, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_DETAILS_TEMP_SNAP.format(snap_num=snap, run_num=run_num)
        fname = os.path.join(self.output_details, fname)
        return fname

    def fname_details_snap(self, snap, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_DETAILS_SNAP.format(snap_num=snap, run_num=run_num)
        fname = os.path.join(self.output_details, fname)
        return fname

    def fname_mergers_temp(self, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_MERGERS_TEMP.format(run_num=run_num)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_mergers_fixed(self, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_MERGERS_FIXED.format(run_num=run_num)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_merger_tree(self, run_num=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        fname = self.FNAME_MERGER_TREE.format(run_num=run_num)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_merger_details(self, run_num=None, max_per_snap=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        if max_per_snap is None:
            max_per_snap = self._core.sets.MAX_DETAILS_PER_SNAP

        fname = self.FNAME_MERGER_DETAILS.format(run_num=run_num, per_snap=max_per_snap)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_remnant_details(self, run_num=None, max_per_snap=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        if max_per_snap is None:
            max_per_snap = self._core.sets.MAX_DETAILS_PER_SNAP

        fname = self.FNAME_REMNANT_DETAILS.format(run_num=run_num, per_snap=max_per_snap)
        fname = os.path.join(self.output, fname)
        return fname

    def fname_remnant_details_fixed(self, run_num=None, max_per_snap=None):
        if run_num is None:
            run_num = self._core.sets.RUN_NUM

        if max_per_snap is None:
            max_per_snap = self._core.sets.MAX_DETAILS_PER_SNAP

        fname = self.FNAME_REMNANT_DETAILS_FIXED.format(run_num=run_num, per_snap=max_per_snap)
        fname = os.path.join(self.output, fname)
        return fname

    @property
    def fname_bh_particles(self):
        return os.path.join(self.OUTPUT, self.FNAME_BH_PARTICLES)


class Core(pycore.Core):
    _CLASS_SETTINGS = Settings
    _CLASS_PATHS = Paths

    def setup_for_ipython(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return

    def _load_cosmology(self):
        import illpy_lib.illcosmo
        if self.sets.TNG:
            cosmo = illpy_lib.illcosmo.Illustris_Cosmology_TNG(self)
        else:
            cosmo = illpy_lib.illcosmo.Illustris_Cosmology_TOS(self)

        return cosmo

    def finalize(self):
        pass
