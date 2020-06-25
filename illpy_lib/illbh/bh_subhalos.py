"""Collect and manage combined BH (particle) and subhalo (catalog) data.
"""

import os
import sys
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

# import corner
import h5py

import illpy as ill

import illpy_lib
from illpy_lib.constants import PARTICLE  # , BOX_VOLUME_MPC3, CONV_ILL_TO_SOL
import illpy_lib.subhalos.particle_hosts

from illpy_lib.illbh import Core

from zcode import plot as zplot
from zcode import math as zmath

from mbhmergers import BASE_PATH_ILLUSTRIS_1
# from mbhmergers.analysis import core

# from binagn import PATH_OUTPUT_BH_AGN_PLOTS, get_bh_subhalo_data_fname

# CONV_ILL_TO_SOL = illpy_lib.constants.CONV_ILL_TO_SOL
# from illpy_lib import illcosmo
# cosmo = illcosmo.Illustris_Cosmology_TOS()
# CONV_ILL_TO_SOL = cosmo.CONV_ILL_TO_SOL
#
# CONV_MASS = CONV_ILL_TO_SOL.MASS
# CONV_MDOT = CONV_ILL_TO_SOL.MDOT
# CONV_VEL = CONV_ILL_TO_SOL.VEL
PT_STAR = illpy_lib.constants.PARTICLE.STAR
PT_BH = illpy_lib.constants.PARTICLE.BH

# CONV_ILL_TO_SOL = COSMO.CONV_ILL_TO_SOL

# ========================
# ====    Settings    ====
# ========================

# DEF_LOG = {'debug': False, 'versioned': False}
# DEF_LOG = {'verbose': True}

# Illustris snapshot number
SNAP_NUM = 135
# Minimum BH Mass to consider
MIN_BH_MASS = 1.0e6  # [Msol]
# Number of star particles to require for subhalos
MIN_NUM_STARS = 100

SUBHALO_CAT_FIELDS = [
    'SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloHalfmassRadType', 'SubhaloLenType',
    'SubhaloMassType', 'SubhaloMassInHalfRadType', 'SubhaloSFRinHalfRad', 'SubhaloSFR',
    'SubhaloStellarPhotometrics', 'SubhaloVelDisp', 'SubhaloVmax'
]

SNAPSHOT_BH_FIELDS = [
    'Masses', 'ParticleIDs', 'HostHaloMass', 'SubfindVelDisp', 'SubfindDensity',
    'BH_Pressure', 'BH_Mdot', 'BH_Mass', 'BH_Hsml', 'BH_Density'
]


BH_SUBHALO_DATA_FNAME = 'bh_subhalos_snap-{:03d}.hdf5'
OUTPUT_BH_AGN = "/Users/lzkelley/Research/working/arepo/illustris/redesign/em_mbhb/output/bh-agn"
OUTPUT_BH_AGN_PLOTS = os.path.join(OUTPUT_BH_AGN, 'plots/')


def get_bh_subhalo_data_fname(snap_num):
    fname = os.path.join(OUTPUT_BH_AGN, BH_SUBHALO_DATA_FNAME.format(snap_num))
    return fname


def main(core=None, cat=None, snap=None, plot=False, show=False, snap_num=None):
    if core is None:
        core = Core(sets=dict(LOG_FILENAME='log_bh-subhalos.log'))

    log = core.log
    log.debug("illpy_lib.illbh.bh_subhalos.main()")
    beg_all = datetime.now()

    if snap_num is None:
        snap_num = SNAP_NUM

    # Load Subhalo Catalog
    # --------------------
    if cat is None:
        log.info("Loading groupcat data")
        beg = datetime.now()
        cat = _load_subhalo_catalog(log, snap_num)
        num_cats = len(cat['SubhaloBHMass'])
        log.after("Loaded {} groupcat entries in snap {}".format(num_cats, snap_num), beg, beg_all)

    # Plot BH-Subhalo Relations from Catalog
    # --------------------------------------
    if plot or show:
        log.info("Plotting bh-subhalo relations")
        beg = datetime.now()
        fig = _plot_bh_subhalo_relations_from_catalog(cat, core)
        log.after("Plotted", beg, beg_all)

        fname = 'bh-agn.pdf'
        fig.save(os.path.join(OUTPUT_BH_AGN_PLOTS, fname), show=show)

    # Load all BH from snapshot
    # -------------------------
    if snap is None:
        log.info("Loading snapshot data")
        beg = datetime.now()
        snap = _load_snapshot_bh(log, snap_num)
        num_snap_bh = len(snap['ParticleIDs'])
        log.after("Loaded {} BH particles from snap {}".format(
            num_snap_bh, snap_num), beg, beg_all)

    # Match each BH to parent subhalo
    # -------------------------------
    log.info("Matching BHs from snapshots to subhalo catalogs")
    beg = datetime.now()
    cat_inds_valid, cat_inds_matched, snap_inds_matched = _match_bh_subhalos(
        core, snap, cat, MIN_NUM_STARS, MIN_BH_MASS)
    log.after("Matched", beg, beg_all)

    # Plot Masses of Selected BH and Subhalos (compared to full populations)
    # ----------------------------------------------------------------------
    if plot or show:
        log.info("Plotting bh-subhalo relations")
        beg = datetime.now()
        fig = _plot_bh_subhalo_relations_from_catalog(cat, log)
        log.after("Plotted", beg)

        fname = 'catalog_and_snapshot_selections.pdf'
        fig.save(os.path.join(OUTPUT_BH_AGN_PLOTS, fname), show=show)

    # Save BH Snapshot and Host Subhalo Catalog data to file
    # ------------------------------------------------------
    log.info("Saving matched data to file")
    beg = datetime.now()
    save_fname = _save_bh_subhalo_data(
        log, snap, cat, cat_inds_matched, snap_inds_matched, MIN_BH_MASS, MIN_NUM_STARS)
    log.after("Saved data to '{}'".format(save_fname), beg, beg_all, lvl=log.WARNING)

    # Load saved data
    # ---------------
    log.info("Loading matched data")
    beg = datetime.now()
    data = load_bh_subhalos(snap_num, core)
    log.after("Loaded", beg, beg_all)

    return log, data, cat, snap, cat_inds_matched, snap_inds_matched


def load_bh_subhalos(snap_num, core=None):
    """Load a preexisting BH-Subhalos data from file.

    Returns
    -------
    data : dict
        Combined snapshot BH data and (matched) parent subhalo catalog data.

    """
    if core is None:
        core = Core()

    log = core.log

    save_fname = get_bh_subhalo_data_fname(snap_num)
    if not os.path.exists(save_fname):
        msg = "File '{}' does not exist!  Run `bh_subhalos.main()`.".format(save_fname)
        if log is not None:
            log.raise_error(msg)
        else:
            raise IOError(msg)

    if log is not None:
        log.info("Loading from filename '{}'".format(save_fname))
    data = h5py.File(save_fname, 'r')

    if log is not None:
        log.info("Loaded {} entries for snap {}, created {}".format(
            data.attrs['num'], data.attrs['snap_num'], data.attrs['created']))

    return data


def _load_subhalo_catalog(log, snap_num):
    """Use `illustris_python` to load subhalo catalog with target parameters

    Returns
    -------
    cat

    """
    log.debug("bh_subhalos._load_subhalo_catalog()")

    cat = ill.groupcat.loadSubhalos(BASE_PATH_ILLUSTRIS_1, snap_num, fields=SUBHALO_CAT_FIELDS)
    cat['snap_num'] = snap_num
    return cat


def _plot_bh_subhalo_relations_from_catalog(cat, core):
    log = core.log
    log.debug("bh_subhalos._plot_bh_subhalo_relations_from_catalog()")
    beg_all = beg = datetime.now()
    CONV_ILL_TO_SOL = core.cosmo.CONV_ILL_TO_SOL

    log.debug("Initializing")
    import corner

    # Select target data
    subh_nbh = cat['SubhaloLenType'][:, PT_BH]
    subh_vel = cat['SubhaloVelDisp']*CONV_ILL_TO_SOL.VEL
    subh_bhmass = cat['SubhaloBHMass']*CONV_ILL_TO_SOL.MASS
    subh_bhmdot = cat['SubhaloBHMdot']*CONV_ILL_TO_SOL.MDOT
    subh_stmass = cat['SubhaloMassInHalfRadType'][:, PT_STAR]*CONV_ILL_TO_SOL.MASS

    num_cats = len(cat['SubhaloBHMass'])
    fig = core.figs.Fig_Grid(log, 1, 3, share_labels=None)
    axes = np.squeeze(fig.axes)

    # Filter for desired subhalos
    # ---------------------------
    idx_1 = (subh_bhmass > 0.0)
    log.frac(np.count_nonzero(idx_1), num_cats, "Nonzero BH masses:")
    idx_2 = (subh_nbh == 1)
    log.frac(np.count_nonzero(idx_2), num_cats, "Singular BH      :")
    idx_3 = (subh_bhmass > MIN_BH_MASS)
    log.frac(np.count_nonzero(idx_3), num_cats, "BH Mass > {:.1e}    :".format(MIN_BH_MASS))
    idx = idx_1 & idx_2 & idx_3
    log.frac(np.count_nonzero(idx), num_cats, "Selecting:")
    subh_vel = subh_vel[idx]
    subh_bhmass = subh_bhmass[idx]
    subh_bhmdot = subh_bhmdot[idx]
    subh_stmass = subh_stmass[idx]

    levels = [0.5] + list(zmath.sigma([1, 2, 3]))
    corner_kwargs = {
        'smooth': True, 'levels': levels, 'no_fill_contours': True, 'plot_density': False,
        'data_kwargs': {'ms': 4.0}, 'contour_kwargs': {'alpha': 0.25}}

    log.after("initialized", beg, beg_all)

    # BH-Mass vs. Velocity Dispersion
    # -------------------------------
    ii = 0
    beg = datetime.now()
    log.debug("Plotting vs. subh-vel")
    corner.hist2d(np.log10(subh_vel), np.log10(subh_bhmass), ax=axes[ii], **corner_kwargs)
    axes[ii].set(xlabel='Subh Velocity', ylabel='BH Mass')
    log.after("plotted subh-vel", beg, beg_all)

    # BH-Mass vs. Subhalo Stellar Mass
    # --------------------------------
    ii = 1
    beg = datetime.now()
    log.debug("Plotting vs. subh-vel")
    corner.hist2d(np.log10(subh_stmass), np.log10(subh_bhmass), ax=axes[ii], **corner_kwargs)
    axes[ii].set(xlabel='Subh Stellar Mass (half-rad)', ylabel='BH Mass')
    log.after("plotted subh-vel", beg, beg_all)

    # BH-Mass vs. BH Mdot
    # -------------------
    ii = 2
    beg = datetime.now()
    log.debug("Plotting vs. BH Mdot")
    corner.hist2d(np.log10(subh_bhmdot), np.log10(subh_bhmass), ax=axes[ii], **corner_kwargs)
    axes[ii].set(xlabel='BH MDot', ylabel='BH Mass')
    log.after("plotted BH MDot", beg, beg_all)

    for ax in axes:
        zplot.set_grid(ax, True)

    return fig


def _load_snapshot_bh(log, snap_num):
    """Use `illustris_python` to load all BH particles in the snapshot with target parameters.
    """
    log.debug("bh_subhalos._load_snapshot_bh()")
    snap = ill.snapshot.loadSubset(BASE_PATH_ILLUSTRIS_1, snap_num, PT_BH,
                                   fields=SNAPSHOT_BH_FIELDS)
    snap['snap_num'] = snap_num
    return snap


def _match_bh_subhalos(core, snap, cat, nstar, bh_mass_min):
    log = core.log
    log.debug("bh_subhalos._match_bh_subhalos()")

    # Select valid (i.e. resolved) subhalos
    #    Finds the subhalo indices for subhalos with a single BH, enough stars,
    #    and BH above required mass
    # -------------------------------------
    cat_inds_valid = _match__select_valid_subhalos(core, cat, nstar, bh_mass_min)

    # Match subhalos and particles
    cat_inds_matched, snap_inds_matched = _match__match_entries(
        core, snap, cat, cat_inds_valid, bh_mass_min)

    snap_num = snap_inds_matched.size
    cat_num = cat_inds_matched.size
    log.info("Matched BH:: snap: {}, cat: {}".format(snap_num, cat_num))
    if snap_num != cat_num:
        log.raise_error("Entry numbers do not match: snap {}, cat {}".format(snap_num, cat_num))

    return cat_inds_valid, cat_inds_matched, snap_inds_matched


def _match__select_valid_subhalos(core, cat, nstar, bh_mass_min, show=False, save=None):
    """Find catalog indices for subhalos with enough star particles, a single BH above given mass.

    Arguments
    ---------
    log : `logging.Logger`
    cat : dict
        Subhalo catalog from `illustris_python.subhalos`
    nstar : int
        Minimum number of stars for a valid subhalo
    bh_mass_min : scalar
        Minimum BH-mass for a valid BH.

    Returns
    -------
    cat_inds_valid : (M,) bool
        True for subhalos in the catalog which are 'valid'.

    """
    log = core.log
    log.debug("bh_subhalos._match__select_valid_subhalos()")
    beg_all = datetime.now()
    CONV_ILL_TO_SOL = core.cosmo.CONV_ILL_TO_SOL

    log.info("Requiring {} Star particles, exactly 1 BH, BH mass >= {:.2e} [Msol]".format(
        nstar, bh_mass_min))

    # Get number of stars and BH in each subhalo
    nparts_type = cat['SubhaloLenType'][:]
    nparts_star = nparts_type[:, PARTICLE.STAR]
    nparts_bh = nparts_type[:, PARTICLE.BH]

    # Find entries with enough stars
    cat_inds_stars = (nparts_star >= nstar)
    num_good = np.count_nonzero(cat_inds_stars)
    log.frac(num_good, nparts_star.size, "Subhalos with more than {} stars".format(nstar))

    # Find entries also with only a single BH
    cat_inds_stars_bh = (cat_inds_stars & (nparts_bh == 1))
    num_good = np.count_nonzero(cat_inds_stars_bh)
    log.frac(num_good, np.count_nonzero(cat_inds_stars), "and exactly 1 BH")

    # Find entries also with the BH above the required mass
    cat_inds_valid = ((cat_inds_stars_bh) &
                      (cat['SubhaloBHMass'][:]*CONV_ILL_TO_SOL.MASS >= bh_mass_min))
    num_good = np.count_nonzero(cat_inds_valid)
    num_all = np.count_nonzero(cat_inds_stars_bh)
    log.frac(num_good, num_all, "and BH Mass >= {:.2e}".format(bh_mass_min))
    log.after("Selected valid subhalos", beg_all)

    # Plot select subhalos
    # --------------------
    if show or (save is not None):
        beg = datetime.now()
        fig = core.figs.Fig(log)
        ax = fig.default()

        label = ""
        label += "$N_\mathrm{{star}} > 10^{{{:.1f}}}$".format(np.log10(nstar))
        label += "\n$N_\mathrm{{BH}} = 1$"
        label += "\n$M_\mathrm{{BH}} > 10^{{{:.1f}}} \, M_\odot$".format(np.log10(bh_mass_min))

        xbins = zmath.spacing(nparts_star, 'log', 50)
        log.info("Bins: " + zmath.str_array(xbins, format=':.1e'))
        lines = []
        names = []
        # All subhalos
        ll = ax.hist(nparts_star, xbins, alpha=0.7)
        lines.append(ll[-1][0])
        names.append('All')
        # Particle count cuts
        ll = ax.hist(nparts_star[cat_inds_stars_bh], xbins, alpha=0.7)
        lines.append(ll[-1][0])
        names.append('Stars & BH')
        # Particle count cuts and BH-Mass cut
        ll = ax.hist(nparts_star[cat_inds_valid], xbins, alpha=0.7)
        lines.append(ll[-1][0])
        names.append('Stars & BH & BH-Mass')

        ax.set(xlabel='Star Particles', xscale='log', ylabel='Number of Subhalos', yscale='log')
        ax.axvline(nstar, color='k', ls='--', lw=2.0, alpha=0.5)
        zplot.text(ax, label, loc='uc')

        # fig._legend(lines, names, art=ax, loc='ur')
        zplot.legend(ax, lines, names, loc='ur')
        if save is not None:
            fig.save(save, show=show)
        elif show:
            plt.show(block=False)
        log.after("Plotted", beg, beg_all)

    return cat_inds_valid


def _match__match_entries(core, snap, cat, cat_inds_valid, bh_mass_min, show=False, save=None):
    """Use BH ID numbers to find parent subhalos.

    Arguments
    ---------
    log : `logging.Logger`
    snap : dict
        Dictionary of snapshot BH-particle data from `illustris_python.snapshot`
    cat : dict
        Dictionary of subhalo catalog data from `illustris_python.subhalos`
    cat_inds_valid : (M,) bool
        Boolean array with element for each subhalo in catalog.  Elements are true if the subhalo
        is 'valid' by matching the required criteria in `_match__select_valid_subhalos`.
    bh_mass_min : scalar
        Minimum mass for a 'valid' BH.
    show
    save

    Returns
    -------
    cat_inds_matched : (M,) int
        Indices for the BH snapshot particles which are valid and matched.
    snap_inds_matched : (M,) int
        Indices for the Subhalo catalog entries which are valid and matched.

    """
    log = core.log
    log.debug("bh_subhalos._match__match_entries()")
    CONV_ILL_TO_SOL = core.cosmo.CONV_ILL_TO_SOL
    beg_all = datetime.now()
    bh_masses = snap['BH_Mass'][:] * CONV_ILL_TO_SOL.MASS
    bh_ids = snap['ParticleIDs'][:]
    # print(bh_masses.size, bh_ids.size)

    # Find BH with valid masses
    log.info("Finding valid BH snapshot indices")
    snap_inds_bh = (bh_masses > 0.0)
    num_bad = bh_masses.size - np.count_nonzero(snap_inds_bh)
    log.frac(num_bad, bh_masses.size, "BH Mass <= 0.0", lvl=log.WARNING)
    snap_inds_bh_mass = (bh_masses > bh_mass_min)
    num_good = np.count_nonzero(snap_inds_bh_mass)
    log.frac(num_good, np.count_nonzero(snap_inds_bh), "and > {:.1e}".format(bh_mass_min))

    # Subhalos for each BH ID
    snap_num_cat = cat['snap_num']
    snap_num_snap = snap['snap_num']
    if snap_num_cat != snap_num_snap:
        log.raise_error("snap_num from `cat` {} does not match `snap` {}!".format(
            snap_num_cat, snap_num_snap))
    log.info("Loading particle hosts data for snapshot '{}'".format(snap_num_cat))
    subhalo_inds_all = illpy_lib.subhalos.particle_hosts.bh_subhalos(1, snap_num_cat, log, bh_ids)

    snap_inds_match = (subhalo_inds_all >= 0)
    log.frac(np.count_nonzero(snap_inds_match), subhalo_inds_all.size, "Matched subhalos")
    snap_inds_bh_match = (snap_inds_match & snap_inds_bh_mass)
    num_good = np.count_nonzero(snap_inds_bh_match)
    log.frac(num_good, snap_inds_match.size, "Matched subhalos with BH Mass")

    bh_inds = np.where(snap_inds_bh_match)[0]
    subhalo_inds = subhalo_inds_all[bh_inds]

    # `cat_inds_valid` (bool) are the good indices of all catalog subhalos based on cat selection
    #    (e.g. num stars, bh mass, etc)
    # Start with all invalid
    cat_inds_matched = np.zeros(cat_inds_valid.size, dtype=bool)
    # Set those that have been matched to valid
    cat_inds_matched[subhalo_inds] = True
    # Set those that do not pass catalog cuts to invalid
    cat_inds_matched[~cat_inds_valid] = False

    # apply the catalog cuts to the BH (snapshot) indices also
    _goods = np.where(cat_inds_matched[subhalo_inds])[0]
    # Get array of indices which are valid and matched (i.e. *not* bool array)
    snap_inds_matched = bh_inds[_goods]
    cat_inds_matched = np.where(cat_inds_matched)[0]
    log.frac(snap_inds_matched.size, bh_ids.size, "Selected and matched BH")
    log.after("Matched", beg_all)

    # Plot catalog vs. snapshot masses for all matched systems
    # --------------------------------------------------------
    if show or (save is not None):
        beg = datetime.now()
        fig = core.figs.Fig(log)
        ax = fig.default()
        zplot.set_grid(ax, True)
        ax.set(xlabel='Subhalo BH Mass', xscale='log', ylabel='Snapshot BH Mass', yscale='log')
        xx = cat['SubhaloBHMass'][subhalo_inds] * CONV_ILL_TO_SOL.MASS
        yy = snap['BH_Mass'][bh_inds] * CONV_ILL_TO_SOL.MASS
        ax.scatter(xx, yy, color='blue', s=10, alpha=0.5, label='All Matches')

        label = "Compare BH snapshot masses to catalog masses."
        label += "\nCorrect matches should be equal."
        label += "\nIncorrect matches can occur for subhalo with $N_\mathrm{BH} > 1$."
        zplot.text(ax, label, 'uc')

        # Plot catalog vs. snapshot for matched systems with catalog cuts (i.e. 1 BH)
        xx = cat['SubhaloBHMass'][cat_inds_matched] * CONV_ILL_TO_SOL.MASS
        yy = snap['BH_Mass'][snap_inds_matched] * CONV_ILL_TO_SOL.MASS
        ax.scatter(xx, yy, color='red', s=10, alpha=0.5, label='Singular BH Matches')
        ax.legend(loc='upper left')
        if save is not None:
            fig.save(save, show=show)
        elif show:
            plt.show(block=False)
        log.after("Plotted", beg, beg_all)

    return cat_inds_matched, snap_inds_matched


def _plot_selected_bh_subhalos(core, snap, cat, cat_inds_valid,
                               cat_inds_matched, snap_inds_matched):
    NBINS = 40
    log = core.log
    fig = core.figs.Fig_Grid(log, 1, 2, share_labels='')
    axes = np.squeeze(fig.axes)
    for ax in axes:
        zplot.set_grid(ax, True)

    CONV_ILL_TO_SOL = core.cosmo.CONV_ILL_TO_SOL

    # Total masses (all particle types) within stellar half-rad, for all galaxies
    xx = cat['SubhaloMassInHalfRadType'][:, :] * CONV_ILL_TO_SOL.MASS
    xx = np.sum(xx, axis=-1)
    xbins = zmath.spacing(xx, 'log', NBINS)
    log.info("xbins: " + zmath.str_array(xbins, log=True))
    log.info("Subhalo masses: " + zmath.stats_str(xx, format=':.2e'))
    axes[0].set(xscale='log', xlabel='Subhalo Masses', yscale='log', ylabel='Number')
    axes[0].hist(xx, xbins, color='blue', histtype='step', lw=2.0, label='All (Total Mass)')
    axes[0].hist(xx[cat_inds_valid], xbins, color='red', histtype='step', lw=2.0, label='Cat Selected')
    axes[0].hist(xx[cat_inds_matched], xbins, color='purple', histtype='step', lw=2.0, label='Selected')

    xx = cat['SubhaloMassInHalfRadType'][:, PARTICLE.STAR] * CONV_ILL_TO_SOL.MASS
    axes[0].hist(xx, xbins, color='blue', histtype='step', lw=2.0, ls='--', label='All (Stellar Mass)')
    axes[0].hist(xx[cat_inds_valid], xbins, color='red', histtype='step', lw=2.0, ls='--')
    axes[0].hist(xx[cat_inds_matched], xbins, color='purple', histtype='step', lw=2.0, ls='--')
    axes[0].legend(loc='upper right')

    # BH Masses
    xx = snap['BH_Mass'][:] * CONV_ILL_TO_SOL.MASS
    xbins = np.logspace(5, 10.5, 45)
    log.info("xbins: " + zmath.str_array(xbins, log=True))
    log.info("BH Masses: " + zmath.stats_str(xx, format=':.2f', log=True))

    axes[1].set(xscale='log', xlabel='BH Masses', yscale='log', ylabel='Number')
    axes[1].hist(xx, xbins, histtype='step', color='red', lw=2.0, label='All')
    nums, bins, patches = axes[1].hist(
        xx[snap_inds_matched], xbins, color='purple', histtype='step', lw=2.0, label='Selected')
    axes[1].legend(loc='upper right')

    return fig


def _save_bh_subhalo_data(log, snap, cat, cat_inds_matched, snap_inds_matched, bh_mass_min, nstar):
    log.debug("bh_subhalos._save_bh_subhalo_data()")
    snap_num = snap['snap_num']
    if snap_num != cat['snap_num']:
        log.raise_error("Snapshot numbers do not match: snap {}, cat {}".format(
            snap_num, cat['snap_num']))

    num_bh = snap_inds_matched.size
    if num_bh != cat_inds_matched.size:
        log.raise_error("Entry numbers do not match: snap {}, cat {}".format(
            num_bh, cat_inds_matched.size))

    desc_str = \
        """This file contains the combined snapshot and subhalo catalog data for BHs
        and their hosts for snapshot {snap}.
        BH have been selected to have masses above {bh_mass_min} [Msol].
        Subhalos have been selected to have more than {nstar} star particles, and exactly one BH
        with the above mass-cut.  {num} matched entries result from {nbh} total blackholes in the
        snapshot, and {nsh} total subhalos in the catalog.
        """.format(snap=snap_num, bh_mass_min=bh_mass_min, nstar=nstar, num=num_bh,
                   nbh=snap['count'], nsh=cat['count'])

    skip_keys = ['count', 'snap_num']

    save_fname = get_bh_subhalo_data_fname(snap_num)
    log.info("Saving to filename '{}'".format(save_fname))
    with h5py.File(save_fname, 'w') as h5file:
        h5file.attrs['snap_num'] = snap_num
        h5file.attrs['num'] = num_bh
        h5file.attrs['desc'] = str(desc_str)
        h5file.attrs['script'] = str(__file__)
        h5file.attrs['created'] = str(datetime.now().ctime())

        # Store indices
        h5file.create_dataset('cat_inds_matched', data=cat_inds_matched)
        h5file.create_dataset('snap_inds_matched', data=snap_inds_matched)

        # Copy catalog entries
        saved_keys = []
        for key, vals in cat.items():
            if key in skip_keys:
                continue
            log.debug("Creating catalog dataset for key '{}'".format(key))
            h5file.create_dataset(key, data=vals[cat_inds_matched])
            saved_keys.append(key)

        # Copy snapshot entries
        for key, vals in snap.items():
            if key in skip_keys:
                continue
            log.debug("Creating snapshot dataset for key '{}'".format(key))
            h5file.create_dataset(key, data=vals[snap_inds_matched])
            saved_keys.append(key)

    fsize = os.path.getsize(save_fname)/1024/1024
    log.info("Saved {} keys: {}".format(len(saved_keys), ", ".join(saved_keys)))
    log.info("Saved to '{}', Size: '{:.2e}' MB".format(save_fname, fsize))
    return save_fname


if __name__ == "__main__":
    print(sys.argv)
    snap_num = None
    if len(sys.argv) > 1:
        snap_num = int(sys.argv[1])
        print("snap_num = ", snap_num)

    main(snap_num=snap_num)
