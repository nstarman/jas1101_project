# -*- coding: utf-8 -*-

"""Utilities.

Routing Listings
----------------
convert_angle
convert_pm_angular
clip_quantile_nd
clip_quantile_1d
profile_binning
get_mean_rbins

"""

__all__ = [
    "convert_angle",
    "convert_pm_angular",
    "clip_quantile_nd",
    "clip_quantile_1d",
    "profile_binning",
    "get_mean_rbins",
]


###############################################################################
# IMPORTS

# GENERAL
import numpy as np

from typing import Optional, Sequence

import astropy.units as u

# plot
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


###############################################################################
# PARAMETERS

rcParams.update(
    {
        "figure.figsize": [7, 5],
        "xtick.major.pad": "5.0",
        "xtick.major.size": "4",
        "xtick.major.width": "1.",
        "xtick.minor.pad": "5.0",
        "xtick.minor.size": "4",
        "xtick.minor.width": "0.8",
        "ytick.major.pad": "5.0",
        "ytick.major.size": "4",
        "ytick.major.width": "1.",
        "ytick.minor.pad": "5.0",
        "ytick.minor.size": "4",
        "ytick.minor.width": "0.8",
        "axes.labelsize": 14,
        "font.size": 14,
    }
)


###############################################################################
# CODE
###############################################################################


@u.quantity_input(angle=u.deg, distance="length")
def convert_angle(angle, distance) -> u.kpc:
    """Convert Angle.

    Parameters
    ----------
    angle : Quantity
        deg
    distance : Quantity
        kpc

    Returns
    -------
    Quantity

    """
    return distance * np.tan(angle)


# /def


# --------------------------------------------------------------------------


@u.quantity_input(angle=u.mas / u.yr, distance="length")
def convert_pm_angular(velocity, distance) -> u.km / u.s:
    """Convert PM in angular coordinates.

    Parameters
    ----------
    velocity : Quantity
        mas/yr
    distance : Quantity
        kpc

    Returns
    -------
    velocity : Quantity

    """
    v = velocity.to(u.mas / u.yr)
    return distance * np.tan(v * u.yr) / u.yr


# /def

# --------------------------------------------------------------------------


def clip_quantile_nd(
    z: np.ndarray,
    z_quantile: Optional[Sequence] = None,
    ind_clip: list = [1, 2],
    return_func: bool = False,
):
    """Clip function based on quantile for N-d array.

    Parameters
    ----------
    z : array-like
        N-d array [N_samples, N_dimensions]
    z_quantile : list
        quantile [lower, upper]  (float: 0 ~ 1)
    ind_clip : array-like
        which columns of z to clip
    return_func : bool
        whether to return a function or the clipped array

    Example
    -------
    good_pm = clip_quantile_1d(np.vstack([r, pmx, pmy]).T)

    Return
    ------
    A function or N-d array.

    """
    if z_quantile is None:
        z_quantile = [0.001, 0.999]

    z_clip = np.quantile(z, z_quantile, axis=0)
    # n_dim = z.shape[1]

    clip = lambda z_: np.logical_and.reduce(
        [
            (z_[:, j] > z_clip[0, j]) & (z_[:, j] < z_clip[1:, j])
            for j in ind_clip
        ],
        axis=0,
    )

    if return_func:
        return clip
    else:
        return clip(z)


# /def


def clip_quantile_1d(
    z: np.ndarray,
    z_quantile: Optional[Sequence] = None,
    return_func: bool = False,
):
    """Clip function based on given quantile.

    Parameters
    ----------
    z : 1d array
    z_quantile : quantile [lower, upper]  (float: 0 ~ 1)
    return_func : whether to return a function or the clipped array

    Example
    -------
    good_pmx = clip_quantile_1d(pmx)

    Return
    ------
    A function or N-d array.

    """
    if z_quantile is None:
        z_quantile = [0.001, 0.999]

    z_clip = np.quantile(z, z_quantile)

    clip = lambda z_: (z_ > z_clip[0]) & (z_ < z_clip[1])

    if return_func:
        return clip
    else:
        return clip(z)


# /def


def profile_binning(
    r: np.ndarray,
    z: np.ndarray,
    bins: np.ndarray,
    z_name: str = "pm",
    z_clip: Optional[dict] = None,
    z_quantile: Optional[Sequence] = None,
    return_bin: bool = True,
    plot: bool = True,
):
    """Bin the given quantity z in r.

    Parameters
    ----------
    r: array-like
    z: array-like
    bins: array-like
    z_name: str
    z_clip: dict, optional
    z_quantile: Sequence, optional
    return_bin: bool, optional
    plot: bool, optional

    Returns
    -------
    r_rbin : array-like
    z_rbin : array-like
    z_bins : dict

    """
    if z_clip is None:
        clip = clip_quantile_1d(z, z_quantile, return_func=True)
    else:
        clip = lambda z_: (z_ > z_clip[0]) & (z_ < z_clip[1])

    z_bins = {}

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Clip by bins
    for k, b in enumerate(bins[:-1]):
        in_bin = (bins[k] <= r) & (r < bins[k + 1])
        clipped = clip(z[in_bin])
        z_in_bin = z[in_bin][clipped]
        r_in_bin = r[in_bin][clipped]

        z_bin = {z_name: z_in_bin, "r": r_in_bin}
        z_bins[k] = z_bin

        if plot:
            lab = "{0:.2f}<r<{1:.2f}".format(bins[k], bins[k + 1])
            sns.distplot(
                z_in_bin,
                hist=False,
                kde_kws={"lw": 2, "alpha": 0.9},
                label=lab,
            )

    r_rbin, z_rbin = get_mean_rbins(z_bins, z_name=z_name)

    z_bins = z_bins if return_bin else None

    return r_rbin, z_rbin, z_bins


# /def

# --------------------------------------------------------------------------


def get_mean_rbins(z_bins, z_name="pm"):
    """Get mean of radial bins.

    Parameters
    ----------
    z_bins
    z_name : str, optional

    Returns
    -------
    r_rbin : array-like
    z_rbin : array-like

    """
    res = np.array(
        [
            [np.mean(val["r"]), np.mean(val[z_name])]
            for key, val in z_bins.items()
        ]
    )
    r_rbin, z_rbin = res[:, 0], res[:, 1]
    return r_rbin, z_rbin


###############################################################################
# Command Line
###############################################################################


###############################################################################
# END
