# -*- coding: utf-8 -*-

"""Plot Utilities.

Routing Listings
----------------

"""

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL
import numpy as np

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import seaborn as sns

# astropy
import astropy.units as u
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


# CUSTOM

# PROJECT-SPECIFIC


###############################################################################
# PARAMETERS

rcParams.update({"figure.figsize": [7, 5]})
rcParams.update({"xtick.major.pad": "5.0"})
rcParams.update({"xtick.major.size": "4"})
rcParams.update({"xtick.major.width": "1."})
rcParams.update({"xtick.minor.pad": "5.0"})
rcParams.update({"xtick.minor.size": "4"})
rcParams.update({"xtick.minor.width": "0.8"})
rcParams.update({"ytick.major.pad": "5.0"})
rcParams.update({"ytick.major.size": "4"})
rcParams.update({"ytick.major.width": "1."})
rcParams.update({"ytick.minor.pad": "5.0"})
rcParams.update({"ytick.minor.size": "4"})
rcParams.update({"ytick.minor.width": "0.8"})
rcParams.update({"axes.labelsize": 14})
rcParams.update({"font.size": 14})


###############################################################################
# CODE
###############################################################################


# def LogNorm():
#     """Custom LogNorm.

#     Returns
#     -------
#     ImageNormalize

#     """
#     return ImageNormalize(stretch=LogStretch())


# # /def


# --------------------------------------------------------------------------


def colobar_non_mappable(fig, ax, cmap="magma", vmin=0, vmax=1):
    """colobar_non_mappable."""
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(
        ax_cb, cmap=plt.cm.get_cmap(cmap), norm=norm, orientation="vertical"
    )
    cb.set_label("M$_{BH}$")
    fig.add_axes(ax_cb)

    return fig


# /def


# --------------------------------------------------------------------------


def plot_gc_hist2d_summary(gc):
    """plot_gc_hist2d.

    Parameters
    ----------
    gc : GlobularCluster

    Returns
    -------
    fig : Figure

    """
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    # spatial
    plt.sca(axs[0, 0])
    sns.distplot(gc.r)
    plt.xlabel("r / rscale")
    plt.ylabel("density")

    plt.sca(axs[1, 0])
    H, xb, yb, _ = axs[1, 0].hist2d(
        gc.x, gc.y, bins=200, cmap="gnuplot2",  # norm=LogNorm(),
    )
    plt.xlabel("x / rscale")
    plt.ylabel("y / rscale")

    # proper motion
    plt.sca(axs[0, 1])
    sns.distplot(gc.pm.value)
    plt.xlabel("PM [{}]".format(gc.pm.unit))
    plt.ylabel("density")

    plt.sca(axs[1, 1])
    H, xb, yb, _ = axs[1, 1].hist2d(
        gc.df["pmx"].value,
        gc.df["pmy"].value,
        bins=100,
        # norm=LogNorm(),
        cmap="gnuplot2",
    )
    plt.xlabel("v_x [{}]".format(gc.df["pmx"].unit))
    plt.ylabel("v_y [{}]".format(gc.df["pmy"].unit))

    plt.tight_layout()

    return fig


# /def

# --------------------------------------------------------------------------

# def plot


###############################################################################
# Command Line
###############################################################################


###############################################################################
# END
