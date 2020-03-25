# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""Plot Utilities.

Routine Listings
----------------
LogNorm
AsinhNorm
HistEqNorm
rand_color
colorbar_non_mappable
plot_clustering
plot_GMM_1d
plot_GMM_2d
plot_BIC_diagnosis
plot_binned_profile
plot_gc_hist2d_summary


"""

__all__ = [
    "LogNorm" "AsinhNorm",
    "HistEqNorm",
    "rand_color",
    "colorbar_non_mappable",
    "plot_clustering",
    "plot_GMM_1d",
    "plot_GMM_2d",
    "plot_BIC_diagnosis",
    "plot_binned_profile",
    "plot_gc_hist2d_summary",
]


###############################################################################
# IMPORTS

# GENERAL
import numpy as np
from scipy import stats

# astropy
from astropy.visualization import LogStretch, AsinhStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns


# PROJECT-SPECIFIC

from .stats import confidence_ellipse
from .utils import profile_binning


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


def LogNorm():
    """Custom LogNorm.

    Returns
    -------
    ImageNormalize

    """
    return ImageNormalize(stretch=LogStretch())


# /def


def AsinhNorm(a=0.1):
    """Custom Arcsinh Norm.

    Parameters
    ----------
    a : float, optional

    Returns
    -------
    ImageNormalize

    """

    return ImageNormalize(stretch=AsinhStretch(a=a))


# /def


def HistEqNorm(data):
    """Custom Histogram Equalization Norm.

    Parameters
    ----------
    data

    Returns
    -------
    ImageNormalize

    """
    return ImageNormalize(stretch=HistEqStretch(data))


# /def


def rand_color(N=1):
    """Random Color.

    Parameters
    ----------
    N : int, optional

    """
    return [np.random.random(size=(3)) for i in range(N)]


# /def


# --------------------------------------------------------------------------


def colorbar_non_mappable(fig, ax, cmap="magma", vmin=0, vmax=1):
    """Colorbar non-mappable.

    Parameters
    ----------
    fig : Figure
    ax : Axes
    cmap : str, optional
    vmin : int, optional
    vmax : int, optional

    Returns
    -------


    """
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
# Clustering related plot


def plot_clustering(
    X,
    n_dim: int,
    labels: list,
    feature_labels: list = ["R", "PMX", "PMY"],
    figsize: tuple = (11, 10),
    ms: int = 3,
    alpha: float = 0.1,
):
    """Plot Clustering.

    Parameters
    ----------
    X : array-like
    n_dim : int
    labels : list
    feature_labels : list, optional
    figsize : tuple, optional
    ms : int, optional
    alpha : float, optional

    Returns
    -------
    fig : Figure

    """
    X = X
    k: int = n_dim - 1

    fig = plt.figure(figsize=figsize)

    i: int
    j: int
    for i in range(k):
        for j in range(k):
            if i > j:
                continue

            plt.subplot2grid((k, k), (k - i - 1, j), rowspan=1, colspan=1)

            plt.scatter(X[:, i], X[:, j + 1], c=labels, s=ms, alpha=alpha)

            plt.xlabel(feature_labels[i])
            plt.ylabel(feature_labels[j + 1])

    plt.tight_layout()

    return fig


# /def


# --------------------------------------------------------------------------
# GMM related plot


def plot_GMM_1d(data, weights, means, sigmas, sample, verbose=True):
    """Plot GMM decomposation in 1D.

    Parameters
    ----------
    data : array-like
    weights : array-like
    means : array-like
    sigmas : array-like
    sample : array-like
    verbose: bool, optional

    Returns
    -------
    ax : Axes

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.distplot(data, kde_kws={"lw": 4}, color="plum", label="data")
    sns.distplot(
        sample,
        hist=False,
        color="k",
        kde_kws={"lw": 4, "alpha": 0.7},
        label="GMM Fit",
    )
    plt.legend(loc="best")

    for w, m, s, in zip(weights, means, sigmas):
        rv = stats.norm(loc=m, scale=s)
        x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
        plt.plot(x, w * rv.pdf(x), "--", color="k", lw=3, alpha=0.7)

    plt.xlim(0, np.quantile(data, 0.999))
    plt.xlabel("Proper Motion (mas)")
    plt.ylabel("PDF")

    return ax


# /def


def plot_GMM_2d(
    data,
    weights,
    means,
    covariances,
    bins=100,
    range=None,
    k_std=2,
    norm=None,
    verbose=True,
):
    """Plot GMM decomposation in 2D.

    Parameters
    ----------
    data : array-like
    weights : array-like
    means : array-like
    covariances : array-like
    bins : int, optional
    range : tuple, optional
    k_std : int, optional
    norm : bool, optional
    verbose : bool, optional

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    xrange = np.quantile(data[:, 0], [0.001, 0.999])
    yrange = np.quantile(data[:, 1], [0.001, 0.999])

    H, xb, yb, _ = plt.hist2d(
        data[:, 0],
        data[:, 1],
        bins,
        range=[xrange, yrange],
        norm=norm,
        cmap="gnuplot2",
    )

    N_comp = len(weights)

    for i, (mean, cov, color, weight) in enumerate(
        zip(means, covariances, rand_color(N_comp), np.argsort(weights))
    ):
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            mean,
            k_std * v[0],
            k_std * v[1],
            180 + angle,
            edgecolor=color,
            facecolor="none",
            lw=4,
            zorder=weight + 1,
        )
        ax.add_artist(ell)

    # Robust confidence ellipse
    mean_tot, v_tot, angle_tot = confidence_ellipse(
        data, robust=True, verbose=verbose
    )

    # Plot an ellipse to show the Gaussian component
    el = mpl.patches.Ellipse(
        mean_tot,
        k_std * v_tot[0],
        k_std * v_tot[1],
        180 + angle_tot,
        edgecolor="w",
        facecolor="none",
        lw=4,
        alpha=1,
        zorder=10,
    )
    ax.add_artist(el)

    plt.xlabel("PMX")
    plt.ylabel("PMY")

    return


# /def


def plot_BIC_diagnosis(
    BIC, ax=None, bbox_to_anchor=(-0.02, -0.2, 1, 1), style="dark_background"
):
    """Plot BIC vs N_comp.

    Parameters
    ----------
    BIC : float
    ax : Axes, optional
    bbox_to_anchor : tuple, optional
    style : str, optional

    Returns
    -------
    axins : Axes

    """
    if BIC is None:
        return None

    if ax is None:
        ax = plt.gca()

    N_comp = np.arange(1, len(BIC) + 1, 1)

    with plt.style.context(style):

        axins = inset_axes(
            ax,
            width="30%",
            height="30%",
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
        )
        axins.plot(N_comp, BIC / BIC.min(), "ro-")
        axins.text(
            0.5,
            0.8,
            "N$_{best}$=%d" % N_comp[np.argmin(BIC)],
            transform=axins.transAxes,
        )
        axins.axhline(1, color="k", ls="--", zorder=1)
        axins.set_ylabel("BIC / BIC$_{min}$", fontsize=12)
        axins.tick_params(axis="both", which="major", labelsize=10)

    return axins


# /def


# --------------------------------------------------------------------------


def plot_binned_profile(r, pm, bins=None, z_clip=None):
    """Plot distribution of pm in each radial bin.

    Parameters
    ----------
    r : array-like
    pm : array-like
    bins : tuple, optional
    z_clip : dict, optional

    Returns
    -------
    fig : Figure

    """
    if bins is None:
        raise Exception("need to pass bins or call bin_profile")

    with sns.color_palette("husl", len(bins)):
        fig = plt.figure()
        profile_binning(
            r, pm, bins=bins, z_clip=z_clip, return_bin=False, plot=True,
        )
        plt.xlabel("pm [mas/yr]")
        plt.ylabel("density")
        plt.show()

    return fig


# /def


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
        gc.table["pmx"].value,
        gc.table["pmy"].value,
        bins=100,
        # norm=LogNorm(),
        cmap="gnuplot2",
    )
    plt.xlabel("v_x [{}]".format(gc.table["pmx"].unit))
    plt.ylabel("v_y [{}]".format(gc.table["pmy"].unit))

    plt.tight_layout()

    return fig


# /def


###############################################################################
# Command Line
###############################################################################


###############################################################################
# END
