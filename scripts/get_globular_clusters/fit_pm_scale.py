# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""**DOCSTRING**.

Warnings
--------
SEE normalize_pm notebook for exploration of Gaussianity (assumed here)
because it is NOT true.


Routine Listings
----------------

"""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]


# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL
import os
import pathlib
import warnings
import argparse
import tqdm
from typing import Optional

import numpy as np

import scipy.optimize as optimize

import astropy.units as u
from astropy.table import Table, QTable
from astropy.stats import SigmaClip

# CUSTOM

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.insert(0, '../../')
# fmt: on

from src.util import gaussfitter


###############################################################################
# PARAMETERS

warnings.simplefilter("always", UserWarning)


###############################################################################
# CODE
###############################################################################


def read_globular_cluster_table(file):
    """Docstring."""
    df = QTable.read(file, format="ascii.commented_header")

    units = {
        "x": u.deg,
        "y": u.deg,
        "pmx": u.mas / u.yr,
        "pmy": u.mas / u.yr,
        "pmx_e": u.mas / u.yr,
        "pmy_e": u.mas / u.yr,
        "g_mag": u.mag,
        "bp_rp": u.mag,
    }

    for name, unit in units.items():
        df[name].unit = unit

    return df


# /def


def read_summary_table(file):
    """Read summary table to be in Astropy format.

    Parameters
    ----------
    file: str
        file to read with QTable

    Returns
    -------
    df : QTable

    """
    df = Table.read(file, format="ascii.commented_header")
    df.add_index("Name")

    units = {
        "ra": u.deg,
        "dec": u.deg,
        "dist": u.kpc,
        "vlos": u.km / u.s,
        "vloserr": u.km / u.s,
        "sigma": u.km / u.s,
        "rmax": u.arcmin,
        "pmra": u.mas / u.yr,
        "pmdec": u.mas / u.yr,
        "pmra_e": u.mas / u.yr,
        "pmdec_e": u.mas / u.yr,
        "rscale": u.arcmin,
        "pmdisp": u.mas / u.yr,
        "pmscale": u.mas / u.yr,
        "pmscale_e": u.mas / u.yr,
    }

    for name, unit in units.items():
        if name in df.columns:
            df[name].unit = unit

    return QTable(df)


# /def

# ------------------------------------------------------------------------
# https://scipy-cookbook.readthedocs.io/items/FittingData.html


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2)
        / 2
    )


# /def


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(
        np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum()
    )
    row = data[int(x), :]
    width_y = np.sqrt(
        np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum()
    )
    height = data.max()
    return height, x, y, width_x, width_y


# /def


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(
        gaussian(*p)(*np.indices(data.shape)) - data
    )
    p, cov, infodict, *errmsg = optimize.leastsq(
        errorfunction, params, full_output=True
    )
    return p, cov, infodict, errmsg


# /def


# ------------------------------------------------------------------------


def scale_values_2d(df, threshold=0.8):
    """

    Use Sturge’s Rule to determine number of bins

    TODO
    ----
    don't choose arbitrary threshold
    don't use histogram?
    not arbitrary rotation threshold

    """

    ismember = df["memberprob"] > threshold

    pmx = df["pmx"][ismember].to_value("mas / yr")
    pmy = df["pmy"][ismember].to_value("mas / yr")

    # Sigma Clip major outliers

    sigclip = SigmaClip(sigma=3, maxiters=1.0)
    resx = sigclip(pmx)
    resy = sigclip(pmy)

    pmx = resx.data[~resx.mask & ~resy.mask]
    pmy = resy.data[~resx.mask & ~resy.mask]

    # Now histogram
    # need equi-spaced bins
    # TODO error estimate from bin size

    data, *edges = np.histogram2d(
        pmx, pmy, bins=int(1 + 3.222 * np.log(len(pmx))), density=True
    )

    # fit 2D Gaussian, with freedom of rotation
    params, cov, infodict, errmsg = gaussfitter.gaussfit(
        data, circle=0, rotate=1, vheight=1, return_all=True
    )

    height, amp, x, y, width_x, width_y, rota = params

    labels = ("height", "amp", "x", "y", "width_x", "width_y", "rota")

    # Check if need to do a rotated system. get better results if don't.
    if rota < 2:  # not rotated
        amp = rota = None
        params, cov, infodict, errmsg = fitgaussian(data)
        height, x, y, width_x, width_y = params

        labels = ("height", "x", "y", "width_x", "width_y")

    return width_x, width_y, cov, labels, edges


# /def


# ------------------------------------------------------------------------


def average_scale_value(width_x, width_y, edges_x, edges_y):

    flag = False

    if not np.allclose(np.diff(edges_x)[:-1], np.diff(edges_x)[1:]):
        warnings.warn("x edges are not equally spaced")
        flag = True
    if not np.allclose(np.diff(edges_y)[:-1], np.diff(edges_y)[1:]):
        warnings.warn("y edges are not equally spaced")
        flag = True

    pm_per_bin_x = np.diff(edges_x)[0] * u.mas / u.yr
    pm_per_bin_y = np.diff(edges_y)[0] * u.mas / u.yr

    pm_scale = (width_x * pm_per_bin_x + width_y * pm_per_bin_y) / 2

    # eror estimate
    pm_scale_err = np.abs(width_x * pm_per_bin_x - width_y * pm_per_bin_y) / 2

    return pm_scale, pm_scale_err, flag


# /def


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Expose parser for ``main``.

    Parameters
    ----------
    inheritable: bool
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    Returns
    -------
    parser: ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="fit_pm_scale",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data",
        help="The data directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="The input data directory",
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(
    args: Optional[list] = None, opts: Optional[argparse.Namespace] = None
):
    """Script Function.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : Namespace, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        parser = make_parser()
        opts = parser.parse_args(args)

    data_dir: str = opts.data_dir
    result_dir = str(pathlib.Path(data_dir).parent)

    # reformat
    data_dir = data_dir if data_dir.endswith("/") else data_dir + "/"
    result_dir = result_dir if result_dir.endswith("/") else result_dir + "/"

    summary = read_summary_table(result_dir + "result.txt")
    summary["pmscale"] = np.NaN * u.mas / u.yr
    summary["pmscale_e"] = np.NaN * u.mas / u.yr

    # globular clusters
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".txt")]

    for file in tqdm.tqdm(files):

        name = file[: -len(".txt")]

        gc = read_globular_cluster_table(data_dir + file)

        # compute scale parameter
        width_x, width_y, cov, labels, edges = scale_values_2d(
            gc, threshold=0.8
        )
        pm_scale, pm_scale_err, flag = average_scale_value(
            width_x, width_y, *edges
        )
        if flag:
            warnings.warn(name + " raised the previous warning")

        if np.isnan(pm_scale):
            warnings.warn(name + " has NaN pm scale")

        summary.loc[name]["pmscale"] = np.round(pm_scale, 3)
        summary.loc[name]["pmscale_e"] = np.round(pm_scale_err, 3)

    # # /for

    summary.write(
        result_dir + "result.txt",
        format="ascii.commented_header",
        overwrite=True,
    )

    summary = read_summary_table(result_dir + "result.txt")

    return


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    print("Running fit_pm_scale script.")

    main(args=None, opts=None)

    print("finished.")

# /if


###############################################################################
# END
