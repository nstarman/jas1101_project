# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : Normalize Globular Clusters
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""**DOCSTRING**.

description

Routine Listings
----------------

"""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]

# __copyright__ = "Copyright 2019, "
# __credits__ = [""]
# __license__ = ""
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

# __all__ = [
#     ""
# ]


##############################################################################
# IMPORTS

# GENERAL

import os
import pathlib
import warnings
import argparse
from typing import Optional

import tqdm

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


# PROJECT-SPECIFIC

from ... import data
from ...GlobularCluster import GlobularCluster
from ... import plot, utils, cluster

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS

FIGURES = str(pathlib.Path(__file__).parent.absolute()) + "/figures/"

if not os.path.isdir(FIGURES):
    os.mkdir(FIGURES)


##############################################################################
# CODE
##############################################################################


def plot_1(GC):
    plt.figure(figsize=(7, 7))
    H, xb, yb, _ = plt.hist2d(
        GC.x,
        GC.y,
        bins=100,
        range=[[-1.5, 1.5], [-1.5, 1.5]],
        norm=plot.LogNorm(),
        cmap="gnuplot2",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_space.pdf")
    plt.close()


# /def


# ------------------------------------------------------------------------


def plot_2(GC):
    plt.figure(figsize=(7, 7))
    sns.distplot(GC.r)
    plt.xlabel("R")
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_dist.pdf")
    plt.close()


# /def


# ------------------------------------------------------------------------


def plot_3(GC):
    plt.figure(figsize=(6, 6))
    H, xb, yb, _ = plt.hist2d(
        GC.pmx,
        GC.pmy,
        bins=100,
        range=[[-15, 15], [-15, 15]],
        norm=colors.LogNorm(),
        cmap="gnuplot2",
    )
    plt.xlabel("PMX")
    plt.ylabel("PMY")
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_pm.pdf")
    plt.close()


# /def

# ------------------------------------------------------------------------


def plot_4(GC, clus):
    clus.plot_clustering()
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_cluster.pdf")
    plt.close()


# ------------------------------------------------------------------------


def plot_5(GC, sel, bins):
    plt.figure()    
    fig = plot.plot_binned_std_profile(r=GC.r[sel],
                                       pm=GC.pm[sel],
                                       bins=bins)
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_sigma_rbin.pdf")
    plt.close()

# ------------------------------------------------------------------------


def plot_6(GC, sel):
    plt.figure()
    stats.probplot(GC.pm[sel], dist="norm", plot=plt.gca())
    plt.savefig(FIGURES + f"{GC.name}/{GC.name}_QQ.pdf")
    plt.close()

##############################################################################
# Command Line
##############################################################################


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
        description="",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "output_dir", type=str, help="The data output directory",
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

    summary = data.load_summary_table()

    for name in tqdm.tqdm(summary["Name"]):
        # print(name)

        if not os.path.isdir(FIGURES + name):
            os.mkdir(FIGURES + name)

        # load GC
        GC = GlobularCluster.from_name(
            name,
            member_threshold=0.0,  # to ensure that all the data is returned
        )

        plot_1(GC)
        plot_2(GC)
        plot_3(GC)

        # Data stack
        X0 = np.vstack([GC.r, GC.pmx, GC.pmy]).T

        # clip major outliers
        good_pm = utils.clip_quantile_nd(
            X0, z_quantile=[0.001, 0.999], ind_clip=[1, 2]
        )
        
        # clip in radius
        r_min, r_max = 0.25, 1
        good_r = (X0[:,0]>r_min) & (X0[:,0]<r_max)

        if sum(good_pm & good_r) == 0:
            print("no good pm")

        X = X0[good_pm & good_r]

        # cluster data
        try:
            clus = cluster.DBSCAN_Clustering(X, verbose=False)
            clus.run_clustering(plot=False)  # if use DBSCAN : eps = 0.5
        except ValueError:  # not enough data point
            print(f"{name} failed")
            member_prob = 1.0
        else:

            plot_4(GC, clus)

            # predict main population from clustering
            is_mp = clus.predict_main_pop()

            # boolean membership probability
            member_prob = np.zeros(len(X0))
            sel = np.where(good_pm & good_r)[0][is_mp]
            member_prob[sel] = 1
            
            # plot std of pm in radial bins
#             plot_5(GC, sel, bins=np.linspace(r_min, r_max, 5))
            
            # qq plot
            plot_6(GC, sel)

        # add to GC table
        df = data.load_globular_cluster(name)
        df["member_prob_DB"] = member_prob

        # save
        output_dir: str = opts.output_dir
        output_dir = (
            output_dir if output_dir.endswith("/") else output_dir + "/"
        )

        df.write(
            output_dir + "gcs/{}.ecsv".format(name),
            format="ascii.ecsv",
            overwrite=True,
        )

    return


# /def


##############################################################################
# END
