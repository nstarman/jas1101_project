# -*- coding: utf-8 -*-

"""A collection of tools for analyzing the data from the Gaia satellite.

Code modified from project written by Eugene Vasiliev

The description below follows the order in which these tools should be used.

    get_mean_pm.py:
A small module providing the eponymous routine for computing the mean proper
motion (PM) and optionally its dispersion for a star cluster, with or without
accounting for spatially correlated systematic errors, as described in the
Appendix of arXiv:1811.05345.
This file can be run as a main program, illustrating a few example fits.
It is also used by run_fit.py to measure the systematic uncertainty in the mean
PM of globular clusters.
DEPENDENCIES: numpy, scipy.

    input.txt:
List of globular clusters in the Milky Way (input for the remaining scripts).
Data is taken from the following catalogues:
Harris 2010 (sky coordinates and distance, except for a couple of corrections);
Baumgardt+ 2019 (line-of-sight velocity and its error estimate, and central
velocity dispersion);
rmax is the maximum distance from cluster center used to query the Gaia archive;
pmra, pmdec is the initial guess for the mean PM of the cluster used in the fit.

    McMillan17.ini:
The Milky Way potential from McMillan 2017, used in orbit integrations
(run_orbits.py)

    query_gaia_archive.py:
Retrieve the data from the Gaia archive (all sources satisfying the maximum
distance from cluster center and a simple parallax cut).
Source data for each cluster is stored in a separate numpy zip file:
"data/[cluster_name].npz".
Additionally, the table for computing the renormalized unit weight error
(an astrometric quality flag) is retrieved from the Gaia website and stored in
"DR2_RUWE_V1/table_u0_2D.txt".
DEPENDENCIES: numpy, scipy, astropy, astroquery (astropy-affiliated package).
RESOURCES: run time: a few minutes (depending on internet speed);
disk space: a few tens of megabytes to store the downloaded data.

    run_fit.py:
The main script performing the membership determination and measuring the mean
PM for each cluster, as described in the Appendix of arXiv:1807.09775.
It reads the data previously stored in "data/*.npz" by query_gaia_archive.py,
performs the fit, estimates the uncertainties on the mean PM (optionally taking
into account systematic errors), writes the summary for each cluster to the file
"result.txt" (same columns as "input.txt", plus the following ones:
uncertainties in mean PMra, PMdec and the correlation coefficient;
number of members (not an integer, since the inference is probabilistic);
scale radius of a Plummer profile of cluster members from the filtered Gaia
catalogue - not the same as the half-mass radius of all stars in the cluster);
internal PM dispersion in the center inferred during the fit (not a robust
measurement, rather a nuisance parameter).
Additionally, the data for all stars from each cluster are written to a file
"data/[cluster_name].txt":
x,y are orthogonally projected coordinates w.r.t. cluster center (in degrees);
pmx, pmy is the quasi-PM in these orthogonally projected coordinates (mas/yr);
pmx_e, pmy_e, pm_corr are their uncertainties and correlation coefficient;
g_mag is G-band magnitude;  bp_rp is the color;
filter is the flag (0/1) specifying whether the star passed the quality filters
on the initial sample (i.e., astrometric noise and photometry);
prob is the cluster membership probability (only for stars with filter==1).
DEPENDENCIES: numpy, scipy; optionally autograd (needed only for computing
the statistical uncertainties, however, the systematic ones are almost always
higher, so "autograd" is not really needed).
RESOURCES: run time: 10-30 minutes; memory: a few gigabytes;
disk space: ~100 Mb to store the results for all stars in all clusters.

    run_orbits.py:
Convert the sky coordinates, distances, mean PM and line-of-sight velocities
of all clusters produced by runfit.py to Galactocentric cartesian coordinates,
sampling from uncertainty covariance matrix of all parameters.
Produces the file "posvel.txt" which contains bootstrapped samples (by default
100 for each cluster) of positions and velocities.
After this first step, compute the galactic orbit for each of these samples,
obtain peri/apocenter distances, orbital energy and actions, and store the
median and 68% confidence intervals on these quantities in a file
"result_orbits.txt".
This second step uses the best-fit potential from McMillan 2017, and employs
the Agama library ( https://github.com/GalacticDynamics-Oxford/Agama ) for
computing the orbits and actions.
For many clusters, these confidence intervals reported in "result_orbits.txt"
are small enough to realistically represent the uncertainties;
however, often the distribution of these parameters is significantly
correlated, elongated and does not resemble an ellipse at all,
hence these results may only serve as a rough guide.
DEPENDENCIES: numpy, astropy; optionall (for the 2nd step) agama.
RESOURCES: run time: ~30 CPU minutes (parallelized - wall-clock time is lower).

"""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]
__credit__ = ["Eugene Vasiliev"]


###############################################################################
# IMPORTS

# GENERAL

import warnings
import argparse
from typing import Optional


# PROJECT-SPECIFIC

from .query_gaia_archive import main as run_query
from .run_fit import main as run_gc_fit
from .run_orbits import main as run_gc_orbits
from .fit_pm_scale import main as fit_pm_scale
from .format_results import main as format_results_function


###############################################################################
# CODE
###############################################################################


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Make parser.

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
        description="get_globular_clusters",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        # default="../../data",
        help="The data directory",
    )
    # parser.add_argument(
    #     "--data_dir",
    #     type=str,
    #     default="data",
    #     help="The input data directory",
    # )

    return parser


# /def


# ------------------------------------------------------------------------


def main(
    args: Optional[list] = None, opts: Optional[argparse.Namespace] = None
):
    """Script Function.

    This function runs the complete set of scripts in this module.
    See README for a description of each component script.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : Namespace, optional
        pre-constructed results of parsed args

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        parser = make_parser()
        opts = parser.parse_args(args)

    # 1) Query Gaia Archive
    run_query(opts=[])

    # 2) Run fit
    run_gc_fit(opts=[])

    # 3) Run orbit
    run_gc_orbits(opts=[])

    # 4) Determine proper motion scale
    fit_pm_scale(opts=[])  # TODO figures directory

    # 5) Reformat results, saving to /data
    format_results_function(opts=opts)

    return

# /def

# ------------------------------------------------------------------------


# if __name__ == "__main__":

#     main(args=None, opts=None)

# # /if


###############################################################################
# END
