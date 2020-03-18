# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : Format Result
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""Format result.txt into an astropy ECSV table."""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]


###############################################################################
# IMPORTS

# GENERAL
import warnings
import argparse
from typing import Optional


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
    # This script was written by Eugene
    from .query_gaia_archive import main as run_query
    run_query()

    # 2) Run fit
    # This script, written by Eugene, runs on import
    from .run_fit import main as run_gc_fit
    run_gc_fit()

    # 3) Run orbit
    # This script, written by Eugene, runs on import
    from .run_orbits import main as run_gc_orbits
    run_gc_orbits()

    # 4) Determine proper motion scale
    from .fit_pm_scale import main as fit_pm_scale
    fit_pm_scale()

    # 5) Reformat results, saving to /data
    from .format_results import main as format_results_function
    format_results_function(opts=opts)

    return

# /def

# ------------------------------------------------------------------------


if __name__ == "__main__":

    main(args=None, opts=None)

# /if


###############################################################################
# END
