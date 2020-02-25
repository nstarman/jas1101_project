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
import argparse
from typing import Union


###############################################################################
# CODE
###############################################################################



###############################################################################
# Command Line
###############################################################################


def main(opts: Union[argparse.ArgumentParser, None]):
    """Script Function.

    This function runs the complete set of scripts in this module.
    See README for a description of each component script.

    Parameters
    ----------
    opts : ArgumentParser or None
        must contain `output_dir`, `data_dir`

    """

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

    # 4) Reformat results, saving to /data
    from .format_results import main as format_results_function

    if opts is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output_dir",
            type=str,
            default="../../data",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data",
        )
        opts, args = parser.parse_args()

    format_results_function(opts)

# /def

# ------------------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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

    options, args = parser.parse_args()

    main(options)

# /def


###############################################################################
# END
