# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : SCripts
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

###############################################################################
# IMPORTS

# GENERAL
import argparse

from typing import Union


###############################################################################
# PARAMETERS


###############################################################################
# CODE
###############################################################################


# --------------------------------------------------------------------------


###############################################################################
# Command Line
###############################################################################


def main(opts: Union[argparse.ArgumentParser, None]):
    """Script Function.

    Parameters
    ----------
    opts : ArgumentParser or None

    """
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
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.80,
        )
        opts, args = parser.parse_args()

    # 1) Get the globular clusters
    from .get_globular_clusters import main as get_gcs
    get_gcs(opts)

    # 2) Do the proper motion normalization
    from .normalize_globular_clusters import main as normalize_gcs
    normalize_gcs(opts)

    # 3) Run stacking pipeline
    # TODO

    # 4) Run analysis pipeline
    # TODO

# /def


# --------------------------------------------------------------------------

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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="The cluster membership probability threshold",
    )
    opts, args = parser.parse_args()

    main(opts)

# /if


###############################################################################
# END
