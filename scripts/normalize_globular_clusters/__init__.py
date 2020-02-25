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
            help="The data directory",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.80,
        )
        opts, args = parser.parse_args()

    print('OK')

    # 1. MAD clip at 5-sigma

    # 2. Do simultaneous GMM fitting

    # 3. Predict Cluster, returning probability(x | threshold)

    # 4. PM scale size from 2D Gaussian
    #     Right now simplify to 1 number, but in future, keep as shape matrix

    # 5. export table of PM scales

    pass


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
        "--threshold",
        dest="threshold",
        default=0.80,
        help="The cluster membership probability threshold",
    )
    opts, args = parser.parse_args()

    main(opts)

# /if


###############################################################################
# END
