# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : Analyze
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

    pass


# /def

# --------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="../../data",
    #     help="The data directory",
    # )
    opts, args = parser.parse_args()

    main(opts)

# /if


###############################################################################
# END
