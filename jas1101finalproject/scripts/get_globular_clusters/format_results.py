# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : Format Result
# AUTHOR  : Nathaniel Starkman
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""Format result.txt into an astropy ECSV table."""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]


###############################################################################
# IMPORTS

# GENERAL
import os
import argparse
import warnings
import pathlib
from typing import Optional

import tqdm

import numpy as np

# astropy
import astropy.units as u
from astropy.table import QTable


###############################################################################
# PARAMETERS

DATA = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


###############################################################################
# CODE
###############################################################################


def format_summary_table(file):
    """Reformat summary table to be in astropy format.

    Parameters
    ----------
    file: str
        file to read with QTable

    Returns
    -------
    df : QTable

    """
    df = QTable.read(file, format="ascii.commented_header")
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
        df[name].unit = unit

    return df


# /def


# ------------------------------------------------------------------------

def format_globular_cluster_table(file):
    """Reformat GC table to be in astropy format.

    Needs to be applied to each GC separately.

    Parameters
    ----------
    file: str
        file to read with QTable

    Returns
    -------
    df : QTable

    """
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

    # Add calculated columns
    # TODO are these approximations?
    df["r"] = np.hypot(df["x"], df["y"])
    df["pm"] = np.hypot(df["pmx"], df["pmy"])

    return df

# /def


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
        description="format_results",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The data output directory",
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

    # get directories from options
    output_dir: str = opts.output_dir
    output_dir = output_dir if output_dir.endswith("/") else output_dir + "/"

    if not os.path.isdir(output_dir + 'gcs'):
        os.mkdir(output_dir + 'gcs')

    # summary table
    df = format_summary_table(DATA + "summary.txt")
    df.write(output_dir + "summary.ecsv", format="ascii.ecsv", overwrite=True)

    # globular clusters
    files = os.listdir(DATA + 'gcts')
    files = [f for f in files if f.endswith(".txt")]

    for file in tqdm.tqdm(files):

        df = format_globular_cluster_table(DATA + 'gcts/' + file)
        df.write(
            output_dir + "gcs/{}.ecsv".format(file[:-4]), format="ascii.ecsv",
            overwrite=True
        )

    # /for

    return

# /def

# ------------------------------------------------------------------------


# if __name__ == "__main__":

#     main(args=None, opts=None)

# # /if


###############################################################################
# END
