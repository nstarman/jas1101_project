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
import pathlib

# astropy
import astropy.units as u
from astropy.table import QTable


###############################################################################
# CODE
###############################################################################


def format_summary_table(file):

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
    }

    for name, unit in units.items():
        df[name].unit = unit

    return df


# /def

# ------------------------------------------------------------------------


def format_globular_cluster_table(file):

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


###############################################################################
# Command Line
###############################################################################


def main(opts):
    """Script Function.

    Parameters
    ----------
    opts : ArgumentParser
        must contain `output_dir`, `data_dir`

    """

    # get directories from options
    output_dir = opts.output_dir
    data_dir = opts.data_dir
    result_dir = str(pathlib.Path(data_dir).parent)

    # reformat
    output_dir = output_dir if output_dir.endswith("/") else output_dir + "/"
    data_dir = data_dir if data_dir.endswith("/") else data_dir + "/"
    result_dir = result_dir if result_dir.endswith("/") else result_dir + "/"

    # summary table
    df = format_summary_table(result_dir + "result.txt")
    df.write(output_dir + "result.ecsv", format="ascii.ecsv")

    # globular clusters
    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue

        df = format_globular_cluster_table(data_dir + file)
        # df.write("../../data/gcs/{}.ecsv", format(file[:-4]))
        df.write(
            output_dir + "gcs/{}.ecsv".format(file[:-4]), format="ascii.ecsv",
        )

    # /for


# /def

# ------------------------------------------------------------------------


if __name__ == "__main__":

    import argparse

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
