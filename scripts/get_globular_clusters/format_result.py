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

__author__ = "Nathaniel Starkman"


###############################################################################
# IMPORTS

# GENERAL
import astropy.units as u
from astropy.table import QTable


###############################################################################
# Command Line
###############################################################################

if __name__ == "__main__":

    df = QTable.read("result.txt", format="ascii.commented_header")
    df.add_index("Name")

    names = (
        "ra",
        "dec",
        "dist",
        "vlos",
        "vloserr",
        "sigma",
        "rmax",
        "pmra",
        "pmdec",
        "pmra_e",
        "pmdec_e",
        "rscale",
        "pmdisp",
    )
    units = (
        u.deg,
        u.deg,
        u.kpc,
        u.km / u.s,
        u.km / u.s,
        u.km / u.s,
        u.arcmin,
        u.mas / u.yr,
        u.mas / u.yr,
        u.mas / u.yr,
        u.mas / u.yr,
        u.arcmin,
        u.mas / u.yr,
    )

    for name, unit in zip(names, units):
        df[name].unit = unit

    df.write("result.ecsv")

# /def


###############################################################################
# END
