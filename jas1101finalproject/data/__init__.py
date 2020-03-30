# -*- coding: utf-8 -*-
# see LICENSE.rst

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""Initialization file.

Routine Listings
----------------

"""

__author__ = ""
# __copyright__ = "Copyright 2018, "
# __credits__ = [""]
# __license__ = ""
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

__all__ = [
    "DATA_PATH",
    "load_summary_table",
    "load_globular_cluster",
    "load_all_globular_clusters",
]


##############################################################################
# IMPORTS

# GENERAL

import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from astropy.coordinates import SkyCoord
from astropy.table import QTable

from astroquery.utils import TableList


# CUSTOM

# PROJECT-SPECIFIC


##############################################################################
# PARAMETERS

DATA_PATH: str = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = DATA_PATH + '/' if not DATA_PATH.endswith('/') else DATA_PATH


##############################################################################
# SETUP
##############################################################################

def _run_get_globular_clusters():

    # run script if data not present
    if not os.path.isfile(DATA_PATH + 'summary.ecsv'):

        from argparse import Namespace

        opts = Namespace()
        opts.output_dir = DATA_PATH

        from ..scripts.get_globular_clusters import main

        main(opts=opts)

        from ..scripts.normalize_globular_clusters import main

        main(opts=opts)

# /def


##############################################################################
# CODE
##############################################################################

def load_summary_table():
    """Read Summary Table.

    Returns
    -------
    df : QTable

    """
    _run_get_globular_clusters()

    df = QTable(QTable.read(DATA_PATH + "summary.ecsv", format="ascii.ecsv"))
    df.add_index("Name")
    df.columns

    # convert to distance units, from angular units
    # this uses the distance, which is assumed to be errorless # TODO

    # storing information
    df["pm"] = np.hypot(df["pmra"], df["pmdec"])

    # making skycoord for ease of use
    df_sc = SkyCoord(
        ra=df["ra"],
        dec=df["dec"],
        distance=df["dist"],
        pm_ra_cosdec=df["pmra"],
        pm_dec=df["pmdec"],
        radial_velocity=df["vlos"],
    )
    df_sc.representation_type = "cartesian"

    # store SkyCoord
    df["sc"] = df_sc

    return df


# /def


# ------------------------------------------------------------------------


def load_globular_cluster(name):
    """Read Summary Table.

    scales by rscale.

    Parameters
    ----------
    name : str
        globular cluster name

    Returns
    -------
    df : QTable

    """
    _run_get_globular_clusters()

    if not name.endswith('.ecsv'):
        name += '.ecsv'

    # GC table
    df = QTable.read(DATA_PATH + 'gcs/' + name, format="ascii.ecsv")

    return df


# /def


# ------------------------------------------------------------------------


def load_all_globular_clusters():
    """Load all Globular Clusters in directory.

    Returns
    -------
    TableList

    """

    files = os.listdir(DATA_PATH + 'gcs')
    datafiles = [f for f in files if f.endswith('.ecsv')]

    gcs = OrderedDict()
    for file in tqdm(datafiles):
        name = file[: -len('.ecsv')]
        gcs[name] = load_globular_cluster(file)

    return TableList(gcs)


# /def


##############################################################################
# END
