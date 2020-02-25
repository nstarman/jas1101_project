# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""initialization file for __________.

description

Routine Listings
----------------
module

"""

__author__ = ""
# __copyright__ = "Copyright 2018, "
# __credits__ = [""]
# __license__ = "MIT"
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

# __all__ = [
#     ""
# ]

##############################################################################
# IMPORTS

# GENERAL

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

from tqdm import tqdm
import os
from collections import OrderedDict
from astroquery.utils import TableList


##############################################################################
# CODE
##############################################################################


def load_summary_table(drct):
    """Read Summary Table.

    Parameters
    ----------
    drct : str
        "data/"

    Returns
    -------
    df : QTable

    TODO
    ----
    not need drct by using environment variable

    """
    df = QTable(QTable.read(drct + "result.ecsv", format="ascii.ecsv"))
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

# --------------------------------------------------------------------------


def load_globular_cluster(file, clip_at=(1, 15 * u.mas / u.yr)):
    """Read Summary Table.

    scales by rscale.

    Parameters
    ----------
    file : str
        "scripts/get_globular_clusters/result.ecsv"
    gc_name : str
        globular cluster name
    clip_at : tuple or False, optional
        the radius and proper motion clipping factors

    Returns
    -------
    df : QTable

    TODO
    ----
    not need file by using environment variable

    """
    # GC table
    df = QTable.read(file, format="ascii.ecsv")

    # Add calculated columns
    # TODO are these approximations?
    df["r"] = np.hypot(df["x"], df["y"])
    df["pm"] = np.hypot(df["pmx"], df["pmy"])

    if clip_at:
        sel = (
            (df["r"] < clip_at[0])
            & (0 * u.mas / u.yr < df["pm"])  # true by default
            & (df["pm"] < clip_at[1])
        )

        dfsub = df[sel]

    else:
        dfsub = None

    return df, dfsub


# /def


# --------------------------------------------------------------------------


def load_all_globular_clusters(drct, ffmt='.ecsv'):

    drct = drct if drct.endswith('/') else drct + '/'

    files = os.listdir(drct)
    datafiles = [f for f in files if f.endswith(ffmt)]
    print(files)

    gcs = OrderedDict()
    for file in tqdm(datafiles):
        name = file[:-len(ffmt)]
        gcs[name], _ = load_globular_cluster(drct + file, clip_at=False)

    return TableList(gcs)




##############################################################################
# END
