# -*- coding: utf-8 -*-

"""Query Gaia for GCs.

Retrieve the data from the Gaia archive (all sources satisfying the
maximum distance from cluster center and a simple parallax cut). Source
data for each cluster is stored in a separate numpy zip file:
"data/[cluster_name].npz". Additionally, the table for computing the
renormalized unit weight error (an astrometric quality flag) is retrieved
from the Gaia website and stored in "DR2_RUWE_V1/table_u0_2D.txt".

Routine Listings
----------------
main

Notes
-----
Dependencies:

    - numpy,
    - scipy,
    - astropy,
    - astroquery (astropy-affiliated package).


Resources:
    run time is a few minutes (depending on internet speed);
    requires a few tens of megabytes to store the downloaded data.

"""

__author__ = "Eugene Vasiliev"
__maintainer__ = "Nathaniel Starkman"


###############################################################################
# IMPORTS

# GENERAL
import os
import shutil
import pathlib
import subprocess
import argparse
from typing import Optional

import numpy as np
import scipy.interpolate

import tqdm

import warnings
import astropy

from astroquery.utils.tap.core import Tap  # basic Gaia query interface


###############################################################################
# PARAMETERS

# silence some irrelevant warnings
warnings.filterwarnings("ignore", category=astropy.io.votable.VOWarning)

DATA = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


###############################################################################
# CODE
###############################################################################


def retrieve(gaia, rint, rint0, ra, dec, radius, filename, parallax_limit):
    """Query Helper.

    Query the Gaia archive for all sources within a certain radius from the given point,
    which have parallax below the given limit (within 3 sigma),
    and save the result as a numpy zip archive.

    """
    job = gaia.launch_job(
        "select top 999999 "
        + "ra, dec, pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr, "
        + "phot_g_mean_mag, bp_rp, "
        + "sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al-5)) as uwe, "
        + "astrometric_excess_noise, phot_bp_rp_excess_factor "
        + "FROM gaiadr2.gaia_source WHERE "
        + "parallax is not null and "
        + "parallax-"
        + str(parallax_limit)
        + "<3*parallax_error and "
        + "contains(point('icrs',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec), "
        + "circle('icrs',"
        + str(ra)
        + ","
        + str(dec)
        + ","
        + str(radius)
        + "))=1"
    )
    table = job.get_results()
    # print("%s => %d" % (filename, len(table)))
    # compute the renormalized unit weight error from the interpolation tables
    g_mag = np.array(table["phot_g_mean_mag"])
    bp_rp = np.array(table["bp_rp"])
    rfac = rint(g_mag, bp_rp, grid=False)
    rfac[np.isnan(bp_rp)] = rint0(g_mag[np.isnan(bp_rp)])
    # save the data as a numpy zip archive
    np.savez_compressed(
        filename,
        ra=np.array(table["ra"]).astype(np.float32),
        dec=np.array(table["dec"]).astype(np.float32),
        pmra=np.array(table["pmra"]).astype(np.float32),
        pmdec=np.array(table["pmdec"]).astype(np.float32),
        pmra_error=np.array(table["pmra_error"]).astype(np.float32),
        pmdec_error=np.array(table["pmdec_error"]).astype(np.float32),
        pmra_pmdec_corr=np.array(table["pmra_pmdec_corr"]).astype(np.float32),
        phot_g_mean_mag=g_mag.astype(np.float32),
        bp_rp=bp_rp.astype(np.float32),
        ruwe=(np.array(table["uwe"]) / rfac).astype(np.float32),
        astrometric_excess_noise=np.array(
            table["astrometric_excess_noise"]
        ).astype(np.float32),
        phot_bp_rp_excess_factor=np.array(
            table["phot_bp_rp_excess_factor"]
        ).astype(np.float32),
    )

    return


# /def


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Expose parser for ``main``.

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
        description="Query Gaua Archive Parser",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(
    args: Optional[list] = None, opts: Optional[argparse.Namespace] = None
):
    """Script Function.

    Retrieve the data from the Gaia archive (all sources satisfying the
    maximum distance from cluster center and a simple parallax cut). Source
    data for each cluster is stored in a separate numpy zip file:
    "data/[cluster_name].npz". Additionally, the table for computing the
    renormalized unit weight error (an astrometric quality flag) is retrieved
    from the Gaia website and stored in "DR2_RUWE_V1/table_u0_2D.txt".
    DEPENDENCIES: numpy, scipy, astropy, astroquery (astropy-affiliated
    package). RESOURCES: run time: a few minutes (depending on internet
    speed); disk space: a few tens of megabytes to store the downloaded data.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : Namespace, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        parser = make_parser()
        opts = parser.parse_args(args)

    if not os.path.isdir(DATA):
        os.mkdir(DATA)

    # download the file with renormalized unit weight error correction tables from the Gaia website
    if not os.path.isdir(DATA + "DR2_RUWE_V1"):
        os.mkdir(DATA + "DR2_RUWE_V1")
    ruwefile = DATA + "DR2_RUWE_V1/table_u0_2D.txt"

    if not os.path.isfile(ruwefile):
        subprocess.call(
            (  # no , b/c combine into 1 string
                "curl https://www.cosmos.esa.int/documents/29201/1769576/"
                "DR2_RUWE_V1.zip/d90f37a8-37c9-81ba-bf59-dd29d9b1438f"
                " > temp.zip"
            ),
            shell=True,
        )
        subprocess.call(
            "unzip temp.zip DR2_RUWE_V1/table_u0_2D.txt", shell=True
        )
        os.remove("temp.zip")
        os.rename("DR2_RUWE_V1/table_u0_2D.txt", ruwefile)
        shutil.rmtree("DR2_RUWE_V1")

    if not os.path.isdir(DATA + "gczs/"):
        os.mkdir(DATA + "gczs/")

    # construct interpolator for renorm unit weight error correction table
    rtab = np.loadtxt(ruwefile, delimiter=",", skiprows=1)
    # correction factor as a function of g_mag and bp_rp
    rint = scipy.interpolate.RectBivariateSpline(
        x=rtab[:, 0], y=np.linspace(-1.0, 10.0, 111), z=rtab[:, 2:], kx=1, ky=1
    )
    # correction factor in case of no bp/rp, as a function of g_mag only
    rint0 = scipy.interpolate.UnivariateSpline(
        x=rtab[:, 0], y=rtab[:, 1], k=1, s=0
    )

    gaia = Tap(url="https://gea.esac.esa.int/tap-server/tap")

    # read the list of clusters and query the Gaia archive for each of them
    lst = np.genfromtxt(DATA + "input.txt", dtype=str)

    for l in tqdm.tqdm(lst):
        filename = DATA + "gczs/" + l[0] + '.npz'
        if not os.path.isfile(filename):
            retrieve(
                gaia=gaia,
                rint=rint,
                rint0=rint0,
                ra=float(l[1]),
                dec=float(l[2]),
                radius=float(l[7]) / 60,  # convert from arcmin to degrees
                filename=filename,
                parallax_limit=1.0 / float(l[3]),
            )

    return


# /def


# --------------------------------------------------------------------------

# if __name__ == "__main__":

#     main(args=None, opts=None)

# /if

###############################################################################
# END
