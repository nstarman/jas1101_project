# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""GC Orbit Solution Script.

Convert the sky coordinates, distances, mean PM and line-of-sight velocities
of all clusters produced by runfit.py to Galactocentric cartesian coordinates,
sampling from uncertainty covariance matrix of all parameters.
Produces the file "posvel.txt" which contains bootstrapped samples (by default
100 for each cluster) of positions and velocities.
After this first step, compute the galactic orbit for each of these samples,
obtain peri/apocenter distances, orbital energy and actions, and store the
median and 68% confidence intervals on these quantities in a file
"result_orbits.txt".
This second step uses the best-fit potential from McMillan 2017, and employs
the Agama library ( https://github.com/GalacticDynamics-Oxford/Agama ) for
computing the orbits and actions.
For many clusters, these confidence intervals reported in "result_orbits.txt"
are small enough to realistically represent the uncertainties;
however, often the distribution of these parameters is significantly
correlated, elongated and does not resemble an ellipse at all,
hence these results may only serve as a rough guide.
DEPENDENCIES: numpy, astropy; optionall (for the 2nd step) agama.
RESOURCES: run time: ~30 CPU minutes (parallelized - wall-clock time is lower).


"""

__author__ = "Eugene Vasiliev"
__maintainer__ = "Nathaniel Starkman"


###############################################################################
# IMPORTS

# GENERAL
import argparse
import warnings
from typing import Optional

import numpy as np

from astropy import units as u
from astropy import coordinates as coord

# CUSTOM
import agama


###############################################################################
# PARAMETERS


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
        description="Run Orbits",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else 'error'
    )

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
        if not None, used ONLY if args is None

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn('Not using `opts` because `args` are given')
        parser = make_parser()
        opts = parser.parse_args(args)
    # STEP 1: create Monte Carlo realizations of position and velocity of each cluster,
    # sampling from their measured uncertainties.

    # this file should have been produced by run_fit.py
    tab = np.loadtxt("result.txt", dtype=str)
    names = tab[:, 0]  # 0th column is the cluster name (string)
    tab = tab[:, 1:].astype(float)  # remaining columns are numbers
    ra0 = tab[:, 0]  # coordinates of cluster centers [deg]
    dec0 = tab[:, 1]
    dist0 = tab[:, 2]  # distance [kpc]
    vlos0 = tab[:, 3]  # line-of-sight velocity [km/s]
    vlose = tab[:, 4]  # its error estimate
    pmra0 = tab[:, 7]  # mean proper motion [mas/yr]
    pmdec0 = tab[:, 8]
    pmrae = tab[:, 9]  # its uncertainty
    pmdece = tab[:, 10]
    pmcorr = tab[
        :, 11
    ]  # correlation coefficient for errors in two PM components
    vlose = np.maximum(
        vlose, 2.0
    )  # assumed error of at least 2 km/s on line-of-sight velocity
    diste = dist0 * 0.46 * 0.1  # assumed error of 0.1 mag in distance modulus

    # create bootstrap samples
    np.random.seed(42)  # ensure repeatability of random samples
    nboot = 100  # number of bootstrap samples for each cluster
    nclust = len(tab)
    ra = np.repeat(ra0, nboot)
    dec = np.repeat(dec0, nboot)
    pmra = np.repeat(pmra0, nboot)
    pmdec = np.repeat(pmdec0, nboot)
    for i in range(nclust):
        # draw PM realizations from a correlated 2d gaussian for each cluster
        A = np.random.normal(size=nboot)
        B = (
            np.random.normal(size=nboot) * (1 - pmcorr[i] ** 2) ** 0.5
            + A * pmcorr[i]
        )
        pmra[i * nboot : (i + 1) * nboot] += pmrae[i] * A
        pmdec[i * nboot : (i + 1) * nboot] += pmdece[i] * B
    vlos = np.repeat(vlos0, nboot) + np.hstack(
        [np.random.normal(scale=e, size=nboot) for e in vlose]
    )
    dist = np.repeat(dist0, nboot) + np.hstack(
        [np.random.normal(scale=e, size=nboot) for e in diste]
    )

    # convert coordinates from heliocentric (ra,dec,dist,PM,vlos) to Galactocentric (kpc and km/s)
    u.kms = u.km / u.s
    c_sky = coord.ICRS(
        ra=ra * u.degree,
        dec=dec * u.degree,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        distance=dist * u.kpc,
        radial_velocity=vlos * u.kms,
    )
    c_gal = c_sky.transform_to(
        coord.Galactocentric(
            galcen_distance=8.2 * u.kpc,
            galcen_v_sun=coord.CartesianDifferential(
                [10.0, 248.0, 7.0] * u.kms
            ),
        )
    )
    pos = np.column_stack((c_gal.x / u.kpc, c_gal.y / u.kpc, c_gal.z / u.kpc))
    vel = np.column_stack(
        (c_gal.v_x / u.kms, c_gal.v_y / u.kms, c_gal.v_z / u.kms)
    )
    # add uncertainties from the solar position and velocity
    pos[:, 0] += np.random.normal(
        scale=0.1, size=nboot * nclust
    )  # uncertainty in solar distance from Galactic center
    vel[:, 0] += np.random.normal(
        scale=1.0, size=nboot * nclust
    )  # uncertainty in solar velocity
    vel[:, 1] += np.random.normal(scale=3.0, size=nboot * nclust)
    vel[:, 2] += np.random.normal(scale=1.0, size=nboot * nclust)
    pos[
        :, 0
    ] *= (
        -1
    )  # revert back to normal orientation of coordinate system (solar position at x=+8.2)
    vel[:, 0] *= -1  # same for velocity
    posvel = np.column_stack((pos, vel)).value
    np.savetxt("posvel.txt", posvel, fmt="%.6g")

    # STEP 2: compute the orbits, min/max galactocentric radii, and actions, for all Monte Carlo samples

    print(
        agama.setUnits(length=1, velocity=1, mass=1)
    )  # units: kpc, km/s, Msun; time unit ~ 1 Gyr
    potential = agama.Potential(
        "resources/McMillan17.ini"
    )  # MW potential from McMillan(2017)

    # compute orbits for each realization of initial conditions,
    # integrated for 100 dynamical times or 20 Gyr (whichever is lower)
    print(
        "Computing orbits for %d realizations of cluster initial conditions"
        % len(posvel)
    )
    inttime = np.minimum(20.0, potential.Tcirc(posvel) * 100)
    orbits = agama.orbit(
        ic=posvel, potential=potential, time=inttime, trajsize=1000
    )[:, 1]
    rmin = np.zeros(len(orbits))
    rmax = np.zeros(len(orbits))
    for i, o in enumerate(orbits):
        r = np.sum(o[:, 0:3] ** 2, axis=1) ** 0.5
        rmin[i] = np.min(r) if len(r) > 0 else np.nan
        rmax[i] = np.max(r) if len(r) > 0 else np.nan
    # replace nboot samples rmin/rmax with their median and 68% confidence intervals for each cluster
    rmin = np.nanpercentile(rmin.reshape(nclust, nboot), [16, 50, 84], axis=1)
    rmax = np.nanpercentile(rmax.reshape(nclust, nboot), [16, 50, 84], axis=1)

    # compute actions for the same initial conditions
    actfinder = agama.ActionFinder(potential)
    actions = actfinder(posvel)
    # again compute the median and 68% confidence intervals for each cluster
    actions = np.nanpercentile(
        actions.reshape(nclust, nboot, 3), [16, 50, 84], axis=1
    )

    # compute the same confidence intervals for the total energy
    energy = potential.potential(posvel[:, 0:3]) + 0.5 * np.sum(
        posvel[:, 3:6] ** 2, axis=1
    )
    energy = np.percentile(energy.reshape(nclust, nboot), [16, 50, 84], axis=1)

    # write the orbit parameters, actions and energy - one line per cluster, with the median and uncertainties
    fileout = open("result_orbits.txt", "w")
    fileout.write(
        "# Name         \t     pericenter[kpc]   \t     apocenter[kpc]    \t"
        + "       Jr[kpc*km/s]    \t       Jz[kpc*km/s]    \t      Jphi[kpc*km/s]   \t    Energy[km^2/s^2]   \n"
    )
    for i in range(nclust):
        fileout.write(
            ("%-15s" + "\t%7.2f" * 6 + "\t%7.0f" * 12 + "\n")
            % (
                names[i],
                rmin[0, i],
                rmin[1, i],
                rmin[2, i],
                rmax[0, i],
                rmax[1, i],
                rmax[2, i],
                actions[0, i, 0],
                actions[1, i, 0],
                actions[2, i, 0],
                actions[0, i, 1],
                actions[1, i, 1],
                actions[2, i, 1],
                actions[0, i, 2],
                actions[1, i, 2],
                actions[2, i, 2],
                energy[0, i],
                energy[1, i],
                energy[2, i],
            )
        )
    fileout.close()


# --------------------------------------------------------------------------

if __name__ == "__main__":
    main(args=None, opts=None)


###############################################################################
# END
