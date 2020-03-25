# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""Best-Fit Solution Script.

The main script performing the membership determination and measuring the mean
PM for each cluster, as described in the Appendix of arXiv:1807.09775.
It reads the data previously stored in "data/*.npz" by query_gaia_archive.py,
performs the fit, estimates the uncertainties on the mean PM (optionally taking
into account systematic errors), writes the summary for each cluster to the file
"result.txt" (same columns as "input.txt", plus the following ones:
uncertainties in mean PMra, PMdec and the correlation coefficient;
number of members (not an integer, since the inference is probabilistic);
scale radius of a Plummer profile of cluster members from the filtered Gaia
catalogue - not the same as the half-mass radius of all stars in the cluster);
internal PM dispersion in the center inferred during the fit (not a robust
measurement, rather a nuisance parameter).
Additionally, the data for all stars from each cluster are written to a file
"data/[cluster_name].txt":
x,y are orthogonally projected coordinates w.r.t. cluster center (in degrees);
pmx, pmy is the quasi-PM in these orthogonally projected coordinates (mas/yr);
pmx_e, pmy_e, pm_corr are their uncertainties and correlation coefficient;
g_mag is G-band magnitude;  bp_rp is the color;
filter is the flag (0/1) specifying whether the star passed the quality filters
on the initial sample (i.e., astrometric noise and photometry);
prob is the cluster membership probability (only for stars with filter==1).
DEPENDENCIES: numpy, scipy; optionally autograd (needed only for computing
the statistical uncertainties, however, the systematic ones are almost always
higher, so "autograd" is not really needed).
RESOURCES: run time: 10-30 minutes; memory: a few gigabytes;
disk space: ~100 Mb to store the results for all stars in all clusters.

"""

__author__ = "Eugene Vasiliev"
__maintainer__ = "Nathaniel Starkman"


###############################################################################
# IMPORTS

# GENERAL
import os
import argparse
import warnings
import pathlib
from typing import Optional
import numpy as np
import scipy.optimize

import tqdm

try:
    # autograd module allows to compute analytically the Hessian of the log-likelihood function
    import autograd
except ImportError:
    # will not be able to compute statistical uncertainties if autograd is not available,
    # however, if use_systematic_error==True, the uncertainties will be computed in a different way
    # which does not use autograd anyway
    autograd = None
    log = np.log
    exp = np.exp
else:
    log = autograd.numpy.log
    exp = autograd.numpy.exp

# PROJECT-SPECIFIC
from .get_mean_pm import get_mean_pm, covfnc1


###############################################################################
# PARAMETERS

# whether to account for spatially correlated systematic errors when
# estimating the uncertainty on mean PM
use_systematic_error = True


DATA = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


###############################################################################
# CODE
###############################################################################


# --------------------------------------------------------------------------


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
        description="Run Fit Parser",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )

    return parser


# /def


# -------------------------------------------------------------------


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
            warnings.warn("Not using `opts` because `args` are given")
        parser = make_parser()
        opts = parser.parse_args(args)

    if not os.path.isdir(DATA + "gcts/"):
        os.mkdir(DATA + "gcts/")

    # ----------------------------------------------------------------------

    foutname = DATA + "summary.txt"

    if not os.path.isfile(foutname):

        filein = open(DATA + "input.txt", "r")
        fileout = open(foutname, "w")

        for linein in tqdm.tqdm(filein):
            # parse the input file (one line per cluster)
            line = linein.strip().split()
            name = line[0]
            if linein[0] == "#":
                fileout.write(linein)
                # print("%s skipped" % name)
                continue
            ra0 = float(line[1])  # degrees
            dec0 = float(line[2])  # degrees
            dist = float(line[3])  # kpc
            vdisp = float(line[6])  # km/s
            rmax = float(line[7])  # arcmin
            pmra0 = float(line[8])  # initial guess for the cluster PMra
            pmdec0 = float(line[9])  # same for PMdec

            # read data file that was previously retrieved from the Gaia archive
            data = np.load(DATA + "/gczs/" + name + ".npz")
            ra = data["ra"].astype(float)
            dec = data["dec"].astype(float)
            pmra = data["pmra"].astype(float)
            pmdec = data["pmdec"].astype(float)
            pmrae = data["pmra_error"].astype(float)
            pmdece = data["pmdec_error"].astype(float)
            pmcorr = data["pmra_pmdec_corr"].astype(float)
            bprp = data["bp_rp"]
            gmag = data["phot_g_mean_mag"]
            ruwe = data["ruwe"]
            aen = data["astrometric_excess_noise"]
            phen = data["phot_bp_rp_excess_factor"]

            # coordinate transformation from sky to tangent plane (orthogonal projection)
            sin = np.sin
            cos = np.cos
            d2r = np.pi / 180  # degrees to radians
            x = (
                cos(dec * d2r) * sin((ra - ra0) * d2r)
            ) / d2r  # x,y are in degrees
            y = (
                sin(dec * d2r) * cos(dec0 * d2r)
                - cos(dec * d2r) * sin(dec0 * d2r) * cos((ra - ra0) * d2r)
            ) / d2r
            # transformation of PM and its uncertainty covariance matrix
            Jxa = cos((ra - ra0) * d2r)
            Jxd = -sin(dec * d2r) * sin((ra - ra0) * d2r)
            Jya = sin(dec0 * d2r) * sin((ra - ra0) * d2r)
            Jyd = cos(dec * d2r) * cos(dec0 * d2r) + sin(dec * d2r) * sin(
                dec0 * d2r
            ) * cos((ra - ra0) * d2r)
            mx = pmra * Jxa + pmdec * Jxd
            my = pmra * Jya + pmdec * Jyd
            Cxx = (
                (Jxa * pmrae) ** 2
                + (Jxd * pmdece) ** 2
                + 2 * Jxa * Jxd * pmcorr * pmrae * pmdece
            )
            Cyy = (
                (Jya * pmrae) ** 2
                + (Jyd * pmdece) ** 2
                + 2 * Jya * Jyd * pmcorr * pmrae * pmdece
            )
            Cxy = (
                Jxa * Jya * pmrae ** 2
                + Jxd * Jyd * pmdece ** 2
                + (Jya * Jxd + Jxa * Jyd) * pmcorr * pmrae * pmdece
            )
            mxe = Cxx ** 0.5
            mye = Cyy ** 0.5
            mcorr = Cxy / (mxe * mye)
            rdist = (
                x ** 2 + y ** 2
            ) ** 0.5 * 60.0  # distance from cluster center in arcmin

            # apply various quality filters
            filt = rdist < rmax  # distance filter
            filt *= (
                pmra ** 2 + pmdec ** 2 < 30 ** 2
            )  # eliminate spurious very large PM
            filt *= (ruwe < 1.2) * (
                aen < 1.0
            )  # eliminate unreliable PM from astrometric excess noise and RUWE
            if not name in [
                "AM_1",
                "Crater",
                "Pal_3",
                "Pal_4",
                "Terzan_6_HP_5",
            ]:
                filt *= (
                    phen < 1.3 + 0.06 * bprp ** 2
                )  # eliminate mostly faint sources in crowded regions

            # filtered PM and their uncertainty covariance matrices for each star
            # (for simplicity, refer to them as "x,y", not "ra,dec")
            errmul = 1.1  # increase the statistical uncertainty by this factor
            star_pmra = pmra[filt]
            star_pmdec = pmdec[filt]
            star_covrr = (pmrae[filt] * errmul) ** 2
            star_covdd = (pmdece[filt] * errmul) ** 2
            star_covrd = pmcorr[filt] * (star_covrr * star_covdd) ** 0.5
            star_rdist = rdist[filt]

            # center and variance of PM distribution of field stars (initially assign from all stars)
            field_pmra0 = np.mean(star_pmra)
            field_pmdec0 = np.mean(star_pmdec)
            field_covrr0 = np.var(star_pmra)
            field_covdd0 = np.var(star_pmdec)
            # same for cluster stars (take the initial guess for the center from the input file,
            # and the initial dispersion - or more correctly, standard deviation - from line-of-sight vel dispersion
            clust_pmra0 = pmra0
            clust_pmdec0 = pmdec0
            clust_disp0 = vdisp / dist / 4.74

            # initial guess for the parameters of the Gaussian mixture likelihood function
            params = np.array(
                [
                    clust_pmra0,
                    clust_pmdec0,  # [0-1] = center of 0th component (cluster) in the PM space
                    field_pmra0,
                    field_pmdec0,  # [2-3] = center of 1st component (field)
                    field_covrr0,
                    field_covdd0,
                    0,  # [4-6] = covariance matrix of 1st component (xx, yy, xy)
                    0.5,  # [7]   = weight of 0st component
                    0.5,  # [8]   = Plummer scale radius of the cluster, normalized to rmax
                    clust_disp0,  # [9]   = PM dispersion of cluster stars
                ]
            )

            def eval_model(params, give_prob=False):
                """
                Construct a mixture model with the given parameters and evaluate the distribution function
                of both cluster and field components for each star.
                This function can be used in two different contexts:
                if give_prob==False,  return -ln(likelihood) - used in minimizer;
                otherwise, return the array of posterior probabilities of cluster membership for each star.
                """
                (
                    clust_pmra,
                    clust_pmdec,
                    field_pmra,
                    field_pmdec,
                    field_covrr,
                    field_covdd,
                    field_covrd,
                    clust_weight,
                    clust_rscale_mult,
                    clust_disp,
                ) = params
                # Plummer scale radius of cluster stars
                clust_rscale = clust_rscale_mult * rmax
                # scale radius for PM dispersion profile - assume a fixed fraction of Rscale
                clust_rsigma = clust_rscale * 0.5
                # number of cluster members in the circle of radius rmax
                clust_count = len(star_rdist) * clust_weight
                # same for field stars
                field_count = len(star_rdist) - clust_count
                # check ranges
                if (
                    clust_count <= 2
                    or field_count <= 0
                    or clust_rscale <= 0
                    or clust_disp <= 0
                    or field_covrr <= 0
                    or field_covdd <= 0
                    or field_covrd ** 2 >= field_covrr * field_covdd
                ):
                    return np.inf

                # spatially dependent squared velocity dispersion of cluster stars (diagonal elements of covar.matrix)
                clust_covar = (
                    clust_disp ** 2
                    / (1 + (star_rdist / clust_rsigma) ** 2) ** 0.5
                )
                # offset between PM of each star and the mean PM of the cluster
                clust_pmra -= star_pmra
                clust_pmdec -= star_pmdec
                # covariance matrices of each star as if it were a cluster member
                clust_covrr = star_covrr + clust_covar
                clust_covdd = star_covdd + clust_covar
                clust_covrd = star_covrd
                # determinants of these covariance matrices for each star
                clust_det = clust_covrr * clust_covdd - clust_covrd ** 2
                # same for each star as if it were a field star
                field_pmra -= star_pmra
                field_pmdec -= star_pmdec
                field_covrr += star_covrr
                field_covdd += star_covdd
                field_covrd += star_covrd
                field_det = field_covrr * field_covdd - field_covrd ** 2
                # amplitude of the surface density of cluster stars (spatially dependent)
                clust_ampl = clust_count
                # value of the distribution function of the 0th component (cluster) for each star,
                # including the spatially-dependent prior multiplying factor
                clust_distr = (
                    clust_count
                    * (1 + (rmax / clust_rscale) ** 2)
                    * (1 + (star_rdist / clust_rscale) ** 2) ** -2
                    * clust_det ** -0.5
                    * exp(
                        -0.5
                        / clust_det
                        * (
                            clust_pmra ** 2 * clust_covdd
                            + clust_pmdec ** 2 * clust_covrr
                            - 2 * clust_pmra * clust_pmdec * clust_covrd
                        )
                    )
                )
                # same for the 1st component (field)
                field_distr = (
                    field_count
                    * field_det ** -0.5
                    * exp(
                        -0.5
                        / field_det
                        * (
                            field_pmra ** 2 * field_covdd
                            + field_pmdec ** 2 * field_covrr
                            - 2 * field_pmra * field_pmdec * field_covrd
                        )
                    )
                )
                if give_prob:
                    # return the posterior membership probability for each star
                    return np.nan_to_num(
                        clust_distr / (clust_distr + field_distr)
                    )
                else:
                    # return the total log-likelihood of the model
                    result = sum(log(clust_distr + field_distr))
                    # prior discourages PM dispersion and scale radius from getting too high
                    prior = -exp(5 * (clust_disp / clust_disp0 - 1)) - exp(
                        5 * (clust_rscale / rmax - 0.75)
                    )
                    return -(
                        result + prior
                    )  # return minus log-likelihood, the function to be minimized

            # minimization of minus log-likelihood - perform several runs of Nelder-Mead algorithm,
            # restarting it each time from the last best-fit position (to prevent stalling in a local minimum)
            llprev = np.inf
            params = scipy.optimize.minimize(
                eval_model, params, method="Nelder-Mead"
            ).x
            llcurr = eval_model(params)
            while llcurr < llprev - 0.1:
                params = scipy.optimize.minimize(
                    eval_model, params, method="Nelder-Mead"
                ).x
                llprev = llcurr
                llcurr = eval_model(params)

            # final values of best-fit parameters
            (
                clust_pmra,
                clust_pmdec,
                field_pmra,
                field_pmdec,
                field_covrr,
                field_covdd,
                field_covrd,
                clust_weight,
                clust_rscale_mult,
                clust_disp,
            ) = params
            clust_rscale = clust_rscale_mult * rmax
            clust_rsigma = clust_rscale * 0.5

            # estimate the uncertainty covariance matrix of all model parameters from the inverse Hessian;
            # which is computed by automatic differentiation (if available)
            covmat = np.eye(len(params)) * 0.0
            if not autograd is None:
                hessian = autograd.hessian(eval_model)(params)
                if not np.any(np.isnan(hessian)):
                    try:
                        covmat = np.linalg.inv(hessian)
                    except:
                        print("Hessian matrix is not positive-definite")
                else:
                    print("Hessian matrix contains invalid elements")
            clust_pmrae = covmat[0, 0] ** 0.5
            clust_pmdece = covmat[1, 1] ** 0.5
            clust_pmcorr = (
                covmat[0, 1] / (covmat[0, 0] * covmat[1, 1] + 1e-200) ** 0.5
            )

            # evaluate the membership probability for each star
            memberprob = np.zeros(
                len(pmra)
            )  # set to zero for stars which did not pass initial filter
            memberprob[filt] = eval_model(params, give_prob=True)
            memberprob[memberprob < 1e-6] = 0.0

            # optional: estimate the uncertainty in mean PM taking into account systematic errors
            if use_systematic_error:
                # select stars likely belonging to the cluster (cannot use probabilistic membership at this stage)
                filtprob = memberprob >= 0.8
                maxused = 10000  # limit the maximum number of stars, as the cost scales as N^3
                if sum(filtprob) > maxused:
                    filtprob *= (
                        gmag < np.sort(gmag[filtprob])[maxused]
                    )  # retain only brighter stars

                # use the routine from the supplementary module to compute the mean PM and its uncertainty
                result = get_mean_pm(
                    ra[filtprob],
                    dec[filtprob],
                    pmra[filtprob],
                    pmdec[filtprob],
                    pmrae[filtprob],
                    pmdece[filtprob],
                    pmcorr[filtprob],
                    sigma=clust_disp,
                    rsigma=clust_rsigma,
                    covfnc=covfnc1,
                )

                # take these values if the uncertainty is larger than inferred without accounting for systematics
                if result[2] + result[3] > clust_pmrae + clust_pmdece:
                    clust_pmra = result[0]
                    clust_pmdec = result[1]
                    clust_pmrae = result[2]
                    clust_pmdece = result[3]
                    clust_pmcorr = result[4]

            # finally, write the summary results for this cluster to the output file and print them to screen
            print(
                "%s:  Total=%d,  PMfilt=%.1f,  PMra=%.3f +- %.3f,  PMdec=%.3f +- %.3f,  corr=%.3f,  rscale=%.2f,  PMdisp=%.3f"
                % (
                    name,
                    sum(filt),
                    sum(memberprob),
                    clust_pmra,
                    clust_pmrae,
                    clust_pmdec,
                    clust_pmdece,
                    clust_pmcorr,
                    clust_rscale,
                    clust_disp,
                )
            )
            line[0] = "%-15s" % name
            line[8] = "%.3f" % clust_pmra
            line[9] = "%.3f" % clust_pmdec
            line += [
                "%.3f" % clust_pmrae,
                "%.3f" % clust_pmdece,
                "%.3f" % clust_pmcorr,
                "%.1f" % (clust_weight * len(star_rdist)),
                "%.2f" % clust_rscale,
                "%.3f" % clust_disp,
            ]
            fileout.write("\t".join(line) + "\n")
            fileout.flush()

            # write the data for all stars into a text file
            np.savetxt(
                DATA + "gcts/" + name + ".txt",
                np.column_stack(
                    (
                        x,
                        y,
                        mx,
                        my,
                        mxe,
                        mye,
                        mcorr,
                        gmag,
                        bprp,
                        filt,
                        memberprob,
                    )
                ),
                fmt="%.7g",
                header="x   y   pmx    pmy    pmx_e  pmy_e  pm_corr g_mag bp_rp filter  memberprob\n"
                + "deg deg mas/yr mas/yr mas/yr mas/yr dimless mag   mag   zeroone float",
            )
        # /for
    
        filein.close()
        fileout.close()

    # /if

    return


# /def


# ------------------------------------------------------------------------


# if __name__ == "__main__":

#     main(args=None, opts=None)

# # /if


###############################################################################
# END
