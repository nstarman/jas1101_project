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

__all__ = ["GlobularCluster"]


##############################################################################
# IMPORTS

# GENERAL
# import numpy as np
import astropy.units as u

# CUSTOM

# PROJECT-SPECIFIC
from .load import load_summary_table, load_globular_cluster
from .utils import profile_binning, convert_angle, convert_pm_angular
from .GMM import GMM_bins


##############################################################################
# PARAMETERS

dmls = u.dimensionless_unscaled


##############################################################################
# CODE
##############################################################################


class GlobularCluster(object):
    """docstring for GlobularCluster"""

    def __init__(
        self, name, property_table, star_table, clip_at=(1, 15 * u.mas / u.yr)
    ):
        """GlobularCluster

        Parameters
        ----------
        name: str
        property_table : QTable
            overall properties
        star_table : QTable
            the stars data table
        clip_at: list
            where to clip the full star table. It has a lot of background.
            TODO isn't it better to weight the data instead of trimming?

        """
        super(GlobularCluster, self).__init__()
        self.name = name
        self._bins = None

        # ----------------------------

        if len(property_table) > 1:  # then full summary table of all GCs

            property_table = property_table.loc[name]  # QTable to not be Row

        # Property table
        self.summary = property_table
        # storing specific-properties
        self.nstar = property_table["nstar"]
        self.rc_ang = property_table["rscale"]
        self.d = property_table["dist"]
        self.rc = convert_angle(property_table["rscale"], self.d)

        # ----------------------------

        self.df_full = star_table
        if clip_at:
            sel = (
                (star_table["r"] < clip_at[0] * self.rc_ang)
                & (0 * u.mas / u.yr < star_table["pm"])  # true by default
                & (star_table["pm"] < clip_at[1])
            )

            self.df = star_table[sel]

        else:
            self.df = star_table
        # storing sub-properties

        # angular
        # adjusted for distance is the same.
        self.x = (self.df["x"] / self.rc_ang).decompose().to_value(dmls)
        self.y = (self.df["y"] / self.rc_ang).decompose().to_value(dmls)
        self.r = (self.df["r"] / self.rc_ang).decompose().to_value(dmls)

        self.pm = self.df["pm"]
        self.vsky = convert_pm_angular(self.df["pm"][:], self.d)

        return None

    # /def

    @classmethod
    def from_directory(cls, name, drct, clip_at=(1, 15 * u.mas / u.yr)):
        """From Directory

        Load from a directory in the following format:
            result.txt
            gcs/"globular "

        Parameters
        ----------
        drct: str
            the data/ path

        """
        summary = load_summary_table(drct)
        star_table, _ = load_globular_cluster(
            drct + "gcs/" + name + ".ecsv", clip_at=False
        )

        return cls(name, summary, star_table, clip_at=clip_at)

    # /def

    # --------------------------------

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        return setattr(self, name, value)

    # --------------------------------
    # Data Processing

    def bin_profile(self, bins, plot=False):
        r_rbin, z_rbin, z_bins = profile_binning(
            self.r,
            #     gcs["r_ang"] * gcP['rscale_ang'].to_value('deg'),  # TODO FIX
            self.pm.value,
            bins=bins,
            z_clip=[0, 12],
            return_bin=True,
            plot=False,
        )

        if plot is not False:
            kw = plot if isinstance(plot, dict) else {}
            self.plot_binned_profile(bins=bins, **kw)

        return r_rbin, z_rbin, z_bins

    def makeGMMs(self, bins, plot=True):
        """Make a GMM of radial vs kind"""
        self._bins = bins
        self.GMM = GMM_bins(self.r, self.df["pmx"], bins)
        return self.GMM

    def runGMM(self, n_comp=None, max_n_comp=6, plot=True, verbose=False):
        self.GMM.run_GMM_bins(
            n_comp=n_comp, max_n_comp=max_n_comp, plot=plot, verbose=verbose
        )

    def predictMainPop(self, data, annulus: int):
        return self.GMM.gmms[annulus].predict_main_pop(data)

    # --------------------------------
    # Plot

    def plot_binned_profile(self, bins=None, z_clip=[0, 12]):
        import seaborn as sns
        import matplotlib.pyplot as plt

        bins = self._bins if bins is None else bins
        if bins is None:
            raise Exception("need to pass bins or call bin_profile")

        sns.set_palette("RdBu", len(bins))

        fig = plt.figure()
        profile_binning(
            self.r,
            self.pm.value,
            bins=bins,
            z_clip=[0, 12],
            return_bin=False,
            plot=True,
        )
        plt.xlabel("pm [mas/yr]")
        plt.ylabel("density")

        return fig

    # /def


##############################################################################
# END
