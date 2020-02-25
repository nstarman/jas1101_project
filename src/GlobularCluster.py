# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""Globular Cluster Class.


Routine Listings
----------------
GlobularCluster

"""

__all__ = ["GlobularCluster"]


##############################################################################
# IMPORTS

# GENERAL
import astropy.units as u

# CUSTOM

# PROJECT-SPECIFIC
from .load import load_summary_table, load_globular_cluster
from .utils import profile_binning, convert_angle, convert_pm_angular
from .plot import plot_binned_profile
from .GMM import GMM_bins


##############################################################################
# PARAMETERS

dmls = u.dimensionless_unscaled  # shortcut


##############################################################################
# CODE
##############################################################################


class GlobularCluster(object):
    """Globular Cluster Class."""

    def __init__(
        self,
        name,
        property_table,
        star_table,
        clip_at=(1.0, 20 * u.mas / u.yr),
    ):
        """Create Globular Cluster from Data.

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

        self.table_full = star_table
        if clip_at:
            sel = (
                (star_table["r"] < clip_at[0] * self.rc_ang)
                & (0 * u.mas / u.yr < star_table["pm"])  # true by default
                & (star_table["pm"] < clip_at[1])
            )

            self.table = star_table[sel]

        else:
            self.table = star_table
        # storing sub-properties

        # angular
        # adjusted for distance is the same.
        self.x = (self.table["x"] / self.rc_ang).decompose().to_value(dmls)
        self.y = (self.table["y"] / self.rc_ang).decompose().to_value(dmls)
        self.r = (self.table["r"] / self.rc_ang).decompose().to_value(dmls)

        # TODO: normalized pm replacing .value
        self.pm = self.table["pm"]
        self.pmx = self.table["pmx"].value
        self.pmy = self.table["pmy"].value
        self.vsky = convert_pm_angular(self.table["pm"][:], self.d)

        return None

    # /def

    @classmethod
    def from_directory(cls, name, drct, clip_at=(1.0, 20 * u.mas / u.yr)):
        """Load From Directory.

        Load from a directory in the following format:
            result.txt
            output/"globular  "

        Parameters
        ----------
        drct: str
            the get_globular_clusters path

        """
        summary = load_summary_table(drct)
        star_table, _ = load_globular_cluster(
            drct + "gcs/" + name + ".ecsv", clip_at=False
        )

        return cls(name, summary, star_table, clip_at=clip_at)

    # /def

    # --------------------------------

    def __getitem__(self, name):
        """__getitem__."""
        return getattr(self, name)

    def __setitem__(self, name, value):
        return setattr(self, name, value)

    # --------------------------------
    # Data Processing

    def bin_profile(self, bins, z_clip=None, plot=False):
        r_rbin, z_rbin, z_bins = profile_binning(
            self.r,
            #     gcs["r_ang"] * gcP['rscale_ang'].to_value('deg'),  # TODO FIX
            self.pm.value,
            bins,
            z_clip=z_clip,
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
        self.GMMx = GMM_bins(self.r, self.table["pmx"], bins)
        return self.GMM

    def runGMM(self, n_comp=None, max_n_comp=6, plot=True, verbose=False):
        self.GMM.run_GMM_bins(
            n_comp=n_comp, max_n_comp=max_n_comp, plot=plot, verbose=verbose
        )

    def predictMainPop(self, data, annulus: int):
        return self.GMM.gmms[annulus].predict_main_pop(data)

    # --------------------------------
    # Plot

    def plot_binned_profile(self, bins=None, z_clip=None):
        bins = self._bins if bins is None else bins
        plot_binned_profile(self.r, self.pm.value, bins, z_clip=z_clip)

    # /def


##############################################################################
# END
