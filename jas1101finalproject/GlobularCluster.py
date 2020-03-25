# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# PROJECT : JAS1101 Final Project
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
import numpy as np
import astropy.units as u
from astropy.table import Table

from typing import Tuple, Any, Sequence, Optional, Union
from typing_extensions import Literal

# CUSTOM

# PROJECT-SPECIFIC
from . import data

# from .data import load_summary_table, load_globular_cluster
from .utils import profile_binning, convert_angle, convert_pm_angular
from .plot import plot_binned_profile
from .GMM import GMM_bins


##############################################################################
# PARAMETERS

dmls: u.Quantity = u.dimensionless_unscaled  # shortcut


##############################################################################
# CODE
##############################################################################


class GlobularCluster(object):
    """Globular Cluster Class."""

    def __init__(
        self,
        name: str,
        property_table: Table,
        star_table: Table,
        # pm_norm_method: Union[Literal["pmscale"], Literal["GMM"]] = "pmscale",
        member_method: Optional[Literal["DBScan"]] = None,
        member_threshold: float = 0.8,
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
        self.name: str = name
        # self._bins = None

        # ----------------------------

        if len(property_table) > 1:  # then full summary table of all GCs
            property_table = property_table.loc[name]  # QTable to not be Row
        else:
            pass

        # Property table
        self.summary: Table = property_table

        # storing specific-properties
        self.nstar: int = property_table["nstar"]

        self.rc_ang: u.Quantity = property_table["rscale"]
        self.pmc_ang: u.Quantity = property_table["pmscale"]

        self.d: u.Quantity = property_table["dist"]
        self.rc: u.Quantity = convert_angle(self.rc_ang, self.d)
        self.pmc: u.Quantity = convert_pm_angular(self.pmc_ang, self.d)

        # ----------------------------

        self.table_full: Table = star_table

        if member_method == None:
            self.table: Table = self.table_full[
                self.table_full["memberprob"] >= member_threshold
            ]
        elif member_method == "GMM":
            self.table: Table = self.table_full[
                self.table_full["memberprob_GMM"] >= member_threshold
            ]
        else:
            raise ValueError("Not an allowed value")

        # storing sub-properties

        # angular
        # adjusted for distance is the same.
        self.x: np.ndarray = (self.table["x"] / self.rc_ang).to_value(dmls)
        self.y: np.ndarray = (self.table["y"] / self.rc_ang).to_value(dmls)
        self.r: np.ndarray = (self.table["r"] / self.rc_ang).to_value(dmls)

        # if pm_norm_method == "pmscale":
        self.pm: float = (self.table["pm"] / self.pmc_ang).to_value(dmls)
        self.pmx: float = (self.table["pmx"] / self.pmc_ang).to_value(dmls)
        self.pmy: float = (self.table["pmy"] / self.pmc_ang).to_value(dmls)
        # elif pm_norm_method == "GMM":
        #     raise ValueError("Not yet implemented")
        # else:
        #     raise ValueError("Not an allowed value")

        self.vsky: u.Quantity = convert_pm_angular(self.table["pm"][:], self.d)

        return None

    # /def

    @classmethod
    def from_name(
        cls,
        name: str,
        member_threshold: float = 0.8,
        # clip_at: Tuple[float, u.Quantity] = (1.0, 20 * u.mas / u.yr),
    ):
        """Load From Directory.

        Load from a directory in the following format:
            result.txt
            output/"globular  "

        Parameters
        ----------
        drct: str
            the get_globular_clusters path

        """
        property_table = data.load_summary_table()
        star_table = data.load_globular_cluster(name + ".ecsv")

        return cls(
            name=name,
            property_table=property_table,
            star_table=star_table,
            member_threshold=member_threshold
            # clip_at=clip_at
        )

    # /def

    # --------------------------------

    def __getitem__(self, name: str):
        """__getitem__."""
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any):
        """__setitem__."""
        return setattr(self, name, value)

    # --------------------------------
    # Data Processing

    # def bin_profile(self, bins: Sequence, z_clip=None, plot: bool = False):
    #     """Bin Profile.

    #     Parameters
    #     ----------
    #     bins : array-like
    #     z_clip : optional
    #     plot : bool, optional
    #         plot result

    #     Returns
    #     -------
    #     r_rbin : array-like
    #     z_rbin : array-like
    #     z_bins : array-like

    #     """
    #     r_rbin, z_rbin, z_bins = profile_binning(
    #         self.r,
    #         #     gcs["r_ang"] * gcP['rscale_ang'].to_value('deg'),  # TODO FIX
    #         self.pm.value,
    #         bins,
    #         z_clip=z_clip,
    #         return_bin=True,
    #         plot=False,
    #     )

    #     if plot is not False:
    #         kw = plot if isinstance(plot, dict) else {}
    #         self.plot_binned_profile(bins=bins, **kw)

    #     return r_rbin, z_rbin, z_bins

    # # /def

    # def makeGMMs(self, bins: Sequence, plot: bool = True):
    #     """Make a GMM of radial vs kind.

    #     Parameters
    #     ----------
    #     bins : array-like
    #     plot : bool

    #     Returns
    #     -------
    #     GMM : GMM_bins

    #     """
    #     self._bins = bins
    #     self.GMM = GMM_bins(self.r, self.table["pmx"], bins)
    #     return self.GMM

    # # /def

    # def runGMM(
    #     self,
    #     n_comp: Optional[int] = None,
    #     max_n_comp: int = 6,
    #     plot: bool = True,
    #     verbose: bool = False,
    # ):
    #     """Runs GMM.

    #     calls run_GMM_bins

    #     Parameters
    #     ----------
    #     n_comp : int, optional
    #         number of components
    #     max_n_com : int, optional
    #         maximum number of components
    #     plot : bool, optional
    #         whether to plot GMM bins
    #     verbose : bool, optional
    #         verbosity of GMM fit

    #     """
    #     self.GMM.run_GMM_bins(
    #         n_comp=n_comp, max_n_comp=max_n_comp, plot=plot, verbose=verbose
    #     )

    # # /def

    # def predictMainPop(self, data, annulus: int):
    #     """Predict whether star belongs to main population or the background.

    #     Parameters
    #     ----------
    #     data
    #     annulus

    #     Returns
    #     -------
    #     array-like

    #     """
    #     return self.GMM.gmms[annulus].predict_main_pop(data)

    # # /def

    # # --------------------------------
    # # Plot

    # def plot_binned_profile(self, bins=None, z_clip=None):
    #     """Plot Binned Profile.

    #     Parameters
    #     ----------
    #     bins
    #     z_clip

    #     Returns
    #     -------
    #     fig : Figure

    #     """
    #     bins = self._bins if bins is None else bins
    #     return plot_binned_profile(self.r, self.pm.value, bins, z_clip=z_clip)

    # # /def


##############################################################################
# END
