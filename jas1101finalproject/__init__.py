# -*- coding: utf-8 -*-

"""JAS1101 Final Project.

A pipeline to analyze Globular Clusters for the presence of IMBHs.

"""

__author__ = ["Nathaniel Starkman", "Qing Liu", "Vivian Ngo"]
__copyright__ = "Copyright 2020"
# __credits__ = [""]
__license__ = "BSD-3"
__version__ = "0.0.0"
__status__ = "Production"

# __all__ = [
#     ""
# ]


##############################################################################
# IMPORTS

# Packages may add whatever they like to this file, but
# should keep this content at the top.
from ._astropy_init import *   # noqa

# GENERAL

import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord


# PROJECT-SPECIFIC

# from .load import (
#     load_summary_table,
#     load_globular_cluster,
#     load_all_globular_clusters,
# )
from .GlobularCluster import GlobularCluster
# from .utils import profile_binning, convert_angle, convert_pm_angular


##############################################################################
# PARAMETERS

dmls: u.Quantity = u.dimensionless_unscaled


##############################################################################
# CODE
##############################################################################


##############################################################################
# END
