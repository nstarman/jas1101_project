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

import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

# CUSTOM

# PROJECT-SPECIFIC

from .load import (
    load_summary_table,
    load_globular_cluster,
    load_all_globular_clusters,
)
from .GlobularCluster import GlobularCluster
from .utils import profile_binning, convert_angle, convert_pm_angular


##############################################################################
# PARAMETERS

dmls = u.dimensionless_unscaled


##############################################################################
# CODE
##############################################################################


##############################################################################
# END
