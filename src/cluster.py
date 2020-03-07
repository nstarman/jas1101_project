# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : Cluster
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""Data Clustering.

Routine Listings
----------------
DBSCAN_Clustering

"""

__all__ = ["DBSCAN_Clustering"]


###############################################################################
# IMPORTS

# GENERAL
import numpy as np
from sklearn import preprocessing

from typing import Optional

try:
    from hdbscan import HDBSCAN as DBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN

# PROJECT-SPECIFIC
from .plot import plot_clustering


###############################################################################
# CODE
###############################################################################


class DBSCAN_Clustering:
    """DBSCAN clustering class.

    HDBSCAN is used if hdbscan is installed.

    Attributes
    ----------
    labels
    core_samples_mask
    n_components
    components
    main_population
    noise

    Methods
    -------
    run_clustering
    predict_main_pop
    plot_clustering

    """

    def __init__(self, data, scale: bool = True, verbose: bool = True):
        """__init__.
        
        Parameters
        ----------
        data : array-like
            N-d array [N_samples, N_dimensions]
        scale : bool, optional
            Whether to scale the data before clustering. Should be True. 
        verbose : bool, optional

        """
        self.data = data
        self.run = False
        self.scaled = scale

        self.n_sample = data.shape[0]
        self.n_dim = data.shape[1]

        X = data[:, None] if self.n_dim == 1 else data

        self.X = X

        if scale:
            X_scale = preprocessing.scale(X)

        self.X_scale = X_scale

        self.verbose = True

    # /def

    def __str__(self):
        """String representation."""
        return "DBSCAN clustering Class"

    # /def

    def __repr__(self):
        """Representation."""
        if self.run:
            return f"Class {self.__class__.__name__}: "
        else:
            return f"{self.__class__.__name__}"

    # /def

    def run_clustering(
        self,
        eps: float = 0.3,
        min_samples: Optional[int] = None,
        min_frac: float = 0.005,
        plot: bool = True,
        *args,
        **kwargs,
    ):
        """Perform DBSCAN clustering.

        Note the clustering does not account for uncertainties.

        Parameters
        -----------
        eps : float, optional
            maximum distance between two samples for one to
            be considered as in the neighborhood of the other.
            default: 0.3 (not used in HDBSCAN)
        min_samples : int, optional
            The number of samples in a neighborhood
            for a point to be considered as a core point.
            (Used as min_cluster_size in HDBSCAN)
            default: None (use min_frac)
        min_frac : float, optional
            The fraction of samples served as min_samples.
            default: 0.5%

        """
        # -----------------
        self.min_samples: int

        # -----------------
        if self.scaled:
            X = self.X_scale
        else:
            X = self.X

        if min_samples is None:
            min_samples: int = int(X.shape[0] * min_frac)

        if DBSCAN.__name__ == "DBSCAN":
            if self.verbose:
                print(
                    "HDBSCAN not installed. Use scikit-learn DBSCAN instead."
                )
                print("A eps parameter is required by user interaction.")

            self.eps: float = eps
            self.min_samples: int = min_samples

            db = DBSCAN(eps=eps, min_samples=min_samples, *args, **kwargs)

        elif DBSCAN.__name__ == "HDBSCAN":
            if self.verbose:
                print("Clustering using HDBSCAN")

            self.min_cluster_size = min_samples

            db = DBSCAN(min_cluster_size=min_samples, *args, **kwargs)

        # /if

        db.fit(X)

        self.db = db

        self.run = True
        if self.verbose:
            print(
                "Clustering Finished: {0:d} components".format(
                    self.n_components
                )
            )

            if max(self.labels) == -1:
                print("Noisy data. Be caution.")

        if plot:
            self.plot_clustering()

        return

    # /def

    @property
    def labels(self):
        """Predicted labels.

        Returns
        -------
        list

        """
        return self.db.labels_

    # /def

    @property
    def core_samples_mask(self):
        """Mask for core samples.

        Returns
        -------
        ndarray

        """
        core_samples_mask = np.zeros(self.n_sample, dtype=bool)
        core_samples_mask[self.db.core_sample_indices_] = True
        return core_samples_mask

    # /def

    @property
    def n_components(self):
        """Number of Components.

        Returns
        -------
        int

        """
        return max(self.labels) + 1

    # /def

    @property
    def components(self):
        """Samples of each component.

        Returns
        -------
        list

        """
        labels = self.labels
        return [self.X[labels == k] for k in range(self.n_components)]

    # /def

    def predict_main_pop(self, ind_radius=0):
        """Whether the sample belongs to main pop or not.

        Parameters
        ----------
        ind_radius : float, optional

        Returns
        -------
        is_main_population : array-like
            boolean array

        """
        if max(self.labels) == -1:
            is_main_population = np.ones(self.n_sample, dtype=bool)

        else:
            means = np.array([comp.mean(axis=0) for comp in self.components])
            r_mean = means[:, ind_radius]
            lab_mp = np.argmin(r_mean)

            is_main_population = np.zeros(self.n_sample, dtype=bool)
            is_main_population[self.labels == lab_mp] = True

        return is_main_population

    # /def

    @property
    def main_population(self, ind_radius=0):
        """Samples of the major component.

        Parameters
        ----------
        ind_radius : float, optional

        Returns
        -------
        array-like

        """
        return self.X[self.predict_main_pop(ind_radius)]

    # /def

    @property
    def noise(self):
        """Samples identified as noise.

        Returns
        -------
        array-like

        """
        return self.X[self.labels == -1]

    # /def

    def plot_clustering(
        self, feature_labels=["R", "PMX", "PMY"], *args, **kwargs
    ):
        """Visualize clustering by pairplot.

        Parameters
        ----------
        feature_labels : list, optional
        args
            passed to plot_clustering
        kwargs
            passed to plot_clustering

        Returns
        -------
        fig : Figure
            matplotlib figure object

        """
        return plot_clustering(
            self.X, self.n_dim, self.labels, feature_labels, *args, **kwargs
        )

    # /def
