import numpy as np
# import matplotlib.pyplot as plt

from sklearn import preprocessing

try:
    from hdbscan import HDBSCAN as DBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN

from .plot import plot_clustering


class DBSCAN_Clustering:
    """DBSCAN clustering class.

    HDBSCAN is used if hdbscan is installed.

    Parameters
    -----------
    data : N-d array [N_samples, N_dimensions]
    scale : Whether to scale the data before clustering. Should be True.

    """

    def __init__(self, data, scale=True, verbose=True):
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

        self.verbose = verbose

    def __str__(self):
        return "DBSCAN clustering Class"

    def __repr__(self):
        if self.run:
            return f"Class {self.__class__.__name__}: "
        else:
            return f"{self.__class__.__name__}"

    def run_clustering(
        self,
        eps=0.3,
        min_samples=None,
        min_frac=0.005,
        plot=True,
        *args,
        **kwargs,
    ):
        """ Perform DBSCAN clustering. 
            Note the clustering does not account for uncertainties.

        Parameters
        -----------
        eps : float, default: 0.3 (not used in HDBSCAN)
            maximum distance between two samples for one to
            be considered as in the neighborhood of the other.

        min_samples : int, default: None (use min_frac)
            The number of samples in a neighborhood
            for a point to be considered as a core point.
            (Used as min_cluster_size in HDBSCAN)

        min_frac : float, default: 0.5%
            The fraction of samples served as min_samples.


        """

        if self.scaled:
            X = self.X_scale
        else:
            X = self.X

        if min_samples is None:
            min_samples = int(X.shape[0] * min_frac)

        if DBSCAN.__name__ == "DBSCAN":
            if self.verbose:
                print(
                    "HDBSCAN not installed. Use scikit-learn DBSCAN instead."
                )
                print("A eps parameter is required by user interaction.")

            self.eps = eps
            self.min_samples = min_samples

            db = DBSCAN(eps=eps, min_samples=min_samples, *args, **kwargs)

        elif DBSCAN.__name__ == "HDBSCAN":
            if self.verbose:
                print("Clustering using HDBSCAN")

            self.min_cluster_size = min_samples

            db = DBSCAN(min_cluster_size=min_samples, *args, **kwargs)

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

    @property
    def labels(self):
        """ Predicted labels """
        return self.db.labels_

    @property
    def core_samples_mask(self):
        """ Mask for core samples """
        core_samples_mask = np.zeros(self.n_sample, dtype=bool)
        core_samples_mask[self.db.core_sample_indices_] = True
        return core_samples_mask

    @property
    def n_components(self):
        return max(self.labels) + 1

    @property
    def components(self):
        """ Samples of each component """
        labels = self.labels
        return [self.X[labels == k] for k in range(self.n_components)]

    def predict_main_pop(self, ind_radius=0):
        """ Whether the sample belongs to main pop or not """

        if max(self.labels) == -1:
            return np.ones(self.n_sample, dtype=bool)

        else:
            means = np.array([comp.mean(axis=0) for comp in self.components])
            r_mean = means[:, ind_radius]
            lab_mp = np.argmin(r_mean)

            is_main_population = np.zeros(self.n_sample, dtype=bool)
            is_main_population[self.labels == lab_mp] = True

            return is_main_population

    @property
    def main_population(self, ind_radius=0):
        """ Samples of the major component """

        return self.X[self.predict_main_pop(ind_radius)]

    @property
    def noise(self):
        """ Samples identified as noise """
        return self.X[self.labels == -1]

    def plot_clustering(
        self, feature_labels=["R", "PMX", "PMY"], *args, **kwargs
    ):
        """ Visualize clustering by pairplot """
        plot_clustering(
            self.X, self.n_dim, self.labels, feature_labels, *args, **kwargs
        )
