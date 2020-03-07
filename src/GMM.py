# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# PROJECT : JAS1101 Final Project
#
# ----------------------------------------------------------------------------

# Docstring
"""Gaussian Mixture Model.

Routine Listings
----------------
GMM_bins
GMM

"""

__all__ = ["GMM_bins", "GMM"]


###############################################################################
# IMPORTS

# GENERAL
import numpy as np

from typing import Optional

from astropy.stats import mad_std

from sklearn import mixture

# plot
import matplotlib.pyplot as plt

from .plot import (
    plot_GMM_1d,
    plot_GMM_2d,
    plot_BIC_diagnosis,
    LogNorm,
    AsinhNorm,
)


###############################################################################
# CODE
###############################################################################


class GMM_bins(object):
    """GMM bins."""

    # TODO make work on astropy

    def __init__(self, data, bins):
        """__init__.

        Parameters
        ----------
        data
        bins

        """
        self.data = data
        self.bins = bins
        self.gmms = []
        self.run = False

    # /def

    def __str__(self):
        """String representation."""
        return "GMM Class in radial bins"

    # /def

    def __repr__(self):
        """Primitive string representation."""
        if self.run:
            if self.n_comp is None:
                return (
                    f"Class {self.__class__.__name__}: "
                    f" Max Component Number = {self.max_n_comp}"
                )
            if self.n_comp is not None:
                return (
                    f"Class {self.__class__.__name__}: "
                    f" Fixed Component Number = {self.n_comp}"
                )
        else:
            return f"{self.__class__.__name__}"

    # /def

    def run_GMM_bins(
        self,
        n_comp: Optional[int] = None,
        max_n_comp: int = 12,
        plot: bool = True,
        verbose: bool = False,
    ):
        """Run GMM bins.

        Parameters
        ----------
        n_comp : int, optional
        max_n_com : int, optional
        plot : bool, optional
        verbose : bool, optional

        """
        # gc = self.gc
        bins = self.bins

        self.max_n_comp = max_n_comp
        self.n_comp = n_comp

        for i in range(len(bins) - 1):
            sel = (bins[i] < self.x) & (self.x < bins[i + 1])
            gmm = GMM(self.y[sel])
            gmm.run_GMM(
                n_comp=n_comp,
                max_n_comp=max_n_comp,
                plot=plot,
                verbose=verbose,
            )
            self.gmms.append(gmm)

        print("Finished")
        self.run = True

    # /def

    def plot_pred_main_pop_1d(self, k_std=2.5, p_thre=0.8):
        """Visualize the GMM discrimination.

        Parameters
        ----------
        k_std : float
        p_thre : float

        """
        if not self.run:
            return None

        bins = self.bins

        for i in range(len(bins) - 1):
            ind = (bins[i] < self.x) & (self.x < bins[i + 1])
            data = self.y[ind]
            print(data)

            gmm = self.gmms[i]

            p_main_pop = gmm.predict_main_pop1d(data, k_std=k_std)
            is_main_pop = p_main_pop > p_thre

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(
                data,
                bins=np.linspace(0, 15, 11),
                histtype="step",
                label="All",
                lw=3,
                alpha=0.8,
            )
            ax.hist(
                data[is_main_pop],
                histtype="step",
                color="navy",
                label="Main",
                lw=3,
                alpha=0.8,
            )
            ax.hist(
                data[~is_main_pop],
                histtype="step",
                color="gold",
                label="Minor",
                lw=3,
                alpha=0.8,
            )
            txt = r"${0:.3f}<r<{1:.3f}$".format(bins[i], bins[i + 1])
            ax.text(0.6, 0.85, txt, fontsize=15, transform=ax.transAxes)
            plt.xlabel("Proper Motion (mas)")
            plt.ylabel("# of Stars")
            plt.show()

    # /def

    def plot_pred_main_pop_1d_tot(self, z_bins, k_std=2.5, p_thre=0.8):
        """Plot predicted main population 1d total.

        Parameters
        ----------
        z_bins : array-like
        k_std : float
        p_thre : float

        """
        if not self.run:
            return None

        bins = self.bins

        for i in range(len(bins) - 1):

            r = z_bins[str(i)]["r"]
            pm = z_bins[str(i)]["pm"]
            gmm = self.gmms[i]

            prob_main_pop = gmm.predict_main_pop(pm, k_std=k_std)
            is_main_pop = prob_main_pop > p_thre

            plt.scatter(
                r[is_main_pop], pm[is_main_pop], color="navy", alpha=0.1, s=2
            )
            plt.scatter(
                r[~is_main_pop], pm[~is_main_pop], color="gold", alpha=0.3, s=5
            )
            plt.axvline(bins[i], color="k", alpha=0.2)
            plt.ylabel("Proper Motion (mas)")
            plt.xlabel("Radius (deg)")

        plt.show()

    # /def


###############################################################################


class GMM(object):
    """Gaussian Mixture Model.

    Notes
    -----
    Visualization are only available < 2d.

    """

    def __init__(self, data):
        """Gaussian Mixture Model.

        Parameters
        ----------
        data: array-like
            1d array or Nd array [N_samples, N_dimensions]

        """
        self.data = data
        self.run = False
        self.ndim = np.ndim(data)

    # /def

    def __str__(self):
        """String representation.

        Returns
        -------
        str

        """
        return "GMM Class"

    # /def

    def __repr__(self):
        """String representation.

        Returns
        -------
        str

        """
        if self.run:
            if self.n_comp is None:
                return (
                    f"Class {self.__class__.__name__}: "
                    f" Max Component Number = {self.max_n_comp}"
                )
            if self.n_comp is not None:
                return (
                    f"Class {self.__class__.__name__}: "
                    f" Fixed Component Number = {self.n_comp}"
                )
        else:
            return f"{self.__class__.__name__}"

    # /def

    def run_GMM(
        self,
        n_comp: Optional[int] = None,
        max_n_comp: int = 10,
        verbose: bool = True,
        plot: bool = True,
        **kwargs,
    ):
        """Perform GMM fitting.

        Yield a scikit-learn Gaussian Mixture Model or a list of them.
        Note the GMM does not account for uncertainties.

        Parameters
        ----------
        n_comp : int, optional
            specified number of components. Use N = argmin(BIC)
            if n_comp is not given
        max_n_comp : int, optional
            maximum number of component.
        verbose : bool, optional
        plot : bool, optional
        kwargs

        """
        self.max_n_comp = max_n_comp
        self.n_comp = n_comp
        data = self.data

        X = data[:, None] if np.ndim(data) == 1 else data
        self.X = X

        if n_comp is not None:
            # number of component specified
            gmm = mixture.GaussianMixture(n_components=n_comp, **kwargs)
            gmm.fit(X)

            BIC = None

        else:
            # run from single component to max_n_comp
            N_comp = np.arange(1, max_n_comp + 1, 1)
            BIC = np.zeros(len(N_comp))
            gmm = []
            for N in N_comp:
                g = mixture.GaussianMixture(n_components=N, **kwargs)
                g.fit(X)
                BIC[N - 1] = g.bic(X)
                gmm.append(g)

            # the optimal fit is given by min BIC
            gmm = gmm[np.argmin(BIC)]

            if verbose:
                print("Optimal # of Components:", N_comp[np.argmin(BIC)])

        self.gmm = gmm

        self.BIC = BIC
        self.run = True

        if plot:
            self.plot_GMM(verbose=verbose)

    # /def

    @property
    def means(self):
        """Means."""
        if self.run:
            return self.gmm.means_

    # /def

    @property
    def covariances(self):
        """Covariances."""
        if self.run:
            return self.gmm.covariances_

    # /def

    @property
    def sigmas(self):
        """Sigmas."""
        if self.run:
            covar = self.covariances
            if covar is not None:
                return np.diagonal(covar, axis1=1, axis2=2)
            else:
                return None

    # /def

    @property
    def weights(self):
        """Weights."""
        if self.run:
            return self.gmm.weights_

    # /def

    def GMM_summary(self, verbose=True):
        """GMM Summary.

        Parameters
        ----------
        verbose : bool, optional

        """
        weights = self.weights
        means = self.means
        sigmas = self.sigmas

        if verbose:
            print(
                "Fitted Parameters for Mixture Gaussian using %d components:"
                % self.gmm.n_components
            )
            print("Weight:", *np.around(weights, 3))
            print("Mean:", *np.around(means, 3))
            print("Sigma:", *np.around(sigmas, 3))

    # /def

    def plot_GMM(self, BIC_style="dark_background", *args, **kwargs):
        """Visualize GMM results.

        Decomposition and BIC vs N_comp

        Parameters
        ----------
        BIC_style : str, optional
        args
        kwargs

        """
        if not self.run:
            return None

        data = self.data
        BIC = self.BIC
        weights = self.weights

        if self.ndim == 1:
            sample = self.gmm.sample(10000)[0]
            means, sigmas = self.means.ravel(), self.sigmas.ravel()

            ax = plot_GMM_1d(data, weights, means, sigmas, sample, **kwargs)

            plot_BIC_diagnosis(BIC, ax)

        if self.ndim == 2:
            ax = plot_GMM_2d(
                data, weights, self.means, self.covariances, *args, **kwargs
            )

            plot_BIC_diagnosis(
                BIC, ax, bbox_to_anchor=(-0.55, 0, 1, 1), style=BIC_style
            )

        plt.show()

    # /def

    def predict_main_pop(self, X, k_std=2.5):
        """Predict probability of main population membership.

        1D: GMM means located in [median +/- k_std * std]

        Parameters
        ----------
        X : array-like
        k_std : float, optional

        Returns
        -------
        array-like or None

        """
        if not self.run:
            return None

        gmm = self.gmm

        if self.ndim == 1:
            means = gmm.means_.ravel()
            med, std = np.median(X), mad_std(X)
            main_pop = abs(means - med) < k_std * std

            prob_main_pop = gmm.predict_proba(X)[:, main_pop].sum(axis=1)

            return prob_main_pop

        elif self.ndim == 2:
            return None

    # /def


###############################################################################
# END
