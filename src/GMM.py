import numpy as np
import scipy.stats as stats
from astropy.stats import mad_std

from sklearn import mixture

import seaborn as sns
import matplotlib.pyplot as plt

from .plot import plot_GMM_1d, plot_GMM_2d, plot_BIC_diagnosis
from .plot import LogNorm, AsinhNorm


class GMM_bins:
    # TODO make work on astropy

    def __init__(self, data, bins):
        # self.gc = gc
        self.data = data
        self.bins = bins
        self.gmms = []
        self.run = False

    def __str__(self):
        return "GMM Class in radial bins"

    def __repr__(self):
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

    def run_GMM_bins(
        self, n_comp=None, max_n_comp=12, plot=True, verbose=False
    ):
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

    def plot_pred_main_pop_1d(self, k_std=2.5, p_thre=0.8):
        """ Visualize the GMM discrimination. """
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

    def plot_pred_main_pop_1d_tot(self, z_bins, k_std=2.5, p_thre=0.8):
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


class GMM:
    """ 
    GMM class
        
    -----------
    data: 1d array or Nd array [N_samples, N_dimensions]
    
    Note: Visualization are only available < 2d.

    """
    
    def __init__(self, data):
        self.data = data
        self.run = False
        self.ndim = np.ndim(data)

    def __str__(self):
        return "GMM Class"

    def __repr__(self):
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

    def run_GMM(
        self, n_comp=None, max_n_comp=10, verbose=True, plot=True, **kwargs
    ):
        """ Perform GMM fitting. Yield a scikit-learn Gaussian Mixture Model
        or a list of them. Note the GMM does not account for uncertainties.
        
        Parameters
        -----------
        n_comp : specified number of components. Use N = argmin(BIC) 
            if n_comp is not given
        max_n_comp : maximum number of component.
        
        """
    
        self.max_n_comp = max_n_comp
        self.n_comp = n_comp
        data = self.data
        
        X = data[:, None] if np.ndim(data)==1 else data
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

    @property
    def means(self):
        if self.run:
            return self.gmm.means_

    @property
    def covariances(self):
        if self.run:
            return self.gmm.covariances_
    
    @property
    def sigmas(self):
        if self.run:
            covar = self.covariances
            if covar is not None:
                return np.diagonal(covar, axis1=1, axis2=2)
            else:
                return None

    @property
    def weights(self):
        if self.run:
            return self.gmm.weights_
    
    def GMM_summary(self, verbose=True):
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
            

    def plot_GMM(self, BIC_style='dark_background', *args, **kwargs):
        """ Visualize GMM results: decomposation and BIC vs N_comp """
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
            ax = plot_GMM_2d(data, weights, self.means, self.covariances, *args, **kwargs)
            
            plot_BIC_diagnosis(BIC, ax, bbox_to_anchor=(-0.55, 0, 1, 1), style=BIC_style)
            
        plt.show()

    def predict_main_pop(self, k_std=2.5):
        """  Predict probability for each sample that it
            belongs to the main population:
        
        1D: GMM means located in [median +/- k_std * std]
        
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
        
        elif self.ndim==2:
            return None
