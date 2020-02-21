import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from astropy.stats import mad_std
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .plot import LogNorm


class GMM_bins:
    # TODO make work on astropy

    def __init__(self, x, y, bins):
        # self.gc = gc
        self.x = x
        self.y = y
        self.bins = bins
        self.gmms = []
        self.run = False

    def __str__(self):
        return "GMM Mixture Class in bins"

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
        self, n_comp=None, max_n_comp=6, plot=True, verbose=False
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

    def plot_pred_main_pop(self, k_std=2.0, p_thre=0.8):
        """ Visualize the GMM discrimination. """
        if not self.run:
            return None

        # gc = self.gc
        bins = self.bins

        for i in range(len(bins) - 1):
            ind = (bins[i] < self.x) & (self.x < bins[i + 1])
            data = self.y[ind]
            print(data)

            gmm = self.gmms[i]

            p_main_pop = gmm.predict_main_pop(data, k_std=k_std)
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

    def plot_pred_main_pop_all(self, z_bins, k_std=2.0, p_thre=0.8):
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
    def __init__(self, data):
        self.data = data
        self.run = False

    def __str__(self):
        return "GMM Mixture Class"

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
        self, n_comp=None, max_n_comp=6, verbose=True, plot=True, **kwargs
    ):
        """ Perform GMM fitting. Return a scikit learn Gaussian Mixture Model or a list of them. """
        from sklearn import mixture

        self.max_n_comp = max_n_comp
        self.n_comp = n_comp
        data = self.data

        X = data[:, None]

        if n_comp is not None:
            gmm = mixture.GaussianMixture(n_components=n_comp, **kwargs)
            gmm.fit(X)

            BIC = None

        else:
            N_comp = np.arange(1, max_n_comp + 1, 1)
            BIC = np.zeros(len(N_comp))
            gmm = []
            for N in N_comp:
                g = mixture.GaussianMixture(n_components=N, **kwargs)
                g.fit(X)
                BIC[N - 1] = g.bic(X)
                gmm.append(g)

            gmm = gmm[np.argmin(BIC)]

            if verbose:
                print("Optimal # of Components:", N_comp[np.argmin(BIC)])

        self.gmm = gmm

        self.BIC = BIC
        self.run = True

        if plot:
            self.plot_GMM(verbose=verbose)

    @property
    def gmm_means(self):
        if self.run:
            return self.gmm.means_.ravel()

    @property
    def gmm_sigmas(self):
        if self.run:
            return self.gmm.covariances_.ravel()

    @property
    def gmm_weights(self):
        if self.run:
            return self.gmm.weights_.ravel()

    def plot_GMM(self, verbose=True):
        """ Visualize GMM results """
        if not self.run:
            return None

        data = self.data
        gmm = self.gmm
        BIC = self.BIC

        means = self.gmm_means
        sigmas = self.gmm_sigmas
        weights = self.gmm_weights

        if verbose:
            print(
                "Fitted Parameters for Mixture Gaussian using %d components:"
                % gmm.n_components
            )
            print("Mean:", *np.around(means, 3))
            print("Sigma:", *np.around(sigmas, 3))
            print("Weight:", *np.around(weights, 3))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.distplot(data, kde_kws={"lw": 4}, color="violet", label="data")
        sns.distplot(
            gmm.sample(10000)[0],
            hist=False,
            color="k",
            kde_kws={"lw": 4, "alpha": 0.7},
            label="GMM Fit",
        )
        plt.legend(loc="best")

        for w, m, s, in zip(weights, means, sigmas):
            rv = stats.norm(loc=m, scale=s)
            x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
            plt.plot(x, w * rv.pdf(x), "--", color="k", lw=3, alpha=0.7)

        plt.xlim(0, np.quantile(data, 0.999))
        plt.xlabel("Proper Motion (mas)")
        plt.ylabel("PDF")

        if BIC is not None:
            N_comp = np.arange(1, len(BIC) + 1, 1)
            axins = inset_axes(
                ax,
                width="35%",
                height="35%",
                bbox_to_anchor=(-0.02, -0.2, 1, 1),
                bbox_transform=ax.transAxes,
            )
            axins.plot(N_comp, BIC / BIC.min(), "ro-")
            axins.text(
                0.6,
                0.8,
                "N$_{best}$=%d" % N_comp[np.argmin(BIC)],
                transform=axins.transAxes,
            )
            axins.axhline(1, color="k", ls="--", zorder=1)
            axins.set_ylabel("BIC / BIC$_{min}$", fontsize=12)
            axins.tick_params(axis="both", which="major", labelsize=10)

        plt.show()

    def predict_main_pop(self, data, k_std=2.0):
        """ 
        Predict probability for each sample that it belongs to the main population:
        with GMM means located in [median +/- k_std * std]
        """
        if not self.run:
            return None

        gmm = self.gmm

        X = data[:, None]
        means = gmm.means_.ravel()
        med, std = np.median(X), np.std(X)
        main_pop = (means >= med - k_std * std) & (means <= med + k_std * std)

        prob_main_pop = gmm.predict_proba(X)[:, main_pop].sum(axis=1)

        return prob_main_pop
