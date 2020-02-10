import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from astropy.stats import mad_std
import seaborn as sns


from matplotlib import rcParams
rcParams.update({'figure.figsize': [7,5]})
rcParams.update({'xtick.major.pad': '5.0'})
rcParams.update({'xtick.major.size': '4'})
rcParams.update({'xtick.major.width': '1.'})
rcParams.update({'xtick.minor.pad': '5.0'})
rcParams.update({'xtick.minor.size': '4'})
rcParams.update({'xtick.minor.width': '0.8'})
rcParams.update({'ytick.major.pad': '5.0'})
rcParams.update({'ytick.major.size': '4'})
rcParams.update({'ytick.major.width': '1.'})
rcParams.update({'ytick.minor.pad': '5.0'})
rcParams.update({'ytick.minor.size': '4'})
rcParams.update({'ytick.minor.width': '0.8'})
rcParams.update({'axes.labelsize': 14})
rcParams.update({'font.size': 14})

def LogNorm():
    from astropy.visualization import LogStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    return ImageNormalize(stretch=LogStretch())

def profile_binning(r, z, bins=np.logspace(np.log10(0.1), np.log10(0.5), 11),
                    z_name="pm", z_clip=[2,15], return_bin=False, plot=True):
    """ Bin the given quantity. """
    clip = lambda z: (z>z_clip[0])&(z<z_clip[1])
    z_bins = {}
    
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        
    # Clip by bins
    for k, b in enumerate(bins[:-1]):
        in_bin = (r>=bins[k])&(r<bins[k+1])
        clipped = clip(z[in_bin])
        z_in_bin = z[in_bin][clipped].values
        r_in_bin = r[in_bin][clipped].values
        
        z_bin = {z_name: z_in_bin,
                 "r": r_in_bin}
        z_bins[str(k)] = z_bin

        if plot:
            lab = "{0:.2f}<r<{1:.2f}".format(bins[k],bins[k+1])
            sns.distplot(z_in_bin, hist=False,  kde_kws={"lw":2, "alpha":0.9}, label=lab)
            
    if return_bin:
        return z_bins
    else:
        r_rbin, z_rbin = get_mean_rbins(z_bins, z_name=z_name)
        return r_rbin, z_rbin
    
def get_mean_rbins(z_bins, z_name="pm"):
    """ Get mean of radial bins. """
    res = np.array([[np.mean(val['r']), np.mean(val[z_name])] for key, val in z_bins.items()])
    r_rbin, z_rbin = res[:,0], res[:,1]
    return r_rbin, z_rbin
    
def GMM(data, n_comp=None, max_n_comp=6, verbose=True, plot=True):
    """ Perform GMM fitting. Return a scikit learn Gaussian Mixture Model or a list of them. """
    from sklearn import mixture
    X = data[:, None]
    
    if n_comp is not None:
        gmm = mixture.GaussianMixture(n_components=n_comp)
        gmm.fit(X) 
        
        BIC = None
        
    else:
        N_comp = np.arange(1,max_n_comp+1,1)
        BIC = np.zeros(len(N_comp))
        gmm = []
        for N in N_comp:
            g = mixture.GaussianMixture(n_components=N)
            g.fit(X) 
            BIC[N-1] = g.bic(X)
            gmm.append(g)
            
        gmm = gmm[np.argmin(BIC)]
        
        if verbose:   
            print("Optimal # of Components:", N_comp[np.argmin(BIC)])
    
    if plot:
        plot_GMM(data, gmm, BIC=BIC, verbose=verbose)
            
    return gmm

def predict_main_pop(data, gmm, k_std=2., p_thre=0.8):
    """ 
    Predict probability for each sample that it belongs to the main population:
    with GMM means located in [median +/- k_std * std]
    """
    X = data[:,None]
    means = gmm.means_.ravel()
    med, std = np.median(X), np.std(X)
    main_pop = (means >= med - k_std * std) & (means <= med + k_std * std)
    
    prob_main_pop = gmm.predict_proba(X)[:,main_pop].sum(axis=1)
    
    return prob_main_pop


def plot_GMM(data, gmm, BIC=None, verbose=True):
    """ Visualize GMM results """
    means = gmm.means_.ravel()
    sigmas = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()
    
    if verbose:
        print("Fitted Parameters for Mixture Gaussian using %d components:"%gmm.n_components)
        print("Mean:", *np.around(means,3))
        print("Sigma:", *np.around(sigmas,3))
        print("Weight:", *np.around(weights,3))
        
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    sns.distplot(data, kde_kws={"lw":4}, color="violet", label="data")
    sns.distplot(gmm.sample(10000)[0], hist=False, color="k", 
                 kde_kws={"lw":4, "alpha":0.7}, label="GMM Fit")
    plt.legend(loc="best")

    for w,m,s, in zip(weights, means, sigmas):
        rv = stats.norm(loc=m, scale=s)
        x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
        plt.plot(x, w*rv.pdf(x), '--', color="k", lw=3, alpha=0.7)
    
    plt.xlim(0, data.quantile(0.999))
    plt.xlabel("Proper Motion (mas)")
    plt.ylabel("PDF")
    
    if BIC is not None:
        N_comp = np.arange(1,len(BIC)+1,1)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="35%", height="35%",
                           bbox_to_anchor=(-0.02,-0.2,1,1), bbox_transform=ax.transAxes)
        axins.plot(N_comp, BIC/BIC.min(), "ro-")
        axins.text(0.6,0.8,"N$_{best}$=%d"%N_comp[np.argmin(BIC)], transform=axins.transAxes)
        axins.axhline(1, color="k", ls="--", zorder=1)
        axins.set_ylabel("BIC / BIC$_{min}$",fontsize=12)
        axins.tick_params(axis='both', which='major', labelsize=10)
        
    plt.show()
    
def plot_predict_main_pop(gc, bins, gmm_bins, p_thre=0.8):
    """ Visualize the GMM discrimination. """
    for i in range(len(bins)-1):
        gcb = gc[(bins[i]<gc.r)&(gc.r<bins[i+1])]
        data = gcb.pm

        gmm = gmm_bins[i]

        p_main_pop = predict_main_pop(data, gmm)
        is_main_pop = p_main_pop > p_thre
        
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.hist(data, bins=np.linspace(0,15,11), histtype="step", label="All", lw=3, alpha=0.8)
        ax.hist(data[is_main_pop], histtype="step", color="navy", label="Main", lw=3, alpha=0.8)
        ax.hist(data[~is_main_pop], histtype="step", color="gold", label="Minor", lw=3, alpha=0.8)
        txt = r"${0:.3f}<r<{1:.3f}$".format(bins[i],bins[i+1])
        ax.text(0.6, 0.85, txt, fontsize=15, transform=ax.transAxes)
        plt.xlabel("Proper Motion (mas)")
        plt.ylabel("# of Stars")
        plt.show()