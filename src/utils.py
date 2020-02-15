import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def colobar_non_mappable(fig, ax, cmap="magma", vmin=0, vmax=1):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1)   
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=plt.cm.get_cmap(cmap), norm=norm, orientation='vertical')
    cb.set_label("M$_{BH}$")
    fig.add_axes(ax_cb)

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