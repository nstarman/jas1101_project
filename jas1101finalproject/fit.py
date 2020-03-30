import os
import time
import numpy as np

import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as constants

import multiprocess as mp

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


def sigmar_2(x, M_gc, r_scale, beta):
    
    """ Compute sigma^2(r).
    
    Parameters
    ----------
    x: np. array
        normalized radius
    M_gc : float
        total mass of globular cluster
    r_scale: float
        scaled radius of globular cluster
    beta: float (0~1)
        ratio of black hole mass and total mass
    
    Return 
    ----------
    sigma^2 ()
    
    """
    
    sig2 = -(M_gc
            * (
                - 6 * beta
                - 36 * x ** 2 * beta
                - 70 * x ** 4 * beta
                - 56 * x ** 6 * beta
                - 16 * x ** 8 * beta
                + 48 * x ** 3 * np.sqrt(1 + x ** 2) * beta
                + 48 * x ** 5 * np.sqrt(1 + x ** 2) * beta
                + 16 * x ** 7 * np.sqrt(1 + x ** 2) * beta
                + x * np.sqrt(1 + x ** 2) * (-1 + 16 * beta)
            )
        ) / (6.0 * r_scale * x * (1 + x ** 2))  # r_scale in pc

    return (constants.G * u.solMass * sig2 / u.pc).to((u.km/u.s)**2)


### Prior Transform ###

def set_prior(kind = 'scale',
              logbeta_max=-1,
              logbeta_min=-6,
              M_max=20, r_max=30,
              scale_max=25,
              mu_ol_max=25,
              r_min_min=0.3,
              r_max_max=1.5,
              delta_r_min=0.2):
    
    """
    Setup prior transforms for models. 
    
    Parameters
    ----------
    
    logbeta_min : min log beta
    logbeta_max : max log beta
    
    kind : str
    
    'physical' - fit by physical units 
        M_max : max M_GC
        r_max : max r_scale
    
    'scale' - fit scale as a parameter
        scale_max : max scale
        
    'scale-outlier' - fit scale + include outlier
        scale_max : max scale
        mu_ol_max : max outlier mean 
        
    'scale-outlier-range' - fit scale + include outlier + fit x range
        r_min_min : min of r_min
        r_max_max : max of r_max
        delta_r_min : min of delta r


    Returns
    ----------
    prior : prior transform function for fitting
    ndim : number of dimension
    
    """
    
    if kind == 'physical':
        
        def prior_1(u):
            """ Prior Transform Function"""
            v = u.copy()

            v[0] = u[0] * M_max
                # M_GC [10^5 M_sun]

            v[1] = u[1] * r_max
                # r_scale [pc]

            logbeta_range = logbeta_max - logbeta_min
            v[2] = u[2] * logbeta_range + logbeta_min       
                # log beta

            return v 

        return prior_1, 3

    elif kind == 'scale':
        
        def prior_2(u):
            """ Prior Transform Function"""
            v = u.copy()

            v[0] = u[0] * 10     
                # M_GC/r_scale [10^5M_sun/pc]
            v[1] = u[1] * scale_max 
                # scale [km/s]
            logbeta_range = logbeta_max - logbeta_min
            v[2] = u[2] * logbeta_range + logbeta_min       
                # log beta

            return v 

        return prior_2, 3
    
    elif kind == 'scale-outlier':
        
        def prior_3(u):
            v = u.copy()

            v[0] = u[0]               
                # M_GC [10^5 M_sun] / r_scale [pc]
            v[1] = u[1] * scale_max 
                # scale [km/s]
            logbeta_range = logbeta_max - logbeta_min
            v[2] = u[2] * logbeta_range + logbeta_min       
                # log beta
            v[3] = u[3] * 4.7 - 5     
                # log prob outlier
            v[4] = (u[4] - 0.5) * 2 * mu_ol_max     
                # mean outlier
            v[5] = u[5] * 100         
                # variance outlier

            return v 
        
        return prior_3, 6
    
    elif kind == 'scale-outlier-range':
        
        def prior_4(u):
            v = u.copy()

            v[0] = u[0]               
                # M_GC [10^5 M_sun] / r_scale [pc]
            v[1] = u[1] * scale_max 
                # scale [km/s]
            logbeta_range = logbeta_max - logbeta_min
            v[2] = u[2] * logbeta_range + logbeta_min       
                # log beta
            v[3] = u[3] * 4.7 - 5     
                # log prob outlier
            v[4] = (u[4] - 0.5) * 2 * mu_ol_max     
                # mean outlier
            v[5] = u[5] * 100         
                # variance outlier
            v[6] = u[6] * (1 - r_min_min) + r_min_min
                # r_min (r_min_min ~ 1)
            r_max_min = v[6] + delta_r_min
            v[7] = u[7] * (r_max_max - r_max_min) + r_max_min
                # r_max (r_min ~ r_max_max)

            return v 
        
        return prior_4, 8


### Log Likelihood ###

def set_likelihood(x, y, y_err=None, kind='scale'):
    
    """
    Setup likelihood function.
    
    Parameters
    ----------
    x : 1d array
    y : 1d array
    y_err : error of y, 1d array
    kind : str
        'physical' - fit by physical units 
        'scale' - fit scale as a parameter
        'scale-outlier' - fit scale + include outlier
        'scale-outlier-range' - fit scale + include outlier + fit x range
    
    Returns
    ----------
    loglike : log-likelihood function for fitting
    
    """
    
    
    if y_err is None:
        y_err = np.zeros_like(y)
        
    x0 = x.copy()
    y0 = y.copy()
    y_err0 = y_err.copy()
    
    if kind == 'physical':
        
        def loglike_1(v):
            
            # parameter
            M_gc, r_scale, logbeta = v

            M_gc = M_gc * 1e5

            beta = 10**logbeta

            # mean value
            ypred = np.mean(y)
            
            # sigma profile in km/s
            sigma2_phy = sigmar_2(x, M_gc, r_scale, beta).to_value((u.km/u.s)**2)

            # normalized by sigma at r = 1
            sigma2_n = sigmar_2(1, M_gc, r_scale, beta).to_value((u.km/u.s)**2)
            
            # scaled sigma profile
            sigma2 = sigma2_phy / sigma2_n

            # total sigma2 with scaled error
            sigma2_tot = sigma2 * (1 + y_err / y)
            
            # residual
            residsq = (ypred - y)**2 / sigma2_tot
            
            # log likelihood
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma2_tot))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike_1

    elif kind == 'scale':
        
        def loglike_2(v):
            """ likelihood function """

            # parameter
            r_scale = 10
            M_gc = v[0] * r_scale * 1e5
            scale = v[1]
            beta = 10**v[2]

            # mean value
            ypred = np.mean(y)

            # sigma profile in km/s
            sigma2_phy = sigmar_2(x, M_gc, r_scale, beta).to_value((u.km/u.s)**2)

            # scaled sigma profile
            sigma2 = sigma2_phy / scale**2
            
            # total sigma2 with scaled error
            sigma2_tot = sigma2 * (1 + y_err / y)

            # residual
            residsq = (ypred - y)**2 / sigma2_tot

            # log likelihood
            loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma2_tot))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike

        return loglike_2
    
    elif kind == 'scale-outlier':
        
        def loglike_3(v):
            """ likelihood function """

            # parameter
            r_scale = 10
            M_gc = v[0] * r_scale * 1e5
            scale = v[1]
            beta = 10**v[2]
            p_ol = 10**v[3]
            y_ol = v[4]
            var_ol = v[5]

            # mean value
            ypred = np.mean(y)

            # sigma profile in km/s
            sigma2_phy = sigmar_2(x, M_gc, r_scale, beta).to_value((u.km/u.s)**2)

            # scaled sigma profile
            sigma2 = sigma2_phy / scale**2
            
            # total sigma2 with scaled error
            sigma2_tot = sigma2 * (1 + y_err / y)

            # residual
            residsq = (ypred - y)**2 / sigma2_tot
            
            # total sigma2 for outlier
            sigma2_tot_ol = sigma2_tot + var_ol
            
            # residual for outlier
            residsq_ol = (y_ol - y)**2 / sigma2_tot_ol
            
            # foreground & background log prob
            logp_fg = np.log(1-p_ol) - 0.5 * np.log(2 * np.pi * sigma2) - 0.5 * residsq
            logp_bg = np.log(p_ol) - 0.5 * np.log(2 * np.pi * sigma2_tot_ol) - 0.5 * residsq_ol

            # log likelihood
            loglike = np.sum(np.logaddexp(logp_fg, logp_bg))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike

        return loglike_3
    
    elif kind == 'scale-outlier-range':
        
        def loglike_4(v):
            """ likelihood function """

            # parameter
            r_scale = 10
            M_gc = v[0] * r_scale * 1e5
            scale = v[1]
            beta = 10**v[2]
            p_ol = 10**v[3]
            y_ol = v[4]
            var_ol = v[5]
            x_min = v[6]
            x_max = v[7]
            
            # clip data in x range
            use_x = (x0>=x_min) & (x0<=x_max)
            x = x0[use_x]
            y = y0[use_x]
            y_err = y_err0[use_x]

            # reject given too few stars in the range
            if len(y)<100:
                loglike = -1e100
                
            ypred = np.nanmean(y)

            # sigma profile in km/s
            sigma2_phy = sigmar_2(x, M_gc, r_scale, beta).to_value((u.km/u.s)**2)

            # scaled sigma profile
            sigma2 = sigma2_phy / scale**2
            
            # total sigma2 with scaled error
            sigma2_tot = sigma2 * (1 + y_err / y)

            # residual
            residsq = (ypred - y)**2 / sigma2_tot
            
            # total sigma2 for outlier
            sigma2_tot_ol = sigma2_tot + var_ol
            
            # residual for outlier
            residsq_ol = (y_ol - y)**2 / sigma2_tot_ol
            
            # foreground & background log prob
            logp_fg = np.log(1-p_ol) - 0.5 * np.log(2 * np.pi * sigma2) - 0.5 * residsq
            logp_bg = np.log(p_ol) - 0.5 * np.log(2 * np.pi * sigma2_tot_ol) - 0.5 * residsq_ol

            # log likelihood
            loglike = np.sum(np.logaddexp(logp_fg, logp_bg))

            if not np.isfinite(loglike):
                loglike = -1e100

            return loglike

        return loglike_4

    
class DynamicNestedSampler:
    
    """ 
    A wrapped dynesty sampler.
    
    Parameters
    ----------
    loglikelihood: function
        log likelihood function
    prior_transform: function
        priot transorm function
    ndim: int
        number of dimension
    
    """
    
    def __init__(self, loglikelihood,
                 prior_transform, ndim,
                 sample='auto', bound='multi',
                 n_cpu=None, n_thread=None):
        
        if n_cpu is None:
            n_cpu = mp.cpu_count()
            
        if n_thread is not None:
            n_thread = max(n_thread, n_cpu-1)
        
        if n_cpu > 1:
            self.open_pool(n_cpu)
            self.use_pool = {'update_bound': False}
        else:
            self.pool = None
            self.use_pool = None
                
        self.prior_tf = prior_transform
        self.loglike = loglikelihood
        self.ndim = ndim
        
        dsampler = dynesty.DynamicNestedSampler(self.loglike,
                                                self.prior_tf,
                                                self.ndim,
                                                sample=sample,
                                                bound=bound,
                                                pool=self.pool,
                                                queue_size=n_thread,
                                                use_pool=self.use_pool)
        self.dsampler = dsampler
        
    ### run ###
    
    def run_fitting(self,
                    nlive_init=100,
                    maxiter=10000,
                    nlive_batch=50,
                    maxbatch=2,
                    wt_kwargs={'pfrac': 0.8},
                    close_pool=True,
                    print_progress=True):
    
        print("Run Nested Fitting for the image... Dim of params: %d"%self.ndim)
        start = time.time()
   
        dlogz = 1e-3 * (nlive_init - 1) + 0.01
        
        self.dsampler.run_nested(nlive_init=nlive_init, 
                                 nlive_batch=nlive_batch, 
                                 maxbatch=maxbatch,
                                 maxiter=maxiter,
                                 dlogz_init=dlogz, 
                                 wt_kwargs=wt_kwargs,
                                 print_progress=print_progress) 
        
        end = time.time()
        self.run_time = (end-start)
        
        print("\nFinish Fitting! Total time elapsed: %.3g s"%self.run_time)
        
        if (self.pool is not None) & close_pool:
            self.close_pool()
        
    def open_pool(self, n_cpu):
        print("\nOpening new pool: # of CPU used: %d"%(n_cpu - 1))
        self.pool = mp.Pool(processes=n_cpu - 1)
        self.pool.size = n_cpu - 1
    
    def close_pool(self):
        print("\nPool Closed.")
        self.pool.close()
        self.pool.join()
    
    
    ### read/save/load result ###
    
    @property
    def results(self):
        res = getattr(self.dsampler, 'results', {})
        return res
    
    def get_params(self, return_sample=False):
        """
        Get parameters meadian/mean/covariance
        return samples if samples_eq is True.
        """
        
        output = get_params_fit(self.results, return_sample)
            
        return output
    
    
    def save_results(self, filename, fit_info=None, save_dir='.'):
        res = {}
        if fit_info is not None:
            for key, val in fit_info.items():
                res[key] = val

        res['run_time'] = self.run_time
        res['fit_res'] = self.results
        
        fname = os.path.join(save_dir, filename)
        save_nested_fitting_result(res, fname)
        
        self.res = res
    
    
    ### plot ###
    
    def cornerplot(self, truths=None, labels=None, figsize=(12,10)):
        
        fig, axes = plt.subplots(self.ndim, self.ndim,
                                 figsize=figsize)
        
        dyplot.cornerplot(self.results, truths=truths, labels=labels, 
                          color="royalblue", truth_color="indianred",
                          title_kwargs={'fontsize':18, 'y': 1.04},
                          label_kwargs={'fontsize':16}, show_titles=True,
                          fig=(fig, axes))
        
    def cornerbound(self, figsize=(10,10), labels=None):
        
        fig, axes = plt.subplots(self.ndim-1, self.ndim-1,
                                 figsize=figsize)
        
        fg, ax = dyplot.cornerbound(self.results, it=1000, labels=labels,
                                    prior_transform=self.prior_tf,
                                    show_live=True, fig=(fig, axes))

            
def Run_Dynamic_Nested_Fitting(loglikelihood,
                               prior_transform, ndim,
                               nlive_init=100, sample='auto', 
                               nlive_batch=50, maxbatch=2,
                               pfrac=0.8, n_cpu=None,
                               print_progress=True):
    
    """ Run Fitting as a Function.
    
    Parameters
    ----------
    loglikelihood: function
        log likelihood function
    prior_transform: function
        priot transorm function
    ndim: int
        number of dimension
        
    """
    
    print("Run Nested Fitting for the image... #a of params: %d"%ndim)
    
    start = time.time()
    
    if n_cpu is None:
        n_cpu = mp.cpu_count()-1
        
    with mp.Pool(processes=n_cpu) as pool:
        print("Opening pool: # of CPU used: %d"%(n_cpu))
        pool.size = n_cpu

        dlogz = 1e-3 * (nlive_init - 1) + 0.01

        pdsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim,
                                                 sample=sample, pool=pool,
                                                 use_pool={'update_bound': False})
        pdsampler.run_nested(nlive_init=nlive_init, 
                             nlive_batch=nlive_batch, 
                             maxbatch=maxbatch,
                             print_progress=print_progress, 
                             dlogz_init=dlogz, 
                             wt_kwargs={'pfrac': pfrac})
        
    end = time.time()
    print("Finish Fitting! Total time elapsed: %.3gs"%(end-start))
    
    return pdsampler


def get_params_fit(results, return_sample=False):
    """ Get median, mean, covairance (and samples) from fitting results """
    samples = results.samples                                 # samples
    weights = np.exp(results.logwt - results.logz[-1])        # normalized weights 
    pmean, pcov = dyfunc.mean_and_cov(samples, weights)       # weighted mean and covariance
    samples_eq = dyfunc.resample_equal(samples, weights)      # resample weighted samples
    pmed = np.median(samples_eq,axis=0)
    
    if return_sample:
        return pmed, pmean, pcov, samples_eq
    else:
        return pmed, pmean, pcov

def save_nested_fitting_result(res, filename='fit.res'):
    import dill
    with open(filename,'wb') as file:
        dill.dump(res, file)
        
def load_nested_fitting_result(filename='fit.res'):        
    import dill
    with open(filename, "rb") as file:
        res = dill.load(file)
    return res
