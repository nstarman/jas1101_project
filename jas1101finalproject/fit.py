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
              scale_max=10, scale_min=0,
              logbeta_max=-2, logbeta_min=-5):
    
    """
    Setup prior transforms for models. 
    
    Parameters
    ----------
    scale_min : min scale
    scale_max : max scale
    
    logbeta_min : min log beta
    logbeta_max : max log beta

    Returns
    ----------
    prior_tf : prior transform function for fitting
    
    """
    
    if kind == 'scale':
        
        def prior_2(u):
            """ Prior Transform FUnction"""
            v = u.copy()

            v[0] = u[0] * 10     
                # M_GC/r_scale [10^5M_sun/pc]

            v[1] = u[1] * scale_max 
                # scale [km/s]

            logbeta_range = logbeta_max - logbeta_min
            v[2] = u[2] * logbeta_range + logbeta_min       
                # log beta

            return v 

        return prior_2


### Log Likelihood ###

def set_likelihood(x, y):
    
    """
    Setup likelihood function.
    
    Parameters
    ----------
    x: 1d array
    y: 1d array
    
    Returns
    ----------
    loglike : log-likelihood function for fitting
    
    """
    
    def loglike_2(v):
        """ likelihood function """
        
        # parameter
        r_scale = 10
        M_gc = v[0] * r_scale * 1e5
        scale = v[1]
        beta = 10**v[2]
        
        # mean value
        ypred = np.mean(y)
        
        # sigma profile
        sigma2 = sigmar_2(x, M_gc, r_scale, beta).to_value((u.km/u.s)**2)

        # scaled sigma profile
        sigma2 = sigma2 / scale**2

        # residual
        residsq = (ypred - y)**2 / sigma2
        
        # log likelihood
        loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * sigma2))

        if not np.isfinite(loglike):
            loglike = -1e100

        return loglike
    
    return loglike_2


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
        
    ## run
    
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
    
    
    ## result
    
    @property
    def results(self):
        res = getattr(self.dsampler, 'results', {})
        return res
    
    def get_params(self, return_sample=False):
        return get_params_fit(self.results, return_sample)
    
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
    
    ## plot
    
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
