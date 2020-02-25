import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import ks_2samp
import sklearn
from sklearn.neighbors import KernelDensity
# import statsmodels
# from statsmodels.distributions.empirical_distribution import ECDF

class Stats_output:
    '''
    This is a script to analyse two different vectors, a prior and a posterior
    '''

    def __init__(self,prior, posterior):
        self.prior=prior
        self.posterior=posterior


    def plots(self):
        # df = pd.DataFrame({'prior': self.prior, 'posterior': posterior})
        #df.plot('self.prior', 'posterior', kind='scatter')
        #df.plot('self.prior', 'posterior', kind='line')

        df1 = pd.DataFrame({'prior': self.prior})
        df1['Name'] = 'prior'
        df1 = df1.rename(columns={"prior": "value"})
        df2 = pd.DataFrame({'posterior': self.posterior})
        df2['Name'] = 'posterior'
        df2 = df2.rename(columns={"posterior": "value"})

        # kernel density plots, overlay
        frames = [df1,df2]
        dflong = pd.concat(frames)
        dflong.groupby(['Name']).value.plot.kde()
        plt.legend()

        plt.figure(0)  # Here's the part I need
        # histograms, overlay
        dflong.groupby(['Name']).value.plot.hist(alpha=0.3)
        plt.legend()

    def bootstrap(self, conf_level):
        '''
        We will use this to find a conf_level % confidence interval for the posterior mean
        confidence level is between 0 and 1
        :param self:
        :type self:
        :return:
        :rtype:
        '''
        n = 10000 # n is the number of bootstrap samples
        sample_means = [0]*n
        for i in range(n):
            x = np.random.choice(self.posterior, size=n, replace=True, p=None)
            sample_means[i] = np.mean(x)
        lbound = np.quantile(x,(1-conf_level)/2) # lower bound of the confidence interval
        ubound = np.quantile(x,1 - (1-conf_level)/2)# upper bound of the confidence interval
        self.lbound= lbound
        self.ubound = ubound

    def comparisons(self):
        '''
        This function calculates the KL divergence between the self.prior and the posterior distribution
        using scipy's entropy function,
        Person's correlatino coefficient,
        2 sample KS divergence test # null hypothesis is that the two distributions are the same
        :param self:
        :type self:
        :return:
        :rtype:
        '''
        KL = scipy.stats.entropy(self.prior, self.posterior)
        corr = scipy.stats.pearsonr(self.prior,self.posterior) #  correlation, pvalue
        KS = ks_2samp(self.prior,self.posterior)
        self.KL = KL
        self.corr = corr[0]
        self.corr_pval = corr[1] # null: correlation is 0
        self.KS_stat = KS.statistic
        self.KS_pval = KS.pvalue

    def ecdf_survive(self, which = "posterior"):
        if which =="posterior":
            vals = self.posterior
        else:
            vals = self.prior
        # ecdf, empirical CDF
        ecdf = ECDF(vals)
        plt.plot(ecdf.x, ecdf.y)
        # empirical survival function
        plt.plot(0)
        plt.plot(ecdf.x, 1-ecdf.y)

# you can run, for example:
# prior = np.random.uniform(low=-10,high=10, size=1000) # generate random uniform
# posterior = np.random.normal(0, 5, size=1000) # generate random normal
# obj = Stats_output(prior,posterior)
# obj.posterior
# obj.prior
# obj.bootstrap(0.95)
# obj.lbound
# obj.ubound
# obj.ubound
# obj.plots()
# obj.comparisons()
# obj.ecdf_survive_post()
# obj.ecdf_survive_post(which="prior")

# --------------------------------------------------------------------------
def get_mean_cov(X, robust=True):
    from astropy.stats.biweight import biweight_midcovariance, biweight_location

    if robust:
        # Robust
        mean_tot = biweight_location(X, axis=0)
        cov_tot = biweight_midcovariance(X.T)
    else:
        # Classical
        mean_tot = np.mean(X, axis=0)
        cov_tot = np.cov(X)
        
    return mean_tot, cov_tot
    
def confidence_ellipse(X, robust=True, verbose=True):
    """Confidence Ellipse describing 2D data"""
    # Robust confidence ellipse
    mean_tot, cov_tot = get_mean_cov(X, robust=robust)
    
    # Eigen value and norm to determine the ellipse shape
    v_tot, w_tot = np.linalg.eigh(cov_tot)
    u_tot = w_tot[0] / np.linalg.norm(w_tot[0])
    angle_tot = np.arctan(u_tot[1] / u_tot[0]) * 180 / np.pi  # convert to degrees
    
    if verbose:
        print("Robust Sample Ellipse:")
        print("Mean: ", *np.around(mean_tot,3))
        print("Semi-axis Length: ", *np.around(v_tot,3))
        print("Angle (deg): ", np.around(angle_tot,3))
    
    return mean_tot, v_tot, angle_tot

def HotellingT2(X, robust=True):
    """Hotelling's T^2 statistics for X [N_samples, N_dimensions]"""
    
    mean_tot, cov_tot = get_mean_cov(X, robust=robust)
    
    return [np.linalg.multi_dot([X[i]-mean_tot,
                                 np.linalg.inv(cov_tot),
                                 (X[i]-mean_tot).T])
            for i in range(X.shape[0])]

def HotellingT2Test(X, robust=True, plot=True):
    """Testing whether the observation is abnormal using Hotellig T^2 at confidence level a = 0.01"""
    
    T2 = HotellingT2(X, robust=robust)
    
    if plot:
        sns.distplot(np.log10(T2), color="plum")
        plt.axvline(np.log10(stats.chi2(df=2).ppf(0.99)),color="k")
        plt.axvline(np.log10(stats.chi2(df=2).ppf(0.95)),color="k",ls="--")
        plt.xlabel(r"$\log\,T^2$")
    
    normal = T2 < stats.chi2(df=X.shape[1]).ppf(0.99)
        
    return normal
