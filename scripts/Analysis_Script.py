import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import ks_2samp
import sklearn
from sklearn.neighbors import KernelDensity

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

    def bootstrap(self, conf_level, n=10000):
        '''
        We will use this to find a conf_level % confidence interval for the posterior mean
        confidence level is between 0 and 1
        n is the number of bootstrap samples
        :param self:
        :type self:
        :return:
        :rtype:
        '''
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


# you can run, for example:
# prior = np.random.uniform(low=-10,high=10, size=1000) # generate random uniform
# posterior = np.random.normal(0, 5, size=1000) # generate random normal
# obj = Stats_output(prior,posterior)
# obj.posterior
# obj.prior
# obj.bootstrap(0.95)
# obj.bootstrap(0.95, 1000)
# obj.lbound
# obj.ubound
# obj.ubound
# obj.plots()
# obj.comparisons()
