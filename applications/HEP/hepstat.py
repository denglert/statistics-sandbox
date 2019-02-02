#!/usr/bin/env python

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.optimize
from parameter_estimation import confidence_interval as ci

def calc_CI_bounds(n_obs, bkg, alpha=0.05):

    lambda_CI_LL_twosided, lambda_CI_UL_twosided = ci.poisson_two_sided(n_obs, alpha=alpha)
    lambda_CI_UL_onesided = ci.poisson_upper(n_obs, alpha=alpha)
    lambda_CI_UL_onesided_CLs = ci.poisson_upper_CLs(n_obs, bkg, alpha=alpha)

    return lambda_CI_LL_twosided, lambda_CI_UL_twosided, lambda_CI_UL_onesided, lambda_CI_UL_onesided_CLs


def show_confidence_intervals_with_CLs(experiments, theta_true, xmin_inf = -1e10, figsize=(10,8)):

    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    ax = axes.flatten()

    yshift = 0.4

    for i, experiment in enumerate(experiments):
        n = experiment[0]
        ll_2s = experiment[1]
        ul_2s = experiment[2]
        ul_1s = experiment[3]
        ul_1s_CLs = experiment[4]
        ax[0].scatter(n, 2*i, c='navy',)
        ax[1].scatter(n, 2*i, c='navy',)
        ax[1].scatter(n, 2*i+yshift, c='navy',)
        ax[0].hlines(2*i, xmin=ll_2s, xmax=ul_2s, colors='r')
        ax[1].hlines(2*i, xmin=xmin_inf, xmax=ul_1s, colors='purple')
        ax[1].hlines(2*i+yshift, xmin=xmin_inf, xmax=ul_1s_CLs, colors='forestgreen')

    ax[0].axvline(theta_true, linestyle='--', label=r"$\theta_{true}$", c='k')
    ax[1].axvline(theta_true, linestyle='--', label=r"$\theta_{true}$", c='k')
    
    ax[0].scatter(experiments[0,0], 0, c='navy', label='$n_{obs}$')
    ax[1].scatter(experiments[0,0], 0, c='navy', label='$n_{obs}$')
    
    ax[0].hlines(9, xmin=experiments[0,1], xmax=experiments[0,2], colors='r', label='Two-sided CI')
    ax[1].hlines(0, xmin=xmin_inf, xmax=experiments[0,3], colors='purple', label='One-sided, upper bound CI')
    ax[1].hlines(yshift, xmin=xmin_inf, xmax=experiments[0,4], colors='forestgreen', label=r'One-sided, upper bound CI from $CL_{s}$')

    ax[0].legend()
    ax[1].legend(loc='upper left',  bbox_to_anchor=(1,1))
    
    ax[0].set_xlim(0.0, 30.0)
    ax[1].set_xlim(0.0, 30.0)

    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[0].set_title(r"Two-sided confidence interval")
    ax[1].set_title(r"One-sided, upper bounded confidence interval")
    
    return fig, ax



#############################################################################

class SimplePoissonModel:

    def __init__(self, mu, b, s0=1.0):
        self.b = b
        self.s0 = s0
        self.mu = mu
        self.poisson_lambda = b + s0*mu

    def set_mu(self, mu):
        self.mu = mu
        self.poisson_lambda = self.b + self.s0*mu

    def loglikelihood(self, n, mu=None):
        if mu is None:
            mu = self.mu
        poisson_lambda = self.b + self.s0*mu
        logpmf = np.sum(scipy.stats.poisson.logpmf(k=n, mu=poisson_lambda))
        return logpmf

    def loglikelihoodmax(self, n):
        logpmf = np.sum(scipy.stats.poisson.logpmf(k=n, mu=n))
        return logpmf

    def calc_muhat(self, n):
        muhat = (n - self.b)/self.s0
        return muhat

    def calc_tmu(self, n, mu=None):
        if mu is None:
            mu = self.mu
        ll = self.loglikelihood(n, mu=mu)
        llmax = self.loglikelihoodmax(n)
        tmu = -2.0 * (ll - llmax)
        return tmu 
    
    def calc_ttildemu(self, n, mu=None):
        if mu is None:
            mu = self.mu
        muhat = self.calc_muhat(n)
        ll = self.loglikelihood(n, mu=mu)
        llmax = self.loglikelihoodmax(n)
        llmu0 = self.loglikelihood(n, mu=0.0)

        if muhat < 0:
            ttildemu = -2.0 * (ll - llmu0)
        else:
            ttildemu = -2.0 * (ll - llmax)
    
        return ttildemu

    def calc_q0(self, n):
        muhat = self.calc_muhat(n)
        ll = self.loglikelihood(n, mu=0)
        llmax = self.loglikelihoodmax(n)
        llmu0 = self.loglikelihood(n, mu=0.0)

        if muhat < 0:
            ttildemu = 0
        else:
            ttildemu = -2.0 * (ll - llmax)
    
        return ttildemu

    def calc_qmu(self, n, mu=None):

        if mu is None:
            mu = self.mu

        muhat = self.calc_muhat(n)
        ll = self.loglikelihood(n, mu=mu)
        llmax = self.loglikelihoodmax(n)

        if muhat > mu:
            qmu = 0
        else:
            qmu = -2.0 * (ll - llmax)
    
        return qmu

    def calc_gmu(self, n, mu=None):

        if mu is None:
            mu = self.mu

        muhat = self.calc_muhat(n)
        lambda_hypo = mu*self.s0 + self.b
        var_muhat = lambda_hypo/(self.s0**2)

        gmu = (muhat-mu)**2.0/var_muhat
        return gmu


    def calc_qtildemu(self, n, mu=None):

        if mu is None:
            mu = self.mu

        muhat = self.calc_muhat(n)
        ll = self.loglikelihood(n, mu=mu)
        llmax = self.loglikelihoodmax(n)
        llmu0 = self.loglikelihood(n, mu=0.0)

        if muhat <= 0:
            qtildemu = -2.0 * (ll - llmu0)
        elif muhat > 0 and muhat < mu:
            qtildemu = -2.0 * (ll - llmax)
        else:
            qtildemu = 0
    
        return qtildemu
