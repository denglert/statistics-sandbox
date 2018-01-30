#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import minimize
import numpy as np


### --- Negative log-likelihoods (NLL)

def NLL_norm(pars, x):
    """Returns -2*ln*L"""
    mu = pars[0]
    sigma = pars[1]
    ll = -0.5 * np.log(2.0*np.pi) - np.log(sigma) - (x-mu)**2/(2.0*sigma**2)
    return - 2.0 * ll


def NLL(pars, X, NLL_func):
    nll_sum = 0
    for xi in X:
        nll = NLL_func(pars, xi)
        nll_sum += nll
    return nll_sum


def NLL_cond(par_opt, par_fix, X, NLL_func):
    pars = (par_opt, par_fix)
    nll_sum = NLL(pars, X, NLL_func)
    return nll_sum


### --- Maximum likelihood estimators (MLEs)

def MLE(NLL, X, NLL_func, init_values, method='Nelder-Mead'):
    args = (X, NLL_func)
    m = minimize(fun=NLL, x0=init_values, args=args, method=method)
    return m


def MLE_cond(NLL_cond, X, NLL_func, par_fix, init_values, method='Nelder-Mead'):
    args = (par_fix, X, NLL_func)
    m = minimize(fun=NLL_cond, x0=init_values, args=args, method=method)
    return m


### --- Plotting

def plot_NLL(X,Y,Z):

    f,a = plt.subplots()
    
    pcm = a.pcolor(X, Y, Z, cmap='viridis_r', norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
    cb = f.colorbar(pcm, ax=a, extend='max')
    cb.set_label(r'$-2\ln \mathcal{L}$')
        
    min_index = np.argmin(Z)
    min_value = np.min(Z)
    a.scatter(X.flatten()[min_index], Y.flatten()[min_index], c='k', marker='x')
    
    a.contour(X,Y,Z, [min_value+1.0, min_value+4.0, min_value+9.0])
    
    label = "Min value: {:.2e}".format(min_value)
    a.text(0.2, 0.1, label, transform=a.transAxes)
    return f,a
