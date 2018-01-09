#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def NLL(pars, x):
    mult = 2.0
    ll = mult*(-0.5 * np.log(2.0*np.pi) - np.log(pars[1]) - (x-pars[0])**2/(2.0*pars[1]**2))
    return -ll

def NLL_comb(pars, X, NLL):
    nll_sum = 0
    for xi in X:
        nll = NLL(pars, xi)
        nll_sum += nll
    return nll_sum


def NLL_comb_conditional(par_opt, par_fix, x, NLL):
    pars = (par_opt, par_fix)
    nll_sum = NLL_comb(pars, x, NLL)
    return nll_sum


def plot_NLL(X,Y,Z):
    f,a = plt.subplots()
    
    pcm = a.pcolor(X, Y, Z, cmap='viridis_r', norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
    cb = f.colorbar(pcm, ax=a)
    cb.set_label(r'$-2\ln \mathcal{L}$')
        
    min_index = np.argmin(Z)
    min_value = np.min(Z)
    a.scatter(X.flatten()[min_index], Y.flatten()[min_index], c='k', marker='x')
    
    a.contour(X,Y,Z, [min_value+1.0, min_value+4.0, min_value+9.0])
    
    label = "Min value: {:.2e}".format(min_value)
    a.text(0.2, 0.1, label, transform=a.transAxes)
    return f,a


def MLE(NLL_comb, x0, args, method='Nelder-Mead'):
    m = minimize(fun=NLL_comb, x0=x0, args=args, method=method)
    return m
