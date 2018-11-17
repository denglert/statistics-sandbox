import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def calc_Sxx(x):
    xbar = np.mean(x)
    Sxx = np.sum( (x - xbar)*x)
    return Sxx


def calc_Sxy(x,y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    Sxy = np.sum( (x-xbar) * (y-ybar) )
    return Sxy


def calc_beta0hat(x, y):
    ybar = np.mean(y)
    xbar = np.mean(x)
    beta1hat = calc_beta1hat(x,y)
    beta0hat = ybar - xbar * beta1hat
    return beta0hat


def calc_beta1hat(x,y):
    Sxx = calc_Sxx(x)
    Sxy = calc_Sxy(x,y)
    beta1hat = Sxy/Sxx
    return beta1hat


def calc_beta0hat_bias(x, epsilon_means):
    Sxx = calc_Sxx(x)
    beta0hat_bias = np.mean(epsilon_means) - np.sum( np.mean(x) * (x - np.mean(x)) * epsilon_means)/Sxx
    return beta0hat_bias


def calc_beta1hat_bias(x, epsilon_means):
    Sxx = calc_Sxx(x)
    beta1hat_bias = np.sum( (x - np.mean(x))*epsilon_means )/Sxx
    return beta1hat_bias


def calc_beta0hat_variance(x, epsilon_variances, eps):
    Sxx = calc_Sxx(x)
    xbar = np.mean(x)
    n = len(x)
    beta0hat_variance = np.sum( epsilon_variances/n**2 +
                                (xbar/(Sxx**2))*np.sum( (x-xbar)**2 * epsilon_variances) +
                                - 2 
                              )
    return beta0hat_variance



def calc_beta1hat_variance(x, epsilon_variances):
    Sxx = calc_Sxx(x)
    xbar = np.mean(x)
    beta1hat_variance = np.sum( (x-xbar)**2 * epsilon_variances )/(Sxx**2)
    return beta1hat_variance


def generate_experiment(xrange, npts, epsilon_means, epsilon_sigmas, beta0_t, beta1_t):
    x = np.linspace(xrange[0], xrange[1], npts)
    epsilon = scipy.stats.norm.rvs(loc=epsilon_means, scale=epsilon_sigmas, size=npts)
    y_t = beta0_t + beta1_t*x
    y = y_t + epsilon
    beta0hat = calc_beta0hat(x,y)
    beta1hat = calc_beta1hat(x,y)
    return x, y_t, y, epsilon, beta0hat, beta1hat


def dashboard(beta0_t, beta1_t, epsilon_means, experiments, figsize=(12,16)):
    
    x   = experiments[0][0]
    y_t = experiments[0][1]
    
    beta0hats = np.asarray(experiments[:,4], dtype=np.float32)
    beta1hats = np.asarray(experiments[:,5], dtype=np.float32)

    beta0hat_bias = calc_beta0hat_bias(x, epsilon_means)
    beta1hat_bias = calc_beta1hat_bias(x, epsilon_means)
    
    beta0hat_mean = np.mean(beta0hats)
    
    beta1hat_mean = np.mean(beta1hats)
    beta1hat_sample_variance = np.var(beta1hats, ddof=1)
    

#   fig = plt.subplots(figsize=figsize, squeeze=False)
    fig,axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    ax = axes.flatten()

#   ax1 = plt.subplot2grid( (2,2), (0,0) )
#   ax2 = plt.subplot2grid( (2,2), (0,1) )
#   ax3 = plt.subplot2grid( (2,2), (1,0) )
#   ax4 = plt.subplot2grid( (2,2), (1,1) )
#   ax = [ax1, ax2, ax3, ax4]

    fig.subplots_adjust(hspace=0.3, bottom=0.3)

    # - Axis 0
    ax[0].plot(x, y_t, linestyle='--', c='k', label=r'$\beta_{0,t} + \beta_{1,t} x$ (true model)')
    ax[0].scatter(x,experiments[0][2], c='C0', rasterized=True, label='Observed samples')

    for experiment in experiments:
        x = experiment[0]
        y = experiment[2]
        ax[0].scatter(x,y, alpha=0.05, c='C0', rasterized=True)
        
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_xlim(-1.0, 11.0)
    ax[0].set_ylim(0.0, 20)
    ax[0].legend()

    
    # - Axis 1
    ax[1].hist(beta0hats,bins=50)
    ax[1].set_xlabel(r"$\hat{\beta}_{0}$")
    label_beta0hat_mean = r"Mean of $\hat{{\beta}}_{{0}}: {:.2f}$".format(beta0hat_mean)
    label_beta0_t = r"True $\beta_{{0}}$: {:.2f}".format(beta0_t)
    ax[1].axvline(beta0hat_mean, linestyle='--', c='navy', label=label_beta0hat_mean)
    ax[1].axvline(beta0_t, linestyle='-', c='k', label=label_beta0_t)
    ax[1].axvline(beta0_t+beta0hat_bias, c='firebrick', label=r'Expected $\hat{\beta}_{0}$')
    exp_eq_latex = r"$E[\hat{\beta}_{0}] = \beta^{t}_{0} + \frac{1}{n} \sum_{i=1}^{n} E[\epsilon_{i}]  - \frac{\bar{x}}{S_{xx}} \sum_{i=1}^{n} (x_{i} - \bar{x}) E[\epsilon_{i}]$"
    ax[1].text(0.5, -0.25, exp_eq_latex, transform=ax[1].transAxes, horizontalalignment='center')
    ax[1].legend()

    # - Axis 2
    ax[2].hist(beta1hats, bins=50)
    ax[2].set_xlabel(r"$\hat{\beta}_{1}$")
    label_beta1hat_mean = r"Mean of $\hat{{\beta}}_{{1}}: {:.2f}$".format(beta1hat_mean)
    label_beta1_t = r"True $\beta_{{1}}$: {:.2f}".format(beta1_t)
    ax[2].axvline(beta1hat_mean, linestyle='--', c='navy', label=label_beta1hat_mean)
    ax[2].axvline(beta1_t, linestyle='-', c='k', label=label_beta1_t)
    ax[2].axvline(beta1_t+beta1hat_bias, c='firebrick', label=r'Expected $\hat{\beta}_{1}$')
    exp_eq_latex = r"$E[\hat{\beta}_{1}] = \beta^{t}_{1} + \frac{1}{S_{xx}} \sum_{i=1}^{n} (x_{i} - \bar{x}) E[\epsilon_{i}]$"
    ax[2].text(0.5, -0.25, exp_eq_latex, transform=ax[2].transAxes, horizontalalignment='center')
    ax[2].legend()
    
    # - Axis 3
    ax[3].scatter(beta0hats, beta1hats, c='C0', rasterized=True)
    ax[3].set_xlabel(r'$\hat{\beta}_{0}$')
    ax[3].set_ylabel(r'$\hat{\beta}_{1}$')
    ax[3].scatter(beta0hat_mean, beta1hat_mean, c='k', rasterized=True, label='Mean')
    ax[3].legend()
    
    return fig, ax
