"""

A program replicating "Main_Yog_priceonly.m" in Python

Version: Python 2.7

"""

from __future__ import division  # Necessary only in Python 2
import os

import numpy as np
import pandas as pd
import scipy.optimize as opt

home = os.environ['HOME']
root = home + '/Dropbox/Instruction/Scraping/maria_ana/for_class'


# ---- Function Definition ---- #

def loglike(param, summation=True):
    """
    Given param, calculates the log likelihood of the data

    Parameters
    ----------
    param : numpy 2d array
    summation : Boolean
        If True, the function returns the log likelihood
        If False, the function returns individual probabilities in a numpy array
        that sum to the log likelihood
    """
    e_y = np.exp(yop.dot(param[[0, 3]]))
    e_d = np.exp(dan.dot(param[[1, 3]]))
    e_h = np.exp(hil.dot(param[[2, 3]]))
    e_w = np.exp(wwt.dot(param[[3]]))

    den = e_y + e_d + e_h + e_w
    py = e_y / den
    pd = e_d / den
    ph = e_h / den
    pw = e_w / den
    p = np.c_[py, pd, ph, pw]
    selbmat = choi * p
    selb = selbmat.sum(axis=1)  # Sum collapsing the second dimension (columns)

    lselb = np.log(selb)

    if not summation:
        return lselb
    else:
        lpr = -lselb.sum()
    return lpr


def serrors_basic_logit(param):
    """
    Given param, calculates standard errors

    Parameters
    ----------
    param : numpy 2d array

    Returns
    -------
    serrors : numpy 2d array

    """
    no_params = param.shape[0]
    H = np.zeros((no_params, no_params))
    di = np.zeros((n, no_params))

    for l in xrange(no_params):
        bj = param.copy()
        bj2 = param.copy()
        bj[l] = .05 + param[l]
        di[:, l] = (loglike(bj2, summation=False) - loglike(bj, summation=False)) / .05

    for i in xrange(n):
        H += di[[i], :].T.dot(di[[i], :])

    serrors = np.sqrt(np.diag(np.linalg.inv(H)))

    return serrors


# ---- Data Extraction ---- #

yogurt = np.loadtxt(root + '/yogurt.txt')

pan = yogurt[:, 0]  # Id Number of Panelists
price = yogurt[:, 14:18]  # Prices for the 4 brands
choi = yogurt[:, 6:10]  # Brand purchase information
n = yogurt.shape[0]  # Shape of the matrix along the first dimension; Equivalent to len(yogurt)
o = np.ones((n, 1))  # Create n by 1 matrix of ones; the first argument of the function is the shape

yop = np.c_[o, price[:, 0]]  # Concatenate columns
dan = np.concatenate((o, price[:, [1]]), axis=1)  # Concatenation along the specified axis (here add a column)
hil = np.c_[o, price[:, 2]]
wwt = price[:, [3]]


# ---- Estimate Model ---- #
X = -np.ones((4, 1))  # Initial values for parameters

bfgs_options = {'maxiter': 100000,
                'maxfun': 100000,
                'ftol': 1e-5,
                'disp': False}

results = opt.minimize(loglike, X, method='L-BFGS-B', options=bfgs_options)
xfinal = results['x']  # The results object is a dictionary of various outputs

serrors = serrors_basic_logit(xfinal)
tstats = xfinal / serrors


# ---- Export Results ---- #

# Prints to active terminal
np_export = np.c_[xfinal, serrors, tstats]
print(np_export)

# Alternative using pandas to easily export to a csv
pd_export = pd.DataFrame(np_export, columns=['coefficients', 'standard_errors', 't_stat'],
                         index=['param_1', 'param_2', 'param_3', 'param_4'])
pd_export.to_csv(root + '/yogurt_params.csv', index=True)
