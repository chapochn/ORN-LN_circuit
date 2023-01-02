#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/2

@author: Nikolai M Chapochnikov

This file creates plots to better understand the behavior of the cdf -
the cumulative density function, called
relative cumulative frequency in the paper.
Plots are not saved on the disk and need to be saved by hand
"""

#%%

from plots_import import *

import params.act3 as par_act3
#%%
DATASET = 3

# there might be a cleaner way to write this...
par_act = None
exec(f'par_act = par_act{DATASET}')

cell_type = 'ORN'
# cell_type = 'PN'

strms = [0, 1, 2, 3]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = False
plot_plots = False

CONC = 'all'
# CONC = '8'
act_sel_ks = [CONC]
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'


# act_pps_k1 = 'tn'  # k like key
act_pps_k1 = 'raw'
# act_pps_key1 = 'model'

# act_pps_k2 = 'raw'
act_pps_k2 = 'mean'
# act_pps_k2 = 'l2'

path_plots = None

odor_sel = None

ACT_PPS = 'o' # this is the preprocessing that we will use on the
# activity data, other options are 'o', 'c', 'n', 'cn'

ORNA = FO.NeurActConAnalysis(DATASET, cell_type, strms, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=odor_sel,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, reduce=True,
                             subfolder=None)


# %%
# Understanding the functional form of the distribution

from scipy.optimize import curve_fit
from scipy.stats import norm

# n_d = 50 # number of dimensions of the vectors
n = 500000
# n = 50

f, axx = plt.subplots(1, 3, figsize=(6,2))
my_bins = np.linspace(-1, 1, 2001, endpoint=True)
for i, n_d in enumerate([10, 21, 50]):
    X1 = FG.get_ctr_norm_np(np.random.randn(n_d, n))
    X2 = FG.get_ctr_norm_np(np.random.randn(n_d, n))
    corrs = np.sum(X1 * X2, axis=0)
    # m = np.mean(corrs)
    # s = np.std(corrs)
    # print(m, s)
    ax = axx[i]
    y, bins, _ = ax.hist(corrs, bins=my_bins, density=True, cumulative=True,
             label='corr. coefs from\nnumerical simulation',
                         histtype='step', lw=2)
    dbin = bins[1] - bins[0]
    mu, sigma = curve_fit(norm.cdf, bins[:-1] + dbin/2, y, p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins, norm.cdf(bins, mu, sigma), label='Gaussian fit')
    # x = np.linspace(-1,1, 100)
    # ax.plot(x, 1/(s*np.sqrt(2*np.pi))*np.exp(-(x/(s*np.sqrt(2)))**2),
    #          label='gaussian')
    ax.set_xlim(-1,1)
    ax.set_xlabel('correlation coefficient')
    ax.set_title(f'in {n_d} dimensions')
    plt.legend(loc='upper left', bbox_to_anchor=[0.02, 1])
axx[0].set_ylabel('probability density')
plt.tight_layout()
plt.show()

#%%

# Here, one should probably choose a specific w,
# so that one shuffles only one vector and takes a gaussian vector on the
# other side.
# we will do this for the 4 LN type vectors.


con_sel_cn = ORNA.con_strms2[STRM]['cn'].loc[:].copy()
con_sel_cn = con_sel_cn.dropna(axis=1)


# Let's first extract the 4 vectors:
n_d = 21 # number of dimensions of the vectors
n = 500000
LNs = ['Broad T M M', 'Broad D M M', 'Keystone M M', 'Picky 0 [dend] M']
titles = ['Broad Trio', 'Broad Duet', 'Keystone', 'Picky 0']
f, axx = plt.subplots(1, 4, figsize=(6,2))
for i, LN in enumerate(LNs):
    vect_con = con_sel_cn[LNs[i]].values
    X1 = FG.get_ctr_norm_np(np.random.randn(n_d, n))
    corrs = np.sum(X1 * vect_con[:, np.newaxis], axis=0)
    m = np.mean(corrs)
    s = np.std(corrs)
    print(m, s)
    ax = axx[i]
    y, bins, _ = ax.hist(corrs, bins=my_bins, density=True, cumulative=True,
            label='from numerical\nsimulations',
                         histtype='step', lw=2)
    dbin = bins[1] - bins[0]
    mu, sigma = curve_fit(norm.cdf, bins[:-1] + dbin/2, y, p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins, norm.cdf(bins, mu, sigma), label='Gaussian fit')
    # x = np.linspace(-1, 1, 100)
    # ax.plot(x, 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-(x / (s * np.sqrt(2))) ** 2),
    #         label='gaussian')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('correlation coefficient')
    ax.set_title(titles[i])
    if i==3:
        ax.legend(loc='upper left', bbox_to_anchor=[0.02, 1])
plt.tight_layout()
plt.show()


#%%
# I am a bit surprised by the results. I would like to confirm it with the
# activity vectors
# So now we are going to use it with act_cn

act_cn = FG.get_ctr_norm(ORNA.act_sels[CONC][ACT_PPS].T, opt=0)

f, axx = plt.subplots(1, 4, figsize=(6,2))
n=10000
n_odors = 170
# n_odors = 34
for i, LN in enumerate(LNs):
    corrs = np.zeros(n * n_odors)
    vect_con = con_sel_cn[LNs[i]].values
    # X1 = act_cn.loc[:, (slice(None), 7)].values
    X1 = act_cn.values
    for j in range(n):
        np.random.shuffle(vect_con)
        corrs[j*n_odors:(j+1)*n_odors] = np.sum(X1 * vect_con[:, np.newaxis], axis=0)
    m = np.mean(corrs)
    s = np.std(corrs)
    print(m, s)
    ax = axx[i]
    y, bins, _ = ax.hist(corrs, bins=my_bins, density=True, cumulative=True,
            label='from numerical\nsimulations',
                         histtype='step', lw=2)
    dbin = bins[1] - bins[0]
    mu, sigma = curve_fit(norm.cdf, bins[:-1] + dbin/2, y, p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins, norm.cdf(bins, mu, sigma), label='Gaussian fit')
    # x = np.linspace(-1, 1, 100)
    # ax.plot(x, 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-(x / (s * np.sqrt(2))) ** 2),
    #         label='gaussian')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('correlation coefficient')
    ax.set_title(titles[i])
    if i==3:
        ax.legend(loc='upper left', bbox_to_anchor=[0.02, 1])
plt.tight_layout()
plt.show()


#%%
# here we are only using a specific concentration,
# and not all the concentrations.
# This graphs was not shown to the reviewer

f, axx = plt.subplots(1, 5, figsize=(6,2))
n=1000
n_odors = 170
n_odors = 34
for i, conc in enumerate([4, 5, 6, 7, 8]):
    corrs = np.zeros(n * n_odors)
    vect_con = con_sel_cn[LNs[3]].values
    X1 = act_cn.loc[:, (slice(None), conc)].values
    for j in range(n):
        np.random.shuffle(vect_con)
        corrs[j*n_odors:(j+1)*n_odors] = np.sum(X1 * vect_con[:, np.newaxis], axis=0)
    m = np.mean(corrs)
    s = np.std(corrs)
    print(m, s)
    ax = axx[i]
    y, bins, _ = ax.hist(corrs, bins=my_bins, density=True, cumulative=True,
            label='from num.\nsim',
                         histtype='step', lw=2)
    dbin = bins[1] - bins[0]
    mu, sigma = curve_fit(norm.cdf, bins[:-1] + dbin/2, y, p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins, norm.cdf(bins, mu, sigma), label='gauss')
    # x = np.linspace(-1, 1, 100)
    # ax.plot(x, 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-(x / (s * np.sqrt(2))) ** 2),
    #         label='gaussian')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('correlation coefficient')
    ax.set_title(f'{titles[3]}, conc {conc}')
    if i==3:
        ax.legend(loc='center left')
plt.tight_layout()
plt.show()