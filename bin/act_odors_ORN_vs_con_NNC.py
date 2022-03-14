#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 4th 2022

@author: Nikolai M Chapochnikov

This file is similar to act_odors_ORN_vs_con_ORN-LN,
but with the connections arising from the NNC instead of the real connections
from the data.

"""


# timing stuff
# start_time = timeit.default_timer()
# do something
# elapsed = timeit.default_timer() - start_time
# print(elapsed)

# %%
# #############################################################################
# ################################# IMPORTS ###################################
# #############################################################################

# import openpyxl
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
import pandas as pd
import importlib
# import datetime
# import sys
# import sklearn.cluster as skc

import params.act3 as par_act3

# import h5py
# import scipy.io as sio  # to read the matlab .mat files
# import sklearn.decomposition as skdd
from functions import general as FG, olfactory as FO, plotting as FP


# %%
# ################################  RELOADS  ##################################
# =============================================================================
importlib.reload(FO)
importlib.reload(FG)

# %%
# #############################################################################
# ########################  CLASS INITIALIZATION  #############################
# #############################################################################

DATASET = 3

# there might be a cleaner way to write this...
par_act = None
exec(f'par_act = par_act{DATASET}')

cell_type = 'ORN'
# cell_type = 'PN'

ORN_order = par_act.ORN_order

strms = [0, 1, 2, 3]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = True
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
                             neur_order=ORN_order, odor_order=None,
                             path_plots=path_plots, reduce=True,
                             subfolder='act_ORN_odors_vs_con_ORN-LN/')


xmin = -1
xmax = 1
bins_pdf = np.linspace(xmin, xmax, 21)
n_bins_cdf = 500
bins_cdf = np.linspace(xmin, xmax, n_bins_cdf + 1)

with open(ORNA.path_plots / f'params.txt', 'w') as f:
    f.write(f'\ndataset: {DATASET}')
    f.write(f'\nact_pps1: {act_pps_k1}')
    f.write(f'\nact_pps2: {act_pps_k2}')
    f.write(f'\nodor_subset: {odor_sel}')
    f.write(f'\nACT_PPS: {ACT_PPS}')
    f.write(f'\nCONC: {CONC}')
    f.write(f'\nact_sel_ks: {act_sel_ks}')

png_opts = {'dpi': 250, 'transparent': True}


# %%
# #############################################################################
# ###########  Importing the W of NNC-4 cells  ################################
# #############################################################################

OLF_PATH = FO.OLF_PATH

RESULTS_PATH = OLF_PATH / 'results'
file = RESULTS_PATH / 'NNC-W_act-all.hdf'
Ws = pd.DataFrame(pd.read_hdf(file))

conc = 'all'
scal = 1  # so this is the same as in the figure 3 of the paper
pps = f'{scal}o'

K = 4
meth = f'NNC_{K}'
# Y[k] = Ys[conc, pps, meth, '', '']/scal
# Y[k] = Y[k].loc[:, ORN_order].reindex(odor_order, level='odor').T
# # Y[k] = Y[k].values
# Z[k] = Zs[conc, pps, meth, '', '']
# Z[k] = Z[k].reindex(odor_order, level='odor').T
# # Z[k] = Z[k].values
# W2[k] = Y[k] @ Z[k].T / N
W = Ws[conc, pps, meth, '', '']/scal
W = W.loc[ORN_order]
# %%
# #############################################################################
# ###########  GLOBAL STUFF USED LATER  #######################################
# #############################################################################
act_cn = FG.get_ctr_norm(ORNA.act_sels[CONC][ACT_PPS].T, opt=0)
W_cn = FG.get_ctr_norm(W, opt=0)

#%%
# calculating the corr and signifiance each odor vs each W from NNC

con_sel_cn = W_cn


# the corr(should be) the same as the one obtained below

cor0 = con_sel_cn.T @ act_cn



# ######################  PDF and CDF FOR ALL CELLS  #########################

CELL_LIST = list(cor0.index)
# getting the true pdfs and cdfs
pdf_true, cdf_true = FG.get_pdf_cdf_2(cor0.values, bins_pdf, bins_cdf,
                                      add_point=True)
pdf_true = pd.DataFrame(pdf_true, index=cor0.index)
cdf_true = pd.DataFrame(cdf_true, index=cor0.index)

# %%
# EXPORTING THE TRUE CDFS

# before explorting this, be sure what is the CELL_LIST, if you want
# the M cells to be there or not

file_begin = (FO.OLF_PATH / f'results/NNC-{K}_con-W_vs_'
              f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_corr_')
cdf_true.to_hdf(f'{file_begin}cdf-true.hdf', 'cdf_true')

# %%
# #############################################################################
# ######################  SIGNIFICANCE WHEN GENERATING  #######################
# ######################       Ws BY SHUFFLING          #######################
# ######################                                #######################
# #############################################################################






# #############################################################################
# ###################  GETTING COLLECTION OF CORR COEF  #######################
# #############################################################################

# the 2 different versions of shuffling:
# 1. each column is shuffled sevarately, so the correlation between connections
# are kept
# 2. all columns are permuted simultaneously/together, thus the
# relations/correlations between the different dimensions are conserved.


# in corr1: n_cells x n_odors
# in cor_col1, N x n_cells x n_odors

N = 50000
    # the order in the function matters, because it is the matrix
    # in the second position that will be shuffled, so it is better
    # to put there the matrix with less columns, as it will thus take
    # less time
    # later i invert the axes so that the odors are in the last dimension
_, cor_col1 = FG.get_signif_corr_v1(act_cn, con_sel_cn, N=N, dist=True)
cor_col1 = np.swapaxes(cor_col1, 1, 2)



if np.isnan(cor_col1).any():
    print("WE HAVE SOME NANs IN cor_col1 which shouldn't be there")

#%%
# Saving the mean and std of the correlation coefficients, so that it
# can be used to define a gaussian.

# %%
# #############################################################################
# ##############  PDF AND CDF OF REAL AND SHUFFLED CORR COEF  #################
# #############################################################################

# N_iter = 10000
N_iter = N
pdfs_shfl, cdfs_shfl = FG.get_pdf_cdf_3(cor_col1[:N_iter],
                                        bins_pdf, bins_cdf, checknan=False)

# mean and std of histograms
# the averages are among the instances
pdfs_shfl_m = pd.DataFrame(pdfs_shfl.mean(axis=0), index=CELL_LIST)
pdfs_shfl_std = pd.DataFrame(pdfs_shfl.std(axis=0), index=CELL_LIST)

cdf_shfl_m = pd.DataFrame(cdfs_shfl.mean(axis=0), index=CELL_LIST)
cdf_shfl_std = pd.DataFrame(cdfs_shfl.std(axis=0), index=CELL_LIST)

# %%
# #############################################################################
# ########################  EXPORTING CDF MEAN, AND STD  ######################

# before explorting this, be sure what is the CELL_LIST, if you want
# the M cells to be there or not
# =============================================================================
file_begin = (f'results/NNC-{K}_con-W_vs_'
              f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_corr_')
cdf_shfl_m.to_hdf(FO.OLF_PATH / f'{file_begin}cdf-shfl-m.hdf', 'cdf_shfl_m')
cdf_shfl_std.to_hdf(FO.OLF_PATH / f'{file_begin}cdf-shfl-std.hdf',
                    'cdf_shfl_std')
# =============================================================================


# %%
# ###################  MAXIMUM DEVIATION IN CDF AND PVAL  #####################

cdf_shfl_diff_min = FG.get_min_diff(cdfs_shfl, cdf_shfl_m.values)
# the max is over the odors, i.e., the last dimension

# cdf_diff_max_true[strm] = np.max(np.abs(cdf_true[strm] - cdf_mean),
# axis=1)
# first dim are the cells, second dim are the odors
# the 2 is because we already have a similar name, where the mean
# comes from the category averages and not the cell per cell shuffled average
cdf_diff_min2 = FG.get_min_diff(cdf_true, cdf_shfl_m)

# cdf_diff_max_true2 contains the true max deviation for each cell

# cdf_diff_max_shfl is N_iter x n_cells
# cdf_diff_max_true2 is n_cells
pv_min = np.mean(cdf_shfl_diff_min >= cdf_diff_min2.values, axis=0)

# cdf_diff_max_true2 = pd.Series(cdf_diff_max_true2, index=cor0[strm].columns)
cdf_diff_min_pv2 = pd.Series(pv_min, index=CELL_LIST)
cdf_diff_min_pv2 = cdf_diff_min_pv2.replace(0, 1./N).T


# %%

# =============================================================================
file_begin = (f'results/NNC-{K}_con-W_vs_'
              f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_corr_'
              f'cdf-shfl-diff-min')
cdf_diff_min2.to_hdf(FO.OLF_PATH / f'{file_begin}.hdf', 'cdf_diff_min')
cdf_diff_min_pv2.to_hdf(FO.OLF_PATH / f'{file_begin}_pv.hdf', 'cdf_diff_min_pv')
# 
# =============================================================================