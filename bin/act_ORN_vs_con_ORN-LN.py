"""
Created on Tue Dec 19 14:55:23 2017

@author: nchapochnikov


This is what is present in this file:
- plotting connections (not sure is that at all useful here, i feel we already)
have it in many other places...
- plotting activity, and the activity correlation, and activity for the new
odors
- calculating the PCA, NMF
- calculating the correlations and the significance tests for each correlation
coefficient independently (no multi-hypothesis testing in the first place)
- rank correlation (nothing interesting there)
- exporting the corr and sign data
- "deep shuffling" - this is a touchy topic. Where i am shuffling the response
for odors, but not shuffling the ORNs. The main result here is that the
first eigenvector is basically not affected. And it is because the 1st
loading vector is quite close to the mean. Have not been using this lately
- correlation of the connections directly with the ORN activations by the
odors
- significance testing for different LN classes for the population of
correlation coefficients
- correlation and significance of PCA/NMF of activity vs connectivity
- size of glomeruli vs activity vectors
- plotting the activity of single odors vs connectivity
- maximum correlation with a subspace of neural activity
- projecting the connectivity vector on PCA subspaces
- correlation between connectivity and activity subspaces
- plotting a scatter plot and a line plot, showing the resemblance of an
activity and a connectivity vector. Seems very similar to a function that we
have in the presentation plots file.

    
    
plotting of the correlation between different connectivity vector to and from
ORNs as well as to and from uPNs
in the beginning of the file we choose if we analyse ORNs or uPNs.
So if we want to have the full analysis,
we need to run this file 2 times, once for ORNs, once for uPNs

The other important part of this file is plotting the radio plots of the
connections vs the activity vectors.
This is something you often might want to have a look at, in order to
understand the correlation plot


"""


# timing stuff
# start_time = timeit.default_timer()
# do something
# elapsed = timeit.default_timer() - start_time
# print(elapsed)

# %%
# ################################# IMPORTS ###################################
# import openpyxl
import numpy as np
import pandas as pd
import importlib
# import datetime
# import sys
import itertools
# import sklearn.cluster as skc

# import h5py
# import scipy.io as sio  # to read the matlab .mat files
# import sklearn.decomposition as skdd
import statsmodels.stats.multitest as smsm  # for the multihypothesis testing

import functions.plotting as FP
import functions.general as FG
import functions.olfactory as FO



# %%
# ################################  RELOADS  ##################################
# =============================================================================
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)


# %%
# #############################################################################
# ########################  CLASS INITIALIZATION  #############################
# #############################################################################

dataset = 3

cell_type = 'ORN'
# cell_type = 'PN'


con_dirs = [0, 1, 2, 3] # feedforward, feedback, feedback divided by indegree
# mean of ff and fb. Here we are only interested in ff
con_dirs = [0]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = True
plot_plots = False
# N_NMF = 2
# act_sel_ks = ['all', 'ec50', '45', '678']
act_sel_ks = ['all']
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'
# however, the 'ec50' works only for dataset 1, as data for 3 cells is missing,
# so we cannot use the same ORN list as for the other things, so it becomes
# a bit a mess.

# act_pps_k1 = 'tn'  # k like key

# for responses generated from EC50
# act_pps_k1 = 'model'
# act_pps_k2 = 'raw'

# for mean responses
act_pps_k1 = 'raw'
act_pps_k2 = 'mean'

# not sure whawt is l2, probably some normalization
# act_pps_k2 = 'l2'

path_plots = None

odor_sel = None
# odor_sel = odor_subset


ORNA = FO.NeurActConAnalysis(dataset, cell_type, con_dirs, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=odor_sel,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, reduce=True)

# the cn pps is useful to calculate the SVD of the ctr-norm activity data
# importantly the cn is in the direction of each ORN, meaning the average
# activity of each ORN becomes 0 and the norm of each ORN activity is 1.

ORNA.add_act_pps('cn', lambda x: FG.get_pps(x, pps='cn')['cn'])
# this function adds the preprocessing defined in the function to the variable
# self.act_sels[k], where k is the key for a certain concentration selection


# because of the normalization, this creates NaN, so i will put 0 instead
# i imagine we should need to iterate over all the conc to have this everywhere
ORNA.act_sels['all']['cn'] = ORNA.act_sels['all']['cn'].fillna(0)
# ORNA.act_sels['4']['cn'] = ORNA.act_sels['4']['cn'].fillna(0)
act_SVD = FG.get_svd_df(ORNA.act_sels['all']['cn'])


with open(ORNA.path_plots / 'params.txt', 'w') as f:
    f.write(f'dataset: {dataset}\n'
            f'con_pps_k: {con_pps_k}\n'
            f'act_pps1: {act_pps_k1}\n'
            f'act_pps2: {act_pps_k2}\n'
            f'odor_subset: {odor_sel}\n')


# %%
# #############################################################################
# #################  CALCULATING PCA OF THE ACTIVITY DATA  ####################
# #############################################################################

# PCA
# ORNA.add_act_pps('n', lambda x: FG.get_pps(x, pps='n')['n'])
# meaning calculating the PCA of the activity with pps 'o' and only calculating
# the top 3 loading vectors
N_PCA = 5
ORNA.calc_act_PCA(act_sel_ks, 'o', k=N_PCA)

file = FO.OLF_PATH / f'results/NNC-W_act-all.hdf'
Ws = pd.read_hdf(file)
Ws_sel = Ws.drop(columns=['0.1o'], level=1)
# # for scaling in [1,2, 10]:
# #     # file = f'../results/W_NNOC_act{scaling}.hdf'
# #     Ws_sel = Ws.loc[:, ('all', f'{scaling}o')]
# #     # Ws_sel.rename(columns={'o': f'{scaling}o'}, inplace=True)
ORNA.act_W = pd.concat([ORNA.act_W, Ws_sel], axis=1)
# ###########################################
# ORNA.add_act_pps('nr', lambda x: FG.rectify(FG.get_pps(x, pps='n')['n']))
# ORNA.calc_act_NMF(act_sel_ks, 'nr', N=3)

# could make sense at a later point to put some ordering, so that it would
# be easier to visualize (as in the file act_compare_NMF_SNMF_NNOC.py)

# we need the normalized version for the calculation of the cos similarity
# we need the centered-normalized version for the calculation of the corr
_, ORNA.act_U_n, ORNA.act_W_cn = FG.get_ctr_norm(ORNA.act_W, opt=2)


# %%
# #############################################################################
# ########################  CALCULATING  CORRELATIONS  ########################
# #############################################################################

# at this stage we have the vectors from the activity as well as
# the connectivity vectors.
# the next stage is to calculate the correlation and the significance
# between them.
# the additional layer of complication relatively to before, is that
# although we just have one matrix of different activity data
# for the connectivity we have L, R and 3 different directions
# we already combined the L and R, but we still didn't combine the different
# directions, and we probably won;t do it, so we'll keep 3 different key
# for the 3 directions

# there is also the quesion of name change, but we skip that for the moment
# the last thing we need is to align the index of the activity and connectivity
# before we calculate correlation and significance

# if you want to have the high quality data for later use, then better put
# 50000, but needs computer with more memory
# N_iter = 50000
N_iter = 10000
ORNA.calc_act_con_cc_pv(N_iter, opt='fast')



# this function fill the dataframe self.cc
# and dictionaries self.cc_pv, self.sign_CS.
# the index are the different methods of processing the activity
# the columns are the different connections with the LNs.
# Later we might want to do differently and change the labels of the index
# or/and columns
# the instances that it fills are cc, cc_pv, CS, CS_pv

# here we have several things coming into play:
# - which cells do we actually keep and in which order
# - in which order do we show the different processing of the activity
# - which odor concentrations/methods do we actually choose to show

# so basically the idea here is that we have done the analysis once and for
# all with a bunch of concentrations, and now we are doing the plotting
# so we are selecting how exactly we want things to appear

# the version combine_pval is slower and can give innacurate results
# ORNA.cc_pv['f'] = FG.combine_pval(ORNA.cc, ORNA.cc_pv['l'], ORNA.cc_pv['r'])
ORNA.cc_pv['f'] = FG.combine_pval2(ORNA.cc_pv['o'], ORNA.cc_pv['l'],
                                   ORNA.cc_pv['r'])

# this function is combining the left and the right significance
# for the SVD (n !=1) (2-tailed test) and keeping just the probability
# on the right (one tailed) for all the other act analysis (like NMF, k-means
# etc..)


# exporting with added NMF and SNMF and NNOC results
for conc_sel in ['all']:
    file = FO.OLF_PATH / (f'results/corr_sign/act{dataset}-{act_pps_k1}-{act_pps_k2}_conc-'
            f'{conc_sel}_SVD-NNC_vs_con-ORN-all_')
    to_export = ORNA.cc.loc[conc_sel]
    to_export.to_hdf(f'{file}cc.hdf', 'cc')

    to_export = ORNA.cc_pv['f'].loc[conc_sel]
    to_export.to_hdf(f'{file}cc_pv.hdf', 'cc_pval')

