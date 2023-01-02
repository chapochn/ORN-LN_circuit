"""
Created on Tue Dec 19 14:55:23 2017

@author: Nikolai M Chapochnikov


This is what is present in this file:
- calculating the PCA
- calculating the correlations and the significance tests for each correlation
coefficient independently (no multi-hypothesis testing in the first place)
- exporting the corr and sign data
- correlation of the connections directly with the ORN activations by the
odors
- correlation and significance of PCA/NMF of activity vs connectivity


"""


# timing stuff
# start_time = timeit.default_timer()
# do something
# elapsed = timeit.default_timer() - start_time
# print(elapsed)

# %%
# ################################# IMPORTS ###################################

import numpy as np
import pandas as pd
import importlib

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

OLF_PATH = FO.OLF_PATH
RESULTS_PATH = OLF_PATH / 'results'
# RESULTS_PATH = OLF_PATH / 'results2'

dataset = 3

cell_type = 'ORN'
# cell_type = 'PN'


con_dirs = [0, 1, 2, 3] # feedforward, feedback, feedback divided by indegree
# mean of ff and fb. Here we are only interested in ff
con_dirs = [0]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = False
plot_plots = False
# N_NMF = 2
# act_sel_ks = ['all', 'ec50', '45', '678']
act_sel_ks = ['all']
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'
# however, the 'ec50' works only for dataset 1, as data for 3 cells is missing

# act_pps_k1 = 'tn'  # k like key

# for responses generated from EC50
# act_pps_k1 = 'model'
# act_pps_k2 = 'raw'

# for mean responses
act_pps_k1 = 'raw'
act_pps_k2 = 'mean'


path_plots = None

odor_sel = None
# odor_sel = odor_subset


ORNA = FO.NeurActConAnalysis(dataset, cell_type, con_dirs, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=odor_sel,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, reduce=True)

# with open(ORNA.path_plots / 'params.txt', 'w') as f:
#     f.write(f'dataset: {dataset}\n'
#             f'con_pps_k: {con_pps_k}\n'
#             f'act_pps1: {act_pps_k1}\n'
#             f'act_pps2: {act_pps_k2}\n'
#             f'odor_subset: {odor_sel}\n')


# %%
# #############################################################################
# #################  CALCULATING PCA OF THE ACTIVITY DATA  ####################
# #############################################################################

# PCA

N_PCA = 5
ORNA.calc_act_PCA(act_sel_ks, 'o', k=N_PCA)

file = FO.OLF_PATH / f'{RESULTS_PATH}/NNC-W_act-all.hdf'
Ws = pd.read_hdf(file)
Ws_sel = Ws.drop(columns=['0.1o'], level=1)

ORNA.act_W = pd.concat([ORNA.act_W, Ws_sel], axis=1)

_, ORNA.act_U_n, ORNA.act_W_cn = FG.get_ctr_norm(ORNA.act_W, opt=2)


# %%
# #############################################################################
# ########################  CALCULATING  CORRELATIONS  ########################
# #############################################################################

# We have the vectors from the activity as well as the connectivity vectors.
# Next we calculate the correlation and the significance between them.


# if you want to have the high quality data for later use, then better put
# 50000, but needs computer with more memory
# N_iter = 50000
N_iter = 50000
ORNA.calc_act_con_cc_pv(N_iter)

# this function fill the dataframe self.cc
# and dictionaries self.cc_pv
# the index are the different methods of processing the activity
# the columns are the different connections with the LNs.

ORNA.cc_pv['f'] = FG.combine_pval(ORNA.cc_pv['o'], ORNA.cc_pv['l'],
                                  ORNA.cc_pv['r'])

# this function is combining the left and the right significance
# for the SVD (n !=1) (2-tailed test) and keeping just the probability
# on the right (one tailed) for all the other act analysis (like NNC)

res_path = RESULTS_PATH / 'corr_sign'
res_path.mkdir(exist_ok=True)

# exporting with added NNC results
for conc_sel in ['all']:
    file = FO.OLF_PATH / (f'{RESULTS_PATH}/corr_sign/act{dataset}-{act_pps_k1}-{act_pps_k2}_conc-'
            f'{conc_sel}_SVD-NNC_vs_con-ORN-all_')
    to_export = ORNA.cc.loc[conc_sel]
    to_export.to_hdf(f'{file}cc.hdf', 'cc')

    to_export = ORNA.cc_pv['f'].loc[conc_sel]
    to_export.to_hdf(f'{file}cc_pv.hdf', 'cc_pval')

print('final done')
