"""
Created Aug 28 2019
@author: Nikolai M Chapochnikov

This file generates the following results that are then for plotting the
model behavior on the ORN dataset:
NNC-Y_act-all.hdf
NNC-Z_act-all.hdf
NNC-W_act-all.hdf

The following takes a while, since there are many repetitions,
in the scale of tens of mins
W_NNC-4.hdf
W_NNC-8.hdf

For the first files, one is using the class NeurActConAnalysis, and the
calculations for the NNC are done inside

For last files, one is directly using the function olf_gd_offline, which
is used under the hood in NeurActConAnalysis.

It is like this only for historical reasons and can be rewritten if needed
"""


# %%
# ################################# IMPORTS ###################################
import importlib

import functions.general as FG
import functions.olfactory as FO
import functions.olf_circ_offline as FOC
import numpy as np
import itertools
import pandas as pd

import params.act3 as par_act
import params.con as par_con

# %%
# import matplotlib
importlib.reload(FG)
importlib.reload(FOC)
importlib.reload(FO)
# %%
# ############################## IMPORTING DATA ###############################

dataset = 3


cell_type = 'ORN'

con_dirs = [0, 1, 2, 3]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = False
plot_plots = False

# act_sel_ks = ['all', 'ec50', '45', '678']
act_sel_ks = ['all']
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'
# however, the 'ec50' works only for dataset 1, as data for 3 cells is missing


act_pps_k1 = 'raw'
act_pps_k2 = 'mean'

path_plots = None
subfolder = None


ORNA = FO.NeurActConAnalysis(dataset, cell_type, con_dirs, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=None,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, subfolder=subfolder)

# with open(ORNA.path_plots / 'params.txt', 'w') as f:
#     f.write(f'dataset: {dataset}'
#             f'act_pps1: {act_pps_k1}'
#             f'act_pps2: {act_pps_k2}')

k_max = 8


# %%
# checking how centered the data is:
X = ORNA.act_sels['all']['o'].T
print('mean', X.mean(axis=1))
print('max', X.max(axis=1))
print('min', X.min(axis=1))


# %%
# alpha is the coefficient by which the gradients are multiplied
alpha = {0.1: 50, 0.35: 50, 1: 50, 2: 50, 10: 50, 100: 1000}
for rect in [(True, True)]:
    for rho, i in itertools.product([0.1, 0.35, 1, 2, 10], range(1, k_max+1)):
    # for scaling, i in itertools.product([100], range(1, k_max+1)):
    # for scaling, i in itertools.product([100], range(1, k_max + 1)):
        ORNA.calc_act_NNC(act_sel_ks, 'o', k=i, alpha=alpha[rho],
                           rho=rho, rtol=1e-10, rectY=rect[0],
                           rectZ=rect[1], beta=0.2, cycle=500)
# %%
# so the calc_act_NNC does several things:
# it calculates Y, Z, W and M. Let's save Y and Z for the 3 scalings
path_results = FO.OLF_PATH / 'results'
ORNA.act_Y.to_hdf(path_results / 'NNC-Y_act-all.hdf', 'Y')
ORNA.act_Z.to_hdf(path_results / 'NNC-Z_act-all.hdf', 'Z')
ORNA.act_W.to_hdf(path_results / 'NNC-W_act-all.hdf', 'W')
# ORNA.act_M.to_hdf('../results/NNC-M_act-all.hdf', 'M')
#
#
#
# %%
# #############################################################################
# #############################################################################
# ###############   FOR FIGURE 4 of the main text    ##########################
# ###############     showing corr of W with W_LN   ###########################
# ###############           and grouping of Ws      ###########################
# #############################################################################
# #############################################################################
# Doing an iteration over different values of rho, and
# saving the W that arise. From there we can do all the statistics one wants.
# Ws are saved as dataframe Ws, for which the rows
# are the ORNs and the columns will be the different alpha and repetitions
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
importlib.reload(FOC)
DATASET = 3
act = FO.get_ORN_act_data(DATASET).T
act_m = act.groupby(axis=1, level=('odor', 'conc')).mean()
act_m = act_m.loc[par_act.ORN, :]
X = act_m.values
N = X.shape[1]
rect = True
mi3 = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                    names=['rho', 'rep', 'i'])
Ws = pd.DataFrame(index=par_act.ORN, columns=mi3)
for K in [4, 8]:
    for p in np.arange(-10, 10.1, 0.5):
    # going through a logarithmic scale for rho, from 0.1 to 10
    # for p in np.arange(10, 10.1, 0.5):
        rho = 10**(p/10)
        print('rho: ', rho)
        for i in range(50):
            print('repetition: ', i)
            Y, Z, _ = FOC.olf_gd_offline(X, K, max_iter=10000, rectY=rect,
                                         rectZ=rect,
                                         alpha=50, cycle=500, rho=rho, beta=0.2)
            W = Y @ Z.T / N
            # CG = sns.clustermap(np.corrcoef(W.T))
            # plt.close()
            # idx = CG.dendrogram_col.reordered_ind
            # W = W[:, idx]
            print(W.shape)
            for j in range(K):
                Ws.loc[:, (p, i, j)] = W[:, j]

    Ws.to_hdf(FO.OLF_PATH / f'results/W_NNC-{K}.hdf', 'Ws')