"""
Created Aug 28 2019
@author: Nikolai M Chapochnikov

This file has as goal to compare the NMF and SNMF decompositions for the
olfactory data
I could also include selections of different parts of the dataset

"""


# %%
# ################################# IMPORTS ###################################
import importlib

import functions.general as FG
import functions.olfactory as FO
import functions.plotting as FP
import functions.olf_circ_offline as FOC
import functions.circuit_simulation as FCS
import functions.nmf as nmf
import scipy.linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import seaborn as sns

import params.act3 as par_act
import params.con as par_con
from typing import List

# %%
# import matplotlib
importlib.reload(FG)
importlib.reload(FCS)
importlib.reload(FOC)
importlib.reload(FO)
importlib.reload(nmf)
# %%
# ############################## IMPORTING DATA ###############################

dataset = 3


cell_type = 'ORN'

con_dirs = [0, 1, 2, 3]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = True
plot_plots = False

# act_sel_ks = ['all', 'ec50', '45', '678']
act_sel_ks = ['all']
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'
# however, the 'ec50' works only for dataset 1, as data for 3 cells is missing,
# so we cannot use the same ORN list as for the other things, so it becomes
# a bit a mess.

act_pps_k1 = 'raw'
act_pps_k2 = 'mean'

path_plots = None

subfolder = 'act_whitening_simulations/'


ORNA = FO.NeurActConAnalysis(dataset, cell_type, con_dirs, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=None,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, subfolder=subfolder)

with open(ORNA.path_plots / 'params.txt', 'w') as f:
    f.write(f'dataset: {dataset}'
            f'act_pps1: {act_pps_k1}'
            f'act_pps2: {act_pps_k2}')

k_max = 8


# %%
# checking how centered the data is:
X = ORNA.act_sels['all']['o'].T
print('mean', X.mean(axis=1))
print('max', X.max(axis=1))
print('min', X.min(axis=1))


# %%


alpha = {0.1: 50, 0.35: 50, 1: 50, 2: 50, 10: 50, 100: 1000}
for rect in [(True, True)]:
    for scaling, i in itertools.product([0.1, 0.35, 1, 2, 10], range(1, k_max+1)):
    # for scaling, i in itertools.product([100], range(1, k_max+1)):
    # for scaling, i in itertools.product([100], range(1, k_max + 1)):
        ORNA.calc_act_NNC(act_sel_ks, 'o', k=i, alpha=alpha[scaling],
                           scaling=scaling, rtol=1e-10, rectY=rect[0],
                           rectZ=rect[1], beta=0.2, cycle=500)
# %%
# so the calc_act_NNC does several things:
# it calculates Y, Z, W and M. Let's save Y and Z for the 3 scalings
ORNA.act_Y.to_hdf(FO.OLF_PATH / 'results/NNC-Y_act-all.hdf', 'Y')
ORNA.act_Z.to_hdf(FO.OLF_PATH / 'results/NNC-Z_act-all.hdf', 'Z')
ORNA.act_W.to_hdf(FO.OLF_PATH / 'results/NNC-W_act-all.hdf', 'W')
# ORNA.act_M.to_hdf('../results/NNC-M_act-all.hdf', 'M')
#
#
#
#