"""
Created on Aug 5 2019

@author: Nikolai M Chapochnikov


"""
# %%

import functions.general as FG
import functions.plotting as FP
import functions.nmf as nmf
import functions.olf_circ_offline as FOC
import functions.olfactory as FO
import params.act3 as par_act
# import seaborn as sns
import scipy.cluster.hierarchy as sch

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.linalg as LA
import importlib

# %%

importlib.reload(FOC)
importlib.reload(nmf)

# %%
# OK, so now i want to do an iteration over different values of rho, and just
# save the W that arise. From there we can do all the statistics one wants.
# How will we encode the W? Let's create a dataframe Ws, for which the rows
# will be the ORNs and the columns will be the different alpha and repetitions
importlib.reload(FOC)
importlib.reload(nmf)
DATASET = 3
act = FO.get_ORN_act_data(DATASET).T
act_m = act.mean(axis=1, level=('odor', 'conc'))
act_m = act_m.loc[par_act.ORN, :]
X = act_m.values
N = X.shape[1]
rect = True
mi3 = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                    names=['rho', 'rep', 'i'])
Ws = pd.DataFrame(index=par_act.ORN, columns=mi3)
K = 4  # also simulations with K = 8 below
# for p in [0]:
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

# %%
# Ws.to_hdf(f'../results/W_NNC-{K}-short.hdf', 'Ws')
Ws.to_hdf(FO.OLF_PATH / f'results/W_NNC-{K}.hdf', 'Ws')

# %%


K = 8
# for p in [0]:
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