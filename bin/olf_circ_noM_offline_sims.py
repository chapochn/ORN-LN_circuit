"""
Created in 2022

@author: Nikolai M Chapochnikov


This file generates that files that are in results/dataset4

"""
# %%

import functions.general as FG
import functions.plotting as FP
import functions.nmf as nmf
import functions.white_circ_offline as FOC
import functions.olfactory as FO
import functions.datasets as FD
import params.act3 as par_act
import scipy.cluster.hierarchy as sch

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.linalg as LA
import importlib
import matplotlib.pyplot as plt

def plot_clustering_results(X, Z, W, x_max=1, Y=None):
    f, axx = plt.subplots(2, 2)
    ax = axx[0, 0]
    im = ax.imshow(Z, aspect='auto', interpolation='nearest',
               vmin=0)
    ax.set_title('Z activity')
    f.colorbar(im, ax=ax)

    ax = axx[1, 0]
    im = ax.imshow(Z.T @ Z, interpolation='nearest')
    f.colorbar(im, ax=ax)
    ax.set_title('Z similarity')

    # W = W/np.max(W, axis=0, keepdims=True)
    W = W/ LA.norm(W, axis=0, keepdims=True)
    ax = axx[0, 1]
    ax.scatter(*X[:2], s=1)
    if Y is not None:
        ax.scatter(*Y[:2], s=1)
    ax.scatter(*x_max*W[:2])
    # ax.set_xlim(0, None)
    # ax.set_ylim(0, None)
    ax.set_aspect(1)
    ax.set_title('X, Y, W')


    ax = axx[1, 1]
    corr = np.corrcoef(W.T)
    links = sch.linkage(corr, method='average', optimal_ordering=True)
    new_order = sch.leaves_list(links)
    im = plt.imshow(corr[new_order][:, new_order], cmap='bwr', vmax=1, vmin=-1)
    f.colorbar(im, ax=ax)
    ax.set_title('correlation among W')

    entries = FG.get_entries(corr, diag=False)
    max_rect_corr = FG.rectify(entries).mean()
    print('mean rectified corr', max_rect_corr)
    return f, max_rect_corr
# %%

importlib.reload(FOC)
importlib.reload(nmf)



#%%
# #############################################################################
# #############################################################################
# #########  FOR analog FIGURES 6, showing the effect of NNC and LC   #########
# ###############   showing clustering by NNC        ##########################
# #############################################################################
# #############################################################################
# #############################################################################
# Plots for understanding the influence of rho and LNs number, with 2 clusters

# First let's create a dataset we want to work with:
# we will have 2 clusters
importlib.reload(FD)


D = 2
n_clusters = 2

data_i, dx = 4, 0
X = FD.create_clusters_data(D, [100, 100], n_clusters, dx=dx, scale=0.3,
                            x_max=2)
file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
# np.save(file, X)
X = np.load(file)  # directly load the already saved dataset

# create uniform data
# N = 100
# X = np.random.normal(size=[D, N])
# X[2:] = 0
# X = X / np.linalg.norm(X, axis=0)
# X += np.random.normal(size=[D, N], scale=0.1)
# X = np.abs(X)


D, N = X.shape


f, axx = plt.subplots(1, 3)
ax = axx[0]
im = ax.imshow(X, aspect='auto', interpolation='nearest', vmin=0)
ax.set_xlabel('samples/time')
ax.set_ylabel('dimensions')
ax.set_title('data')
f.colorbar(im, ax=ax)

ax = axx[1]
im = ax.imshow(X.T @ X, interpolation='nearest')
ax.set_xlabel('samples/time')
ax.set_ylabel('samples/time')
ax.set_title('data similarity matrix')
f.colorbar(im, ax=ax)

ax = axx[2]
ax.scatter(*X[:2], s=1)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.set_title('data first 2 dimensions')
ax.set_aspect(1)
plt.tight_layout()
plt.show()


#%%
# results from the original similarity matching function
# this is just to see that the results below make sense
importlib.reload(nmf)
K = 2
output = nmf.get_snmf_best_np(X, k=K, max_iter=100000)
Z = output[0]
W = X @ Z.T

f, max_rect_corr = plot_clustering_results(X, Z, W, x_max=1)
f.suptitle(f'SNMF, corr:{max_rect_corr}')
f.show()
#%%
# results from the circuit with given rho similarity matching function
importlib.reload(FOC)
rect = True
# rect = False

K = 2
# There is a bit of tinkering needed if terms of alpha as far as I remember
# alpha = 50  # seems to be a bit faster
alpha = 10 # trying things out
# alpha = 1  # maybe slower but more precise
# for rho in [0.1, 1, 10]:  # the parameters needed in the paper
rho = 1
Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                   rectZ=rect,
                                   # init='pre_init',
                                   init='random',
                             alpha=alpha, cycle=100000, rho=rho, beta=0.1,
                                   output_n=1000, sigma=0.1,
                                   rtol=1e-7)
W1 = Y1 @ Z1.T / N

file = FO.OLF_PATH / f'results/dataset{data_i}/whiteNNC_Y_K{K}_rho{rho}'
np.save(file, Y1)
file = FO.OLF_PATH / f'results/dataset{data_i}/whiteNNC_Z_K{K}_rho{rho}'
np.save(file, Z1)


f, rect_corr_crt = plot_clustering_results(X, Z1, W1, Y=Y1)
f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
f.show()
print(rect_corr_crt)
