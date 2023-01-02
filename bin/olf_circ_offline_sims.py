"""
Created in 2021

@author: Nikolai M Chapochnikov

this generates several results that are then plotted in the paper:
the files in dataset1 and dataset2 (related to changing rho and K
to see the effect of the correlation of W) and dataset3 (use for figure 6)

"""

# %%
import itertools

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import importlib

import functions.general as FG
import functions.plotting as FP
import functions.nmf as nmf
import functions.olf_circ_offline as FOC
import functions.olfactory as FO
import functions.datasets as FD

# %%

importlib.reload(FOC)
importlib.reload(nmf)


#%%
# #############################################################################
# #############################################################################
# ###############   FOR FIGURES 9&10 of supplement    ##########################
# ###############   showing clustering by NNC        ##########################
# #############################################################################
# #############################################################################
# #############################################################################

importlib.reload(FD)

# Analysis for understanding the influence of rho and LNs number
# We create 2 datasets, each with 2 clusters:
# 1. clusters close to the edge, i.e., clusters further apart
# 2. clusters further from the edge, clusters closer together
D = 10
n_clusters = 2
X1 = {}
for data_i, dx in [(1, 0), (2, 0.4)]:
    data_path = FO.OLF_PATH / f'results/dataset{data_i}'
    data_path.mkdir(exist_ok=True)
    file = data_path / f'dataset_{D}D_{n_clusters}clusters.npy'
    try:
        X = np.load(file/'asdf')
        print('dataset already exists, it is loaded')
    except:
        print('dataset does not exist yet, it will be created and saved')
        X = FD.create_clusters_data(D, [100, 150], n_clusters, dx=dx, scale=0.15)
        np.save(file, X)  # saving the dataset, so that it can be reused.

    X1[data_i] = X
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
    plt.suptitle(f'dataset {data_i}')
    plt.show()


#%%
# results from the original similarity matching function
# this is just to check if the results below make sense
# importlib.reload(nmf)

# X = X1[1]
# # X = X1[2]
#
# K = 3
# K = 2
# output = nmf.get_snmf_best_np(X, k=K, max_iter=100000)
# Z = output[0]
# W = X @ Z.T
#
# f, max_rect_corr = plot_clustering_results(X, Z, W, x_max=1)
# f.suptitle(f'SNMF, corr:{max_rect_corr}')
# f.show()
#%%
# results from the circuit with given rho similarity matching function
importlib.reload(FOC)
rect = True

alpha = 50
# for data_i, rho, K in itertools.product([2], [0.1, 1, 10], [2, 3]):
for data_i, rho, K in itertools.product([1, 2], [0.1, 1, 10], [2, 3]):
    X = X1[data_i]
    Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                       rectZ=rect,
                                       # init='pre_init',
                                       init='full_rand',
                                 alpha=alpha, cycle=1000, rho=rho, beta=0.5,
                                       output_n=1000, sigma=0.1)
    W1 = Y1 @ Z1.T / N

    file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}'
    np.save(file, Y1)
    file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}'
    np.save(file, Z1)


    f, rect_corr_crt = FP.plot_clustering_results(X, Z1, W1, Y=Y1)
    f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
    f.show()
    print(rect_corr_crt)

#%%
# plotting all the results obtained above:
# D = 10
# n_clusters = 2
# for data_i, rho, K in itertools.product([1, 2], [0.1, 1, 10], [2, 3]):
#     file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
#     X = np.load(file)
#     D, N = X.shape
#
#     file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}.npy'
#     Y1 = np.load(file)
#     file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}.npy'
#     Z1 = np.load(file)
#     W1 = Y1 @ Z1.T / N
#     f, rect_corr_crt = FP.plot_clustering_results(X, Z1, W1, Y=Y1)
#     f.suptitle(f'K:{K}, rho:{rho}, corr:{rect_corr_crt}')
#     f.show()
#     print(data_i, rho, K, rect_corr_crt)


#%%
# #############################################################################
# #############################################################################
# ###############   FOR FIGURES 6, showing the effect of NNC and LC   #########
# ###############   showing clustering by NNC        ##########################
# #############################################################################
# #############################################################################
# #############################################################################
importlib.reload(FD)

# dataset in 2d with 2 clusters
D = 2
n_clusters = 2

data_i, dx = 3, 0.4

data_path = FO.OLF_PATH / f'results/dataset{data_i}'
data_path.mkdir(exist_ok=True)
file = data_path / f'dataset_{D}D_{n_clusters}clusters.npy'
try:
    X = np.load(file)
    print('dataset already exists, it is loaded')
except:
    print('dataset does not exist yet, it will be created and saved')
    X = FD.create_clusters_data(D, [100, 100], n_clusters, dx=dx, scale=0.17)
    np.save(file, X)  # saving the dataset, so that it can be reused.

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
# importlib.reload(nmf)
# K = 2
# output = nmf.get_snmf_best_np(X, k=K, max_iter=100000)
# Z = output[0]
# W = X @ Z.T
#
# f, max_rect_corr = FP.plot_clustering_results(X, Z, W, x_max=1)
# f.suptitle(f'SNMF, corr:{max_rect_corr}')
# f.show()
#%%
# results from the circuit with given rho similarity matching function
importlib.reload(FOC)
rect = True

K = 2

alpha = 50
rho = 1
Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                   rectZ=rect,
                                   # init='pre_init',
                                   init='full_rand',
                             alpha=alpha, cycle=1000, rho=rho, beta=0.5,
                                   output_n=1000, sigma=0.1)
W1 = Y1 @ Z1.T / N

file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}'
np.save(file, Y1)
file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}'
np.save(file, Z1)


f, rect_corr_crt = FP.plot_clustering_results(X, Z1, W1, Y=Y1)
f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
f.show()
print(rect_corr_crt)

print('done final')

#%%
# doing a full simulation with the dataset above:
# #############################################################################
# #############################################################################
# this is for the case if you want a more fine grained depiction
# of what is happening for different rho. It is not used in the paper,
# As I think showing the results for rho=0.1,1,10 is enough to get an understanding


# rect = True
#
# ps = np.linspace(-10, 10, 21)
# rhos = 10.**(ps/10)
# ps = np.round(ps).astype(int)
# # rhos = np.logspace(-1, 1, 21)
# # ps = np.round((np.log10(rhos)*10)).astype(int)
# # ps = [-2, -1]
# # rhos = [10**-0.2, 10**-0.1]
# # ps = [-1]
# # rhos = [10**-ps[0]]
# print(rhos)
# print(ps)
# N_rep = 10
#
# mi3 = pd.MultiIndex.from_product([ps, range(N_rep), range(K)],
#                     names=['p', 'rep', 'i'])
# Ws = pd.DataFrame(index=np.arange(D), columns=mi3)
#
# K = 8
#
# # for p in [0]:
# for p, rho in zip(ps, rhos):
# # going through a logarithmic scale for rho, from 0.1 to 10
# # for p in np.arange(10, 10.1, 0.5):
#     for i in range(N_rep):
#         print('rho: ', rho)
#         print('repetition: ', i)
#         Y, Z, _ = FOC.olf_gd_offline(X, K, max_iter=10000, rectY=rect,
#                                      rectZ=rect, init='full_rand',
#                                      alpha=50, cycle=500, rho=rho, beta=0.2)
#         W = Y @ Z.T / N
#
#         for j in range(K):
#             Ws.loc[:, (p, i, j)] = W[:, j]
#
# #%%
# # Ws.to_hdf(FO.OLF_PATH / f'results/W_NNC-dataset_{n_clusters}clusters.hdf', 'Ws')
# #%%
# n_clusters = 4
# Ws = pd.read_hdf(FO.OLF_PATH / f'results/W_NNC-dataset_{n_clusters}clusters.hdf')
# #%%
# mi2 = pd.MultiIndex.from_product([ps, range(N_rep)],
#                     names=['p', 'rep'])
# corr_W_nnc_s = pd.Series(index=mi2, dtype=float)
# # gathering the results from above:
# for p in ps:
#     for i in range(N_rep):
#         W_crt = Ws.loc[:, (p, i)]
#         corr = np.corrcoef(W_crt.T)
#         df1_ent = FG.get_entries(corr, diag=False)
#         corr_W_nnc_s.loc[(p, i)] = FG.rectify(df1_ent).mean()
#
# y = corr_W_nnc_s.groupby('p').mean()
# e = corr_W_nnc_s.groupby('p').std()
# # print(corr_W_nnc_s)
#
# x = ps/10
# f, ax = plt.subplots()
# ax.plot(x, y, lw=1, c='k')
# ax.fill_between(x, y-e, y+e, alpha=0.5, facecolor='k', label='NNC-8')
# ax.hlines(max_rect_corr, -1, 1)
# ax.set_ylim(0, None)
# plt.show()
#%%