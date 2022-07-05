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
    f.colorbar(im, ax=ax)

    ax = axx[1, 0]
    im = ax.imshow(Z.T @ Z, interpolation='nearest')
    f.colorbar(im, ax=ax)

    W = W/np.max(W, axis=0, keepdims=True)
    ax = axx[0, 1]
    ax.scatter(*X[:2], s=1)
    if Y is not None:
        ax.scatter(*Y[:2], s=1)
    ax.scatter(*x_max*W[:2])
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_aspect(1)

    ax = axx[1, 1]
    corr = np.corrcoef(W.T)
    links = sch.linkage(corr, method='average', optimal_ordering=True)
    new_order = sch.leaves_list(links)
    im = plt.imshow(corr[new_order][:, new_order], cmap='bwr', vmax=1, vmin=-1)
    f.colorbar(im, ax=ax)

    entries = FG.get_entries(corr, diag=False)
    max_rect_corr = FG.rectify(entries).mean()
    print('mean rectified corr', max_rect_corr)
    return f, max_rect_corr
# %%

importlib.reload(FOC)
importlib.reload(nmf)

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
#%%
# #############################################################################
# #############################################################################
# ###############   FOR FIGURES 8&9 of supplement    ##########################
# ###############   showing clustering by NNC        ##########################
# #############################################################################
# #############################################################################
# #############################################################################
# Plots for understanding the influence of rho and LNs number, with 2 clusters

# First let's create a dataset we want to work with:
# we will have 2 clusters
importlib.reload(FD)

# D = 21
D = 10
# D = 2
n_clusters = 2


data_i, dx = 1, 0  # clusters close to the edge, i.e., clusters further apart
data_i, dx = 2, 0.4  # clusters further from the edge, clusters closer together
X = FD.create_clusters_data(D, [100, 150], n_clusters, dx=dx, scale=0.15)
file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
# np.save(file, X)  # I save the dataset, so that it can be reused.
X = np.load(file)

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
K = 3
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

# choose one of the two (K=2 or K=3), both are needed for the paper plots
K = 3
K = 2
# There is a bit of tinkering needed if terms of alpha as far as I remember
alpha = 50  # seems to be a bit faster
alpha = 1  # maybe more slower but more precise
# for rho in [0.1, 1, 10]:  # the parameters needed in the paper
for rho in [.1]:
    Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                       rectZ=rect,
                                       # init='pre_init',
                                       init='full_rand',
                                 alpha=alpha, cycle=100000, rho=rho, beta=0.1,
                                       output_n=1000, sigma=0.1)
    W1 = Y1 @ Z1.T / N

    file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}'
    np.save(file, Y1)
    file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}'
    np.save(file, Z1)


    f, rect_corr_crt = plot_clustering_results(X, Z1, W1, Y=Y1)
    f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
    f.show()
    print(rect_corr_crt)

#%%
# plotting the results obtained above:
D = 10
K = 3
n_clusters = 2
# for data_i in [1, 2]:
for data_i in [2]:
    file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
    X = np.load(file)
    D, N = X.shape
    for rho in [0.1, 1, 10]:
    # for rho in [1]:
        file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}.npy'
        Y1 = np.load(file)
        file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}.npy'
        Z1 = np.load(file)
        W1 = Y1 @ Z1.T / N
        f, rect_corr_crt = plot_clustering_results(X, Z1, W1, Y=Y1)
        f.suptitle(f'K:{K}, rho:{rho}, corr:{rect_corr_crt}')
        f.show()
        print(data_i, rho, rect_corr_crt)
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

#%%
# #############################################################################
# #############################################################################
# ###############   FOR FIGURES 6, showing the effect of NNC and LC   #########
# ###############   showing clustering by NNC        ##########################
# #############################################################################
# #############################################################################
# #############################################################################
# Plots for understanding the influence of rho and LNs number, with 2 clusters

# First let's create a dataset we want to work with:
# we will have 2 clusters
importlib.reload(FD)

# D = 21
D = 2
# D = 2
n_clusters = 2


# data_i, dx = 1, 0  # clusters close to the edge, i.e., clusters further apart
data_i, dx = 3, 0.2  # clusters further from the edge, clusters closer together
X = FD.create_clusters_data(D, [100, 100], n_clusters, dx=dx, scale=0.15)
file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
np.save(file, X)  # I save the dataset, so that it can be reused.
# X = np.load(file)

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

K = 2
# There is a bit of tinkering needed if terms of alpha as far as I remember
alpha = 50  # seems to be a bit faster
alpha = 1  # maybe more slower but more precise
# for rho in [0.1, 1, 10]:  # the parameters needed in the paper
rho = 1
Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                   rectZ=rect,
                                   # init='pre_init',
                                   init='full_rand',
                             alpha=alpha, cycle=100000, rho=rho, beta=0.1,
                                   output_n=1000, sigma=0.1)
W1 = Y1 @ Z1.T / N

file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}'
np.save(file, Y1)
file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}'
np.save(file, Z1)


f, rect_corr_crt = plot_clustering_results(X, Z1, W1, Y=Y1)
f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
f.show()
print(rect_corr_crt)
