"""
Created in 2022

@author: Nikolai M Chapochnikov


This file generates the files that are in results/dataset4

"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import importlib

import functions.nmf as nmf
import functions.white_circ_offline as FOC  ### !!! different FOC !!!
import functions.olfactory as FO
import functions.datasets as FD
import functions.plotting as FP


# %%

importlib.reload(FOC)
importlib.reload(nmf)



#%%
# #############################################################################
# #############################################################################
# #########  For supplementary figure showing circuit with no M   #############
# ##################   DIFFERENT OPTIMIZATION PROBLEM  ########################
# #############################################################################
# #############################################################################

# Dataset with 2 clusters
importlib.reload(FD)

path_res = FO.OLF_PATH / 'results'

D = 2
n_clusters = 2

data_i, dx = 4, 0
data_path = FO.OLF_PATH / f'results/dataset{data_i}'
data_path.mkdir(exist_ok=True)

file = path_res / f'dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'

try:
    X = np.load(file)
    print('dataset already exists, it is loaded')
except:
    print('dataset does not exist yet, it will be created and saved')
    X = FD.create_clusters_data(D, [100, 100], n_clusters, dx=dx, scale=0.3,
                                x_max=2)
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
# results from the circuit with given rho similarity matching function
importlib.reload(FOC)
rect = True

K = 2

alpha = 10
rho = 1
Y1, Z1, costs = FOC.olf_gd_offline(X, K, max_iter=100000, rectY=rect,
                                   rectZ=rect,
                                   # init='pre_init',
                                   init='random',
                             alpha=alpha, cycle=1000, rho=rho, beta=0.5,
                                   output_n=1000, sigma=0.1,
                                   rtol=1e-7)
W1 = Y1 @ Z1.T / N

file = path_res / f'dataset{data_i}/whiteNNC_Y_K{K}_rho{rho}'
np.save(file, Y1)
file = path_res / f'dataset{data_i}/whiteNNC_Z_K{K}_rho{rho}'
np.save(file, Z1)


f, rect_corr_crt = FP.plot_clustering_results(X, Z1, W1, Y=Y1)
f.suptitle(f'rho:{rho}, alpha:{alpha}, cost:{costs[-1]}, corr:{rect_corr_crt}')
f.show()
print(rect_corr_crt)
