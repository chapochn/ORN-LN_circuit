#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:39:20 2019

@author: nchapochnikov

Here we are testing that the circuit is indeed doing what it is supposed to do
In particular that it is converging to the theoretical pca solution in the
linear case
TODO: nonlinear case?
"""

import functions.white_circ_offline as FOC
import functions.datasets as FD
import functions.general as FG
from functions import nmf
import sklearn.decomposition as skd

import numpy as np
import matplotlib.pylab as plt
import time
import scipy.linalg as LA
import importlib
import collections
import pytest
from typing import Dict
from joblib import Parallel, delayed


# %%

# ##################  running the circuit with learning  ######################
# circ_alg = 'inv'
NN = False
verbose = False
verbose = True
D = 30
D = 20
# D = 7
K = 7
list_ds = [('SC', K), ('clus1', K), ('clus2', K), ('olf', 1), ('olf', 2),
           ('olf', 3), ('olf', 4), ('olf', 5), ('olf', 6)]
# list_ds = [('SC', K)]
# list_ds = [('clus2', K)]
# list_ds = [('clus1', K)]
# list_ds = [('olf', 1)]

GAMMA = 0.8
GAMMA = 1.2
GAMMA = 1


@pytest.fixture(scope='module')
def create_datasets():
    datasets = FD.create_datasets(D, 1000, K, options={'a': .5, 'b': 2},
                                  rho1=0.1, rho2=0.1)
    return datasets

@pytest.fixture(scope='module')
def get_res_th(create_datasets):  # theoretical results
    res_svd_off: Dict[str, Dict[int, Dict[str, np.ndarray]]] = \
        collections.defaultdict(dict)

    for name, k in list_ds:
        ds = create_datasets[name]
        X = ds['ds']  # the data
        N = X.shape[1]
        U, s, Vt = LA.svd(X, full_matrices=False)
        s_y = s.copy()
        s_y[:k] = FOC.damp_sx(s_y[:k], N)
        Y = U @ np.diag(s_y) @ Vt
        Q_U = U[:, :k]
        s_z = s.copy()
        s_z = FOC.sz_from_sx(s_z, N, rho=1, gamma=1)
        Z = np.diag(s_z[:k]) @ Vt[:k]  # without any rotation
        res_svd_off[name][k] = {'U': Q_U, 'Z': Z, 'Y': Y, 's_x':s}

    return res_svd_off


def simulate1(name, K, ds):
    X = ds['ds']  # the data
    D, N = X.shape
    message = f'Working on dataset {name}; {D}, {N}; {K}.\n'
    rect = False

    time_1 = time.time()
    cycle = 100000
    alpha = 10
    if name == 'olf' and K==1: # has issues converging
        cycle = 30000
        alpha = 100
    Y, Z, costs = FOC.olf_gd_offline(X, K, max_iter=300000, rectY=rect,
                                     rectZ=rect, alpha=alpha, rtol=1e-7,
                                     gamma=GAMMA, cycle=cycle)  # rho is 1 i suppose
    time_2 = time.time() - time_1
    message += f'Elapsed time: {time_2}\n'

    print(message)
    # plt.figure()
    # plt.plot(costs)
    # plt.show()
    return name, K, Y, Z


def simulate2(name, K, ds):
    X = ds['ds']  # the data
    D, N = X.shape
    message = f'Working on dataset {name}; {D}, {N}; {K}.\n'
    rect = False

    time_1 = time.time()
    alpha = 1
    Z, costs = FOC.olf_gd_offline2(X, K, max_iter=1000, rect=rect, cycle=10000,
                                   sigma=0.005, alpha=alpha,
                                   rtol=0)  # not sure why sigma needs to be so small here...
    time_2 = time.time() - time_1
    message += f'Elapsed time: {time_2}\n'
    Y = X @ LA.inv(Z.T @ Z / N + np.eye(N))

    print(message)
    # plt.figure()
    # plt.plot(costs)
    # plt.show()
    return name, K, Y, Z


@pytest.fixture(scope='module', params=[simulate1, simulate2])
# @pytest.fixture(scope='module', params=[simulate1])
# @pytest.fixture(scope='module', params=[simulate2])
def get_res_off(create_datasets, request):
    simulate = request.param
    dss = create_datasets
    res_on = collections.defaultdict(dict)
    for res in Parallel(9)(delayed(simulate)(name, k, dss[name])
                           for name, k in list_ds):
        res_on[res[0]][res[1]] = (res[2], res[3])
    return res_on


@pytest.mark.parametrize("ds", list_ds)
def test_offline_simul(create_datasets, get_res_th, get_res_off, ds):
    name, k = ds
    print(f'Dataset {name} {k}')
    X = create_datasets[name]['ds']
    res_th = get_res_th[name][k]
    U_th = res_th['U']  # needed to get the subspace
    Z_th = res_th['Z']  # needed to get the cost
    Y_th = res_th['Y']  # needed to get the cost
    print(U_th.shape, Z_th.shape, Y_th.shape)
    Y_off, Z_off = get_res_off[name][k]
    W = X @ Z_off.T / X.shape[1]
    U_off, _, _ = LA.svd(W, full_matrices=False)

    Y_overlap = FG.subspc_overlap(U_off, U_th, scaled=1)
    print(f'overlap: {Y_overlap}')
    diff_Y = np.max(np.abs(Y_th - Y_off))
    print(f' max diff between theoretical and sim Y: {diff_Y}')

    # the diff in Z doesn't mean much, because there can be a rotation in Z
    # diff_Z = np.max(np.abs(Z_th - Z_off))
    # print(f' max diff between theoretical and sim Z: {diff_Z}')
    cost_th = FOC.olf_cost(X, Y_th, Z_th)
    cost_off = FOC.olf_cost(X, Y_off, Z_off, gamma=GAMMA)
    cost_ratio = (cost_th-cost_off)/cost_th
    print(f'costs theoretical, offline: {cost_th}, {cost_off}')
    print(f'costs relative error: {cost_ratio}')
    cost_th = FOC.olf_cost2(X, Z_th)
    cost_off = FOC.olf_cost2(X, Z_off)
    cost_ratio = (cost_th-cost_off)/cost_th
    print(f'costs theoretical, offline: {cost_th}, {cost_off}')
    print(f'costs relative error: {cost_ratio}')

    # N = X.shape[1]
    # Y_new = X @ LA.inv(Z_th.T @ Z_th / N + np.eye(N))

    _,  S_Y_off, _  = LA.svd(Y_off, full_matrices=False)
    _,  S_Y_th, _  = LA.svd(Y_th, full_matrices=False)
    # _,  S_Y_new, _  = LA.svd(Y_new, full_matrices=False)
    print('X singular values original:', res_th['s_x'])
    print('Y singular values theoretical:', S_Y_th)
    print('Y singular values offline:', S_Y_off)
    # print('Y singular values new:', S_Y_new)
    print('Y singular values difference:', S_Y_off- S_Y_th)

    _,  S_Z_off, _  = LA.svd(Z_off, full_matrices=False)
    _,  S_Z_th, _  = LA.svd(Z_th, full_matrices=False)
    print('Z singular values theoretical:', S_Z_th)
    print('Z singular values offline:', S_Z_off)
    # print('singular values difference:', S_Y_off- S_Y_th)

    assert Y_overlap < 5e-3
    assert diff_Y < 0.07

# it could make sense to actually have the results per dataset instead of
# of the global constrains

# # %%
# for name, k in list_ds:
#     ds = datasets[name]
#     X = ds['ds']  # the data
#     Z_off = res_svd_off[name][k]['Z']
#     sm, mon = results[name][k]
#     FCS.display_sm_results(sm, X, Z_off, mon, plot=True, name=f'{name}_{k}')
# basically, as "result" is that all of the above examples should "converge"
# apart from the olf3 and olf6

