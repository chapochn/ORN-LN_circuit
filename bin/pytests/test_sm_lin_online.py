#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:39:20 2019

@author: nchapochnikov

Here we are testing that the circuit is indeed doing what it is supposed to do
In particular that it is converging to the theoretical pca solution in the
linear case and to the SNMF solution in the non-negative case

# there 1 or 2 tests often don't pass exactly...
"""

import functions.circuit_simulation as FCS
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
# -----------------------------------------------------------------------------
# ########## 1. Testing the convergence of principal subspaces ################
# =============================================================================

# ##################  running the circuit with learning  ######################
circ_alg = 'inv'
NN = False
verbose = False
D = 30
K = 7
list_ds = [('SC', K), ('clus1', K), ('clus2', K), ('olf', 1), ('olf', 2),
           ('olf', 3), ('olf', 4), ('olf', 5), ('olf', 6)]

@pytest.fixture(scope='module', autouse=True)
def create_datasets():
    datasets = FD.create_datasets(D, 1000, K, options={'a': .5, 'b': 1},
                                  rho1=0.1, rho2=0.1)
    return datasets

@pytest.fixture(scope='module', autouse=True)
def get_res_off(create_datasets):
    res_svd_off: Dict[str, Dict[int, Dict[str, np.ndarray]]] = \
        collections.defaultdict(dict)

    for name, k in list_ds:
        ds = create_datasets[name]
        X = ds['ds']  # the data
        U, _, _ = LA.svd(X, full_matrices=False)
        Q_U = U[:, 0:k]
        Z = Q_U.T @ X  # ideal reconstruction error
        res_svd_off[name][k] = {'U': Q_U, 'Z': Z}

    return res_svd_off


def simulate(name, K, ds):
    X = ds['ds']  # the data
    D, N = X.shape
    message = f'Working on dataset {name}; {D}, {N}; {K}.\n'

    mon_funcs = {}
    W0, M0 = FCS.get_initial_W_M(X, K, scale=1000, type='random')
    n_epoch = int(20000 / N)
    # n_epoch = 1
    message += f'number of epochs: {n_epoch}.\n'

    time_1 = time.time()
    sm, _ = FCS.run_simulation(FCS.SM, W0, M0, X, n_epoch,
                               monitoring_functions=mon_funcs,
                               circ_alg=circ_alg)
    time_2 = time.time() - time_1
    message += f'Elapsed time: {time_2}\n'

    print(message)
    return name, K, sm


@pytest.fixture(scope='module', autouse=True)
def get_res_on(create_datasets):
    dss = create_datasets
    res_on = collections.defaultdict(dict)
    for res in Parallel(9)(delayed(simulate)(name, k, dss[name])
                           for name, k in list_ds):
        res_on[res[0]][res[1]] = res[2]
    return res_on


@pytest.mark.parametrize("ds", list_ds)
def test_online_simul(create_datasets, get_res_off, get_res_on, ds):
    name, k = ds
    print(f'Dataset {name} {k}')
    X = create_datasets[name]['ds']
    res_off = get_res_off[name][k]
    Q_U = res_off['U']  # needed to get the
    Z_off = res_off['Z']  # needed to get the cost
    sm = get_res_on[name][k]
    cost_on, cost_off, OL, perp = FCS.inspect_sm_results(sm, X, Q_U, Z_off)
    cost_ratio = (cost_on-cost_off)/cost_off
    message = (f'subspace overlap: {OL}\n'
                f'perpendicularity error: {perp}\n'
                f'Cost online, offline: {cost_on}, {cost_off}\n'
                f'cost ratio: {cost_ratio}')
    print(message)
    assert cost_ratio < 1e-3
    assert perp < 0.002
    assert OL < 0.005

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

