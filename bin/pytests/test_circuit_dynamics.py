#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:39:20 2019

@author: nchapochnikov


Testing if sm_output_online gives the same results with the methods
inv, GD and CD

Starting with some random M and W matrices and some random x vectors
and see if it always converges to the same y.

I think the best would be to test all the methods in circuit dynamics
meaning also testing that CD_NN works well, is it possible to test that Minden
works well?
You are not planning to use Minden, so maybe it doesn't matter too much
but maybe would be good to for example test that CD_NN gives a better
minimization result than the other algorithm while rectifying the y.

You can do all this for random vectors, but also for your olfactory condition,
so as to be sure that that particular case works as well.

What this function is doing is the following:
generating a spd matrix M and a random matrix W and a random vector x
then calculating:
y1: found using the inverse
y2: found using gradient descent
y3: found using coordinate descent
y4: found using GD and a non-negative (NN) constrain on each step
y5: found using CD and a NN constraint on each step
y6 = rectify(y1)

we check that y1, y2, and y3 are close to each other
that y4 and y5 are close to each other
(however i am not really sure that y4 and y5 should really always be the same)
that err1 < err4 < err6

"""

import functions.circuit_simulation as FCS
import functions.general as FG

import numpy as np
# import time
import importlib
import pytest
# from joblib import Parallel, delayed

# %%
importlib.reload(FCS)

# %%

np.random.seed(654342)
# verbose = True
verbose = False
check_M = True


def cost_fun(W: np.ndarray, M: np.ndarray, x: np.ndarray, y: np.ndarray)\
        -> float:
    """
    cost function related to the minimization of y
    """
    return - 4 * W @ y @ x + 2 * M @ y @ y


def cost_fun_olf(W: np.ndarray, M: np.ndarray, x: np.ndarray, y: np.ndarray,
             z: np.ndarray, rho) \
        -> float:
    """
    cost function related to the minimization of y
    """
    return -x @ y + 1/2 * y @ y + W @ z @ y - 1/(2*rho) * M @ z @ z


def get_ys_sm(x, W, M):
    atol_conv = 1e-9  # conv as convergence
    y1 = FCS.sm_output_online(x, W, M, method='inv', verbose=verbose,
                              check_M=check_M)

    y2 = FCS.sm_output_online(x, W, M, method='GD', verbose=verbose,
                              atol=atol_conv)
    y3 = FCS.sm_output_online(x, W, M, method='CD', verbose=verbose,
                              n_iter_max=1000,
                              atol=atol_conv)
    y4 = FCS.sm_output_online(x, W, M, method='CD_NN', verbose=verbose,
                              atol=atol_conv)
    y5 = FCS.sm_output_online(x, W, M, method='GD_NN', verbose=verbose,
                              n_iter_max=1000,
                              atol=atol_conv)
    y6 = FG.rectify(y1)  # this shows that just taking the element wise
    # minimum does not get us where we want, the CD finds a better value for y
    return y1, y2, y3, y4, y5, y6


def get_sol_olf_circ(x, W, M, n_iter_max=int(1e6)):
    atol_conv = 1e-9  # conv as convergence
    y1, z1 = FCS.olf_output_online(x, W, W, M, method='inv', verbose=verbose,
                                   check_M=check_M, rho=1)

    y2, z2 = FCS.olf_output_online(x, W, W, M, method='GD', verbose=verbose,
                                   atol=atol_conv, rho=1,
                                   n_iter_max=n_iter_max)
    # y3 = FCS.sm_output_online(x, W, M, method='CD', verbose=verbose,
    #                           n_iter_max=1000, atol=atol_conv)
    # y4 = FCS.sm_output_online(x, W, M, method='CD_NN', verbose=verbose,
    #                           atol=atol_conv)
    y5, z5 = FCS.olf_output_online(x, W, W, M, method='GD_NN', verbose=verbose,
                                   rho=1, atol=atol_conv)
    # y6 = FG.rectify(y1)  # this shows that just taking the element wise
    # # minimum does not get us where we want, the CD finds a better value for y
    return (y1, z1), (y2, z2), (y5, z5) #, y3, y4, y5, y6


def get_data(D, K):
    eig_vals = np.abs(np.random.randn(K)) * 10. + 1
    # if you want to make it crash, put -1 instead of +1
    M = FG.get_random_spd_matrix(K, eig_vals)
    W = np.random.randn(D, K) * 10
    x = np.random.randn(D) * 10
    # M = np.asarray(M, dtype=np.float32)
    # W = np.asarray(W, dtype=np.float32)
    # x = np.asarray(x, dtype=np.float32)
    return x, W, M


D = 96
K = 30
N = 100
D = 21
K = 8
N = 10
atol_test = 1e-8


# it could have been cleaner to use the params keyword for the fixture
# but then it would not be possible to run the thing in parallel.
@pytest.fixture(scope='module', params=np.arange(N))
def get_dataset_sm():
    print('creating datasets')
    # the parallel should not be used for the input, as it creates the problem
    # of similar outputs as different
    # inputs = list(Parallel(8)(delayed(get_data)(D, K) for _ in range(N)))
    # inputs = [get_data(D, K) for _ in range(N)]
    # outputs = [get_ys(*inputs[i]) for i in range(N)]
    # outputs = list(Parallel(10)(delayed(get_ys)(*inputs[i]) for i in range(N)))
    # # outputs = None
    # return inputs, outputs
    input = get_data(D, K)
    return input, get_ys_sm(*input)

@pytest.fixture(scope='module', params=np.arange(N))
def get_dataset_olf():
    print('creating datasets')
    # the parallel should not be used for the input, as it creates the problem
    # of similar outputs as different
    # inputs = list(Parallel(8)(delayed(get_data)(D, K) for _ in range(N)))
    # inputs = [get_data(D, K) for _ in range(N)]
    # outputs = [get_ys(*inputs[i]) for i in range(N)]
    # outputs = list(Parallel(10)(delayed(get_ys)(*inputs[i]) for i in range(N)))
    # # outputs = None
    # return inputs, outputs
    input = get_data(D, K)
    return input, get_sol_olf_circ(*input, n_iter_max=1000000)
np.random.seed()

# %%
# this is to check the improvement in speed when using the Parallel in
# get_dataset function
# def get_dataset():
#     print('creating datasets')
#     # inputs = list(Parallel(10)(delayed(get_data)(D, K) for _ in range(N)))
#     # need to check if the parallel returns same stuff
#     inputs = [get_data(D, K) for _ in range(N)]
#     outputs = [get_ys(*inputs[i]) for i in range(N)]
#     # outputs = list(Parallel(10)(delayed(get_ys)(*inputs[i]) for i in range(N)))
#     return inputs, outputs
# D = 100
# K = 30
# N = 100
# atol_test = 1e-8
# time0 = time.time()
# get_dataset()
# time1 = time.time()
# print(time1-time0)

# %%
# no need to check the randomness if we generate the samples independently
# def test_dataset_randomness(get_dataset):
#     """
#     this tests that no 2 generated vectors are the same in the randomly
#     generated vectors, This can happen when one uses the Parallel for
#     generating samples, as they start with the same seed
#     Parameters
#     ----------
#     get_dataset
#
#     Returns
#     -------
#
#     """
#     X = np.empty((D, N))
#     for i in range(N):
#         X[:, i] = get_dataset[0][i][0]
#     corr = np.corrcoef(X.T)
#     corr -= np.diag(np.diag(corr))
#     print(np.sum(corr.flatten() == 1))
#     assert not any(corr.flatten() == 1)


def test_ys1(get_dataset_sm):
    # x, W, M = get_dataset_sm[0]
    y1, y2, y3, y4, y5, y6 = get_dataset_sm[1]
    # print(cost_fun(W, M, x, y1), cost_fun(W, M, x, y2))
    assert np.allclose(y1, y2, atol=atol_test), y1-y2


def test_ys2(get_dataset_sm):
    # x, W, M = get_dataset_sm[0]
    y1, y2, y3, y4, y5, y6 = get_dataset_sm[1]
    # print(cost_fun(W, M, x, y1), cost_fun(W, M, x, y3))
    assert np.allclose(y1, y3, atol=atol_test), y1-y3


def test_ys3(get_dataset_sm):
    # x, W, M = get_dataset_sm[0]
    y1, y2, y3, y4, y5, y6 = get_dataset_sm[1]
    # print(cost_fun(W, M, x, y4), cost_fun(W, M, x, y5))
    assert np.allclose(y4, y5, atol=atol_test), y4-y5


def test_ys4(get_dataset_sm):
    x, W, M = get_dataset_sm[0]
    y1, y2, y3, y4, y5, y6 = get_dataset_sm[1]
    assert cost_fun(W, M, x, y1) < cost_fun(W, M, x, y4)


def test_ys5(get_dataset_sm):
    x, W, M = get_dataset_sm[0]
    y1, y2, y3, y4, y5, y6 = get_dataset_sm[1]
    assert cost_fun(W, M, x, y4) < cost_fun(W, M, x, y6)


# Although the test "fails" the error is quite small
# the main issue is with the eta in the olf_output_online
# function
def test_y_olf_circ(get_dataset_olf):
    (y1, z1), (y2, z2), _ = get_dataset_olf[1]
    assert np.allclose(y1, y2, atol=atol_test), y1-y2
    # assert np.allclose(z2, z1, atol=atol_test), z1-z2

# On the other hand, these pass for some reason...
def test_z_olf_circ(get_dataset_olf):
    (y1, z1), (y2, z2), _ = get_dataset_olf[1]
    # assert np.allclose(y2, y1, atol=atol_test), y1-y2
    assert np.allclose(z1, z2, atol=atol_test), z1-z2
