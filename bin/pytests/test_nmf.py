#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 16, 2019

@author: nchapochnikov


testing the offline NMF
"""
import pytest
import functions.nmf as nmf
import functions.general as FG
import numpy as np
import pandas as pd
import scipy.linalg as LA
import itertools as it
# import importlib

#
# # %%
# A = FG.rectify(np.random.rand(25, 200) + 1)
# n = 3
# # %%
# importlib.reload(FG)
#
#
#
# W, H, err, n_iter = functions.nmf.get_nmf_np(A, n=n)
# assert err == LA.norm(W @ H - A), 'error'
# print(err, err/LA.norm(A), n_iter)
#
# # %%
# importlib.reload(FG)
#
# W1, H1, err1, init_s = functions.nmf.get_nmf_best_np(A, n=n)
# print(init_s, err1)
# assert err1 == LA.norm(W1 @ H1 - A), 'error'
# assert err1 <= err, f'error {err1} should be smaller than {err}'
#
#
# if __name__ == '__main__':
#     unittest.main()
#


Ns1 = [1, 2, 4]
Ns2 = [1, 2, 3, 4, 5]  # when using the random abc
n1 = 6
n2 = 20


# #############################################################################
# ##################  with NP  ################################################
# #############################################################################

def get_abc(n):
    print(f'get_abc called with n={n}')
    A = np.arange(0, n1 * n).reshape(n1, n) / n
    B = np.arange(0, n2 * n).reshape(n, n2) / n
    return A, B, A @ B


def get_abc_rand(n):
    print(f'get_abc_rand called with n={n}')
    A = FG.rectify(np.random.randn(n1, n) + 1)
    B = FG.rectify(np.random.randn(n, n2) + 1)
    return A, B, A @ B


@pytest.mark.parametrize("n", Ns1)
def test_abc(n):
    A, B, C = get_abc(n)
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1]


@pytest.mark.parametrize("n", Ns1)
def test_abc_rand(n):
    A, B, C = get_abc_rand(n)
    print(A.shape[0], C.shape[0])
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1]


@pytest.fixture(scope='module', params=list(it.product(Ns1,
                                    [nmf.get_nmf_np, nmf.get_nmf_best_np])))
def get_nmf_in_out(request):
    print(f'generating data get_nmf_in_out with {request.param}')
    n, get_nmf = request.param
    A, B, C = get_abc(n)
    A1, B1, err, p = get_nmf(C, k=n)
    return C, A1, B1, err, p


def test_nmf1(get_nmf_in_out):
    C, A_hat, B_hat, err, p = get_nmf_in_out
    print(err, p)
    assert err == LA.norm(A_hat @ B_hat - C)


def test_nmf2(get_nmf_in_out):
    C, A_hat, B_hat, err, p = get_nmf_in_out
    assert np.max(np.abs(A_hat @ B_hat - C)) < 1e-3


@pytest.mark.parametrize("n", Ns2)
def test_nmf_best_3(n):
    A, B, C = get_abc_rand(n)
    A1, B1, err1, p1 = nmf.get_nmf_np(C, k=n)
    A2, B2, err2, p2 = nmf.get_nmf_best_np(C, k=n)
    print(err1 - err2)
    assert err1 - err2 >= -1e-14


# #############################################################################
# ##################  with DF  ################################################
# #############################################################################


def get_abc_df(n):
    A, B, C = get_abc(n)
    print(f'get_abc_df called with n={n}')
    n1 = A.shape[0]
    n2 = B.shape[1]
    A = pd.DataFrame(A, index=np.arange(n1), columns=np.arange(n) + 10)
    B = pd.DataFrame(B, index=np.arange(n) + 10, columns=np.arange(n2))
    C = pd.DataFrame(C, index=np.arange(n1), columns=np.arange(n2))
    return A, B, C


def get_abc_rand_df(n):
    A, B, C = get_abc_rand(n)
    print(f'get_abc_rand_df called with n={n}')
    n1 = A.shape[0]
    n2 = B.shape[1]
    A = pd.DataFrame(A, index=np.arange(n1), columns=np.arange(n) + 10)
    B = pd.DataFrame(B, index=np.arange(n) + 10, columns=np.arange(n2))
    C = pd.DataFrame(C, index=np.arange(n1), columns=np.arange(n2))
    return A, B, C


@pytest.mark.parametrize("n", Ns1)
def test_abc_df(n):
    A, B, C = get_abc_df(n)
    pd.testing.assert_frame_equal(A @ B, C, check_exact=True)


@pytest.mark.parametrize("n", Ns1)
def test_abc_rand_df(n):
    A, B, C = get_abc_rand_df(n)
    pd.testing.assert_frame_equal(A @ B, C, check_exact=True)


@pytest.fixture(scope='module', params=list(it.product(Ns1,
                                    [nmf.get_nmf_df, nmf.get_nmf_best_df])))
def get_nmf_in_out_df(request):
    print(f'generating data get_nmf_in_out_df with {request.param}')
    n, get_nmf = request.param
    A, B, C = get_abc_df(n)
    A1, B1, err, p = get_nmf(C, k=n)
    return C, A1, B1, err, p


def test_nmf_df1(get_nmf_in_out_df):
    C, A_hat, B_hat, err, p = get_nmf_in_out_df
    print(err, p)
    assert err == LA.norm(A_hat @ B_hat - C)


def test_nmf_df2(get_nmf_in_out_df):
    C, A_hat, B_hat, err, p = get_nmf_in_out_df
    assert np.max(np.abs(A_hat @ B_hat - C).values) < 1e-3


def test_nmf_df3(get_nmf_in_out_df):
    C, A_hat, B_hat, err, p = get_nmf_in_out_df
    pd.testing.assert_frame_equal(A_hat @ B_hat, C, check_less_precise=True)


@pytest.mark.parametrize("n", Ns2)
def test_nmf_best_df(n):
    A, B, C = get_abc_rand_df(n)
    A1, B1, err1, p1 = nmf.get_nmf_np(C, k=n)
    A2, B2, err2, p2 = nmf.get_nmf_best_np(C, k=n)
    print(err1 - err2)
    assert err1 - err2 >= -1e-14
