"""
Testing that the SNMF functions is doing what we want it to do on different
datasets

"""


import numpy as np
import pytest
from scipy import linalg as LA

import functions.datasets as FD
from functions import nmf
from functions.general import rectify
import pandas as pd
import functions.olfactory as FO
import functions.circuit_simulation as FCS

Ns = [1, 2, 4]
Ns2 = [1, 4]
n1 = 15  # this is the dimensionality of the space
rnd = np.random.RandomState(1232345)

# %%

# ##################### numpy versions  #######################################


def get_ab(n):
    """
    for some reason snmf has a hard time with this function
    Parameters
    ----------
    n

    Returns
    -------

    """
    A = rnd.permutation(np.arange(0, n1 * n).reshape(n, n1)) / n*n1
    return A, A.T @ A


def get_ab_rand(n):
    A = rectify(rnd.randn(n, n1) + 1)
    return A, A.T @ A


@pytest.fixture(scope='module', params=Ns)
def get_snmf_in_out(request):
    n = request.param
    print('creating snmf_np data')
    input = get_ab_rand(n)
    output = nmf.get_snmf_best_np(input[0], k=n, max_iter=100000, rtol=1e-7)
    return input, output


def test_snmf1(get_snmf_in_out):
    (_, B), (A_hat, err, m) = get_snmf_in_out
    print(m)
    err1 = LA.norm(A_hat.T @ A_hat - B)
    assert err == err1


def test_snmf2(get_snmf_in_out):
    (_, B), (A_hat, err, m) = get_snmf_in_out
    print(np.max(np.abs(A_hat.T @ A_hat - B)))
    assert np.max(np.abs(A_hat.T @ A_hat - B)) < 1e-4


# ##################### pandas dataframe versions  ############################

def get_ab_rand_df(n):
    A = rectify(np.random.randn(n, n1) + 1)
    A = pd.DataFrame(A, index=np.arange(n), columns=np.arange(n1) + 100)
    return A, A.T @ A


@pytest.fixture(scope='module', params=Ns2)
def get_snmf_in_out_df(request):
    n = request.param
    print('creating snmf_df data')
    input = get_ab_rand_df(n)
    output = nmf.get_snmf_best_df(input[0], k=n, max_iter=100000)
    return input, output


def test_snmf_df1(get_snmf_in_out_df):
    (_, B), (A_hat, err, m) = get_snmf_in_out_df
    err1 = LA.norm(A_hat.T @ A_hat - B)
    assert err == err1


def test_snmf_df2(get_snmf_in_out_df):
    (_, B), (A_hat, err, m) = get_snmf_in_out_df
    print(np.max(np.abs(A_hat.T @ A_hat - B).values))
    assert np.max(np.abs(A_hat.T @ A_hat - B).values) < 1e-4

# %%
# ##########################  LARGE DATASETS TESTS  ###########################


@pytest.fixture(scope='module', autouse=True)
def create_datasets():
    seed = 4321
    datasets = FD.create_datasets(D=30, N=500, K=7, options={'a': .5, 'b': 1},
                                  seed=seed, rho1=0.1, rho2=0.1)
    return datasets


@pytest.fixture(scope='module')
def get_errors():
    """
    would have been even better if you has some theoretical estimate about
    the errors, instead of relying on some particular instance of the
    random seed
    These errors have been precalculated
    Returns
    -------

    """
    mi = pd.MultiIndex(levels=[['SC', 'clus1', 'clus2', 'olf'],
                               [1, 2, 3, 4, 5, 6, 7]],
                       codes=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                               2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
                              [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1,
                               2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]])
    errors = [1137.53607409, 1099.70399457, 1067.71706905, 1041.11524297,
              1020.38990012, 1005.6874046 ,  992.39401361, 1673.68864709,
              1514.81597576, 1353.85784684, 1198.55893055, 1034.15481344,
              844.01617062,  617.41345614, 1898.08417399, 1453.49453563,
              1158.0339563 ,  961.89684678,  727.12743164,  468.76539715,
              23.22192233,  721.66398627,  432.70151956,  349.89787922,
              281.66545468,  237.71148972,  204.03424606,  170.24030922]
    errors = np.asarray(errors) * 1.005
    return pd.Series(errors, index=mi)


# def calculate_errors(A, k):
#     nmf.get_snmf_best_np(A, k, rtol=1e-6, max_iter=10000)


@pytest.mark.parametrize("k", np.arange(7) + 1)
@pytest.mark.parametrize("dataset", ['SC', 'clus1', 'clus2', 'olf'])
# @pytest.mark.parametrize("k", [2])
# @pytest.mark.parametrize("dataset", ['SC'])
def test_snmf_errors(create_datasets, get_errors, k, dataset):
    err1 = get_errors.loc[dataset, k]
    dataset = create_datasets[dataset]
    X = dataset['ds']
    # A = X.T @ X
    Y, err2, m = nmf.get_snmf_best_np(X, k, rtol=1e-6, max_iter=100000)
    print(m)
    print(k, ', error1 vs error2:', err1, err2)
    assert err1 >= err2
