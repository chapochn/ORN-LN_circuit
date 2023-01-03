"""
@author: Nikolai M Chapochnikov
Created in 2021
Testing the subspc_overlap functions

ALL TESTS PASS
"""


import functions.general as FG
import numpy as np
import scipy.linalg as LA
import pytest



NS = [(200, 20, 10), (200, 1, 1), (124, 1, 10), (70, 7, 7)]

@pytest.fixture(scope='module', params=NS)
def get_ab(request):
    n1, n2, n3 = request.param
    A = np.random.randn(n1, n2)
    A, _ = LA.qr(A, mode='economic')
    B = np.random.randn(n1, n3)
    B, _ = LA.qr(B, mode='economic')
    err1 = LA.norm(A @ A.T - B @ B.T)**2
    return A, B, err1

def test_subspc_overlap0(get_ab):
    A, B, err1 = get_ab
    err2 = FG.subspc_overlap(A, B, scaled=0)
    err = abs(err1-err2)
    print(err)
    assert err < 1e-13

def test_subspc_overlap1(get_ab):
    A, B, err1 = get_ab
    m = np.abs(A.shape[1] - B.shape[1])
    M = A.shape[1] + B.shape[1]
    err2 = FG.subspc_overlap(A, B, scaled=1)
    assert m/M <= err2
    assert err2 <= 1
    err = abs(err1/M-err2)
    print(err)
    assert err < 1e-13


def test_subspc_overlap2(get_ab):
    A, B, err1 = get_ab
    err2 = FG.subspc_overlap(A, B, scaled=2)
    assert 0 <= err2
    assert err2 <= 1
# N = 10
# print(timeit.timeit('FCS.subspc_overlap(A, B, relative_error_flag=False)',
#       globals=globals(), number=N))
# print(timeit.timeit('FG.subspc_overlap(A, B, scaled=False)',
#                     globals=globals(), number=N))



