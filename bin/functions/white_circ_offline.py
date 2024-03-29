"""
@author: Nikolai M Chapochnikov
Created in 2022
"""


import numpy as np
from scipy import linalg as LA
from functions.nmf import get_init_matrix
from functions.general import rectify, FLOAT_TYPE, EPSILON
from typing import Tuple

def mat_mul(A, B, alpha=1):
    """
    https://www.benjaminjohnston.com.au/matmul
    """
    return alpha * np.matmul(A, B)


def damp_sx(sx, T, rho=1):
    """
    dampening the singular values as expected in the linear whitening circuit
    Parameters
    ----------
    sx
        vector of singular values
    T
        number of time points
    Returns
    -------

    """
    max_value = np.sqrt(T)/rho
    return np.minimum(sx, max_value)

def sz_from_sx(sx, T, rho=1, gamma=1):
    """

    :param sx:
    :param T:
    :param rho:
    :param gamma:
    :return:
    """
    value = (sx * rho/np.sqrt(T) - 1) * T/gamma**2
    sz2 = np.maximum(value, 0)
    return np.sqrt(sz2)


def olf_cost(X, Y, Z, rho=1, gamma=1.):
    N = X.shape[1]
    XmY = X-Y
    ret1 = 1/(2*N) * np.sum(XmY**2)

    YZT = mat_mul(Y, Z.T)
    # ZZT = mat_mul(Z, Z.T)
    ret2 = 1/(2 * N**2) * gamma**2 * np.sum(YZT**2)
    ret3 = -1/(2 * N * rho**2) * gamma**2 * np.sum(Z**2)
    return ret1 + ret2 + ret3

def get_grad_Y(X, Y, Z, gamma=1):
    N = X.shape[1]
    return (Y-X) / N + gamma**2 * mat_mul(mat_mul(Y, Z.T), Z) / (N ** 2)


def get_grad_Z(Y, Z, rho=1, gamma=1):
    ZYT = mat_mul(Z, Y.T)
    N = Z.shape[1]
    return gamma**2 / (N ** 2) * (mat_mul(ZYT, Y)) \
           - gamma**2 * Z/(N * rho**2)


def gd_step_Y(X, Y_old, Z, cost_old, sigma, beta, alpha=1, rect=False, rho=1,
              gamma=1):
    """
    gradient descent in Y
    Parameters
    ----------
    sigma:
        acceptance parameter. coefficient by which the gradient is multiplied
        to set the expected cost change
    beta:
        reduction parameter, coefficient by which alpha is multiplied at each
        iteration, so that the gradient step becomes smaller
    alpha: float
        original coefficient by which the gradient is multiplied
    rho:
        the rho that is in the cost function, which is equivalent in terms of
        effect it has on the computation with scaling the input
    Returns
    -------

    """
    # alpha = 1
    if rect:
        func = rectify
    else:
        func = lambda x: x
    success = False
    grad_Y = get_grad_Y(X, Y_old, Z, gamma=gamma)
    while alpha > EPSILON*0.000001:
        Y_new = func(Y_old - alpha * grad_Y)
        expected_cost_decrease = sigma * np.sum(grad_Y * (Y_new - Y_old))
        cost_new = olf_cost(X, Y_new, Z, rho=rho, gamma=gamma)
        # print('GD in Y, expected_cost_decrease vs true',
        #       expected_cost_decrease, cost_new - cost_old)
        if cost_new - cost_old < expected_cost_decrease:
            success = True
            break
        alpha *= beta
    # print(f'alpha in gd_step_Y: {alpha}')
    if not success:
        return Y_old, cost_old, success, alpha
    else:
        return Y_new, cost_new, success, alpha


def gd_step_Z(X, Y, Z_old, cost_old, sigma, beta, alpha=1, rect=False, rho=1,
              gamma=1):
    """
    gradient ascent in Z
    Parameters
    ----------
    sigma:
        acceptance parameter. coefficient by which the gradient is multiplied
        to set the expected cost change
    beta:
        reduction parameter, coefficient by which alpha is multiplied at each
        iteration, so that the gradient step becomes smaller
    alpha: float
        original coefficient by which the gradient is multiplied
    rho:
        the rho that is in the cost function, which is equivalent in terms of
        effect it has on the computation with scaling the input
    Returns
    -------

    """
    # alpha = 1
    if rect:
        func = rectify
    else:
        func = lambda x: x
    success = False
    grad_Z = get_grad_Z(Y, Z_old, rho=rho, gamma=gamma)
    while alpha > EPSILON*0.000001:
        Z_new = func(Z_old + alpha * grad_Z)
        expected_cost_increase = sigma * np.sum(grad_Z * (Z_new - Z_old))
        cost_new = olf_cost(X, Y, Z_new, rho=rho, gamma=gamma)
        # print('GD in Z, expected_cost_increase vs true',
        #       expected_cost_increase, cost_new - cost_old)
        if cost_new - cost_old > expected_cost_increase:
            success = True
            break
        alpha *= beta
    # print(f'alpha in gd_step_Z: {alpha}')
    if not success:
        return Z_old, cost_old, success, alpha
    else:
        return Z_new, cost_new, success, alpha




def olf_gd_offline(X: np.ndarray, k: int, rtol: float = 1e-6,
                   max_iter: int = 100000,
                   rectY: bool = False, rectZ: bool = False,
                   init: str = 'random', Y0=None, Z0=None,
                   verbose: bool = False, alpha=1,
                   cycle=500, rho=1, beta=0.1, gamma=1, output_n=100, sigma=0.1) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradient descent to calculate the solution of the olfactory
    cost function

    Parameters
    ----------
    X:
        input data matrix
    k:
        number of LNs
    etamax:
        the initial learning rate
    rtol:
        relative tolerance
    init:
        initialization for the Y and Z matrices
    sigma:
        acceptance parameter

    Returns
    ------
    Y and Z
        Matrices minimizing the olf cost function
    """

    D, N = X.shape  # n_s is the number of samples
    if init != 'given':
        Y = get_init_matrix(D, N, method=init, A=X, scale=100, rect=rectY)
        # Y = X  # i wonder if you should do the simulation with the random matrix,
        # so that is it not "cheating" by already taking a solution which resembles
        # what you think the solution should be.
        if rectY:
            Y = rectify(Y)
        # Y = X
        Z = get_init_matrix(k, N, method=init, A=X, scale=100, rect=rectZ)/gamma
        # Y = np.array(Y, dtype=FLOAT_TYPE)
        # A = np.array(A, dtype=FLOAT_TYPE)
        # X = np.array(X, dtype=FLOAT_TYPE, order='F')
        # Y = np.array(Y, dtype=FLOAT_TYPE, order='F')
        # Z = np.array(Z, dtype=FLOAT_TYPE, order='F')
    else:
        Y = Y0
        Z = Z0
    cost0 = olf_cost(X, Y, Z, rho=rho, gamma=gamma)
    cost2 = cost0
    # print(f'initial reconstruction error {LA.norm(A - Y.T @ Y)}')

    # eye = np.eye(N)/N
    costs = np.zeros(max_iter)
    # eye = np.array(eye, dtype=FLOAT_TYPE, order='F')
    for i in range(max_iter):
        costs[i] = cost2
        Y_old = Y.copy()
        Z_old = Z.copy()
        cost_old = cost2.copy()
        mess = f'i: {i}\ncost1: {cost2}, '  # concatenating messages
        Y, cost1, successY, a1 = gd_step_Y(X, Y, Z, cost2, sigma, beta, alpha,
                                           rectY, rho=rho, gamma=gamma)
        mess += f'cost2: {cost1}, '

        Z, cost2, successZ, a2 = gd_step_Z(X, Y, Z, cost1, sigma, beta, alpha,
                                           rectZ, rho=rho, gamma=gamma)
        mess += f'cost3: {cost2}, '

        conv1 = np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))
        conv2 = np.amax(np.abs(Z-Z_old) / np.abs(Z_old + 1e-2))
        # i imagine the 1e-2 here is to not have any divergence problem?
        mess += f'convergences y, z : {conv1}, {conv2}, alphas: {a1}, {a2}, '

        if not successY and not successZ:
            print(mess)
            print('stopped because both gd steps were unsuccessfull')
            break

        d_cost1 = np.abs(cost_old - cost1) / np.abs(cost_old)
        d_cost2 = np.abs(cost1 - cost2) / np.abs(cost1)
        mess += f'deltacosts after y, z = {d_cost1}, {d_cost2}'
        if d_cost1 < rtol and d_cost2 < rtol and conv1 < rtol and conv2 < rtol:
            print(mess)
            print('stopped because costs and Y and Z stopped changing')
            break
        # ctol = 10**-5
        # if :
        #     print(mess)
        #     print('stopped because Y and Z stopped changing')
        #     break

        if i % output_n == 0:
            print(mess)

        if i % cycle == 0 and i > 0:
            alpha *= beta
            # print('i, costs:', cost_old, cost1, cost2)
    print(f'i: {i}, costs: {cost0}, {cost2}')

    return Y, Z, costs[:i+1]


# #############################################################################
# #############################################################################
# !!!!!!!!!!!!!!!!!!!   algorithm only in Z    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def olf_cost2(X, Z, rho=1) -> float:
    """
    this is the cost function that needs to be minimized in the case when
    Y has been factored out and the function has been multiplied by N
    1/2 X\T X (Z\TZ/N + I)^-1 + 1/(N*4*rho^2) Z\TZZ\TZ)
    Parameters
    ----------
    X
    Z
    rho
    Returns
    -------

    """
    N = X.shape[1]
    ZTZ = mat_mul(Z.T, Z)
    inv = LA.inv(ZTZ/N + np.eye(N))
    ret1 =  mat_mul(X.T, X) @ inv
    ret2 =  ZTZ

    return 1/2 * np.trace(ret1) + 1/(2 * rho**2) * np.trace(ret2)
#TODO:check this formula


def get_grad_Z2(X, Z, rho=1):
    """
    we will do a gradient descent on the funtion above, but we will put back
    the 1/N everywhere.
    Parameters
    ----------
    X
    Z

    Returns
    -------

    """
    N = X.shape[1]
    ZTZ = mat_mul(Z.T, Z)
    inv = LA.inv(ZTZ/N + np.eye(N))
    grad = - Z @ inv @ mat_mul(X.T, X) @ inv/N + Z /(rho**2)
    #TODO: check this formula
    return grad


def gd_step_Z2(X, Z_old, cost_old: float, sigma: float, beta: float,
               alpha: float = 1, rect: bool = False, rho: float=1):
    """
    gradient descent in Z for the olfactory function which depends on
    Z only, and where the dependence on Y has been removed. Works only if
    the cost function is linear in Y, but can be non-negative or linear in
    Z.
    Parameters
    ----------
    X:
        data matrix
    Z_old:
        previous Z matrix
    cost_old:
        previous cost
    sigma:
        factor by which the linear expectation of the decrease in cost
        is multiplied before comparing with the actual cost decrease
    beta:
        reduction factor, by which alpha is multiplied at each iteration
    alpha:
        initial step size, by which the gradient is multiplied
    rect:
        whether Z is rectified or not
    rho:
        strength of the inhibition in the cost function


    Returns
    -------

    """
    # alpha = 1
    success = False
    if rect:
        func = rectify
    else:
        func = lambda x: x
    grad_Z = get_grad_Z2(X, Z_old, rho)
    for i in range(50):
        Z_new = func(Z_old - alpha * grad_Z)
        expected_cost_decrease = sigma * np.sum(grad_Z * (Z_new - Z_old))
        cost_new = olf_cost2(X, Z_new, rho)
        print('alpha, cost decrease: expected, actual:', alpha, expected_cost_decrease,
              cost_new - cost_old)
        if cost_new - cost_old < expected_cost_decrease:
            success = True
            break
        alpha *= beta
    print(f'alpha in gd_step_Z: {alpha}')
    if not success:
        return Z_old, cost_old, success
    else:
        return Z_new, cost_new, success


def olf_gd_offline2(X: np.ndarray, k: int, rtol: float = 1e-6,
                    max_iter: int = 100000,
                    init: str = 'random', verbose: bool = False,
                    rect: bool=False, alpha: float = 1, cycle: int = 500,
                    rho: float = 1, sigma=0.1, beta = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient descent to calculate the solution of the olfactory
    cost function when it only depends on Z, and where the dependence on
    Y has been eliminated. This can only be used in the linear case. In the
    case where Y is rectified, Y cannot be replaced in the cost function.
    In the case where only Z is rectified, this scheme can be used.

    Parameters
    ----------
    X:
        input data matrix
    k:
        number of LNs
    rtol:
        relative tolerance for the decrease in the cost after which the
        algorithm stops
    max_iter:
        maximum number of iteration of the algorithm
    init:
        initialization for the Z matrices
    rect:
        whether Z needs to be rectified or not
    alpha:
        the initial learning rate
    cycle:
        every how many iterations alpha is decreased by beta (here 0.1)

    Returns
    ------
    Z, consts
        Matrice minimizing the olf cost function, list of costs
    """

    D, N = X.shape  # n_s is the number of samples

    Z = get_init_matrix(k, N, method=init, A=X, rect=rect)
    # Z = np.ones((k, N))/1000#get_init_matrix(k, N, method=init, A=X, rect=rect)
    cost0 = olf_cost2(X, Z)
    cost1 = cost0
    # print(f'initial reconstruction error {LA.norm(A - Y.T @ Y)}')
    # sigma = 0.1


    costs = np.zeros(max_iter)
    for i in range(max_iter):
        costs[i] = cost1
        Z_old = Z.copy()
        cost_old = cost1.copy()
        print(f'i: {i}\ncost1: {cost1}')
        Z, cost1, success = gd_step_Z2(X, Z, cost1, sigma, beta,
                                       alpha=alpha, rect=rect, rho=rho)
        print(f'cost2: {cost1}')

        conv1 = np.amax(np.abs(Z-Z_old) / np.abs(Z_old + 1e-2))
        # conv as convergence
        print(f'convergences in Z {conv1}\n')

        if not success:
            print('stopped because the gradient descent was not successful')
            break

        # d_Z = np.max(np.abs(Z_old - Z))
        # print('change in Z:', d_Z)
        d_cost = np.abs(cost_old - cost1) / np.abs(cost_old)
        if  d_cost < rtol:
            print('stopped because the decrease in cost was too small:', d_cost)
            break

        # This is actually a very important part of the algorithm, otherwise
        # the algorithm can just be turning in circles
        if (i+1) % cycle == 0: # the + 1 is so that it doesn't change the
            # alpha at the first iteration
            alpha *= beta
    print(f'i: {i}, costs: {cost0}, {cost1}')
    return Z, costs[:i+1]


