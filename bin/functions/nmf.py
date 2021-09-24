"""
Contains functions that are for NMF and SNMF
@author: Nikolai M Chapochnikov

"""
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import linalg as LA
from sklearn import decomposition as skd

from functions.general import rectify, FLOAT_TYPE, mat_mul, EPSILON,\
    subspc_overlap, get_norm_np



# #############################################################################
# ##########################  NMF ANALYSIS  ###################################
# #############################################################################


def get_nmf_np(A: np.ndarray, k: int = 2, beta: float = 2,
               init: str = 'nndsvd', rnd=None, max_iter: int = 10000):
    solver = 'cd'
    if beta != 2:
        solver = 'mu'
    print(f'NMF with k={k}. solver: {solver}, beta={beta}')
    NMFmodel = skd.NMF(n_components=k, random_state=rnd, beta_loss=beta,
                       solver=solver, init=init, max_iter=max_iter, alpha=0)

    W = NMFmodel.fit_transform(A)  # not used later, but needed to calculate
    H = NMFmodel.components_
    err = NMFmodel.reconstruction_err_
    n_iter = NMFmodel.n_iter_
    return W, H, err, n_iter


def get_nmf_best_np(A: np.ndarray, k: int = 2, beta: float = 2,
                    n_rnd: int = 10, max_iter: int = 10000):
    """
    N is the order of the NMF
    beta is the parameter in the optimization function of the NMF
    n_rnd is the number of random initialization in addition on the
    deterministic initializations to try in order to get the best NMF
    (i.e., with the smallest error according to the objective function)
    """
    # different initializations
    inits = ['nndsvd', 'nndsvda', 'nndsvdar'] + list(range(n_rnd))
    n_inits = len(inits)

    Ws = np.zeros((n_inits, A.shape[0], k))
    Hs = np.zeros((n_inits, k, A.shape[1]))
    errs = np.zeros(n_inits)

    for i, init in enumerate(inits):
        if isinstance(init, int):
            init = 'random'
        Ws[i], Hs[i], errs[i], _ = get_nmf_np(A, k=k, beta=beta, init=init,
                                              max_iter=max_iter)
    idxmin = np.argmin(errs)
    if 'nndsvd' not in str(inits[idxmin]):
        init_s = 'random'
    else:
        init_s = inits[idxmin]
    print(f'best initialization: {init_s}')
    return Ws[idxmin], Hs[idxmin], errs[idxmin], init_s


def get_nmf_df(A: pd.DataFrame, **kwargs)\
        -> Tuple[pd.DataFrame, pd.DataFrame, float, int]:
    """
    calculates the NMF
    Parameters
    ----------
    A
    k
    beta
    init
    rnd

    Returns
    -------

    """
    W, H, err, n_iter = get_nmf_np(A, **kwargs)
    k = kwargs['k']
    W_df = pd.DataFrame(W, index=A.index, columns=np.arange(k) + 1)
    W_df.index.name = A.index.name
    W_df.columns.name = 'N'

    H_df = pd.DataFrame(H, columns=A.columns, index=np.arange(k) + 1)
    H_df.columns.name = A.columns.name
    H_df.index.name = 'N'

    return W_df, H_df, err, n_iter


def get_nmf_best_df(A: pd.DataFrame, **kwargs):
    """
    wrapper around get_nmf_best_np
    ----------
    A
    k
    beta
    n_rnd

    Returns
    -------

    """
    res = get_nmf_best_np(A, **kwargs)
    k = kwargs['k']
    W, H, err, init = res
    W_df = pd.DataFrame(W, index=A.index, columns=np.arange(k) + 1)
    W_df.index.name = A.index.name
    W_df.columns.name = 'N'

    H_df = pd.DataFrame(H, index=np.arange(k) + 1, columns=A.columns)
    H_df.columns.name = A.columns.name
    H_df.index.name = 'N'

    return W_df, H_df, err, init


def beta_div(x, y, b):  # b is beta
    """
    basically same function as a similar function in pandas NMF to calculate
    the error
    """
    x = x.ravel()
    y = y.ravel()
    EPSILON = np.finfo(np.float32).eps
    y_e = y.copy()
    if b <= 1:
        y_e[y == 0] = EPSILON

    if b in [0, 1]:
        y_0 = y.copy()
        x_0 = x.copy()
        y_0 = y_0[x_0 != 0]
        x_0 = x_0[x_0 != 0]
        y_0_e = y_0.copy()
        y_0_e[y_0_e == 0] = EPSILON

    if b == 1:
        div = np.sum(x_0 * np.log(x_0 / y_0_e)) - np.sum(x) + np.sum(y)
    elif b == 0:
        div = x_0 / y_0_e - np.log(x_0 / y_0_e) - 1
    else:
        div = (x**b + (b-1) * y**b - b * x * y_e**(b-1)) / (b*(b-1))

    # to get the same result as in the sklearn NMF one should
    # multiply by 2 and take the sqrt
    # return np.sqrt(2*np.sum(div))
    return np.sum(div)


def my_nmf(A: np.ndarray, k):
    n1, n2 = A.shape
    W = get_init_matrix(n1, k, method='random', A=A)
    H = get_init_matrix(k, n2, method='random', A=A)

    n_iter = 100
    for i in range(n_iter):
        # H = get_norm_np(H.T).T
        W = rectify(A @ H.T @ LA.inv(H @ H.T))
        H = rectify(A.T @ W @ LA.inv(W.T @ W)).T
    return W, H



# #############################################################################
# ##########################  SNMF ANALYSIS  ##################################
# #############################################################################



def snmf_via_nmf_np(A: np.ndarray, k: int = 2, beta=2,
                    init: str = 'nndsvd', rnd=None, max_iter: int = 10000):
    """

    :param A:
    :param n:
    :param beta:
    :param init:
    :param rnd:
    :return:
    W
    H
    error
    n_int
    """
    solver = 'cd'
    if beta != 2:
        solver = 'mu'
    NMFmodel = skd.NMF(n_components=k, random_state=rnd, beta_loss=beta,
                       solver=solver, init=init, max_iter=max_iter)

    W = NMFmodel.fit_transform(A)  # not used later, but needed to calculate
    H = NMFmodel.components_
    n_iter = NMFmodel.n_iter_
    S = (W.T + H) / 2
    error = LA.norm(A - S.T @ S)
    return S, error, n_iter


def snmf_via_nmf_best_np(A: np.ndarray, k: int = 2, beta=2,
                         n_rnd: int = 10, max_iter: int = 10000):
    """
    N is the order of the NMF
    beta is the parameter in the optimization function of the NMF
    n_rnd is the number of random initialization in addition on the
    deterministic initializations to try in order to get the best NMF
    (i.e., with the smallest error according to the objective function)
    """
    # different initializations
    inits = ['nndsvd', 'nndsvda', 'nndsvdar'] + list(range(n_rnd))

    mi = pd.MultiIndex(levels=[[], []], codes=[[], []],
                       names=['init', 'n'])
    SNMFs_S = pd.DataFrame(columns=mi)
    err = pd.Series()

    for init in inits:
        init_s = str(init)
        if isinstance(init, int):
            init = 'random'
        S, e, n_iter = snmf_via_nmf_np(A, k=k, beta=beta, init=init,
                                       max_iter=max_iter)
        for n in np.arange(k):
            SNMFs_S.loc[:, (init_s, n)] = S[n]
        err[init_s] = e

    # finding the best reconstruction
    idxmin = err.idxmin()
    if 'nndsvd' not in idxmin:
        init_s = 'random'
    else:
        init_s = idxmin
    return SNMFs_S[idxmin].values.T, err[idxmin], init_s


def get_init_matrix(n1: int, n2: int, method: str, A: np.ndarray = None,
                    scale=100, rect=False)\
        -> np.ndarray:
    """
    it must have been checked before that A is symmetric

    Parameters
    ----------
    n1:
        number of rows of the matrix Y to return
    n2:
        number of columns of the matrix Y to return
    method:
        circ_alg of initialization: random, pca, pre_init
    A:
        in case the circ_alg of initialization in pre_init or pca
        then the Matrix A is required
    Returns
    -------
    Y:
        return the initialization matrix
    """

    if method != 'random' and A is None:
        raise ValueError('Matrix A required')
        n_s1, n_s2 = A.shape
        if n_s1 != n_s2:
            raise ValueError('matrix A is not square and it should be!')
        if n_s1 != n2:
            raise ValueError('n2 should matrix the dimension of A')

    if method == 'random':
        # creating a random initialization for the output of dims n_y x n_s
        std = np.std(A)/scale
        # print('std in random initialization: ', scale)
        Y = np.random.normal(0, std, (n1, n2))
        # version of Cengiz:
        # Y = np.random.normal(0, 1/np.sqrt(n_s), (n_y, n_s))
    elif method == 'full':
        Y = np.full((n1, n2), 1 / (np.std(A) * np.sqrt(n2)))
    elif method == 'full_rand':
        Y = np.random.normal(1 / np.sqrt(n2),
                             1 / (np.sqrt(n2) * np.std(A)), (n1, n2))
        Y = rectify(Y)
    elif method == 'pca':  # not sure this initialization is good at all,
        # probably should not be used.
        w, U = LA.eig(A)
        U = U[:, 0: 1] * np.sqrt(w[0: n1])
        U = U/np.sign(np.mean(U, axis=0))
        Y = rectify(U.T)
    elif method == 'pre_init':
        # pre initialization that comes form the CD algorithm
        # this makes convergence much quicker
        # however sometimes it fails, because there is some inversion problem
        # in those cases i think i would like to then call the random
        # circ_alg

        Y = np.random.normal(1, 1 / np.sqrt(n2), (n1, n2))
        # Y = np.random.normal(10, 1, (n1, n2))
        Y = rectify(Y)
        try:
            for i in range(100):
                Y = LA.inv(Y @ Y.T) @ Y @ A
                Y = rectify(Y)
                # print(i)
        except:
            print('!!!!!!!!  circ_alg pre_init failed, trying random  !!!!!!!!')
            Y = get_init_matrix(n1, n2, 'random', A)

    else:
        raise ValueError('init should be either random, pca or pre_init')

    if rect:
        # Y = rectify(Y)
        Y = np.abs(Y)
    return Y


def snmf_gd_offline(X: np.ndarray, n_y: int, rtol: float = 1e-4,
                    max_iter: int = 100000, etamax: float = 1e-3,
                    init: bool = 'random', verbose: bool = False)\
                    -> np.ndarray:
    """
    Gradient descent to calculate a non negative symmetric matrix factorization

    Parameters
    ----------
    X:
       input matrix
    n_y:
        number of output dimensions
    etamax:
        the initial learning rate
    rtol:
        relative tolerance
    init:
        initialization for the Y matrix

    Returns
    ------
    Y:
        Matrix which minimizes ||S - Y.T Y||_F^2
    """
    A = X.T @ X
    n_s, n_s2 = A.shape  # n_s is the number of samples
    if n_s != n_s2:
        raise ValueError(f'A should be a square matrix, given: {A.shape}')
    if not np.array_equal(A, A.T):
        raise ValueError('A should be symmetric')

    Y = get_init_matrix(n_y, n_s, method=init, A=A)
    # Y = np.array(Y, dtype=FLOAT_TYPE)
    # A = np.array(A, dtype=FLOAT_TYPE)
    Y = np.array(Y, dtype=FLOAT_TYPE, order='F')
    A = np.array(A, dtype=FLOAT_TYPE, order='F')

    print(f'initial reconstruction error {LA.norm(A - Y.T @ Y)}')

    eta = etamax
    # super counter, just a way of updating the learning rate
    scounter = 1
    # print(eta.dtype)
    conv = 1  # convergence error
    for i in range(max_iter):
        Y_old = Y.copy()
        # Y += eta * Y @ (A - Y.T @ Y)   # + eta*(Y*A-Y*Y'*Y)
        # Y += eta * Y @ (A - np.matmul(Y.T, Y))
        # Y += eta * mat_mul(Y, (A - mat_mul(Y.T, Y)))
        Y += mat_mul(Y, (A - mat_mul(Y.T, Y)), alpha=eta)
        # Y += mat_mul_s((A - mat_mul(Y.T, Y)), Y, alpha=eta)
        Y = rectify(Y)

        conv = np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))/eta
        # i imagine the 1e-2 here is to not have any divergence problem?

        if conv < rtol:
            break

        # then resetting the counter, adding 0.1 to scounter
        # and decreasing even more eta, which is the learning rate.
        if i%10000 == 0:
            eta = etamax/scounter
            scounter += 0.1
            print(f'reset counter in SNMF_GD_offline, convergence: {conv}'
                  f', cost: {LA.norm(A - Y.T @ Y)}')
            if verbose:
                print(np.amax(np.abs(Y - Y_old)))
                print(np.amax(np.abs(Y - Y_old) / np.abs(Y_old + 1e-2)))
                print(np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))/eta)
    print(f'convergence: {conv}, i: {i}, scounter: {scounter}, eta: {eta}')
    return Y


def snmf_cd_offline(X: np.ndarray, n_y: int, rtol: float = 1e-4,
                    init: str = 'random') -> np.ndarray:
    """
    Coordinate descent to calculate a non negative symmetric matrix
    factorization
    from Ding et al., 2005
    This does not get to converge, but can be used as an initial stage
    for starting the algorithm

    Parameters
    ----------
    X:
       input matrix
    n_y:
        number of output dimensions
    rtol:
        relative tolerance
    init:
        initialization for the Y matrix

    Returns
    ------
    Y:
        Matrix which minimizes ||S - Y.T Y||_F^2
    """

    A = X.T @ X
    n_s, n_s2 = A.shape  # n_s is the number of samples
    if n_s != n_s2:
        raise ValueError(f'K should be a square matrix, given: {A.shape}')
    if not np.array_equal(A, A.T):
        raise ValueError('K should be symmetric')

    Y = get_init_matrix(n_y, n_s, method=init, A=A)

    print(f'initial reconstruction error {LA.norm(A - Y.T @ Y)}')

    convergence = False  # bool variable telling if the algorithm converged
    counter = 0
    while not convergence:
        counter += 1
        Y_old = Y.copy()
        Y = LA.inv(Y @ Y.T) @ Y @ A
        Y = rectify(Y)

        er = np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))
        # i imagine the 1e-2 here is to not have any divergence problem?

        if er < rtol:
            convergence = True

        # then resetting the counter, adding 0.1 to scounter
        # and decreasing even more eta, which is the learning rate.
        if counter % 1000 == 0:
            print(f'reset counter in SNMF_CD_offline, err: {er}')
        if counter > 10000:
            break
    print(er, counter)
    return Y


def snmf_cd2_offline(X: np.ndarray, n_y: int, rtol: float = 1e-4,
                     max_iter: int = 100000, init: str = 'random')\
                     -> np.ndarray:
    """
    Maybe could be optimized in terms of matrix multiplication
    Coordinate descent to calculate a non negative symmetric matrix
    factorization
    The recommended initialization is either 'full' or 'pre_init'
    the initialization 'random' usually runs into some division by 0
    because there are many 0 values

    Parameters
    ----------
    X:
       input matrix
    n_y:
        number of output dimensions
    rtol:
        relative tolerance
    init:
        initialization for the Y matrix

    Returns
    ------
    Y:
        Matrix which minimizes ||A - Y.T Y||_F^2
    """

    A = X.T @ X
    n_s, n_s2 = A.shape  # n_s is the number of samples
    if n_s != n_s2:
        raise ValueError(f'K should be a square matrix, given: {A.shape}')
    if not np.array_equal(A, A.T):
        raise ValueError('K should be symmetric')

    Y = get_init_matrix(n_y, n_s, method=init, A=A)

    # seem to only make things slower, very strange...
    # Y = np.array(Y, dtype=FLOAT_TYPE)
    # A = np.array(A, dtype=FLOAT_TYPE)

    # Y = np.array(Y, dtype=FLOAT_TYPE, order='F')
    # A = np.array(A, dtype=FLOAT_TYPE, order='F')

    print(f'initial reconstruction error {LA.norm(A - Y.T @ Y)}')

    beta = 0.5
    conv = 1  # convergence error
    for i in range(max_iter):
        Y_old = Y.copy()

        # SYT = A @ Y.T
        # Y3 = Y.T @ Y @ Y.T
        for j, k in itertools.product(range(n_y), range(n_s)):
            Y[j, k] *= 1 - beta + beta * (Y[j] @ A[:, k]) / (Y[j] @ Y.T @ Y[:, k])
            # doesn't make any noticable difference in performance
            # div = np.vdot(Y[j], mv_mul(Y.T, Y[:, k]))
            # Y[j, k] *= 1 - beta + beta * np.vdot(Y[j], A[:, k]) / div
            # Y[i, j] = rectify(Y[i, j])  # not sure when is this needed
        Y = rectify(Y)

        conv = np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))
        # i imagine the 1e-2 here is to not have any divergence problem?

        if conv < rtol:
            break

        # then resetting the counter, adding 0.1 to scounter
        # and decreasing even more eta, which is the learning rate.
        if i % 100 == 0:
            print(f'reset counter in SNMF_CD_offline, err: {conv}')
        if i > 10000:
            break
    print(f'convergence: {conv}, i: {i}')
    return Y


def projected_gradient(gradient: np.ndarray, x: np.ndarray):
    """
    from the paper Projected Gradient Methods for Non-negative Matrix
    Factorization by Chih-Jen Lin
    :param gradient:
    :param x:
    :return:
    """
    on_edge = x == 0
    # print(on_edge)
    # print(gradient)
    gradient2 = gradient.copy()
    gradient2[on_edge] = np.minimum(0, gradient2[on_edge])
    # print(gradient2)
    return gradient2


def my_costA(A: np.ndarray, Y: np.ndarray):
    YTY = mat_mul(Y.T, Y, alpha=1)
    # YTY = np.matmul(Y.T, Y)
    return np.sum((A - YTY) ** 2)
    # return LA.norm(S - Y.T @ Y)**2


def my_costX(X: np.ndarray, Y: np.ndarray):
    XXT = mat_mul(X, X.T)
    YYT = mat_mul(Y, Y.T)
    YXT = mat_mul(Y, X.T)
    return np.sum(XXT**2) + np.sum(YYT**2) - 2 * np.sum(YXT**2)


# def my_costX(X: np.ndarray, Y: np.ndarray):
#     YYT = mat_mul(Y, Y.T)
#     YXT = mat_mul(Y, X.T)
#     return np.sum(YYT**2) - 2 * np.sum(YXT**2)


# def get_grad_Y(A, Y):
#     return mat_mul(Y, (mat_mul(Y.T, Y) - A), alpha=4)


def get_grad_Y(X, Y):
    YYT = mat_mul(Y, Y.T)
    YXT = mat_mul(Y, X.T)
    return mat_mul(YYT, Y, 4) - mat_mul(YXT, X, 4)


def gd_step_Y(X, Y_old, cost_old, sigma, beta, alpha=1):
    """
    gradient descent in Y
    Returns
    -------

    """
    # alpha = 1
    success = False
    grad_Y = get_grad_Y(X, Y_old)
    while alpha > EPSILON*0.0001:
        Y_new = rectify(Y_old - alpha * grad_Y)
        expected_cost_decrease = sigma * np.sum(grad_Y * (Y_new - Y_old))
        # cost_new = my_costA(A, Y_new)
        cost_new = my_costX(X, Y_new)
        # print(expected_cost_decrease, cost_new - cost_old)
        if cost_new - cost_old < expected_cost_decrease:
            success = True
            break
        alpha *= beta
    # print(f'alpha in gd_step_Y: {alpha}')
    if not success:
        return Y_old, cost_old, alpha, success
    else:
        return Y_new, cost_new, alpha, success


def snmf_kuang_offline(X: np.ndarray, n_y: int, rtol: float = 1e-7,
                       max_iter: int = 100000, init: str = 'random',
                       verbose: bool = False) -> Tuple[np.ndarray, float, str]:
    """
    Gradient descent to calculate a non negative symmetric matrix factorization
    with the scaling matrix being identity

    From paper Kuang et al., 2012

    Parameters
    ----------
    X:
       input matrix
    n_y:
        number of output dimensions
    rtol:
        relative tolerance
    max_iter:
        maximum number of iterations
    init:
        initialization for the Y matrix
    verbose:
        outputs some info about how the simulation went

    Returns
    ------
    Y:
        Matrix which minimizes ||A - Y.T Y||_F^2
    cost:
        final cost
    message:
        str
    """

    n_s = X.shape[1]
    Y = get_init_matrix(n_y, n_s, method=init, A=X)
    # Y = np.array(Y, dtype=FLOAT_TYPE, order='F')
    # X = np.array(X, dtype=FLOAT_TYPE, order='F')
    # Y = np.array(Y, dtype=FLOAT_TYPE)
    # S = np.array(S, dtype=FLOAT_TYPE)
    # A = np.array(A, dtype=FLOAT_TYPE, order='F')
    # Y = np.array(Y, order='F')
    # S = np.array(S, order='F')


    sigma = 0.1  # acceptance parameter
    beta = 0.1  # reduction factor

    alpha = 1
    cost0 = my_costX(X, Y)  # initial cost
    # cost0 = my_costA(A, Y)  # initial cost
    cost1 = cost0
    # grad0 = get_grad_Y(A, Y)
    # grad0_norm = LA.norm(grad0)
    # grad = grad0.copy()
    alpha = 1
    for i in range(max_iter):
        # print(Y)
        # Y_old = Y.copy()  # only needed if i care about convergence of Y
        cost_old = cost1.copy()
        Y, cost1, alpha, success = gd_step_Y(X, Y, cost1, sigma, beta, alpha)
        # if i % 10 == 0:
        #     print(f'i: {i}, cost1: {cost1}, alpha: {alpha}')

        # conv = np.amax(np.abs(Y-Y_old) / np.abs(Y_old + 1e-2))
        # print(f'i: {i}, cost: {cost1}, convergence: {conv}')

        if not success or np.abs(cost_old - cost1) / np.abs(cost_old) < rtol:
            break

    message = (f'init: {init}, i: {i}, alpha: {alpha}, '
              f'costs: {cost0}, {cost1}')
    if verbose:
        print(message)
    return Y, cost1, message


def compare_snmf(Y1: np.ndarray, Y2: np.ndarray, A: np.ndarray):
    """
    Compares the 2 decompositions and checks that they both have the same
    subspace
    """
    Q_Y1 = LA.qr(Y1.T, mode='economic')[0]  # become column vectors
    Q_Y2 = LA.qr(Y2.T, mode='economic')[0]
    _, cos_sim, _ = LA.svd(Q_Y1.T @ Q_Y2, full_matrices=False)
    print(f'cos angles between subspaces: {cos_sim}')
    subspace_overlap = subspc_overlap(Q_Y1, Q_Y2)
    print(f'subspaces overlap: {subspace_overlap}')

    # checking that basically that the decomposition is that same
    # modulo a permutation
    Y1_n = get_norm_np(Y1.T)
    Y2_n = get_norm_np(Y2.T)
    perm_mat = Y1_n.T @ Y2_n
    # print(f'{perm_mat}')
    perm_mat[perm_mat < 0.9] = 0
    print('checking that the Y vectors coming for the 2 methods '
          'are similar and thus their inner product in a permutation matrix.\n'
          'The following numbers should all be near 0: ', end='')
    print(perm_mat.sum(axis=0) - 1, perm_mat.sum(axis=1) - 1)

    err1 = LA.norm(A - Y1.T @ Y1)
    err2 = LA.norm(A - Y2.T @ Y2)
    print(f'costs: {err1}, {err2}\n')

    return cos_sim, subspace_overlap, perm_mat, (err1, err2)


def check_snmf(Y: np.ndarray, Y0: np.ndarray, A: np.ndarray):
    """
    """
    err = LA.norm(A - Y.T @ Y)
    err0 = LA.norm(A - Y0.T @ Y0)
    if (err-err0)/err > 1e-6:
        print(f'TEST FAILED: error is not small enough: {err}, {err0}')
    else:
        print('TEST PASSED')
    compare_snmf(Y, Y0, A)


def get_snmf_best_np(X: np.ndarray, k: int, rtol: float = 1e-6,
                     max_iter: int = 10000) -> Tuple[np.ndarray, float, str]:
    """
    returns either the best SNMF calculated via gradient descent with Kuang or with

    if the difference in error between Kuang and via MNF is minor, the
    Kuang solution is outputted


    Parameters
    ----------
    X:
       input matrix
    k:
        number of dimensions in output
    rtol:
        relative tolerence
    max_iter:
        maximum number of iterations

    Returns
    -------
    Y0:
        Y matrix
    err0:
        error
    message:
        print message

    """
    A = X.T @ X
    message: str = f'calculating SNMF with k: {k}\n'
    # print(message)
    A_NN = rectify(A)
    Y0, err0a, init = snmf_via_nmf_best_np(A_NN, k=k, max_iter=10000)
    err0 = LA.norm(Y0.T @ Y0 - A)
    assert err0a == LA.norm(Y0.T @ Y0 - A_NN), 'Problem'
    # here this assetion works because in the function get_snmf_best_np we were
    # really actually using the LA.norm function
    message += f'error to beat {err0}, init: {init}\n'
    # print(message)

    inits = ['pre_init', 'pre_init', 'full_rand', 'random', 'random']
    n = len(inits)  # number of different initializations we do the simulations
    errs = np.zeros(n)
    Ys = np.zeros((n, k, len(A)), dtype=FLOAT_TYPE)

    for i in range(n):
        Y, cost, mes = snmf_kuang_offline(X, k, rtol=rtol, init=inits[i],
                                          max_iter=max_iter)
        # print(mes)
        # 'F' order is necessary to get exactly the same error
        # as it can give slight differences
        err2 = my_costX(np.array(X, dtype=FLOAT_TYPE, order='F'), Y)
        # this is the same function as in snmf kuang
        # print(err1, err2, cost)
        # assert cost == err2, f'problem {(cost - err2)/cost}, {(cost - err2)}'

        err1 = LA.norm(Y.T @ Y - A) # not sure if this one is to be trusted
        # more than the one done by mat_mul from blas
        errs[i] = err1
        Ys[i] = Y
        message += mes + '\n'

    err_min_id = errs.argmin()
    err_min = errs[err_min_id]
    err_ratio = (err0 - err_min)/err0
    message += f'via NMF: {err0}, Kuang: {err_min}, ratio: {err_ratio}\n'
    # print(message)
    if err_ratio >= -2e-7:
        message += f'Kuang better or same as SNMF via NMF, {err_min_id} won.\n'
        Y0 = Ys[err_min_id]
        err_out = err_min
    else:
        message += 'SNMF via NMF better than Kuang\n'
        err_out = err0
    err_relative = err_out/LA.norm(A)
    message += f'relative error: {err_relative}'
    print(message)
    return Y0, err_out, message


def get_snmf_best_df(X: pd.DataFrame, **kwargs)\
        -> Tuple[pd.DataFrame, float, str]:
    """
    this is basically a wrapper around the function get_snmf above.
    Parameters
    ----------
    A
    kwargs
        same as for get_snmf
    Returns
    Y: matrix
    err: SNMF cost
    -------

    """
    k = kwargs['k']
    Y, err, m = get_snmf_best_np(X.values, **kwargs)
    Y = pd.DataFrame(Y, index=np.arange(k)+1, columns=X.columns)
    Y.index.name = 'n'
    return Y, err, m



# def get_hessian(A, Y):
#     """
#     not implemented
#     :param A:
#     :param Y:
#     :return:
#     """
#     return None
#
#
# def SNMF_Kuang_H_offline(A: np.ndarray, n_y: int, rtol: float = 1e-4,
#                          max_iter: int = 100000, init: bool = 'random',
#                          verbose: bool = False) -> np.ndarray:
#     """
#     NOT IMPLEMENTED, NOT HIGH PRIORITY AT THIS MOMENT
#     Gradient descent to calculate a non negative symmetric matrix factorization
#     using the Hessian
#
#     Parameters
#     ----------
#     A:
#         symmetric matrix
#     n_y:
#         number of output dimensions
#     etamax:
#         the initial learning rate
#     rtol:
#         relative tolerance
#     init:
#         initialization for the Y matrix
#
#     Returns
#     ------
#     Y:
#         Matrix which minimizes ||A - Y.T Y||_F^2
#     """
#
#     n_s, n_s2 = A.shape  # n_s is the number of samples
#     if n_s != n_s2:
#         raise ValueError(f'A should be a square matrix, given: {A.shape}')
#     if not np.array_equal(A, A.T):
#         raise ValueError('A should be symmetric')
#
#     Y = get_init_matrix(n_y, n_s, circ_alg=init, A=A)
#     # Y = np.array(Y, dtype=FLOAT_TYPE)
#     # A = np.array(A, dtype=FLOAT_TYPE)
#     Y = np.array(Y, dtype=FLOAT_TYPE, order='F')
#     A = np.array(A, dtype=FLOAT_TYPE, order='F')
#     # Y = np.array(Y, order='F')
#     # A = np.array(A, order='F')
#
#     Y_old = Y.copy()
#
#
#     sigma = 0.1  # acceptance parameter
#     beta = 0.1  # reduction factor
#
#     alpha = 1
#     cost_old = my_cost(A, Y_old)
#     gradient0 = 2 * mat_mul(Y_old, (mat_mul(Y_old.T, Y_old) - A), alpha=1)
#     grad0_norm = LA.norm(gradient0)
#     print(f'norm gradient: {grad0_norm}')
#     gradient = gradient0.copy()
#     print(f'initial reconstruction error {np.sqrt(cost_old)}')
#     for i in range(max_iter):
#         # print(Y)
#         Y = rectify(Y_old - alpha * gradient)
#         cost = my_cost(A, Y)
#         # print(f'norms: {norm}, {norm_old}')
#         expected_cost_decrease = sigma * np.sum(gradient * (Y - Y_old))
#         cond = ((cost - cost_old) <= expected_cost_decrease)
#         # cond = ((cost - cost_old) <= 0)
#         # print(norm - norm_old, some_number)
#
#
#         if not cond:
#             alpha *= beta
#         else:
#             # print(f'alpha: {alpha}')
#             Y_old = Y.copy()
#             cost_old = cost.copy()
#             # print(f'new cost: {np.sqrt(cost)}')
#             gradient = 4 * mat_mul(Y, (mat_mul(Y.T, Y) - A), alpha=1)
#             grad_proj = LA.norm(projected_gradient(gradient, Y))
#             # print(f'norm gradient: {grad_proj}')
#             # if grad_proj <= rtol * grad0_norm or alpha < 1e-8:
#             #    break
#             # print(f'norm gradient: {grad_proj}')
#             if alpha < 1e-7:  # needed because doing float64
#                 break
#             alpha=1
#     print(f'cost: {cost}, i: {i}')
#     return Y_old


