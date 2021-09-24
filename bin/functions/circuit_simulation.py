#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:15:11 2019

@author: nchapochnikov


In this file are functions to simulate the original/default similarity
matching circuit sa well as the olfactory circuit.
There there are classes for the default similarity matching and the
olfactory circuit that keep the values of W and M and have functions that
update their values.


"""


##############################
# Imports
import numpy as np
import scipy.linalg as LA
import functions.general as FG
import matplotlib.pyplot as plt
from typing import Tuple
from abc import ABC, abstractmethod

##############################


def test_atol(y1, y2, tol=1e-8):
    """
    absolute tolerance test
    Parameters
    ----------
    y1
    y2
    tol

    Returns
    -------

    """
    return np.amax(np.abs(y1-y2)) < tol
    # return np.allclose(y1, y2, atol=atol, rtol=rtol)

def test_rtol(y1, y2, tol=1e-8):
    """
    relative tolerance test
    Parameters
    ----------
    y1
    y2
    tol

    Returns
    -------

    """
    return np.amax(np.abs(y1-y2)/(np.abs(y1) + tol)) < tol
    # return np.allclose(y1, y2, atol=atol, rtol=rtol)


def sm_output_online(x: np.ndarray, W: np.ndarray, M: np.ndarray,
                     method: str = 'inv', NN: bool = False,
                     atol: float = 1e-9, rtol: float = 0,
                     n_iter_max: int = int(1e6), verbose: bool = False,
                     check_M: bool = False) -> np.ndarray:
    """
    calculating the output y using matrix inversion, based on the input and on
    different the synaptic weights
    available methods are inv and GD

    I wonder if one should do some tests here before running the simulation
    Or at least maybe it should be said here that:


    Parameters
    -----------
    M:
        symmetrical (or?) and k x k
    W:
        should be n x k
    x:
        should be n
    method:
        which circ_alg should be used for calculating the circuit dynamics
        possible options are 'inv', 'GD', 'CD', 'CD_NN', 'Minden'
    NN:
        if we want that the result to be non negative, nothing fancy, just
        making the end result non-negative, does not change the convergence
        circ_alg
    atol:
        absolute tolerance for the convergence
    rtol:
        relative tolerance for the convergence
        NOT IMPLEMENTED, VALUE IS DOES NOT HAVE ANY EFFECT
    n_iter_max:
        maximum number of iterations
    verbose:
        prints at what iteration the algorithm stopped and the error
    check_M:
        checks if M is positive definite

    Returns
    -------
    y:
        vector of length k
    """
    if check_M and not np.all(LA.eigvals(M) > 0):
        raise ValueError('Matrix M is not positive definite, thus the circuit '
                         'dynamics will not converge to a stable fix point')

    i = 0
    if method == 'inv':
        y2 = np.linalg.solve(M, W.T @ x)

    elif method == 'GD':
        Wx = W.T.dot(x)
        eta = 0.01
        # if i make delta larger, y often goes in the wrong direction and
        # diverges
        # y = Wx.copy()
        y2 = np.zeros(len(M), dtype=M.dtype)
        for i in range(n_iter_max):
            # print(y2)
            y = y2.copy()
            y2 += eta * (Wx - M.dot(y2))
            # print(- 4 * (W.T @ y) @ x + 2 * M @ y @ y)
            # TODO: for checking the convergence, it would make sense that the
            # error is multiplied by the eta
            if test_rtol(y, y2, rtol):
            # if tol_test(y, y2, atol=atol, rtol=rtol):
                break

    elif method == 'GD_NN':
        Wx = W.T @ x
        eta = 0.01
        # if i make delta larger, y often goes in the wrong direction and
        # diverges
        # y = Wx.copy()
        y2 = np.zeros(len(M), dtype=M.dtype)
        for i in range(n_iter_max):
            # print(y2)
            y = y2.copy()
            y2 = FG.rectify(y2 + eta * (Wx - M.dot(y2)))
            # print(- 4 * (W.T @ y) @ x + 2 * M @ y @ y)
            # TODO: for checking the convergence, it would make sense that the
            # error is multiplied by the eta
            if test_rtol(y, y2, rtol):
                break

    elif method == 'CD':
        # print(M)
        # print(W)
        # print(x)
        W1 = W/np.diag(M)
        M1 = M/(np.diag(M)[:, None])
        W1x = W1.T.dot(x)
        np.fill_diagonal(M1, 0)  # puts 0 on the diagonal
        k = len(M)
        y2 = np.zeros(k, dtype=M.dtype)
        # initialization from Minden et al., 2018, not sure it makes anything
        # faster...
        # =============================================================================
        #         Md_inv = np.diag(1./np.diag(M))
        #         Mo = M.copy()
        #         np.fill_diagonal(Mo, 0)
        #         y2 = Md_inv @ W @ x
        #         y2 -= Md_inv @ Mo @ y2
        # =============================================================================
        for i in range(n_iter_max):
            # print(y2)
            y = y2.copy()  # keeping a copy of y at the previous iteration
            for j in range(k):
                y2[j] = W1x[j] - M1[j] @ y2
            # print(y2-y)
            if test_rtol(y, y2, rtol):
            # if tol_test(y, y2, atol=atol, rtol=rtol):
                break

    elif method == 'CD_NN':
        W1 = W/np.diag(M)
        M1 = M/(np.diag(M)[:, None])
        W1x = W1.T.dot(x)
        np.fill_diagonal(M1, 0)
        k = len(M)
        y2 = np.zeros(k, dtype=M.dtype)
        for i in range(n_iter_max):
            # print(y2)
            y = y2.copy()  # keeping a copy of y at the previous iteration
            for j in range(k):
                y2[j] = max(0, W1x[j] - np.vdot(M1[j], y2))
            if test_rtol(y, y2, rtol):
            # if tol_test(y, y2, atol=atol, rtol=rtol):
                break

    elif method == 'Minden':
        Md_inv = np.diag(1./np.diag(M))
        Mo = M.copy()
        np.fill_diagonal(Mo, 0)
        y2 = Md_inv @ W.T @ x
        y2 -= Md_inv @ Mo @ y2

    else:
        raise ValueError(f'circ_alg {method} is not implemented. Use inv,'
                         ' GD, CG_NN, CD, CD_NN, Minden')

    if verbose and method not in ['inv', 'Minden']:
        print(f'end {method} at i: {i}, error: {np.amax(np.abs(y-y2))}')

    if NN:
        return FG.rectify(y2)
    else:
        return y2


def sm_output_offline(X, W, M, method, NN=False):
    K = W.shape[0]
    D, N = X.shape
    # just our of speed considerations, although not big difference
    # in particular as K does play a big role as well
    if method == 'inv' and N > D:
        Minv = LA.inv(M)
        Y = Minv @ W.T @ X
    else:
        Y = np.zeros((K, N))
        for i in range(N):
            Y[:, i] = sm_output_online(X[:, i], W, M, method=method)
    if NN:
        return FG.rectify(Y)
    else:
        return Y


def olf_output_online(x: np.ndarray, Wff: np.ndarray, Wfb: np.ndarray,
                      M: np.ndarray, rho=1,
                      method: str = 'inv', NN: bool = False,
                      atol: float = 1e-10, rtol: float = 1e-10,
                      n_iter_max: int = int(1e6), verbose: bool = False,
                      check_M: bool = False)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    calculating the output y and z using matrix inversion,
    based on the input and on
    different the synaptic weights

    available methods are inv


    Parameters
    -----------
    M:
        symmetrical (or?) and k x k
    W:
        should be k x n
    x:
        should be n
    method:
        which circ_alg should be used for calculating the circuit dynamics
        possible options are 'inv', 'GD', 'GD_NN'
    atol:
        absolute tolerance for the convergence
    rtol:
        relative tolerance for the convergence
        NOT IMPLEMENTED, VALUE IS DOES NOT HAVE ANY EFFECT
    n_iter_max:
        maximum number of iterations
    verbose:
        prints at what iteration the algorithm stopped and the error
    check_M:
        checks if M is positive definite

    Returns
    -------
    y:
        vector of length D
    z:
        vector of length k
    """
    if check_M and not np.all(LA.eigvals(M) > 0):
        raise ValueError('Matrix M is not positive definite, thus the circuit '
                         'dynamics will not converge to a stable fix point')

    if method == 'inv':
        # could use LA.solve instead of the inverse, i guess
        # would be faster/cleaner/more exact...
        Minv = LA.inv(M)
        # this is with solve, but we need to calculate the inverse of M
        # anyway so not sure if it is really helpful
        # y = LA.solve(np.eye(W.shape[1]) + W.T @ Minv @ W, x)
        y2 = LA.inv(np.eye(Wff.shape[0]) + rho**2 * Wfb @ Minv @ Wff.T) @ x
        z2 = rho**2 * Minv @ Wff.T @ y2

    elif method == 'GD':
        # probably a quite conservative estimate for eta
        est = (np.max([1, np.max(np.abs(Wff))]) *
               np.max([1, np.max(np.abs(M))]) *
               np.max([1, np.max(np.abs(x))]))
        if est < 1.01:
            est = np.max([np.max(np.abs(Wff.T @ x)),
                          np.max(np.abs(M @ Wff.T @ x))])

        print('estimate ', est)
        eta1 = 1. / est
        # if i make delta larger, y often goes in the wrong direction and
        # diverges
        # y = Wx.copy()
        y2 = np.zeros(len(x), dtype=M.dtype)
        # y2 = x
        z2 = np.zeros(len(M), dtype=M.dtype)
        for i in range(n_iter_max):
            eta = 1000*eta1 / (i+1000)
            # if i % 1000 == 0:
            #     print(y2)
            #     print(z2)
            y = y2.copy()
            z = z2.copy()
            y2 += eta * (x - y - Wfb.dot(z))
            z2 += eta * (rho ** 2 * Wff.T.dot(y) - M.dot(z))
            # TODO: for checking the convergence, it would make sense that the
            # tol is multiplited by eta
            if (test_rtol(y, y2, rtol) and (test_rtol(z, z2, rtol))):
            # if (tol_test(y, y2, atol=atol, rtol=rtol) and
            #         tol_test(z, z2, atol=atol, rtol=rtol)):
                # print('converged at ', i)
                break

    elif method == 'GD_NN':
        # probably a quite conservative estimate for eta
        est = (np.max([1, np.max(np.abs(Wff))]) *
               np.max([1, np.max(np.abs(M))]) *
               np.max([1, np.max(np.abs(x))]))
        if est < 1.01:
            est = np.max([np.max(np.abs(Wff.T @ x)),
                          np.max(np.abs(M @ Wff.T @ x))])
        print('estimate ', est)
        eta1 = 1./est
        # if i make delta larger, y often goes in the wrong direction and
        # diverges
        # y = Wx.copy()
        y2 = np.zeros(len(x), dtype=M.dtype)
        # y2 = x
        z2 = np.zeros(len(M), dtype=M.dtype)
        for i in range(n_iter_max):
            eta = 1000*eta1 / (i+1000)
            # if i % 1000 == 0:
            #     print(y2)
            #     print(z2)
            y = y2.copy()
            z = z2.copy()
            y2 = FG.rectify(y + eta * (x - y - Wfb.dot(z)))
            z2 = FG.rectify(z + eta * (rho**2 * Wff.T.dot(y) - M.dot(z)))
            if (test_rtol(y, y2, rtol) and (test_rtol(z, z2, rtol))):
            # if (tol_test(y, y2, atol=atol, rtol=rtol) and
            #         tol_test(z, z2, atol=atol, rtol=rtol)):
                break
        print('converged at ', i)

    else:
        raise ValueError(f'circ_alg {method} is not implemented. Use inv, GD'
                         f'or GD_NN')

    # only relevant for iterative methods, where one can compare y and y2
    # meaning the last 2 steps of the iteration
    # if verbose and circ_alg not in ['inv']:
    #     print(f'end {circ_alg} at i: {i}, error: {np.amax(np.abs(y-y2))}')

    if NN:
        y2 = FG.rectify(y2)
        z2 = FG.rectify(z2)

    return y2, z2


def olf_output_offline(X, Wff, Wfb, M, rho=1, method='inv',
                       n_iter_max=int(1e6)):
    """
    just doing the same as the function above, but with a matrix instead
    of on thing at a time
    Parameters
    ----------
    X
    W
    M
    method

    Returns
    -------

    """
    K = Wff.shape[1]
    D, N = X.shape
    # just out of speed considerations, although not big difference
    # in particular as K does play a big role as well
    if method == 'inv':
        Y, Z = olf_output_online(X, Wff, Wfb, M, rho, method)
    else:
        Y = np.zeros((D, N))
        Z = np.zeros((K, N))
        for i in range(N):
            y, z = olf_output_online(X[:, i], Wff, Wfb, M, rho=rho,
                                     method=method, n_iter_max=n_iter_max)
            Y[:, i], Z[:, i] = y, z
    return Y, Z


# these functions are not used anywhere and where written in
# Giovannuci/Minden paper. Not sure if they could be of any use for me
# def compute_errors(error_options, Uhat, t, errs):
#     """
#     Parameters
#     ----------
#     error_options:
#         A struct of error options
#     Uhat:
#         The approximation Uhat of an orthonormal basis for the PCA subspace
#         of size D by K
#     t:
#         The current iteration index
#     errs:
#         An output dict in which to put the computed errs
#
#     Returns:
#     ------
#
#
#     """
#     if t % error_options['n_skip']:
#         return
#
#     for i, (fname, f) in enumerate(error_options['error_func_list']):
#         errs[fname][t] = f(Uhat)
#
#
def initialize_errors(error_options, n_its):
    """
    Build a dictionary for storing the error information for each specified
    error function
    """
    return {fun_name: np.zeros(n_its) for (fun_name, _)
            in error_options['error_func_list']}

def eta(t):
    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return 1.0 / (t + 5)

def eta2(t, a, b):
    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return np.max((1.0 / (b*t + a), 1e-4))
# #############################################################################
# #################  PARENT CIRCUIT CLASS  ####################################
# #############################################################################

class Circuit(ABC):
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M,
                        must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W,
                        must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M
                        (multiplier of the W learning rate)

    Methods:
    ========
    fit_next()
    get_components()
    get_components2()
    get_components3()

    """

    # initialization of the class
    def __init__(self, K, D, M0=None, W0=None, learning_rate=eta, tau=0.5,
                 circ_alg='inv', NN=False):

        if M0 is not None:
            if M0.shape != (K, K):
                raise ValueError(f"The shape of the initial guess Minv0"
                                 " must be (K,K)=({K},{K})")
            M = M0
        else:
            M = np.eye(K)

        if W0 is not None:
            if W0.shape != (D, K):
                raise ValueError(f"The shape of the initial guess W0"
                                 " must be (K,D)=({K},{D})")
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(D), size=(K, D))

        self.eta = learning_rate
        self.t = 0

        self.K = K
        self.D = D
        self.tau = tau
        self.M = M
        self.W = W

        # Storage variables to allocate memory and optimize outer product time
        self.outer_W = np.empty_like(W)
        self.outer_M = np.empty_like(M)
        self.method = circ_alg
        self.NN = NN
        self.x = 0
        self.y = 0
        self.z = 0

    # these are not really necessary, as they just add more calculations
    # but give the same results
    # def get_components(self, orthogonalize: bool = True):
    #     """
    #     Extract components from object
    #
    #     orthogonalize: bool
    #         whether to orthogonalize the components before returning
    #
    #     Returns
    #     -------
    #     components: ndarray
    #     """
    #
    #     components = np.asarray(np.linalg.solve(self.M, self.W).T)
    #     # the solve solves M x = W, basically components = M^-1 W
    #     # same as:
    #     # components = (np.linalg.inv(self.M) @ self.W).T
    #     # but probably faster because doesn't need to calculate the inverse
    #     if orthogonalize:
    #         components, _ = np.linalg.qr(components)
    #
    #     return components
    #
    # def get_components2(self, orthogonalize: bool = True):
    #     """
    #     Extract components from object
    #
    #     orthogonalize: bool
    #         whether to orthogonalize the components before returning
    #
    #     Returns
    #     -------
    #     components: ndarray
    #     """
    #
    #     components = (np.linalg.inv(self.M) @ self.W).T
    #     if orthogonalize:
    #         components, _ = np.linalg.qr(components)
    #
    #     return components

    def update_W_M(self, x, y, step):
        # before was written as (not sure if it makes it faster or something
        # it is supposed to make it faster, but i don't really see a change
        # in performance):
        # Plasticity, using gradient ascent/descent
        # TODO: the factor of 2 can go away probably...
        # W <- W + 2 eta(t) * (y*x' - W)
        np.outer(x, 2 * step * y, out=self.outer_W)
        # self.outer_W = np.outer(2 * step * y, x)
        W = (1 - 2 * step) * self.W + self.outer_W

        # M <- M + eta(self.t)/tau * (y*y' - M)
        step = step / self.tau
        np.outer(step * y, y, out=self.outer_M)
        # self.outer_M = np.outer(step * y, y)
        M = (1 - step) * self.M + self.outer_M

        return W, M


    def get_components3(self, orthogonalize: bool = True):
        """
        Extract components from object

        orthogonalize: bool
            whether to orthogonalize the components before returning

        Returns
        -------
        components: ndarray
        """

        components = self.W
        if orthogonalize:
            components, _ = LA.qr(components, mode='economic')

        return components

    def get_overlap(self, Q_U):
        return FG.subspc_overlap(self.get_components3(), Q_U, scaled=1)

    def get_perp(self):
        Minv = np.linalg.inv(self.M)
        return LA.norm(Minv @ self.W @ self.W.T @ Minv - np.eye(self.K))

    @abstractmethod
    def transform_1(self, x):
        pass

    @abstractmethod
    def fit_next(self, x):
        pass

    @abstractmethod
    def transform_batch(self, X):
        pass

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

# #############################################################################
# #####################  SIMILARITY MATCHING CLASS  ###########################
# #############################################################################



class SM(Circuit):
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M,
                        must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W,
                        must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M
                        (multiplier of the W learning rate)

    Methods:
    ========
    fit_next()
    get_components()
    get_components2()
    get_components3()

    """

    # initialization of the class

    def transform_1(self, x):
        y = sm_output_online(x, self.W, self.M, method=self.method,
                             NN=self.NN, atol=1e-9, rtol=0,
                             n_iter_max=int(1e6), verbose=False, check_M=False)
        return y

    def transform_batch(self, X):
        return sm_output_offline(X, self.W, self.M, self.method, NN=self.NN)

    def fit_next(self, x):

        if x.shape != (self.D,):
            raise ValueError(f'x should have the shape of D, however {x.shape}'
                             ' is different from {(self.D,)}.')

        self.x = x
        self.z = self.transform_1(self.x)
        
        # print(f'error in comparison with inv: {np.max(np.abs(y-y1))}')

        step = self.eta(self.t)

        self.W, self.M = self.update_W_M(self.x, self.z, step)
        self.t += 1


        
# #############################################################################
# ##########################  OLFACTORY CLASS  ################################
# #############################################################################

class OlfSim(Circuit):
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M,
                        must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W,
                        must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M
                        (multiplier of the W learning rate)

    Methods:
    ========
    fit_next()
    get_components()
    get_components2()
    get_components3()

    """

    def transform_1(self, x):
        y, z = olf_output_online(x, self.W, self.M, method=self.method,
                                 NN=self.NN, atol=1e-9, rtol=0, verbose=False,
                                 n_iter_max=int(1e6), check_M=False)
        return y, z

    def transform_batch(self, X):
        return olf_output_offline(X, self.W, self.W, self.M, self.method)

    def fit_next(self, x):
        """
        for the moment just implementing the linear case
        Parameters
        ----------
        x

        Returns
        -------

        """

        if x.shape != (self.D,):
            raise ValueError(f'x should have the size of D, however {x.shape}'
                             ' is different from {(self.D,)}.')
        self.x = x
        self.y, self.z = self.transform_1(self.x)

        # print(f'error in comparison with inv: {np.max(np.abs(y-y1))}')

        step = self.eta(self.t)

        self.W, self.M = self.update_W_M(self.y, self.z, step)

        self.t += 1




# #############################################################################
# #######################  functions inspecting/monitoring  ###################
# #######################  the evolution of circuits  #########################
# #############################################################################

# i want to have a dictionary of "inspection"
# functions which will take as input the class and make some calculation
# for each function, there will be an array outputted


def initialize_monitors(funcs, n_its):
    """
    Build a dictionary for storing the error information for each specified
    error function
    """
    return {name: np.zeros((v[0], n_its)) for name, v in funcs.items()}


def compute_monitors(monitors, sm, funcs, i):
    for k in monitors.keys():
        monitors[k][:, i] = funcs[k][1](sm)


def get_rec_err_per_epoch(X, Y, n_epochs):
    """
    X is the input over several epochs, Y is the output over several epochs
    Parameters
    ----------
    X
    Y

    Returns
    -------

    """
    N1 = X.shape[1]
    N2 = Y.shape[1]
    if N1 != N2:
        raise ValueError('number of times points in Y not same as in X')
    N = int(N1/n_epochs)

    if N * n_epochs != N1:
        raise ValueError('number of epochs not correct')

    recs = np.zeros(n_epochs)
    X1 = X[:, : N]
    norm = LA.norm(X1.T @ X1)
    for i in range(n_epochs):
        Y1 = Y[:, i*N: (i+1)*N]
        X1 = X[:, i*N: (i+1)*N]
        recs[i] = LA.norm(X1.T @ X1 - Y1.T @ Y1)

    return recs/norm


def run_simulation(circ, W, M, dataset:np.ndarray, n_epoch: int,
                   monitoring_functions={}, **kwargs):
    """
    for the case of non-negative circuit, there is no need to do any
    verbose or give any Q_U because there is no subspace to be inspected
    and also no perpendicularity

    Parameters
    ----------
    circ:
        the class we are using for circuit simulations
    W:
        initial guess for W
    M:
        initial guess for M
    dataset:
        the dataset on which we want to work
    n_epoch:
        number of epochs for the simulation
    NN:
        if to the non-negative
    verbose:
        if to return details of err, perp and recs
    Q_U:
        the actual expected subspace

    Returns
    sm:
        similarity matching class output
    monitors:
        monitored_values
    -------

    """
    X = dataset
    N = dataset.shape[1]
    K, D = W.shape
    if M.shape != (K, K):
        raise ValueError(f"The shape of M must be (K,K)=({K},{K})")
    sequence = np.arange(N)
    sm = circ(K, D, **kwargs)
    n_iters = N * n_epoch
    monitors = initialize_monitors(monitoring_functions, n_iters)
    z_sum = np.zeros(K)
    n = 0
    for n_e in range(n_epoch):
        seq_rand = np.random.permutation(sequence)
        for i, ir in enumerate(seq_rand):  # ir like i random
            n = n_e * N + i
            sm.fit_next(X[:, ir])
            compute_monitors(monitors, sm, monitoring_functions, n)

            z_sum += sm.z
            if n>0 and n % 1000 == 0:
                print(f'n= {n}, z = {z_sum}')
                if any(z_sum == 0):
                    js_0 = np.argwhere(z_sum == 0)
                    print('all weights will be changed')
                    sm.W, sm.M = get_initial_W_M(X, K, scale=100, type='random')
                    # print(sm.W)
                    # for j in js_0:
                    #     # sm.W[j] = -sm.W[j]
                    #     sm.W[j] = np.random.randn(len(sm.W[j]))*np.mean(sm.W[j-1])
                    # print(sm.W)
                z_sum = np.zeros(K)
    return sm, monitors


def continue_simulation(circ, dataset:np.ndarray, n_epoch: int,
                   monitoring_functions={}):
    """
    for the case of non-negative circuit, there is no need to do any
    verbose or give any Q_U because there is no subspace to be inspected
    and also no perpendicularity

    Parameters
    ----------
    circ:
        the class we are using for circuit simulations
    W:
        initial guess for W
    M:
        initial guess for M
    dataset:
        the dataset on which we want to work
    n_epoch:
        number of epochs for the simulation
    NN:
        if to the non-negative
    verbose:
        if to return details of err, perp and recs
    Q_U:
        the actual expected subspace

    Returns
    sm:
        similarity matching class output
    monitors:
        monitored_values
    -------

    """
    sm = circ
    X = dataset
    N = dataset.shape[1]
    sequence = np.arange(N)
    n_iters = N * n_epoch
    monitors = initialize_monitors(monitoring_functions, n_iters)
    z_sum = np.zeros(sm.K)
    n = 0
    for n_e in range(n_epoch):
        seq_rand = np.random.permutation(sequence)
        for i, ir in enumerate(seq_rand):  # ir like i random
            n = n_e * N + i
            circ.fit_next(X[:, ir])
            compute_monitors(monitors, sm, monitoring_functions, n)

            z_sum += sm.z
            if n>0 and n % 100 == 0:
                print(f'n= {n}')
                if any(z_sum == 0):
                    js_0 = np.argwhere(z_sum == 0)
                    print('weights will be inverted')
                    print(sm.W)
                    for j in js_0:
                        sm.W[j] = -sm.W[j]
                    print(sm.W)
                z_sum = np.zeros(sm.K)
    return sm, monitors

def inspect_sm_results(circ: Circuit, X: np.ndarray, U_off: np.ndarray,
                       Z_off: np.ndarray):
    """

    Parameters
    ----------
    X:
        initial data
    U:
        expected solution, at lesat in terms of subspace
    Returns
    -------

    """
    # U is the actual subspace, we don't use the theoretical part U_th
    # creating an orthonormal basis in case they do not come from PCA

    Z = circ.transform_batch(X)  # Y output with frozen circuit
    rel_cost_on = LA.norm(Z.T @ Z - X.T @ X) / LA.norm(X.T @ X)
    rel_cost_off = LA.norm(Z_off.T @ Z_off - X.T @ X) / LA.norm(X.T @ X)

    # this error compares the subspaces, based on W and Q_U
    err = circ.get_overlap(U_off)
    perp = circ.get_perp()
    return rel_cost_on, rel_cost_off, err, perp


def display_sm_results(circ: Circuit, X: np.ndarray, Z_off: np.ndarray, monitors,
                       plot=True, name=''):
    """

    Parameters
    ----------
    X:
        initial data
    U:
        expected solution, at lesat in terms of subspace
    errs:
        errors during the whole simulation
    perp:
        measure of perpendicularity
    recs:
        reconstruction at each epoch
    plot:
        to plot or not to plot
    Y2:
        this is the ideal reconstruction

    Returns
    -------

    """
    errs = monitors['overlap'][0]
    perps = monitors['perp'][0]
    X1 = monitors['x']
    Z1 = monitors['z']
    n_epoch = int(X1.shape[1]/X.shape[1])
    recs = get_rec_err_per_epoch(X1, Z1, n_epoch)
    n_epoch = len(recs)
    # U is the actual subspace, we don't use the theoretical part U_th
    # creating an orthonormal basis in case they do not come from PCA
    print('\n-----------------------------\n', name)
    print(f'Final subspace error: {errs[-1]}')
    print(f'Final orthogonality error: {perps[-1]}')
    Z = circ.transform_batch(X)  # Y output with frozen circuit
    rel_rec_error_frozen = LA.norm(Z.T @ Z - X.T @ X)/LA.norm(X.T @ X)
    rel_rec_error_best = LA.norm(Z_off.T @ Z_off - X.T @ X)/LA.norm(X.T @ X)
    print(f'Reconstruction last epoch, frozen, best: {recs[-1]}, '
          f'{rel_rec_error_frozen}, {rel_rec_error_best}')

    # ############## plotting
    if plot:
        f, axx = plt.subplots(1, 3, figsize=(15, 5))
        ax = axx[0]
        ax.semilogy(errs)
        ax.set(ylabel='Relative subspace error', xlabel='Samples (t)')

        ax = axx[1]
        ax.semilogy(perps)
        ax.set(ylim=(None, 10), ylabel='orthonormality error',
               xlabel='Samples (t)')

        ax = axx[2]
        ax.semilogy(np.arange(n_epoch) + 1, recs)
        ax.semilogy([1, n_epoch], [rel_rec_error_best, rel_rec_error_best])
        ax.set(ylabel='relative reconstruction error', xlabel='epoch')

        plt.suptitle(name)
        plt.show()


def get_initial_W_M(X, K, scale=100, type='random'):
    # Initial guess for the W and the M matrices so that Uhat0 @ M0 = Qs
    # scal is used to fix initial values for W and M
    if type == 'X':
        W = X[:, :K].T / np.sqrt((X[:, :K] ** 2).sum(0)) / scale
    elif type == 'random':
        W = np.random.normal(0, 1, (K, X.shape[0])) /scale
    elif type == 'svd':
        U, _, _ = LA.svd(X[:1000], full_matrices=False)
        W = U[:, 0:K].T/scale
    else:
        raise ValueError('no such type for initializing W and M')
    # w is a matrix of shape D x K
    M = np.eye(K) / scale  # diagonal matrix
    return W, M
