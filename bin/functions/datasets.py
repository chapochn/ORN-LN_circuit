from typing import Tuple, Dict

import numpy as np
from scipy import linalg as LA

from functions import general as FG, olfactory as FO, nmf as nmf


def generate_samples(D: int = 50, N: int = 100, K: int = 10,
                     method: str = 'spiked_covariance', options: dict = {},
                     scale_data: bool = False, ctr_data: bool = False,
                     rho: float = 0.01, seed: int = None)\
        -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, float]]:
    """
    This function is an amazing function that generates sample data

    Parameters
    ----------

    D: int
        number of features
        NMC: that is the dimensionality of the space we are working with
        so in the case of olfaction, it would be 21 ORNs

    K: int
        number of components
        it is the number of dimensions in the top principal subspace,
        which we are trying to find. It is the dimensions of the
        new subspace. It is the number of LNs

    N: int
        number of samples

    method: str
        so far 'spiked_covariance' or 'clusters'

    options: dict
        specific of each circ_alg (see code)

    scale_data: bool
        scaling data so that average sample norm is one
        scaled data means that on average a sample vector has a norm of 1
        meaning that on average the a column vector has a norm of 1

    ctr_data: bool
        centering the data so that the mean is 0

    rho: float
        the variance of the noise in level in 'spiked_covariance' and variance
        in the cluster clouds in 'clusters'

    Returns
    -------
    X : np.ndarray
        generated samples, X in R^{DxN}

    U : np.ndarray
        ground truth eigenvectors

    (avg, scale_factor) : np.ndarray, float
        mean of X. the amount by which the data was scaled. Both values are
        always returned.


    Examples
    --------
    Here is a good example
    """

    # Generate synthetic data samples from a specified model or load real
    # datasets
    # here making sure that we use the right n when including n_test frames
    rnd = np.random.RandomState(seed)
    if not options:  # checks if the dictionary is empty
        options = {
            'a': 0.5,
            'b': 1,  # parameters to create a linspace of eigenvalues
            'c': 2
        }
    if method == 'spiked_covariance':

        # normalized mean that the highest eigenvalue is 1 + rho
        a = options['a']
        b = options['b']
        # lambda_K is in a way the gap, and all the eigenvalues
        # will be between lambda_K and 1
        sigma = np.sqrt(np.linspace(a, b, K))

        # the ground truth is created by creating a random matrix
        # of D row and K columns, and then making a QR decomposition
        # and taking the K vectors that arise
        # U are the direction of the true top eigenspace
        U, _ = np.linalg.qr(rnd.normal(0, 1, (D, K)))
        # this actually does not give an exacty random matrix from a
        # uniformly distributed according to Haar measure:
        # https://arxiv.org/abs/math-ph/0609050

        # w are the vectors used to generate the direction in the
        # principal subspace, z in the paper
        w = rnd.normal(0, 1, (K, N))
        # creating the background signal:
        X = np.sqrt(rho) * rnd.normal(0, 1, (D, N))

        X += U.dot((w.T * sigma).T)

    elif method == 'clusters':
        # first generate K clusters centers in D dimensional space
        # however they are not perpendicular
        U = options['c'] * rnd.rand(D, K) - 1
        # now generated N points in D dimensions;
        X = rnd.normal(0, rho, (D, N))
        for i in range(N):
            X[:, i] += U[:, i % K]

    else:
        raise ValueError(f'circ_alg {method} does not exist, choose between '
                         'spiked_covariance or clusters')

    avg = X.mean(1)[:, None]  # same as X.mean(1)[:, np.newaxis]
    scale_factor = get_scale_data_factor(X)
    # center data
    if ctr_data:
        X -= avg
        # centering the data could influence the true eigenvectors...
    if scale_data:
        X = X * scale_factor
        # U doesn't change with the scaling, as it is the eigenvalues

    return X, U, (avg, scale_factor)




def create_datasets(D: int = 30, N: int = 1000, K: int = 7,
                    options: Dict = {'a': 2, 'b': 5}, seed: int = None,
                    rho1: float = 0.1, rho2: float = 0.1)\
        -> Dict[str, Dict[str, np.ndarray]]:
    """
    Function creating different datasets
    K = 7
    # D: dimensionality of the full space, N: number of samples
    D, N = 30, 1000
    rho1 is for the background variance of the SC
    rho2 is the variance for the clusters
    Returns
    -------

    """
    print('creating datasets')
    datasets = {}
    # Size of PCA subspace to recover, or number of clusters
    scale_data = False  # they say that scaling the data makes convergence better
    # but it is not stricly necessary

    # ---------- spiked covariance dataset
    method = 'spiked_covariance'
    X, U_th, _ = generate_samples(D=D, N=N, K=K, method=method,
                                  scale_data=scale_data, rho=rho1,
                                  ctr_data=False,
                                  options=options, seed=seed)
    # U_th is the theoretical U, used to generate the data, but it is not the
    # actual one from this particular sample
    # the default options here are: lambda_K=0.5, rho:0.002

    datasets['SC'] = {'ds': X, 'g': U_th}

    # ------------ clusters dataset:
    method = 'clusters'
    X, U_th, _ = generate_samples(D=D, N=N, K=K, method=method,
                                  scale_data=scale_data, rho=rho2,
                                  options={}, seed=seed)
    datasets['clus1'] = {'ds': X, 'g': U_th}

    # ------------ clusters dataset in positive quadrant:
    # putting the data in the positive quadrant
    X2 = FG.rectify(X + 1)
    datasets['clus2'] = {'ds': X2, 'g': U_th + 1}

    # ----------- olfactory dataset
    DATASET = 3
    act = FO.get_ORN_act_data(DATASET).T
    X = act.mean(axis=1, level=('odor', 'conc')).values
    U, _, _ = LA.svd(X, full_matrices=False)
    datasets['olf'] = {'ds': X, 'g': U}

    # -------------- END
    print('datasets created\n')
    return datasets


def get_scale_data_factor(X: np.ndarray) -> float:
    """
    Scaling for convergence reasons

    The scaling is such that the average norm is 1
    Not sure why this is helpful for convergence.

    Parameters
    ----------
    X
    Returns
    -------

    """
    # np.sqrt(np.sum(X ** 2, 0)) is the norm of each vector
    # norm_fact is the mean norm
    norm_fact = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale_factor = 1 / norm_fact

    return scale_factor


# i think i found an other more logical way to deal with this.
# def add_snmf_to_datasets(datasets, max_iter:int = 100000):
#     for name, data in datasets.items():
#         X = data[0]
#         Y, _, mes = nmf.get_snmf_best_np(X, data[2].shape[1],
#                                          max_iter=max_iter)
#         print(mes)
#         datasets[name] = (X, data[1], data[2], Y)
#     # return datasets

# code for 2 clusters:
def create_clusters_data(D=21, n=100, K=2, dx=0.5, scale=0.1, x_max=1):
    """

    :param D: number of dimensions
    :param n: number of samples for each cluster, can be a list of dimension K
    :param K: number of clusters
    :param dx: shift from 0 for the K dimensions where the clusters are
    :param scale: sd of the gaussian
    :return:
    """

    cls = {}
    if isinstance(n, int): # if n is an int
        ns = [n] * K
    elif len(n) <= K:   # if n in a list (hopefully, not really tested thoroughly)
        ns = n
    else:
        raise ValueError(f'unclear what n is here {n}, should be an integer'
                         f'or list of length K or less')
    for i in range(K): # creating each cluster
        ctr = np.zeros(D)
        ctr[:K] += dx
        ctr[i] = x_max  # the dimensions for the cluster i
        cls[i] = np.random.normal(size=[D, ns[i]], scale=scale) \
                 + ctr[:, np.newaxis]

    X = np.abs(np.concatenate(list(cls.values()), axis=1))
    return X
