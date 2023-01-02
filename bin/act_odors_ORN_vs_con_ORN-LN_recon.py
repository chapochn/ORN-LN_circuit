#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 10.2022

@author: Nikolai M Chapochnikov


"""


# timing stuff
# start_time = timeit.default_timer()
# do something
# elapsed = timeit.default_timer() - start_time
# print(elapsed)

# %%
# #############################################################################
# ################################# IMPORTS ###################################
# #############################################################################

import numpy as np
import pandas as pd
import importlib

import params.act3 as par_act3
from sklearn import linear_model
import scipy.linalg as LA
from functions import general as FG, olfactory as FO




# %%
# ################################  RELOADS  ##################################
# =============================================================================
importlib.reload(FO)
importlib.reload(FG)

# %%
# #############################################################################
# ########################  CLASS INITIALIZATION  #############################
# #############################################################################

DATASET = 3

# there might be a cleaner way to write this...
par_act = None
exec(f'par_act = par_act{DATASET}')

cell_type = 'ORN'
# cell_type = 'PN'

strms = [0, 1, 2, 3]
con_pps_k = 'cn'  # options: 'cn', 'n'; c and o also possible, but not really
# meaningful, as then one cannot really see the correlation with the activity
# and even the 'n' not sure that is even interesting.

save_plots = True
plot_plots = False

CONC = 'all'
# CONC = '8'
act_sel_ks = [CONC]
# we are deciding upfront which concentration we want to look at
# options are: 'ec50', 'all', '45', '678', '4', '5', '6', '7', '8'


# act_pps_k1 = 'tn'  # k like key
act_pps_k1 = 'raw'
# act_pps_key1 = 'model'

# act_pps_k2 = 'raw'
act_pps_k2 = 'mean'
# act_pps_k2 = 'l2'

path_plots = None

odor_sel = None

ACT_PPS = 'o' # this is the preprocessing that we will use on the
# activity data, other options are 'o', 'c', 'n', 'cn'

ORNA = FO.NeurActConAnalysis(DATASET, cell_type, strms, con_pps_k,
                             save_plots, plot_plots, act_sel_ks,
                             act_pps_k1, act_pps_k2,
                             odor_sel=odor_sel,
                             neur_order=None, odor_order=None,
                             path_plots=path_plots, reduce=True,
                             subfolder='act_ORN_odors_vs_con_ORN-LN/')


# getting only the real cells, i think they are the same independent
# of the stream
STRM = 0
CELLS_REAL =  ['Broad T1 L',
                 'Broad T2 L',
                 'Broad T3 L',
                 'Broad T1 R',
                 'Broad T2 R',
                 'Broad T3 R',
                 'Broad T M M',
                 'Broad D1 L',
                 'Broad D2 L',
                 'Broad D1 R',
                 'Broad D2 R',
                 'Broad D M M',
                 'Keystone L L',
                 'Keystone L R',
                 'Keystone R R',
                 'Keystone R L',
                 'Keystone M M',
                 'Picky 0 [dend] L',
                 'Picky 0 [dend] R',
                 'Picky 0 [dend] M'
                 ]
# decomment this line if you don't want the "fake" cells", which are
# the average of categories
# CELLS_REAL = [name for name in CELLS_REAL if ' M' not in name]

xmin = -1
xmax = 1
bins_pdf = np.linspace(xmin, xmax, 21)
n_bins_cdf = 500
bins_cdf = np.linspace(xmin, xmax, n_bins_cdf + 1)

with open(ORNA.path_plots / f'params.txt', 'w') as f:
    f.write(f'\ndataset: {DATASET}')
    f.write(f'\nact_pps1: {act_pps_k1}')
    f.write(f'\nact_pps2: {act_pps_k2}')
    f.write(f'\nodor_subset: {odor_sel}')
    f.write(f'\nstrm: {STRM}')
    f.write(f'\nACT_PPS: {ACT_PPS}')
    f.write(f'\nCONC: {CONC}')
    f.write(f'\nact_sel_ks: {act_sel_ks}')

png_opts = {'dpi': 250, 'transparent': True}
print('done')


#%%

act_o = ORNA.act_sels[CONC][ACT_PPS].T
act_o_np = act_o.values
act_o_np

con_o = ORNA.con_strms2[STRM]['o'].loc[:, CELLS_REAL].copy()
con_o


#%%



lasso = True
# lasso = False

if lasso:
    l_norm = 1
    alphas1 = np.logspace(-4, -1, 20)
    alphas2 = np.logspace(-4, -1, 20)

    N = 1000
    def model(alpha):
        return linear_model.Lasso(alpha=alpha, max_iter=int(1e6))#, tol=1e-10)
else:
    l_norm = 2
    alphas1 = np.logspace(-3, 2, 20)
    alphas2 = np.logspace(-3, 2, 20)
    N = 1000
    def model(alpha):
        return linear_model.Ridge(alpha=alpha)

act_n_np = act_o_np / LA.norm(act_o_np, axis=0)
for LN_i in [6, 11, 16, 19]:
    w = con_o.values[:, LN_i]
    LN = con_o.columns[LN_i]
    print(LN)
    w = w/LA.norm(w)
    w1 = w

    res_rand = pd.DataFrame(index = np.arange(len(alphas2)*N),
                           columns=['alpha', 'error', 'coef_norm', 'w'],
                            dtype=float)

    for j in range(N):
        print(j)
        w1 = np.random.permutation(w)
        for i, alph in enumerate(alphas2):
            clf = model(alph)
            clf.fit(act_n_np, w1)
            bias = clf.intercept_
            error = LA.norm(w1 - act_n_np @ clf.coef_ - bias)
            coef_norm = LA.norm(clf.coef_, ord=l_norm)
            res_rand.loc[i*N + j, 'alpha'] = alph
            res_rand.loc[i*N + j, 'error'] = error
            res_rand.loc[i*N + j, 'coef_norm'] = coef_norm
            res_rand.loc[i*N + j, 'w'] = j
            # coef_norms[i] = LA.norm(clf.coef_, ord=l_norm)
        # print(clf.intercept_, clf.coef_[:10])
    print('done')


    file_begin = (f'results/{cell_type}_con{STRM}_vs_'
                  f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_{LN}_recon_rand')
    res_rand.to_hdf(FO.OLF_PATH / f'{file_begin}.h5', 'recon')
#%%

###############################################################################
###############################################################################
###############################################################################
# this does what is above, but one LN at a time
#
# from sklearn import linear_model
# import scipy.linalg as LA
# act_n_np = act_o_np / LA.norm(act_o_np, axis=0)
# LN_i = 6
# # LN_i = 11
# # LN_i = 16
# # LN_i = 19
# w = con_o.values[:, LN_i]
# LN = con_o.columns[LN_i]
# print(LN)
# w = w/LA.norm(w)
# w1 = w
# # w1 = np.random.permutation(w)
#
# lasso = True
# # lasso = False
#
# if lasso:
#     l_norm = 1
#     alphas1 = np.logspace(-4, -1, 20)
#     alphas2 = np.logspace(-4, -1, 20)
#
#     N = 1000
#     def model(alpha):
#         return linear_model.Lasso(alpha=alpha, max_iter=int(1e6))#, tol=1e-10)
# else:
#     l_norm = 2
#     alphas1 = np.logspace(-3, 2, 20)
#     alphas2 = np.logspace(-3, 2, 20)
#     N = 1000
#     def model(alpha):
#         return linear_model.Ridge(alpha=alpha)
#
# # ALPHA = .0001
#
# # this calculates the reconstruction error with the true LN
# results = pd.DataFrame(index = np.arange(len(alphas1)),
#                        columns=['alpha', 'error', 'coef_norm'])
# for i, alph in enumerate(alphas1):
#     clf = model(alph)
#     clf.fit(act_n_np, w1)
#     bias = clf.intercept_
#     error = LA.norm(w1 - act_n_np @ clf.coef_ - bias)
#     coef_norm = LA.norm(clf.coef_, ord=l_norm)
#     results.loc[i, 'alpha'] = alph
#     results.loc[i, 'error'] = error
#     results.loc[i, 'coef_norm'] = coef_norm
#     # print(error, bias, coef_norm, clf.n_iter_, clf.n_features_in_, len(clf.coef_))
# print('done')
#
# #%%
#
# # this calculates the reconstruction error with different shuffled LNs
# res_rand = pd.DataFrame(index = np.arange(len(alphas2)*N),
#                        columns=['alpha', 'error', 'coef_norm', 'w'],
#                         dtype=float)
#
# for j in range(N):
#     print(j)
#     w1 = np.random.permutation(w)
#     for i, alph in enumerate(alphas2):
#         clf = model(alph)
#         clf.fit(act_n_np, w1)
#         bias = clf.intercept_
#         error = LA.norm(w1 - act_n_np @ clf.coef_ - bias)
#         coef_norm = LA.norm(clf.coef_, ord=l_norm)
#         res_rand.loc[i*N + j, 'alpha'] = alph
#         res_rand.loc[i*N + j, 'error'] = error
#         res_rand.loc[i*N + j, 'coef_norm'] = coef_norm
#         res_rand.loc[i*N + j, 'w'] = j
#         # coef_norms[i] = LA.norm(clf.coef_, ord=l_norm)
#     # print(clf.intercept_, clf.coef_[:10])
# print('done')

#%%
###############################################################################
###############################################################################
###############################################################################

# in this version instead of first shuffling the w and calculating the
# reconstruction for different alpha, One first chooses an alpha
# and then

# res_rand = pd.DataFrame(index = np.arange(len(alphas2)*N),
#                        columns=['alpha', 'error', 'coef_norm'])
# for i, alph in enumerate(alphas2):
#     print(i, alph)
#     clf = model(alph)
#     for j in range(N):
#         w1 = np.random.permutation(w)
#         clf.fit(act_n_np, w1)
#         bias = clf.intercept_
#         error = LA.norm(w1 - act_n_np @ clf.coef_ - bias)
#         coef_norm = LA.norm(clf.coef_, ord=l_norm)
#         res_rand.loc[i*N + j, 'alpha'] = alph
#         res_rand.loc[i*N + j, 'error'] = error
#         res_rand.loc[i*N + j, 'coef_norm'] = coef_norm
#         # coef_norms[i] = LA.norm(clf.coef_, ord=l_norm)
#     # print(clf.intercept_, clf.coef_[:10])
# print('done')





#%%

###############################################################################
###############################################################################
###############################################################################
# From here on, it is analysis and plotting that is directly done in the paper
# import matplotlib.pyplot as plt





# for LN in LNs_MM:
#     file_begin = (f'results/{cell_type}_con{STRM}_vs_'
#                   f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_{LN}_recon_rand')
#     res_rand[LN] = pd.read_hdf(FO.OLF_PATH / f'{file_begin}.h5')
#%%
# plt.figure()
# # plt.scatter(errors, coef_norms)
# groups = res_rand.groupby('alpha')
# gr = results.groupby('alpha')
# for i, alph in enumerate(alphas1):
#     res_rand_sel = groups.get_group(alph)
#     plt.scatter(res_rand_sel['coef_norm'], res_rand_sel['error'], color=f'C{i}')
#
# for i, alph in enumerate(alphas2):
#     res_sel = gr.get_group(alph)
#     plt.scatter(res_sel['coef_norm'], res_sel['error'], color=f'C{i}', s=50,
#                 marker='s', edgecolors= 'k')
# plt.xlabel('norm of v')
# plt.ylabel('reconstruction error')
# plt.title(LN)
# # plt.yscale('log')
 #%%
# plt.figure()
# # plt.scatter(errors, coef_norms)
# plt.scatter(res_rand['coef_norm'], res_rand['error']*res_rand['coef_norm'])
# plt.scatter(results['coef_norm'], results['error']*results['coef_norm'])
# # plt.yscale('log')
#
#

#%%


# gr = res_rand.groupby('w')
# y_interps = [scipy.interpolate.interp1d(gr.get_group(i)['coef_norm'], gr.get_group(i)['error'], fill_value='extrapolate')
#              for i in range(N)]
#
# v_p = []
# for i in range(0, len(alphas1)-1):
#     x = results['coef_norm'].iloc[i]
#     y = results['error'].iloc[i]
#     ys = np.array([y_interps[i](x) for i in range(N)])
#     p = np.mean(ys<y)
#     print(alphas1[i], x, p)
#     v_p.append([x, p])
# v_p = np.array(v_p)


#%%
# plt.figure()
# # plt.scatter(errors, coef_norms)
# groups = res_rand.groupby('w')
# for i in range(N):
#     res_rand_sel = groups.get_group(i)
#     plt.plot(res_rand_sel['coef_norm'], res_rand_sel['error'], color=f'C{i}', lw=0.5)
#
# plt.plot(results['coef_norm'], results['error'], color='k', lw=1)
# plt.plot(v_p[:, 0], v_p[:, 1], color='k', lw=1)
# plt.xlabel('norm of v')
# plt.ylabel('reconstruction error')
# plt.title(LN)
# # plt.yscale('log')


#%%
###############################################################################
###############################################################################
###############################################################################

# this part uses reconstruction by only allowing a certain number of
# of entries, it's not as clean as the next one.
# although surprisingly it shoes better reconstuction with the real
# cells than shuffled cells

#
# import scipy.linalg as LA
# w = con_o.values[:, 6]
# w = con_o.values[:, 11]
# w = con_o.values[:, 16]
# w = con_o.values[:, 19]
#
# lmbd_n = 21
# # w = np.abs(np.random.randn(21))
#
# w = w/LA.norm(w)
#
# results = np.zeros((lmbd_n, 1000))
# for lmbd in range(1, lmbd_n):
#     vect = np.zeros(170, dtype=bool)
#     vect[:lmbd] = True
#     print('<<<<<<<<', lmbd, '>>>>>>>>')
#     for i in range(1000):
#         rank = 0
#         while rank != lmbd:
#             v = np.random.permutation(vect)
#             act_sel = act_o_np[:, v]
#             rank = np.linalg.matrix_rank(act_sel)
#         res = LA.lstsq(act_sel, w)
#         # print(res[1], res[2], LA.norm(w - act_sel @ res[0]))
#
#         results[lmbd, i] = res[1]
#
# print('done')
# #%%
# import seaborn as sns
# f, ax = plt.subplots()
# sns.violinplot(results.T, ax=ax)
# plt.show()
# #%%
#
# lmbd_n = 21
#
#
# results1 = np.zeros((lmbd_n, 10000))
#
# for i in range(100):
#     # w = np.abs(np.random.randn(21))
#     # w = w/LA.norm(w)
#     w1 = np.random.permutation(w)
#     print('<<<<<<<<', i, '>>>>>>>>')
#     for lmbd in range(1, lmbd_n):
#         vect = np.zeros(170, dtype=bool)
#         vect[:lmbd] = True
#         for j in range(100):
#             rank = 0
#             while rank != lmbd:
#                 v = np.random.permutation(vect)
#                 act_sel = act_o_np[:, v]
#                 rank = np.linalg.matrix_rank(act_sel)
#             res = LA.lstsq(act_sel, w1)
#             # print(res[1], res[2], LA.norm(w - act_sel @ res[0]))
#
#             results1[lmbd, i*100 + j] = res[1]
#
# print('done')
# #%%
#
# import seaborn as sns
# f, ax = plt.subplots()
# sns.violinplot(results[:20].T, ax=ax, color='r')
# sns.violinplot(results1[:20].T, ax=ax, color='b')
# plt.show()
#
# #%%
# res_df = pd.DataFrame(results.T, columns=np.arange(21)).unstack().reset_index(level=0)
# res_df['cell'] = 'P0'
# res1_df = pd.DataFrame(results1.T, columns=np.arange(21)).unstack().reset_index(level=0)
# res1_df['cell'] = 'random'
#
# res_all_df = pd.concat([res_df, res1_df])
# res_all_df.columns = ['lambda', 'error', 'cell']
#
# f, ax = plt.subplots()
# sns.violinplot(data = res_all_df, ax=ax, x='lambda', y='error', hue='cell', split='True',
#                inner="quart")
# plt.show()
