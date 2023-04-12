#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2022

@author: Nikolai M Chapochnikov


files used:
data_i = 4
dataset_{D}D_{n_clusters}clusters.npy
dataset{data_i}/whiteNNC_Y_K{K}_rho{rho}.npy
dataset{data_i}/whiteNNC_Z_K{K}_rho{rho}.npy

FROM: olf_circ_noM-offline_sims.py
"""

# %%
# ################################# IMPORTS ###################################

from plots_import import *

import functions.white_circ_offline as FOC  # changing from the
# usual olfactory circuit to the circuit without LN-LN connections
# only used for the function damp_sx
# %%
# ################################  RELOADS  ##################################
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)
# importlib.reload(par_con)
importlib.reload(par_act)
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ##############################                  #############################
# ##############################                  #############################
# ######################### LC and NNC computation   ##########################
# ##############################                  #############################
# #############################################################################
# #############################################################################
# #############################################################################
# graphs for figure 6, showing the transformation for LC and NNC
# on an artificial dataset
# Theoretical plots showing the relationship between s_x and s_y
# for different rho.
# There is also gamma and T, so we'll need to fix that too.

x = np.linspace(0, 6, 100)
rhos = {0.1: 'C1', 0.2: 'C2', 0.4: 'C3', 1: 'C4', 2: 'C5', 10: 'C6',
        50: 'C7'}


pads = (0.37, 0.15, 0.3, 0.2)
# graph_width = pads[0] + pads[1] + 15 * SQ
fs, axs = FP.calc_fs_ax(pads, 15 * SQ, 15 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.plot(x, x, c='C0')
ax.text(6.2, 6, 0)
rhos_sel = [0.1, 0.2, 0.4, 1, 2, 10]
for rho in rhos_sel:
    ax.plot(x, FOC.damp_sx(x, 1, rho=rho), c=rhos[rho])
    ax.text(6.2, FOC.damp_sx(6, 1, rho=rho), rho)
ax.text(6.2, 6.8, r'$\rho:$')
ax.set(ylabel=r'axon, PCA SD ($\sigma_Y$)',
       xlabel=r'soma, PCA SD ($\sigma_X$)',
       xticks=[0, 3, 6], yticks=[0, 3, 6])

file = f'{PP_THEORY_NOM}/sy_sx.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
#
#
#
#

x_end = 100
x = np.logspace(-1.5, 2, 100)
# rhos = [0.1, 1, 10, 50]
pads = (0.37, 0.15, 0.3, 0.2)

fs, axs = FP.calc_fs_ax(pads, 15 * SQ, 15 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.loglog(x, x, c='C0')
ax.text(x_end + 50, x_end, 0)
rhos_sel = [0.1, 1, 10, 50]
for rho in rhos_sel:
    ax.loglog(x, FOC.damp_sx(x, 1, rho=rho), c=rhos[rho])
    ax.text(x_end + 50, FOC.damp_sx(x_end, 1, rho=rho), rho)
ax.text(x_end + 50, x_end + 200, r'$\rho:$')

x = np.logspace(-0.8, 0.9, 100)
ax.loglog(x, 10*x, ls='--', lw=1, c='k')
ax.text(1, 12, r'$\sigma_Y\propto \sigma_X$', ha='right', va='bottom')

ax.set(ylabel=r'axon, PCA SD ($\sigma_Y$)',
       xlabel=r'soma, PCA SD ($\sigma_X$)',
       xticks=[0.1, 1, 10, 100], yticks=[0.1, 1, 10, 100],
       xticklabels=[0.1, 1, 10, 100],
       yticklabels=[0.1, 1, 10, 100])
ax.minorticks_off()
# ax.set_xticks([0.1, 1, 10, 100])

file = f'{PP_THEORY_NOM}/sy_sx_log.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

 # %%
# #############################################################################
# #####################  SHOWING EXAMPLE OF TRANSFORMATION  ###################
# #############################################################################


data_i = 4
D = 2
n_clusters = 2
file = RESULTS_PATH / f'dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
X = np.load(file)
# df = pd.DataFrame(X)
D, N = X.shape
K = 2
rho = 1
file = RESULTS_PATH / f'dataset{data_i}/whiteNNC_Y_K{K}_rho{rho}.npy'
Y_NNC = np.load(file)
file = RESULTS_PATH / f'dataset{data_i}/whiteNNC_Z_K{K}_rho{rho}.npy'
Z_NNC = np.load(file)


U_X, s_X, Vt_X = LA.svd(X, full_matrices=False)
U_X = -U_X
Vt_X = -Vt_X
# s_Y_LC = s_X.copy()
s_Y_LC = FOC.damp_sx(s_X, len(X.T), 1)
Y_LC = U_X @ np.diag(s_Y_LC)@ Vt_X
Z_LC = np.diag(s_Y_LC[:K]) @ Vt_X[:K] # this Z_LC is the one that is the special case without rotation
# let's make one with a rotation:
rot_angle = 3.2*np.pi/8  # used for the paper
# rot_angle = np.pi/4  # gives the same as for the NNC

# rot_angle = 0
rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)]])
Z_LC = rot @ Z_LC

# this gives the same result, just coming from numerical simulations
# file = RESULTS_PATH / f'dataset{data_i}/whiteLC_Y_K{K}_rho{rho}.npy'
# Y_LC = np.load(file)
# file = RESULTS_PATH / f'dataset{data_i}/whiteLC_Z_K{K}_rho{rho}.npy'
# Z_LC = np.load(file)

v_max = 3
cb_title = ''

size=1
lw=0
pad_up = 0.2
pad_l = 0.3
pad_r = 0.3

# df_width = graph_width - pad_l - pad_r
# squeeze = df_width/SQ/N
squeeze = 0.063
# df_width = squeeze * SQ * N
labels = {'X': 'X', 'LC': r'LC\textsuperscript{*}', 'NNC': r'NNC\textsuperscript{*}'}
for data, label in [(X, 'X'), (Z_NNC, 'NNC'), (Z_LC, 'LC')]:
    df = pd.DataFrame(data)

    if label == 'LC':
        act_map, divnorm = FP.get_div_color_map(-v_max, -v_max, v_max)
        cb_ticks = [-2, 2]
    else:
        divnorm = mpl.colors.Normalize(0, v_max)
        act_map = 'Oranges'
        cb_ticks = [0, 2]
    if label == 'X':
        ylabel = r'$x_i$'
        title = f'        ORN soma activity {Xtstex}'
    else:
        ylabel = r'$z_i$'
        title = f'     {labels[label]}, LN activity {Ztex}'

    if label == 'NNC':
        xlabel = f'sample (T={N})'
        pad_down = 0.2
    else:
        xlabel = ''
        pad_down = 0.1
        pad_down = 0.1
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                                     cb_ticks, pads=[pad_l, pad_r, pad_down, pad_up],
                                     extend='neither',
                                     squeeze=squeeze, do_vert_spl=False, SQ=SQ, CB_DX=0.06,
                                     cb_title_font=cb_title_font)
    ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel=xlabel)

    file = f'{PP_THEORY_NOM}/dataset{data_i}_{label}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


# %%
# #############################################################################
# #############################################################################
# ###########################   LC    #########################################
# #############################################################################

# n = 500
# x = np.random.randn(2, n)
# x[0] = x[0] * 2
# x[1] = x[1] * 0.75
# rot = SS.ortho_group.rvs(2, random_state=0)
# x = rot @ x
## x[0, x[0, :]<0] = 0
## x[1, x[1, :]<0] = 0
## x = x[:, np.sum(x, axis=0)>0]

# x = x[:, x[0, :]>0]
# x = x[:, x[1, :]>0]
# v_max = np.max(x)
# v_max = 5
# print(x.shape)
# not working for now
# https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib
# pgf_with_latex = {
#     "text.usetex": True,            # use LaTeX to write all text
#     "pgf.rcfonts": False,           # Ignore Matplotlibrc
#     "pgf.preamble": [
#         r'\usepackage{color}'     # xcolor for colours
#     ]
# }
# matplotlib.rcParams.update(pgf_with_latex)
# importlib.reload(mpl)
alpha = 0.8
n1 = -1
# U, s, Vt = LA.svd(x, full_matrices=False)
sigmax = s_X/np.sqrt(len(X.T))
sigmay = s_Y_LC/np.sqrt(len(X.T))


W0 = Y_LC @ Z_LC.T / N  # original W
W = W0 / LA.norm(W0, axis=0, keepdims=True)

size = 2
pads = (0.3, 0.1, 0.27, 0.25)
fs, axs = FP.calc_fs_ax(pads, 15 * SQ, 15 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.hlines(0, -1, 10, zorder=1, lw=0.3, color='gray', ls='--')
ax.vlines(0, -1, 10, zorder=1, lw=0.3, color='gray', ls='--')
FP.plot_scatter(ax, X[0], X[1], r'$x_1$, $y_1$',
                r'$x_2$, $y_2$', xticks=[0, 1, 2],
                yticks=[0, 1, 2],
                pca_line_scale1=0., pca_line_scale2=0.,
                show_cc=False,
                s=size, c='k', lw=lw, label=f'soma {Xttex}')
s2 = ax.scatter(*Y_LC[:2], s=size, c='magenta', lw=lw, label=f"{labels['LC']}, axon {Yttex}",
                alpha=0.9)

for i in range(N):
    ax.plot([X[0, i], Y_LC[0, i]],[X[1, i], Y_LC[1, i]], c='k', lw=0.3, ls='-',
            alpha=0.2)
# l1, = ax.plot([0, U_X[0, 0]*sigmax[0]], [0, U_X[0, 1]*sigmax[0]], c='salmon', lw=2,
#               label=r'$\sigma_{X, 1} \mathbf{u}_{X, 1}$', alpha=alpha, clip_on=False)
# l2, = ax.plot([0, U_X[1, 0]*sigmax[1]], [0, U_X[1, 1]*sigmax[1]], c='mediumpurple', lw=2,
#               label=r'$\sigma_{X, 2} \mathbf{u}_{X, 2}$', alpha=alpha, clip_on=False)
# l3, = ax.plot([0, U_X[0, 0]*sigmay[0]], [0, U_X[0, 1]*sigmay[0]], c='r',
#               label=r'$\sigma_{Y, 1} \mathbf{u}_{Y, 1}$', clip_on=False)
# l4, = ax.plot([0, U_X[1, 0]*sigmay[1]], [0, U_X[1, 1]*sigmay[1]], c='darkviolet',
#               label=r'$\sigma_{Y, 2} \mathbf{u}_{Y, 2}$', clip_on=False)
l5, = ax.plot([-1, -1], [-0.5, -0.5], label=r'$\mathbf{w}_1, \mathbf{w}_2$',
              alpha=0.9, c='g') # not visible, just for legend

ax.set_xlim(-.25, v_max)
ax.set_ylim(-.25, v_max)

kwargs = {'length_includes_head':True, 'width':0.04, 'color':'g',
          'alpha':0.9, 'head_width':0.1, 'lw':0, 'clip_on':False}
# ax.arrow(0, 0, W[0, 0], W[1, 0], **kwargs)
for k in range(K):
    ax.arrow(0, 0, W[0, k], W[1, k], **kwargs)

ax.legend(ncol=2, loc='upper right',
          bbox_to_anchor=(1.09, 1.21), labelspacing=0.3)


file = f'{PP_THEORY_NOM}/dataset{data_i}_scatterXY_LC.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# #############################################################################
# ########################    NNC          ####################################
# #############################################################################

# now let's plot the results

n_clusters = 2

ylabel = r'$z_i$'
cb_title = ''
cb_ticks = [0, 1]
wtexs = {}
wtexs[1] = r'$\mathbf{w}_1$'
wtexs[2] = r'$\mathbf{w}_2$'
wtexs[3] = r'$\mathbf{w}_3$'

W0 = Y_NNC @ Z_NNC.T / N  # original W
W = W0 / LA.norm(W0, axis=0, keepdims=True)


fs, axs = FP.calc_fs_ax([0.3, 0.1, 0.27, 0.25], 15 * SQ, 15 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.hlines(0, -1, 10, zorder=1, lw=0.3, color='gray', ls='--')
ax.vlines(0, -1, 10, zorder=1, lw=0.3, color='gray', ls='--')
FP.plot_scatter(ax, X[0], X[1], r'$x_1$, $y_1$',
                r'$x_2$, $y_2$', xticks=[0, 1, 2],
                yticks=[0, 1, 2],
                pca_line_scale1=0., pca_line_scale2=0.,
                show_cc=False,
                s=size, c='k', lw=lw, label=f'soma {Xttex}')
ax.scatter(*Y_NNC[:2], s=size, c='dodgerblue', lw=lw, label=f"{labels['NNC']}, axon {Yttex}",
           alpha=0.9)

ax.plot([-1, -1], [-0.5, -0.5], label=r'$\mathbf{w}_1, \mathbf{w}_2$',
        alpha=0.9, c='g') # not visible, just for legend



for i in range(N):
    ax.plot([X[0, i], Y_NNC[0, i]],[X[1, i], Y_NNC[1, i]], c='k', lw=0.3, ls='-',
            alpha=0.2)
# ax.scatter(*W[:2], s=size, c='g', lw=3, label=r'$\mathbf{w}_{k}$',
#            marker='+')

ax.set_xlim(-.15, v_max)
ax.set_ylim(-.15, v_max)
kwargs = {'length_includes_head':True, 'width':0.04, 'color':'g',
          'alpha':0.9, 'head_width':0.1, 'lw':0}
# ax.arrow(0, 0, W[0, 0], W[1, 0], **kwargs)
for k in range(K):
    ax.arrow(0, 0, W[0, k], W[1, k], **kwargs)


ax.legend(ncol=2, loc='upper right',
          bbox_to_anchor=(1.09, 1.21), labelspacing=0.3)

file = f'{PP_THEORY_NOM}/dataset{data_i}_scatterXY_NNC.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


print('Final done')
 #%%