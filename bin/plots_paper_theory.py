#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:56:59 2019

@author: Nikolai M Chapochnikov
"""

# %%
# ################################# IMPORTS ###################################

from plots_paper_import import *
# %%
# ################################  RELOADS  ##################################
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)
importlib.reload(FCS)
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
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(ylabel=r'axon, PCA s.d. ($\sigma_Y$)',
       xlabel=r'soma, PCA s.d. ($\sigma_X$)',
       xticks=[0, 3, 6], yticks=[0, 3, 6])
ax.tick_params(axis='both',  pad=1)

file = f'{PP_THEORY}/sy_sx.'
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
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)




x = np.logspace(-0.8, 0.9, 100)
ax.loglog(x, 10*x, ls='--', lw=1, c='k')
ax.text(1, 12, r'$\sigma_Y\propto \sigma_X$', ha='right', va='bottom')

x = np.logspace(0.2, 2, 100)
ax.loglog(x, 0.02*x**(1/3), ls='--', lw=1, c='k')
ax.text(9, 0.06, r'$\sigma_Y\propto \sigma_X^{1/3}$',
        ha='left', va='top')

ax.set(ylabel=r'axon, PCA s.d. ($\sigma_Y$)',
       xlabel=r'soma, PCA s.d. ($\sigma_X$)',
       xticks=[0.1, 1, 10, 100], yticks=[0.1, 1, 10, 100],
      xticklabels=[0.1, 1, 10, 100],
      yticklabels=[0.1, 1, 10, 100])
ax.tick_params(axis='both',  pad=1)
ax.minorticks_off()
# ax.set_xticks([0.1, 1, 10, 100])

file = f'{PP_THEORY}/sy_sx_log.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #############################################################################
# #####################  SHOWING EXAMPLE OF TRANSFORMATION  ###################
# #############################################################################


data_i = 3
D = 2
n_clusters = 2
file = FO.OLF_PATH / \
       f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
X = np.load(file)
# df = pd.DataFrame(X)
D, N = X.shape
K = 2
rho = 1
file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}.npy'
Y_NNC = np.load(file)
file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}.npy'
Z_NNC = np.load(file)


U_X, s_X, Vt_X = LA.svd(X, full_matrices=False)
U_X = -U_X
Vt_X = -Vt_X
# s_Y_LC = s_X.copy()
s_Y_LC = FOC.damp_sx(s_X, len(X.T), 1)
Y_LC = U_X @ np.diag(s_Y_LC)@ Vt_X
Z_LC = np.diag(s_Y_LC[:K]) @ Vt_X[:K] # this Z_LC is the one that is the special case without rotation
# let's make one with a rotation:
rot_angle = np.pi/15  # used for the paper
# rot_angle = np.pi/4  # gives the same as for the NNC

# rot_angle = 0
rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)]])
Z_LC = rot @ Z_LC


v_max = 1.5
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
for data, label in [(X, 'X'), (Z_NNC, 'NNC'), (Z_LC, 'LC')]:
    df = pd.DataFrame(data)

    if label == 'LC':
        act_map, divnorm = FP.get_div_color_map(-v_max, -v_max, v_max)
        cb_ticks = [-1, 1]
    else:
        divnorm = matplotlib.colors.Normalize(0, v_max)
        act_map = 'Oranges'
        cb_ticks = [0, 1]
    if label == 'X':
        ylabel = r'$x_i$'
        title = f'        ORN soma activity {Xtstex}'
    else:
        ylabel = r'$z_i$'
        title = f'     {label}, LN activity {Ztex}'

    if label == 'NNC':
        xlabel = f'samples (T={N})'
        pad_down = 0.2
    else:
        xlabel = ''
        pad_down = 0.1
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                       cb_ticks, pads=[pad_l, pad_r, pad_down, pad_up],
                                     extend='neither',
                       squeeze=squeeze, do_vert_spl=False, SQ=SQ, CB_DX=0.06,
                                     cb_title_font=cb_title_font)
    ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel=xlabel)

    file = f'{PP_THEORY}/dataset{data_i}_{label}.'
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
# importlib.reload(matplotlib)
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
                r'$x_2$, $y_2$', xticks=[0, 1],
                yticks=[0, 1],
                pca_line_scale1=0., pca_line_scale2=0.,
                show_cc=False,
                s=size, c='k', lw=lw, label=f'soma {Xttex}')
s2 = ax.scatter(*Y_LC[:2], s=size, c='magenta', lw=lw, label=f'LC, axon {Yttex}',
           alpha=0.9)
legend1 = ax.legend(frameon=False, ncol=1, columnspacing=0.8, borderpad=0,
          loc='upper left', handletextpad=0.4, handlelength=0.8,
          bbox_to_anchor=(-.1, 1.25), labelspacing=0.2)
for i in range(N):
    ax.plot([X[0, i], Y_LC[0, i]],[X[1, i], Y_LC[1, i]], c='k', lw=0.3, ls='-',
            alpha=0.2)
l1, = ax.plot([0, U_X[0, 0]*sigmax[0]], [0, U_X[0, 1]*sigmax[0]], c='salmon', lw=2,
        label=r'$\sigma_{X, 1} \mathbf{u}_{X, 1}$', alpha=alpha, clip_on=False)
l2, = ax.plot([0, U_X[1, 0]*sigmax[1]], [0, U_X[1, 1]*sigmax[1]], c='mediumpurple', lw=2,
        label=r'$\sigma_{X, 2} \mathbf{u}_{X, 2}$', alpha=alpha, clip_on=False)
l3, = ax.plot([0, U_X[0, 0]*sigmay[0]], [0, U_X[0, 1]*sigmay[0]], c='r',
        label=r'$\sigma_{Y, 1} \mathbf{u}_{Y, 1}$', clip_on=False)
l4, = ax.plot([0, U_X[1, 0]*sigmay[1]], [0, U_X[1, 1]*sigmay[1]], c='darkviolet',
        label=r'$\sigma_{Y, 2} \mathbf{u}_{Y, 2}$', clip_on=False)
l5, = ax.plot([-1, -1], [-0.5, -0.5], label=r'$\mathbf{w}_1, \mathbf{w}_2$',
        alpha=0.9, c='g') # not visible, just for legend
plot_lines = [l1, l2, l3, l4, l5]
ax.set_xlim(-.15, v_max)
ax.set_ylim(-.15, v_max)

kwargs = {'length_includes_head':True, 'width':0.04, 'color':'g',
          'alpha':0.9, 'head_width':0.1, 'lw':0, 'clip_on':False}
# ax.arrow(0, 0, W[0, 0], W[1, 0], **kwargs)
for k in range(K):
    ax.arrow(0, 0, W[0, k], W[1, k], **kwargs)


legend2 = plt.legend(handles=plot_lines,
                          frameon=False, ncol=1, columnspacing=0.8, borderpad=0,
                            loc='upper right', handletextpad=0.4, handlelength=0.8,
                            bbox_to_anchor=(1.1, 1.25), labelspacing=0.7
                          )

# Add the legend manually to the current Axes.
ax.add_artist(legend1)
ax.add_artist(legend2)



# font_size = 9
# ax.text(0, 3, r'$\sigma_{X, 1} \mathbf{u}_1$', c='r', fontsize=font_size)
# ax.text(0.5, -0.7, r'$\sigma_{X, 2} \mathbf{u}_2$', c='g', fontsize=font_size)

file = f'{PP_THEORY}/dataset{data_i}_scatterXY_LC.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


# %%
# #############################################################################
# #############################################################################
# ########################    NNC          ####################################
# #############################################################################

# now let's plot the results

n_clusters = 2


act_map = 'Oranges'
v_max = 1.5
divnorm = matplotlib.colors.Normalize(0, v_max)
title = f'LN activity patterns {Ztex}'
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
                r'$x_2$, $y_2$', xticks=[0, 1],
                yticks=[0, 1],
                pca_line_scale1=0., pca_line_scale2=0.,
                show_cc=False,
                s=size, c='k', lw=lw, label=f'soma {Xttex}')
ax.scatter(*Y_NNC[:2], s=size, c='dodgerblue', lw=lw, label=f'NNC, axon {Yttex}',
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

# ax.legend(frameon=False, ncol=3, columnspacing=1., borderpad=0,
#           loc='upper right', handletextpad=0.4, handlelength=1,
#           bbox_to_anchor=(1.15, 1.3))
ax.legend(frameon=False, ncol=2, columnspacing=1.2, borderpad=0,
          handletextpad=0.2,
          loc='upper center', scatterpoints=1,
          bbox_to_anchor=(0.5, 1.25),
          handlelength=0.8, labelspacing=0.2)

file = f'{PP_THEORY}/dataset{data_i}_scatterXY_NNC.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #############################################################################
# #############################################################################
# ########################    variances        ################################
# #############################################################################
# s_X
# s_Y_LC
_, s_Y_NNC, _ = LA.svd(Y_NNC, full_matrices=False)


k1 = 2
# Y_str = r', $\mathbf{Y}$'
Y_str = f', {Ytex}'

Nl1 = f'NNC' + Y_str
Ll1 = f'LC' + Y_str

# now plotting the variances of the uncentered PCA instead of singular values
def plot_sv1(datas, order=None):
    x_i = np.arange(1, 3)
    pads = (0.37, 0.4, 0.3, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 10*SQ, 15*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    for data in datas:
        s_data = data[0]**2/200
        ax.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                c=data[4], label=data[5], markeredgewidth=data[6])
    ax.set(ylabel='variance', xlabel=f'PCA direction',
           xticks=[1, 2], ylim=[0 , None], xlim=[0.75, 2.25],
           yticks=[0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both',  pad=1)
    handles, labels = ax.get_legend_handles_labels()
    if order is None:
        order = np.arange(len(handles))
    leg = ax.legend([handles[idx] for idx in order],
                    [labels[idx] for idx in order],
                    bbox_to_anchor=(1.6, 1.15), loc='upper right',
                    frameon=False, borderpad=0,
                    handletextpad=0.4
                    )
    leg.get_frame().set_linewidth(0.0)
    # leg = ax.legend()
    # leg.get_frame().set_linewidth(0.0)
    return f


datas = [[s_X, '.-', 2, 4, 'k', Xtstex, None],
         [s_Y_LC, 's-', 2, 4, 'magenta', Ll1, 0],
         [s_Y_NNC, 's--', 1.5, 3, 'dodgerblue', Nl1, 0]]

f = plot_sv1(datas, [0, 1, 2])
file = f'{PP_THEORY}/variances'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)






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
# ############################## NNC CLUSTERING   #############################
# ##############################                  #############################
# #############################################################################
# #############################################################################
# #############################################################################
# graphs in the supplement, showing relationship between n_cluster, rho, K

# importlib.reload(FP)
n_clusters = 2
D = 10
act_map = 'Oranges'
v_max = 1.5
divnorm = matplotlib.colors.Normalize(0, v_max)
title = f'Input activity patterns {Xtstex}'
cb_title = ''
cb_ticks = [0, 0.5, 1]
ylabel = r'$x_i$'
size=1
lw=0
pad_up = 0.2
# for data_i in [1, 2, 3]:  # the 3rd dataset is plotted above
for data_i in [1, 2]:
# for data_i in [1]:
    if data_i == 3:
        D = 2
    else:
        D = 10
    file = FO.OLF_PATH / \
           f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
    X = np.load(file)
    df = pd.DataFrame(X)
    D, N = X.shape
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                       cb_ticks, pads=[0.2, 0.4, 0.3, pad_up], extend='neither',
                       squeeze=0.1, do_vert_spl=False, SQ=SQ/10*9)
    ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel=f'samples (T={N})')

    file = f'{PP_WrhoK}/dataset{data_i}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

    fs, axs = FP.calc_fs_ax([0.3, 0.3, 0.3, 0.15], 9 * SQ, 9 * SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    FP.plot_scatter(ax, X[0], X[1], r'$x_1$',
                    r'$x_2$', xticks=[0, 1],
                    yticks=[0, 1],
                    pca_line_scale1=0., pca_line_scale2=0.,
                    show_cc=False,
                    s=size, c='k', lw=lw)
    ax.set_xlim(0, v_max)
    ax.set_ylim(0, v_max)
    # ax.axis('off')
    file = f'{PP_WrhoK}/dataset{data_i}_scatter.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

#%%
# now let's plot the results

n_clusters = 2


act_map = 'Oranges'
v_max = 1.5
divnorm = matplotlib.colors.Normalize(0, v_max)
title = f'LN activity patterns {Ztex}'
ylabel = r'$z_i$'
cb_title = ''
cb_ticks = [0, 1]
cb_ticks = None
wtexs = {}
wtexs[1] = r'$\mathbf{w}_1$'
wtexs[2] = r'$\mathbf{w}_2$'
wtexs[3] = r'$\mathbf{w}_3$'
# for data_i in [1, 2, 3]: # in case you also want to do the 3rd dataset
for data_i in [1, 2]:
# for data_i in [3]:
# for data_i in [1]:
    if data_i == 3:
        iters = [(1, 2)]
        D = 2
    else:
        iters = itertools.product([0.1, 1, 10], [2, 3])
        D = 10
    file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
    X = np.load(file)
    D, N = X.shape
    for rho, K in iters:
    # for rho, K in itertools.product([0.1], [2]):
        file = FO.OLF_PATH / f'results/dataset{data_i}/Y_K{K}_rho{rho}.npy'
        Y = np.load(file)
        file = FO.OLF_PATH / f'results/dataset{data_i}/Z_K{K}_rho{rho}.npy'
        Z = np.load(file)


        df = pd.DataFrame(Z)
        D, N = X.shape
        W0 = Y @ Z.T / N  # original W
        W = W0/LA.norm(W0, axis=0, keepdims=True)

        corr = np.corrcoef(W0.T)
        links = sch.linkage(corr, method='average', optimal_ordering=True)
        new_order = sch.leaves_list(links)


        divnorm = matplotlib.colors.Normalize(0, df.values.max())
        f, ax, _ = FP.plot_full_activity(df.iloc[new_order], act_map, divnorm,
                                         title, cb_title,
                                         cb_ticks, pads=[0.2, 0.4, 0.3, pad_up],
                                         extend='neither', squeeze=0.1,
                                         do_vert_spl=False,
                                         set_ticks_params=False,SQ=SQ*9/10)
        ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel='samples')

        file = f'{PP_WrhoK}/dataset{data_i}_K{K}_rho{rho}_Z.'
        FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
        FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


        fs, axs = FP.calc_fs_ax([0.3, 0.3, 0.3, 0.01], 9 * SQ, 9 * SQ)
        f = plt.figure(figsize=fs)
        ax = f.add_axes(axs)
        ax.plot([-1, -1], [-0.5, -0.5], label=r'$\mathbf{w}_{k}$',
                alpha=0.9, c='g')
        FP.plot_scatter(ax, X[0], X[1], r'$x_1$, $y_1$',
                        r'$x_2$, $y_2$', xticks=[0, 1],
                        yticks=[0, 1],
                        pca_line_scale1=0., pca_line_scale2=0.,
                        show_cc=False,
                        s=size, c='k', lw=lw, label=f'input {Xttex}')
        ax.scatter(*Y[:2], s=size, c='r', lw=lw, label=f'output {Yttex}')
        # ax.scatter(*W[:2], s=size, c='g', lw=3, label=r'$\mathbf{w}_{k}$',
        #            marker='+')
        kwargs = {'length_includes_head':True, 'width':0.04, 'color':'g',
                  'alpha':0.9, 'head_width':0.1, 'lw':0}
        ax.arrow(0, 0, W[0, 0], W[1, 0], **kwargs)
        for k in range(1, K):
            ax.arrow(0, 0, W[0, k], W[1, k], **kwargs)
        ax.set_xlim(0, v_max)
        ax.set_ylim(0, v_max)
        ax.legend(frameon=False, borderpad=0, handletextpad=0.2,
                  loc='upper left', scatterpoints=1,
                  bbox_to_anchor=(0.6, 1.05), labelspacing=0.,
                  handlelength=1)

        file = f'{PP_WrhoK}/dataset{data_i}_K{K}_rho{rho}_XYW_scatter.'
        FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
        FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

        # correlation between W plot
        fs, axs = FP.calc_fs_ax([0.2, 0.4, 0.3,pad_up], 7 * SQ, 7 * SQ)
        f = plt.figure(figsize=fs)
        ax = f.add_axes(axs)
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0],
                            axs[1], CB_W / fs[0], axs[3]])
        corr = pd.DataFrame(corr[new_order][:, new_order])
        cp = FP.imshow_df2(corr, ax, cmap=corr_cmap, vlim=1,
                           splits_x=[], splits_y=[], show_lab_x=True,
                           rot=0)
        ax.set_xticklabels([wtexs[i] for i in range(1, K + 1)])
        ax.set_yticklabels([wtexs[i] for i in range(1, K + 1)])

        add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])

        entries = FG.get_entries(corr, diag=False)
        max_rect_corr = FG.rectify(entries).mean()
        print(data_i, K, rho, 'mean rectified corr', max_rect_corr)
        ax.set_title(r'$\overline{r}_+$' + f'={max_rect_corr:.2f}')

        file = f'{PP_WrhoK}/dataset{data_i}_K{K}_rho{rho}_W_corr.'
        FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
        FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


print('Final done')
#%%