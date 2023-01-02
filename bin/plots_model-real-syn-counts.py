#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 2022

@author: Nikolai M Chapochnikov

this file is very similar to plots_models-ORN-act
but modified slightly to print just the output of one model

files used:
sims_real-synaptic-counts/...
FROM olf_circ_online_sims_real-syn-counts.py
"""

# %%
# ################################# IMPORTS ###################################
from plots_import import *
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
# ############################## WHITENING GRAPHS #############################
# ##############################                  #############################
# #############################################################################
# #############################################################################
# #############################################################################

# %%
# #############################################################################
# #############################################################################
# #############################################################################
Y = {}
Z = {}
for side in ['L', 'R']:
    file = RESULTS_PATH / 'sims_real-synaptic-counts'/ f'Y_{side}.h5'
    Y[side] = pd.DataFrame(pd.read_hdf(file))
    file = RESULTS_PATH / 'sims_real-synaptic-counts'/ f'Z_{side}.h5'
    Z[side] = pd.DataFrame(pd.read_hdf(file))
Y = (Y['L'] + Y['R'])/2
# Z = Z1['L'].copy()
LN_names = [n[:-2] for n in Z['L'].index]
LN_names[-1] = 'Picky 0'
# Z[:] = (Z1['L'].values + Z1['R'].values)/2
Z['L'].index = LN_names
Z['R'].index = LN_names

odor_order = par_act.odor_order
# ORN_order = par_act.ORN  # this is just the alphabetical order
ORN_order = par_act.ORN_order
DATASET = 3

# sf = 'CS'
# def sim_func(X):
#     if isinstance(X, pd.DataFrame):
#         X = X.values
#     X = FG.get_norm_np(X.T)
#     return X.T @ X

sf = 'corr'  # sf as similarity function (measure)
def sim_func(X: Union[pd.DataFrame, np.ndarray]):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = FG.get_ctr_norm_np(X.T)
    return X.T @ X

def get_cc_data(X: Union[pd.DataFrame, np.ndarray],
                X_sel: Union[pd.DataFrame, np.ndarray]):
    X_cc_c = FG.get_entries(sim_func(X))
    X_cc_p = FG.get_entries(sim_func(X.T))
    X_cc_p_sel = FG.get_entries(sim_func(X_sel.T))
    X_cc_p_o = sim_func(X.T)[~big_block]
    return X_cc_c, X_cc_p, X_cc_p_sel, X_cc_p_o

def order_act_df(X):
    conc_levels = X.columns.unique(level='conc')
    idx = pd.MultiIndex.from_product([odor_order, np.sort(conc_levels)[::-1]])
    return X.loc[:, idx]

def get_important_data(Y, Z):
    # problem seems to be here already
    Y = order_act_df(Y.loc[ORN_order])
    # Y_sel = Y.xs(conc_sel, axis=1, level='conc').copy()
    Y_sel = Y.loc[:, (slice(None), conc_sel)].copy()
    Z = order_act_df(Z)
    # Z_sel = Z.xs(conc_sel, axis=1, level='conc').copy()
    Z_sel = Z.loc[:, (slice(None), conc_sel)].copy()
    W = Y @ Z.T / N
    M = Z @ Z.T / N
    return Y, Y_sel, Z, Z_sel, W, M


conc_sel = (5, 4)  # this is just for the plotting -> new version
# conc_sel = 4  # this is just for the plotting
act = FO.get_ORN_act_data(DATASET).T  # matrix ORNs x stimuli
act_m = act.groupby(axis=1, level=('odor', 'conc')).mean()
# act_m = act.mean(axis=1, level=('odor', 'conc'))  # old version
X = order_act_df(act_m.loc[ORN_order])
# X = act_m.loc[ORN_order].T.reindex(odor_order, level='odor').T

# preparation for looking at the correlation outside of the same odor
block = np.ones((5, 5))
n_odors = int(X.shape[1]/5)
big_block = LA.block_diag(*[block for _ in range(n_odors)]) - np.eye(n_odors*5)
big_block = big_block.astype(bool)



U_X, s_X, Vt_X = LA.svd(X, full_matrices=False)
# X_sel = X.xs(conc_sel, axis=1, level='conc').copy()
X_sel = X.loc[:, (slice(None), conc_sel)].copy()
N = X.shape[1]
X_cc_c, X_cc_p, X_cc_p_sel, X_cc_p_o = get_cc_data(X, X_sel)



conc = 'all'

if SAVE_PLOTS:
    PP_WHITE = PATH_PLOTS / f'simulations-real-syn-counts'
    PP_WHITE.mkdir(exist_ok=True)


Y, Y_sel, Z['L'], _, _, _ = get_important_data(Y, Z['L'])
Y, Y_sel, Z['R'], _, _, _ = get_important_data(Y, Z['R'])

Y_cc_c, Y_cc_p, Y_cc_p_sel, Y_cc_p_o = get_cc_data(Y, Y_sel)

y_real = f'{Ytex}'
col = 'r'

print('done')

# %%
# #############################################################################
# ######################  change in left singular vectors #####################
# #####################  with and without M=0  ###############################
# #############################################################################

def plot_XYtrans(data):
    """
    transformtion of the left singular vectors from X to Y
    Parameters
    ----------
    data

    Returns
    -------

    """
    U_X = LA.svd(data[0])[0]
    U = LA.svd(data[1])[0]
    title = data[2]
    name = data[3]
    cond = data[4]
    myset = []
    product = U.T @ U_X
    mysum = np.sum(product)
    for i in range(len(U_X)):
        U_new = U.copy()
        U_new[:, i] *= -1
        product2 = U_new.T @ U_X
        if np.max(product2[i]) > np.max(product[i]):
            myset = myset + [i]
    U[:, myset] *= -1
    toplot = U.T @ U_X

    pads = (0.35, 0.1, 0.3, 0.2)
    if cond:
        pads = (0.35, 0.4, 0.3, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(toplot, ax, vlim=[-1, 1], show_lab_y=False,
                       title=title, show_lab_x=False,
                       cmap=corr_cmap)
    ax.set_xticks([0, 4, 9, 14, 20], [1, 5, 10, 15, 21])
    ax.set_yticks([0, 4, 9, 14, 20], [1, 5, 10, 15, 21])
    if title == 'LC-8' or title == "LC'-8":
        ax.set_xticks([0, 4, 7, 9, 14, 20], [1, 5, 8, 10, 15, 21])
    ax.tick_params('both', which='minor', length=0)
    # ax.tick_params('both', which='major', bottom=False, left=False, direction='in')
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # [i.set_linewidth(0.5) for i in ax.spines.values()]
    ax.set_xlabel(r'PCA direction $\mathbf{u}_{X,i}$')
    ax.set_ylabel(r'PCA direction $\mathbf{u}_{Y,i}$')
    ax.set_xticks(np.arange(-.5, 21, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 21, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    # ax.set_yticks(np.arange(len(toplot)))
    if cond:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = add_colorbar_crt(cp, ax_cb, '', [-1, 0, 1])

    file = f'{PP_WHITE}/ORN_act_{name}_LSV'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


title = ''
datas = [[X, Y, 'real' + title, f'X-real', False]]


for data in datas:
    plot_XYtrans(data)

print('done')

# %%

# only change is cross correlation on the diagonal
def plot_XYcrosscorr_diag(data):
    """
    cross correlation between X and Y
    Parameters
    ----------
    data

    Returns
    -------

    """
    for i in range(len(data[1])):
        assert np.array_equal(data[0].index, data[1][i].index)
        assert np.array_equal(data[0].columns, data[1][i].columns)

    data[0] = FG.get_ctr_norm_np(data[0].T)
    for i in range(len(data[1])):
        data[1][i] = FG.get_ctr_norm_np(data[1][i].T)
        data[1][i] = np.diag(data[1][i].T @ data[0])
    colors = data[2]
    ls = data[3]
    labels = data[4]
    title = data[5]
    name = data[6]

    pads = (0.4, 0.1, 0.2, 0.2)

    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 21, gh=SQ * 15)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    for i in range(len(data[1])):
        ax.plot(data[1][i], c=colors[i], ls=ls[i], label=labels[i])
    ax.set_xlabel('ORNs')
    ax.set_ylabel(r'corr. coef. $r$')
    ax.set_yticks([0.8, 0.9, 1])
    ax.set_xticks(np.arange(data[0].shape[1]), [])
    ax.tick_params(axis='x', which='both', bottom=False, direction='in')
    ax.xaxis.grid()

    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              handlelength=2)

    file =  f'{PP_WHITE}/ORN_act_{name}_crosscorr_diag'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


k1 = 1
k2 = 8
title = r', corr($\mathbf{X}$, $\mathbf{Y}$)'

datas = [[X, [Y], [col],
          ['-'], ['real'],
          f'LC-{k1}' + title, f'X-YLC']]

for data in datas:
    plot_XYcrosscorr_diag(data)


print('done')
# %%
# ################## activity plots for different dampening  ##################


datas = [['X', X.copy(), False, False, True,
          7, f'ORN soma activity {Xdatatex}',
          [-2, 0, 2, 4, 6]],
         ['real', Y.copy(), False, False, True, 7,
          f'ORN axon activity {Ytex}',
          [-2, 0, 2, 4, 6]]
         ]


# odor_order = par_act.odor_order_o  # if we want the original order
odor_order = par_act.odor_order
# ORN_order = par_act.ORN  # this is just the alphabetical order
ORN_order = par_act.ORN_order

# colors
vmin_col = -2
vmin_to_show = -2
vmax = 6
# act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)

pads = [0.2, 0.4, 0.15, 0.2]
for data in datas:
    name = data[0]
    act_sel = data[1]
    show_x = data[2]
    show_y = data[3]
    cb = data[4]
    v_max = data[5]
    title = data[6]
    cb_ticks = data[7]

    act_map, divnorm = FP.get_div_color_map(cb_ticks[0], cb_ticks[0], v_max)

    # preparing the data:
    act_sel = act_sel.loc[ORN_order, odor_order]
    print('min', np.min(act_sel.values))
    print('max', np.max(act_sel.values))
    # removing 'ORN' from the cell names
    # ORN_list = [name[4:] for name in act_sel.index]
    # act_sel.index = ORN_list
    # act_sel.columns.name = 'odors'
    # act_sel.index.name = 'ORNs'
    # plotting
    _pads = pads.copy()
    if not cb:
        _pads[1] = 0.01
    # if ds != 'X':
    #     _pads[0] = 0.01
    df = act_sel
    # _, fs, axs = FP.calc_fs_ax_df(df, _pads, sq=SQ*0.7)
    # splx = np.arange(2, len(df.T), 2)
    splx = np.arange(5, len(df.T), 5)
    fs, axs = FP.calc_fs_ax(_pads, SQ * len(df.T) * 0.24, SQ * len(df) * 0.5)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    # cp = FP.imshow_df2(df, ax, vlim=[vmin_to_show, v_max], show_lab_y=show_y,
    cp = FP.imshow_df2(df, ax, vlim=None, show_lab_y=show_y, aspect='auto',
                       title=title, show_lab_x=show_x, cmap=act_map,
                       splits_x=splx, splits_c='gray', lw=0.5,
                       **{'norm': divnorm},)
    ax.set_xlabel('stimuli')#, labelpad=0)
    # if ds == 'X':
    #     ax.set_ylabel('ORNs')
    ax.set_ylabel('ORNs')#, labelpad=0)
    print('figure size:', fs)

    if cb:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = add_colorbar_crt(cp, ax_cb, 'a.u.', cb_ticks,
                               extend='max')
        # clb = add_colorbar_crt(cp, ax_cb, r'$\Delta F/F$', [-2, 0, 2, 4, 6])
        # clb.ax.set_yticklabels(['0', '2', '4', '6'])

    file = f'{PP_WHITE}/ORN_act_conc-all_{name}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#%%


datas = [['realLN-L', Z['L'].copy(), False, True, True, 8,
          f'LN activity {Ztex}, left', [0, 4, 8]],
         ['realLN-R', Z['R'].copy(), False, True, True, 8,
          f'LN activity {Ztex}, right', [0, 4, 8]]
         ]

pads = [0.6, 0.4, 0.15, 0.2]
for data in datas:
    name = data[0]
    act_sel = data[1]
    show_x = data[2]
    show_y = data[3]
    cb = data[4]
    v_max = data[5]
    title = data[6]
    cb_ticks = data[7]

    divnorm = mpl.colors.Normalize(0, v_max)
    # divnorm = None
    act_map = 'Oranges'

    print('min', np.min(act_sel.values))
    print('max', np.max(act_sel.values))
    # removing 'ORN' from the cell names
    # ORN_list = [name[4:] for name in act_sel.index]
    # act_sel.index = ORN_list
    # act_sel.columns.name = 'odors'
    # act_sel.index.name = 'ORNs'
    # plotting
    _pads = pads.copy()
    if not cb:
        _pads[1] = 0.01
    # if ds != 'X':
    #     _pads[0] = 0.01
    df = act_sel
    # _, fs, axs = FP.calc_fs_ax_df(df, _pads, sq=SQ*0.7)
    # splx = np.arange(2, len(df.T), 2)
    splx = np.arange(5, len(df.T), 5)
    fs, axs = FP.calc_fs_ax(_pads, SQ * len(df.T) * 0.24, SQ * len(df) * 1)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    # cp = FP.imshow_df2(df, ax, vlim=[vmin_to_show, v_max], show_lab_y=show_y,
    cp = FP.imshow_df2(df, ax, vlim=None, show_lab_y=show_y, aspect='auto',
                       title=title, show_lab_x=show_x, cmap=act_map,
                       splits_x=splx, splits_c='gray', lw=0.5,
                       **{'norm': divnorm})
    ax.set_xlabel('stimuli')
    # if ds == 'X':
    #     ax.set_ylabel('ORNs')
    ax.set_ylabel('LNs')
    print('figure size:', fs)

    if cb:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = add_colorbar_crt(cp, ax_cb, 'a.u.', cb_ticks,
                               extend='max')
        # clb = add_colorbar_crt(cp, ax_cb, r'$\Delta F/F$', [-2, 0, 2, 4, 6])
        # clb.ax.set_yticklabels(['0', '2', '4', '6'])

    file = f'{PP_WHITE}/ORN_act_conc-all_{name}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')


# %%
# #####################  PCA VARIANCE PLOTS  ##################################
# #############################################################################


# now plotting the variances of the uncentered PCA instead of singular values
def plot_sv1(datas, order=None):
    x_i = np.arange(1, 22)
    pads = (0.4, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    axs1 = axs.copy()
    axs2 = axs.copy()
    axs1[3] -= 0.12
    dh1 = axs1[3]
    axs2[1] = axs1[1] + axs1[3] + 0.02
    axs2[3] = 0.1
    dh2 = axs2[3]
    ax1 = f.add_axes(axs1)
    ax2 = f.add_axes(axs2)
    print(axs, axs1)
    for data in datas:
        s_data = U_X.T @ data[0]
        # s_data = LA.norm(s_data, axis=1)**2/len(data[0].T)
        s_data = np.mean(s_data*s_data, axis=1)  # that corresponds to the variance
        # print(s_data)
        # if you want to have the variance, instead of the SD
        # s_data = s_data**2
        ax1.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                 c=data[4], label=data[5], markeredgewidth=data[6])
        ax2.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                 c=data[4], label=data[5], markeredgewidth=data[6])
    ax1.grid()
    ax1.set(ylabel='variance', xlabel=f'PCA direction of unctr. {Xtstex}  ',
            xticks=[1, 5, 10, 15, 21], ylim=[-0.2 , 3.5],
            yticks=[0, 3])

    ax2.grid()
    ax2.set(xticks=[1, 5, 10, 15, 21], ylim=[7-3.7/dh1*dh2 , 7],
            yticks=[7])
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(bottom=False, labelbottom=False)

    handles, labels = plt.gca().get_legend_handles_labels()
    if order is None:
        order = np.arange(len(handles))
    leg = ax2.legend([handles[idx] for idx in order],
                     [labels[idx] for idx in order],
                     bbox_to_anchor=(1.0, 1), loc='upper right', frameon=True)
    # leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
                  linestyle="none", color='k', mec='k', mew=0.75, clip_on=False)
    ax1.plot([0], [1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)

    return f


datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         [Y, 's-', 1, 2, col, y_real, 0]]

f = plot_sv1(datas)
file = f'{PP_WHITE}/ORN_act_X_real_S_orderedbyX'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #################  EIGENVALUES OF THE COVARIANCE MATRIX #####################
# ##################  DIVIDED BY MEAN  ########################################
# #################  INSTEAD OF SINGULAR VLAUES  ##############################
# #############################################################################

def plot_cov_ev2(datas, order=None):
    x_i = np.arange(1, 22)
    pads = (0.45, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    ax.grid(axis='x')
    if order is None:
        order = np.arange(len(datas))
    rect = mpl.patches.Rectangle((0.45, 1), 0.55,
                                        -(len(datas) + 1)*0.11,
                                        linewidth=0,
                                        facecolor='white', edgecolor='white',
                                        alpha=0.7,
                                        transform=ax.transAxes)

    # Add the patch to the Axes
    ax.add_patch(rect)
    labels = {}
    for i, data in enumerate(datas):
        cov = np.cov(data[0])
        # s_data = np.sqrt(LA.eigvalsh(cov)[::-1])
        s_data = LA.eigvalsh(cov)[::-1]
        s_data /= np.mean(s_data)
        CV = np.std(s_data)/np.mean(s_data)
        labels[i] = data[5] + f": {CV:0.1f}"
        ax.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                c=data[4],# label=label,
                markeredgewidth=data[6])
    for i, data in enumerate(datas):
        ax.text(1, 8/9-order[i]/9, labels[i], transform=ax.transAxes, ha='right',
                va='top', color=data[4])
    ax.text(1, 1, r'CV$_\sigma:$', transform=ax.transAxes, ha='right',
            va = 'top')
    ax.set(ylabel='scaled variance', xlabel='PCA direction',
           xticks=[1, 5, 10, 15, 21], ylim=[None, None],
           yticks=[0, 1, 2, 4, 6])
    # [line.set_zorder(10) for line in ax.lines]
    ax.set_axisbelow(True)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    return f


datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         [Y, 's-', 1, 2, col, y_real, 0]]

f = plot_cov_ev2(datas)
file = f'{PP_WHITE}/ORN_act_X_real_cov-ev_div-mean'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)

print('done')

# %%
# #############################################################################
# ############################ NORM/VARIANCE PLOTS  ###########################
# ############################    DEFINITIONS       ###########################
# #############################################################################
# #############################################################################
# #############################################################################

l_norm = 2

k1 = 1
k2 = 8
legend_opt = {'loc':'upper left', 'bbox_to_anchor': (0, 1.0)}

def scatter_norm_plot(datas, axis, xmax, ticks, xlab, ylab, do_fit=False,
                      VAR=False):
    pads = (0.5, 0.15, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 14 * SQ, 14 * SQ)
    # title = 'norm of activity patterns  '
    # title = ''
    # xlab = r'||$X_{:, i}$||'
    # ylab = r'||$Y_{:, i}$||'

    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    for data in datas:
        if not VAR:
            data[0] = LA.norm(data[0], axis=axis, ord=l_norm)
            data[1] = LA.norm(data[1], axis=axis, ord=l_norm)
        else:
            data[0] = np.sum(data[0]**2, axis=axis)/(data[0].shape[axis]-1)
            data[1] = np.sum(data[1]**2, axis=axis)/(data[1].shape[axis]-1)

        ax.scatter(data[0], data[1], label=data[2], s=data[3], marker=data[4],
                   c=data[5], lw=data[6], alpha=data[7])

    if do_fit:
        x_lin = np.linspace(0, xmax, 50)
        for data in datas:
            popt, _ = curve_fit(FOC.damp_sx, data[0], data[1])
            print(popt)
            plt.plot(x_lin, FOC.damp_sx(x_lin, *popt), c='w', ls='-', lw=1.5,
                     alpha=data[8])
            plt.plot(x_lin, FOC.damp_sx(x_lin, *popt), c=data[5], ls='--', lw=1,
                     alpha=data[9])
            # plt.plot(x_lin, FOC.damp_sx(x_lin, *popt), c=data[5], ls='--', lw=1,
            # alpha=0.7)

    ax.set(xlabel=xlab, ylabel=ylab, ylim=(None, xmax), xlim=(None, xmax),
           xticks=ticks, yticks=ticks)

    _, xmax = ax.get_xlim()
    ax.plot([0, xmax], [0, xmax], lw=0.5, color='k', ls='--')
    # plt.legend(**legend_opt)
    return f

s1 = 5
s2 = 2
lw1 = 0.5
lw2 = 0
m1 = '+'
m2 = 's'

print('done')
# %%
# #############################################################################
# ######################### CHANNEL VARIANCE PLOTS  ###########################
# #############################################################################


# xmax = 20
xmax = None
# ticks = [0, 10, 20]
ticks = [0, 1, 2]
labx = 'variance        '
laby = 'variance'
xlab = f'ORN soma {Xtstex} {labx}'
ylab = f'ORN axon {Ytex}\n{laby}'

a1 = 0.8
a2 = 0.5
a3 = 1


datas = [[X, Y, y_real, s1, m1, col, lw1, a1, a2, a3]]

f = scatter_norm_plot(datas, 1, xmax, ticks, xlab, ylab, VAR=True)
file = f'{PP_WHITE}/ORN_act-X-vs-real-Y_variance_ch'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# ##########################  BOX PLOTS  ######################################
dss = {Xtstex: X.copy(),
       y_real: Y.copy()}

axis = 1
for i, ds in dss.items():
    # dss[i] = LA.norm(dss[i], axis=1, ord=l_norm)
    dss[i] = np.sum(dss[i]**2, axis=axis)/(dss[i].shape[axis]-1)
    dss[i] /= dss[i].mean()
    # dss[i] /= ds.mean()
    print(np.std(dss[i])/np.mean(dss[i]))
dss_df = pd.DataFrame(dss)
colors = ['k', col]


pads = (0.45, 0.1, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

lineprops = {'edgecolor': 'k'}
lineprops2 = {'color': 'k'}
# sns.boxplot(data=dss_df, color=colors)
sns.swarmplot(ax=ax, data=dss_df, palette=['gray'], size=2)
bplot = sns.boxplot(ax=ax, data=dss_df, showfliers=False, linewidth=1,
                    width=0.5, boxprops=lineprops,
                    medianprops=lineprops2, whiskerprops=lineprops2,
                    capprops=lineprops2)
for i in range(0, 2):
    # mybox = bplot.artists[i]
    mybox = bplot.patches[i]
    mybox.set_facecolor('None')
    mybox.set_edgecolor(colors[i])
    for j in range(i*5, i*5+5):
        line = ax.lines[j]
        line.set_color(colors[i])
        # line.set_mfc('k')
        # line.set_mec('k')

bplot.set_xticklabels(bplot.get_xticklabels(),rotation=25, ha='right')
ax.tick_params(axis='x', which='both', pad=-1)
ax.set_ylabel('scaled\nORN variance')
# ax.set_title('scaled channel norm')
ax.set_ylim(0, 2.7)
ax.set_yticks([0, 1, 2])
ax.text(-0.3, 2.6, 'CV:', color='k', ha='left')
for i in range(2):
    ax.text(i, 2.3, f"{np.std(dss_df.iloc[:, i].values):0.2}",
            color='k', ha='center', va='bottom')

file = f'{PP_WHITE}/ORN_act-X-vs-real-Y_variance_ch_box'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#
#
#%%
# #############################################################################
# ############################ PATTERN NORM PLOTS  ############################
# #############################################################################

# STIMULUS NORM
xmax = 12.5
xmax = None
ticks = [0, 6, 12]
labx = 'pattern magnitude        '
laby = 'pattern magnitude'
xlab = f'soma {Xtstex} {labx}'
xlab =  r'a\hspace{10em}'+xlab +r'\hspace{12em}a'
ylab = f'axon {Ytex}\n{laby}'

a1 = 0.8
a2 = 0.5
a3 = 1


datas = [[X, Y, y_real, s1, m1, col, lw1, a1, a2, a3]]

f = scatter_norm_plot(datas, 0, xmax, ticks, xlab, ylab, VAR=False)
file = f'{PP_WHITE}/ORN_act-X-vs-real-Y_norm'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)

print('done')

# %%
# ##########################  BOX PLOTS  ######################################
dss = {Xtstex: X_sel.copy(),
       y_real: Y_sel.copy()}

for i, ds in dss.items():
    dss[i] = LA.norm(dss[i], axis=0, ord=l_norm)
    dss[i] /= dss[i].mean()
    print(np.std(dss[i])/np.mean(dss[i]))
    print(np.std(dss[i]), 'should be same')
dss_df = pd.DataFrame(dss)
colors = ['k', col]


pads = (0.45, 0.1, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

lineprops = {'edgecolor': 'k'}
lineprops2 = {'color': 'k'}
# sns.boxplot(data=dss_df, color=colors)
sns.swarmplot(ax=ax, data=dss_df, palette=['gray'], size=1.5)
bplot = sns.boxplot(ax=ax, data=dss_df, showfliers=False, linewidth=1,
                    width=0.5, boxprops=lineprops,
                    medianprops=lineprops2, whiskerprops=lineprops2,
                    capprops=lineprops2)

# changing the look of the box plot
for i in range(0, 2):
    # mybox = bplot.artists[i]  # old matplotlib
    mybox = bplot.patches[i]
    mybox.set_facecolor('None')
    mybox.set_edgecolor(colors[i])
    for j in range(i * 5, i * 5 + 5):
        line = ax.lines[j]
        line.set_color(colors[i])


bplot.set_xticklabels(bplot.get_xticklabels(), rotation=25, ha='right')
ax.tick_params(axis='x', which='both', pad=-1)
ax.set_ylabel('scaled\npattern magnitude')
# ax.set_title('scaled channel norm')
ax.set_ylim(0, 2.5)
ax.set_yticks([0, 1, 2])
ax.text(-0.3, 2.45, 'CV:', color='k', ha='left')
for i in range(2):
    ax.text(i, 2.2, f"{np.std(dss_df.iloc[:, i].values):0.2}",
            color='k', ha='center', va='bottom')

file = f'{PP_WHITE}/ORN_act-X-vs-real-Y_norm_box'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')

# %%
# #############################################################################
# ##################### CORRELATION MATRICES PLOTS  ###########################
# #############################################################################

# importlib.reload(FP)
# what would make sense i guess is to plot all of the
# channel and patter correlation plots. And then you can decide which ones
# you actually show. Maybe some in the main text, some in the supplement
# per channel
k1 = 1
k2 = 8
pads = (0.2, 0.4, 0.35, 0.2)

subtitles = [Xtstex, y_real]
# titles = [name + ', channel corr.' for name in subtitles]
titles = [f'{Xtstex}; {name}' for name in subtitles]
# titles[2] = titles[2] + '  '
names = ['X', 'realY']


# for i, data in enumerate([X, Y_nnc[k1], Y_nnc[k2], Y_lc[k1], Y_lc[k2],
#                           Y_nnc_noM[k2], Y_lc_noM[k2]]):
X_corr = sim_func(X)
for i, data in enumerate([X, Y]):
    cond = (i==1)
    if cond:
        pads = (0.2, 0.4, 0.35, 0.2)
    else:
        pads = (0.2, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)

    df = sim_func(data)
    df[np.tril_indices_from(df)] = X_corr[np.tril_indices_from(X_corr)]
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(df, ax, vlim=[-1, 1], show_lab_y=False,
                       title=titles[i], show_lab_x=False,
                       cmap=corr_cmap)
    ax.set(xlabel='ORNs', ylabel='ORNs')

    if cond:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])
    ax.plot([0, 1], [1, 0], transform=ax.transAxes, c='w', lw=0.5)
    file = f'{PP_WHITE}/ORN_act_{names[i]}_{sf}_channel'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# per pattern
# titles = [name + ', pattern corr.' for name in subtitles]
titles = [f'{Xtstex}; {name}' for name in subtitles]
# titles[2] = titles[2] + '  '


X_corr = sim_func(X.T)
# X_corr = sim_func(X_sel.T)  # in case one only wants to look at the higher concentration
for i, data in enumerate([X, Y]):
    # for i, data in enumerate([X_sel, Y_nnc_sel[k1], Y_nnc_sel[k2], Y_nnc_sel[k1], Y_nnc_sel[k2]]):
    cond = False
    if cond:
        pads = (0.2, 0.4, 0.35, 0.2)
    else:
        pads = (0.2, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)
    df = sim_func(data.T)
    df[np.tril_indices_from(df)] = X_corr[np.tril_indices_from(X_corr)]
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(df, ax, vlim=[-1, 1], show_lab_y=False,
                       title=titles[i], show_lab_x=False,
                       cmap=corr_cmap)
    ax.set(xlabel='stimuli', ylabel='stimuli')

    # ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
    #                     CB_W / fs[0], axs[3]])
    # clb = add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])
    ax.plot([0, 1], [1, 0], transform=ax.transAxes, c='w', lw=0.2)
    file = f'{PP_WHITE}/ORN_act_{names[i]}_{sf}_pattern'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# #############################################################################
# #########################  HISTOGRAM PLOTS  #################################
# #########################  FOR CORRELATIONS  ################################
# #############################################################################

xlabel = r'corr. coef. $r$'
ylabel = 'rel. frequency density'

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# LC
# channel correlation
def plot_dists(datas, xlabel, ylabel, title, yticks, file, bw=1,
               xlim=(-0.5, 1.01), ylim=(0, None), legend=True, fs=[21, 15],
               zoom=False, square=None):
    hist_bl = False
    pads = (0.4, 0.15, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, fs[0]*SQ, fs[1]*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    for data in datas:
        sns.kdeplot(data[0], ax=ax, color=data[1],# hist=hist_bl,
                    label=data[2])#bw_adjust=bw,)
    # ax.axhline(0, clip_on=False)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
           yticks=yticks, title=title)
    if legend:
        plt.legend(bbox_to_anchor=(1.08, 1.04),loc='lower right', ncol=3)
    else:
        ax.get_legend().remove()

    if zoom:
        axin = ax.inset_axes([0.58, 0.37, 0.5, 0.5])
        # axin = inset_axes(ax, 0.5, 0.5)
        # axin = ax.inset_axes([0.35, 0.05, 0.7, 0.1])
        for data in datas:
            sns.kdeplot(data[0], ax=axin, color=data[1], #hist=hist_bl,
                        bw_adjust=bw, label=data[2])
        axin.set(xlabel='', ylabel='', xlim=square[0], ylim=square[1],
                 xticks=[], yticks=[], facecolor='bisque')
        # axin.get_legend().remove()
        ax.indicate_inset_zoom(axin, edgecolor='grey', alpha=0.7, lw=1)
        # mark_inset(ax, axin, loc1=2, loc2=3, fc="none", ec="0.5")


    FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)


x0 = 0.25


datas = [[X_cc_c, 'k', Xtstex],
         [Y_cc_c, col, y_real]]

plot_dists(datas, xlabel, ylabel, '', [0, 3, 6],
           f'{PP_WHITE}/ORN_act_real_{sf}-chan_dist',
           ylim=(0, 6), zoom=True, square=[[x0, 1], [0., 0.8]])

datas = [[X_cc_p_sel, 'k', Xtstex],
         [Y_cc_p_sel, col, y_real]]

plot_dists(datas, xlabel, ylabel, '', [0, 1, 2, 3],
           f'{PP_WHITE}/ORN_act-conc{conc_sel}_real_{sf}'
           f'-patt_dist', ylim=(0, 3.5), zoom=True, square=[[x0, 1], [0, 0.7]])



print('Final done')
