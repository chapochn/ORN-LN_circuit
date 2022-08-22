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
# ####################  ACTIVITY PLOTS  #######################################
# #############################################################################

# #############################################################################
# #####################  ACTIVITY PLOT, different conc separately  ############
# #############################################################################

# this is not used in the publication anymore

# the first plot shows the raw actiity
# show = {8: False, 7: True, 6: False, 5: True, 4: True}
# show_x = {8: True, 7: True, 6: False, 5: False, 4: False}
#
# pads = [0.01, 0.01, 0.05, 0.2]
#
# odor_order = par_act.odor_order_o # if we want the original order
# odor_order = par_act.odor_order
# ORN_order = par_act.ORN  # this is just the alphabetical order
# ORN_order = par_act.ORN_order
#
# # colors
# vmin_col = -2
# vmin_to_show = -1
# vmax = 6
# act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)
#
#
# for conc in [8, 7, 6, 5, 4]:
#     # preparing the data:
#     act_sel = act_m.xs(conc, axis=1, level='conc').copy()
#     # putting the odors and cells in the same order as in the paper
#
#     act_sel = act_sel.loc[ORN_order, odor_order]
#     print('max', np.max(act_sel.values))
#     # removing 'ORN' from the cell names
#     ORN_list = [name[4:] for name in act_sel.index]
#     act_sel.index = ORN_list
#     act_sel.columns.name = 'odors'
#     act_sel.index.name = 'ORNs'
#     # plotting
#     _pads = pads.copy()
#     if show[conc]:
#         _pads[0] = 0.55
#     if show_x[conc]:
#         _pads[2] = 1.3
#     if conc == 4:
#         _pads[1] = 0.4
#     df = act_sel
#     _, fs, axs = FP.calc_fs_ax_df(df, _pads, sq=SQ)
#     f = plt.figure(figsize=fs)
#     ax = f.add_axes(axs)
#     cp = FP.imshow_df2(df, ax, vlim=None, show_lab_y=show[conc],
#                        title=r'dilution $10^{-%d}$' % conc, cmap=act_map,
#                        show_lab_x=show_x[conc], **{'norm': divnorm})
#     print(fs)
#
#     if conc == 4:
#         ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1],
#                             CB_W/fs[0], axs[3]])
#         clb = FP.add_colorbar(cp, ax_cb, r'$\Delta F/F$', [-1, 0, 2, 4, 6],
#                               extend='max')
#         # clb.ax.set_yticklabels(['0', '2', '4', '6'])
#
#     file = f'{PP_ACT}/ORN_act_conc-{conc}'
#     FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
#     FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# prepare colors
vmin_col = -2
vmin_to_show = -1
vmax = 6
act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)
# for logarithmic scale
# act_map = 'plasma'
# divnorm = matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.01,
#                                               vmin=-2.0, vmax=8.0, base=10)

# prepare dataframe and order of odors and ORNs
odor_order = par_act.odor_order
ORN_order = par_act.ORN_order
# not sure if there is a quicker way of doing it
idx = pd.MultiIndex.from_product([odor_order, [8, 7, 6, 5, 4]])
df = act_m.loc[ORN_order, idx]
ORN_list = [name[4:] for name in df.index]
df.index = ORN_list


title = f'ORN soma activity patterns {Xdatatex}'
cb_title = r'$\Delta F/F_0$'
# cb_title = r'$\Delta$'
cb_ticks = [-1, 0, 2, 4, 6]
# cb_ticks = [-1, -0.1, 0, 0.1, 1, 8]  # for the log scale
pads = [0.55, 0.4, 1.32, 0.2]
f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title, cb_ticks,
                              pads=pads, cb_title_font=cb_title_font)
# ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                                 cb_ticks, cb_title_font=cb_title_font)
# ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')
ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act_nolabels.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')

# %%
# same as above, but each odor normalized by the max
# prepare colors

divnorm = None
act_map = 'Oranges'

# prepare dataframe and order of odors and ORNs
odor_order = par_act.odor_order
ORN_order = par_act.ORN_order
# not sure if there is a quicker way of doing it
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
df = act_m.loc[ORN_order, idx].copy()
ORN_list = [name[4:] for name in df.index]
df.index = ORN_list
# scaling between 0 and 1
df = df - np.min(df)
df = df/np.max(df)


title = f'Scaled ORN soma activity patterns {Xdatatex}'
cb_title = ''
cb_ticks = [0, 0.5, 1]
f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title, cb_ticks,
                              extend='neither', cb_title_font=cb_title_font)
ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act_scaled_max.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# #############################################################################
# ####################  DECOMPOSION OF ACTIVITY IN PCA AND NNC  ###############
# #############################################################################

# ####################         SVD          ###################################

SVD = FG.get_svd_df(act_m)
ORN_order = par_act.ORN_order


# plotting the PC strength
pads = (0.4, 0.1, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 18*SQ, 15*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)


var = pd.Series(np.diag(SVD['s'])**2, index=SVD['s'].index)
perc_var_explained = var/np.sum(var)*100
ax.plot(perc_var_explained, '.-', lw=0.5, markersize=3, c='k')
ax.grid()
ax.set(ylabel='% variance explained', xlabel='principal component',
       xticks=[1, 5, 10, 15, 21],
       title=f'PCA of ORN activity {Xdatatex}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

file = f'{PP_ACT}/ORN_act_SVD_s'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# plotting the actual PC vectors, in imshow
pads = (0.6, 0.45, 0.35, 0.2)
SVD['U'].iloc[:, 0] *= np.sign(SVD['U'].iloc[0, 0])

print(np.max(SVD['U'].values), np.min(SVD['U'].values))

df = SVD['U'].loc[ORN_order, 1:5]
ORN_list = [name[4:] for name in df.index]
df.index = ORN_list
# act_sel.columns.name = 'odors'
df.index.name = 'ORNs'
df.columns.name = 'loading vector'
_, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

cp = FP.imshow_df2(df, ax, vlim=[-0.85, 0.85], rot=0, cmap=plt.cm.plasma,
                   title='PCA')

cb_x = axs[0] + axs[2] + CB_DX/fs[0]
ax_cb = f.add_axes([cb_x, axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, '', [-0.8, -0.4, 0, 0.4, 0.8])

file = f'{PP_ACT}/ORN_act_SVD_U'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%

print(perc_var_explained)
print(np.cumsum(perc_var_explained))
print('done')

# %%

# ####################         NNC          ##################################
strm = 0
ORN_order = par_act.ORN_order
W_data = con_strms3.loc[:, strm]
W_data = W_data.loc[ORN_order, LNs_MM]

# file = f'../results/NNC-Y_act-all.hdf'
# Ys = pd.read_hdf(file)
# file = f'../results/NNC-Z_act-all.hdf'
# odor_order = par_act.odor_order
# ORN_order = par_act.ORN  # this is just the alphabetical order
ORN_order = par_act.ORN_order
# DATASET = 3

# Zs = pd.read_hdf(file)
file = RESULTS_PATH / 'NNC-W_act-all.hdf'
Ws = pd.DataFrame(pd.read_hdf(file))
# act = FO.get_ORN_act_data(DATASET).T
# act_m = act.mean(axis=1, level=('odor', 'conc'))
# X = act_m.loc[ORN_order].T.reindex(odor_order, level='odor').T
# N = X.shape[1]

conc = 'all'
scal = 1
# scal = 0.35
pps = f'{scal}o'

# Y = {}
# Z = {}
# W2 = {}
W = {}
for k in [4, 5]:
    meth = f'NNC_{k}'
    # Y[k] = Ys[conc, pps, meth, '', '']/scal
    # Y[k] = Y[k].loc[:, ORN_order].reindex(odor_order, level='odor').T
    # # Y[k] = Y[k].values
    # Z[k] = Zs[conc, pps, meth, '', '']
    # Z[k] = Z[k].reindex(odor_order, level='odor').T
    # # Z[k] = Z[k].values
    # W2[k] = Y[k] @ Z[k].T / N
    W[k] = Ws[conc, pps, meth, '', '']/scal
    W[k] = W[k].loc[ORN_order]
print('done')
# %%
if scal == 1:
    order_LN = {4: [3, 4, 1, 2], 5: [3, 1, 4, 5, 2]}
else:
    order_LN = {4: [4, 2, 1, 3], 5: [2, 5, 3, 4, 1]}

for k in [4, 5]:
    # plotting the actual PC vectors, as imshow
    pads = (0.1, 0.4, 0.35, 0.2)
    print(np.max(W[k].values), np.min(W[k].values))

    df = W[k].loc[ORN_order, order_LN[k]]
    df.columns = np.arange(1, k+1)

    print(FG.get_ctr_norm(df).T @ FG.get_ctr_norm(W_data))

    ORN_list = [name[4:] for name in df.index]
    df.index = ORN_list
    # act_sel.columns.name = 'odors'
    df.index.name = 'ORNs'
    df.columns.name = r'$\mathbf{w}_k$'
    _, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    cp = FP.imshow_df2(df, ax, vlim=[0, None], rot=0, cmap=plt.cm.plasma,
                       show_lab_y=False, title=f'NNC-{k}')

    cb_x = axs[0] + axs[2] + CB_DX/fs[0]
    ax_cb = f.add_axes([cb_x, axs[1], CB_W/fs[0], axs[3]])
    add_colorbar_crt(cp, ax_cb, '', [0, 0.5])

    file = f'{PP_CON_PRED}/ORN_act-{pps}_NNC{k}_W'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ##############           END OF ACTIVITY PLOTS          #####################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############            BEGINNING ACT VS CON PLOTS        ##################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################

# %%
# #############################################################################
# #################  2 AXIS PLOT ORN ACT FOR ODOR VS CON  ####################
# #############################################################################

importlib.reload(FP)
ORN_order = par_act.ORN_order
# ORN_order = par_act.ORN
pads = (0.4, 0.3, 0.55, 0.15)
fs, axs = FP.calc_fs_ax(pads, 21*SQ, 11*SQ)

# side = 'L'
# LN = f'Broad T1 {side}'
LN = 'Broad T M M'
con_w = con_strms3[0][LN]
# con_w = con[side].loc[ORNs_side[side], LN].copy()
# con_w.index = [name[:-2] for name in con_w.index]
con_w = con_w.loc[ORN_order]
ORN_list = [name[4:] for name in ORN_order]

conc = 4

ylim2 = (-0.2, 5.4)
ylim = (-2, 45)
c1 = 'k'
c2 = 'g'
i = 2
odors = ['2-acetylpyridine', 'isoamyl acetate', '2-heptanone']
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
odor = odors[i]
act_vect = act_m.loc[ORN_order, (odor, conc)].copy()
ax, ax2, lns = FP.plot_line_2yax(ax, con_w.values, act_vect.values,
                                 None, None, ORN_list, 'ORNs',
                                 c1=c1, c2=c2, m1=',', m2=',')
# ax.set_xticks(np.arange(len(ORN_list)))
# ax.set_xticklabels(ORN_list, rotation=70, ha='right')
odor = odors[0]
act_vect = act_m.loc[ORN_order, (odor, conc)].copy()
ln3 = ax2.plot(act_vect.values, c=c2, label=odor, ls='dashed')
lns = lns + ln3
ax2.set_ylim(ylim2)
ax2.set_xlim((-1, 21))
ax.set_ylim(ylim)
ax.set_yticks([0, 20, 40])
ax2.set_yticks([0, 2, 4])
ax2.set_ylabel(r'ORN $\Delta F/F_0$', color=c2)
ax.set_ylabel(r'\# of syn. ORNs$\rightarrow$BT', color=c1)
labs = ['\# syn.', 'odor A', 'odor B']
leg = ax.legend(lns, labs, ncol=3, loc=10,
                bbox_to_anchor=(0.1, 1.01, .8, 0.1), frameon=False,
                handletextpad=0.5, columnspacing=1)
leg.get_frame().set_linewidth(0.0)

file = f'{PP_ODOR_CON}/{LN}_2odors-{conc}_2axplot'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# #################  SCATTER PLOT ORN ACT FOR ODOR VS CON  ####################
# #############################################################################

pads = {0: (0.15, 0.15, 0.55, 0.2), 1: (0.15, 0.15, 0.55, 0.2),
        2: (0.15, 0.15, 0.55, 0.2)}

# side = 'L'
# LN = f'Broad T1 {side}'
# con_w = con[side].loc[ORNs_side[side], LN]
LN = f'Broad T M M'
con_w = con_strms3[0][LN].loc[ORN_order]

conc = 4

scl1 = [0, 0.3, 0.3]
scl2 = [0, 0.8, 0.8]
# title = odors
title = ['odor B', 'odor C', 'odor A']
for i, odor in enumerate(odors):

    fs, axs = FP.calc_fs_ax(pads[i], 11*SQ, 11*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    act_vect = act_m.loc[ORN_order, (odor, conc)].copy()
    FP.plot_scatter(ax, con_w.values, act_vect.values, '\# of synapses',
                    '', c1, c2, xticks=[0, 20, 40],
                    yticks=[0, 2, 4],
                    pca_line_scale1=scl1[i], pca_line_scale2=scl2[i],
                    s=5, c='indigo')
    ax.set_title(title[i], color=c2)
    ax.set_xlim(ylim)
    ax.set_ylim(ylim2)
    # ax.set_ylim(-0.5, None)
    FP.set_aspect_ratio(ax, 1)
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    # FP.save_plot(f, PATH_PLOTS + 'PCA1+conBrT1L_scatter_3.pdf', SAVE_PLOTS,
    #              **pdf_opts)
    file = f'{PP_ODOR_CON}/{LN}_{odor}-{conc}_scatter'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# plotting the same as above but with using PCA

def plot_2lines(LN, LN_label, vect, vect_label, ylims,
                yticks1=[0, 20, 40], yticks2=[0, 0.5]):
    ORN_order = par_act.ORN_order
    # ORN_order = par_act.ORN
    pads = (0.4, 0.25, 0.55, 0.15)
    fs, axs = FP.calc_fs_ax(pads, 21 * SQ, 11 * SQ)

    con_w = con_strms3[0][LN]
    # con_w = con[side].loc[ORNs_side[side], LN].copy()
    # con_w.index = [name[:-2] for name in con_w.index]
    con_w = con_w.loc[ORN_order]
    ORN_list = [name[4:] for name in ORN_order]

    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    ylim2 = ylims[1]
    ylim = ylims[0]
    c1 = 'k'
    c2 = 'g'

    ax, ax2, lns = FP.plot_line_2yax(ax, con_w.values, vect.values,
                                     None, None, ORN_list, 'ORNs',
                                     c1=c1, c2=c2, m1=',', m2=',')
    ax2.set_ylim(ylim2)
    ax2.set_xlim((-1, 21))
    ax.set_ylim(ylim)
    ax.set_yticks(yticks1)
    ax2.set_yticks(yticks2)
    # ax2.set_ylabel(r'ORN $\Delta$F/F', color=c2)
    ax2.set_ylabel('', color=c2)
    ax.set_ylabel(r'\# of syn. ORNs$\rightarrow$' + LN_label, color=c1)
    labs = ['\# syn.', vect_label]
    leg = ax.legend(lns, labs, ncol=3, loc=10,
                    bbox_to_anchor=(0.1, 1.01, .8, 0.1), frameon=False,
                    handletextpad=0.5, columnspacing=1)
    leg.get_frame().set_linewidth(0.0)
    return f



LN = f'Broad T M M'
SVD = FG.get_svd_df(act_m)
act_vect = SVD['U'].loc[:, 1]
ORN_order = par_act.ORN_order
act_vect = -act_vect.loc[ORN_order]
ylims = [(-2, 45), (-0.02, 0.55)]
LN_label = 'BT'
c1 = 'k'
f = plot_2lines(LN, 'BT', act_vect, 'PCA 1', ylims)
ax = f.get_axes()[0]
# couldn't find an easier solution
ax.set_ylabel(r'a\hspace{10em}'+r'\# of syn. ORNs$\rightarrow$' + LN_label+r'\hspace{12em}a', color=c1)
# x = ax[0].yaxis.set_label_coords(-0.1, 0.5)
# print(x)
# ax.yaxis.set_label_coords()
file = f'{PP_COMP_CON}/{LN}_PCA1_2axplot'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# ########## scatter plot ##########

importlib.reload(FP)
c1 = 'k'
c2 = 'g'
pads = [0.4, 0.15, 0.55, 0.1]

LN = f'Broad T M M'
con_w = con_strms3[0][LN]
con_w = con_w.loc[ORN_order]

SVD = FG.get_svd_df(act_m)
act_vect = SVD['U'].loc[:, 1]
act_vect = -act_vect.loc[ORN_order]

scl1 = 0.5
scl2 = 0.7

fs, axs = FP.calc_fs_ax(pads, 11 * SQ, 11 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

FP.plot_scatter(ax, con_w.values, act_vect.values,
                r'\# of syn. ORNs$\rightarrow$BT',
                'ORN activity PCA 1     ', c1, c2, xticks=[0, 20, 40],
                yticks=[0, 0.5], show_cc=False,
                pca_line_scale1=scl1, pca_line_scale2=scl2,
                c='indigo', s=5)
corr_coef = np.corrcoef(con_w.values, act_vect.values)[0, 1]
ax.text(0.65, 0.05, r'$r$'+" = %0.2f" % corr_coef, transform=ax.transAxes)
# ax.set_title('PCA 1', color=c1)
ax.set_xlim(-2, 45)
ax.set_ylim(-0.02, 0.55)
# ax.set_ylim(-0.5, None)
FP.set_aspect_ratio(ax, 1)
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
# FP.save_plot(f, PATH_PLOTS + 'PCA1+conBrT1L_scatter_3.pdf', SAVE_PLOTS,
#              **pdf_opts)
file = f'{PP_COMP_CON}/{LN}_PCA1_scatter'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# #############################################################################
# ###################  CDF OF 1 LN AND OF MEAN FROM SHUFFLED  #################
# #############################################################################

# ###################         IMPORTING                      ##################

STRM = 0
CONC = 'all'

file_begin = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-')

cdfs_true = pd.DataFrame(pd.read_hdf(f'{file_begin}true.hdf'))
cdfs_shfl_m = pd.DataFrame(pd.read_hdf(f'{file_begin}shfl-m.hdf'))
cdfs_shfl_std = pd.DataFrame(pd.read_hdf(f'{file_begin}shfl-std.hdf'))

xmin = -1
xmax = 1
n_bins_cdf = 100
n_bins_cdf = 500
bins_cdf = np.linspace(xmin, xmax, n_bins_cdf + 1)


# also adding the pvalue on the graph directly

file_begin = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')

cdf_diff_min = pd.DataFrame(pd.read_hdf(f'{file_begin}.hdf'))
cdf_diff_min_pv = pd.DataFrame(pd.read_hdf(f'{file_begin}_pv.hdf'))

LN_idx = LNs_MM
pvals = cdf_diff_min_pv.loc[LN_idx].squeeze()  # converts into a series
alpha = 0.05
reject, pvals_corrected, _, _ = smsm.multipletests(pvals, method='fdr_bh',
                                                   alpha=alpha)

print('done')
# %%
# #########################  PLOTTING   #######################################
# ########################  CDFs with a cdf from shuffling  ###################
# for the paper I will use the graph that shows all lines: mean, true, fake
# also i will separate the 2 graphs, just for simplicity.

# side = 'L'
# LN = f'Broad T1 {side}'
# LN = f'Broad T M M'
LNs_m = {'BT': 'Broad T M M',
         'BD': 'Broad D M M',
         'KS': 'Keystone M M',
         'P0': 'Picky 0 [dend] M'}

# adding the gaussian fitting:
from scipy.optimize import curve_fit
from scipy.stats import norm



for LN_i, LN_m in enumerate(LNs_m):
    LN = LNs_m[LN_m]
    pval_crt = pvals_corrected[LN_i]

    cdf_mean = cdfs_shfl_m.loc[LN]
    cdf_std = cdfs_shfl_std.loc[LN]
    lw = 1

    # Plotting the 2 plots separately

    pads = (0.52, 0.05, 0.35, 0.1)
    fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    ax.plot(bins_cdf, cdf_mean, drawstyle='steps-post', label='mean', lw=lw,
            color='k')
    ax.fill_between(bins_cdf, cdf_mean - cdf_std, cdf_mean + cdf_std,
                    facecolor='grey', step='post', label='s.d.')
    # this might not be exactly correct, to verify
    dbin = bins_cdf[1] - bins_cdf[0]
    mu, sigma = curve_fit(norm.cdf, bins_cdf[:-1] + dbin/2, cdf_mean[:-1],
                          p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins_cdf, norm.cdf(bins_cdf, mu, sigma), label='gauss', lw=0.5,
            color='c')

    ax.plot(bins_cdf, cdfs_true.loc[LN], drawstyle='steps-post', c='r',
            label='true', lw=lw)



    ax.set(xlabel=r'corr. coef. $r$', ylabel='relative cumulative\nfrequency '+
                                             r'($RCF$)',
           xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    # order = [0, 3, 2, 1]
    order = [0, 1, 3, 2]
    # order = [0, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              frameon=False, bbox_to_anchor=(-0.04, 1.1), loc='upper left',
              handlelength=1, handletextpad=0.4)

    ax.text(0.4, 0.12, f'LN type: {LN_m}', transform=ax.transAxes)
    ax.text(0.4, 0.03, f"pv = {pval_crt:.1}", transform=ax.transAxes)

    file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std1'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


    # pads = (0.5, 0.05, 0.35, 0.1)
    fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    cdf_diff = cdfs_true.loc[LN] - cdf_mean
    ax.plot([-1, 1], [0, 0], label='mean', lw=lw, c='k')
    ax.fill_between(bins_cdf, - cdf_std, + cdf_std, facecolor='grey', step='post',
                    label='s.d.')
    ax.plot(bins_cdf, cdf_diff, drawstyle='steps-post', c='r', label='true',
            lw=lw)

    ax.set(xlabel=r'corr. coef. $r$', ylabel=r'$RC F - \overline{RCF}$',
           xticks=[-1, 0, 1], yticks=[-0.4, -0.2, 0, 0.2], xlim=(-1, 1))

    i_min = np.argmin(cdf_diff.values)
    col_ann = 'magenta'
    plt.annotate('', xy=(bins_cdf[i_min], 0), xycoords='data',
                 xytext=(bins_cdf[i_min], cdf_diff[i_min]), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': col_ann})
    plt.text(bins_cdf[i_min] + 0.05, cdf_diff[i_min]/2, 'max dev.',
             color=col_ann)
    ax.text(0.4, 0.12, f'LN type: {LN_m}', transform=ax.transAxes)
    ax.text(0.4, 0.03, f"pv = {pval_crt:.1}", transform=ax.transAxes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std2'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)



LN_m = 'BT'
LN = f'Broad T M M'

cdf_mean = cdfs_shfl_m.loc[LN]
cdf_std = cdfs_shfl_std.loc[LN]
lw = 1

# Plotting the 2 plots separately

pads = (0.52, 0.05, 0.35, 0.1)
fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

ax.plot(bins_cdf, cdf_mean, drawstyle='steps-post', label='mean', lw=lw,
        color='k')
ax.fill_between(bins_cdf, cdf_mean - cdf_std, cdf_mean + cdf_std,
                facecolor='grey', step='post', label='s.d.')

ax.plot(bins_cdf, cdfs_true.loc[LN], drawstyle='steps-post', c='r',
        label='true', lw=lw)



ax.set(xlabel=r'corr. coef. $r$', ylabel='relative cumulative\nfrequency '+
                                         r'($RCF$)',
       xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# legend
handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 2]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          frameon=False, bbox_to_anchor=(-0.04, 1.1), loc='upper left',
          handlelength=1, handletextpad=0.4)

ax.text(0.4, 0.025, f'LN type: {LN_m}', transform=ax.transAxes)

file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std1_nofit'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# ##################  CDF OF 1 NNC-4 W AND OF MEAN FROM SHUFFLED  #############
# #############################################################################
# basically same as above, but for the W of NNC-4
# ###################         IMPORTING                      ##################

K = 4
CONC = 'all'

file_begin = (f'{RESULTS_PATH}/NNC-{K}_con-W_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-')

cdfs_true = pd.DataFrame(pd.read_hdf(f'{file_begin}true.hdf'))
cdfs_shfl_m = pd.DataFrame(pd.read_hdf(f'{file_begin}shfl-m.hdf'))
cdfs_shfl_std = pd.DataFrame(pd.read_hdf(f'{file_begin}shfl-std.hdf'))

xmin = -1
xmax = 1
n_bins_cdf = 100
n_bins_cdf = 500
bins_cdf = np.linspace(xmin, xmax, n_bins_cdf + 1)


# also importing p-values to put them directly in the graph:
CONC = 'all'

file_begin = (f'{RESULTS_PATH}/NNC-{K}_con-W_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')

cdf_diff_min = pd.DataFrame(pd.read_hdf(f'{file_begin}.hdf'))
cdf_diff_min_pv = pd.DataFrame(pd.read_hdf(f'{file_begin}_pv.hdf'))

pvals = cdf_diff_min_pv.squeeze()  # converts into a series
alpha = 0.05
reject, pvals_corrected, _, _ = smsm.multipletests(pvals, method='fdr_bh',
                                                   alpha=alpha)
pvals_fdr = pvals.copy()
pvals_fdr[:] = pvals_corrected

print('done')
# %%
# #########################  PLOTTING   #######################################
# ########################  CDFs with a cdf from shuffling  ###################
# for the paper I will use the graph that shows all lines: mean, true, fake
# also i will separate the 2 graphs, just for simplicity.

# adding the gaussian fitting:
from scipy.optimize import curve_fit
from scipy.stats import norm

LN_order = [3, 4, 1, 2]
LN_text = {1:r'$\mathbf{w}_1$', 2: r'$\mathbf{w}_2$',
           3:r'$\mathbf{w}_3$', 4: r'$\mathbf{w}_4$'}

for LN_i in range(1, K+1):
    LN_i_new = LN_order[LN_i-1]
    LN_text_crt = f'NNC-{K}, ' + LN_text[LN_i_new]
    pval_crt = pvals_fdr[LN_i]
    cdf_mean = cdfs_shfl_m.loc[LN_i]
    cdf_std = cdfs_shfl_std.loc[LN_i]
    lw = 1

    # Plotting the 2 plots separately

    pads = (0.52, 0.05, 0.35, 0.1)
    pads = (0.52, 0.15, 0.35, 0.1)
    fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    ax.plot(bins_cdf, cdf_mean, drawstyle='steps-post', label='mean', lw=lw,
            color='k')
    ax.fill_between(bins_cdf, cdf_mean - cdf_std, cdf_mean + cdf_std,
                    facecolor='grey', step='post', label='s.d.')
    # this might not be exactly correct, to verify
    dbin = bins_cdf[1] - bins_cdf[0]
    mu, sigma = curve_fit(norm.cdf, bins_cdf[:-1] + dbin/2, cdf_mean[:-1],
                          p0=[0, 0.3])[0]
    print(mu, sigma)
    ax.plot(bins_cdf, norm.cdf(bins_cdf, mu, sigma), label='gauss', lw=0.5,
            color='c')

    ax.plot(bins_cdf, cdfs_true.loc[LN_i], drawstyle='steps-post', c='r',
            label='true', lw=lw)



    ax.set(xlabel=r'corr. coef. $r$', ylabel='relative cumulative\nfrequency '+
                                             r'($RCF$)',
           xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 3, 2]
    # order = [0, 3, 2, 1]
    # order = [0, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              frameon=False, bbox_to_anchor=(-0.04, 1.1), loc='upper left',
              handlelength=1, handletextpad=0.4)

    ax.text(0.5, 0.12, LN_text_crt, transform=ax.transAxes)
    ax.text(0.5, 0.03, f"pv = {pval_crt:.1}", transform=ax.transAxes)

    file = f'{PP_ODOR_CON}/cors_W{LN_i_new}_rcf-m-std1'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


    # pads = (0.5, 0.05, 0.35, 0.1)
    fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    cdf_diff = cdfs_true.loc[LN_i] - cdf_mean
    ax.plot([-1, 1], [0, 0], label='mean', lw=lw, c='k')
    ax.fill_between(bins_cdf, - cdf_std, + cdf_std, facecolor='grey', step='post',
                    label='s.d.')
    ax.plot(bins_cdf, cdf_diff, drawstyle='steps-post', c='r', label='true',
            lw=lw)

    ax.set(xlabel=r'corr. coef. $r$', ylabel=r'$RC F - \overline{RCF}$',
           xticks=[-1, 0, 1], yticks=[-0.4, -0.2, 0, 0.2], xlim=(-1, 1))

    i_min = np.argmin(cdf_diff.values)
    col_ann = 'magenta'
    plt.annotate('', xy=(bins_cdf[i_min], 0), xycoords='data',
                 xytext=(bins_cdf[i_min], cdf_diff[i_min]), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': col_ann})
    plt.text(bins_cdf[i_min] + 0.05, cdf_diff[i_min]/2, 'max dev.',
             color=col_ann)
    ax.text(0.5, 0.12, LN_text_crt, transform=ax.transAxes)
    ax.text(0.5, 0.03, f"pv = {pval_crt:.1}", transform=ax.transAxes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    file = f'{PP_ODOR_CON}/cors_W{LN_i_new}_rcf-m-std2'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ###########  CORRELATION COEFFICIENTS FOR ALL ODORS AND ALL LNs  ############
# #############################################################################
wLNtex = r'$\mathbf{w}_{\mathrm{LN}}$'

# calculation of the correlation coefficients
# con_W = con_strms3.loc[:, 0].loc[:, LNs_sel_LR_d]
con_W = con_strms3.loc[:, 0].loc[:, LNs_sel_LRM_d]
con_W = con_W.rename(columns={'Broad T M M': 'Broad T',
                              'Broad D M M': 'Broad D',
                              'Keystone M M': 'Keystone',
                              'Picky 0 [dend] M': 'Picky 0 [dend]'})
con_W_cn = FG.get_ctr_norm(con_W)
act_cn = FG.get_ctr_norm(act_m)

# the alignment (which changes the names of the ORNs so that both
# datasets have the same names) is necessary so that the
# correlation coefficient calculation is done correctly
act_cn2, con_W_cn2 = FG.align_indices(act_cn, con_W_cn)
cors = con_W_cn2.T @ act_cn2

# ordering of the concentration so that it is as before
odor_order = par_act.odor_order
# not sure if there is a quicker way of doing it
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
df = cors.loc[:, idx]

# plotting
splx = np.arange(5, len(idx), 5)
# sply = [6, 6+4, 6+4+4]
sply = [7, 7+5, 7+5+5]
pads = (0.95, 0.41, 1.32, 0.2)
fs, axs = FP.calc_fs_ax(pads, SQ*len(df.T)*0.4, SQ*len(df))  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
cp = FP.imshow_df2(df, ax, vlim=[-1, 1], cmap=corr_cmap, splits_x=splx,
                   splits_y=sply, aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       xlabel=f'ORN activation patterns {Xttex} to odors at different dilutions',
       ylabel=wLNtex)
       # ylabel = r'data, ' + wLNtex)

plt.title(f'Correlation between ORN activity patterns {Xttex} '
          r'and ORNs$\rightarrow$LN connection weight vectors '
          r'$\mathbf{w}_{\mathrm{LN}}$')

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/ORN_con0_vs_act3_raw_mean_all-odors_corr'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# just for LN M
# calculation of the correlation coefficients
wLNtex = r'$\mathbf{w}_\mathrm{LNtype}$'

# con_W = con_strms3.loc[:, 0].loc[:, LNs_sel_LR_d]
con_W = con_strms3.loc[:, 0].loc[:, LNs_MM]
con_W = con_W.rename(columns={'Broad T M M': 'BT',
                              'Broad D M M': 'BD',
                              'Keystone M M': 'KS',
                              'Picky 0 [dend] M': 'P0'})
con_W_cn = FG.get_ctr_norm(con_W)
act_cn = FG.get_ctr_norm(act_m)

# the alignment (which changes the names of the ORNs so that both
# datasets have the same names) is necessary so that the
# correlation coefficient calculation is done correctly
act_cn2, con_W_cn2 = FG.align_indices(act_cn, con_W_cn)
cors = con_W_cn2.T @ act_cn2

# ordering of the concentration so that it is as before
odor_order = par_act.odor_order
# not sure if there is a quicker way of doing it
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
df = cors.loc[:, idx]

# plotting
splx = np.arange(5, len(idx), 5)
pads = (0.55, 0.4, 0.2, 0.2)
fs, axs = FP.calc_fs_ax(pads, SQ*len(df.T)*0.4, SQ*len(df))  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
cp = FP.imshow_df2(df, ax, vlim=[-1, 1], cmap=corr_cmap, splits_x=splx,
                   aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=[])
ax.set_xlabel(f'ORN activation patterns {Xttex} to odors at different dilutions')
ax.set_ylabel(wLNtex)
# ax.set_ylabel('from\nORNs\nto', rotation=0, fontsize=ft_s_tk, labelpad=3,
#               va='center', ha='right')
# f.text(0.005, 0.5, wLNtex + ':', rotation=90, va='center', ha='left')
# ax.annotate('', xy=(-6.5, -1), xytext=(-6.5, 4), xycoords='data',
#             arrowprops={'arrowstyle': '-', 'lw': 0.5},
#             annotation_clip=False)

plt.title(f'Correlation between ORN activity patterns {Xttex}'
          r' and ORNs$\rightarrow$LN conn. weight vectors '+ f'{wLNtex}')
ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/ORN_con0-LN-M_vs_act3_raw_mean_all-odors_corr'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# Histogram for each LN
con_W = con_strms3.loc[:, 0].loc[:, LNs_MM]
con_W = con_W.rename(columns={'Broad T M M': 'BT',
                              'Broad D M M': 'BD',
                              'Keystone M M': 'KS',
                              'Picky 0 [dend] M': 'P0'})
con_W_cn = FG.get_ctr_norm(con_W)
act_cn = FG.get_ctr_norm(act_m)

# the alignment (which changes the names of the ORNs so that both
# datasets have the same names) is necessary so that the
# correlation coefficient calculation is done correctly
act_cn2, con_W_cn2 = FG.align_indices(act_cn, con_W_cn)
cors = con_W_cn2.T @ act_cn2
print('done')
#
#
#
# %%
pads = (0.4, 0.1, 0.35, 0.15)
fs, axs = FP.calc_fs_ax(pads, SQ*11, SQ*14)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.plot([0, 170 * 2 - 1], [0, 0], c='gray', lw=0.5)
for i in range(4):
    # plt.figure()
    my_sorted = np.sort(cors.iloc[i].values)[::-1]
    # sorted2 = np.concatenate([sorted, sorted[::-1]])
    ax.plot(my_sorted, label=cors.index[i], lw=1)
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#           frameon=False, bbox_to_anchor=(0.5, 1.1), loc='upper left',
#           handlelength = 1, handletextpad=0.4)
plt.legend(frameon=False, loc='upper left', handlelength=1,
           handletextpad=0.4, bbox_to_anchor=(0.5, 1.1))
ax.set_ylim(-0.5, 0.9)
ax.set_xlim(-5, 170)
ax.set_yticks([-0.5, 0, 0.5, 1])
ax.set_xticks([0, 50, 100, 150])
ax.set_ylabel(r'corr. coef. $r$', labelpad=-2)
ax.set_xlabel('ordered stimuli')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

file = f'{PP_ODOR_CON}/LN_tuning_curves.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#
#
#
# %%
# #############################################################################
# #########################  CDF DIFF MIN AND PVAL  ###########################
# #############################################################################

# ###################         IMPORTING             ###########################

CONC = 'all'

file_begin = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')

cdf_diff_min = pd.DataFrame(pd.read_hdf(f'{file_begin}.hdf'))
cdf_diff_min_pv = pd.DataFrame(pd.read_hdf(f'{file_begin}_pv.hdf'))
print('done')
# %%
# ##################      PLOTTING FOR INDIVIDUAL CELLS      ##################
LN_idx = LNs_sel_LR_d
pvals = cdf_diff_min_pv.loc[LN_idx].squeeze()  # converts into a series
alpha = 0.05
reject, pvals_corrected, _, _ = smsm.multipletests(pvals, method='fdr_bh',
                                                   alpha=alpha)
pvals_fdr = pvals.copy()
pvals_fdr[:] = pvals_corrected
x = np.arange(len(LN_idx), dtype=float)
x[6:] += 0.5
x[10:] += 0.5
x[14:] += 0.5

pads = (0.4, 0.35, 0.9, 0.15)
fs, axs = FP.calc_fs_ax(pads, SQ*(len(x)+1.5), SQ*15)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

axx = FP.plot_double_series_unevenX(ax, x, cdf_diff_min.loc[LN_idx],
                                    -np.log10(pvals_fdr),
                                    'magenta', 'b', 'RCF max deviation (\enspace)',
                                    r'$-\log_{10}$(p-value) (\enspace)',
                                    ylim1=(0, 0.37), ylim2=(0, 2.1))
# axx[1].annotate('', xy=(-2, -1.5), xytext=(18.5, -1.5), xycoords='data',
#                 arrowprops={'arrowstyle': '-', 'lw': 0.5},
#                 annotation_clip=False)
axx[1].plot([20.85], [1.76], ls='None', marker='+', color='b', markersize=5,
            clip_on=False)
axx[0].plot([-5.3], [0.335], ls='None', marker='.', color='magenta',
            markersize=5, clip_on=False)
axx[0].set_yticks([0, 0.1, 0.2, 0.3])
axx[1].set_yticks([0, 1, 2])
axx[1].set_xlabel(r'$\mathbf{w}_\mathrm{LN}$')
# f.text(0.5, 0., 'conn. weight vectors from ORNs to',
#    fontsize=matplotlib.rcParams['axes.labelsize'], va='bottom', ha='center')
# axx[1].set_xlabel('ORNs -> LN weight vectors')

# adding the info about the significance
# pvals = cdf_diff_min_pv.loc[LN_idx].values
for alpha, y, sign in [[0.05, 2.1, '*']]:
    FP.add_sign_stars(axx[1], pvals, alpha, x, y, sign, fontdict={'size': 14})

file = (f'{PP_ODOR_CON}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_corr_rcf_min_diff')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
 # %%
# #############      PLOTTING FOR INDIVIDUAL SUMMARY CELLS      ###############

# strm = 0
# con_ff_sel = con_strms3.loc[:, strm]
# con_W = con_ff_sel.loc[:, LNs_MM].copy()
# con_W_cn = FG.get_ctr_norm(con_W)
# act_cn = FG.get_ctr_norm(act_m)
# res = FG.characterize_raw_responses(act_cn, con_W_cn, N1=10, N2=50000)
# corr, corr_pv, corr_m_per_cell, corr_m_per_cell_pv = res[0:4]
# cdf_diff_max, cdf_diff_max_pv, cdfs_true, cdfs = res[4:]
# pvals = cdf_diff_max_pv

LN_idx = LNs_MM
pvals = cdf_diff_min_pv.loc[LNs_MM].squeeze()

alpha = 0.05
_, pvals_corrected, _, _ = smsm.multipletests(pvals, method='fdr_bh',
                                              alpha=alpha)
pvals_fdr = pvals.copy()
pvals_fdr[:] = pvals_corrected
# df1 = cdf_diff_max.copy()
df1 = cdf_diff_min.loc[LN_idx].copy()
df2 = -np.log10(pvals_fdr)
print('done')
# %%

LN_name_change_dict = {'Broad T M M': 'BT',
                       'Broad D M M': 'BD',
                       'Picky 0 [dend] M': 'P0',
                       'Keystone M M': 'KS'}
df1 = df1.rename(index=LN_name_change_dict)
df2 = df2.rename(index=LN_name_change_dict)

x = np.arange(len(LNs_MM), dtype=float)
x[1:] += 0.5
x[2:] += 0.5
x[3:] += 0.5

pads = (0.4, 0.35, 0.35, 0.15)
fs, axs = FP.calc_fs_ax(pads, SQ*(len(x)+2), SQ*14)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

axx = FP.plot_double_series_unevenX(ax, x, df1, df2, 'magenta', 'b',
                  r'RCF max deviation (\enspace)', r'$-\log_{10}$(p-value) (\enspace)',
                   ylim1=(0, 0.37), ylim2=(0, 2.5))
axx[1].plot([8.75], [2.155], ls='None', marker='+', color='b', markersize=5,
            clip_on=False)
axx[0].plot([-5.2], [0.345], ls='None', marker='.', color='magenta',
            markersize=5, clip_on=False)
axx[0].set_yticks([0, 0.1, 0.2, 0.3])
axx[1].set_yticks([0, 1, 2])
axx[1].set_xlabel('from ORNs to', fontsize=ft_s_tk)
axx[1].set_xlabel(r'$\mathbf{w}_\mathrm{LNtype}$')
axx[1].set_xticklabels(df1.index, rotation=45, ha='center')
# axx[1].annotate('', xy=(-1.5, -0.5), xytext=(6, -0.5), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# f.text(0.5, 0., 'ORNs -> LN vect.',
#    fontsize=matplotlib.rcParams['axes.labelsize'], va='bottom', ha='center')


for alpha, y, sign in [[0.05, 2.4, '*']]:
    FP.add_sign_stars(axx[1], pvals, alpha, x, y, sign, fontdict={'size': 14})

file = (f'{PP_ODOR_CON}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_corr_rcf_min_diff_sum2')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#
# =============================================================================
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ######################  CORR OF CON VS COMP OF ACTIVITY  ####################
# #############################################################################

# ######################  IMPORTING  ##########################################

# NMF_n = 4
# NMF_name = f'NMF_{NMF_n}'
# SVD_n = 6
RESULTS_PATH = OLF_PATH / 'results'
# file = (f'{RESULTS_PATH}/corr_sign/act{DATASET}-{act_pps1}-{act_pps2}'
#        f'_conc-all_NMF{NMF_n}-SVD{SVD_n}_vs_con-ORN-all_')
file = (f'{RESULTS_PATH}/corr_sign/act{DATASET}-{act_pps1}-{act_pps2}'
        f'_conc-all_SVD-NNC_vs_con-ORN-all_')

file_cc = f'{file}cc.hdf'
file_cc_pv = f'{file}cc_pv.hdf'
# =============================================================================
# file_CS = f'{file}CS.hdf'
# file_CS_pv = f'{file}CS_pv.hdf'
# =============================================================================

data_cc = pd.DataFrame(pd.read_hdf(file_cc))[STRM]
data_cc = data_cc.reset_index(level=['par1', 'par2'], drop=True)

data_cc_pv = pd.DataFrame(pd.read_hdf(file_cc_pv))[STRM]
data_cc_pv = data_cc_pv.reset_index(level=['par1', 'par2'], drop=True)

# =============================================================================
# data_CS = pd.read_hdf(file_CS)[STRM]
# data_CS = data_CS.reset_index(level=['par1', 'par2'], drop=True)
#
# data_CS_pv = pd.read_hdf(file_CS_pv)[STRM]
# data_CS_pv = data_CS_pv.reset_index(level=['par1', 'par2'], drop=True)
# =============================================================================

# =============================================================================
# corr_data = corr_data.rename(columns=LN_name_change_dict)
# sign_data = sign_data.rename(columns=LN_name_change_dict)
# =============================================================================

# corr_data = corr_data.loc[:, (slice(None), LNs1)]
# sign_data = sign_data.loc[:, (slice(None), LNs1)]
print('done')
# %%


scal = 1
pps = f'{scal}o'
SVD_n = 5
NNC_n = 4

idy = {}
ylabels_new = {}

if scal == 1:
    order_LN = {4: [3, 4, 1, 2], 5: [3, 1, 4, 5, 2]}
else:
    order_LN = {4: [4, 2, 1, 3], 5: [2, 5, 3, 4, 1]}

idy[0] = [('o', 'SVD', i) for i in range(1, SVD_n + 1)]
idy[1] = [(pps, f'NNC_{NNC_n}', i) for i in order_LN[4]]


ylabels_new[0] = [f'{i}' for i in range(1, SVD_n + 1)]
ylabels_new[1] = [f'{i}' for i in range(1, NNC_n + 1)]

NNC_n = 5
idy[2] = [(pps, f'NNC_{NNC_n}', i) for i in order_LN[5]]
ylabels_new[2] = [f'{i}' for i in range(1, NNC_n + 1)]

for k in [2, 3]:
    idy[k + 9] = [(pps, f'NNC_{k}', i) for i in range(1, k + 1)]
    ylabels_new[k + 9] = [f'{i}' for i in range(1, k + 1)]

ylabel = [f'PCA directions\nof {Xdatatex}',
          r'NNC-4, $\mathbf{w}_k$',
          r'NNC-5, $\mathbf{w}_k$']

xlabel = {0: r'$\mathbf{w}_\mathrm{LNtype}$',
          1: 'connection on ORNs from',
          2: 'connection on ORNs from',
          3: 'connection of ORNs with'}

xlabel_short = {0: 'from ORNs to',
                1: 'conn. on ORNs from',
                2: 'conn. on ORNs from',
                3: 'conn. of ORNs with'}
print('done')
# %%
splx = {'raw': [6, 6+4, 10+4], 'M': [1, 2, 3]}
spl = SVD_n
LNs_k = {'raw': LNs_sel_LR_d, 'M': LNs_MM}

pads = [0.4, 0.45, 0.35, 0.2] # for the correlation plot
pads1 = [0.1, 0.47, 0.35, 0.2] # for the pv plot
d_h = 0.05

show_lab_x = {0: False, 1: False, 2: True, 3: True, 4: True, 5: True, 6: True,
              7: True, 8: True, 9: True, 10: True, 11: True, 12: True}
# show_lab_x = {0: False, 1: False, 2: True}

def plot_pv_measure(df1, df2, splx, cmap, vlim, i=None):
    """
    measure stands for either corr coef or CS, yeah it's a weird name
    """
    _pads = pads.copy()
    _pads1 = pads1.copy()
    sq = 2.2 * SQ
    showlabx = True
    if len(df1.columns) > 5:
        _pads = [0.4, 0.45, 0.9, 0.2]  # for the correlation plot
        _pads1 = [0.04, 0.47, 0.9, 0.2]  # for the pv plot
        sq = 1.1 * SQ
        showlabx = show_lab_x[i]
        if i != 2:
            _pads = [0.4, 0.45, 0.2, 0.2]  # for the correlation plot
            _pads1 = [0.04, 0.47, 0.2, 0.2]  # for the pv plot
    if len(df1.columns) < 5 and i == 0:
        _pads[2] = 0.55
        _pads1[2] = 0.55

    _, fs, axs = FP.calc_fs_ax_df(df1, _pads, sq=sq)
    f = plt.figure(figsize=fs)
    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0],
                        axs[1], CB_W / fs[0], axs[3]])
    ax1 = f.add_axes(axs)  # left, bottom, width, height

    cp = FP.imshow_df2(df1, ax1, vlim=vlim, cmap=cmap, splits_x=splx,
                       lw=0.5, show_lab_x=False)
    if show_lab_x and len(df1.columns) < 5:
        ax1.set_xticks(np.arange(len(df1.T)) + 0.5)
        ax1.set_xticklabels(list(df1.columns), rotation=45, ha='right')
        ax1.tick_params('x', bottom=False, pad=-1)
        ax1.set_xlabel(df1.columns.name)
        # ax1.set_xlabel(df1.columns.name, fontsize=ft_s_tk,
        #                labelpad=2, rotation_mode='default', ha='center',
        #                va='top')
        # ax1.annotate('', xy=(-1, len(df1.index) + 1.3),
        #              xytext=(len(df1.columns), len(df1.index) + 1.3),
        #              arrowprops={'arrowstyle': '-', 'lw': 0.5},
        #              xycoords='data', annotation_clip=False)

    if show_lab_x and len(df1.columns) > 5 and i == 2:
        ax1.set_xticks(np.arange(len(df1.T)))
        ax1.set_xticklabels(list(df1.columns), rotation=90, ha='center')
        ax1.tick_params('x', bottom=False, pad=-1)
        ax1.set_xlabel(r'$\mathbf{w}_\mathrm{LN}$')
        # ax1.set_xlabel(df1.columns.name, fontsize=ft_s_tk,
        #                labelpad=2, rotation_mode='default', ha='center',
        #                va='top')
        # ax1.annotate('', xy=(-1, len(df1.index) + 9),
        #              xytext=(len(df1.columns), len(df1.index) + 9),
        #              arrowprops={'arrowstyle': '-', 'lw': 0.5},
        #              xycoords='data', annotation_clip=False)

    add_colorbar_crt(cp, ax_cb, r'$r$', [-1 , 0, 1])
    f1 = (f, ax1, ax_cb, cp)

    # p value plot
    _, fs, axs = FP.calc_fs_ax_df(df2, _pads1, sq=sq)
    f = plt.figure(figsize=fs)
    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0],
                        axs[1], CB_W/fs[0], axs[3]])
    ax1 = f.add_axes(axs)  # left, bottom, widgth, height
    cp = FP.imshow_df2(df2, ax1, vlim=[-3, -1], lw=0.5,
                       cmap=plt.cm.viridis_r, splits_x=splx, show_lab_y=True,
                       show_lab_x=False)
    if show_lab_x and len(df1.columns) < 5:
        ax1.set_xticks(np.arange(len(df1.T)) + 0.5)
        ax1.set_xticklabels(list(df1.columns), rotation=45, ha='right')
        ax1.tick_params('x', bottom=False, pad=-1)
        ax1.set_xlabel(df1.columns.name)
        # ax1.set_xlabel(df1.columns.name, rotation=0, fontsize=ft_s_tk,
        #                labelpad=2, rotation_mode='default', ha='center',
        #                va='top')
        # ax1.annotate('', xy=(-1, len(df1.index) + 1.3),
        #              xytext=(len(df1.columns), len(df1.index) + 1.3),
        #              arrowprops={'arrowstyle': '-', 'lw': 0.5}, xycoords='data',
        #              annotation_clip=False)
    if show_lab_x and len(df1.columns) > 5 and i == 2:
        ax1.set_xticks(np.arange(len(df1.T)))
        ax1.set_xticklabels(list(df1.columns), rotation=90, ha='center')
        ax1.tick_params('x', bottom=False, pad=-1)
        ax1.set_xlabel(r'$\mathbf{w}_\mathrm{LN}$')
        # ax1.set_xlabel(df1.columns.name)
        # ax1.set_xlabel(df1.columns.name, fontsize=ft_s_tk,
        #                labelpad=2, rotation_mode='default', ha='center',
        #                va='top')
        # ax1.annotate('', xy=(-1, len(df1.index) + 9),
        #              xytext=(len(df1.columns), len(df1.index) + 9),
        #              arrowprops={'arrowstyle': '-', 'lw': 0.5}, xycoords='data',
        #              annotation_clip=False)

    clb = add_colorbar_crt(cp, ax_cb, 'pv', [-3, -2, -1], extend='both')
    cp.cmap.set_over('k')
    clb.set_ticklabels([0.001, 0.01, 0.1])
    f2 = (f, ax1, ax_cb, cp)
    return f1, f2


# #############################  PLOTTING  ####################################

dict_M = {'Broad T M M': 'BT',
          'Broad D M M': 'BD',
          'Keystone M M': 'KS',
          'Picky 0 [dend] M': 'P0'}


for i, k in itertools.product(range(3), ['raw', 'M']):  # k as key
    print(i, k)
    df1 = data_cc.loc[idy[i], LNs_k[k]].copy()
    if i == 0:
        df1.iloc[1] = -df1.iloc[1]
    df1.index = ylabels_new[i]
    df1.index.name = ylabel[i]
    df1.columns.name = xlabel[0]

    pvals = data_cc_pv.loc[idy[i], LNs_k[k]]
    alpha = 0.05
    _, pvals2, _, _ = smsm.multipletests(pvals.values.flatten(),
                                         method='fdr_bh', alpha=alpha)
    pvals_fdr = pvals.copy()
    pvals_fdr[:] = pvals2.reshape(pvals_fdr.shape)

    stars = pd.DataFrame(pvals_fdr < alpha, dtype=str)
    stars = stars.replace({'True': '*', 'False': ''})

    # df2 = np.log10(pvals_fdr)
    df2 = np.log10(pvals)
    df2.index = ylabels_new[i]
    df2.columns.name = xlabel[0]
    file = f'{PP_COMP_CON}/act-{pps}-comps{i}_con{STRM}-{k}_cc'
    if i >= 1:
        file = f'{PP_CON_PRED}/act-{pps}-comps{i}_con{STRM}-{k}_cc'

    df1 = df1.rename(dict_M, axis='columns')
    df2 = df2.rename(dict_M, axis='columns')

    f1, f2 = plot_pv_measure(df1, df2, splx[k], corr_cmap,
                             [-1, 1], i)

    for (m, n), label in np.ndenumerate(stars):
        f1[1].text(n, m+0.1, label, ha='center', va='center',
                   size=matplotlib.rcParams['font.size']*0.8**0)

    FP.save_plot(f1[0], f'{file}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f1[0], f'{file}.pdf', SAVE_PLOTS, **pdf_opts)
    # FP.save_plot(f2[0], f'{file}_pv_adjusted.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f2[0], f'{file}_pv.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f2[0], f'{file}_pv.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# significance testing with the p-values with the mean connections.
k = 'M'
i = 0
pvals = data_cc_pv.loc[idy[i], LNs_k[k]].values.flatten()
alpha = 0.05
reject, pvals_fdr, _, _ = smsm.multipletests(pvals, method='fdr_bh',
                                             alpha=alpha)
print(reject)
print(pvals_fdr)
print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #########  comparing activity with the whole set of Ws from NNC  ############
# #############################################################################
# #############################################################################


# Ws = pd.read_hdf('../results/W_NNC-4.hdf')
Ws = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-4.hdf'))
print(Ws.shape)
Ws_cn = FG.get_ctr_norm(Ws).loc[par_act.ORN_order]
# Ws = pd.read_hdf('../results/W_NNC-5_short.hdf')
LNs_M = ['Broad T M M', 'Broad D M M', 'Keystone M M', 'Picky 0 [dend] M']
LNs_short = ['BT', 'BD', 'KS', 'P0']
con_sel = con_strms3_cn[0][LNs_M].loc[par_act.ORN_order]
corr = (con_sel.T @ Ws_cn).T
corr_grp = corr.groupby(['rho', 'rep']).max()
y_mean = corr_grp.groupby('rho').mean()
y_std = corr_grp.groupby('rho').std()
print('done')
# %%
pads = (0.4, 0.1, 0.35, 0.04)
fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*10)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
for i in range(4):
    # x = (corr.index.unique('rho')+(i-1.5)/10)/10
    x = (corr.index.unique('rho'))/10
    # ax.errorbar(x, y_mean[LNs_M[i]], yerr=y_std[LNs_M[i]],
    # label=LNs_short[i],
    #             lw=1)
    y = y_mean[LNs_M[i]]
    e = y_std[LNs_M[i]]
    ax.plot(x, y, lw=1, label=LNs_short[i])
    ax.fill_between(x, y-e, y+e, alpha=0.5)
ax.set_yticks([0, 0.4, 0.8])
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([0.1, 1, 10])
ax.set_ylim(0, 0.8)
ax.set_ylabel(r'corr. coef. $r$')
ax.set_xlabel(r'model inhibition strength $\rho$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(ncol=4, frameon=False, columnspacing=1, handlelength=1,
           handletextpad=0.4)

file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-4_rho-range')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%

# Now i want to calculate the p values for each individual case
cor, pv_o, pv_l, pv_r = FG.get_signif_corr_v2(Ws_cn, con_sel, N=20000)
alpha = 0.05
pv_fdr = pv_o.copy()
for rho in pv_fdr.index.unique('rho'):
    for rep in pv_fdr.index.unique('rep'):
        pvs = pv_fdr.loc[(rho, rep)]
        _, pvs2, _, _ = smsm.multipletests(pvs.values.flatten(),
                                           method='fdr_bh', alpha=alpha)
        pv_fdr.loc[(rho, rep)][:] = pvs2.reshape(pvs.shape)
# Here is the structure of pv_fdr:
# it has 4 columns, each column is one of the mean LNtype
# the rows are ordered by a multiindex
# 1: rho
# 2: repetition
# 3: one of the 4 w_k
# pvs_grp = pv_fdr.groupby(['rho', 'rep']).min(axis=1)
pvs_grp = pv_fdr.groupby(['rho', 'rep']).min()
n_cells_signif = (pvs_grp < 0.05).sum(axis=1)
print('done')
# %%

x = pv_fdr.index.unique('rho')
y = n_cells_signif.groupby('rho').mean()
e = n_cells_signif.groupby('rho').std()
# plt.figure()
# plt.errorbar(x, y_mean, yerr=y_std, label=LNs_M[i])
# plt.show()

pads = (0.4, 0.1, 0.085, 0.1)
fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*3.5)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
x = corr.index.unique('rho')/10
ax.plot(x, y, lw=1, c='k')
ax.fill_between(x, y-e, y+e, alpha=0.5, facecolor='k')
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels([0, '', 2, '', 4])
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([])
ax.set_ylim(0, 4)
ax.set_ylabel('\# signif.')
# ax.set_xlabel(r'$\rho$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-4_rho-range_pv')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ######################  SUBSPACES OVERLAP  ##################################
# #############################################################################
# #############################################################################
# #############################################################################
strm = 0
SVD = FG.get_svd_df(act_m.loc[par_act.ORN])
SVD_N = 5
act_subspc = SVD['U'].loc[:, 1:SVD_N]
act_subspc_v = act_subspc.values

con_ff_sel = con_strms3.loc[:, strm]
con_W = con_ff_sel.loc[:, LNs_MM].copy()
con_W_v = con_W.values

act_Q, _ = LA.qr(act_subspc, mode='economic')  # stays the same for PCA
con_Q, _ = LA.qr(con_W, mode='economic')
scaled = 0
overlap_true = FG.subspc_overlap(act_Q, con_Q, scaled=scaled)
print(overlap_true)
print((act_Q.shape[1] + con_Q.shape[1] - overlap_true)/2)
# print(FG.subspc_overlap(act_Q, con_Q, scaled=1))
# print(FG.subspc_overlap(act_Q, con_Q, scaled=2))
print('done')
# %%
#  confirming that the correlation coefficients are still the same as before
print(FG.get_corr(FG.get_ctr_norm_np(Ws), FG.get_ctr_norm_np(con_W)))

print(FG.get_corr(FG.get_ctr_norm_np(act_subspc), FG.get_ctr_norm_np(con_W)))



print('done')
# %%

N = 50000  # number of iterations
overlaps_rdm = np.zeros((6, N), dtype=float)

# doing significance testing using gaussian random
for i in range(N):
    A = np.random.randn(21, SVD_N)
    B = np.random.randn(21, 4)
    A_Q, _ = LA.qr(A, mode='economic')
    B_Q, _ = LA.qr(B, mode='economic')
    overlaps_rdm[0, i] = FG.subspc_overlap(A_Q, B_Q, scaled=scaled)

# =============================================================================
#     A = np.abs(A)
#     A_Q, _ = LA.qr(A, mode='economic')
#     B_Q, _ = LA.qr(B, mode='economic')
#     overlaps_rdm[1, i] = FG.subspc_overlap(A_Q, B_Q, scaled=scaled)
#
#     B[:, 0] = np.abs(B[:, 0])
#     B_Q, _ = LA.qr(B, mode='economic')
#     overlaps_rdm[2, i] = FG.subspc_overlap(A_Q, B_Q, scaled=scaled)
#
#     B = np.abs(B)
#     B_Q, _ = LA.qr(B, mode='economic')
#     overlaps_rdm[3, i] = FG.subspc_overlap(A_Q, B_Q, scaled=scaled)
# =============================================================================

A1 = act_subspc_v.copy()
A_Q1, _ = LA.qr(A1, mode='economic')
for i in range(N):
    B = FG.shuffle_matrix(con_W_v)
    B_Q, _ = LA.qr(B, mode='economic')
    overlaps_rdm[5, i] = FG.subspc_overlap(A_Q1, B_Q, scaled=scaled)

# =============================================================================
#     A = FG.shuffle_matrix(act_subspc_v)
#     A_Q, _ = LA.qr(A, mode='economic')
#     overlaps_rdm[4, i] = FG.subspc_overlap(A_Q, B_Q, scaled=scaled)
# =============================================================================

# overlaps_pvs = np.mean(overlaps_rdm >= overlap_true, axis=1)
overlaps_pvs = np.mean(overlaps_rdm <= overlap_true, axis=1)
print(overlaps_pvs)
print('done')
# %%
# quick plot
# =============================================================================
# bins=np.linspace(0, 1, 101)
# plt.figure()
# plt.hist(overlaps_rdm[0], bins=bins, alpha=0.5, color='C0')
# # plt.hist(overlaps[2], bins=bins, alpha=0.5, color='C1')
# plt.hist(overlaps_rdm[5], bins=bins, alpha=0.5, color='C2')
# plt.plot([overlap_true, overlap_true], [0, 1000], color='k')
# =============================================================================

pads = [0.35, 0.1, 0.35, 0.1]
fs, ax1 = FP.calc_fs_ax(pads, 18*SQ, 12*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(ax1)
bins = np.linspace(0, 4, 101)
ax.hist((9-overlaps_rdm[0])/2, bins=bins, alpha=0.7, color='grey',
        density=True, label='gaussian')
ax.hist((9-overlaps_rdm[5])/2, bins=bins, alpha=0.7, color='darkslategray',
        density=True, label='shuffled')
ax.plot([(9-overlap_true)/2, (9-overlap_true)/2], [0, 1.8], color='k', lw=1,
        label='true')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set(xlabel=r'~ \# of aligned dimensions $\Gamma$',
       ylabel='probability density',
       xticks=[0, 1, 2, 3, 4], xticklabels=[0, '', 2, '', 4], yticks=[0, 1, 2],
       xlim=(0, 4))
# legend
handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 0]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          frameon=False, bbox_to_anchor=(1.1, 1.1), loc='upper right',
          handlelength=1, handletextpad=0.4)

ax.text(0.65, 0.17, f"pv = {overlaps_pvs[0]:.0e}", transform=ax.transAxes,
        color='grey', fontsize=5)
ax.text(0.65, 0.05, f"pv = {overlaps_pvs[5]:.0e}", transform=ax.transAxes,
        color='darkslategray', fontsize=5)

file = f'{PP_COMP_CON}/con_act{SVD_N}_subspc_overlap_hist3'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print(np.mean((9-overlaps_rdm[0])/2), np.mean((9-overlaps_rdm[5])/2))
print((9-overlap_true)/2)
print('done final')