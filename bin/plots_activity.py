#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:56:59 2019

@author: Nikolai M Chapochnikov

External files that are read and used for this plotting:
in plots_paper_import:
f'cons/cons_full_{k}.hdf'
/act3.hdf' FROM preprocess-activity.py
'cons/cons_ORN_all.hdf' FROM preprocess-connectivity.py

In this file (all inside RESULTS_PATH):
'NNC-W_act-all.hdf' FROM olf_circ_offline_sims_ORN-data.py

(f'{CELL_TYPE}_con{STRM}_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')
FROM act_odors_ORN_vs_con_ORN-LN.py

file = (f'/corr_sign/act{DATASET}-{act_pps1}-{act_pps2}'
        f'_conc-all_SVD-NNC_vs_con-ORN-all_')
file_cc = f'{file}cc.hdf'
file_cc_pv = f'{file}cc_pv.hdf'
FROM act_ORN_vs_con_ORN-LN.py

(f'NNC-{K}_con-W_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')
FROM act_odors_ORN_vs_con_NNC.py

(f'{cell_type}_con{STRM}_vs_'
                  f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_{LN}_recon_rand')
FROM act_odors_ORN_vs_con_ORN-LN_recon.py
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
# ####################  ACTIVITY PLOTS  #######################################
# #############################################################################

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
cb_title = '$\Delta F/F_0$'
# cb_title = r'$\Delta$'
cb_ticks = [-1, 0, 2, 4, 6]
# cb_ticks = [-1, -0.1, 0, 0.1, 1, 8]  # for the log scale
pads = [0.55, 0.4, 1.32, 0.2]
f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title, cb_ticks,
                                 pads=pads, cb_title_font=cb_title_font,
                                 squeeze=0.5)
# ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                                 cb_ticks, cb_title_font=cb_title_font,
                                 squeeze=0.5)
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
                                 extend='neither', cb_title_font=cb_title_font,
                                 squeeze=0.5)
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

cp = FP.imshow_df(df, ax, vlim=[-0.85, 0.85], rot=0, cmap=plt.cm.plasma,
                  title='PCA')

cb_x = axs[0] + axs[2] + CB_DX/fs[0]
ax_cb = f.add_axes([cb_x, axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, ticks=[-0.8, -0.4, 0, 0.4, 0.8])

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
    order_LN = {4: [3, 1, 4, 2], 5: [3, 4, 1, 5, 2]}
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

    cp = FP.imshow_df(df, ax, vlim=[0, None], rot=0, cmap=plt.cm.plasma,
                      show_lab_y=False, title=f'NNC-{k}')

    cb_x = axs[0] + axs[2] + CB_DX/fs[0]
    ax_cb = f.add_axes([cb_x, axs[1], CB_W/fs[0], axs[3]])
    add_colorbar_crt(cp, ax_cb, ticks=[0, 0.5])

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

# importlib.reload(FP)
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

ylim2 = (-0.2, 5.3)
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
                                 c1=c1, c2=c2, m1=',', m2=',',
                                 label1=r'\# of syn. ORNs$\rightarrow$BT',
                                 label2='ORN $\Delta F/F_0$')
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
labs = ['\# syn.', 'odor A', 'odor B']
leg = ax.legend(lns, labs, ncol=3, loc='lower center',
                bbox_to_anchor=(0.5, 1.01, 0, 0.), handlelength=2)

file = f'{PP_ODOR_CON}/{LN}_2odors-{conc}_2axplot'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# #################  SCATTER PLOT ORN ACT FOR ODOR VS CON  ####################
# #############################################################################

# importing the p-values of all LNs vs all odors:
STRM = 0
CONC = 'all'
file = (f'{CELL_TYPE}_con{STRM}_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')

pvals = pd.read_hdf(RESULTS_PATH / file)


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
                    yticks=[0, 2, 4], clip_on=False,
                    pca_line_scale1=scl1[i], pca_line_scale2=scl2[i],
                    s=5, c='indigo', pvalue=None, show_cc=False)
    # corr coef and pvalues:
    corr_coef = np.corrcoef(con_w.values, act_vect.values)[0, 1]
    ax.text(0.01, 1, r'$r$' + " = %0.2f" % corr_coef, transform=ax.transAxes,
            va='top')
    ax.text(0.01, 0.9, "pv = %0.1g" % pvals.loc[LN, (odor, conc)],
            transform=ax.transAxes, va='top')


    ax.set_title(title[i], color=c2)
    ax.set_xlim(ylim)
    ax.set_ylim(ylim2)
    FP.set_aspect_ratio(ax, 1)
    file = f'{PP_ODOR_CON}/{LN}_{odor}-{conc}_scatter'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print('done')
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# plotting the same as above but with using PCA

# importing some data
file = (f'{RESULTS_PATH}/corr_sign/act{DATASET}-{act_pps1}-{act_pps2}'
        f'_conc-all_SVD-NNC_vs_con-ORN-all_')

file_cc = f'{file}cc.hdf'
file_cc_pv = f'{file}cc_pv.hdf'

data_cc = pd.DataFrame(pd.read_hdf(file_cc))[STRM]
data_cc = data_cc.reset_index(level=['par1', 'par2'], drop=True)

data_cc_pv = pd.DataFrame(pd.read_hdf(file_cc_pv))[STRM]
data_cc_pv = data_cc_pv.reset_index(level=['par1', 'par2'], drop=True)



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
                    bbox_to_anchor=(0.1, 1.01, .8, 0.1), handlelength=2)
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

# importlib.reload(FP)
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
assert np.abs(corr_coef - data_cc.loc[('o', 'SVD', 1), LN]) < 1e-10
# ax.text(0.65, 0.05, r'$r$'+" = %0.2f" % corr_coef, transform=ax.transAxes)
ax.text(0.01, 1, r'$r$' + " = %0.2f" % corr_coef, transform=ax.transAxes,
        va='top')
ax.text(0.01, 0.9, "pv = %0.1g" % data_cc_pv.loc[('o', 'SVD', 1), LN],
        transform=ax.transAxes, va='top')
# ax.set_title('PCA 1', color=c1)
ax.set_xlim(-2, 45)
ax.set_ylim(-0.02, 0.55)
# ax.set_ylim(-0.5, None)
FP.set_aspect_ratio(ax, 1)

# FP.save_plot(f, PATH_PLOTS + 'PCA1+conBrT1L_scatter_3.pdf', SAVE_PLOTS,
#              **pdf_opts)
file = f'{PP_COMP_CON}/{LN}_PCA1_scatter'
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
import statsmodels.stats.multitest as mt

# https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
def highlight_cell(x,y, ax=None, linewidth=0.5, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, linewidth=linewidth, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

# ordering of the concentration so that it is as before, maybe quicker way of doing it
odor_order = par_act.odor_order
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]

# importing the p-values of all LNs vs all odors:
STRM = 0
CONC = 'all'
file = (f'{CELL_TYPE}_con{STRM}_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')

pvals = pd.read_hdf(RESULTS_PATH / file).loc[:, idx]
pvals_fdr = pvals.copy()
for i in range(pvals.shape[0]):
    pvals_fdr.iloc[i] = mt.multipletests(pvals_fdr.iloc[i].values, method='fdr_bh')[1]


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
df = cors.loc[:, idx]


# plotting
splx = np.arange(5, len(idx), 5)
# sply = [6, 6+4, 6+4+4]
sply = [7, 7+5, 7+5+5]
pads = (0.95, 0.41, 1.32, 0.2)
fs, axs = FP.calc_fs_ax(pads, SQ*len(df.T)*0.45, SQ*len(df))  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
cp = FP.imshow_df(df, ax, vlim=[-1, 1], cmap=corr_cmap, splits_x=splx,
                  splits_y=sply, aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       xlabel=f'ORN activation patterns {Xttex} to odors at different dilutions',
       ylabel=wLNtex)
for (y, x) in np.argwhere(pvals.values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.5, color="green")

for (y, x) in np.argwhere(pvals_fdr.values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.75, color="yellow")

# doesn't do anything like I want
# x_lin = np.arange(pvals.shape[1])
# y_lin = np.arange(pvals.shape[0])
# X, Y = np.meshgrid(x_lin, y_lin)
# ax.contour(X, Y, pvals, [0.05])

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
fs, axs = FP.calc_fs_ax(pads, SQ*len(df.T)*0.5, SQ*len(df))  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
cp = FP.imshow_df(df, ax, vlim=[-1, 1], cmap=corr_cmap, splits_x=splx,
                  aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=[])
ax.set_xlabel(f'ORN activation patterns {Xttex} to odors at different dilutions')
ax.set_ylabel(wLNtypetex)
# ax.set_ylabel('from\nORNs\nto', rotation=0, fontsize=ft_s_tk, labelpad=3,
#               va='center', ha='right')
# f.text(0.005, 0.5, wLNtypetex + ':', rotation=90, va='center', ha='left')
# ax.annotate('', xy=(-6.5, -1), xytext=(-6.5, 4), xycoords='data',
#             arrowprops={'arrowstyle': '-', 'lw': 0.5},
#             annotation_clip=False)

for (y, x) in np.argwhere(pvals.loc[LNs_MM].values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.5, color="green")

for (y, x) in np.argwhere(pvals_fdr.loc[LNs_MM].values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.75, color="yellow")

plt.title(f'Correlation between ORN activity patterns {Xttex}'
          r' and ORNs$\rightarrow$LN conn. weight vectors '+ f'{wLNtypetex}')
ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/ORN_con0-LN-M_vs_act3_raw_mean_all-odors_corr'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# Cumulative distribution of the correlations
# coefficients for each LN to then plot the "tuning curves" for each LN
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
pads = (0.4, 0.1, 0.55, 0.15)
fs, axs = FP.calc_fs_ax(pads, SQ*15, SQ*11)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.plot([-10, 180], [0, 0], c='gray', lw=0.5)
for i in range(4):
    # plt.figure()
    my_sorted = np.sort(cors.iloc[i].values)[::-1]
    # sorted2 = np.concatenate([sorted, sorted[::-1]])
    ax.plot(my_sorted, label=cors.index[i], lw=1)
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#           bbox_to_anchor=(0.5, 1.1), loc='upper left',
#           handlelength = 1, handletextpad=0.4)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
ax.set_ylim(-0.5, 0.85)
ax.set_xlim(-5, 170)
ax.set_yticks([-0.5, 0, 0.5])
ax.set_xticks([0, 50, 100, 150])
ax.set_ylabel(r'corr. coef. $r$')
ax.set_xlabel('ordered stimuli')


file = f'{PP_ODOR_CON}/LN_tuning_curves.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#
#
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #########  CORRELATION COEFFICIENTS FOR ALL ODORS AND W from NNC-4 ##########
# #############################################################################
import statsmodels.stats.multitest as mt

# https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
def highlight_cell(x,y, ax=None, linewidth=0.5, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, linewidth=linewidth, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

# ordering of the concentration so that it is as before, maybe quicker way of doing it
odor_order = par_act.odor_order
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]

# importing the p-values of all W from NNC vs all odors:
CONC = 'all'
K = 4
file = (f'NNC-{K}_con-W_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')

pvals = pd.read_hdf(RESULTS_PATH / file).loc[:, idx]
pvals = pvals.loc[order_LN[4], :]
pvals_fdr = pvals.copy()
for i in range(pvals.shape[0]):
    pvals_fdr.iloc[i] = mt.multipletests(pvals_fdr.iloc[i].values, method='fdr_bh')[1]


# calculation of the correlation coefficients
# con_W = con_strms3.loc[:, 0].loc[:, LNs_sel_LR_d]
file = RESULTS_PATH / 'NNC-W_act-all.hdf'
Ws = pd.DataFrame(pd.read_hdf(file))
conc = 'all'
scal = 1  # so this is the same as in the figure 3 of the paper
pps = f'{scal}o'
meth = f'NNC_{K}'
W = Ws[conc, pps, meth, '', '']/scal
con_W_cn = FG.get_ctr_norm(W)
con_W_cn = con_W_cn.loc[:, order_LN[4]]
con_W_cn.columns = [1, 2, 3, 4]
act_cn = FG.get_ctr_norm(act_m)

# the alignment (which changes the names of the ORNs so that both
# datasets have the same names) is necessary so that the
# correlation coefficient calculation is done correctly
act_cn2, con_W_cn2 = FG.align_indices(act_cn, con_W_cn)
cors = con_W_cn2.T @ act_cn2
df = cors.loc[:, idx]


# plotting
splx = np.arange(5, len(idx), 5)
pads = (0.25, 0.4, 0.2, 0.2)
fs, axs = FP.calc_fs_ax(pads, SQ*len(df.T)*0.5, SQ*len(df))  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
cp = FP.imshow_df(df, ax, vlim=[-1, 1], cmap=corr_cmap, splits_x=splx,
                  aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=[],
       xlabel=f'ORN activation patterns {Xttex} to odors at different dilutions',
       ylabel='$\mathbf{w}_k$')
for (y, x) in np.argwhere(pvals.values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.5, color="green")

for (y, x) in np.argwhere(pvals_fdr.values < 0.05):
    highlight_cell(x, y, ax=ax, linewidth=0.75, color="yellow")

# doesn't do anything like I want
# x_lin = np.arange(pvals.shape[1])
# y_lin = np.arange(pvals.shape[0])
# X, Y = np.meshgrid(x_lin, y_lin)
# ax.contour(X, Y, pvals, [0.05])

plt.title(f'Correlation between ORN activity patterns {Xttex} '
          r'and ORNs$\rightarrow$LN connection weight vectors '
          r'$\mathbf{w}_k$ from NNC-4')

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/NNC{K}w_vs_act3_raw_mean_all-odors_corr'
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
# #####  PVALUES OF CORRELATION COEFFICIENTS FOR ALL ODORS AND ALL LNs  #######
# #############################################################################
# importing the p-values of all LNs vs all odors:
STRM = 0
CONC = 'all'
file = (f'{CELL_TYPE}_con{STRM}_vs_'
              f'act-{act_pps1}-{act_pps2}-{ACT_PPS}-conc-{CONC}_pvals.h5')

pvals = pd.read_hdf(RESULTS_PATH / file)

pvals_sel = pvals.loc[LNs_MM]
# cors_sel = cors.loc[['Broad T', 'Broad D', 'Keystone', 'Picky 0 [dend]']]
#%%
# let's do 2 plots, one showing the raw pvalues and a violin plot or histogram
# on top

pads = (0.3, 0.1, 0.3, 0.2)
fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*14)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

ax.violinplot(pvals_sel.T, bw_method=0.1, positions=np.arange(len(pvals_sel)),
              showextrema=False)
sns.swarmplot(ax=ax, data=pvals_sel.T, color='black', size=0.5)
ax.set_xticks([0, 1, 2, 3], ['BT', 'BD', 'KS', 'P0'])
ax.set_yticks([0, 0.1, 0.5, 1])
ax.set_ylabel('p-value')
ax.set_xlabel(wLNtypetex)
file = f'{PP_ODOR_CON}/LN_p-values.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
#%%

# doing additional tests on the distribution of p-values.

# import scipy.stats as stats
# for i in range(0,4):
#     print(stats.kstest(pvals_sel.iloc[i].loc[(slice(None), 5)], 'uniform', alternative='greater')[1])
#
#
#
# #%%
# for i in range(0,4):
#     pv1 = pvals_sel.iloc[i]
#     print(np.sum(pv1<0.05)/len(pv1))
#
# #%%
# idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
# cors_sel = cors_sel.loc[:, idx]
# pvals_sel = pvals_sel.loc[:, idx]
# for i in range(0,4):
#     pv1 = -np.log10(pvals_sel.iloc[i])
#     cr1 = cors_sel.iloc[i]
#     plt.figure()
#     plt.scatter(cr1, pv1)
#     plt.show()
#
#     # print(np.sum(pv1<0.05)/len(pv1))
#
# #%%
#
# df = cors_sel.loc[:, idx]
# plt.figure()
# plt.imshow(df, aspect='auto', vmin=-1, vmax=1)
# plt.show()
# #%%
# idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
# df = -np.log10(pvals_sel.loc[:, idx])
# plt.figure()
# plt.imshow(df, aspect='auto', vmin=0, vmax=3)
# plt.show()

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
# ########################  RECON of CON with Activity  #######################
# #############################################################################

from sklearn import linear_model
import scipy.interpolate
import scipy.linalg as LA

ORN_order = par_act.ORN_order
act_o_np = act_m.loc[ORN_order, :].values
act_n_np = act_o_np / LA.norm(act_o_np, axis=0)
# act_o_np

con_o = con_strms3.loc[ORN_order, 0]



cell_type = 'ORN'
act_pps_k1 = 'raw'
act_pps_k2 = 'mean'
CONC = 'all'



lasso = True
# lasso = False

if lasso:
    l_norm = 1
    alphas1 = np.logspace(-4, -1, 50)
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

res_rand = {}
results = {}
v_pval = {}
for LN in LNs_MM:
    print(LN)
    w = con_o[LN].values
    w1 = w / LA.norm(w)

    res = pd.DataFrame(index=np.arange(len(alphas1)),
                           columns=['alpha', 'error', 'coef_norm'])
    for i, alph in enumerate(alphas1):
        clf = model(alph)
        clf.fit(act_n_np, w1)
        bias = clf.intercept_
        error = LA.norm(w1 - act_n_np @ clf.coef_ - bias)
        coef_norm = LA.norm(clf.coef_, ord=l_norm)
        res.loc[i, 'alpha'] = alph
        res.loc[i, 'error'] = error
        res.loc[i, 'coef_norm'] = coef_norm
    results[LN] = res.copy()


    # reading the data that was generated in the file
    # act_odors_ORN_vs_con_ORN-LN_recon.py
    file_begin = (f'{cell_type}_con{STRM}_vs_'
                  f'act-{act_pps_k1}-{act_pps_k2}-{ACT_PPS}-conc-{CONC}_{LN}_recon_rand')
    res_rand[LN] = pd.read_hdf(RESULTS_PATH / f'{file_begin}.h5')

    gr = res_rand[LN].groupby('w')
    y_interps = [
        scipy.interpolate.interp1d(gr.get_group(i)['coef_norm'],
                                   gr.get_group(i)['error'],
                                   fill_value='extrapolate')
        for i in range(N)]

    v_p = []
    for i in range(0, len(alphas1) - 1):
        x = res['coef_norm'].iloc[i]
        if x == 0:
            continue
        y = res['error'].iloc[i]
        ys = np.array([y_interps[i](x) for i in range(N)])
        p = np.mean(ys < y)
        print(alphas1[i], x, p)
        v_p.append([x, p])
    v_p = np.array(v_p)
    v_pval[LN] = v_p.copy()
#%%

dict_LN = {'Broad T M M': 'BT',
                              'Broad D M M': 'BD',
                              'Keystone M M': 'KS',
                              'Picky 0 [dend] M': 'P0'}

for LN in LNs_MM:
    pads = (0.35, 0.4, 0.3, 0.2)
    fs, axs = FP.calc_fs_ax(pads, SQ * 11, SQ * 14)  # pads, gw, gh
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    # plt.scatter(errors, coef_norms)
    groups = res_rand[LN].groupby('w')
    for i in range(N):
        res_rand_sel = groups.get_group(i)
        # if i ==0:
        #     plt.plot(res_rand_sel['coef_norm'], res_rand_sel['error'],color='gray',
        #              lw=0.2, alpha=0.1, label='shuffled $\mathbf{w}$')
        # else:
        plt.plot(res_rand_sel['coef_norm'], res_rand_sel['error'],color='gray',
                     lw=0.2, alpha=0.1)

    ax.plot(results[LN]['coef_norm'], results[LN]['error'], color='gray', lw=0.2,
            label='shuffled $\mathbf{w}$')
    ax.plot(results[LN]['coef_norm'], results[LN]['error'], color='k', lw=0.75,
            label='true $\mathbf{w}$')
    ax.legend(borderaxespad=0.2)
    ax.set_xlabel('norm of $\mathbf{v}$')
    ax.set_ylabel('reconstruction error')
    ax.set_title(dict_LN[LN])
    col = 'r'
    ax2 = ax.twinx()
    ax2.plot(v_pval[LN][:, 0], v_pval[LN][:, 1], color=col, lw=0.75)
    ax2.set_ylim(0, 0.2)
    ax2.spines['right'].set(visible=True, color=col)
    ax2.set_ylabel('p-value', color=col)
    ax2.tick_params('y', colors=col)
    file = f'{PP_ODOR_CON}/recon_{LN}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
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

idy = {}


SVD_n = 5
NNC_n = 4
idy[0] = [('o', 'SVD', i) for i in range(1, SVD_n + 1)]
idy[1] = [(pps, f'NNC_{NNC_n}', i) for i in order_LN[NNC_n]]
NNC_n = 5
idy[2] = [(pps, f'NNC_{NNC_n}', i) for i in order_LN[NNC_n]]

ylabel = [f'PCA directions\nof {Xdatatex}',
          'NNC-4, $\mathbf{w}_k$',
          'NNC-5, $\mathbf{w}_k$']

xlabel = {'raw': '$\mathbf{w}_\mathrm{LN}$',
         'M':'$\mathbf{w}_\mathrm{LNtype}$'}
splx = {'raw': [6, 6+4, 10+4], 'M': [1, 2, 3]}
LNs_k = {'raw': LNs_sel_LR_d, 'M': LNs_MM}
sqs = {'raw': 1.1 * SQ, 'M': 2.2 * SQ}
pads = [0.4, 0.45, 0.35, 0.2] # for the correlation plot
pads1 = [0.4, 0.5, 0.35, 0.2] # for the pv plot
print('done')
# %%

def plot_pv_measure(df1, df2, splx, cmap, vlim, i=None):
    """
    measure stands for either corr coef or CS, yeah it's a weird name
    """
    _pads = pads.copy()
    _pads1 = pads1.copy()

    if len(df1.columns) > 5:  # for the raw plots
        _pads[2] = 0.9  # for the correlation plot
        _pads1[2] = 0.9 # for the PV plot
        state = 'raw'
        if i < 2: # not NNC-5
            _pads[2] = 0.2  # for the correlation plot
    if len(df1.columns) < 5:  # for the mean plots
        state = 'M'
        if i == 0: # for the PCA plot
            _pads[2] = 0.55

    _, fs, axs = FP.calc_fs_ax_df(df1, _pads, sq=sqs[state])
    f = plt.figure(figsize=fs)
    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0],
                        axs[1], CB_W / fs[0], axs[3]])
    ax = f.add_axes(axs)  # left, bottom, width, height

    if state == 'M':
        kwargs = {'x_offset': 0.5, 'ha':'right', 'rot': 45}
    elif state == 'raw' and i==2:
        kwargs = {}
    else:
        kwargs = {'show_lab_x':False}

    cp = FP.imshow_df(df1, ax, vlim=vlim, cmap=cmap, splits_x=splx,
                      lw=0.5, **kwargs)

    add_colorbar_crt(cp, ax_cb, r'$r$', [-1 , 0, 1])
    f1 = (f, ax, ax_cb, cp)

    # p value plot, not shown in paper
    _, fs, axs = FP.calc_fs_ax_df(df2, _pads1, sq=sqs[state])
    f = plt.figure(figsize=fs)
    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0],
                        axs[1], CB_W/fs[0], axs[3]])
    ax = f.add_axes(axs)  # left, bottom, widgth, height
    cp = FP.imshow_df(df2, ax, vlim=[-3, -1], lw=0.5,
                      cmap=plt.cm.viridis_r, splits_x=splx)

    clb = add_colorbar_crt(cp, ax_cb, 'pv', [-3, -2, -1], extend='both')
    cp.cmap.set_over('k')
    clb.set_ticklabels([0.001, 0.01, 0.1])
    f2 = (f, ax, ax_cb, cp)
    return f1, f2


# #############################  PLOTTING  ####################################

dict_M = {'Broad T M M': 'BT',
          'Broad D M M': 'BD',
          'Keystone M M': 'KS',
          'Picky 0 [dend] M': 'P0'}

# i = 0: PCA
# i = 1: NNC-4
# i = 2: NNC-5
# raw: all original connectivity
# M: the LNtype connectivity
for i, k in itertools.product(range(3), ['raw', 'M']):  # k as key
    print(i, k)
    df1 = data_cc.loc[idy[i], LNs_k[k]].copy()
    if i == 0:
        df1.iloc[1] = -df1.iloc[1]
    ylabels_new = np.arange(1, df1.shape[0]+1, dtype=int)
    df1.index = ylabels_new
    df1.index.name = ylabel[i]
    df1.columns.name = xlabel[k]

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
    df2.index = ylabels_new
    df2.index.name = ylabel[i]
    df2.columns.name = xlabel[k]

    df1 = df1.rename(dict_M, axis='columns')
    df2 = df2.rename(dict_M, axis='columns')

    f1, f2 = plot_pv_measure(df1, df2, splx[k], corr_cmap,
                             [-1, 1], i)

    for (m, n), label in np.ndenumerate(stars):
        f1[1].text(n, m, label, ha='center', va='top',
                   size=ft_s_lb)

    path_crt = PP_COMP_CON
    if i >= 1:
        path_crt = PP_CON_PRED
    file = f'{path_crt}/act-{pps}-comps{i}_con{STRM}-{k}_cc'
    FP.save_plot(f1[0], f'{file}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f1[0], f'{file}.pdf', SAVE_PLOTS, **pdf_opts)
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
ax.set_xticks([-1, 0, 1], [0.1, 1, 10])
ax.set_ylim(0, 0.8)
ax.set_ylabel(r'corr. coef. $r$')
ax.set_xlabel(r'model inhibition strength $\rho$')

plt.legend(ncol=4, bbox_to_anchor=[0.5, 0.05], loc='lower center')

file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-4_rho-range')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%

# Now i want to calculate the p values for each individual case
cor, pv_o, pv_l, pv_r = FG.get_signif_v1(Ws_cn, con_sel, N=20000)
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
ax.set_yticks([0, 1, 2, 3, 4], [0, '', 2, '', 4])
ax.set_xticks([-1, 0, 1], [])
ax.set_ylim(0, 4)
ax.set_ylabel('\# signif.')
# ax.set_xlabel(r'$\rho$')

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

ax.set(xlabel=r'~ \# of aligned dimensions $\Gamma$',
       ylabel='probability density',
       xticks=[0, 1, 2, 3, 4], xticklabels=[0, '', 2, '', 4], yticks=[0, 1, 2],
       xlim=(0, 4))
# legend
handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 0]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          bbox_to_anchor=(1, 1), loc='upper right')

ax.text(0.65, 0.17, f"pv = {overlaps_pvs[0]:.0e}", transform=ax.transAxes,
        color='grey')
ax.text(0.65, 0.05, f"pv = {overlaps_pvs[5]:.0e}", transform=ax.transAxes,
        color='darkslategray')

file = f'{PP_COMP_CON}/con_act{SVD_N}_subspc_overlap_hist3'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print(np.mean((9-overlaps_rdm[0])/2), np.mean((9-overlaps_rdm[5])/2))
print((9-overlap_true)/2)
print('done final')