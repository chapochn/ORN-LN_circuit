#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:56:59 2019

@author: Nikolai M Chapochnikov

These plots are for the supplements

External files that are read and used for this plotting:
in plots_paper_import:
f'results/cons/cons_full_{k}.hdf'
'results/act3.hdf' FROM act_preprocess.py
'results/cons/cons_ORN_all.hdf' FROM con_preprocess.py

(f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-')

(f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')
FROM act_odors_ORN_vs_con_ORN-LN.py

(f'{RESULTS_PATH}/NNC-{K}_con-W_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-')

(f'{RESULTS_PATH}/NNC-{K}_con-W_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')
FROM act_odors_ORN_vs_con_NNC.py
"""

# %%
# ################################# IMPORTS ###################################
import importlib

from plots_import import *
# %%
# ################################  RELOADS  ##################################
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)
importlib.reload(FCS)
# importlib.reload(par_con)
importlib.reload(par_act)

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
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# ###################  CDF OF 1 LN AND OF MEAN FROM SHUFFLED  #################
# #############################################################################

# ###################         IMPORTING                      ##################

STRM = 0
CONC = 'all'

file = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-')

cdfs_true = pd.DataFrame(pd.read_hdf(f'{file}true.hdf'))
cdfs_shfl_m = pd.DataFrame(pd.read_hdf(f'{file}shfl-m.hdf'))
cdfs_shfl_std = pd.DataFrame(pd.read_hdf(f'{file}shfl-std.hdf'))

xmin = -1
xmax = 1
n_bins_cdf = 100
n_bins_cdf = 500
bins_cdf = np.linspace(xmin, xmax, n_bins_cdf + 1)


# also adding the pvalue on the graph directly

file = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')

cdf_diff_min = pd.DataFrame(pd.read_hdf(f'{file}.hdf'))
cdf_diff_min_pv = pd.DataFrame(pd.read_hdf(f'{file}_pv.hdf'))

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
# plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage[helvet]{sfmath} \usepackage{setspace} \setstretch{0.65} \centering'
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

    pads = (0.42, 0.05, 0.3, 0.1)
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



    ax.set(xlabel='corr. coef. $r$',
           ylabel='relative cumulative \n frequency ($RCF$)',
           xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))

    # legend
    handles, labels = ax.get_legend_handles_labels()
    # order = [0, 3, 2, 1]
    order = [0, 1, 3, 2]
    # order = [0, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              bbox_to_anchor=(0.05, 1.), loc='upper left')

    ax.text(1.0, 0.01, f'LN type: {LN_m}',
            transform=ax.transAxes, ha='right', va='bottom')

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

    ax.set(xlabel=r'corr. coef. $r$', ylabel=r'$RCF - \overline{RCF}$',
           xticks=[-1, 0, 1], yticks=[-0.4, -0.2, 0, 0.2], xlim=(-1, 1))

    i_min = np.argmin(cdf_diff.values)
    col_ann = 'magenta'
    plt.annotate('', xy=(bins_cdf[i_min], 0), xycoords='data',
                 xytext=(bins_cdf[i_min], cdf_diff[i_min]), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': col_ann})
    plt.text(bins_cdf[i_min] + 0.05, cdf_diff[i_min]/2, 'max dev.',
             color=col_ann)
    ax.text(0.02, 0.01, '\\begin{flushleft}' +
            f'pv = {pval_crt:.1}\\newline LN type: {LN_m}' +'\\end{flushleft}',
            transform=ax.transAxes, ha='left', va='bottom')
    # ax.text(0.4, 0.03, f"pv = {pval_crt:.1}", transform=ax.transAxes)


    file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std2'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# this is not used anymore in the main text
# LN_m = 'BT'
# LN = f'Broad T M M'
#
# cdf_mean = cdfs_shfl_m.loc[LN]
# cdf_std = cdfs_shfl_std.loc[LN]
# lw = 1
#

# Plotting the 2 plots separately

# pads = (0.52, 0.05, 0.35, 0.1)
# fs, axs = FP.calc_fs_ax(pads, 12*SQ, 14*SQ)
# f = plt.figure(figsize=fs)
# ax = f.add_axes(axs)
#
# ax.plot(bins_cdf, cdf_mean, drawstyle='steps-post', label='mean', lw=lw,
#         color='k')
# ax.fill_between(bins_cdf, cdf_mean - cdf_std, cdf_mean + cdf_std,
#                 facecolor='grey', step='post', label='s.d.')
#
# ax.plot(bins_cdf, cdfs_true.loc[LN], drawstyle='steps-post', c='r',
#         label='true', lw=lw)
#
#
#
# ax.set(xlabel='corr. coef. $r$',
#        ylabel='relative cumulative \linebreak frequency ($RCF$)',
#        xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))
#
#
# # legend
# handles, labels = ax.get_legend_handles_labels()
# order = [0, 1, 2]
# ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
#           bbox_to_anchor=(0.05, 1), loc='upper left')
#
# ax.text(1, 0.01, f'LN type: {LN_m}', transform=ax.transAxes,
#         ha='right', va='bottom')
#
# file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std1_nofit'
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

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
LN_text = {1:'$\mathbf{w}_1$', 2: '$\mathbf{w}_2$',
           3:'$\mathbf{w}_3$', 4: '$\mathbf{w}_4$'}

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


    ax.set(xlabel='corr. coef. $r$',
           ylabel='relative cumulative \n frequency ($RCF$)',
    # ax.set(xlabel=r'corr. coef. $r$', ylabel='relative cumulative \linebreak frequency',
           xticks=[-1, 0, 1], yticks=[0, 0.5, 1], xlim=(-1, 1))

    # legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 3, 2]
    # order = [0, 3, 2, 1]
    # order = [0, 2, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              bbox_to_anchor=(0.05, 1.), loc='upper left')

    ax.text(1, 0.01, LN_text_crt, transform=ax.transAxes,
            va='bottom', ha='right')

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

    ax.set(xlabel='corr. coef. $r$', ylabel='$RCF - \overline{RCF}$',
           xticks=[-1, 0, 1], yticks=[-0.4, -0.2, 0, 0.2], xlim=(-1, 1))

    i_min = np.argmin(cdf_diff.values)
    col_ann = 'magenta'
    plt.annotate('', xy=(bins_cdf[i_min], 0), xycoords='data',
                 xytext=(bins_cdf[i_min], cdf_diff[i_min]), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': col_ann})
    plt.text(bins_cdf[i_min] + 0.05, cdf_diff[i_min]/2, 'max dev.',
             color=col_ann)
    ax.text(0.02, 0.01, f'pv = {pval_crt:.1}\n{LN_text_crt}',
            transform=ax.transAxes, ha='left', va='bottom')


    file = f'{PP_ODOR_CON}/cors_W{LN_i_new}_rcf-m-std2'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


print('done')

#
# %%
# #############################################################################
# #########################  CDF DIFF MIN AND PVAL  ###########################
# #############################################################################

# ###################         IMPORTING             ###########################
#
CONC = 'all'

file_begin = (f'{RESULTS_PATH}/{CELL_TYPE}_con{STRM}_vs_act-{act_pps1}'
              f'-{act_pps2}-{ACT_PPS}-conc-{CONC}_corr_cdf-shfl-diff-min')

cdf_diff_min = pd.DataFrame(pd.read_hdf(f'{file_begin}.hdf'))
cdf_diff_min_pv = pd.DataFrame(pd.read_hdf(f'{file_begin}_pv.hdf'))
print('done')
# # %%
# # ##################      PLOTTING FOR INDIVIDUAL CELLS      ##################
#
# importlib.reload(FP)
#
# LN_idx = LNs_sel_LR_d
# pvals = cdf_diff_min_pv.loc[LN_idx].squeeze()  # converts into a series
# alpha = 0.05
# reject, pvals_corrected, _, _ = smsm.multipletests(pvals, method='fdr_bh',
#                                                    alpha=alpha)
# pvals_fdr = pvals.copy()
# pvals_fdr[:] = pvals_corrected
# x = np.arange(len(LN_idx), dtype=float)
# x[6:] += 0.5
# x[10:] += 0.5
# x[14:] += 0.5
#
# pads = (0.4, 0.35, 0.9, 0.15)
# fs, axs = FP.calc_fs_ax(pads, SQ*(len(x)+1.5), SQ*15)  # pads, gw, gh
# f = plt.figure(figsize=fs)
# ax = f.add_axes(axs)
#
# axx = FP.plot_double_series_unevenX(ax, x, cdf_diff_min.loc[LN_idx],
#                                     -np.log10(pvals_fdr),
#                                     'magenta', 'b', 'RCF max deviation (\enspace)',
#                                     '$-\log_{10}$(p-value) (\enspace)',
#                                     ylim1=(0, 0.37), ylim2=(0, 2.1))
# # axx[1].annotate('', xy=(-2, -1.5), xytext=(18.5, -1.5), xycoords='data',
# #                 arrowprops={'arrowstyle': '-', 'lw': 0.5},
# #                 annotation_clip=False)
# axx[1].plot([19.95], [1.77], ls='None', marker='+', color='b', markersize=5,
#             clip_on=False)
# axx[0].plot([-4.5], [0.335], ls='None', marker='.', color='magenta',
#             markersize=5, clip_on=False)
# axx[0].set_yticks([0, 0.1, 0.2, 0.3])
# axx[1].set_yticks([0, 1, 2])
# axx[0].set_xlabel('$\mathbf{w}_\mathrm{LN}$')
# # f.text(0.5, 0., 'conn. weight vectors from ORNs to',
# #    fontsize=matplotlib.rcParams['axes.labelsize'], va='bottom', ha='center')
# # axx[1].set_xlabel('ORNs -> LN weight vectors')
#
# # adding the info about the significance
# # pvals = cdf_diff_min_pv.loc[LN_idx].values
# for alpha, y, sign in [[0.05, 2.1, '*']]:
#     FP.add_sign_stars(axx[1], pvals, alpha, x, y, sign, fontdict={'size': 14})
#
# file = (f'{PP_ODOR_CON}/{CELL_TYPE}_con{STRM}_vs_act'
#         f'-{act_pps1}-{act_pps2}-{ACT_PPS}_corr_rcf_min_diff')
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
# print('done')
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

pads = (0.32, 0.28, 0.3, 0.15)
fs, axs = FP.calc_fs_ax(pads, SQ*(len(x)+2), SQ*14)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

axx = FP.plot_double_series_unevenX(ax, x, df1, df2, 'magenta', 'b',
                                    r'RCF max deviation (\enspace)', r'$-\log_{10}$(p-value) (\enspace)',
                                    ylim1=(0, 0.37), ylim2=(0, 2.5))
axx[1].plot([7.85], [2.155], ls='None', marker='+', color='b', markersize=5,
            clip_on=False)
axx[0].plot([-4.3], [0.345], ls='None', marker='.', color='magenta',
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
print('final done')
#
