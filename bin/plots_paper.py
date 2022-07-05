#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:56:59 2019

@author: Nikolai M Chapochnikov
"""

# %%
# ################################# IMPORTS ###################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # for colors and gradients
import matplotlib.colors  # for colors and gradients
import pandas as pd
import pathlib
import importlib
import seaborn as sns
# import fitz # to rotate a pdf page, packaged needed to install: pymupdf
# -> pip install pymupdf
# could be replaced by PyPDF2
from typing import Union

import itertools

import statsmodels.stats.multitest as smsm  # for the multihypothesis testing

import scipy.linalg as LA

import scipy.stats as SS
import scipy.cluster.hierarchy as sch
from scipy.optimize import curve_fit


import functions.olfactory as FO
import functions.olf_circ_offline as FOC
import functions.circuit_simulation as FCS
import functions.plotting as FP
import functions.general as FG
import params.act3 as par_act
import params.con as par_con


# %%
# ################################  RELOADS  ##################################
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)
importlib.reload(FCS)
# importlib.reload(par_con)
importlib.reload(par_act)

# %%
# #################################  OPTIONS  #################################

OLF_PATH = FO.OLF_PATH

DATASET = 3
RESULTS_PATH = OLF_PATH / 'results'
# RESULTS_PATH = OLF_PATH / 'results2'

SAVE_PLOTS = True
PLOT_PLOTS = False

PATH_PLOTS = ''
if SAVE_PLOTS:
    PATH_PLOTS = FG.create_path(OLF_PATH / 'plots/plots_paper/')
    print('path plots:', PATH_PLOTS)
    PP_CONN = PATH_PLOTS / 'conn'
    PP_CONN.mkdir(exist_ok=True)
    PP_THEORY = PATH_PLOTS / 'theory'
    PP_THEORY.mkdir(exist_ok=True)
    PP_ACT = PATH_PLOTS / 'activity'
    PP_ACT.mkdir(exist_ok=True)
    PP_COMP_CON = PATH_PLOTS / 'act-comps_vs_con'
    PP_COMP_CON.mkdir(exist_ok=True)
    PP_ODOR_CON = PATH_PLOTS / 'act-odors_vs_con'
    PP_ODOR_CON.mkdir(exist_ok=True)
    PP_CON_PRED = PATH_PLOTS / 'conn_predictions'
    PP_CON_PRED.mkdir(exist_ok=True)
    PP_ROT = PATH_PLOTS / 'rotationX-Y'
    PP_ROT.mkdir(exist_ok=True)
    # defined lower
    # PP_WHITE = PATH_PLOTS / 'whitening'
    # PP_WHITE.mkdir()
    PP_Z = PATH_PLOTS / 'clusteringZ'
    PP_Z.mkdir(exist_ok=True)

    PP_WrhoK = PATH_PLOTS / 'W_vs_rho_K'
    PP_WrhoK.mkdir(exist_ok=True)


FP.set_plotting(PLOT_PLOTS)

CELL_TYPE = 'ORN'

# this is the preprocessing related to how the trials are averaged
act_pps1 = 'raw'
act_pps2 = 'mean'  # what is used in the whole paper

# this is the preprocessing about if the data is centered, normalized, etc...
ACT_PPS = 'o'  # o is original

STRM = 0  # this is the first stream, from ORN to LNs

SIDES = ['L', 'R']

# this is useful for having all the LNs so that later i can order them
LNs = ['Broad T1',
       'Broad T2',
       'Broad T3',
       'Broad D1',
       'Broad D2',
       'Keystone L',
       'Keystone R',
       'Picky 0 [dend]',
       'Picky 0 [axon]',
       'Picky 1 [dend]',
       'Picky 1 [axon]',
       'Picky 2 [dend]',
       'Picky 2 [axon]',
       'Picky 3 [dend]',
       'Picky 3 [axon]',
       'Picky 4 [dend]',
       'Picky 4 [axon]',
       'Choosy 1 [dend]',
       'Choosy 1 [axon]',
       'Choosy 2 [dend]',
       'Choosy 2 [axon]',
       'Ventral LN']

LNs_sel = ['Broad T1',
           'Broad T2',
           'Broad T3',
           'Broad D1',
           'Broad D2',
           'Keystone L',
           'Keystone R',
           'Picky 0 [dend]',
           'Picky 0 [axon]']

# a like axon of the Picky
LNs_sel_a = ['Broad T1',
             'Broad T2',
             'Broad T3',
             'Broad D1',
             'Broad D2',
             'Keystone L',
             'Keystone R',
             'Picky 0 [axon]']

# d like dendrite of the Picky
LNs_sel_d = ['Broad T1',
             'Broad T2',
             'Broad T3',
             'Broad D1',
             'Broad D2',
             'Keystone L',
             'Keystone R',
             'Picky 0 [dend]']

LNs_sel_short = ['BT 1',
                 'BT 2',
                 'BT 3',
                 'BD 1',
                 'BD 2',
                 'KS L',
                 'KS R',
                 'P0'
                 # 'P0a'
                 ]

# merging the axon and dendrite for the picky
LNs_sel_ad = ['Broad T1',
              'Broad T2',
              'Broad T3',
              'Broad D1',
              'Broad D2',
              'Keystone L',
              'Keystone R',
              'Picky 0 S']

LNs_sel_ad2 = ['Broad T1',
               'Broad T2',
               'Broad T3',
               'Broad D1',
               'Broad D2',
               'Keystone L',
               'Keystone R',
               'Picky 0']


LNs_sel_LR_d = ['Broad T1 L',
                'Broad T2 L',
                'Broad T3 L',
                'Broad T1 R',
                'Broad T2 R',
                'Broad T3 R',
                'Broad D1 L',
                'Broad D2 L',
                'Broad D1 R',
                'Broad D2 R',
                'Keystone L L',
                'Keystone L R',
                'Keystone R R',
                'Keystone R L',
                'Picky 0 [dend] L',
                'Picky 0 [dend] R',
                # 'Picky 0 [axon] L',
                # 'Picky 0 [axon] R'
                ]

LNs_sel_LRM_d = ['Broad T1 L',
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
                 'Picky 0 [dend] M',
                 # 'Picky 0 [axon] L',
                 # 'Picky 0 [axon] R'
                 ]


LNs_sel_LR_a = ['Broad T1 L',
                'Broad T2 L',
                'Broad T3 L',
                'Broad T1 R',
                'Broad T2 R',
                'Broad T3 R',
                'Broad D1 L',
                'Broad D2 L',
                'Broad D1 R',
                'Broad D2 R',
                'Keystone L L',
                'Keystone L R',
                'Keystone R R',
                'Keystone R L',
                # 'Picky 0 [dend] L',
                # 'Picky 0 [dend] R',
                'Picky 0 [axon] L',
                'Picky 0 [axon] R'
                ]


# not sure i would use that, but could be used if one want to put abbreviations
LNs_sel_LR_short = ['BT 1 L',
                    'BT 2 L',
                    'BT 3 L',
                    'BT 1 R',
                    'BT 2 R',
                    'BT 3 R',
                    'BD 1 L',
                    'BD 2 L',
                    'BD 1 R',
                    'BD 2 R',
                    'KS L L',
                    'KS L R',
                    'KS R R',
                    'KS R L',
                    'P0 L',
                    'P0 R',
                    # 'P0a L',
                    # 'P0a R'
                    ]

LN_cats = ['Broad T',
           'Broad D',
           'Keystone',
           'Picky 0'
           ]

LNs_MM = ['Broad T M M',
          'Broad D M M',
          'Keystone M M',
          'Picky 0 [dend] M'
          ]

# =============================================================================
# LNs_MM = ['Broad T M M',
#           'Broad D M M',
#           'Keystone M M',
#           'Picky 0 M M'
#            ]
# =============================================================================

LN_cats2 = ['Broad T',
            'Broad D',
            'Keystone',
            'Picky 0',
            'Picky'
           ]

LNs_side = {S: [f'{name} {S}' for name in LNs] for S in SIDES}
LNs_sel_side = {S: [f'{name} {S}' for name in LNs_sel] for S in SIDES}
LNs_sel_a_side = {S: [f'{name} {S}' for name in LNs_sel_a] for S in SIDES}
LNs_sel_d_side = {S: [f'{name} {S}' for name in LNs_sel_d] for S in SIDES}
LNs_sel_ad_side = {S: [f'{name} {S}' for name in LNs_sel_ad] for S in SIDES}
LNs_sel_short_side = {S: [f'{name} {S}' for name in LNs_sel_short]
                      for S in SIDES}


ft_s_tk = 6  # fontsize ticks
ft_s_lb = 7  # fontsize labels
png_opts = {'dpi': 400, 'transparent': True}
pdf_opts = {'dpi': 800, 'transparent': True}

# I wonder if something like this is needed in the save_plot:
# bbox_inches='tight', pad_inches = 0
# it seems not, then it doesn't respect the size you've put

CB_W = 0.1
CB_DX = 0.11
SQ = 0.07

# this updates the values for all the plots
matplotlib.rcParams['font.size'] = ft_s_tk  # default 10
matplotlib.rcParams['axes.labelsize'] = 'large'  # default 'medium'
matplotlib.rcParams['axes.titlesize'] = 'large'  # default 'medium'

with open(f'{PATH_PLOTS}/params.txt', 'w') as f:
    f.write(f'\ndataset: {DATASET}')
    f.write(f'\nact_pps1: {act_pps1}')
    f.write(f'\nact_pps2: {act_pps2}')
    f.write(f'\nACT_PPS: {ACT_PPS}')
    f.write(f'\nCELL_TYPE: {CELL_TYPE}')
    f.write(f'\nstrm: {STRM}')
#    f.write(f'\nACT_PPS: {ACT_PPS}')
#    f.write(f'\nCONC: {CONC}')
#    f.write(f'\nact_sel_ks: {act_sel_ks}')

Xdatatex = r'$\{\mathbf{x}^{(t)}\}_\mathrm{data}$'
Xtstex = r'$\{\mathbf{x}^{(t)}\}$'
Xttex = r'$\mathbf{x}^{(t)}$'
Yttex = r'$\mathbf{y}^{(t)}$'
Wdatatex = r'$\mathbf{W}_\mathrm{data}$'
# Ytex = r'$\mathbf{Y}$'
# Ztex = r'$\mathbf{Z}$'
Ytex = r'$\{\mathbf{y}^{(t)}\}$'
Ztex = r'$\{\mathbf{z}^{(t)}\}$'

# %%
# #############################################################################
# ##############################  IMPORTS  ####################################
# #############################################################################
# importlib.reload(FO)
con = FO.get_con_data()
ORNs = par_con.ORN
ORNs_short = [name[4:] for name in ORNs]
ORNs_sorted = par_act.ORN_order
ORNs_sorted_short = [name[4:] for name in ORNs_sorted]

act = FO.get_ORN_act_data(DATASET).T
act_m = act.mean(axis=1, level=('odor', 'conc'))
act_m = act_m.loc[par_act.ORN, :]
act_mr = FG.rectify(act_m)

# #############################################################################
# ####################  ORN CONNECTIVITY IMPORTS  #############################
# #############################################################################

# basically only using this to get the Picky summed for axon and dendrite
# it doesn't change anything for the already existing cells.
con_S = FO.create_summary_LNs(con, func='sum', S='S')
ORNs_side = {}
con_ff = {}
con_fb = {}
cells_all = {}
cells_all2 = {}
cells_no_LN = {}
for s in SIDES:
    ORNs_side[s] = FG.get_match_names(ORNs, list(con[s].index))
    con_ff[s] = con[s].loc[ORNs_side[s]]
    con_fb[s] = con[s].loc[:, ORNs_side[s]].T
    con_ff[s].index = ORNs_short
    con_fb[s].index = ORNs_short

    cells_all[s] = con_ff[s].columns
    cells_no_LN[s] = FG.get_strs_in_cats(cells_all[s], LNs_side[s], compl=True)

    cells_all2[s] = cells_no_LN[s] + LNs_side[s]

con_strms3 = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'cons/cons_ORN_all.hdf'))
con_strms3_cn = FG.get_ctr_norm(con_strms3)

# this are the parameters in the first plot, which fixes the SQ for the rest
# of the file, maybe would make sense to just fix the SQ


# %%
# #############################################################################
# ###################  PLOTTING ORN FULL CONNECTIVITY  ########################
# #############################################################################

# plotting ff and fb plots together, and also putting some more details
# onto the plots, as the categories of the neurons
importlib.reload(FG)

pads = [0.6, 0.55, 0.1, 0.2]
d_h = 1.4

arr_y = 21 + 13  # arrow y position
txt_y = arr_y + 0.5  # annotation text y position
arr_stl = "<->, head_width=0.3"
kwargs = {'annotation_clip': False,
          'arrowprops': dict(arrowstyle=arr_stl, color='C0')}
kwargs2 = {'ha': 'center', 'va': 'top', 'color': 'C0'}
for s in ['L', 'R']:
    # top plot
    df1 = con_ff[s].loc[:, cells_all2[s][21:]]
    df2 = con_fb[s].loc[:, cells_all2[s][21:]]
    fs, axs1, axs2, _ = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, 0, 0)

    # _, fs, axs = FP.calc_fs_ax_df(df1, pads1, sq=SQ)
    # print(SQ, SQ*np.array(df.shape), fs, axs)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs1)
    cp = FP.imshow_df2(df1, ax1, vlim=[0, 120],
                       title=r'feedforward connections: ORNs$\rightarrow$'
                             r'other neurons')
    ax1.set(ylabel='ORNs', xlabel='')
    # arrows
    # plt.arrow(0.5, 35, 19.5, 0, lw=2, clip_on=False,
    # length_includes_head=True)
    ax1.annotate('', xy=(-0.9, arr_y), xytext=(21.9, arr_y), **kwargs)
    ax1.annotate('', xy=(21.1, arr_y), xytext=(39.9, arr_y), **kwargs)
    ax1.annotate('', xy=(39.1, arr_y), xytext=(52.9, arr_y), **kwargs)
    ax1.annotate('', xy=(52.1, arr_y), xytext=(74.9, arr_y), **kwargs)
    ax1.text(21/2, txt_y, 'uniglomerular\nprojection neurons (uPN)', **kwargs2)
    ax1.text((21 + 40)/2, txt_y, 'multiglomerular\nprojection neurons (mPN)',
             **kwargs2)
    ax1.text((39 + 53)/2, txt_y, 'other neurons', **kwargs2)
    ax1.text((52 + 75)/2, txt_y, 'Inhibitory\nlocal neurons (LN)', **kwargs2)
    # colorbar
    cb_x = axs1[0] + axs1[2] + CB_DX/fs[0]
    ax_cb = f.add_axes([cb_x, axs1[1], CB_W/fs[0], axs1[3]])
    FP.add_colorbar(cp, ax_cb, '# of synapses', [0, 40, 80, 120])

    # bottom plot
    # _, _, axs = FP.calc_fs_ax_df(df, pads2, sq=SQ)
    ax2 = f.add_axes(axs2)
    cp = FP.imshow_df2(df2, ax2, vlim=[0, 20], show_lab_x=False,
                       title=r'feedback connections: other neurons'
                             r'$\rightarrow$ORNs')
    ax2.set(ylabel='ORNs', xlabel='')
    # colorbar
    ax_cb = f.add_axes([cb_x, axs2[1], CB_W/fs[0], axs2[3]])
    FP.add_colorbar(cp, ax_cb, '', [0, 10, 20])

    # rectangle
    rect = matplotlib.patches.Rectangle((53 - 0.6, -41.8), 9.2, 62.5, lw=1,
                                        clip_on=False, ls='--',
                                        edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    file = f'{PP_CONN}/con_ORN_{s}_all.'
    FP.save_plot(f, file + 'png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + 'pdf', SAVE_PLOTS, **pdf_opts)

    # this seems to try to make it even tighter than what you set it up for
    # pdf_opts1 = {'dpi': 800, 'transparent': True, 'bbox_inches':'tight',
    #              'pad_inches': 0}
    # file1 = f'{PP_CONN}/con_ORN_{s}_all2.'
    # FP.save_plot(f, file1 + 'pdf', SAVE_PLOTS, **pdf_opts1)



# %%
# #############################################################################
# ###################  PLOTTING ORN SEL. CONNECTIVITY  ########################
# #############################################################################

# keeping the same height as the plot above and letting the width adapt
# basically keeping the same square size

# it is also possible to look at the feedback stream, then you would also
# need to change the name of the file

strm = 0

pads = (0.6, 0.55, 0.81, 0.32)  # l, r, b, t
df = con_strms3.loc[:, strm].copy()
df = df.loc[ORNs_sorted, LNs_sel_LR_d]
df.index = ORNs_sorted_short
# df.columns = LNs_sel2_short
_, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)

splx = [6, 6+4, 6+4+4]

cp = FP.imshow_df2(df, ax1, vlim=[0, 55], title='', splits_x=splx, lw=0.5)
ax1.set(ylabel='ORNs', xlabel='LNs', title=r'ORNs$\rightarrow$LNs'+
                                           f'\n syn. counts '
                                        +r'$\mathbf{w}_\mathrm{LN}^\mathrm{ff}$')
ax1.xaxis.set_label_coords(0.5, -0.45)

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
FP.add_colorbar(cp, ax_cb, '# of syn.', [0, 20, 40])

file = f'{PP_CONN}/con_ORN_ff_LN_sel.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


strm = 1

pads = (0.6, 0.55, 0.81, 0.32)  # l, r, b, t
df = con_strms3.loc[:, strm].copy()
df = df.loc[ORNs_sorted, LNs_sel_LR_a]
df.index = ORNs_sorted_short
# df.columns = LNs_sel2_short
_, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)

splx = [6, 6+4, 6+4+4]

cp = FP.imshow_df2(df, ax1, vlim=[0, 20], title='', splits_x=splx, lw=0.5)
ax1.set(ylabel='ORNs', xlabel='LNs', title=r'LNs$\rightarrow$ORNs'
                                           +f'\n syn. counts '
                                        +r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
ax1.xaxis.set_label_coords(0.5, -0.45)

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
FP.add_colorbar(cp, ax_cb, '# of syn.', [0, 10, 20])

file = f'{PP_CONN}/con_ORN_fb_LN_sel.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #############################################################################
# ###################  SHOWING THE VARIANCE FOR EACH CON CAT  #################
# #############################################################################
# from PIL import Image


# LNs_cat = ['BT', 'BD', 'KS', 'P0']

strm = 0

col1 = 'k'
col_e = 'orangered'
col_e = 'k'


LNs_cat = {'BT': ['Broad T1 L', 'Broad T2 L', 'Broad T3 L',
                  'Broad T1 R', 'Broad T2 R', 'Broad T3 R'],
           'BD': ['Broad D1 L', 'Broad D2 L', 'Broad D1 R', 'Broad D2 R'],
           'KS': ['Keystone L L', 'Keystone L R', 'Keystone R R',
                  'Keystone R L'],
           'P0': ['Picky 0 [dend] L', 'Picky 0 [dend] R']}

LNs_m = {'BT': 'Broad T M M',
         'BD': 'Broad D M M',
         'KS': 'Keystone M M',
         'P0': 'Picky 0 [dend] M'}

LNs_name = {'BT': 'Broad Trio (BT, 6)',
            'BD': 'Broad Duet (BD, 4)',
            'KS': 'Keystone (KS, 4)',
            'P0': 'Picky 0 (P0, 2)'}


# adding in the background lines showing the variance from a Poisson process.
LNs_ymax = {'BT': 56,
            'BD': 42,
            'KS': 33,
            'P0': 45}

LNs_yticks = {'BT': [0, 20, 40],
              'BD': [0, 20, 40],
              'KS': [0, 20],
              'P0': [0, 20, 40]}

for LN, LN_list in LNs_cat.items():

    LN_m = LNs_m[LN]
    data = con_strms3.loc[:, strm]
    data = data.loc[ORNs_sorted, LN_m].copy()
    data.index = ORNs_sorted_short
    cell_list = data.index

    data1 = con_strms3.loc[:, strm].loc[ORNs_sorted, LN_list].copy()
    data_std = data1.std(axis=1)  # normalized by n-1, which is the unbiased
    # version of the std
    data_m = data1.mean(axis=1)

    print(np.max(np.abs(data.values-data_m.values)))

    if LN == 'BT':
        pads = [0.2, 0.35, 0.55, 0.12]
    else:
        pads = [0.2, 0.35, 0.12, 0.12]

    fs, ax1 = FP.calc_fs_ax(pads, 21*SQ, 10*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(ax1)

    mean = data_m.mean()
    std = SS.poisson.std(mean)
    n_total = data_m.sum()
    std2 = SS.binom.std(n_total, p=1/21)
    print(std, std2)

    # ax.fill_between([-0.5, len(data)-0.5], [mean - std]*2, [mean + std]*2,
    #                  facecolor='red', alpha=0.3, zorder=10)
    # ax.plot([-0.5, len(data)-0.5], [mean]*2, c='red', lw=0.5, zorder=15)

    ax.errorbar(np.arange(len(data)), data, c=col1, yerr=data_std,
                elinewidth=0.5, ecolor=col_e, capsize=1, lw=1, zorder=20)
    ax.plot(data1, lw=0.5)
    ax.set_ylim(-2, LNs_ymax[LN])
    ax.set_xlim(-0.5, 20.5)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines['right'].set_edgecolor(col1)
    # ax.set_title('connection vector from ORNs to ' + lbl)

    plt.sca(ax)
    y_max = np.ceil(data.max()/10) * 10
    # ytcks = [0, int(y_max/2), int(y_max)]
    ytcks = LNs_yticks[LN]
    plt.yticks(ytcks, ytcks, va='center')
    ax.tick_params(axis='x', which='both', bottom=False, pad=-1,
                   labelrotation=90)
    ax.tick_params(axis='y', which='both', pad=1, rotation=90, colors=col1)

    f.text(-0.1, 0.5, LNs_name[LN], rotation=90, transform=ax.transAxes,
           fontsize=ft_s_lb*0.9, va='center', ha='left')

    if LN == 'BT':
        ax.set_xlabel('ORNs', rotation=180)
        plt.sca(ax)
        plt.xticks(np.arange(len(cell_list)), cell_list, rotation='vertical')
        ax.set_ylabel(r'# of syn. ORNs$\rightarrow$LN    ', color=col1)
    else:
        plt.sca(ax)
        plt.xticks(np.arange(len(cell_list)), ['']*21)


    # adjusting the borders
    ax.xaxis.grid(zorder=0)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    file = f'{PP_CONN}/con_{LN}_rot'
    FP.save_plot(f, file+'.png', SAVE_PLOTS, **png_opts)
    fimg = plt.imread(file + '.png')
    fimg = np.rot90(fimg, k=3)
    fimg = fimg.copy(order='C')  # not sure why I need this, but
    # otherwise imsave complains...
    plt.imsave(file + '2.png', fimg, dpi=400)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
    doc = fitz.open(file + '.pdf')
    page = doc[0]
    page.setRotation(90)
    doc.save(file + '2.pdf')
    doc.close()


# %%
# #############################################################################
# #########################  CONNECTION CORRELATIONS  #########################
# #############################################################################
# here, to conserve the dimensions, i would like to have the height and the
# width exactly corresponding to the width of the previous graph, which
# contains all the LNs
wLNtex = r'$\mathbf{w}_{\mathrm{LN}}$'

for strm in [0, 1]:
    # Correlation coefficient per category
    cat = {}
    cat['BT'] = ['BT 1 L', 'BT 2 L', 'BT 3 L', 'BT 1 R', 'BT 2 R',
                 'BT 3 R']
    cat['BD'] = ['BD 1 L', 'BD 2 L', 'BD 1 R', 'BD 2 R']
    cat['KS'] = ['KS L L', 'KS L R', 'KS R R', 'KS R L']
    cat['P0'] = ['P0 L', 'P0 R']
    if strm == 0:
        LNs_sel_LR = LNs_sel_LR_d
        title = (r'corr. among ORNs$\rightarrow$LN'
                 + f'\nconn. weight vectors {wLNtex}')
        xylabel = 'ORNs\n' + r'$\rightarrow$'
        ylabel = r'$\mathbf{w}_\mathrm{LN}$'
        xlabel = r'$\mathbf{w}_\mathrm{LN}$'
        # ylabel = r'$\mathbf{w}_\mathrm{LN}$:' + '\nfrom\nORNs\nto'
        # xlabel = r'$\mathbf{w}_\mathrm{LN}$: from ORNs to'
    else:
        LNs_sel_LR = LNs_sel_LR_a
        title = (r'corr. among LN$\rightarrow$ORNs' + f'\n'
                 + r'conn. weight vectors $\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
        # xylabel = r'...$\rightarrow$ ORNs'
        xylabel = 'ORNs\n' + r'$\leftarrow$'
        ylabel = r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$'
        xlabel = r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$'
        # ylabel = 'to\nORNs\nfrom'
        # xlabel = 'to ORNs from'

    con_ff_sel = con_strms3.loc[:, strm]
    con_ff_sel = con_ff_sel.loc[:, LNs_sel_LR]
    con_ff_sel.columns = LNs_sel_LR_short
    # if strm == 0:
    #     labels = [r'ORNs $\rightarrow$ '+ LN for LN in LNs_sel_LR_short]
    #     con_ff_sel.columns = labels
    # else:
    #     con_ff_sel.columns = LNs_sel_LR_short
    con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
    grammian = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)

    # pads = (0.5, 0.45, 0.5, 0.2)  # l, r, b, t
    pads = (0.6, 0.35, 0.52, 0.32)  # l, r, b, t
    _, fs, axs = FP.calc_fs_ax_df(grammian, pads, sq=SQ)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs)
    cp = FP.imshow_df2(grammian, ax1, cmap=plt.cm.bwr, vlim=1,
                       splits_x=[6, 10, 14], splits_y=[6, 10, 14],
                       show_lab_x=False)
    ax1.set_xticks(np.arange(len(grammian.T)) + 0.5)
    ax1.set_xticklabels(list(grammian.columns), rotation=70, ha='right')
    ax1.tick_params('x', bottom=False, pad=-1)
    # ax1.set(xlabel=xylabel, ylabel=xylabel, title=title)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    # ax1.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
    #                rotation_mode='default', ha='center', va='top')
    # ax1.set_ylabel(ylabel, rotation=0, fontsize=ft_s_tk, labelpad=3,
    #                va='center', ha='right')
    # ax1.annotate('', xy=(-2, 20.3), xytext=(17, 20.3), xycoords='data',
    #              arrowprops={'arrowstyle': '-', 'lw':0.5},
    #              annotation_clip=False)
    # ax1.annotate('', xy=(-5.3, -1), xytext=(-5.3, 16), xycoords='data',
    #              arrowprops={'arrowstyle': '-', 'lw':0.5},
    #              annotation_clip=False)

    # ax1.xaxis.set_label_coords(0.5, -0.6)
    # ax1.yaxis.set_label_coords(-0.6, 0.5)

    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0],
                        axs[3]])

    FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
    file = (f'{PP_CONN}/{CELL_TYPE}_con{strm}_cn_grammian')
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)



    gram_cat = pd.DataFrame(index=cat.keys(), columns=cat.keys(), dtype=float)
    for c1, c2 in itertools.product(cat.keys(), cat.keys()):
        mat = grammian.loc[cat[c1], cat[c2]].values
        if c1 == c2:
            n = len(cat[c1])
            triu = np.triu_indices(n, 1)
            gram_cat.loc[c1, c2] = np.mean(FG.rectify(mat[triu]))
        else:
            gram_cat.loc[c1, c2] = np.mean(FG.rectify(mat))


    # plotting
    # pads = (0.5, 0.3, 0.5, 0.5)  # l, r, b, t
    pads = (0.25, 0.25, 0.45, 0.5)  # l, r, b, t
    _, fs, axs = FP.calc_fs_ax_df(gram_cat, pads, sq=SQ*1.5)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs)

    FP.imshow_df2(gram_cat, ax1, cmap=plt.cm.bwr, vlim=1,
                       splits_x=[1, 2, 3], splits_y=[1, 2, 3], rot=50)
    ax1.set(xlabel='', ylabel='', title=r'mean corr. coef. $r$' + '\nwithin and '
                                        'across\n LN types')

    file = (f'{PP_CONN}/{CELL_TYPE}_con{strm}_cn_grammian_cat')
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# %%
# plot correlation of feedforward vs feedforward so that we have that in the
# paper and have it as argument for not showing the analysis for feedback.

strm = 0
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel_LR_d]
con_ff_sel.columns = LNs_sel_LR_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
strm = 1
con_fb_sel = con_strms3.loc[:, strm]
con_fb_sel = con_fb_sel.loc[:, LNs_sel_LR_a]
con_fb_sel.columns = LNs_sel_LR_short
con_fb_sel_cn = FG.get_ctr_norm(con_fb_sel)
grammian = FG.get_corr(con_ff_sel_cn, con_fb_sel_cn)

# pads = (0.8, 0.45, 0.8, 0.2)  # l, r, b, t
# pads = (0.6, 0.35, 0.45, 0.2)  # l, r, b, t
pads = (0.6, 0.35, 0.52, 0.32)  # l, r, b, t
_, fs, axs = FP.calc_fs_ax_df(grammian, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)
cp = FP.imshow_df2(grammian, ax1, cmap=plt.cm.bwr, vlim=1, show_lab_x=False,
                   splits_x=[6, 10, 14], splits_y=[6, 10, 14])
ax1.set_xticks(np.arange(len(grammian.T))+0.5)
ax1.set_xticklabels(list(grammian.columns), rotation=70, ha='right')
ax1.set_title(f'corr. between\n'+ r'$\mathbf{w}_\mathrm{LN}^\mathrm{ff}$ and '
               + r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
ax1.set_xlabel(r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
ax1.set_ylabel(r'$\mathbf{w}_\mathrm{LN}^\mathrm{ff}$')
# ax1.set_xlabel('to ORNs from', rotation=0, fontsize=ft_s_tk, labelpad=2)
# ax1.set_ylabel('from\nORNs\nto', rotation=0, fontsize=ft_s_tk,
#                labelpad=3, va='center', ha='right')
ax1.tick_params('x', bottom=False, pad=-1)
# ax1.xaxis.set_label_coords(0.5, -0.6)
# ax1.yaxis.set_label_coords(-0.6, 0.5)
# ax1.annotate('', xy=(-2, 20.3), xytext=(17, 20.3), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax1.annotate('', xy=(-5.3, -1), xytext=(-5.3, 16), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
file = f'{PP_CONN}/{CELL_TYPE}_con0vs1_cn_grammian'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# %%
# this is not used in the paper

# left and right side separately
# strm = 0
# xlabel = 'from ORNs to'
# ylabel = 'from\nORNs\nto'
# # ylabel = {'L': r'ORNs $\rightarrow$ ...', 'R': ''}
# pads = {'L': (0.51, 0.1, 0.45, 0.2),
#         'R': (0.1, 0.45, 0.45, 0.2)}  # l, r, b, t
# show_ylabel = {'L':  True, 'R': False}
# title = {'L': 'left', 'R': 'right'}
# for side in ['L', 'R']:
#     # Correlation coefficient per category
#     LNs_sel1 = LNs_sel_d_side[side]
#
#     con_ff_sel = con_strms3.loc[:, strm]
#     con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
#     con_ff_sel.columns = LNs_sel_short
#     con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
#     grammian = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
#
#     _, fs, axs = FP.calc_fs_ax_df(grammian, pads[side], sq=SQ)
#     f = plt.figure(figsize=fs)
#     ax1 = f.add_axes(axs)
#     cp = FP.imshow_df2(grammian, ax1, cmap=plt.cm.bwr, vlim=1,
#                        show_lab_x=True, show_lab_y=show_ylabel[side])
#     # splits_x=[3, 5, 7], splits_y=[3, 5, 7])
#     ax1.set_title(f'W data, {title[side]}')
#     ax1.set_xticks(np.arange(len(grammian.T)) + 0.5)
#     ax1.set_xticklabels(list(grammian.columns), rotation=70, ha='right')
#     if side == 'L':
#         ax1.set_ylabel(ylabel, rotation=0, fontsize=ft_s_tk, labelpad=3,
#                        va='center', ha='right')
#     ax1.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
#                    rotation_mode='default', ha='center', va='top')
#     ax1.annotate('', xy=(-2, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#                  arrowprops={'arrowstyle': '-', 'lw': 0.5},
#                  annotation_clip=False)
#     ax1.annotate('', xy=(-4.1, -1), xytext=(-4.1, 8), xycoords='data',
#                  arrowprops={'arrowstyle': '-', 'lw': 0.5},
#                  annotation_clip=False)
#     # ax1.xaxis.set_label_coords(0.5, -1.2)
#     # ax1.yaxis.set_label_coords(-1.2, 0.5)
#
#     ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0],
#                         axs[3]])
#     FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
#     file = (f'{PP_CONN}/{CELL_TYPE}_con{strm}{side}_cn_grammian')
#     FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
#     FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# left and right side separately, but on same figure
d_h = 0.15
strm = 0
# xlabel = 'from ORNs to'
xlabel = r'$\mathbf{w}_\mathrm{LN}$'
# ylabel = 'from\nORNs\nto'
ylabel = xlabel
pads = [0.40, 0.45, 0.37, 0.35]  # l, r, b, t
title = {'L': 'left', 'R': 'right'}
side = 'L'
# Correlation coefficient per category
LNs_sel1 = LNs_sel_d_side[side]

con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df1 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
side = 'R'
# Correlation coefficient per category
LNs_sel1 = LNs_sel_d_side[side]

con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df2 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
d_x = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts_side(df1, df2, pads, d_h, SQ,
                                                  CB_DX, CB_W)

f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df2(df1, ax1, cmap=plt.cm.bwr, vlim=1,
                   show_lab_x=True, show_lab_y=True)
ax1.set_title('left side', pad=2, fontsize=ft_s_lb)
ax1.set_xticks(np.arange(len(df1.T)) + 0.5)
ax1.set_xticklabels(list(df1.columns), rotation=70, ha='right')
ax1.set_ylabel(ylabel)
# ax1.set_ylabel(ylabel, rotation=0, fontsize=ft_s_tk, labelpad=3,
#                va='center', ha='right')
# ax1.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
#                rotation_mode='default', ha='center', va='top')
# ax1.annotate('', xy=(-4.1, -1), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

cp = FP.imshow_df2(df2, ax2, cmap=plt.cm.bwr, vlim=1,
                   show_lab_x=True, show_lab_y=False)
ax2.set_title('right side', pad=2, fontsize=ft_s_lb)
ax2.set_xticks(np.arange(len(df2.T)) + 0.5)
ax2.set_xticklabels(list(df2.columns), rotation=70, ha='right')
# ax2.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
#                rotation_mode='default', ha='center', va='top')
# ax2.annotate('', xy=(-12, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
f.text(0.5, 0., xlabel, rotation=0, fontsize=ft_s_lb, va='bottom',
       ha='center')
# ax1.xaxis.set_label_coords(0.5, -1.2)
# ax1.yaxis.set_label_coords(-1.2, 0.5)

FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
plt.suptitle(r'corr. among $\mathbf{w}_\mathrm{LN}}$ on each side')
file = f'{PP_CONN}/{CELL_TYPE}_con{strm}LR_cn_grammian'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
#%%
# #############################################################################
# #############################################################################
# not used in paper
# plot above with angle instead of correlation coefficient...:
#
# # left and right side separately, but on same figure
# d_h = 0.15
# strm = 0
# # xlabel = 'from ORNs to'
# xlabel = r'$\mathbf{w}_\mathrm{LN}$'
# # ylabel = 'from\nORNs\nto'
# ylabel = xlabel
# pads = [0.40, 0.45, 0.37, 0.35]  # l, r, b, t
# title = {'L': 'left', 'R': 'right'}
# side = 'L'
# # Correlation coefficient per category
# LNs_sel1 = LNs_sel_d_side[side]
#
# con_ff_sel = con_strms3.loc[:, strm]
# con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
# con_ff_sel.columns = LNs_sel_short
# # con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
# con_ff_sel_n = FG.get_norm(con_ff_sel)
# # df1 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
# df1 = FG.get_cos_sim(con_ff_sel_n, con_ff_sel_n)
# # this makes sure that there is no numerical problem with any
# # number larger than 1.
# df1[:] = np.minimum.reduce([df1.values, np.ones((len(df1), len(df1)))])
# df1 = np.arccos(df1)
# side = 'R'
# # Correlation coefficient per category
# LNs_sel1 = LNs_sel_d_side[side]
#
# con_ff_sel = con_strms3.loc[:, strm]
# con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
# con_ff_sel.columns = LNs_sel_short
# # con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
# con_ff_sel_n = FG.get_norm(con_ff_sel)
# # df2 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
# df2 = FG.get_cos_sim(con_ff_sel_n, con_ff_sel_n)
# df2[:] = np.minimum.reduce([df2.values, np.ones((len(df2), len(df2)))])
# df2 = np.arccos(df2)
# d_x = 0.15  # delta height between the 2 imshows
# fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts_side(df1, df2, pads, d_h, SQ,
#                                                   CB_DX, CB_W)
#
# f = plt.figure(figsize=fs)
# ax1 = f.add_axes(axs1)
# ax2 = f.add_axes(axs2)
# ax_cb = f.add_axes(axs_cb)
#
# cp = FP.imshow_df2(df1, ax1, cmap=plt.cm.magma, vlim=[0,np.pi/2],
#                    show_lab_x=True, show_lab_y=True)
# ax1.set_title('left side', pad=2, fontsize=ft_s_lb)
# ax1.set_xticks(np.arange(len(df1.T)) + 0.5)
# ax1.set_xticklabels(list(df1.columns), rotation=70, ha='right')
# ax1.set_ylabel(ylabel)
#
# cp = FP.imshow_df2(df2, ax2, cmap=plt.cm.magma, vlim=[0,np.pi/2],
#                    show_lab_x=True, show_lab_y=False)
# ax2.set_title('right side', pad=2, fontsize=ft_s_lb)
# ax2.set_xticks(np.arange(len(df2.T)) + 0.5)
# ax2.set_xticklabels(list(df2.columns), rotation=70, ha='right')
#
# f.text(0.5, 0., xlabel, rotation=0, fontsize=ft_s_lb, va='bottom',
#        ha='center')
#
#
# FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
# plt.suptitle(r'corr. among $\mathbf{w}_\mathrm{LN}}$ on each side')
# file = f'{PP_CONN}/{CELL_TYPE}_con{strm}LR_cn_grammian_CS'
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
#
# print(np.mean(FG.get_entries(df1, diag=False)))
# print(np.mean(FG.get_entries(df2, diag=False)))

# %%
# #############################################################################
# ######################  COMPARISON M WITH WWT  ##############################
# #############################################################################

# ###########################          M         ##############################
# this is pretty much the same as the M plot above, just with Picky merged
# into 1 cell and changed padding at the bottom and at the top

s = 'L'
df1 = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
df2 = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
df2.columns.name = 'Postsynaptic LNs'


pads = [0.4, 0.4, 0.41, 0.45]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df2(df1, ax1, vlim=[0, 110], show_lab_x=False)
ax1.set_title('left side', pad=2, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df2(df2, ax2, vlim=[0, 110])
ax2.set_xticks(np.arange(len(df2.T)) + 0.5)
ax2.set_xticklabels(list(df2.columns), rotation=70, ha='right')
ax2.set_title('right side', pad=2, fontsize=ft_s_lb)

# y label
f.text(0.01, 0.55, 'Presynaptic LNs', rotation=90,
       fontsize=matplotlib.rcParams['axes.labelsize'], va='center', ha='left')

clb = FP.add_colorbar(cp, ax_cb, '', [0, 50, 100])
clb.ax.set_title('# syn.', pad=2, fontsize=ft_s_tk)
plt.suptitle("LN-LN connections\n synaptic counts " + r"$\mathbf{M}$")
file = f'{PP_CONN}/con_M_a-d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


# %%
# ###########################          WTW       ##############################
# this is pretty much the same as the W plot above, just with Picky dend and
# axon merged into 1 cell and changed padding at the bottom and at the top

xylabel = r'$\mathbf{w}_\mathrm{LN}$'

s = 'L'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df1 = con_ff_sel.T.dot(con_ff_sel)
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df2 = con_ff_sel.T.dot(con_ff_sel)
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
# df2.columns.name = r'ORNs$\rightarrow$...'

# df1 = df1 - np.diag(np.diag(df1))
# df2 = df2 - np.diag(np.diag(df2))

print(np.max(df1.values), np.max(df2.values))

pads = [0.5, 0.5, 0.41, 0.45]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

FP.imshow_df2(df1/1000, ax1, vlim=[0, 15], show_lab_x=False)
ax1.set_title('left side', pad=2, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df2(df2/1000, ax2, vlim=[0, 15])
ax2.set_xticks(np.arange(len(df2.T)) + 0.5)
ax2.set_xticklabels(list(df2.columns), rotation=70, ha='right')
ax2.set_title('right side', pad=2, fontsize=ft_s_lb)
ax2.set_xlabel(xylabel)
# ax2.annotate('', xy=(-2, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax2.annotate('', xy=(-4.1, -11), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

f.text(0.1, 0.5, xylabel, rotation=90, fontsize=ft_s_lb,
       va='center',ha='right')
# f.text(0.15, 0.55, 'from\nORNs\nto', rotation=0, fontsize=ft_s_tk, va='center',
#        ha='right')

clb = FP.add_colorbar(cp, ax_cb, '', [0, 10])
clb.set_ticklabels([0, r'$10^4$'])
clb.ax.set_title(r'(# syn.)$^2$', pad=2, fontsize=ft_s_tk)
# clb.ax.set_title('1e3', pad=2, fontsize=ft_s_tk)
plt.suptitle(r'ORNs$\rightarrow$LN dot products' + '\n'
             r'$\mathbf{W}^\mathrm{\top}\mathbf{W} = $'+
             r'$\{\mathbf{w}_\mathrm{LNi}^\mathrm{\top}\mathbf{w}_\mathrm{LNj}\}$')

file = f'{PP_CONN}/con_WtW_d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


#
#
#
# %%
# #########################        sqrt(WTW)       ############################
# this is pretty much the same as the W plot above, just with Picky dend and
# axon merged into 1 cell and changed padding at the bottom and at the top

xylabel = r'$\mathbf{w}_\mathrm{LN}$'

s = 'L'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df1 = con_ff_sel.T.dot(con_ff_sel)
df1[:] = LA.sqrtm(df1.values)
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df2 = con_ff_sel.T.dot(con_ff_sel)
df2[:] = LA.sqrtm(df2.values)
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
# df2.columns.name = r'ORNs$\rightarrow$...'

# df1 = df1 - np.diag(np.diag(df1))
# df2 = df2 - np.diag(np.diag(df2))

print(np.max(df1.values), np.max(df2.values))

# pads = [0.5, 0.5, 0.41, 0.45]
pads = [0.5, 0.5, 0.41, 0.32]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df2(df1, ax1, vlim=[0, 90], show_lab_x=False)
ax1.set_title('left side', pad=2, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df2(df2, ax2, vlim=[0, 90])
ax2.set_xticks(np.arange(len(df2.T)) + 0.5)
ax2.set_xticklabels(list(df2.columns), rotation=70, ha='right')
ax2.set_title('right side', pad=2, fontsize=ft_s_lb)
ax2.set_xlabel(xylabel)
# ax2.annotate('', xy=(-2, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax2.annotate('', xy=(-4.1, -11), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

f.text(0.1, 0.5, xylabel, rotation=90, fontsize=ft_s_lb, va='center',
       ha='right')

clb = FP.add_colorbar(cp, ax_cb, '', [0, 40, 80])
clb.ax.set_title('# syn.', pad=2, fontsize=ft_s_tk)
plt.suptitle(r'$(\mathbf{W}^\mathrm{\top} \mathbf{W})^{1/2}$')

file = f'{PP_CONN}/con_sqrtWtW_d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


# %%
# #####################  SCATTER  PLOTS AND    ################################
# #####################     SIGNIFICANCE       ################################

# this is not used in the paper
#
# scatter plot between WTW and M^2
#
# importlib.reload(FP)
#
# power = 1
#
# s = 'L'
# ML = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].values
# ML2 = ML @ ML
# WL = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
# WTW_L = WL.T @ WL
# s = 'R'
# MR = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].values
# MR2 = MR @ MR
# WR = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
# WTW_R = WR.T @ WR
#
# diag = False
# W_entries = np.concatenate([FG.get_entries(WTW_L, diag=diag),
#                             FG.get_entries(WTW_R, diag=diag)])
#
# M_entries = np.concatenate([FG.get_entries(ML2, diag=diag),
#                             FG.get_entries(MR2, diag=diag)])
# n_pts = int(len(M_entries)/2)
#
# cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
# print(cc_real)  # 0.75
#
# # the only thing that is important that it matches is the bottom in order
# # to compare with the M and WTW plots
# # b = 0.41
# b = 0.35
# pads = (0.4, 0.1, b, 0.15)
# # then, what about the actual size of the graph?
#
# fs, ax1 = FP.calc_fs_ax(pads, 15*SQ, 15*SQ)
# f = plt.figure(figsize=fs)
# ax = f.add_axes(ax1)
# ax.scatter(M_entries[:n_pts]/1000, W_entries[:n_pts]/1000, c='indigo', s=5,
#            label='left', alpha=0.7, lw=0)
# FP.plot_scatter(ax, M_entries[n_pts:]/1000, W_entries[n_pts:]/1000,
#                 r'$\mathbf{M}^2$ entries ($10^3$)',
#                 r'$\mathbf{W}^\top \mathbf{W}$'
#                 r' entries ($10^3$)', pca_line_scale1=0.15,
#                 pca_line_scale2=0.65, show_cc=False, s=5, c='teal',
#                 label='right', alpha=0.7, lw=0)
# ax.legend(frameon=False, borderpad=0, handletextpad=0, loc='upper left',
#           bbox_to_anchor=(-0.07, 1.04))
# ax.text(0.65, 0.11, r'$r$' + " = %0.2f" % cc_real, transform=ax.transAxes)
# ax.text(0.65, 0.03, "pv = %0.0e" % 0.01, transform=ax.transAxes)
# # popt, _ = curve_fit()
# ax.set(ylim=(0, None), xlim=(-.5, None))
#
# file = f'{PP_CONN}/con_WtW_M2_scatter.'
# FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# significance is calculated in the file con_analysis_MvsW2.py

# %%
# scatter plot between sqrt(WTW) and M

importlib.reload(FP)

power = 1

s = 'L'
ML = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()**power
WL = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
WTW_L: np.ndarray = LA.sqrtm(WL.T @ WL)
s = 'R'
MR = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()**power
WR = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
WTW_R: np.ndarray = LA.sqrtm(WR.T @ WR)

diag = False
W_entries = np.concatenate([FG.get_entries(WTW_L, diag=diag),
                            FG.get_entries(WTW_R, diag=diag)])

M_entries = np.concatenate([FG.get_entries(ML, diag=diag),
                            FG.get_entries(MR, diag=diag)])
n_pts = int(len(M_entries)/2)

cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
print(cc_real)  # 0.73

# the only thing that is important that it matches is the bottom in order
# to compare with the M and WTW plots
b = 0.35
pads = (0.4, 0.1, b, 0.15)
# then, what about the actual size of the graph?

fs, ax1 = FP.calc_fs_ax(pads, 15*SQ, 15*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(ax1)
ax.scatter(M_entries[:n_pts], W_entries[:n_pts], c='indigo', s=5,
           label='left', alpha=0.7, lw=0)
FP.plot_scatter(ax, M_entries[n_pts:], W_entries[n_pts:],
                r'$\mathbf{M}$ entries',
                r'$(\mathbf{W}^\mathrm{\top}'
                r' \mathbf{W})^{1/2}$ entries',
                pca_line_scale1=0.18,
                pca_line_scale2=0.68, show_cc=False, s=5, c='teal',
                label='right', alpha=0.7, lw=0)
ax.legend(frameon=False, borderpad=0, handletextpad=0, loc='upper left',
          bbox_to_anchor=(-0.07, 1.04))
ax.text(0.65, 0.11, r'$r$' + " = %0.2f" % cc_real, transform=ax.transAxes)
ax.text(0.65, 0.03, "pv = %0.0e" % 0.006, transform=ax.transAxes)
ax.set(ylim=(0, None), xlim=(-5, None))

file = f'{PP_CONN}/con_sqrtWtW_M_scatter'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# significance is calculated in the file con_analysis_MvsW2.py

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
# ##################  END OF CONNECTIVITY PLOTS  #############################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
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
idx = [(o, c) for o, c in itertools.product(odor_order, [8, 7, 6, 5, 4])]
df = act_m.loc[ORN_order, idx]
ORN_list = [name[4:] for name in df.index]
df.index = ORN_list


title = f'ORN soma activity patterns {Xdatatex}'
cb_title = r'$\Delta F/F$'
cb_ticks = [-1, 0, 2, 4, 6]
# cb_ticks = [-1, -0.1, 0, 0.1, 1, 8]  # for the log scale
pads = [0.55, 0.4, 1.32, 0.2]
f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title, cb_ticks,
                              pads=pads)
# ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title, cb_ticks)
# ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')
ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act_nolabels.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


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
                              extend='neither')
ax.set(xticks=[], ylabel='ORNs', xlabel='odors at different dilutions')

file = f'{PP_ACT}/ORN_act_scaled_max.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
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
FP.add_colorbar(cp, ax_cb, '', [-0.8, -0.4, 0, 0.4, 0.8])

file = f'{PP_ACT}/ORN_act_SVD_U'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
# %%

print(perc_var_explained)
print(np.cumsum(perc_var_explained))


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
    FP.add_colorbar(cp, ax_cb, '', [0, 0.5])

    file = f'{PP_CON_PRED}/ORN_act-{pps}_NNC{k}_W'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


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
LN = f'Broad T M M'
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
ax2.set_ylabel(r'ORN $\Delta$F/F', color=c2)
ax.set_ylabel(r'# of syn. ORNs$\rightarrow$BT', color=c1)
labs = ['# syn.', 'odor A', 'odor B']
leg = ax.legend(lns, labs, ncol=3, loc=10,
                bbox_to_anchor=(0.1, 1.01, .8, 0.1), frameon=False,
                handletextpad=0.5, columnspacing=1)
leg.get_frame().set_linewidth(0.0)

file = f'{PP_ODOR_CON}/{LN}_2odors-{conc}_2axplot'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
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
    FP.plot_scatter(ax, con_w.values, act_vect.values, '# of synapses',
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

# %%
# #############################################################################
# #############################################################################
# #############################################################################
# plotting the same as above but with using PCA

def plot_2lines(LN, LN_label, vect, vect_label, ylims, file_label,
                yticks1=[0, 20, 40], yticks2=[0, 0.5], PP=PATH_PLOTS):
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
    ax.set_ylabel(r'# of syn. ORNs$\rightarrow$' + LN_label, color=c1)
    labs = ['# syn.', vect_label]
    leg = ax.legend(lns, labs, ncol=3, loc=10,
                    bbox_to_anchor=(0.1, 1.01, .8, 0.1), frameon=False,
                    handletextpad=0.5, columnspacing=1)
    leg.get_frame().set_linewidth(0.0)
    file = f'{PP}/{LN}_{file_label}_2axplot'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


LN = f'Broad T M M'
SVD = FG.get_svd_df(act_m)
act_vect = SVD['U'].loc[:, 1]
ORN_order = par_act.ORN_order
act_vect = -act_vect.loc[ORN_order]
ylims = [(-2, 45), (-0.02, 0.55)]
plot_2lines(LN, 'BT       ', act_vect, 'PCA 1', ylims, 'PCA1', PP=PP_COMP_CON)

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
                r'# of syn. ORNs$\rightarrow$BT',
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
    order = [0, 3, 2, 1]
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
order = [0, 2, 1]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          frameon=False, bbox_to_anchor=(-0.04, 1.1), loc='upper left',
          handlelength=1, handletextpad=0.4)

ax.text(0.4, 0.025, f'LN type: {LN_m}', transform=ax.transAxes)

file = f'{PP_ODOR_CON}/cors_{LN}_rcf-m-std1_nofit'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


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
    order = [0, 3, 2, 1]
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
cp = FP.imshow_df2(df, ax, vlim=[-1, 1], cmap=plt.cm.bwr, splits_x=splx,
                   splits_y=sply, aspect='auto', splits_c='gray', lw=0.5)
ax.set(xticks=np.arange(2, len(idx), 5), xticklabels=odor_order,
       xlabel=f'ORN activation patterns {Xttex} to odors at different dilutions',
       ylabel=wLNtex)
       # ylabel = r'data, ' + wLNtex)

plt.title(f'Correlation between ORN activity patterns {Xttex} '
          r'and ORNs$\rightarrow$LN connection weight vectors '
          r'$\mathbf{w}_{\mathrm{LN}}$')
ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/ORN_con0_vs_act3_raw_mean_all-odors_corr'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# just for LN M
# calculation of the correlation coefficients
wLNtex = r'$\mathbf{w}_{\mathregular{LNtype}}$'

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
cp = FP.imshow_df2(df, ax, vlim=[-1, 1], cmap=plt.cm.bwr, splits_x=splx,
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
FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])

file = f'{PP_ODOR_CON}/ORN_con0-LN-M_vs_act3_raw_mean_all-odors_corr'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
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
ax.set_xticks([])
ax.set_ylabel(r'corr. coef. $r$', labelpad=-2)
ax.set_xlabel('ordered stimuli')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

file = f'{PP_ODOR_CON}/LN_tuning_curves.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

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
                                    'magenta', 'b', 'RCF max deviation (  )',
                                    r'$-log_{10}$(p-value) (  )',
                                    ylim1=(0, 0.37), ylim2=(0, 2.1))
# axx[1].annotate('', xy=(-2, -1.5), xytext=(18.5, -1.5), xycoords='data',
#                 arrowprops={'arrowstyle': '-', 'lw': 0.5},
#                 annotation_clip=False)
axx[1].plot([20.85], [1.86], ls='None', marker='+', color='b', markersize=5,
            clip_on=False)
axx[0].plot([-5.7], [0.356], ls='None', marker='.', color='magenta',
            markersize=5, clip_on=False)
axx[0].set_yticks([0, 0.1, 0.2, 0.3])
axx[1].set_yticks([0, 1, 2])
axx[1].set_xlabel(r'$\mathbf{w}_\mathrm{LN}$')
# f.text(0.5, 0., 'conn. weight vectors from ORNs to',
#    fontsize=matplotlib.rcParams['axes.labelsize'], va='bottom', ha='center')
# axx[1].set_xlabel('ORNs -> LN weight vectors')

# adding the info about the significance
# pvals = cdf_diff_min_pv.loc[LN_idx].values
for alpha, y, sign in [[0.05, 2.2, '*']]:
    FP.add_sign_stars(axx[1], pvals, alpha, x, y, sign)

file = (f'{PP_ODOR_CON}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_corr_rcf_min_diff')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

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
                  r'RCF max deviation (  )', r'$-log_{10}$(p-value) (  )',
                   ylim1=(0, 0.37), ylim2=(0, 2.5))
axx[1].plot([8.75], [2.28], ls='None', marker='+', color='b', markersize=5,
            clip_on=False)
axx[0].plot([-5.5], [0.37], ls='None', marker='.', color='magenta',
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


for alpha, y, sign in [[0.05, 2.6, '*']]:
    FP.add_sign_stars(axx[1], pvals, alpha, x, y, sign)

file = (f'{PP_ODOR_CON}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_corr_rcf_min_diff_sum2')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
#
#
#
#
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

    FP.add_colorbar(cp, ax_cb, r'$r$', [-1 , 0, 1])
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

    clb = FP.add_colorbar(cp, ax_cb, 'pv', [-3, -2, -1], extend='both')
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

    f1, f2 = plot_pv_measure(df1, df2, splx[k], plt.cm.bwr,
                             [-1, 1], i)

    for (m, n), label in np.ndenumerate(stars):
        f1[1].text(n, m+0.1, label, ha='center', va='center',
                   size=matplotlib.rcParams['font.size']*0.8**0)

    FP.save_plot(f1[0], f'{file}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f1[0], f'{file}.pdf', SAVE_PLOTS, **pdf_opts)
    # FP.save_plot(f2[0], f'{file}_pv_adjusted.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f2[0], f'{file}_pv.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f2[0], f'{file}_pv.pdf', SAVE_PLOTS, **pdf_opts)

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
ax.set_xlabel(r'$\rho$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(ncol=4, frameon=False, columnspacing=1, handlelength=1,
           handletextpad=0.4)

file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-4_rho-range')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

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
ax.set_ylabel('# signif.')
# ax.set_xlabel(r'$\rho$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-4_rho-range_pv')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

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

# %%
#  confirming that the correlation coefficients are still the same as before
print(FG.get_corr(FG.get_ctr_norm_np(Ws), FG.get_ctr_norm_np(con_W)))

print(FG.get_corr(FG.get_ctr_norm_np(act_subspc), FG.get_ctr_norm_np(con_W)))




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
ax.set(xlabel=r'~ # of aligned dimensions $\Gamma$',
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
# %% Theoretical plots showing the relationship between s_x and s_y
# for different rho.
# There is also gamma and T, so we'll need to fix that too.

x = np.linspace(0, 6, 100)
rhos = {0.1: 'C1', 0.2: 'C2', 0.4: 'C3', 1: 'C4', 2: 'C5', 10: 'C6',
        50: 'C7'}


pads = (0.35, 0.25, 0.35, 0.2)
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
ax.set(ylabel=r'$\sigma_Y$', xlabel=r'$\sigma_X$',
       xticks=[0, 3, 6], yticks=[0, 3, 6])

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
pads = (0.5, 0.25, 0.35, 0.2)
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
ax.set(ylabel=r'$\sigma_Y$', xlabel=r'$\sigma_X$')
       # xticks=[0, 3, 6], yticks=[0, 3, 6])


x = np.logspace(-0.8, 0.5, 100)
ax.loglog(x, 10*x, ls='--', lw=1, c='k')
ax.text(0.1, 13, r'$\sigma_Y\propto \sigma_X$')

x = np.logspace(0.2, 2, 100)
ax.loglog(x, 0.02*x**(1/3), ls='--', lw=1, c='k')
ax.text(10, 0.02, r'$\sigma_Y\propto \sigma_X^{1/3}$')
ax.minorticks_off()

file = f'{PP_THEORY}/sy_sx_log.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #############################################################################
# #####################  SHOWING EXAMPLE OF TRANSFORMATION  ###################
# #############################################################################
# from scipy.stats import multivariate_normal
n = 500
x = np.random.randn(2, n)
x[0] = x[0] * 2
x[1] = x[1] * 0.75
rot = SS.ortho_group.rvs(2, random_state=0)
x = rot @ x
# x[0, x[0, :]<0] = 0
# x[1, x[1, :]<0] = 0
# x = x[:, np.sum(x, axis=0)>0]

x = x[:, x[0, :]>0]
x = x[:, x[1, :]>0]
v_max = np.max(x)
v_max = 5
print(x.shape)
c_u1x = 'orange'
alpha = 0.8
n1 = 75
U, s, Vt = LA.svd(x, full_matrices=False)
sigmax = s/np.sqrt(len(x.T))

pads = (0.35, 0.25, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 15 * SQ, 15 * SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.scatter(x[0, :n1], x[1, :n1], c='k', s=0.5, label=f'input {Xttex}')
ax.plot([0, -U[0, 0]*sigmax[0]], [0, -U[0, 1]*sigmax[0]], c=c_u1x, lw=2,
        label=r'$\sigma_{X, 1} \mathbf{u}_1$', alpha=alpha)
ax.plot([0, -U[1, 0]*sigmax[1]], [0, -U[1, 1]*sigmax[1]], c='g',
        label=r'$\sigma_{X, 2} \mathbf{u}_2$', alpha=alpha)
ax.set_xlim(-1, v_max)
ax.set_ylim(-1, v_max)
FP.set_aspect_ratio(ax, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0, 2, 4])
ax.set_yticks([0, 2, 4])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.legend(frameon=False, ncol=2, columnspacing=1., borderpad=0,
          loc='upper right', handletextpad=0.4, handlelength=1,
          bbox_to_anchor=(1.15, 1.15))
# font_size = 9
# ax.text(0, 3, r'$\sigma_{X, 1} \mathbf{u}_1$', c='r', fontsize=font_size)
# ax.text(0.5, -0.7, r'$\sigma_{X, 2} \mathbf{u}_2$', c='g', fontsize=font_size)

file = f'{PP_THEORY}/example_x.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


# here I am calculating the transformation from x to y with
# the above generated dataset and rho = 1
s_y = s.copy()
s_y[0] = FOC.damp_sx(s[0], len(x.T), 1)
y = U @ np.diag(s_y)@ Vt
sigmay = s_y/np.sqrt(len(x.T))
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.scatter(y[0], y[1], c='k', s=0.5)
ax.plot([0, -U[0, 0]*sigmay[0]], [0, -U[0, 1]*sigmay[0]], c='r',
        label=r'$\sigma_{Y, 1} \mathbf{u}_1$')
ax.plot([0, -U[1, 0]*sigmay[1]], [0, -U[1, 1]*sigmay[1]], c='g',
        label=r'$\sigma_{Y, 2} \mathbf{u}_2$')
ax.set_xlim(-1, v_max)
ax.set_ylim(-1, v_max)
FP.set_aspect_ratio(ax, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0, 2, 4])
ax.set_yticks([0, 2, 4])
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
ax.legend(frameon=False)
# font_size = 9
# ax.text(0, 3, r'$\sigma_{Y, 1} \mathbf{u}_1$', c='r', fontsize=font_size)
# ax.text(0.5, -0.7, r'$\sigma_{Y, 2} \mathbf{u}_2$', c='g', fontsize=font_size)

file = f'{PP_THEORY}/example_y.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


pads = (0.5, 0.25, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 15 * SQ, 15 * SQ)

f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.scatter(x[0, :n1], x[1, :n1], c='gray', s=0.5, alpha=0.2,
           label=f'input {Xttex}')
ax.scatter(y[0, :n1], y[1, :n1], c='k', s=4, marker='+', lw=.6, alpha=0.7,
           label=f'output {Yttex}')
for i in range(n1):
    ax.plot([x[0, i], y[0, i]],[x[1, i], y[1, i]], c='k', lw=0.4, ls='--',
            alpha=0.3)
ax.plot([0, -U[0, 0]*sigmax[0]], [0, -U[0, 1]*sigmax[0]], c=c_u1x,
        label=r'$\sigma_{X, 1} \mathbf{u}_1$', lw=2, alpha=alpha)
ax.plot([0, -U[0, 0]*sigmay[0]], [0, -U[0, 1]*sigmay[0]], c='red',
        label=r'$\sigma_{Y, 1} \mathbf{u}_1$', alpha=alpha)
ax.plot([0, -U[1, 0]*sigmay[1]], [0, -U[1, 1]*sigmay[1]], c='g', alpha=alpha,
        label=r'$\sigma_{X, 2} \mathbf{u}_2=\sigma_{Y, 2} \mathbf{u}_2$')

ax.set_xlim(-1, v_max)
ax.set_ylim(-1, v_max)
FP.set_aspect_ratio(ax, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0, 2, 4])
ax.set_yticks([0, 2, 4])
ax.set_xlabel(r'$x_1, y_1$')
ax.set_ylabel(r'$x_2, y_2$')
ax.legend(frameon=False, ncol=2, columnspacing=-2., borderpad=0,
          loc='upper right', handletextpad=0.4, handlelength=1,
          bbox_to_anchor=(1.15, 1.15))
# font_size = 9
# ax.text(0, 3, r'$\sigma_{Y, 1} \mathbf{u}_1$', c='r', fontsize=font_size)
# ax.text(0.5, -0.7, r'$\sigma_{Y, 2} \mathbf{u}_2$', c='g', fontsize=font_size)

file = f'{PP_THEORY}/example_xy.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)




# %%
# #############################################################################
# #############################################################################
# creating and saving random orthogonal matrices
for k in range(2, 9):
    U_random = SS.ortho_group.rvs(k)
    file = f'results/orthogonal_matrix_{k}.npy'
    np.save(FO.OLF_PATH / file, U_random)
# %%
# #############################################################################
# #############################################################################
# #############################################################################
file = RESULTS_PATH / 'NNC-Y_act-all.hdf'
Ys_nnc = pd.DataFrame(pd.read_hdf(file)).T
file = RESULTS_PATH / 'NNC-Z_act-all.hdf'
Zs_nnc = pd.DataFrame(pd.read_hdf(file)).T
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

sf = 'corr'
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

def get_important_data(Y, Z):
    # problem seems to be here already
    Y = Y.loc[ORN_order].T.reindex(odor_order, level='odor').T
    Y_sel = Y.xs(conc_sel, axis=1, level='conc').copy()
    Z = Z.T.reindex(odor_order, level='odor').T
    Z_sel = Z.xs(conc_sel, axis=1, level='conc').copy()
    W = Y @ Z.T / N
    M = Z @ Z.T / N
    return Y, Y_sel, Z, Z_sel, W, M


conc_sel = 4  # this is just for the plooting
act = FO.get_ORN_act_data(DATASET).T
act_m = act.mean(axis=1, level=('odor', 'conc'))
X = act_m.loc[ORN_order].T.reindex(odor_order, level='odor').T

# preparation for looking at the correlation outside of the same odor
block = np.ones((5, 5))
n_odors = int(X.shape[1]/5)
big_block = LA.block_diag(*[block for _ in range(n_odors)]) - np.eye(n_odors*5)
big_block = big_block.astype(bool)



U_X, s_X, Vt_X = LA.svd(X, full_matrices=False)
X_sel = X.xs(conc_sel, axis=1, level='conc').copy()
N = X.shape[1]
X_cc_c, X_cc_p, X_cc_p_sel, X_cc_p_o = get_cc_data(X, X_sel)


conc = 'all'
# ###################### IMPORTANT VARIABLE #################################
SCAL_W = 10  # in the paper for supplementary materials
# SCAL_W = 2  # in the paper for the main results
# SCAL_W = 0.35
pps = f'{SCAL_W}o'

if SAVE_PLOTS:
    PP_WHITE = PATH_PLOTS / f'whitening{SCAL_W}'
    PP_WHITE.mkdir(exist_ok=True)

Y_nnc = {}
Y_nnc_sel = {}
Z_nnc = {}
Z_nnc_sel = {}
W_nnc = {}
M_nnc = {}
M_nnc_diag = {}
Y_nnc_cc_c = {}  # channels
Y_nnc_cc_p = {}  # pattens
Y_nnc_cc_p_sel = {}  # pattens
Y_nnc_cc_p_o = {}  # patterns outside same odor

Y_lc = {}
Y_lc_sel = {}
Z_lc = {}
Z_lc2 = {}  # this is the activity without multiplying by a random
# orthogonal matrix on the left
Z_lc_sel = {}
W_lc = {}
M_lc = {}
M_lc_diag = {}
s_Y_lc = {}  # singular values of LC
Y_lc_cc_c = {}
Y_lc_cc_p = {}
Y_lc_cc_p_sel = {}
Y_lc_cc_p_o = {}

Y_lc_noM = {}
Y_lc_noM_sel = {}
Z_lc_noM = {}
W_lc_noM = {}
M_lc_noM = {}
Y_lc_noM_cc_c = {}
Y_lc_noM_cc_p = {}
Y_lc_noM_cc_p_sel = {}

Y_nnc_noM = {}
Y_nnc_noM_sel = {}
Z_nnc_noM = {}
W_nnc_noM = {}
M_nnc_noM = {}
Y_nnc_noM_cc_c = {}
Y_nnc_noM_cc_p = {}
Y_nnc_noM_cc_p_sel = {}

U_random = {} ##############  check if you want real random below #########

for k in range(1, 9):
    meth = f'NNC_{k}'
    Y_nnc[k] = Ys_nnc.loc['all', pps, meth, '', '']
    Z_nnc[k] = Zs_nnc.loc['all', pps, meth, '', '']
    res = get_important_data(Y_nnc[k], Z_nnc[k])
    Y_nnc[k], Y_nnc_sel[k], Z_nnc[k], Z_nnc_sel[k], W_nnc[k], M_nnc[k] = res
    res = get_cc_data(Y_nnc[k], Y_nnc_sel[k])
    Y_nnc_cc_c[k], Y_nnc_cc_p[k], Y_nnc_cc_p_sel[k], Y_nnc_cc_p_o[k] = res

    s_Y_lc[k] = s_X.copy()
    s_Y_lc[k][:k] = FOC.damp_sx(s_Y_lc[k][:k], N, SCAL_W)
    Y_lc[k] = X.copy()
    Y_lc[k][:] = U_X @ np.diag(s_Y_lc[k]) @ Vt_X
    if k >= 2:
        U_random[k] = np.load(RESULTS_PATH / f'orthogonal_matrix_{k}.npy')
        # U_random[k] = np.eye(k)  # this removes the random
    else:
        U_random[k] = [[1]]  # can't rotate a scalar...

    Z_lc[k] = Z_nnc[k].copy()
    Z_lc2[k] = Z_nnc[k].copy()
    Z_lc2[k][:] = np.diag(s_Y_lc[k][:k]) @ Vt_X[:k]
    Z_lc[k][:] = U_random[k] @ Z_lc2[k]
    res = get_important_data(Y_lc[k], Z_lc[k])
    Y_lc[k], Y_lc_sel[k], Z_lc[k], Z_lc_sel[k], W_lc[k], M_lc[k] = res

    # Y_lc_sel[k] = Y_lc[k].xs(conc_sel, axis=1, level='conc').copy()
    res = get_cc_data(Y_lc[k], Y_lc_sel[k])
    Y_lc_cc_c[k], Y_lc_cc_p[k], Y_lc_cc_p_sel[k], Y_lc_cc_p_o[k] = res
    #
    # W_lc[k] = 1 / N * Y_lc[k] @ Z_lc[k].T
    # M_lc[k] = 1 / N * Z_lc[k] @ Z_lc[k].T
    #
    M_lc_diag[k] = np.diag(np.diag(M_lc[k]))
    M_lc_noM[k] = M_lc_diag[k].copy()
    W_lc_noM[k] = W_lc[k].copy()
    Y_lc_noM[k] = Y_lc[k].copy()
    Z_lc_noM[k] = Z_lc[k].copy()
    #
    Y_lc_noM[k][:], Z_lc_noM[k][:] = FCS.olf_output_offline(X.values,
                            W_lc_noM[k].values, W_lc_noM[k].values,
                            M_lc_noM[k], SCAL_W, method='inv')

    res = get_important_data(Y_lc_noM[k], Z_lc_noM[k])
    Y_lc_noM[k], Y_lc_noM_sel[k], Z_lc_noM[k], _, _, _ = res
    res = get_cc_data(Y_lc_noM[k], Y_lc_noM_sel[k])
    Y_lc_noM_cc_c[k], Y_lc_noM_cc_p[k], Y_lc_noM_cc_p_sel[k], _ = res

    M_nnc_diag[k] = np.diag(np.diag(M_nnc[k]))
    M_nnc_noM[k] = M_nnc_diag[k].copy()
    W_nnc_noM[k] = W_nnc[k].copy()
    Y_nnc_noM[k] = Y_nnc[k].copy()
    Z_nnc_noM[k] = Z_nnc[k].copy()
    #
    res = FCS.olf_output_offline(X.values,
                                 W_nnc_noM[k].values, W_nnc_noM[k].values,
                                 M_nnc_noM[k], SCAL_W, method='GD_NN')
    Y_nnc_noM[k][:], Z_nnc_noM[k][:] = res

    res = get_important_data(Y_nnc_noM[k], Z_nnc_noM[k])
    Y_nnc_noM[k], Y_nnc_noM_sel[k], Z_nnc_noM[k], _, _, _ = res
    res = get_cc_data(Y_nnc_noM[k], Y_nnc_noM_sel[k])
    Y_nnc_noM_cc_c[k], Y_nnc_noM_cc_p[k], Y_nnc_noM_cc_p_sel[k], _ = res


# c_nn1 = 'darkred'
# c_nn1 = 'saddlebrown'
c_nn1 = 'sienna'
c_nn2 = 'tomato'
# c_nn2 = 'coral'
c_l1 = 'blueviolet'
c_l2 = 'dodgerblue'

# ideally the results of noM should be saved, so that they are not
# calculated each time, which makes you have to redo all the plots again


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

    pads = (0.2, 0.1, 0.2, 0.2)
    if cond:
        pads = (0.2, 0.4, 0.2, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(toplot, ax, vlim=[-1, 1], show_lab_y=False,
                       title=title, show_lab_x=False,
                       cmap=plt.cm.bwr)
    for spine in ax.spines.values():
        spine.set_visible(True)
    [i.set_linewidth(0.5) for i in ax.spines.values()]
    ax.set_xlabel(r'$\mathbf{U}_X[:, i]$')
    ax.set_ylabel(r'$\mathbf{U}_Y[:, j]$')
    # ax.set_yticks(np.arange(len(toplot)))
    if cond:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = FP.add_colorbar(cp, ax_cb, '', [-1, 0, 1])

    file = f'{PP_ROT}/ORN_act-{pps}_{name}_LSV'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


k1 = 1
k2 = 8
title = r', $\mathbf{U}_Y^\top \cdot \mathbf{U}_X$'
datas = [[X, Y_lc[k1], f'LC-{k1}' + title, f'X-YLC{k1}', False],
         [X, Y_lc[k2], f'LC-{k2}' + title, f'X-YLC{k2}', False],
         [X, Y_lc_noM[k2], f"LC'-{k2}" + title, f'X-YLCnoM{k2}', False],
         [X, Y_nnc[k1], f"NNC-{k1}" + title, f'X-YNNC{k1}', False],
         [X, Y_nnc[k2], f"NNC-{k2}" + title, f'X-YNNC{k2}', True],
         [X, Y_nnc_noM[k2], f"NNC'-{k2}" + title, f'X-YNNCnoM{k2}', True]]

if SCAL_W == 2:
    for data in datas:
        plot_XYtrans(data)
# %%
# #############################################################################
# ######################  change in the X-Y cross correlation #################
# #####################  with the absence of M  ###############################
# #############################################################################


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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('ORNs')
    ax.set_ylabel(r'corr. coef. $r$')
    ax.set_yticks([0.8, 0.9, 1])
    ax.set_xticks(np.arange(data[0].shape[1]))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False, pad=-2)
    ax.xaxis.grid()

    ax.legend(ncol=3, loc=10,
              bbox_to_anchor=(0.1, 1.01, .8, 0.1), frameon=False,
              handletextpad=0.5, columnspacing=1)

    file =  f'{PP_ROT}/ORN_act-{pps}_{name}_crosscorr_diag'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


k1 = 1
k2 = 8
title = r', corr($\mathbf{X}$, $\mathbf{Y}$)'
# datas = [[X, [Y_lc[k1], Y_lc[k2], Y_lc_noM[k2]], [c_l1, c_l2, c_l2],
#           ['-', '-', '--'], [f'LC-{k1}', f'LC-{k2}', f"LC'-{k2}"],
#           title + f', LC-{k1}', f'X-YLC'],
#          [X, [Y_nnc[k1], Y_nnc[k2], Y_nnc_noM[k2]], [c_nn1, c_nn2, c_nn2],
#           ['-', '-', '--'], [f'NNC-{k1}', f'NNC-{k2}', f"NNC'-{k2}"],
#           title + f", NNC-{k1}", f'X-YNNC']]
datas = [[X, [Y_lc[k2], Y_lc_noM[k2]], [c_l2, c_l2],
          ['-', '--'], [f'LC-{k2}', f"LC'-{k2}"],
          f'LC-{k1}' + title, f'X-YLC'],
         [X, [Y_nnc[k2], Y_nnc_noM[k2]], [c_nn2, c_nn2],
          ['-', '--'], [f'NNC-{k2}', f"NNC'-{k2}"],
          f", NNC-{k1}" + title, f'X-YNNC']]

if SCAL_W == 2:
    for data in datas:
        plot_XYcrosscorr_diag(data)



# %%
# ################## activity plots for different dampening  ##################

if SCAL_W == 10:
    v_max = {'X': 7, 'LC8': 4, 'NNC8': 2}


datas = [['X', X_sel.copy(), False, False, True,
          7, f'input activity {Xdatatex}' + r' (dil. $10^{-4}$)   '],
         ['LC8', Y_lc_sel[8].copy(), False, False, True, 4,
          f'LC-8, output activity {Ytex}'],
         ['NNC8', Y_nnc_sel[8].copy(), False, False, True, 4,
          f'NNC-8, output activity {Ytex}'],
#         ['LC-noM8', Y_lc_noM_sel[8].copy(), False, False, True, 4,
#          f"LC'-8, output activity {Ytex}"],
#         ['NNC-noM8', Y_nnc_noM_sel[8].copy(), False, False, True, 4,
#          f"NNC'-8, output activity {Ytex}"]
         ]

# odor_order = par_act.odor_order_o  # if we want the original order
odor_order = par_act.odor_order
# ORN_order = par_act.ORN  # this is just the alphabetical order
ORN_order = par_act.ORN_order

# colors
vmin_col = -2
vmin_to_show = -2
vmax = 6
act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)

pads = [0.2, 0.4, 0.2, 0.2]
for data in datas:
    name = data[0]
    act_sel = data[1]
    show_x = data[2]
    show_y = data[3]
    cb = data[4]
    v_max = data[5]
    title = data[6]
    # preparing the data:
    act_sel = act_sel.loc[ORN_order, odor_order]
    print('min', np.min(act_sel.values))
    print('max', np.max(act_sel.values))
    # removing 'ORN' from the cell names
    ORN_list = [name[4:] for name in act_sel.index]
    act_sel.index = ORN_list
    act_sel.columns.name = 'odors'
    act_sel.index.name = 'ORNs'
    # plotting
    _pads = pads.copy()
    if not cb:
        _pads[1] = 0.01
    # if ds != 'X':
    #     _pads[0] = 0.01
    df = act_sel
    _, fs, axs = FP.calc_fs_ax_df(df, _pads, sq=SQ*0.7)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(df, ax, vlim=[vmin_to_show, v_max], show_lab_y=show_y,
                       title=title, show_lab_x=show_x, cmap=act_map,
                       **{'norm': divnorm})
    ax.set_xlabel('odors')
    # if ds == 'X':
    #     ax.set_ylabel('ORNs')
    ax.set_ylabel('ORNs')
    print('figure size:', fs)

    if cb:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = FP.add_colorbar(cp, ax_cb, 'a.u.', [-2, 0, 2, 4, 6],
                              extend='max')
        # clb = FP.add_colorbar(cp, ax_cb, r'$\Delta F/F$', [-2, 0, 2, 4, 6])
        # clb.ax.set_yticklabels(['0', '2', '4', '6'])

    file = f'{PP_WHITE}/ORN_act-{pps}_conc-{conc_sel}_{name}.'
    FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

# %%
# ###############  activity in LNs showing all the dilutions  #################
# #############################################################################


# in NNC

LN_order = {}
LN_order[1] = [1]
LN_order[2] = [1, 2]
LN_order[3] = [1, 3, 2]
LN_order[4] = [3, 2, 1, 4]
LN_order[8] = [8, 5, 4, 3, 1, 6, 2, 7]

for K in [1, 2, 3, 4, 8]:
    name = f'NNC{K}Z'
    df = Z_nnc[K].copy()
    df = df.loc[LN_order[K], :]

    # vmin_col = -2
    # vmin_to_show = -2
    vmax = np.ceil(np.max(df.values))
    # act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)
    divnorm = matplotlib.colors.Normalize(0, vmax)
    # divnorm = None
    act_map = 'Oranges'

    title = f'NNC-{K}, LNs activity {Ztex}'
    ylabel = 'LNs'
    if K ==1:
        title = f'NNC-{K}, ' + r'LN activity $\{z^{(t)}\}$'
        ylabel = 'LN'
    cb_ticks = [0, vmax]
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, 'a.u.', cb_ticks,
                                  extend='neither')
    ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel='')

    file_name = f'{PP_Z}/ORN_act-{pps}_conc-all_{name}'
    FP.save_plot(f, f'{file_name}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file_name}.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# In LC
# the issue here is that for the LC, we multiplied the activity Z
# by a random matrix

for K in [1, 8]:
    name = f'LC{K}Z'
    df =  Z_lc2[K].copy()
    df.iloc[0, :] = -df.iloc[0, :].values
    # df = df.loc[LN_order[K], :]

    vmax = np.ceil(np.max(df.values))
    vmin_col = -vmax
    vmin_to_show = -vmax
    act_map, divnorm = FP.get_div_color_map(vmin_col, vmin_to_show, vmax)

    # divnorm = None
    # act_map = 'Oranges'

    title = f'LC-{K}, LNs activity {Ztex}'
    cb_ticks = [-vmax, 0, vmax]
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, 'a.u.', cb_ticks,
                                  extend='neither')
    ax.set(xticks=[], yticks=[], ylabel='LNs', xlabel='')

    file_name = f'{PP_Z}/ORN_act-{pps}_conc-all_{name}'
    FP.save_plot(f, f'{file_name}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file_name}.pdf', SAVE_PLOTS, **pdf_opts)



# %%
# #####################  PCA VARIANCE PLOTS  ##################################
# #############################################################################
# let's try if cramming 5 lines on that graph is possible or if it is too much
k1 = 1
k2 = 8
# Y_str = r', $\mathbf{Y}$'
Y_str = f', {Ytex}'

Nl1 = f'NNC-{k1}' + Y_str
Nl2 = f'NNC-{k2}' + Y_str
Nl2M = f"NNC'-{k2}" + Y_str
Ll1 = f'LC-{k1}' + Y_str
Ll2 = f'LC-{k2}' + Y_str
Ll2M = f"LC'-{k2}" + Y_str

# now plotting the variances of the uncentered PCA instead of singular values
def plot_sv1(datas, order=None):
    x_i = np.arange(1, 22)
    pads = (0.4, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    for data in datas:
        s_data = U_X.T @ data[0]
        # s_data = LA.norm(s_data, axis=1)**2/len(data[0].T)
        s_data = np.mean(s_data*s_data, axis=1)  # that corresponds to the variance
        # if you want to have the variance, instead of the SD
        # s_data = s_data**2
        ax.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                c=data[4], label=data[5], markeredgewidth=data[6])
    ax.grid()
    ax.set(ylabel='variance', xlabel=f'PCA directions of unctr. {Xtstex}  ',
           xticks=[1, 5, 10, 15, 21], ylim=[-0.2 , None],
           yticks=[0, 3, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    if order is None:
        order = np.arange(len(handles))
    leg = ax.legend([handles[idx] for idx in order],
                    [labels[idx] for idx in order],
                    bbox_to_anchor=(1.05, 1.12), loc='upper right')
    leg.get_frame().set_linewidth(0.0)
    # leg = ax.legend()
    # leg.get_frame().set_linewidth(0.0)
    return f


datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         [Y_nnc[k1], 's-', 1, 2, c_nn1, Nl1, 0],
         [Y_nnc[k2], 's--', 1, 2, c_nn2, Nl2, 0],
         [Y_lc[k1], '+-', 1, 5, c_l1, Ll1, 0.5],
         [Y_lc[k2], '+--', 1, 5, c_l2, Ll2, 0.5]]

f = plot_sv1(datas, [0, 3, 4, 1, 2])
file = f'{PP_WHITE}/ORN_act-{pps}_X_LC-NNC-{k1}-{k2}_S_orderedbyX'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)
# ax.set_ylim(0, None)
datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         # [Y_nnc_noM[k1], 's-', 1, 2, c_nn1, Nl1, 0],
         [Y_nnc_noM[k2], 's--', 1, 2, c_nn2, Nl2M, 0],
         # [Y_lc_noM[k1], '+-', 1, 5, c_l1, Ll1, 0.5],
         [Y_lc_noM[k2], '+--', 1, 5, c_l2, Ll2M, 0.5]]

f = plot_sv1(datas, [0, 2, 1])
file = f'{PP_WHITE}/ORN_act-{pps}_X_LC-NNC-noM-{k1}-{k2}_S_orderedbyX'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #################  EIGENVALUES OF THE COVARIANCE MATRIX #####################
# ##################  DIVIDED BY MEAN  ########################################
# #################  INSTEAD OF SINGULAR VLAUES  ##############################
# #############################################################################

k1 = 1
k2 = 8
Y_str = f', {Ytex}'
Nl1 = f'NNC-{k1}' + Y_str
Nl2 = f'NNC-{k2}' + Y_str
Nl2M = f"NNC'-{k2}" + Y_str
Ll1 = f'LC-{k1}' + Y_str
Ll2 = f'LC-{k2}' + Y_str
Ll2M = f"LC'-{k2}" + Y_str


def plot_cov_ev2(datas, order=None):
    x_i = np.arange(1, 22)
    pads = (0.4, 0.1, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    ax.grid(axis='x')
    if order is None:
        order = np.arange(len(datas))
    rect = matplotlib.patches.Rectangle((0.45, 1), 0.55,
                                         -(len(datas) + 1)*0.11,
                                         linewidth=0,
                                         facecolor='white', edgecolor='white',
                                         alpha=0.7,
                                         transform=ax.transAxes)

    # Add the patch to the Axes
    ax.add_patch(rect)
    for i, data in enumerate(datas):
        cov = np.cov(data[0])
        # s_data = np.sqrt(LA.eigvalsh(cov)[::-1])
        s_data = LA.eigvalsh(cov)[::-1]
        s_data /= np.mean(s_data)
        CV = np.std(s_data)/np.mean(s_data)
        label = data[5] + f": {CV:0.1f}"
        ax.plot(x_i, s_data, data[1], lw=data[2], markersize=data[3],
                c=data[4], label=label,
                markeredgewidth=data[6])
        ax.text(1, 8/9-order[i]/9, label, transform=ax.transAxes, ha='right',
                va='top', color=data[4])
    ax.text(1, 1, r'CV$_\sigma:$', transform=ax.transAxes, ha='right',
            va = 'top')
    ax.set(ylabel='scaled variance', xlabel='PCA direction',
           xticks=[1, 5, 10, 15, 21], ylim=[None, None],
           yticks=[0, 1, 2, 4, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # [line.set_zorder(10) for line in ax.lines]
    ax.set_axisbelow(True)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    return f


datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         [Y_nnc[k1], 's-', 1, 2, c_nn1, Nl1, 0],
         [Y_nnc[k2], 's--', 1, 2, c_nn2, Nl2, 0],
         [Y_lc[k1], '+-', 1, 5, c_l1, Ll1, 0.5],
         [Y_lc[k2], '+--', 1, 5, c_l2, Ll2, 0.5]]

f = plot_cov_ev2(datas, [0, 3, 4, 1, 2])
file =f'{PP_WHITE}/ORN_act-{pps}_X_LC-NNC-{k1}-{k2}_cov-ev_div-mean'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)
# ax.set_ylim(0, None)
datas = [[X.values, '.-', 1.5, 3, 'k', Xtstex, None],
         # [Y_nnc_noM[k1], 's-', 1, 2, c_nn1, Nl1, 0],
         [Y_nnc_noM[k2], 's--', 1, 2, c_nn2, Nl2M, 0],
         # [Y_lc_noM[k1], '+-', 1, 5, c_l1, Ll1, 0.5],
         [Y_lc_noM[k2], '+--', 1, 5, c_l2, Ll2M, 0.5]]

f = plot_cov_ev2(datas, [0, 2, 1])
file = f'{PP_WHITE}/ORN_act-{pps}_X_LC-NNC-noM-{k1}-{k2}_cov-ev_div-mean'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)




# %%
# ############################# PATTERN NORM PLOTS  ###########################
# #############################################################################

k1 = 1
k2 = 8
legend_opt = {'handletextpad':0., 'frameon':False, 'loc':'upper left',
              'borderpad':0, 'bbox_to_anchor': (-0.07, 1.04)}

def scatter_norm_plot(datas, axis, xmax, ticks, lab, do_fit=False):
    pads = (0.5, 0.15, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, 14 * SQ, 14 * SQ)
    x_lin = np.linspace(0, xmax, 50)
    # title = 'norm of activity patterns  '
    title = ''
    # xlab = r'||$X_{:, i}$||'
    # ylab = r'||$Y_{:, i}$||'
    xlab = f'input {Xtstex} {lab} norm'
    ylab = f'output {Ytex}\n{lab} norm'
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    for data in datas:
        data[0] = LA.norm(data[0], axis=axis)
        data[1] = LA.norm(data[1], axis=axis)
        ax.scatter(data[0], data[1], label=data[2], s=data[3], marker=data[4],
                   c=data[5], lw=data[6], alpha=data[7])

    if do_fit:
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
           xticks=ticks, yticks=ticks, title=title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(**legend_opt)
    return f

#%%
Nl1 = f'NNC-{k1}'
Nl2 = f'NNC-{k2}'
Ll1 = f'LC-{k1}'
Ll2 = f'LC-{k2}'
Nl2M = f"NNC'-{k2}"
Ll2M = f"LC'-{k2}"
s1 = 5
s2 = 2
lw1 = 0.5
lw2 = 0
m1 = '+'
m2 = 's'
# STIMULUS NORM
xmax = 12.5
ticks = [0, 6, 12]
lab = 'pattern'

a1 = 0.8
a2 = 0.5
a3 = 1


datas = [[X, Y_lc[k1], Ll1, s1, m1, c_l1, lw1, a1, a2, a3],
         [X, Y_lc[k2], Ll2, s1, m1, c_l2, lw1, a1, a2, a3],
         [X, Y_nnc[k1], Nl1, s2, m2, c_nn1, lw2, a1, a2, a3],
         [X, Y_nnc[k2], Nl2, s2, m2, c_nn2, lw2, a1, a2, a3]]

f = scatter_norm_plot(datas, 0, xmax, ticks, lab)
file = f'{PP_WHITE}/ORN_act-{pps}-X-vs-LC-NNC-{k1}-{k2}-Y_norm'
FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)


# these are not used in the publication
# datas = [  # [X, Y_lc_noM[k1], Ll1, s1, m1, c_l1, lw1, a1, a2, a3],
#          [X, Y_lc_noM[k2], Ll2M, s1, m1, c_l2, lw1, a1, a2, a3],
#          # [X, Y_nnc_noM[k1], Nl1, s2, m2, c_nn1, lw2, a1, a2, a3],
#          [X, Y_nnc_noM[k2], Nl2M, s2, m2, c_nn2, lw2, a1, a2, a3]]
#
# f = scatter_norm_plot(datas, 0, xmax, ticks, lab)
# file = f'{PATH_PLOTS}ORN_act-{pps}-X-vs-LC-NNC-noM-{k2}-Y_norm'
# FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# ##########################  BOX PLOTS  ######################################
dss = {Xtstex: X_sel.copy(),
       f'LC-1, {Ytex}': Y_lc_sel[k1].copy(),
       f'LC-8, {Ytex}': Y_lc_sel[k2].copy(),
       f'NNC-1, {Ytex}': Y_nnc_sel[k1].copy(),
       f'NNC-8, {Ytex}': Y_nnc_sel[k2].copy()}
# dss = {'X': X_sel_norm.copy(),
#        'LC-1': Y_lc_sel_norm[k1].copy(),
#        'LC-8': Y_lc_sel_norm[k2].copy(),
#        'NNC-1': Y_nnc_sel_norm[k1].copy(),
#        'NNC-8': Y_nnc_sel_norm[k2].copy()}
for i, ds in dss.items():
    dss[i] = LA.norm(dss[i], axis=0)
    dss[i] /= dss[i].mean()
    print(np.std(dss[i])/np.mean(dss[i]))
    print(np.std(dss[i]))
dss_df = pd.DataFrame(dss)
colors = ['k', c_l1, c_l2, c_nn1, c_nn2]


pads = (0.4, 0.1, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

lineprops = {'edgecolor': 'k'}
lineprops2 = {'color': 'k'}
# sns.boxplot(data=dss_df, color=colors)
sns.swarmplot(ax=ax, data=dss_df, color='dimgray', size=1.5)
bplot = sns.boxplot(ax=ax, data=dss_df, showfliers=False, linewidth=1,
                    width=0.5, boxprops=lineprops,
                    medianprops=lineprops2, whiskerprops=lineprops2,
                    capprops=lineprops2)
for i in range(0, 5):
    mybox = bplot.artists[i]
    mybox.set_facecolor('None')
    mybox.set_edgecolor(colors[i])
    for j in range(i * 5, i * 5 + 5):
        line = ax.lines[j]
        line.set_color(colors[i])
        # line.set_mfc('k')
        # line.set_mec('k')

bplot.set_xticklabels(bplot.get_xticklabels(),
                      rotation=25, horizontalalignment='right')
ax.tick_params(axis='x', which='both', pad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('scaled pattern norm')
# ax.set_title('scaled channel norm')
ax.set_ylim(0, 2.3)
ax.set_yticks([0, 1, 2])
ax.text(-0.3, 2.35, 'CV:', color='k', ha='left', fontsize=6)
for i in range(5):
    ax.text(i, 2.05, f"{np.std(dss_df.iloc[:, i].values):0.2}",
            color='k', ha='center', va='bottom', fontsize=5)

file = f'{PP_WHITE}/ORN_act-{pps}-X-vs-LC-NNC-{k1}-{k2}-Y_norm_box'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# %%
# ############################# CHANNEL NORM PLOTS  ###########################
# #############################################################################
Nl1 = f'NNC-{k1}'
Nl2 = f'NNC-{k2}'
Ll1 = f'LC-{k1}'
Ll2 = f'LC-{k2}'
Nl2M = f"NNC'-{k2}"
Ll2M = f"LC'-{k2}"

xmax = 20
ticks = [0, 10, 20]
lab = 'channel'

a1 = 0.8
a2 = 0.5
a3 = 1


datas = [[X, Y_lc[k1], Ll1, s1, m1, c_l1, lw1, a1, a2, a3],
         [X, Y_lc[k2], Ll2, s1, m1, c_l2, lw1, a1, a2, a3],
         [X, Y_nnc[k1], Nl1, s2, m2, c_nn1, lw2, a1, a2, a3],
         [X, Y_nnc[k2], Nl2, s2, m2, c_nn2, lw2, a1, a2, a3]]

f = scatter_norm_plot(datas, 1, xmax, ticks, lab)
file = f'{PP_WHITE}/ORN_act-{pps}-X-vs-LC-NNC-{k1}-{k2}-Y_norm_ch'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# not used for the paper

# datas = [  # [X, Y_lc_noM[k1], Ll1, s1, m1, c_l1, lw1, a1, a2, a3],
#          [X, Y_lc_noM[k2], Ll2M, s1, m1, c_l2, lw1, a1, a2, a3],
#          # [X, Y_nnc_noM[k1], Nl1, s2, m2, c_nn1, lw2, a1, a2, a3],
#          [X, Y_nnc_noM[k2], Nl2M, s2, m2, c_nn2, lw2, a1, a2, a3]]
#
# f = scatter_norm_plot(datas, 1, xmax, ticks, lab)
# file = f'{PATH_PLOTS}ORN_act-{pps}-X-vs-LC-NNC-noM-{k2}-Y_norm_ch'
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)


# %%
# ##########################  BOX PLOTS  ######################################
dss = {Xtstex: X.copy(),
       f'LC-1, {Ytex}': Y_lc[k1].copy(),
       f'LC-8, {Ytex}': Y_lc[k2].copy(),
       f'NNC-1, {Ytex}': Y_nnc[k1].copy(),
       f'NNC-8, {Ytex}': Y_nnc[k2].copy()}

for i, ds in dss.items():
    dss[i] = LA.norm(dss[i], axis=1)
    dss[i] /= dss[i].mean()
    # dss[i] /= ds.mean()
    print(np.std(dss[i])/np.mean(dss[i]))
dss_df = pd.DataFrame(dss)
colors = ['k', c_l1, c_l2, c_nn1, c_nn2]


pads = (0.4, 0.1, 0.35, 0.2)
fs, axs = FP.calc_fs_ax(pads, 18*SQ, 14*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)

lineprops = {'edgecolor': 'k'}
lineprops2 = {'color': 'k'}
# sns.boxplot(data=dss_df, color=colors)
sns.swarmplot(ax=ax, data=dss_df, color='dimgray', size=2)
bplot = sns.boxplot(ax=ax, data=dss_df, showfliers=False, linewidth=1,
                    width=0.5, boxprops=lineprops,
                    medianprops=lineprops2, whiskerprops=lineprops2,
                    capprops=lineprops2)
for i in range(0, 5):
    mybox = bplot.artists[i]
    mybox.set_facecolor('None')
    mybox.set_edgecolor(colors[i])
    for j in range(i*5, i*5+5):
        line = ax.lines[j]
        line.set_color(colors[i])
        # line.set_mfc('k')
        # line.set_mec('k')

bplot.set_xticklabels(bplot.get_xticklabels(),
                      rotation=25, horizontalalignment='right')
ax.tick_params(axis='x', which='both', pad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('scaled channel norm')
# ax.set_title('scaled channel norm')
ax.set_ylim(0, 2)
ax.set_yticks([0, 1, 2])
ax.text(-0.3, 2, 'CV:', color='k', ha='left', fontsize=6)
for i in range(5):
    ax.text(i, 1.75, f"{np.std(dss_df.iloc[:, i].values):0.2}",
            color='k', ha='center', va='bottom', fontsize=5)

file = f'{PP_WHITE}/ORN_act-{pps}-X-vs-LC-NNC-{k1}-{k2}-Y_norm_ch_box'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

#
#

# %%
# #############################################################################
# ##################### CORRELATION MATRICES PLOTS  ###########################
# #############################################################################

importlib.reload(FP)
# what would make sense i guess is to plot all of the
# channel and patter correlation plots. And then you can decide which ones
# you actually show. Maybe some in the main text, some in the supplement
# per channel
k1 = 1
k2 = 8
pads = (0.2, 0.4, 0.35, 0.2)

subtitles = [Xtstex, f'NNC-{k1}, {Ytex}', f'NNC-{k2}, {Ytex}',
             f'LC-{k1}, {Ytex}', f'LC-{k2}, {Ytex}',
             f"NNC'-{k2}, {Ytex}", f"LC'-{k2}, {Ytex}"]
# titles = [name + ', channel corr.' for name in subtitles]
titles = [name for name in subtitles]
# titles[2] = titles[2] + '  '
names = ['X', f'Y-NNC{k1}', f'Y-NNC{k2}', f'Y-LC{k1}', f'Y-LC{k2}',
         f'Y-NNC{k2}noM', f'Y-LC{k2}noM']


# for i, data in enumerate([X, Y_nnc[k1], Y_nnc[k2], Y_lc[k1], Y_lc[k2],
#                           Y_nnc_noM[k2], Y_lc_noM[k2]]):
for i, data in enumerate([X, Y_nnc[k1], Y_nnc[k2], Y_lc[k1], Y_lc[k2]]):
    cond = (i in [2, 4] and SCAL_W in [1, 2]) or (i == 2 and SCAL_W == 10)
    if cond:
        pads = (0.2, 0.4, 0.35, 0.2)
    else:
        pads = (0.2, 0.2, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)
    df = sim_func(data)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(df, ax, vlim=[-1, 1], show_lab_y=False,
                       title=titles[i], show_lab_x=False,
                       cmap=plt.cm.bwr)
    ax.set_xlabel('ORNs')
    ax.set_ylabel('ORNs')

    if cond:
        ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                            CB_W / fs[0], axs[3]])
        clb = FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])
    file = f'{PP_WHITE}/ORN_act-{pps}_{names[i]}_{sf}_channel'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# per pattern
# titles = [name + ', pattern corr.' for name in subtitles]
titles = [name for name in subtitles]
# titles[2] = titles[2] + '  '

# for i, data in enumerate([X, Y_nnc[k1], Y_nnc[k2], Y_lc[k1], Y_lc[k2],
#                           Y_nnc_noM[k2], Y_lc_noM[k2]]):
for i, data in enumerate([X, Y_nnc[k1], Y_nnc[k2], Y_lc[k1], Y_lc[k2]]):
    if (i in [2, 4] and SCAL_W in [1, 2]) or (i == 2 and SCAL_W == 10):
        pads = (0.2, 0.4, 0.35, 0.2)
    else:
        pads = (0.2, 0.2, 0.35, 0.2)
    fs, axs = FP.calc_fs_ax(pads, gw=SQ * 15, gh=SQ * 15)
    df = sim_func(data.T)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)
    cp = FP.imshow_df2(df, ax, vlim=[-1, 1], show_lab_y=False,
                       title=titles[i], show_lab_x=False,
                       cmap=plt.cm.bwr)
    ax.set_xlabel('stimuli')
    ax.set_ylabel('stimuli')
    # ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
    #                     CB_W / fs[0], axs[3]])
    # clb = FP.add_colorbar(cp, ax_cb, '', [-1, 0, 1])
    file = f'{PP_WHITE}/ORN_act-{pps}_{names[i]}_{sf}_pattern'
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

# %%
# #############################################################################
# #########################  HISTOGRAM PLOTS  #################################
# #########################  FOR CORRELATIONS  ################################
# #############################################################################
k1 = 1
k2 = 8
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        plt.legend(frameon=False, handletextpad=0.4, borderpad=0,
                   handlelength=1, bbox_to_anchor=(1.1, 1.2),
                   loc='upper right',
                   ncol=3, columnspacing=0.7)
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
        axin.spines['top'].set_visible(False)
        axin.spines['right'].set_visible(False)
        # axin.get_legend().remove()
        ax.indicate_inset_zoom(axin, edgecolor='grey', alpha=0.7, lw=1)
        # mark_inset(ax, axin, loc1=2, loc2=3, fc="none", ec="0.5")


    FP.save_plot(f, f'{file}.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, f'{file}.pdf', SAVE_PLOTS, **pdf_opts)


x0 = 0.25
Ll1 = f'LC-{k1}, {Ytex}'
Ll2 = f'LC-{k2}, {Ytex}'
Nl1 = f'NNC-{k1}, {Ytex}'
Nl2 = f'NNC-{k2}, {Ytex}'

# LC
datas = [[X_cc_c, 'k', Xtstex],
         [Y_lc_cc_c[k1], c_l1, Ll1],
         [Y_lc_cc_c[k2], c_l2, Ll2]]
# plot_dists(datas, xlabel, ylabel, 'channel correlation', [0, 2, 4],
plot_dists(datas, xlabel, ylabel, '', [0, 2.5, 5],
           f'{PP_WHITE}/ORN_act-{pps}_LC-{k1}-{k2}_{sf}-chan_dist',
           ylim=(0, 5.2), zoom=True, square=[[x0, 1], [0., 0.8]])
# plot_dists(datas, xlabel, '', '', [0, 0.5, 1.],
#            f'{PATH_PLOTS}ORN_act-{pps}_LC-{k1}-{k2}_{sf}-chan_dist_zoom',
#            xlim=(0.2, 1), ylim=(0.0, 1.1), legend=False, fs=[18/1.5*0.8, 15])

# datas = [[X_cc_p, 'k', 'X'],
#          [Y_lc_cc_p[k1], c_l1, Ll1],
#          [Y_lc_cc_p[k2], c_l2, Ll2]]
# datas = [[X_cc_p_o, 'k', 'X'],
#          [Y_lc_cc_p_o[k1], c_l1, Ll1],
#          [Y_lc_cc_p_o[k2], c_l2, Ll2]]
datas = [[X_cc_p_sel, 'k', Xtstex],
         [Y_lc_cc_p_sel[k1], c_l1, Ll1],
         [Y_lc_cc_p_sel[k2], c_l2, Ll2]]
# plot_dists(datas, xlabel, ylabel, 'pattern correlation', [0, 1, 2],
plot_dists(datas, xlabel, ylabel, '', [0, 1, 2],
           f'{PP_WHITE}/ORN_act-{pps}-conc{conc_sel}_LC-{k1}-{k2}_{sf}'
           f'-patt_dist', ylim=(0, 2.4), zoom=True, square=[[x0, 1], [0, 0.7]])



# NNC
datas = [[X_cc_c, 'k', Xtstex],
         [Y_nnc_cc_c[k1], c_nn1, Nl1],
         [Y_nnc_cc_c[k2], c_nn2, Nl2]]
# plot_dists(datas, xlabel, ylabel, 'channel correlation', [0, 2.5, 5],
plot_dists(datas, xlabel, ylabel, '', [0, 2.5, 5],
           f'{PP_WHITE}/ORN_act-{pps}_NNC-{k1}-{k2}_{sf}-chan_dist',
           ylim=(0, 5.7), zoom=True, square=[[x0, 1], [0, 0.8]])

yticks = {1: [0, 1, 2], 2: [0, 1.5, 3], 10: [0, 2.5, 5]}
datas = [[X_cc_p_sel, 'k', Xtstex],
         [Y_nnc_cc_p_sel[k1], c_nn1, Nl1],
         [Y_nnc_cc_p_sel[k2], c_nn2, Nl2]]
# plot_dists(datas, xlabel, ylabel, 'pattern correlation', yticks[SCAL_W],
plot_dists(datas, xlabel, ylabel, '', yticks[SCAL_W],
           f'{PP_WHITE}/ORN_act-{pps}-conc{conc_sel}_NNC-{k1}-{k2}_{sf}-'
           f'patt_dist', ylim=(0, 3.), zoom=True, square=[[x0, 1], [0, 0.7]])

# LC and NNC together, this is used for rho=10
yticks={1: [0, 2.5, 5], 2:[0, 3, 6], 10: [0, 3, 6]}
ylim = {2: 6, 10: 6}
datas = [[X_cc_c, 'k', Xtstex],
         [Y_lc_cc_c[k2], c_l2, Ll2],
         [Y_nnc_cc_c[k2], c_nn2, Nl2]]
# plot_dists(datas, xlabel, ylabel, 'channel correlation', yticks[SCAL_W],
plot_dists(datas, xlabel, ylabel, '', yticks[SCAL_W],
           f'{PP_WHITE}/ORN_act-{pps}_LC-NNC-{k2}_{sf}-chan_dist',
           ylim=(0, ylim[SCAL_W]), zoom=True, square=[[x0, 1], [0, 0.8]])


yticks = {1: [0, 1, 2], 2: [0, 2, 4], 10: [0, 1.5, 3]}
ylim = {2: 4, 10: 3.2}
datas = [[X_cc_p_sel, 'k', Xtstex],
         [Y_lc_cc_p_sel[k2], c_l2, Ll2],
         [Y_nnc_cc_p_sel[k2], c_nn2, Nl2]]
# plot_dists(datas, xlabel, ylabel, 'pattern correlation', yticks[SCAL_W],
plot_dists(datas, xlabel, ylabel, '', yticks[SCAL_W],
           f'{PP_WHITE}/ORN_act-{pps}-conc{conc_sel}_LC-NNC-{k2}_{sf}-'
           f'patt_dist',
           ylim=(0, ylim[SCAL_W]), zoom=True, square=[[x0, 1], [0, 0.7]])

# this is not used in the paper
# datas = [[X_cc_c, 'k', 'X'],
#          [Y_lc_noM_cc_c[k2], c_l2, f"Y, LC'-{k2}"],
#          [Y_nnc_noM_cc_c[k2], c_nn2, f"Y, NNC'-{k2}"]]
# plot_dists(datas, xlabel, ylabel, 'channel correlation', [0, 3, 6],
#            f'{PATH_PLOTS}ORN_act-{pps}-conc{conc_sel}_LC-NNC-noM-{k2}_{sf}'
#            f'-chan_dist', ylim=(0, 6), zoom=True, square=[[x0, 1], [0, 0.8]])
#
# datas = [[X_cc_p_sel, 'k', 'X'],
#          [Y_lc_noM_cc_p_sel[k2], c_l2, f"Y, LC'-{k2}"],
#          [Y_nnc_noM_cc_p_sel[k2], c_nn2, f"Y, NNC'-{k2}"]]
# plot_dists(datas, xlabel, ylabel, 'pattern correlation', [0, 2, 4],
#            f'{PATH_PLOTS}ORN_act-{pps}-conc{conc_sel}_LC-NNC-noM-{k2}_{sf}'
#            f'-patt_dist', ylim=(0, 4), zoom=True, square=[[x0, 1], [0, 0.7]])


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
# #########################  W clustering in NNC  #############################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################

Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
# Ws_cn = FG.get_ctr_norm(Ws).loc[par_act.ORN_order].T


# %%
# Plotting the previous graphs on a single figure
k = 8
lab_y = {-1: True, -0.45: False, 0: False, 1: False}
pads = [0.3, 0.075, 0.37, 0.35]
title = {-1: '0.1', -0.45: '0.35', 0: '1', 1: '10'}
n_ax = len(title)
fs, axs = FP.calc_fs_ax(pads, gw=SQ * k * n_ax + (n_ax-1) * 2 * SQ, gh=SQ * k)
f = plt.figure(figsize=fs)
axs_coords = {-1: [axs[0], axs[1], SQ * k / fs[0], axs[3]],
             -0.45:[axs[0] + SQ * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
              0: [axs[0] + SQ * 2 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
              1: [axs[0] + SQ * 3 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]]}
for s in lab_y.keys():
    rho = 10**s
    pps_local = f'{title[s]}o'
    print('rho:', rho)
    W_nncT = Ws_nnc.loc[:, (s*10, 1)].values.T

    links = sch.linkage(np.corrcoef(W_nncT), method='average', optimal_ordering=True)
    new_order = sch.leaves_list(links)
    df = pd.DataFrame(np.corrcoef(W_nncT[new_order]))
    #
    # CG = sns.clustermap(np.corrcoef(W_nncT), cmap=plt.cm.bwr, vmin=-1, vmax=1)
    # idx = CG.dendrogram_col.reordered_ind
    # df = pd.DataFrame(np.corrcoef(W_nncT[idx]))


    ax = f.add_axes(axs_coords[s])
    cp = FP.imshow_df2(df, ax, vlim=[-1, 1], show_lab_y=lab_y[s],
                       show_lab_x=True, cmap=plt.cm.bwr, rot=0)
    ax.set_title(r'$\rho$ = ' + f'{title[s]}', pad=2, fontsize=ft_s_lb)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(np.arange(1, 9))
    # ax.set_xlabel(r'NNC-8, $\mathbf{w}_k$', labelpad=1, rotation_mode='default',
    #               ha='center')
    if lab_y[s]:
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels(np.arange(1, 9))
        ax.set_ylabel(r'$\mathbf{w}_k$', labelpad=6, va='center')
    # ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
    #                     CB_W / fs[0], axs[3]])
    # clb = FP.add_colorbar(cp, ax_cb, '', [-1, 0, 1])
    print(W_nncT[idx].sum(axis=1))
plt.suptitle(r'NNC-8, corr. among $\mathbf{w}_k$')
f.text(0.54, 0.1, r'$\mathbf{w}_k$', rotation=0, fontsize=ft_s_lb, va='bottom',
       ha='center')
file = f'{PP_CON_PRED}/ORN_act_NNC{k}_corrW_all'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)



# %%
# #############################################################################
# #################### LOOKING AT A WHOLE SET OF W ############################
# #################### AND COMPARING THE CLUSTERING TO THE DATA ###############
# #############################################################################
strm = 0
side = 'L'
LNs_sel1 = LNs_sel_d_side[side]
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df1 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)

side = 'R'
LNs_sel1 = LNs_sel_d_side[side]
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df2 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
# %%
th = 0.45
df1_ent = FG.get_entries(df1, diag=False)
corr_L = FG.rectify(df1_ent).mean()
df2_ent = FG.get_entries(df2, diag=False)
corr_R = FG.rectify(df2_ent).mean()
print(corr_L, corr_R)
# %%
Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
Ws_nnc_cn = FG.get_ctr_norm(Ws_nnc).loc[par_act.ORN_order].T
mi2 = pd.MultiIndex(levels=[[], []], codes=[[], []],
                    names=['rho', 'rep'])
corr_W_nnc_s = pd.Series(index=mi2)
clust_W_nnc_s = pd.Series(index=mi2)

for p in np.arange(-10, 10.1, 0.5):
    # rho = 10**(p/10)
    for i in range(50):
        W_nnc = Ws_nnc_cn.loc[(p, i)]
        corr = W_nnc @ W_nnc.T
        df1_ent = FG.get_entries(corr, diag=False)
        corr_W_nnc_s.loc[(p, i)] = FG.rectify(df1_ent).mean()

# %%
rho_special = 0.35
x_special = np.log10(rho_special)
x = corr_W_nnc_s.index.unique('rho')/10
y = corr_W_nnc_s.groupby('rho').mean()
e = corr_W_nnc_s.groupby('rho').std()

pads = (0.4, 0.1, 0.35, 0.1)
# fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*10)
fs, axs = FP.calc_fs_ax(pads, SQ*12, SQ*12)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.plot(x, y, lw=1, c='k')
ax.fill_between(x, y-e, y+e, alpha=0.5, facecolor='k', label='NNC-8')
ax.fill_between([min(x), max(x)], [corr_L, corr_L], [corr_R, corr_R],
                alpha=0.5, label='data')
ax.plot([x_special, x_special], [0, 0.4], lw=0.5, color='gray', ls='--')
ax.set_yticks([0, 0.2, 0.4])
# ax.set_yticklabels([0, '', 2, '', 4])
ax.set_xticks([-1, x_special, 0, 1])
ax.set_xticklabels([0.1, rho_special, 1, 10])
ax.set_ylim(0, 0.4)
ax.set_ylabel(r'$\overline{r}_+$')
ax.set_xlabel(r'$\rho$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False, loc='lower left')
file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-8_W-corr')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **png_opts)

# %%
# finding where the data and the model lines intersect
# the value of x around where the lines intersect
corr_M = (corr_L + corr_R) / 2
print(x[np.sum(y > corr_M)])

#%%
# #############################################################################
# #############################################################################
# #############################################################################
# just for testing, doing the same as above, but with the angle between
# vectors

Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
# Ws_cn = FG.get_ctr_norm(Ws).loc[par_act.ORN_order].T

#
# # %%
# # Plotting the previous graphs on a single figure
# k = 8
# lab_y = {-1: True, -0.45: False, 0: False, 1: False}
# pads = [0.3, 0.075, 0.37, 0.35]
# title = {-1: '0.1', -0.45: '0.35', 0: '1', 1: '10'}
# n_ax = len(title)
# fs, axs = FP.calc_fs_ax(pads, gw=SQ * k * n_ax + (n_ax-1) * 2 * SQ, gh=SQ * k)
# f = plt.figure(figsize=fs)
# axs_coords = {-1: [axs[0], axs[1], SQ * k / fs[0], axs[3]],
#              -0.45:[axs[0] + SQ * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
#               0: [axs[0] + SQ * 2 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
#               1: [axs[0] + SQ * 3 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]]}
# for s in lab_y.keys():
#     rho = 10**s
#     pps_local = f'{title[s]}o'
#     print('rho:', rho)
#     W_nncT = Ws_nnc.loc[:, (s*10, 1)].values.T
#     CG = sns.clustermap(np.corrcoef(W_nncT), cmap=plt.cm.bwr, vmin=-1, vmax=1)
#     idx = CG.dendrogram_col.reordered_ind
#     W_nnc_n = FG.get_norm_np(W_nncT[idx].T)
#     df = pd.DataFrame(W_nnc_n.T@W_nnc_n)
#     df = np.arccos(df)
#
#     ax = f.add_axes(axs_coords[s])
#     cp = FP.imshow_df2(df, ax, vlim=[0, np.pi/2], show_lab_y=lab_y[s],
#                        show_lab_x=True, cmap=plt.cm.magma, rot=0)
#     ax.set_title(r'$\rho$ = ' + f'{title[s]}', pad=2, fontsize=ft_s_lb)
#     ax.set_xticks(np.arange(8))
#     ax.set_xticklabels(np.arange(1, 9))
#     # ax.set_xlabel(r'NNC-8, $\mathbf{w}_k$', labelpad=1, rotation_mode='default',
#     #               ha='center')
#     if lab_y[s]:
#         ax.set_yticks(np.arange(8))
#         ax.set_yticklabels(np.arange(1, 9))
#         ax.set_ylabel(r'$\mathbf{w}_k$', labelpad=6, va='center')
#     # ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
#     #                     CB_W / fs[0], axs[3]])
#     # clb = FP.add_colorbar(cp, ax_cb, '', [-1, 0, 1])
#     print(W_nncT[idx].sum(axis=1))
# plt.suptitle(r'NNC-8, corr. among $\mathbf{w}_k$')
# f.text(0.54, 0.1, r'$\mathbf{w}_k$', rotation=0, fontsize=ft_s_lb, va='bottom',
#        ha='center')
# file = f'{PP_CON_PRED}/ORN_act_NNC{k}_angle_W_all'
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
#
# #%%
# # #############################################################################
# # #############################################################################
#
#
# strm = 0
# side = 'L'
# LNs_sel1 = LNs_sel_d_side[side]
# con_ff_sel = con_strms3.loc[:, strm]
# con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
# con_ff_sel.columns = LNs_sel_short
# con_ff_sel_n = FG.get_norm(con_ff_sel)
# df1 = FG.get_cos_sim(con_ff_sel_n, con_ff_sel_n)
# # this makes sure that there is no numerical problem with any
# # number larger than 1.
# df1[:] = np.minimum.reduce([df1.values, np.ones((len(df1), len(df1)))])
# df1 = np.arccos(df1)
#
# side = 'R'
# LNs_sel1 = LNs_sel_d_side[side]
# con_ff_sel = con_strms3.loc[:, strm]
# con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
# con_ff_sel.columns = LNs_sel_short
# con_ff_sel_n = FG.get_norm(con_ff_sel)
# df2 = FG.get_cos_sim(con_ff_sel_n, con_ff_sel_n)
# df2[:] = np.minimum.reduce([df2.values, np.ones((len(df2), len(df2)))])
# df2 = np.arccos(df2)
# # %%
# th = 0.45
# df1_ent = FG.get_entries(df1, diag=False)
# corr_L = FG.rectify(df1_ent).mean()
# df2_ent = FG.get_entries(df2, diag=False)
# corr_R = FG.rectify(df2_ent).mean()
# print(corr_L, corr_R)
# # %%
# Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
# Ws_nnc_n = FG.get_norm(Ws_nnc).loc[par_act.ORN_order].T
# mi2 = pd.MultiIndex(levels=[[], []], codes=[[], []],
#                     names=['rho', 'rep'])
# corr_W_nnc_s = pd.Series(index=mi2)
# clust_W_nnc_s = pd.Series(index=mi2)
#
# for p in np.arange(-10, 10.1, 0.5):
#     rho = 10**(p/10)
#     for i in range(50):
#         W_nnc = Ws_nnc_n.loc[(p, i)].T
#         CS =  FG.get_cos_sim(W_nnc, W_nnc)
#         angle = np.arccos(CS)
#         df1_ent = FG.get_entries(angle, diag=False)
#         corr_W_nnc_s.loc[(p, i)] = FG.rectify(df1_ent).mean()
#
# # %%
# rho_special = 0.35
# x_special = np.log10(rho_special)
# x = corr_W_nnc_s.index.unique('rho')/10
# y = corr_W_nnc_s.groupby('rho').mean()
# e = corr_W_nnc_s.groupby('rho').std()
#
# pads = (0.4, 0.1, 0.35, 0.1)
# # fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*10)
# fs, axs = FP.calc_fs_ax(pads, SQ*12, SQ*12)  # pads, gw, gh
# f = plt.figure(figsize=fs)
# ax = f.add_axes(axs)
# ax.plot(x, y, lw=1, c='k')
# ax.fill_between(x, y-e, y+e, alpha=0.5, facecolor='k', label='NNC-8')
# ax.fill_between([min(x), max(x)], [corr_L, corr_L], [corr_R, corr_R],
#                 alpha=0.5, label='data')
# ax.plot([x_special, x_special], [0, 0.4], lw=0.5, color='gray', ls='--')
# ax.set_yticks([0, 0.2, 0.4])
# # ax.set_yticklabels([0, '', 2, '', 4])
# ax.set_xticks([-1, x_special, 0, 1])
# ax.set_xticklabels([0.1, rho_special, 1, 10])
# # ax.set_ylim(0, 0.4)
# ax.set_ylabel(r'$\overline{r}_+$')
# ax.set_xlabel(r'$\rho$')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.legend(frameon=False, loc='lower left')
# file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
#         f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-8_W-angle')
# FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
# FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **png_opts)
#
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
importlib.reload(FP)
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
for data_i in [1, 2]:
# for data_i in [1]:
    file = FO.OLF_PATH / \
           f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
    X = np.load(file)
    df = pd.DataFrame(X)
    D, N = X.shape
    f, ax, _ = FP.plot_full_activity(df, act_map, divnorm, title, cb_title,
                       cb_ticks, pads=[0.2, 0.4, 0.3, pad_up], extend='neither',
                       squeeze=0.1, do_vert_spl=False, SQ=SQ/10*9)
    ax.set(xticks=[], yticks=[], ylabel=ylabel, xlabel=f'samples (N={N})')

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
D = 10
K = 2

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
for data_i in [1, 2]:
# for data_i in [1]:
    file = FO.OLF_PATH / f'results/dataset{data_i}/dataset_{D}D_{n_clusters}clusters.npy'
    X = np.load(file)
    D, N = X.shape
    for rho, K in itertools.product([0.1, 1, 10], [2, 3]):
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
        cp = FP.imshow_df2(corr, ax, cmap=plt.cm.bwr, vlim=1,
                           splits_x=[], splits_y=[], show_lab_x=True,
                           rot=0)
        ax.set_xticklabels([wtexs[i] for i in range(1, K + 1)])
        ax.set_yticklabels([wtexs[i] for i in range(1, K + 1)])

        FP.add_colorbar(cp, ax_cb, r'$r$', [-1, 0, 1])

        entries = FG.get_entries(corr, diag=False)
        max_rect_corr = FG.rectify(entries).mean()
        print(data_i, K, rho, 'mean rectified corr', max_rect_corr)
        ax.set_title(r'$\overline{r}_+$' + f'={max_rect_corr:.2f}')

        file = f'{PP_WrhoK}/dataset{data_i}_K{K}_rho{rho}_W_corr.'
        FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
        FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)



#%%