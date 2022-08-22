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
import fitz # to rotate a pdf page, packaged needed to install: pymupdf
# -> pip install pymupdf
# could be replaced by PyPDF2
from typing import Union

import itertools

from functools import partial

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
ft_s_tl = 8  # fontsize title
png_opts = {'dpi': 400}
pdf_opts = {'dpi': 800}

# I wonder if something like this is needed in the save_plot:
# bbox_inches='tight', pad_inches = 0
# it seems not, then it doesn't respect the size you've put

CB_W = 0.1
CB_DX = 0.11
SQ = 0.07

# this updates the values for all the plots
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
matplotlib.rcParams['font.size'] = ft_s_tk
matplotlib.rcParams['axes.labelsize'] = ft_s_lb
matplotlib.rcParams['axes.titlesize'] = ft_s_tl
matplotlib.rcParams['figure.titlesize'] = ft_s_tl
matplotlib.rcParams['xtick.labelsize'] = ft_s_tk
matplotlib.rcParams['ytick.labelsize'] = ft_s_tk
# matplotlib.rcParams['legend.fontsize'] = 'medium' #ft_s_tk
# matplotlib.rcParams['legend.title_fontsize'] = None#ft_s_lb
matplotlib.rcParams['savefig.transparent'] = True

cb_title_font = None  # if you don't want any changes
cb_title_font = {'size':ft_s_lb}

def add_colorbar_crt(cp, ax, cbtitle='', ticks=[], pad=1, extend='neither',
                 title_font=cb_title_font):
    return FP.add_colorbar(cp, ax, cbtitle=cbtitle, ticks=ticks, pad=pad,
                           extend=extend, title_font=title_font)

# these options make the plotting much slower unfortunately
# https://matplotlib.org/stable/tutorials/text/usetex.html
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"})

# If you want sans serif fonts for the math
# the helvet does not render \Delta correctly for some reason
# https://felix11h.github.io/blog/matplotlib-tgheros
# https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib/20709149#20709149
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath'
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage[helvet]{sfmath}'
# plt.rcParams['text.latex.preamble'] = r'\renewcommand{\familydefault}{\sfdefault} \usepackage{helvet} \usepackage[helvet]{sfmath}'# \everymath={\sf}'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{tgheros} \usepackage{sansmath} \sansmath'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{newtxtext,newtxmath,sansmath} \usepackage{sansmath} \sansmath'

# corr_cmap = plt.cm.RdBu_r
corr_cmap = plt.cm.bwr

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

print('DONE IMPORTING')