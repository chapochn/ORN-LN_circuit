"""
Contains functions and classes that are olfactory-specific.

"""
# ################################# IMPORTS ###################################


import copy
import itertools
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as LA
import scipy.stats as sps
import pathlib

import functions.nmf as nmf
import functions.general as FG
import functions.plotting as FP
import functions.olf_circ_offline as FOC
import params.act2 as par_act2
import params.act3 as par_act3
import params.con as par_con
import os
from typing import List, Tuple, Any

from functions.plotting import plot_cov  # since we are not using any other
# plotting function

PATH = os.path.realpath(f"{os.path.expanduser('~')}/ORN-LN_circuit") + '/'
OLF_PATH = pathlib.Path.home() / 'ORN-LN_circuit'

# ###################### FUNCTIONS CONNECTIVITY DATA ##########################

def get_labels(sheet, n_begin: int, n: int):
    all_labels_row = np.array(FG.get_labels_row(sheet, n_begin, n, 1))
    all_labels_column = np.array(FG.get_labels_clmn(sheet, n_begin, n, 1))
    if np.sum((all_labels_row == all_labels_column)-1) != 0:
        print('left headers rows and columns are not the same!')

    all_labels = all_labels_row  # both labels are the same
    # to get names similar to the activity data
    all_labels = FG.replace_substr_np(all_labels, par_con.dict_str_replace)
    return all_labels


def combine_labels(labels_left, labels_right, lab):
    """
    this function combines the left and right cell names.
    It first replaces left and L and right by R if it is the same label on
    both sides
    Then if on the right there is the string right and on the left there is the
    string left, it is replaced by the string lab
    """
    labels_combined = np.zeros(len(labels_left), dtype='U40')

    for i in range(len(labels_combined)):
        if labels_left[i] == labels_right[i]:
            labels_combined[i] = FG.repl_add(labels_left[i], ' left', ' L')
            labels_combined[i] = FG.repl_add(labels_combined[i], ' right', ' R')
            labels_combined[i] += lab
        else:
            if ('left' in labels_left[i]) and ('right' in labels_right[i]):
                labels_combined[i] = FG.repl_add(labels_left[i], ' left', lab)
            else:
                labels_combined[i] = labels_left[i] + labels_right[i]
                print('something weird in the function combine_labels')
                print('labels are', labels_left[i], labels_right[i])
    return labels_combined


# We need to find a new way to combine the cells, being caresulf with the
# broad cells
# that means we cannot anymore just sum cells
def import_con_data(keys=['L', 'R', 'M']):
    """
    returns the connectivity data
    this function makes several transformations to the initial dataset
    which is encoded in an excel sheet. Transformations that are made
    are around the names of the variables, and also the creation of
    the M dataset, which are the mean connections from L and R

    keys should be a list or an array indicating which 'sides' we want to get
    options are: 'L', 'R', 'S', 'M'
    the option None return them all, no option returns L, R, M
    """



    dict_str = {'bilateral': 'bil.',
                # 'left': 'L',
                # 'right': 'R',
                'dendrites': 'dend',
                'octopaminergic': 'oct.',
                'Olfactory': 'olf.',
                'LOWER': 'low.',
                'UPPER': 'up.',
                'Descending': 'Desc.'}

    sheet_con_L = FG.get_sheet(OLF_PATH / par_con.file_L)
    sheet_con_R = FG.get_sheet(OLF_PATH / par_con.file_R)

    cons = {}
    cons['L'] = FG.get_data(sheet_con_L, par_con.all_begin, par_con.all_n,
                            par_con.all_begin, par_con.all_n)
    cons['R'] = FG.get_data(sheet_con_R, par_con.all_begin, par_con.all_n,
                            par_con.all_begin, par_con.all_n)

    cells = {}
    cells['L'] = get_labels(sheet_con_L, par_con.all_begin, par_con.all_n)
    cells['R'] = get_labels(sheet_con_R, par_con.all_begin, par_con.all_n)

    # changing the position of ORN and PN in the cell names
    for i, s in itertools.product(range(len(cells['L'])), ['L', 'R']):
        cells[s][i] = FG.repl_preadd(cells[s][i], ' ORN', 'ORN ')
        cells[s][i] = FG.repl_preadd(cells[s][i], ' PN', 'PN ')

    cells['S'] = combine_labels(cells['L'], cells['R'], ' S')
    cells['M'] = combine_labels(cells['L'], cells['R'], ' M')

    for cells1 in cells.values():
        cells1 = FG.replace_substr_np(cells1, dict_str)

    for i in range(len(cells['L'])):
        cells['L'][i] = FG.repl_add(cells['L'][i], ' left', ' L')
        cells['R'][i] = FG.repl_add(cells['R'][i], ' left', ' L')
        cells['L'][i] = FG.repl_add(cells['L'][i], ' right', ' R')
        cells['R'][i] = FG.repl_add(cells['R'][i], ' right', ' R')

        cells_bil = ['Keystone', 'PN 35a bil.']  # bilateral cells

        for cl in cells_bil:
            if cells['L'][i] == f'{cl} L':
                cells['L'][i] = f'{cl} L L'

            if cells['L'][i] == f'{cl} R':
                cells['L'][i] = f'{cl} R L'

            if cells['R'][i] == f'{cl} R':
                cells['R'][i] = f'{cl} R R'

            if cells['R'][i] == f'{cl} L':
                cells['R'][i] = f'{cl} L R'

    cons['L'] = pd.DataFrame(cons['L'], index=cells['L'], columns=cells['L'])
    cons['R'] = pd.DataFrame(cons['R'], index=cells['R'], columns=cells['R'])
    # the sum of the connectivities
    cons['S'] = pd.DataFrame(cons['L'].values + cons['R'].values,
                             index=cells['S'], columns=cells['S'])
    # the mean connectivity
    cons['M'] = pd.DataFrame(cons['S'].values/2,
                             index=cells['M'], columns=cells['M'])

    for con in cons.values():
        con.columns.name = 'Post cells'
        con.index.name = 'Pre cells'

    if keys is None:
        return cons
    else:
        return {k: cons[k] for k in keys}


def get_con_data(keys=['L', 'R', 'M']):
    cons = {}
    for k in keys:
        cons[k] = pd.read_hdf(f'{PATH}results/cons/cons_full_{k}.hdf')
    return cons


def create_summary_LNs(cons1, func='mean', S='M'):
    """
    this function adds columns and lines in the connectivity matrix
    which correspond to the sum/mean of broad trios, duets and keystones,
    picky (averaging the dendrite and axon) and choosy (averaging the dend and
    axons).
    
    """
    # without the following line, the function changes the input cons
    cons = copy.deepcopy(cons1)
    if func == 'mean':
        f = pd.DataFrame.mean
    elif func == 'sum':
        f = pd.DataFrame.sum
    else:
        raise ValueError(f'func should be \'mean\' or \'sum\', got {func}')

    for s in cons.keys():
        # for broads
        (B_T1, B_T3) = (f'Broad T1 {s}', f'Broad T3 {s}')
        (B_D1, B_D2) = (f'Broad D1 {s}', f'Broad D2 {s}')

        cons[s].loc[:, f'Broad T {S} {s}'] = f(cons[s].loc[:, B_T1: B_T3],
                                               axis=1)
        cons[s].loc[:, f'Broad D {S} {s}'] = f(cons[s].loc[:, B_D1: B_D2],
                                               axis=1)

        cons[s].loc[f'Broad T {S} {s}'] = f(cons[s].loc[B_T1: B_T3])
        cons[s].loc[f'Broad D {S} {s}'] = f(cons[s].loc[B_D1: B_D2])

        for i in range(5):
            (P01, P02) = (f'Picky {i} [dend] {s}', f'Picky {i} [axon] {s}')
            cons[s].loc[:, f'Picky {i} {S} {s}'] = f(cons[s].loc[:, P01: P02],
                                                     axis=1)
            cons[s].loc[f'Picky {i} {S} {s}'] = f(cons[s].loc[P01: P02])

        for i in range(1, 3):
            (P01, P02) = (f'Choosy {i} [dend] {s}', f'Choosy {i} [axon] {s}')
            cons[s].loc[:, f'Choosy {i} {S} {s}'] = f(cons[s].loc[:, P01: P02],
                                                  axis=1)
            cons[s].loc[f'Choosy {i} {S} {s}'] = f(cons[s].loc[P01: P02])

        # for keystones
        (key1, key2) = (f'Keystone L {s}', f'Keystone R {s}')
        cons[s].loc[:, f'Keystone {S} {s}'] = f(cons[s].loc[:, key1: key2],
                                                axis=1)
        cons[s].loc[f'Keystone {S} {s}'] = f(cons[s].loc[key1: key2])

#    s = 'S'
#    key1, key2 = ('Keystone L', 'Keystone R')
#    cons[s].loc[:, 'Keystone ' + s] = cons[s].loc[:, key1: key2].sum(axis=1)
#    cons[s].loc['Keystone ' + s] = cons[s].loc[key1: key2].sum()
#
    return cons


# probably this function and the one above can be easily merged,
# but don't want to go through the trouble now
# =============================================================================
# def create_mean_broads_keystone(cons):
#     """
#     this function adds columns and rows in the connectivity matrix
#     which correspond to the mean of broad trios, duets and keystoes
#     """
#     for s in cons.keys():
#         # for broads
#         (BT1, BT3) = ('Broad T1 ' + s, 'Broad T3 ' + s)
#         (BD1, BD2) = ('Broad D1 ' + s, 'Broad D2 ' + s)
#         cons[s].loc[:, f'Broad T {s}'] = cons[s].loc[:, BT1:BT3].mean(axis=1)
#         cons[s].loc[:, f'Broad D {s}] = cons[s].loc[:, BD1:BD2].mean(axis=1)
#         cons[s].loc['Broad T ' + s] = cons[s].loc[BT1: BT3].mean()
#         cons[s].loc['Broad D ' + s] = cons[s].loc[BD1: BD2].mean()
# 
#         # for keystones
#         (key1, key2) = ('Keystone L ' + s, 'Keystone R ' + s)
#         cons[s].loc[:, 'Keystone M ' + s] = (cons[s].loc[:, key1: key2]
#                                                .mean(axis=1))
#         cons[s].loc['Keystone M ' + s] = cons[s].loc[key1: key2].mean()
# 
#     return cons
# =============================================================================


def merge_con_data(con_L, con_R):
    """
    this creates one huge matrix from the left and right datasets
    """
    func = np.max  # some of the data is repeated from cell to cell
    shape = con_L.shape
    con = np.block([[con_L, np.zeros(shape)], [np.zeros(shape), con_R]])
    all_cells = np.concatenate((list(con_L.index), list(con_R.index)))
    con = pd.DataFrame(con, index=all_cells, columns=all_cells)

    con = con.groupby(by=[con.index], axis=0).transform(func)
    con = con.groupby(by=[con.columns], axis=1).transform(func)
    con = con.loc[~con.index.duplicated(keep='first')]
    con = con.loc[:, ~con.columns.duplicated(keep='first')]

    # these 2 lines are just to check that you did the merge correctly
    print('errors when merging:',
          np.abs(con_R - con.loc[con_R.index, con_R.columns]).max().max(),
          np.abs(con_L - con.loc[con_L.index, con_L.columns]).max().max())

    # i would still like to control the following: that in duplicates:
    # either they are the same, either one is zero
    # for that you can choose groups that are larger than 1, and then
    # somehow compare

    # the following lines work but it messes up the order of
    # cells, which is a bit annoying
    # con = con.groupby(by=[con.index], axis = 0).first()
    # con = con.groupby(by=[con.columns], axis = 1).first()
    return con


def get_con_pps(con, labels):
    """
    this function creates all the streams of connections
    0: feedforward, from labels to all others
    1: feedback, from all others to labels
    2: feedback, with some normalization, using the total input to that label
    3: average of feedforward and feedback, i.e., of 0 and 1
    Parameters
    ----------
    con
    labels

    Returns
    -------

    """
    X2A = con.loc[labels]  # selection to all
    A2X = con.loc[:, labels]  # all to selection
    X_A = X2A + A2X.T  # mean connection (maybe you would even need to do
    # an other type of normalization...)

    (X2A_c, X2A_n, X2A_cn) = FG.get_ctr_norm(X2A, opt=2)
    (A2X_c, A2X_n, A2X_cn) = FG.get_ctr_norm(A2X.T, opt=2)
    (X_A_c, X_A_n, X_A_cn) = FG.get_ctr_norm(X_A, opt=2)

    # divide by the indegree
    A2X2 = A2X/A2X.sum()
    (A2X2_c, A2X2_n, A2X2_cn) = FG.get_ctr_norm(A2X2.T, opt=2)

    return {0: {'o': X2A, 'c': X2A_c, 'n': X2A_n, 'cn': X2A_cn},
            1: {'o': A2X.T, 'c': A2X_c, 'n': A2X_n, 'cn': A2X_cn},
            2: {'o': A2X2.T, 'c': A2X2_c, 'n': A2X2_n, 'cn': A2X2_cn},
            3: {'o': X_A, 'c': X_A_c, 'n': X_A_n, 'cn': X_A_cn}}

# =============================================================================
# def get_con_pps_x(con, labels):
#     X2A = con.loc[labels]  # selection to all
#     A2X = con[labels]  # all to selection
#
#     (X2A_c, X2A_n, X2A_cn) = FG.get_ctr_norm(X2A, opt=2)
#     (A2X_c, A2X_n, A2X_cn) = FG.get_ctr_norm(A2X.T, opt=2)
#
#     # divide by the indegree
#     A2X2 = A2X/A2X.sum()
#     (A2X2_c, A2X2_n, A2X2_cn) = FG.get_ctr_norm(A2X2.T, opt=2)
#
#     return {0: {'o': X2A, 'c': X2A_c, 'n': X2A_n, 'cn': X2A_cn},
#             1: {'o': A2X, 'c': A2X_c.T, 'n': A2X_n.T, 'cn': A2X_cn.T},
#             2: {'o': A2X2, 'c': A2X2_c.T, 'n': A2X2_n.T, 'cn': A2X2_cn.T}}
# =============================================================================


def get_con_norm_by_in_degree(A2X):
    A2X2 = A2X/A2X.sum()
    A2X2_c = A2X2.subtract(A2X2.mean(axis=1), axis=0)
    A2X2_cn = A2X2_c.divide(LA.norm(A2X2_c.values, axis=1), axis=0)
    return A2X2, A2X2_c, A2X2_cn


# #############################################################################
# ########################### PLOTTING CONNECTIVITY ###########################
# #############################################################################

def plot_gram(data, ax=None, name='', splits=[], fs=(13, 10), adj=0.2,
              ctr=True):
    """
    this function does not use the latest technologies of plotting,
    i.e., not using the function imshow_df which already has all the needed
    functionality
    """
    # data_c = data # if you don't want normalization
    cmap = plt.bwr
    if ctr:
        data = FG.get_ctr_norm(data)
        vmin, vmax = (-1, 1)
        gram = FG.get_corr(data, data)
    else:
        data = FG.get_norm(data)
        gram = FG.get_cos_sim(data, data)
        vmin, vmax = (np.nanmin(gram.values), np.nanmax(gram.values))

    print(vmin, vmax)

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fs)
        # print('creating new figure in plot_gram')
    plt.sca(ax)
    plt.subplots_adjust(bottom=adj)
    masked_array = np.ma.array(gram, mask=np.isnan(gram))

    cmap.set_bad('black', 1.)
    cp = ax.imshow(masked_array, cmap=cmap, vmin=vmin, vmax=vmax)
    for s in splits:
        ax.plot([-0.5, len(data.columns)-0.5], [s-0.5, s-0.5], c='k')
        ax.plot([s-0.5, s-0.5], [-0.5, len(data.columns)-0.5], c='k')
    plt.xticks(np.arange(len(gram)), list(gram.index), rotation='vertical')
    plt.yticks(np.arange(len(gram)), list(gram.index))
    ax.set_title('grammian' + name)

    plt.colorbar(cp, ax=ax)
    return plt.gcf(), ax


def plot_con_full(save_plots, plot_plots, path_plots,
                  cell_sort=par_con.ORN, cell_name='ORN',
                  sides=['L', 'R', 'M'], ppss=['o', 'c', 'n', 'cn'],
                  strms=[0, 1, 2, 3], show_values=True, show_n_syn=True):
    """
    Plots all the connection directions and all the connection pps,
    e.g., o, c, n, cn (original, centered, normalized, cent-normalized)
    for the cells cell_sort
    """
    print(f"""sides to plot {sides}\npps to plot {ppss}\n
          streams to plot {strms}""")
    FP.set_plotting(plot_plots)

    # ################  ANALYSIS  #########
    cons = get_con_data()

    # calcualting the number of connections and synapses.
    # It is something that is relevant only for streams 0, 1, 3
    # and for L, R, M, and for pps 'o'

    # n_con containts the number of connections from or towards
    # any cell with the ORN/uPN set
    # n_syn containts the number of connections from or towards
    # any cell with the ORN/uPN set

    mi = pd.MultiIndex(levels=[[], []], codes=[[], []],
                       names=['side', 'stream'])

    n_con = pd.DataFrame(columns=mi)
    n_syn = pd.DataFrame(columns=mi)  # this is the number of synapses

    for side in sides:
        chosen_cells = FG.get_match_names(cell_sort, list(cons[side].index))
        con_sel = get_con_pps(cons[side], chosen_cells)

        for strm in strms:
            con_o = con_sel[strm]['o']
            n_con[side, strm] = np.sign(con_o).sum().values.astype(int)
            n_syn[side, strm] = con_o.sum().values.astype(int)

    # con_m = of.merge_con_data(con_L, con_R)

    titles = [f'connections from {cell_name} to all',
              f'connections from all to {cell_name}',
              f'connections from all to {cell_name}, scaled by in-degree',
              f'connections of {cell_name} with all',
              ]

    # iterating over the different "sides" (L, R, M)
    for side in sides:
        chosen_cells = FG.get_match_names(cell_sort, list(cons[side].index))
        print(chosen_cells)
        con_sel = get_con_pps(cons[side], chosen_cells)
        # 0: sel2A, 1: A2sel, 2:A2sel: normalized by the input of each neuron

        # plots of the connectivity, using lines
# =============================================================================
#     of.plot_con_data(ORN2A, A2ORN)
#     of.plot_con_data(ORN2A_cn, A2ORN_cn)
# =============================================================================

        # iterating over the different pps of the connections
        # a different figure/file for each
        for pps in ppss:
            # if pps_key != 'o':
            #     continue
            # plots of the cells connectivity, using imshow
            f, axx = plt.subplots(len(strms), 1, figsize=(25, 22))
            # iterating over the 4 streams (ff, fb, fb-indegr, (ff+fb))
            # which are all in the same figure
            for i, strm in enumerate(strms):
                FP.imshow_df(con_sel[strm][pps], ax=axx[i],
                             title=f'{titles[strm]}, {pps}',
                             show_values=show_values)
                
                if show_n_syn:
                    # adding the information about the number of connections
                    # and the number of synapses above the heatmap
                    x_n = len(n_con.index)
                    for j in range(x_n):
                        axx[i].text((j + 0.5)/x_n, 1.05,
                           (f'{n_con[side, strm].loc[j]},'
                           f' {n_syn[side, strm].loc[j]}'),
                           ha='center', va='bottom',
                           transform=axx[i].transAxes, rotation=90)
            plt.tight_layout()
            FP.save_plot(f, f'{path_plots}{cell_name}_con_{side}_{pps}.png',
                         save_plots)


# #############################################################################
# ################  PLOTTING SVD ANALYSIS OF CONNECTIONS  #####################
# #############################################################################

# need to check the resemblance with the SVD/PCA of the activity function
def plot_SVD_con(data, sets=[], vlim=None, title=''):
    SVD1 = FG.get_svd_df(data)
    SVD1_s = np.diag(SVD1['s'])
    x = SVD1_s[0] * SVD1['Vh'].iloc[0]
    y = SVD1_s[1] * SVD1['Vh'].iloc[1]
    z = SVD1_s[2] * SVD1['Vh'].iloc[2]
    print(SVD1_s)

    if vlim is None:
        vlim = max([x.abs().max(), y.abs().max(), z.abs().max()])*1.1

    if len(sets) == 0:
        sets = [np.arange(len(data.T))]

    f, axx = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle(title)

    FP.imshow_df(data, ax=axx[0, 0])

    plot_gram(data, ax=axx[0, 1])

    ax = axx[1, 0]
    ax.plot(np.arange(1, 1+len(SVD1_s)), SVD1_s)
    ax.set_xlabel('principal component number')
    plt.sca(ax)
    plt.xticks(np.arange(1, 1+len(SVD1_s)))
    ax.set_ylim(0, None)

    ax = axx[1, 1]
    for s in sets:
        ax.scatter(x.iloc[s], y.iloc[s])
    ax.set_xlim(-vlim, vlim)
    ax.set_ylim(-vlim, vlim)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    for cell_label in list(SVD1['Vh'].columns):
        ax.annotate(cell_label, (x.loc[cell_label], y.loc[cell_label]))

    ax = axx[1, 2]
    for s in sets:
        ax.scatter(y.iloc[s], z.iloc[s])
    ax.set_xlim(-vlim, vlim)
    ax.set_ylim(-vlim, vlim)
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC3')
    for cell_label in list(SVD1['Vh'].columns):
        ax.annotate(cell_label, (y.loc[cell_label], z.loc[cell_label]))

    axx[0, 2].axis('off')
    plt.tight_layout()
    return f, axx


def get_ORN_act_data_2(dropna: bool = True, fast: bool = True) -> pd.DataFrame:
    """
    this function reads out the data from the second provided dataset, which
    has more odors, and for which all the ORNs have been activated
    The initial data was provided in a mat file, but it was exported to a
    csv file so that python could read it, with separators as ';', because
    other separators like spaces and commas are present in the odor names
    The dropna option is weather or not to drop the rows that have NA
    The lines that have NA means that actually all the ORNs were not
    recorded simultaneously.
    the option fast reads the stored copy of the activity, it is quicker
    than reading it from scratch from the csv file
    """
    if fast:
        act_ORN_df = pd.read_hdf(par_act2.file_hdf)
    else:
        # reading the raw table
        # would be good to compare with the calculation they did...
        # needed to use the ';' separator because there are commas in the odors
        act_ORN_df = pd.read_csv(OLF_PATH / par_act2.file_tbl, sep=';',
                                 index_col=[0, 1, 2])
        # Changing the names form ['Odor', 'Exp_ID', 'Concentration']
        act_ORN_df.index.names = ['odor', 'exp_id', 'conc']

        concs_new = -np.log10(act_ORN_df.index.levels[2]).astype(int)
        act_ORN_df.index = act_ORN_df.index.set_levels(concs_new, 'conc')

        ORNs = np.array(act_ORN_df.columns)
        # the next line removes the columns.name, so it needs to come after
        act_ORN_df.columns = FG.replace_substr_np(ORNs,
                                                  {'_': '/', 'Or': 'ORN '})

        # correcing small typos in the original dataset
        act_ORN_df.rename(inplace=True,
            index={'4-pheny-2-butanol': '4-phenyl-2-butanol',
                   '4-methylcyclohexane': '4-methylcyclohexanol'})
    # whether or not to drop the rows that have NA
    if dropna:
        act_ORN_df = act_ORN_df.dropna()

    act_ORN_df.columns.name = 'cell'

    # instead of just dropping the values that are not available, we could also
    # combine them from different trials... however i am not sure that it is
    # what you want. We can just leave it as it is right now.
    print(act_ORN_df.columns.name)
    print(act_ORN_df.index.names)

    return act_ORN_df


def get_ORN_act_data_3(dropna: bool = False, fast: bool = True)\
        -> pd.DataFrame:
    """
    We start with dataset 2, with nan values that were kept
    then we calculate the mean of the dataset, in order to use the mean
    values to populate the nan values of the available initial dataset
    This dataset is quite redundant, because the mean values are
    repeated several times.

    Parameters
    ----------
    dropna:
        not used here
    fast:
        if True: reads the stored copy of the activity, which is quick
        if False: reads the data from scratch from the csv file and then
        calculates the mean, and the filling all the missing values

    Return
    ------
    act: pd.DataFrame
        activity of dataset 3

    """

    if fast:
        act = pd.read_hdf(OLF_PATH / par_act3.file_hdf)
        # the object stored is a DataFrame. Must be a clean way to check
        # that it is indeed the case i imagine
        return act

    # this code is for fast==False
    act = get_ORN_act_data_2(dropna=False, fast=False)
    # raw dataset, nothing averaged

    # removing the 9, 10, 11 concentrations, as they are not used later anyway
    act = act.loc[(slice(None), slice(None), [4, 5, 6, 7, 8]), :]
    # act = act.dropna(thresh=10)  # if there are more than 5 nan, the row is
    # dropped
    # rows_nans = act.isna().sum(axis=1)>5

    # using the mean dataset based on the dataset 2
    act_mean = act.mean(level=('odor', 'conc'))
    act_mean = ORN_act_populate_missing_values(act_mean)
    act_mean = act_mean.dropna(axis=0)  # removes any row that contains nan
    # now we have the nice averages. We want to now populate all the missing
    # values in the raw dataset act

    # now i would like to itereate over all the values of act, and if there
    # is a nan, replace it by the value given in act_mean

    cells = list(act.columns)

    # iterating through all the cells and odors and concentrations and trials
    # maybe there is a faster way, but this is the most straight forward
    for cel in cells:
        for i in range(len(act.index)):
            if pd.isnull(act.loc[:, cel].iloc[i]):
                idx = act.iloc[i].name[0: 3: 2]
                act.loc[:, cel].iloc[i] = act_mean.loc[idx, cel]

    return act


ORN_data_func_dict = {#1: get_ORN_act_data_1,
                      2: get_ORN_act_data_2,
                      3: get_ORN_act_data_3}


def get_ORN_act_data(dataset: int, dropna: bool = True, fast: bool = True) \
        -> pd.DataFrame:
    """
    i don't sure we really need to maintain the first dataset
    because the first dataset is just anyway a subset of the first subset
    maybe at some moment we can stop maintaining it

    Parameters
    ----------
    dataset:
        which dataset, can be 1, 2 or 3
    dropna:
        removed the stimuli with unavailable data
    fast:
        if True: reads the stored copy of the activity, which is quick
        if False: reads the data from scratch from the csv file and then
        calculates the mean, and the filling all the missing values

    Return
    ------
    act: pd.DataFrame
        activity of dataset
    """
    return ORN_data_func_dict[dataset](dropna=dropna, fast=fast)



def get_cell_names_act(sheet, n_begin, n, n_row):
    cells_act = np.array(FG.get_labels_row(sheet, n_begin, n, n_row),
                         dtype='U30')
    cells_act = FG.replace_substr_np(cells_act, {'_': '/', 'Or': 'ORN '})
    return cells_act


def remove_inactive_neurons(act_df: pd.DataFrame) -> pd.DataFrame:
    # this is faster than using the pandas sum
    active_neurons = act_df.values.sum(axis=0) != 0
    # print('number of active neurons: ', sum(active_neurons))
    act_df = act_df.loc[:, active_neurons]
    # print(data.shape)
    return act_df


def get_normalized_act_per_trial(act_df: pd.DataFrame) -> pd.DataFrame:
    """
    we use dataframees to do the normalization
    the first step is to find the maximum in each trial
    a trial is a given odor and a given experiment id
    Parameters
    ----------
    act_df

    Returns
    -------

    """
    max_per_exp = act_df.max(axis=1).max(level=('odor', 'exp_id'))
    # finding the maximum among all the neurons and
    # all the concenctration for a given exp_id and odor
    # maybe it is possible to use groupby here?
    data_normalized = act_df.unstack(level='conc')
    data_normalized = data_normalized.divide(max_per_exp, axis=0).stack()
    # moving the level of experiemets fo the columns to make the division
    return data_normalized



def get_ORN_act_pps(dataset: int, pps1: str, pps2: str, dropna: bool = True,
                    fast: bool = True) -> pd.DataFrame:
    """
    the first paramater can be tn, raw or model
    tn is trial normalized, what is done in the paper
    raw means that we don't change the data, just inverting 2 levels
    model, means that we produce the data from the hill equation and ec50 data

    the second parameter can be mean, raw, l2
    mean: we average among trials
    raw, we don't change anything
    l2 we normalize each response

    the usual settings are:
    tn, mean - as in the paper
    raw, raw, taking all the raw data
    raw, mean, taking the raw data and averaging among trials
    model, raw: model data produced from ec50
    """
    act0 = get_ORN_act_data(dataset, dropna=dropna, fast=fast)
#    print(act0.columns.name)
#    print(act0.index.names)
    # act1 is the raw data, there are many ways we can normalize the data
    # before analysing it

    if pps1 == 'tn':
        # normalizing each trial by the maximum response
        # this is not used further on
        act1 = get_normalized_act_per_trial(act0)
    elif pps1 == 'raw':
        act1 = act0.copy()
        act1 = act1.swaplevel('conc', 'exp_id')
    # elif pps1 == 'model':
    #     # generating data using the hill function and EC50 data
    #     # in this cases there are no tirals, only conc and odors
    #     act1 = get_ORN_act_from_EC50(dataset)
    else:
        print('act_pps1 is not well defined')
        sys.exit(0)

    if pps2 == 'mean':
        # averaging data between the repetitions
        act2 = act1.mean(level=('odor', 'conc'))
    elif pps2 == 'raw':
        act2 = act1.copy()
    elif pps2 == 'l2':
        act2 = act1.copy()
        act2 = (act2.T / LA.norm(act2, axis=1)).T
        # removing all the places where the activity was 0
        # as it causes NA because of the division
        act2 = act2.dropna(axis=0, how='any')
    else:
        print('act_pps2 is not well defined')
        sys.exit(0)

#    print(act2.columns.name)
#    print(act2.index.names)

    return act2


def ORN_act_populate_missing_values(act: pd.DataFrame) -> pd.DataFrame:
    """
    this function fills the missing values at concentration 6, 5, 4
    from the activity dataset 2, after averaging.
    We put the saturated value for odors:
    - 2-heptanone
    - methyl salicytate
    and cells 85c and 22c, respectively
    We ignore the low concentrations (which go down to 10^-11)

    We use a function that fills values for a given concentration
    by a given value. This function first checks that the previous values
    were nan and that it is not overwriting anything.
    """

    act_filled = act.copy()
    odors = ['2-heptanone', 'methyl salicylate']
    cells = ['ORN 85c', 'ORN 22c']
    # would be good that the function finds itself the cell that
    # needs to be filled

    concs = [4, 5, 6]  # concentrations that need to be filled
    for od, cell in zip(odors, cells):
        val = np.nanmax(act_filled.loc[od, cell].values)
        # print(val)
        act_filled = fill_values_act(act_filled, od, cell, concs, val)

    # when all odors are present, we might need to create a new
    # parameter file which contains the updated list of odors
    # and maybe also a new order of ORNs, if adding those odors has an effect
    return act_filled


def fill_values_act(act, odor, cell, concs, val) -> pd.DataFrame:
    """
    This function fills in the activity data, for an odor 'odor', above the
    concentration conc, the value that is used is for the concentration
    just below
    """
    # the first step is to check that indeed the values where we are going
    # to put values are nan, so that we don't overwrite anything
    if act.loc[(odor, concs), cell].isnull().all():
        act.loc[(odor, concs), cell] = val
    return act


# maybe would make it even easier to fill automatically by making a recurrent
# function:
# find a nan, in the data. look at the previous concentration. If it exists
# use it, if not go to a concentration lower. If there are not concentration
# lower, put 0. During the process, tell which cells and odors you are filling


# ################## ACTIVITY SVD ANALYSIS FUNCTIONS ##########################


def get_SVD_cells(data_sets, PCs=np.arange(1, 3)) -> pd.DataFrame:
    SVDs = {k1: {k2: FG.get_svd_df(v2.T) for k2, v2 in v1.items()}
            for k1, v1 in data_sets.items()}
# here before we had the sorting, i don't think we need it now...

# putting in a single vector all the principal vectors we are going to use
    mi = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []],
                       names=['decomp', 'conc', 'pps', 'n'])
    U_all = pd.DataFrame(columns=mi)
    for k1, v1 in SVDs.items():
        for k2, v2 in v1.items():
            for PC in PCs:
                U_all['PCA', k1, k2, PC] = v2['U'][PC]
    return U_all


# ##########################  NMF ANALYSIS  ###################################
# this is not very flexible, especially if one want to have different
# preprocessing for different betas and different initializations for
# differnet betas...
def get_NMF_set(data_sets, N=2, betas=[0.01, 2, 3],
                inits=['nndsvd', 'nndsvda'], rnd=None) -> pd.DataFrame:
    """
    data_sets should be a dictionary of the following form:
    data_sets['concentration/dataset']['preprocessing']
    """
    Ns = np.arange(N)+1

    mi = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []],
                       names=['decomp', 'conc', 'pps', 'n'])
    U_all = pd.DataFrame(columns=mi)
    for (beta, init) in itertools.product(betas, inits):
        for conc, datas in data_sets.items():
            for pps, data in datas.items():
                data1 = FG.rectify(data)
                # data1 = data-data.min().min()
                _, H, _, _ = nmf.get_nmf_df(data1, k=N, beta=beta,
                                                      init=init, rnd=rnd)

                # for easier comparison, we will order them in a specific order
                # sorted by the first value
                H = FG.get_ctr_norm(H.T, opt=2)[1]
                # H = H.sort_values(by=H.index[0], axis=1)

                for n in Ns:
                    U_all['NMF_' + str(round(beta)) + '_' + init,
                          conc, pps, n] = H.iloc[:, n-1]
    return U_all


# ##########################  PLOTTING ACTIVITY DATA  #########################

def plot_activity_old(act_df, cell_order, odor_order, norm=True, title=''):
    """
    Plot the activity
    the rows are stimuli (odors), the columns are ORNs
    (just like in the plot in the paper)
    a plot for each concentration
    if norm is True, then it is the same scaling for each concentration
    if norm is False, then each concentration has its own colorbar
    not yet tested with the new dataset
    """
    print(cell_order)
    print(odor_order)
    act_df = act_df[cell_order].reindex(odor_order, level='odor')

    conc_list = act_df.index.get_level_values('conc').unique()
    odor_list = act_df.index.get_level_values('odor').unique()
    n_conc = len(conc_list)
    cell_list = list(act_df)
    # cell_list = [name.strip(' ORN L&R') for name in cell_list]

    vmax = act_df.max(axis=0, level='conc').max(axis=1)
    vmin = act_df.min(axis=0, level='conc').min(axis=1)
    if norm is True:
        vmax[:] = vmax.max()
        vmin[:] = vmin.min()
        n_plots = n_conc+1
    else:
        n_plots = n_conc

    f, axx = plt.subplots(1, n_plots, figsize=(n_plots*2, len(odor_list)/10+1))
    for i in range(n_conc):
        ax = axx[i]
        cp = ax.imshow(act_df.xs(conc_list[i], level='conc'),
                       vmin=vmin.iloc[i], vmax=vmax.iloc[i], cmap='jet')
        plt.sca(ax)
        plt.xticks(np.arange(len(cell_list)), cell_list, rotation='vertical')

        # putting the odor labels only on the most left graph
        if i == 0:
            ax.set_yticks(np.arange(len(odor_list)))
            ax.set_yticklabels(odor_list)
        else:
            ax.set_yticks([])
        ax.set_title('concentration 1e-' + str(conc_list[i]))
        if norm is False:
            f.colorbar(cp, ax=ax)
    if norm is True:
        ax = axx[-1]
        ax.axis('off')
        f.colorbar(cp, ax=ax)

    plt.tight_layout()
    plt.suptitle(title)
#    plt.show()
    return f, axx


def plot_activity(act_df, cell_order, odor_order, norm=True, title='',
                  cmap=plt.cm.viridis):
    """
    Plot the activity
    the rows are stimuli (odors), the columns are ORNs
    (just like in the plot in the paper)
    a plot for each concentration
    if norm is True, then it is the same scaling for each concentration
    if norm is False, then each concentration has its own colorbar
    not yet tested with the new dataset
    """
    if not norm:
        plt.rc('font', size=5)
    else:
        plt.rc('font', size=6)
    print(cell_order)
    print(odor_order)
    act_df = act_df[cell_order].reindex(odor_order, level='odor')

    conc_list = act_df.index.get_level_values('conc').unique()
    print(conc_list)
    odor_list = act_df.index.get_level_values('odor').unique()
    n_conc = len(conc_list)
    # cell_list = list(act_df)
    # cell_list = [name.strip(' ORN L&R') for name in cell_list]

    vmax = act_df.max(axis=0, level='conc').max(axis=1)
    vmin = act_df.min(axis=0, level='conc').min(axis=1)
    if norm is True:
        vmax[:] = vmax.max()
        vmin[:] = vmin.min()
    n_plots = n_conc

    f_w = n_plots*1.7
    f_h = len(odor_list)/12+0.4

    f, axx = plt.subplots(1, n_plots, figsize=(f_w, f_h))
    for i in range(n_conc):
        if n_conc > 1:
            ax = axx[i]
        else:
            ax = axx
        FP.imshow_df(act_df.xs(conc_list[i], level='conc'), ax=ax,
                     cmap=cmap, vlim=[vmin.iloc[i], vmax.iloc[i]],
                     title='concentration 1e-' + str(conc_list[i]),
                     cb=not norm, cb_frac=0.066)
        ax.set_xlabel('')
        if i > 0:
            ax.set_yticks([])
            ax.set_ylabel('')

    if norm is True:
        ri = 1 - 0.3/f_w
        bot = 0.7/f_h
        top = 1-0.4/f_h
        plt.subplots_adjust(left=1.3/f_w, bottom=bot,
                            right=ri, top=top, wspace=0.01/f_w)
        cbaxes = f.add_axes([ri + 0.1/f_w, bot, 0.1/f_w, top-bot])
        scale = mpl.colors.Normalize(vmin=vmin.iloc[0],
                                            vmax=vmax.iloc[0])
        clb = mpl.colorbar.ColorbarBase(cbaxes, norm=scale, cmap=cmap)
        clb.outline.set_linewidth(0.00)
        clb.ax.tick_params(size=2, direction='in', pad=1.5)
    else:
        plt.subplots_adjust(left=1.3/f_w, bottom=0.2,
                            right=1 - 0.3/f_w, top=0.9, wspace=1.4/f_w)
        plt.rc('font', size=5)
# =============================================================================
#         cp = ax.imshow(act_df.xs(conc_list[i], level='conc'),
#                        vmin=vmin.iloc[i], vmax=vmax.iloc[i], cmap='jet')
#         plt.sca(ax)
#         plt.xticks(np.arange(len(cell_list)), cell_list, rotation='vertical')
# 
#         # putting the odor labels only on the most left graph
#         if i == 0:
#             ax.set_yticks(np.arange(len(odor_list)))
#             ax.set_yticklabels(odor_list)
#         else:
#             ax.set_yticks([])
#         ax.set_title('concentration 1e-' + str(conc_list[i]))
#         if norm is False:
#             f.colorbar(cp, ax=ax)
#     if norm is True:
#         ax = axx[-1]
#         ax.axis('off')
#         f.colorbar(cp, ax=ax)
# =============================================================================

    # plt.tight_layout()

    plt.suptitle(title)
#    plt.show()
    plt.rc('font', size=6)
    return f, axx


def plot_activity2(act_df, cell_order, odor_order, title=''):
    """
    here we plot all the responses in one graph
    the rows are different cells
    the columns are different stimuli
    in the top graph we plot the stimuli ordered by odors
    in the lower graph, we plot stimuli first ordered by concentration
    for each concentration, it looks just like a flipped version of the plots
    above.
    """
    plt.rc('font', size=6)
    f, axx = plt.subplots(2, 1, figsize=(11, 6))

    act_df = act_df[cell_order].reindex(odor_order, level='odor')
    FP.imshow_df(act_df.T, ax=axx[0], cmap=plt.cm.jet)

    act_df = act_df.sort_index(axis=0, level='conc')
    FP.imshow_df(act_df.T, ax=axx[1], cmap=plt.cm.jet)
    plt.tight_layout()
    plt.suptitle(title)
    return f, axx



# #############################################################################
# #############################################################################
# ###########################  CLASSES DEFINITIONS   ##########################
# #############################################################################


# ###############################  LN ANALYSIS   ##############################

def plot_hist_cor_pred_vs_fake(pred, fake, verbose=True):
    """
    pred and fake must be a matrices with the same number of columns
    Plots 2 graphs on 1 figure. One is the distribution of the fake
    correlation coefficients and on top the predicted ones
    The other plots is the distribution of the max for each row
    """

    if pred.shape[1] != fake.shape[1]:
        print("shouldn't pred and fake have the same number of columns?")
        return

    pred_max = pred.max(axis=1)
    fake_max = fake.max(axis=1)

    ttest_all = sps.ttest_ind(pred.flatten(), fake.flatten())
    ttest_max = sps.ttest_ind(pred_max.flatten(), fake_max.flatten())

    f, axx = plt.subplots(1, 2, figsize=(5, 2.5))
    ax = axx[0]

    ax.hist(fake.flatten(), density=True, bins=100,
            label='from shuffled data')
    ax.hist(pred.flatten(), density=True, histtype='step', bins=20,
            label='from prediction')
    ax.set_title('all corr. coef.')
    ax.set_ylabel('density')
    ax.legend(prop={'size': 8}, loc=3)
    ax.text(0.03, 0.8, 'p-value: \n' + '{:.0e}'.format(ttest_all.pvalue),
            transform=ax.transAxes)

    ax = axx[1]
    ax.hist(fake_max, density=True, bins=100)
    ax.hist(pred_max, density=True, histtype='step',
            bins=10)
    ax.set_title('best corr. coef.')
    ax.text(0.03, 0.80, 'p-value: \n' + '{:.0e}'.format(ttest_max.pvalue),
            transform=ax.transAxes)

    if verbose is True:
        print('ttest all:')
        print(ttest_all)
        print('ttest max:')
        print(ttest_max)

    return f, axx


def ctr_divide(x):
    x = FG.get_ctr_norm(x, opt=1)[0]
    return x/((x**2).sum())


# #####################  ORN/PN ACT VS ACT  ###################################

# =============================================================================


# as i will now do everything in a class, i think it makes sense to separate
# some stuff of what was done before in a single instance.
# before i was trying to analyse all at once, but it was creating
# some difficulties. for example analyzing ORNs and uPNs at once was not easy
# to manage, so maybe it would make sense to have 2 instances of this in a
# class
# i don't remember if it makes any sense at all to analyse the 2 cell types
# at the same time. Is there any analysis that is at all in common?
# i guess since things are getting a bit more complicated, it would be good
# to have a plan/description of what we want to achieve, so that we can
# code in a better way.

class NeurActConAnalysis:
    '''
    this class is relevant for ORNs and for uPNs
    '''

    sides = ['L', 'R', 'M']

    def __init__(self, dataset: int, cell_type: str, strms: List[int],
                 con_pps_k: str, save_plots: bool, plot_plots: bool,
                 act_sel_ks: List[str], act_pps_k1: str, act_pps_k2: str,
                 neur_order: List[str] = None, odor_order: List[str] = None,
                 odor_sel: List[Tuple[str, str]] = None,
                 path_plots: str = None, reduce: bool = False,
                 subfolder: str = 'act_ORN_vs_con_ORN-LN/') -> Any:
        """
        after the initialization con_strms, con_strms2 and act will
        all have the same ordering of neurons, and it will be ordered by
        neur_order, if neur_order is empty, the order will me imposed by
        the order that is already in the activity matrix

        the reduce option is for the case when the 2 act_pps options are
        'raw', and we are taking the same number of points (trials)
        for each odor

        Parameters
        ----------
        dataset
        cell_type
        strms
        con_pps_k
        save_plots
        plot_plots
        act_sel_ks
        act_pps_k1
        act_pps_k2
        neur_order
        odor_order
        odor_sel
        path_plots
        reduce
        subfolder
        """


        # plotting default
        plt.rc('font', size=6)

        self.dataset = dataset
        print('dataset used: ', str(self.dataset))

        self.save_plots = save_plots
        self.plot_plots = plot_plots
        self.path_plots = path_plots

        if self.save_plots is True:
            if path_plots is None:
                self.path_plots = FG.create_path(OLF_PATH / f'plots/{subfolder}')
            print(f'plots will be saved in {self.path_plots}')
        else:
            print('plots will not be saved')

        FP.set_plotting(self.plot_plots)

        self.cell_type = cell_type
        self.strms = strms  # stream, i.e., connectivity directions
        self.con_pps_k = con_pps_k
        self.fs = (16, 12)

        print(f'cell type: {self.cell_type}\n'
              f'streams: {self.strms}\n'
              f'connectivity data pps: {self.con_pps_k}')

        # #####################################################################
        # ##############  INITIALIZING THE ACTIVITY DATA  #####################
        # #####################################################################
        # here we will need to have some options about choosing the ORN dataset
        self.act_pps_k1 = act_pps_k1
        self.act_pps_k2 = act_pps_k2
        if self.cell_type == 'ORN':
            self.act = get_ORN_act_pps(self.dataset, act_pps_k1, act_pps_k2,
                                       fast=True)
            # the fast option means that it is read directly from a hdf
            # file
        # elif self.cell_type == 'PN':
        #     self.act = get_uPN_act_pps(self.dataset, act_pps_k1, act_pps_k2,
        #                                bin_con=True, side='S')
        else:
            raise ValueError(f'no cell type {self.cell_type}, exiting.')

        # print('EC50 data: retrieving')
        # self.EC50 = get_ORN_EC50(dataset=dataset)
        # self.EC50.fillna(0, inplace=True)  # puts 0 instead of nan
        # print('EC50 data: done retrieving')

        # there should be a check here that the names we chose are related to
        # the names in the activity
        # these are mainly used as sets, not so much for their order
        self.neur_act = np.array(self.act.columns)
        self.odor_act = np.array(self.act.index
                                 .get_level_values('odor').unique())

        # setting the internal variables neur_order and odor_order
        # if neur_order is empty is not compatible with what is in the activity
        # dataset, the order will be imposed by what is in the activity data
        self.set_neur_order(neur_order)
        self.set_odor_order(odor_order)

        self.odor_sel = odor_sel
        if odor_sel is not None:
            self.act = self.act.loc[odor_sel]
        # this puts in the activity the order that is in neur_order
        self.act = self.act.loc[:, self.neur_order]

        # this is reducing the activity
        if act_pps_k1 == 'raw' and act_pps_k2 == 'raw' and reduce is True:
            print(f'shape of act: {self.act.shape}')
            print(self.act.groupby('odor').size())
            min_trials = self.act.groupby('odor').size().min()
            print(f'min # of trials per odor: {min_trials}')
            # so that there is the same number of samples for each odor
            self.act = self.act.groupby('odor').head(min_trials)

        # this could be very much simplified and streamed if the ec50 was
        # not there, not sure what to do about it
        # for the moment we are not using ec50 anyway
        self.act_sels = {#'ec50': -self.EC50,
                         'all':  self.act,
                         '45':   self.act.loc(axis=0)[:, [4, 5]],
                         '678':  self.act.loc(axis=0)[:, [6, 7, 8]],
                         '4':    self.act.loc(axis=0)[:, 4],
                         '5':    self.act.loc(axis=0)[:, 5],
                         '6':    self.act.loc(axis=0)[:, 6],
                         '7':    self.act.loc(axis=0)[:, 7],
                         '8':    self.act.loc(axis=0)[:, 8]}
        self.act_sel_ks = act_sel_ks
        # keeping only those selections that are indicated by the input
        # and adding a layer in the dictionnary with key 'o', as later
        # we will be adding new pps for the act.
        if not set(act_sel_ks) <= self.act_sels.keys():
            raise ValueError(f'keys {act_sel_ks} are not available.\n'
                    f'Available keys are {list(self.act_sels.keys())}')

        self.act_sels = {k: {'o': self.act_sels[k]} for k in self.act_sel_ks}
        print('Done initializing main activity data.')
        # #####################################################################
        # ##############  IMPORTING GRAPE APPLE ACT DATA  #####################
        # #####################################################################
        # print('importing grape and apple activity data')
        # # at a later point, we might explore the raw responses, for now
        # # let's just work with the means
        # self.act_apl_gr_raw = pd.read_hdf(par_act3.file_apple_grape_hdf)
        # self.act_apl_gr = self.act_apl_gr_raw.mean(level=('odor', 'conc'))
        #
        # # this is shifting the concentrations so that it is in the same
        # # range as all the rest
        # self.act_apl_gr.rename(index={0: 4, 1: 5, 2: 6, 3: 7, 4: 8},
        #                        inplace=True)
        # print('Done importing grape and apple activity data')

        # #####################################################################
        # #############  INITIALIZING CONNECTIVITY DATA  ######################
        # #####################################################################
        print('Initializing connectivity data')
        # there is one more thing to be careful about here, it is that
        # the activity data might not have the same neurons as the
        # connectivity data, so you should actually only look at the neurons
        # that are present in the activity
        # also i wonder if i should be careful about the order here...

        path = f'{PATH}results/cons/'

        self.con_strms2 = {}
        for s in self.strms:
            # this is the old version, where only a subset of LN is there
# =============================================================================
#             self.con_strms2[s] = pd.read_hdf(f'../results/cons/cons_'
#                                              f'{cell_type}_{s}.hdf')
# =============================================================================
            self.con_strms2[s] = pd.read_hdf(f'{path}cons_'
                                             f'{cell_type}_{s}_all.hdf')
            self.con_strms2[s] = self.con_strms2[s].loc[self.neur_order]
            self.con_strms2[s] = FG.get_pps(self.con_strms2[s],
                                            ['o', 'cn', 'n'])

        # self.con_strms3 = pd.read_hdf(f'{path}cons_{cell_type}.hdf')
        self.con_strms3 = pd.read_hdf(f'{path}cons_{cell_type}_all.hdf')
        self.con_strms3 = self.con_strms3.loc[self.neur_order]
        self.con_strms3 = FG.get_pps(self.con_strms3, ['o', 'cn', 'n'])

        # if one wants to, one could rearrange the LNs in con_stream3:
        # LN_names = ['Broad T1 L', 'Broad T2 L', 'Broad T3 L']
        # (con_strms3.swaplevel(axis=1).stack().loc[:, LN_names]
        # .unstack().swaplevel(axis=1))

        # i don't remember what is used where and what is needed where
        # i guess i will just debug when the time will be right

        self.LNs = ['Broad T1 L', 'Broad T2 L', 'Broad T3 L', 'Broad T M L',
                    'Broad T1 R', 'Broad T2 R', 'Broad T3 R', 'Broad T M R',
                    'Broad T M M',
                    'Broad D1 L', 'Broad D2 L', 'Broad D M L',
                    'Broad D1 R', 'Broad D2 R', 'Broad D M R',
                    'Broad D M M',
                    'Keystone L L', 'Keystone R L',
                    'Keystone L R', 'Keystone R R',
                    'Keystone M M',
                    'Picky 0 [dend] L', 'Picky 0 [dend] R', 'Picky 0 [dend] M',
                    'Picky 0 [axon] L', 'Picky 0 [axon] R', 'Picky 0 [axon] M']

        self.splits_LN = [9, 9 + 7, 9 + 7 + 5]
        # these are the pps that are outputted by get_con_pps
        self.pps = ['o', 'n', 'cn']

        # glomeruli data that is not used in the paper
        # self.con_gloms = pd.read_hdf(f'{path}cons_gloms_M.hdf')
        # idx = self.con_gloms.index
        # idx = [f'{cell_type} {name}' for name in idx]
        # self.con_gloms.index = idx
        # self.con_gloms = self.con_gloms.loc[self.neur_order]
        # self.con_gloms = FG.get_pps(self.con_gloms, ['o', 'cn'])

        print('done initializing connectivity data.')
        # #####################################################################
        # #############  DATABASES THAT WILL BE USED  #########################
        # #####################################################################
        mi6 = pd.MultiIndex(levels=[[], [], [], [], [], []],
                            codes=[[], [], [], [], [], []],
                        names=['conc', 'pps', 'decomp', 'par1', 'par2', 'n'])
        self.act_W = pd.DataFrame(columns=mi6)
        self.act_W_cn = pd.DataFrame(columns=mi6)
        self.act_H = pd.DataFrame(columns=mi6)
        self.act_Y = pd.DataFrame(columns=mi6)
        self.act_Z = pd.DataFrame(columns=mi6)
        self.act_M = {}

        mi3 = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                            names=['conc', 'decomp', 'k'])
        self.errors = pd.Series(index=mi3)
        # the additional parameter i here is the identifier for the shuffling
        mi7 = pd.MultiIndex(levels=[[], [], [], [], [], [], []],
                            codes=[[], [], [], [], [], [], []],
                    names=['conc', 'pps', 'i', 'decomp', 'par1', 'par2', 'n'])

        # container for the PCA/NMF calculated on the shuffled/fake data
        self.act_fk_U = pd.DataFrame(columns=mi7)
        self.act_fk_U_cn = pd.DataFrame(columns=mi7)

        # the way these dictionaries will be organized is the following way:
        # keys are the streams of the connectivity:
        # 0 for ORNs to LNs
        # 1 for LNs to ORNs
        # 2 for LNs to ORNs with in-degree scaled
        # the elements of the dictionary are DataFrames, which the index
        # being different methods to analyse activity, and the columns
        # being different LNs, i.e., the connection vectors
        self.cc = {}  # cc is correlation coefficient
        self.cc_pv = {}  # pv is p-value
        
        # cosine similarity
        self.CS = {}
        self.CS_pv = {}

        self.subspc_cc = {}
        self.subspc_cc_pv = {}
        self.subspc_CS = {}
        self.subspc_CS_pv = {}
        
        # subspace overlap
        self.subspc_OL = {}
        self.subspc_OL_pv = {}

        print('Done creating instance of class NeurActConAnalysis')
        

    # #########################################################################
    # ######################  ADMIN FUNCTIONS  ################################
    # #########################################################################
    def set_neur_order(self, neur_order):
        """
        setting the internal variable neur_order and checking that all those
        cells are indeed present in the activity data
        """
        self.neur_order = self.neur_act if neur_order is None else neur_order

        if not set(self.neur_order).issubset(set(self.neur_act)):
            print('ERROR: provided list of neuron is not contained in the'
                  + ' activity data')
            self.neur_order = self.neur_act

    def set_odor_order(self, odor_order):
        """
        setting the internal variable odor_order and checking that all those
        odors are indeed present in the activity data
        """
        self.odor_order = self.odor_act if odor_order is None else odor_order
        if not set(self.odor_order).issubset(set(self.odor_act)):
            print('ERROR: provided list of odors is not contained in the'
                  + ' activity data')
            self.odor_order = self.odor_act

    # #########################################################################
    # ######################  PLOTTING ACTIVITY  ##############################
    # #########################################################################
    def plot_act(self, ps_str, act=None, neur_order=None, odor_order=None):
        """
        plot where each concentration where the graph is scaled to its own max
        plot where the responses for all concentations are on the same scale
        plot where all responses are all on one plot
        """

        # because in the raw case odors will have duplicates, before
        # plotting we need to clean up the data by removing the duplicates:
        if act is None:
            act = self.act.copy()
            act = act[~act.index.duplicated(keep='first')]

        if neur_order is None:
            neur_order = self.neur_order

        if odor_order is None:
            odor_order = self.odor_order

        for norm in [False, True]:
            f, _ = plot_activity(act, neur_order, odor_order,
                        norm=bool(norm), title=f'{self.cell_type} activity')
            file_str = (self.path_plots / f'{self.cell_type}_act'
                        f'{self.dataset}_scale{int(norm)}'
                        f'{ps_str}.png')
            FP.save_plot(f, file_str, self.save_plots, dpi=300)
        f, _ = plot_activity2(act, neur_order, odor_order,
                              title=f'{self.cell_type} activity')
        file_str = (self.path_plots / f'{self.cell_type}_act'
                    f'{self.dataset}{ps_str}.png')
        FP.save_plot(f, file_str, self.save_plots)

    def plot_cor(self, conc, pps, ps_str):
        """
        plotting the ORN and odor covariance matrices
        choosing which concentrations we using
        ps_str is just a string that we append to the end of the filename
        """

        # correlation between cells
        f, _, _ = plot_cov(self.act_sels[conc][pps], self.neur_order)
        file_str = (self.path_plots / f'{self.cell_type}_act'
                    f'{self.dataset}_{conc}_{pps}_neur-cov{ps_str}.png')
        FP.save_plot(f, file_str, self.save_plots, dpi=300)

        # correlation between odors
        # using the unstack makes that we average between concentrations

        # if we want to have correlations for the odors, then we need to
        # ctr and normalize in the ther other direction than for the
        # cells

        # because in the raw case odors will have duplicates, before
        # plotting we need to clean up the data by removing the duplicates:
        act = self.act_sels[conc]['o'].copy()
        act = act[~act.index.duplicated(keep='first')]
        # i don't think that removing dupliates has any effect
        act = act.unstack().T
        if pps == 'cn':
            act = FG.get_ctr_norm(act)

        f, _, _ = plot_cov(act, self.odor_order, norm=False)
        file_str = (self.path_plots / f'{self.cell_type}_act'
                    f'{self.dataset}_{conc}_{pps}_odor-cov{ps_str}.png')
        FP.save_plot(f, file_str, self.save_plots, dpi=300)

    # #########################################################################
    # ##################  SAVING THE CONNECTIVITY IN HDF5  ####################
    # #########################################################################
    # exporting connectivity data
    # there is already a function in con_preprocess_2.py that does the same
    # thing. So not sure if this is not redundant...
    def export_con_data(self):
        """
        this exports the con_stream2 dataset into 3 hdf5 files
        """
        path = f'{PATH}results/'
        ct = self.cell_type
        self.con_strms2[0]['o'].to_hdf(f'{path}con_{ct}2LN.hdf', f'{ct}2LN')
        self.con_strms2[1]['o'].to_hdf(f'{path}con_LN2{ct}.hdf', f'LN2{ct}')
        self.con_strms2[2]['o'].to_hdf(f'{path}con_LN2{ct}_indeg.hdf',
                                       f'LN2{ct}2_indeg')

    # #########################################################################
    # ######################  PLOTTING CONNECTIVITY  ##########################
    # #########################################################################
    def plot_con(self, LNs, pps):
        ct = self.cell_type
        titles = [f'{ct}s to LNs',
                  f'LNs to {ct}s',
                  f'LNs to {ct}s, in-degree scaled',
                  f'{ct}s with LNs, average ff and fb' ]
        # print('hello')
        f, axx = plt.subplots(1, len(self.strms), figsize=(7, 2.5))
        for i in self.strms:
            ax = axx[i]
            data = self.con_strms2[i][pps].loc[:, LNs].copy()
            # ordering the ORNs or uPNs as implied by the internal variable
            data = data.loc[self.neur_order]
            FP.imshow_df(data, ax=ax, splits_x=self.splits_LN, cb_frac=0.042)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(titles[i])
            if i > 0:
                ax.set_yticks([])

        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.22, top=0.95, left=0.1, right=0.95)
        FP.save_plot(f, self.path_plots / f'con_{self.cell_type}-LN_{pps}.png',
                     self.save_plots, dpi=300)

    # #########################################################################
    # ##############  CALCULATING DIFFERENT ANALYSIS OF ACTIVITY  #############
    # #########################################################################
    # here we have to decide on what data do we exactly calculate the PCA
    # of the data. Here are some choices that we have:
    # - which concentration selection
    # - which preprocessing of the data
    # actually we could be quite open and just decide here which
    # combinations of these things we want. That want we don't need to make
    # anything complicated. It will just be added to a global structure
    # which contains all the processing (PCA, NMF) of the activity
    # And then at a later stage, you can still choose what from all these
    # processing you want to correlate/plot
    # So at this stage we are just calculating some pps of the act
    # and we are choosing which conc selection and pps
    # Actually the thing is that we don't have at this point all the pps
    # of all the different activity data
    # actually seeing what i did before, i see that i need some home-made
    # ppsing of the data before doing the analysis

    # so maybe a useful function would be to create a pps of the data
    # which would iterate over all the concentrations and just add
    def add_act_pps(self, key, func):
        if key in self.act_sels[next(iter(self.act_sels))].keys():
            print('the key ' + key + ' exists already! Exiting.')
            return
        for k in self.act_sels.keys():
            self.act_sels[k][key] = func(self.act_sels[k]['o'])

    def calc_act_PCA(self, concs, act_pps, k: int = 3):
        """
        calculates the PCA loading vectors for the selected concentratiosn
        and the selected pps of the activity data
        If a loading vector has all the values of the same size, then
        that vector will be made positive
        Parameters
        ----------
        """
        print(f'data selections for which PCA will be calculated: {concs}')
        for conc in concs:
            SVD = FG.get_svd_df(self.act_sels[conc][act_pps].T)
            W = SVD['U']
            H = SVD['Vh']
            for n in range(1, k + 1):

                # making the eigenvector positive if all the values are <=0
                # it is usually only something that would happen to the
                # first component
                sign = -1 if np.sum(W[n] <= 0) == len(W[n]) else 1
                self.act_W[conc, act_pps, 'SVD', '', '', n] = sign * W[n]
                self.act_H[conc, act_pps, 'SVD', '', '', n] = H.T[n]

    def calc_act_NMF(self, concs, act_pps, k: int = 2, beta=2,
                     n_rnd: int = 10):
        """
        N is the order of the NMF
        beta is the parameter in the optimization function of the NMF
        n_rnd is the number of random initialization in addition on the
        deterministic initializations to try in order to get the best NMF
        (i.e., with the smallest error according to the objective function)
        """
        meth = 'NMF_' + str(k)
        print(f'data selections for which NMF will be calculated: {concs}')
        for conc in concs:
            data = self.act_sels[conc][act_pps].T
            # this is the usual way of having data: columns odors, rows: ORNs
            W, H, e, init = nmf.get_nmf_best_df(data, k=k, beta=beta,
                                                n_rnd=n_rnd)
            self.errors[conc, 'NMF', k] = e
            for n in range(1, k + 1):
                self.act_W[conc, act_pps, meth, beta, init, n] = W[n]
                self.act_H[conc, act_pps, meth, beta, init, n] = H.T[n]
            print(f'finished conc: {conc}, error: {e}')


    def calc_act_SNMF(self, concs, act_pps, k: int = 3):
        """
        Calculating the W that would arise from SNMF
        """
        meth = 'SNMF_' + str(k)
        print(f'data selections for which SNMF will be calculated: {concs}')
        ks = np.arange(k) + 1
        for conc in concs:
            print(f'working on conc: {conc}')
            data = self.act_sels[conc][act_pps].T
            Z, e, m = nmf.get_snmf_best_df(data, k=k, rtol=1e-6,
                                        max_iter=int(1e6))
            # print(m)
            self.errors[conc, 'SNMF', k] = e
            W = data @ Z.T / data.shape[1]
            M = Z @ Z.T / data.shape[1]
            M = pd.DataFrame(M, index=ks, columns=ks)
            for n in range(1, k + 1):
                self.act_W[conc, act_pps, meth, '', '', n] = W[n]
                self.act_H[conc, act_pps, meth, '', '', n] = Z.T[n]
            self.act_M.setdefault(conc, {}).setdefault(act_pps, {})[meth] = M
            print(f'finished conc: {conc}, error: {e}')


    def calc_act_NNC(self, concs, act_pps, k: int = 3, alpha=50, cycle=500,
                      scaling=1, rtol=1e-7, rectY: bool = True,
                      rectZ: bool = True, beta=0.2):
        """
        Calculating the W that would arise from the non-negative olfactory
        circuit
        adding the scaling as it also influences the W
        Parameters
        ----------
        alpha:
            50 usually works well, that the beginning value for parameter
            in the gradient
        """
        if rectY and rectZ:
            meth = 'NNC'
        elif rectY:
            meth = 'NNYC'
        elif rectZ:
            meth = 'NNZC'
        else:
            meth = 'LOC'
        meth = f'{meth}_{k}'
        print(f'data for which {meth} will be calculated: conc: {concs},'
              f'scaling: {scaling}, act_pps: {act_pps}')
        ks = np.arange(k) + 1
        for conc in concs:
            print(f'working on conc: {conc}')
            data = self.act_sels[conc][act_pps].T
            # here data is oriented as the usual X
            pps = f'{scaling}{act_pps}'
            Y, Z, costs = FOC.olf_gd_offline(data.values, k, max_iter=10000,
                                             rectY=rectY, rectZ=rectZ,
                                             rtol=rtol, rho=scaling,
                                             alpha=alpha, cycle=cycle,
                                             beta=beta)
            self.errors[conc, 'NNC', k] = costs[-1]
            W = Y @ Z.T / data.shape[1]
            W = pd.DataFrame(W, index=data.index, columns=ks)
            M = Z @ Z.T / data.shape[1]
            M = pd.DataFrame(M, index=ks, columns=ks)

            print(W.corr())

            Z = pd.DataFrame(Z, index=np.arange(k) + 1, columns=data.columns)
            Y = pd.DataFrame(Y, index=data.index, columns=data.columns)
            W.columns.name = 'n'
            for n in range(1, k + 1):
                self.act_W.loc[:, (conc, pps, meth, '', '', n)] = W.loc[:, n]
                self.act_Z.loc[:, (conc, pps, meth, '', '', n)] = Z.loc[n]
            for orn in data.index:
                self.act_Y.loc[:, (conc, pps, meth, '', '', orn)] = Y.loc[orn]
            self.act_M.setdefault(conc, {}).setdefault(pps, {})[meth] = M
            print(f'finished conc: {conc}, cost: {costs[-1]}')


    def calc_act_NNZC(self, concs, act_pps, k: int = 3, alpha=100, cycle=500,
                      scaling=1, rtol=1e-7):
        """
        Calculating the W that would arise from the non-negative olfactory
        circuit
        adding the scaling as it also influences the W
        """
        meth = 'NNZC_' + str(k)
        print(f'data selections for which NNZC will be calculated: {concs},'
              f'scaling: {scaling}, act_pps: {act_pps}')
        ks = np.arange(k) + 1
        rect = True
        for conc in concs:
            print(f'working on conc: {conc}')
            data = self.act_sels[conc][act_pps].T*scaling
            N = data.shape[1]
            # here data is oriented as the usual X
            pps = f'{scaling}{act_pps}'
            Z, costs = FOC.olf_gd_offline2(data.values, k, max_iter=10000,
                                             rect=rect, rtol=rtol,
                                             alpha=alpha, cycle=cycle)
            Y = data @ LA.inv(Z.T @ Z / N + np.eye(N))
            self.errors[conc, 'NNZC', k] = costs[-1]
            W = Y @ Z.T / data.shape[1]
            W = pd.DataFrame(W, index=data.index, columns=ks)
            M = Z @ Z.T / data.shape[1]
            M = pd.DataFrame(M, index=ks, columns=ks)

            print(W.corr())

            Z = pd.DataFrame(Z, index=np.arange(k) + 1, columns=data.columns)
            Y = pd.DataFrame(Y, index=data.index, columns=data.columns)
            W.columns.name = 'n'
            for n in range(1, k + 1):
                self.act_W.loc[:, (conc, pps, meth, '', '', n)] = W.loc[:, n]
                self.act_Z.loc[:, (conc, pps, meth, '', '', n)] = Z.loc[n]
            for orn in data.index:
                self.act_Y.loc[:, (conc, pps, meth, '', '', orn)] = Y.loc[orn]
            self.act_M.setdefault(conc, {}).setdefault(pps, {})[meth] = M
            print(f'finished conc: {conc}, cost: {costs[-1]}')

    # #########################################################################
    # ###########  CALCULATING THE CORR AND SIGNIF OF THE ACT VS CON ##########
    # #########################################################################
    # def calc_act_con_corr_signif__old(self, N_iter=10, opt='fast'):
    #     """
    #     calculates the significance for every correlation coefficient
    #     that was already calculated
    #     The fast and slow versions differ in whether we are shuffling
    #     every single column separately of the streams2 matrix, or
    #     if we are shuffling them all at once, with the same permutation.
    #     """
    #     if opt == 'fast':
    #         signif_f = FG.get_signif_corr_v2
    #     elif opt == 'slow':
    #         signif_f = FG.get_signif_corr_v1
    #     else:
    #         raise ValueError('no input option opt: ' + opt)
    #
    #     N = len(self.con_strms2[0]['o'].index)  # the number of ORNs/PNs
    #     for strm in self.strms:
    #         # it seems that this function might be aligning the ORN order
    #         cor, sign_l, sign_r = signif_f(self.act_W_cn,
    #                                        self.con_strms2[strm]['cn'], N=N_iter)
    #         sign_l = sign_l.replace(0, 1./N_iter)
    #         sign_r = sign_r.replace(0, 1./N_iter)
    #         self.corr[strm] = cor
    #         self.sign_l[strm] = sign_l
    #         self.sign_r[strm] = sign_r
    #
    #         # formula for signicance testing of the correlation coefficient
    #         # putting the abs makes us calculate one sided prob
    #         t = np.abs(cor.T) / np.sqrt((1 - cor.T**2) / (N - 2))  # t value
    #         self.sign_t[strm] = t.applymap(lambda x: sps.t.sf(x, N - 2))

    # in this new version we work directly with strms3
    def calc_act_con_cc_pv(self, N_iter=10, opt='fast'):
        """
        calculates the significance for every correlation coefficient
        that was already calculated
        The fast and slow versions differ in whether we are shuffling
        every single column separately of the strms3 matrix, or
        if we are shuffling them all at once, with the same permutation.
        """
        if opt == 'fast':
            signif_f = FG.get_signif_corr_v2
        elif opt == 'slow':
            signif_f = FG.get_signif_corr_v1
        else:
            raise ValueError('no input option opt: ' + opt)

        N = len(self.con_strms3['o'].index)  # the number of ORNs/PNs
        # This function aligns the ORN order, i.e., in the row dimension
        # the function shuffles the second dataset, here con_strms3, so
        # it is better if it has less columns than the other...
        cor, pv_o, pv_l, pv_r = signif_f(self.act_W_cn,
                                         self.con_strms3['cn'], N=N_iter)

        pv_o = pv_o.replace(0, 1./N_iter)
        pv_l = pv_l.replace(0, 1./N_iter)
        pv_r = pv_r.replace(0, 1./N_iter)
        self.cc = cor
        self.cc_pv['o'] = pv_o
        self.cc_pv['l'] = pv_l
        self.cc_pv['r'] = pv_r

        # formula for significance testing of the correlation coefficient
        # putting the abs makes us calculate one sided prob
        # http://vassarstats.net/rsig.html
        t = np.abs(cor) / np.sqrt((1 - cor**2) / (N - 2))  # t value
        self.cc_pv['t'] = t.applymap(lambda x: sps.t.sf(x, N - 2))

    # calculating the cosine similarity
    def calc_act_con_CS_pv(self, N_iter=10, opt='fast'):
        """
        calculates the significance for every cosine similarity
        that was already calculated
        The fast and slow versions differ in whether we are shuffling
        every single column separately of the strms3 matrix, or
        if we are shuffling them all at once, with the same permutation.
        """
        if opt == 'fast':
            signif_f = FG.get_signif_CS_v2
        elif opt == 'slow':
            signif_f = FG.get_signif_CS_v1
        else:
            raise ValueError('no input option opt: ' + opt)

        # it seems that this function might be aligning the ORN order
        CS, pv_o, pv_l, pv_r = signif_f(self.act_U_n,
                                        self.con_strms3['n'], N=N_iter)
        # for the cosine similarity we need the normalized version of vectors

        pv_o = pv_o.replace(0, 1./N_iter)
        pv_l = pv_l.replace(0, 1./N_iter)
        pv_r = pv_r.replace(0, 1./N_iter)
        self.CS = CS
        self.CS_pv['o'] = pv_o
        self.CS_pv['l'] = pv_l
        self.CS_pv['r'] = pv_r
        
        # i don't think there is a formula that gives the significance
        # for cos similarity in the same way as there was a formula for
        # for the correlation coefficient

    # #########################################################################
    # ##########  CALCULATING THE MAXIMUM CORR WITH THE ACT SUBSPACE  #########
    # #########################################################################
    # what are the inputs, and what do you want to get with this function

    def get_corr_act_subspace(self, act_subspc):
        """
        input: act vectors, label of the decomp (e.g., SVD_4, NMF_3...)
        
        output corr with all the streams of the connections (strms3,
        which contains all directions)

        Description of the calculation:
        The goal of this calculation is to get max correlation coefficient
        between w, which is a connection vector and a1*v1 + a2*v2 + ... + ai*vi
        which is a composition of activity vectors.
        We start with the activity vectors. It can be coming from the SVD
        or the NMF. The activity vectors are not centered.
        However, we want a vector that is inside the subspace created by the
        original SVD or NMF vectors.

        First just the simplest theory. If you have a vector v and a subspace S
        and you want to find the vector in S that has the smallest angle with v
        Here is what you do in such a case: you project the vector v onto
        the subspace and the obtained vector with have the smallest angle with
        v and will be part of S.
        How to do it practically.
        In the case when the subspace is already described by orthonormal
        vectors, one just needs to create a matrix M with these vectors as
        columns and then calculate M . M^T . v
        M^T . v gives the projected components of v onto the orthogonal
        vectors and multiplying by M gives the final vector.
        We can then just use the newly obtained vector to calculate the actual
        angle between it and v.

        In the case when we vectors that are not orthonormal, there are 2
        options. First find the equivalent orthonormal base for this subspace.
        This can be done with a QR decomposition. The matrix Q will contain
        the orthonormal basis, and the we just do the same as before.
        The second option is to directly calculate the projector, which is:
        B.dot(LA.inv(B.T.dot(B))).dot(B.T)
        If the columns of B are orthonormal, then the middle part is simply
        the identity and we get back to the formula above.

        This was for finding the vector with the smallest angle. Now we want
        to find the vector, in the subspace S, that has the highest correlation
        with the vector v. Say that the vector v is not centered.
        
        The want to find the minimum angle with a centered version of a
        vector of the subspace (which corresponds to the correlation coef).
        The trick here is that having a linear
        combination of the vectors of the subspace and then centering is
        equivalent to first centering the vectors of the subspace and
        then finding a linear combination of them.
        
        It is important to note that the space of centered vectors is a
        vector subspace. So when projecting on it, is it the same as first
        subtracting the mean of the vector and then projecting.
        
        Centering is a projection onto a subspace. It is a n-1 dimensional
        subspace
        
        Or an other way of putting it. You are trying to find the maximum
        correlation between a vector and a subspace. Each time you chose
        a vector in that subspace, you first project into the subspace
        of centered vectors. So basically there is a subspace inside the
        subspace of centered vectors which contains all the potential
        vectors. Thus you directly work in the subspace of centered vectors
        """

        # centering the subspace, since we are going to project onto the
        # centered subspace
        act_c = act_subspc - act_subspc.mean()
        # columns of act_c contain centered version of the original vectors
        A = act_c.values

        # orthogonal projector onto the subspace formed by the centered
        # activity vectors
        P = A.dot(LA.inv(A.T.dot(A))).dot(A.T)

        # projecting the connection vector onto the subspace
        proj = pd.DataFrame(P.dot(self.con_strms3['o']))
        # at this point, the projecion is already centered
        # this can be checked with the centering function:
        # (can be removed at a later point)
        if not FG.is_mean_0(proj):
            print('calc_corr_act_subspace: proj not centered')
            print(np.abs(np.sum(A)).argmax(), np.abs(np.sum(A)).max())
        
        # proj contains the projections of the connectivity vectors onto
        # the activity subspace
        
        # normalizing the projections so that one calculates
        # the correlation coefficients. The centering happened throught the
        # projection, as one was projecting on centered vectors
        proj_cn = FG.get_norm(proj)
        # proj_cn contains the vector with the highest correlation coefficient
        # within the subspace
        
        # element-wise multiplication with the sum give the corr.coefs
        return np.multiply(self.con_strms3['cn'], proj_cn).sum()
        # selecting only the corr coef that are interesting for us
#
#

    def get_something(self, act_subspc, label):
        """
        maybe here doing the projection and also findout the contribution
        of different components in the best projection

        the QR decomposition is done once and for all, it is only dependent
        on the subspace we are working with
        The Q R decomposition gives the following:
        - the columns of Q are the orthonormal vector forming the same subspace
        as the original vectors that were provided
        - the columns of R give the coefficients needed recreate the
        original vector
        Then, using the QR decomposition, we then take one by one the
        connectivity vectors we are interested in, the different streams,
        the different LNs, etc...
        For each connectivity w, R^-1 Q.T w gives the coefficients
        in terms of the original act vectors, that give the projection
        (i.e., the vector with the highest correlation) in the subspace
        of the original vectors

        for just the projection on the subspace, we only need Q Q.T w

        To get the contributions one needs R^-1 Q.T w
        """

    # #########################################################################
    # #################  SIGNIFICANCE TEST WITH DEEP SHUFFLING  ###############
    # #########################################################################
    def get_shuffled_act(self, conc, pps, opt=3):
        """
        this function creates a new dataset, with shuffled entries.
        takes as input the keys of the dataset we want to work with,
        which entails the conc selection as well as the pps

        there are 3 options to do shuffling of the data:
        1. shuffle completely
        2. shuffle within concentrations
        3. shuffle by blocks, keeping the concentration together for
        a cell/odor pair

        option 1 is the fastest, because i guess the others require to do
        the unstacking and stacking... would probably be advantageous
        to stack/unstack only once... still needs to be implemented

        an other option that might needed to be implemented is when the
        shuffling also mIxes responses from different cells
        """
        if conc not in list(self.act_sels.keys()):
            raise ValueError('concentration ' + conc + ' not available.')
        if pps not in list(self.act_sels[conc].keys()):
            raise ValueError('pps ' + pps + ' not available for conc' + conc)
        data = self.act_sels[conc][pps].copy()

        if opt == 1:
            for c in data.columns:
                data.loc[:, c] = np.random.permutation(data.loc[:, c])
        elif opt == 2:
            # the unstacking moves teh concentration index to the columns
            # so that now each concentration is shuffled separetely
            data = data.unstack()
            for c in data.columns:
                data.loc[:, c] = np.random.permutation(data.loc[:, c])
            data = data.stack()
        elif opt == 3:
            # the unstacking moves the concentration index to the columns
            # and then the shuffling happens per cell
            cells = data.columns
            data = data.unstack()
            for c in cells:
                data.loc[:, c] = np.random.permutation(data.loc[:, c])
            data = data.stack()
        else:
            raise ValueError('value ' + str(opt) + ' not available for opt')

        return data

    def calc_act_shuffled_PCA(self, conc, pps, opt=3, N=3, i=0):
        """
        I think it would be better if this function would be written in such a
        way that it returns something, so that it is clear what this function
        is changing.


        adds the top N loading vectors of the shuffled data
        of conc and pps, shuffling happening with opt.
        i is an additional id parameter so that different
        shuffles have unique identifiers in act_fk_U
        If a loading vector has all the values of the same size, then
        that vector will be made positive
        """
        data_shfl = self.get_shuffled_act(conc, pps, opt=opt)
        SVD1 = FG.get_svd_df(data_shfl.T)['U']
        for n in range(1, N+1):
            sign = -1 if np.sum(SVD1[n] <= 0) == len(SVD1[n]) else 1
            self.act_fk_U[conc, pps, i, 'SVD', '', '', n] = sign * SVD1[n]

    def calc_act_shuffled_PCA_batch(self, conc, pps, opt=3, N=3, n_batch=10):
        """
        calculates several (n_batch) shuffled data and resulting PCAs N top
        loading vectors. conc and pps keyworkds are choosing the act dataset
        opt, chooses the type of shuffling, see get_shuffled_act
        """
        for i in range(n_batch):
            self.calc_act_shuffled_PCA(conc, pps, opt, N, i)

    def calc_act_shuffled_NMF(self, conc, pps, opt=3, N=3, beta=2, n_rnd=10,
                              i=0):
        """
        adds the Nth order NMF decomposition of the shuffled data
        of conc and pps, shuffling happening with opt.
        i is an additional id parameter so that different
        shuffles have unique identifiers in act_fk_U
        """
        meth = 'NMF_' + str(N)
        data_shfl = self.get_shuffled_act(conc, pps, opt=opt)
        _, H, _, init = nmf.get_nmf_best_df(data_shfl, k=N, beta=beta,
                                             n_rnd=n_rnd)
        for n in range(1, N+1):
            self.act_fk_U[conc, pps, i, meth, beta, init, n] = H.T[n]

    def calc_act_shuffled_NMF_batch(self, conc, pps, opt=3, N=3, beta=2,
                                    n_rnd=10, n_batch=10):
        """
        calculates several (n_batch) shuffled data and resulting PCAs N top
        loading vectors. conc and pps keyworkds are choosing the act dataset
        opt, chooses the type of shuffling, see get_shuffled_act
        """
        for i in range(n_batch):
            self.calc_act_shuffled_NMF(conc, pps, opt, N, beta, n_rnd, i)

    def clean_shuffled_comps(self):
        """
        puts act_fk_U and act_fk_U_cn to empty DataFrame values
        comps as in components of PCA or NMF
        """
        self.act_fk_U = pd.DataFrame(columns=self.mi7)
        self.act_fk_U_cn = pd.DataFrame(columns=self.mi7)

    def calc_fk_act_comp_vs_conn_corr(self):
        """
        calculates first act_fk_U_cn, centered normalized version of
        act_fk_U, then calculated the correlation with the usual
        3 connectivity streams
        """
        # calculating the centered and normalized version, so that one can
        # calculate the cc just after

        self.act_fk_U_cn = FG.get_ctr_norm(self.act_fk_U)
        for strm in self.strms:
            self.corr_fk[strm] = FG.get_corr(self.act_fk_U_cn,
                                     self.con_strms2[strm]['cn'])

    def calc_sign_d(self, conc, meth, pps, comps=[1, 2, 3]):
        """
        calculates the sign_dl and sign_dr for the right and left
        significance.
        and corr_fk_m, which is the mean fk correlations
        (i could also calculate that for the other shuffling..
        so not sure it is super useful here)
        probably need both one-tailed and two-tailed

        this function relies on the fact that the real correlations have
        already been calculated and also that we already have a set of
        fake correlation coefficients with the method meth, conc, and pps
        comps are the components of PCA/NMF
        """
        # the list of LNs that are is the real corr-coef dataset
        LNs_sel = list(self.corr[0].columns)

        for strm, n, ln in itertools.product(self.strms, comps, LNs_sel):
            # collection of fake correlation coefficients
            ccs = self.corr_fk[strm].loc[(conc, pps, slice(None), meth,
                              slice(None), slice(None), n), ln].values

            # actual correlation coefficient for the same parameters
            ccs_real = self.corr[strm].loc[(conc, pps, meth, slice(None),
                                            slice(None), n), ln]
            idx = list(ccs_real.index)[0]
            ccs_real = ccs_real.values
            sign_r = np.sum(ccs >= np.abs(ccs_real))/np.sum(len(ccs))
            sign_l = np.sum(ccs <= -np.abs(ccs_real))/np.sum(len(ccs))
            self.sign_dr[strm].loc[idx, ln] = sign_r
            self.sign_dl[strm].loc[idx, ln] = sign_l
            self.corr_fk_m[strm].loc[idx, ln] = np.mean(ccs)

        # the len of css, which is the number of trials, is nomrally
        # the same for all streams, unless something very strange happened...
        for strm in self.strms:
            self.sign_dl[strm] = self.sign_dl[strm].replace(0, 1./len(ccs))
            self.sign_dr[strm] = self.sign_dr[strm].replace(0, 1./len(ccs))

    # #########################################################################
    # ###############  PLOTTING CORR ODOR VS CONN  ############################
    # #########################################################################
    def plot_act_odors_vs_conn_corr(self, act, act_comp, con_pps, ps_str,
                                    neur_sel, odor_order=None, **kwargs):
        """
        act is the activity data, it will be centered and normalized inside
        the function, before calculating the correlation coefficient
        act_comp are the activity component that come for SVD of NMF
        act will be concatenated with act_comp, so that we have a whole
        matrix
        the correlation of the concatenated data will be calculated with the
        all the streams of connections.
        Each stream will be plotted separately
        ps_str is a string to be added to the file name so that the file
        name can be easily differentiated
        neur_sel are the neurons that we want to include in the plot and for
        which the correlatoin will be calculated.
        """

        if odor_order is None:
            odor_order = self.odor_order
        ns = self.neur_order  # like neuron sort
        A = act.loc[:, ns]  # sorting necessary before the concatenation below
        A = A.unstack().loc[odor_order].stack()  # there is a bug
        # and for some reason it doesn't work without the unstack/stack
        # this order corresponds to the 2nd component of the PCA


        C = act_comp.loc[ns]
        A = pd.concat([A, C.T], axis=0)

        for strm in self.strms:
            B = self.con_strms2[strm][con_pps].loc[ns].loc[:, neur_sel]
            B = pd.concat([C, B], axis=1)

            corr_all_odors_con = A.dot(B)
            f, ax, _ = FP.imshow_df(corr_all_odors_con.T, tight=True,
                                    title=self.titles_d()[strm], **kwargs)

            file_name = (f'{self.cell_type}_con_'
                         f'{self.con_pps_k}{strm}_vs_act{self.dataset}_'
                         f'{self.act_pps_k1}_{self.act_pps_k2}_'
                         f'{ps_str}.png')
            FP.save_plot(f, self.path_plots / file_name, self.save_plots)

    # #########################################################################
    # ##################  PLOTTING CORR AND SIGNIF  ###########################
    # #########################################################################
    def plot_corr_signif(self, sel, strm, neur_sel, cat_drop, list_df, ext='',
                         **kwargs):
        """
        cat_drop are the categories to drop in the index before plotting
        ext is the extension that we put at the end of the filename
        
        what is in kwargs is transmitted to im
        """
        # the following is just some text extraction to be put in the file
        # name
        concs = [item[0] for item in sel]
        concs = set(concs)
        pps = [item[1] for item in sel]
        pps = set(pps)
        meths = [item[2] for item in sel]
        meth_names = [''.join([i for i in meth if not i.isdigit()])
                      for meth in meths]
        meth_names = set(meth_names)
        meth_numbers = [''.join([i for i in meth if i.isdigit()])
                       for meth in meths]
        meth_numbers = list(filter(None, meth_numbers))
        meth_min = min(meth_numbers)
        meth_max = max(meth_numbers)
        # meth can be quite long to be included in the file name, making it
        # shorter

        
        sel = [(s[0], s[1], s[2], i+1) for s in sel for i in range(s[3])]
        
        for (df, cm, lim, k) in list_df:
            to_plot = df.loc[:, strm].copy()
            to_plot = to_plot.reset_index(cat_drop, drop=True)
            to_plot = to_plot.loc[sel, neur_sel]
            
            # print(to_plot)

            # if we are plotting the correlation, we are changing the sign
            # of the correlation coef on the row so that they are positive for
            # the first Broad neuron

            if k == 'c':
                for i in range(len(to_plot.index)):
                    if (to_plot.iloc[i, 0] < 0 and
                        'SVD' in to_plot.iloc[i].name):
                        to_plot.iloc[i] = - to_plot.iloc[i]

            f, ax, clb = FP.imshow_df(to_plot, vlim=lim, cmap=cm, **kwargs)
            f.tight_layout()
            file_name = (self.path_plots / f'{self.cell_type}_con_'
                         f'{self.con_pps_k}{strm}_vs_act{self.dataset}_'
                         f'{self.act_pps_k1}_{self.act_pps_k2}_'
                         f'{concs}_{pps}_{meth_names}{meth_min}-{meth_max}'
                         f'_{k}{ext}')
            FP.save_plot(f, file_name + '.png', self.save_plots,
                         dpi=400, transparent=True)

    # #########################################################################
    # #################  PLOTTING 1 CON AND 1 ACT VECTOR  #####################
    # #########################################################################
    def plot_1act_1con(self, strm, LN_sel, act_w, lbl_con, lbl_act):
        """
        this function puts on the same figure a line plots and a scatter plots
        comparing a component of the activity and one connectivity vector
        strm is the stream, can be from 0 to 2
        LN_sel is the name of the LN
        act_w is the label for activity vector, like PCA/NMF ect
        lbl_con and lbl_act are just the string that we want to write in the
        plot
        """

        f, axx = plt.subplots(1, 2, figsize=(6, 3))

        ax = axx[0]
        act_plt = self.act_W.T.reset_index(level=['par1', 'par2'], drop=True).T

        # act_plt = act_plt.loc[:, act_w]
        # con_plt = self.con_strms2[strm]['o'].loc[:, LN_sel]

        act_plt = act_plt.loc[self.neur_order, act_w]
        con_plt = self.con_strms2[strm]['o'].loc[self.neur_order, LN_sel]

        cell_list = list(act_plt.index)
        if cell_list != list(con_plt.index):
            print('ERROR: the cell order in act and con is not aligned!')

        cell_list = [name[len(self.cell_type) + 1:] for name in cell_list]
        # print(cell_list)
        FP.plot_line_2yax(ax, act_plt.values, con_plt.values, lbl_act, lbl_con,
                          cell_list, self.cell_type + 's', 'b', 'r')

        ax = axx[1]
        FP.plot_scatter(ax, con_plt, act_plt, lbl_con, lbl_act, 'r', 'b')

        f.tight_layout()
        file_name = (self.cell_type + '_con_strm'
                     + str(strm) + '_' + LN_sel + '_vs_act' + str(self.dataset)
                     + '_' + str(act_w) + '.png')
        FP.save_plot(f,  self.path_plots / file_name, self.save_plots)
        plt.show()

    # #########################################################################
    # ##################  PLOTTING CONs AND ACT VECTORs  ######################
    # #########################################################################
    # title of the graphs
    def titles_d(self):
        return {0: f'from {self.cell_type} to LNs',
                1: f'from LNs to {self.cell_type}',
                2: f'from LNs to {self.cell_type}, in-degree-scaled',
                3: f'{self.cell_type} with LNs, ff and fb averaged'}

    def plot_act_con_table(self, meths, conc, PS=''):
        """
        This function plots several figures on a grid. In the top row there are
        the PCA or NMF or something else of the activity
        and the below graphs show the connectivities.
        Also all the graphs can include the connectviity from ORN to uPN
        """
# this is if we want to include the strength of the connections ORN -> PN
# we won't do it for now, maybe later.
# =============================================================================
# con_ORN2uPN1 = (cons['L'].loc[cells_full['ORN']['L'], cells_full['uPN']['L']]
#                 .sum(axis=1)
# con_ONR2uPN2 = (cons['L'].loc[cells_full['ORN']['L'], cells_full['uPN']['L']]
#                 .sum(axis=0)
# 
# con_O2u = {'ORN': con_ORN2uPN1, 'uPN': con_ONR2uPN2}
# con_O2u_n = {k: v/v.mean() for k, v in con_O2u.items()}
# 
# # normalization based on the size of the glomeruli
# norm = {'ORN': con_ORN2uPN1.values, 'uPN': con_ONR2uPN2.values}
# =============================================================================

        # this removes the normalization
        # (for now we do without this normalization)
# =============================================================================
#         normalize = False
#         norm_str = '_norm'
#         if normalize is False:
#             norm = np.ones(len(self.neur_act))
#             norm_str = ''
# =============================================================================

        n_o = self.neur_order
        f, axx = plt.subplots(6, 3, figsize=(15, 20), sharex='all')

        for k in range(len(meths)):  # k is the decomposition here
            H = self.act_W.loc[n_o, meths[k]]
            H = H.T.reset_index(level=['par1', 'par2'], drop=True).T
            # for adding on the plots the number of connections from ORN to uPN
            # con_O2u_n[ct].plot(ax=axx[0, drct], style=':', color='grey')
            H.plot(ax=axx[0, k])
            axx[0, k].set_facecolor('lightyellow')
            axx[0, k].set_title(meths[k])

        for k in self.strms:  # k is the stream here
            axx[1, k].set_title(self.titles_d()[k])
            cons = self.con_strms2[k]['o']
            for a, s in enumerate(['L', 'R']):
                con_T = cons.loc[n_o, 'Broad T1 ' + s: 'Broad T3 ' + s].copy()
                # con_T = con_T.divide(norm, axis=0)
                con_D = cons.loc[n_o, 'Broad D1 ' + s: 'Broad D2 ' + s].copy()
                # con_D = con_D.divide(norm, axis=0)
                con_T.plot(ax=axx[1 + a, k])
                con_T.mean(axis=1).plot(ax=axx[1 + a, k], color='k')

                con_D.plot(ax=axx[3 + a, k])
                con_D.mean(axis=1).plot(ax=axx[3 + a, k], color='k')
# =============================================================================
#         (con_O2u_n[ct]*np.mean(con_T.values)).plot(ax=axx[1+a, strm],
#                                                    style=':', color='grey')
#         (con_O2u_n[ct]*np.mean(con_D.values)).plot(ax=axx[3+a, strm],
#                                                    style=':', color='grey')
# =============================================================================

            # con_O2u_n[ct].plot(ax=axx[5, strm], style=':', color='gray')
            s = 'M'
            con_T = cons.loc[n_o, 'Broad T1 ' + s: 'Broad T3 ' + s].copy()
            # con_T = con_T.divide(norm, axis=0)
            con_D = cons.loc[n_o, 'Broad D1 ' + s: 'Broad D2 ' + s].copy()
            # con_D = con_D.divide(norm, axis=0)

            con_T = con_T.mean(axis=1)
            con_T /= con_T.mean()
            con_T.plot(ax=axx[5, k], color='b', label='Broad T', legend=True)

            con_D = con_D.mean(axis=1)
            con_D /= con_D.mean()
            con_D.plot(ax=axx[5, k], color='r', legend=True,
                       xticks=np.arange(len(con_D)), rot=90, label='Broad D')

        for i, j in itertools.product(range(0, 6), range(3)):
            axx[i, j].xaxis.grid()
            if i != 0:
                axx[i, j].set_ylim((0, None))

        plt.tight_layout()

        FP.save_plot(f,  self.path_plots / (self.cell_type + '_act'
                     + str(self.dataset) + conc + str(meths)
                     + '_vs_conn' + PS + '.png'), self.save_plots)

