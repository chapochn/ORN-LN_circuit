"""
Contains functions and classes that are olfactory-specific.
@author: Nikolai M Chapochnikov
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

# import functions.nmf as nmf
import functions.general as FG
import functions.plotting as FP
import functions.olf_circ_offline as FOC
import params.act2 as par_act2
import params.act3 as par_act3
import params.con as par_con
import os
from typing import List, Tuple, Any

# from functions.plotting import plot_cov  # since we are not using any other
# plotting function

# OLF_PATH = os.path.realpath(f"{os.path.expanduser('~')}/ORN-LN_circuit") + '/'
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
        cons[k] = pd.read_hdf(OLF_PATH / f'results/cons/cons_full_{k}.hdf')
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


# def remove_inactive_neurons(act_df: pd.DataFrame) -> pd.DataFrame:
#     # this is faster than using the pandas sum
#     active_neurons = act_df.values.sum(axis=0) != 0
#     # print('number of active neurons: ', sum(active_neurons))
#     act_df = act_df.loc[:, active_neurons]
#     # print(data.shape)
#     return act_df


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
        # act2 = act1.mean(level=('odor', 'conc'))
        act2 = act1.groupby(level=('odor', 'conc')).mean()
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


# #############################################################################
# #############################################################################
# ###########################  CLASSES DEFINITIONS   ##########################
# #############################################################################


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
        neur_order, if neur_order is empty, the order will be imposed by
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

        path = OLF_PATH / f'results/cons/'

        self.con_strms2 = {}
        for s in self.strms:
            # this is the old version, where only a subset of LN is there
# =============================================================================
#             self.con_strms2[s] = pd.read_hdf(f'../results/cons/cons_'
#                                              f'{cell_type}_{s}.hdf')
# =============================================================================
            self.con_strms2[s] = pd.read_hdf(path / f'cons_'
                                             f'{cell_type}_{s}_all.hdf')
            self.con_strms2[s] = self.con_strms2[s].loc[self.neur_order]
            self.con_strms2[s] = FG.get_pps(self.con_strms2[s],
                                            ['o', 'cn', 'n'])

        # self.con_strms3 = pd.read_hdf(f'{path}cons_{cell_type}.hdf')
        self.con_strms3 = pd.read_hdf(path / f'cons_{cell_type}_all.hdf')
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
        self.errors = pd.Series(index=mi3, dtype=float)
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
    # ##################  SAVING THE CONNECTIVITY IN HDF5  ####################
    # #########################################################################
    # exporting connectivity data
    # there is already a function in con_preprocess_2.py that does the same
    # thing. So not sure if this is not redundant...
    def export_con_data(self):
        """
        this exports the con_stream2 dataset into 3 hdf5 files
        """
        path = OLF_PATH / 'results'
        ct = self.cell_type
        self.con_strms2[0]['o'].to_hdf(path / f'con_{ct}2LN.hdf', f'{ct}2LN')
        self.con_strms2[1]['o'].to_hdf(path / f'con_LN2{ct}.hdf', f'LN2{ct}')
        self.con_strms2[2]['o'].to_hdf(path / f'con_LN2{ct}_indeg.hdf',
                                       f'LN2{ct}2_indeg')


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

# not used for the paper
    # def calc_act_NMF(self, concs, act_pps, k: int = 2, beta=2,
    #                  n_rnd: int = 10):
    #     """
    #     N is the order of the NMF
    #     beta is the parameter in the optimization function of the NMF
    #     n_rnd is the number of random initialization in addition on the
    #     deterministic initializations to try in order to get the best NMF
    #     (i.e., with the smallest error according to the objective function)
    #     """
    #     meth = 'NMF_' + str(k)
    #     print(f'data selections for which NMF will be calculated: {concs}')
    #     for conc in concs:
    #         data = self.act_sels[conc][act_pps].T
    #         # this is the usual way of having data: columns odors, rows: ORNs
    #         W, H, e, init = nmf.get_nmf_best_df(data, k=k, beta=beta,
    #                                             n_rnd=n_rnd)
    #         self.errors[conc, 'NMF', k] = e
    #         for n in range(1, k + 1):
    #             self.act_W[conc, act_pps, meth, beta, init, n] = W[n]
    #             self.act_H[conc, act_pps, meth, beta, init, n] = H.T[n]
    #         print(f'finished conc: {conc}, error: {e}')


    # def calc_act_SNMF(self, concs, act_pps, k: int = 3):
    #     """
    #     Calculating the W that would arise from SNMF
    #     """
    #     meth = 'SNMF_' + str(k)
    #     print(f'data selections for which SNMF will be calculated: {concs}')
    #     ks = np.arange(k) + 1
    #     for conc in concs:
    #         print(f'working on conc: {conc}')
    #         data = self.act_sels[conc][act_pps].T
    #         Z, e, m = nmf.get_snmf_best_df(data, k=k, rtol=1e-6,
    #                                     max_iter=int(1e6))
    #         # print(m)
    #         self.errors[conc, 'SNMF', k] = e
    #         W = data @ Z.T / data.shape[1]
    #         M = Z @ Z.T / data.shape[1]
    #         M = pd.DataFrame(M, index=ks, columns=ks)
    #         for n in range(1, k + 1):
    #             self.act_W[conc, act_pps, meth, '', '', n] = W[n]
    #             self.act_H[conc, act_pps, meth, '', '', n] = Z.T[n]
    #         self.act_M.setdefault(conc, {}).setdefault(act_pps, {})[meth] = M
    #         print(f'finished conc: {conc}, error: {e}')


    def calc_act_NNC(self, concs, act_pps, k: int = 3, alpha=50, cycle=500,
                      rho=1, rtol=1e-7, rectY: bool = True,
                      rectZ: bool = True, beta=0.2):
        """
        Calculating the W that would arise from the non-negative olfactory
        circuit
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
              f'rho: {rho}, act_pps: {act_pps}')
        ks = np.arange(k) + 1
        for conc in concs:
            print(f'working on conc: {conc}')
            data = self.act_sels[conc][act_pps].T
            # here data is oriented as the usual X
            pps = f'{rho}{act_pps}'
            Y, Z, costs = FOC.olf_gd_offline(data.values, k, max_iter=10000,
                                             rectY=rectY, rectZ=rectZ,
                                             rtol=rtol, rho=rho,
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


    # in this new version we work directly with strms3
    def calc_act_con_cc_pv(self, N_iter=10, opt='slow'):
        """
        calculates the significance for every correlation coefficient
        that was already calculated
        The fast and slow versions differ in whether we are shuffling
        every single column separately of the strms3 matrix, or
        if we are shuffling them all at once, with the same permutation.
        """
        if opt == 'fast':
            # signif_f = FG.get_signif_v2
            raise ValueError('fast is not anymore implemented/maintained')
        elif opt == 'slow':
            signif_f = FG.get_signif_v1
        else:
            raise ValueError('no input option opt: ' + opt)

        N = len(self.con_strms3['o'].index)  # the number of ORNs/PNs
        # This function aligns the ORN order, i.e., in the row dimension
        # the function shuffles the second dataset, here con_strms3, so
        # it is better if it has less columns than the other...
        cor, pv_o, pv_l, pv_r = signif_f(self.act_W_cn, self.con_strms3['cn'],
                                         N=N_iter, measure='corr')

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
