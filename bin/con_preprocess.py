#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:12:28 2018

@author: nchapochnikov

better would be to create some functions here to make to code clearer
but will leave for later if needed

the point of this file is just to read the connectivity data and to export
it into formats that are readable. So basically i would do it just once
and then i would only use the exported datasets, which are already
organized the way i want


description of what is happening in this file
of the connectivity data
1. the connectivity data is read and put into a dict with L, R, M
2. the variable con_strms is created, which has the following
organization [L/R/M][0/1/2][o/n/c/cn][ORN:LN]
3. the variable con_strms2 is created:
[0/1/2][o/n/c/cn][ORN:LN], the L, R, and M have been put together
4. finally, the variable strms3 is created (which i think
should be used everywhere, but because the code has not been updated
i still mainly use strms2): [o/n/c/cn][ORN:(0/1/2,LN)]
as above, the L, R, and M have been put together in the LNs
in con_strms3, there has already been a selection and ordering,
however i believe this should be done only later for more clarity

in this version we don't cut on any targets, all targets are kept.
I guess one should preferable use this version in the future

"""

import pandas as pd
import itertools
import importlib

from functions import general as FG, olfactory as FO
import params.con as par_con

# %%
importlib.reload(FO)
importlib.reload(par_con)

# %%
# #############################################################################
# ####################  CREATING AND EXPORTING CONS  ##########################
# #############################################################################

SIDES = ['L', 'R', 'M', 'S']


cons = FO.import_con_data(SIDES)  # get con. data
# con = cons[key_con_1]  # M: mean, L:left, R:right
# %%
# exporting all the connection datasets to hdfs files
for k, v in cons.items():
    v.to_hdf(FO.OLF_PATH / f'results/cons/cons_full_{k}.hdf', f'cons_{k}')


# %%

# #############################################################################
# #############################  CREATING STRMS1  #############################
# #############################################################################
SIDES = ['L', 'R', 'M']
cons = FO.import_con_data(SIDES)  # get con data

# here we need to choose which neuron type we wnat to work with, ORN or uPN:
cell = 'ORN'
# cell = 'uPN'
if cell == 'ORN':
    neur_order = par_con.ORN
elif cell == 'uPN':
    neur_order = par_con.uPN
STRMS = [0, 1, 2, 3]
# Adding "cells" which correspond to the sums/means of connectivity
# of the Broad Trio and Broad Duet
cons = FO.create_summary_LNs(cons, func='mean', S='M')

# in neur_full are the ORN/uPN names with the full name as present
# in the connectivity data, but only the subselection that is actually
# found in the activity data
neur_full = {}

# in con_strms are the sides, the directions and the ppps, but
# selected for the ORNs that we are actually looking at.
con_strms = {}

# n_con containts the number of connections from or towards
# any cell with the ORN/uPN set
# =============================================================================
# mi = pd.MultiIndex(levels=[[], []], labels=[[], []],
#                    names=['side', 'direction'])
# n_con = pd.DataFrame(columns=mi)
# =============================================================================

for s in SIDES:
    neur_names_con = list(cons[s].index)
    # print(neur_names_con)
    # print(neur_order)

    # the following function will have that neur_full has the same
    # order as neur_order
    neur_full[s] = FG.get_match_names(neur_order, neur_names_con)
    con_strms[s] = FO.get_con_pps(cons[s], neur_full[s])
    # 0: sel2A, 1: A2sel, 2:A2sel: scaled by the input of each neuron
# =============================================================================
#     for d in strms:
#         con_o = con_strms[s][d]['o']
#         n_con[s, d] = np.sign(con_o).sum().values.astype(int)
# =============================================================================

# these are the pps that are outputted by get_con_pps
PPS = ['o', 'c', 'n', 'cn']
# the structure of con_strms is con_strms[s][d][pps]
# %%
# #####################################################################
# #####################  SELECTING TARGETS  ##########################
# #####################################################################
# we are selecting which neurons we want to keep. We are basically selecting
# all the targets after the uPNs

for s, d in itertools.product(SIDES, STRMS):
    con_strms[s][d] = con_strms[s][d]['o'].iloc[:, 43:]

# %%
# #############################################################################
# #############################  CREATING STRMS2  #############################
# #############################################################################
# =============================================================================
# self.n_con = self.n_con.loc[neur_targets]
# =============================================================================

# in con_strms2, the L, R sides are together, meaning that
# the L and R index is now blended
con_strms2 = {}
for d in STRMS:
    dfs = {}  # dict of dataframes
    for s in SIDES:
        dfs[s] = con_strms[s][d].copy()
        # in neur_sort_shorter there are no more side reference
        neur_sort_shorter = [name[:-2] for name in list(dfs[s].index)]
        dct = dict(zip(dfs[s].index, neur_sort_shorter))
        dfs[s].rename(index=dct, inplace=True)
    con_strms2[d] = pd.concat([dfs[s] for s in SIDES], axis=1)


for k, df in con_strms2.items():
    names = list(df.columns)
    seen = {}
    for i, x in enumerate(names):
        if x not in seen:
            seen[x] = 1
        else:
            seen[x] += 1
            names[i] = f'{names[i]} {seen[x]}'

    df.columns = names
    df.to_hdf(FO.OLF_PATH / f'results/cons/cons_{cell}_{k}_all.hdf',
              f'con_{cell}_{k}')

# choosing which neurons we want to work with, from the already
# selected cells above. Actually just putting an order
# this could be a parameter

# to decide from outside which neurons to choose, we can
# call: self.con_strms2[0]['o'].columns
# right now we just have a hardcoded version

# %%
# #############################################################################
# #############################  CREATING STRMS3  #############################
# #############################################################################
# =============================================================================
# self.LNs2 = [7, 17, 8, 18, 9, 19]
# self.LNs = [0, 1, 2, 7, 10, 11, 12, 17, 27,  # Trios
#             3, 4, 8, 13, 14, 18, 28,  # Duets
#             5, 6, 15, 16, 25, 26, 29]  # Keystones
# self.splits_LN = [9, 9 + 7]
# =============================================================================
# splits for the graphs between the different cell categories


# this dataset puts all the directions together
# and only works with the LNs in self.LNs
# might be restrictive, but seems easier to get what we want
d0 = con_strms2[0].copy().T  # .iloc[LNs]
d0['strm'] = 0
d1 = con_strms2[1].copy().T  # .iloc[LNs]
d1['strm'] = 1
d2 = con_strms2[2].copy().T  # .iloc[LNs]
d2['strm'] = 2
d3 = con_strms2[3].copy().T  # .iloc[LNs]
d3['strm'] = 3
dnew = pd.concat([d0, d1, d2, d3])
dnew = dnew.reset_index()
dnew = dnew.rename(columns={'index': 'cells'})
dnew = dnew.set_index(['strm', 'cells'])
con_strms3 = dnew.copy().T

con_strms3.to_hdf(FO.OLF_PATH / f'results/cons/cons_{cell}_all.hdf',
                  f'con_{cell}')

