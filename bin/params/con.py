#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:24:44 2017

@author: nchapochnikov
"""

file_L = 'data/Berck_2016/elife-14859-supp1-v2_no_blanks.xlsx'
file_R = 'data/Berck_2016/elife-14859-supp2-v2_no_blanks.xlsx'

# this is some information from the excel sheet
# not sure if it is meaningful to put it here, or somewhere else
# or to read out from some parameter file. Let's leave it here for the moment
ORN_begin = 2
ORN_end = 22
ORN_n = ORN_end - ORN_begin + 1

uPN_begin = 23
uPN_end = 44
uPN_n = uPN_end - uPN_begin + 1

Broad_begin = 45
Broad_end = 49
Broad_n = Broad_end - Broad_begin + 1

Picky_begin = 52
Picky_end = 61
Picky_n = Picky_end - Picky_begin + 1

Choosy_begin = 62
Choosy_end = 65
Choosy_n = Choosy_end - Choosy_begin + 1

mPN_begin = 67
mPN_end = 84
mPN_n = mPN_end - mPN_begin + 1

all_begin = 2
all_end = 97
all_n = all_end - all_begin + 1


ORN = ['1a', '13a', '22c', '24a', '30a', '33a', '35a', '42a', '42b', '45a',
       '45b', '33b/47a', '49a', '59a', '63a', '67b', '74a', '82a', '83a',
       '85c', '94a/94b']

ORN = ['ORN ' + name for name in ORN]
ORN_L = [name + ' L' for name in ORN]
ORN_R = [name + ' R' for name in ORN]
ORN_A = [name + ' L&R' for name in ORN]  # A being all

uPN = ['1a', '13a', '22c', '24a', '30a', '33a', '35a bil. L',
       '35a bil. R', '42a', '42b', '45a', '45b', '33b/47a', '49a', '59a',
       '63a', '67b', '74a', '82a', '83a', '85c', '94a/94b']

uPN_merg = ['1a', '13a', '22c', '24a', '30a', '33a', '35a bil. L',
            '42a', '42b', '45a', '45b', '33b/47a', '49a', '59a',
            '63a', '67b', '74a', '82a', '83a', '85c', '94a/94b']

uPN = ['PN ' + name for name in uPN]
uPN_merg = ['PN ' + name for name in uPN_merg]

LN = ['Broad T1', 'Broad T2', 'Broad T3', 'Broad D1', 'Broad D2',
      'Keystone L', 'Keystone R', 'Picky 0 [dend]',
      'Picky 0 [axon]', 'Picky 1 [dend]', 'Picky 1 [axon]',
      'Picky 2 [dend]', 'Picky 2 [axon]',
      'Picky 3 [dend]', 'Picky 3 [axon]',
      'Picky 4 [dend]', 'Picky 4 [axon]',
      'Choosy 1 [dend]', 'Choosy 1 [axon]',
      'Choosy 2 [dend]', 'Choosy 2 [axon]', 'Ventral LN']

dict_str_replace = {'47a & 33b': '33b/47a', '94a & 94b': '94a/94b'}
