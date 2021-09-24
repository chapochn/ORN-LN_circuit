#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:46:14 2017

@author: nchapochnikov
"""


file_tbl = 'data/Si_2019/ORN_data_table.csv'
file_hdf = 'results/act2.hdf'

n_cells = 21

n_conc = 5

n_odors_EC = 34

# this is the order based on the 2nd loading vector of the PCA of ORN activity
# when the full activity data (all concentrations) is centered and normalized
# the pps is raw and mean, that gives the best separation on the correlation
# plot
ORN_order = ['45b', '59a', '30a', '83a', '22c', '24a', '94a/94b', '67b', '1a',
             '63a', '49a', '33a', '82a', '74a', '42a', '33b/47a', '42b', '35a',
             '13a', '85c', '45a']


# this is the order that was in the paper, where there are just 18 cells
# the 3 last ones are are the cells for which there was no activity in the
# first dataset, now the are elicited in the new dataset
ORN_order_Si = ['42b', '74a', '42a', '45a', '82a', '13a', '85c', '33b/47a',
                '35a', '1a', '67b', '30a', '22c', '24a', '94a/94b', '45b',
                '59a', '83a', '33a', '49a', '63a']

# this is just the alphabetical order
ORN = ['1a', '13a', '22c', '24a', '30a', '33a', '33b/47a', '35a', '42a', '42b',
       '45a', '45b', '49a', '59a', '63a', '67b', '74a', '82a', '83a', '85c',
       '94a/94b']

ORN_order = ['ORN '+ name for name in ORN_order]
ORN_order_Si = ['ORN '+ name for name in ORN_order_Si]
ORN = ['ORN '+ name for name in ORN]


# this is the ordered by the 2nd loading vector of the PCA of ORN activity:
odor_order = ['4-methyl-5-vinylthiazole', 'methyl phenyl sulfide',
              'benzaldehyde', '4,5-dimethylthiazole', 'anisole',
              '2-methoxyphenyl acetate', '2-acetylpyridine',
              '2,5-dimethylpyrazine', '4-phenyl-2-butanol', 'menthol',
              'ethyl butyrate', 'myrtenal', '4-methylcyclohexanol',
              '2-phenyl ethanol', 'benzyl acetate', '3-pentanol',
              'pentyl acetate', 'linalool', 'isoamyl acetate', 'nonane',
              '2-nonanone', '1-pentanol', 'acetal', 'butyl acetate',
              'ethyl acetate', 'trans-3-hexen-1-ol', 'geranyl acetate',
              '6-methyl-5-hepten-2-ol', 'trans,trans-2,4-nonadienal',
              '3-octanol', 'hexyl acetate', '4-hexen-3-one']

# this is not any particular order, it is just the order that arises
# when importing the data from the data they gave me - original order
odor_order_o = ['1-pentanol', '3-pentanol', '6-methyl-5-hepten-2-ol',
                '3-octanol', 'trans-3-hexen-1-ol', 'methyl phenyl sulfide',
                'anisole', '2-acetylpyridine', '2,5-dimethylpyrazine',
                'pentyl acetate', 'geranyl acetate', '2-methoxyphenyl acetate',
                'trans,trans-2,4-nonadienal', '4-methyl-5-vinylthiazole',
                '4,5-dimethylthiazole', '4-hexen-3-one', '2-nonanone',
                'acetal', '2-phenyl ethanol', 'butyl acetate', 'ethyl acetate',
                'benzaldehyde', 'ethyl butyrate', 'isoamyl acetate',
                '4-methylcyclohexanol', 'hexyl acetate', 'linalool',
                'benzyl acetate', '4-phenyl-2-butanol', 'myrtenal', 'menthol',
                'nonane']
