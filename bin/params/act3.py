#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:27:11 2018

@author: Nikolai M Chapochnikov
"""

file_hdf = 'results/act3.hdf'

n_cells = 21

n_conc = 5

n_odors_EC = 34

# this is the order based on the 2nd loading vector of the PCA of ORN activity
# when the full activity data (all concentrations) is centered and normalized
# the pps is raw and mean, that gives the best separation on the correlation
# plot
ORN_order = ['45b', '24a', '59a', '30a', '83a', '22c', '94a/94b', '1a', '63a',
             '67b', '49a', '33a', '33b/47a', '82a', '13a', '35a', '85c', '74a',
             '45a', '42a', '42b']


# this is the order that was in the paper
ORN_order_Si = [
    '49a',
    '94a/94b',
    '83a',
    '59a',
    '45b',
    '30a',
    '24a',
    '22c',
    '63a',
    '1a',
    '67b',
    '35a',
    '85c',
    '13a',
    '33b/47a',
    '45a',
    '74a',
    '42a',
    '42b',
    '82a',
    '33a',
]

# this is just the alphabetical order
ORN = ['1a', '13a', '22c', '24a', '30a', '33a', '35a', '42a', '42b', '45a',
      '45b', '33b/47a', '49a', '59a', '63a', '67b', '74a', '82a', '83a', '85c',
      '94a/94b']

ORN_order = [f'ORN {name}' for name in ORN_order]
ORN_order_Si = [f'ORN {name}' for name in ORN_order_Si]
ORN = [f'ORN {name}' for name in ORN]


# this is the ordered by the 2nd loading vector of the PCA of ORN activity:
odor_order = ['methyl phenyl sulfide', '4-methyl-5-vinylthiazole',
       'methyl salicylate', 'benzaldehyde', '4,5-dimethylthiazole', 'anisole',
       '2-methoxyphenyl acetate', '2-acetylpyridine', '4-phenyl-2-butanol',
       'pentyl acetate', '2,5-dimethylpyrazine', 'menthol', 'benzyl acetate',
       '2-nonanone', 'ethyl butyrate', 'myrtenal', '2-phenyl ethanol',
       '4-methylcyclohexanol', '3-pentanol', 'isoamyl acetate', 'linalool',
       '1-pentanol', '6-methyl-5-hepten-2-ol', 'butyl acetate', '3-octanol',
       'hexyl acetate', 'trans-3-hexen-1-ol', 'nonane', 'geranyl acetate',
       'acetal', '2-heptanone', 'ethyl acetate', 'trans,trans-2,4-nonadienal',
       '4-hexen-3-one']

# from their paper
odor_order_Si = [
    '2-phenyl ethanol',
    'menthol',
    '2-methoxyphenyl acetate',
    'myrtenal',
    '4-methylcyclohexanol',
    '4,5-dimethylthiazole',
    '4-methyl-5-vinylthiazole',
    'anisole',
    '2-acetylpyridine',
    '2,5-dimethylpyrazine',
    'methyl phenyl sulfide',
    'benzaldehyde',
    'methyl salicylate',
    '4-phenyl-2-butanol',
    'benzyl acetate',
    'linalool',
    '1-pentanol',
    'trans-3-hexen-1-ol',
    'pentyl acetate',
    '6-methyl-5-hepten-2-ol',
    '2-heptanone',
    '3-octanol',
    'butyl acetate',
    'hexyl acetate',
    'isoamyl acetate',
    '2-nonanone',
    'trans,trans-2,4-nonadienal',
    'ethyl butyrate',
    '4-hexen-3-one',
    '3-pentanol',
    'ethyl acetate',
    'geranyl acetate',
    'nonane',
    'acetal',
              ]

# this is not any particular order, it is just the order that arises
# when importing the data from the data they gave me - original order
odor_order_o = ['1-pentanol', '3-pentanol', '6-methyl-5-hepten-2-ol',
                '3-octanol', 'trans-3-hexen-1-ol', 'methyl phenyl sulfide',
                'anisole', '2-acetylpyridine', '2,5-dimethylpyrazine', 
                'pentyl acetate', 'geranyl acetate', '2-methoxyphenyl acetate',
                'trans,trans-2,4-nonadienal', '4-methyl-5-vinylthiazole',
                '4,5-dimethylthiazole', '4-hexen-3-one', '2-nonanone',
                'acetal', '2-phenyl ethanol', 'butyl acetate', 'ethyl acetate',
                'benzaldehyde', '2-heptanone', 'methyl salicylate',
                'ethyl butyrate', 'isoamyl acetate', '4-methylcyclohexanol',
                'hexyl acetate', 'linalool', 'benzyl acetate',
                '4-phenyl-2-butanol','myrtenal', 'menthol', 'nonane']