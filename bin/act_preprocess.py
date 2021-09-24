#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:45:19 2018

@author: nchapochnikov

functions to import and preprocess the activity data, like the ORN data,
the LN data, the additional odors data, the EC50 etc.. and export them here
so that one can immediatly import the hdf5  when one do the analysis,
and so that these steps are well separated.
"""

# %%

from functions import general as FG, olfactory as FO
import pandas as pd
import numpy as np
import scipy.io as sio 
import importlib

# %%
importlib.reload(FO)

# %%
# #############################################################################
# #############  IMPORTING AND SAVING THE ACTIVITY DATA  ######################
# #############################################################################

act3 = FO.get_ORN_act_data_3(fast=False)
# %%

act3.to_hdf(FO.OLF_PATH / 'results/act3.hdf', 'act3')
