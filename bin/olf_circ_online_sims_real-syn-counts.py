"""
Calculating the output of a model circuit
where the connections weights are proportional to the connection counts
from the connectome

@author: Nikolai M Chapochnikov
"""
#%%

import numpy as np
import pandas as pd
import functions.olfactory as FO
import params.act3 as par_act
import scipy.linalg as LA
import functions.circuit_simulation as FCS
import importlib

# %%


OLF_PATH = FO.OLF_PATH
RESULTS_PATH = OLF_PATH / 'results'

SIDES = ['L', 'R']
LNs_sel = {}
LNs_sel1 = ['Broad T1',
            'Broad T2',
            'Broad T3',
            'Broad D1',
            'Broad D2',
            'Keystone L',
            'Keystone R']

# a like axon of the Picky
LNs_sel['a'] = LNs_sel1 + ['Picky 0 [axon]']
# d like dendrite of the Picky
LNs_sel['d'] = LNs_sel1 + ['Picky 0 [dend]']

def LNs(key, S):
    return [f'{name} {S}' for name in LNs_sel[key]]

con = FO.get_con_data()
con_strms3 = pd.read_hdf(RESULTS_PATH / 'cons/cons_ORN_all.hdf').loc[par_act.ORN_order]
W_f = {}  # feedforward
W_b = {}  # feedback
M = {}
for s in ['L', 'R']:
    M[s] = con[s].loc[LNs('a', s), LNs('d', s)].copy()
    W_f[s] = con_strms3[0][LNs('d', s)]
    W_b[s] = con_strms3[1][LNs('a', s)]


odor_order = par_act.odor_order
ORN_order = par_act.ORN_order

DATASET = 3
act = FO.get_ORN_act_data(DATASET).T
act_m = act.groupby(axis=1, level=('odor', 'conc')).mean()
act_m = act_m.loc[ORN_order, :]

def order_act_df(X):
    conc_levels = X.columns.unique(level='conc')
    idx = pd.MultiIndex.from_product([odor_order, np.sort(conc_levels)[::-1]])
    return X.loc[:, idx]

X = order_act_df(act_m)

# this is what motivates to put the scalings below
for side in ['L', 'R']:
    norms = LA.norm(W_f[side].values, axis=0)
    print(side, 'W_ff', norms, 'mean norm:', np.mean(norms))#, np.mean(W_f[side].values))
    norms = LA.norm(W_b[side].values, axis=0)
    print(side, 'W_fb', norms, 'mean norm:', np.mean(norms))#, np.mean(W_b[side].values))
    norms = LA.norm(M[side].values, axis=0)
    print(side, 'M', norms, 'mean norm:', np.mean(norms))#, np.mean(M[side].values))
# %%
# the activity is the same as when we do the simulations with the LC or NNC
# the only thing we need to do is scale the W_f

factor_ff = 80
factor_fb = 30
factor_LN = 60

# factor_ff = 80
# factor_fb = 30
# factor_LN = 120


res_path = RESULTS_PATH / 'sims_real-synaptic-counts'
res_path.mkdir(exist_ok=True)

for side in ['L', 'R']:

    # normalization based on the above
    W_f_crt = W_f[side].values/factor_ff
    W_b_crt = W_b[side].values/factor_fb
    M_crt = (M[side].values + 1*np.diag(M[side].max(axis=0)))/factor_LN  # fixes the values on the diagonal

    Y = X.copy()
    Y[:], Z = FCS.olf_output_online_bulk(X.values, W_f_crt, W_b_crt, M_crt, rho=1,
                                         method='GD_NN')

    Z = pd.DataFrame(Z, index=M[side].index, columns=Y.columns)

    file = res_path / f'Y_{side}.h5'
    Y.to_hdf(file, 'Y')
    file = res_path / f'Z_{side}.h5'
    Z.to_hdf(file, 'Z')

print('final done')


#%%
# comparing with previous simulation
# import pathlib
# res_path_bu = '/Users/chapon01/ORN-LN_circuit/results_bu/sims_real-synaptic-counts'
# for side in ['L', 'R']:
#     Y_bu = pd.read_hdf(pathlib.Path(res_path_bu) / f'Y_{side}.h5')
#     Y = pd.read_hdf(pathlib.Path(res_path) / f'Y_{side}.h5')
#     print(np.max(np.abs(Y - Y_bu).values))
#     Z_bu = pd.read_hdf(pathlib.Path(res_path_bu) / f'Z_{side}.h5')
#     Z = pd.read_hdf(pathlib.Path(res_path) / f'Z_{side}.h5')
#     print(np.max(np.abs(Z - Z_bu).values))