"""
created in 2019

@author: Nikolai M Chapochnikov

Calculating the significance of the relationship between
M and sqrt(W.T*W)
This files doesn't save any data, one should just use the significance values
 that are printed in the prompt


"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions.olfactory as FO
import functions.general as FG
import scipy.linalg as LA

# %%

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
# LNs_sel['a'] = LNs_sel1
# d like dendrite of the Picky
LNs_sel['d'] = LNs_sel1 + ['Picky 0 [dend]']
# LNs_sel['d'] = LNs_sel1

def LNs(key, S):
    return [f'{name} {S}' for name in LNs_sel[key]]

con = FO.get_con_data()

con_strms3 = pd.read_hdf(FO.OLF_PATH / 'results/cons/cons_ORN_all.hdf')

# %%

s = 'L'
M_L = con[s].loc[LNs('a', s), LNs('d', s)].copy()
W_f_L = con_strms3[0][LNs('d', s)]
W_b_L = con_strms3[1][LNs('a', s)]


s = 'R'
M_R = con[s].loc[LNs('a', s), LNs('d', s)].copy()
W_f_R = con_strms3[0][LNs('d', s)]
W_b_R = con_strms3[1][LNs('a', s)]

WTW_L = W_f_L.T @ W_f_L
WTW_R = W_f_R.T @ W_f_R

def signif_sqrtWTW_M(W, M_entries, N, diag):
    cc_shfl = np.zeros(N)
    for i in range(N):
        W_shfl = FG.shuffle_matrix(W)
        WTW_shfl = LA.sqrtm(W_shfl.T @ W_shfl)
        W_entr_shfl = FG.get_entries(WTW_shfl, diag=diag)
        cc_shfl[i] = np.corrcoef(W_entr_shfl, M_entries)[0, 1]
    return cc_shfl

def signif_sqrtWTW_M_LR(WL, WR, M_entries, N, diag):
    # shuffling the columns of W
    cc_shfl = np.zeros(N)
    for i in range(N):
        WL_shfl = FG.shuffle_matrix(WL)
        WR_shfl = FG.shuffle_matrix(WR)
        WTW_L_shfl = LA.sqrtm(WL_shfl.T @ WL_shfl)
        WTW_R_shfl = LA.sqrtm(WR_shfl.T @ WR_shfl)
        W_entr_shfl = np.concatenate([FG.get_entries(WTW_L_shfl, diag=diag),
                                      FG.get_entries(WTW_R_shfl, diag=diag)])
        cc_shfl[i] = np.corrcoef(W_entr_shfl, M_entries)[0, 1]
    return cc_shfl

def signif_sqrtWTW_M_LR2(WL, WR, M_entries, N, diag):
    # shuffling the rows of W, meaning that the connectivity of each single ORN
    # remains the same
    cc_shfl = np.zeros(N)
    for i in range(N):
        WL_shfl = FG.shuffle_matrix(WL.T).T
        WR_shfl = FG.shuffle_matrix(WR.T).T
        WTW_L_shfl = LA.sqrtm(WL_shfl.T @ WL_shfl)
        WTW_R_shfl = LA.sqrtm(WR_shfl.T @ WR_shfl)
        W_entr_shfl = np.concatenate([FG.get_entries(WTW_L_shfl, diag=diag),
                                      FG.get_entries(WTW_R_shfl, diag=diag)])
        cc_shfl[i] = np.corrcoef(W_entr_shfl, M_entries)[0, 1]
    return cc_shfl


def signif_sqrtWTW_M_LRbis(WL, WR, M_entries, N, diag):
    # shuffling both W and M
    cc_shfl = np.zeros(N)
    for i in range(N):
        WL_shfl = FG.shuffle_matrix(WL)
        WR_shfl = FG.shuffle_matrix(WR)
        WTW_L_shfl = LA.sqrtm(WL_shfl.T @ WL_shfl)
        WTW_R_shfl = LA.sqrtm(WR_shfl.T @ WR_shfl)
        W_entr_shfl = np.concatenate([FG.get_entries(WTW_L_shfl, diag=diag),
                                      FG.get_entries(WTW_R_shfl, diag=diag)])
        cc_shfl[i] = np.corrcoef(W_entr_shfl,
                                 np.random.permutation(M_entries))[0, 1]
    return cc_shfl

# %%
# relationship between sqrt(WtW) and M (as theory, excluding the diagonal
# to which we have no access)

N = 10000
diag = False

# there is a unique symmetic square root
W_entries = np.concatenate([FG.get_entries(LA.sqrtm(WTW_L), diag=diag),
                            FG.get_entries(LA.sqrtm(WTW_R), diag=diag)])

M_entries = np.concatenate([FG.get_entries(M_L, diag=diag),
                            FG.get_entries(M_R, diag=diag)])
plt.figure()
plt.scatter(M_entries, W_entries)
plt.title('WTW vs M, L&R')
plt.show()
cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
cc_shfl = signif_sqrtWTW_M_LR(W_f_L.values, W_f_R.values,
                              M_entries, N, diag)
pval = np.mean(cc_shfl > cc_real)
print('sqrt(WTW) vs M, on LR, cc', cc_real, 'pval', pval)

cc_shfl = signif_sqrtWTW_M_LRbis(W_f_L.values, W_f_R.values,
                              M_entries, N, diag)
pval = np.mean(cc_shfl > cc_real)
print('sqrt(WTW) vs M, on LR, shuffling M as well, cc', cc_real, 'pval', pval)

# this is not used in the paper:

# shuffling the rows of W instead of the columns, so that the total synaptic
# strenght of each ORN remains the same
# cc_shfl = signif_sqrtWTW_M_LR2(W_f_L.values, W_f_R.values,
#                               M_entries, N, diag)
# pval = np.mean(cc_shfl > cc_real)
# print('sqrt(WTW) vs M, on LR, cc', cc_real, 'pval', pval)



# W_entries = FG.get_entries(LA.sqrtm(WTW_L), diag=diag)
# M_entries = FG.get_entries(M_L, diag=diag)
# plt.figure()
# plt.scatter(M_entries, W_entries)
# plt.title('sqrt(WTW) vs M, L')
# plt.show()
# cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
# cc_shfl = signif_sqrtWTW_M(W_f_L.values, M_entries, N, diag)
# pval = np.mean(cc_shfl > cc_real)
# print('sqrt(WTW) vs M, on L, cc', cc_real, 'pval', pval)
#
#
# W_entries = FG.get_entries(LA.sqrtm(WTW_R), diag=diag)
# M_entries = FG.get_entries(M_R, diag=diag)
# plt.figure()
# plt.scatter(M_entries, W_entries)
# plt.title('sqrt(WTW) vs M, R')
# plt.show()
# cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
# cc_shfl = signif_sqrtWTW_M(W_f_R.values, M_entries, N, diag)
# pval = np.mean(cc_shfl > cc_real)
# print('sqrt(WTW) vs M, on R, cc', cc_real, 'pval', pval)

