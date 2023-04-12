#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:56:59 2019

@author: Nikolai M Chapochnikov

This plots all the connectivity related plots in the paper

External files that are read and used for this plotting:
in plots_paper_import:
f'results/cons/cons_full_{k}.hdf'
'results/act3.hdf' FROM preprocess-activity.py
'results/cons/cons_ORN_all.hdf' FROM preprocess-connectivity.py

'W_NNC-8.hdf'  FROM olf_circ_offline_sims_ORN-data.py

for sqrt(WW) vs M: significance is calculated in the file con_analysis_MvsW.py
"""

# %%
# ################################# IMPORTS ###################################

from plots_import import *
# %%
# ################################  RELOADS  ##################################
importlib.reload(FO)
importlib.reload(FG)
importlib.reload(FP)
importlib.reload(FCS)
# importlib.reload(par_con)
importlib.reload(par_act)
# %%
# #############################################################################
# ###################  PLOTTING ORN FULL CONNECTIVITY  ########################
# #############################################################################

# plotting ff and fb plots together, and also putting some more details
# onto the plots, as the categories of the neurons
# importlib.reload(FG)

pads = [0.6, 0.55, 0.1, 0.2]
d_h = 1.4

arr_y = 21 + 13  # arrow y position
txt_y = arr_y + 0.5  # annotation text y position
arr_stl = "<->, head_width=0.3"
kwargs = {'annotation_clip': False,
          'arrowprops': dict(arrowstyle=arr_stl, color='C0')}
kwargs2 = {'ha': 'center', 'va': 'top', 'color': 'C0', 'size':ft_s_lb}
for s in ['L', 'R']:
    # top plot
    df1 = con_ff[s].loc[:, cells_all2[s][21:]]
    df2 = con_fb[s].loc[:, cells_all2[s][21:]]
    fs, axs1, axs2, _ = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, 0, 0)

    # _, fs, axs = FP.calc_fs_ax_df(df1, pads1, sq=SQ)
    # print(SQ, SQ*np.array(df.shape), fs, axs)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs1)
    cp = FP.imshow_df(df1, ax1, vlim=[0, 125],
                      title=r'feedforward connections: ORNs$\rightarrow$'
                             r'other neurons')
    ax1.set(ylabel='ORN', xlabel='')
    # arrows
    # plt.arrow(0.5, 35, 19.5, 0, lw=2, clip_on=False,
    # length_includes_head=True)
    ax1.annotate('', xy=(-0.9, arr_y), xytext=(21.9, arr_y), **kwargs)
    ax1.annotate('', xy=(21.1, arr_y), xytext=(39.9, arr_y), **kwargs)
    ax1.annotate('', xy=(39.1, arr_y), xytext=(52.9, arr_y), **kwargs)
    ax1.annotate('', xy=(52.1, arr_y), xytext=(74.9, arr_y), **kwargs)
    ax1.text(21/2, txt_y, 'uniglomerular\nprojection neurons (uPNs)', **kwargs2)
    ax1.text((21 + 40)/2, txt_y, 'multiglomerular\nprojection neurons (mPNs)',
             **kwargs2)
    ax1.text((39 + 53)/2, txt_y, 'other neurons', **kwargs2)
    ax1.text((52 + 75)/2, txt_y, 'Inhibitory\nlocal neurons (LNs)', **kwargs2)
    # colorbar
    cb_x = axs1[0] + axs1[2] + CB_DX/fs[0]
    ax_cb = f.add_axes([cb_x, axs1[1], CB_W/fs[0], axs1[3]])
    add_colorbar_crt(cp, ax_cb, '\# of synapses', [0, 40, 80, 120])

    # bottom plot
    # _, _, axs = FP.calc_fs_ax_df(df, pads2, sq=SQ)
    ax2 = f.add_axes(axs2)
    cp = FP.imshow_df(df2, ax2, vlim=[0, 21], show_lab_x=False,
                      title=r'feedback connections: other neurons'
                             r'$\rightarrow$ORNs')
    ax2.set(ylabel='ORN', xlabel='')
    # colorbar
    ax_cb = f.add_axes([cb_x, axs2[1], CB_W/fs[0], axs2[3]])
    add_colorbar_crt(cp, ax_cb, '\# of synapses', [0, 10, 20])

    # rectangle
    rect = mpl.patches.Rectangle((53 - 0.6, -41.8), 9.2, 62.5, lw=1,
                                        clip_on=False, ls='--',
                                        edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    file = f'{PP_CONN}/con_ORN_{s}_all.'
    FP.save_plot(f, file + 'png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + 'pdf', SAVE_PLOTS, **pdf_opts)

    # this seems to try to make it even tighter than what you set it up for
    # pdf_opts1 = {'dpi': 800, 'transparent': True, 'bbox_inches':'tight',
    #              'pad_inches': 0}
    # file1 = f'{PP_CONN}/con_ORN_{s}_all2.'
    # FP.save_plot(f, file1 + 'pdf', SAVE_PLOTS, **pdf_opts1)

print('done')

# %%
# #############################################################################
# ###################  PLOTTING ORN SEL. CONNECTIVITY  ########################
# #############################################################################

# keeping the same height as the plot above and letting the width adapt
# basically keeping the same square size

# it is also possible to look at the feedback stream, then you would also
# need to change the name of the file

strm = 0

pads = (0.6, 0.55, 0.80, 0.35)  # l, r, b, t
df = con_strms3.loc[:, strm].copy()
df = df.loc[ORNs_sorted, LNs_sel_LR_d]
df.index = ORNs_sorted_short
# df.columns = LNs_sel2_short
_, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)

splx = [6, 6+4, 6+4+4]

cp = FP.imshow_df(df, ax1, vlim=[0, 55], splits_x=splx, lw=0.5,
                  title=r'ORNs$\rightarrow$LNs' + f'\n syn. counts')
                      #   + r'$\mathbf{w}_\mathrm{LN}^\mathrm{ff}$')
ax1.set(ylabel='ORN', xlabel='LN')

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, '\# of syn.', [0, 20, 40])

file = f'{PP_CONN}/con_ORN_ff_LN_sel.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)


strm = 1

# pads = (0.6, 0.55, 0.81, 0.32)  # l, r, b, t
df = con_strms3.loc[:, strm].copy()
df = df.loc[ORNs_sorted, LNs_sel_LR_a]
df.index = ORNs_sorted_short
# df.columns = LNs_sel2_short
_, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)

splx = [6, 6+4, 6+4+4]

cp = FP.imshow_df(df, ax1, vlim=[0, 21], splits_x=splx, lw=0.5,
                  title=r'LNs$\rightarrow$ORNs'+ f'\n syn. counts')
                 #        + r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
ax1.set(ylabel='ORN', xlabel='LN')

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, '\# of syn.', [0, 10, 20])

file = f'{PP_CONN}/con_ORN_fb_LN_sel.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# #############################################################################
# ###################  SHOWING THE VARIANCE FOR EACH CON CAT  #################
# #############################################################################
# from PIL import Image


# LNs_cat = ['BT', 'BD', 'KS', 'P0']

strm = 0

col1 = 'k'
col_e = 'orangered'
col_e = 'k'


LNs_cat = {'BT': ['Broad T1 L', 'Broad T2 L', 'Broad T3 L',
                  'Broad T1 R', 'Broad T2 R', 'Broad T3 R'],
           'BD': ['Broad D1 L', 'Broad D2 L', 'Broad D1 R', 'Broad D2 R'],
           'KS': ['Keystone L L', 'Keystone L R', 'Keystone R R',
                  'Keystone R L'],
           'P0': ['Picky 0 [dend] L', 'Picky 0 [dend] R']}

LNs_m = {'BT': 'Broad T M M',
         'BD': 'Broad D M M',
         'KS': 'Keystone M M',
         'P0': 'Picky 0 [dend] M'}

LNs_name = {'BT': 'Broad Trio (BT, 6)',
            'BD': 'Broad Duet (BD, 4)',
            'KS': 'Keystone (KS, 4)',
            'P0': 'Picky 0 (P0, 2)'}


# adding in the background lines showing the variance from a Poisson process.
LNs_ymax = {'BT': 56,
            'BD': 42,
            'KS': 33,
            'P0': 45}

LNs_yticks = {'BT': [0, 20, 40],
              'BD': [0, 20, 40],
              'KS': [0, 20],
              'P0': [0, 20, 40]}

for LN, LN_list in LNs_cat.items():

    LN_m = LNs_m[LN]
    data = con_strms3.loc[:, strm]
    data = data.loc[ORNs_sorted, LN_m].copy()
    data.index = ORNs_sorted_short
    cell_list = data.index

    data1 = con_strms3.loc[:, strm].loc[ORNs_sorted, LN_list].copy()
    data_std = data1.std(axis=1)  # normalized by n-1, which is the unbiased
    # version of the std
    data_m = data1.mean(axis=1)

    assert np.array_equal(data.values,data_m.values)

    if LN == 'BT':
        pads = [0.55, 0.12, 0.35, 0.2]
    else:
        pads = [0.12, 0.12, 0.35, 0.2]

    fs, ax1 = FP.calc_fs_ax(pads, 10*SQ, 21*SQ)
    f = plt.figure(figsize=fs)
    ax = f.add_axes(ax1)

    # mean = data_m.mean()
    # std = SS.poisson.std(mean)
    # n_total = data_m.sum()
    # std2 = SS.binom.std(n_total, p=1/21)
    # print(std, std2)

    ax.errorbar(data, np.arange(len(data)), c=col1, xerr=data_std,
                elinewidth=0.5, ecolor=col_e, capsize=1, lw=1, zorder=20)
    for i in range(data1.shape[1]):
        ax.plot(data1.iloc[:, i], range(data1.shape[0]), lw=0.5)
    ax.set_xlim(-2, LNs_ymax[LN])
    ax.set_ylim(-0.5, 20.5)
    ax.set_title(LNs_name[LN])
    ax.invert_yaxis()

    ax.set_xticks(LNs_yticks[LN])
    ax.set_yticks(np.arange(len(cell_list)), cell_list)
    if LN == 'BT':
        ax.tick_params(axis='y', left=False, direction='in')
        ax.set_ylabel('ORN')
    else:
        ax.tick_params(axis='y', left=False, labelleft=False)


    ax.yaxis.grid(zorder=0)


    file = f'{PP_CONN}/con_{LN}'
    FP.save_plot(f, file+'.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #############################################################################
# #########################  CONNECTION CORRELATIONS  #########################
# #############################################################################
# here, to conserve the dimensions, i would like to have the height and the
# width exactly corresponding to the width of the previous graph, which
# contains all the LNs
wLNtex = r'$\mathbf{w}_{\mathrm{LN}}$'

for strm in [0, 1]:
    # Correlation coefficient per category
    cat = {}
    cat['BT'] = ['BT 1 L', 'BT 2 L', 'BT 3 L', 'BT 1 R', 'BT 2 R',
                 'BT 3 R']
    cat['BD'] = ['BD 1 L', 'BD 2 L', 'BD 1 R', 'BD 2 R']
    cat['KS'] = ['KS L L', 'KS L R', 'KS R R', 'KS R L']
    cat['P0'] = ['P0 L', 'P0 R']
    if strm == 0:
        LNs_sel_LR = LNs_sel_LR_d
        title = (r'corr. among ORNs$\rightarrow$LN'+ f'\n'
                 + 'syn. count vectors \{'+ f'{wLNtex}' + '\}')
        xylabel = 'ORNs\n' + r'$\rightarrow$'
        ylabel = r'$\mathbf{w}_\mathrm{LN}$'
        xlabel = r'$\mathbf{w}_\mathrm{LN}$'
        # ylabel = r'$\mathbf{w}_\mathrm{LN}$:' + '\nfrom\nORNs\nto'
        # xlabel = r'$\mathbf{w}_\mathrm{LN}$: from ORNs to'
    else:
        LNs_sel_LR = LNs_sel_LR_a
        title = (r'corr. among LN$\rightarrow$ORNs' + f'\n'
                 + r'syn. count vectors \{$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$\}')
        # xylabel = r'...$\rightarrow$ ORNs'
        xylabel = 'ORNs\n' + r'$\leftarrow$'
        ylabel = r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$'
        xlabel = r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$'
        # ylabel = 'to\nORNs\nfrom'
        # xlabel = 'to ORNs from'

    con_ff_sel = con_strms3.loc[:, strm]
    con_ff_sel = con_ff_sel.loc[:, LNs_sel_LR]
    con_ff_sel.columns = LNs_sel_LR_short
    # if strm == 0:
    #     labels = [r'ORNs $\rightarrow$ '+ LN for LN in LNs_sel_LR_short]
    #     con_ff_sel.columns = labels
    # else:
    #     con_ff_sel.columns = LNs_sel_LR_short
    con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
    grammian = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)

    # pads = (0.5, 0.45, 0.5, 0.2)  # l, r, b, t
    pads = (0.6, 0.35, 0.52, 0.35)  # l, r, b, t
    _, fs, axs = FP.calc_fs_ax_df(grammian, pads, sq=SQ)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs)
    cp = FP.imshow_df(grammian, ax1, cmap=corr_cmap, vlim=1,
                      splits_x=[6, 10, 14], splits_y=[6, 10, 14], ha='right',
                      show_lab_x=True, title=title, x_offset=0.5, rot=70)
    # ax1.set_xticks(np.arange(len(grammian.T)) + 0.5)  # the +0.5 is needed
    # because of the rotation
    # ax1.set_xticklabels(list(grammian.columns), rotation=70, )
    ax1.set(xlabel=xlabel, ylabel=ylabel)

    # ax1.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
    #                rotation_mode='default', ha='center', va='top')
    # ax1.set_ylabel(ylabel, rotation=0, fontsize=ft_s_tk, labelpad=3,
    #                va='center', ha='right')
    # ax1.annotate('', xy=(-2, 20.3), xytext=(17, 20.3), xycoords='data',
    #              arrowprops={'arrowstyle': '-', 'lw':0.5},
    #              annotation_clip=False)
    # ax1.annotate('', xy=(-5.3, -1), xytext=(-5.3, 16), xycoords='data',
    #              arrowprops={'arrowstyle': '-', 'lw':0.5},
    #              annotation_clip=False)

    # ax1.xaxis.set_label_coords(0.5, -0.6)
    # ax1.yaxis.set_label_coords(-0.6, 0.5)

    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0],
                        axs[3]])

    add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])
    file = (f'{PP_CONN}/{CELL_TYPE}_con{strm}_cn_grammian')
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)



    gram_cat = pd.DataFrame(index=cat.keys(), columns=cat.keys(), dtype=float)
    for c1, c2 in itertools.product(cat.keys(), cat.keys()):
        mat = grammian.loc[cat[c1], cat[c2]].values
        if c1 == c2:
            n = len(cat[c1])
            triu = np.triu_indices(n, 1)
            gram_cat.loc[c1, c2] = np.mean(FG.rectify(mat[triu]))
        else:
            gram_cat.loc[c1, c2] = np.mean(FG.rectify(mat))


    # plotting
    # pads = (0.5, 0.3, 0.5, 0.5)  # l, r, b, t
    pads = (0.25, 0.25, 0.45, 0.5)  # l, r, b, t
    _, fs, axs = FP.calc_fs_ax_df(gram_cat, pads, sq=SQ*1.5)
    f = plt.figure(figsize=fs)
    ax1 = f.add_axes(axs)

    FP.imshow_df(gram_cat, ax1, cmap=corr_cmap, vlim=1,
                 splits_x=[1, 2, 3], splits_y=[1, 2, 3], rot=50,
                 title = r'mean corr. coef. $r$' + '\nwithin and '
                                                    'across\n LN types'
                 )
    ax1.set(xlabel='', ylabel='')

    file = (f'{PP_CONN}/{CELL_TYPE}_con{strm}_cn_grammian_cat')
    FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
    FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# %%
# plot correlation of feedforward vs feedforward so that we have that in the
# paper and have it as argument for not showing the analysis for feedback.

strm = 0
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel_LR_d]
con_ff_sel.columns = LNs_sel_LR_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
strm = 1
con_fb_sel = con_strms3.loc[:, strm]
con_fb_sel = con_fb_sel.loc[:, LNs_sel_LR_a]
con_fb_sel.columns = LNs_sel_LR_short
con_fb_sel_cn = FG.get_ctr_norm(con_fb_sel)
grammian = FG.get_corr(con_ff_sel_cn, con_fb_sel_cn)

# pads = (0.8, 0.45, 0.8, 0.2)  # l, r, b, t
# pads = (0.6, 0.35, 0.45, 0.2)  # l, r, b, t
pads = (0.6, 0.35, 0.52, 0.35)  # l, r, b, t
_, fs, axs = FP.calc_fs_ax_df(grammian, pads, sq=SQ)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs)
cp = FP.imshow_df(grammian, ax1, cmap=corr_cmap, vlim=1, x_offset=0.5,
                  splits_x=[6, 10, 14], splits_y=[6, 10, 14],
                  title=f'corr. between\n'+ r'$\{\mathbf{w}_\mathrm{LN}^\mathrm{ff}\}$ and '
                         + r'$\{\mathbf{w}_\mathrm{LN}^\mathrm{fb}\}$', rot=70, ha='right')

ax1.set_xlabel(r'$\mathbf{w}_\mathrm{LN}^\mathrm{fb}$')
ax1.set_ylabel(r'$\mathbf{w}_\mathrm{LN}^\mathrm{ff}$')
# ax1.set_xlabel('to ORNs from', rotation=0, fontsize=ft_s_tk, labelpad=2)
# ax1.set_ylabel('from\nORNs\nto', rotation=0, fontsize=ft_s_tk,
#                labelpad=3, va='center', ha='right')
# ax1.xaxis.set_label_coords(0.5, -0.6)
# ax1.yaxis.set_label_coords(-0.6, 0.5)
# ax1.annotate('', xy=(-2, 20.3), xytext=(17, 20.3), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax1.annotate('', xy=(-5.3, -1), xytext=(-5.3, 16), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX/fs[0], axs[1], CB_W/fs[0], axs[3]])
add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])
file = f'{PP_CONN}/{CELL_TYPE}_con0vs1_cn_grammian'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')

# %%
# left and right side separately, but on same figure
d_h = 0.15
strm = 0
# xlabel = 'from ORNs to'
xlabel = r'ORNs$\rightarrow$LN synaptic count vector $\mathbf{w}_\mathrm{LN}$'
# ylabel = 'from\nORNs\nto'
ylabel = r'$\mathbf{w}_\mathrm{LN}$'
pads = [0.40, 0.45, 0.37, 0.35]  # l, r, b, t
title = {'L': 'left', 'R': 'right'}
side = 'L'
# Correlation coefficient per category
LNs_sel1 = LNs_sel_d_side[side]

con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df1 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
side = 'R'
# Correlation coefficient per category
LNs_sel1 = LNs_sel_d_side[side]

con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df2 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
d_x = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts_side(df1, df2, pads, d_h, SQ,
                                                  CB_DX, CB_W)

f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df(df1, ax1, cmap=corr_cmap, vlim=1, x_offset=0.5,
                  rot=70, ha='right')
ax1.set_title('left side', pad=2)
ax1.set_ylabel(ylabel)
# ax1.set_ylabel(ylabel, rotation=0, fontsize=ft_s_tk, labelpad=3,
#                va='center', ha='right')
# ax1.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
#                rotation_mode='default', ha='center', va='top')
# ax1.annotate('', xy=(-4.1, -1), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

cp = FP.imshow_df(df2, ax2, cmap=corr_cmap, vlim=1, x_offset=0.5,
                  show_lab_y=False, rot=70, ha='right')
ax2.set_title('right side', pad=2)

# ax2.set_xlabel(xlabel, rotation=0, fontsize=ft_s_tk, labelpad=2,
#                rotation_mode='default', ha='center', va='top')
# ax2.annotate('', xy=(-12, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
f.text(0.5, 0., xlabel, rotation=0, fontsize=ft_s_lb, va='bottom',
       ha='center')
# ax1.xaxis.set_label_coords(0.5, -1.2)
# ax1.yaxis.set_label_coords(-1.2, 0.5)

add_colorbar_crt(cp, ax_cb, r'$r$', [-1, 0, 1])
plt.suptitle(r'connectome, corr. among $\{\mathbf{w}_\mathrm{LN}\}$')
file = f'{PP_CONN}/{CELL_TYPE}_con{strm}LR_cn_grammian'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')

# %%
# #############################################################################
# ######################  COMPARISON M WITH WWT  ##############################
# #############################################################################

# ###########################          M         ##############################
# this is pretty much the same as the M plot above, just with Picky merged
# into 1 cell and changed padding at the bottom and at the top

s = 'L'
df1 = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
df2 = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
df2.columns.name = 'Postsynaptic LN'


pads = [0.4, 0.4, 0.41, 0.45]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df(df1, ax1, vlim=[0, 110], show_lab_x=False)
ax1.set_title('left side', pad=2)#, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df(df2, ax2, vlim=[0, 110], x_offset=0.5, rot=70, ha='right')
ax2.set_title('right side', pad=2)#, fontsize=ft_s_lb)

# y label
f.text(0.01, 0.55, 'Presynaptic LN', rotation=90,
       fontsize=ft_s_lb, va='center', ha='left')

clb = add_colorbar_crt(cp, ax_cb, '\# syn.', [0, 50, 100])
# clb.ax.set_title(, pad=2, fontsize=ft_s_tk)
plt.suptitle("LN-LN connections\n synaptic counts " + r"$\mathbf{M}$")
file = f'{PP_CONN}/con_M_a-d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# ###########################          WTW       ##############################
# this is pretty much the same as the W plot above, just with Picky dend and
# axon merged into 1 cell and changed padding at the bottom and at the top

xlabel = r'$\mathbf{w}_\mathrm{LN}$'
ylabel = r'ORNs$\rightarrow$LN synaptic count vector $\mathbf{w}_\mathrm{LN}$'

s = 'L'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df1 = con_ff_sel.T.dot(con_ff_sel)
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df2 = con_ff_sel.T.dot(con_ff_sel)
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
# df2.columns.name = r'ORNs$\rightarrow$...'

# df1 = df1 - np.diag(np.diag(df1))
# df2 = df2 - np.diag(np.diag(df2))

print(np.max(df1.values), np.max(df2.values))

pads = [0.5, 0.5, 0.41, 0.45]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

FP.imshow_df(df1 / 1000, ax1, vlim=[0, 15], show_lab_x=False)
ax1.set_title('left side', pad=2)#, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df(df2 / 1000, ax2, vlim=[0, 15], x_offset=0.5, rot=70, ha='right')
ax2.set_title('right side', pad=2)#, fontsize=ft_s_lb)
ax2.set_xlabel(xlabel)
# ax2.annotate('', xy=(-2, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax2.annotate('', xy=(-4.1, -11), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

f.text(0.1, 0.5, ylabel, rotation=90, fontsize=ft_s_lb,
       va='center',ha='right')
# f.text(0.15, 0.55, 'from\nORNs\nto', rotation=0, fontsize=ft_s_tk, va='center',
#        ha='right')

clb = add_colorbar_crt(cp, ax_cb, '', [0, 10])
clb.set_ticklabels([0, r'$10^4$'])
clb.ax.set_title(r'(\# syn.)$^2$', pad=2, fontsize=ft_s_tk)
# clb.ax.set_title('1e3', pad=2, fontsize=ft_s_tk)
plt.suptitle(r'ORNs$\rightarrow$LN dot products' + '\n'
                                                   r'$\mathbf{W}^\mathrm{\top}\mathbf{W} = $'+
             r'$\{\mathbf{w}_\mathrm{LNi}^\mathrm{\top}\mathbf{w}_\mathrm{LNj}\}$')

file = f'{PP_CONN}/con_WtW_d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)
#
print('done')
#
# %%
# #########################        sqrt(WTW)       ############################
# this is pretty much the same as the W plot above, just with Picky dend and
# axon merged into 1 cell and changed padding at the bottom and at the top

xylabel = r'$\mathbf{w}_\mathrm{LN}$'

s = 'L'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df1 = con_ff_sel.T.dot(con_ff_sel)
df1[:] = LA.sqrtm(df1.values)
# df1.index = LNs_sel_ad2
df1.index = LNs_sel_short
# df1.index.name = 'Presynaptic'
s = 'R'
con_ff_sel = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
df2 = con_ff_sel.T.dot(con_ff_sel)
df2[:] = LA.sqrtm(df2.values)
# df2.index = LNs_sel_ad2
df2.index = LNs_sel_short
# df2.index.name = 'Presynaptic'
# df2.columns = LNs_sel_ad2
df2.columns = LNs_sel_short
# df2.columns.name = r'ORNs$\rightarrow$...'

# df1 = df1 - np.diag(np.diag(df1))
# df2 = df2 - np.diag(np.diag(df2))

print(np.max(df1.values), np.max(df2.values))

# pads = [0.5, 0.5, 0.41, 0.45]
pads = [0.5, 0.5, 0.41, 0.32]
d_h = 0.15  # delta height between the 2 imshows
fs, axs1, axs2, axs_cb = FP.calc_fs_ax_2plts(df1, df2, pads, d_h, SQ, CB_DX,
                                             CB_W)
f = plt.figure(figsize=fs)
ax1 = f.add_axes(axs1)
ax2 = f.add_axes(axs2)
ax_cb = f.add_axes(axs_cb)

cp = FP.imshow_df(df1, ax1, vlim=[0, 90], show_lab_x=False)
ax1.set_title('left side', pad=2)#, fontsize=ft_s_lb)

# bottom plot
cp = FP.imshow_df(df2, ax2, vlim=[0, 90], x_offset=0.5, rot=70, ha='right')
ax2.set_title('right side', pad=2) #  , fontsize=ft_s_lb)
ax2.set_xlabel(xylabel)
# ax2.annotate('', xy=(-2, 8 + 3.1), xytext=(9, 8 + 3.1), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)
# ax2.annotate('', xy=(-4.1, -11), xytext=(-4.1, 8), xycoords='data',
#              arrowprops={'arrowstyle': '-', 'lw': 0.5},
#              annotation_clip=False)

f.text(0.1, 0.5, xylabel, rotation=90, fontsize=ft_s_lb, va='center',
       ha='right')

clb = add_colorbar_crt(cp, ax_cb, '', [0, 40, 80])
clb.ax.set_title('\# syn.', pad=2, fontsize=ft_s_tk)
plt.suptitle(r'$(\mathbf{W}^\mathrm{\top} \mathbf{W})^{1/2}$')

file = f'{PP_CONN}/con_sqrtWtW_d.'
FP.save_plot(f, f'{file}png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, f'{file}pdf', SAVE_PLOTS, **pdf_opts)

print('done')
# %%
# #####################  SCATTER  PLOTS AND    ################################
# #####################     SIGNIFICANCE       ################################

# scatter plot between sqrt(WTW) and M

# importlib.reload(FP)

power = 1

s = 'L'
ML = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()**power
WL = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
WTW_L: np.ndarray = LA.sqrtm(WL.T @ WL)
s = 'R'
MR = con_S[s].loc[LNs_sel_a_side[s], LNs_sel_d_side[s]].copy()**power
WR = con_S[s].loc[ORNs_side[s], LNs_sel_d_side[s]]
WTW_R: np.ndarray = LA.sqrtm(WR.T @ WR)

diag = False
W_entries = np.concatenate([FG.get_entries(WTW_L, diag=diag),
                            FG.get_entries(WTW_R, diag=diag)])

M_entries = np.concatenate([FG.get_entries(ML, diag=diag),
                            FG.get_entries(MR, diag=diag)])
n_pts = int(len(M_entries)/2)

cc_real = np.corrcoef(W_entries, M_entries)[0, 1]
print(cc_real)  # 0.73

# the only thing that is important that it matches is the bottom in order
# to compare with the M and WTW plots
b = 0.35
pads = (0.4, 0.1, b, 0.15)
# then, what about the actual size of the graph?

fs, ax1 = FP.calc_fs_ax(pads, 15*SQ, 15*SQ)
f = plt.figure(figsize=fs)
ax = f.add_axes(ax1)
ax.scatter(M_entries[:n_pts], W_entries[:n_pts], c='indigo', s=5,
           label='left', alpha=0.7, lw=0)
FP.plot_scatter(ax, M_entries[n_pts:], W_entries[n_pts:],
                r'$\mathbf{M}$ entry (\# syn.)',
                r'$(\mathbf{W}^\mathrm{\top}'
                r' \mathbf{W})^{1/2}$ entry (\# syn.)',
                pca_line_scale1=0.18,
                pca_line_scale2=0.68, show_cc=False, s=5, c='teal',
                label='right', alpha=0.7, lw=0)
ax.legend(loc='upper left')
ax.text(0.65, 0.11, r'$r$' + " = %0.2f" % cc_real, transform=ax.transAxes)
ax.text(0.65, 0.03, "pv = %0.0e" % 0.006, transform=ax.transAxes)
ax.set(ylim=(None , None), xlim=(-5, None))

file = f'{PP_CONN}/con_sqrtWtW_M_scatter'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)
print('done')
# significance is calculated in the file con_analysis_MvsW.py
# %%
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #########################  W clustering in NNC  #############################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################

Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
# Ws_cn = FG.get_ctr_norm(Ws).loc[par_act.ORN_order].T


# %%
# Plotting the previous graphs on a single figure
k = 8
lab_y = {-1: True, -0.45: False, 0: False, 1: False}
pads = [0.3, 0.075, 0.37, 0.35]
title = {-1: '0.1', -0.45: '0.35', 0: '1', 1: '10'}
n_ax = len(title)
fs, axs = FP.calc_fs_ax(pads, gw=SQ * k * n_ax + (n_ax-1) * 2 * SQ, gh=SQ * k)
f = plt.figure(figsize=fs)
axs_coords = {-1: [axs[0], axs[1], SQ * k / fs[0], axs[3]],
              -0.45:[axs[0] + SQ * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
              0: [axs[0] + SQ * 2 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]],
              1: [axs[0] + SQ * 3 * (k + 2)/fs[0], axs[1], SQ * k / fs[0], axs[3]]}
for s in lab_y.keys():
    rho = 10**s
    pps_local = f'{title[s]}o'
    print('rho:', rho)
    W_nncT = Ws_nnc.loc[:, (s*10, 1)].values.T

    links = sch.linkage(np.corrcoef(W_nncT), method='average', optimal_ordering=True)
    new_order = sch.leaves_list(links)
    df = pd.DataFrame(np.corrcoef(W_nncT[new_order]))
    #
    # CG = sns.clustermap(np.corrcoef(W_nncT), cmap=corr_cmap, vmin=-1, vmax=1)
    # idx = CG.dendrogram_col.reordered_ind
    # df = pd.DataFrame(np.corrcoef(W_nncT[idx]))


    ax = f.add_axes(axs_coords[s])
    cp = FP.imshow_df(df, ax, vlim=[-1, 1], show_lab_y=lab_y[s],
                      show_lab_x=True, cmap=corr_cmap, rot=0)
    ax.set_title(r'$\rho$ = ' + f'{title[s]}', pad=2)
    ax.set_xticks(np.arange(8), np.arange(1, 9))
    # ax.set_xticklabels()
    # ax.set_xlabel(r'NNC-8, $\mathbf{w}_k$', labelpad=1, rotation_mode='default',
    #               ha='center')
    if lab_y[s]:
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels(np.arange(1, 9))
        ax.set_ylabel(r'$\mathbf{w}_k$', labelpad=6, va='center')
    # ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
    #                     CB_W / fs[0], axs[3]])
    # clb = add_colorbar_crt(cp, ax_cb, '', [-1, 0, 1])
    print(W_nncT.sum(axis=1))
    # print(W_nncT[idx].sum(axis=1))
plt.suptitle(r'NNC-8 model, corr. among $\{\mathbf{w}_k\}$')
# f.text(0.54, 0.1, r'$\mathbf{w}_k$', rotation=0, fontsize=ft_s_lb, va='bottom',
#        ha='center')
f.text(0.54, 0.1, r'ORNs$\rightarrow$LN conn. weight vector $\mathbf{w}_k$',
       rotation=0, fontsize=ft_s_lb, va='bottom',
       ha='center')
file = f'{PP_CON_PRED}/ORN_act_NNC{k}_corrW_all'
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **pdf_opts)

print('done')

# %%
# #############################################################################
# #################### LOOKING AT A WHOLE SET OF W ############################
# #################### AND COMPARING THE CLUSTERING TO THE DATA ###############
# #############################################################################
strm = 0
side = 'L'
LNs_sel1 = LNs_sel_d_side[side]
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df1 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)

side = 'R'
LNs_sel1 = LNs_sel_d_side[side]
con_ff_sel = con_strms3.loc[:, strm]
con_ff_sel = con_ff_sel.loc[:, LNs_sel1]
con_ff_sel.columns = LNs_sel_short
con_ff_sel_cn = FG.get_ctr_norm(con_ff_sel)
df2 = FG.get_corr(con_ff_sel_cn, con_ff_sel_cn)
print('done')
# %%
th = 0.45
df1_ent = FG.get_entries(df1, diag=False)
corr_L = FG.rectify(df1_ent).mean()
df2_ent = FG.get_entries(df2, diag=False)
corr_R = FG.rectify(df2_ent).mean()
print(corr_L, corr_R)
print('done')
# %%
Ws_nnc = pd.DataFrame(pd.read_hdf(RESULTS_PATH / 'W_NNC-8.hdf'))
Ws_nnc_cn = FG.get_ctr_norm(Ws_nnc).loc[par_act.ORN_order].T
mi2 = pd.MultiIndex(levels=[[], []], codes=[[], []],
                    names=['rho', 'rep'])
corr_W_nnc_s = pd.Series(index=mi2)
clust_W_nnc_s = pd.Series(index=mi2)

for p in np.arange(-10, 10.1, 0.5):
    # rho = 10**(p/10)
    for i in range(50):
        W_nnc = Ws_nnc_cn.loc[(p, i)]
        corr = W_nnc @ W_nnc.T
        df1_ent = FG.get_entries(corr, diag=False)
        corr_W_nnc_s.loc[(p, i)] = FG.rectify(df1_ent).mean()
print('done')
# %%
rho_special = 0.35
x_special = np.log10(rho_special)
x = corr_W_nnc_s.index.unique('rho')/10
y = corr_W_nnc_s.groupby('rho').mean()
e = corr_W_nnc_s.groupby('rho').std()

pads = (0.4, 0.1, 0.35, 0.1)
# fs, axs = FP.calc_fs_ax(pads, SQ*18, SQ*10)
fs, axs = FP.calc_fs_ax(pads, SQ*12, SQ*12)  # pads, gw, gh
f = plt.figure(figsize=fs)
ax = f.add_axes(axs)
ax.plot(x, y, lw=1, c='k')
ax.fill_between(x, y-e, y+e, alpha=0.5, facecolor='k', label='NNC-8')
ax.fill_between([min(x), max(x)], [corr_L, corr_L], [corr_R, corr_R],
                alpha=0.5, label='data')
ax.plot([x_special, x_special], [0, 0.4], lw=0.5, color='gray', ls='--')
ax.set_yticks([0, 0.2, 0.4])
# ax.set_yticklabels([0, '', 2, '', 4])
ax.set_xticks([-1, x_special, 0, 1], [0.1, rho_special, 1, 10])
ax.set_ylim(0, 0.4)
ax.set_ylabel(r'$\overline{r}_+$')
ax.set_xlabel(r'$\rho$')
plt.legend(loc='upper right')
file = (f'{PP_CON_PRED}/{CELL_TYPE}_con{STRM}_vs_act'
        f'-{act_pps1}-{act_pps2}-{ACT_PPS}_NNC-8_W-corr')
FP.save_plot(f, file + '.png', SAVE_PLOTS, **png_opts)
FP.save_plot(f, file + '.pdf', SAVE_PLOTS, **png_opts)
print('done')
# %%
# finding where the data and the model lines intersect
# the value of x around where the lines intersect
corr_M = (corr_L + corr_R) / 2
print(x[np.sum(y > corr_M)])
print('Final done')
