# -*- coding: utf-8 -*-

"""
@author: Nikolai M Chapochnikov
"""

# #############################################################################
# ################################# IMPORTS ###################################
# #############################################################################
import numpy as np
import pandas as pd
import matplotlib
# import itertools
import scipy.linalg as LA
import ast
import sklearn.decomposition as skd
import statsmodels.stats.multitest as smsm  # for the multihypothesis testing
import matplotlib.pyplot as plt
import functions.general as FG
from typing import Tuple

# #############################################################################
# ###########################  GENERAL FUNCTIONS ##############################
# #############################################################################

def set_plotting(plot_plots: bool):
    if plot_plots:
        plt.ion()
    else:
        # this weird construction is needed otherwise ioff gets hijacked
        # with
        # plt.ioff()
        f = plt.figure()
        plt.close(f)
        plt.ioff()
    return 0


def save_plot(f: plt.Figure, path: str, cond: bool, **kwargs):
    """
    saves the figure only if cond is True, otherwise doesn't do anything
    can take any options that is then transmitted to savefig
    This is basically a simple wrapper for savefig
    """
    if cond:
        f.savefig(path, **kwargs)
        plt.close(f)


# function for setting back the default style
# can be useful after messing up with different styles.
def set_default_plot_params():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # to get the available styles: plt.style.available
    # to apply one of the available styles:
    # plt.style.use('seaborn-paper')


def unpack_vlim(vlim) -> Tuple[float, float]:
    """
    unpack vlim
    Parameters
    ----------
    vlim

    Returns
    -------

    """
    if type(vlim) is not list and vlim is not None:
        (vmin, vmax) = (-vlim, vlim)
    elif vlim is None:
        (vmin, vmax) = (vlim, vlim)
    elif len(vlim) == 2:
        (vmin, vmax) = vlim
    else:
        raise ValueError(f'there is a problem with vlim: {vlim}')
    return vmin, vmax


def set_aspect_ratio(ax, alpha):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0)/alpha)

# could be replaced by vlines
def add_x_splits(ax, splits, n, **plot_args):
    for s in splits:
        ax.plot([s-0.5, s-0.5], [-0.5, n-0.5], **plot_args)


def add_y_splits(ax, splits, n, **plot_args):
    for s in splits:
        ax.plot([-0.5, n-0.5], [s-0.5, s-0.5], **plot_args)

# not used in paper
# def imshow_df(df: pd.DataFrame, ax=None, cb=True, title='', vlim=None,
#               cmap=plt.cm.viridis, figsize=None, tight=False,
#               splits_x=[], splits_y=[], rot=90, ha='center', lw=1,
#               show_lab_x: bool = True, show_lab_y: bool = True,
#               cb_frac: float = 0.043,
#               cbtitle: str = '', show_values: bool = False):
#     """
#     very general wrapper around imshow to plot dataframes with labels
#     splits_x and split_y adds lines to separate different categories in the
#     data
#     """
#     if ax is None:
#         f, ax = plt.subplots(1, 1, figsize=figsize)
#
#     (vmin, vmax) = unpack_vlim(vlim)
#     print(f'vmin and vmax in imshow_df: {vmin}, {vmax}')
#     cmap.set_bad('black')
#     cp = ax.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
#     ax.set_title(title)
#
#     plt.sca(ax)
#     if show_lab_x is True:
#         plt.xticks(np.arange(len(df.T)), list(df.columns), rotation=rot, ha=ha)
#         ax.tick_params('x', bottom=False, pad=0)
#
#         label = ''
#         # for a multiindex creates a label from the mi names
#         if df.columns.names[0] is not None:
#             for i in range(len(df.columns.names)):
#                 label = label + ', ' + df.columns.names[i]
#             label = label[2:]
#         ax.set_xlabel(label)
#     else:
#         plt.xticks([], [])
#
#     if show_lab_y is True:
#         plt.yticks(np.arange(len(df)), list(df.index))
#         ax.tick_params('y', left=False, pad=0)
#
#         label = ''
#         if df.index.names[0] is not None:
#             for i in range(len(df.index.names)):
#                 label = label + ', ' + df.index.names[i]
#             label = label[2:]
#         ax.set_ylabel(label)
#     else:
#         plt.yticks([], [])
#
#     add_x_splits(ax, splits_x, len(df.index), c='w', lw=lw)
#     add_y_splits(ax, splits_y, len(df.columns), c='w', lw=lw)
#
#     if cb is True:
#         clb = plt.colorbar(cp, ax=ax, fraction=cb_frac, pad=0.04)
#         clb.outline.set_linewidth(0.00)
#         clb.ax.tick_params(size=2, direction='in', pad=1.5)
#         clb.ax.set_title(cbtitle)
#     else:
#         clb = None
#
#     if tight is True:
#         plt.tight_layout()
#
#     # removing the borders
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
#     # printing the actual values inside the squares.
#     if show_values:
#         for (j, i), label in np.ndenumerate(df):
#             ax.text(i, j, int(label), ha='center', va='center',
#                     size=matplotlib.rcParams['font.size']*0.8**2)
#
#     return (plt.gcf(), ax, clb)


def imshow_df2(df, ax, title='', vlim=None, cmap=plt.cm.viridis,
               splits_x=[], splits_y=[], rot=90, ha='center', lw=1,
               show_lab_x=True, show_lab_y=True, show_values=False,
               aspect='equal', splits_c='w', title_font=None, x_offset=0,
               **kwargs):
    """
    very general wrapper around imshow to plot dataframes with labels
    splits_x and split_y adds lines to separate different categories in the
    data
    """

    (vmin, vmax) = unpack_vlim(vlim)
    print(f'vmin and vmax in imshow_df: {vmin}, {vmax}')
    #    cmap.set_bad('black')
    cp = ax.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
                   **kwargs)
    ax.set_title(title, fontdict=title_font)

    ax.tick_params('x', bottom=False, direction='in')
    ax.tick_params('y', left=False, direction='in')
    if show_lab_x:
        ax.set_xticks(np.arange(len(df.T)) + x_offset,
                      list(df.columns), rotation=rot, ha=ha)
        label = ''
        # for a multiindex creates a label from the mi names
        if df.columns.names[0] is not None:
            for i in range(len(df.columns.names)):
                label = label + ', ' + df.columns.names[i]
            label = label[2:]
        ax.set_xlabel(label)
    else:
        ax.set_xticks([], [])

    if show_lab_y:
        ax.set_yticks(np.arange(len(df)), list(df.index))

        label = ''
        if df.index.names[0] is not None:
            for i in range(len(df.index.names)):
                label = label + ', ' + df.index.names[i]
            label = label[2:]
        ax.set_ylabel(label)
    else:
        ax.set_yticks([], [])

    add_x_splits(ax, splits_x, df.shape[0], c=splits_c, lw=lw)
    add_y_splits(ax, splits_y, df.shape[1], c=splits_c, lw=lw)

    # removing the borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # printing the actual values inside the squares.
    if show_values:
        for (j, i), label in np.ndenumerate(df):
            ax.text(i, j, int(label), ha='center', va='center',
                    size=matplotlib.rcParams['font.size']*0.8**2)

    return cp


def add_colorbar(cp, ax, cbtitle='', ticks=None, pad_title=1, extend='neither',
                 title_font=None):
    """
    add a colorbar at the location specified by ax and for the plot referenced
    by cp. The pad is for the title position of the colorbar
    """
    clb = plt.colorbar(cp, cax=ax, extend=extend)
    clb.ax.set_yscale('linear')  # new option needed
    clb.outline.set_linewidth(0.00)
    # clb.ax.tick_params(size=2, direction='in', pad=-1)
    # clb.ax.tick_params(size=2, direction='in', pad=1.5)
    clb.ax.tick_params(direction='in')
    clb.ax.set_title(cbtitle, pad=pad_title, fontdict=title_font)
    if ticks is not None:
        clb.set_ticks(ticks)
    return clb


def calc_fs_ax_df(df, pads, gw=None, gh=None, sq=None):
    """
    calculates the size of the figure based on the requirement on the padding
    and of the size of the data
    """
    l, r, b, t = pads
    h_n, w_n = df.shape
    if gw is not None:
        sq = gw/w_n
        gh = sq*h_n
    elif gh is not None:
        sq = gh/h_n
        gw = sq*w_n
    elif sq is not None:
        gh = sq*h_n
        gw = sq*w_n
    else:
        raise ValueError(f'gw or gh or sq needed, got {gw}, {gh}, {sq}')
    fs = (l + r + gw, b + t + gh)
    axes = [l/fs[0], b/fs[1], 1 - (l+r)/fs[0], 1 - (b+t)/fs[1]]
    return sq, fs, axes


def calc_fs_ax(pads, gw, gh):
    """
    calculates the size of the figure based on the requirement on the padding
    and of the size of graph we want
    gw is the graph width
    gh is the graph height
    """
    l, r, b, t = pads
    fs = (l + r + gw, b + t + gh)
    axes = [l/fs[0], b/fs[1], 1 - (l+r)/fs[0], 1 - (b+t)/fs[1]]
    return fs, axes


def calc_fs_ax_2plts(df1, df2, pads, d_h, sq, cb_dx, cb_w):
    """
    if one want to plot 2 plots on top of each other
    pads are for the both plots together
    d_h is the distance between the 2 plots in inches
    sq is the size of the square in inches
    cb_dx is the distance from the plot to the colorbar
    cb_w is the width of the colorbar
    """
    pads1 = pads.copy() # for the top plot, changing the bottom padding
    pads1[2] = pads[2] + d_h + sq*len(df2.index)

    pads2 = pads.copy()  # for the bottom plot, changing the top padding
    pads2[3] = pads[3] + d_h + sq*len(df1.index)

    _, fs, axs1 = calc_fs_ax_df(df1, pads1, sq=sq)
    _, _, axs2 = calc_fs_ax_df(df2, pads2, sq=sq)
    # ax : [x, y, dx, dy]

    cb_x = axs1[0] + axs1[2] + cb_dx/fs[0] # x position of colorbar
    # corresponds to x + dx + bd_dx
    axs_cb = [cb_x, axs2[1], cb_w/fs[0], axs1[3] + axs2[3] + d_h/fs[1]]
    # [x_position, y_position,
    return fs, axs1, axs2, axs_cb


def calc_fs_ax_2plts_side(df1, df2, pads, d_x, sq, cb_dx, cb_w):
    """
    if one want to plot 2 plots side to side of each other
    pads are for the both plots together
    d_x is the distance between the 2 plots in inches
    sq is the size of the square in inches
    cb_dx is the distance from the plot to the colorbar
    cb_w is the width of the colorbar
    """
    pads1 = pads.copy()  # for the left plot
    pads1[1] = pads[1] + d_x + sq*len(df2.columns)

    pads2 = pads.copy()  # for the right plot
    pads2[0] = pads[0] + d_x + sq*len(df1.columns)

    _, fs, axs1 = calc_fs_ax_df(df1, pads1, sq=sq)
    _, _, axs2 = calc_fs_ax_df(df2, pads2, sq=sq)

    cb_x = axs2[0] + axs2[2] + cb_dx/fs[0]
    axs_cb = [cb_x, axs2[1], cb_w/fs[0], axs2[3]]

    return fs, axs1, axs2, axs_cb


def add_pca_line(x, y, scale1, scale2, ax):
    """
    this function adds (usually to a scatter plot) a line which is the
    principal direction of the dataset in these 2 dimensions
    the scale is a parameter setting how extended shold be the line from
    the center of the cloud
    """
    x1 = x - x.mean()
    x_l2 = LA.norm(x1)
    x2 = x1/x_l2

    y1 = y - y.mean()
    y_l2 = LA.norm(y1)
    y2 = y1/y_l2

    X = np.array([x2, y2]).T
    pca = skd.PCA(n_components=1)
    pca.fit(X)
    m = [x.mean(), y.mean()]
    v = pca.components_
    v = v * np.array([x_l2, y_l2])
    ax.plot([m[0] - scale1*v[0, 0], m[0] + scale2*v[0, 0]],
            [m[1] - scale1*v[0, 1], m[1] + scale2*v[0, 1]], '--', c='gray')


# ####  FUNCTION USED IN GRANT AND PRESENTATION PLOTS  ########################
# one idea would be that these 2 functions would be part of plotting class
# that has as parameters different font sizes and other adjustments
# so that they are fixed onces for all then can be references from
# within the functino


def plot_scatter(ax, data1, data2, lblx, lbly, c1='k', c2='k',
                 xticks=None, yticks=None, pvalue=None,
                 pca_line_scale1=0.8, pca_line_scale2=0.8, show_cc=True,
                 **kwargs):
    """
    this function is used to plot article quality figure of a scatter plot
    we need to provide 2 datasets, the labels, the colors, and the ticks
    on the y axis, which was historically used for the vector describing
    activity
    """
    # to make things more modulable, the adjustments should be done outside
    # of the functino
    # adj_l = 0.15
    # adj_r = 0.85
    # adj_b = 0.2
    # adj_t = 0.9

    corr_coef = np.corrcoef(data1, data2)[0, 1]
    ax.scatter(data1, data2, **kwargs)
    # here we are adding a line showing the pca directino of the data-set
    add_pca_line(data1, data2, pca_line_scale1, pca_line_scale2, ax)
    # ax.set_xlim(0, None)  # in the csae of PCA it can go below 0
    # ax.set_ylim(0, None)
    # set_aspect_ratio(ax, 1)
    ax.set_xlabel(lblx, color=c1)  # , fontsize=ft_s_lb)
    ax.set_ylabel(lbly, color=c2)  # , fontsize=ft_s_lb)

    # This is to allow the authomatic ticking if you don't put any parameters
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)

    ax.tick_params('x', colors=c1)
    ax.tick_params('y', colors=c2)

    ax.spines['bottom'].set_color(c1)
    ax.spines['left'].set_color(c2)

    # plt.subplots_adjust(left=adj_l, right=adj_r, bottom=adj_b, top=adj_t)
    if show_cc:
        ax.text(0.65, 0.11, r'$r$' + " = %0.2f" % corr_coef, transform=ax.transAxes)
    if pvalue is not None:
        ax.text(0.65, 0.03, "pv = %0.0e" % pvalue, transform=ax.transAxes)

# =============================================================================
#     regr = skllin.LinearRegression()
#     regr.fit(data1.reshape(-1, 1), data2)
#     print(regr.coef_)
#     d1min = 0
#     d1max = np.max(data1)*2
#     ax.plot([d1min, d1max], regr.predict([[d1min], [d1max]]), '--', c='gray')
# =============================================================================
# ax.plot()


def plot_line_2yax(ax, data1, data2, lbl1, lbl2, cell_list, ct,
                   c1='k', c2='k', m1=',', m2=',', label1 = '', label2='',
                   LS='-', title='', rot_y=0):
    """
    plot activity components and a connectivity with a line plot,
    where the 2 sides have a different scaling
    m1 and m2 are markers
    Now the function is tuned to for connectivity and activity, but if it will
    be used wider, it can definitely be generalized
    LS is an option related to the linetype, if we want lines or not between
    points
    Parameters
    ----------
    ct: x label

    """
    # ideally one should scale the 2 datasets a bit better so that
    # mean1 = np.mean(data1)
    # mean2 = np.mean(data2)
    ln1 = ax.plot(data1, c=c1, label=lbl1, ls=LS, marker=m1)

    ax.set_title(title)

    # ax.set_ylim(0, None)  # in the case of PCA it can be below 0
    ax.set_xlabel(ct)
    ax.set_ylabel(label1, color=c1)

    ax.tick_params('y', colors=c1, rotation=rot_y)
    # ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(np.arange(len(cell_list)), cell_list)
    ax.tick_params(axis='x', bottom=False, direction='in', labelrotation=90)

    ax2 = ax.twinx()
    ln2 = ax2.plot(data2, c=c2, label=lbl2, ls=LS, marker=m2)
    ax2.set_ylim(0, None)
    ax2.set_ylabel(label2, color=c2)
    ax2.tick_params('y', colors=c2, rotation=rot_y)

    # adjusting the borders
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set(visible=True, color=c2)
    ax.spines['left'].set_color(c1)

    ax.xaxis.grid()

    # legend, if one puts ax instead of ax, then the legend will not be on the
    # top layer
    lns = ln1 + ln2

    # ax2.legend(lns, labs, loc=4, prop={'size': ft_s_tk})
    # adding a legend if lbl1 and lbl2 are not None
    if (lbl1 is not None) and (lbl2 is not None):
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, ncol=3, loc='lower center',
                         bbox_to_anchor=(0.5, 1.01))
    return ax, ax2, lns

# this is not used as the function below is more flexible
# def plot_double_series(data1, data2, col1, col2, ylab1, ylab2,
#                        ylim1=None, ylim2=None, figsize=(9, 2.5)):
#     """
#     the weird order of ax and ax2 comes from the fact i want the corr points to
#     be above the signif points
#     used for plotting of p-values and mad diff of cdf
#     """
#
#     f, ax = plt.subplots(1, 1, figsize=figsize)
#     ax2 = ax.twinx()
#     ax2.yaxis.set_label_position("left")
#     ax2.yaxis.tick_left()
#     ax2.plot([-1, len(data1)], [0, 0], c='gray', lw=0.5)
#     ax2.plot(data1, ls='None', marker='.', markersize=5)
#     # ax.set_zorder(2)
#     ax2.set_ylabel(ylab1, color=col1)
#     ax2.spines['left'].set_edgecolor(col1)
#     ax2.tick_params('y', colors=col1)
#     ax2.set_xlim(-0.7, len(data1) - 1 + 0.7)
#     ax2.set_ylim(ylim1)
#
#     ax.plot(data2, ls='None', marker="+", c=col2, markersize=5)
#     ax.yaxis.set_label_position("right")
#     ax.yaxis.tick_right()
#     ax.set_ylabel(ylab2, color=col2)
#     ax.spines['right'].set_edgecolor(col2)
#     ax.spines['right'].set_visible(True)
#     ax.tick_params('y', colors=col2)
#     sign = -np.log10(0.05)
#     ax.plot([-0.5, len(data2)-0.5], [sign, sign], c=col2, lw=0.5)
#     ax.xaxis.grid()
#     ax.set_ylim(ylim2)
#
#     ax.set_xticklabels(data1.index, rotation=90)
#
#     plt.tight_layout()
#     return f, (ax2, ax)


def plot_double_series_unevenX(ax, x, data1, data2, col1, col2, ylab1, ylab2,
                               ylim1=None, ylim2=None):
#https://discourse.matplotlib.org/t/zorder-with-twinx-grid/10574
# zorder does not work when you have 2 axis
# that's why we need the weird order of axis
    if not data1.index.equals(data2.index):
        raise ValueError('data1 and data2 should have the same index')

    ax2 = ax.twinx()
    ax2.plot([-1, np.max(x) + 1], [0, 0], c='gray', lw=0.5)
    ax2.plot(x, data1.values, ls='None', marker='.', markersize=5, c=col1)
    # ax.set_zorder(2)

    ax2.spines['left'].set_color(col1)

    ax2.set_ylabel(ylab1, color=col1)
    ax2.tick_params('y', colors=col1)
    ax2.set_ylim(ylim1)
    ax2.set_xlim(-0.7, np.max(x) + 0.7)
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()




    ax.plot(x, data2.values, ls='None', marker="+", c=col2, markersize=5,
             zorder=5)
    ax.set_ylabel(ylab2, color=col2)
    ax.spines['right'].set(visible=True, color=col2)
    ax.tick_params('y', colors=col2)
    sign = -np.log10(0.05)
    ax.plot([-1, np.max(x) + 1], [sign, sign], c=col2, lw=0.5)

    ax.set_ylim(ylim2)

    ax.set_xticks(x)
    ax.set_xticklabels(data1.index, rotation=90)
    ax.tick_params(axis='x', which='both', bottom=False, direction='in')
    ax.xaxis.grid()


    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    return ax2, ax


def plot_pdf_cdf(bins_pdf, pdf_true, pdf_mean, pdf_std,
                 bins_cdf, cdf_true, cdf_mean, cdf_std):
    f, axx = plt.subplots(1, 3, figsize=(15, 4))
    ax = axx[0]
    ax.step(bins_pdf, pdf_mean, where='post', lw=3)
    ax.fill_between(bins_pdf, pdf_mean - pdf_std, pdf_mean + pdf_std,
                    facecolor='grey', step='post')
    ax.step(bins_pdf, pdf_true, where='post')

    ax = axx[1]
    ax.step(bins_cdf, cdf_mean, where='post', lw=3)
    ax.fill_between(bins_cdf, cdf_mean - cdf_std, cdf_mean + cdf_std,
                    facecolor='grey', step='post')
    ax.step(bins_cdf, cdf_true, where='post')

    ax = axx[2]
    ax.step(bins_cdf, cdf_true - cdf_mean, where='post', lw=3, c='C1')
    ax.fill_between(bins_cdf, - cdf_std, + cdf_std,
                    facecolor='grey', step='post')
    # ax.step(bins_cdf, cdf_true, where='post')
    return f, axx


def pvalue2stars(v, init: str = "", sign: str = "*"):
    stars = np.full(len(v), init, dtype=object)
    for i in range(len(v)):
        if v[i]:
            stars[i] = sign
    return stars


# this is actually a plotting function...
def add_sign_stars(ax, pvals, alpha, x, y, sign, fontdict=None):
    """
    computes the significance with a certain alpha and puts the stars
    at the position x and y
    sign is the mark used for the siginifiance indication, for example
    a "*" or "**" or whatever one wants.
    """
    reject, _, _, _ = smsm.multipletests(pvals, method='fdr_bh', alpha=alpha)
    sign_stars = pvalue2stars(reject, sign=sign)
    for i in range(len(x)):
        ax.text(x[i], y, sign_stars[i], horizontalalignment='center',
                fontdict=fontdict)


# #############################################################################
# ##########################  PLOTTING ACTIVITY DATA  #########################
# #############################################################################


# not sure if this is used anywhere
def plot_conVSact_scatter(d1, d2, lab1='activity principal vector, ctr, norm',
                          lab2='connectivity, ctr, norm'):
    """

    Parameters
    ----------
    d1
    d2
    lab1
    lab2

    Returns
    -------

    """
    # Making sure that the 2 datasets are alinged
    cell_srt = d1.index
    d2 = d2.loc[cell_srt]

    f, ax = plt.subplots(1, 1)
    d0 = pd.concat([d1, d1], axis=1)
    corr = d1.values @ d2.values
    corr_str = "{:.2f}".format(corr)

    v_max = np.max(np.abs(d0.values))*1.1
    ax.set_xlim(-v_max, v_max)
    ax.set_ylim(-v_max, v_max)
    ax.plot([-v_max, v_max], [0, 0], c='k')
    ax.plot([0, 0], [-v_max, v_max], c='k')
    ax.scatter(d1, d2)
    for i, txt in enumerate(list(d1.index)):
        ax.annotate(txt, (d1.iloc[i], d2.iloc[i]))
    ax.set_xlabel(lab1)
    ax.set_ylabel(lab2)
    ax.set_aspect('equal')
    ax.grid()

    plt.text(0.05, 0.95, 'corr = ' + corr_str, horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes,
             color='r')
    return None


def _update_annotations(ax, corr_txt, annotations, curve):
    (d1, d2) = curve.get_data()
    corr = d1.dot(d2)
    corr_str = "{:.2f}".format(corr)
    corr_txt.set_text('corr = ' + corr_str)
    for i, cell_label in enumerate(list(d1.index)):
        if annotations[i] != 0:
            annotations[i].remove()
        annotations[i] = ax.annotate(cell_label, (d1.iloc[i], d2.iloc[i]))


def plot_scatter_radio(df1, df2, title='', xlab='', ylab='', vlim=1):
    """

    Parameters
    ----------
    df1
    df2
    title
    xlab
    ylab
    vlim

    Returns
    -------

    """
    # Making sure that the 2 datasets are alinged
    vmin, vmax = unpack_vlim(vlim)
    (df1, df2) = FG.align_indices(df1, df2)

    df1_multi = isinstance(df1.columns, pd.core.index.MultiIndex)
    df2_multi = isinstance(df2.columns, pd.core.index.MultiIndex)

    def update_curve(label, df, df_multi, x):
        if df_multi:
            d_new = df[ast.literal_eval(label)]
        else:
            d_new = df[label]
        if x:
            curve.set_xdata(d_new)
        else:
            curve.set_ydata(d_new)
        plt.draw()
        _update_annotations(ax, corr_txt, annotations, curve)

    names1 = list(df1.columns)
    names2 = list(df2.columns)

    f, ax = plt.subplots(1, 1, figsize=(16, 16))
    plt.subplots_adjust(left=0.25, right=0.75)

    axcol = 'lightgoldenrodyellow'
    rax1 = plt.axes([-0.05, 0.05, 0.5, 0.9],
                    facecolor=axcol, frameon=False)
    radio1 = matplotlib.widgets.RadioButtons(rax1, names1)
    for circle in radio1.circles:  # adjusting radius. The default is 0.05
        circle.set_radius(0.01)

    rax2 = plt.axes([0.75, 0.05, 0.5, 0.9],
                    facecolor=axcol, frameon=False)
    radio2 = matplotlib.widgets.RadioButtons(rax2, names2)
    for circle in radio2.circles:  # adjusting radius. The default is 0.05
        circle.set_radius(0.01)

    d1 = df1[names1[0]]
    d2 = df2[names2[0]]
    if vmax is None:
        vmax = max(np.max(np.abs(df1.values)), np.max(np.abs(df2.values)))
        vmin = -vmax
    # d0 = pd.concat([d1, d1], axis=1)
    # v_max = np.max(np.abs(d0.values))*1.1
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin, vmax], [0, 0], c='k')
    ax.plot([0, 0], [vmin, vmax], c='k')
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    corr_txt = plt.text(0.05, 0.95, '', horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes,
                        color='r')

    curve, = ax.plot(d1, d2, '.', markersize=10)
    curve.set_data(d1, d2)  # this formulation keeps the pandas information
    annotations = [0]*len(d1.index)
    _update_annotations(ax, corr_txt, annotations, curve)

    radio1.on_clicked(lambda x: update_curve(x, df1, df1_multi, True))
    radio2.on_clicked(lambda x: update_curve(x, df2, df2_multi, False))

    return radio1, radio2

# not used in paper
# def plot_cov(data: pd.DataFrame, cell_sort=None, norm: bool = False,
#              title: str = '') -> (plt.Figure, plt.Axes, plt.colorbar):
#     """
#     plots the covariance of the input data
#
#
#     Parameters
#     ----------
#     data
#         input data
#     cell_sort
#         most probably in list or np.array format
#     norm
#         whether or not to divide by the number of samples
#     title
#         plot title
#
#     Returns
#     -------
#     f
#         reference to the figure
#     ax
#         reference to the axis
#     clb
#         reference to the colorbar
#     """
#
#     cov = data.T.dot(data)
#     if norm is True:
#         cov /= len(data)
#     if cell_sort is not None:
#         cov = cov.loc[cell_sort, cell_sort]
#     f, ax, clb = imshow_df(cov, vlim=np.max(cov.values),
#                               cmap=plt.cm.bwr, title=title, tight=True)
#     return f, ax, clb


# #############################################################################
# ####################  ACTIVITY PLOTS  #######################################
# #############################################################################
# definition used for actiivty plots
def get_div_color_map(vmin_col, vmin_to_show, vmax):
    colors_neg = plt.cm.PuBu(np.linspace(vmin_to_show / vmin_col, 0, 256))
    colors_pos = plt.cm.Oranges(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_neg, colors_pos))
    act_map = matplotlib.colors.LinearSegmentedColormap.from_list('act_map',
                                                                  all_colors)
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=vmin_to_show, vcenter=0,
                                             vmax=vmax)
    return act_map, divnorm


def plot_full_activity(df, act_map, divnorm, title='', cb_title='',
                       cb_ticks=None, pads=[0.55, 0.4, 0.2, 0.2], extend='max',
                       squeeze=0.45, do_vert_spl=True,
                       SQ=0.07, CB_W=0.1, CB_DX=0.11, title_font=None,
                       cb_title_font=None):
    # _, fs, axs = FP.calc_fs_ax_df(df, pads, sq=SQ)
    fs, axs = calc_fs_ax(pads, SQ * len(df.T) * squeeze, SQ * len(df))  # pads, gw, gh
    f = plt.figure(figsize=fs)
    ax = f.add_axes(axs)

    if do_vert_spl:
        splx = np.arange(5, len(df.T), 5)
    else:
        splx = []
    cp = imshow_df2(df, ax, vlim=None, show_lab_y=True,
                    title=title,
                    cmap=act_map, splits_x=splx,
                    show_lab_x=True, aspect='auto', splits_c='gray', lw=0.5,
                    title_font=title_font,
                    **{'norm': divnorm})
    ax_cb = f.add_axes([axs[0] + axs[2] + CB_DX / fs[0], axs[1],
                        CB_W / fs[0], axs[3]])
    clb = add_colorbar(cp, ax_cb, cb_title, cb_ticks, extend=extend,
                       title_font=cb_title_font)
    # clb = plt.colorbar(cp, cax=ax_cb, extend=extend)
    # clb.ax.set_yscale('linear')  # new option needed
    # clb.outline.set_linewidth(0.00)
    # clb.ax.tick_params(size=2, direction='in', pad=1.5)
    # clb.ax.set_title(cb_title, pad=1)
    # if set_ticks_params:
    #     clb.set_ticks(cb_ticks)
    #     clb.set_ticklabels(cb_ticks)
    # clb.set_ticks(cb_ticks)  # needs to be called again because of the linea command
    return f, ax, clb