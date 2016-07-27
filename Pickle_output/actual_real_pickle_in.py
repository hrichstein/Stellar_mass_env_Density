from __future__ import division, absolute_import

import astropy.stats
import cPickle as pickle
import glob
import math
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import os
import pandas as pd
from scipy import integrate,optimize,spatial

plt.rc('font', **{'family': 'serif', 'serif':['Computer Modern']})

plt.rc('text', usetex=True)

###############################################################################

pickle_in = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
pickle_in+= r"\Pickle_output"

###############################################################################

pickle_in_rats = pickle_in
pickle_in_rats+=r"\ratio_bands.p"

rats_vals = pickle.load(open(pickle_in_rats,"rb"))

one_dex_ratios = rats_vals[0]
two_dex_ratios = rats_vals[1]
three_dex_ratios = rats_vals[2]

one_dex_rat_dict = {1:one_dex_ratios['1_4'],5:one_dex_ratios['5_4'],\
            20:one_dex_ratios['20_4']}

two_dex_rat_dict = {1:two_dex_ratios['1_4'],5:two_dex_ratios['5_4'],\
            20:two_dex_ratios['20_4']} 

three_dex_rat_dict = {1:three_dex_ratios['1_4'],5:three_dex_ratios['5_4'],\
            20:three_dex_ratios['20_4']}

all_rat_dict = {1:one_dex_rat_dict,2:two_dex_rat_dict,3:three_dex_rat_dict}

###############################################################################

pickle_in_meds = pickle_in
pickle_in_meds+=r"\med_bands.p"

meds_vals = pickle.load(open(pickle_in_meds,"rb"))

one_dex_meds = meds_vals[0]
two_dex_meds = meds_vals[1]
three_dex_meds = meds_vals[2]

# one_dm_slim = {1:one_dex_meds['1'],5:one_dex_meds['5'],20:one_dex_meds['20']}
# two_dm_slim = {1:two_dex_meds['1'],5:two_dex_meds['5'],20:two_dex_meds['20']}
# three_dm_slim = {1:three_dex_meds['1'],5:three_dex_meds['5'],\
#         20:three_dex_meds['20']}

one_dex_meds_dict = {1:one_dex_meds['1'],5:one_dex_meds['5'],\
            20:one_dex_meds['20']}

two_dex_meds_dict = {1:two_dex_meds['1'],5:two_dex_meds['5'],\
            20:two_dex_meds['20']} 

three_dex_meds_dict = {1:three_dex_meds['1'],5:three_dex_meds['5'],\
            20:three_dex_meds['20']}

all_meds_dict = {1:one_dex_meds_dict,2:two_dex_meds_dict,3:three_dex_meds_dict}

##dictionaries with [['10', '20', '1', '3', '2', '5']] keys
##yields a list with two arrays (upper and lower bounds)

###############################################################################

pickle_in_hists = pickle_in
pickle_in_hists+=r"\hist_bands.p"

hists_vals = pickle.load(open(pickle_in_hists,"rb"))

two_dex_hists_low = hists_vals[2]

hists_dict_low = {1:two_dex_hists_low['1_4'],5:two_dex_hists_low['5_4'],\
            20:two_dex_hists_low['20_4']}

two_dex_hists_high = hists_vals[3]

hists_dict_high = {1:two_dex_hists_high['1_4'],5:two_dex_hists_high['5_4'],\
            20:two_dex_hists_high['20_4']}

# for ii in neigh_vals:
#     for tt in range (2):
#         print len(hists_dict[ii][tt])            

###############################################################################


##eco_low,eco_high,eco_ratio_info, eco_final_bins,eco_medians
pickle_in_eco = pickle_in
pickle_in_eco+=r"\eco_data.p"

eco_vals = pickle.load(open(pickle_in_eco,"rb"))

eco_low_hist = eco_vals[0]
eco_high_hist = eco_vals[1]
eco_ratio = {1:eco_vals[2][0][0][4],5:eco_vals[2][3][0][4],\
            20:eco_vals[2][5][0][4]}
eco_rat_err = {1:eco_vals[2][0][1][1],5:eco_vals[2][3][1][1],\
            20:eco_vals[2][5][1][1]}
eco_bins = {1:eco_vals[3][0][1],5:eco_vals[3][3][1],20:eco_vals[3][5][1]}
eco_meds = {1:eco_vals[4][0],5:eco_vals[4][3],20:eco_vals[4][5]}

bins = np.arange(9.1,11.9,0.2)

bin_centers= 0.5*(bins[:-1]+bins[1:])

##eco_meds... eventually 3 arrays. First is median line, second and third are \
##low and high bootstrap
###############################################################################
pickle_in_eco_hists = pickle_in
pickle_in_eco_hists+=r"\eco_hists.p"

eco_hists = pickle.load(open(pickle_in_eco_hists,"rb"))

eco_high_counts = {1:(eco_hists[1][1][4],eco_hists[1][1]['err_4']),\
        5:(eco_hists[1][5][4],eco_hists[1][5]['err_4']),\
        20:(eco_hists[1][20][4],eco_hists[1][20]['err_4'])}
eco_low_counts  = {1:(eco_hists[0][1][4],eco_hists[0][1]['err_4']),\
        5:(eco_hists[0][5][4],eco_hists[0][5]['err_4']),\
        20:(eco_hists[0][20][4],eco_hists[0][20]['err_4'])}

eco_high_bins = {1:eco_hists[3][1][4],5:eco_hists[3][5][4],\
        20:eco_hists[3][20][4]}
eco_low_bins  = {1:eco_hists[2][1][4],5:eco_hists[2][5][4],\
        20:eco_hists[2][20][4]}     

###############################################################################


# def plot_bands(bin_centers,upper,lower,ax,plot_idx,color='silver',label=None):
#     """
#     """
#     # ax.set_yscale('symlog')
#     ax.set_ylim(0,4)
#     ax.set_xlim(9.1,11.8)
#     ax.set_xticks(np.arange(9.5, 12., 0.5))
#     ax.set_yticks([0,1,2,3,4])
#     ax.tick_params(axis='both', labelsize=12)
#     ax.fill_between(bin_centers,upper,lower,color=color,alpha=0.1,label=label)
#     if plot_idx == 0:
#         ax.legend(loc='best')
#     plot_neigh_dict = {0:1,1:5,2:20}
#     title = 'n = {0}'.format(plot_neigh_dict[plot_idx])
#     ax.text(0.05, 0.05, title,horizontalalignment='left',\
#                     verticalalignment='bottom',transform=ax.transAxes,fontsize=18)

# ###############################################################################        


# def plot_eco_rats(bin_centers,y_vals,y_err,neigh_val,ax,frac_val,plot_idx):
#     """
#     """
#     if plot_idx     ==1:
#         ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
#     ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
#     ax.errorbar(bin_centers,y_vals,yerr=y_err,\
#                     color='darkmagenta',linewidth=1,label='ECO')

###############################################################################


def plot_every_rat(bin_cens,upper,lower,ax,plot_idx,neigh_val,eco_bins,\
    eco_vals,eco_err,color='silver',label=None,eco=False,alpha=0.1):

    ax.set_ylim(0,4)
    ax.set_xlim(9.1,11.8)
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.set_yticks([0,1,2,3,4])
    ax.tick_params(axis='both', labelsize=18)
    ax.fill_between(bin_cens,upper,lower,color=color,alpha=alpha,label=label)
    plot_neigh_dict = {0:1,1:5,2:20}
    title = 'n = {0}'.format(neigh_val)
    ax.text(0.05, 0.05, title,horizontalalignment='left',\
                    verticalalignment='bottom',transform=ax.transAxes,\
                    fontsize=18)
    if eco == True:
        if plot_idx     ==1:
            ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
        ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
        ax.errorbar(eco_bins,eco_vals,yerr=eco_err,\
                    color='darkmagenta',linewidth=1,label='ECO')
    if plot_idx == 1:
        ax.legend(loc='best',numpoints=1)



###############################################################################

nrow_num = int(1)
ncol_num = int(3)



dict_to_neigh = {1:1,5:2,20:3}
dict_to_zz  = {1:0,5:1,20:2}

neigh_vals = np.array([1,5,20])

fig,axes = plt.subplots(nrows=nrow_num,ncols=ncol_num,figsize=(14,4),\
    sharey=True)

figure_title = fig.suptitle\
(r"Abundance Ratio of Galaxies in Top/Bottom 25\% Density Regions", \
    fontsize=20)

figure_title.set_y(1.0)
fig.subplots_adjust(bottom=0.2, right=0.9, top=0.85, hspace=0, wspace=0)

axes_flat= axes.flatten()

zz       = int(0)
while zz == 0:
    for yy in range(1,4):
        for xx in neigh_vals:
            upper = all_rat_dict[yy][xx][0]
            lower = all_rat_dict[yy][xx][1]
            if xx == 1:
                ax_as = axes_flat[0]
            if xx == 5:
                ax_as = axes_flat[1]
            if xx == 20:
                ax_as = axes_flat[2]
            if yy == 1:
                color = 'mediumspringgreen'
                label = '0.1 dex'
                alpha =  0.3
                eco   = True
            if yy ==2:
                color = 'mediumslateblue'
                label = '0.2 dex'
                alpha =  0.2
                eco   = True
            if yy ==3:
                color = 'orchid'
                label = '0.3 dex'
                alpha =  0.1
                eco   = True
            plot_every_rat(bin_centers[:-1],upper,lower,ax_as,yy,xx,\
                eco_bins[xx],eco_ratio[xx],eco_rat_err[xx],\
                color=color,label=label,eco=eco,alpha=alpha)

    zz+=1

plt.tight_layout()
plt.show()    

###############################################################################

def plot_every_med(bin_cens,upper,lower,ax,plot_idx,\
    eco_vals,neigh_val,color='silver',label=None,eco=False,alpha=0.1):

    # ax.set_ylim(0,4)
    ax.set_yscale('symlog')
    ax.set_xlim(9.1,11.8)
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(0,10**1.3)
    ax.set_yticks(np.arange(0,12,1))  
    ax.set_yticklabels(np.arange(1,11,2))
    ax.fill_between(bin_cens,upper,lower,color=color,alpha=alpha,label=label)
    title = 'n = {0}'.format(neigh_val)
    ax.text(0.05, 0.05, title,horizontalalignment='left',\
                    verticalalignment='bottom',transform=ax.transAxes,\
                    fontsize=18)
    if eco == True:
        if plot_idx  ==1:
            ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
        ax.errorbar(bin_cens,eco_vals[0],yerr=0.1,lolims=eco_vals[1],\
            uplims=eco_vals[2],\
                    color='darkmagenta',linewidth=1,label='ECO')
        ax.legend(loc='best',numpoints=1)


###############################################################################

nrow_num = int(1)
ncol_num = int(3)


neigh_vals = np.array([1,5,20])

fig,axes = plt.subplots(nrows=nrow_num,ncols=ncol_num,figsize=(14,4),\
    sharey=True)

figure_title = fig.suptitle(r"Median Distance to Nth Nearest Neighbor", \
    fontsize=20)
figure_title.set_y(1.0)
fig.subplots_adjust(bottom=0.2, right=0.9, top=0.85, hspace=0, wspace=0)

axes_flat= axes.flatten()
zz       = int(0)
while zz <= 2:
    for xx in neigh_vals:
        for yy in all_meds_dict:
            upper = all_meds_dict[yy][xx][0]
            lower = all_meds_dict[yy][xx][1]
            if xx == 1:
                ax_as = axes_flat[0]
            if xx == 5:
                ax_as = axes_flat[1]
            if xx == 20:
                ax_as = axes_flat[2]
            if yy == 1:
                color = 'violet'
                label = '0.1 dex'
                alpha =  0.3
                eco = False
            if yy == 2:
                color = 'lightsteelblue'
                label = '0.2 dex'
                alpha =  0.2
                eco = False
            if yy == 3:
                color = 'orchid'
                label = '0.3 dex'
                alpha =  0.1
                eco   = True
            plot_every_med(bin_centers[:-1],upper,lower,ax_as,zz,\
                eco_meds[xx],xx,color=color,label=label,eco=eco,alpha=alpha)

    zz+=1

plt.show()    

###############################################################################

def plot_eco_hists(bins_high,bins_low,high_counts,low_counts,               \
    high_counts_err,low_counts_err,ax,plot_idx,bin_centers,\
    upper_h,lower_h,upper_l,lower_l,text=False):
        ax.set_yscale('log')
        ax.set_xticks(np.arange(9.5,12,0.5))
        ax.errorbar(bins_high,high_counts,\
                            yerr=high_counts_err,drawstyle='steps-mid',\
                                    color='royalblue',label='Higher Density')
        ax.errorbar(bins_low,low_counts,\
                            yerr=low_counts_err,drawstyle='steps-mid',\
                                        color='crimson',label='Lower Density')
        ax.fill_between(bin_centers,upper_h,lower_h,color='lightsteelblue',alpha=0.2)
        ax.fill_between(bin_centers,upper_l,lower_l,color='lightsteelblue',alpha=0.2)                                               
        if plot_idx == 0:
            ax.set_ylabel(r'$\log\ (\frac{N_{gal/bin}}{N_{total}*dlogM})$')
        if plot_idx     ==1:
            ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)                                          
        if text == True:
            ax.legend(loc='best')

        

eco_low_hist = eco_vals[0]
eco_high_hist = eco_vals[1]


fig,axes = plt.subplots(nrows=nrow_num,ncols=ncol_num,figsize=(14,4),\
    sharey=True)

figure_title = fig.suptitle(r"Abundance of Galaxies in Top/Bottom 25\% Density Regions", \
    fontsize=20)
figure_title.set_y(1.0)
fig.subplots_adjust(bottom=0.2, right=0.9, top=0.85, hspace=0, wspace=0)

axes_flat = axes.flatten()

text = False
zz = int(0)
while zz <= 2:
    for xx in neigh_vals:
        if xx == 1:
            ax_as = axes_flat[0]
        if xx == 5:
            ax_as = axes_flat[1]
        if xx == 20:
            ax_as = axes_flat[2]
            if zz == 2:
                text = True
            else:
                text = False
        plot_eco_hists(eco_high_bins[xx],eco_low_bins[xx],\
            eco_high_counts[xx][0],eco_low_counts[xx][0],\
            eco_high_counts[xx][1],eco_low_counts[xx][1],\
            ax_as,zz,bin_centers[:-1],hists_dict_high[xx][0],\
            hists_dict_high[xx][1],hists_dict_low[xx][0],\
            hists_dict_low[xx][1],text=text)  
    zz+=1          

plt.show()