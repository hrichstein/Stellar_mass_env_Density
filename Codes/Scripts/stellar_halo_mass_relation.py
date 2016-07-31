from __future__ import division, absolute_import

import astropy.stats
import glob
import math
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import os
import pandas as pd
from scipy import integrate,optimize,spatial

from matplotlib import rc,rcParams
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

class Vars(object):
    size_xlabel = 24
    size_ylabel = 24
    size_text   = 18
    size_tick   = 18

va = Vars()

# In[252]:

__author__     =['Victor Calderon']
__copyright__  =["Copyright 2016 Victor Calderon, Index function"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']

def Index(directory, datatype):
    """
    Indexes the files in a directory `directory' with a
    specific data type.

    Parameters
    ----------
    directory: str
            Absolute path to the folder that is indexed.

    datatype: str
            Data type of the files to be indexed in the folder.

    Returns
    -------
    file_array: array_like 
            np.array of indexed files in the folder 'directory' 
            with specific datatype.

    Examples
    --------
    >>> Index('~/data', '.txt')
    >>> array(['A.txt', 'Z'.txt', ...])
    """
    assert(os.path.exists(directory))
    files = np.array(glob.glob('{0}/*{1}'.format(directory, datatype)))

    return files

dirpath  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU"
dirpath += r"\Stellar_mass_env_density\Catalogs"
dirpath += r"\Resolve_plk_5001_so_mvir_scatter_ECO_Mocks_scatter_mocks"
dirpath += r"\Resolve_plk_5001_so_mvir_scatter0p3_ECO_Mocks"

# dirpath  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU"
# dirpath += r"\Stellar_mass_env_Density\Catalogs"
# dirpath += r"\Mocks_Scatter_Abundance_Matching"
# dirpath += r"\Resolve_plk_5001_so_mvir_scatter0p2_ECO_Mocks"

usecols  = (3,5,7,13)

# In[260]:

ECO_cats = (Index(dirpath,'.dat'))

names    = ['Mr','logMhalo','cent_sat','logMstar']

PD = [[] for ii in range(len(ECO_cats))]

for ii in range(len(ECO_cats)):
    temp_PD = (pd.read_csv(ECO_cats[ii],sep="\s+", usecols= usecols,\
        header=None,skiprows=2,names=names))
    # col_list = list(temp_PD)
    # col_list[2], col_list[3], col_list[4] = \
    # col_list[3], col_list[4], col_list[2]
    # temp_PD.ix[:,col_list]
    PD[ii] = temp_PD

PD_comp_1  = [(PD[ii][PD[ii].logMstar >= 9.1]) for ii in range(len(ECO_cats))]

PD_comp_2  = [(PD_comp_1[ii][PD_comp_1[ii].logMstar <= 11.77]) \
    for ii in range(len(ECO_cats))]
PD_comp    = [(PD_comp_2[ii][PD_comp_2[ii].Mr <= -17.33]) \
    for ii in range(len(ECO_cats))]

PD_comp_sats = [(PD_comp[ii][PD_comp[ii].cent_sat==0]) \
    for ii in range(len(PD_comp))]

PD_comp_cent = [(PD_comp[ii][PD_comp[ii].cent_sat==1]) \
    for ii in range(len(PD_comp))]    

logMhalo_arr_sats  = np.array([(PD_comp_sats[ii].logMhalo) \
    for ii in range(len(PD_comp))])  

logmstar_arr_sats  = np.array([(PD_comp_sats[ii].logMstar) \
    for ii in range(len(PD_comp))])

logMhalo_arr_cent  = np.array([(PD_comp_cent[ii].logMhalo) \
    for ii in range(len(PD_comp))])  

logmstar_arr_cent  = np.array([(PD_comp_cent[ii].logMstar) \
    for ii in range(len(PD_comp))])


def mean_mass(mhalo,mstar,bins):
    """
    """

    digitized    = np.digitize(mhalo,bins)
    digitized   -= int(1)

    bin_nums          = np.unique(digitized)
    
    bin_nums_list = list(bin_nums)

    if (len(bins)-1) not in bin_nums_list:
        bin_nums_list.append(len(bins)-1)

    if (len(bins)) not in bin_nums_list:
        bin_nums_list.append(len(bins))
        
    bin_nums = np.array(bin_nums_list)
    
    for ii in bin_nums:

        if len(mstar[digitized==ii]) == 0:
#             temp_list = list(mass_dist.T[kk]\
#                                              [digitized==ii])
#             temp_list.append(np.nan)
            mstar[digitized==ii] = np.nan

    # print bin_nums
    # print len(bin_nums)
    
    mean  = np.array([np.nanmean(mstar[digitized==ii]) \
                for ii in bin_nums])

    return mean

# Behroozi 2010 Model - SMHM Relation
def Behroozi_relation(log_stellar_mass):
    """
    Computes the Behroozi relation for Halo Mass

    Parameters
    ----------
    log_stellar_mass: array_like
        array of stellar masses in h=0.7 units

    Returns
    -------
    log_halo_mass: array_like
        array of halo masses in h=0.7 units
    """
    littleh = 0.7
    param_dict = {
        'smhm_m0_0': 10.72,
        'smhm_m0_a': 0.59,
        'smhm_m1_0': 12.35,
        'smhm_m1_a': 0.3,
        'smhm_beta_0': 0.43,
        'smhm_beta_a': 0.18,
        'smhm_delta_0': 0.56,
        'smhm_delta_a': 0.18,
        'smhm_gamma_0': 1.54,
        'smhm_gamma_a': 2.52}
    # Halo mass calculation
    redshift = 0.
    # Mstar in h=0.7
    stellar_mass = 10**log_stellar_mass
    # stellar_mass = (10.**log_stellar_mass)/(littleh**2)
    a = 1./(1. + redshift)

    logm0 =  param_dict['smhm_m0_0'] + param_dict['smhm_m0_a']*(a - 1)
    m0 = 10.**logm0
    logm1 = param_dict['smhm_m1_0'] + param_dict['smhm_m1_a']*(a - 1)
    beta = param_dict['smhm_beta_0'] + param_dict['smhm_beta_a']*(a - 1)
    delta = param_dict['smhm_delta_0'] + param_dict['smhm_delta_a']*(a - 1)
    gamma = param_dict['smhm_gamma_0'] + param_dict['smhm_gamma_a']*(a - 1)

    stellar_mass_by_m0 = stellar_mass/m0
    term3_numerator = (stellar_mass_by_m0)**delta
    term3_denominator = 1 + (stellar_mass_by_m0)**(-gamma)

    log_halo_mass = logm1 + beta*np.log10(stellar_mass_by_m0) + (term3_numerator/term3_denominator) - 0.5

    return np.log10(10.**log_halo_mass)

log_mstar_arr_h07 = np.linspace(9.0, 12.0, 100) # In h=0.7 units
log_mhalo_arr_h07 = Behroozi_relation(log_mstar_arr_h07) # In h=0.7 units

scatter_line = Behroozi_relation(10.5)

little_h = 0.7
bins = np.arange(10.8,14.6,0.1) # In h=0.7 units
bins_h1 = bins * little_h # In h=1 units
bin_centers_h07 = 0.5*(bins[:-1]+bins[1:]) # in h=0.7 units
bin_centers_h1  = 0.5*(bins_h1[:-1]+bins_h1[1:]) # In h=1 units

# test_mean = mean_mass(logMhalo_arr_cent[0],logmstar_arr_cent[0],bins)    

# mean_sm_h1  = model.mean_stellar_mass(prim_haloprop = 10**bin_centers_h1)
# mean_sm_h07 = mean_sm_h1 / (little_h**2) 

# mean_sm = (model.mean_stellar_mass(prim_haloprop = (10**(bin_centers)/0.49)))/0.7
# mean_sm_line = (model.mean_stellar_mass(prim_haloprop = (10**(12)/0.49)))/0.7


fig, ax = plt.subplots()

ax.set_ylim(9.1,12)
ax.set_xlim(10.7,14.5)

# ax.scatter(logMhalo_arr_cent[0],logmstar_arr_cent[0],color='royalblue',marker='o',\
#     s=1,label='Centrals')
# ax.scatter(logMhalo_arr_sats[0],logmstar_arr_sats[0],color='crimson',marker='o',\
#     s=1,label='Satellites')

# ax.plot(bin_centers,test_mean,color='darkmagenta',linewidth=2,\
    # label='Mean (Single Mock)')

ax.plot(log_mhalo_arr_h07, log_mstar_arr_h07,linewidth=2,\
    color='lime',label='Behroozi et al. 2010')
# ax.errorbar(scatter_line,10.5,yerr=0.1,lolims=(10.5-0.15),uplims\
#     =(10.5+0.15),color='fuchsia',linewidth=2,marker='.')

ax.set_xlabel(r'$\log\ (M_{Halo}/M_{\odot})$',fontsize=va.size_xlabel)
ax.set_ylabel(r'$\log\ (M_{*}/M_{\odot})$',\
    fontsize=va.size_ylabel)
ax.set_xticks(np.arange(11,14.5,0.5))
ax.set_yticks(np.arange(9.5,12,0.5))
ax.tick_params(axis='both', labelsize=va.size_tick)
# ax.text(0.05,0.7,'Scatter = 0.3 dex',fontsize=va.size_text,transform=ax.transAxes,\
#     weight='bold',verticalalignment='bottom')
ax.legend(loc='upper left',fontsize=14)
plt.tight_layout()
# ax.set_title('Galaxy Stellar-to-Halo Mass Relation',fontsize=20)
plt.show()