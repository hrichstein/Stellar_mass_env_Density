# neigh_dict and nn_dict are the same thing. 
# Need to combine/change variables at some point

# In[251]:

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


# In[253]:

def myceil(x, base=10):
    """
    Returns the upper-bound integer of 'x' in base 'base'.

    Parameters
    ----------
    x: float
        number to be approximated to closest number to 'base'

    base: float
        base used to calculate the closest 'largest' number

    Returns
    -------
    n_high: float
        Closest float number to 'x', i.e. upper-bound float.

    Example
    -------
    >>>> myceil(12,10)
      20
    >>>>
    >>>> myceil(12.05, 0.1)
     12.10000 
    """
    n_high = float(base*math.ceil(float(x)/base))

    return n_high

###############################################################################    

def myfloor(x, base=10):
    """
    Returns the lower-bound integer of 'x' in base 'base'

    Parameters
    ----------
    x: float
        number to be approximated to closest number of 'base'

    base: float
        base used to calculate the closest 'smallest' number

    Returns
    -------
    n_low: float
        Closest float number to 'x', i.e. lower-bound float.

    Example
    -------
    >>>> myfloor(12, 5)
    >>>> 10
    """
    n_low = float(base*math.floor(float(x)/base))

    return n_low

###############################################################################

def Bins_array_create(arr, base=10):
    """
    Generates array between [arr.min(), arr.max()] in steps of `base`.

    Parameters
    ----------
    arr: array_like, Shape (N,...), One-dimensional
        Array of numerical elements

    base: float, optional (default=10)
        Interval between bins

    Returns
    -------
    bins_arr: array_like
        Array of bin edges for given arr

    """
    base = float(base)
    arr  = np.array(arr)
    assert(arr.ndim==1)
    arr_min  = myfloor(arr.min(), base=base)
    arr_max  = myceil( arr.max(), base=base)
    bins_arr = np.arange(arr_min, arr_max+0.5*base, base)

    return bins_arr


# In[254]:

def sph_to_cart(ra,dec,cz):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    ra: array-like
        right-ascension of galaxies in degrees
    dec: array-like
        declination of galaxies in degrees
    cz: array-like
        velocity of galaxies in km/s

    Returns
    -------
    coords: array-like, shape = N by 3
        x, y, and z coordinates
    """
    cz_dist = cz/70. #converts velocity into distance
    x_arr   = cz_dist*np.cos(np.radians(ra))*np.cos(np.radians(dec))
    y_arr   = cz_dist*np.sin(np.radians(ra))*np.cos(np.radians(dec))
    z_arr   = cz_dist*np.sin(np.radians(dec))
    coords  = np.column_stack((x_arr,y_arr,z_arr))

    return coords

############################################################################

def calc_dens(n_val,r_val):
    """
    Returns densities of spheres with radius being the distance to the 
        nth nearest neighbor.

    Parameters
    ----------
    n_val = integer
        The 'N' from Nth nearest neighbor
    r_val = array-like
        An array with the distances to the Nth nearest neighbor for
        each galaxy

    Returns
    -------
    dens: array-like
        An array with the densities of the spheres created with radii
        to the Nth nearest neighbor.
    """
    dens = np.array([(3.*(n_val+1)/(4.*np.pi*r_val[hh]**3))\
        for hh in range(len(r_val))])

    return dens


# In[255]:

def plot_calcs(mass,bins,dlogM):
    """
    Returns values for plotting the stellar mass function and 
        mass ratios

    Parameters
    ----------
    mass: array-like
        A 1D array with mass values, assumed to be in order
    bins: array=like
        A 1D array with the values which will be used as the bin edges
        by the histogram function
    dlogM: float-like
        The log difference between bin edges

    Returns
    -------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    mass-freq: array-like
        Contains the number density values of each mass bin
    ratio_dict: dictionary-like
        A dictionary with three keys, corresponding to the divisors
        2,4, and 10 (as the percentile cuts are based on these 
        divisions). Each key has the density-cut, mass ratios for
        that specific cut (50/50 for 2; 25/75 for 4; 10/90 for 10).
    """

    mass_counts, edges = np.histogram(mass,bins)
    bin_centers        = 0.5*(edges[:-1]+edges[1:])

    mass_freq  = mass_counts/float(len(mass))/dlogM
    
#     non_zero   = (mass_freq!=0)

    ratio_dict = {}
    frac_val   = [2,4,10]

    yerr = []
    bin_centers_fin = []

    for ii in frac_val:
        ratio_dict[ii] = {}
        frac_data      = int(len(mass)/ii)
        
        # Calculations for the lower density cut
        frac_mass      = mass[0:frac_data]
        counts, edges  = np.histogram(frac_mass,bins)

        # Calculations for the higher density cut
        frac_mass_2       = mass[-frac_data:]
        counts_2, edges_2 = np.histogram(frac_mass_2,bins)

        # Ratio determination
        ratio_counts   = (1.*counts_2)/(1.*counts)
        
        non_zero = np.isfinite(ratio_counts)

        ratio_counts_1 = ratio_counts[non_zero]
        
#         print 'len ratio_counts: {0}'.format(len(ratio_counts_1))
        
        ratio_dict[ii] = ratio_counts_1
        
        temp_yerr = (counts_2*1.)/(counts*1.)*            \
        np.sqrt(1./counts + 1./counts_2)
            
        temp_yerr_1 = temp_yerr[non_zero]
        
#         print 'len yerr: {0}'.format(len(temp_yerr_1))

        yerr.append(temp_yerr_1)
        
        bin_centers_1 = bin_centers[non_zero]
        
#         print 'len bin_cens: {0}'.format(len(bin_centers_1))
        
        bin_centers_fin.append(bin_centers_1)
        

    mass_freq_list     = [[] for xx in xrange(2)]
    mass_freq_list[0]  = mass_freq
    mass_freq_list[1]  = np.sqrt(mass_counts)/float(len(mass))/dlogM
    mass_freq          = np.array(mass_freq_list)

    ratio_dict_list    = [[] for xx in range(2)]
    ratio_dict_list[0] = ratio_dict
    ratio_dict_list[1] = yerr
    ratio_dict         = ratio_dict_list

    return bin_centers, mass_freq, ratio_dict, bin_centers_fin


# In[366]:

def bin_func(mass_dist,bins,kk,bootstrap=False):
    """
    Returns median distance to Nth nearest neighbor

    Parameters
    ----------
    mass_dist: array-like
        An array with mass values in at index 0 (when transformed) and distance 
        to the Nth nearest neighbor in the others
        Example: 6239 by 7
            Has mass values and distances to 6 Nth nearest neighbors  
    bins: array=like
        A 1D array with the values which will be used as the bin edges     
    kk: integer-like
        The index of mass_dist (transformed) where the appropriate distance 
        array may be found

    Optional
    --------
    bootstrap == True
        Calculates the bootstrap errors associated with each median distance 
        value. Creates an array housing arrays containing the actual distance 
        values associated with every galaxy in a specific bin. Bootstrap error
        is then performed using astropy, and upper and lower one sigma values 
        are found for each median value.  These are added to a list with the 
        median distances, and then converted to an array and returned in place 
        of just 'medians.'

    Returns
    -------
    medians: array-like
        An array with the median distance to the Nth nearest neighbor from 
        all the galaxies in each of the bins
    """
    
    edges        = bins
    bin_centers  = 0.5*(edges[:-1]+edges[1:])

    # print 'length bins:'
    # print len(bins)

    digitized    = np.digitize(mass_dist.T[0],edges)
    digitized   -= int(1)

    bin_nums          = np.unique(digitized)
    
    bin_nums_list = list(bin_nums)

    if (len(bin_centers)) in bin_nums_list:
        bin_nums_list.remove(len(bin_centers))
        
    bin_nums = np.array(bin_nums_list)
    
#     print bin_nums

    non_zero_bins = []
    for ii in bin_nums:
        if (len(mass_dist.T[kk][digitized==ii]) != 0):
            non_zero_bins.append(bin_centers[ii])
#     print len(non_zero_bins)
    
    for ii in bin_nums:

        if len(mass_dist.T[kk][digitized==ii]) == 0:
#             temp_list = list(mass_dist.T[kk]\
#                                              [digitized==ii])
#             temp_list.append(np.nan)
            mass_dist.T[kk][digitized==ii] = np.nan

    # print bin_nums
    # print len(bin_nums)
    
    medians  = np.array([np.nanmedian(mass_dist.T[kk][digitized==ii])\
        for ii in bin_nums])

    # print len(medians)

    if bootstrap == True:
        dist_in_bin    = np.array([(mass_dist.T[kk][digitized==ii])\
                         for ii in bin_nums])
        for vv in range(len(dist_in_bin)):
            if len(dist_in_bin[vv]) == 0:
#                 dist_in_bin_list = list(dist_in_bin[vv])
#                 dist_in_bin[vv] = np.zeros(len(dist_in_bin[0]))
                dist_in_bin[vv] = np.nan
        low_err_test   = np.array([np.percentile(astropy.stats.bootstrap\
                         (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),16)\
                                            for vv in range(len(dist_in_bin))])
        high_err_test  = np.array([np.percentile(astropy.stats.bootstrap\
                        (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),84)\
                                         for vv in range(len(dist_in_bin))])

        med_list    = [[] for yy in range(3)]
        med_list[0] = medians
        med_list[1] = low_err_test
        med_list[2] = high_err_test
        medians     = np.array(med_list)
        
#     print len(medians)
#     print len(non_zero_bins)

    return medians, np.array(non_zero_bins)
    


# In[257]:

def hist_calcs(mass,bins,dlogM):
    """
    Returns dictionaries with the counts for the upper
        and lower density portions; calculates the 
        three different percentile cuts for each mass
        array given
    
    Parameters
    ----------
    mass: array-like
        A 1D array with log stellar mass values, assumed
        to be an order which corresponds to the ascending 
        densities; (necessary, as the index cuts are based 
        on this)
    bins: array-like
        A 1D array with the values which will be used as the bin edges   
    dlogM: float-like
        The log difference between bin edges
        
    Returns
    -------
    hist_dict_low: dictionary-like
        A dictionary with three keys (the frac vals), with arrays
        as values. The values for the lower density cut
    hist_dict_high: dictionary like
        A dictionary with three keys (the frac vals), with arrays
        as values. The values for the higher density cut
    """
    hist_dict_low  = {}
    hist_dict_high = {}
    bin_cens_low   = {}
    bin_cens_high  = {}
    frac_val  = np.array([2,4,10])
    frac_dict = {2:0,4:1,10:2}
    
    edges = bins
    
    bin_centers = 0.5 * (edges[:-1]+edges[1:])
    
    low_err   = [[] for xx in xrange(len(frac_val))]
    high_err  = [[] for xx in xrange(len(frac_val))]
    
    for ii in frac_val:
#         hist_dict_low[ii]  = {}
#         hist_dict_high[ii] = {}
    
        frac_data     = int(len(mass)/ii)
        
        frac_mass     = mass[0:frac_data]
        counts, edges = np.histogram(frac_mass,bins)
        low_counts    = (counts/float(len(frac_mass))/dlogM)
        
        non_zero = (low_counts!=0)
        low_counts_1 = low_counts[non_zero]
        hist_dict_low[ii]  = low_counts_1
        bin_cens_low[ii]   = bin_centers[non_zero]
        
        ##So... I don't actually know if I need to be calculating error
        ##on the mocks. I thought I didn't, but then, I swear someone
        ##*ahem (Victor)* said to. So I am. Guess I'm not sure they're
        ##useful. But I'll have them if necessary. And ECO at least
        ##needs them.
        
        low_err = np.sqrt(counts)/len(frac_mass)/dlogM
        low_err_1 = low_err[non_zero]
        err_key = 'err_{0}'.format(ii)
        hist_dict_low[err_key] = low_err_1
        
        frac_mass_2        = mass[-frac_data:]
        counts_2, edges_2  = np.histogram(frac_mass_2,bins)
        high_counts        = (counts_2/float(len(frac_mass_2))/dlogM)
        
        non_zero = (high_counts!=0)
        high_counts_1 = high_counts[non_zero]
        hist_dict_high[ii] = high_counts_1
        bin_cens_high[ii]  = bin_centers[non_zero]
        
        high_err = np.sqrt(counts_2)/len(frac_mass_2)/dlogM
        high_err_1 = high_err[non_zero]
        hist_dict_high[err_key] = high_err_1
    
    return hist_dict_low, hist_dict_high, bin_cens_low, bin_cens_high


# In[258]:

def mean_bin_mass(mass_dist,bins,kk):
    """
    Returns mean mass of galaxies in each bin

    Parameters
    ----------
    mass_dist: array-like
        An array with mass values in at index 0 (when transformed) 
    bins: array=like
        A 1D array with the values which will be used as the bin edges     

    Returns
    -------

    """
    edges        = bins

    digitized    = np.digitize(mass_dist.T[0],edges)
    digitized   -= int(1)
    
    bin_nums          = np.unique(digitized)
    
    for ii in bin_nums:
        if len(mass_dist.T[kk][digitized==ii]) == 0:
            mass_dist.T[kk][digitized==ii] = np.nan



    mean_mass = np.array([np.nanmean(mass_dist.T[0][digitized==ii])\
                     for ii in bin_nums])

    return mean_mass 


# In[259]:

# dirpath  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density"
# dirpath += r"\Catalogs\Resolve_plk_5001_so_mvir_scatter_ECO_Mocks_"
# dirpath += r"scatter_mocks\Resolve_plk_5001_so_mvir_scatter0p1_ECO_Mocks"

dirpath  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU"
dirpath += r"\Stellar_mass_env_Density\Catalogs"
dirpath += r"\Mocks_Scatter_Abundance_Matching"
dirpath += r"\Resolve_plk_5001_so_mvir_scatter0p1_ECO_Mocks"


# figsave_path = r"C:\Users\Hannah\Desktop\Vanderbilt_REU"
# figsave_path+= r"\Stellar_mass_env_Density\Plots"
# figsave_path+= r"\Abundance_matched"
# figsave_path+= r"\three_dec"


usecols  = (0,1,4,7,8,13)
dlogM    = 0.2
neigh_dict = {1:0,2:1,3:2,5:3,10:4,20:5}


# In[260]:

ECO_cats = (Index(dirpath,'.dat'))

names    = ['ra','dec','Halo_ID','cen_sat_flag','cz','logMstar']

PD = [[] for ii in range(len(ECO_cats))]

for ii in range(len(ECO_cats)):
    temp_PD = (pd.read_csv(ECO_cats[ii],sep="\s+", usecols= usecols,\
        header=None,skiprows=2,names=names))
    col_list = list(temp_PD)
    col_list[2], col_list[3], col_list[4] = \
    col_list[3], col_list[4], col_list[2]
    temp_PD.ix[:,col_list]
    PD[ii] = temp_PD

PD_comp_1  = [(PD[ii][PD[ii].logMstar >= 9.1]) for ii in range(len(ECO_cats))]
PD_comp_2  = [(PD_comp_1[ii][PD_comp_1[ii].logMstar <=11.77]) \
for ii in range(len(ECO_cats))]

PD_comp = [(PD_comp_2[ii][PD_comp_2[ii].cen_sat_flag == 0]) \
for ii in range(len(ECO_cats))]

[(PD_comp[ii].reset_index(drop=True,inplace=True)) \
for ii in range(len(ECO_cats))]

min_max_mass_arr = []

for ii in range(len(PD_comp)):
    min_max_mass_arr.append(max(PD_comp[ii].logMstar))
    min_max_mass_arr.append(min(PD_comp[ii].logMstar))

min_max_mass_arr = np.array(min_max_mass_arr)

bins = Bins_array_create(min_max_mass_arr,dlogM)
bins+= 0.1
bins_list = list(bins)
for ii in bins:
    if ii > 11.77:
        bins_list.remove(ii)

bins = np.array(bins_list)

bin_centers = 0.5 * (bins[:-1]+bins[1:])

num_of_bins = int(len(bins) - 1) 

ra_arr  = np.array([(PD_comp[ii].ra)     for ii in range(len(PD_comp))])

dec_arr  = np.array([(PD_comp[ii].dec)     for ii in range(len(PD_comp))])

cz_arr  = np.array([(PD_comp[ii].cz)     for ii in range(len(PD_comp))])

mass_arr  = np.array([(PD_comp[ii].logMstar)     for ii in range(len(PD_comp))])

halo_id_arr  = np.array([(PD_comp[ii].Halo_ID)     for ii in range(len(PD_comp))])


coords_test = np.array([sph_to_cart(ra_arr[vv],dec_arr[vv],cz_arr[vv])\
                 for vv in range(len(ECO_cats))])

neigh_vals  = np.array([1,2,3,5,10,20])

nn_arr_temp = [[] for uu in xrange(len(coords_test))]
nn_arr      = [[] for xx in xrange(len(coords_test))]
nn_arr_nn   = [[] for yy in xrange(len(neigh_vals))]
nn_idx      = [[] for zz in xrange(len(coords_test))]

for vv in range(len(coords_test)):
    nn_arr_temp[vv] = spatial.cKDTree(coords_test[vv])
    nn_arr[vv] = np.array(nn_arr_temp[vv].query(coords_test[vv],21)[0])
    nn_idx[vv] = np.array(nn_arr_temp[vv].query(coords_test[vv],21)[1])
    

nn_specs       = [(np.array(nn_arr).T[ii].T[neigh_vals].T) for ii in\
                     range(len(coords_test))]
nn_mass_dist   = np.array([(np.column_stack((mass_arr[qq],nn_specs[qq])))\
                     for qq in range(len(coords_test))])

nn_neigh_idx      = np.array([(np.array(nn_idx).T[ii].T[neigh_vals].T) \
                    for ii in range(len(coords_test))])

truth_vals = {}
for ii in range(len(halo_id_arr)):
    truth_vals[ii] = {}
    for jj in neigh_vals:
        halo_id_neigh = halo_id_arr[ii][nn_neigh_idx[ii].T[neigh_dict[jj]]].values
        truth_vals[ii][jj] = halo_id_neigh==halo_id_arr[ii].values


# In[265]:

halo_frac = {}
for ii in range(len(mass_arr)):
    halo_frac[ii] = {}
    mass_binning = np.digitize(mass_arr[ii],bins)
    bins_to_use = list(np.unique(mass_binning))
    if (len(bins)-3) not in bins_to_use:
        bins_to_use.append(len(bins)-3)
    if (len(bins)-2) not in bins_to_use:
        bins_to_use.append(len(bins)-2)
    if (len(bins)-1) not in bins_to_use:
        bins_to_use.append(len(bins)-1)
    if len(bins) in bins_to_use:
        bins_to_use.remove(len(bins))
    if (len(bins)+1) in bins_to_use:
        bins_to_use.remove((len(bins)+1))
    for jj in neigh_vals:
        one_zero = truth_vals[ii][jj].astype(int)
        frac = []
        for xx in bins_to_use:
            truth_binning = one_zero[mass_binning==xx]
            num_in_bin = len(truth_binning)
            if num_in_bin == 0:
                num_in_bin = np.nan
            num_same_halo = np.count_nonzero(truth_binning==1)
            frac.append(num_same_halo/(1.*num_in_bin))
        print len(frac)
        halo_frac[ii][jj] = frac


# In[266]:

nn_dict   = {1:0,2:1,3:2,5:3,10:4,20:5}

mean_mock_halo_frac = {}

for ii in neigh_vals:
    for jj in range(len(halo_frac)):
        bin_str = '{0}'.format(ii)
        oo_arr = halo_frac[jj][ii]
        n_o_elem = len(oo_arr)
        if jj == 0:
            oo_tot = np.zeros((n_o_elem,1))
        oo_tot = np.insert(oo_tot,len(oo_tot.T),oo_arr,1)
    oo_tot = np.array(np.delete(oo_tot,0,axis=1))
    oo_tot_mean = [np.nanmean(oo_tot[uu]) for uu in xrange(len(oo_tot))]
    oo_tot_std  = [np.nanstd(oo_tot[uu])/np.sqrt(len(halo_frac)) \
    for uu in xrange(len(oo_tot))]
    mean_mock_halo_frac[bin_str] = [oo_tot_mean,oo_tot_std]


def plot_halo_frac(bin_centers,y_vals,ax,plot_idx,text=False):
    titles = [1,2,3,5,10,20]
    ax.set_xlim(9.1,11.9)
    ax.set_xticks(np.arange(9.5,12.,0.5)) 
    ax.tick_params(axis='x', which='major', labelsize=16)
    if text == True:
        title_here = 'n = {0}'.format(titles[plot_idx])
        ax.text(0.05, 0.95, title_here,horizontalalignment='left',            \
            verticalalignment='top',transform=ax.transAxes,fontsize=18)
    if plot_idx == 4:
        ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=20)
    ax.plot(bin_centers,y_vals,color='silver')
    
def plot_mean_halo_frac(bin_centers,mean_vals,ax,std):
    ax.errorbar(bin_centers,mean_vals,yerr=std,color='maroon')



# In[56]:

nrow = int(2)
ncol = int(3)

fig,axes = plt.subplots(nrows=nrow,ncols=ncol,                        \
    figsize=(100,200),sharex=True)
axes_flat = axes.flatten()

zz = int(0)
while zz <=4:
    for jj in neigh_vals:
        for kk in range(len(halo_frac)):
            if kk == 0:
                value = True
            else:
                value = False
            plot_halo_frac(bin_centers,halo_frac[kk][jj],axes_flat[zz],zz,\
                text = value)
        nn_str = '{0}'.format(jj)
        plot_mean_halo_frac(bin_centers,mean_mock_halo_frac[nn_str][0],\
            axes_flat[zz],mean_mock_halo_frac[nn_str][1])
        # save_means = open("halo_frac_means.txt", "a")
        # save_means.write\
        # (("{0} + \n + 'nn_val' + {1} + 'mean' {2} \n + 'error' {3}")\
        # .format(dirpath,jj,mean_mock_halo_frac[nn_str][0],\
        #     mean_mock_halo_frac[nn_str][1]))
        # save_means.close()
        zz += 1

plt.subplots_adjust(top=0.97,bottom=0.1,left=0.03,right=0.99,hspace=0.10,\
    wspace=0.12)     

# plt.savefig(figsave_path + r"\halo_frac_means")

plt.show()      


# In[342]:


# nn_dist    = {}
nn_dens    = {}
mass_dat   = {}
ratio_info = {}
bin_cens_diff = {}

mass_freq  = [[] for xx in xrange(len(coords_test))]

for ii in range(len(coords_test)):
#     nn_dist[ii]    = {}
    nn_dens[ii]    = {}
    mass_dat[ii]   = {}
    ratio_info[ii] = {}
    bin_cens_diff[ii] = {}

#     nn_dist[ii]['mass'] = nn_mass_dist[ii].T[0]

    for jj in range(len(neigh_vals)):
#         nn_dist[ii][(neigh_vals[jj])]  = np.array(nn_mass_dist[ii].T\
#                                             [range(1,len(neigh_vals)+1)[jj]])
        nn_dens[ii][(neigh_vals[jj])]  = np.column_stack((nn_mass_dist[ii].T\
                                                [0],calc_dens(neigh_vals[jj],\
                        nn_mass_dist[ii].T[range(1,len(neigh_vals)+1)[jj]])))

        idx = np.array([nn_dens[ii][neigh_vals[jj]].T[1].argsort()])
        mass_dat[ii][(neigh_vals[jj])] = (nn_dens[ii][neigh_vals[jj]]\
                                                    [idx].T[0])

        bin_centers, mass_freq[ii], ratio_info[ii][neigh_vals[jj]],\
        bin_cens_diff[ii][neigh_vals[jj]] = \
                        plot_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)

all_mock_meds = {}
mock_meds_bins = {}
all_mock_mass_means = {}

for vv in range(len(nn_mass_dist)):
    all_mock_meds[vv] = {}
    mock_meds_bins[vv]= {}
    all_mock_mass_means[vv] = {}
    for jj in range(len(nn_mass_dist[vv].T)-1):
        all_mock_meds[vv][neigh_vals[jj]],mock_meds_bins[vv][neigh_vals[jj]]\
         = (bin_func(nn_mass_dist[vv],bins,(jj+1)))
        all_mock_mass_means[vv][neigh_vals[jj]] =\
         (mean_bin_mass(nn_mass_dist[vv],bins,(jj+1))) 


# In[358]:

med_plot_arr = {}

for ii in range(len(neigh_vals)):
    med_plot_arr[neigh_vals[ii]] = {}
    for jj in range(len(nn_mass_dist)):
        med_plot_arr[neigh_vals[ii]][jj] = all_mock_meds[jj][neigh_vals[ii]]    

# for ii in range(len(neigh_vals)):
#     for jj in range(len(nn_mass_dist)):
#         print len(all_mock_meds[jj][ii])

mass_freq_plot  = (np.array(mass_freq))
max_lim = [[] for xx in range(len(mass_freq_plot.T))]
min_lim = [[] for xx in range(len(mass_freq_plot.T))]
for jj in range(len(mass_freq_plot.T)):
    max_lim[jj] = max(mass_freq_plot.T[jj][0])
    min_lim[jj] = min(mass_freq_plot.T[jj][0])

global bins_curve_fit

bins_curve_fit = bins.copy()
# global bins_curve_fit

# In[281]:

eco_path  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density"
eco_path += r"\Catalogs\ECO_true"
eco_cols  = np.array([0,1,2,4])


# In[282]:

ECO_true = (Index(eco_path,'.txt'))
names    = ['ra','dec','cz','logMstar']
PD_eco   = pd.read_csv(ECO_true[0],sep="\s+", usecols=(eco_cols),header=None, \
               skiprows=1,names=names)
eco_comp = PD_eco[PD_eco.logMstar >= 9.1]

ra_eco   = (np.array(eco_comp)).T[0]
dec_eco  = (np.array(eco_comp)).T[1] 
cz_eco   = (np.array(eco_comp)).T[2] 
mass_eco = (np.array(eco_comp)).T[3]

coords_eco        = sph_to_cart(ra_eco,dec_eco,cz_eco)
eco_neighbor_tree = spatial.cKDTree(coords_eco)
eco_tree_dist     = np.array(eco_neighbor_tree.query(coords_eco,        \
            (neigh_vals[-1]+1))[0])

eco_mass_dist = np.column_stack((mass_eco,eco_tree_dist.T[neigh_vals].T))
##range 1,7 because of the six nearest neighbors (and fact that 0 is mass)
##the jj is there to specify which index in the [1,6] array
eco_dens = ([calc_dens(neigh_vals[jj],            (eco_mass_dist.T[range(1,7)\
    [jj]])) for jj in range (len(neigh_vals))])

eco_mass_dens = [(np.column_stack((mass_eco,eco_dens[ii]))) for ii in\
                 range(len(neigh_vals))]
eco_idx  = [(eco_mass_dens[jj].T[1].argsort()) for jj in \
            range(len(neigh_vals))]
eco_mass_dat  = [(eco_mass_dens[jj][eco_idx[jj]].T[0]) for jj in\
                 range(len(neigh_vals))]

eco_ratio_info    = [[] for xx in xrange(len(eco_mass_dat))]
eco_final_bins    = [[] for xx in xrange(len(eco_mass_dat))]


for qq in range(len(eco_mass_dat)):
    bin_centers, eco_freq, eco_ratio_info[qq],eco_final_bins[qq] = \
    plot_calcs(eco_mass_dat[qq],bins,dlogM)

eco_medians   = [[] for xx in xrange(len(eco_mass_dat))]    
eco_med_bins   = [[] for xx in xrange(len(eco_mass_dat))]    
eco_mass_means   = [[] for xx in xrange(len(eco_mass_dat))] 

for jj in (range(len(eco_mass_dat))):
    eco_medians[jj],eco_med_bins[jj] = np.array(bin_func(eco_mass_dist,\
        bins,(jj+1),bootstrap=True))
    eco_mass_means[jj] = (mean_bin_mass(eco_mass_dist,bins,(jj+1))) 


# In[283]:

hist_low_info  = {}
hist_high_info = {}
hist_low_bins  = {}
hist_high_bins = {}

for ii in xrange(len(coords_test)):
    hist_low_info[ii]  = {}
    hist_high_info[ii] = {}
    hist_low_bins[ii]  = {}
    hist_high_bins[ii] = {}
    
    for jj in range(len(neigh_vals)):
        hist_low_info[ii][neigh_vals[jj]],\
                hist_high_info[ii][neigh_vals[jj]],\
                         hist_low_bins[ii][neigh_vals[jj]],\
                                 hist_high_bins[ii][neigh_vals[jj]]\
                         = hist_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)
        
frac_vals     = [2,4,10]
hist_low_arr  = [[[] for yy in xrange(len(nn_mass_dist))] for xx in\
     xrange(len(neigh_vals))]
hist_high_arr = [[[] for yy in xrange(len(nn_mass_dist))] for xx in\
     xrange(len(neigh_vals))]


# In[284]:

hist_low_info[0][1]


# In[285]:

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        hist_low_arr[ii][jj]  = (hist_low_info[jj][neigh_vals[ii]])
        hist_high_arr[ii][jj] = (hist_high_info[jj][neigh_vals[ii]])
        
##I unindented the below two "lines". Because they don't \
# seem to need to be called iteratively        
        
plot_low_hist  = [[[[] for yy in xrange(len(nn_mass_dist))]\
                          for zz in xrange(len(frac_vals))] for xx in\
                                                    xrange(len(hist_low_arr))]
        
plot_high_hist = [[[[] for yy in xrange(len(nn_mass_dist))]\
                  for zz in xrange(len(frac_vals))] for xx in\
                                    xrange(len(hist_high_arr))]

for jj in range(len(nn_mass_dist)):
    for hh in range(len(frac_vals)):
        for ii in range(len(neigh_vals)):
            plot_low_hist[ii][hh][jj]  = hist_low_arr[ii][jj][frac_vals[hh]]        
            plot_high_hist[ii][hh][jj] = hist_high_arr[ii][jj][frac_vals[hh]] 


# In[286]:

eco_mass_means


# In[287]:

eco_low  = {}
eco_high = {}
eco_low_bins = {}
eco_high_bins = {}
for jj in range(len(neigh_vals)):
    eco_low[neigh_vals[jj]]  = {}
    eco_high[neigh_vals[jj]] = {}
    eco_low_bins[neigh_vals[jj]]  = {}
    eco_high_bins[neigh_vals[jj]] = {}
    eco_low[neigh_vals[jj]], eco_high[neigh_vals[jj]],\
         eco_low_bins[neigh_vals[jj]], eco_high_bins[neigh_vals[jj]]=\
              hist_calcs(eco_mass_dat[jj],bins,dlogM)


# In[288]:

eco_low[1]


# In[289]:

def perc_calcs(mass,bins,dlogM):
    mass_counts, edges = np.histogram(mass,bins)
    mass_freq          = mass_counts/float(len(mass))/dlogM
    
    bin_centers        = 0.5*(edges[:-1]+edges[1:])
    
    non_zero = (mass_freq!=0)
    
    mass_freq_1 = mass_freq[non_zero]
    
    smf_err            = np.sqrt(mass_counts)/float(len(mass))/dlogM
    
    smf_err_1   = smf_err[non_zero] 
    
    bin_centers_1 = bin_centers[non_zero]

    return mass_freq_1, smf_err_1, bin_centers_1


# In[290]:

def quartiles(mass):
    dec_val     =  int(len(mass)/4)
    res_list      =  [[] for bb in range(4)]

    for aa in range(0,4):
        if aa == 3:
            res_list[aa] = mass[aa*dec_val:]
        else:
            res_list[aa] = mass[aa*dec_val:(aa+1)*dec_val]

    return res_list


# In[291]:

def deciles(mass):
    dec_val     =  int(len(mass)/10)
    res_list      =  [[] for bb in range(10)]

    for aa in range(0,10):
        if aa == 9:
            res_list[aa] = mass[aa*dec_val:]
        else:
            res_list[aa] = mass[aa*dec_val:(aa+1)*dec_val]

    return res_list


# In[412]:

def mean_perc_mass(mass,bins):
    """
    Returns mean mass of galaxies in each bin

    Parameters
    ----------
    mass_dist: array-like
        An array with mass values in at index 0 (when transformed) 
    bins: array=like
        A 1D array with the values which will be used as the bin edges     

    Returns
    -------

    """
    edges        = bins

    digitized    = np.digitize(mass,edges)
    digitized   -= int(1)
    
    bin_nums          = np.unique(digitized)
    
    for ii in bin_nums:
        if len(mass[digitized==ii]) == 0:
            mass[digitized==ii] = np.nan



    mean_mass = np.array([np.nanmean(mass[digitized==ii])\
                    for ii in bin_nums])

    return mean_mass 


eco_dec = {}
for cc in range(len(eco_mass_dat)):
    eco_dec[neigh_vals[cc]] = deciles(eco_mass_dat[cc])
    
eco_dec_smf = {}
eco_dec_err = {}
eco_dec_bin = {}

for ss in neigh_vals:
    eco_dec_smf[ss] = {}
    eco_dec_err[ss] = {}
    eco_dec_bin[ss] = {}
    for tt in range(len(eco_dec[ss])):
        eco_dec_smf[ss][tt], eco_dec_err[ss][tt], eco_dec_bin[ss][tt] = \
        perc_calcs(eco_dec[ss][tt],bins,dlogM)    

# # Stellar Mass Function

# In[294]:

fig,ax  = plt.subplots(figsize=(8,8))
ax.set_title('Mass Distribution',fontsize=18)
ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
ax.set_ylabel(r'$\log\ (\frac{N_{gal}}{N_{total}*dlogM_{*}})$',fontsize=20)
ax.set_yscale('log')
ax.set_xlim(9.1,11.9)
ax.tick_params(axis='both', labelsize=14)

for ii in range(len(mass_freq)):
    ax.plot(bin_centers,mass_freq[ii][0],color='silver')
    ax.fill_between(bin_centers,max_lim,min_lim,color='silver',alpha=0.1)
ax.errorbar(bin_centers,eco_freq[0],yerr=eco_freq[1],color='maroon',\
            linewidth=2,label='ECO')
ax.legend(loc='best')

plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.94,\
                    hspace=0.2,wspace=0.2)

# plt.savefig(figsave_path + r"\stellar_mass_func")

plt.show()


# # The Mess I am unlovingly referring to as Schechter Functions

# In[432]:

def schechter_real_func(mean_of_mass_bin,phi_star,alpha,Mstar):
    """
    
    mean_of_mass_bin: array-like
        Unlogged x-values
    phi-star: float-like
        Normalization value
    alpha: float-like
        Low-mass end slope
    Mstar: float-like
        Unlogged value where function switches from power-law to exponential
    
    """
#     M_over_mstar = (10**mean_of_mass_bin)/Mstar
    M_over_mstar = (mean_of_mass_bin)/Mstar
    res_arr    = (phi_star) * (M_over_mstar**(alpha)) *\
                             np.exp(- M_over_mstar)

    return res_arr


# In[40]:

def schechter_log_func(stellar_mass,phi_star,alpha,m_star):
    """
    Returns a plottable Schechter function for the 
        stellar mass functions of galaxies
    
    Parameters
    ----------
    stellar_mass: array-like
        An array of unlogged stellar mass values which 
        will eventually be the x-axis values the function
        is plotted against
    phi_star: float-like
        A constant which normalizes (?) the function;
        Moves the graph up and down
    alpha: negative integer-like
        The faint-end, or in this case, low-mass slope;
        Describes the power-law portion of the curve
    m_star: float-like
        Unlogged value of the characteristic (?) stellar
        mass; the "knee" of the function, where the 
        power-law gives way to the exponential portion
        
    Returns
    -------
    res: array-like
        Array of values to be plotted on a log
        scale to display the Schechter function
        
    """
    constant = np.log(10) * phi_star
    log_M_Mstar = np.log10(stellar_mass/m_star)
    res = constant * 10**(log_M_Mstar * (alpha+1)) *\
             np.exp(-10**log_M_Mstar)
        
    return res


# In[41]:

def schech_integral(edge_1,edge_2,phi_star,alpha,Mstar):
    bin_integral = (integrate.quad(schechter_real_func,edge_1,edge_2,\
        args=(phi_star,alpha,Mstar))[0])
#     tot_integral = (integrate.quad(schechter_real_func,9.1,11.7,\
# args=(phi_star,alpha,Mstar)))[0]
# #     
#     result = bin_integral/tot_integral/0.2
    
    return bin_integral


def schech_step_3(xdata,phi_star,alpha,Mstar):
    """
    xdata: array-like
        Unlogged x-values
    Mstar:
        unlogged
    """
    test_int = []
    for ii in range(len(xdata)):
        test_int.append((schech_integral(10**bins_curve_fit[ii],\
            10**bins_curve_fit[ii+1],phi_star,alpha,Mstar)))
    return test_int

# In[44]:

def find_params(bin_int,mean_mass,count_err):
    """
    Parameters
    ----------
    bin_int: array-like
        Integral (number of counts) in each bin of width dlogM
    mean_mass: array-like
        Logged values (?)

    Returns
    -------
    opt_v: array-like
        Array with three values: phi_star, alpha, and M_star
    res_arr: array-like
        Array with two values: alpha and log_M_star


    """
    xdata = 10**mean_mass
#     xdata = mean_mass
    p0    = (1.5,-1.05,10**10.64)
    opt_v,est_cov = optimize.curve_fit(schech_step_3,xdata,\
                            bin_int,p0=p0,sigma=count_err,check_finite=True)
    alpha   = opt_v[1]
    log_m_star    = np.log10(opt_v[2])
    
    res_arr = np.array([alpha,log_m_star])
    
    perr = np.sqrt(np.diag(est_cov))

    return opt_v, res_arr, perr, est_cov

# fig, ax = plt.subplots()
# ax.set_yscale('log')
# ax.set_xscale('log')
# # ax.plot(eco_mass_means[0][:-3],test)
# ax.plot(10**bin_centers,schech_vals_graph)
# ax.plot(10**eco_mass_means[0][:-3],(eco_dec_smf[1][0]))
# plt.show()      


# # Regular Plotting Reintroduced

# In[104]:

def plot_all_rats(bin_centers,y_vals,neigh_val,ax,col_num,plot_idx,text=False):
    """
    Returns a plot showing the density-cut, mass ratio.  Optimally
        used with a well-initiated for-loop
        
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    y_vals: array-like
        An array containing the ratio values for each mass bin
    neigh_val: integer-like
        Value which will be inserted into the text label of each plot
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    col_num: integer-like
        Integer which specifies which column is currently being 
        plotted. Used for labelling subplots
    plot_idx: integer-like
        Specifies which subplot the figure is plotted to. Used for
        labeling the x-axis
        
    Returns
    -------
    Figure with three subplots showing appropriate ratios
    """
    if plot_idx     ==16:
        ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)

    if text == True:
        if col_num      ==0:
            title_label = 'Mass Ratio 50/50, {0} NN'.format(neigh_val)
            frac_val    = 10
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                        verticalalignment='top',transform=ax.transAxes,\
                        fontsize=12)
        elif col_num    ==1:
            title_label = 'Mass Ratio 25/75, {0} NN'.format(neigh_val)
            frac_val    = 4
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                        verticalalignment='top',transform=ax.transAxes,\
                        fontsize=12)
        elif col_num    ==2:
            title_label = 'Mass Ratio 10/90, {0} NN'.format(neigh_val)
            frac_val    = 2
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                        verticalalignment='top',transform=ax.transAxes,\
                        fontsize=12)
    ax.set_xlim(9.1,11.9)
#     ax.set_ylim([0,5])
    ax.set_ylim(0,7)
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.set_yticks([1.,3.])
    ax.tick_params(axis='both', labelsize=12)
    ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
    ax.plot(bin_centers,y_vals,color='silver')


# In[103]:

def plot_eco_rats(bin_centers,y_vals,neigh_val,ax,col_num,plot_idx,only=False):
    """

    Returns subplots of ECO density-cut,mass ratios
    
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    y_vals: array-like
        An array containing the ratio values for each mass bin
    neigh_val: integer-like
        Value which will be inserted into the text label of each plot
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    col_num: integer-like
        Integer which specifies which column is currently being 
        plotted. Used for labelling subplots
    plot_idx: integer-like
        Specifies which subplot the figure is plotted to. Used for
        labeling the x-axis
    
    Optional
    --------
    only == True
        To be used when only plotting the ECO ratios, no mocks.
        Will add in the additional plotting specifications that
        would have been taken care of previously in a for-loop
        which plotted the mocks as well
        
    Returns
    -------
    ECO ratios plotted to any previously initialized figure
    """
    if only == True:
        if plot_idx     ==16:
            ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
        if col_num      ==0:
            title_label = 'Mass Ratio 50/50, {0} NN'.format(neigh_val)
            frac_val    = 10
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        elif col_num    ==1:
            title_label = 'Mass Ratio 25/75, {0} NN'.format(neigh_val)
            frac_val    = 4
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        elif col_num    ==2:
            title_label = 'Mass Ratio 10/90, {0} NN'.format(neigh_val)
            frac_val    = 2
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        ax.set_xlim(9.1,11.9)
        ax.set_ylim(0,7)
#         ax.set_ylim([0,5])
        ax.set_xticks(np.arange(9.5, 12., 0.5))
        ax.set_yticks([1.,3.])
        ax.tick_params(axis='both', labelsize=12)
        ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
    frac_vals = np.array([2,4,10])
    y_vals_2 = y_vals[0][frac_vals[hh]]
    ax.errorbar(bin_centers,y_vals_2,yerr=y_vals[1][hh],\
                    color='maroon',linewidth=2)


# In[83]:

frac_vals   = [2,4,10]
nn_plot_arr = [[[] for yy in xrange(len(nn_mass_dist))] for xx in\
     xrange(len(neigh_vals))]

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        nn_plot_arr[ii][jj] = (ratio_info[jj][neigh_vals[ii]])
        
plot_frac_arr = [[[[] for yy in xrange(len(nn_mass_dist))]\
                  for zz in xrange(len(frac_vals))] for xx in\
                                    xrange(len(nn_plot_arr))]
frac_err_arr = [[[[] for yy in xrange(len(nn_mass_dist))]\
                  for zz in xrange(len(frac_vals))] for xx in\
                                    xrange(len(nn_plot_arr))]

for jj in range(len(nn_mass_dist)):
    for hh in range(len(frac_vals)):
        for ii in range(len(neigh_vals)):
            plot_frac_arr[ii][hh][jj] = nn_plot_arr[ii][jj][0][frac_vals[hh]]
            frac_err_arr[ii][hh][jj] = nn_plot_arr[ii][jj][1][hh]


# In[105]:

np.seterr(divide='ignore',invalid='ignore')

nrow_num = int(6)
ncol_num = int(3)
zz       = int(0)

fig, axes = plt.subplots(nrows=nrow_num, ncols=ncol_num,\
             figsize=(100,200), sharex= True,sharey=True)
axes_flat = axes.flatten()


fig.text(0.01, 0.5, 'High Density Counts/Lower Density Counts', ha='center',\
     va='center',rotation='vertical',fontsize=20)
# fig.suptitle("Percentile Trends", fontsize=18)

while zz <= 16:
    for ii in range(len(eco_ratio_info)):
        for hh in range(len(eco_ratio_info[0][1])):
            for jj in range(len(nn_mass_dist)):
                if jj == 0:
                    value = True
                else:
                    value = False
                plot_all_rats(bin_cens_diff[jj][neigh_vals[ii]][hh],\
                    (plot_frac_arr[ii][hh][jj]),\
                    neigh_vals[ii],axes_flat[zz],hh,zz,text=value)
            plot_eco_rats(eco_final_bins[ii][hh],(eco_ratio_info[ii]),\
                neigh_vals[ii],axes_flat[zz],hh,zz)
            zz += 1

plt.subplots_adjust(left=0.04, bottom=0.09, right=0.98, top=0.98,\
                    hspace=0,wspace=0)

# plt.savefig(figsave_path + r"\ratios")

plt.show()


# In[145]:

def plot_hists(bins_high,bins_low,high_counts,low_counts,\
               neigh_val,ax,col_num,plot_idx,text=False):
    """
    Returns a plot showing the density-cut, mass counts.
    
    Parameters
    ----------
    mass: array-like
        A 1D array with log stellar mass values
    neigh_val: integer-like
        Value which will be inserted into the text label of each plot
    bins: array-like
        A 1D array with the values which will be used as the bin edges   
    dlogM: float-like
        The log difference between bin edges
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    col_num: integer-like
        Integer which specifies which column is currently being 
        plotted. Used for labelling subplots
    plot_idx: integer-like
        Specifies which subplot the figure is plotted to. Used for
        labeling the x-axis
        
    Returns
    -------
    Figure with two curves, optionally (if uncommented) plotted in step
    
    """
    ax.set_yscale('log')
    if text == True:
        if col_num==0:
            title_label = 'Mass 50/50, {0} NN'.format(neigh_val)
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        elif col_num==1:
            title_label = 'Mass 25/75, {0} NN'.format(neigh_val)
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        elif col_num==2:
            title_label = 'Mass 10/90, {0} NN'.format(neigh_val)
            ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=12)
        
    if plot_idx == 16:
        ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)                      
    ax.set_xlim(9.1,11.9)
    ax.set_ylim([10**-3,10**1])
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.set_yticks([10**-2,10**0])
    
    ax.plot(bins_high,high_counts,color = 'lightslategrey',alpha=0.2)
    ax.plot(bins_low,low_counts,color = 'lightslategray',alpha=0.2)


def plot_eco_hists(bins_high,bins_low,high_counts,low_counts,               \
    frac_val,ax,plot_idx):
        err_key = 'err_{0}'.format(frac_val)
        
        ax.errorbar(bins_high[frac_val],high_counts[frac_val],\
                            yerr=high_counts[err_key],drawstyle='steps-mid',\
                                    color='royalblue',label='Higher Density')
        ax.errorbar(bins_low[frac_val],low_counts[frac_val],\
                            yerr=low_counts[err_key],drawstyle='steps-mid',\
                                        color='crimson',label='Lower Density')
        
        if plot_idx == 0:
            ax.legend(loc='best')


# In[147]:

nrow_num  = int(6)
ncol_num  = int(3)

frac_dict = {2:0,4:1,10:2}

fig, axes = plt.subplots(nrows=nrow_num, ncols=ncol_num,\
             figsize=(150,200), sharex= True,sharey=True)
axes_flat = axes.flatten()

fig.text(0.02, 0.5,r'$\log\ (\frac{N_{gal}}{N_{total}*dlogM_{*}})$', \
    ha='center',va='center',rotation='vertical',fontsize=20)


for ii in range(len(mass_dat)):
    zz = 0
    for jj in range(len(neigh_vals)):
        for hh in frac_vals:
            if ii == 0:
                value = True
            else:
                value = False
            plot_hists(hist_high_bins[ii][neigh_vals[jj]][hh],\
                            hist_low_bins[ii][neigh_vals[jj]][hh],\
                            hist_high_info[ii][neigh_vals[jj]][hh],\
                            hist_low_info[ii][neigh_vals[jj]][hh],\
                            neigh_vals[jj],axes_flat[zz],frac_dict[hh],zz,\
                            text=value)
            if ii == 0:
                plot_eco_hists(eco_high_bins[neigh_vals[jj]],\
                   eco_low_bins[neigh_vals[jj]],eco_high[neigh_vals[jj]],\
                       eco_low[neigh_vals[jj]],hh,axes_flat[zz],zz)
            zz += int(1)         

plt.subplots_adjust(left=0.07, bottom=0.09, right=0.98, top=0.98,\
                    hspace=0, wspace=0)  


# plt.savefig(figsave_path + r"\histograms")                              

plt.show()


# In[372]:

def plot_all_meds(bin_centers,y_vals,ax,plot_idx,text=False):
    """
    Returns six subplots showing the median distance to 
        the Nth nearest neighbor for each mass bin.  Assumes a
        previously defined figure.  Best used in a for-loop
    
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    y_vals: array-like
        An array containing the median distance values for each mass bin
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    plot_idx: integer-like
        Specifies which subplot the figure is plotted to. Used for
        the text label in each subplot
        
    Returns
    -------
    Subplots displaying the median distance to Nth nearest neighbor
    trends for each mass bin
    
    """
    titles = [1,2,3,5,10,20]
    ax.set_ylim(0,10**1.5)
    ax.set_xlim(9.1,11.9)
    ax.set_yscale('symlog')
    ax.set_xticks(np.arange(9.5,12.,0.5))  
    ax.set_yticks(np.arange(0,12,1))  
    ax.set_yticklabels(np.arange(1,11,2))
    ax.tick_params(axis='x', which='major', labelsize=16)
    if text == True:
        title_here = 'n = {0}'.format(titles[plot_idx])
        ax.text(0.05, 0.95, title_here,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=18)
    if plot_idx == 4:
        ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=20)
    ax.plot(bin_centers,y_vals,color='silver')


# In[153]:

def plot_eco_meds(bin_centers,y_vals,low_lim,up_lim,ax,plot_idx,only=False):
    """
    Returns six subplots showing the median Nth nearest neighbor distance for 
        ECO galaxies in each mass bin
        
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    y_vals: array-like
        An array containing the median distance values for each mass bin
    low_lim: array-like
        An array with the lower cut-off of the bootstrap errors for each median
    up_lim: array-like
        An array with the upper cut-off of the bootstrap errors for each median
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    plot_idx: integer-like
        Specifies which subplot the figure is plotted to. Used for
        the text label in each subplot
    
    Optional
    --------
    only == False
        To be used when only plotting the ECO median trends, 
        no mocks.  Will add in the additional plotting 
        specifications that would have been taken care of 
        previously in a for-loop which plotted the mocks as well
    
    Returns
    -------
    Subplots displaying the median distance to Nth nearest neighbor
    trends for each mass bin, with the bootstrap errors
    
    """
    if only == True:
        titles = [1,2,3,5,10,20]
        ax.set_ylim(0,10**1.5)
        ax.set_xlim(9.1,11.9)
        ax.set_yscale('symlog')
        ax.set_xticks(np.arange(9.5,12.,0.5))  
        ax.tick_params(axis='both', which='major', labelsize=16)
        title_here = 'n = {0}'.format(titles[plot_idx])
        ax.text(0.05, 0.95, title_here,horizontalalignment='left',\
                    verticalalignment='top',transform=ax.transAxes,fontsize=18)
        if plot_idx == 4:
            ax.set_xlabel('$\log\ (M_{*}/M_{\odot})$',fontsize=18)
    ax.errorbar(bin_centers,y_vals,yerr=0.1,lolims=low_lim,        \
        uplims=up_lim,color='maroon',label='ECO')
    # if plot_idx == 5:
    #     ax.legend(loc='best')


# In[378]:

nrow_num_mass = int(2)
ncol_num_mass = int(3)

fig, axes = plt.subplots(nrows=nrow_num_mass, ncols=ncol_num_mass,         \
    figsize=(100,200), sharex= True, sharey = True)
axes_flat = axes.flatten()
fig.text(0.01, 0.5, 'Distance to Nth Neighbor (Mpc)', ha='center',     \
    va='center',rotation='vertical',fontsize=20)

zz = int(0)
while zz <=4:
    for ii in range(len(med_plot_arr)):
        for vv in range(len(nn_mass_dist)):
            if vv == 0:
                value = True
            else:
                value = False
            plot_all_meds(mock_meds_bins[vv][neigh_vals[ii]],\
                med_plot_arr[neigh_vals[ii]][vv],axes_flat[zz],\
                                zz,text=value)
            plot_eco_meds(eco_med_bins[ii],eco_medians[ii][0],\
                                      eco_medians[ii][1],eco_medians[ii][2],\
                                                            axes_flat[zz],zz)
        zz   += 1
        
plt.subplots_adjust(left=0.05, bottom=0.09, right=0.98, top=0.98,\
            hspace=0,wspace=0)

# plt.savefig(figsave_path + r"\median_distances")

plt.show() 

