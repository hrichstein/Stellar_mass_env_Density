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



pickle_out = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
pickle_out+= r"\Pickle_output"
iter_num = 2

if iter_num == 0:
    ##original Behroozi
    dirpath = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
    dirpath+= r"\Catalogs\Resolve_plk_5001_so_mvir_scatter_ECO_Mocks_scatter_mocks"
    dirpath+= r"\Resolve_plk_5001_so_mvir_scatter0p2_ECO_Mocks"
elif iter_num == 1:
    ## M1 change
    ##changed from no_ab to ab
    dirpath = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
    dirpath+= r"\Catalogs\Beta_M1_Behroozi\ab_matching"
    dirpath+= r"\Resolve_plk_5001_so_mvir_hod1_scatter0p2_mock1_ECO_Mocks"
elif iter_num == 2:
    ##Beta change
    ##changed from no_ab to ab
    dirpath = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
    dirpath+= r"\Catalogs\Beta_M1_Behroozi\ab_matching"
    dirpath+= r"\Resolve_plk_5001_so_mvir_hod1_scatter0p2_mock2_ECO_Mocks"

# In[2]:

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


# In[3]:

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


# In[4]:

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
    dens = np.array([(3.*(n_val+1)/(4.*np.pi*r_val[hh]**3))                      for hh in range(len(r_val))])

    return dens


# In[5]:

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
        
        temp_yerr = (counts_2*1.)/(counts*1.)*            np.sqrt(1./counts + 1./counts_2)
            
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


# In[6]:

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
    
    medians  = np.array([np.nanmedian(mass_dist.T[kk][digitized==ii])                 for ii in bin_nums])

    # print len(medians)

    if bootstrap == True:
        dist_in_bin    = np.array([(mass_dist.T[kk][digitized==ii])                 for ii in bin_nums])
        for vv in range(len(dist_in_bin)):
            if len(dist_in_bin[vv]) == 0:
#                 dist_in_bin_list = list(dist_in_bin[vv])
#                 dist_in_bin[vv] = np.zeros(len(dist_in_bin[0]))
                dist_in_bin[vv] = np.nan
        low_err_test   = np.array([np.percentile(astropy.stats.bootstrap                        (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),16)                         for vv in range(len(dist_in_bin))])
        high_err_test  = np.array([np.percentile(astropy.stats.bootstrap                        (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),84)                         for vv in range(len(dist_in_bin))])

        med_list    = [[] for yy in range(4)]
        med_list[0] = medians
        med_list[1] = low_err_test
        med_list[2] = high_err_test
        medians     = np.array(med_list)
        
#     print len(medians)
#     print len(non_zero_bins)

    return medians, np.array(non_zero_bins)
    


# In[7]:

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
    frac_val  = np.array([2,4,10])
    frac_dict = {2:0,4:1,10:2}
    
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
        
        high_err = np.sqrt(counts_2)/len(frac_mass_2)/dlogM
        high_err_1 = high_err[non_zero]
        hist_dict_high[err_key] = high_err_1
    
    return hist_dict_low, hist_dict_high


# In[8]:

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



    mean_mass = np.array([np.nanmean(mass_dist.T[0][digitized==ii])                 for ii in bin_nums])

    return mean_mass 

###note, this should use the same bin_centers as provided by the 
#median from bin_func


###############################################################################

#########obsolete because I changed the hist_calcs function to calculate error
def eco_hist_calcs(mass,bins,dlogM):
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


###############################################################################
###############################################################################
###############################################################################
eco_path  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density"
eco_path += r"\Catalogs\ECO_true"

##ra, dec,cz,absrmag,logMstar,group identifier,fc (cent/sat),logMh (halo)
##2: logmh, 4:dec, 10:fc, 15: group, 16: absrmag, 19:cz. 20:ra, 21: logmstar


eco_cols  = np.array([2,4,10,15,16,19,20,21])


# In[282]:

# ECO_true = (Index(eco_path,'.txt'))
ECO_true = (Index(eco_path,'.csv'))
names    = ['logMhalo','dec','cent_sat','group_ID','Mr','cz','ra','logMstar']
PD_eco   = pd.read_csv(ECO_true[0], usecols=(eco_cols),header=None, \
               skiprows=1,names=names)
eco_comp = PD_eco[PD_eco.logMstar >= 9.1]

ra_eco   = eco_comp.ra
dec_eco  = eco_comp.dec
cz_eco   = eco_comp.cz
mass_eco = eco_comp.logMstar
logMhalo = eco_comp.logMhalo
cent_sat = eco_comp.cent_sat
group_ID = eco_comp.group_ID
Mr_eco   = eco_comp.Mr
###############################################################################

usecols  = (0,1,2,4,13)
dlogM    = 0.2
neigh_dict = {1:0,2:1,3:2,5:3,10:4,20:5}


# In[10]:

ECO_cats = (Index(dirpath,'.dat'))

names    = ['ra','dec','cz','Halo_ID','logMstar']

PD = [[] for ii in range(len(ECO_cats))]

for ii in range(len(ECO_cats)):
    temp_PD = (pd.read_csv(ECO_cats[ii],sep="\s+", usecols= usecols,header=None,                   skiprows=2,names=names))
    col_list = list(temp_PD)
    col_list[2], col_list[3], col_list[4] = col_list[3], col_list[4], col_list[2]
    temp_PD.ix[:,col_list]
    PD[ii] = temp_PD

PD_comp_1  = [(PD[ii][PD[ii].logMstar >= 9.1]) for ii in range(len(ECO_cats))]
PD_comp  = [(PD_comp_1[ii][PD_comp_1[ii].logMstar <=11.77]) for ii in range(len(ECO_cats))]

[(PD_comp[ii].reset_index(drop=True,inplace=True)) for ii in range(len(ECO_cats))]


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

num_of_bins = int(len(bins) - 1) 

ra_arr  = np.array([(PD_comp[ii].ra)     for ii in range(len(PD_comp))])

dec_arr  = np.array([(PD_comp[ii].dec)     for ii in range(len(PD_comp))])

cz_arr  = np.array([(PD_comp[ii].cz)     for ii in range(len(PD_comp))])

mass_arr  = np.array([(PD_comp[ii].logMstar)     for ii in range(len(PD_comp))])

halo_id_arr  = np.array([(PD_comp[ii].Halo_ID)     for ii in range(len(PD_comp))])


coords_test = np.array([sph_to_cart(ra_arr[vv],dec_arr[vv],cz_arr[vv])                 for vv in range(len(ECO_cats))])

neigh_vals  = np.array([1,2,3,5,10,20])

nn_arr_temp = [[] for uu in xrange(len(coords_test))]
nn_arr      = [[] for xx in xrange(len(coords_test))]
nn_arr_nn   = [[] for yy in xrange(len(neigh_vals))]
nn_idx      = [[] for zz in xrange(len(coords_test))]

for vv in range(len(coords_test)):
    nn_arr_temp[vv] = spatial.cKDTree(coords_test[vv])
    nn_arr[vv] = np.array(nn_arr_temp[vv].query(coords_test[vv],21)[0])
    nn_idx[vv] = np.array(nn_arr_temp[vv].query(coords_test[vv],21)[1])
    

nn_specs       = [(np.array(nn_arr).T[ii].T[neigh_vals].T) for ii in                     range(len(coords_test))]
nn_mass_dist   = np.array([(np.column_stack((mass_arr[qq],nn_specs[qq])))                     for qq in range(len(coords_test))])

nn_neigh_idx      = np.array([(np.array(nn_idx).T[ii].T[neigh_vals].T) for ii in                     range(len(coords_test))])

truth_vals = {}
for ii in range(len(halo_id_arr)):
    truth_vals[ii] = {}
    for jj in neigh_vals:
        halo_id_neigh = halo_id_arr[ii][nn_neigh_idx[ii].T[neigh_dict[jj]]].values
        truth_vals[ii][jj] = halo_id_neigh==halo_id_arr[ii].values


# In[15]:

halo_frac = {}
for ii in range(len(mass_arr)):
    halo_frac[ii] = {}
    mass_binning = np.digitize(mass_arr[ii],bins)
    bins_to_use = list(np.unique(mass_binning))
    if (len(bins)-1) not in bins_to_use:
        bins_to_use.append(len(bins)-1)
    if len(bins) in bins_to_use:
        bins_to_use.remove(len(bins))
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
        halo_frac[ii][jj] = frac


# In[16]:

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
    oo_tot_std  = [np.nanstd(oo_tot[uu])/np.sqrt(len(halo_frac)) for uu in xrange(len(oo_tot))]
    mean_mock_halo_frac[bin_str] = [oo_tot_mean,oo_tot_std]

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
        nn_dens[ii][(neigh_vals[jj])]  = np.column_stack((nn_mass_dist[ii].T                                            [0],calc_dens(neigh_vals[jj],                                            nn_mass_dist[ii].T[range(1,len                                                (neigh_vals)+1)[jj]])))

        idx = np.array([nn_dens[ii][neigh_vals[jj]].T[1].argsort()])
        mass_dat[ii][(neigh_vals[jj])] = (nn_dens[ii][neigh_vals[jj]]                                            [idx].T[0])

        bin_centers, mass_freq[ii], ratio_info[ii][neigh_vals[jj]],bin_cens_diff[ii][neigh_vals[jj]] =                             plot_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)

# pickle_out_smf = pickle_out
# pickle_out_smf+=r"\mock_smfs.p"

# if iter_num == 0:
#     mock_smf_data = [mass_freq]
#     pickle.dump(mock_smf_data, open(pickle_out_smf, "wb"))
# else:
#     mock_smf_data_new = pickle.load(open(pickle_out_smf, "rb"))
#     mock_smf_data_new.append(mass_freq)
#     pickle.dump(mock_smf_data_new,open(pickle_out_smf,"wb"))


###############################################################################        

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

med_plot_arr = {}

for ii in range(len(neigh_vals)):
    med_plot_arr[neigh_vals[ii]] = {}
    for jj in range(len(nn_mass_dist)):
        med_plot_arr[neigh_vals[ii]][jj] = all_mock_meds[jj][neigh_vals[ii]]

#########For finding bands for ratios
nn_dict   = {1:0,2:1,3:2,5:3,10:4,20:5}
coln_dict = {2:0,4:1,10:2}



frac_vals   = [2,4,10]
nn_plot_arr = {}

for ii in range(len(neigh_vals)):
    nn_plot_arr[neigh_vals[ii]] = {}
    for jj in range(len(nn_mass_dist)):
        nn_plot_arr[neigh_vals[ii]][jj] = (ratio_info[jj][neigh_vals[ii]])

plot_frac_arr = {}



for ii in (neigh_vals):
    plot_frac_arr[ii] = {}
    for hh in (frac_vals):
        plot_frac_arr[ii][hh] = {}
        for jj in range(len(nn_mass_dist)):
            plot_frac_arr[ii][hh][jj] = \
                nn_plot_arr[ii][jj][0][hh]


A = {}


nn_keys  = np.sort(nn_dict.keys())
col_keys = np.sort(coln_dict.keys())
zz_num   = len(plot_frac_arr[1][2])

for nn in neigh_vals:
    for vv in frac_vals:
        bin_str    = '{0}_{1}'.format(nn,vv)
        for cc in range(zz_num):
            zz_arr = plot_frac_arr[nn][vv][cc]
            if len(zz_arr) == num_of_bins:
                n_elem = len(zz_arr)
            else:
                while len(zz_arr) < num_of_bins:
                    zz_arr_list = list(zz_arr)
                    zz_arr_list.append(np.nan)
                    zz_arr = np.array(zz_arr_list)
                n_elem = len(zz_arr)
            if cc == 0:
                zz_tot = np.zeros((n_elem,1))
            zz_tot = np.insert(zz_tot,len(zz_tot.T),zz_arr,1)
        zz_tot = np.array(np.delete(zz_tot,0,axis=1))
        for kk in xrange(len(zz_tot)):
            zz_tot[kk][zz_tot[kk] == np.inf] = np.nan
        zz_tot_max = [np.nanmax(zz_tot[kk]) for kk in xrange(len(zz_tot))]
        zz_tot_min = [np.nanmin(zz_tot[kk]) for kk in xrange(len(zz_tot))]
        A[bin_str] = [zz_tot_max,zz_tot_min]     

pickle_out_rats = pickle_out
pickle_out_rats+=r"\ratio_bands.p"

if iter_num == 0:
    rat_band_data = [A]
    pickle.dump(rat_band_data, open(pickle_out_rats, "wb"))
else:
    rat_band_data_new = pickle.load(open(pickle_out_rats, "rb"))
    rat_band_data_new.append(A)
    pickle.dump(rat_band_data_new,open(pickle_out_rats,"wb"))



### dict A now houses the upper and lower limits needed for the bands
#####Bands for stellar mass function    
 

#Bands for median distances

B = {}
yy_num = len(med_plot_arr[neigh_vals[0]])

for nn in neigh_vals:
    for ii in range(yy_num):
        med_str  = '{0}'.format(nn)
        yy_arr   = med_plot_arr[nn][ii]
        if len(yy_arr) == num_of_bins:
            n_y_elem = len(yy_arr)
        else:
            while len(yy_arr) < num_of_bins:
                yy_arr_list = list(yy_arr)
                yy_arr_list.append(np.nan)
                yy_arr = np.array(yy_arr_list)
            n_y_elem = len(yy_arr)
        if ii == 0:
            yy_tot = np.zeros((n_y_elem,1))
        yy_tot = np.insert(yy_tot,len(yy_tot.T),yy_arr,1)
    yy_tot = np.array(np.delete(yy_tot,0,axis=1))
    yy_tot_max = [np.nanmax(yy_tot[kk]) for kk in xrange(len(yy_tot))]
    yy_tot_min = [np.nanmin(yy_tot[kk]) for kk in xrange(len(yy_tot))]
    B[med_str] = [yy_tot_max,yy_tot_min]

pickle_out_meds = pickle_out
pickle_out_meds+=r"\med_bands.p"

if iter_num == 0:
    med_band_data = [B]
    pickle.dump(med_band_data, open(pickle_out_meds, "wb"))
else:
    med_band_data_new = pickle.load(open(pickle_out_meds, "rb"))
    med_band_data_new.append(B)
    pickle.dump(med_band_data_new,open(pickle_out_meds,"wb"))    


global bins_curve_fit

bins_curve_fit = bins.copy()


# In[28]:

# eco_path  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density"
# eco_path += r"\Catalogs\ECO_true"
# eco_cols  = np.array([0,1,2,4])


# # In[29]:

# ECO_true = (Index(eco_path,'.txt'))
# names    = ['ra','dec','cz','logMstar']
# PD_eco   = pd.read_csv(ECO_true[0],sep="\s+", usecols=(eco_cols),header=None,                skiprows=1,names=names)
# eco_comp = PD_eco[PD_eco.logMstar >= 9.1]

# ra_eco   = (np.array(eco_comp)).T[0]
# dec_eco  = (np.array(eco_comp)).T[1] 
# cz_eco   = (np.array(eco_comp)).T[2] 
# mass_eco = (np.array(eco_comp)).T[3]

coords_eco        = sph_to_cart(ra_eco,dec_eco,cz_eco)
eco_neighbor_tree = spatial.cKDTree(coords_eco)
eco_tree_dist     = np.array(eco_neighbor_tree.query(coords_eco,                    (neigh_vals[-1]+1))[0])

eco_mass_dist = np.column_stack((mass_eco,eco_tree_dist.T[neigh_vals].T))
##range 1,7 because of the six nearest neighbors (and fact that 0 is mass)
##the jj is there to specify which index in the [1,6] array
eco_dens = ([calc_dens(neigh_vals[jj],            (eco_mass_dist.T[range(1,7)[jj]])) for jj in range            (len(neigh_vals))])

eco_mass_dens = [(np.column_stack((mass_eco,eco_dens[ii]))) for ii in                 range(len(neigh_vals))]
eco_idx  = [(eco_mass_dens[jj].T[1].argsort()) for jj in             range(len(neigh_vals))]
eco_mass_dat  = [(eco_mass_dens[jj][eco_idx[jj]].T[0]) for jj in                 range(len(neigh_vals))]

eco_ratio_info    = [[] for xx in xrange(len(eco_mass_dat))]
eco_final_bins    = [[] for xx in xrange(len(eco_mass_dat))]


for qq in range(len(eco_mass_dat)):
    bin_centers, eco_freq, eco_ratio_info[qq],eco_final_bins[qq] = plot_calcs(eco_mass_dat[qq],                                    bins,dlogM)

# pickle_out_eco_smf = pickle_out
# pickle_out_eco_smf+=r"\eco_smf.p"

# if iter_num == 0:
#     eco_smf_data = [eco_freq]
#     pickle.dump(eco_smf_data, open(pickle_out_eco_smf, "wb"))

eco_medians   = [[] for xx in xrange(len(eco_mass_dat))]    
eco_med_bins   = [[] for xx in xrange(len(eco_mass_dat))]    
eco_mass_means   = [[] for xx in xrange(len(eco_mass_dat))] 

for jj in (range(len(eco_mass_dat))):
    eco_medians[jj],eco_med_bins[jj] = np.array(bin_func(eco_mass_dist,bins,(jj+1),        bootstrap=True))
    eco_mass_means[jj] = (mean_bin_mass(eco_mass_dist,bins,(jj+1))) 

###ECO hists

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
              eco_hist_calcs(eco_mass_dat[jj],bins,dlogM)
###############################################################################            

pickle_out_eco = pickle_out
pickle_out_eco+=r"\eco_hists.p"

if iter_num == 0:
    eco_bins_data = [eco_low,eco_high,eco_low_bins,eco_high_bins]
    pickle.dump(eco_bins_data, open(pickle_out_eco, "wb"))


pickle_out_eco_1 = pickle_out
pickle_out_eco_1+=r"\eco_data.p"


if iter_num == 0:
    eco_band_data = [eco_low,eco_high,eco_ratio_info, eco_final_bins,eco_medians]
    pickle.dump(eco_band_data, open(pickle_out_eco_1, "wb"))

else:
    eco_band_data_new = pickle.load(open(pickle_out_eco, "rb"))
    eco_band_data_new.append([])
    pickle.dump(eco_band_data_new,open(pickle_out_eco,"wb"))   




#########Histograms

hist_low_info  = {}
hist_high_info = {}

for ii in xrange(len(coords_test)):
    hist_low_info[ii]  = {}
    hist_high_info[ii] = {}
    
    for jj in range(len(neigh_vals)):
        hist_low_info[ii][neigh_vals[jj]],hist_high_info[ii][neigh_vals[jj]] \
                = hist_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)
        
frac_vals     = [2,4,10]
hist_low_arr  = [[[] for yy in xrange(len(nn_mass_dist))] for xx in     \
xrange(len(neigh_vals))]
hist_high_arr = [[[] for yy in xrange(len(nn_mass_dist))] for xx in     \
xrange(len(neigh_vals))]

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        hist_low_arr[ii][jj]  = (hist_low_info[jj][neigh_vals[ii]])
        hist_high_arr[ii][jj] = (hist_high_info[jj][neigh_vals[ii]])

plot_low_hist = {}
plot_high_hist = {}


for ii in range(len(neigh_vals)):
    plot_low_hist[ii] = {}
    plot_high_hist[ii] = {}
    for hh in range(len(frac_vals)):
        plot_low_hist[ii][hh] = {}
        plot_high_hist[ii][hh] = {}
        for jj in range(len(nn_mass_dist)):
            plot_low_hist[ii][hh][jj]  = hist_low_arr[ii][jj][frac_vals[hh]]        
            plot_high_hist[ii][hh][jj] = hist_high_arr[ii][jj][frac_vals[hh]]  

###Histogram bands                  
C = {}
D = {}
nn_dict   = {1:0,2:1,3:2,5:3,10:4,20:5}
coln_dict = {2:0,4:1,10:2}

nn_keys   = np.sort(nn_dict.keys())
col_keys  = np.sort(coln_dict.keys())

vv_num    = len(plot_low_hist[nn_dict[1]][coln_dict[2]])

for nn in nn_keys:
    for coln in col_keys:
        bin_str    = '{0}_{1}'.format(nn,coln)
        for cc in range(vv_num):
            vv_arr = np.array(plot_low_hist[nn_dict[nn]][coln_dict[coln]][cc])
            if len(vv_arr) == num_of_bins:
                n_elem = len(vv_arr)
            else:
                while len(vv_arr) < num_of_bins:
                    vv_arr_list = list(vv_arr)
                    vv_arr_list.append(np.nan)
                    vv_arr = np.array(vv_arr_list)
                n_elem = len(vv_arr)
            if cc == 0:
                vv_tot = np.zeros((n_elem,1))
            vv_tot = np.insert(vv_tot,len(vv_tot.T),vv_arr,1)
        vv_tot = np.array(np.delete(vv_tot,0,axis=1))
        for kk in xrange(len(vv_tot)):
            vv_tot[kk][vv_tot[kk] == np.inf] = np.nan
        vv_tot_max = [np.nanmax(vv_tot[kk]) for kk in xrange(len(vv_tot))]
        vv_tot_min = [np.nanmin(vv_tot[kk]) for kk in xrange(len(vv_tot))]
        C[bin_str] = [vv_tot_max,vv_tot_min]
        
hh_num   = len(plot_high_hist[nn_dict[1]][coln_dict[2]])

for nn in nn_keys:
    for coln in col_keys:
        bin_str    = '{0}_{1}'.format(nn,coln)
        for cc in range(hh_num):
            hh_arr = np.array(plot_high_hist[nn_dict[nn]][coln_dict[coln]][cc])
            if len(hh_arr) == num_of_bins:
                n_elem = len(hh_arr)
            else:
                while len(hh_arr) < num_of_bins:
                    vv_arr_list = list(hh_arr)
                    vv_arr_list.append(np.nan)
                    hh_arr = np.array(vv_arr_list)
                n_elem = len(hh_arr)
            if cc == 0:
                hh_tot = np.zeros((n_elem,1))
            hh_tot = np.insert(hh_tot,len(hh_tot.T),hh_arr,1)
        hh_tot     = np.array(np.delete(hh_tot,0,axis=1))
        for kk in xrange(len(hh_tot)):
            hh_tot[kk][hh_tot[kk] == np.inf] = np.nan
        hh_tot_max = [np.nanmax(hh_tot[kk]) for kk in xrange(len(hh_tot))]
        hh_tot_min = [np.nanmin(hh_tot[kk]) for kk in xrange(len(hh_tot))]
        D[bin_str] = [hh_tot_max,hh_tot_min]        

pickle_out_hists = pickle_out
pickle_out_hists+=r"\hist_bands.p"

if iter_num == 0:
    hist_band_data = [C,D]
    pickle.dump(hist_band_data, open(pickle_out_hists, "wb"))
else:
    hist_band_data_new = pickle.load(open(pickle_out_hists, "rb"))
    hist_band_data_new.append(C)
    hist_band_data_new.append(D)
    pickle.dump(hist_band_data_new,open(pickle_out_hists,"wb"))        