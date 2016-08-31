##technically, this means that I don't have to force floats in my division
from __future__ import division, absolute_import

##this makes the plot lines thicker, darker, etc.
from matplotlib import rc,rcParams
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')

##importing the needed modules
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

##for plotting 3D
from mpl_toolkits.mplot3d import Axes3D

##making global-like text sizes
class Vars(object):
    size_xlabel = 22
    size_ylabel = 22
    size_text   = 20
    size_tick   = 20
    size_legend = 20   

va = Vars()

##Victor's functions
##used for importing the needed data files
##used for creating bins for histograms, etc. automatically (not hard coding)

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

###############################################################################

def sph_to_cart(ra,dec,cz,h):
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
    h: float-like
        the Hubble constant; 70 for experimental, 100 for theoretical

    Returns
    -------
    coords: array-like, shape = N by 3
        x, y, and z coordinates
    """
    cz_dist = cz/float(h) #converts velocity into distance
    x_arr   = cz_dist*np.cos(np.radians(ra))*np.cos(np.radians(dec))
    y_arr   = cz_dist*np.sin(np.radians(ra))*np.cos(np.radians(dec))
    z_arr   = cz_dist*np.sin(np.radians(dec))
    coords  = np.column_stack((x_arr,y_arr,z_arr))

    return coords

###############################################################################

##actually a rather obsolete function, as I could just sort by distance, since
##N is the same for all cases
##Need something like this for Density in a Sphere method

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

###############################################################################

def plot_calcs(mass,bins,dlogM):
    """
    Returns values for plotting the stellar mass function and 
        mass ratios

    Parameters
    ----------
    mass: array-like
        A 1D array with mass values, assumed to be in order based on the 
        density of the environment
    bins: array-like
        A 1D array with the values which will be used as the bin edges
        by the histogram function
    dlogM: float-like
        The log difference between bin edges

    Returns
    -------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    mass_freq: array-like
        Contains the number density values of each mass bin
    ratio_dict: dictionary-like
        A dictionary with three keys, corresponding to the divisors
        2,4, and 10 (as the percentile cuts are based on these 
        divisions). Each key has the density-cut, mass ratios for
        that specific cut (50/50 for 2; 25/75 for 4; 10/90 for 10).
    bin_centers_fin: array-like
        An array with the mean mass values of the galaxies in each mass bin
    """

    mass_counts, edges = np.histogram(mass,bins)
    bin_centers        = 0.5*(edges[:-1]+edges[1:])

    mass_freq  = mass_counts/float(len(mass))/dlogM


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

###############################################################################

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
    bins: array-like
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
    np.array(non_zero_bins): array-like
        An array with the medians of the non-empty bins; so these would be the 
        x-coords at which to plot the medians

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

###############################################################################

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
        as values. The values for the lower density cut; also has three error
        keys
    hist_dict_high: dictionary-like
        A dictionary with three keys (the frac vals), with arrays
        as values. The values for the higher density cut; also has three error 
        keys
    """
    hist_dict_low  = {}
    hist_dict_high = {}
    bin_cens_low   = {}
    bin_cens_high  = {}
    frac_val  = np.array([2,4,10])

    ##given the fractional value, returns the index
    frac_dict = {2:0,4:1,10:2}
    
    edges = bins
    
    bin_centers = 0.5 * (edges[:-1]+edges[1:])
    
    low_err   = [[] for xx in xrange(len(frac_val))]
    high_err  = [[] for xx in xrange(len(frac_val))]
    
    for ii in frac_val:
    
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
        high_counts        = counts_2/float(len(frac_mass_2))/dlogM
        
        non_zero = (high_counts!=0)
        high_counts_1 = high_counts[non_zero]
        hist_dict_high[ii] = high_counts_1
        bin_cens_high[ii]  = bin_centers[non_zero]
        
        high_err = np.sqrt(counts_2)/len(frac_mass_2)/dlogM
        high_err_1 = high_err[non_zero]
        hist_dict_high[err_key] = high_err_1
    
    return hist_dict_low, hist_dict_high, bin_cens_low, bin_cens_high

###############################################################################

def mean_bin_mass(mass_dist,bins,kk):
    """
    Returns mean mass of galaxies in each bin; similar operation done in 
    plot_calcs, but this is for more general use

    Parameters
    ----------
    mass_dist: array-like
        An array with mass values in at index 0 (when transformed) 
    bins: array-like
        A 1D array with the values which will be used as the bin edges
    kk: integer-like
        The index of mass_dist (transformed) where the appropriate distance 
        array may be found

    Returns
    -------
    An array of the mean mass of the galaxies in each mass bin

    """
    edges        = bins

    digitized    = np.digitize(mass_dist.T[0],edges)
    digitized   -= int(1)
    
    bin_nums          = np.unique(digitized)
    
    for ii in bin_nums:
        if len(mass_dist.T[kk][digitized==ii]) == 0:
            mass_dist.T[kk][digitized==ii] = np.nan

    mean_mass = np.array([np.nanmean(mass_dist.T[0][digitized==ii])                 
        for ii in bin_nums])

    return mean_mass 

###note, this should use the same bin_centers (for x-coord) as provided by the 
#median from bin_func


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

##Importing the ECO Data!!!

eco_path  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density"
eco_path += r"\Catalogs\ECO_true"

eco_cols  = np.array([2,4,10,15,16,19,20,21])

##the newest file from the online database
ECO_true = (Index(eco_path,'.csv'))
names    = ['logMhalo','dec','cent_sat','group_ID','Mr','cz','ra','logMstar']

##the [0] is necessary for it to take the string. Otherwise, ECO_true is a
##numpy array
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

dirpath = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_Density"
dirpath+= r"\Catalogs\Beta_M1_Behroozi\ab_matching"
dirpath+= r"\Resolve_plk_5001_so_mvir_hod1_scatter0p2_mock1_ECO_Mocks"

usecols  = (0,1,2,4,7,13,20,25)
dlogM    = 0.2
neigh_dict = {1:0,2:1,3:2,5:3,10:4,20:5}

ECO_cats = (Index(dirpath,'.dat'))

names    = ['ra','dec','cz','Halo_ID','halo_cent_sat','logMstar',
    'group_ID','group_cent_sat']

PD = [[] for ii in range(len(ECO_cats))]

##creating a list of panda dataframes
for ii in range(len(ECO_cats)):
    temp_PD = (pd.read_csv(ECO_cats[ii],sep="\s+", usecols= usecols,
        header=None, skiprows=2,names=names))
    PD[ii] = temp_PD


##making stellar mass cuts, with stellar mass completeness limit (all should 
##pass, but just in case), and max limit being that of ECO's most massive 
##galaxy

PD_comp_1  = [(PD[ii][PD[ii].logMstar >= 9.1]) for ii in range(len(ECO_cats))]
PD_comp  = [(PD_comp_1[ii][PD_comp_1[ii].logMstar <=11.77]) 
    for ii in range(len(ECO_cats))]

##this resets the indices in the pd dataframe, meaning they will be 
##counted/indexed as if the galaxies with stellar masses outside of the limit 
##were never there
[(PD_comp[ii].reset_index(drop=True,inplace=True)) 
    for ii in range(len(ECO_cats))]

##finding the min and max stellar mass galaxy of each catalog
min_max_mass_arr = []

for ii in range(len(PD_comp)):
    min_max_mass_arr.append(max(PD_comp[ii].logMstar))
    min_max_mass_arr.append(min(PD_comp[ii].logMstar))

min_max_mass_arr = np.array(min_max_mass_arr)

##bins inherently use even decimal points, so this forces odd
bins = Bins_array_create(min_max_mass_arr,dlogM)
bins+= 0.1
bins_list = list(bins)
##double checking to make sure that there is no mass bin edge past 11.7
for ii in bins:
    if ii > 11.77:
        bins_list.remove(ii)
bins = np.array(bins_list)

##calulating the number of bins for later reference
num_of_bins = int(len(bins) - 1) 

ra_arr  = np.array([(PD_comp[ii].ra) for ii in range(len(PD_comp))])
dec_arr  = np.array([(PD_comp[ii].dec) for ii in range(len(PD_comp))])
cz_arr  = np.array([(PD_comp[ii].cz) for ii in range(len(PD_comp))])
mass_arr  = np.array([(PD_comp[ii].logMstar) for ii in range(len(PD_comp))])

halo_id_arr  = np.array([(PD_comp[ii].Halo_ID) for ii in range(len(PD_comp))])
halo_cent_sat_arr  = np.array([(PD_comp[ii].halo_cent_sat) 
    for ii in range(len(PD_comp))])
group_id_arr  = np.array([(PD_comp[ii].group_ID) 
        for ii in range(len(PD_comp))])
group_cent_sat_arr  = np.array([(PD_comp[ii].group_cent_sat) 
    for ii in range(len(PD_comp))])