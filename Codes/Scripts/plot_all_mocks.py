from __future__ import division, absolute_import

import astropy.stats
import glob
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import os
import pandas as pd
from scipy import optimize,spatial

#########################################################################

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

############################################################################

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
    dens = np.array([(3.*(n_val+1)/(4.*np.pi*r_val[hh]**3)) \
                     for hh in range(len(r_val))])

    return dens

#################################################################################

def plot_calcs(mass,bins,dlogM,mass_err=False,ratio_err=False):
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
    
    Optional
    --------
    mass_err  == True
        Calculates the Poisson errors on the stellar mass function.
        Returns mass_freq as a list with 2 array elements, the first being
        the stellar mass function values, the second being the errors.
    ratio_err == True
        Calculates the Poisson errors on the density-based, mass ratios.
        Creates empty list and appends ratio error arrays to it as they 
        are generated. Returns ratio_dict as a list. The first element is 
        a dictionary with the ratio values to be plotted. The second is a
        three element list. Each element is an array with the error values 
        for each of the three density-cut ratios.

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

    ratio_dict = {}
    frac_val   = [2,4,10]
    
    if ratio_err == True:
        yerr = []

    for ii in frac_val:
        ratio_dict[ii] = {}

        # Calculations for the lower density cut
        frac_data      = int(len(mass)/ii)
        frac_mass      = mass[0:frac_data]
        counts, edges  = np.histogram(frac_mass,bins)

        # Calculations for the higher density cut
        frac_mass_2       = mass[-frac_data:]
        counts_2, edges_2 = np.histogram(frac_mass_2,bins)

        # Ratio determination
        ratio_counts   = (1.*counts_2)/(1.*counts)
        ratio_dict[ii] = ratio_counts

        if ratio_err == True:
            yerr.append((counts_2*1.)/(counts*1.)*\
                np.sqrt(1./counts + 1./counts_2))

    if mass_err == True:
        mass_freq_list     = [[] for xx in xrange(2)]
        mass_freq_list[0]  = mass_freq
        mass_freq_list[1]  = np.sqrt(mass_counts)/float(len(mass))/dlogM
        mass_freq          = np.array(mass_freq_list)

    if ratio_err == True:
        ratio_dict_list    = [[] for xx in range(2)]
        ratio_dict_list[0] = ratio_dict
        ratio_dict_list[1] = yerr
        ratio_dict         = ratio_dict_list

    return bin_centers, mass_freq, ratio_dict

##############################################################################

def bin_func(mass_dist,bins,kk,bootstrap=False):
    """
    Returns median distance to Nth nearest neighbor

    Parameters
    ----------
    mass_dist: array-like
        An array with mass values in at index 0 (when transformed) and distance to the Nth 
        nearest neighbor in the others
        Example: 6239 by 7
            Has mass values and distances to 6 Nth nearest neighbors  
    bins: array=like
        A 1D array with the values which will be used as the bin edges     
    kk: integer-like
        The index of mass_dist (transformed) where the appropriate distance array may be found

    Optional
    --------
    bootstrap == True
        Calculates the bootstrap errors associated with each median distance value.
        Creates an array housing arrays containing the actual distance values
        associated with every galaxy in a specific bin. Bootstrap error is then 
        performed using astropy, and upper and lower one sigma values are found
        for each median value.  These are added to a list with the median distances, 
        and then converted to an array and returned in place of just 'medians.'

    Returns
    -------
    medians: array-like
        An array with the median distance to the Nth nearest neighbor from 
        all the galaxies in each of the bins
    """
    edges        = bins
    digitized    = np.digitize(mass_dist.T[0],edges)
    digitized   -= int(1)

    bin_nums     = np.unique(digitized)
    bin_nums     = list(bin_nums)

    if 14 in bin_nums:
        bin_nums.remove(14)
        bin_nums = np.array(bin_nums)
    
    medians  = np.array([np.median(mass_dist.T[kk][digitized==ii]) \
                for ii in bin_nums])

    if bootstrap == True:
        dist_in_bin    = np.array([(mass_dist.T[kk][digitized==ii]) \
                for ii in bin_nums])
        for vv in range(len(dist_in_bin)):
            if len(dist_in_bin[vv]) == 0:
                dist_in_bin_list = list(dist_in_bin[vv])
                dist_in_bin[vv] = np.zeros(len(dist_in_bin[0]))
        low_err_test   = np.array([np.percentile(astropy.stats.bootstrap\
                        (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),16) \
                        for vv in range(len(dist_in_bin))])
        high_err_test  = np.array([np.percentile(astropy.stats.bootstrap\
                        (dist_in_bin[vv],bootnum=1000,bootfunc=np.median),84) \
                        for vv in range(len(dist_in_bin))])

        med_list    = [[] for yy in range(len(frac_vals))]
        med_list[0] = medians
        med_list[1] = low_err_test
        med_list[2] = high_err_test
        medians     = np.array(med_list)

    return medians    

#####################################################################################
def hist_calcs(mass,bins,dlogM,eco=False):
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
    
    if eco == True:
        low_err   = [[] for xx in xrange(len(frac_val))]
        high_err  = [[] for xx in xrange(len(frac_val))]
    
    for ii in frac_val:
        hist_dict_low[ii]  = {}
        hist_dict_high[ii] = {}
    
        frac_data     = int(len(mass)/ii)
        
        frac_mass     = mass[0:frac_data]
        counts, edges = np.histogram(frac_mass,bins)
        low_counts    = (counts/float(len(frac_mass))/dlogM)
        
        if eco == True:
            low_err[frac_dict[ii]]    = np.sqrt(counts)/len(frac_mass)/dlogM
        
        hist_dict_low[ii]  = low_counts

        frac_mass_2        = mass[-frac_data:]
        counts_2, edges_2  = np.histogram(frac_mass_2,bins)
        high_counts        = (counts_2/float(len(frac_mass_2))/dlogM)
        
        if eco == True:
            high_err[frac_dict[ii]]   = np.sqrt(counts_2)/len(frac_mass_2)/dlogM
        
        hist_dict_high[ii] = high_counts
        
    if eco == True:
        hist_dict_low['low_err']   = low_err
        hist_dict_high['high_err'] = high_err
    
    return hist_dict_low, hist_dict_high
#####################################################################################

def plot_all_rats(bin_centers,y_vals,neigh_val,ax,col_num,plot_idx):
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
        ax.set_xlabel('$\log\ M_{*}$',fontsize=18)
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
    ax.set_ylim([0,5])
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.set_yticks([1.,3.])
    ax.tick_params(axis='both', labelsize=12)
    ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
    ax.plot(bin_centers,y_vals,color='silver')

###################################################################################

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
            ax.set_xlabel('$\log\ M_{*}$',fontsize=18)
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
        ax.set_ylim([0,5])
        ax.set_xticks(np.arange(9.5, 12., 0.5))
        ax.set_yticks([1.,3.])
        ax.tick_params(axis='both', labelsize=12)
        ax.axhline(y=1,c="darkorchid",linewidth=0.5,zorder=0)
    frac_vals = np.array([2,4,10])
    y_vals_2 = y_vals[0][frac_vals[hh]]
    ax.errorbar(bin_centers,y_vals_2,yerr=y_vals[1][hh],\
                color='deeppink',linewidth=2)

#######################################################################################

def plot_hists(mass,neigh_val,bins,dlogM,ax,col_num,plot_idx):
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
    if col_num==0:
        title_label = 'Mass 50/50, {0} NN'.format(neigh_val)
        frac_val    = 2
        ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                verticalalignment='top',transform=ax.transAxes,fontsize=12)
    elif col_num==1:
        title_label = 'Mass 25/75, {0} NN'.format(neigh_val)
        frac_val    = 4
        ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                verticalalignment='top',transform=ax.transAxes,fontsize=12)
    elif col_num==2:
        title_label = 'Mass 10/90, {0} NN'.format(neigh_val)
        frac_val    = 10
        ax.text(0.05, 0.95, title_label,horizontalalignment='left',\
                verticalalignment='top',transform=ax.transAxes,fontsize=12)
    ax.set_xlim(9.1,11.9)
    ax.set_ylim([10**-3,10**1])
    ax.set_xticks(np.arange(9.5, 12., 0.5))
    ax.set_yticks([10**-2,10**0])
    
    frac_data     = (len(mass)/frac_val)

    frac_mass     = mass[0:frac_data]
    counts, edges = np.histogram(frac_mass,bins)
    low_counts    = (counts/float(len(frac_mass))/dlogM)
    
    bins_cens     = .5*(edges[:-1]+edges[1:])
#     ax.step(bins_cens, low_counts, color='lightslategrey',where='mid',alpha=0.1)
    ax.plot(bins_cens, low_counts, color='lightslategrey',alpha=0.1)

    frac_mass_2       =  mass[-frac_data:]
    counts_2, edges_2 =  np.histogram(frac_mass_2,bins)
    high_counts       = (counts_2/float(len(frac_mass_2))/dlogM)
#     ax.step(bins_cens, high_counts, color='lightslategray',where='mid',alpha=0.1)
    ax.plot(bins_cens, high_counts, color='lightslategray',alpha=0.1)
#######################################################################################

def plot_eco_hists(mass,bins,dlogM,ax,col):
    if   col==0:
        frac_val    = 2
    elif col==1:
        frac_val    = 4
    elif col==2:
        frac_val    = 10
    
    frac_data     = (len(mass)/frac_val)

    frac_mass     = mass[0:frac_data]
    counts, edges = np.histogram(frac_mass,bins)
    bins_cens     = .5*(edges[:-1]+edges[1:])
    ax.step(bins_cens, (counts/float(len(frac_mass))/dlogM), color='lime',where='mid')

    frac_mass_2       = mass[-frac_data:]
    counts_2, edges_2 = np.histogram(frac_mass_2,bins)
    ax.step(bins_cens, (counts_2/float(len(frac_mass_2))/dlogM), color='deeppink',where='mid')

#######################################################################################

def plot_all_meds(bin_centers,y_vals,ax,plot_idx):
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
    ax.tick_params(axis='both', which='major', labelsize=16)
    title_here = 'n = {0}'.format(titles[plot_idx])
    ax.text(0.05, 0.95, title_here,horizontalalignment='left',\
            verticalalignment='top',transform=ax.transAxes,fontsize=18)
    if plot_idx == 4:
        ax.set_xlabel('$\log\ M_{*}$',fontsize=18)
    ax.plot(bin_centers,y_vals,color='silver')

#############################################################################              

def plot_eco_meds(bin_centers,y_vals,low_lim,up_lim,ax,plot_idx,only=False):
    """
    Returns six subplots showing the median Nth nearest neighbor distance for ECO
        galaxies in each mass bin
        
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
            ax.set_xlabel('$\log\ M_{*}$',fontsize=18)
    ax.errorbar(bin_centers,y_vals,yerr=0.1,lolims=low_lim,\
        uplims=up_lim,color='deeppink')

##############################################################################

def plot_bands(bin_centers,upper,lower,ax):
    """
    Returns an overlayed, fill-between plot, creating a band
        between the different mock catalog values plotted
    
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    upper: array-like
        Array with the max y-values among all the mocks
        for each mass bin
    lower: array-like
        Array with the min y-values among all the mocks
        for each mass bin
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    
    Returns
    -------
    A semi-transparent band overlaying the area of the plot 
    bordered by the mocks
    """
    ax.fill_between(bin_centers,upper,lower,color='silver',alpha=0.1)

#######################################################################################

def plot_med_range(bin_centers,low_lim,up_lim,ax,alpha,color='gray'):
    """
    Returns a plot with a transparent band highlighting a range of
        values.
    
    Parameters
    ----------
    bin_centers: array-like
        An array with the medians mass values of the mass bins
    low_lim: array-like
        Array with the min y-values among all the mocks
        for each mass bin
    up_lim: array-like
        Array with the max y-values among all the mocks
        for each mass bin 
    ax: axis-like
        A value which specifies which axis each subplot is to be 
        plotted to
    alpha: float-like
        A value which will determine the tranparency of the band
    color: str
        Any color which Python recognizes; sets the color of the band
        
    Returns
    -------
    A colored band spanning from the max y-values to the minimum.
    
    """
    ax.fill_between(bin_centers,low_lim,up_lim,color=color,alpha=alpha)    

##############################################################################
##############################################################################
##############################################################################

dirpath  = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density\Catalogs\Resolve_plk_5001_so_mvir_hod1_Hannah_newdens_ECO_Mocks\Resolve_plk_5001_so_mvir_hod1_Hannah_newdens_ECO_Mocks"
usecols  = (0,1,8,13)

bins  = np.linspace(9.1,11.9,15)
dlogM = (11.9-9.1)/14

##############################################################################
##############################################################################
##############################################################################

ECO_cats = (Index(dirpath,'.dat'))

names    = ['ra','dec','cz','logMstar']
PD       = [(pd.read_csv(ECO_cats[ii],sep="\s+", usecols= usecols,header=None,\
                   skiprows=2,names=names)) for ii in range(len(ECO_cats))]
PD_comp  = [(PD[ii][PD[ii].logMstar >= 9.1]) for ii in range(len(ECO_cats))]

ra_arr   = np.array([(np.array(PD_comp[ii])).T[0] for ii in range(len(PD_comp))])
dec_arr  = np.array([(np.array(PD_comp[ii])).T[1] for ii in range(len(PD_comp))])
cz_arr   = np.array([(np.array(PD_comp[ii])).T[2] for ii in range(len(PD_comp))])
mass_arr = np.array([(np.array(PD_comp[ii])).T[3] for ii in range(len(PD_comp))])

coords_test = np.array([sph_to_cart(ra_arr[vv],dec_arr[vv],cz_arr[vv]) \
                for vv in range(len(ECO_cats))])

neigh_vals  = np.array([1,2,3,5,10,20])

nn_arr      = [[] for xx in xrange(len(coords_test))]
nn_arr_nn   = [[] for yy in xrange(len(neigh_vals))]

for vv in range(len(coords_test)):
    nn_arr[vv] = spatial.cKDTree(coords_test[vv])
    nn_arr[vv] = np.array(nn_arr[vv].query(coords_test[vv],21)[0])

nn_specs       = [(np.array(nn_arr).T[ii].T[neigh_vals].T) for ii in \
                    range(len(coords_test))]
nn_mass_dist   = np.array([(np.column_stack((mass_arr[qq],nn_specs[qq]))) \
                    for qq in range(len(coords_test))])

###################################################################################
sat_cols    = (13,25)
  
sat_names   = ['logMstar','cent_sat_flag']
SF_PD       = [(pd.read_csv(ECO_cats[ii],sep="\s+", usecols= sat_cols,header=None,\
                     skiprows=2,names=sat_names)) for ii in range(8)]
SF_PD_comp  = [(SF_PD[ii][SF_PD[ii].logMstar >= 9.1]) for ii in \
                range(len(ECO_cats))]


sats_num  = np.array([(len(SF_PD_comp[ii][SF_PD_comp[ii].cent_sat_flag==0])) \
            for ii in range(len(SF_PD_comp))])
cents_num = np.array([(len(SF_PD_comp[ii][SF_PD_comp[ii].cent_sat_flag==1])) \
            for ii in range(len(SF_PD_comp))])
gal_tot   = np.array([(len(SF_PD_comp[ii])) for ii in range(len(SF_PD_comp))])
# print sats_num/gal_tot

###################################################################################

nn_dist    = {}
nn_dens    = {}
mass_dat   = {}
ratio_info = {}

mass_freq  = [[] for xx in xrange(len(coords_test))]

for ii in range(len(coords_test)):
    nn_dist[ii]    = {}
    nn_dens[ii]    = {}
    mass_dat[ii]   = {}
    ratio_info[ii] = {}

    nn_dist[ii]['mass'] = nn_mass_dist[ii].T[0]

    for jj in range(len(neigh_vals)):
        nn_dist[ii][(neigh_vals[jj])]  = np.array(nn_mass_dist[ii].T\
                                            [range(1,len(neigh_vals)+1)[jj]])
        nn_dens[ii][(neigh_vals[jj])]  = np.column_stack((nn_mass_dist[ii].T\
                                            [0],calc_dens(neigh_vals[jj],\
                                            nn_mass_dist[ii].T[range(1,len\
                                                (neigh_vals)+1)[jj]])))

        idx = np.array([nn_dens[ii][neigh_vals[jj]].T[1].argsort()])
        mass_dat[ii][(neigh_vals[jj])] = (nn_dens[ii][neigh_vals[jj]][idx].T[0])

        bin_centers, mass_freq[ii], ratio_info[ii][neigh_vals[jj]] = \
                            plot_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)

all_mock_meds = [[] for xx in range(len(nn_mass_dist))]

for vv in range(len(nn_mass_dist)):
    all_mock_meds[vv] = np.array([bin_func(nn_mass_dist[vv],bins,(jj+1)) \
                                for jj in range(len(nn_mass_dist[vv].T)-1)])
    
med_plot_arr = [([[] for yy in xrange(len(nn_mass_dist))]) \
                                            for xx in xrange(len(neigh_vals))]

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        med_plot_arr[ii][jj] = all_mock_meds[jj][ii]    

mass_freq_plot  = (np.array(mass_freq))
max_lim = [[] for xx in range(len(mass_freq_plot.T))]
min_lim = [[] for xx in range(len(mass_freq_plot.T))]
for jj in range(len(mass_freq_plot.T)):
    max_lim[jj] = max(mass_freq_plot.T[jj])
    min_lim[jj] = min(mass_freq_plot.T[jj])

###################################################################################
# ordered_mass = nn_mass_dist[0].T[0][(nn_mass_dist[0].T[0].argsort())]
dist_cont = [[[[] for zz in xrange(len(bins)-1)] for yy in \
              xrange(len(nn_mass_dist))] for xx in xrange(1,len(nn_mass_dist[0].T))]

for ii in xrange(len(nn_mass_dist)):
    sorting_test  = np.digitize(nn_mass_dist[ii].T[0],bins)
    bin_nums      = np.unique(sorting_test)
    bin_nums_list = list(bin_nums)
    if 15 in bin_nums_list:
        bin_nums_list.remove(15)
        bin_nums  = np.array(bin_nums_list)
    for jj in xrange(1,len(nn_mass_dist[ii].T)):
        for hh in bin_nums:
            dist_cont[jj-1][ii][hh-1] = (nn_mass_dist[ii].T[jj][sorting_test==hh])

###################################################################################
top_68 = [[[[]for ll in xrange(len(bin_nums))]for yy in \
            xrange(len(nn_mass_dist))] for xx in xrange(len(neigh_vals))]
low_68 = [[[[]for ll in xrange(len(bin_nums))]for yy in \
            xrange(len(nn_mass_dist))] for xx in xrange(len(neigh_vals))]
top_95 = [[[[]for ll in xrange(len(bin_nums))]for yy in \
            xrange(len(nn_mass_dist))] for xx in xrange(len(neigh_vals))]
low_95 = [[[[]for ll in xrange(len(bin_nums))]for yy in \
            xrange(len(nn_mass_dist))] for xx in xrange(len(neigh_vals))]
med_50 = [[[[]for ll in xrange(len(bin_nums))]for yy in \
            xrange(len(nn_mass_dist))] for xx in xrange(len(neigh_vals))]

for aa in xrange(len(neigh_vals)):
    for bb in xrange(len(nn_mass_dist)):
        for cc in xrange(len(dist_cont[aa][bb])):
            top_68[aa][bb][cc] = np.percentile(dist_cont[aa][bb][cc],84)
            low_68[aa][bb][cc] = np.percentile(dist_cont[aa][bb][cc],16)
            top_95[aa][bb][cc] = np.percentile(dist_cont[aa][bb][cc],97.5)
            low_95[aa][bb][cc] = np.percentile(dist_cont[aa][bb][cc],2.5)
            med_50[aa][bb][cc] = np.median((dist_cont[aa][bb][cc]))
            
top_68 = np.array(top_68)
low_68 = np.array(low_68)
top_95 = np.array(top_95)
low_95 = np.array(low_95)
med_50 = np.array(med_50)

###################################################################################

frac_vals   = [2,4,10]
nn_plot_arr = [[[] for yy in xrange(len(nn_mass_dist))] for xx in \
    xrange(len(neigh_vals))]

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        nn_plot_arr[ii][jj] = (ratio_info[jj][neigh_vals[ii]])
        plot_frac_arr = [[[[] for yy in xrange(len(nn_mass_dist))] \
                         for zz in xrange(len(frac_vals))] for xx in \
                         xrange(len(nn_plot_arr))]

for jj in range(len(nn_mass_dist)):
    for hh in range(len(frac_vals)):
        for ii in range(len(neigh_vals)):
            plot_frac_arr[ii][hh][jj] = nn_plot_arr[ii][jj][frac_vals[hh]]

######################################################################################
######################################################################################
######################################################################################

eco_path = r"C:\Users\Hannah\Desktop\Vanderbilt_REU\Stellar_mass_env_density\Catalogs\ECO_true"
eco_cols = np.array([0,1,2,4])

######################################################################################
######################################################################################
######################################################################################

ECO_true = (Index(eco_path,'.txt'))
names    = ['ra','dec','cz','logMstar']
PD_eco   = pd.read_csv(ECO_true[0],sep="\s+", usecols=(eco_cols),header=None,\
                skiprows=1,names=names)
eco_comp = PD_eco[PD_eco.logMstar >= 9.1]

ra_eco   = (np.array(eco_comp)).T[0]
dec_eco  = (np.array(eco_comp)).T[1] 
cz_eco   = (np.array(eco_comp)).T[2] 
mass_eco = (np.array(eco_comp)).T[3]

coords_eco        = sph_to_cart(ra_eco,dec_eco,cz_eco)
eco_neighbor_tree = spatial.cKDTree(coords_eco)
eco_tree_dist     = np.array(eco_neighbor_tree.query(coords_eco,\
                    (neigh_vals[-1]+1))[0])

eco_mass_dist = np.column_stack((mass_eco,eco_tree_dist.T[neigh_vals].T))
eco_dens = ([calc_dens(neigh_vals[jj],\
            (eco_mass_dist.T[range(1,7)[jj]])) for jj in range(len(neigh_vals))])

eco_mass_dens = [(np.column_stack((mass_eco,eco_dens[ii]))) for ii in \
                range(len(neigh_vals))]
eco_idx  = [(eco_mass_dens[jj].T[1].argsort()) for jj in range(len(neigh_vals))]
eco_mass_dat  = [(eco_mass_dens[jj][eco_idx[jj]].T[0]) for jj in \
                range(len(neigh_vals))]

eco_ratio_info    = [[] for xx in xrange(len(eco_mass_dat))]


for qq in range(len(eco_mass_dat)):
    bin_centers, eco_freq, eco_ratio_info[qq] = plot_calcs(eco_mass_dat[qq],\
                                    bins,dlogM,mass_err=True,ratio_err=True)

eco_medians   = [[] for xx in xrange(len(eco_mass_dat))]    

for jj in (range(len(eco_mass_dat))):
    eco_medians[jj] = np.array(bin_func(eco_mass_dist,bins,(jj+1),bootstrap=True))

#####################################################################################
#####################################################################################

fig,ax  = plt.subplots(figsize=(8,8))
ax.set_title('Mass Distribution',fontsize=18)
ax.set_xlabel('$\log\ M_{*}$',fontsize=17)
ax.set_yscale('log')
ax.set_xlim(9.1,12.1)
ax.tick_params(axis='both', labelsize=14)

for ii in range(len(mass_freq)):
    ax.plot(bin_centers,mass_freq[ii],color='silver')
    ax.fill_between(bin_centers,max_lim,min_lim,color='silver',alpha=0.1)
ax.errorbar(bin_centers,eco_freq[0],yerr=eco_freq[1],color='deeppink',\
            linewidth=2,label='ECO')
ax.legend(loc='best')

plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.94,\
                    hspace=0.2,wspace=0.2)
plt.show()

####################################################################################

A = {}
nn_dict   = {1:0,2:1,3:2,5:3,10:4,20:5}
coln_dict = {2:0,4:1,10:2}

nn_keys  = np.sort(nn_dict.keys())
col_keys = np.sort(coln_dict.keys())
zz_num   = len(plot_frac_arr[nn_dict[1]][coln_dict[2]])

for nn in nn_keys:
    for coln in col_keys:
        bin_str    = '{0}_{1}'.format(nn,coln)
        for cc in range(zz_num):
            zz_arr = np.array(plot_frac_arr[nn_dict[nn]][coln_dict[coln]][cc])
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

####################################################################################

np.seterr(divide='ignore',invalid='ignore')

nrow_num = int(6)
ncol_num = int(3)
zz       = int(0)

fig, axes = plt.subplots(nrows=nrow_num, ncols=ncol_num, \
            figsize=(100,200), sharex= True,sharey=True)
axes_flat = axes.flatten()
# fig.suptitle("Percentile Trends", fontsize=18)

while zz <= 16:
    for ii in range(len(eco_ratio_info)):
        for hh in range(len(eco_ratio_info[0][1])):
            for jj in range(len(nn_mass_dist)):
                upper = A['{0}_{1}'.format(neigh_vals[ii],frac_vals[hh])][0]
                lower  = A['{0}_{1}'.format(neigh_vals[ii],frac_vals[hh])][1]
                plot_bands(bin_centers,upper,lower,axes_flat[zz] )
                plot_all_rats(bin_centers,(plot_frac_arr[ii][hh][jj]),\
                              neigh_vals[ii],axes_flat[zz],hh,zz)
            plot_eco_rats(bin_centers,(eco_ratio_info[ii]),neigh_vals[ii],\
                                    axes_flat[zz],hh,zz)
            zz += 1

plt.subplots_adjust(left=0.02, bottom=0.09, right=1.00, top=1.00,\
                    hspace=0,wspace=0)

plt.show()

######################################################################################

B = {}
yy_num = len(med_plot_arr[0])

for nn in range(len(med_plot_arr)):
    for ii in range(yy_num):
        med_str  = '{0}'.format(nn)
        yy_arr   = med_plot_arr[nn][ii]
        n_y_elem = len(yy_arr)
        if ii == 0:
            yy_tot = np.zeros((n_y_elem,1))
        yy_tot = np.insert(yy_tot,len(yy_tot.T),yy_arr,1)
    yy_tot = np.array(np.delete(yy_tot,0,axis=1))
    yy_tot_max = [np.nanmax(yy_tot[kk]) for kk in xrange(len(yy_tot))]
    yy_tot_min = [np.nanmin(yy_tot[kk]) for kk in xrange(len(yy_tot))]
    B[med_str] = [yy_tot_max,yy_tot_min]

#####################################################################################

nrow_num_mass = int(2)
ncol_num_mass = int(3)

fig, axes = plt.subplots(nrows=nrow_num_mass, ncols=ncol_num_mass, \
        figsize=(100,200), sharex= True, sharey = True)
axes_flat = axes.flatten()

zz = int(0)
while zz <=4:
    for ii in range(len(med_plot_arr)):
        for vv in range(len(nn_mass_dist)):
                lower_m  = B['{0}'.format(ii)][0]
                upper_m  = B['{0}'.format(ii)][1]
                plot_med_range(bin_centers,top_95[ii][vv],low_95[ii][vv],\
                    axes_flat[zz],0.05,color='lightsteelblue')
                plot_med_range(bin_centers,top_68[ii][vv],low_68[ii][vv],\
                    axes_flat[zz],0.15,color='gainsboro')
                plot_bands(bin_centers,upper_m,lower_m,axes_flat[zz])
                plot_all_meds(bin_centers,med_plot_arr[ii][vv],axes_flat[zz],zz)
                plot_eco_meds(bin_centers,eco_medians[ii][0],\
                              eco_medians[ii][1],eco_medians[ii][2],\
                                                axes_flat[zz],zz)
        zz   += 1
        
plt.subplots_adjust(left=0.05, bottom=0.09, right=1.00, top=1.00,\
            hspace=0,wspace=0)
plt.show()

####################################################################################                

hist_low_info  = {}
hist_high_info = {}

for ii in xrange(len(coords_test)):
    hist_low_info[ii]  = {}
    hist_high_info[ii] = {}
    
    for jj in range(len(neigh_vals)):
        hist_low_info[ii][neigh_vals[jj]], \
        hist_high_info[ii][neigh_vals[jj]] = \
        hist_calcs(mass_dat[ii][neigh_vals[jj]],bins,dlogM)
        
frac_vals     = [2,4,10]
hist_low_arr  = [[[] for yy in xrange(len(nn_mass_dist))] for xx in \
    xrange(len(neigh_vals))]
hist_high_arr = [[[] for yy in xrange(len(nn_mass_dist))] for xx in \
    xrange(len(neigh_vals))]

for ii in range(len(neigh_vals)):
    for jj in range(len(nn_mass_dist)):
        hist_low_arr[ii][jj]  = (hist_low_info[jj][neigh_vals[ii]])
        hist_high_arr[ii][jj] = (hist_high_info[jj][neigh_vals[ii]])
        
        
        plot_low_hist  = [[[[] for yy in xrange(len(nn_mass_dist))] \
                         for zz in xrange(len(frac_vals))] for xx in \
                         xrange(len(hist_low_arr))]
        
        plot_high_hist = [[[[] for yy in xrange(len(nn_mass_dist))] \
                 for zz in xrange(len(frac_vals))] for xx in \
                 xrange(len(hist_high_arr))]

for jj in range(len(nn_mass_dist)):
    for hh in range(len(frac_vals)):
        for ii in range(len(neigh_vals)):
            plot_low_hist[ii][hh][jj]  = hist_low_arr[ii][jj][frac_vals[hh]]        
            plot_high_hist[ii][hh][jj] = hist_high_arr[ii][jj][frac_vals[hh]]        

###############################################################################

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

###############################################################################

nrow_num  = int(6)
ncol_num  = int(3)

fig, axes = plt.subplots(nrows=nrow_num, ncols=ncol_num, \
            figsize=(150,200), sharex= True,sharey=True)
axes_flat = axes.flatten()

for ii in range(len(mass_dat)):
    zz = 0
    for jj in range(len(neigh_vals)):
        for hh in range(3):
            upper    = C['{0}_{1}'.format(neigh_vals[jj],frac_vals[hh])][0]
            lower    = C['{0}_{1}'.format(neigh_vals[jj],frac_vals[hh])][1]
            upper_2  = D['{0}_{1}'.format(neigh_vals[jj],frac_vals[hh])][0]
            lower_2  = D['{0}_{1}'.format(neigh_vals[jj],frac_vals[hh])][1]
            plot_bands(bin_centers,upper,lower,axes_flat[zz])
            plot_bands(bin_centers,upper_2,lower_2,axes_flat[zz])
            plot_hists(mass_dat[ii][neigh_vals[jj]],neigh_vals[jj],bins,dlogM,\
                axes_flat[zz], hh, zz)
            if ii == 0:
                plot_eco_hists(eco_mass_dat[jj],bins,dlogM,\
                axes_flat[zz],hh)
            zz += int(1)         

plt.subplots_adjust(left=0.02, bottom=0.09, right=0.98, top=0.98,\
                    hspace=0, wspace=0)            

plt.show()
###############################################################################

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
        Array of values prepared to be plotted on a log
        scale to display the Schechter function
        
    """
    constant = np.log(10) * phi_star
    log_M_Mstar = np.log10(stellar_mass/m_star)
    res = constant * 10**(log_M_Mstar * (alpha+1)) * \
        np.exp(-10**log_M_Mstar)
        
    return res

###############################################################################


xdata = 10**bin_centers
p0    = (1,-1.05,10**10.64)
param_arr = [[] for ii in range(len(mass_freq)+1)]

fig,axes  = plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,\
                        figsize=(150,200))
axes_flat = axes.flatten()

for ii in range(len(mass_freq)+1):
    if ii == 8:
        ydata = eco_freq[0]
        opt_v, est_cov = optimize.curve_fit(schechter_log_func,xdata,ydata,\
                            p0=p0,sigma=eco_freq[1])
    else:
        ydata = (mass_freq[ii])
        opt_v, est_cov = optimize.curve_fit(schechter_log_func,xdata,ydata,p0=p0)
    schech_vals   = schechter_log_func(10**bin_centers,opt_v[0],opt_v[1],opt_v[2])
    param_arr[ii] = opt_v
    param_arr = np.array(param_arr)
    
    ax = axes_flat[ii]
    ax.set_yscale('log')
    ax.set_ylim([10**-3,10**0])
    ax.set_xlim([9.1,11.9])
    ax.set_yticks([10**-2,10**-1,10**0])
    ax.plot(bin_centers,schech_vals,label='Schechter',color='silver')
    if ii == 8:
        ax.errorbar(bin_centers,ydata,yerr=eco_freq[1],color='deeppink',\
            label='ECO')
    else:
        ax.plot(bin_centers,ydata,label='Mock',color='darkorchid')
    if ii == 0 or ii == 8:
        ax.legend(loc='best')
    if ii == 7:
        ax.set_xlabel('$\log\ M_{*}$',fontsize=18)
    
plt.subplots_adjust(left=0.03, bottom=0.08, right=0.99, top=0.99,\
                    hspace=0,wspace=0)                
    
plt.show()   

###############################################################################

eco_low  = {}
eco_high = {}
for jj in range(len(neigh_vals)):
    eco_low[neigh_vals[jj]]  = {}
    eco_high[neigh_vals[jj]] = {}
    eco_low[neigh_vals[jj]], eco_high[neigh_vals[jj]] = hist_calcs\
    (eco_mass_dat[jj],bins,dlogM,eco=True)


###############################################################################

fig,ax = plt.subplots()

ax.set_yscale('log')
ax.plot(bin_centers,eco_low[1][2])
plt.show()

###############################################################################

def param_finder(hist_counts,bin_centers,hist_err):
    """
    Returns
    -------
    opt_v: array-like
        Array with three values: phi_star, alpha, and M_star
    res_arr: array-like
        Array with two values: alpha and M_star


    """
    xdata = 10**bin_centers
    p0    = (1,-1.05,10**10.64)
    opt_v,est_cov = optimize.curve_fit(schechter_log_func,xdata,\
                            hist_counts,p0=p0)
    alpha   = opt_v[1]
    log_m_star  = np.log10(opt_v[2])
    res_arr = np.array([alpha,log_m_star])

    return opt_v, res_arr

###############################################################################
###Test that param_finder is working

opt_v,test_arr = param_finder(eco_low[1][2],bin_centers,eco_low[1]['low_err'][0])
schech_vals   = schechter_log_func(10**bin_centers,opt_v[0],opt_v[1],\
                    opt_v[2])

fig,ax = plt.subplots()

ax.set_yscale('log')
ax.plot(bin_centers,eco_low[1][2])
ax.plot(bin_centers,schech_vals)
plt.show()

###############################################################################
param_dict_low  = {}
param_dict_high = {}


for dd in neigh_vals:
    param_dict_low[dd]  = {}
    param_dict_high[dd] = {}
    for ee in frac_vals:
        param_dict_low[dd][ee]  = {}
        param_dict_high[dd][ee] = {}
        opt_v, param_dict_low[dd][ee] = param_finder(eco_low[dd][ee],\
            bin_centers,eco_low[dd]['low_err'][coln_dict[ee]])
        opt_v, param_dict_high[dd][ee] = param_finder(eco_high[dd][ee],\
            bin_centers,eco_high[dd]['high_err'][coln_dict[ee]])

over_alpha_dict = {}
over_log_m_star = {}
for dd in neigh_vals:
    temp_list_alpha = []
    temp_list_logMstar = []
    over_alpha_dict[dd] = {}
    over_log_m_star[dd] = {}

    low_idx  = np.array(list(reversed(np.sort(param_dict_low[dd].keys()))))
    high_idx = np.sort(param_dict_high[dd].keys())

    for ff in range(len(low_idx)):
        temp_list_alpha.append(param_dict_low[dd][low_idx[ff]][0])
        temp_list_logMstar.append(param_dict_low[dd][low_idx[ff]][1])
    for ff in range(len(high_idx)):
        temp_list_alpha.append(param_dict_high[dd][high_idx[ff]][0])
        temp_list_logMstar.append(param_dict_low[dd][low_idx[ff]][1])
    over_alpha_dict[dd] = temp_list_alpha
    over_log_m_star[dd] = temp_list_logMstar

#10,25,low_50,high_50,75,90
perc_arr = (10,25,49,51,75,90)


fig,ax = plt.subplots()
for jj in neigh_vals:
    ax.plot(perc_arr,over_log_m_star[jj],marker='o',label='{0}'.format(jj),linestyle='--')
ax.set_xlim([0,100])
ax.legend(loc='best', numpoints=1)
ax.set_xlabel('Percentage')
ax.set_ylabel(r'$\log\ M_{*}$')
plt.show()


fig,ax = plt.subplots()
for jj in neigh_vals:
    ax.plot(perc_arr,over_alpha_dict[jj],marker='o',label='{0}'.format(jj),linestyle='--')
ax.set_xlim([0,100])
ax.legend(loc='best', numpoints=1)
ax.set_xlabel('Percentage')
ax.set_ylabel(r'$\alpha$')
plt.show()


###############################################################################

diff_dict_m_star = {}
diff_dict_alpha = {}

for dd in neigh_vals:
    diff_dict_m_star[dd] = {}
    diff_dict_alpha[dd] = {}
    for jj in frac_vals:
        temp_list_diff_m_star = []
        temp_list_diff_alpha = []
        diff_dict_alpha[dd][jj] = {}
        diff_dict_m_star[dd][jj] = {}
        temp_list_diff_m_star.append((param_dict_high[dd][jj][1] - \
            param_dict_low[dd][jj][1]))
        temp_list_diff_alpha.append((param_dict_high[dd][jj][0] - \
            param_dict_low[dd][jj][0]))
        diff_dict_alpha[dd][jj]  = np.array(temp_list_diff_alpha)
        diff_dict_m_star[dd][jj] = np.array(temp_list_diff_m_star)

dict_revamp_mstar = {}
for dd in neigh_vals:
    dict_revamp_mstar[dd] = []
    for jj in frac_vals:
        dict_revamp_mstar[dd].append(diff_dict_m_star[dd][jj])

dict_revamp_alpha = {}
for dd in neigh_vals:
    dict_revamp_alpha[dd] = []
    for jj in frac_vals:
        dict_revamp_alpha[dd].append(diff_dict_alpha[dd][jj])


discrete_x = np.array([1,2,3])

fig,ax = plt.subplots()

for ii in neigh_vals:
    ax.plot(discrete_x,dict_revamp_mstar[ii],marker='o',\
        linestyle = '--',label='{0}'.format(ii))
ax.set_xlim(0,4)        
ax.set_xlabel('Fractional Cut')
ax.set_xticks([1,2,3])
ax.set_ylabel('Difference in $\log\ M_{*}$')
ax.legend(loc='best',numpoints=1)
plt.show()

######

fig,ax = plt.subplots()

for ii in neigh_vals:
    ax.plot(discrete_x,dict_revamp_alpha[ii],marker='o',\
        linestyle = '--',label='{0}'.format(ii))
ax.set_xlim(0,4)        
ax.set_xlabel('Fractional Cut')
ax.set_xticks([1,2,3])
ax.set_ylabel(r'Difference in $\alpha$')
ax.legend(loc='best',numpoints=1)
plt.show()


#50/50,25/75,10/90

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################