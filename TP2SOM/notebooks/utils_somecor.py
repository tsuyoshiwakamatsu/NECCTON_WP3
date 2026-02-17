from pathlib import Path
import random

import yaml
import pickle
import numpy as np
import numpy.ma as ma
import calendar
import inspect
import sys
import os

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

from netCDF4 import Dataset
import netCDF4 as nc
import gzip
import shutil
import tempfile

import scipy.ndimage as ndi
from scipy import ndimage
from sklearn_som.som import SOM

class NetCDFHandler:
    def __init__(self, filename):
        self.filename = filename
        self.ncfile = None
        self.tempfile_name = None

    def __enter__(self):
        self.ncfile = self._open_netcdf()
        return self.ncfile

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ncfile is not None:
            self.ncfile.close()
        if self.tempfile_name is not None and os.path.exists(self.tempfile_name):
            os.unlink(self.tempfile_name)

    def _open_netcdf(self):
        if self.filename.endswith(".gz"):
            return self._open_gzipped_netcdf()
        else:
            return nc.Dataset(self.filename, 'r')

    def _open_gzipped_netcdf(self):
        with gzip.open(self.filename, 'rb') as gzfile:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(gzfile, tmp)
                self.tempfile_name = tmp.name
        return nc.Dataset(self.tempfile_name, 'r')

def inspect_array(var,isplot):
    """inspect type and shape of a given array"""
    frame = inspect.currentframe().f_back
    try:
        var_name = None
        for name, value in frame.f_locals.items():
            if value is var:
                var_name = name
                break
        print(f"{var_name}: type={type(var)}, shape={var.shape}")
        if isplot and var.ndim == 2:
            plt.imshow(var)
            plt.title(var_name)
            plt.colorbar()
            plt.show()
    finally:
        del frame

def inspect_dictionary(dict):
    """inspect type and shape of a given dictionary"""
    frame = inspect.currentframe().f_back
    try:
        for key, value in dict.items():
            print(f"Key: {key}, Type: {type(value).__name__}")        
    finally:
        del frame

def normalize_array(arr):
    # Calculate the standard deviation of the array
    std = ma.std(arr)
    mean = ma.mean(arr)
    
    # Normalize the array
    normalized_arr = ma.masked_array((arr-mean)/std,mask=arr.mask)
    
    return normalized_arr
        
def extract_month_data(data, month, year):
    # Calculate the start and end day for the specified month
    start_day = sum(calendar.monthrange(year, i)[1] for i in range(1, month))
    end_day = start_day + calendar.monthrange(year, month)[1]

    # Extract data for the specified month (0-based indexing)
    month_data = data[start_day:end_day, :, :]
    
    return month_data

def gaussian_kernel(size: int, sigma: float, mask):
    """
    Generate a 2D Gaussian kernel.
    
    Parameters:
        size: The size of the kernel (odd number).
        sigma: The standard deviation of the Gaussian kernel.
        mask: 2D boolean array of size*size
    
    Returns:
        A 2D Gaussian kernel with land mask.
    """
    # Generate a 1D Gaussian kernel
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Apply boolean mask to the Gaussian kernel
    kernel_mskd = masked_data = np.ma.masked_array(kernel, mask)
    
    # Normalize the kernel to make it sum to 1
    kernel = kernel_mskd / np.sum(kernel_mskd)
    return kernel

def convolution_2D(data,ksize=5,sigma=1.0):
    """
    Applies padding to a 2D numpy masked array while maintaining masked elements.

    Parameters:
        ksize (int): Kernel size (assumes odd value).
        sigma (float): Standard deviation of the Gaussian kernel (not used here, but retained for flexibility).
        data (np.ma.MaskedArray): Input 2D masked array.

    Returns:
        np.ma.MaskedArray: Padded masked array.
    """
    pad = ksize // 2  # Calculate padding size
    
    # Pad the data and the mask separately
    padded_data = np.pad(data.data, pad_width=pad, mode='constant', constant_values=data.fill_value)
    padded_mask = np.pad(data.mask, pad_width=pad, mode='constant', constant_values=True)
    
    # Combine them back into a masked array
    data_padded = np.ma.masked_array(padded_data, mask=padded_mask, fill_value=data.fill_value)    
    
    # Convolution to interior indecies

    convolved_data = np.zeros_like(data)
    
    rows, cols = data.shape
    for i in range(pad, rows + pad -1):
        for j in range(pad, cols + pad -1):
            if data_padded.mask[i, j]:
                convolved_value = data_padded[i,j]
            else:
                # Extract the local region around the current pixel
                local_data = data_padded[i-pad:i+pad+1,j-pad:j+pad+1]
                local_mask = local_data.mask
                kernel = gaussian_kernel(size=ksize, sigma=sigma, mask=local_mask)
                convolved_value = np.sum(local_data * kernel)

            convolved_data[i-pad,j-pad] = convolved_value
    
    return convolved_data

def iterate_convolution_2D(data,ksize=5,sigma=1.0,iterations=5):
    smoothed_data = data.copy()
    for _ in range(iterations):
        # Apply Gaussian filter while preserving land mask                                                                                      
        smoothed_data = convolution_2D(smoothed_data,ksize,sigma)
    return smoothed_data

def create_interpolated_colormap(color_list, nclss):
    """
    Creates a colormap by interpolating the base colors to fit nclss.
    
    Parameters:
    - color_list: List of base colors to interpolate.
    - nclss: int, number of desired colors in the colormap.
    
    Returns:
    - cmap_var: A LinearSegmentedColormap with nclss colors.
    """
    # Create a colormap with interpolated colors
    cmap_var = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=nclss)
    return cmap_var

def plot_array(ax, a, title=None, cmap=None):
    if title is None:
        im = ax.imshow(a)
    else:
        im = ax.imshow(a,cmap=cmap)
    ax.figure.colorbar(im,ax=ax)
    if title is not None:
        ax.set_title(title)
        
def plot_cluster_TP2(ax,lons_som,lats_som,clss_som,title=None):
    from mpl_toolkits.basemap import Basemap, cm
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cmx
    import pylab as pl

    # count number of classes
    
    unique_values = np.unique(clss_som) # skip land mask
    nclss  = len(unique_values)

    print("number of classes:",nclss)
    
    # define colormap

    # Base color list
    base_colors = [
        'tab:orange', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'mediumblue', 'darkcyan',
        'darksalmon', 'khaki', 'thistle'
    ]

    # Generate the interpolated colormap
    cmap_var = create_interpolated_colormap(base_colors, nclss)
    
    #fig, ax = plt.subplots()

    #define projection    
    m = Basemap(projection='npaeqd',resolution='c',boundinglat=50.0,lon_0=0,ax=ax)
    #m = Basemap(projection='cass',resolution='i',llcrnrlon=-20,llcrnrlat=55,urcrnrlon=60,urcrnrlat=75,lon_0=0,lat_0=70)
    
    # plot ecoregion on TP2 domain                                                                                                                                      
    x_plot, y_plot = m(lons_som,lats_som)
    cs = m.scatter(x_plot,y_plot,s=4,c=clss_som,edgecolors='none',marker='o',alpha=1.0,cmap=cmap_var,vmin=0.5,vmax=float(nclss)+0.5)

    # colorbar settings
    cb = m.colorbar(cs,location='right',pad='13%')
    cb.set_ticks([float(i) for i in range(1, nclss + 1)])

    # draw coastlines, country boundaries, fill continents.                                                                                                             
    m.drawcoastlines(linewidth=0.5,color='dimgray')
    m.fillcontinents(color='whitesmoke',lake_color='whitesmoke')

    # draw meridians and parallelss                                                                                                                                     
    lbs_lon=[1, 0, 0, 1]
    lbs_lat=[0, 1, 1, 0]
    m.drawmeridians(range(-180,180,20),labels=lbs_lon,color='gray');
    m.drawparallels(range(-90,90,10),labels=lbs_lat,color='gray');

    # title
    if title is not None:
        plt.title(title,y=1.06)

#    # save figure
#    if outfile is not None:
#        pl.savefig(outfile,dpi=300)
#        
#    plt.show()
    #plt.clf()

def extract_and_discretize_partial_colormap(cmap_name, start=0.0, end=1.0, n_colors=10):
    """
    Extract a partial colormap and discretize it into a given number of colors.
    
    Parameters:
        cmap_name: str, name of the existing colormap (e.g., 'viridis', 'plasma').
        start: float, start of the range (0.0 to 1.0).
        end: float, end of the range (0.0 to 1.0).
        n_colors: int, number of discrete colors in the new colormap.
    
    Returns:
        A discretized colormap object.
    """
    base_cmap = plt.get_cmap(cmap_name)
    # Sample evenly from the partial range
    colors = base_cmap(np.linspace(start, end, n_colors))
    # Create a discretized colormap
    discrete_cmap = ListedColormap(colors, name=f"{cmap_name}_partial_{n_colors}")

    return discrete_cmap

def plot_ecoregion_TP2(ax,lons_som,lats_som,ecor_som,title=None):
    from mpl_toolkits.basemap import Basemap, cm
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cmx
    import pylab as pl

    isdebug=False

    # define colormap

    unique_values = np.unique(ecor_som.data)
    count_unique = len(unique_values) # we do not count land mask (=0)
    #count_unique = len(unique_values) - 1 # we do not count land mask (=0)
    if isdebug:
        print(unique_values)
        print(count_unique)
    cmap_var = extract_and_discretize_partial_colormap('nipy_spectral', start=0.1, end=0.9, n_colors=count_unique)
    
    #define projection    
    m = Basemap(projection='npaeqd',resolution='i',boundinglat=50.0,lon_0=0,ax=ax)
    
    # plot ecoregion on TP2 domain                                                                                                                                      
    x_plot, y_plot = m(lons_som,lats_som)
    cs = m.scatter(x_plot,y_plot,s=4,c=ecor_som,edgecolors='none',marker='o',alpha=1.0,cmap=cmap_var,vmin=0.5,vmax=float(count_unique)+0.5)

    # colorbar settings
    cb = m.colorbar(cs,location='right',pad='13%')
    cb.set_ticks([float(i) for i in range(1, count_unique + 1)])

    # draw coastlines, country boundaries, fill continents.                                                                                                             
    #m.drawcoastlines(linewidth=0.5,color='dimgray')
    m.fillcontinents(color='whitesmoke',lake_color='whitesmoke')

    # draw meridians and parallelss                                                                                                                                     
    lbs_lon=[1, 0, 0, 1]
    lbs_lat=[0, 1, 1, 0]
    m.drawmeridians(range(-180,180,20),labels=lbs_lon,color='gray');
    m.drawparallels(range(-90,90,10),labels=lbs_lat,color='gray');

    # title
    if title is not None:
        plt.title(title,y=1.06)

#    # save figure
#    if outfile is not None:
#        pl.savefig(outfile,dpi=300)
        
#    plt.show()
    #plt.clf()

# Function to perform the majority filter while preserving land mask                                                                            
# Moving window size can be changed with the size parameter, e.g. size=5 meaning 5x5 pixels                                                     
def majority_filter_preserve_land(data, mask, size=5,pad_value=-9999, land_value=1.e+20):
    pad_size = size // 2
    padded_data = np.pad(data, pad_size, mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, pad_size, mode='constant', constant_values=True)
    result = data.copy()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #print(i,j,mask[i,j])                                                                                                               
            if mask[i, j]:  # Skip land pixels                                                                                                  
                result[i, j] = land_value
            else:
                # Calculate indices for centered window                                                                                         
                start_i = i - size//2 + pad_size
                end_i = i + size//2 + pad_size + 1
                start_j = j - size//2 + pad_size
                end_j = j + size//2 + pad_size + 1

                window = padded_data[start_i:end_i, start_j:end_j]
                window_mask = padded_mask[start_i:end_i, start_j:end_j]

                # Filter out land pixels from window                                                                                            
                non_land_pixels = window[window_mask == False]

                # Exclude padding values                                                                                                        
                valid_pixels = non_land_pixels[non_land_pixels != pad_value]

                # Find the most common value among valid pixels in the window                                                                   
                values, counts = np.unique(valid_pixels, return_counts=True)
                majority_value = values[np.argmax(counts)]
                result[i, j] = majority_value
    return result
    
# Function to perform smoothing multiple times while preserving land mask                                                                       
def apply_smoothing_preserve_land(data, mask, iterations, size=5):
    smoothed_data = data.copy()
    for _ in range(iterations):
        # Apply majority filter while preserving land mask                                                                                      
        smoothed_data = majority_filter_preserve_land(smoothed_data, mask, size)
    return smoothed_data

