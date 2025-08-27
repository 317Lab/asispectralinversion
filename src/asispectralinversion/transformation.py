import numpy as np
import scipy
from apexpy import Apex
from scipy.interpolate import griddata, NearestNDInterpolator
import matplotlib.pyplot as plt
from .preparation import prepare_data
from .inversion import load_lookup_tables_directory, calculate_E0_Q_v2, calculate_E0_Q_v2_RGonly, calculate_Sig
import os
import h5py
import sys

if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files


"""
Purpose of this script:
    - takes energies and conductivities in geomagnetic coordinates on a regularized grid,
      interpolating over NaN'd areas to handle non-physical GLOW results
    - smooths over the interpolated area to handle coarse/sharp gradients that may have been
      brought about by interpolating
    - sets up a grid space for the actual area of interest for the data
    - takes interpolated, smoothed energies and conductivities in geomagnetic
      coordinates and transforms them onto a non-regularized geodetic grid
    - regularizes the data onto a regularized geodetic grid
    
Function steps:
    1. Interpolation - in geomagnetic coordinates to fill in NaNs
    2. Smoothing - smooths over interpolated area
    3. Writing - writes output data file for geomag data
    4. Coordinate Transformation - going from apex geomag to geodetic
    5. Regularize Grid - de-warps the geodetic data onto a regular grid
    6. Writing - writes output data file for geodetic data
"""

def interp_data_nans(inarray):
    mask = np.where(~np.isnan(inarray))
    interp = NearestNDInterpolator(np.transpose(mask), inarray[mask])
    filled = inarray.copy()
    unmask = np.where(np.isnan(inarray))
    filled[unmask] = interp(np.transpose(unmask))
    return filled



def interp_data_zeros(inarray):
    mask = np.nonzero(inarray)
    interp = NearestNDInterpolator(np.transpose(mask), inarray[mask])
    filled = inarray.copy()
    unmask = np.where(inarray == 0)
    filled[unmask] = interp(np.transpose(unmask))
    return filled



def smooth_data(inarray):

    # Set smoothing factors for smoothing process (setting either >1 distorts things)
    smooth_1 = 0.5 # smoothing factor towards NaNs
    smooth_2 = 0.1 # smoothing factor away from NaNs
    outarray = scipy.ndimage.gaussian_filter(inarray, sigma=(smooth_1, smooth_2))

    return outarray



# Interpolate to a regular geodetic grid
def regularize_data(dtdate, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth):
    """
    Purpose:
          - regularizes inverted, interpolated, smoothed, and transformed ASI data onto a regularized geodetic grid
    """

    # Set up Apex object info
    apex_object = Apex(date=dtdate)
    # Perform geomagnetic to geodetic coordinate conversion for whole data set
    geo_lat_grid, geo_lon_grid = apex_object.convert(maglat_dec, maglon_dec, 'apex', 'geo', height=110)

    print("Putting geodetic data onto a regular geodetic grid...")
    
    # Create regular grid space
    lat_min, lat_max = geo_lat_grid.min(), geo_lat_grid.max()
    lon_min, lon_max = geo_lon_grid.min(), geo_lon_grid.max()
    grid_lat, grid_lon = np.mgrid[lat_min:lat_max:256j, lon_min:lon_max:256j]
    
    # Interpolate/Regularize filtered data onto the regular grid
    geo_lat_grid_flat = geo_lat_grid.flatten()
    geo_lon_grid_flat = geo_lon_grid.flatten()
    Q_smooth_flat = Q_smooth.flatten()
    E0_smooth_flat = E0_smooth.flatten()
    SigP_smooth_flat = SigP_smooth.flatten()
    SigH_smooth_flat = SigH_smooth.flatten()
    
    Q_reg = griddata((geo_lat_grid_flat, geo_lon_grid_flat), Q_smooth_flat, (grid_lat, grid_lon), method='cubic')
    E0_reg = griddata((geo_lat_grid_flat, geo_lon_grid_flat), E0_smooth_flat, (grid_lat, grid_lon), method='cubic')
    SigP_reg = griddata((geo_lat_grid_flat, geo_lon_grid_flat), SigP_smooth_flat, (grid_lat, grid_lon), method='cubic')
    SigH_reg = griddata((geo_lat_grid_flat, geo_lon_grid_flat), SigH_smooth_flat, (grid_lat, grid_lon), method='cubic')
    
    geo_lat_grid = geo_lat_grid[~np.isnan(geo_lat_grid)]
    geo_lon_grid = geo_lon_grid[~np.isnan(geo_lon_grid)]
    
    return grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg



def write_output(dtdate, gdlon, gdlat, Qgd, E0gd, SigPgd, SigHgd, gmlon, gmlat, Qgm, E0gm, SigPgm, SigHgm, out_fn):
    
   print("Writing/saving file for Q, E0, SigP, and SigH in geodetic coordinates...")
    
   #out_fn = "geodetic_Q_E0.h5"
    
   with h5py.File(out_fn, "w") as hdf:
        hdf.create_group("Geodetic")
        hdf.create_dataset("Geodetic/Longitude", data=gdlon)
        hdf.create_dataset("Geodetic/Latitude", data=gdlat)
        hdf.create_dataset("Geodetic/Q", data=Qgd)
        hdf.create_dataset("Geodetic/E0", data=E0gd)
        hdf.create_dataset("Geodetic/SigP", data=SigPgd)
        hdf.create_dataset("Geodetic/SigH", data=SigHgd)
    
        hdf.create_group("Geomagnetic")
        hdf.create_dataset("Geomagnetic/Longitude", data=gmlon)
        hdf.create_dataset("Geomagnetic/Latitude", data=gmlat)
        hdf.create_dataset("Geomagnetic/Q", data=Qgm)
        hdf.create_dataset("Geomagnetic/E0", data=E0gm)
        hdf.create_dataset("Geomagnetic/SigP", data=SigPgm)
        hdf.create_dataset("Geomagnetic/SigH", data=SigHgm)

        hdf.create_dataset("TimeStamp", data=dtdate.isoformat())
    

   
def feed_data(dtdate, foi_0428, foi_0558, foi_0630, folder, output_file, method='rgb', skymap_file=None, plot=False, prep_kwarg=dict()):
    # This generates a SINGLE output file
    """
    Purpose:
        - pipeline for feeding processed, inverted ASI data into the series of functions in this script
    Input:
        dtdate: datetime.date
            Date the inversion is calculated on
        foi_0428: list of str
            List of filenames (png) of 0428 images to include in the inversion
        foi_0558: list of str
            List of filenames (png) of 0558 images to include in the inversion
        foi_0630: list of str
            List of filenames (png) of 0630 images to include in the inversion
        folder: str
            Directory where inversion lookup tables generated by glow are saved
        output_file: str
            Output file name and path
        method: str
            Method to use for inversion, either 'rgb' (default) for standard method or 'rg' for only using red/green images
        skymap_file: str (optional)
            Path to skymap file when not using default (PKR)
        plot: bool (optional)
            Whether or not to generate intermediate plots (default=False)
        prep_kwarg: dict
            Additional optional keyword arguments for prepare_data()
    """
    
    print("")

    # Call prepare_data from preparation.py to get the necessary inputs for everything in this script

    if not skymap_file:
        print("Skymap file not specified! Using default for Poker Flat.")
        skymap_file = files('asispectralinversion').joinpath('skymap.mat')

    print("Prepare Data")

    # Requires skymap file - site specific
    redraydec, greenraydec, blueraydec, maglon_dec, maglat_dec = prepare_data(dtdate, foi_0630, foi_0558, foi_0428, skymap_file, plot=plot, **prep_kwarg)

    # Approximate site magnetic latitude with avarage of mlat grid
#    maglatsite = np.mean(maglat_dec)

    # Load lookup tables
    # These are generated by seperate FORTRAN program - site, time, and geomag condition specific
    v = load_lookup_tables_directory(folder, plot=plot)
    
    print("Calculating Q and E0...")
    
    if method == 'rgb':
        qout, e0out = calculate_E0_Q_v2(redraydec, greenraydec, blueraydec, v, minE0=150, generous=True, plot=plot)
    elif method == 'rg':
        qout, e0out = calculate_E0_Q_v2_RGonly(redraydec, greenraydec, blueraydec, v, minE0=150, generous=True, plot=plot)
    else:
        raise ValueError(f'Input inversion method {method} is not valid!')
    
    print("Calculating conductivities given Q and E0...")
    
    # Calculate conductivities AFTER calculating Q and E0
    SigP, SigH = calculate_Sig(qout, e0out, v, generous=True, plot=plot)


    # Call each of the functions in this script that talk to each other internally
    # interp_data (for NaNs)
    Q_filled = interp_data_nans(qout)
    E0_filled = interp_data_nans(e0out)
    SigP_filled = interp_data_nans(SigP)
    SigH_filled = interp_data_nans(SigH)
    
    # interp again (for zeros)
    Q_filled = interp_data_zeros(Q_filled)
    E0_filled = interp_data_zeros(E0_filled)
    SigP_filled = interp_data_zeros(SigP_filled)
    SigH_filled = interp_data_zeros(SigH_filled)
    
    # smooth_data
    Q_smooth = smooth_data(Q_filled)
    E0_smooth = smooth_data(E0_filled)
    SigP_smooth = smooth_data(SigP_filled)
    SigH_smooth = smooth_data(SigH_filled)
    
    # regularize data
    grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg = regularize_data(dtdate, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth)

    # write output file
    write_output(dtdate, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth, output_file)
   
    print("Returning all necessary data to funnel into Lompe and GEMINI...")
    return dtdate, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg
