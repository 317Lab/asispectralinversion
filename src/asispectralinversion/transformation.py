import numpy as np
import scipy
from scipy.interpolate import NearestNDInterpolator
from apexpy import Apex
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from .preparation import prepare_data
from .inversion import load_lookup_tables_directory, calculate_E0_Q_v2, calculate_Sig
import os
import h5py

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
    

   
def feed_data(dtdate, maglatsite, folder, foi_0428, foi_0558, foi_0630, output_file):
    # This generates a SINGLE output file
    """
    Purpose:
        - pipeline for feeding processed, inverted ASI data into the series of functions in this script
    """
    
    print("")

    # Call prepare_data from preparation.py to get the necessary inputs for everything in this script

    print("Prepare Data")

    redraydec, greenraydec, blueraydec, maglon_dec, maglat_dec = prepare_data(dtdate, foi_0630, foi_0558, foi_0428, 'test_data_20230314/skymap.mat')
    
    # Load lookup tables
    v = load_lookup_tables_directory(folder, maglatsite)
    
    print("Calculating Q and E0...")
    
    qout, e0out, minq, maxq, mine0, maxe0 = calculate_E0_Q_v2(redraydec, greenraydec, blueraydec, v, minE0 = 150, generous = True)
    
    print("Calculating conductivities given Q and E0...")
    
    # Calculate conductivities AFTER calculating Q and E0
    SigP, SigH = calculate_Sig(qout, e0out, v, generous = True)


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
    
    # regularize_data
    grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg = regularize_data(dtdate, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth)

    # write_geodetic
    write_output(dtdate, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth, output_file)
   
    print("Returning all necessary data to funnel into Lompe and GEMINI...")
    return dtdate, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg
