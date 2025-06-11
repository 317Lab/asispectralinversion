import numpy as np
import scipy
from scipy.interpolate import NearestNDInterpolator
from apexpy import Apex
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from .preparation import prepare_data
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

def interp_data_nans(dtdate, group_outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH):
    """
    Purpose:
        - finds NaNs from inverted ASI data, masking over them
        - interpolates over NaNs to fill in areas where GLOW inversion could not be performed
    """

    # Determine where NaNs are and mask over them
    print("Finding NaNs...")
    Q_mask = np.where(~np.isnan(qout))
    E0_mask = np.where(~np.isnan(e0out))
    SigP_mask = np.where(~np.isnan(SigP))
    SigH_mask = np.where(~np.isnan(SigH))
    
    # Interpolate over masked area
    print("Interpolating over NaN'd areas...")
    Q_interp = NearestNDInterpolator(np.transpose(Q_mask), qout[Q_mask])
    E0_interp = NearestNDInterpolator(np.transpose(E0_mask), e0out[E0_mask])
    SigP_interp = NearestNDInterpolator(np.transpose(SigP_mask), SigP[SigP_mask])
    SigH_interp = NearestNDInterpolator(np.transpose(SigH_mask), SigH[SigH_mask])
    
    # Make copies of the variables to maintain shapes/sizes
    Q_filled = qout.copy()
    E0_filled = e0out.copy()
    SigP_filled = SigP.copy()
    SigH_filled = SigH.copy()
    
    # Determine areas where NaNs are NOT
    print("Filling in NaNs with interpolated values...")
    Q_unmask = np.where(np.isnan(qout))
    E0_unmask = np.where(np.isnan(e0out))
    SigP_unmask = np.where(np.isnan(SigP))
    SigH_unmask = np.where(np.isnan(SigH))
    
    # Apply interpolation to fill in the NaNs
    Q_filled[Q_unmask] = Q_interp(np.transpose(Q_unmask))
    E0_filled[E0_unmask] = E0_interp(np.transpose(E0_unmask))
    SigP_filled[SigP_unmask] = SigP_interp(np.transpose(SigP_unmask))
    SigH_filled[SigH_unmask] = SigH_interp(np.transpose(SigH_unmask))
    
    # Troubleshooting plots - INTERPOLATED DATA IN REGULARIZED GEOMAGNETIC COORDINATES
    plt.title('Map of Q in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, Q_filled, cmap='plasma')
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_interp.png'
    Q_out = os.path.join(group_outdir, Q_fn)
    plt.savefig(Q_out)
    plt.close()

    plt.title('Map of E0 in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, E0_filled, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_interp.png'
    E0_out = os.path.join(group_outdir, E0_fn)
    plt.savefig(E0_out)
    plt.close()

    return dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled


def interp_data_zeros(dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled):
    """
    Purpose:
        - finds NaNs from inverted ASI data, masking over them
        - interpolates over NaNs to fill in areas where GLOW inversion could not be performed
    """

    # Determine where data points are non-zero and mask over them
    print("Finding non-zero data points...")
    Q_mask = np.nonzero(Q_filled)
    E0_mask = np.nonzero(E0_filled)
    SigP_mask = np.nonzero(SigP_filled)
    SigH_mask = np.nonzero(SigH_filled)
    
    # Interpolate over masked area
    print("Interpolating over NaN'd areas...")
    Q_interp = NearestNDInterpolator(np.transpose(Q_mask), Q_filled[Q_mask])
    E0_interp = NearestNDInterpolator(np.transpose(E0_mask), E0_filled[E0_mask])
    SigP_interp = NearestNDInterpolator(np.transpose(SigP_mask), SigP_filled[SigP_mask])
    SigH_interp = NearestNDInterpolator(np.transpose(SigH_mask), SigH_filled[SigH_mask])
    
    # Make copies of the variables to maintain shapes/sizes
    Q_filled = Q_filled.copy()
    E0_filled = E0_filled.copy()
    SigP_filled = SigP_filled.copy()
    SigH_filled = SigH_filled.copy()
    
    # Determine areas where data points are zero
    print("Filling in zero data points with interpolated values...")
    Q_unmask = np.where(Q_filled == 0)
    E0_unmask = np.where(E0_filled == 0)
    SigP_unmask = np.where(SigP_filled == 0)
    SigH_unmask = np.where(SigH_filled == 0)
    
    # Apply interpolation to fill in the NaNs
    Q_filled[Q_unmask] = Q_interp(np.transpose(Q_unmask))
    E0_filled[E0_unmask] = E0_interp(np.transpose(E0_unmask))
    SigP_filled[SigP_unmask] = SigP_interp(np.transpose(SigP_unmask))
    SigH_filled[SigH_unmask] = SigH_interp(np.transpose(SigH_unmask))
    
    # Troubleshooting plots - INTERPOLATED DATA IN REGULARIZED GEOMAGNETIC COORDINATES
    plt.title('Map of Q in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, Q_filled, cmap='plasma')
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_interp.png'
    Q_out = os.path.join(group_outdir, Q_fn)
    plt.savefig(Q_out)
    plt.close()

    plt.title('Map of E0 in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, E0_filled, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_interp.png'
    E0_out = os.path.join(group_outdir, E0_fn)
    plt.savefig(E0_out)
    plt.close()

    return dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled


def smooth_data(dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled):
    """
    Purpose:
        - smoothes over interpolated areas to handle sharp gradients brought on by interpolation
    """

    # Set smoothing factors for smoothing process (setting either >1 distorts things)
    smooth_1 = 0.5 # smoothing factor towards NaNs
    smooth_2 = 0.1 # smoothing factor away from NaNs
    
    # Smooth over interpolated area
    print("Smoothing interpolated data...")
    Q_smooth = scipy.ndimage.gaussian_filter(Q_filled, sigma=(smooth_1, smooth_2))
    E0_smooth = scipy.ndimage.gaussian_filter(E0_filled, sigma=(smooth_1, smooth_2))
    SigP_smooth = scipy.ndimage.gaussian_filter(SigP_filled, sigma=(smooth_1, smooth_2))
    SigH_smooth = scipy.ndimage.gaussian_filter(SigH_filled, sigma=(smooth_1, smooth_2))
    
    # Troubleshooting plots - INTERPOLATED AND SMOOTHED DATA IN REGULARIZED GEOMAGNETIC COORDINATES
    plt.title('Map of Q in Geomagnetic Coordinates (Smoothed)')
    plt.pcolormesh(maglon_dec, maglat_dec, Q_smooth, cmap='plasma')
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_smooth.png'
    Q_out = os.path.join(group_outdir, Q_fn)
    plt.savefig(Q_out)
    plt.close()

    plt.title('Map of E0 in Geomagnetic Coordinates (Smoothed)')
    plt.pcolormesh(maglon_dec, maglat_dec, E0_smooth, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_smooth.png'
    E0_out = os.path.join(group_outdir, E0_fn)
    plt.savefig(E0_out)
    plt.close()

    return dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth


def write_geomag(dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth):
    
    print("Writing/saving file for Q, E0, SigP, and SigH in geomagnetic coordinates...")
      
    out_fn = "magnetic_Q_E0.h5"
    
    with h5py.File(group_outdir + out_fn, "w") as hdf:
        hdf.create_dataset("Decimated Magnetic Longitude", data=maglon_dec)
        hdf.create_dataset("Decimated Magnetic Latitude", data=maglat_dec)
        hdf.create_dataset("Q", data=Q_smooth)
        hdf.create_dataset("E0", data=E0_smooth)
        hdf.create_dataset("SigP", data=SigP_smooth)
        hdf.create_dataset("SigH", data=SigH_smooth)
    
    return dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth


def transform_data(dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth):
    """
    Purpose:
        - converts interpolated, inverted ASI data from geomagnetic (apex) to geodetic coordinates
        - trims out data that falls outside of area of interest
        - transforms this data onto a geodetic, non-regularized grid
    """
    
    # Set up Apex object info
    print("Setting up Apex inputs...")
    apex_date = dtdate
    print("Making first actual Apex call...")
    apex_object = Apex(date=apex_date)
    
    # Perform geomagnetic to geodetic coordinate conversion for whole data set
    geo_lat_grid, geo_lon_grid = apex_object.convert(maglat_dec, maglon_dec, 'apex', 'geo', height=110)
    
    # Troubleshooting Plots - WHOLE SPACE NON-REGULARIZED DATA IN GEODETIC COORDINATES
    plt.title('Map of Q in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, Q_smooth, cmap='plasma')
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    Q_fn = 'Q_geod_irreg.png'
    Q_out = os.path.join(group_outdir, Q_fn)
    plt.savefig(Q_out)
    plt.close()

    plt.title('Map of E0 in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, E0_smooth, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    E0_fn = 'E0_geod_irreg.png'
    E0_out = os.path.join(group_outdir, E0_fn)
    plt.savefig(E0_out)
    plt.close()
    
    return dtdate, group_outdir, geo_lon_grid, geo_lat_grid, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth


def regularize_data(dtdate, group_outdir, geo_lon_grid, geo_lat_grid, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth):
    """
    Purpose:
          - regularizes inverted, interpolated, smoothed, and transformed ASI data onto a regularized geodetic grid
    """
    
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
    
    # Troubleshooting Plots - INTERPOLATED, SMOOTHED, TRANSFORMED, AND MAPPED ONTO A REGULAR GRID
    plt.title('Map of Q in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, Q_reg, cmap='plasma')
    plt.colorbar(label = 'mW/m$^2$')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    Q_fn = 'Q_geod_reg.png'
    Q_out = os.path.join(group_outdir, Q_fn)
    plt.savefig(Q_out)
    plt.close()
    
    plt.title('Map of E0 in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, E0_reg, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    E0_fn = 'E0_geod_reg.png'
    E0_out = os.path.join(group_outdir, E0_fn)
    plt.savefig(E0_out)
    plt.close()

    return dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg


def write_geodetic(dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg):
    
   print("Writing/saving file for Q, E0, SigP, and SigH in geodetic coordinates...")
    
   out_fn = "geodetic_Q_E0.h5"
    
   with h5py.File(group_outdir + out_fn, "w") as hdf:
        hdf.create_dataset("Geodetic Longitude", data=grid_lon)
        hdf.create_dataset("Geodetic Latitude", data=grid_lat)
        hdf.create_dataset("Q", data=Q_reg)
        hdf.create_dataset("E0", data=E0_reg)
        hdf.create_dataset("SigP", data=SigP_reg)
        hdf.create_dataset("SigH", data=SigH_reg)
    
   return dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg

   
def feed_data(date, maglatsite, folder, foi_0428, foi_0558, foi_0630, group_outdir, group_number):
    """
    Purpose:
        - pipeline for feeding processed, inverted ASI data into the series of functions in this script
    """
    
    print("")

    # Call prepare_data from preparation.py to get the necessary inputs for everything in this script
    dtdate, group_outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH = prepare_data(date, 
                                                                                         maglatsite, 
                                                                                         folder, 
                                                                                         foi_0428, 
                                                                                         foi_0558, 
                                                                                         foi_0630, 
                                                                                         group_outdir, 
                                                                                         group_number)
    
    # Call each of the functions in this script that talk to each other internally
    # interp_data (for NaNs)
    dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled = interp_data_nans(dtdate, 
                                                                                                                   group_outdir, 
                                                                                                                   maglon_dec, 
                                                                                                                   maglat_dec, 
                                                                                                                   qout, 
                                                                                                                   e0out, 
                                                                                                                   SigP, 
                                                                                                                   SigH)
    # interp again (for zeros)
    dtdate, group_outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled = interp_data_zeros(dtdate, 
                                                                                                                    group_outdir, 
                                                                                                                    maglon_dec, 
                                                                                                                    maglat_dec, 
                                                                                                                    Q_filled, 
                                                                                                                    E0_filled, 
                                                                                                                    SigP_filled, 
                                                                                                                    SigH_filled)
    # smooth_data
    dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth = smooth_data(dtdate, 
                                                                                                              group_outdir, 
                                                                                                              maglon_dec, 
                                                                                                              maglat_dec, 
                                                                                                              Q_filled, 
                                                                                                              E0_filled, 
                                                                                                              SigP_filled, 
                                                                                                              SigH_filled)
    # write_geomag
    dtdate, group_outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth = write_geomag(dtdate, 
                                                                                                               group_outdir, 
                                                                                                               maglon_dec, 
                                                                                                               maglat_dec, 
                                                                                                               Q_smooth, 
                                                                                                               E0_smooth, 
                                                                                                               SigP_smooth, 
                                                                                                               SigH_smooth)
    # transform_data
    dtdate, group_outdir, geo_lon_grid, geo_lat_grid, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth = transform_data(dtdate, 
                                                                                                                     group_outdir, 
                                                                                                                     maglon_dec, 
                                                                                                                     maglat_dec, 
                                                                                                                     Q_smooth, 
                                                                                                                     E0_smooth, 
                                                                                                                     SigP_smooth, 
                                                                                                                     SigH_smooth)
    # regularize_data
    dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg = regularize_data(dtdate, 
                                                                                                  group_outdir, 
                                                                                                  geo_lon_grid, 
                                                                                                  geo_lat_grid, 
                                                                                                  Q_smooth, 
                                                                                                  E0_smooth, 
                                                                                                  SigP_smooth, 
                                                                                                  SigH_smooth)
    # write_geodetic
    dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg = write_geodetic(dtdate, 
                                                                                                 group_outdir, 
                                                                                                 grid_lon, 
                                                                                                 grid_lat, 
                                                                                                 Q_reg, 
                                                                                                 E0_reg, 
                                                                                                 SigP_reg, 
                                                                                                 SigH_reg)
    
    print("Returning all necessary data to funnel into Lompe and GEMINI...")
    return dtdate, group_outdir, grid_lon, grid_lat, Q_reg, E0_reg, SigP_reg, SigH_reg
