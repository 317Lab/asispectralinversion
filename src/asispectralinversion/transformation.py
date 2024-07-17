import numpy as np
import scipy
from scipy.interpolate import NearestNDInterpolator
from apexpy import Apex
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from preparation import prepare_data
import os
from asi_gemini import gemini_coord_transformation
from asi_gemini import write_asi_gemini

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
"""

def interp_data(dtdate, outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH):
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
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_interp.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()

    plt.title('Map of E0 in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, E0_filled, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_interp.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()

    plt.title('Map of SigP in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, SigP_filled, cmap='magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigP_fn = 'SigP_geomag_interp.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()

    plt.title('Map of SigH in Geomagnetic Coordinates (Interpolated)')
    plt.pcolormesh(maglon_dec, maglat_dec, SigH_filled, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigH_fn = 'SigH_geomag_interp.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()

    return dtdate, outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled


def smooth_data(dtdate, outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled):
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
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_smooth.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()

    plt.title('Map of E0 in Geomagnetic Coordinates (Smoothed)')
    plt.pcolormesh(maglon_dec, maglat_dec, E0_smooth, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_smooth.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()

    plt.title('Map of SigP in Geomagnetic Coordinates (Smoothed)')
    plt.pcolormesh(maglon_dec, maglat_dec, SigP_smooth, cmap='magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigP_fn = 'SigP_geomag_smooth.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()

    plt.title('Map of SigH in Geomagnetic Coordinates (Smoothed)')
    plt.pcolormesh(maglon_dec, maglat_dec, SigH_smooth, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigH_fn = 'SigH_geomag_interp.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()

    return dtdate, outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth


def transform_trim_data(dtdate, outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth):
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
    
    # Create mask for area of interest
    filter_mask = (maglat_dec >= 64.5) & (maglat_dec <= 67.5) & (maglon_dec >= -99.5) & (maglon_dec <= -83.5)
    
    # Apply mask to data
    filtered_maglat_dec = maglat_dec[filter_mask]
    filtered_maglon_dec = maglon_dec[filter_mask]
    filtered_Q_smooth = Q_smooth[filter_mask]
    filtered_E0_smooth = E0_smooth[filter_mask]
    filtered_SigP_smooth = SigP_smooth[filter_mask]
    filtered_SigH_smooth = SigH_smooth[filter_mask]
    
    # Perform geomagnetic to geodetic coordinate conversion for area of interest
    filtered_geo_lat_grid, filtered_geo_lon_grid = apex_object.convert(filtered_maglat_dec, filtered_maglon_dec, 'apex', 'geo', height=110)
    
    # Troubleshooting Plots - WHOLE SPACE NON-REGULARIZED DATA IN GEODETIC COORDINATES
    plt.title('Map of Q in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, Q_smooth, cmap='plasma')
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    Q_fn = 'Q_geod_irreg.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()

    plt.title('Map of E0 in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, E0_smooth, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    E0_fn = 'E0_geod_irreg.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()

    plt.title('Map of SigP in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, SigP_smooth, cmap='magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigP_fn = 'SigP_geod_irreg.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()

    plt.title('Map of SigH in Geodetic Coordinates (Irregular Grid)')
    plt.pcolormesh(geo_lon_grid, geo_lat_grid, SigH_smooth, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigH_fn = 'SigH_geod_irreg.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()
    
    return dtdate, outdir, maglon_dec, maglat_dec, geo_lon_grid, geo_lat_grid, filtered_maglon_dec, filtered_maglat_dec, filtered_Q_smooth, filtered_E0_smooth, filtered_SigP_smooth, filtered_SigH_smooth, filtered_geo_lat_grid, filtered_geo_lon_grid


def regularize_data(dtdate, outdir, maglon_dec, maglat_dec, geo_lon_grid, geo_lat_grid, filtered_maglon_dec, filtered_maglat_dec, filtered_Q_smooth, filtered_E0_smooth, filtered_SigP_smooth, filtered_SigH_smooth, filtered_geo_lat_grid, filtered_geo_lon_grid):
    """
    Purpose:
          - regularizes inverted, interpolated, smoothed, and transformed ASI data onto a regularized geodetic grid
    """
    
    # Create regular grid space
    lat_min, lat_max = filtered_geo_lat_grid.min(), filtered_geo_lat_grid.max()
    lon_min, lon_max = filtered_geo_lon_grid.min(), filtered_geo_lon_grid.max()
    grid_lat, grid_lon = np.mgrid[lat_min:lat_max:100j, lon_min:lon_max:100j]
    
    # Interpolate/Regularize filtered data onto the regular grid
    Q_smooth_grid = griddata((filtered_geo_lat_grid, filtered_geo_lon_grid), filtered_Q_smooth, (grid_lat, grid_lon), method='cubic')
    E0_smooth_grid = griddata((filtered_geo_lat_grid, filtered_geo_lon_grid), filtered_E0_smooth, (grid_lat, grid_lon), method='cubic')
    SigP_smooth_grid = griddata((filtered_geo_lat_grid, filtered_geo_lon_grid), filtered_SigP_smooth, (grid_lat, grid_lon), method='cubic')
    SigH_smooth_grid = griddata((filtered_geo_lat_grid, filtered_geo_lon_grid), filtered_SigH_smooth, (grid_lat, grid_lon), method='cubic')
    
    filtered_geo_lat_grid = filtered_geo_lat_grid[~np.isnan(filtered_geo_lat_grid)]
    filtered_geo_lon_grid = filtered_geo_lon_grid[~np.isnan(filtered_geo_lon_grid)]
    
    print("Creating geodetic regular grid...")
    geodetic_lat_min = 63.5
    geodetic_lat_max = 68.5
    geodetic_lon_min = -102.5
    geodetic_lon_max = -80.5

    geodetic_lat_res = 0.01
    geodetic_lon_res = 0.01

    geodetic_lat_grid = np.arange(geodetic_lat_min, geodetic_lat_max + geodetic_lat_res, geodetic_lat_res)
    geodetic_lon_grid = np.arange(geodetic_lon_min, geodetic_lon_max + geodetic_lon_res, geodetic_lon_res)

    lon_grid, lat_grid = np.meshgrid(geodetic_lon_grid, geodetic_lat_grid)
    
    # Troubleshooting Plots - INTERPOLATED, SMOOTHED, AND FILTERED
    plt.title('Map of Q in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, Q_smooth_grid, cmap='plasma')
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    Q_fn = 'Q_geod_reg.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()
    
    plt.title('Map of E0 in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, E0_smooth_grid, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    E0_fn = 'E0_geod_reg.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()
    
    plt.title('Map of SigP in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, SigP_smooth_grid, cmap='magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigP_fn = 'SigP_geod_reg.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()
    
    plt.title('Map of SigH in Geodetic Coordinates')
    plt.pcolormesh(grid_lon, grid_lat, SigH_smooth_grid, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigH_fn = 'SigH_geod_reg.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()

    return dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid


def smooth_reg_data(dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid):
    """
    Purpose:
        - smoothes over the edges of the trimmed grid from the initial given ASI data
    """
    # Width of the edge regions to smooth  over
    sigma = 10
    edge_width = int(sigma)  # 3 times the sigma value
    
    # Create a mask for the edges
    edge_mask = np.zeros_like(Q_smooth_grid, dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True
    
    # Smooth each variable grid over the edges
    Q_final = Q_smooth_grid.copy()
    E0_final = E0_smooth_grid.copy()
    SigP_final = SigP_smooth_grid.copy()
    SigH_final = SigH_smooth_grid.copy()
    
    Q_final[edge_mask] = scipy.ndimage.gaussian_filter(Q_smooth_grid, sigma)[edge_mask]
    E0_final[edge_mask] = scipy.ndimage.gaussian_filter(E0_smooth_grid, sigma)[edge_mask]
    SigP_final[edge_mask] = scipy.ndimage.gaussian_filter(SigP_smooth_grid, sigma)[edge_mask]
    SigH_final[edge_mask] = scipy.ndimage.gaussian_filter(SigH_smooth_grid, sigma)[edge_mask]
    
    # Troubleshooting Plots - INTERPOLATED, SMOOTHED, FILTERED, AND EDGE-SMOOTHED
    plt.title('Map of Q in Geodetic Coordinates (Edges Smoothed)')
    plt.pcolormesh(grid_lon, grid_lat, Q_final, cmap='magma')
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    plt.title('Smoothed Q on Geodetic Grid')
    Q_fn = 'Q_final.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()
    
    plt.title('Map of E0 in Geodetic Coordinates (Edges Smoothed)')
    plt.pcolormesh(grid_lon, grid_lat, E0_final, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    E0_fn = 'E0_final.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()
    
    plt.title('Map of SigP in Geodetic Coordinates (Edges Smoothed)')
    plt.pcolormesh(grid_lon, grid_lat, SigP_final, cmap='plasma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigP_fn = 'SigP_final.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()
    
    plt.title('Map of SigH in Geodetic Coordinates (Edges Smoothed)')
    plt.pcolormesh(grid_lon, grid_lat, SigH_final, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geodetic Longitude')
    plt.ylabel('Geodetic Latitude')
    SigH_fn = 'SigH_final.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()


    return dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_final, E0_final, SigP_final, SigH_final

   
def feed_data(date, maglatsite, folder, outdir):
    """
    Purpose:
        - pipeline for feeding processed, inverted ASI data into the series of functions in this script
    """

    # Call prepare_data from preparation.py to get the necessary inputs for everything in this script
    dtdate, outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH = prepare_data(date, maglatsite, folder, outdir)
    
    # Call each of the functions in this script that talk to each other internally
    dtdate, outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled = interp_data(dtdate, outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH)
    
    dtdate, outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth = smooth_data(dtdate, outdir, maglon_dec, maglat_dec, Q_filled, E0_filled, SigP_filled, SigH_filled)
    
    dtdate, outdir, maglon_dec, maglat_dec, geo_lon_grid, geo_lat_grid, filtered_maglon_dec, filtered_maglat_dec, filtered_Q_smooth, filtered_E0_smooth, filtered_SigP_smooth, filtered_SigH_smooth, filtered_geo_lat_grid, filtered_geo_lon_grid = transform_trim_data(dtdate, outdir, maglon_dec, maglat_dec, Q_smooth, E0_smooth, SigP_smooth, SigH_smooth)
    
    dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid = regularize_data(dtdate, outdir, maglon_dec, maglat_dec, geo_lon_grid, geo_lat_grid, filtered_maglon_dec, filtered_maglat_dec, filtered_Q_smooth, filtered_E0_smooth, filtered_SigP_smooth, filtered_SigH_smooth, filtered_geo_lat_grid, filtered_geo_lon_grid)
    
    dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_final, E0_final, SigP_final, SigH_final = smooth_reg_data(dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid)
    
    dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid = gemini_coord_transformation(dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid)
    
    dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid = write_asi_gemini(dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid)
    
    print("Returning all necessary data to funnel into Lompe and GEMINI...")
    return dtdate, outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid
