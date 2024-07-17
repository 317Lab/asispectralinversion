from gemini3d.grid.convert import geog2geomag
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py


def gemini_coord_transformation(dtdate, outdir, maglon_dec, maglat_dec, grid_lon, grid_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid):
    
    # Call pygemini function for converting lon and lat from geodetic to gemini's magnetic coords
    print("Converting from geodetic coordinates to GEMINI's internal magnetic coordinate system...")
    gemini_mag_lon, gemini_mag_lat = geog2geomag(grid_lon, grid_lat)
    
    # Build grid (for plotting)
    print("Creating geomagnetic grid for variables...")
    #gemini_mag_lon_grid, gemini_mag_lat_grid = np.meshgrid(gemini_mag_lon, gemini_mag_lat)
    
    print("Lat, Lon array shapes: ")
    print("Geodetic Lat, Lon: ", grid_lat.shape, "  &  ", grid_lon.shape)
    print("GEMINI Geomagnetic Lat, Lon: ", gemini_mag_lat.shape, "  &  ", gemini_mag_lon.shape)
    

    # Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI
    plt.title('Map of Q in GEMINI Geomagnetic Coordinates')
    plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, cmap='plasma')
    #plt.pcolormesh(gemini_mag_lon_grid, gemini_mag_lat_grid, Q_smooth_grid, cmap='plasma')
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag_gemini.png' 
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()

    plt.title('Map of E0 in GEMINI Geomagnetic Coordinates')
    plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, cmap='viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag_gemini.png' 
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()

    plt.title('Map of SigP in GEMINI Geomagnetic Coordinates')
    plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, cmap='magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigP_fn = 'SigP_geomag_gemini.png' 
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()

    plt.title('Map of SigH in GEMINI Geomagnetic Coordinates')
    plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, cmap='cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigH_fn = 'SigH_geomag_gemini.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()
    
    return dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid


def write_asi_gemini(dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid):
    
    out_fn = "gemini_Q_E0.h5"
    
    with h5py.File(outdir + out_fn, "w") as hdf:
        hdf.create_dataset("gemini_mag_lon", data=gemini_mag_lon)
        hdf.create_dataset("gemini_mag_lat", data=gemini_mag_lat)
        hdf.create_dataset("Q_smooth_grid", data=Q_smooth_grid)
        hdf.create_dataset("E0_smooth_grid", data=E0_smooth_grid)
        hdf.create_dataset("SigP_smooth_grid", data=SigP_smooth_grid)
        hdf.create_dataset("SigH_smooth_grid", data=SigH_smooth_grid)
    
    return dtdate, outdir, gemini_mag_lon, gemini_mag_lat, Q_smooth_grid, E0_smooth_grid, SigP_smooth_grid, SigH_smooth_grid
