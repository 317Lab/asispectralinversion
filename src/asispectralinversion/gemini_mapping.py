from gemini3d.grid.convert import geog2geomag
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy
import os

asifn = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/6UT/grouped_54geodetic_Q_E0.h5'
outdir = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/6UT/grouped_54/gemini_out/'

with h5py.File(asifn,"r") as h5:
    lon = h5['Geodetic Longitude'][:]
    lat = h5['Geodetic Latitude'][:]
    Q = h5['Q'][:]
    E0 = h5['E0'][:]
    
    print("Lat Shape: ")
    print(lat.shape)
    
    print("Lon: ")
    print(lon.shape)
    
    print("Q shape: ")
    print(Q.shape)
    
    print("E0 shape: ")
    print(E0.shape)

# Call pygemini function for converting lon and lat from geodetic to gemini's magnetic coords
print("Converting from geodetic coordinates to GEMINI's internal magnetic coordinate system...")
#gemini_mag_lon, gemini_mag_lat = geog2geomag(grid_lon, grid_lat)
phi, theta = geog2geomag(lon, lat)

# Q, E0 mag lat, lon sites (for source data)
gemini_mag_lon = phi * 180 / np.pi
gemini_mag_lat = 90 - (theta * 180 / np.pi)
    
# Grid sampling steps / Creation of target grid/set of locations
mloni = np.linspace(gemini_mag_lon.min(), gemini_mag_lon.max(), Q.shape[0])
mlati = np.linspace(gemini_mag_lat.min(), gemini_mag_lat.max(), Q.shape[1])
MLONi, MLATi = np.meshgrid(mloni, mlati, indexing="ij")

gemini_mag_lat_flat = gemini_mag_lat.flatten()
gemini_mag_lon_flat = gemini_mag_lon.flatten()
Q_flat = Q.flatten()
E0_flat = E0.flatten()

Q_gridding = scipy.interpolate.griddata((gemini_mag_lon_flat, gemini_mag_lat_flat), Q_flat, (MLONi, MLATi), fill_value=0)
E0_gridding = scipy.interpolate.griddata((gemini_mag_lon_flat, gemini_mag_lat_flat), E0_flat, (MLONi, MLATi), fill_value=0)    

# Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI
plt.title('Map of Q in GEMINI Geomagnetic Coordinates')
plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, Q, cmap='plasma')
plt.colorbar(label = 'mW/m$^2$')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
Q_fn = 'Q_geomag_gemini.png' 
Q_out = os.path.join(outdir, Q_fn)
plt.savefig(Q_out, dpi=300, bbox_inches='tight')
plt.close()

plt.title('Map of E0 in GEMINI Geomagnetic Coordinates')
plt.pcolormesh(gemini_mag_lon, gemini_mag_lat, E0, cmap='viridis')
plt.colorbar(label = 'eV')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
E0_fn = 'E0_geomag_gemini.png' 
E0_out = os.path.join(outdir, E0_fn)
plt.savefig(E0_out, dpi=300, bbox_inches='tight')
plt.close()

# Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI - Regridded
plt.title('Map of Q in GEMINI Geomagnetic Coordinates - Regridded')
plt.pcolormesh(MLONi, MLATi, Q_gridding, cmap='plasma')
plt.colorbar(label = 'mW/m$^2$')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
Q_fn_rg = 'Q_geomag_gemini_regridded.png' 
Q_out_rg = os.path.join(outdir, Q_fn_rg)
plt.savefig(Q_out_rg, dpi=300, bbox_inches='tight')
plt.close()

plt.title('Map of E0 in GEMINI Geomagnetic Coordinates - Regridded')
plt.pcolormesh(MLONi, MLATi, E0_gridding, cmap='viridis')
plt.colorbar(label = 'eV')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
E0_fn_rg = 'E0_geomag_gemini_regridded.png' 
E0_out_rg = os.path.join(outdir, E0_fn_rg)
plt.savefig(E0_out_rg, dpi=300, bbox_inches='tight')
plt.close()

ilatmin = np.argmin(abs(mlati - 63.5))
ilatmax = np.argmin(abs(mlati - 68.5))

ilonmin=np.argmin(abs(mloni - 250.5))
ilonmax = np.argmin(abs(mloni - 266.5))

Q_out = Q_gridding[ilonmin:ilonmax, ilatmin:ilatmax]
E0_out = E0_gridding[ilonmin:ilonmax, ilatmin:ilatmax]
lat_out = mlati[ilatmin:ilatmax]
lon_out = mloni[ilonmin:ilonmax]

# Troubleshooting Plots - Visulaziation of what is about to get fed into GEMINI - Regridded
plt.title('Map of Q in GEMINI Format')
plt.pcolormesh(lon_out, lat_out, Q_out.transpose(), cmap='plasma', shading="gouraud")
plt.colorbar(label = 'mW/m$^2$')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
Q_fn_gemini = 'Q_gemini_final.png'
Q_out_gemini = os.path.join(outdir, Q_fn_gemini)
plt.savefig(Q_out_gemini, dpi=300, bbox_inches='tight')
plt.close()

plt.title('Map of E0 in GEMINI Format')
plt.pcolormesh(lon_out, lat_out, E0_out.transpose(), cmap='viridis', shading="gouraud")
plt.colorbar(label = 'eV')
plt.xlabel('Geomagnetic Longitude')
plt.ylabel('Geomagnetic Latitude')
E0_fn_gemini = 'E0_gemini_final.png' 
E0_out_gemini = os.path.join(outdir, E0_fn_gemini)
plt.savefig(E0_out_gemini, dpi=300, bbox_inches='tight')
plt.close()

print("Writing/saving file for Q, E0, SigP, and SigH in GEMINI's geomagnetic coordinates...")

out_fn = "gemini_Q_E0_mlat_mlon.h5"

with h5py.File(outdir + out_fn, "w") as hdf:
    hdf.create_dataset("MLAT", data=lat_out)
    hdf.create_dataset("MLON", data=lon_out)
    hdf.create_dataset("Q", data=Q_out)
    hdf.create_dataset("E0", data=E0_out)