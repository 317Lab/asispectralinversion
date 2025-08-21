import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import os
from apexpy import Apex
from PIL import Image
from .preprocessing import common_grid, interpolate_reggrid, background_brightness, wavelet_denoise, gaussian_denoise, to_rayleighs 

"""
Purpose of this script:
    - takes in ASI/GLOW information, preparing for preprocessing and inversion
    - runs preprocessing and inversion functions
    - returns Q, E0, SigmaP, and SigmaH in regularized, geomagnetic coordinates
"""



def prepare_data(dtdate, redimgs, greenimgs, blueimgs, skymap_file, plot=True):
    """
    Purpose: 
        - prepares Q, E0, SigP, and SigH given ASI data
    """
    # Main inversion??
    
    print("Pulling information from data files and lookup tables...")

    #dtdate = datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])) # creating datetime object from given date
    # These should be function parameters
    blur_deg_EW = 0.4 # gaussian blur width in degrees maglon
    blur_deg_NS = 0.04 # gaussian blur width in degrees maglat
    n_shifts = 50 # integer determining shift-invariance of wavelets
    background_method = 'corners' # set to 'patches' or 'corners'

    ## Load lookup tables
    #v = load_lookup_tables_directory(folder, maglatsite)
    
    # Load PNGs
    redims = list()
    for src_file in redimgs:
        with Image.open(src_file) as img:
            redims.append(np.array(img))

    greenims = list()
    for src_file in greenimgs:
        with Image.open(src_file) as img:
            greenims.append(np.array(img))

    blueims = list()
    for src_file in blueimgs:
        with Image.open(src_file) as img:
            blueims.append(np.array(img))


    # Load skymap file
    with h5py.File(skymap_file, 'r') as h5:
        skymapred = [h5['/magnetic_footpointing/180km/lat'][:],
                     h5['/magnetic_footpointing/180km/lon'][:]]
        skymapgreen = [h5['/magnetic_footpointing/110km/lat'][:],
                       h5['/magnetic_footpointing/110km/lon'][:]]
        skymapblue = [h5['/magnetic_footpointing/107km/lat'][:],
                      h5['/magnetic_footpointing/107km/lon'][:]]

    # Coadd images
    redimcoadd = sum(redims)/len(redims)
    greenimcoadd = sum(greenims)/len(greenims)
    blueimcoadd = sum(blueims)/len(blueims)

    # Plot coadded images
    if plot:
        plt.imshow(redimcoadd)
        plt.title('Red Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        #red_fn = 'red_imagery.png'
        ##red_out = os.path.join(group_outdir, red_fn)
        #plt.savefig(red_fn)
        #plt.close()
        
        plt.imshow(greenimcoadd)
        plt.title('Green Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        #green_fn = 'green_imagery.png'
        ##green_out = os.path.join(group_outdir, green_fn)
        #plt.savefig(green_fn)
        #plt.close()
        
        plt.imshow(blueimcoadd)
        plt.title('Blue Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        #blue_fn = 'blue_imagery.png'
        ##blue_out = os.path.join(group_outdir, blue_fn)
        #plt.savefig(blue_fn)
        #plt.close()

        plt.show()



    # Define masks where image not defined
    bmask = np.isnan(skymapblue[0])
    gmask = np.isnan(skymapgreen[0])
    rmask = np.isnan(skymapred[0])




    # Calculate background brightness
    bluebgbright, sig = background_brightness(blueimcoadd, bmask)
    greenbgbright, sig = background_brightness(greenimcoadd, gmask)
    redbgbright, sig = background_brightness(redimcoadd, rmask)

    # Calculate new magnetic grid

    # Map everything to magnetic coordinates?
    A = Apex(date = dtdate)
    bmlat, bmlon = A.geo2apex(skymapblue[0], skymapblue[1], height=107)
    gmlat, gmlon = A.geo2apex(skymapgreen[0], skymapgreen[1], height=110)
    rmlat, rmlon = A.geo2apex(skymapred[0], skymapred[1], height=180)

    # Define common, regular grid
    gridmlat, gridmlon = common_grid(bmlat, bmlon, gmlat, gmlon, rmlat, rmlon)
    # Footpoint new grid (this is ONLY needed for the internal plotting done in this function)
    lat0, lon0, _ = A.apex2geo(gridmlat, gridmlon, height=110)


    # Interpolate images to new magnetic grid
    blueimreg = interpolate_reggrid(blueimcoadd, bmlon, bmlat, gridmlon, gridmlat)
    greenimreg = interpolate_reggrid(greenimcoadd, gmlon, gmlat, gridmlon, gridmlat)
    redimreg = interpolate_reggrid(redimcoadd, rmlon, rmlat, gridmlon, gridmlat)


    # Plot Regridded Images
    if plot:
        plt.pcolormesh(lon0, lat0, redimreg)
        plt.title('Red Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(lon0, lat0, greenimreg)
        plt.title('Green Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(lon0, lat0, blueimreg)
        plt.title('Blue Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()


    # Grid steps for our new footpointed grid - the new grid is very nearly Cartesian in footlat/footlon
    dlon = np.mean(np.diff(gridmlon, axis=0))
    dlat = np.mean(np.diff(gridmlat, axis=1))


    # Wavelet Denoise
    blueimdenoise = wavelet_denoise(blueimreg, dlat, dlon, bluebgbright, nshifts=30)
    greenimdenoise = wavelet_denoise(greenimreg, dlat, dlon, greenbgbright, nshifts=30)
    redimdenoise = wavelet_denoise(redimreg, dlat, dlon, redbgbright, nshifts=30)

    # Plot Wavelet Denoise Images
    if plot:
        plt.pcolormesh(lon0, lat0, redimdenoise)
        plt.title('Red Wavelet Denoise')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(lon0, lat0, greenimdenoise)
        plt.title('Green Wavelet Denoise')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(lon0, lat0, blueimdenoise)
        plt.title('Blue Wavelet Denoise')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()


    # Gaussian Denoise
    blueimdenoise = gaussian_denoise(blueimdenoise, dlat, dlon, bluebgbright, EW_deg=blur_deg_EW, NS_deg=blur_deg_NS)

    # Plot Gaussian Denoise Images
    if plot:
        plt.pcolormesh(lon0, lat0, blueimdenoise)
        plt.title('Blue Gaussian Denoise')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()


    ##########
    ngreen = (1 / np.std(greenimreg[np.where(~np.isnan(greenimreg))])) ** (6.5 / 8)
    nred = (1 / np.std(redimreg[np.where(~np.isnan(redimreg))])) ** (6.5 / 8)
    nblue = (1 / np.std(blueimreg[np.where(~np.isnan(blueimreg))])) ** (6.5 / 8)

    greenframe = np.copy(greenimreg)
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimreg)
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimreg)
    redframe[np.where(np.isnan(redframe))] = redbgbright

    greenmin = np.amin(greenframe)
    bluemin = np.amin(blueframe)
    redmin = np.amin(redframe)

    colormat = np.asarray([nred * (redframe - redmin), ngreen * (greenframe - greenmin), nblue * (blueframe - bluemin)]).astype(float)
    maxbright = np.amax(colormat)
    colormat /= maxbright

    if plot:
        plt.pcolormesh(lon0, lat0, colormat.transpose(1,2,0))
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()

   
    redray, greenray, blueray = to_rayleighs(redimdenoise, greenimdenoise, blueimdenoise, redbgbright, greenbgbright, bluebgbright)
    
    badrange = np.where(np.isnan(redray + blueray + greenray))
    redray[badrange] = np.nan
    greenray[badrange] = np.nan
    blueray[badrange] = np.nan

    negatives = np.zeros_like(blueray)
    negatives[np.where( (redray < 0) | (blueray < 0) | (greenray < 0) )] = 1
    negatives[np.where(np.isnan(blueray))] = np.nan

    redray[np.where(redray < 0)] = 0
    greenray[np.where(greenray < 0)] = 0
    blueray[np.where(blueray < 0)] = 0
    
    print("Decimating images...")

    dec = 2 # 'dec = 2' returns given resolution

    redraydec = redray[::dec, ::dec]
    blueraydec = blueray[::dec, ::dec]
    greenraydec = greenray[::dec, ::dec]

    greenframe = np.copy(greenimdenoise)[::dec, ::dec]
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimdenoise)[::dec, ::dec]
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimdenoise)[::dec, ::dec]
    redframe[np.where(np.isnan(redframe))] = redbgbright

    colormat = np.asarray([nred * (redframe - redmin), ngreen * (greenframe - greenmin), nblue * (blueframe - bluemin)]).astype(float)
    colormat /= maxbright

    maglon_dec = gridmlon[::dec, ::dec]
    maglat_dec = gridmlat[::dec, ::dec]

    if plot:
        plt.pcolormesh(maglon_dec, maglat_dec, colormat.transpose(1,2,0))
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()


    return redraydec, greenraydec, blueraydec, maglon_dec, maglat_dec


