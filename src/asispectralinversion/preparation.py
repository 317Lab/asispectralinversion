import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import os
from apexpy import Apex
from .inversion import load_lookup_tables_directory
from .inversion import calculate_E0_Q_v2
from .preprocessing import wavelet_denoise_resample
from .preprocessing import gaussian_denoise_resample
from .preprocessing import to_rayleighs
from .inversion import calculate_Sig

"""
Purpose of this script:
    - takes in ASI/GLOW information, preparing for preprocessing and inversion
    - runs preprocessing and inversion functions
    - returns Q, E0, SigmaP, and SigmaH in regularized, geomagnetic coordinates
"""

#def copy_h5(vtest):
#    """
#    Purpose: 
#        - copies an HDF5 structure to a python dict recursively
#    """
#    
#    dicttest = {}
#    keyslist = list(vtest.keys())
#    for key in keyslist:
#        if type(vtest[key]) == h5py._hl.dataset.Dataset:
#            if vtest[key].shape[1] == 1:
#                if vtest[key].shape[0] == 1:
#                    dicttest[key] = vtest[key][0][0]
#                else:
#                    dicttest[key] = np.asarray(vtest[key]).flatten()
#            else:
#                dicttest[key] = np.asarray(vtest[key])
#        else:
#            dicttest[key] = copy_h5(vtest[key])
#            
#    return dicttest


#def prepare_data(date, maglatsite, folder, foi_0428, foi_0558, foi_0630, group_outdir, group_number):
from PIL import Image

def prepare_data(dtdate, redimgs, greenimgs, blueimgs, skymap_file, plot=False):
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
    background_method = 'patches' # set to 'patches' or 'corners'

    ## Load lookup tables
    #v = load_lookup_tables_directory(folder, maglatsite)
    
    # load images
    #redims = copy_h5(h5py.File(folder + '/' + foi_0630))
    #greenims = copy_h5(h5py.File(folder + '/' + foi_0558))
    #blueims = copy_h5(h5py.File(folder + '/' + foi_0428))

    # Convert PNG to numpy array
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


    # load skymap files
    #vskymap = copy_h5(h5py.File(folder + 'skymap.mat')['magnetic_footpointing'])
    #skymapred = [vskymap['180km']['lat'], vskymap['180km']['lon']]
    #skymapgreen = [vskymap['110km']['lat'], vskymap['110km']['lon']]
    #skymapblue = [vskymap['107km']['lat'], vskymap['107km']['lon']]

    with h5py.File(skymap_file, 'r') as h5:
        skymapred = [h5['/magnetic_footpointing/180km/lat'][:],
                     h5['/magnetic_footpointing/180km/lon'][:]]
        skymapgreen = [h5['/magnetic_footpointing/110km/lat'][:],
                       h5['/magnetic_footpointing/110km/lon'][:]]
        skymapblue = [h5['/magnetic_footpointing/107km/lat'][:],
                      h5['/magnetic_footpointing/107km/lon'][:]]

    ## Prepare data with coadding
    #greenimcoadd = (greenims['frame_1'] + greenims['frame_2'] + greenims['frame_3']) / 3
    #blueimcoadd = (blueims['frame_1'] + blueims['frame_2'] + blueims['frame_3']) / 3
    #redimcoadd = (redims['frame_1'] + redims['frame_2'] + redims['frame_3']) / 3

    redimcoadd = sum(redims)/len(redims)
    greenimcoadd = sum(greenims)/len(greenims)
    blueimcoadd = sum(blueims)/len(blueims)

    # Create pngs of coadded images (this should be optional)
    if plot:
        plt.imshow(redimcoadd)
        plt.title('Red Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        red_fn = 'red_imagery.png'
        #red_out = os.path.join(group_outdir, red_fn)
        plt.savefig(red_fn)
        plt.close()
        
        plt.imshow(greenimcoadd)
        plt.title('Green Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        green_fn = 'green_imagery.png'
        #green_out = os.path.join(group_outdir, green_fn)
        plt.savefig(green_fn)
        plt.close()
        
        plt.imshow(blueimcoadd)
        plt.title('Blue Imagery')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        blue_fn = 'blue_imagery.png'
        #blue_out = os.path.join(group_outdir, blue_fn)
        plt.savefig(blue_fn)
        plt.close()

    # Map everything to magnetic coordinates?
    A = Apex(date = dtdate)
    bmla, bmlo = A.convert(skymapblue[0].reshape(-1), np.mod(skymapblue[1].reshape(-1), 360), 'geo', 'apex', height = 110)

    minmlat = np.amin(bmla[np.where(~np.isnan(bmla))])
    maxmlat = np.amax(bmla[np.where(~np.isnan(bmla))])
    
    minmlon = np.amin(bmlo[np.where(~np.isnan(bmlo))])
    maxmlon = np.amax(bmlo[np.where(~np.isnan(bmlo))])

    interplonvec = np.linspace(minmlon, maxmlon, 1024)
    interplatvec = np.linspace(minmlat, maxmlat, 1024)
    
    print("Denoising images...")

    # regridding happens somewhere in here
    # Also image processing/smoothing/denoising/gap-filling
    # There's a bunch of image copying in here that's probably memory heavy - is it really necessary?
    # Can we seperate red, green, and blue processing into their own functions? (Low Priority)
    # Somehow this function generates the maglon and maglat grids - WHY?
    blueimdenoisewavelet, blueimreg, lon110, lat110, maglon, maglat, bluebgbright, bluesig = wavelet_denoise_resample(blueimcoadd, dtdate, skymapblue[1], skymapblue[0], 
                                                                                                                      interplonvec, interplatvec, 110, nshifts = n_shifts, 
                                                                                                                      background_method = background_method, plot = True)

    blueimdenoisegauss, _, _, _, _, _, _, _ = gaussian_denoise_resample(blueimcoadd, dtdate, skymapblue[1], skymapblue[0], interplonvec, 
                                                                        interplatvec, 110, blur_deg_EW, NS_deg = blur_deg_NS, plot = True)

    noise = np.copy(skymapblue[0]).reshape(-1)
    noiseadd = (np.random.randn(len(noise[np.where(~np.isnan(noise))])) * bluesig)
    noise[np.where(~np.isnan(noise))] = noiseadd
    noise = noise.reshape(skymapblue[0].shape)

    redimdenoise, redimreg, _, _, _, _, redbgbright, redsig = wavelet_denoise_resample(redimcoadd, dtdate, skymapred[1], skymapred[0], 
                                                                                       interplonvec, interplatvec, 110, nshifts = n_shifts, 
                                                                                       background_method = background_method, plot = True)
    greenimdenoise, greenimreg, _, _, _, _, greenbgbright, greensig = wavelet_denoise_resample(greenimcoadd, dtdate, skymapgreen[1], skymapgreen[0], 
                                                                                               interplonvec, interplatvec, 110, nshifts = n_shifts, 
                                                                                               background_method = background_method, plot = True)
    blueimdenoise = np.copy(blueimdenoisewavelet)

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

    greenframe = np.copy(greenimdenoise)
    greenframe[np.where(np.isnan(greenframe))] = greenbgbright

    blueframe = np.copy(blueimdenoise)
    blueframe[np.where(np.isnan(blueframe))] = bluebgbright

    redframe = np.copy(redimdenoise)
    redframe[np.where(np.isnan(redframe))] = redbgbright

    colormat = np.asarray([nred * (redframe - redmin), ngreen * (greenframe - greenmin), nblue * (blueframe - bluemin)]).astype(float)
    colormat /= maxbright
    
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

    
    ## Actual inversion and Conductance calculations happen here
    ## Probably makes sense to shift these into a new function so this function is limited to image preprocessing and prep
    #print("Calculating Q and E0...")

    #qout, e0out, minq, maxq, mine0, maxe0 = calculate_E0_Q_v2(redraydec, greenraydec, blueraydec, v, minE0 = 150, generous = True)
    #
    #print("Calculating conductivities given Q and E0...")
    #
    ## Calculate conductivities AFTER calculating Q and E0
    #SigP, SigH = calculate_Sig(qout, e0out, v, generous = True)
    #
    maglon_dec = maglon[::dec, ::dec]
    maglat_dec = maglat[::dec, ::dec]

    return redraydec, greenraydec, blueraydec, maglon_dec, maglat_dec

    ## Make plotting optional
    ## Troubleshooting plots
    #plt.title('Map of Q in Geomagnetic Coordinates')
    #plt.pcolormesh(maglon_dec, maglat_dec, qout, cmap = 'plasma')
    #plt.colorbar(label = 'mW/m$^2$')
    #plt.xlabel('Geomagnetic Longitude')
    #plt.ylabel('Geomagnetic Latitude')
    #Q_fn = 'Q_geomag.png'
    #Q_out = os.path.join(group_outdir, Q_fn)
    #plt.savefig(Q_out)
    #plt.close()

    #plt.title('Map of E0 in Geomagnetic Coordinates')
    #plt.pcolormesh(maglon_dec, maglat_dec, e0out, cmap = 'viridis')
    #plt.colorbar(label = 'eV')
    #plt.xlabel('Geomagnetic Longitude')
    #plt.ylabel('Geomagnetic Latitude')
    #E0_fn = 'E0_geomag.png'
    #E0_out = os.path.join(group_outdir, E0_fn)
    #plt.savefig(E0_out)
    #plt.close()

    #plt.title('Map of SigP in Geomagnetic Coordinates')
    #plt.pcolormesh(maglon_dec, maglat_dec, SigP, cmap = 'magma')
    #plt.colorbar(label = 'mho ($\mho$)')
    #plt.xlabel('Geomagnetic Longitude')
    #plt.ylabel('Geomagnetic Latitude')
    #SigP_fn = 'SigP_geomag.png'
    #SigP_out = os.path.join(group_outdir, SigP_fn)
    #plt.savefig(SigP_out)
    #plt.close()
    #
    #plt.title('Map of SigH in Geomagnetic Coordinates')
    #plt.pcolormesh(maglon_dec, maglat_dec, SigH, cmap = 'cividis')
    #plt.colorbar(label = 'mho ($\mho$)')
    #plt.xlabel('Geomagnetic Longitude')
    #plt.ylabel('Geomagnetic Latitude')
    #SigH_fn = 'SigH_geomag.png'
    #SigH_out = os.path.join(group_outdir, SigH_fn)
    #plt.savefig(SigH_out)
    #plt.close()
    #
    #return dtdate, group_outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH
