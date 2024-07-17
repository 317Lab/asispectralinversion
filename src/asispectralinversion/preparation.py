import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
from apexpy import Apex
from inversion import load_lookup_tables_directory
from inversion import calculate_E0_Q_v2
from preprocessing import wavelet_denoise_resample
from preprocessing import gaussian_denoise_resample
from preprocessing import to_rayleighs
from inversion import calculate_Sig
import os

"""
Purpose of this script:
    - takes in ASI/GLOW information, preparing for preprocessing and inversion
    - runs preprocessing and inversion functions
    - returns Q, E0, SigmaP, and SigmaH in regularized, geomagnetic coordinates
"""

def copy_h5(vtest):
    """
    Purpose: 
        - copies an HDF5 structure to a python dict recursively
    """
    
    dicttest = {}
    keyslist = list(vtest.keys())
    for key in keyslist:
        if type(vtest[key]) == h5py._hl.dataset.Dataset:
            if vtest[key].shape[1] == 1:
                if vtest[key].shape[0] == 1:
                    dicttest[key] = vtest[key][0][0]
                else:
                    dicttest[key] = np.asarray(vtest[key]).flatten()
            else:
                dicttest[key] = np.asarray(vtest[key])
        else:
            dicttest[key] = copy_h5(vtest[key])
            
    return dicttest


def prepare_data(date, maglatsite, folder, outdir):
    """
    Purpose: 
        - prepares Q, E0, SigP, and SigH given ASI data
    """
    
    print("Pulling information from data files and lookup tables...")

    dtdate = datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])) # creating datetime object from given date
    blur_deg_EW = 0.4 # gaussian blur width in degrees maglon
    blur_deg_NS = 0.04 # gaussian blur width in degrees maglat
    n_shifts = 50 # integer determining shift-invariance of wavelets
    background_method = 'patches' # set to 'patches' or 'corners'

    v = load_lookup_tables_directory(folder, maglatsite)

    redims = copy_h5(h5py.File(folder + 'reddata.mat'))
    greenims = copy_h5(h5py.File(folder + 'greendata.mat'))
    blueims = copy_h5(h5py.File(folder + 'bluedata.mat'))

    vskymap = copy_h5(h5py.File(folder + 'skymap.mat')['magnetic_footpointing'])
    skymapred = [vskymap['180km']['lat'], vskymap['180km']['lon']]
    skymapgreen = [vskymap['110km']['lat'], vskymap['110km']['lon']]
    skymapblue = [vskymap['107km']['lat'], vskymap['107km']['lon']]

    greenimcoadd = (greenims['frame1'] + greenims['frame2'] + greenims['frame3']) / 3
    blueimcoadd = (blueims['frame1'] + blueims['frame2'] + blueims['frame3']) / 3
    redimcoadd = (redims['frame1'] + redims['frame2'] + redims['frame3']) / 3

    plt.imshow(redimcoadd)
    plt.title('Red Imagery')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    red_fn = 'red_imagery.png'
    red_out = os.path.join(outdir, red_fn)
    plt.savefig(red_out)
    plt.show()
    
    plt.imshow(greenimcoadd)
    plt.title('Green Imagery')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    green_fn = 'green_imagery.png'
    green_out = os.path.join(outdir, green_fn)
    plt.savefig(green_out)
    plt.show()
    
    plt.imshow(blueimcoadd)
    plt.title('Blue Imagery')
    plt.xlabel('E-W')
    plt.ylabel('N-S')
    blue_fn = 'blue_imagery.png'
    blue_out = os.path.join(outdir, blue_fn)
    plt.savefig(blue_out)
    plt.show()

    A = Apex(date = dtdate)
    bmla, bmlo = A.convert(skymapblue[0].reshape(-1), np.mod(skymapblue[1].reshape(-1), 360), 'geo', 'apex', height = 110)

    minmlat = np.amin(bmla[np.where(~np.isnan(bmla))])
    maxmlat = np.amax(bmla[np.where(~np.isnan(bmla))])
    
    minmlon = np.amin(bmlo[np.where(~np.isnan(bmlo))])
    maxmlon = np.amax(bmlo[np.where(~np.isnan(bmlo))])

    interplonvec = np.linspace(minmlon, maxmlon, 1024)
    interplatvec = np.linspace(minmlat, maxmlat, 1024)
    
    print("Denoising images...")

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

    dec = 15 # 'dec = 2' returns given resolution

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
    
    print("Calculating Q and E0...")

    qout, e0out, minq, maxq, mine0, maxe0 = calculate_E0_Q_v2(redraydec, greenraydec, blueraydec, v, minE0 = 150, generous = True)
    
    print("Calculating conductivities given Q and E0...")
    
    # Calculate conductivities AFTER calculating Q and E0
    SigP, SigH = calculate_Sig(qout, e0out, v, generous = True)
    
    maglon_dec = maglon[::dec, ::dec]
    maglat_dec = maglat[::dec, ::dec]
    
    # Troubleshooting plots
    plt.title('Map of Q in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, qout, cmap = 'plasma')
    plt.colorbar(label = 'W/m$^2$')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    Q_fn = 'Q_geomag.png'
    Q_out = os.path.join(outdir, Q_fn)
    plt.savefig(Q_out)
    plt.show()

    plt.title('Map of E0 in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, e0out, cmap = 'viridis')
    plt.colorbar(label = 'eV')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    E0_fn = 'E0_geomag.png'
    E0_out = os.path.join(outdir, E0_fn)
    plt.savefig(E0_out)
    plt.show()

    plt.title('Map of SigP in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, SigP, cmap = 'magma')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigP_fn = 'SigP_geomag.png'
    SigP_out = os.path.join(outdir, SigP_fn)
    plt.savefig(SigP_out)
    plt.show()
    
    plt.title('Map of SigH in Geomagnetic Coordinates')
    plt.pcolormesh(maglon_dec, maglat_dec, SigH, cmap = 'cividis')
    plt.colorbar(label = 'mho ($\mho$)')
    plt.xlabel('Geomagnetic Longitude')
    plt.ylabel('Geomagnetic Latitude')
    SigH_fn = 'SigH_geomag.png'
    SigH_out = os.path.join(outdir, SigH_fn)
    plt.savefig(SigH_out)
    plt.show()
    
    return dtdate, outdir, maglon_dec, maglat_dec, qout, e0out, SigP, SigH
