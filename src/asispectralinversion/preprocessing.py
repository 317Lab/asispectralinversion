import numpy as np
from apexpy import Apex
import scipy.interpolate
from skimage.restoration import denoise_wavelet, cycle_spin, estimate_sigma

#def reggrid(oldlon, oldlat, newlonvec, newlatvec):
#    """
#    Purpose: 
#        - interpolates an image, given a grid, onto a new regular grid
#        - takes the image, the old lat/lon grid, and lat/lon vectors for the new mesh grid
#    """
#    
#    # Masks out NaNs
#    lonmasked = np.ma.masked_invalid(oldlon)
#    latmasked = np.ma.masked_invalid(oldlat)
#
#    # Pulls out unmasked part of old grid
#    longood = lonmasked[~lonmasked.mask]
#    latgood = latmasked[~lonmasked.mask]
#    
#    # Creates the new mesh grid
#    newlat, newlon = np.meshgrid(newlatvec, newlonvec)
#
#    return newlon, newlat


def common_grid(bmlat, bmlon, gmlat, gmlon, rmlat, rmlon):
    """
    Purpose:
        - generate common grid from intial magnetic blue, green, and red grid
    """

    minmlat = np.max([np.nanmin(bmlat), np.nanmin(gmlat), np.nanmin(rmlat)])
    maxmlat = np.min([np.nanmax(bmlat), np.nanmax(gmlat), np.nanmax(rmlat)])
    print('GRID LAT LIMS:', minmlat, maxmlat)
    
    minmlon = np.max([np.nanmin(bmlon), np.nanmin(gmlon), np.nanmin(rmlon)])
    maxmlon = np.min([np.nanmax(bmlon), np.nanmax(gmlon), np.nanmax(rmlon)])
    print('GRID LON LIMS:', minmlon, maxmlon)

    interplonvec = np.linspace(minmlon, maxmlon, 1024)
    interplatvec = np.linspace(minmlat, maxmlat, 1024)

    gridmlat, gridmlon = np.meshgrid(interplatvec, interplonvec)

    return gridmlat, gridmlon



def interpolate_reggrid(im, oldlon, oldlat, newlon, newlat):
    """
    Purpose: 
        - interpolates an image, given a grid, onto a new regular grid
        - takes the image, the old lat/lon grid, and lat/lon vectors for the new mesh grid
    """

    print('interpolate image...')
    
    # Masks out NaNs
    lonmasked = np.ma.masked_invalid(oldlon)
    latmasked = np.ma.masked_invalid(oldlat)

    # Pulls out unmasked part of old grid
    longood = lonmasked[~lonmasked.mask]
    latgood = latmasked[~lonmasked.mask]
    
    # Pulls out part of image corresponding to valid lon/lat coords
    imgood = im[~lonmasked.mask]

    # Creates the new mesh grid
    #newlat, newlon = np.meshgrid(newlatvec, newlonvec)

    # Interpolates the image onto the new grid
    newimvec = scipy.interpolate.griddata(np.asarray([longood, latgood]).T, imgood, np.asarray([newlon.reshape(-1), newlat.reshape(-1)]).T, method='linear')

    #return newimvec.reshape(newlat.shape), newlon, newlat
    return newimvec.reshape(newlat.shape)


# A gaussian function
def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def background_brightness_darkpatches(im, mask):
    """
    Purpose: 
        - given an unmapped image, finds dark patches of sky to estimate background brightness
          and gaussian noise level
    """
    
    im[mask] = np.nan
    
    # First, break the image up into patches:
    rows, cols = im.shape
    
    # Size of patches - they will be of size *step* x *step*
    step = 5
    
    # Indices to iterate through
    rowinds = np.arange(0, rows, step)
    colinds = np.arange(0, cols, step)
    
    # Construct a list of median brightnesses for each valid patch
    # Can this be done through sliding window functions?
    medvec = [] # initialize vector keeping track of medians
    for i in rowinds:
        for j in colinds:
            imbin = im[i: i + step + 1, j: j + step + 1] # extract patch
            #if len(imbingood)>=((step+1)**2)/2: # discard patch if it is more than half invalid.
            if np.sum(~np.isnan(imbin))>=((step+1)**2)/2: # discard patch if it is more than half invalid.
                medi = np.median(imbin) 
                medvec.append(medi)
            else: # otherwise, calculate the median
                continue
    medvec = np.asarray(medvec)


    # Choose a maximum cutoff brightness, using quantiles or a specified absolute brightness
    medbright = np.sort(medvec)[1] # we will have at least two patches kept

    # Construct a list of all the brightness points from patches below the max brightness cutoff
    imvec = [] # vector keeping track of all pixels from patches dimmer than medbright

    # Find sufficiently dim patches
    for i in rowinds:
        for j in colinds:
            imbin = im[i: i + step + 1, j: j + step + 1] # extract patch
            if np.sum(~np.isnan(imbin))>=((step+1)**2)/2: # discard patch if it is more than half invalid.
                if np.median(imbin) <= medbright:
                    imvec.extend(list(imbin))
            else: # otherwise, calculate the median
                continue
    imvec = np.asarray(imvec)

    # Bin up the brightnesses to make a histogram
    bins = np.arange(np.amin(imvec), np.amax(imvec) + 1) # choose the bins as integer numbers of counts
    plotbins = np.arange(np.amin(imvec), np.amax(imvec) + 5, 5)
    bincenters = [np.mean(bins[i: i + 2]) for i in range(len(bins) - 1)] # centers of bins
    
    # Histogram for fitting
    hist, _ = np.histogram(imvec, bins)
    hist = np.asarray(hist)
    
    # Histogram for plotting/initial guesses
    plothist, _ = np.histogram(imvec, plotbins)
    plothist = np.asarray(plothist)
    
    # An initial guess of the center of the brightness distribution
    centerguess = np.median(imvec)
    sigguess = estimate_sigma(im[np.where(~np.isnan(im))])
    
    # Initialize binary to keep track of whether the fit was successful
    fitfailed = False
    try:
        [peak, cent, sig] = scipy.optimize.curve_fit(gauss, bincenters, hist, p0=[np.amax(plothist) / 5, centerguess, sigguess])[0]

    except Exception as e:
        print('Fit failed! Failed at {e}')
        cent, sig = background_brightness_corners(im, mask)
            
    # Range of the data
    hrange = bins[-1] - bins[0]
    
    # Sanity check on parameters - they are almost surely wrong if they fail this
    if (sig>(hrange/2)) | ((cent-centerguess)>(hrange/4)):
        print('Fit failed!')
        fitfailed = True

    if fitfailed:
        cent = np.copy(centerguess) # keep the median estimate for background brightness
        sig = sigguess # skimage noise estimate
 
#    print('bg=' + str(cent))
#    print('sig=' + str(sig))
    
    return cent, sig

     
def background_brightness_corners(im, mask):
    """
    Purpose: 
        - given an unmapped image, uses corners of sky to estimate background brightness
          and gaussian noise level
    """
    
    imvec = im[np.where(mask)]
    
    # Bin up the brightnesses to make a histogram
    bins = np.arange(np.amin(imvec), np.amax(imvec) + 1) # choose the bins as integer numbers of counts
    bincenters = [np.mean(bins[i: i + 2]) for i in range(len(bins) - 1)]
    hist, _ = np.histogram(imvec, bins)

    # An initial guess of the peak of the brightness distribution - this is simply the mode of the histogram
    centerguess = bincenters[np.where(hist == np.amax(hist))[0][0]]

    # An initial guess at the width of the distribution, found by assuming it to be gaussian and finding the 
    sigguess = (centerguess - bincenters[np.where(hist >= np.amax(hist)/2)[0][0]])/np.sqrt(np.log(2)) # crossing point of half max on the lower brightness end

    # Fit the the histogram to a gaussian curve
    try:
        [peak, cent, sig] = scipy.optimize.curve_fit(gauss, bincenters, hist, p0=[np.amax(hist), centerguess, sigguess])[0]
    except Exception as e:
        print('Fit failed! Failed at {e}')
        cent = centerguess
        sig = sigguess
    #print('sig=' + str(sig))
    
    return cent, sig


def background_brightness(im, mask, background_method='patches'):
    # Find background brightness and estimated gaussian noise level
    if background_method=='patches':
        bgbright, sig = background_brightness_darkpatches(im, mask)
    elif background_method=='corners':
        bgbright, sig = background_brightness_corners(im, mask)

#    # Estimate sigma using skimage
#    sigma_est = estimate_sigma(im)
#    print('skimage estimated sig=' + str(sigma_est))

    return bgbright, sig


def gaussian_denoise(im, dlat, dlon, bgbright, EW_deg=0, NS_deg=0): 
    """
    Purpose: 
        - resample an image onto a uniform grid and denoise it
        - project a red, green, and a blue image onto the same new uniform grid
        - specify a new uniform grid in mlon, mlat; resample image onto it, footpoint the grid to 110 km to match the convention used
        - denoising is done through gaussian blurring
    Takes in:
        - image
        - date (for Apex footpointing)
        - old geolon grid
        - old geolat grid
        - new maglon vector
        - new maglat vector
        - altitude [km] of old map
        - blur width in degrees lat
        - blur width in degrees lon
        - plot
    """

    print('gaussian denoise...')

    # Denoise
    regimblur = np.copy(im)
    
    # Fill in the region of the image that does not map to the sky with the background brightness value
    regimblur[np.where(np.isnan(im))] = bgbright

    # Blur E-W
    if EW_deg != 0:
        regimblur = scipy.ndimage.gaussian_filter1d(regimblur, EW_deg / dlon, axis = 0)

    # Blur N-S
    if NS_deg != 0: # latitudinal gaussian width
        # Blur N-S
        regimblur = scipy.ndimage.gaussian_filter1d(regimblur, NS_deg / dlat, axis = 1)

    regimblur[np.where(np.isnan(im))] = np.nan

    return regimblur


def wavelet_denoise(im, dlat, dlon, bgbright, nshifts=50):
    """
    Purpose: 
        - denoising is done through Bayesian thresholding of a nearly shift-invariant discrete wavelet transform
        - 'nshifts' effectively parameterizes the shift-invariance, the larger it is set, the longer the function
          takes to run but the more shift invariant the wavelets are (better quality denoising)
    Takes in:
        - image
        - date (for Apex footpointing)
        - old geolon grid
        - old geolat grid
        - new maglon vector
        - new maglat vector
        - altitude [km] of old map
        - nshifts
        - plot
    """
    

    print('wavelet denoise...')

    ridnfill = np.copy(im)
    
    # Fill in the region of the image that does not map to the sky with the background brightness value
    ridnfill[np.isnan(im)] = bgbright
    
    # Denoise
    imdenoise = cycle_spin(ridnfill, func = denoise_wavelet, max_shifts = nshifts)

   
    # Perform small gaussian blur
    regimblur = scipy.ndimage.gaussian_filter1d(imdenoise, 0.1 / dlon, axis = 0)
    regimblur = scipy.ndimage.gaussian_filter1d(np.copy(regimblur), 0.01 / dlat, axis = 1)
    regimblur[np.where(np.isnan(im))] = np.nan

    return regimblur

    
def to_rayleighs(redcutin,greencutin,bluecutin,redbg,greenbg,bluebg):
    """
    Purpose: 
        - uses radioactive source calibrations to convert counts to rayleighs
        - specific to the Poker DASC
    """
    
    # Divide by integration time to get counts per second
    redcut = (np.copy(redcutin) - redbg) / 1.5
    greencut = (np.copy(greencutin) - greenbg) / 1
    bluecut = (np.copy(bluecutin) - bluebg) / 1
    
    # Conversion factors, rayleighs / (counts/second)
    redcut *= 23.8
    greencut *= 24.2
    bluecut *= 69.8
    
    return redcut, greencut, bluecut
