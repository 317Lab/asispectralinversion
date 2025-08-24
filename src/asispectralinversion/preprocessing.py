import numpy as np
from apexpy import Apex
import scipy.interpolate
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet, cycle_spin, estimate_sigma


def common_grid(bmlat, bmlon, gmlat, gmlon, rmlat, rmlon):
    """
    Purpose:
        - generate common grid from intial magnetic blue, green, and red grid
    """

    minmlat = np.max([np.nanmin(bmlat), np.nanmin(gmlat), np.nanmin(rmlat)])
    maxmlat = np.min([np.nanmax(bmlat), np.nanmax(gmlat), np.nanmax(rmlat)])
    #print('GRID LAT LIMS:', minmlat, maxmlat)
    
    minmlon = np.max([np.nanmin(bmlon), np.nanmin(gmlon), np.nanmin(rmlon)])
    maxmlon = np.min([np.nanmax(bmlon), np.nanmax(gmlon), np.nanmax(rmlon)])
    #print('GRID LON LIMS:', minmlon, maxmlon)

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


def background_brightness_darkpatches(im, mask, plot=True):
    """
    Purpose: 
        - given an unmapped image, finds dark patches of sky to estimate background brightness
          and gaussian noise level
    """
    
#<<<<<<< HEAD
## Given an unmapped image, finds dark patches of sky to estimate background brightness and gaussian noise level
#def background_brightness_darkpatches(im,lon,lat,plot=False):
#    # A gaussian function
#    def gauss(x, A, x0, sigma):
#        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
#=======
    im[mask] = np.nan
    
    # First, break the image up into patches:
    rows, cols = im.shape
#>>>>>>> production
    
    # Size of patches - they will be of size *step* x *step*
    #step = 5
    step = 10
    # print('patch size='+str(step**2))
    
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
#<<<<<<< HEAD
#    # If the fit failed
#    if fitfailed:
#        # We keep the median estimate for background brightness
#        cent = np.copy(centerguess)
#        # Skimage noise estimate
#        sig = sigguess
# 
#    # print('bg='+str(cent))
#    # print('sig='+str(sig))
#    
#    if plot:
#        # Plot the best guess gaussian
#        if ~fitfailed:
#            plt.plot(bins,5*gauss(bins,peak,cent,sig))
#        plt.show()
#        # Plot the image on a balanced colormap where white is the background brightness we found
#        # Possibly useful to assess validity of results
#        plt.pcolor(lon,lat,im,vmin=cent-4*sig,vmax=cent+4*sig,cmap='seismic')
#        plt.colorbar()
#        plt.title('white is fitted background brightness')
#        plt.show()
#    return cent,sig
#    
#    
#def background_brightness_corners(im,lon,lat,plot=False):
#    # A gaussian function
#    def gauss(x, A, x0, sigma):
#        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
#    
#    imvec = im[np.where(np.isnan(lon+lat))]
#=======
#>>>>>>> production

    if fitfailed:
        cent = np.copy(centerguess) # keep the median estimate for background brightness
        sig = sigguess # skimage noise estimate
 
#    print('bg=' + str(cent))
#    print('sig=' + str(sig))
    
    if plot:
        # Plot the best guess gaussian
        if ~fitfailed:
            plt.plot(bins,5*gauss(bins,peak,cent,sig))
        plt.show()
        # Plot the image on a balanced colormap where white is the background brightness we found
        # Possibly useful to assess validity of results
        #plt.pcolor(lon,lat,im,vmin=cent-4*sig,vmax=cent+4*sig,cmap='seismic')
        plt.pcolor(im,vmin=cent-4*sig,vmax=cent+4*sig,cmap='seismic')
        plt.colorbar()
        plt.title('white is fitted background brightness')
        plt.show()
    
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
    
#<<<<<<< HEAD
    # Plot histograms and initial guess gaussian
    if plot:
        plt.scatter(bincenters,hist)
        plt.plot(bins,gauss(bins,np.amax(hist),centerguess,sigguess))
        plt.xlim(np.amin(imvec)-10,np.amax(imvec)+10)
        plt.title('noise fit')

    # We fit the the histogram to a gaussian curve
    [peak,cent,sig] = scipy.optimize.curve_fit(gauss,bincenters,hist,p0=[np.amax(hist),centerguess,sigguess])[0]
    #print('bg='+str(cent))
    # print('sig='+str(sig))
    
    # Plot the image on a balanced colormap where white is the background brightness we found
    # Possibly useful to assess validity of results
    if plot:
        plt.plot(bins,gauss(bins,peak,cent,sig))
        plt.show()
        plt.pcolor(lon,lat,im,vmin=cent-3*sig,vmax=cent+3*sig,cmap='seismic')
        plt.colorbar()
        plt.title('white is fitted background brightness')
        plt.show()
    #return cent,sig

## Resample an image onto a uniform grid and denoise it
#
## Alex: this is confusing but to try to clarify... We want to project a red, green, and a blue image onto the same new uniform grid.
## To to that we specify a new uniform grid in mlon,mlat, resample our image onto it, then footpoint the grid to 110 km to match the convention we use.
#
#
## Takes in: image, date (for Apex footpointing), old geolon grid, old geolat grid,
## new maglon vector, new maglat vector, altitude in km of old map, blur width in degrees lon, blur width in degrees lat, plot
#
## Denoising is done through straightforward gaussian blurring - width_deg and NS_deg specify longitudinal and latitudinal gaussian widths
## in degrees, respectively.
#def gaussian_denoise_resample(im,date,lon,lat,newmlonvec,newmlatvec,mapalt_km,width_deg,NS_deg=0,background_method='patches',plot=False): 
#    # Used for setting the bounds of plots
#    minlon = np.amin(lon[np.where(~np.isnan(lon))])
#    maxlon = np.amax(lon[np.where(~np.isnan(lon))])
#
#    minlat = np.amin(lat[np.where(~np.isnan(lon))])
#    maxlat = np.amax(lat[np.where(~np.isnan(lon))])
#
#    # Set the date for apex coordinates
#    # A = Apex(date=date)
#
#    # Convert
#    maglat, maglon = apex_convert(lat, lon, 'geo', 'apex', date, height=mapalt_km)
#    # maglat, maglon = A.convert(lat.reshape(-1), np.mod(lon.reshape(-1),360), 'geo', 'apex', height=mapalt_km)
#    # maglon = maglon.reshape(lon.shape)
#    # maglat = maglat.reshape(lon.shape)
#        
#    # Interpolate onto regular grid        
#    regim,regmaglon,regmaglat = interpolate_reggrid(im,maglon,maglat,newmlonvec,newmlatvec)
#=======
    return cent, sig


def background_brightness(im, mask, background_method='patches'):
#>>>>>>> production
    # Find background brightness and estimated gaussian noise level
    if background_method=='patches':
        bgbright, sig = background_brightness_darkpatches(im, mask)
    elif background_method=='corners':
#<<<<<<< HEAD
#        bgbright,sig = background_brightness_corners(im,lon,lat,plot=plot)
#        
#    # Estimate sigma using skimage
#    sigma_est = estimate_sigma(im[np.where(~np.isnan(lon+lat))])
#    # print('skimage estimated sig='+str(sigma_est))
#    
#    # Grid steps for our new footpointed grid
#    # Note that the new grid is very nearly Cartesian in footlat/footlon
#    dlon = np.mean(np.diff(regmaglon,axis=0))
#    dlat = np.mean(np.diff(regmaglat,axis=1))
#
#    # Denoise
#    # If we want to blur by a nonzero amount
#    if width_deg != 0:
#        regimfill = np.copy(regim)
#        # Fill in the region of the image that does not map to the sky with the background brightness value
#        regimfill[np.where(np.isnan(regim))] = bgbright
#        
#        # Plot image in magnetic coords
#        if plot:
#            plt.pcolormesh(regmaglon,regmaglat,regimfill)
#            plt.title('Image in Mag Coords')
#            plt.show()
#
#        # Blur E-W
#        regimblur = scipy.ndimage.gaussian_filter1d(regimfill,width_deg/dlon,axis=0)
#        if NS_deg != 0:
#            #print('blurring N-S!!!')
#            # Blur N-S
#            regimblur = scipy.ndimage.gaussian_filter1d(np.copy(regimblur),NS_deg/dlat,axis=1)
#        regimblur[np.where(np.isnan(regim))] = np.nan
#    else:
#        # If EW blurring degrees == 0, we just return the resampled image
#        regimblur = regim
#
#    # Footpoint our new grid
#    lat110, lon110 = apex_convert(regmaglat, regmaglon, 'apex', 'geo', date, height=110)
#    # lat110,lon110 = A.convert(regmaglat.reshape(-1), np.mod(regmaglon.reshape(-1),360), 'apex', 'geo', height=110)
#    # lat110 = lat110.reshape(regmaglat.shape)
#    # lon110 = lon110.reshape(regmaglon.shape)
#
#    if plot:
#        plt.pcolor(lon,lat,im,vmin=np.amin(regimblur[np.where(~np.isnan(regimblur))]),vmax=np.amax(regimblur[np.where(~np.isnan(regimblur))]))
#        plt.title('original image')
#        plt.show()
#
#        if width_deg != 0:
#            plt.pcolormesh(lon110,lat110,regimblur)
#            plt.xlim(minlon,maxlon)
#            plt.ylim(minlat,maxlat)
#            plt.title('blurred image')
#            plt.show()
#    
#    return regimblur,regim,lon110,lat110,regmaglon,regmaglat,bgbright,sig
#
#
## Takes in: image, date (for Apex footpointing), old geolon grid, old geolat grid,
## new maglon vector, new maglat vector, altitude in km of old map, nshifts, plot
#
## Denoising is done through Bayesian thresholding of a nearly shift-invariant discrete wavelet transform
## nshifts effectively parameterizes the shift-invariance, the larger it is set, the longer the function
## takes to run but the more shift invariant the wavelets are (better quality denoising). Its default value
## of 50 is already probably too high, one could reduce it to 30 with no problem. There is no good reason
## to set it <5, since that should take less than a second to run.
#
#def wavelet_denoise_resample(im,date,lon,lat,newmlonvec,newmlatvec,mapalt_km,nshifts=50,background_method='patches',plot=False):
#    # Used for setting the bounds of plots
#    minlon = np.amin(lon[np.where(~np.isnan(lon))])
#    maxlon = np.amax(lon[np.where(~np.isnan(lon))])
#
#    minlat = np.amin(lat[np.where(~np.isnan(lon))])
#    maxlat = np.amax(lat[np.where(~np.isnan(lon))])
#
#    # Set the date for apex coordinates
#    # A = Apex(date=date)
#
#    # Convert
#    maglat, maglon = apex_convert(lat, lon, 'geo', 'apex', date, height=mapalt_km)
#    # maglat, maglon = A.convert(lat.reshape(-1), np.mod(lon.reshape(-1),360), 'geo', 'apex', height=mapalt_km)
#    # maglon = maglon.reshape(lon.shape)
#    # maglat = maglat.reshape(lon.shape)
#        
#    # Interpolate original image onto regular grid        
#    regim,regmaglon,regmaglat = interpolate_reggrid(im,maglon,maglat,newmlonvec,newmlatvec)
#    # Find background brightness and estimated gaussian noise level
#    if background_method=='patches':
#        bgbright,sig = background_brightness_darkpatches(im,lon,lat,plot=plot)
#    elif background_method=='corners':
#        bgbright,sig = background_brightness_corners(im,lon,lat,plot=plot)
#
#    # Estimate sigma using skimage
#    sigma_est = estimate_sigma(im[np.where(~np.isnan(lon+lat))])
#    # print('skimage estimated sig='+str(sigma_est))
#
#    # Grid steps for our new footpointed grid
#    # Note that the new grid is very nearly Cartesian in footlat/footlon
#    dlon = np.mean(np.diff(regmaglon,axis=0))
#    dlat = np.mean(np.diff(regmaglat,axis=1))
#
#    # Denoise
#    imdenoise = cycle_spin(im, func=denoise_wavelet, max_shifts=nshifts)
#    # Interpolate denoised image onto regular grid
#    regimdenoise,_,_ = interpolate_reggrid(imdenoise,maglon,maglat,newmlonvec,newmlatvec)
#    
#    ridnfill = np.copy(regimdenoise)
#    # Fill in the region of the image that does not map to the sky with the background brightness value
#    ridnfill[np.where(np.isnan(regimdenoise))] = bgbright
#    
#    # NEW!!!! DO A SMALL GAUSSIAN BLUR!!!
#    regimblur = scipy.ndimage.gaussian_filter1d(ridnfill,0.1/dlon,axis=0)
#    regimblur = scipy.ndimage.gaussian_filter1d(np.copy(regimblur),0.01/dlat,axis=1)
#    regimblur[np.where(np.isnan(regimdenoise))] = np.nan
#    #regimblur = np.copy(regimdenoise)
#    
#    # Footpoint our new grid
#    # lat110,lon110 = A.convert(regmaglat.reshape(-1), np.mod(regmaglon.reshape(-1),360), 'apex', 'geo', height=110)
#    # lat110 = lat110.reshape(regmaglat.shape)
#    # lon110 = lon110.reshape(regmaglon.shape)
#    lat110, lon110 = apex_convert(regmaglat, regmaglon, 'apex', 'geo', date, height=110)
#=======
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
#>>>>>>> production

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
    
#<<<<<<< HEAD
    return redcut,greencut,bluecut

def apex_convert(lat, lon, source, dest, date, height=0):
    A = Apex(date=date)
    lat_flat = lat.reshape(-1)
    lon_flat = np.mod(lon.reshape(-1), 360)
    ids = np.where(~np.isnan(lat_flat))
    lat_in = lat_flat[ids]
    lon_in = lon_flat[ids]
    if np.shape(height) == ():
        height_in = height
    else:
        height_flat = height.reshape(-1)
        height_in = height_flat[ids]

    lat_out, lon_out = A.convert(lat_in, lon_in, source, dest, height=height_in)
    lat_out_nans = np.empty(lat_flat.shape)
    lon_out_nans = np.empty(lat_flat.shape)
    lat_out_nans[:] = np.nan
    lon_out_nans[:] = np.nan
    lat_out_nans[ids] = lat_out
    lon_out_nans[ids] = lon_out
    return lat_out_nans.reshape(lat.shape), lon_out_nans.reshape(lon.shape)
#=======
#    return redcut, greencut, bluecut
#
#
#>>>>>>> production
