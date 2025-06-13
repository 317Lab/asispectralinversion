import numpy as np
import scipy.integrate
import scipy.interpolate
import glob


def sig_integrator(sigmat, altvec, maglat):
    """
    Purpose: 
        - height-integrates a 3d conductivity datacube to get a 2d conductance matrix
        - accounts for magnetic field angle from vertical to first order
    """
    
    # Cumulative trapezoidal integration
    #Sigmat = scipy.integrate.cumtrapz(sigmat, altvec / 100, axis = 0)[-1]
    Sigmat = scipy.integrate.cumulative_trapezoid(sigmat, altvec / 100, axis = 0)[-1]   # Added by LL 2025-06-12 - cumtrapz was replaced with cumlative_trapezoid in scipy version 1.6
    
    # First order account for magnetic field angle from vertical
    Sigmat /= np.sin(maglat * np.pi / 180)
    
    return Sigmat


def load_lookup_tables(fname_red, fname_green, fname_blue, fname_sigp, fname_sigh, maglat, plot = True):
    """
    Purpose: 
        - given a set of filenames, reads in the GLOW lookup tables and packages them into a struct
    """
    
    # Read in: run parameters, Q vector, E0 vector, green brightness matrix from bin file
    params, Qvec, E0vec, greenmat = process_brightbin(fname_green, plot=plot)
    
    # Read in: red and blue brightness matrices from bin files
    _, _, _, redmat = process_brightbin(fname_red, plot = plot)
    _, _, _, bluemat = process_brightbin(fname_blue, plot = plot)
    
    # Read in: altitude vector and Pedersen conductivity datacube from bin file
    _, _, _, altvec, sigPmat = process_sig3dbin(fname_sigp)
    
    # Read in: Hall conductivity datacube from bin file
    _, _, _, _, sigHmat = process_sig3dbin(fname_sigh)
    
    # Height-integrate conductivity datacubes to get conductance - first order correction for magnetic field angle
    SigPmat = sig_integrator(sigPmat, altvec, maglat)
    SigHmat = sig_integrator(sigHmat, altvec, maglat)
    
    # Put everything into a Python dict
    lookup_table = {
        'Params': params,
    	'Qvec': Qvec,
    	'E0vec': E0vec,
    	'greenmat': greenmat,
    	'redmat': redmat,
    	'bluemat': bluemat,
    	'altvec': altvec,
    	'sigPmat': sigPmat,
    	'sigHmat': sigHmat,
    	'SigPmat': SigPmat,
    	'SigHmat': SigHmat
    }
    return lookup_table

    
def load_lookup_tables_directory(folder, maglat, plot = True):
    """
    Purpose: 
        - given a directory, reads in the GLOW lookup tables and packages them into a struct
    """
    
    fnamered = glob.glob(folder + 'I6300*.bin')[0]
    fnamegreen = glob.glob(folder + 'I5577*.bin')[0]
    fnameblue = glob.glob(folder + 'I4278*.bin')[0]
    fnameped = glob.glob(folder + 'ped3d*.bin')[0]
    fnamehall = glob.glob(folder + 'hall3d*.bin')[0]
    
    # Airglow data
    fnamereda = glob.glob(folder +'I6300*.bin')[0]
    fnamegreena = glob.glob(folder + 'I5577*.bin')[0]
    fnamebluea = glob.glob(folder + 'I4278*.bin')[0]
    fnamepeda = glob.glob(folder + 'ped3d*.bin')[0]
    fnamehalla = glob.glob(folder + 'hall3d*.bin')[0]
    
    v = load_lookup_tables(fnamered, fnamegreen, fnameblue, fnameped, fnamehall, maglat, plot = plot)
    va = load_lookup_tables(fnamereda, fnamegreena, fnamebluea, fnamepeda, fnamehalla, maglat, plot = False)
    v['redbright_airglow'] = va['redmat']
    v['bluebright_airglow'] = va['bluemat']
    v['greenbright_airglow'] = va['greenmat']
    
    v['sigP_bg'] = va['sigPmat']
    v['sigH_bg'] = va['sigHmat']
    
    v['SigP_bg'] = va['SigPmat']
    v['SigH_bg'] = va['SigHmat']
    
    return v


def calculate_E0_Q_v2(redbright, greenbright, bluebright, inlookup_table, minE0 = 150, generous = False):
    """
    Purpose: 
        - given RGB brightness arrays (calibrated, in Rayleighs) and a lookup table for the correct night, estimates E0 and Q
    Notes:
        - setting minE0 constrains uncertainty values in Q to account for non-physical quantities at the bottom of the lookup tables
        - we often assume that visual signatures are insignificant below 150 eV, but that parameter can be set lower or higher as desired
        - the 'generous' option sets Q, E0 to zero instead of NaN when inversion fails but certain conditions are met (very dim pixels)
    """

    # Subtract out background brightnesses from the lookup table:
    lookup_table = inlookup_table.copy()
    lookup_table['redmat'] -= lookup_table['redbright_airglow'][0][0]
    lookup_table['greenmat'] -= lookup_table['greenbright_airglow'][0][0]
    lookup_table['bluemat'] -= lookup_table['bluebright_airglow'][0][0]
    
    # Save the initial shape of arrays - they will be flattened and later reshaped back to this
    shape = greenbright.shape

    # Reshape brightness arrays to vectors
    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    # Cut off the lookup table appropriately
    minE0ind = np.where(lookup_table['E0vec']>minE0)[0][0]
    
    # Estimates Q from blue brightness, along with error bars
    qvec, maxqvec, minqvec = q_interp(lookup_table['bluemat'], lookup_table['Qvec'], lookup_table['E0vec'], bluevec, minE0ind=minE0ind, maxbluebright='auto', interp='linear', plot=False)

    # Estimates E0 from red/green ratio and estimated Q value
    e0vec = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), qvec)
    e0vecext1 = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), maxqvec)
    e0vecext2 = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), minqvec)

    mine0vec = np.minimum(e0vecext1, e0vecext2)
    maxe0vec = np.maximum(e0vecext1, e0vecext2)

    if generous:
        qvec[np.where(bluevec<np.amin(lookup_table['bluemat']))] = 0
        e0vec[np.where((redvec == 0) | (greenvec == 0))] = 0
        e0vec[np.where((redvec/greenvec) > np.amax(lookup_table['redmat'] / lookup_table['greenmat']))] = 0

    return qvec.reshape(shape), e0vec.reshape(shape), minqvec.reshape(shape), maxqvec.reshape(shape), mine0vec.reshape(shape), maxe0vec.reshape(shape)


def calculate_E0_Q(redbright, greenbright, bluebright, lookup_table, minE0 = 150, generous = False):
    """
    Purpose: 
        - deprecated version of 'calculate_E0_Q_v2'
    """
    
    # Save the initial shape of arrays. They will be flattened and later reshaped back to this
    shape = greenbright.shape

    # Reshape brightness arrays to vectors
    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    # Cuts off the lookup table appropriately
    minE0ind = np.where(lookup_table['E0vec']>minE0)[0][0]
    
    # Estimates Q from blue brightness, along with error bars
    qvec, maxqvec, minqvec = q_interp(lookup_table['bluemat'], lookup_table['Qvec'], lookup_table['E0vec'], bluevec, minE0ind = minE0ind, maxbluebright = 'auto', interp = 'linear', plot = False)
 
    # Estimates E0 from red/green ratio and estimated Q value
    e0vec = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), qvec)
    e0vecext1 = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), maxqvec)
    e0vecext2 = e0_interp_general(lookup_table['redmat'] / lookup_table['greenmat'], lookup_table['Qvec'], lookup_table['E0vec'], (redvec / greenvec), minqvec)

    mine0vec = np.minimum(e0vecext1, e0vecext2)
    maxe0vec = np.maximum(e0vecext1, e0vecext2)

    if generous:
        qvec[np.where(bluevec<np.amin(lookup_table['bluemat']))] = 0
        e0vec[np.where((redvec == 0) | (greenvec == 0))] = 0
        e0vec[np.where((redvec/greenvec) > np.amax(lookup_table['redmat'] / lookup_table['greenmat']))] = 0

    return qvec.reshape(shape), e0vec.reshape(shape), minqvec.reshape(shape), maxqvec.reshape(shape), mine0vec.reshape(shape), maxe0vec.reshape(shape)


def calculate_Sig(q, e0, lookup_table, generous = False):
    """
    Purpose: 
        - given a processed lookup table dict from load_lookup_tables and arrays of  Q and E0, interpolates to calculate conductances
    Notes:
        - the 'generous' option tries to make sense of zeros in Q/E0 arrays by setting conductances to their minimum values
        - this function may throw an error when provided with a Q/E0 value that is nonzero but below the mininum entry in the table
    """
    
    # Saves shape of input
    shape = q.shape
    
    # Reshapes inputs to vectors
    qvec = q.reshape(-1)
    e0vec = e0.reshape(-1)
    
    # Linearly interpolates conductances
    SigP_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'], lookup_table['Qvec']], lookup_table['SigPmat'])
    SigH_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'], lookup_table['Qvec']], lookup_table['SigHmat'])

    # Initializes conductance vectors
    SigPout = np.zeros_like(qvec)
    SigHout = np.zeros_like(qvec)

    # Removes nans or zeros from Q and E0 vecs that would cause the interpolator to throw an error
    mask = (np.isnan(qvec) | np.isnan(e0vec)) | ((qvec == 0) | (e0vec == 0))

    # Reshapes input to the format the interpolator wants
    invec = np.asarray([e0vec[np.where(~mask)], qvec[np.where(~mask)]]).T
    
    # Puts NaNs where the interpolator would have failed
    SigPout[np.where(mask)] = np.nan
    SigPout[np.where(~mask)] = SigP_interp(invec)

    SigHout[np.where(mask)] = np.nan
    SigHout[np.where(~mask)] = SigH_interp(invec)


    # Tries to make sense of zeros in Q/E0 vectots
    if generous:
        SigPout[np.where( (qvec == 0) | (e0vec == 0) )] = np.amin(SigPout)
        SigHout[np.where( (qvec == 0) | (e0vec == 0) )] = np.amin(SigHout)
    return SigPout.reshape(shape),SigHout.reshape(shape)


def process_brightbin(fname, plot = True):
    """
    Purpose: 
        - process one of the GLOW brightness lookup tables
    """
    
    with open(fname) as f:
        recs = np.fromfile(f, dtype='float32') # open file
        params = recs[:20] # parameters
        nq = int(recs[20]) # dimensions of Q
        ne = int(recs[21]) # dimensions of E0
        Qvec = recs[22: 22 + nq] # Q vector
        E0vec = recs[22 + nq: 22 + nq + ne] # E0 vector
        bright = recs[22 + nq + ne:].reshape(ne, nq) # brightness matrix

        return params, Qvec, E0vec, bright

    
def process_sig3dbin(fname):
    """
    Purpose: 
        - process one of the GLOW conductance lookup tables
    """
    
    with open(fname) as f:
        
        recs = np.fromfile(f, dtype='float32') # open file
        
        params = recs[:20] # parameters
        nq = int(recs[20]) # dimensions of Q
        ne = int(recs[21]) # dimensions of E0
        nalt = int(recs[22]) # dimensions of alt
        Qvec = recs[23: 23 + nq] # Q vector
        E0vec = recs[23 + nq: 23 + nq + ne] # E0 vector
        altvec = recs[23 + nq + ne: 23 + nq + ne + nalt] # alt vector
        sig3d = recs[23+nq+ne+nalt:].reshape(nalt,ne,nq) # brightness data cube

        return params, Qvec, E0vec, altvec, sig3d
 
     
def q_interp(bright428 ,Qvec, E0vec, bluevec, minE0ind=0, maxbluebright='auto', interp='linear', plot = False):
    """
    Purpose: 
        - uses a GLOW lookup table to estimate Q from blue-line brightness
    Notes:
        - specify a minimum E0 index to crop the lookup table to
        - the uncertainty values returned are valid only under the assumption that true E0 is greater than or equal to the chosen cutoff
        - 'maxbluebright'' should be kept at 'auto' unless there's a good reason to change it - changing 'maxbluebright' only affects the error bars on returned Q, not Q itself
        - the Q interpolation has a built-in routine to estimate the max blue brightness that works well
        - this function recursively calls itself (once), when used with the 'auto' parameter for maxbluebright
    """
    
    # Automatically estimate where the inversion table "runs out of room" for very bright blue values
    if maxbluebright == 'auto':
        # Generate 50 blue brightnesses and invert to Q
    	testbluevec = np.linspace(0, np.amax(bright428), 50)
    	_, testmaxqvec, _ =  q_interp(bright428, Qvec, E0vec, testbluevec, minE0ind = minE0ind, maxbluebright = np.inf, interp = interp, plot = False)
    	
        # Find where the upper Q bound hits a ceiling, and mark it as the maximum blue brightness where upper Q bound can accurately be determined
    	medval = np.median(np.diff(testmaxqvec[np.where(~np.isnan(testmaxqvec))]))
    	firstbadind = np.where((np.diff(testmaxqvec)<(medval / 2)))[0][0]
    	maxbluebright = testbluevec[firstbadind]

    # Initialize vector of Q values
    qvec = []
    
    # Initialize vectors keeping track of max/min Q values due to lack of knowledge of E0 
    maxqvec = []
    minqvec = []

    # Iterate through blue brightness data points
    for blue in bluevec:
        qcross = [] # initialize vector storing all possible values Q could take for the given blue brightness
        e0cross = [] # initialize vector keeping track of the corresponding E0s for those Q values
        
        # Iterate through all E0s and find the Qs that correspond to the blue brightness data point
        for e0i in range(minE0ind, len(E0vec)):
            try:
                if interp == 'nearest':
                    qcross.append(Qvec[np.where(np.diff(np.sign(bright428[e0i, :] - blue)))[0][0]])
                    e0cross.append(E0vec[e0i])
                
                elif interp == 'linear': # linearly interpolate between Q values
                    guessind = np.where(np.diff(np.sign(bright428[e0i, :] - blue)))[0][0] # the index immediately before the interpolation range 
                    m = (bright428[e0i,guessind+1] - bright428[e0i,guessind]) / (Qvec[guessind + 1] - Qvec[guessind]) # the forward difference slope of brightness vs Q for this index
                    qi = (blue - bright428[e0i, guessind]) / m + Qvec[guessind] # the interpolated value of Q
                    qcross.append(qi)
                    e0cross.append(E0vec[e0i])
            except: # no Q consistent with this E0
                pass 
        qcross = np.asarray(qcross)

        if len(qcross)>0: # if at least one valid solution was found
            qvec.append(qcross[-1]) # take the Q value for very high E0 
            if blue>maxbluebright: # keep track of how much error we may have accrued by choosing that value
            	maxqvec.append(np.nan)
            else:
            	maxqvec.append(np.amax(qcross))
            minqvec.append(np.amin(qcross))
        else: # if no good solution was found
            if blue < np.amin(bright428): # extremely dim pixel
            	qvec.append(0.)
            	maxqvec.append(Qvec[0])
            	minqvec.append(0.)
            else:
            	qvec.append(np.nan)
            	maxqvec.append(np.nan)
            	minqvec.append(np.nan)
                
                # Troubleshooting print statements - sizes of vecs

    return np.asarray(qvec), np.asarray(maxqvec), np.asarray(minqvec)


def e0_interp_general(testmat, Qvec, E0vec, testvec, qinvec, degen_bounds = None):
    """
    Purpose: 
        - interpolates E0 from a "general" lookup table, aka any function F(Qvec,E0vec)
    Notes:
        - highly recommend that this be used with the table red_brightness/green_brightness
        - if you specify 'degen_bounds = [min_E0, max_E0]', they will be used in the event of a degeneracy (multiple valid E0 values found)
        - in the event that the degeneracy is not resolved, E0 will be set to NaN
    """
    
    e0out = []
    indvec = []
    
    for qin in qinvec: # closest Q index that undershoots the actual Q
        try:
            indvec.append(np.where(Qvec<qin)[0][-1])
        except:
            indvec.append(np.nan)
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else: # linear interpolation in Q
            fracind = (qinvec[i]-Qvec[indvec[i]])/(Qvec[indvec[i]+1]-Qvec[indvec[i]])
            curve0 = (1-fracind)*testmat[:,indvec[i]] + (fracind)*testmat[:,indvec[i]+1]
            curve = np.copy(curve0) - testvec[i]
            try:
                crossings = np.where(np.diff(np.sign(curve)))[0]
                guessind = crossings[0]
                if len(crossings)>1:
                    if (len(crossings)>2) or (np.diff(crossings)[0] != 1):
                        if degen_bounds == None:
                            guessind = np.nan
                        else: #attempt to resolve the degeneracy
                            crossings_mask = (E0vec[crossings]>degen_bounds[-1]) | (E0vec[crossings]<degen_bounds[0])
                            if len(crossings[~crossings_mask]) == 1:
                                print('degeneracy broken!')
                                guessind = crossings[~crossings_mask][0]
                            else:
                                print('failed to break degeneracy')
                                print(E0vec[crossings])
                                guessind = np.nan        
                # Linear interpolation in E0
                m = (curve[guessind+1] - curve[guessind])/(E0vec[guessind+1]-E0vec[guessind])
                e0i = -curve[guessind]/m + E0vec[guessind]
            except: 
                e0i = np.nan
            e0out.append(e0i)
            
    return np.asarray(e0out)


def e0_interp_general_nearest(testmat, Qvec, E0vec, testvec, qinvec):
    """
    Purpose: 
        - interpolates E0 using nearest neighbor interpolation
    """
    
    e0out = []
    indvec = np.asarray([np.argmin(np.abs(Qvec - qin)) for qin in qinvec])
    
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else:
            curve = testmat[:,indvec[i]]-testvec[i] # nearest neighbor interpolation
            try:
                e0i = E0vec[np.where(np.diff(np.sign(curve)))[0][0]]
            except:
                e0i = np.nan
            e0out.append(e0i)
            
    return np.asarray(e0out)
