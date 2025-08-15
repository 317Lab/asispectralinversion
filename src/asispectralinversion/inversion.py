import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate
import glob
import copy


# Height-integrate a 3d conductivity datacube to get a 2d conductance matrix
# Accounts for magnetic field angle from vertical to first order
def sig_integrator(sigmat,altvec):
    # Cumulative trapezoidal integration
    Sigmat = scipy.integrate.cumulative_trapezoid(sigmat,altvec/100,axis=0)[-1]
    # First order account for magnetic field angle from vertical
    return Sigmat

# Given a set of filenames, reads in the GLOW lookup tables and packages them into a struct
def load_lookup_tables(fname_red, fname_green, fname_blue, fname_sigp, fname_sigh, fname_edens, plot=False):
    # Read in: run parameters,Q vector, E0 vector, green brightness matrix from bin file
    params, Qvec, E0vec, greenmat = process_brightbin(fname_green,plot=plot)
    # Read in red and blue brightness matrices from bin files
    _, _, _, redmat = process_brightbin(fname_red,plot=plot)
    _, _, _, bluemat = process_brightbin(fname_blue,plot=plot)
    # Read in altitude vector and Pedersen conductivity datacube from bin file
    _, _, _, altvec, sigPmat = process_sig3dbin(fname_sigp)
    # Read in Hall conductivity datacube from bin file
    _, _, _, _, sigHmat = process_sig3dbin(fname_sigh)
    # Read in electron density datacube from bin file
    _, _, _, _, edensmat = process_sig3dbin(fname_edens)
    
    # Height-integrate conductivity datacubes to get conductance. First order correction for magnetic field angle.
    SigPmat = sig_integrator(sigPmat, altvec)
    SigHmat = sig_integrator(sigHmat, altvec)
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
        'SigHmat': SigHmat,
        'edensmat': edensmat
    }
    return lookup_table
    
# Given a directory, reads in the GLOW lookup tables and packages them into a struct
def load_lookup_tables_directory(directory, plot=False):
    fnamered = glob.glob(directory+'I6300*.bin')[0]
    fnamegreen = glob.glob(directory+'I5577*.bin')[0]
    fnameblue = glob.glob(directory+'I4278*.bin')[0]
    fnameped = glob.glob(directory+'ped3d*.bin')[0]
    fnamehall = glob.glob(directory+'hall3d*.bin')[0]
    fnameedens = glob.glob(directory+'edens*.bin')[0]
        
    # Airglow data
    fnamereda = glob.glob(directory+'airglow/'+'I6300*.bin')[0]
    fnamegreena = glob.glob(directory+'airglow/'+'I5577*.bin')[0]
    fnamebluea = glob.glob(directory+'airglow/'+'I4278*.bin')[0]
    fnamepeda = glob.glob(directory+'airglow/'+'ped3d*.bin')[0]
    fnamehalla = glob.glob(directory+'airglow/'+'hall3d*.bin')[0]
    fnameedensa = glob.glob(directory+'airglow/'+'edens*.bin')[0]
    
    v = load_lookup_tables(fnamered, fnamegreen, fnameblue, fnameped, fnamehall, fnameedens, plot=plot)
    va = load_lookup_tables(fnamereda, fnamegreena, fnamebluea, fnamepeda, fnamehalla, fnameedensa, plot=False)
    v['redbright_airglow'] = va['redmat']
    v['bluebright_airglow'] = va['bluemat']
    v['greenbright_airglow'] = va['greenmat']
    
    v['sigP_bg'] = va['sigPmat']
    v['sigH_bg'] = va['sigHmat']
    
    v['SigP_bg'] = va['SigPmat']
    v['SigH_bg'] = va['SigHmat']
    
    return v

# Given RGB brightness arrays (calibrated, in Rayleighs) and a lookup table for the correct night, estimates E0 and Q
# Setting minE0 constrains uncertainty values in Q, since for some nights some strange stuff happens at the bottom of the lookup tables.
# We often assume that visual signatures are insignificant below 150 eV, but that parameter can be set lower or higher as desired
# The generous option sets Q,E0 to zero instead of NaN when inversion fails but certain conditions are met (very dim pixels)
def calculate_E0_Q_v2(redbright,greenbright,bluebright,inlookup_table,minE0=150,secondorder=True,generous=False,plot=False):

    # Subtract out background brightnesses from the lookup table:
    
    # This was a shallow copy!!! We must replace it with a deep copy
    #lookup_table = inlookup_table.copy()
    lookup_table = copy.deepcopy(inlookup_table)
    lookup_table['redmat'] -= lookup_table['redbright_airglow'][0][0]
    lookup_table['greenmat'] -= lookup_table['greenbright_airglow'][0][0]
    lookup_table['bluemat'] -= lookup_table['bluebright_airglow'][0][0]
    
    # Save the initial shape of arrays. They will be flattened and later reshaped back to this
    shape = greenbright.shape

    # Reshape brightness arrays to vectors
    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    # Cuts off the lookup table appropriately
    minE0ind = np.where(lookup_table['E0vec']>minE0)[0][0]
    
    # Estimates Q from blue brightness, along with error bars
    qvec, maxqvec, minqvec = q_interp(lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],bluevec,minE0ind=minE0ind,maxbluebright='auto',interp='linear',plot=plot)

    # Estimates E0 from red/green ratio and estimated Q value
    e0vec = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),qvec,generous=generous)
    e0vecext1 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),maxqvec,generous=generous)
    e0vecext2 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),minqvec,generous=generous)

    mine0vec = np.minimum(e0vecext1,e0vecext2)
    maxe0vec = np.maximum(e0vecext1,e0vecext2)
    
    # Perform a second fit for Q using our E0 bounds
    if secondorder:
    	qvec, maxqvec, minqvec = q_interp_constrained(lookup_table['bluemat'],mine0vec,maxe0vec,lookup_table['Qvec'],lookup_table['E0vec'],bluevec,backupminE0ind=minE0ind,interp='linear',plot=plot)    
    	e0vec = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),qvec,generous=generous)
    	e0vecext1 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),maxqvec,generous=generous)
    	e0vecext2 = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),minqvec,generous=generous)
    if generous:
        qvec[np.where(bluevec<np.amin(lookup_table['bluemat']))] = 0#lookup_table['E0vec'][0]
        #e0vec[np.where(bluevec<np.amin(lookup_table['bluemat']))] = 0
        #e0vec[np.where((redvec == 0) | (greenvec == 0))] = 0#lookup_table['E0vec'][0]
        e0vec[np.where((redvec/greenvec) > np.amax(lookup_table['redmat']/lookup_table['greenmat']))] = 0#lookup_table['E0vec'][0]

    return qvec.reshape(shape),e0vec.reshape(shape),minqvec.reshape(shape),maxqvec.reshape(shape),mine0vec.reshape(shape),maxe0vec.reshape(shape)

# like calculate_E0_Q_v2 but estimates both Q and E0 without explicitly using blue brightness
# Note that this does not include error bars
def calculate_E0_Q_v2_RGonly(redbright,greenbright,bluebright,inlookup_table,minE0=150,checkagreement=True,cutoffgoodness=0.3,generous=False,plot=False):

    # Subtract out background brightnesses from the lookup table:
    
    # This was a shallow copy!!! We must replace it with a deep copy
    #lookup_table = inlookup_table.copy()
    lookup_table = copy.deepcopy(inlookup_table)
    lookup_table['redmat'] -= lookup_table['redbright_airglow'][0][0]
    lookup_table['greenmat'] -= lookup_table['greenbright_airglow'][0][0]
    lookup_table['bluemat'] -= lookup_table['bluebright_airglow'][0][0]
    
    # Save the initial shape of arrays. They will be flattened and later reshaped back to this
    shape = greenbright.shape

    # Reshape brightness arrays to vectors
    redvec = redbright.reshape(-1)
    greenvec = greenbright.reshape(-1)
    bluevec = bluebright.reshape(-1)

    # Cuts off the lookup table appropriately
    minE0ind = np.where(lookup_table['E0vec']>minE0)[0][0]
    
    # # Estimates Q from blue brightness, along with error bars
    # qvec, maxqvec, minqvec = q_interp(lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],bluevec,minE0ind=minE0ind,maxbluebright='auto',interp='linear',plot=plot)
    qmat = q_interp_RG(lookup_table['redmat'],lookup_table['greenmat'],lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],redbright,greenbright,bluebright,checkagreement=checkagreement,cutoffgoodness=cutoffgoodness,plot=plot)
    qvec = np.copy(qmat).reshape(-1)
    
    #testqvec, _, _ = q_interp(lookup_table['bluemat'],lookup_table['Qvec'],lookup_table['E0vec'],bluevec,minE0ind=minE0ind,maxbluebright='auto',interp='linear',plot=plot)
    #print(len(testqvec))
    # Estimates E0 from red/green ratio and estimated Q value
    e0vec = e0_interp_general(lookup_table['redmat']/lookup_table['greenmat'],lookup_table['Qvec'],lookup_table['E0vec'],(redvec/greenvec),qvec,generous=generous)
    if generous:
        e0vec[np.where((redvec/greenvec) > np.amax(lookup_table['redmat']/lookup_table['greenmat']))] = 0#lookup_table['E0vec'][0]
    return qmat,e0vec.reshape(shape)




# Given a processed lookup table dict from load_lookup_tables and arrays of  Q and E0, interpolates to calculate conductances
# The generous option tries to make sense of zeros in Q/E0 arrays by setting conductances to their minimum values
# Note that this function may throw an error when provided with a Q/E0 value that is nonzero but below the mininum entry in the table.
# This should be fixed soon! -Alex
def calculate_Sig(q,e0,lookup_table,generous=False):
    # Saves shape of input
    shape = q.shape
    
    # Reshapes inputs to vectors
    qvec = q.reshape(-1)
    e0vec = e0.reshape(-1)
    
    # Linearly interpolates conductances
    SigP_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'],lookup_table['Qvec']],lookup_table['SigPmat'])
    SigH_interp = scipy.interpolate.RegularGridInterpolator([lookup_table['E0vec'],lookup_table['Qvec']],lookup_table['SigHmat'])

    # Initializes conductance vectors
    SigPout = np.zeros_like(qvec)
    SigHout = np.zeros_like(qvec)

    # Removes nans or extreme values from Q and E0 vecs that would cause the interpolator to throw an error
    nancond = np.isnan(qvec) | np.isnan(e0vec)
    lessercond = (qvec < np.amin(lookup_table['Qvec'])) | (e0vec < np.amin(lookup_table['E0vec']))
    greatercond = ((qvec > np.amax(lookup_table['Qvec'])) | (e0vec > np.amax(lookup_table['E0vec'])))
    mask = (nancond | lessercond) | greatercond

    # Reshapes input to the format the interpolator wants
    invec = np.asarray([e0vec[np.where(~mask)],qvec[np.where(~mask)]]).T
    
    # Puts NaNs where the interpolator would have failed
    SigPout[np.where(mask)] = np.nan
    SigPout[np.where(~mask)] = SigP_interp(invec)

    SigHout[np.where(mask)] = np.nan
    SigHout[np.where(~mask)] = SigH_interp(invec)


    # Tries to make sense of zeros in Q/E0 vectors
    if generous:
        SigPout[np.where( lessercond )] = lookup_table['sigP_bg']
        SigHout[np.where( lessercond )] = lookup_table['sigH_bg']
    return SigPout.reshape(shape),SigHout.reshape(shape)



# Process one of the GLOW brightness lookup tables
def process_brightbin(fname,plot=False):
    with open(fname) as f:
        # Open file
        recs = np.fromfile(f, dtype='float32')
        # Parameters
        params = recs[:20]
        # Dimensions of Q,E0
        nq = int(recs[20])
        ne = int(recs[21])
        # Q vector and E0 vector
        Qvec = recs[22:22+nq]
        E0vec = recs[22+nq:22+nq+ne]
        # Brightness matrix
        bright = recs[22+nq+ne:].reshape(ne,nq)
        # Pcolor as a sanity check
        if plot:
            plt.pcolormesh(Qvec,E0vec,bright,shading='auto')
            plt.xlabel('Q')
            plt.ylabel('E0')
            plt.title(fname.split('/')[-1][1:5])
            plt.colorbar()
            plt.show()
        # Return data
        return params,Qvec,E0vec,bright
    
# Process one of the GLOW conductance lookup tables
def process_sig3dbin(fname):
    with open(fname) as f:
        # Open file
        recs = np.fromfile(f, dtype='float32')
        # Parameters
        params = recs[:20]
        # Dimensions of Q,E0,alt
        nq = int(recs[20])
        ne = int(recs[21])
        nalt = int(recs[22])
        # Q vector, E0 vector, alt vector
        Qvec = recs[23:23+nq]
        E0vec = recs[23+nq:23+nq+ne]
        altvec = recs[23+nq+ne:23+nq+ne+nalt]
        # Data cube
        sig3d = recs[23+nq+ne+nalt:].reshape(nalt,ne,nq)
        # Return data
        return params,Qvec,E0vec,altvec,sig3d
    
    
# Uses a GLOW lookup table to estimate Q from blue line brightness
# We also specify a minimum E0 index to crop the lookup table to. 

# Note that the uncertainty values returned are valid only under the assumption that
# true E0 is greater than or equal to the chosen cutoff

# maxbluebright should be kept at 'auto' unless there's a good reason to change it.
# The Q interpolation has a built-in routine to estimate the max blue brightness that works well
# Changing maxbluebright only affects the error bars on returned Q, not Q itself
def q_interp(bright428,Qvec,E0vec,bluevec,minE0ind=0,maxbluebright='auto',interp='linear',plot=False):
    # Automatically estimate where the inversion table "runs out of room" for very bright blue values
    # This involves a recursive evaluation...
    if maxbluebright == 'auto':
        # Generate 50 blue brightnesses and invert to Q
        # Note that this function recursively calls itself (once), when used with the 'auto' parameter for maxbluebright
        # This lets it determine a reasonable bound for blue brightness, above which inversions may be inaccurate
        testbluevec = np.linspace(0,np.amax(bright428),50)
        _,testmaxqvec,_ =  q_interp(bright428,Qvec,E0vec,testbluevec,minE0ind=minE0ind,maxbluebright=np.inf,interp=interp,plot=False)
        # Find where the upper Q bound hits a ceiling, and mark it as the maximum blue brightness where upper Q bound can accurately be determined
        medval = np.median(np.diff(testmaxqvec[np.where(~np.isnan(testmaxqvec))]))
        firstbadind = np.where((np.diff(testmaxqvec)<(medval/2)))[0][0]
        maxbluebright = testbluevec[firstbadind]
        if plot:
            plt.scatter(testbluevec,testmaxqvec,color='black')
            plt.scatter(testbluevec[firstbadind:],testmaxqvec[firstbadind:],color='red')
            plt.title('Max possible Q, ceiling hit in red')
            plt.xlabel('blue brightness')
            plt.show()
    
    # Initialize vector of Q values
    qvec = []
    # Initialize vectors keeping track of max/min Q values due to lack of knowledge of E0 
    maxqvec = []
    minqvec = []

    # Pcolor the whole lookup table
    if plot:
        plt.pcolormesh(Qvec,E0vec,bright428,shading='auto')
        plt.xlabel('Q')
        plt.ylabel('E0')

    # Iterate through blue brightness data points
    for blue in bluevec:
    
        # Check to see if pixel is too bright
        if (blue < np.amin(bright428)) | (blue > np.amax(bright428)):
            qvec.append(np.nan)
            maxqvec.append(np.nan)
            minqvec.append(np.nan)
            continue
            
        # Initialize vector storing all possible values Q could take for the given blue brightness
        qcross = []
        # Initialize vector keeping track of the corresponding E0s for those Q values
        e0cross = []
        # Iterate through all E0s and find the Qs that correspond to the blue brightness data point
        for e0i in range(minE0ind,len(E0vec)):
            try:
                if interp=='nearest':
                    qcross.append(Qvec[np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]])
                    e0cross.append(E0vec[e0i])
                # We linearly interpolate between Q values
                elif interp=='linear':
                    # The index immediately before the interpolation range 
                    guessind = np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]
                    # The forward difference slope of brightness vs Q for this index
                    m = (bright428[e0i,guessind+1] - bright428[e0i,guessind])/(Qvec[guessind+1]-Qvec[guessind])
                    # The interpolated value of Q
                    qi = (blue-bright428[e0i,guessind])/m + Qvec[guessind]
                    qcross.append(qi)
                    e0cross.append(E0vec[e0i])
            except:
                # no Q consistent with this E0
                pass
        qcross = np.asarray(qcross)

        # If at least one valid solution was found
        if len(qcross)>0:
            # We take the Q value for very high E0
            qvec.append(qcross[-1])
            # We keep track of how much error we may have accrued by choosing that value
            if blue>maxbluebright:
                maxqvec.append(np.nan)
            else:
                maxqvec.append(np.amax(qcross))
            minqvec.append(np.amin(qcross))
            # Plot the curves of possible Q solutions for each data point.
            # For large-dimensional input, this gets busy/useless/slow pretty fast!
            if plot:
                plt.plot(qcross,e0cross)
                plt.scatter(qcross[-1],e0cross[np.argmin(np.abs(qcross-qcross[-1]))],color='black',s=50)
        # If no good solution was found
        else:
            qvec.append(np.nan)
            maxqvec.append(np.nan)
            minqvec.append(np.nan)

    # Plot the minimum value of E0 considered for the inversion
    if plot:
        plt.plot([Qvec[0],Qvec[-1]],[E0vec[minE0ind],E0vec[minE0ind]],color='black',linewidth=5)
        plt.title('Solution sets for a given blue line brightness ')

    return np.asarray(qvec),np.asarray(maxqvec),np.asarray(minqvec)
    
# Same as q_interp, but now we specify our E0 bounds to constrain what Q could be more accurately
def q_interp_constrained(bright428,minE0vec,maxE0vec,Qvec,E0vec,bluevec,backupminE0ind=0,interp='linear',plot=False):    
    # Initialize vector of Q values
    qvec = []
    # Initialize vectors keeping track of max/min Q values due to lack of knowledge of E0 
    maxqvec = []
    minqvec = []

    # Pcolor the whole lookup table
    if plot:
        plt.pcolormesh(Qvec,E0vec,bright428,shading='auto')
        plt.xlabel('Q')
        plt.ylabel('E0')

    # Iterate through blue brightness data points
    for i in range(len(bluevec)):
        blue = bluevec[i]
        
        if np.isnan(minE0vec[i]):
            minE0ind = backupminE0ind
        else:
            minE0ind = np.where(E0vec<=minE0vec[i])[0][-1]
        if np.isnan(maxE0vec[i]):
            maxE0ind = len(E0vec)-1
        else:
            maxE0ind = np.where(E0vec>=maxE0vec[i])[0][0]
            
        # Initialize vector storing all possible values Q could take for the given blue brightness
        qcross = []
        # Initialize vector keeping track of the corresponding E0s for those Q values
        e0cross = []
        # Iterate through all E0s and find the Qs that correspond to the blue brightness data point
        for e0i in range(minE0ind,maxE0ind+1):
            try:
                if interp=='nearest':
                    qcross.append(Qvec[np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]])
                    e0cross.append(E0vec[e0i])
                # We linearly interpolate between Q values
                elif interp=='linear':
                    # The index immediately before the interpolation range 
                    guessind = np.where(np.diff(np.sign(bright428[e0i,:]-blue)))[0][0]
                    # The forward difference slope of brightness vs Q for this index
                    m = (bright428[e0i,guessind+1] - bright428[e0i,guessind])/(Qvec[guessind+1]-Qvec[guessind])
                    # The interpolated value of Q
                    qi = (blue-bright428[e0i,guessind])/m + Qvec[guessind]
                    qcross.append(qi)
                    e0cross.append(E0vec[e0i])
            except:
                # no Q consistent with this E0
                pass
        qcross = np.asarray(qcross)

        # If at least one valid solution was found
        if len(qcross)>0:
            # We take the median Q value
            qvec.append(np.median(qcross))
            # We keep track of how much error we may have accrued by choosing that value
            if (minE0vec[i]>=E0vec[0]) & (maxE0vec[i]<=E0vec[-1]):
                maxqvec.append(np.amax(qcross))
                minqvec.append(np.amin(qcross))
            else:
                maxqvec.append(np.nan)
                minqvec.append(np.nan)
                
            # Plot the curves of possible Q solutions for each data point.
            # For large-dimensional input, this gets busy/useless/slow pretty fast!
            if plot:
                plt.plot(qcross,e0cross)
                plt.scatter(qcross[-1],e0cross[np.argmin(np.abs(qcross-qcross[-1]))],color='black',s=50)
        # If no good solution was found
        else:
            qvec.append(np.nan)
            maxqvec.append(np.nan)
            minqvec.append(np.nan)

    # Plot the minimum value of E0 considered for the inversion
    if plot:
        plt.plot([Qvec[0],Qvec[-1]],[E0vec[minE0ind],E0vec[minE0ind]],color='black',linewidth=5)
        plt.title('Solution sets for a given blue line brightness ')

    return np.asarray(qvec),np.asarray(maxqvec),np.asarray(minqvec)

# This is an experimental routine to calculate Q from red and green brightness without using blue
# One might use it when blue data are unavailable or of low quality
# It can, optionally, assess the consistency of each RGB pixel to what GLOW expects, and
# mask out data that is sufficiently inconsistent.
# It takes in: 

# bright630,bright558,bright428: red,green,blue lookup tables
# Qvec,E0vec: Q and E0 values for those lookup tables
# redbright,greenbright,bluebright: red,green,blue pixel brightnesses in Rayleighs, to be inverted
# checkagreement: assess the consistency of the data to what GLOW predicts
# cutoffgoodness: very roughly, cutoffgoodness=0.3 discards data points with brightnesses that diverge from 
#     GLOW by a factor of 2. Higher cutoffs discard more points, lower ones discard less
# plot: set to True to include diagnostic plots
def q_interp_RG(bright630,bright558,bright428,Qvec,E0vec,redbright,greenbright,bluebright,checkagreement=True,cutoffgoodness=0.3,plot=False):
    # A new coordinate system for R,G,B brightness
    def newcoords(r,g,b):
        newx = (np.log(g) - np.log(r))
        newy = (np.log(g) - np.log(b))
        newz = (np.log(r) + np.log(g) + np.log(b))
    
        return newx,newy,newz
    
    ################################################################
    # Extracting the part of the lookup table that is relevant to this inversion
    ################################################################
    
    # Pulling out the brightest pixels that could be inverted
    goodrange = np.where(~np.isnan((redbright+greenbright+bluebright).reshape(-1)))[0]
    
    # Brightest R,G,B pixels
    redmax = 1*np.amax(redbright.reshape(-1)[goodrange])
    greenmax = 1*np.amax(greenbright.reshape(-1)[goodrange])
    bluemax = 1*np.amax(bluebright.reshape(-1)[goodrange])
    
    # Extracting the lookup table brightnesses as vectors
    redin = bright630.reshape(-1)
    greenin = bright558.reshape(-1)
    bluein = bright428.reshape(-1)
    
    # Specifying the part of the lookup table that's useful for this inversion
    goodrange = np.where((redin<redmax) & (greenin<greenmax) & (bluein<bluemax))[0]
    
    # Decimate if the lookup table is too big to interpolate
    if len(goodrange)>20000:
    	dec = int(np.ceil(np.sqrt(len(goodrange)/20000)))
    	#print('decimating')
    	redin_q = bright630[::dec,::dec].reshape(-1)
    	greenin_q = bright558[::dec,::dec].reshape(-1)
    	bluein_q = bright428[::dec,::dec].reshape(-1)
    	goodrange = np.where((redin_q<redmax) & (greenin_q<greenmax) & (bluein_q<bluemax))[0]
    	
    	# Meshgrid Q,E0 to get the full set of tuples
    	Qmat,E0mat = np.meshgrid(Qvec[::dec],E0vec[::dec])
    else:
    	dec=1
    	redin_q = bright630.reshape(-1)
    	greenin_q = bright558.reshape(-1)
    	bluein_q = bright428.reshape(-1)
    	# Meshgrid Q,E0 to get the full set of tuples
    	Qmat,E0mat = np.meshgrid(Qvec,E0vec)
    # Trimming lookup table
    redplot = redin_q[goodrange]
    greenplot = greenin_q[goodrange]
    blueplot = bluein_q[goodrange]
    
    E0plot = E0mat.reshape(-1)[goodrange]
    Qplot = Qmat.reshape(-1)[goodrange]
    
    ################################################################
    ################################################################
    
    # Converting lookup table to new coords
    newx,newy,newz = newcoords(redplot,greenplot,blueplot)
    # Converting data to new coords
    newxdata,newydata,newzdata = newcoords(redbright,greenbright,bluebright)
    
    # Function to invert red,green brightness to Q
    Qinterp = scipy.interpolate.RBFInterpolator(np.asarray([np.log(redplot),np.log(greenplot)]).T,Qplot,kernel='cubic')
    
    # Evaluate function on data
    Qdata = Qinterp(np.asarray([np.log(redbright).reshape(-1),np.log(greenbright).reshape(-1)]).T).reshape(newxdata.shape)
    
    if plot & len(Qdata.shape)>1:
        print('Qdata shape='+str(Qdata.shape))
        plt.pcolormesh(Qdata,vmin=0,vmax=12)
        plt.colorbar()
        plt.title('Q from log r,log g, cubic')
        plt.show()
    
    if checkagreement:
        ################################################################
        # Evaluating the agreement of data to GLOW - when can we trust it
        ################################################################
        
        ################ We use median binning to fit the GLOW-derived 
        ################ x and y coords to a single curve
        nbins = int(np.round(35))
        #print('nbins= '+str(nbins))
        newxs0 = np.linspace(np.amin(newx),np.amax(newx),nbins)
        newys = np.zeros(len(newxs0)-1)
        
        for i in range(len(newys)):
            xrange = np.where( (newx>=newxs0[i]) & (newx < newxs0[i+1]) )[0]
            newys[i] = np.median(newy[xrange])
        
        # We want newxs to index the middle of each bin, not the start
        newxs = np.asarray([(newxs0[i]+newxs0[i+1])/2 for i in range(len(newxs0)-1)])
        
        # Remove NaNs
        goodrange = np.where(~np.isnan(newys))[0]
        newxs = newxs[goodrange]
        newys = newys[goodrange]
        
        ################ Now we construct an interpolator to determine how
        ################ consistent a data point is with the GLOW model
        
        # Cubic spline interpolate GLOW curve
        cs = scipy.interpolate.CubicSpline(newxs,newys)
        
        # Evaluate cubic spline on finer grid
        xcurve = np.linspace(np.amin(newxs),np.amax(newxs),1000)
        ycurve = cs(xcurve)
        
        # Construct a new x-y grid
        newxfordiffvec = np.linspace(np.amin(xcurve)-2,np.amax(xcurve)+2,50)
        newyfordiffvec = np.linspace(np.amin(ycurve)-2,np.amax(ycurve)+2,51)
        newyfordiff,newxfordiff = np.meshgrid(newyfordiffvec,newxfordiffvec)
        
        # Evaluate the distance from the GLOW curve, everywhere on this grid
        distmat = np.zeros_like(newxfordiff)
        for i in range(len(newxfordiffvec)):
            for j in range(len(newyfordiffvec)):
                distmat[i,j] = np.amin((ycurve-newyfordiff[i,j])**2 + (xcurve-newxfordiff[i,j])**2)
        
        if plot:
            plt.title('distances')
            plt.pcolormesh(newxfordiff,newyfordiff,distmat,vmax=2)
            plt.colorbar()
            #plt.show()
            plt.scatter(xcurve,ycurve,color='orange')
            plt.show()
        
        # Construct the interpolator
        distinterp = scipy.interpolate.RBFInterpolator(np.asarray([newxfordiff.reshape(-1),newyfordiff.reshape(-1)]).T,distmat.reshape(-1),kernel='cubic')
        
        # Evaluate it on our data
        distdata = distinterp(np.asarray([newxdata.reshape(-1),newydata.reshape(-1)]).T).reshape(newxdata.shape)
        
        if plot:
            plt.scatter(newxdata.reshape(-1),newydata.reshape(-1),zorder=-1,s=0.1,c=(1-distdata).reshape(-1),cmap='plasma',vmin=cutoffgoodness,vmax=1)
            plt.colorbar()
            plt.plot(xcurve,ycurve,color='orange')
            plt.title('Agreement with GLOW')
            plt.xlabel('log(green/red)')
            plt.ylabel('log(green/blue)')
            plt.show()
            
            if len(Qdata.shape)>1:
                plt.pcolormesh(1-distdata,vmin=cutoffgoodness,cmap='plasma')
                plt.title('goodness of agreement with GLOW')
                plt.colorbar()
                plt.show()
        
        # Mask out regions where data are inconsistent with GLOW
        badrange = np.where( ((1-distdata) < cutoffgoodness) | np.isnan(bluebright+distdata) )
        Qdata[badrange] = np.nan
        
        if plot & len(Qdata.shape)>1:
            plt.pcolormesh(Qdata,vmin=0,vmax=20)
            plt.title('Q, trustable')
            plt.colorbar()
    return Qdata


# Interpolates E0 from a "general" lookup table, aka any function F(Qvec,E0vec).
# We highly recommend that this be used with the table red_brightness/green_brightness.

# If you specify degen_bounds = [min_E0, max_E0], they will be used in the event of a degeneracy
# (multiple valid E0 values found). In the event that the degeneracy is not resolved, E0 will
# be set to NaN.
def e0_interp_general(testmat,Qvec,E0vec,testvec,qinvec,generous=False,degen_bounds=None):
    e0out = []
    
    # indvec = np.asarray([np.argmin(np.abs(Qvec-qin)) for qin in qinvec])
    
    # Closest Q index that undershoots the actual Q
    indvec = []
    for qin in qinvec:
        try:
            indvec.append(np.where(Qvec<qin)[0][-1])
        except:
            indvec.append(np.nan)
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else:
            #curve = testmat[:,indvec[i]]-testvec[i]
            # Linear interpolation in Q
            if indvec[i] == len(Qvec) - 1:
                print('new behaviour'+str(qin))
                if generous:
                    curve = np.copy(testmat[:,-1]) - testvec[i]
                else:
                    e0out.append(np.nan)
                    continue
            else:
                fracind = (qinvec[i]-Qvec[indvec[i]])/(Qvec[indvec[i]+1]-Qvec[indvec[i]])
                curve0 = (1-fracind)*testmat[:,indvec[i]] + (fracind)*testmat[:,indvec[i]+1]
                curve = np.copy(curve0) - testvec[i]
            
            try:
                crossings = np.where(np.diff(np.sign(curve)))[0]
                guessind = crossings[0]
                
                # Hopefully there is only one nontrivial solution for E0
                if len(crossings)>1:
                    # Multiple nontrivial crossings. We (maybe) cannot trust this result
                    if (len(crossings)>2) or (np.diff(crossings)[0] != 1):
                        # print('bad, mult cross')
                        # print(E0vec[crossings])
                        
                        if degen_bounds == None:
                            #print('degenerate output')
                            # This crashes the evaluation of e0i
                            guessind = np.nan
                        else:
                            # print('trying to break degeneracy')
                            # We attempt to resolve the degeneracy
                            crossings_mask = (E0vec[crossings]>degen_bounds[-1]) | (E0vec[crossings]<degen_bounds[0])
                            #print(E0vec[crossings])
                            if len(crossings[~crossings_mask]) == 1:
                                print('degeneracy broken!')
                                guessind = crossings[~crossings_mask][0]
                            else:
                                print('failed to break degeneracy')
                                print(E0vec[crossings])
                                # This crashes the evaluation of e0i
                                guessind = np.nan
                        
                # Linear interpolation in E0
                m = (curve[guessind+1] - curve[guessind])/(E0vec[guessind+1]-E0vec[guessind])
                e0i = -curve[guessind]/m + E0vec[guessind]

            except:
                e0i = np.nan
                
            e0out.append(e0i)
            
    return np.asarray(e0out)

def e0_interp_general_nearest(testmat,Qvec,E0vec,testvec,qinvec):
    e0out = []
    
    # Nearest neighbor interpolation
    indvec = np.asarray([np.argmin(np.abs(Qvec-qin)) for qin in qinvec])
    
    for i in range(len(testvec)):
        if np.isnan(indvec[i]):
            e0out.append(np.nan)
        else:
            # Nearest neighbor interpolation
            curve = testmat[:,indvec[i]]-testvec[i]
            try:
                e0i = E0vec[np.where(np.diff(np.sign(curve)))[0][0]]
                #print(np.diff(np.sign(curve)))
                #print(np.where(np.diff(np.sign(curve)))[0])
            except:
                e0i = np.nan
            e0out.append(e0i)
    return np.asarray(e0out)
