import numpy as np
import datetime as dt
import os
import glob
import re
from bs4 import BeautifulSoup
import requests
import wget
from .transformation import feed_data

"""
Purpose of this script:
    - handle input data
        - GLOW lookup table for a specific hour of an event
        - ASI PNGs from Don's archive (http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/)
    - gets everything into a nice h5 format before preprocessing and inversion
    - contains main function call that feeds processed h5 files into inversion pipeline
"""

def seconds_since_midnight(time):
    """
    Purpose:
        - Turns 24 hour string time into float of seconds past midnight
    """
    
    #print("Finding seconds since midnight...")
    
    return 60 * 60 * time.hour + 60 * time.minute + time.second


 
def genlinks(date, starttime, endtime):
    """
    Purpose:
        - Given a date, start and end time, finds links to every DASC frame

    Input:
        date: datetime.date
            Date of this inversion
        starttime: datetime.time
            Start of time interval to get links for
        endtime: datetime.time
            End of time interval to get links for
    """
    
    print("Building link to archive...")
    
    startsecs = seconds_since_midnight(starttime) # seconds from midnight of start time
    endsecs = seconds_since_midnight(endtime) # seconds from midnight of end time

    # Construct base url
    url = 'http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/' + f'{date:%Y}/{date:%Y%m%d}/{starttime:%H}'
    print(url)
    
    # Pull and process file list
    soup = BeautifulSoup(requests.get(url).text,'html.parser')
    # First 5 links are not actual events
    rawlinks = soup.find_all('a')[5:]
    # Turning into a numpy array
    links = np.asarray(rawlinks).flatten()

    if len(links)==0:
        print('no imagery found')
        raise Exception('no links')

    # Extracting the necessary information
    # Each row is: 
    # [seconds since midnight, color, time]

    # Initializing array with zeros
    newlinks = np.zeros([len(links), 3])
    for i in range(len(links)):
        time,color = links[i].split('.')[0].split('_')[2:] # time and color from filename
        t0 = dt.datetime.strptime(time, '%H%M%S')
        newlinks[i, 0] = seconds_since_midnight(t0) # seconds past midnight
        newlinks[i, 1] = color
        newlinks[i, 2] = time

    # Finding time of each frame, separated by color
    bluesecs = newlinks[:, 0][np.where(newlinks[:, 1] == 428)]
    greensecs = newlinks[:, 0][np.where(newlinks[:, 1] == 558)]
    redsecs = newlinks[:, 0][np.where(newlinks[:, 1] == 630)]
    
    # We arbitrarily choose our first frame to be blue so that the frames are taken in 'b,g,r' order:
    try:
        s0 = bluesecs[np.where(bluesecs < startsecs)[0][-1]]
    except:
        s0 = bluesecs[0]
    # Therefore our last frame must be red
    try:
        s1 = redsecs[np.where(redsecs > endsecs)[0][0]]
    except:
        s1 = redsecs[-1]
        
    # We pull out the indices of the first and last frames we want to download
    startind = np.where(newlinks[:, 0] >= s0)[0][0]
    endind = np.where(newlinks[:, 0] <= s1)[0][-1]

    linkstrim = list(links[startind: endind + 1])
    linksout = [url + '/' + link for link in linkstrim]
    return linksout, linkstrim


def download_imagery(date, starttime, endtime, folder):
    """
    Purpose:
        - Pulls all DASC png imagery between <starttime> and <endtime> (UT) on the date of 
          <date>

    Input:
        date: datetime.date
            Date of this inversion
        starttime: datetime.time
            Start of time of download interval
        endtime: datetime.time
            End of time of download interval
        folder: str
            Output folder to save the downloaded images to
    """
    
    print("Downloading imagery from archive...")

    print(date, starttime, endtime)
    
    # folder should already exist, but if not:
    if not os.path.exists(folder):
        os.makedirs(folder)
    

    if starttime.hour == endtime.hour:
        # If start and end time in same hour, do single pull
        links, fnames = genlinks(date, starttime, endtime)
    else:
        # If start and end time in different hours, do multiple pulls
        links = list()
        fnames = list()
        for h in range(starttime.hour, endtime.hour+1):
            if h == starttime.hour:
                st = starttime
                et = starttime.replace(minute=59, second=59)
            elif h == endtime.hour:
                st = endtime.replace(minute=0, second=0)
                et = endtime
            else:
                st = dt.time(h, 0, 0)
                et = dt.time(h, 59, 59)
            links0, fnames0 = genlinks(date, st, et)
            links.extend(links0)
            fnames.extend(fnames0)


    # Only download file if it doesn't already exist    
    for i in range(len(links)):
        file_path = os.path.join(folder, fnames[i])
        #print(file_path)
        if os.path.exists(file_path):
            continue
        else:
            wget.download(links[i], out=file_path)
        
            

def file_data(date, starttime, endtime, folder):
    """
    Purpose:
        - download data and generate lists of files for automatic processing

    Input:
        date: datetime.date
            Date of this inversion
        starttime: datetime.time
            Start of time of download interval
        endtime: datetime.time
            End of time of download interval
        folder: str
            Output folder to save the downloaded images to
    """ 
    
    print("Running all data wrangling processes...")
    
    # Main function calls
    download_imagery(date, starttime, endtime, folder)

    grouped_files = dict()
   
    # Sort png by timestamp in filenames
    def get_timestamp(file):
        match = re.search(r'_(\d{8}_\d{6})', file)
        #return match.group(1) if match else '' # just an empty string if nothing there
        tstmp = dt.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return tstmp

    # Sort images into groups of 3
    for color in ['0428', '0558', '0630']:
        file_list = sorted(glob.glob(os.path.join(folder, 'PKR_*_'+color+'.png')))
        
        # Group into sets of 3
        gf = [file_list[i: i+3] for i in range(0, len(file_list), 3)]
        grouped_files[color] = gf

    # generate list of timestamps
    time_range = list()
    for i in range(len(grouped_files['0428'])):
        bluefiles = grouped_files['0428'][i]
        blue_tstmp = [get_timestamp(f)  for c in ['0428', '0558', '0630'] for f in grouped_files[c][i]]
        time_range.append(min(blue_tstmp))

    return time_range, grouped_files['0428'], grouped_files['0558'], grouped_files['0630']




def process_grouped_files(tstmps, files0428, files0558, files0630, folder, base_outdir, method='rgb', skymap_file=None, prep_kwarg=dict()):
    """
    Purpose:
        - automatically process groups of png files
        - this is what allows for the process to be time varying!!

    Input:
        tstmps: list of datetime.datetime
            List of timestamp of each file group
        files0428: list
            List of list of the 0428 files in each processing block
        files0558: list
            List of list of the 0558 files in each processing block
        files0630: list
            List of list of the 0630 files in each processing block
        folder: str
            Directory where inversion lookup tables generated by glow are saved
        base_outdir: str
            Directory to save the output files to
        method: str
            Method to use for inversion, either 'rgb' (default) for standard method or 'rg' for only using red/green images
        skymap_file: str (optional)
            Path to skymap file when not using default (PKR)
        prep_kwarg: dict
            Additional optional keyword arguments for prepare_data()
            
    """ 

    # Iterate through groups of files and compute inversion on each
    for i in range(len(tstmps)):
        print(tstmps[i])
        output_name = os.path.join(base_outdir, f'asi_invert_{tstmps[i]:%Y%m%d_%H%M%S}.h5')
        feed_data(tstmps[i], folder, files0428[i], files0558[i], files0630[i], output_name, method=method, skymap_file=skymap_file, **prep_kwarg)
 

