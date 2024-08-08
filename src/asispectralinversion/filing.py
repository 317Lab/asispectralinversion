import numpy as np
import h5py
import os
from PIL import Image
import glob
import re
import shutil
from bs4 import BeautifulSoup
import requests
from os.path import exists
import wget
from transformation import feed_data
import pandas as pd

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
    
    print("Finding seconds since midnight...")
    
    return 60 * 60 * float(time[:2]) + 60 * float(time[2:4]) + float(time[4:])


def shift_time(timestr, shift_min):
    """
    Purpose:
        - Shifts a string of time 'hhmmss' by some number of minutes
        - Can use fractional minutes (not intended for that) but should
        - Make a whole number of seconds, then write string to pass along
    """
    
    print("Accounting for time shift...")
    
    hr = float(timestr[:2])
    mn = float(timestr[2:4])
    sc = float(timestr[4:])
    
    t = 60 * 60 * hr + 60 * mn + sc
    t += 60 * shift_min
    
    hr = np.floor( t / (60 * 60) )
    mn = np.floor( (t - 60 * 60 * hr) / 60 )
    sc = np.floor( t - 60 * 60 * hr - 60 * mn )
    
    def num2str(num):
        """
        Purpose:
            - Converts type num to str
        """
        if num >= 10:
            strout = str(int(num))
        else:
            strout = '0' + str(int(num))
        
        return strout
    
    return num2str(hr) + num2str(mn) + num2str(sc)

 
def genlinks(date, starttime, endtime):
    """
    Purpose:
        - Given a date, start and end time, finds links to every DASC frame
    """
    
    print("Building link to archive...")
    
    startsecs = seconds_since_midnight(starttime) # seconds from midnight of start time
    endsecs = seconds_since_midnight(endtime) # seconds from midnight of end time
    hr = starttime[:2] # hour in string form
    year = date[:4] # year in string form

    # Construct base url
    url = 'http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/' + year + '/' + date + '/' + hr
    
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
        newlinks[i, 0] = seconds_since_midnight(time) # seconds past midnight
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
          <date>. An example input would be download_imagery('20230314','0730','0745')
    """
    
    print("Downloading imageery from archive...")
    
    # folder should already exist, but if not:
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Start hour and end hour are the same
    if starttime[:2] == endtime[:2]:
        links, fnames = genlinks(date, starttime, endtime)
    # We do two pulls
    else:
        # End of the first hour
        endtime0 = starttime[:2] + '5959'
        print(starttime)
        print(endtime0)

        links0, fnames0 = genlinks(date, starttime, endtime0)
        # Start of the second hour
        starttime1 = endtime[:2] + '0000'
        links1, fnames1 = genlinks(date, starttime1, endtime)
        
        print(starttime1)
        print(endtime)
        # Concatenate
        links = links0 + links1
        fnames = fnames0 + fnames1
    #try:
        #os.mkdir(date)
    #except:
        #pass
    
    for i in range(len(links)):
        #if exists(date + '/' + fnames[i]):
            #print('file exists')
            #continue
        #else:
            #wget.download(links[i], out=date)
        file_path = os.path.join(folder, fnames[i])
        if exists(file_path):
            continue
        else:
            wget.download(links[i], out=file_path)
        
            
    return date, starttime, endtime, folder


def sort_pngs(folder):
    """
    Moves PNG files into subfolders based on the wavelength in their filenames.
    """
    
    print("Sorting PNGs...")
    
    # Creating pattern to read PNG files and sort them
    os.chdir(folder) # go into folder if not already there
    files = os.listdir(folder) # pulls all files in directory (should just be PNGs for the hour)
    pattern = re.compile(r'_(\d{4})\.png$') # sort by wavelength
    
    for file in files:
        if file.endswith('.png'): # pull anything with PNG extension (should be everything in folder)
            match = pattern.search(file) # pull wavelength from filename
            if match:
                wavelength = match.group(1)
                subfolder = wavelength
                if not os.path.exists(subfolder): # create subfolder for all wavelengths
                    os.makedirs(subfolder)
                shutil.move(file, os.path.join(subfolder, file)) # move PNGs into respective wavelenght subfolder
    
    return folder


def png_2_h5(folder):
    """
    Purpose:
        - reads sorted PNG files
        - creates a new subfolder to store hdf5 files
        - converts each PNG file into an hdf5 file
        - stores converted files into new hdf5 folder
    """

    print("Converting PNGs to HDF5s...")
    
    lambdas = ['0630', '0558', '0428'] # red, green, blue wavelengths
    h5_dirs = [] # initalize empty directory

    # Go by each wavelength subfolder
    for lam in lambdas:
        lambda_folder = os.path.join(folder, lam)
        h5_folder = os.path.join(lambda_folder, 'h5_files')
        os.makedirs(h5_folder, exist_ok=True) # bypass if already exists, make if not
        h5_dirs.append(h5_folder)
        
        # Process PNG files
        for file in os.listdir(lambda_folder):
            if file.endswith('.png'): # pulls all files with PNG extension
                src_file = os.path.join(lambda_folder, file)
                h5_file = os.path.join(h5_folder, file.replace('.png', '.h5'))
                
                # Convert PNG to numpy array
                with Image.open(src_file) as img:
                    img_array = np.array(img)

                # Write numpy array to HDF5 file
                with h5py.File(h5_file, 'w') as h5:
                    h5.create_dataset('data', data=img_array)

    return folder


def group_frames(folder):
    """
    Purpose:
        - groups H5 files by threes, renames them sequentially, and saves them into respective wavelength folders.
    """
    
    print("Grouping frames for co-adding...")
    
    lambdas = ['0630/h5_files/', '0558/h5_files/', '0428/h5_files/']  # red, green, blue wavelengths, but with subfolder
    for lam in lambdas:
        folder_lam = os.path.join(folder, lam)
        
        # Check if directory exists (do not want to duplicate from prev function call)
        if not os.path.isdir(folder_lam):
            continue
        
        # List all h5 files in the directory and sort them by date/time based on filename
        h5_list = sorted(glob.glob(os.path.join(folder_lam, '*_*.h5')), key=lambda x: re.search(r'_(\d{8}_\d{6})', x).group(1) if re.search(r'_(\d{8}_\d{6})', x) else '')
    
        # Group into 3 h5s for 3 frames to later co-add
        grouped_h5 = [h5_list[i: i + 3] for i in range(0, len(h5_list), 3)]
    
        # Rename grouped h5 files (so regardless of wavelength, they conveniently share the same fn based on group number)
        for idx, files in enumerate(grouped_h5, start=1):
            if len(files) == 3:
                grouped = 'grouped_h5_files/'
                folder_grouped = os.path.join(folder_lam, grouped)
                os.makedirs(folder_grouped, exist_ok=True)
                output_h5 = os.path.join(folder_grouped, f"grouped_{idx}.h5")
            
                # Write the new grouped files
                with h5py.File(output_h5, 'w') as h5:
                    for i, file in enumerate(files):
                        with h5py.File(file, 'r') as h5src:
                            # Copy data from source files to new combined file
                            data = h5src['data'][:]  # shape of 512 x 512 
                            h5.create_dataset(f'frame_{i + 1}', data=data)
            else:
                continue
                
    return folder


def make_time_list(folder, output_txt):
    """
    Purpose:
        - creates a list of time ranges from the files in the specified folder and saves it to a text file
    """ 
    
    print("Making list of time ranges between image captures...")
    
    lambdas = ['0428', '0558', '0630']
    file_lists = []

    for lam in lambdas:
        folder_lam = os.path.join(folder, lam)
        file_list = sorted(glob.glob(os.path.join(folder_lam, '*_*.png')), key=lambda x: re.search(r'_(\d{8}_\d{6})', x).group(1))
        file_lists.append(file_list)

    grouped_files = []

    for i in range(0, len(file_lists[0]), 3):
        group = []
        for file_list in file_lists:
            group.extend(file_list[i:i + 3])
        grouped_files.append(group)
        
    time_ranges = []

    for idx, group in enumerate(grouped_files):
        if len(group) == 9: # 3 frames per wavelength
            times = []
            for file in group:
                match = re.search(r'_(\d{8}_\d{6})', file)
                if match:
                    time_str = match.group(1)[8:]
                    times.append(time_str)
            if times:
                min_time = min(times)
                max_time = max(times)
                range_str = f"{min_time}-{max_time}"
                time_ranges.append(range_str)
    
    # Write the time ranges to a text file
    with open(output_txt, 'w') as f:
        for time_range in time_ranges:
            f.write(f"{time_range}\n")

    return time_ranges


def make_time_spreadsheet(output_txt, base_outdir):
    """
    Purpose:
        - creates a spreadsheet that details:
            - when first image from file group was taken
            - when last image from file group was taken
            - total time from first frame to last in co-adding
    """
    
    # Read txt file
    with open(output_txt, 'r') as file:
        lines = file.readlines()
        
    # Initialize to store
    starts = []
    ends = []
    diffs = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        start, end = line.split('-')
        start = start.lstrip('_')
        end = end.lstrip('_')
        
        # Convert from text to integer
        start_int = int(start)
        end_int = int(end)
        
        # Find time of co-added groups (time cadence between frames)
        diff = end_int - start_int
        
        # Append into columns
        starts.append(start_int)
        ends.append(end_int)
        diffs.append(diff)
        
        # Make into pds dataframe
        df = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Difference': diffs})
        
        output_csv_fn = 'time_ranges.csv'
        output_csv = os.path.join(base_outdir, output_csv_fn)
        
        # Save as csv
        df.to_csv(output_csv, index=False)
    
    return df


def file_data(date, starttime, endtime, maglatsite, folder, output_txt, base_outdir):
    """
    Purpose:
        - runs all filing processes from functions above
    """ 
    
    print("Running all data wrangling processes...")
    
    # Main function calls
    download_imagery(date, starttime, endtime, folder)
    folder = sort_pngs(folder)
    folder = png_2_h5(folder)
    folder = group_frames(folder)
    make_time_list(folder, output_txt)
    df = make_time_spreadsheet(output_txt, base_outdir)

    lambdas = ['0428/', '0558/', '0630/']
    grouped_files = []

    for lam in lambdas:
        lam_folder = os.path.join(folder, lam, 'h5_files')
        h5_files = glob.glob(os.path.join(lam_folder, '*_*.h5'))
        
        # Sort h5s by timestamp in filenames
        def get_timestamp(file):
            match = re.search(r'_(\d{8}_\d{6})', file)
            return match.group(1) if match else '' # just an empty string if nothing there
        
        h5_list = sorted(h5_files, key=get_timestamp)
        
        # Group into sets of 3
        grouped_h5 = [h5_list[i: i + 3] for i in range(0, len(h5_list), 3)]
        
        for files in grouped_h5:
            if len(files) > 0:
                times = []
                for file in files:
                    match = re.search(r'_(\d{8}_\d{6})', file)
                    if match:
                        time_str = match.group(1)[8:]
                        times.append(time_str)

                if times:
                    min_time = min(times)
                    max_time = max(times)
                    range_str = f"{min_time}-{max_time}"
                    grouped_files.append(os.path.join(lam_folder, f"{range_str}.h5"))

    return date, starttime, endtime, maglatsite, folder, base_outdir, lambdas


def get_grouped_files(folder, lambdas):
    """
    Purpose:
        - pulls together all of the grouped files
    """ 
    
    grouped_files = {lam: sorted(glob.glob(os.path.join(folder, lam, 'h5_files', 'grouped_h5_files', '*.h5')), key=lambda x: int(re.search(r'_(\d+)', os.path.basename(x)).group(1))) for lam in lambdas}
    
    return grouped_files


def process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas):
    """
    Purpose:
        - pulls info from get_grouped_files and feeds all of the processed h5s into feed_data function
        - this is what allows for the process to be time varying!!
    """ 
    grouped_files = get_grouped_files(folder, lambdas)
    
    # Find the maximum number of groups across all wavelengths
    max_groups = max(len(files) for files in grouped_files.values())
    
    # Iterate over each group index - time varying!!
    for idx in range(max_groups):
        foi_files = {}
        
        for lam in lambdas:
            if idx < len(grouped_files[lam]):
                foi = grouped_files[lam][idx]
                foi_files[lam] = os.path.join(lam, 'h5_files', 'grouped_h5_files', os.path.basename(foi))
            else:
                foi_files[lam] = None  # handle cases where there may be less files for some wavelengths
                
        print("FOI Files:", foi_files)

        # Ensure all necessary files collected before calling feed_data
        if all(foi is not None for foi in foi_files.values()):
            group_number = f"grouped_{idx + 1}"
            group_outdir = os.path.join(base_outdir, group_number)
            os.makedirs(group_outdir, exist_ok=True)
            
            feed_data(date, maglatsite, folder,
                      foi_files['0428/'],
                      foi_files['0558/'],
                      foi_files['0630/'],
                      group_outdir, 
                      group_number)
        else:
            continue
