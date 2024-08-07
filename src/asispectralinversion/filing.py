import numpy as np
import h5py
import os
from PIL import Image
import glob
import re
import shutil
from transformation import feed_data

"""
Purpose of this script:
    - handle input data
        - GLOW lookup table for a specific hour of an event
        - ASI PNGs from Don's archive (http://optics.gi.alaska.edu/amisr_archive/PKR/DASC/PNG/)
    - gets everything into a nice h5 format before preprocessing and inversion
    - contains main function call that feeds processed h5 files into inversion pipeline
"""

def sort_pngs(folder):
    """
    Moves PNG files into subfolders based on the wavelength in their filenames.
    """
    
    print("Sorting PNGs...")
    
    # Creating pattern to read PNG files and sort them
    os.chdir(folder) # go into folder
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


def make_time_list(folder, output_file):
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
    with open(output_file, 'w') as f:
        for time_range in time_ranges:
            f.write(f"{time_range}\n")

    return time_ranges


def file_data(folder, output_file, base_outdir):
    """
    Purpose:
        - runs all filing processes from functions above
    """ 
    
    print("Running all data wrangling processes...")
    
    # Main function calls
    folder = sort_pngs(folder)
    folder = png_2_h5(folder)
    folder = group_frames(folder)
    make_time_list(folder, output_file)

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

    return folder, base_outdir, grouped_files


def get_grouped_files(folder, lambdas):
    """
    Purpose:
        - pulls together all of the grouped files
    """ 
    
    grouped_files = {lam: sorted(glob.glob(os.path.join(folder, lam, 'h5_files', 'grouped_h5_files', '*.h5')), key=lambda x: int(re.search(r'_(\d+)', os.path.basename(x)).group(1))) for lam in lambdas}
    
    return grouped_files


def process_grouped_files(date, maglatsite, folder, base_outdir, lambdas):
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

        # Ensure all necessary files collected before calling feed_data
        if all(foi is not None for foi in foi_files.values()):
            group_number = f"grouped_{idx + 1}_"
            group_outdir = os.path.join(base_outdir, group_number)
            os.makedirs(group_outdir, exist_ok=True)
            
            feed_data(date, maglatsite, folder,
                      foi_files['0428'],
                      foi_files['0558'],
                      foi_files['0630'],
                      group_outdir, 
                      group_number)
        else:
            continue


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
