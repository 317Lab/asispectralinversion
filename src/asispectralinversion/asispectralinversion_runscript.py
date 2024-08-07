from filing import file_data
from filing import process_grouped_files

"""
Purpose of this script:
    - This is the top-level run script to go through the entire process of:
        - sorting ASI data inputs
        - preprocessing images to denoise them
        - invert imagery + GLOW lookup table to produce maps of Q, E0, SigmaP, and SigmaH
        - perform a series of interpolations, smoothing, and transformations on these maps
    - Inputs:
        - date
        - latitude of imager site
        - wavelengths of imager filters
        - file path to folder containing ASI PNGs and GLOW lookup tables
        - file path to folder for storing outputs
        - file path + name for txt file containing information about framerate/time between processed images
    - Outputs:
        - 2D maps (lat, lon) of Q, E0, SigmaP, and SigmaH for all times
        - h5 files containing information about Q, E0, SigmaP, and Sigma H for all times in geodetic and geomagnetic coords
        - txt file containing information about framerate/time between processed images
"""

# Tweakable inputs
date = '20230319' # date in the form of YYYYMMDD
maglatsite = 65.8 # site of camera in magnetic latitude
lambdas = ['0428', '0558', '0630'] # wavelengths (nm) of imager filters
folder = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_inputs/asi/images/6ut/' # folder that holds all image files and GLOW outputs for an hour's worth of an event
base_outdir = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/6UT/' # output directory to store all output figures and h5s
output_file = '/Users/clevenger/Projects/data_assimilation2/test_dates/02162023/data_product_outputs/asi_spectral_inversion/6UT/time_ranges.txt' # output for txt file that shows time cadence

# Main function calls to run through entire process
folder, base_outdir, grouped_files = file_data(folder, output_file, base_outdir)
process_grouped_files(date, maglatsite, folder, base_outdir, lambdas)