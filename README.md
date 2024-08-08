# asispectralinversion
Inverting precipitation spectra (Q and E0) from RGB all-sky imagery

To install, download this repo, cd to src, and run "pip install ." to install the asispectralinversion library. 

## Dependencies

To download dependencies separately, use installer of choice to install:
- pandas
- numpy
- scipy
- h5py
- os
- PIL
- glob
- re
- shutil
- bs4
- requests
- wget
- matplotlib
- datetime
- apexpy
- scipy
- skimage
- gemini3d

## Running the asispectralinversion library

Runscript using example data can be found in src/asispectralinversion/asispectralinversion_runscript.py -- tweak inputs like your output directory and where you have ASI data/GLOW lookup tables saved, as in the following template:

```
# Tweakable inputs
date = 'YYMMDD' # date in the form of YYYYMMDD
starttime = 'HHMMSS' # start time in the format of HHMMSS
endtime = 'HHMMSS' # end time in the format of HHMMSS

maglatsite = 65.8 # site of camera in magnetic latitude

lambdas = ['0428', '0558', '0630'] # wavelengths (nm) of imager filters

folder = '/path_to_GLOW_lookup_tables_for_hour/' # folder that holds all image files and GLOW outputs for an hour's worth of an event
base_outdir = '/path_to_output_directory/' # output directory to store all output figures and h5s
output_txt = '/path_to_output_directory/time_ranges.txt' # output for txt file that shows time cadence

# Main function calls to run through entire process
date, starttime, endtime, maglatsite, folder, base_outdir, lambdas = file_data(date, starttime, endtime, maglatsite, folder, output_txt, base_outdir)
process_grouped_files(date, starttime, endtime, maglatsite, folder, base_outdir, lambdas)

```

## Data products output using the asispectralinversion library

Running this process gives a user:
1. Imagery from ASI
   
2. 2D Maps of Characteristic Energy
3. 2D Maps of Energy Flux
4. 2D Maps of Pedersen Conductance
5. 2D Maps of Hall Conductance
6. An output file that shows the time between processed 2D maps
7. Output files with mapped information to plug into external libraries (Lompe, GEMINI)

All maps are given in both geodetic and geomagnetic coordinates.
