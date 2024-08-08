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
   
   ![red_imagery](https://github.com/user-attachments/assets/854905a8-28ba-4c9f-aa56-1618f97833d8) ![green_imagery](https://github.com/user-attachments/assets/667b9b68-1fb1-48fd-afdc-31482b0d84b1) ![blue_imagery](https://github.com/user-attachments/assets/b6bff157-d891-4bda-a6fb-d2f8da3a0cb4)

2. 2D Maps of Characteristic Energy
   
   ![E0_geomag](https://github.com/user-attachments/assets/d39c8273-d570-4cc3-9d4d-d5b42c5791cb) ![E0_geod_reg_{time_str}](https://github.com/user-attachments/assets/7e365d08-9e2b-4df2-acde-eca053b8024d)

3. 2D Maps of Energy Flux
   
   ![Q_geomag](https://github.com/user-attachments/assets/6c451c56-4193-4592-a13e-08839d972e3a) ![Q_geod_reg_{time_str}](https://github.com/user-attachments/assets/1cf513ec-6a70-4359-b611-85126ae9b03e)

4. 2D Maps of Pedersen Conductance
   
   ![SigP_geomag](https://github.com/user-attachments/assets/5a0631eb-946b-44bf-b80a-aab9807d1c2a)

5. 2D Maps of Hall Conductance
    
   ![SigH_geomag](https://github.com/user-attachments/assets/a4cbc41c-2361-4ac2-acec-b58d2654ab7f)

6. An output file that shows the time between processed 2D maps in .csv format
   
7. Output files with mapped information to plug into external libraries (Lompe, GEMINI) in .h5 format


All maps are given in both geodetic and geomagnetic coordinates.
