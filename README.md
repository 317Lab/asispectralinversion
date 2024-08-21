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

Note that the required inputs are a date to pull PNG files for, a GLOW lookup table for the corresponding time(s), and a skymap.mat file. See instructions below for generating GLOW lookup tables and downloading the necessary skymap.mat file to go with each run.

## Data products obtained using the asispectralinversion library

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


## Using this library to format maps to plug into GEMINI

The gemini_mapping.py script located in asispectralinversion/src/ takes the 2D maps achieved by running asispectralinversion_runscript.py and formats them in a way that the GEMINI model can use to generate model precipitation inputs.


## Obtaining GLOW Lookup Tables

You can either obtain the base code for generating these lookup tables from https://github.com/NCAR/GLOW and add a few files to the directory where you have this saved, or you can pull all of the files located [here](https://www.dropbox.com/home/Hayley%20Clevenger/GLOW/invert_GLOW) and put them in one location. If you grab everything from NCAR/GLOW, you will need to include the files/folders from the shared location entitiled "output," "glow_invert_tables_v3.exe," "glow_invert_tables_v3.f90," "glow_invert_tables_v3.o," and any of the files starting with "in.invert"

You will want to download a fortran compiler -- I personally use gfortran on MacOS M1, though other options are fine depending on the combination of compilers you have for your specific machine.

Once all of these are downloaded, you will want to navigate to where you have all of this stored on your machine and:

1. Create the makefile/compile:

   ```make -f make_invert_tables.v3```

3. Edit the makefile to match your compiler:

   For instance, when I compile, my makefile has the line: "FC = gfortran" (so if you are not using gfortran, fill that in with your compiler)

5. Duplicate one of the in.invert files, rename it for the date you are running, and fill in the appropriate parameters:

   The following example is for a file for March 19, 2023 (assuming you name the file in.invert.<YYDOY>, in.invert.23078)

   - YYDOY: 23078
   - ut time (seconds): 29760
   - glat (degrees): 65.12
   - glon (degrees): 212.8
   - f10.7a (average solar flux): 159.9
   - f10.7 (solar flux for this particular day): 143.0
   - f10.7p (solar flux for the previous day): 140.0
   - Ap: 10.0
   - Ec (dummy): 1.0
   - Qc (dummy): 1000.0

7. Run the executable program "glow_invert_tables_v3.exe"

   ```./glow_invert_tables_v3.exe < in.invert.23078```

   Depending on your machine, this may take a while. It takes my machine about 20 minutes to generate one table.

9. Check outut/v3/ folder

   There should be 11 files that are generated and stored in this folder:

   - edens3d_23078_29760.bin
   - eta4278_23078_29760.bin
   - eta5577_23078_29760.bin
   - eta6300_23078_29760.bin
   - eta8446_23078_29760.bin
   - hall3d_23078_29760.bin
   - I4278_23078_29760.bin
   - I5577_23078_29760.bin
   - I6300_23078_29760.bin
   - I8446_23078_29760.bin
   - ped3d_23078_29760.bin
  
11. Copy/move GLOW output files

    If you are keeping the same format for the runscript as above, you will want to store these files in the same spot you plan to store the PNG files, with the variable "folder."


## Obtaining the skymap.mat file

For all runs specifically with the Poker DASC (which this repo is supporting), the same skymap.mat file will be used for every single run. It can be found [here](https://www.dropbox.com/home/Hayley%20Clevenger/GLOW). If the DASC gets moved at all, the skymap.mat file contents will change and the new version will be needed for the relevant dates.   
