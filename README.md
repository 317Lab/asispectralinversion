# asispectralinversion
Inverting precipitation spectra (Q and E0) from RGB all-sky imagery

## Installation

1. Clone this repo
```
git clone https://github.com/317Lab/asispectralinversion.git
```

2. Enter the repo root
```
cd amisrspectralinversion
```

3. Install with pip
```
pip install .
```

### Dependencies

The following dependiences should be installed automatically.  If errors occur, follow the pacakge-specific installation instructions.
- [numpy](https://numpy.org)
- [scipy](https://scipy.org)
- [h5py](https://docs.h5py.org/en/latest/index.html)
- [matplotlib](https://matplotlib.org)
- [pandas](https://pandas.pydata.org)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [scikit-image](https://scikit-image.org)
- [skyfield](https://rhodesmill.org/skyfield/)
- [apexpy](https://apexpy.readthedocs.io/en/latest/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/index.html)
- [importlib_resources](https://importlib-resources.readthedocs.io/en/latest/index.html)

## Usage

The script `example_runscript.py` shows a simple example of how to call this package from a python script.  This script demonstrates two use cases - running a single inversion and generating a series of inversions over the length of an event.  In both cases, you must [generate GLOW lookup tables](#obtaining-glow-lookup-tables) BEFORE running this code.

### Single Inversion
A single inversion can be performed by calling the `feed_data` function. Input for this function includes:

- date, as a `datetime.date` object
- list of specific red input png files
- list of specific green input png files
- list of specific blue input png files
- path to directory holding the GLOW lookup tables
- output filename

Because input files are passed into this function explicity, it does not matter where they are stored or what they are named so long as the full file paths are provided.  Additional optional parameters can be specified to customize how the inversion is performed and whether or not plots are generated.

### Multiple Inversions
A series of inversions for all images over the course of an evant can be performed with the `file_data` and `process_grouped_files` functions.  The `file_data` function downloads and organizes input image files.  Input for this function incudes:

- date, as a `datetime.date` object
- start time as a `datetime.time` object
- end time as a `datetime.time` object
- path to directory to save downloaded image files


The `process_grouped_files` function runs through automatically calling `feed_data` to process groups of files in multiple inversions.  Inputs include:

- time stamps of each file group as `datetime.datetime` objects
- list of red input png file groups
- list of green input png file groups
- list of blue input png file groups
- path to directory holding the GLOW lookup tables
- path to directory to save output files

The first four inputs can taken directly from the output of `file_data`.

### Inversions at Other Sites
By default, this code has been designed to work on the Poker Flat Digital All-Sky Camera (PKR DASC).  The inversion method itself is general, but the built in utilities to download and organize images are specific to this site.  To use this package with a different site, you can only use the "Single Inversion" option discussed above.  Furthermore, you will have to specify a site-specific skymap file that maps image pixels to geographic locations using the optional `skymap_file` parameter in `feed_data`.


## Data products obtained using the asispectralinversion library

1. Imagery from ASI (if automatically downloaded)
   
   ![red_imagery](https://github.com/user-attachments/assets/854905a8-28ba-4c9f-aa56-1618f97833d8) ![green_imagery](https://github.com/user-attachments/assets/667b9b68-1fb1-48fd-afdc-31482b0d84b1) ![blue_imagery](https://github.com/user-attachments/assets/b6bff157-d891-4bda-a6fb-d2f8da3a0cb4)

2. 2D Maps of Characteristic Energy
   
   ![E0_geomag](https://github.com/user-attachments/assets/d39c8273-d570-4cc3-9d4d-d5b42c5791cb) ![E0_geod_reg_{time_str}](https://github.com/user-attachments/assets/7e365d08-9e2b-4df2-acde-eca053b8024d)

3. 2D Maps of Energy Flux
   
   ![Q_geomag](https://github.com/user-attachments/assets/6c451c56-4193-4592-a13e-08839d972e3a) ![Q_geod_reg_{time_str}](https://github.com/user-attachments/assets/1cf513ec-6a70-4359-b611-85126ae9b03e)

4. 2D Maps of Pedersen Conductance
   
   ![SigP_geomag](https://github.com/user-attachments/assets/5a0631eb-946b-44bf-b80a-aab9807d1c2a)

5. 2D Maps of Hall Conductance
    
   ![SigH_geomag](https://github.com/user-attachments/assets/a4cbc41c-2361-4ac2-acec-b58d2654ab7f)

6. Output files with mapped information to plug into external libraries (Lompe, GEMINI) in .h5 format


All maps are given in both geodetic and geomagnetic coordinates.


## Using this library to format maps to plug into GEMINI

The gemini_mapping.py script located in asispectralinversion/src/ takes the 2D maps achieved by running asispectralinversion_runscript.py and formats them in a way that the GEMINI model can use to generate model precipitation inputs.


## Obtaining GLOW Lookup Tables

GLOW lookup tables are required to perform the inversion.  These are generated by Fortran routines that are included in this repo.  The [NCAR GLOW](https://github.com/NCAR/GLOW) model is independently maintained, but has been added as a linked submodule so should be automatically included when cloning this repo.

You will need a [fortran compiler](https://fortran-lang.org/compilers/) to build this code.  If you do not already have a fotran compiler on your system, we recommend installing [gfortran](https://fortran-lang.org/learn/os_setup/install_gfortran/).

### Building Executable
Note: Presuming no changes in the look-up table source code, you should only have to do this part ONCE.  After the executables are build and available on your system, you can use all to generate all look-up tables regardless of time and locaiton.

1. Confirm that you have a [fortran compiler](https://fortran-lang.org/compilers/) installed on your system.

2. Enter the `glow_invert` source code directory.  From the root of this repo, run the following.
```
cd src/glow_invert
```

3. Run the `makefile` to generate the executables.  This can be done with the `make` command.
```
make tables airglow
```
This should generate the executables `glow_invert_tables.exe` and `glow_invert_airglow.exe`.

4. Make sure the shell script `generate_tables.sh` has execution permissions.  This can ususally be done by running
```
chmod +x generate_tables.sh
```

#### Troubleshooting
You may have to edit the makefile for it to work on your particular system.  For instance, by default the makefile specifies gfortran as the compiler with the line:
```
FC = gfortran
```
If you are using a different compiler, it will have to be specified here.  Some of the other flags and options listed in the makefile may also need to be customized for different systems.

If you are running the `make` command multiple times, it may be helpful to run `make clean` in between each build attempt to avoid attempting to build with mismatched compile options.

### Generate Lookup Tables
A new lookup table will need to be generated anytime the input paramters change.

1. Make sure you are in the `glow_invert` source code directory.
```
cd path/to/asispectralinversion/src/glow_invert
```
2. Copy `in.invert`.  This file can be stored anywhere on your system, but keep track of the full path.  It is often helpful to rename it with the event date as `in.invert.YYMMDD`

3. Modify your copy of `in.invert` to specify the appropriate parameters.  Parameters should be listed in a single row with white space seperating each.

   The following example is for a file for March 19, 2023 (assuming you name the file in.invert.<YYDOY>, in.invert.23078)

   - YYDOY: 23078
   - ut time (seconds): 29760
   - glat (degrees): 65.12
   - glon (degrees): 212.8
   - f10.7a (average solar flux): 159.9
   - f10.7 (solar flux for this particular day): 143.0
   - f10.7p (solar flux for the previous day): 140.0
   - Ap: 10.0
   - output directory: path/to/output/directory/for/lookup/tables

4. Run the shell script `generate_tables.sh` with your customized `in.invert` file as a command line argument to generate all lookup tables required for the ASI inversion code

```./generate_tables.sh path/to/in.invert```

   Depending on your machine, this may take a while. It takes my machine about 20 minutes to generate one set oftables.

5. Check your output directory (specified in in.invert)

   There should be 11 files that are generated and stored in this folder, pluss the subfolder `airglow` with 11 more files contained within:

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
   - airglow/edens3d_23078_29760.bin
   - airglow/eta4278_23078_29760.bin
   - airglow/eta5577_23078_29760.bin
   - airglow/eta6300_23078_29760.bin
   - airglow/eta8446_23078_29760.bin
   - airglow/hall3d_23078_29760.bin
   - airglow/I4278_23078_29760.bin
   - airglow/I5577_23078_29760.bin
   - airglow/I6300_23078_29760.bin
   - airglow/I8446_23078_29760.bin
   - airglow/ped3d_23078_29760.bin


