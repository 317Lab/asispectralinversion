from transformation import feed_data

"""
Purpose of this script:
    - top-level/run script for the entire process that takes you from GLOW lookup tables to Lompe and/or GEMINI conductivity inputs
"""

# Inputs tweakable to user
date = '20230314' # date in the form of YYYYMMDD
maglatsite = 65.8 # site of camera in magnetic latitude
folder = '/Users/clevenger/Projects/data_assimilation/march_14/input_data/ASI/' # folder where GLOW inverted data is stored
outdir = '/Users/clevenger/Projects/asi_inversion/hc_fork/asi_inversion_outputs/single_frame/mar_14/' # output directory for plots and h5 files

# Main function call
feed_data(date, maglatsite, folder, outdir)

# Adding in optional stuff to create output files that can be used specicially for Lompe and GEMINI
#gemini = True
#lompe = True
