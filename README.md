# asispectralinversion
Inverting precipitation spectra (Q and E0) from RGB all-sky imagery

To install, download this repo, cd to src, and run "pip install ." to install the asispectralinversion library. 

Runscript using example data can be found in src/asispectralinversion/asi_inversion_runscript.py -- tweak inputs like your output directory and where you have ASI data/GLOW lookup tables saved:

```
from transformation import feed_data

date = '20230314' # date in the form of YYYYMMDD
maglatsite = 65.8 # site of camera in magnetic latitude
folder = '/Users/clevenger/Projects/data_assimilation/march_14/input_data/ASI/' # folder where GLOW inverted data is stored
outdir = '/Users/clevenger/Projects/asi_inversion/hc_fork/asi_inversion_outputs/single_frame/mar_14/' # output directory for plots and h5 files

feed_data(date, maglatsite, folder, outdir)

```
