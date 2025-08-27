from asispectralinversion.filing import file_data, process_grouped_files
from asispectralinversion.transformation import feed_data
import datetime as dt


# Set date and time range
dtdate = dt.date(2023,3,14)
starttime = dt.time(6,48)
endtime = dt.time(6,52)


# Set directory paths
data_dir = 'test_data_20230314/' # folder that holds image files
lookup_dir = 'GLOW_lookup_data/' # folder that holds GLOW lookup tables
out_dir = 'test_out_20230314/' # output directory to store all output figures and h5s


# 1. Process a single set of files
foi_0630 = [data_dir+'PKR_20230314_064904_0630.png',
            data_dir+'PKR_20230314_064912_0630.png',
            data_dir+'PKR_20230314_064920_0630.png']

foi_0558 = [data_dir+'PKR_20230314_064902_0558.png',
            data_dir+'PKR_20230314_064910_0558.png',
            data_dir+'PKR_20230314_064918_0558.png']

foi_0428 = [data_dir+'PKR_20230314_064859_0428.png',
            data_dir+'PKR_20230314_064907_0428.png',
            data_dir+'PKR_20230314_064915_0428.png']

output_file = out_dir+'test_out.hdf5'

feed_data(dtdate, foi_0428, foi_0558, foi_0630, lookup_dir, output_file, plot=True)

# Customize options for the preparation step
#   This particular example heavily decimates the image so it should run much more quickly
#feed_data(dtdate, foi_0428, foi_0558, foi_0630, lookup_dir, output_file, prep_kwarg={'dec':32}, plot=True)



# 2. Process a range of times automatically
tstmps, files0428, files0558, files0630 = file_data(dtdate, starttime, endtime, data_dir)
process_grouped_files(tstmps, files0428, files0558, files0630, lookup_dir, out_dir)



