import cttools as tom
import os, glob
import numpy as np
from skimage.io import *
from sys import argv


p = []

#os.chdir('/mnt/heap/Boeing/2318/0_cycles/01_HUTCH_scans/01_raw_data/3D_printed_support/20191031_HUTCH_2318_CH_IN23_4-5-24_2-3_55um [2019-10-31 15.18.45]/')
os.chdir('/mnt/data4/nwh1v18/2238/01_raw_data/20190403_HMX_2238_MF_thick_fast_mo [2019-04-03 15.45.58]/')
#os.chdir('/mnt/data4/nwh1v18/20190212_HMX_2178_ES/01_raw_data/20190212_HMX_2178_ES_overview [2019-02-12 09.47.31]/')
#os.chdir('/mnt/heap/Boeing/2318/800_cycles/20191114_HUTCH_2318_CH_02_IN23_2-4-24_06um [2019-11-14 22.06.45]')
for f in sorted(glob.glob("*.tif"))[::int(argv[1])]:
    p.append(tom.read_image(f, flat_corrected=True))

#config = tom.config_from_xtekct('20191031_HUTCH_2318_CH_IN23_4-5-24_2-3_55um_01.xtekct')
config = tom.config_from_xtekct('20190403_HMX_2238_MF_thick_fast_mo_01.xtekct')
#config = tom.config_from_xtekct('20190212_HMX_2178_ES_overview.xtekct')
#config = tom.config_from_xtekct('20191114_HUTCH_2318_CH_02_IN23_2-4-24_06um.xtekct')

out = tom.recon(p, config, single_threaded=True, slice=1250);imshow(out, cmap='Greys');show()
