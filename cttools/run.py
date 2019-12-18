import cttools as tom
import os, glob
import numpy as np
from skimage.io import *
from sys import argv


p = []

os.chdir('/mnt/heap/Boeing/2318/0_cycles/01_HUTCH_scans/01_raw_data/3D_printed_support/20191031_HUTCH_2318_CH_IN23_4-5-24_2-3_55um [2019-10-31 15.18.45]/')

for f in sorted(glob.glob("*.tif"))[::int(argv[1])]:
    p.append(tom.read_image(f, flat_corrected=True))

config = tom.config_from_xtekct('20191031_HUTCH_2318_CH_IN23_4-5-24_2-3_55um_01.xtekct')

out = tom.recon(p, config);imshow(out, cmap='Greys');show()
