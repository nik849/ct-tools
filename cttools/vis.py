import glob, os
import time
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import exposure


def load_img_seq(data=None):
    if data is not None:
        imfiles = sorted([os.path.join(data, f) for f in os.listdir(data) if f.endswith('.tif')])
    else:
        imfiles = sorted(glob.glob("*.tif"))

    im = Image.open(imfiles[0])
    imgs = np.zeros((len(imfiles),im.size[1],im.size[0]))
    i = 0
    for imfile in imfiles:
        im = Image.open(imfile)
        imgs[i,:,:] = np.array(im)
        i+=1
    return imgs


def histogram(image):
    histo = exposure.histogram(image)
    plt.plot(histo[1], histo[0], label='data')
    plt.title('Stacked Histogram of voxel values')


def load_raw_vol(vol):
    pass
