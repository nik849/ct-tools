import axitom
import numpy as np
import glob
from os.path import *
import matplotlib.pyplot as plt


class data(object):


    def __init__(self, dataset=None, xtekct=None, debug=None, **kwargs):
        self.debug = debug
        if dataset is not None:
            self.dataset = sorted([os.path.join(data, f) for f in os.listdir(data) if f.endswith('.tif')])
        else:
            self.dataset = sorted(glob.glob(".tif"))
        if xtekct is not None:
            self.config = axitom.config_from_xtekct(xtekct)
        else:
            self.config = axitom.config_from_xtekct(glob.glob("*.xtekct")[0])
        if self.debug:
            print(self.dataset)
            print(self.config)


    def load_dataset(self, corrected=True):
        self.stack = [np.array(axitom.read_image(image, flat_corrected=corrected)) for image in self.dataset]
        if self.debug:
            print(self.stack)
        return self.stack


    def COR(self, stack, slice_number=None, background=0.9):
        if slice_number is not None:
            self.slice = slice_number
        else:
            if len(self.dataset) >= 2:
                self.slice = int(len(self.dataset)/2)
        _, center_offset = axitom.object_center_of_rotation(stack[self.slice], self.config, background_internsity=background)
        self.config.center_of_rot_y = center_offset
        self.config.update()
        if self.debug:
            print(center_offset)


    def recon(self, stack):
        tomo = [axitom.fdk(slice, self.config) for slice in self.stack]
        plt.title("1st Reconstructed Radial slice")
        plt.imshow(tomo[0].transpose(), cmap=plt.cm.cividis)
