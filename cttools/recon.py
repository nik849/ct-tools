import numpy as np
import glob
import os
from scipy.ndimage import map_coordinates
from .utilities import rotate_coordinates, read_image
from .filtering import ramp_filter_and_weight
from .config import Config
from scipy.interpolate import griddata, interp2d, interp1d, RegularGridInterpolator
from skimage.io import *
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from sys import getsizeof
from scipy.fftpack import fft, ifft, fftshift
import psutil
import ray


def panel_coords(x, y, z, theta, config):

    print('Calculating det_a ...')
    det_a = config.source_to_detector_dist * ((-x * np.sin(theta)) + (y * np.cos(theta))) / (config.source_to_detector_dist + (x * np.cos(theta)) + (y * np.sin(theta)))
    print('Calculating det_b ...')
    det_b = z * (config.source_to_detector_dist * (config.source_to_detector_dist + (x * np.cos(theta)) + (y * np.sin(theta))))

    return det_a, det_b


@ray.remote
def fdk_slice(projections, config, slice):

    proj_width = projections[0][0].shape[0]
    proj_height = projections[0][0].shape[1]
    recon = np.zeros((proj_width, proj_height), dtype=np.float32)
    angles = np.linspace(0, (2 * np.pi), len(projections))

    print(f'Angles: {angles}')
    for projection, angle in tqdm(zip(projections, angles), total=len(projections)):
        radius = proj_width / 2.
        x = np.arange(proj_width) - radius
        x_r, y_r = np.mgrid[:config.n_voxels_x, :config.n_voxels_y] - radius
        x_r = x_r + config.center_of_rot_y
        t = -x_r * np.cos(angle) + y_r * np.sin(angle)
        if slice is not None:
            interpolant = partial(np.interp, xp=x, fp=projection[0][:, int(slice)], left=0, right=0)
        else:
            interpolant = partial(np.interp, xp=x, fp=projection[0][:, int(proj_height / 2)], left=0, right=0)
        recon = recon + interpolant(t)
    return recon
    #out = recon / np.float(len(projections))
    #out.tofile('output.raw')
    #return out


def fdk_vol(projections, config, **kwargs):
    output_file = 'output.raw'
    with open(output_file, 'wb') as f:
        pool = Pool()
        #ray.init()
        temp = []
        func = partial(fdk_slice, projections, config)
        num_imgs = list(range(int(config.n_voxels_z)))
        print(f'Processing {config.n_voxels_z} slices ...')
        for slice in pool.imap(func, slice=slice)
            temp.append(slice)
        print(f'Writing out ...')
        for slice in tqdm(sorted(temp), total=len(temp)):
            f.write(slice)


def read_projections(path, into_ram=True, flat_corrected=True, n_proj=3142):
    proj_stack = []
    if into_ram is True:
        print(f'Reading {n_proj} Projections into RAM...')
        for f in tqdm(sorted(glob.glob('*.tif'))[:n_proj], total=len(glob.glob('*.tif')[:n_proj])):
            proj_stack.append(np.array(read_image(f, flat_corrected=True)))
    else:
        pass
    return proj_stack


def filter_projections(param, projections):

    filtered_stack = []

    for projection in projections, :
        projection = -np.log(projection)
        projection_filtered = np.squeeze(ramp_filter_and_weight(projection[:, :, np.newaxis], param), 2)
        filtered_stack.append(projection_filtered)

    return filtered_stack


def recon(projections, param, single_slice=True, slice=None):
    filtered_stack = []
    pool = Pool()
    print('Filtering Projections...')
    func = partial(filter_projections, param)
    for proj in pool.imap(func, projections):
        filtered_stack.append(proj)#list(tqdm(pool.apply_async(filter_projections(projections, param), projections), total=len(projections)))
    pool.close()
    #for proj in projections:
    #    filtered_stack.append(filter_projections(proj))
    print(f'Filtered {len(filtered_stack)} Projections.')
    if single_slice:
        return fdk_slice(filtered_stack, param, slice)
    else:
        return fdk_vol(filtered_stack, param)
    #return filtered_stack
