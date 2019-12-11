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
#import psutil
import ray


def panel_coords(x, y, z, theta, config):

    print('Calculating det_a ...')
    det_a = config.source_to_detector_dist * ((-x * np.sin(theta)) + (y * np.cos(theta))) / (config.source_to_detector_dist + (x * np.cos(theta)) + (y * np.sin(theta)))
    print('Calculating det_b ...')
    det_b = z * (config.source_to_detector_dist * (config.source_to_detector_dist + (x * np.cos(theta)) + (y * np.sin(theta))))

    return det_a, det_b


def fdk_slice(projections, config, **kwargs):

    proj_width = projections[0][0].shape[0]
    proj_height = projections[0][0].shape[1]
    recon = np.zeros((proj_width, proj_height), dtype=np.float32)
    angles = np.linspace(0, (2 * np.pi), len(projections))

    print(f'Angles: {angles}')
    for projection, angle in tqdm(zip(projections, angles), total=len(projections)):
        x_proj = projection[0].T[:, 0]
        y_proj = projection[0].T[0, :]
        z = 0

        x_proj += config.center_of_rot_y
        U = (config.source_to_detector_dist + (x_proj * np.cos(angle)) + (y_proj * np.sin(angle)))
        ratio = (config.source_to_detector_dist ** 2) // (U ** 2)
        radius = proj_width / 2.
        x = np.arange(proj_width) - radius
        x_r, y_r = np.mgrid[:config.n_voxels_x, :config.n_voxels_y] - radius
        det_a = config.source_to_detector_dist * ((-x_r * np.sin(angle)) + (y_r * np.cos(angle))) // (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle)))
        #det_b = z * (config.source_to_detector_dist * (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle))))
        for col in projection[0].T:
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
            recon = recon + interpolant(det_a)# * ratio
            #recon_slice = recon_slice + ratio * interpolant(det_a)

    out = recon // np.float(len(projections))
    out.tofile('output.raw')
    return out


def fdk_slice_threaded(projections, config, initial_angle=0, **kwargs):

    #num_cpus = psutil.cpu_count(logical=False)
    #if len(projections) <= num_cpus:
    #    num_cpus = len(projections)
    ray.init()
    proj_width = projections[0][0].shape[0]
    proj_height = projections[0][0].shape[1]
    recon = np.zeros((proj_width, proj_height), dtype=np.float64)
    angles = np.linspace(0, (2 * np.pi), len(projections)) + np.deg2rad(initial_angle)
    print(f'Angles: {angles}')
    #proj_mem = ray.put(projections[0])
    temp = []
    for projection, angle in zip(projections, angles):
        temp.append(_fdk_slice.remote(projection, angle, config))
    for slice in tqdm(temp, total=len(temp)):
        recon += ray.get(slice)
    ray.shutdown()
    return recon // np.float(len(projections))


@ray.remote
def _fdk_slice(projection, angle, config):

        x_proj = projection[0].T[0, :]
        y_proj = projection[0].T[:, 0]
        z = 0
        proj_width = projection[0].shape[0]
        proj_height = projection[0].shape[1]
        recon = np.zeros((len(x_proj), len(y_proj)), dtype=np.float64)
        #x_proj = x_proj + config.center_of_rot_y
        U = (config.source_to_detector_dist + (x_proj * np.cos(angle)) + (y_proj * np.sin(angle)))
        #ratio = config.source_to_object_dist // config.source_to_detector_dist
        ratio = (config.source_to_object_dist ** 2.) / (U) ** 2
        #R = (ratio + (x_proj * np.cos(angle)) + (y_proj * np.sin(angle)))
        #projection = projection[0] + ratio
        radius = proj_width / 2.
        x = np.arange(proj_width) - radius
        x += config.center_of_rot_y
        x_r, y_r = np.mgrid[:config.n_voxels_x, :config.n_voxels_y] - radius
        det_a = config.source_to_detector_dist * ((-x_r * np.sin(angle)) + (y_r * np.cos(angle))) / (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle)))
        #det_a = config.source_to_detector_dist * ((-x_proj * np.sin(angle)) + (y_proj * np.cos(angle))) / (config.source_to_detector_dist + (x_proj * np.cos(angle)) + (y_proj * np.sin(angle)))

        #det_b = z * (config.source_to_detector_dist * (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle))))
        for col in projection[0].T:
            t = y_r * np.cos(angle) - x_r * np.sin(angle)
            #interpolant = map_coordinates(projection[0], [det_a], cval=0., order=1, prefilter=False)
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
            #interpolant = interp2d()
            recon = recon + U[:, np.newaxis] * interpolant(det_a)# * ratio
        return recon


def fdk_vol(projections, config, **kwargs):

    proj_width = projections[0][0].shape[0]
    proj_height = projections[0][0].shape[1]
    recon_vol = np.zeros((2000, 2000, 2000))
    angles = np.linspace(0, 360, len(projections))

    for projection, angle in tqdm(zip(projections, angles), total=len(projections)):
        angle = np.deg2rad(angle)
        x_proj = projection[0][:, 0]
        y_proj = projection[0][0, :]

        y_proj += config.center_of_rot_y
        U = (config.source_to_detector_dist + (x_proj * np.cos(angle)) + (y_proj * np.sin(angle)))
        ratio = (config.source_to_detector_dist ** 2) // (U ** 2)

        x_r, y_r, z_r = np.meshgrid(config.object_xs, config.object_ys, config.object_zs, sparse=True)
        #x_r = np.array(config.object_xs)#x_r, y_r, z_r = np.meshgrid(config.object_xs, config.object_ys, config.object_zs, sparse=True)
        #y_r = np.array(config.object_ys)
        #z_r = np.array(config.object_zs)

        #print(getsizeof(x_r))
        det_a = config.source_to_detector_dist * ((-x_r * np.sin(angle)) + (y_r * np.cos(angle))) // (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle)))
        det_b = z_r * (config.source_to_detector_dist * (config.source_to_detector_dist + (x_r * np.cos(angle)) + (y_r * np.sin(angle))))

        x = np.arange(proj_width) - proj_width // 2
        y = np.arange(proj_height) - proj_height // 2

        for col in projection[0]:
            interpolant_x = partial(np.interp, xp=x, fp=col, left=0, right=0)
            interpolant_y = interp2d()


            #interpolant_z = partial(np.interp, xp=z_r, fp=col, left=0, right=0)
            #
            #interpolant = map_coordinates(col, [det_a, det_b], cval=0., order=1)
            #recon_vol += map_coordinates(col, [det_a, det_b], cval=0., order=1)#interpolant#_x(det_a)
            recon_vol += interpolant_x(det_a)
            recon_vol += interpolant_y(det_b)
            #recon_vol_z += interpolant_y(det_b)
            #interpolant = interp1d(det_a, projection[0], kind=interpolation, bounds_error=False, fill_value=0)
        #recon_vol += interpolant

    return np.array((recon_vol_x, recon_vol_y, recon_vol_z))


def read_projections(path, virtual_stack=False, flat_corrected=True, n_proj=3142):

    proj_stack = []
    print(f'Reading {n_proj} Projections')
    for f in tqdm(sorted(glob.glob('*.tif'))[:n_proj], total=len(glob.glob('*.tif')[:n_proj])):
        proj_stack.append(np.array(read_image(f, flat_corrected=True)))

    return proj_stack


def filter_projections(param, projections):

    filtered_stack = []

    for projection in projections, :
        projection = -np.log(projection)
        projection_filtered = np.squeeze(ramp_filter_and_weight(projection[:, :, np.newaxis], param), 2)
        filtered_stack.append(projection_filtered)

    return filtered_stack


def recon(projections, param, single_slice=True, **kwargs):
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
        return fdk_slice_threaded(filtered_stack, param, initial_angle=0)
    else:
        return fdk_vol(filtered_stack, param)
    #return filtered_stack
