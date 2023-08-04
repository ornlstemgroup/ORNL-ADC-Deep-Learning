import os.path
import random
import imageio
import cv2

import numpy as np
from math import exp
from skimage.util import random_noise
from skimage.exposure import match_histograms
from skimage.morphology import disk
from scipy.ndimage import filters
from skimage.filters import rank
from utils import *

def calculate_kernel(dimension=(256, 256), sigma=10.0): # gaussian kernel
    kernel = np.zeros(dimension)
    row, col = dimension
    for r in range(row):
        for c in range(col):
            offset_x = r - (row>>1)
            offset_y = c - (col>>1)
            kernel[r][c] = exp(-(offset_x*offset_x+offset_y*offset_y) / (sigma+sigma))

    return kernel

def convolution(image, kernel): # naive 2D convolution
    assert 2 == len(image.shape)
    assert 2 == len(kernel.shape)
    return cv2.filter2D(src=image, kernel=kernel, ddepth=-1)

def normalize(image) :
    return (image-np.amin(image))/(np.amax(image)-np.amin(image)+1.0e-10) # scale to [0, 1]

def simulate(dimension=(512, 512), range_of_columns=(20, 50), sigma=45, max_intensity=5, offset=32, reference=None):
    # random choose some points and intensities for these points
    columns = int(random.uniform(*range_of_columns))
    row, col = dimension
    row_coords = np.random.randint(offset, row-offset, columns)
    col_coords = np.random.randint(offset, col-offset, columns)
    intensities = np.random.randint(1, max_intensity, columns)
    print(intensities)

    image = np.zeros(dimension)
    gt = np.zeros(dimension)

    coords = []
    for idx in range(columns):
        image[row_coords[idx]][col_coords[idx]] = intensities[idx]
        # gt[row_coords[idx]][col_coords[idx]] = 1.0
        mask = create_circular_mask(gt.shape[0], gt.shape[1], (col_coords[idx],row_coords[idx]), 8)
        # att = create_circular_att(mask, gt.shape[0], gt.shape[1], (col_coords[idx],row_coords[idx]), 10)
        coords.append([row_coords[idx], col_coords[idx]])
        gt = np.maximum(gt, mask)
    # gt = gt / columns

    # imageio.imsave('stage1.jpg', image)
    # gt = filters.gaussian_filter(gt, sigma=4)
    # imageio.imsave('gt.jpg', gt*255)

    kernel = calculate_kernel(sigma=sigma, dimension=(row>>1, col>>1))  # 512 * 512 -> 256 * 256
    image = convolution(image=image, kernel=kernel)
    image = filters.gaussian_filter(image, sigma=1)
    image = random_noise(image, mode='localvar',clip=False)
    image = random_noise(image, mode='poisson',clip=False)
    image = random_noise(image, mode='speckle',clip=False)
    image = random_noise(image, mode='s&p',clip=False)
    # histogram matching
    # image = match_histograms(image, reference, channel_axis=-1)
    #image = normalize(image)
    # image = rank.equalize(image, disk(20)) / 255
    # imageio.imsave('image.jpg', image)
    # exit(0)

    return image, gt, coords

def simulate_images(images_to_generate, dimension=(512, 512), range_of_columns=(30, 70), sigmas=(9,100), max_intensity=5, reference=None):
    row, col = dimension
    random_sigma = np.random.randint(sigmas[0], sigmas[1], images_to_generate)
    images = np.zeros((images_to_generate, row, col))
    gts = np.zeros((images_to_generate, row, col))

    all_coords = []
    for idx in range(images_to_generate):
        images[idx,:,:], gts[idx,:,:], coords = simulate(dimension=dimension,
                                                           range_of_columns=range_of_columns,
                                                           sigma=random_sigma[idx],
                                                           max_intensity=max_intensity,
                                                           offset=32,
                                                           reference=reference)
        all_coords.append(coords)

    images = normalize(images)
    images = np.array(images*65535.0, dtype='uint16')
    # imageio.imsave('stage2.jpg', image)

    return images, gts, all_coords


if __name__ == '__main__':
    save_path = './Synthetic_data'
    images_to_generate = 1
    reference = imageio.imread('./Data/temp0.png')


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    simulated_images, gt, all_coords = simulate_images(images_to_generate=images_to_generate,
                                                               dimension=(512, 512),
                                                               range_of_columns=(20, 50),
                                                               sigmas=(5, 40),
                                                               max_intensity=4,
                                                               reference=reference)

    for i in range(images_to_generate):
        # image
        file_name = f'{save_path}/{str(i).zfill(4)}.tif'
        imageio.imsave(file_name, simulated_images[i,:, :])

        # gt
        file_name = f'{save_path}/{str(i).zfill(4)}_gt.tif'
        imageio.imsave(file_name, gt[i,:, :])

        # coords
        txt_name = f'{save_path}/{str(i).zfill(4)}.txt'
        with open(txt_name, 'w') as f:
            for j in range(len(all_coords[i])):
                f.write(f'{all_coords[i][j]}' + '\n')