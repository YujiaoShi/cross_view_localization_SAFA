
import numpy as np
from scipy.misc import imread, imsave
import os


def sample_within_bounds(signal, x, y, bounds):

    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2


############################ Apply Polar Transform to Aerial Images in CVUSA Dataset #############################
S = 750  # Original size of the aerial image
height = 112  # Height of polar transformed aerial image
width = 616   # Width of polar transformed aerial image

i = np.arange(0, height)
j = np.arange(0, width)
jj, ii = np.meshgrid(j, i)

y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

input_dir = '../Data/CVUSA/bingmap/19/'
output_dir = '../Data/CVUSA/polarmap/19/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

for img in images:
    signal = imread(input_dir + img)
    image = sample_bilinear(signal, x, y)
    imsave(output_dir + img.replace('jpg', 'png'), image)


############################ Apply Polar Transform to Aerial Images in CVACT Dataset #############################
S = 1200
height = 112
width = 616

i = np.arange(0, height)
j = np.arange(0, width)
jj, ii = np.meshgrid(j, i)

y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)


input_dir = '../Data/ANU_data_small/satview_polish/'
output_dir = '../Data/CVACT/polarmap/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

for img in images:
    signal = imread(input_dir + img)
    image = sample_bilinear(signal, x, y)
    imsave(output_dir + img.replace('jpg','png'), image)


############################ Prepare Street View Images in CVACT to Accelerate Training Time #############################
import cv2
input_dir = '../Data/ANU_data_small/streetview/'
output_dir = '../Data/CVACT/streetview/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

for img in images:
    signal = imread(input_dir + img)

    start = int(832 / 4)
    image = signal[start: start + int(832 / 2), :, :]
    image = cv2.resize(image, (616, 112), interpolation=cv2.INTER_AREA)
    imsave(output_dir + img.replace('jpg', 'png'), image)
