# /usr/bin/env python

"""
Pretty much obsoleted script due to different data generation
method during training.

Main script for preprocessing data into training, testing
and validating sets. 

Divides data into 256x256 windows with 64px stride.
Performs data augmentation by flipping both horizontally
and vertically and rotating clockwise and counterclockwise.

Patches zero values with median values.

Usage:
python preprocess_data.py <band_path> <index_path> <mask_path> <out_folder>
"""

import numpy as np 
import xarray as xr
import sys
import pickle

from sklearn.model_selection import train_test_split
from utils import preprocessing_utils as pre
from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)
np.set_printoptions(suppress=True)

def main(*args):
    band_file = args[0]
    index_file = args[1]
    mask_file = args[2]


    # Open files and patch missing data with median values
    print('Opening files')
    band_data = xr.open_rasterio(band_file).values
    index_data = xr.open_rasterio(index_file).values
    mask_data = xr.open_rasterio(mask_file).values[0]


    # Set some variables
    # Try to have this area similarly distributed than whole 
    # dataset. These testing and validating images are only
    # for testing purposes. Final test is performed for the 
    # whole dataset.
    x_start = 3248 # x-coordinate to start from
    x_end = 5040 # x-coordinate to end to
    y_start = 732 # y-coordinate to start from
    y_end = 1500 # y-coordinate to end to
    dx = 256 # width of one image
    dy = 256 # height of one image
    stride = 64 # how many pixels to stride at once
    n_classes = 13

    # Generate subimages
    print('Starting to generate smaller images')
    pre.process_images(band_data, index_data, mask_data, n_classes, x_start,
                                       x_end, dx, y_start, y_end, dy, stride, args[3])
    #band_data = index_data = mask_data = None
    #images = images.astype('float32')
    #masks = masks.astype('float32')
    print('\nImages generated')

    # Split into training, testing and validation sets.
    #train_images, test_images, train_masks, test_masks = train_test_split(images, masks, random_state=42,
    #                                                                      test_size = 0.2)
    #train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, 
    #                                                                    random_state=42, test_size = 0.2)

    # Normalize with train set mean and standard deviation
    #print('\nSets generated, normalizing')
    #train_mean = np.mean(train_images, axis = 0)
    #train_std = np.std(train_images, axis = 0)
    #train_images = (train_images - train_mean) / (train_std + 1e-8)
    #test_images = (test_images - train_mean) / (train_std + 1e-8)
    #val_images = (val_images - train_mean) / (train_std + 1e-8)

    # Save train_mean and train_std
    #with open('train_means.obj', 'wb') as vals:
    #    pickle.dump((train_mean, train_std), vals, protocol=pickle.HIGHEST_PROTOCOL) 

    # Save images as .npy files for training and testing
    #print('\nImages normalized, starting to save images.')
    #for i in range(train_images.shape[0]):
    #    np.save('{}/train/{}.npy'.format(images_out_folder, i), train_images[i])
    #    np.save('{}/train/{}.npy'.format(masks_out_folder, i), train_masks[i])
    #for i in range(test_images.shape[0]):
    #    np.save('{}/test/{}.npy'.format(images_out_folder, i), test_images[i])
    #    np.save('{}/test/{}.npy'.format(masks_out_folder, i), test_masks[i])
    #for i in range(val_images.shape[0]):
    #    np.save('{}/val/{}.npy'.format(images_out_folder, i), val_images[i])
    #    np.save('{}/val/{}.npy'.format(masks_out_folder, i), val_masks[i])

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Error!')
        print('Usage: python preprocess_data.py <band_path> <index_path> <mask_path> <out_folder>')
        sys.exit(0)
    main(*sys.argv[1:])
