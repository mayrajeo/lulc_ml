import numpy as np
import xarray as xr
import os
import random
import keras
import cv2
import random
from osgeo import gdal

from rasterio import logging
from keras.utils import to_categorical
from itertools import product
from keras import Model

"""
Collection of data processing utilities to use in various stages.
"""

log = logging.getLogger()
log.setLevel(logging.ERROR)
np.set_printoptions(suppress=True)



def sub_image_generator(img, x_start, x_end, dx, y_start, y_end, dy, channels):
    """
    Generates test batches from full image. Works well with predict_generator from keras
    """
    for x, y in product(range(x_start, x_end, dx),
                        range(y_start, y_end, dy)):
        data = img[:, y:y+dy, x:x+dx]
        yield np.swapaxes(data, 0, 2).reshape((-1, dx, dy, channels))

def create_full_image(preds, x_start, x_end, y_start, y_end):
    """
    Creates full image from smaller images generated with sub_image_generator
    """
    row_len = (x_end - x_start)
    col_len = (y_end - y_start)

    n, nrows, ncols = preds.shape
    return (preds.reshape(row_len//nrows, -1, nrows, ncols)
            .swapaxes(1,2)
            .reshape(row_len, col_len))

def untile_preds(preds, x_start, x_end, y_start, y_end):
    """
    Like create_full_image but for multidimensional array 
    """
    row_len = x_end - x_start
    col_len = y_end  - y_start
    
    n, nrows, ncols, dim = preds.shape
    return (preds.reshape(row_len//nrows, -1, nrows, ncols, dim)
            .swapaxes(1,2).reshape(row_len, col_len, dim))

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator for training. Not as useful as DataGeneratorImage though.

    Credit: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, image_folder, mask_folder, list_IDs, batch_size=4, dim=(256,256), 
                 n_chans = 14, n_classes = 13, shuffle=True, 
                 bands=list(range(14)), single_label=None):
        self.dim = dim
        self.batch_size = batch_size
        self.mask_folder = mask_folder
        self.image_folder = image_folder
        self.image_IDs = list_IDs
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.bands = bands
        self.single_label = single_label
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        IDs_temp = [self.image_IDs[k] for k in indexes]
        img, mask = self.__data_generation(IDs_temp)
        return img, mask

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, IDs_temp):
        # Create numpy arrays for batch
        images = np.empty((self.batch_size, *self.dim, self.n_chans))
        masks = np.empty((self.batch_size, *self.dim, self.n_classes))

        if self.n_classes == 1:
            for i, ID in enumerate(IDs_temp):
                images[i, ] = np.load(self.image_folder + str(ID) + '.npy')[:,:,self.bands]
                masks[i, ] = np.load(self.mask_folder + str(ID) + '.npy')[:,:,self.single_label].reshape(*self.dim, 1)
        else:
            for i, ID in enumerate(IDs_temp):
                images[i, ] = np.load(self.image_folder + str(ID) + '.npy')[:,:,self.bands]
                masks[i, ] = np.load(self.mask_folder + str(ID) + '.npy') 

        return images, masks


class DataGeneratorImage(keras.utils.Sequence):
    """
    ImageDataGenerator that reads .npy -file and generates randomly augmented images 
    from it. Augmentations include random rotations and reflections. 

    Parameters:

    dim: Dimensions of output images. Must be divisible by 16
    batch_size: how many images to generate for one batch
    mask_files: list of paths to mask files
    image_files: list of paths to image files. must have same order than mask_files
    n_chans: number of channels in image
    n_classes: number of classes to predict
    bands: list of bands to use for classification
    single_label: flag for binary or multiclass classification
    rotation: number of equally sized possible rotations to perform
    reflection: flag for if random reflections are performed

    Credit where it's due:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    """
    def __init__(self, image_files, mask_files, batch_size=4, dim=(256,256), 
                 n_chans = 14, n_classes = 13, bands=list(range(14)), 
                 single_label=None, rotation = 16, reflection=True, epoch_len= 8000):
        self.dim = dim
        self.batch_size = batch_size
        self.masks = [np.load(f).astype('float32') for f in mask_files]
        # Here we replace missing values with median and make labels start from 0
        for mask in self.masks:
            mask[np.isnan(mask)] = np.round(np.median(mask))
            mask -= 1
        self.images = [np.load(f).astype('float32') for f in image_files]
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.bands = bands
        self.single_label = single_label
        self.rotation = rotation
        self.reflection = reflection
        self.epoch_len = epoch_len

    def __len__(self):
        return int(self.epoch_len)

    def __getitem__(self, index):
        img, mask = self.__data_generation()
        return img, mask

    def __data_generation(self):
        """
        Reads images from the file and performs random rotation
        and flips to it. Ensures that output size valid
        """
        # Create numpy arrays for batch
        batch_dim = random.sample(self.dim, 1)[0]
        batch_images = np.empty((self.batch_size, batch_dim, batch_dim, self.n_chans))
        batch_masks = np.empty((self.batch_size, batch_dim, batch_dim, self.n_classes))
        index = np.random.randint(0, len(self.images), 1)

        for i in range(self.batch_size):
            # If rotation is not performed, just set some values
            if not self.rotation or self.rotation == 1:
                crop_diff = 0
                crop_size_new = self.dim
            # Else calculate random rotation and rotation matrix for it
            else:
                angle = 360. * np.random.randint(0, self.rotation) / self.rotation
                radian = 2.*np.pi * angle / 360.
                crop_size_new = int(np.ceil(float(batch_dim) * (abs(np.sin(radian)) + 
                                                                abs(np.cos(radian)))))
                rot_mtx = cv2.getRotationMatrix2D((float(crop_size_new) / 2.,
                                                float(crop_size_new) / 2.),
                                                angle, 1.)
                crop_diff = int((crop_size_new -batch_dim)/2.)
        
            # Select random location from image
            x_base = np.random.randint(0, self.images[index].shape[1] - crop_size_new)
            y_base = np.random.randint(0, self.images[index].shape[0] - crop_size_new)
            # Crop image according to previous
            img_crop = np.squeeze(self.images[index][y_base:y_base+crop_size_new,
                                             x_base:x_base+crop_size_new, :])
            mask_crop = np.squeeze(self.masks[index][y_base:y_base+crop_size_new,
                                             x_base:x_base+crop_size_new])

            if not self.rotation or self.rotation == 1:
                img_rot = img_crop
                mask_rot = mask_crop
            # If rotation is performed, do affine transformation to image
            else:
                img_rot = cv2.warpAffine(img_crop, rot_mtx, (crop_size_new, crop_size_new))
                mask_rot = cv2.warpAffine(mask_crop, rot_mtx, (crop_size_new, crop_size_new))
            # If random reflections are performed, select here if it's done
            x_step = 1 if not self.reflection else [-1, 1][np.random.randint(0,2)]
            y_step = 1 if not self.reflection else [-1, 1][np.random.randint(0,2)]

            # Crop images to have suitable size
            batch_images[i,] = img_rot[crop_diff:crop_diff+batch_dim, crop_diff:crop_diff+batch_dim, :][::y_step,::x_step,:]
            batch_masks[i,] = to_categorical(mask_rot[crop_diff: crop_diff+batch_dim:, crop_diff: crop_diff+batch_dim:][::y_step,::x_step], self.n_classes)

        return batch_images, batch_masks


