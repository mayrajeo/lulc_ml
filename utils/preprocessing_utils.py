import numpy as np
import xarray as xr
import sys

from rasterio import logging
from itertools import product

"""
Mostly obsoleted because of different way to generate training data
"""

log = logging.getLogger()
log.setLevel(logging.ERROR)
np.set_printoptions(suppress=True)

def to_cat(y, num_classes=None, dtype='float32'):
    """Copied here from keras source because of reasons
    
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_train_data(img, x, dx ,y, dy):
    data = np.copy(img[x:x+dx,y:y+dy,:])
    return data

def get_train_mask(img, x, dx, y, dy, classes):
    mask = np.copy(img[x:x+dx,y:y+dy])
    # If for some reason there is NaN replace with median
    mask[np.isnan(mask)] = np.median(mask)
    mask -= 1
    mask = to_cat(mask, classes)
    mask = mask.reshape(dx, dy, classes)
    return mask

def augment(data):
    """Perform data-augmentation. Seven-folds available data"""
    augs = [data]
    #clockwise
    augs.append(np.rot90(data, 1))
    #counterclockwise
    augs.append(np.rot90(data, 3))
    #horizontal flip
    augs.append(data[:,::-1])
    #vertical flip
    augs.append(data[::-1, :])
    #CW and horizontal flip
    augs.append(np.rot90(data[:,::-1], 1))
    #CWW and horizontal  flip
    augs.append(np.rot90(data[:,::-1], 3))
    augs = np.array(augs)
    return augs

def normalize_image(data):
    """
    Normalize by zero-centering and dividing by standard deviation
    1e-8 prevents dividing by zero
    """
    data_mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - data_mean) / (std_dev + 1e-8)

def patch_data(data):
    """Patch missing data with median values"""
    for p in range(data.shape[0]):
        data[p, data[p] == 0] = np.median(data[p])
    return data


def process_images(bands, ind, masks, n_classes, x_start, x_end, dx,
                   y_start, y_end, dy, stride, target_folder):
    """
    Processes large geotiff-images into smaller chunks.
    Combines band and index data into one file.
    """
    # Reshape to get channels last
    bands = np.swapaxes(bands, 0, 2)
    bands = np.swapaxes(bands, 0, 1)
    ind = np.swapaxes(ind, 0, 2)
    ind = np.swapaxes(ind, 0, 1)

    # Stack into one image
    vals = np.dstack((bands, ind))
    #masks = np.swapaxes(masks, 1,2)
    # Initialize array containing the images
    #train_imgs = np.empty((0, dx, dy, vals.shape[2]))
    #train_masks = np.empty((0, dx, dy, n_classes))
    i = 0
    tot_images = ((x_end - dx - x_start) // (stride)) * ((y_end - dy - y_start) // (stride)) 
    # Generate subimages
    for x, y in product(range(x_start, x_end-dx, stride), 
                        range(y_start, y_end-dy, stride)):
        data = get_train_data(vals, x, dx, y, dy)
        mask = get_train_mask(masks, x, dx, y, dy, n_classes)
        data = data.astype('float32')
        mask = mask.astype('float32')
        data[...,:9] /= 10000
        data[...,9:] /= 200
        # Augment images
        #train_imgs = np.vstack((train_imgs, augment(data)))
        #train_masks = np.vstack((train_masks, augment(mask)))
        train_imgs = augment(data)
        train_masks = augment(mask)

        for j in range(5):
            np.save('{}/images/{}.npy'.format(target_folder, 5*i+j), train_imgs[j])
            np.save('{}/masks/{}.npy'.format(target_folder, 5*i+j), train_masks[j])
        i += 1
        sys.stdout.write('\r')
        sys.stdout.write('{}/{} images generated'.format(i*5, tot_images*5))
        sys.stdout.flush()
 
    return #train_imgs, train_masks
