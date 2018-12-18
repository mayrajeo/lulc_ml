# /usr/bin/env/python

"""
This script is a command line alternative to classification notebook. This makes land cover 
classifications from 9 channel CORINE mosaic and 5 channel spectral index mosaic.

Assumes that input mosaics have no missing values.

Usage:
python make_classifications.py <model_weights> <band_mosaic> <index_mosaic> <outfile>

Arguments:
    model_weights:  Path to model weights to use
    band_mosaic:    GeoTIFF, NetCDF4 or ERDAS imagine file containing band reflectances (9 channels)
    index_mosaic:   GeoTIFF, NetCDF4 or ERDAS imagine file containing spectral indices (5 channels)
    outfile:        Path for outfile. Note that GDAL might not work if filename already exists
"""

import xarray as xr 
import numpy as np 
from keras.models import load_model
import keras
import sys
import math
from osgeo import gdal
from osgeo import osr

from utils import data_utils as du 
from unet import unet_builder

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)
np.set_printoptions(suppress=True)


def normalize_channel(channel_values):
    """Normalize one channel """
    chan_mean = np.mean(channel_values)
    chan_std = np.std(channel_values)
    normalized = (channel_values - chan_mean) / (chan_std + 1e-8)
    return normalized

def generate_padding(stack, width=256, height=256):
    """Pads array to get it into suitable shape"""
    x_pad = (math.ceil(stack.shape[2] / width) * width - stack.shape[2]) // 2
    y_pad = (math.ceil(stack.shape[1] / width) * width - stack.shape[1]) // 2
    padded_values = np.pad(stack, ((0,0), (y_pad, y_pad), (x_pad, x_pad)), 'reflect')
    return x_pad, y_pad, padded_values

def get_projection(file):
    """Extract projection from file"""
    orig_file= gdal.Open(file, gdal.GA_ReadOnly)
    proj = osr.SpatialReference()
    proj.ImportFromWkt(orig_file.GetProjectionRef())
    indices_for_projection = None
    return proj

def main(*args):
    # Read arguments
    weights = args[0]
    bandfile = args[1]
    indexfile = args[2]
    outfile = args[3]

    # Set width and height for sliding window
    width = 256
    height = 256

    bands = xr.open_rasterio(bandfile)
    indices = xr.open_rasterio(indexfile)
    vals = np.vstack((bands.values, indices.values))

    # Read x_min and y_max for geotransform
    x_min = indices.attrs['transform'][2]
    y_max = indices.attrs['transform'][-1]

    bands = None
    indices = None

    # Make shape be even
    if vals.shape[2] % 2 == 1:
        vals = vals[:,:,:-1]
    if vals.shape[1] % 2 == 1:
        vals = vals[:,:-1,:]

    vals = vals.astype('float32')
    for i in range(vals.shape[0]):
        vals[i] = normalize_channel(vals[i])
    
    x_pad, y_pad, padded_vals = generate_padding(vals)    

    model = unet_builder.build_unet(14,13, activation='softmax')
    model.load_weights(weights)

    # Calculate number of steps for generator
    steps = (padded_vals.shape[2] // width) * (padded_vals.shape[1] // height)

    # Make classifications
    pred = model.predict_generator(du.sub_image_generator(padded_vals, 0, padded_vals.shape[2], width, 
                                                          0, padded_vals.shape[1], height, 
                                                          padded_vals.shape[0]), 
                                   steps=steps, workers=1, max_queue_size=20, verbose=1)
    full_image = du.create_full_image(np.argmax(pred, axis=-1), 0, padded_vals.shape[2], 
                                      0, padded_vals.shape[1])

    # Remove padding and make predictions start from 1
    unpadded_image = full_image[x_pad:-x_pad, y_pad:-y_pad]
    unpadded_image = unpadded_image.T
    unpadded_image += 1

    # Untile predictions
    full_preds = du.untile_preds(pred, 0, padded_vals.shape[2], 0, padded_vals.shape[1])
    full_preds = full_preds[x_pad:-x_pad, y_pad:-y_pad,...].swapaxes(0,1)

    # Save predictions and activation maps as Geotiff with correct geotransform and projection
    proj = get_projection(indexfile)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfile, unpadded_image.shape[1], unpadded_image.shape[0], 
                       full_preds.shape[-1] + 1, gdal.GDT_Float32)

    ds.SetGeoTransform((x_min, 10, 0, y_max, 0, -10))
    ds.SetProjection(proj.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(unpadded_image)
    for i in range(2, 15):
        ds.GetRasterBand(i).WriteArray(full_preds[...,i-2])
    ds = None
    return


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Error!')
        print('Usage: python make_classifications <model_weights> <band_mosaic> <index_mosaic> <outfile>')
        sys.exit(0)
    main(*sys.argv[1:])