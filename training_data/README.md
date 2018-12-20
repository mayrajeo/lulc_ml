# Training and validation data

This folder contains small amount of training data, stored as .npy files. It also contains pickled .obj -file that has the means and standard deviations for unnormalizing the images.

Training data for images has shape \[x, y, chans\], masks have \[x, y\]

Unnormalizing is done like this:
```
import numpy as np
import pickle as pkl

image = '<path_for_image>.npy'

image = np.load(image)
with open('train_mean_and_std.obj', 'rb') as f:
    means, stds = pkl.load(f)

for i in range(image.shape[2]):
    image[...,i] *= stds[i]
    image[...,i] += means[i]
```
