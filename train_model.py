"""
This is the script for training the model.

You should read keras documentation before 
"""
import numpy as np 
import matplotlib.pyplot as plt
import keras
import sys
import pickle

from keras.callbacks import ModelCheckpoint
from unet import unet_builder
from unet import losses
from utils import data_utils as du
from utils.unet_utils import ModelMGPU

def train_unet():

    # Set the hyperparameters and store them to dictionary
    batch_size = 8 # how many images to process before updating the gradient
    n_classes = 13 # number of classes
    n_chans = 14 # number of channels
    epochs = 5 # number of epochs to train the model
    rotation = 16 # How many different rotations to perform
    reflection = True # Are random flips included in augmentations
    # Next generate lists containing paths to training and validation images and masks
    train_images = ['data/training_data/train_data_image_' + str(i) + '.npy' for i in range(8)]
    train_masks = ['data/training_data/train_masks_' + str(i) + '.npy' for i in range(8)]
    val_images = ['data/training_data/val_data_image_' + str(i) + '.npy' for i in range(2)]
    val_masks = ['data/training_data/val_masks_' + str(i) + '.npy' for i in range(2)]

    # Set parameters for train and test datagenerators
    train_params = {'dim': [64,128,256,384],
                    'image_files': train_images,
                    'mask_files': train_masks,
                    'batch_size': batch_size,
                    'n_classes': n_classes,
                    'n_chans': n_chans,
                    'bands': list(range(n_chans)),
                    'single_label': None,
                    'rotation':rotation,
                    'reflection':reflection,
                    'epoch_len':10} # number of batches in one epoch

    test_params = {'dim': [128,256],
                   'image_files': val_images,
                   'mask_files': val_masks,
                   'batch_size': batch_size,
                   'n_classes': n_classes,
                   'n_chans': n_chans,
                   'bands': list(range(n_chans)),
                   'single_label': None,
                   'rotation':rotation,
                   'reflection':reflection,
                   'epoch_len':4}

    # Datagenerators for model
    train_gen = du.DataGeneratorImage(**train_params)
    val_gen = du.DataGeneratorImage(**test_params)

    # Save weights from each epoch
    cb = ModelCheckpoint('models/test-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=False)

    # Use Nadam as optimizer. Other possibilities can be found https://keras.io/optimizers/
    opt = keras.optimizers.Nadam()

    # Loss and activation functions depend on binary or multiclass
    # If binary classification, then activation is sigmoid and loss is binary crossentropy
    #activation = 'sigmoid'
    #loss = keras.losses.binary_crossentropy 

    # If multiclass classification, then activation is softmax and loss is categorical_crossentropy
    activation = 'softmax'
    loss = keras.losses.categorical_crossentropy

    # We have also other possible losses, such as focal and Lovasz-Softmax, but they still need tuning and stuff
    # Don't try them if you don't know what you are doing
    #loss = losses.focal()
    
    # Build and compile network
    model = unet_builder.build_unet(n_chans, n_classes, activation=activation)
    parallel_model = model

    #If multiple gpus are possible and sensible to use, then you can use ModelMGPU
    #parallel_model = ModelMGPU(model, gpus=2)

    parallel_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    # print(model.summary())

    # Train the network. 
    model_train = parallel_model.fit_generator(train_gen, epochs=epochs, 
                                      verbose =1, validation_data = val_gen,
                                      callbacks=[cb], use_multiprocessing=True, workers = 4)


    # Save model history JUST IN CASE something about graphs goes wrong
    with open('histories/test_history.obj', 'wb') as history:
        pickle.dump(model_train.history, history, protocol=pickle.HIGHEST_PROTOCOL) 

    # Plot and save the training metrics
    # Change savefig locations accordingly
    plt.figure()
    acc = model_train.history['acc']
    val_acc = model_train.history['val_acc']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epoc = range(len(acc))
    plt.plot(epoc, acc, 'bo', label='Training accuracy')
    plt.plot(epoc, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('data/graphs/test_training_acc.pdf', bbox_inches='tight')
    plt.figure()
    plt.plot(epoc, loss, 'bo', label='Training loss')
    plt.plot(epoc, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('data/graphs/test_training_loss.pdf' , bbox_inches='tight')



if __name__ == '__main__':
    train_unet()
