from keras.layers import Conv2DTranspose, Conv2D, UpSampling2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Concatenate

"""
Slightly changed from https://github.com/qubvel/segmentation_models

Additions are handle_encoder_block_names and Encoder2D_block

Building blocks for U-net. All blocks use ReLU as their activation function
If you want to use some other, like LReLU or ELU, then you must import 
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU and
replace Activation('relu', name=relu_name) with for example
ELU(name=relu_name)
"""


def handle_decoder_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_up'.format(stage)
    return conv_name, bn_name, relu_name, up_name

def handle_encoder_block_names(stage):
    conv_name = 'encoder_stage{}_conv'.format(stage)
    bn_name = 'encoder_stage{}_bn'.format(stage)
    relu_name = 'encoder_stage{}_relu'.format(stage)
    pool_name = 'encoder_stage{}_pool'.format(stage)
    return conv_name, bn_name, relu_name, pool_name

def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', 
             bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding='same', name=conv_name, 
                   use_bias=not(use_batchnorm), kernel_initializer = 'he_normal')(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer

def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

       	conv_name, bn_name, relu_name, up_name = handle_decoder_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

       	if skip is not None:
            x = Concatenate()([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer

def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_decoder_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

       	x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer

def Encoder2D_block(filters, stage, kernel_size=(3,3), pool_size=(2,2),
                    use_batchnorm=False, has_pooling=True):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, pool_name = handle_encoder_block_names(stage)

       	x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', 
                     relu_name=relu_name + '1')(input_tensor)
        
       	x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        if has_pooling:
            x = MaxPooling2D(pool_size, name=pool_name)(x)

        return x
    return layer
