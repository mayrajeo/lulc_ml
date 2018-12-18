import numpy as np
from keras.layers import Conv2D, Input
from keras.layers import Activation
from keras.models import Model

from .unet_blocks import Transpose2D_block
from .unet_blocks import Upsample2D_block
from .unet_blocks import Encoder2D_block

"""
Builds unet from blocks

Slightly changed from https://github.com/qubvel/segmentation_models
so that backbone is not preloaded but rather generated.
Otherwise identical.
"""


def build_unet(channels, classes, 
               encoder_filters=(32,64,128,256,512),    
               decoder_filters=(512,256,128,64,32),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=4,
               block_type='transpose',
               activation='sigmoid',
               use_batchnorm=True):

    input = Input((None, None, channels))
    # Create encoding path
    for i in range(n_upsample_blocks):
        if i == 0:
            x = Encoder2D_block(encoder_filters[i], i, pool_size=(2,2),
                                use_batchnorm=use_batchnorm)(input)
        elif i == 4:
            x = Encoder2D_block(encoder_filters[-1], 4, pool_size=(2,2),
                        use_batchnorm=use_batchnorm, has_pooling=False)(x)
        else:
            x = Encoder2D_block(encoder_filters[i], i, pool_size=(2,2),
                                use_batchnorm=use_batchnorm)(x)

    # Define if upsampling is done with transpose convolution or not
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block
    backbone = Model(input, x)

    # convert layer names to indices
    skip_connection_layers = ['encoder_stage3_relu2', 'encoder_stage2_relu2',
                              'encoder_stage1_relu2', 'encoder_stage0_relu2']
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    # Generate decoder
    for i in range(n_upsample_blocks):
        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)
    return model


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
	    return (x, x)
    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
	index of layer

    Raises:
	ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))
