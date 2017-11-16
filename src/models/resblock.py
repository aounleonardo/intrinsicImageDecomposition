# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Conv2D, Add
from keras.layers import Activation, BatchNormalization, Input

img_height = 224
img_width = 224
img_channels = 3


def residual_block(_in, nchannels=(64, 32), _strides=(1, 1), _project_shortcut=False):
    shortcut = _in
    _in = Conv2D(nchannels[0], (3, 3), input_shape=(224, 224, 3), padding="same")(_in)
    _in = BatchNormalization()(_in)
    _in = Activation('relu')(_in)

    _in = Conv2D(nchannels[1], (3, 3), input_shape=(224, 224, 3), padding="same")(_in)
    _in = BatchNormalization()(_in)

    _in = Add()([shortcut, _in])
    _in = Activation('relu')(_in)

    return _in