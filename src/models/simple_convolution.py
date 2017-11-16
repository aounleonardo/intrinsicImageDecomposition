import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from sys import path
sys.path.append('/home/aoun/intrinsicImageDecomposition/')

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K

FLAG = '/cvlabdata1/cvlab/datasets_aoun/flag_2/'
IMAGES = 'images/'
ALBEDOS = 'albedos/'
SHADINGS = 'shadings/'
SYNTHS = 'synths/'

TYPE = 'b31_tl_tr-cotton'
NAME = f'sh-{TYPE}_t-cat_flowers_'
TYPE += '/'

import src.reader.cloths as data
X, Y, count = data.load_data()
print(count)

img_width, img_height = 224, 224

train_data_dir = FLAG + SYNTHS + TYPE
validation_data_dir = FLAG + SHADINGS + TYPE
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(input_shape)


model = Sequential()
model.add(Conv2D(3, (3, 3), input_shape=input_shape, padding="same"))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(3, (3, 3)))
#
# model.add(Activation('relu'))

# add last layer width 1

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])


while True:
    model.train_on_batch(X[0:64], Y[0:64])
    print('Batch done')
