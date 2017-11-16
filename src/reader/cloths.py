import numpy as np
from scipy import misc
import os

FLAG = '/cvlabdata1/cvlab/datasets_aoun/flag_2/'
IMAGES = 'images/'
ALBEDOS = 'albedos/'
SHADINGS = 'shadings/'
SYNTHS = 'synths/'

TYPE = 'b31_tl_tr-cotton'
NAME = f'sh-{TYPE}_t-cat_flowers_'
TYPE += '/'


def load_data():
    count = 0
    x_tr = np.empty([2352, 224, 224, 3])
    y_tr = np.empty([2352, 224, 224, 3])
    for filename in os.listdir(FLAG + SYNTHS + TYPE):
        if filename.startswith(NAME):
            x = misc.imread(FLAG + SYNTHS + TYPE + filename, mode='RGB')
            y = misc.imread(FLAG + SHADINGS + TYPE + filename, mode='RGB')
            x_tr[count] = x / 255.0
            y_tr[count] = y / 255.0
            count += 1

    return x_tr, y_tr, count
