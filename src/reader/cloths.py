import numpy as np
from scipy import misc

FLAG = '/cvlabdata1/cvlab/datasets_aoun/flag_2/'
IMAGES = 'images/'
ALBEDOS = 'albedos/'
SHADINGS = 'shadings/'
SYNTHS = 'synths/'

TYPE = 'b31_tl_tr-cotton'
NAME = f'sh-{TYPE}_t-cat_flowers_'
TYPE += '/'

def load_data():
    x = misc.imread(FLAG + IMAGES + TYPE + NAME + '00008.tiff')
    arr = np.empty([2352, 224, 224, 3])
    arr[0] = x / 255.0

    return arr
