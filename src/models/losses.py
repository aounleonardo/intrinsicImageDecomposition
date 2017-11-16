from keras import backend as K
img_height = 224

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true[0:img_height]), axis=-1)