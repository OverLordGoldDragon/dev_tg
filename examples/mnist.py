# -*- coding: utf-8 -*-
import os
import numpy as np

import keras.utils
from keras.datasets import mnist

from deeptrain.util.preprocessing import numpy_data_to_numpy_sets

basedir = ''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train)
x_test  = np.expand_dims(x_test,  -1)
y_test  = keras.utils.to_categorical(y_test)

axes = tuple(range(1, x_train.ndim))
x_train = x_train / x_train.max(axis=axes, keepdims=True)
x_test  = x_test  / x_test.max(axis=axes, keepdims=True)

kw = dict(batch_size=128, data_basename='128batch')
savedir = os.path.join(basedir, "examples", "data", "image", "train")
numpy_data_to_numpy_sets(savedir, x_train, y_train, **kw)
savedir = os.path.join(basedir, "examples", "data", "image", "val")
numpy_data_to_numpy_sets(savedir, x_test, y_test, **kw)
