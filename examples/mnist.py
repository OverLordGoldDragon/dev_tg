# -*- coding: utf-8 -*-
import numpy as np

from keras.datasets import mnist
import keras.utils

from train_generatorr.util.preprocessing import numpy_data_to_numpy_sets
#%%

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
y_train = keras.utils.to_categorical(y_train)
x_test  = np.expand_dims(x_test,  -1)
y_test  = keras.utils.to_categorical(y_test)

axes = tuple(range(1, x_train.ndim))
x_train = x_train / x_train.max(axis=axes, keepdims=True)
x_test  = x_test  / x_test.max(axis=axes, keepdims=True)

#%%

kw = dict(batch_size=128, data_basename='128batch')
savedir = (r"C:\Desktop\School\Deep Learning\DL_code\dev_tg\train_generatorr\\"
           r"data\image\train")
numpy_data_to_numpy_sets(savedir, x_train, y_train, **kw)
savedir = (r"C:\Desktop\School\Deep Learning\DL_code\dev_tg\train_generatorr\\"
           r"data\image\val")
numpy_data_to_numpy_sets(savedir, x_test, y_test, **kw)
#%%