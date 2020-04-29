# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from deeptrain.preprocessing import data_to_hdf5

batch_shape = (64, 50, 8)
n_batches = 12
save_batch_size = 128
overwrite = False

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "timeseries")

def make_data(batch_shape, n_batches):
    X = np.random.randn(n_batches, *batch_shape)
    Y = np.random.randint(0, 2, (n_batches, batch_shape[0], 1))
    return X, Y

###############################################################
x_train, y_train = make_data(batch_shape, n_batches)
x_test,  y_test  = make_data(batch_shape, n_batches // 2)

kw = dict(batch_size=save_batch_size, overwrite=overwrite)
data_to_hdf5(join(basedir, "train", "data.h5"),   data=x_train, **kw)
data_to_hdf5(join(basedir, "train", "labels.h5"), data=y_train, **kw)
data_to_hdf5(join(basedir, "val",   "data.h5"),   data=x_test,  **kw)
data_to_hdf5(join(basedir, "val",   "labels.h5"), data=y_test,  **kw)
