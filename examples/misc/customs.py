# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
   - Setting custom loss and metrics
   - Setting custom data loader
"""
import sys
import inspect
from pathlib import Path
# ensure `examples` directory path is on top of Python's module search
filedir = str(Path(inspect.stack()[0][1]).parents[1])

if sys.path[0] != filedir:
    if filedir in sys.path:
        sys.path.pop(sys.path.index(filedir))  # avoid dudplication
    sys.path.insert(0, filedir)

from utils import AE_CONFIGS, make_autoencoder, init_session

import numpy as np
from tensorflow.keras import backend as K
from deeptrain.metrics import _weighted_loss, _standardize

#%%# Configuration ###########################################################
def mean_L_error(y_true, y_pred, sample_weight=1):
    L = 1.5  # configurable
    y_true, y_pred, sample_weight = _standardize(y_true, y_pred, sample_weight)
    return _weighted_loss(np.mean(np.abs(y_true - y_pred) ** L, axis=-1),
                          sample_weight)

def mLe(y_true, y_pred):
    exp = 1.5  # configurable
    return K.mean(K.pow(K.abs(y_true - y_pred), exp), axis=-1)

def numpy_loader(self, set_num):
    # allow_pickle is irrelevant here, just for demo
    return np.load(self._path(set_num), allow_pickle=True)

AE_CONFIGS['model']['loss'] = mLe
AE_CONFIGS['datagen']['data_loader'] = numpy_loader
AE_CONFIGS['traingen']['custom_metrics'] = {'mLe': mean_L_error}
tg = init_session(AE_CONFIGS, make_autoencoder)

#%%# Train ##################################################################
tg.train()
