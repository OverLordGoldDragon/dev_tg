# -*- coding: utf-8 -*-
import os
import sys
import inspect
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(inspect.stack()[0][1])
if sys.path[0] != filedir:
    if filedir in sys.path:
        sys.path.pop(sys.path.index(filedir))  # avoid dudplication
    sys.path.insert(0, filedir)

import pytest

from time import time
from copy import deepcopy

from backend import BASEDIR, tempdir, notify, pyxfail
from backend import _init_session, make_autoencoder, _do_test_load

pytestmark = pyxfail


#### CONFIGURE TESTING #######################################################
batch_size = 128
width, height = 28, 28
channels = 1
batch_shape = (batch_size, width, height, channels)
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image_lz4f')

MODEL_CFG = dict(
    batch_shape=batch_shape,
    loss='mse',
    metrics=None,
    optimizer='adam',
    num_classes=10,
    activation=['relu'] * 4 + ['sigmoid'],
    filters=[2, 2, 1, 2, 1],
    kernel_size=[(3, 3)] * 5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    data_loader='numpy-lz4f',
    data_loader_dtype='float64',
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    full_batch_shape=batch_shape,
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    data_loader='numpy-lz4f',
    data_loader_dtype='float64',
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    full_batch_shape=batch_shape,
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=2,
    val_freq={'epoch': 1},
    input_as_labels=True,
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load', 'predict')}
autoencoder = make_autoencoder(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_autoencoder)
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        _test_main(C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


def _test_main(C, new_model=False):
    if new_model:
        tg = init_session(C)
    else:
        tg = init_session(C, model=autoencoder)
    tg.train()
    _test_load(tg, C)


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_predict():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        C['traingen']['eval_fn_name'] = 'predict'
        _test_main(C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
