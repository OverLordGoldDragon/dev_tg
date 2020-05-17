# -*- coding: utf-8 -*-
import os
import pytest

from time import time
from copy import deepcopy

from tests.backend import BASEDIR, tempdir, notify, make_classifier
from tests.backend import _init_session, _do_test_load


#### CONFIGURE TESTING #######################################################
batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=[8, 16],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=32,
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    superbatch_dir=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    dynamic_predict_threshold_min_max=(.35, .95),
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load', 'predict',
                                      'group_batch', 'recursive_batch')}
classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        C['traingen']['epochs'] = 2
        _test_main(C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


def _test_main(C, new_model=False):
    if new_model:
        tg = init_session(C)
    else:
        tg = init_session(C, model=classifier)
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


@notify(tests_done)
def test_group_batch():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        for name in ('traingen', 'datagen', 'val_datagen'):
            C[name]['batch_size'] = 64
        C['model']['batch_shape'] = (64, width, height, channels)
        _test_main(C, new_model=True)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_recursive_batch():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        for name in ('traingen', 'datagen', 'val_datagen'):
            C[name]['batch_size'] = 256
        C['model']['batch_shape'] = (256, width, height, channels)
        _test_main(C, new_model=True)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
