# -*- coding: utf-8 -*-
import os
import pytest

from pathlib import Path
from termcolor import cprint
from time import time

from .backend import Input, Dense, LSTM
from .backend import Model
from .backend import BASEDIR, tempdir
from deeptrain import TrainGenerator, SimpleBatchgen


datadir = os.path.join(BASEDIR, 'tests', 'data', 'timeseries')
batch_size = 32

MODEL_CFG = dict(
    batch_shape=(batch_size, 25, 16),
    units=16,
    optimizer='adam',
    loss='binary_crossentropy'
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.csv'),
    batch_size=batch_size,
    data_category='timeseries',
    shuffle=True,
    preprocessor_configs=dict(batch_timesteps=100, window_size=25),
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_dir=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.csv'),
    batch_size=batch_size,
    data_category='timeseries',
    shuffle=False,
    preprocessor_configs=dict(batch_timesteps=100, window_size=25),
)
TRAINGEN_CFG = dict(
    epochs=2,
    reset_statefuls=True,
    max_is_best=False,
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    best_subset_size=3,
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load', 'weighted_slices')}


def test_main():
    t0 = time()
    with tempdir(CONFIGS['traingen']['logs_dir']), tempdir(
            CONFIGS['traingen']['best_models_dir']):
        tg = _init_session(CONFIGS)
        tg.train()
        _test_load(tg, CONFIGS)
    print("\nTime elapsed: {:.3f}".format(time() - t0))
    _notify('main', tests_done)


def test_weighted_slices():
    t0 = time()
    CONFIGS['traingen'].update(dict(eval_fn_name='predict',
                                    loss_weighted_slices_range=(.5, 1.5),
                                    pred_weighted_slices_range=(.5, 1.5)))
    with tempdir(CONFIGS['traingen']['logs_dir']), tempdir(
            CONFIGS['traingen']['best_models_dir']):
        tg = _init_session(CONFIGS)
        tg.train()
        _destroy_session(tg)
    print("\nTime elapsed: {:.3f}".format(time() - t0))
    _notify('weighted_slices', tests_done)


def _test_load(tg, CONFIGS):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    _destroy_session(tg)

    weights_path, loadpath = _get_latest_paths(logdir)
    tg = _init_session(CONFIGS, weights_path, loadpath)

    _notify('load', tests_done)


def _make_model(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'units', 'optimizer')
        return [kw[key] for key in expected_kw]

    batch_shape, loss, units, optimizer = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, return_sequences=False, stateful=True)(ipt)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def _init_session(CONFIGS, weights_path=None, loadpath=None):
    model = _make_model(weights_path, **CONFIGS['model'])
    dg  = SimpleBatchgen(**CONFIGS['datagen'])
    vdg = SimpleBatchgen(**CONFIGS['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath,
                         **CONFIGS['traingen'])
    return tg


def _destroy_session(tg):
    def _clear_data(tg):
        tg.datagen.batch = []
        tg.datagen.superbatch = {}
        tg.val_datagen.batch = []
        tg.val_datagen.superbatch = {}

    _clear_data(tg)
    [delattr(tg, name) for name in ('model', 'datagen', 'val_datagen')]
    del tg


def _notify(name, tests_done):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        cprint("<< TIMESERIES TEST PASSED >>\n", 'green')

if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
