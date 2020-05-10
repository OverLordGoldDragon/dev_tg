# -*- coding: utf-8 -*-
import os
os.environ['IS_MAIN'] = '1' * (__name__ == '__main__')
import pytest

from pathlib import Path
from time import time
from copy import deepcopy

from tests.backend import Input, Dense, LSTM
from tests.backend import l2
from tests.backend import Model
from tests.backend import BASEDIR, tempdir, notify
from deeptrain.callbacks import predictions_per_iteration_cb
from deeptrain.callbacks import predictions_distribution_cb
from deeptrain.callbacks import comparative_histogram_cb
from deeptrain import TrainGenerator, SimpleBatchgen


datadir = os.path.join(BASEDIR, 'tests', 'data', 'timeseries')
batch_size = 32

MODEL_CFG = dict(
    batch_shape=(batch_size, 4, 6),
    units=6,
    optimizer='adam',
    loss='binary_crossentropy'
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.csv'),
    batch_size=batch_size,
    data_category='timeseries',
    shuffle=True,
    preprocessor_configs=dict(batch_timesteps=20, window_size=4),
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_dir=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.csv'),
    batch_size=batch_size,
    data_category='timeseries',
    shuffle=False,
    preprocessor_configs=dict(batch_timesteps=20, window_size=4),
)
TRAINGEN_CFG = dict(
    epochs=2,
    reset_statefuls=True,
    max_is_best=False,
    dynamic_predict_threshold_min_max=(.35, .9),
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    best_subset_size=3,
    model_configs=MODEL_CFG,
    callbacks={'ch': {'val_end': comparative_histogram_cb},
               'ppi': {'val_end': predictions_per_iteration_cb},
               'pd': {'val_end': predictions_distribution_cb}},
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load', 'weighted_slices',
                                      'predict')}


@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = _init_session(C)
        tg.train()
        _test_load(tg, C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_weighted_slices():
    t0 = time()
    C = deepcopy(CONFIGS)
    C['traingen'].update(dict(eval_fn_name='predict',
                              loss_weighted_slices_range=(.5, 1.5),
                              pred_weighted_slices_range=(.5, 1.5)))
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = _init_session(C)
        tg.train()
        _destroy_session(tg)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_predict():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        C['traingen'].update(dict(eval_fn_name='predict',
                                  key_metric='f1_score',
                                  val_metrics=('tnr', 'tpr'),
                                  plot_first_pane_max_vals=1,
                                  metric_printskip_configs={'val': 'f1_score'},
                                  class_weights={0: 1, 1: 5},
                                  ))
        tg = _init_session(C)
        tg.train()
        _test_load(tg, C)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def _test_load(tg, C):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    _destroy_session(tg)

    weights_path, loadpath = _get_latest_paths(logdir)
    tg = _init_session(C, weights_path, loadpath)


def _make_model(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'units', 'optimizer')
        return [kw[key] for key in expected_kw]

    batch_shape, loss, units, optimizer = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, return_sequences=False, stateful=True,
               kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4),
               bias_regularizer=l2(1e-4))(ipt)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def _init_session(C, weights_path=None, loadpath=None):
    model = _make_model(weights_path, **C['model'])
    dg  = SimpleBatchgen(**C['datagen'])
    vdg = SimpleBatchgen(**C['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath,
                         **C['traingen'])
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


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
