# -*- coding: utf-8 -*-
import os
import pytest

from pathlib import Path
from termcolor import cprint
from time import time
from copy import deepcopy

from tests.backend import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tests.backend import Model
from tests.backend import BASEDIR, tempdir, features_2D
from deeptrain import TrainGenerator, SimpleBatchgen
from deeptrain.callbacks import TraingenLogger, make_callbacks


batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')
logger_savedir = os.path.join(BASEDIR, 'tests', '_outputs', '_logger_outs')

MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=[32, 64],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=128,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    dynamic_predict_threshold_min_max=(.35, .95),
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    superbatch_dir=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    data_category='image',
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    data_category='image',
    shuffle=False,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load')}


def _make_logger_cb():
    log_configs = {
        'weights': ['conv2d'],
        'outputs': ['conv2d'],
        'gradients': ['conv2d'],
        'outputs-kw': dict(learning_phase=0),
        'gradients-kw': dict(learning_phase=0),
    }
    callbacks_init = {
        'logger': lambda cls: TraingenLogger(cls, logger_savedir, log_configs)
        }
    save_fn = lambda cls: TraingenLogger.save(cls, _id=cls.tg.epoch)
    callbacks = {
        'logger':
            {'save': (save_fn, TraingenLogger.clear),
             'load': (TraingenLogger.clear, TraingenLogger.load),
             'train:epoch': TraingenLogger.log,
            }
    }
    return callbacks, callbacks_init


def _make_2Dviz_cb():
    class Viz2D():
        def __init__(self, traingen):
            self.tg=traingen

        def viz(self):
            data = self._get_data()
            features_2D(data, tight=True, title_mode=False, cmap='hot',
                        norm=None, show_xy_ticks=[0,0], w=1.1, h=.55, n_rows=4)

        def _get_data(self):
            lg = self.tg._callback_objs['logger']
            last_key = list(lg.outputs.keys())[-1]
            outs = list(lg.outputs[last_key][0].values())[0]
            return outs[0].T

    callbacks_init = {'viz_2d': lambda cls: Viz2D(cls)}
    callbacks = {'viz_2d': {('on_val_end', 'train:epoch'): Viz2D.viz}}
    return callbacks, callbacks_init


def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']), tempdir(logger_savedir):
        cb_makers = [_make_logger_cb, _make_2Dviz_cb]
        callbacks, callbacks_init = make_callbacks(cb_makers)
        C['traingen'].update({'callbacks': callbacks,
                              'callbacks_init': callbacks_init})
        tg = _init_session(C)
        tg.train()
        _test_load(tg, C)

    print("\nTime elapsed: {:.3f}".format(time() - t0))
    _notify('main')


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
    _notify('load')


def _make_model(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'metrics', 'optimizer',
                       'num_classes', 'filters', 'kernel_size',
                       'dropout', 'dense_units')
        return [kw[key] for key in expected_kw]

    (batch_shape, loss, metrics, optimizer, num_classes, filters,
     kernel_size, dropout, dense_units) = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = ipt

    for f, ks in zip(filters, kernel_size):
        x = Conv2D(f, ks, activation='relu', padding='same')(x)

    x   = MaxPooling2D(pool_size=(2, 2))(x)
    x   = Dropout(dropout[0])(x)
    x   = Flatten()(x)
    x   = Dense(dense_units, activation='relu')(x)

    x   = Dropout(dropout[1])(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def _init_session(C, weights_path=None, loadpath=None):
    model = _make_model(weights_path, **C['model'])
    dg  = SimpleBatchgen(**C['datagen'])
    vdg = SimpleBatchgen(**C['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath, **C['traingen'])
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


def _notify(name):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        test_name = Path(__file__).stem.replace('_', ' ').upper()
        cprint(f"<< {test_name} PASSED >>\n", 'green')


if __name__ == '__main__':
    os.environ['IS_MAIN'] = '1'
    pytest.main([__file__, "-s"])
