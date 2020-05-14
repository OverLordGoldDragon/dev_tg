# -*- coding: utf-8 -*-
import os
import pytest

from pathlib import Path
from time import time
from copy import deepcopy

from tests.backend import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tests.backend import Model
from tests.backend import BASEDIR, tempdir, notify
from deeptrain.util.logging import _log_init_state
from deeptrain.util.misc import pass_on_error
from deeptrain import TrainGenerator, DataGenerator


batch_size = None
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
    superbatch_dir=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    dynamic_predict_threshold_min_max=(.35, .95),
    logs_use_full_model_name=False,
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in ('main', 'load', 'checkpoint')}


@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        C['traingen'].update(dict(
            val_freq={'batch': 20},
            plot_history_freq={'val': 2},
            unique_checkpoint_freq={'val': 2},
            optimizer_save_configs={'include': ['updates', 'crowbar']},
            max_one_best_save=True,
            max_checkpoints=3,
            ))
        tg = _init_session(C)
        tg.train()
        _test_load(tg, C)

        C['traingen'].update(dict(
            val_freq={'iter': 20},
            temp_checkpoint_freq={'val': 3},
            optimizer_save_configs={'exclude': ['iterations']},
            optimizer_load_configs={'include': ['momentum', 'momentam']},
            ))
        tg = _init_session(C)
        with tempdir() as savedir:
            _log_init_state(tg, savedir=savedir, verbose=1)
        tg.train()
        _test_load(tg, C)

        tg = _init_session(C)
        tg.train()
        tg._has_trained = True
        tg.train()

        tg.plot_configs['1']['vhlines']['v'] = 'invalid_vlines'
        pass_on_error(tg._get_history_fig)

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


@notify(tests_done)
def test_checkpoint():
    def _get_nfiles(logdir):
        return len([f for f in Path(logdir).iterdir()
                    if f.is_file() and not f.name[0] == '.'])  # omit dir & hidden

    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']
                 ), tempdir(C['traingen']['best_models_dir']):
        C['traingen']['max_checkpoints'] = 2
        tg = _init_session(C)
        tg.train()

        nfiles_1 = _get_nfiles(tg.logdir)
        tg.checkpoint(forced=True, overwrite=True)
        nfiles_2 = _get_nfiles(tg.logdir)
        assert (nfiles_2 == nfiles_1), (
            "Number of files in `logdir` changed with `overwrite`==True on "
            "second checkpoint w/ `max_checkpoints`==2 ({} -> {})".format(
                nfiles_1, nfiles_2))

        tg.checkpoint(forced=True, overwrite=False)
        nfiles_3 = _get_nfiles(tg.logdir)
        assert (nfiles_3 == 2 * nfiles_2), (
            "Number of files didn't double in `logdir` after checkpointing "
            "below `max_checkpoints` checkpoints ({} -> {})".format(
                nfiles_2, nfiles_3))

        tg.checkpoint(forced=True, overwrite=False)
        nfiles_4 = _get_nfiles(tg.logdir)
        assert (nfiles_3 == nfiles_4), (
            "Number of files changed in `logdir` after checkpointing at "
            "`max_checkpoints` checkpoints ({} -> {})".format(
                nfiles_3, nfiles_4))

        tg.max_checkpoints = 0
        tg.checkpoint(forced=True, overwrite=False)
        nfiles_5 = _get_nfiles(tg.logdir)
        assert (nfiles_5 == nfiles_1), (
            "`max_checkpoints`==0 should behave like ==1, but number of "
            "files in `logdir` differs from that in first checkpoint "
            "({} -> {})".format(nfiles_1, nfiles_5))


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
    dg  = DataGenerator(**C['datagen'])
    vdg = DataGenerator(**C['val_datagen'])
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


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
