# -*- coding: utf-8 -*-
import os
import pytest

from pathlib import Path
from termcolor import cprint
from time import time

from .backend import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from .backend import Model
from .backend import BASEDIR, tempdir
from deeptrain import TrainGenerator, SimpleBatchgen


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
    filters=[32, 64],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=128,
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
    superbatch_dir=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    data_category='image',
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG, 
          'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}


def test_datagen_exceptions():
    t0 = time()
    with tempdir(CONFIGS['traingen']['logs_dir']), tempdir(
            CONFIGS['traingen']['best_models_dir']):
        tg = _init_session(CONFIGS)
        tg.train()
        
        dg = tg.datagen
        dg.advance_batch()
        dg.batch = dg.batch[:1]
        _pass_on_fail(dg.advance_batch)
        dg.batch_loaded = True
        dg.advance_batch(forced=False)
        
        dg.shuffle = True        
        dg.all_data_exhausted = True
        dg._validate_batch()

        dg.batch = []
        dg.batch_exhausted = True        
        dg._validate_batch()
        
        dg.set_nums_to_process = dg.set_nums_original.copy()
        _pass_on_fail(dg._set_class_params, ['99', '100'], ['100', '101'])
        _pass_on_fail(dg._set_class_params, ['1', '2'], ['100', '101'])
        dg.superbatch_dir = None
        _pass_on_fail(dg._set_class_params, ['1', '2'], ['1', '2'])

        dg._set_preprocessor(None, {})
        dg._set_preprocessor("x", {})
        
        _pass_on_fail(dg._infer_and_get_data_info, dg.data_dir,
                      data_format="x")
        dg._infer_and_get_data_info(dg.data_dir, data_format="hdf5")
        

    print("\nTime elapsed: {:.3f}".format(time() - t0))
    cprint("<< IMAGE TEST PASSED >>\n", 'green')


def _test_load(tg, CONFIGS):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    _destroy_session(tg)

    weights_path, state_path = _get_latest_paths(logdir)
    tg = _init_session(CONFIGS, weights_path, state_path)
    print("\n>LOAD TEST PASSED")


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


def _init_session(CONFIGS, weights_path=None, state_path=None):
    if state_path is not None:
        CONFIGS['traingen']['loadpath'] = state_path

    model = _make_model(weights_path, **CONFIGS['model'])
    dg  = SimpleBatchgen(**CONFIGS['datagen'])
    vdg = SimpleBatchgen(**CONFIGS['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, **CONFIGS['traingen'])
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


def _pass_on_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print("Errmsg", e)


if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
