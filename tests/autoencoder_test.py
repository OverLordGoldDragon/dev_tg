# -*- coding: utf-8 -*-
import os
import pytest

from pathlib import Path
from termcolor import cprint
from time import time

from .backend import Input, Conv2D, UpSampling2D
from .backend import Model
from .backend import BASEDIR, tempdir
from deeptrain import TrainGenerator, SimpleBatchgen


batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'examples', 'data', 'image')

MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,#['mape'],
    optimizer='adam',
    num_classes=10,
    activation=['relu']*4 + ['sigmoid'],
    filters=[2, 2, 1, 2, 1],
    kernel_size=[(3, 3)]*5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
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
    epochs=2,
    val_freq={'epoch': 1},
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
    input_as_labels=True,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG, 
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}


def test_main():
    t0 = time()
    with tempdir(CONFIGS['traignen']['logs_dir']), tempdir(
            CONFIGS['traingen']['best_models_dir']):
        _test_main()
    
    print("\nTime elapsed: {:.3f}".format(time() - t0))
    cprint("<< AUTOENCODER TEST PASSED >>\n", 'green')


def _test_main():
    tg = _init_session(CONFIGS)
    tg.train()
    
    _test_load(tg, CONFIGS)


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
                       'activation', 'filters', 'kernel_size', 'strides', 
                       'up_sampling_2d')
        return [kw[key] for key in expected_kw]

    (batch_shape, loss, metrics, optimizer, activation, filters, kernel_size,
     strides, up_sampling_2d) = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = ipt
    
    configs = (activation, filters, kernel_size, strides, up_sampling_2d)
    for act, f, ks, s, ups in zip(*configs):
        if ups is not None:
            x = UpSampling2D(ups)(x)
        x = Conv2D(f, ks, strides=s, activation=act, padding='same')(x)
    out = x

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


if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
