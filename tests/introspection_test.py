# -*- coding: utf-8 -*-
import os
import pytest

from copy import deepcopy

from tests.backend import Input, Conv2D, UpSampling2D
from tests.backend import Model
from tests.backend import Adam
from tests.backend import BASEDIR, notify
from deeptrain import introspection
from deeptrain import TrainGenerator, SimpleBatchgen


batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,
    optimizer='adam',
    num_classes=10,
    activation=['relu'] * 4 + ['sigmoid'],
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
    epochs=1,
    val_freq={'epoch': 1},
    input_as_labels=True,
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {name: None for name in
              ('gather_over_dataset', 'print_dead_nan')}


@notify(tests_done)
def test_gather_over_dataset():
    C = deepcopy(CONFIGS)
    tg = _init_session(C)
    tg.train()

    introspection.gradient_norm_over_dataset(tg, n_iters=5)
    introspection.gradients_sum_over_dataset(tg, n_iters=5)


@notify(tests_done)
def test_print_dead_nan():
    def _test_print_nan_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e50)
        tg = _init_session(C)
        tg.train()
        tg.check_health()

    def _test_print_dead_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e-4)
        tg = _init_session(C)
        tg.train()
        tg.check_health(dead_threshold=.1)
        tg.check_health(notify_detected_only=False)

    _test_print_nan_weights()
    _test_print_dead_weights()


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


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
