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

from copy import deepcopy

from backend import Adam
from backend import BASEDIR, notify, make_autoencoder
from backend import _init_session, _get_test_names


#### CONFIGURE TESTING #######################################################
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
    kernel_size=[(3, 3)] * 5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
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
    input_as_labels=True,
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {}
model = make_autoencoder(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_autoencoder)
###############################################################################

@notify(tests_done)
def test_gather_over_dataset():
    C = deepcopy(CONFIGS)
    tg = init_session(C, model=model)
    tg.train()

    tg.gradient_norm_over_dataset(n_iters=None, prog_freq=3)
    tg.gradients_sum_over_dataset(n_iters=5, prog_freq=3)

    x, y, sw = tg.get_data()
    tg.compute_gradients_norm(x, y, sw)  # not gather, but test anyway


@notify(tests_done)
def test_print_dead_nan():
    def _test_print_nan_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e9)
        tg = init_session(C)
        tg.train()
        tg.check_health()

    def _test_print_dead_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e-4)
        tg = init_session(C)
        tg.train()
        tg.check_health(dead_threshold=.1)
        tg.check_health(notify_detected_only=False)
        tg.check_health(notify_detected_only=False, dead_threshold=.5,
                        dead_notify_above_frac=2)

    _test_print_nan_weights()
    _test_print_dead_weights()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
