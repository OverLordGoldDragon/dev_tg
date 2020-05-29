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
import numpy as np

from time import time
from copy import deepcopy

from backend import BASEDIR, tempdir, notify, make_classifier
from backend import _init_session, _do_test_load, _get_test_names
from see_rnn import get_weights, features_2D
from deeptrain.callbacks import TraingenLogger, make_callbacks
from deeptrain.callbacks import make_layer_hists_cb


#### CONFIGURE TESTING #######################################################
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
    filters=[8, 16],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=32,
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
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {}
model = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)


def _make_logger_cb(get_data_fn=None, get_labels_fn=None, gather_fns=None):
    log_configs = {
        'weights': ['conv2d'],
        'outputs': 'conv2d',
        'gradients': ('conv2d',),
        'outputs-kw': dict(learning_phase=0),
        'gradients-kw': dict(learning_phase=0),
    }
    callbacks_init = {
        'logger': lambda self: TraingenLogger(self, logger_savedir, log_configs,
                                              get_data_fn=get_data_fn,
                                              get_labels_fn=get_labels_fn,
                                              gather_fns=gather_fns)
        }
    save_fn = lambda self: TraingenLogger.save(self, _id=self.tg.epoch)
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
            lg = self.tg.callback_objs['logger']
            last_key = list(lg.outputs.keys())[-1]
            outs = list(lg.outputs[last_key][0].values())[0]
            return outs[0].T

    callbacks_init = {'viz_2d': lambda self: Viz2D(self)}
    callbacks = {'viz_2d': {('val_end', 'train:epoch'): Viz2D.viz}}
    return callbacks, callbacks_init


layer_hists_cbs = {
    'lhgo': {'val_end': make_layer_hists_cb(mode='gradients:outputs')},
    'lhgw': {'val_end': make_layer_hists_cb(mode='gradients:weights')},
    'lho':  {'val_end': make_layer_hists_cb(mode='weights')},
    'lhw':  {'val_end': make_layer_hists_cb(mode='outputs',
                                            configs={'title': dict(fontsize=13),
                                                     'plot': dict(annot_kw=None)},
                                            )},
}
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']), tempdir(logger_savedir):
        cb_makers = [_make_logger_cb, _make_2Dviz_cb]
        callbacks, callbacks_init = make_callbacks(cb_makers)
        callbacks.update(layer_hists_cbs)

        C['traingen']['callbacks'] = callbacks
        C['traingen']['callbacks_init'] = callbacks_init

        tg = init_session(C, model=model)
        tg.train()
        _test_load(tg, C)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_traingen_logger():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']), tempdir(logger_savedir):
        batch_shape = (batch_size, width, height, channels)

        cb_makers = [lambda: _make_logger_cb(
            get_data_fn=lambda: np.random.randn(*batch_shape),
            get_labels_fn=lambda: np.random.randint(
                0, 2, (batch_size, C['model']['num_classes'])),
            gather_fns={'weights': get_weights},
            )]
        callbacks, callbacks_init = make_callbacks(cb_makers)

        C['traingen'].update({'callbacks': callbacks,
                              'callbacks_init': callbacks_init})
        tg = init_session(C, model=model)
        tg.train()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
