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
from backend import CL_CONFIGS
from see_rnn import get_weights, features_2D
from deeptrain.callbacks import TraingenCallback, TraingenLogger
from deeptrain.callbacks import make_layer_hists_cb


#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(CL_CONFIGS)
batch_size, width, height, channels = CONFIGS['model']['batch_shape']
logger_savedir = os.path.join(BASEDIR, 'tests', '_outputs', '_logger_outs')

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

    def on_save(self, stage=None):
        self.save(_id=self.tg.epoch)

    def on_load(self, stage=None):
        self.clear()
        self.load()

    def on_train_epoch_end(self, stage=None):
        self.log()

    TGL = TraingenLogger
    TGL.on_save = on_save
    TGL.on_load = on_load
    TGL.on_train_epoch_end = on_train_epoch_end
    tgl = TGL(logger_savedir, log_configs,
              get_data_fn=get_data_fn,
              get_labels_fn=get_labels_fn,
              gather_fns=gather_fns)
    return tgl


def _make_2Dviz_cb():
    class Viz2D(TraingenCallback):
        def __init__(self):
            pass

        def init_with_traingen(self, traingen):
            self.tg=traingen

        def on_val_end(self, stage=None):
            if stage == ('val_end', 'train:epoch'):
                self.viz()

        def viz(self):
            data = self._get_data()
            features_2D(data, tight=True, title_mode=False, cmap='hot',
                        norm=None, show_xy_ticks=[0,0], w=1.1, h=.55, n_rows=4)

        def _get_data(self):
            lg = None
            for cb in self.tg.callbacks:
                if getattr(cb.__class__, '__name__', '') == 'TraingenLogger':
                    lg = cb
            if lg is None:
                raise Exception("TraingenLogger not found in `callbacks`")

            last_key = list(lg.outputs.keys())[-1]
            outs = list(lg.outputs[last_key][0].values())[0]
            return outs[0].T

    viz2d = Viz2D()
    return viz2d


layer_hists_cbs = [
    {'train:epoch': [make_layer_hists_cb(mode='gradients:outputs'),
                     make_layer_hists_cb(mode='gradients:weights')]
     },
    # two dicts can be used to organize code, but will execute same as one
    {('val_end', 'train:epoch'): make_layer_hists_cb(mode='weights'),
     'val_end': make_layer_hists_cb(mode='outputs',
                                    configs={'title': dict(fontsize=13),
                                             'plot': dict(annot_kw=None)},
    )},
]
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']), \
            tempdir(logger_savedir):
        callbacks = [_make_logger_cb(), _make_2Dviz_cb(), *layer_hists_cbs]
        C['traingen']['callbacks'] = callbacks

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
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']), \
            tempdir(logger_savedir):
        batch_shape = (batch_size, width, height, channels)

        n_classes = C['model']['num_classes']
        callbacks = [_make_logger_cb(
            get_data_fn=lambda: np.random.randn(*batch_shape),
            get_labels_fn=lambda: np.random.randint(0, 2,
                                                    (batch_size, n_classes)),
            gather_fns={'weights': get_weights},
            )]
        C['traingen']['callbacks'] = callbacks
        tg = init_session(C, model=model)
        tg.train()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
