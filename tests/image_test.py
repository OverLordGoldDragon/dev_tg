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

from time import time
from copy import deepcopy

from backend import CL_CONFIGS, tempdir, notify, make_classifier
from backend import _init_session, _do_test_load, _get_test_names


#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(CL_CONFIGS)
batch_size, width, height, channels = CONFIGS['model']['batch_shape']

classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen']['epochs'] = 2
        C['traingen']['final_fig_dir'] = C['traingen']['best_models_dir']
        _test_main(C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


def _test_main(C, new_model=False):
    if new_model:
        tg = init_session(C)
    else:
        tg = init_session(C, model=classifier)
    tg.train()
    _test_load(tg, C)


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_predict():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen']['eval_fn'] = 'predict'
        # tests misc._validate_traingen_configs
        C['traingen']['val_metrics'] = ['loss', 'acc']
        _test_main(C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_group_batch():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        for name in ('traingen', 'datagen', 'val_datagen'):
            C[name]['batch_size'] = 64
        C['model']['batch_shape'] = (64, width, height, channels)
        _test_main(C, new_model=True)

    print("\nTime elapsed: {:.3f}".format(time() - t0))



@notify(tests_done)
def test_recursive_batch():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        for name in ('traingen', 'datagen', 'val_datagen'):
            C[name]['batch_size'] = 256
        C['model']['batch_shape'] = (256, width, height, channels)
        _test_main(C, new_model=True)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
