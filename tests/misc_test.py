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

from pathlib import Path
from time import time
from copy import deepcopy

from backend import CL_CONFIGS, BASEDIR, tempdir, notify, make_classifier
from backend import _init_session, _do_test_load, _get_test_names
from deeptrain.util.logging import _log_init_state
from deeptrain.util.misc import pass_on_error
from deeptrain import metrics


#### CONFIGURE TESTING #######################################################
batch_size = None
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

tests_done = {}
CONFIGS = deepcopy(CL_CONFIGS)
CONFIGS['model']['batch_shape'] = (batch_size, width, height, channels)
CONFIGS['datagen']['batch_size'] = batch_size
CONFIGS['val_datagen']['batch_size'] = batch_size
CONFIGS['traingen']['logs_use_full_model_name'] = False

classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)
###############################################################################

@notify(tests_done)
def test_main(monkeypatch):
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen'].update(dict(
            val_freq={'batch': 20},
            plot_history_freq={'val': 2},
            unique_checkpoint_freq={'val': 2},
            optimizer_save_configs={'include': ['updates', 'crowbar']},
            max_one_best_save=True,
            max_checkpoints=3,
            ))
        tg = init_session(C, model=classifier)
        tg.train()
        _test_load(tg, C)

        C['traingen'].update(dict(
            val_freq={'iter': 20},
            temp_checkpoint_freq={'val': 3},
            optimizer_save_configs={'exclude': ['iterations']},
            optimizer_load_configs={'include': ['momentum', 'momentam']},
            eval_fn='predict',
            key_metric='catco_custom',
            custom_metrics={'catco_custom': metrics.categorical_crossentropy},
            ))
        tg = init_session(C, model=classifier)
        with tempdir() as savedir:
            _log_init_state(tg, savedir=savedir, verbose=1)
        tg.train()
        _test_load(tg, C)

        tg = init_session(C, model=classifier)
        tg.train()
        tg._has_trained = True
        tg.train()

        tg.plot_configs[0]['vhlines']['v'] = 'invalid_vlines'
        pass_on_error(tg.get_history_fig)
        tg.clear_cache(reset_val_flags=True)
        tg._should_do({}, forced=True)

        monkeypatch.setattr('builtins.input', lambda x: 'y')
        tg.destroy(confirm=False)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_checkpoint():
    def _get_nfiles(logdir):
        return len([f for f in Path(logdir).iterdir()
                    if f.is_file() and not f.name[0] == '.'])  # omit dir & hidden

    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen']['max_checkpoints'] = 2
        tg = init_session(C, model=classifier)
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


@notify(tests_done)
def test_custom_metrics():
    C = deepcopy(CONFIGS)
    def f05_score(y_true, y_pred, pred_threshold=.5):
        return metrics.f1_score(y_true, y_pred, pred_threshold, beta=.5)

    def _test_eval_mode():
        with tempdir(C['traingen']['logs_dir']), \
            tempdir(C['traingen']['best_models_dir']):
            C['traingen']['custom_metrics'] = {'f.5_score': f05_score}
            C['traingen']['val_metrics'] = ['*', 'f.5_score']
            C['traingen']['eval_fn'] = 'evaluate'
            tg = init_session(C, model=classifier)

            # should be dropped in _validate_traingen_configs:_validate_metrics
            assert 'f.5 score' not in tg.val_metrics
            tg.train()

    def _test_predict_mode():
        with tempdir(C['traingen']['logs_dir']), \
            tempdir(C['traingen']['best_models_dir']):
            C['traingen']['custom_metrics'] = {'f.5_score': f05_score}
            C['traingen']['val_metrics'] = ['*', 'f.5_score']
            C['traingen']['eval_fn'] = 'predict'
            tg = init_session(C, model=classifier)

            # model metrics should be inserted at wildcard
            assert tg.val_metrics[-1] == 'f.5_score'
            tg.train()

    _test_eval_mode()
    _test_predict_mode()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
