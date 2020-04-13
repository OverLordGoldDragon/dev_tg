# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from unittest.mock import patch
from pathlib import Path
from termcolor import cprint
from time import time
from copy import deepcopy

from tests.backend import Input, Conv2D, UpSampling2D
from tests.backend import Model
from tests.backend import BASEDIR, tempdir
from deeptrain import util
from deeptrain.util.misc import pass_on_error
from deeptrain import TrainGenerator, SimpleBatchgen


batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

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
    epochs=1,
    val_freq={'epoch': 1},
    input_as_labels=True,
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
          'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {f'{name}_exceptions': None for name in ('datagen', 'util')}


def test_datagen():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = _init_session(C)
        tg.train()

        dg = tg.datagen
        dg.advance_batch()
        dg.batch = dg.batch[:1]
        dg.batch_loaded = False
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
    _notify('datagen_exceptions', tests_done)

def test_util():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        configs_orig = deepcopy(C)
        # _validate_savelist() + _validate_metrics() [util.misc]
        C['traingen']['savelist'] = ['{labels}']
        C['traingen']['train_metrics'] = ('loss',)
        tg = _init_session(C)

        # save_best_model() [util.saving]
        tg.train()
        with patch('os.remove') as mock_remove:
            mock_remove.side_effect = OSError('Permission Denied')
            util.saving.save_best_model(tg, del_previous_best=True)

        # _update_temp_history() [util.training]
        tg.val_temp_history['loss'] = (1, 2, 3)
        util.training._update_temp_history(tg, metrics=(4,), val=True)
        tg.val_temp_history['loss'] = []
        util.training._update_temp_history(tg, metrics=(4,), val=True)
        tg.temp_history['binary_accuracy'] = []
        pass_on_error(util.training._update_temp_history,
                      tg, metrics=dict(a=1, b=2), val=False)
        tg.temp_history['f1_score'] = []
        pass_on_error(util.training._update_temp_history,
                      tg, metrics=[1, 2], val=False)

        # _get_sample_weight() [util.training]
        labels = np.random.randint(0, 2, (32, 3))
        tg.class_weights = {0: 1, 1: 2, 2: 3}
        util.training._get_sample_weight(tg, labels)

        # _get_api_metric_name() [util.training]
        util.training._get_api_metric_name(
            'accuracy', 'categorical_crossentropy')
        util.training._get_api_metric_name(
            'acc', 'sparse_categorical_crossentropy')
        util.training._get_api_metric_name('acc', 'binary_crossentropy')

        # _validate_weighted_slices_range() [util.misc]
        tg.datagen.slices_per_batch = None
        util.misc._validate_traingen_configs(tg)  ##
        del tg
        C['traingen']['max_is_best'] = True  # elsewhere
        C['traingen']['pred_weighted_slices_range'] = (.1, 1.1)
        C['traingen']['eval_fn_name'] = 'evaluate'
        pass_on_error(_init_session, C)  ##
        C['traingen']['eval_fn_name'] = 'predict'
        pass_on_error(_init_session, C)  ##

        # _validate_directories() [util.misc]
        C = deepcopy(configs_orig)
        C['traingen']['best_models_dir'] = None
        pass_on_error(_init_session, C)
        C = deepcopy(configs_orig)
        C['traingen']['logs_dir'] = None
        pass_on_error(_init_session, C)
        C = deepcopy(configs_orig)
        C['traingen']['best_models_dir'] = None
        C['traingen']['logs_dir'] = None
        pass_on_error(_init_session, C)

        # _validate_optimizer_saving_configs() [util.misc]
        C = deepcopy(configs_orig)
        C['traingen']['optimizer_save_configs'] = {
            'include': 'weights', 'exclude': 'updates'}
        pass_on_error(_init_session, C)

        # _validate_class_weights() [util.misc]
        C = deepcopy(configs_orig)
        C['traingen']['class_weights'] = {'0': 1, 1: 2}
        pass_on_error(_init_session, C)
        C['traingen']['class_weights'] = {0: 1}
        pass_on_error(_init_session, C)

        # _validate_best_subset_size() [util.misc]
        C = deepcopy(configs_orig)
        C['traingen']['best_subset_size'] = 5
        C['val_datagen']['shuffle_group_samples'] = True
        pass_on_error(_init_session, C)

    print("\nTime elapsed: {:.3f}".format(time() - t0))
    _notify('util_exceptions', tests_done)


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


def _init_session(C, weights_path=None, loadpath=None):
    model = _make_model(weights_path, **C['model'])
    dg  = SimpleBatchgen(**C['datagen'])
    vdg = SimpleBatchgen(**C['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath,
                         **C['traingen'])
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


def _notify(name, tests_done):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        cprint("<< EXCEPTIONS TEST PASSED >>\n", 'green')


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
