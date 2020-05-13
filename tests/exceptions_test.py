# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np


from unittest.mock import patch
from pathlib import Path
from time import time
from copy import deepcopy

from tests.backend import Input, Conv2D, UpSampling2D
from tests.backend import Dense, LSTM
from tests.backend import l2
from tests.backend import Model
from tests.backend import BASEDIR, tempdir, notify
from deeptrain import util
from deeptrain import metrics
from deeptrain import preprocessing
from deeptrain.util.misc import pass_on_error
from deeptrain.visuals import layer_hists
from deeptrain import TrainGenerator, DataGenerator


batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

AE_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,
    optimizer='adam',
    num_classes=10,
    activation=['relu']*4 + ['sigmoid'],
    filters=[2, 2, 1, 2, 1],
    kernel_size=[(3, 3)]*5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
)
CL_CFG = dict(
    batch_shape=(batch_size, 25, 16),
    units=16,
    optimizer='adam',
    loss='binary_crossentropy'
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
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=AE_CFG,
)

CONFIGS = {'model': AE_CFG, 'datagen': DATAGEN_CFG,
          'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {f'{name}_exceptions': None for name in (
    'datagen', 'visuals', 'util', 'data_to_hdf5')}


@notify(tests_done)
def test_datagen():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = _init_session(C, _make_autoencoder)
        tg.train()

        dg = tg.datagen
        dg.advance_batch()
        dg.batch = dg.batch[:1]
        dg.batch_loaded = False
        pass_on_error(dg.advance_batch)
        dg.batch_loaded = True
        dg.advance_batch(forced=False)

        dg.shuffle = True
        dg.all_data_exhausted = True
        dg._validate_batch()

        dg.batch = []
        dg.batch_exhausted = True
        dg._validate_batch()

        dg.set_nums_to_process = dg.set_nums_original.copy()
        pass_on_error(dg._set_class_params, ['99', '100'], ['100', '101'])
        pass_on_error(dg._set_class_params, ['1', '2'], ['100', '101'])
        dg.superbatch_dir = None
        pass_on_error(dg._set_class_params, ['1', '2'], ['1', '2'])

        dg._set_preprocessor(None, {})
        pass_on_error(dg._set_preprocessor, "x", {})

        pass_on_error(dg._infer_data_info, dg.data_dir,
                      data_loader="x")
        dg._infer_data_info(dg.data_dir, data_loader="hdf5")

        C['datagen']['invalid_kwarg'] = 5
        pass_on_error(DataGenerator, **C['datagen'])

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_visuals():
    def _layer_hists(model):
        pass_on_error(layer_hists, model, '*', mode='gradients')
        pass_on_error(layer_hists, model, '*', mode='outputs')
        pass_on_error(layer_hists, model, '*', mode='skeletons')

    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = _init_session(C, _make_autoencoder)
        model = tg.model

        _layer_hists(model)


@notify(tests_done)
def test_util():
    t0 = time()

    def _util_make_autoencoder(C):
        C['model'] = AE_CFG
        C['traingen']['model_configs'] = AE_CFG
        C['traingen']['input_as_labels'] = True
        return _init_session(C, _make_autoencoder)

    def _util_make_classifier(C):
        C['model'] = CL_CFG
        C['traingen']['model_configs'] = CL_CFG
        C['traingen']['input_as_labels'] = False
        return _init_session(C, _make_classifier)

    def _save_best_model(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.train()
        with patch('os.remove') as mock_remove:
            mock_remove.side_effect = OSError('Permission Denied')
            util.saving._save_best_model(tg, del_previous_best=True)
        with patch('deeptrain.train_generator.TrainGenerator.generate_report'
                   ) as mock_report:
            mock_report.side_effect = Exception()
            util.saving._save_best_model(tg)

    def checkpoint_model(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.train()
        tg.max_checkpoint_saves = -1
        with patch('os.remove') as mock_remove:
            mock_remove.side_effect = OSError('Permission Denied')
            util.saving.checkpoint_model(tg)

        tg.logdir = None
        pass_on_error(util.saving.checkpoint_model, tg)

    def save(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.model.loss = 'mean_squared_error'
        tg.train()
        tg.datagen.set_nums_to_process = [9001]
        tg.final_fig_dir = tg.logdir

        pass_on_error(tg.load)
        tg.save()
        tg._save_history()
        tg._save_history()
        tg.optimizer_load_configs = {'exclude': ['weights']}
        tg.load()

        tg._history_fig = 1
        tg._save_history()

        tg.use_passed_dirs_over_loaded = True
        tg.load()

        tg.optimizer_save_configs = {'include': []}
        tg.save()

        with patch('tests.backend.K.get_value') as mock_get_value:
            mock_get_value.side_effect = Exception()
            tg.save()

        tg.optimizer_save_configs = {'include': ['leaking_rate']}
        tg.datagen.group_batch = []
        with patch('pickle.dump') as mock_dump:
            mock_dump.side_effect = Exception()
            tg.save()

        tg.logdir = 'abc'
        pass_on_error(tg.load)
        tg.logdir = None
        pass_on_error(tg.load)

    def _get_sample_weight(C):  # [util.training]
        tg = _util_make_autoencoder(C)
        labels = np.random.randint(0, 2, (32, 3))
        tg.class_weights = {0: 1, 1: 2, 2: 3}
        util.training._get_sample_weight(tg, labels)

    def _get_api_metric_name(C):  # [util.training]
        util.training._get_api_metric_name(
            'accuracy', 'categorical_crossentropy')
        util.training._get_api_metric_name(
            'acc', 'sparse_categorical_crossentropy')
        util.training._get_api_metric_name('acc', 'binary_crossentropy')

    def _validate_weighted_slices_range(C):  # [util.misc]
        C['traingen']['pred_weighted_slices_range'] = (.5, 1.5)
        C['traingen']['eval_fn_name'] = 'evaluate'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        tg = _util_make_autoencoder(C)
        tg.pred_weighted_slices_range = (.5, 1.5)
        tg.eval_fn_name = 'predict'
        tg.datagen.slices_per_batch = None
        tg.val_datagen.slices_per_batch = None
        pass_on_error(util.misc._validate_traingen_configs, tg)

        C['traingen']['max_is_best'] = True
        C['traingen']['eval_fn_name'] = 'evaluate'
        C['traingen']['pred_weighted_slices_range'] = (.1, 1.1)
        pass_on_error(_util_make_classifier, C)

        C['traingen']['eval_fn_name'] = 'predict'
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['datagen'].pop('slices_per_batch', None)
        pass_on_error(_util_make_classifier, C)

    def _get_best_subset_val_history(C):  # [util.training]
        C['traingen']['best_subset_size'] = 2
        tg = _util_make_classifier(C)

        tg.val_datagen.slices_per_batch = 4
        tg._labels_cache = np.random.randint(0, 2, (3, 4, batch_size, 1))
        tg._preds_cache = np.random.uniform(0, 1, (3, 4, batch_size, 1))
        tg._sw_cache = np.random.randint(0, 2, (3, 4, batch_size, 1))
        tg._class_labels_cache = tg._labels_cache.copy()
        tg._val_set_name_cache = ['1', '2', '3']
        tg.key_metric = 'f1_score'
        tg.val_temp_history = {'f1_score': []}
        tg.key_metric_fn = metrics.f1_score
        tg.eval_fn_name = 'predict'
        tg.dynamic_predict_threshold_min_max = None

        util.training._get_best_subset_val_history(tg)

        tg.eval_fn_name = 'superfit'
        pass_on_error(util.training._get_best_subset_val_history, tg)

    def _update_temp_history(C):  # [util.training]
        tg = _util_make_classifier(C)

        tg.val_temp_history['loss'] = (1, 2, 3)
        util.training._update_temp_history(tg, metrics=(4,), val=True)
        tg.val_temp_history['loss'] = []
        util.training._update_temp_history(tg, metrics=(4,), val=True)

        tg.datagen.slice_idx = 1
        tg.datagen.slices_per_batch = 2
        tg.temp_history = {'binary_accuracy': []}
        tg.train_metrics = ['binary_accuracy']
        pass_on_error(util.training._update_temp_history,
                      tg, metrics=[1], val=False)

        pass_on_error(util.training._update_temp_history,
                      tg, metrics=[dict(a=1, b=2)], val=False)

        util.training._update_temp_history(tg, [[1]], val=False)

        tg.temp_history = {'f1_score': []}
        tg.train_metrics = ['f1_score']
        pass_on_error(util.training._update_temp_history,
                      tg, metrics=[[1, 2]], val=False)

    def _validate_metrics(C):  # [util.misc]
        C['traingen']['eval_fn_name'] = 'evaluate'
        C['traingen']['key_metric'] = 'hinge'
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['val_metrics'] = 'goblin'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        C['traingen']['key_metric'] = 'swordfish'
        C['traingen']['key_metric_fn'] = None
        C['traingen']['eval_fn_name'] = 'predict'
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['key_metric'] = 'loss'
        C['traingen']['max_is_best'] = True
        _util_make_autoencoder(C)

        C = deepcopy(CONFIGS)
        C['traingen']['eval_fn_name'] = 'predict'
        C['traingen']['train_metrics'] = None
        C['traingen']['val_metrics'] = 'cosine_proximity'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        C['traingen']['eval_fn_name'] = 'predict'
        tg = _util_make_autoencoder(C)
        tg.model.loss = 'hl2'
        pass_on_error(util.misc._validate_traingen_configs, tg)

        tg.train_metrics = ['tnr', 'tpr']
        tg.val_metrics = ['tnr', 'tpr']
        tg.key_metric = 'tnr'
        pass_on_error(util.misc._validate_traingen_configs, tg)


    def _validate_directories(C):  # [util.misc]
        C['traingen']['best_models_dir'] = None
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['traingen']['logs_dir'] = None
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['traingen']['best_models_dir'] = None
        C['traingen']['logs_dir'] = None
        pass_on_error(_util_make_classifier, C)

    def _validate_optimizer_saving_configs(C):  # [util.misc]
        C['traingen']['optimizer_save_configs'] = {
            'include': 'weights', 'exclude': 'updates'}
        pass_on_error(_util_make_classifier, C)

    def _validate_class_weights(C):  # [util.misc]
        C['traingen']['class_weights'] = {'0': 1, 1: 2}
        pass_on_error(_util_make_classifier, C)

        C['traingen']['class_weights'] = {0: 1}
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        tg = _util_make_classifier(C)
        tg.model.loss = 'categorical_crossentropy'
        tg.class_weights = {0: 1, 2: 5, 3: 6}
        util.misc._validate_traingen_configs(tg)

    def _validate_best_subset_size(C):  # [util.misc]
        C['traingen']['best_subset_size'] = 5
        C['val_datagen']['shuffle_group_samples'] = True
        pass_on_error(_util_make_classifier, C)

    def _validate_metric_printskip_configs(C):  # [util.misc]
        C['traingen']['metric_printskip_configs'] = {'val': ('loss',)}
        _util_make_autoencoder(C)

    def _validate_savelist_and_metrics(C):  # [util.misc]
        C['traingen']['savelist'] = ['{labels}']
        C['traingen']['train_metrics'] = ('loss',)
        pass_on_error(_util_make_autoencoder, C)

    tests_all = [_save_best_model,
                  checkpoint_model,
                  save,
                  _get_sample_weight,
                  _get_api_metric_name,
                  _validate_weighted_slices_range,
                  _get_best_subset_val_history,
                  _update_temp_history,
                  _validate_metrics,
                  _validate_directories,
                  _validate_optimizer_saving_configs,
                  _validate_class_weights,
                  _validate_best_subset_size,
                  _validate_metric_printskip_configs,
                  _validate_savelist_and_metrics,
                  ]
    for _test in tests_all:
        with tempdir(CONFIGS['traingen']['logs_dir']), tempdir(
                CONFIGS['traingen']['best_models_dir']):
            C = deepcopy(CONFIGS)  # reset dict
            _test(C)
            print("Passed", _test.__name__)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_data_to_hdf5(monkeypatch):  # [util.preprocessing]
    """Dedicated test since it uses monkeypatch"""
    C = deepcopy(CONFIGS)
    # set preemptively in case data.h5 somehow found in dir
    monkeypatch.setattr('builtins.input', lambda x: 'y')

    with tempdir(C['traingen']['logs_dir']) as loaddir:
        with open(os.path.join(loaddir, "data.txt"), 'w') as txt:
            txt.write("etc")
        savepath = os.path.join(loaddir, "data.h5")
        pass_on_error(preprocessing.data_to_hdf5, savepath.replace('.h5', ''),
                      batch_size=32, loaddir=loaddir)

        data = np.random.randn(1, 32, 100)
        np.save(os.path.join(loaddir, "data.npy"), data)
        pass_on_error(preprocessing.data_to_hdf5, savepath=savepath,
                      batch_size=32, loaddir=loaddir)

        kw = dict(savepath=savepath, data=data, batch_size=32)
        pass_on_error(preprocessing.data_to_hdf5, **kw)

        os.remove(os.path.join(loaddir, "data.txt"))
        preprocessing.data_to_hdf5(**kw)

        monkeypatch.setattr('builtins.input', lambda x: 'y')
        preprocessing.data_to_hdf5(**kw)

        monkeypatch.setattr('builtins.input', lambda x: 'n')
        pass_on_error(preprocessing.data_to_hdf5, **kw)

        preprocessing.data_to_hdf5(overwrite=True, **kw)

        pass_on_error(preprocessing.data_to_hdf5, overwrite=False, **kw)

        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], loaddir=None, data=None)

        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], loaddir=loaddir, data=data)

        _data = [data[0], data[0, :31]]
        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], data=_data, overwrite=True)

        _data = [np.vstack([data[0], data[0]])]
        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], data=_data, overwrite=True)


@notify(tests_done)
def _test_load(tg, C, make_model_fn):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    _destroy_session(tg)

    weights_path, loadpath = _get_latest_paths(logdir)
    tg = _init_session(C, make_model_fn, weights_path, loadpath)


def _make_autoencoder(weights_path=None, **kw):
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


def _make_classifier(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'units', 'optimizer')
        return [kw[key] for key in expected_kw]

    batch_shape, loss, units, optimizer = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, return_sequences=False, stateful=True,
               kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4),
               bias_regularizer=l2(1e-4))(ipt)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def _init_session(C, make_model_fn, weights_path=None, loadpath=None):
    model = make_model_fn(weights_path, **C['model'])
    dg  = DataGenerator(**C['datagen'])
    vdg = DataGenerator(**C['val_datagen'])
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


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
