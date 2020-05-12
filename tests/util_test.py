# -*- coding: utf-8 -*-
import os
import pytest
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from types import LambdaType
from time import time

from deeptrain.util import searching
from deeptrain.util import misc
from deeptrain.util import logging
from deeptrain.util import training
from deeptrain.util import configs
from deeptrain.util import _default_configs
from deeptrain.util.misc import pass_on_error, deeplen
from deeptrain import metrics
from deeptrain import preprocessing
from tests import _methods_dummy
from tests.backend import BASEDIR, tempdir, notify, ModelDummy, TraingenDummy


tests_done = {name: None for name in ('searching', 'misc', 'configs',
                                      'training', 'preprocessing', 'logging',
                                      'deeplen', 'introspection')}


@notify(tests_done)
def test_searching():
    labels = np.random.randint(0, 2, (32,))
    preds = np.random.uniform(0, 1, (32,))
    metric_fn = metrics.f1_score
    searching.find_best_predict_threshold(labels, preds, metric_fn, verbosity=2)

    assert True


@notify(tests_done)
def test_misc():
    def _test_nCk():
        assert misc.nCk(10, 2) == 45
        assert misc.nCk(4, 5) == 1

    def _test_ordered_shuffle():
        ls = [1, 2, 3, 4, 'a']
        x = np.array([5, 6, 7, 8, 9])
        dc = {'a': 1, 5: ls, (2, 3): x, '4': None, None: {1: 2}}
        ls, x, dc = misc.ordered_shuffle(ls, x, dc)

        assert len(ls) == len(x) == len(dc) == 5
        assert isinstance(ls, list)
        assert isinstance(x, np.ndarray)
        assert isinstance(dc, dict)

    def _test_train_on_batch_dummy():
        model = ModelDummy()
        model.loss = 'sparse_categorical_crossentropy'
        misc._train_on_batch_dummy(model)
        model.loss = 'hinge'
        misc._train_on_batch_dummy(model)
        model.loss = 'poisson'
        misc._train_on_batch_dummy(model)
        model.loss = 'categorical_crossentropy'
        misc._train_on_batch_dummy(model, class_weights={0: 1, 1: 5})
        misc._train_on_batch_dummy(model, class_weights=None)
        model.loss = 'goblin'
        pass_on_error(misc._train_on_batch_dummy, model)

    def _test_make_plot_configs_from_metrics():
        tg = TraingenDummy()
        tg.train_metrics = ['binary_crossentropy', 'hinge']
        tg.val_metrics = ['loss', 'f1_score', 'tnr', 'tpr', 'kld']
        tg.plot_first_pane_max_vals = 1
        tg.key_metric = 'loss'
        misc._make_plot_configs_from_metrics(tg)
        tg.key_metric = 'f1_score'
        misc._make_plot_configs_from_metrics(tg)

    def _test_get_module_methods():
        mm = misc.get_module_methods(_methods_dummy)
        assert len(mm) == 1 and 'fn1' in mm
        assert isinstance(mm['fn1'], LambdaType)

    _test_nCk()
    _test_ordered_shuffle()
    _test_train_on_batch_dummy()
    _test_make_plot_configs_from_metrics()
    _test_get_module_methods()


@notify(tests_done)
def test_logging():
    def _testget_unique_model_name():
        tg = TraingenDummy()
        tg.model_name_configs = {'datagen.shuffle': ''}
        os.mkdir(os.path.join(tg.logs_dir, 'M0'))
        logging.get_unique_model_name(tg)

    def _test_log_init_state():
        class SUV():
            pass

        tg = TraingenDummy()
        tg.SUV = SUV()
        logging._log_init_state(tg, source_lognames=['swordfish', 'SUV'])
        logging._log_init_state(tg, source_lognames='*')
        logging._log_init_state(tg, source_lognames=None)

    def _test_get_report_text():
        tg = TraingenDummy()
        tg.report_configs = {'model': dict(stuff='staff'), 'saitama': None}
        pass_on_error(logging.get_report_text, tg)

        tg.report_configs = {'model': {'genos': [1]}}
        pass_on_error(logging.get_report_text, tg)

        tg.report_configs = {'model': {'include': [], 'exclude': []}}
        pass_on_error(logging.get_report_text, tg)

        tg.report_configs = {'model': {'exclude_types': ['etc']}}
        tg.model_configs = {'a': 1}
        pass_on_error(logging.get_report_text, tg)

    def _test_generate_report():
        tg = TraingenDummy()
        tg.report_configs = {'model': {'include': []}}
        tg.logdir = ''
        pass_on_error(logging.generate_report, tg, '')

    logs_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_logs')
    best_models_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_models')
    with tempdir(logs_dir), tempdir(best_models_dir):
        _testget_unique_model_name()

    _test_log_init_state()
    _test_get_report_text()
    _test_generate_report()


@notify(tests_done)
def test_deeplen():
    def _make_bignest():
        arrays = [np.random.randn(100, 100), np.random.uniform(30, 40, 10)]
        lists = [[1, 2, '3', '4', 5, [6, 7]] * 555, {'a': 1, 'b': arrays[0]}]
        dicts = {'x': [1, {2: [3, 4]}, [5, '6', {'7': 8}]*99] * 55,
                 'b': [{'a': 5, 'b': 3}] * 333, ('k', 'g'): (5, 9, [1, 2])}
        tuples = (1, (2, {3: np.array([4., 5.])}, (6, 7, 8, 9) * 21) * 99,
                  (10, (11,) * 5) * 666)
        return {'arrays': arrays, 'lists': lists,
                'dicts': dicts, 'tuples': tuples}

    def _print_report(bignest, t0):
        t = time() - t0
        print("{:.5f} / iter ({} iter avg, total time: {:.3f}); sizes:".format(
            t / iters, iters, t))
        print("bignest:", deeplen(bignest))
        print(("{} {}\n" * len(bignest)).format(
            *[x for k, v in bignest.items()
              for x in ((k + ':').ljust(8), deeplen(v))]))

    iters = 2
    bignest = _make_bignest()
    t0 = time()
    for _    in range(iters):
        deeplen(bignest)
    _print_report(bignest, t0)


@notify(tests_done)
def test_training():
    def _test_unroll_into_samples():
        outs_ndim = (16, 100)
        arrs = [np.random.randn(16, 100)] * 2
        training._unroll_into_samples(outs_ndim, *arrs)

    def _test_weighted_normalize_preds():
        tg = TraingenDummy()
        tg.val_datagen.slices_per_batch = 1
        preds_all = np.random.uniform(0, 1, (3, 1, 16, 100))
        training._weighted_normalize_preds(tg, preds_all)

    def _test_validate_data_shapes():
        tg = TraingenDummy()
        tg.model.batch_size = None
        tg.batch_size = 16
        tg.model.output_shape = (None, 100)
        data = {'preds_all': np.random.uniform(0, 1, (3, 16, 100))}
        training._validate_data_shapes(tg, data)

        data = {'preds_all': np.random.uniform(0, 1, (3, 1, 16, 100))}
        pass_on_error(training._validate_data_shapes, tg, data)

    def _test_validate_class_data_shapes():
        tg = TraingenDummy()
        tg.model.batch_size = None
        tg.batch_size = 16
        tg.model.output_shape = (None, 100)
        data = {'class_labels_all': np.random.uniform(0, 1, (3, 16, 100))}
        training._validate_class_data_shapes(tg, data, validate_n_slices=True)


    _test_unroll_into_samples()
    _test_weighted_normalize_preds()
    _test_validate_data_shapes()
    _test_validate_class_data_shapes()


@notify(tests_done)
def test_configs():
    for name_fn in (configs._NAME_PROCESS_KEY_FN,
                    _default_configs._DEFAULT_NAME_PROCESS_KEY_FN):
        cfg = dict(init_lr=[2e-4, 2e-4, 2e-4, 1e-4],
                   eta_t=(.9, 1.1, 2),
                   timesteps=13500,
                   name='leaf',
                   a=.5,
                   best_key_metric=0.91,
                   )
        assert name_fn('init_lr',   'lr',   cfg) == '_lr2e-4x3_1e-4'
        assert name_fn('eta_t',     'et',   cfg) == '_et.9_1.1_2'
        assert name_fn('timesteps', '',     cfg) == '_13.5k'
        assert name_fn('name',      'name', cfg) == '_name'
        assert name_fn('a',         'a',    cfg) == '_a.5'
        assert name_fn('best_key_metric', 'max',  cfg) == '_max.910'

    names = ['PLOT_CFG', 'BINARY_CLASSIFICATION_PLOT_CFG',
             'MODEL_NAME_CFG', 'REPORT_CFG', 'TRAINGEN_SAVE_LIST',
             'METRIC_PRINTSKIP_CFG', 'METRIC_TO_ALIAS', 'ALIAS_TO_METRIC',
             'TRAINGEN_CFG', 'DATAGEN_CFG']
    for config in (configs, _default_configs):
        for name in names:
            name = [x for x in vars(config) if name in x][0]
            _ = getattr(config, name)
            assert True


@notify(tests_done)
def test_preprocessing(monkeypatch):
    def _test_numpy_data_to_numpy_sets(datadir):
        with tempdir(datadir):
            data = np.random.randn(161, 2)
            labels = np.random.randint(0, 2, (161,))
            preprocessing.numpy_data_to_numpy_sets(
                datadir, data, labels, batch_size=32, shuffle=True,
                data_basename='ex', oversample_remainder=True)

            paths = [str(x) for x in Path(datadir).iterdir() if
                     x.suffix == '.npy']
            assert (len(paths) == 6), ("%s paths" % len(paths))  # 160 / 32

        os.mkdir(datadir)
        data = np.random.randn(161, 2)
        labels = np.random.randint(0, 2, (161,))

        preprocessing.numpy_data_to_numpy_sets(
            datadir, data, labels, batch_size=32, shuffle=True,
            data_basename='ex', oversample_remainder=False)
        os.remove(os.path.join(datadir, "labels.h5"))

        paths = [str(x) for x in Path(datadir).iterdir() if
                 x.suffix == '.npy']
        assert (len(paths) == 5), ("%s paths" % len(paths))  # 160 / 32

        return paths

    def _test_data_to_hdf5(datadir, paths):
        X = np.array([np.load(path) for path in paths])
        kw = dict(savepath=os.path.join(datadir, "data.h5"), batch_size=32,
                  shuffle=True, compression='lzf', overwrite=None)

        monkeypatch.setattr('builtins.input', lambda x: "y")
        preprocessing.data_to_hdf5(loaddir=datadir, **kw)
        preprocessing.data_to_hdf5(data=X, **kw)

        kw.update(**dict(overwrite=True, load_fn=lambda x: x))
        preprocessing.data_to_hdf5(data=X, **kw)

        shutil.rmtree(datadir)

    def _test_numpy2D_to_csv(datadir):
        def _batches_from_df(savepath):
            df = pd.read_csv(savepath)
            assert df.shape == (32, 2)
            x0 = df['0'][:16].to_numpy()
            x1 = df['0'][16:].to_numpy()
            return x0, x1

        def _test_batch_dim_0(savepath):
            X = np.random.randint(0, 2, (16, 4))
            preprocessing.numpy2D_to_csv(data=X, savepath=savepath,
                                         batch_size=32, batch_dim=0)
            x0, x1 = _batches_from_df(savepath)
            assert np.sum(np.abs(x0 - X[:, 0])) == 0
            assert np.sum(np.abs(x1 - X[:, 1])) == 0

        def _test_batch_dim_1(savepath):
            X = np.random.randint(0, 2, (4, 16))
            preprocessing.numpy2D_to_csv(data=X, savepath=savepath,
                                         batch_size=32, batch_dim=1)
            x0, x1 = _batches_from_df(savepath)
            assert np.sum(np.abs(x0 - X[0])) == 0
            assert np.sum(np.abs(x1 - X[1])) == 0

        os.mkdir(datadir)
        savepath = os.path.join(datadir, "labels.csv")
        _test_batch_dim_0(savepath)
        _test_batch_dim_1(savepath)
        shutil.rmtree(datadir)

    datadir = os.path.join(BASEDIR, "_data")

    paths = _test_numpy_data_to_numpy_sets(datadir)
    _test_data_to_hdf5(datadir, paths)
    _test_numpy2D_to_csv(datadir)


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
