# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from tests.backend import BASEDIR, tempdir, notify
from deeptrain.util.misc import pass_on_error, ordered_shuffle
from deeptrain.util import data_loaders, labels_preloaders
from deeptrain.util import TimeseriesPreprocessor
from deeptrain import DataGenerator


datadir = os.path.join(BASEDIR, 'tests', 'data')

DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'image', 'train'),
    labels_path=os.path.join(datadir, 'image', 'train', 'labels.h5'),
    batch_size=128,
    shuffle=True,
)

tests_done = {name: None for name in ('advance_batch', 'shuffle', 'kwargs',
                                      'data_loaders', 'labels_preloaders',
                                      'infer_data_info')}


@notify(tests_done)
def test_advance_batch():
    C = deepcopy(DATAGEN_CFG)
    C['superbatch_dir'] = os.path.join(datadir, 'image', 'train')
    dg = DataGenerator(**C)
    dg.advance_batch()

    C['batch_size'] = 31
    dg = DataGenerator(**C)
    pass_on_error(dg.advance_batch)

    C['batch_size'] = 256
    dg = DataGenerator(**C)
    dg.set_nums_to_process = []
    pass_on_error(dg.advance_batch)

    C['data_loader'] = 'pigeon'
    pass_on_error(DataGenerator, **C)


@notify(tests_done)
def test_shuffle():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['superbatch_dir'] = os.path.join(datadir, 'image', 'train')
    C['batch_size'] = 64
    dg = DataGenerator(**C)
    dg.preload_superbatch()
    dg.advance_batch()


@notify(tests_done)
def test_kwargs():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['shuffle_group_samples'] = True
    DataGenerator(**C)


@notify(tests_done)
def test_data_loaders():
    def _test_auto_hdf5(C):
        dg = DataGenerator(**C)
        dg.advance_batch()

    def _test_hdf5(C):
        C['data_loader'] = data_loaders.hdf5_loader
        dg = DataGenerator(**C)
        dg.advance_batch()

    def _test_exceptions(C):
        C['data_loader'] = 'invalid_loader'
        pass_on_error(DataGenerator, **C)

        C['data_loader'] = None
        dg = DataGenerator(**C)
        pass_on_error(dg._set_data_loader, 'invalid_loader')


    C = deepcopy(DATAGEN_CFG)
    C['data_dir'] = os.path.join(datadir, 'timeseries_split', 'train')
    C['labels_path'] = os.path.join(datadir, 'timeseries_split', 'train',
                                    'labels.h5')
    C['batch_size'] = 128
    C['base_name'] = 'batch32_'

    _test_auto_hdf5(C)
    _test_hdf5(C)
    _test_exceptions(C)


@notify(tests_done)
def test_labels_preloaders():
    def _test_no_preloader():
        C = deepcopy(DATAGEN_CFG)
        C['labels_preloader'] = None
        C['labels_path'] = None
        DataGenerator(**C)

    def _test_hdf5_preloader():
        C = deepcopy(DATAGEN_CFG)
        C['labels_preloader'] = labels_preloaders.hdf5_preloader
        DataGenerator(**C)

    _test_no_preloader()
    _test_hdf5_preloader()


@notify(tests_done)
def test_preprocessors():
    def _test_uninstantiated():
        C = deepcopy(DATAGEN_CFG)
        C['preprocessor'] = TimeseriesPreprocessor
        C['preprocessor_configs'] = dict(window_size=5, batch_timesteps=10)
        DataGenerator(**C)

    def _test_instantiated():
        C = deepcopy(DATAGEN_CFG)
        C['preprocessor'] = TimeseriesPreprocessor(window_size=5,
                                                   batch_timesteps=10)

    def _test_incomplete():
        class IncompletePreprocessor():
            def __init__(self):
                pass

        C = deepcopy(DATAGEN_CFG)
        C['preprocessor'] = IncompletePreprocessor()
        pass_on_error(DataGenerator, **C)

    _test_uninstantiated()
    _test_instantiated()
    _test_incomplete()


@notify(tests_done)
def test_shuffle_group_batches():
    """Ensure reshape doesn't mix batch and spatial dimensions"""
    group_batch = np.random.randn(128, 28, 28, 1)
    labels = np.random.randint(0, 2, (128, 10))
    gb, lb = group_batch, labels

    batch_size = 64
    x0, x1 = gb[:64], gb[64:]
    y0, y1 = lb[:64], lb[64:]

    gb_shape, lb_shape = gb.shape, lb.shape
    gb = gb.reshape(-1, batch_size, *gb_shape[1:])
    lb = lb.reshape(-1, batch_size, *lb_shape[1:])
    x0adiff = np.sum(np.abs(gb[0] - x0))
    x1adiff = np.sum(np.abs(gb[1] - x1))
    y0adiff = np.sum(np.abs(lb[0] - y0))
    y1adiff = np.sum(np.abs(lb[1] - y1))
    assert x0adiff == 0, ("x0 absdiff: %s" % x0adiff)
    assert x1adiff == 0, ("x1 absdiff: %s" % x1adiff)
    assert y0adiff == 0, ("y0 absdiff: %s" % y0adiff)
    assert y1adiff == 0, ("y1 absdiff: %s" % y1adiff)

    gb, lb = ordered_shuffle(gb, lb)
    gb, lb = gb.reshape(*gb_shape), lb.reshape(*lb_shape)
    assert (gb.shape == gb_shape) and (lb.shape == lb_shape)


@notify(tests_done)
def test_infer_data_info():
    def _test_empty_data_dir():
        C = deepcopy(DATAGEN_CFG)
        with tempdir() as dirpath:
            C['data_dir'] = dirpath
            pass_on_error(DataGenerator, **C)

    def _test_no_supported_file_ext():
        C = deepcopy(DATAGEN_CFG)
        with tempdir() as dirpath:
            plt.plot([0, 1])
            plt.gcf().savefig(os.path.join(dirpath, "img.png"))
            C['data_dir'] = dirpath
            pass_on_error(DataGenerator, **C)

    _test_empty_data_dir()
    _test_no_supported_file_ext()


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
