# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from copy import deepcopy

from tests.backend import BASEDIR, notify
from deeptrain.util.misc import pass_on_error, ordered_shuffle
from deeptrain import SimpleBatchgen


datadir = os.path.join(BASEDIR, 'tests', 'data')

DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'image', 'train'),
    labels_path=os.path.join(datadir, 'image', 'train', 'labels.h5'),
    batch_size=128,
    data_category='image',
    shuffle=True,
)

tests_done = {name: None for name in ('advance_batch', 'shuffle',
                                      'data_loaders',)}


@notify(tests_done)
def test_advance_batch():
    C = deepcopy(DATAGEN_CFG)
    C['superbatch_dir'] = os.path.join(datadir, 'image', 'train')
    dg = SimpleBatchgen(**C)
    dg.advance_batch()

    C['batch_size'] = 31
    dg = SimpleBatchgen(**C)
    pass_on_error(dg.advance_batch)

    C['batch_size'] = 256
    dg = SimpleBatchgen(**C)
    dg.set_nums_to_process = []
    pass_on_error(dg.advance_batch)

    C['data_format'] = 'pigeon'
    pass_on_error(SimpleBatchgen, **C)


@notify(tests_done)
def test_shuffle():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['superbatch_dir'] = os.path.join(datadir, 'image', 'train')
    C['batch_size'] = 64
    dg = SimpleBatchgen(**C)
    dg.preload_superbatch()
    dg.advance_batch()


@notify(tests_done)
def test_data_loaders():
    C = deepcopy(DATAGEN_CFG)
    C['data_dir'] = os.path.join(datadir, 'timeseries_split', 'train')
    C['labels_path'] = os.path.join(datadir, 'timeseries_split', 'train',
                                    'labels.h5')
    C['batch_size'] = 128
    C['base_name'] = 'batch32_'
    dg = SimpleBatchgen(**C)
    dg.advance_batch()


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


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
