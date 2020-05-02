# -*- coding: utf-8 -*-
import os
os.environ['IS_MAIN'] = '1' * (__name__ == '__main__')
import pytest

from pathlib import Path
from termcolor import cprint
from copy import deepcopy

from tests.backend import BASEDIR
from deeptrain.util.misc import pass_on_error
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

    _notify('advance_batch')


def test_shuffle():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['superbatch_dir'] = os.path.join(datadir, 'image', 'train')
    C['batch_size'] = 64
    dg = SimpleBatchgen(**C)
    dg.preload_superbatch()
    dg.advance_batch()

    _notify('shuffle')


def test_data_loaders():
    C = deepcopy(DATAGEN_CFG)
    C['data_dir'] = os.path.join(datadir, 'timeseries_split', 'train')
    C['labels_path'] = os.path.join(datadir, 'timeseries_split', 'train',
                                    'labels.h5')
    C['batch_size'] = 128
    C['base_name'] = 'batch128_'
    dg = SimpleBatchgen(**C)
    dg.advance_batch()

    _notify('data_loaders')


def _notify(name):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        test_name = Path(__file__).stem.replace('_', ' ').upper()
        cprint(f"<< {test_name} PASSED >>\n", 'green')


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
