# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
import shutil

from pathlib import Path
from termcolor import cprint

from deeptrain.util import metrics
from deeptrain.util import searching
from deeptrain.util import misc
from deeptrain.util import preprocessing
from tests.backend import BASEDIR, tempdir


tests_done = {name: None for name in ('searching', 'misc', 'preprocessing')}


def test_searching():
    labels = np.random.randint(0, 2, (32,))
    preds = np.random.uniform(0, 1, (32,))
    metric_fn = metrics.f1_score
    searching.get_best_predict_threshold(labels, preds, metric_fn, verbosity=2)

    assert True
    _notify('searching')


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

    _test_nCk()
    _test_ordered_shuffle()
    _notify('misc')


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

        preprocessing.data_to_hdf5(loaddir=datadir, **kw)
        preprocessing.data_to_hdf5(data=X, **kw)

        monkeypatch.setattr('builtins.input', lambda: "y")
        kw.update(dict(overwrite=True, load_fn=lambda x: x))
        preprocessing.data_to_hdf5(data=X, **kw)


    datadir = os.path.join(BASEDIR, "_data")
    paths = _test_numpy_data_to_numpy_sets(datadir)
    _test_data_to_hdf5(datadir, paths)
    _notify('preprocessing')


def _notify(name):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        cprint("<< UTIL TEST PASSED >>\n", 'green')


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
