# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from pathlib import Path
from termcolor import cprint

from deeptrain.util import metrics
from deeptrain.util import searching
from deeptrain.util import misc
from deeptrain.util import preprocessing
from .backend import BASEDIR, tempdir


tests_done = {name: None for name in ('searching', 'misc', 'preprocessing')}


def test_searching():
    labels = np.random.randint(0, 2, (32,))
    preds = np.random.uniform(0, 1, (32,))
    metric_fn = metrics.f1_score
    searching.get_best_predict_threshold(labels, preds, metric_fn, verbosity=2)

    assert True
    _notify('searching')


def test_misc():
    assert misc.nCk(10, 2) == 45
    assert misc.nCk(4, 5) == 1
    _notify('misc')


def test_preprocessing():
    datadir = os.path.join(BASEDIR, "_data")
    with tempdir(datadir):
        data = np.random.randn(160, 2)
        labels = np.random.randint(0, 2, (160,))
        preprocessing.numpy_data_to_numpy_sets(
            datadir, data, labels, batch_size=32, shuffle=True,
            data_basename='ex', oversample_remainder=True)
        os.remove(os.path.join(datadir, "labels.h5"))

        paths = [str(x) for x in Path(datadir).iterdir() if x.suffix == '.npy']
        assert (len(paths) == 5), ("%s paths" % len(paths))  # 160 / 32
        X = np.array([np.load(path) for path in paths])

        kw = dict(savepath=os.path.join(datadir, "data.h5"), batch_size=32,
                  shuffle=True, compression='lzf', overwrite=True)
        preprocessing.data_to_hdf5(loaddir=datadir, **kw)
        preprocessing.data_to_hdf5(data=X, **kw)

    assert True
    _notify('preprocessing')


def _notify(name):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        cprint("<< UTIL TEST PASSED >>\n", 'green')


if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
