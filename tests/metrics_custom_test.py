# -*- coding: utf-8 -*-
import pytest
import numpy as np
import sklearn.metrics

from deeptrain.util.training import _get_val_history, _weighted_normalize_preds
from deeptrain.util.metrics import (
    f1_score,
    f1_score_multi_th,
    tnr,
    tpr,
    tnr_tpr,
    binary_informedness,
    roc_auc_score
    )


def test_f1_score():
    def _test_basic():
        y_true = [0,     0,   1,   0,   0, 0, 1, 1]
        y_pred = [.01, .93, .42, .61, .15, 0, 1, .5]
        assert abs(f1_score(y_true, y_pred) - 1 / 3) < 1e-15

    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .65, .7, .8]
        assert f1_score(y_true, y_pred) == 0.5

    def _test_no_positive_predictions():
        y_true = [0, 0, 1]
        y_pred = [0, 0, 0]
        assert f1_score(y_true, y_pred) == 0

    def _test_vs_sklearn():
        y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
        np.random.shuffle(y_true)
        y_pred = np.random.uniform(0, 1, 32)

        test_score = f1_score(y_true, y_pred, pred_threshold=.5)
        sklearn_score = sklearn.metrics.f1_score(y_true, y_pred > .5)
        adiff = abs(test_score - sklearn_score)
        assert (adiff < 1e-10), ("sklearn: {:.15f}\ntest:    {:.15f}"
                                "\nabsdiff: {}".format(
                                    sklearn_score, test_score, adiff))
    _test_basic()
    _test_no_positive_labels()
    _test_no_positive_predictions()
    _test_vs_sklearn()


def test_f1_score_multi_th():
    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .65, .7, .8]
        pred_thresholds = [.4, .6]
        assert np.all(f1_score_multi_th(y_true, y_pred, pred_thresholds)
                      == [.5, .5])

    def _test_nan_handling():
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0]
        pred_thresholds = [.4, .6]
        assert np.all(f1_score_multi_th(y_true, y_pred, pred_thresholds) == 0)

    def _compare_against_f1_score():
        y_true = np.random.randint(0, 2, (64,))
        y_pred = np.random.uniform(0, 1, (64,))
        pred_thresholds = [.01, .05, .1, .2, .4, .5, .6, .8, .95, .99]

        single_scores = [f1_score(y_true, y_pred, th) for th in pred_thresholds]
        parallel_scores = f1_score_multi_th(y_true, y_pred, pred_thresholds)
        assert np.allclose(single_scores, parallel_scores, atol=1e-15)

    _test_no_positive_labels()
    _test_nan_handling()
    _compare_against_f1_score()


def test_binaries():
    y_true = [0,  0,  0,  0,  1,  1,  1,  1]
    y_pred = [0, .6, .7, .9,  1,  0, .8, .6]

    assert tnr(y_true, y_pred) == .25
    assert tpr(y_true, y_pred) == .75
    assert tnr_tpr(y_true, y_pred) == [.25, .75]
    assert binary_informedness(y_true, y_pred) == 0.


def test_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    np.random.shuffle(y_true)
    y_pred = np.random.uniform(0, 1, 32)

    test_score = roc_auc_score(y_true, y_pred)
    sklearn_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    adiff = abs(test_score - sklearn_score)
    assert (adiff < 1e-10), ("sklearn: {:.15f}\ntest:    {:.15f}"
                             "\nabsdiff: {}".format(
                                 sklearn_score, test_score, adiff))


class TraingenDummy():
    """Proxy class to test util.metrics methods via `cls`"""

    class Datagen():
        def __init__(self):
            self.x = 0

    class Model():
        def __init__(self):
            self.loss = 'mse'
            self.output_shape = (8, 1)

    def __init__(self):
        self.model = TraingenDummy.Model()
        self.datagen = TraingenDummy.Datagen()
        self.val_datagen = TraingenDummy.Datagen()

        self.eval_fn_name = 'predict'
        self.key_metric = 'f1_score'
        self.key_metric_fn = f1_score
        self.class_weights = None
        self.val_class_weights = None
        self.batch_size = 8
        self._inferred_batch_size = 8

        self.best_subset_size = None
        self.pred_weighted_slices_range = None
        self.predict_threshold = .5
        self.dynamic_predict_threshold_min_max = None
        self.loss_weighted_slices_range = None
        self.pred_weighted_slices_range = None

        self.val_metrics = []
        self._sw_cache = []

    def set_shapes(self, batch_size):
        self.batch_size = batch_size
        self._inferred_batch_size = batch_size
        self.model.output_shape = (batch_size, 1)


def test_sample_unrolling():
    """Compare results from internal (TrainGenerator) data transforms vs. those
    of correct transforms done explicitly. Formats included:
        - (samples, outs[1:])
        - (slices, samples, outs[1:])
        - (batches, samples, outs[1:])
        - (batches, slices, samples, outs[1:])
        - Weighted & unweighted slices
        - Linear & nonlinear metrics

    outs == model.output_shape
    'Nonlinear' metrics == cannot be simple-averaged across samples, i.e.
    [fn(samples1) + fn(samples2)] / 2 != fn([samples1, samples2])
    """
    def _test_binaries():
        def _make_traingen(y_true, y_pred, batch_size):
            tg = TraingenDummy()
            tg._labels_cache = y_true.copy()
            tg._preds_cache = y_pred.copy()
            tg._sw_cache = np.ones(y_true.shape)
            tg.val_metrics = ['f1_score', 'tnr', 'tpr', 'tnr_tpr',
                              'binary_informedness', 'roc_auc_score']
            tg.loss = 'binary_crossentropy'
            tg.set_shapes(batch_size)
            return tg

        def _compare(m, yt, yp):
            assert m['f1_score'] == f1_score(yt, yp)
            assert m['tnr'] == tnr(yt, yp)
            assert m['tpr'] == tpr(yt, yp)
            assert m['tnr_tpr'] == tnr_tpr(yt, yp)
            assert all([a == b for a, b in
                        zip(m['tnr_tpr'], tnr_tpr(yt, yp))])
            assert m['binary_informedness'] == binary_informedness(yt, yp)
            assert m['roc_auc_score'] == roc_auc_score(yt, yp)

        def _test_unrolling(y_true, y_pred, tg):
            m = _get_val_history(tg, for_current_iter=False)
            yt, yp = y_true.ravel(), y_pred.ravel()
            _compare(m, yt, yp)

        def _test_batch_unrolling():
            y_true = np.random.randint(0, 2, (2, 8, 1))
            y_pred = np.random.uniform(0, 1, (2, 8, 1))

            tg = _make_traingen(y_true, y_pred, y_true.shape[1])
            _test_unrolling(y_true, y_pred, tg)

        def _test_unweighted_slice_unrolling():
            # unfixed labels across samples also tests fixed case
            y_true = np.random.randint(0, 2, (2, 8, 1))
            y_pred = np.random.uniform(0, 1, (2, 8, 1))

            tg = _make_traingen(y_true, y_pred, y_true.shape[1])
            tg.val_datagen.slices_per_batch = y_true.shape[0]
            _test_unrolling(y_true, y_pred, tg)

        def _test_weighted_slice_unrolling():
            # unfixed labels invalid for weighted slices since preds are normed
            y_true = np.random.randint(0, 2, (1, 8, 1))
            y_pred = np.random.uniform(0, 1, (2, 8, 1))
            y_true = np.vstack([y_true, y_true])  # labels fixed along slices

            tg = _make_traingen(y_true, y_pred, y_true.shape[1])
            tg.val_datagen.slices_per_batch = y_true.shape[0]
            tg.loss_weighted_slices_range = (.2, 1.8)
            tg.pred_weighted_slices_range = (.2, 1.8)
            y_pred = np.expand_dims(y_pred, 0)
            y_pred = _weighted_normalize_preds(tg, y_pred)
            y_true = y_true[0]

            _test_unrolling(y_true, y_pred, tg)

        def _test_sample_slice_unrolling():
            y_true = np.random.randint(0, 2, (3, 2, 8, 1))
            y_pred = np.random.uniform(0, 1, (3, 2, 8, 1))

            tg = _make_traingen(y_true, y_pred, y_true.shape[2])
            tg.val_datagen.slices_per_batch = y_true.shape[1]
            _test_unrolling(y_true, y_pred, tg)

        _test_batch_unrolling()
        _test_unweighted_slice_unrolling()
        _test_weighted_slice_unrolling()
        _test_sample_slice_unrolling()

    _test_binaries()


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
