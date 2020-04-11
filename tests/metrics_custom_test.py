# -*- coding: utf-8 -*-
import pytest
import numpy as np
import sklearn.metrics

from deeptrain.util import metrics


to_test = ['f1_score', 'f1_score_multi_th',
            'tnr',
            'tpr',
            'tnr_tpr',
            'binary_informedness',
            'roc_auc_score',
            ]

[f1_score, f1_score_multi_th, tnr, tpr, tnr_tpr, binary_informedness,
 roc_auc_score] = [getattr(metrics, name) for name in to_test]


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
    y_true = [0,  0,  0,  0, 1, 1,  1,  1]
    y_pred = [0, .6, .7, .9, 1, 0, .8, .6]

    assert tnr(y_true, y_pred) == .25
    assert tpr(y_true, y_pred) == .75
    assert tnr_tpr(y_true, y_pred) == [.25, .75]
    assert binary_informedness(y_true, y_pred) == 0.


def test_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    np.random.shuffle(y_true)
    y_pred = np.random.uniform(0, 1, 32)

    test_score = roc_auc_score(y_true, y_pred, visualize=True)
    sklearn_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    adiff = abs(test_score - sklearn_score)
    assert (adiff < 1e-10), ("sklearn: {:.15f}\ntest:    {:.15f}"
                             "\nabsdiff: {}".format(
                                 sklearn_score, test_score, adiff))

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
