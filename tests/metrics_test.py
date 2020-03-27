# -*- coding: utf-8 -*-
import pytest
import numpy as np

from .backend import K
from .backend import keras_losses, keras_metrics
from deeptrain.util import metrics


to_test = ['binary_crossentropy',
           'categorical_crossentropy',
           'sparse_categorical_crossentropy',
           'mean_squared_error',
           'mean_absolute_error',
           'mean_squared_logarithmic_error',
           'mean_absolute_percentage_error',
           'squared_hinge',
           'hinge',
           'categorical_hinge',
           'logcosh',
           'kullback_leibler_divergence',
           'poisson',
           'cosine_proximity',
           'binary_accuracy',
           'categorical_accuracy',
           'sparse_categorical_accuracy',
           ]

KERAS_METRICS = ['binary_accuracy', 'categorical_accuracy',
                 'sparse_categorical_accuracy']

np.random.seed(0)


def _make_test_fn(name):
    def _keras_metric(name):
        if name not in KERAS_METRICS:
            def _weighted_loss(y_true, y_pred, sample_weight):
                losses = K.get_value(getattr(keras_losses, name)(
                    K.variable(y_true), K.variable(y_pred)))
                if isinstance(sample_weight, np.ndarray) and losses.ndim > 1:
                    sample_weight = sample_weight.reshape(
                        len(losses), *[1]*losses.ndim)
                return np.mean(losses * sample_weight)
            return _weighted_loss
        else:
            # sample_weight makes no sense for keras `metrics`
            return lambda y_true, y_pred: (
                K.get_value(getattr(keras_metrics, name)(K.variable(y_true), 
                                                         K.variable(y_pred))))

    def _test_metric(name):
        if name not in KERAS_METRICS:
            return lambda y_true, y_pred, sample_weight: (
                getattr(metrics, name)(y_true, y_pred, sample_weight))
        else:
            # sample_weight makes no sense for keras `metrics`
            return lambda y_true, y_pred: getattr(metrics, name)(y_true, y_pred)
    
    _test_metric_fn = _test_metric(name)
    _keras_metric_fn = _keras_metric(name)

    def _test_fn(y_true, y_pred, sample_weight=1):
        args = (y_true, y_pred, sample_weight)
        if name in KERAS_METRICS:
            args = args[:-1]
        return _test_metric_fn(*args), _keras_metric_fn(*args)
    return _test_fn


def _make_data_fn(name):
    def _data_fn(name):
        batch_size = 16
        n_classes = 5

        if name in ('binary_crossentropy', 'binary_accuracy'):
            y_true = np.random.randint(0, 2, (batch_size, 1))
            y_pred = np.random.uniform(0, 1, (batch_size, 1))
        elif name in ('categorical_crossentropy', 'categorical_accuracy'):
            class_labels = np.random.randint(0, n_classes, batch_size)
            y_true = np.eye(n_classes)[class_labels]
            y = np.random.uniform(0, 1, (batch_size, n_classes))
            # sample-normalize to 1
            y_pred = y / y.sum(axis=1).reshape(-1, 1)
        elif name in ('sparse_categorical_crossentropy',
                      'sparse_categorical_accuracy'):
            y_true = np.random.randint(0, n_classes, (batch_size, 1))
            y = np.random.uniform(0, 1, (batch_size, n_classes))
            # sample-normalize to 1
            y_pred = y / y.sum(axis=1).reshape(-1, 1)
        elif name in ('mean_squared_error', 'mean_absolute_error',
                      'mean_squared_logarithmic_error',
                      'mean_absolute_percentage_error',
                      'logcosh', 'kullback_leibler_divergence'):
            y_true = np.random.randn(batch_size, 10, 4)
            y_pred = np.random.randn(batch_size, 10, 4)
        elif name in ('squared_hinge', 'hinge', 'categorical_hinge'):
            y_true = np.array([-1, 1])[np.random.randint(0, 2, (batch_size, 1))]
            y_pred = np.random.uniform(-1, 1, (batch_size, 1))
        elif name in ('poisson', 'cosine_proximity'):
            y_true = np.random.uniform(0, 10, batch_size)
            y_pred = np.random.uniform(0, 10, batch_size)
        else:
            raise ValueError("unknown metric: '{}'".format(name))

        if y_true.ndim == 1:
            y_true == y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred == y_pred.reshape(-1, 1)
        sample_weight = np.random.uniform(0, 10, batch_size)
        return lambda: (y_true, y_pred, sample_weight)
    return _data_fn(name)


def _test_unweighted(name):
    y_true, y_pred, _ = _make_data_fn(name)()
    results = _make_test_fn(name)(y_true, y_pred)
    fail_msg = (name, "unweighted", 
                "diff: {}tested: {}, keras: {}".format(
                    np.abs(results[1] - results[0]), results[0], results[1]))
    return np.allclose(*results, atol=1e-3, rtol=1e-5), fail_msg


def _test_sample_weighted(name):
    if name in KERAS_METRICS:
        # sample_weight makes no sense for keras `metrics`
        return True, ''
    y_true, y_pred, sample_weight = _make_data_fn(name)()
    results = _make_test_fn(name)(y_true, y_pred, sample_weight)
    fail_msg = (name, "sample_weighted", 
                "diff: {}tested: {}, keras: {}".format(
                    np.abs(results[1] - results[0]), results[0], results[1]))
    return np.allclose(*results, atol=1e-3, rtol=1e-5), fail_msg


def _assert(fn, *args, **kwargs):
    def _do_asserting(fn_out):
        if isinstance(fn_out, (list, tuple)):
            value, fail_msg = fn_out
            assert value, fail_msg
        else:
            assert fn_out  # value only
    return lambda x: _do_asserting(fn(*args, **kwargs))


def _to_test_name(txt):  # snake_case -> CamelCase, prepend "Test"
    return "Test" + ''.join(x.capitalize() or '_' for x in txt.split('_'))


(TestBinaryCrossentropy,
 TestCategoricalCrossentropy,
 TestSparseCategoricalCrossentropy,
 TestMeanSquaredError,
 TestMeanAbsoluteError,
 TestMeanSquaredLogarithmicError,
 TestMeanAbsolutePercentageError,
 TestSquaredHinge,
 TestHinge,
 TestCategoricalHinge,
 TestLogcosh,
 TestKullbackLeiblerDivergence,
 TestPoisson,
 TestCosineProximity,
 TestBinaryAccuracy,
 TestCategoricalAccuracy,
 TestSparseCategoricalAccuracy,
 ) = [type(_to_test_name(name), (), 
           {'test_unweighted': _assert(_test_unweighted, name),
            'test_sample_weighted': _assert(_test_sample_weighted, name)}
          ) for name in to_test]


custom_to_test = ['f1_score', 'f1_score_multi_th',
                  'tnr',
                  'tpr',
                  'binary_accuracies',
                  'binary_informedness',
                  ]

[f1_score, f1_score_multi_th, tnr, tpr, binary_accuracies, binary_informedness
 ] = [getattr(metrics, name) for name in custom_to_test]


def test_f1_score():
    def _test_basic():
        y_true = [0,     0,   1,   0,   0, 0, 1, 1]
        y_pred = [.01, .93, .42, .61, .15, 0, 1, .5]    
        assert abs(f1_score(y_true, y_pred) - 1 / 3) < 1e-15
    
    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .6, .7, .8]
        assert f1_score(y_true, y_pred) == 0.5
        
    def _test_no_positive_predictions():
        y_true = [0, 0, 1]
        y_pred = [0, 0, 0]
        assert f1_score(y_true, y_pred) == 0
    
    _test_basic()
    _test_no_positive_labels()
    _test_no_positive_predictions()


def test_f1_score_multi_th():
    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .6, .7, .8]
        assert f1_score_multi_th(y_true, y_pred) == [0, 0]

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
    
    _test_nan_handling()
    _compare_against_f1_score()


def test_binaries():
    y_true = [0, 0,  0,  0, 1, 1,  1 , 1]
    y_pred = [0, 1, .1, .9, 1, 0, .8, .2]
    
    assert tnr(y_true, y_pred) == .5
    assert tpr(y_true, y_pred) == .5
    assert binary_accuracies(y_true, y_pred) == [.5, .5]
    assert binary_informedness(y_true, y_pred) == 0.
    
    
if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
