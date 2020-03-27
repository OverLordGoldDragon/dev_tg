# -*- coding: utf-8 -*-
import numpy as np


EPS = 1e-7  # epsilon (keras default, K.epsilon())


def _standardize(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true).astype('float32')
    y_pred = np.clip(np.asarray(y_pred), EPS, 1 - EPS).astype('float32')

    if sample_weight:
        if isinstance(sample_weight, (list, np.ndarray)):
            sample_weight = np.asarray(sample_weight).squeeze().astype('float32')
        return y_true, y_pred, sample_weight
    return y_true, y_pred


def f1_score(y_true, y_pred, pred_threshold=0.5, beta=1):
    y_true, y_pred = map(np.squeeze, _standardize(y_true, y_pred))
    y_pred = y_pred > pred_threshold

    TP = np.sum((y_true == 1) * (y_pred == 1))
    TN = np.sum((y_true == 0) * (y_pred == 0))
    FP = np.sum((y_true == 0) * (y_pred == 1))
    FN = np.sum((y_true == 1) * (y_pred == 0))

    precision   = TP / (TP + FP) if not (TP == 0 and FP == 0) else 0
    recall      = TP / (TP + FN) if not (TP == 0 and FN == 0) else 0
    specificity = TN / (TN + FP) if not (TN == 0 and FP == 0) else 0

    if not (precision == 0 and recall == 0):
        return (1 + beta) * precision * recall / (beta * precision + recall)
    if y_true.sum() == 0:
        return specificity  # '1' labels absent,  return '0' class accuracy
    return 0                # '1' labels present, none guessed


def f1_score_multi_th(y_true, y_pred, pred_thresholds=[.4, .6], beta=1):
    def _div_then_zero_nans(A, B):
        res = A / (A + B)
        res[np.where(np.isnan(res) + np.isinf(res))] = 0
        return res

    y_true, y_pred = _standardize(y_true, y_pred)
    y_pred = y_pred.reshape(-1, 1) > np.asarray(pred_thresholds).reshape(1, -1)
    y_true = y_true.reshape(-1, 1).repeat(y_pred.shape[-1], -1)

    TP = np.sum((y_true == 1) * (y_pred == 1), axis=0)
    TN = np.sum((y_true == 0) * (y_pred == 0), axis=0)
    FP = np.sum((y_true == 0) * (y_pred == 1), axis=0)
    FN = np.sum((y_true == 1) * (y_pred == 0), axis=0)

    precision   = _div_then_zero_nans(TP, FP)
    recall      = _div_then_zero_nans(TP, FN)
    specificity = _div_then_zero_nans(TN, FP)

    f1score = np.zeros(len(precision))
    for idx, (p, r) in enumerate(zip(precision, recall)):
        if not (p == 0) and (r == 0) and y_true.sum() == 0:
            f1score[idx] = (1 + beta) * p * r / (beta * p + r)
        elif y_true.sum() == 0:
            f1score[idx] = specificity[idx]
    return f1score


def _weighted_loss(losses, sample_weight):
    losses = np.asarray(losses)
    if isinstance(sample_weight, np.ndarray) and losses.ndim > 1:
        sample_weight = sample_weight.reshape(len(losses), *[1]*losses.ndim)
    return np.mean(losses * sample_weight)


def binary_crossentropy(y_true, y_pred, sample_weight=1):
    y_true, y_pred, sample_weight = _standardize(y_true, y_pred, sample_weight)
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()

    logits = np.log(y_pred) - np.log(1 - y_pred)  # sigmoid inverse
    # neg_abs_logits = -np.abs(logits)
    neg_abs_logits = np.where(logits >= 0, -logits, logits)
    # relu_logits    = (logits >= 0) * logits
    relu_logits    = np.where(logits >= 0, logits, 0)
    
    loss_vec = relu_logits - logits * y_true + np.log(1 + np.exp(neg_abs_logits))
    return _weighted_loss(loss_vec, sample_weight)


def categorical_crossentropy(y_true, y_pred, sample_weight=1):
    y_true, y_pred, sample_weight = _standardize(y_true, y_pred, sample_weight)
    y_pred = y_pred.reshape(1, -1) if y_pred.ndim == 1 else y_pred
    
    losses = []
    for label, pred in zip(y_true, y_pred):
        pred /= pred.sum(axis=-1, keepdims=True)
        losses.append(np.sum(label * -np.log(pred), axis=-1, keepdims=False))
    return _weighted_loss(losses, sample_weight)


def sparse_categorical_crossentropy(y_true, y_pred, sample_weight=1):
    num_classes = np.asarray(y_pred).shape[-1]
    y_true = np.eye(num_classes)[np.asarray(y_true).squeeze()]  # to categorical

    return categorical_crossentropy(y_true, y_pred, sample_weight)


def mean_squared_error(y_true, y_pred, sample_weight=1):
    return _weighted_loss(np.mean((y_true - y_pred)**2, axis=-1), 
                          sample_weight)


def mean_absolute_error(y_true, y_pred, sample_weight=1):
    return _weighted_loss(np.mean(np.abs(y_true - y_pred), axis=-1),
                          sample_weight)


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=1):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), EPS, None))
    return _weighted_loss(100. * np.mean(diff, axis=-1), sample_weight)


def mean_squared_logarithmic_error(y_true, y_pred, sample_weight=1):
    first_log = np.log(np.clip(y_pred, EPS, None) + 1.)
    second_log = np.log(np.clip(y_true, EPS, None) + 1.)
    return _weighted_loss(np.mean((first_log - second_log)**2, axis=-1),
                          sample_weight)


def squared_hinge(y_true, y_pred, sample_weight=1):
    return _weighted_loss(np.mean(np.maximum(1. - y_true * y_pred, 0.)**2, 
                                  axis=-1), sample_weight)


def hinge(y_true, y_pred, sample_weight=1):
    return _weighted_loss(np.mean(np.maximum(1. - y_true * y_pred, 0.), 
                                  axis=-1), sample_weight)


def categorical_hinge(y_true, y_pred, sample_weight=1):
    pos = np.sum(y_true * y_pred, axis=-1)
    neg = np.max((1. - y_true) * y_pred, axis=-1)
    return _weighted_loss(np.maximum(0., neg - pos + 1.), sample_weight)


def logcosh(y_true, y_pred, sample_weight=1):
    def _softplus(x):
        return np.log(np.exp(x) + 1)

    def _logcosh(x):
        return x + _softplus(-2. * x) - np.log(2.)
    return _weighted_loss(np.mean(_logcosh(y_pred - y_true), axis=-1),
                          sample_weight)


def kullback_leibler_divergence(y_true, y_pred, sample_weight=1):
    y_true = np.clip(y_true, EPS, 1)
    y_pred = np.clip(y_pred, EPS, 1)
    return _weighted_loss(np.sum(y_true * np.log(y_true / y_pred), axis=-1
                                 ), sample_weight)


def poisson(y_true, y_pred, sample_weight=1):
    return _weighted_loss(np.mean(y_pred - y_true * np.log(y_pred + EPS),
                                  axis=-1), sample_weight)


def cosine_proximity(y_true, y_pred, sample_weight=1):
    def _l2_normalize(x, axis=-1, eps=1e-7):
        return x / np.sqrt(np.maximum(np.sum(x**2), eps))
    
    y_true = _l2_normalize(y_true, axis=-1)
    y_pred = _l2_normalize(y_pred, axis=-1)
    return _weighted_loss(-np.sum(y_true * y_pred, axis=-1), sample_weight)


def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return np.equal(np.argmax(y_true, axis=-1),
                    np.argmax(y_pred, axis=-1)).astype('float32')


def sparse_categorical_accuracy(y_true, y_pred):
    # flatten y_true in case it's in shape (num_samples, 1)
    # instead of (num_samples,)
    return np.equal(y_true.flatten(),
                    np.argmax(y_pred, axis=-1).astype('float32')
                    ).astype('float32')


def tpr(y_true, y_pred, pred_threshold=0.5):
    ones_preds = y_pred[np.where(y_true == 1)]
    return np.mean(np.ceil(ones_preds - pred_threshold) == 1)


def tnr(y_true, y_pred, pred_threshold=0.5):
    zeros_preds = y_pred[np.where(y_true == 0)]
    return np.mean(np.ceil(zeros_preds - pred_threshold) == 1)


def binary_accuracies(y_true, y_pred, pred_threshold=0.5):
    return [tnr(y_true, y_pred, pred_threshold),
            tpr(y_true, y_pred, pred_threshold)]


def binary_informedness(y_true, y_pred, pred_threshold=0.5):
    return sum(binary_accuracies(y_true, y_pred, pred_threshold)) - 1
