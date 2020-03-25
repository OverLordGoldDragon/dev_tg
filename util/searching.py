# -*- coding: utf-8 -*-
from .metrics import f1_score, f1_score_multi_th
import numpy as np
from functools import reduce
from itertools import combinations

def get_best_predict_threshold(labels, preds, metric_function, 
                               search_interval=0.05, search_min_max=(0,1),
                               verbosity=0, return_best_metric=False,
                               threshold_preference=0.5):
    _min    = search_min_max[0] if search_min_max[0]!=0 else search_interval
    _max    = search_min_max[1] if search_min_max[1]!=1 else .9999
    th_pref = threshold_preference

    th       = _min
    best_th  = th
    best_acc = 0
    if verbosity == 2:  print("th", "acc", "best", sep="\t")

    while (th >= _min) and (th <= _max):
        acc = metric_function(labels, preds, th)
        if acc >= best_acc:
            if acc > best_acc or (abs(th - th_pref) < abs(best_th - th_pref)):
                best_th = round(th, 2)               # ^ find best closest to .5
            best_acc = acc

        if verbosity == 2:
            print("%.2f\t%.2f\t%.2f" % (th, acc, best_acc))
        th += search_interval
        
    if verbosity >= 1:
        print("Best predict th: %.2f w/ %.2f best acc" % (best_th, best_acc))
    
    return (best_th, best_acc) if return_best_metric else best_th


def find_best_subset(labels_all, preds_all, metric_fn=f1_score,
                     search_interval=.01, subset_size=5):
    def _get_batch_metrics(labels_all, preds_all, th, metric_fn):
        batch_metrics = []
        for (batch_labels, batch_preds) in zip(labels_all, preds_all):
            bl_flat, bp_flat = _flat(batch_labels, batch_preds)
            batch_metrics += [metric_fn(bl_flat, bp_flat, th)]
        return batch_metrics

    def _flat(*lists):
        return [np.asarray(_list).flatten() for _list in lists]

    def _compute_th_metrics(labels_all, preds_all, metric_fn, 
                            search_interval=.01, init_th=.05):
        th = np.linspace(init_th, 1, round((1 - init_th) / search_interval) + 1)
        return th, metric_fn(*_flat(labels_all, preds_all), th)

    def _find_best_batch_idx(labels_all, preds_all, metric_fn):
        th, th_metrics = _compute_th_metrics(labels_all, preds_all, 
                                             metric_fn=f1_score_multi_th,
                                             search_interval=search_interval)
        best_th_metric_idx = list(th_metrics).index(max(th_metrics))
        best_th = th[best_th_metric_idx]

        best_batch_metrics = _get_batch_metrics(labels_all, preds_all,
                                                best_th, metric_fn)
        best_batch_idx  = np.where(best_batch_metrics==
                                   np.max(best_batch_metrics))[0][0]
        return best_batch_idx

    labels_all, preds_all = list(labels_all), list(preds_all)
    batch_idxs = list(range(len(preds_all)))
    best_labels, best_preds, best_batch_idxs = [], [], []

    while len(best_batch_idxs) < subset_size:
        best_batch_idx   = _find_best_batch_idx(labels_all, preds_all, metric_fn)
        best_labels     += [labels_all.pop(best_batch_idx)]
        best_preds      += [preds_all.pop(best_batch_idx)]
        best_batch_idxs += [batch_idxs.pop(best_batch_idx)]

    best_labels_flat, best_preds_flat = _flat(best_labels, best_preds)
    best_th, best_metric = get_best_predict_threshold(best_labels_flat,
                                    best_preds_flat, metric_fn,
                                    search_interval, return_best_metric=True)
    return best_batch_idxs, best_th, best_metric


def nCk(n, k):  # n-Choose-k
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom

# TODO collapse slices as means, since searching per batch not per slice
def find_best_subset_from_history(metric, subset_size=5, max_is_best=False):
    def _find_best(metric, subset_size):
        indices = list(range(len(metric)))
        idx_metric_pairs = [[i, m] for i, m in zip(indices, metric)]
        idx_metric_pairs.sort(key=lambda x: x[1])
        return idx_metric_pairs[0][:subset_size]

    metric = np.asarray(metric)
    if metric.ndim > 1:  # (batches, slices)
        metric = metric.mean(axis=1)

    return _find_best(metric, subset_size)
