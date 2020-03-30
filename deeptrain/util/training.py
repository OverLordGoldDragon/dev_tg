# -*- coding: utf-8 -*-
from .searching import get_best_predict_threshold, find_best_subset
from .searching import find_best_subset_from_history
from .introspection import l1l2_weight_loss
from . import metrics as metric_fns
from . import NOTE, WARN
from . import TF_KERAS

import numpy as np


def _update_temp_history(cls, metrics, val=False):
    def _validate_temp_history_and_names(temp_history, name_aliases):
        for name in name_aliases:
            if name not in temp_history:
                name_aliases.pop(name_aliases.index(name))
                continue
            if not isinstance(temp_history[name], list):
                print(NOTE, "`temp_history` is non-list; attempting casting")
                temp_history[name] = list(temp_history[name])
                assert isinstance(temp_history[name], list)
        return temp_history

    def _try_append_with_fix(temp_history):
        try:
            temp_history[name][-1].append(value)
        except:
            print(WARN, "unable to append to `temp_history`; OK if right "
                  "after load() -- attempting fix via append()...")
            temp_history[name].append(value)

    metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]

    if TF_KERAS:
        metric_names = ['loss', cls.model.metrics[0]._name]  # TODO multi
    else:
        metric_names = ['loss', *cls.model.metrics]
    assert len(metric_names) == len(metrics)
    name_aliases = [cls._alias_to_metric_name(n) for n in metric_names]

    for name in cls.train_metrics:
        assert (name in name_aliases)

    temp_history = cls.val_temp_history if val else cls.temp_history
    datagen = cls.val_datagen if val else cls.datagen
    no_slices = getattr(datagen, 'slice_idx', None) is None
    temp_history = _validate_temp_history_and_names(temp_history, name_aliases)

    for name, value in zip(name_aliases, metrics):
        if no_slices or datagen.slice_idx == 0:
            temp_history[name].append([])
        _try_append_with_fix(temp_history)

        if not no_slices and datagen.slice_idx == (datagen.slices_per_batch - 1):
            temp_history[name][-1] = np.mean(temp_history[name][-1])


def _get_sample_weight(cls, labels, val=False):
    cw = cls.val_class_weights if val else cls.class_weights
    labels = np.asarray(labels).squeeze()
    if cw is None:
        return np.ones(len(labels))

    if sum(cw.keys()) > 1 and labels.ndim == 2:
        labels = labels.argmax(axis=1)  # one-hot to dense
    return np.array([cw[int(y)] for y in labels])


def _get_weighted_sample_weight(cls, labels_all, val=False,
                                 weight_range=(0.5, 1.5), slice_idx=None):        
    def _standardize_labels(cls, labels_all, val, slice_idx):
        # CASES:
        #  a. (batches, slices, samples, X)
        #  b. (1, slices, samples, X)
        #  c. (slices, samples, X)
        #  d. (1, samples, X)
        #  e. (samples, X)
        #
        #  1-5:   X = n_classes
        #  6-10:  X = 1
        #  11-15: X = None (missing)

        def _infer_task(cls, labels_all, val):
            cw = cls.val_class_weights if val else cls.class_weights
            if cw is None:  # some checks will skip
                return None
            if sum(cw.keys()) > 1:
                if labels_all.ndim == 1:
                    return 'sparse_categorical'
                elif labels_all.ndim == 2:
                    return 'categorical'
            return 'binary'

        labels_all = np.asarray(labels_all)
        task = _infer_task(cls, labels_all, val)

        if task == 'sparse_categorical':
            assert (labels_all.shape[-1] != 1), ("last dim must be > 1 for "
                                                 "'sparse_categorical' labels")
        elif task in ('binary', 'categorical'):
            if labels_all.shape[-1] != 1:
                labels_all = np.expand_dims(labels_all, -1)
        while labels_all.ndim < 4:  # ensure dims (batches, slices, samples, y)
            labels_all = np.expand_dims(labels_all, 0)

        if slice_idx is None:  # `None` as in not passed in, not dg-absent
            dg = cls.val_datagen if val else cls.datagen
            assert (labels_all.shape[1] == dg.slices_per_batch), (
                "num slices in `labels_all` doesn't match num slices in datagen")
        return labels_all

    def _sliced_sample_weight(cls, labels_all, val):
        sw_all = []
        for batch_labels in labels_all:
            sw_all.append([])
            for slice_labels in batch_labels:
                sw = _get_sample_weight(cls, slice_labels, val)
                sw_all[-1].append(sw)
        sw_all = np.asarray(sw_all)
        if sw_all.ndim == 3:
            sw_all = sw_all.squeeze(axis=0)
        return sw_all
    
    labels_all = _standardize_labels(cls, labels_all, val, slice_idx)

    dg = cls.val_datagen if val else cls.datagen
    n_slices = dg.slices_per_batch

    sw_weights = np.zeros((n_slices, 1, 1))
    sw_weights[:, 0, 0] = np.linspace(*weight_range, n_slices)

    sw = _sliced_sample_weight(cls, labels_all, val) * sw_weights
    sw = sw if slice_idx is None else sw[slice_idx]
    return sw.squeeze()


def _compute_metric(metric_name, y_true, y_pred, sample_weight=None,
                    pred_threshold=None):
    kw1 = dict(y_true=y_true, y_pred=y_pred, pred_threshold=pred_threshold)
    kw2 = dict(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    non_loss = ('tpr', 'tnr', 'f1-score', 'categorical_accuracy',
                'sparse_categorical_accuracy', 'binary_accuracy', 
                'tnr_tpr', 'binary_informedness')
    
    if metric_name in non_loss:
        return getattr(metric_fns, metric_name)(**kw1)
    else:
        return getattr(metric_fns, metric_name)(**kw2)


def _set_predict_threshold(cls, predict_threshold, for_current_iter=False):
    if cls.use_dynamic_predict_threshold:
        if not for_current_iter:
            cls.dynamic_predict_threshold = predict_threshold
        cls.predict_threshold = predict_threshold
    else:
        cls.predict_threshold = cls.static_predict_threshold 
            

def _get_val_history(cls, for_current_iter=False):    
    if (cls.best_subset_size != 0) and not for_current_iter:
        return _get_best_subset_val_history(cls)

    if cls.eval_fn_name == 'evaluate':
        return {metric: np.mean(values) for metric, values in
                cls.val_temp_history.items()}

    def _unpack_data(cls):
        if for_current_iter:
            labels_all = cls._labels_cache[-1].copy()
            preds_all  = cls._preds_cache[-1].copy()
            sample_weight_all = cls._sw_cache[-1].copy()
        else:
            labels_all = cls._labels_cache.copy()
            preds_all  = cls._preds_cache.copy()
            sample_weight_all = cls._sw_cache.copy()
        return {'labels_all': labels_all, 'preds_all': preds_all,
                'sample_weight_all': sample_weight_all}
    
    def _get_api_metric_name(name, model):
        if name == 'loss':
            api_name = model.loss
        elif name in ('accuracy', 'acc'):
            if model.loss[0] == 'categorical_crossentropy':
                api_name = 'categorical_accuracy'
            elif model.loss[0] == 'sparse_categorical_crossentropy':
                api_name = 'sparse_categorical_accuracy'
            else:
                api_name = 'binary_accuracy'
        else:
            api_name = name
        return api_name

    d = _unpack_data(cls)
    (labels_all, preds_all, sample_weight_all, 
     preds_all_norm, labels_all_norm) = _transform_eval_data(
         cls, d['labels_all'], d['preds_all'], d['sample_weight_all'],
         return_as_dict=False)
    
    pred_threshold, key_metric = get_best_predict_threshold(
        labels_all_norm, preds_all_norm, cls.key_metric_fn,
        search_interval=.01,
        search_min_max=cls.dynamic_predict_threshold_min_max,
        return_best_metric=True)
    _set_predict_threshold(cls, pred_threshold, for_current_iter)

    metric_names = cls.val_metrics.copy()
    metric_names.remove(cls.key_metric)  # already computed
    metrics = {}
    for name in metric_names:
        api_name = _get_api_metric_name(name, cls.model)
        kw = dict(metric_name=api_name, y_true=labels_all_norm, 
                  y_pred=preds_all_norm,
                  sample_weight=sample_weight_all,
                  pred_threshold=cls.predict_threshold)

        if name == 'loss':
            kw['_y_true'], kw['_y_pred'] = labels_all, preds_all
            metrics[name] = _compute_metric(**kw)
            metrics[name] += l1l2_weight_loss(cls.model)
        else:
            metrics[name] = _compute_metric(**kw)

    metrics[cls.key_metric] = key_metric
    return metrics


def _get_best_subset_val_history(cls):
    def _unpack_data(cls):
        def _restore_flattened(data):
            batch_size = cls.batch_size
            n_slices  = cls.val_datagen.slices_per_batch or 1
            n_batches = len(cls.val_datagen.superbatch)
            restored = []
            for x in data.values():
                if x.size == (batch_size * n_slices * n_batches):
                    restored += [x.reshape(n_batches, batch_size, n_slices)]
                else:
                    restored += [x.reshape(n_batches, batch_size)]
            return {name: x for name, x in zip(data, restored)}
    
        labels_all = cls._labels_cache.copy()
        preds_all  = cls._preds_cache.copy()
        sample_weight_all = cls._sw_cache.copy()
        return _restore_flattened(
            *_transform_eval_data(cls, labels_all, preds_all, sample_weight_all))

    def _find_best_subset_from_preds(cls, d):
        best_subset_idxs, pred_threshold, _ = find_best_subset(
            d['labels_all_norm'], d['preds_all_norm'], 
            search_interval=.01,
            search_min_max=cls.dynamic_predict_threshold_min_max,
            metric_fn=getattr(metric_fns, cls.key_metric),
            subset_size=cls.best_subset_size)
        return best_subset_idxs, pred_threshold

    def _find_best_subset_from_history(cls):
        metric = cls.val_temp_history[cls.key_metric]
        best_subset_idxs = find_best_subset_from_history(
            metric, cls.best_subset_size, cls.max_is_best)
        return best_subset_idxs
    
    def _best_subset_metrics_from_history(cls, best_subset_idxs):
        return {name: np.asarray(metric)[best_subset_idxs].mean()
                for name, metric in cls.val_temp_history.items()}
        
    def _best_subset_metrics_from_preds(cls, d, best_subset_idxs):
        def _filter_by_indices(indices, *arrs):
            return [np.asarray([x[idx] for idx in indices]) for x in arrs]
    
        def _flat(*arrs):
            return [np.asarray(x).flatten() for x in arrs]

        ALL = _filter_by_indices(best_subset_idxs, d['labels_all'], 
                                 d['preds_all'], d['sample_weight_all'], 
                                 d['preds_all_norm'], d['labels_all_norm'])
        (labels_all, preds_all, sample_weight_all, preds_all_norm,
         labels_all_norm) = _flat(*ALL)
    
        metric_names = cls.val_metrics.copy()
        metrics = {}
        for name in metric_names:
            kw = dict(metric_name=name, y_true=labels_all_norm, 
                      y_pred=preds_all_norm,
                      sample_weight=sample_weight_all, 
                      pred_threshold=cls.predict_threshold)
            if name == 'loss':
                kw['_y_true'], kw['_y_pred'] = labels_all, preds_all
                kw['metric_name'] = cls.model.loss  # TODO multiple losses
                metrics[name] = _compute_metric(**kw)
                metrics[name] += l1l2_weight_loss(cls.model)
            else:
                metrics[name] = _compute_metric(**kw)
        return metrics
    
    if cls.eval_fn_name == 'evaluate':
        best_subset_idxs = _find_best_subset_from_history(cls)
    elif cls.eval_fn_name == 'predict':
        d = _unpack_data(cls)
        best_subset_idxs, pred_threshold = _find_best_subset_from_preds(cls, d)
        _set_predict_threshold(cls, pred_threshold)
    else:
        raise ValueError("unknown `eval_fn_name`: %s" % cls.eval_fn_name)

    cls.best_subset_nums = np.array(cls._val_set_name_cache)[best_subset_idxs]
    if cls.eval_fn_name == 'evaluate':
        return _best_subset_metrics_from_history(cls, best_subset_idxs)
    else:
        return _best_subset_metrics_from_preds(cls, d, best_subset_idxs)


def _transform_eval_data(cls, labels_all, preds_all, sample_weight_all,
                         return_as_dict=True):
    if cls.weighted_slices_range is not None:
        def _check_and_fix_dims(labels_all, preds_all):
            labels_all, preds_all = np.array(labels_all), np.array(preds_all)
            if labels_all.ndim == 1:
                labels_all = labels_all.reshape(-1, 1)
            if preds_all.ndim == 1:
                preds_all = preds_all.reshape(-1, 1)
            
            assert labels_all.shape == preds_all.shape
            if getattr(cls.val_datagen, 'slices_per_batch', None) is not None:
                assert cls.val_datagen.slices_per_batch in preds_all.shape
            return labels_all, preds_all

        ##TODO: sample weights
        labels_all, preds_all = _check_and_fix_dims(labels_all, preds_all)
        if cls.val_datagen.slices_per_batch is not None:
            preds_all_norm = cls._normalize_preds(preds_all)
        else:
            preds_all_norm = preds_all
        sample_weight_all = cls.get_sample_weight(labels_all, val=True)
        labels_all_norm = np.array([labels[0] for labels in labels_all])
    else:
        preds_all  = np.vstack(preds_all).flatten()
        labels_all = np.vstack(labels_all).flatten()
        sample_weight_all = np.asarray(sample_weight_all).flatten()
        preds_all_norm = preds_all
        labels_all_norm = labels_all
    
    if cls.eval_fn_name == 'predict':
        assert (preds_all_norm.max() <= 1) and (preds_all_norm.min() >= 0)
    
    data = (labels_all, preds_all, sample_weight_all, 
            preds_all_norm, labels_all_norm)
    if return_as_dict:
        names = ('labels_all', 'preds_all', 'sample_weight_all',
                 'preds_all_norm', 'labels_all_norm')
        return {name: x.flatten() for name, x in zip(names, data)}
    else:
        return data

def _normalize_preds(cls, preds_all):
    n_slices = preds_all.shape[1]
    assert (n_slices == cls.val_datagen.slices_per_batch), ("`n_slices` "
            "inferred from `preds_all` != `val_datagen.slices_per_batch`")
    if n_slices == 1:
        return preds_all
    slice_weights = np.linspace(*cls.weighted_slices_range, n_slices)

    weighted_preds = []
    for preds in preds_all:
        weighted_preds.append([])
        for slice_idx, pred in enumerate(preds):
            additive_pred = pred - .5
            weighted_preds[-1] += [additive_pred * slice_weights[slice_idx]]

    weight_norm = np.sum(slice_weights)
    preds_norm = np.sum(np.array(weighted_preds), axis=1) / weight_norm + .5
    
    # fix possible float imprecision
    if np.min(preds_norm) < 0:
        preds_norm -= np.min(preds_norm)
    if np.max(preds_norm) > 1:
        preds_norm -= np.max(preds_norm)
    return preds_norm
