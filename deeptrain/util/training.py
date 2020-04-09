# -*- coding: utf-8 -*-
from .searching import find_best_predict_threshold, find_best_subset
from .searching import find_best_subset_from_history
from .introspection import l1l2_weight_loss
from . import metrics as metric_fns
from . import NOTE, WARN
from . import TF_KERAS

import numpy as np


def _update_temp_history(cls, metrics, val=False):
    def _get_metric_names(metrics):
        # TODO `metrics` should come pre-packed with correct names,
        # validate in misc.validate_kwargs instead
        # TODO redundant check, e.g. 'f1_score' can't be in model.metrics
        if TF_KERAS:
            metric_names = ['loss', cls.model.metrics[0]._name]  # TODO multi
        else:
            metric_names = ['loss', *cls.model.metrics]
        metric_names = [cls._alias_to_metric_name(n) for n in metric_names]

        if cls.eval_fn_name == 'evaluate':
            assert len(metric_names) == len(metrics)
            check_metrics = cls.val_metrics if val else cls.train_metrics
            for name in check_metrics:
                assert (name in metric_names)
        return metric_names

    def _get_temp_history(val):
        temp_history = cls.val_temp_history if val else cls.temp_history
        temp_history = _validate_temp_history(temp_history)
        return temp_history

    def _get_slice_info(val):
        datagen = cls.val_datagen if val else cls.datagen
        slice_idx = getattr(datagen, 'slice_idx', None)
        no_slices = slice_idx is None
        slices_per_batch = getattr(datagen, 'slices_per_batch', None)
        return no_slices, slice_idx, slices_per_batch

    def _validate_temp_history(temp_history):
        for name, value in temp_history.items():
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

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]
    metric_names = _get_metric_names(metrics)
    temp_history = _get_temp_history(val)
    no_slices, slice_idx, slices_per_batch = _get_slice_info(val)

    for name, value in zip(metric_names, metrics):
        if no_slices or slice_idx == 0:
            temp_history[name].append([])
        _try_append_with_fix(temp_history)

        if not no_slices and slice_idx == (slices_per_batch - 1):
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
    def _sliced_sample_weight(labels_all, val):
        sw_all = []
        for batch_labels in labels_all:
            sw_all.append([])
            for slice_labels in batch_labels:
                sw = _get_sample_weight(cls, slice_labels, val)
                sw_all[-1].append(sw)
        sw_all = np.asarray(sw_all)
        if sw_all.ndim >= 3 and sw_all.shape[0] == 1:
            sw_all = sw_all.squeeze(axis=0)
        return sw_all

    do_val = slice_idx is None  # `None` as in not passed in, not dg-absent
    labels_all = _validate_data_shapes(cls, {'labels_all': labels_all},
                                       validate_n_slices=do_val)['labels_all']
    dg = cls.val_datagen if val else cls.datagen
    n_slices = dg.slices_per_batch

    sw = _sliced_sample_weight(labels_all, val)
    sw = _validate_sample_weight_shape(cls, sw, validate_n_slices=do_val)
    sw_weights = np.linspace(*weight_range, n_slices).reshape(
        [1, n_slices] + [1]*(sw.ndim - 2))

    sw = sw * sw_weights
    if slice_idx is not None:
        sw = sw[:, slice_idx]
    return sw.squeeze()


def _get_val_history(cls, for_current_iter=False):
    if cls.best_subset_size and not for_current_iter:
        return _get_best_subset_val_history(cls)

    if cls.eval_fn_name == 'evaluate':
        return {metric: np.mean(values) for metric, values in
                cls.val_temp_history.items()}

    def _unpack_data():
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

    def _find_and_set_predict_threshold():
        pred_threshold = find_best_predict_threshold(
            labels_all_norm, preds_all_norm, cls.key_metric_fn,
            search_interval=.01,
            search_min_max=cls.dynamic_predict_threshold_min_max)
        _set_predict_threshold(cls, pred_threshold, for_current_iter)

    d = _unpack_data()
    (labels_all, preds_all, sample_weight_all,
     preds_all_norm, labels_all_norm) = _transform_eval_data(
         cls, d['labels_all'], d['preds_all'], d['sample_weight_all'],
         return_as_dict=False)

    if cls.dynamic_predict_threshold_min_max is not None:
        _find_and_set_predict_threshold()

    return _compute_metrics(cls, labels_all, preds_all, sample_weight_all,
                            labels_all_norm, preds_all_norm)


def _get_best_subset_val_history(cls):
    def _unpack_data():
        def _restore_flattened(data):
            batch_size = cls.batch_size or cls._inferred_batch_size
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
            _transform_eval_data(cls, labels_all, preds_all, sample_weight_all))

    def _find_best_subset_from_preds(d):
        metric_fn = getattr(
            metric_fns, _get_api_metric_name(cls.key_metric, cls.model.loss))
        if 'pred_threshold' not in metric_fn.__code__.co_varnames:
            search_min_max = None
        elif cls.dynamic_predict_threshold_min_max is None:
            search_min_max = (cls.predict_threshold, cls.predict_threshold)
        else:
            search_min_max = cls.dynamic_predict_threshold_min_max

        best_subset_idxs, pred_threshold, _ = find_best_subset(
            d['labels_all_norm'], d['preds_all_norm'],
            search_interval=.01,
            search_min_max=search_min_max,
            metric_fn=metric_fn,
            subset_size=cls.best_subset_size)
        return best_subset_idxs, pred_threshold

    def _find_best_subset_from_history():
        metric = cls.val_temp_history[cls.key_metric]
        best_subset_idxs = find_best_subset_from_history(
            metric, cls.best_subset_size, cls.max_is_best)
        return best_subset_idxs

    def _best_subset_metrics_from_history(best_subset_idxs):
        return {name: np.asarray(metric)[best_subset_idxs].mean()
                for name, metric in cls.val_temp_history.items()}

    def _best_subset_metrics_from_preds(d, best_subset_idxs):
        def _filter_by_indices(indices, *arrs):
            return [np.asarray([x[idx] for idx in indices]) for x in arrs]

        ALL = _filter_by_indices(best_subset_idxs, d['labels_all'],
                                 d['preds_all'], d['sample_weight_all'],
                                 d['preds_all_norm'], d['labels_all_norm'])
        (labels_all, preds_all, sample_weight_all, preds_all_norm,
         labels_all_norm) = _unroll_samples(cls.model.output_shape, *ALL)

        return _compute_metrics(cls, labels_all, preds_all, sample_weight_all,
                                labels_all_norm, preds_all_norm)

    if cls.eval_fn_name == 'evaluate':
        best_subset_idxs = _find_best_subset_from_history()
    elif cls.eval_fn_name == 'predict':
        d = _unpack_data()
        best_subset_idxs, pred_threshold = _find_best_subset_from_preds(d)
        if cls.dynamic_predict_threshold_min_max is not None:
            _set_predict_threshold(cls, pred_threshold)
    else:
        raise ValueError("unknown `eval_fn_name`: %s" % cls.eval_fn_name)

    cls.best_subset_nums = np.array(cls._val_set_name_cache)[best_subset_idxs]
    if cls.eval_fn_name == 'evaluate':
        return _best_subset_metrics_from_history(best_subset_idxs)
    else:
        return _best_subset_metrics_from_preds(d, best_subset_idxs)


def _compute_metric(data, metric_name=None, metric_fn=None):
    def _del_if_not_in_metric_fn(name, data, metric_fn):
        if name in data and name not in metric_fn.__code__.co_varnames:
            del data[name]

    if metric_name is not None:
        metric_fn = getattr(metric_fns, metric_name)
    _del_if_not_in_metric_fn('pred_threshold', data, metric_fn)
    _del_if_not_in_metric_fn('sample_weight', data, metric_fn)
    return metric_fn(**data)


def _get_api_metric_name(name, loss_name):
    if name == 'loss':
        api_name = loss_name
    elif name in ('accuracy', 'acc'):
        if loss_name == 'categorical_crossentropy':
            api_name = 'categorical_accuracy'
        elif loss_name == 'sparse_categorical_crossentropy':
            api_name = 'sparse_categorical_accuracy'
        else:
            api_name = 'binary_accuracy'
    else:
        api_name = name
    return api_name


def _compute_metrics(cls, labels_all, preds_all, sample_weight_all,
                     labels_all_norm, preds_all_norm):
    metric_names = cls.val_metrics.copy()
    metrics = {}
    for name in metric_names:
        api_name = _get_api_metric_name(name, cls.model.loss)
        data = dict(y_true=labels_all_norm,  # TODO remove `preds_all` cases?
                    y_pred=preds_all_norm,
                    sample_weight=sample_weight_all,
                    pred_threshold=cls.predict_threshold)

        if name == 'loss' or name == cls.key_metric:
            if 'sample_weight' in cls.key_metric_fn.__code__.co_varnames:
                data['y_true'], data['y_pred'] = labels_all, preds_all

            metrics[name] = _compute_metric(data, metric_fn=cls.key_metric_fn)
            if name == 'loss' or name[-1] == '*':
                metrics[name] += l1l2_weight_loss(cls.model)
            data['y_true'], data['y_pred'] = labels_all_norm, preds_all_norm
        else:
            metrics[name] = _compute_metric(data, metric_name=api_name)
    return metrics


def _unroll_samples(output_shape, *arrs):
    out_ndim = len(output_shape)
    ls = []
    for x in arrs:
        while x.shape[0] == 1:
            x = x.squeeze(axis=0)
        if x.shape != output_shape and x.shape[-out_ndim:] == output_shape:
            x = x.reshape(-1, *output_shape)
        ls.append(x)
    return ls


def _transform_eval_data(cls, labels_all, preds_all, sample_weight_all,
                         return_as_dict=True):
    d = {'labels_all': labels_all, 'preds_all': preds_all}
    d = _validate_data_shapes(cls, d)
    sample_weight_all = _validate_sample_weight_shape(cls, sample_weight_all)
    labels_all, preds_all = d['labels_all'], d['preds_all']

    if cls.pred_weighted_slices_range is not None:  # TODO check sw_all shape
        if cls.pred_weighted_slices_range is not None:
            preds_all_norm = _weighted_normalize_preds(cls, preds_all)
        else:
            preds_all_norm = preds_all
        sample_weight_all = cls.get_sample_weight(labels_all, val=True)
        labels_all_norm = np.array([labels[0] for labels in labels_all])
        assert (preds_all_norm.max() <= 1) and (preds_all_norm.min() >= 0)
    else:
        preds_all  = np.vstack(preds_all)
        labels_all = np.vstack(labels_all)
        sample_weight_all = np.vstack(sample_weight_all)
        preds_all_norm = preds_all
        labels_all_norm = labels_all

    data = (labels_all, preds_all, sample_weight_all,
            preds_all_norm, labels_all_norm)
    data = _unroll_samples(cls.model.output_shape, *data)

    if return_as_dict:
        names = ('labels_all', 'preds_all', 'sample_weight_all',
                 'preds_all_norm', 'labels_all_norm')
        return {name: x for name, x in zip(names, data)}
    else:
        return data


def _weighted_normalize_preds(cls, preds_all):
    def _validate_data(preds_all, n_slices):
        spb = cls.val_datagen.slices_per_batch
        assert (n_slices == spb), ("`n_slices` inferred from `preds_all` differs"
                                   " from `val_datagen.slices_per_batch` "
                                   "(%s != %s)" % (n_slices, spb))
        assert (np.asarray(preds_all).max() < 1.01), (
            "`max(preds_all) > 1`; can only normalize in (0, 1) range")
        assert (np.asarray(preds_all).min() > -.01), (
            "`min(preds_all) < 0`; can only normalize in (0, 1) range")

    n_slices = preds_all.shape[1]
    # validate even if n_slices == 1 to ensure expected behavior
    # in metrics computation
    _validate_data(preds_all, n_slices)

    if n_slices == 1:
        return preds_all
    slice_weights = np.linspace(*cls.pred_weighted_slices_range, n_slices)

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


def _validate_data_shapes(cls, data, validate_n_slices=True):
    """Ensures `data` entires are shaped (batches, slices, *model.output_shape)
    """
    def _validate_batch_size(data, outs_shape):
        batch_size = outs_shape[0]
        if batch_size is None:
            batch_size = cls.batch_size or cls._inferred_batch_size
            assert batch_size is not None

        for name, x in data.items():
            assert (batch_size in x.shape), (
                f"`{name}.shape` must include batch_size (={batch_size})")
        return batch_size

    def _validate_last_dim(data, outs_shape):
        for name, x in data.items():
            if x.shape[-1] != outs_shape[-1]:
                assert (outs_shape[-1] == 1), (
                    f"`{name}` shapes must contain model.output_shape "
                    "[%s != %s]" % (x.shape, outs_shape))
                data[name] = np.expand_dims(x, -1)
        return data

    def _validate_iter_ndim(data, ndim):
        if getattr(cls.val_datagen, 'slices_per_batch', None) is not None:
            expected_iter_ndim = ndim + 2  # +(batches, slices)
        else:
            expected_iter_ndim = ndim + 1  # +(batches,)

        for name in data:
            dndim = data[name].ndim
            if dndim > expected_iter_ndim:
                raise Exception(f"{name}.ndim exceeds `expected_iter_ndim` "
                                f"({dndim} > {expected_iter_ndim}) "
                                f"-- {data[name].shape}")
            while data[name].ndim < expected_iter_ndim:
                data[name] = np.expand_dims(data[name], 0)
        return data

    def _validate_last_dims_match_outs_shape(data, outs_shape, ndim):
        for name, x in data.items():
            assert (x.shape[-ndim:] == outs_shape), (
                f"last dims of `{name}` must equal model.output_shape "
                "[%s != %s]" % (x.shape[-ndim:], outs_shape))

    def _validate_equal_shapes(data):
        x = data[list(data.keys())[0]]
        assert all([y.shape == x.shape for y in data.values()])

    def _validate_n_slices(data):
        x = data[list(data.keys())[0]]
        if getattr(cls.val_datagen, 'slices_per_batch', None) is not None:
            assert cls.val_datagen.slices_per_batch in x.shape

    for name in data:
        data[name] = np.asarray(data[name])

    outs_shape = list(cls.model.output_shape)
    ndim = len(outs_shape)

    outs_shape[0] = _validate_batch_size(data, outs_shape)
    outs_shape = tuple(outs_shape)

    data = _validate_last_dim(data, outs_shape)
    data = _validate_iter_ndim(data, ndim)

    _validate_last_dims_match_outs_shape(data, outs_shape, ndim)
    _validate_equal_shapes(data)
    if validate_n_slices:
        _validate_n_slices(data)

    return data


# TODO
def _validate_sample_weight_shape(cls, sample_weight_all,
                                  validate_n_slices=False):
    def _validate_batch_size(x, outs_shape):
        batch_size = outs_shape[0]
        if batch_size is None:
            batch_size = cls.batch_size or cls._inferred_batch_size
            assert batch_size is not None

        assert (batch_size in x.shape), (
            f"`sample_weight_all.shape` must include batch_size (={batch_size})")
        return batch_size

    def _validate_last_dim(x, outs_shape):
        if x.shape[-1] != outs_shape[-1]:
            x = np.expand_dims(x, -1)
        return x

    def _validate_iter_ndim(x, ndim):
        if cls.pred_weighted_slices_range is not None:
            expected_iter_ndim = ndim + 2  # (batches, slices)+
        else:
            expected_iter_ndim = ndim + 1  # (batches,)+

        while x.ndim < expected_iter_ndim:
            x = np.expand_dims(x, 0)
        return x

    def _validate_last_dims_match_outs_shape(x, outs_shape, ndim):
        if x.shape[-1] != outs_shape[-1]:
            if outs_shape[-1] != 1:
                x_shape = x.shape[-ndim:-1]
                check_shape = outs_shape[:-1]
            else:
                x_shape = x.shape
                check_shape = outs_shape
            assert (x_shape == check_shape), (
                "last dims of `sample_weight_all` must equal model.output_shape"
                " [%s != %s]" % (x.shape, outs_shape))

    def _validate_n_slices(x):
        if getattr(cls.val_datagen, 'slices_per_batch', None) is not None:
            assert cls.val_datagen.slices_per_batch in x.shape

    x = np.asarray(sample_weight_all)
    outs_shape = list(cls.model.output_shape)
    ndim = len(outs_shape)

    outs_shape[0] = _validate_batch_size(x, outs_shape)
    outs_shape = tuple(outs_shape)

    x = _validate_last_dim(x, outs_shape)
    x = _validate_iter_ndim(x, ndim)

    _validate_last_dims_match_outs_shape(x, outs_shape, ndim)
    if validate_n_slices:
        _validate_n_slices(x)
    return x


def _set_predict_threshold(cls, predict_threshold, for_current_iter=False):
    if not for_current_iter:
        cls.dynamic_predict_threshold = predict_threshold
    cls.predict_threshold = predict_threshold
