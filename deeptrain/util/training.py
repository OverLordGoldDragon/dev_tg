# -*- coding: utf-8 -*-
import numpy as np

from .searching import find_best_predict_threshold, find_best_subset
from .searching import find_best_subset_from_history
from .introspection import l1l2_weight_loss
from . import metrics as metric_fns
from . import NOTE, WARN


def _update_temp_history(cls, metrics, val=False):
    def _get_metric_names(metrics, val):
        metric_names = cls.val_metrics if val else cls.train_metrics
        if not val or (val and cls.eval_fn_name == 'evaluate'):
            assert len(metric_names) == len(metrics)
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

    def _handle_non_scalar(name, value):
        supported = ('binary_accuracy', 'categorical_accuracy',
                     'sparse_categorical_accuracy')
        assert (name in supported), (
            f"got non-scalar value ({type(value)}) for metric {name}; "
            "supported metrics for non-scalars: %s" % ', '.join(supported))

        if not isinstance(value, (list, np.ndarray)):
            raise ValueError(("unexpected non-scalar metric type: {}; must be "
                              "one of: list, np.ndarray").format(type(value)))
        value = np.asarray(value)
        assert (value.ndim <= 1), ("unfamiliar metric.ndim: %s" % value.ndim)
        return value.mean()

    def _try_append_with_fix(temp_history):
        try:
            temp_history[name][-1].append(value)
        except:
            print(WARN, "unable to append to `temp_history`; OK if right "
                  "after load() -- attempting fix via append()...")
            temp_history[name].append(value)

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]
    metric_names = _get_metric_names(metrics, val)
    temp_history = _get_temp_history(val)
    no_slices, slice_idx, slices_per_batch = _get_slice_info(val)

    for name, value in zip(metric_names, metrics):
        if np.ndim(value) != 0 or isinstance(value, dict):
            value = _handle_non_scalar(name, value)

        if no_slices or slice_idx == 0:
            temp_history[name].append([])
        _try_append_with_fix(temp_history)

        if not no_slices and slice_idx == (slices_per_batch - 1):
            temp_history[name][-1] = np.mean(temp_history[name][-1])


def _get_sample_weight(cls, class_labels, val=False, slice_idx=None,
                       force_unweighted=False):
    def _get_unweighted(cls, class_labels, val):
        class_labels = _unroll_into_samples(len(cls.model.output_shape),
                                            class_labels)
        cw = cls.val_class_weights if val else cls.class_weights

        if cw is not None:
            if sum(cw.keys()) > 1 and class_labels.ndim == 2:
                class_labels = class_labels.argmax(axis=1)  # one-hot to dense
            return np.asarray([cw[int(l)] for l in class_labels])
        return np.ones(class_labels.shape[0])

    def _get_weighted(cls, class_labels, val, slice_idx):
        return _get_weighted_sample_weight(
            cls, class_labels, val, cls.loss_weighted_slices_range, slice_idx)

    loss_weighted = bool(cls.loss_weighted_slices_range)
    pred_weighted = bool(cls.pred_weighted_slices_range)
    either_weighted = loss_weighted or pred_weighted
    get_weighted = loss_weighted or (val and either_weighted)

    if force_unweighted or not get_weighted:
        return _get_unweighted(cls, class_labels, val)
    else:
        return _get_weighted(cls, class_labels, val, slice_idx)


def _get_weighted_sample_weight(cls, class_labels_all, val=False,
                                weight_range=(0.5, 1.5), slice_idx=None):
    def _sliced_sample_weight(class_labels_all, slice_idx, val):
        sw_all = []
        for batch_labels in class_labels_all:
            sw_all.append([])
            for slice_labels in batch_labels:
                sw = _get_sample_weight(
                    cls, slice_labels, val, slice_idx,
                    force_unweighted=True)  # break recursion
                sw_all[-1].append(sw)
        sw_all = np.asarray(sw_all)
        if sw_all.ndim >= 3 and sw_all.shape[0] == 1:
            sw_all = sw_all.squeeze(axis=0)
        return sw_all

    # `None` as in not passed in, not datagen-absent
    validate_n_slices = slice_idx is None
    class_labels_all = _validate_class_data_shapes(
        cls, {'class_labels_all': class_labels_all},
        validate_n_slices, val)

    dg = cls.val_datagen if val else cls.datagen
    n_slices = dg.slices_per_batch

    sw = _sliced_sample_weight(class_labels_all, slice_idx, val)
    sw = _validate_class_data_shapes(cls, {'sample_weight_all': sw},
                                     validate_n_slices, val)
    sw_weights = np.linspace(*weight_range, n_slices).reshape(
        [1, n_slices] + [1]*(sw.ndim - 2))

    sw = sw * sw_weights
    if slice_idx is not None:
        sw = sw[:, slice_idx]
    return sw.squeeze()


def _set_predict_threshold(cls, predict_threshold, for_current_iter=False):
    if not for_current_iter:
        cls.dynamic_predict_threshold = predict_threshold
    cls.predict_threshold = predict_threshold


def _get_val_history(cls, for_current_iter=False):
    if cls.best_subset_size and not for_current_iter:
        return _get_best_subset_val_history(cls)

    if cls.eval_fn_name == 'evaluate':
        return {metric: np.mean(values) for metric, values in
                cls.val_temp_history.items()}

    def _find_and_set_predict_threshold():
        pred_threshold = find_best_predict_threshold(
            labels_all_norm, preds_all_norm, cls.key_metric_fn,
            search_interval=.01,
            search_min_max=cls.dynamic_predict_threshold_min_max)
        _set_predict_threshold(cls, pred_threshold, for_current_iter)

    def _unpack_and_transform_data(for_current_iter):
        if for_current_iter:
            labels_all = cls._labels_cache[-1].copy()
            preds_all  = cls._preds_cache[-1].copy()
            sample_weight_all = cls._sw_cache[-1].copy()
            class_labels_all = cls._class_labels_cache[-1].copy()
        else:
            labels_all = cls._labels_cache.copy()
            preds_all  = cls._preds_cache.copy()
            sample_weight_all = cls._sw_cache.copy()
            class_labels_all = cls._class_labels_cache.copy()
        return _transform_eval_data(cls, labels_all, preds_all,
                                    sample_weight_all, class_labels_all,
                                    return_as_dict=False)

    # `class_labels_all` currently unused; may be useful in the future
    (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all
     ) = _unpack_and_transform_data(for_current_iter)

    if cls.dynamic_predict_threshold_min_max is not None:
        _find_and_set_predict_threshold()

    return _compute_metrics(cls, sample_weight_all,
                            labels_all_norm, preds_all_norm)


def _get_best_subset_val_history(cls):
    def _unpack_and_transform_data():
        labels_all = cls._labels_cache.copy()
        preds_all  = cls._preds_cache.copy()
        sample_weight_all = cls._sw_cache.copy()
        class_labels_all = cls._class_labels_cache.copy()
        return _transform_eval_data(cls, labels_all, preds_all,
                                    sample_weight_all, class_labels_all,
                                    unroll_into_samples=False)

    def _find_best_subset_from_preds(d):
        def _merge_slices_samples(*arrs):
            ls = []
            for x in arrs:
                ls.append(x.reshape(x.shape[0], x.shape[1] * x.shape[2],
                                    *x.shape[3:]))
            return ls

        if 'pred_threshold' not in cls.key_metric_fn.__code__.co_varnames:
            search_min_max = None
        elif cls.dynamic_predict_threshold_min_max is None:
            search_min_max = (cls.predict_threshold, cls.predict_threshold)
        else:
            search_min_max = cls.dynamic_predict_threshold_min_max

        la_norm, pa_norm = _merge_slices_samples(d['labels_all_norm'],
                                                 d['preds_all_norm'])
        best_subset_idxs, pred_threshold, _ = find_best_subset(
            la_norm, pa_norm,
            search_interval=.01,
            search_min_max=search_min_max,
            metric_fn=cls.key_metric_fn,
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

        assert best_subset_idxs, "`best_subset_idxs` is empty"

        ALL = _filter_by_indices(best_subset_idxs, d['labels_all_norm'],
                                 d['preds_all_norm'], d['sample_weight_all'])
        (sample_weight_all, preds_all_norm, labels_all_norm
         ) = _unroll_into_samples(len(cls.model.output_shape), *ALL)
        return _compute_metrics(cls, sample_weight_all, labels_all_norm,
                                preds_all_norm)

    if cls.eval_fn_name == 'evaluate':
        best_subset_idxs = _find_best_subset_from_history()
    elif cls.eval_fn_name == 'predict':
        d = _unpack_and_transform_data()
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


def _get_api_metric_name(name, loss_name, alias_to_metric_name_fn=None):
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
    if alias_to_metric_name_fn is not None:
        api_name = alias_to_metric_name_fn(api_name)
    return api_name


def _compute_metric(data, metric_name=None, metric_fn=None):
    def _del_if_not_in_metric_fn(name, data, metric_fn):
        if name in data and name not in metric_fn.__code__.co_varnames:
            del data[name]

    if metric_name is not None:
        metric_fn = getattr(metric_fns, metric_name)
    _del_if_not_in_metric_fn('pred_threshold', data, metric_fn)
    _del_if_not_in_metric_fn('sample_weight', data, metric_fn)
    return metric_fn(**data)


def _compute_metrics(cls, sample_weight_all, labels_all_norm, preds_all_norm):
    def _ensure_scalar_metrics(metrics):
        def _ensure_is_scalar(metric):
            if np.ndim(metric) != 0:
                assert (metric.ndim <= 1), (
                    "unfamiliar metric.ndim: %s" % metric.ndim)
                metric = metric.mean()
            return metric

        for name, metric in metrics.items():
            if isinstance(metric, list):
                for i, m in enumerate(metric):
                    metrics[name][i] = _ensure_is_scalar(m)
            else:
                metrics[name] = _ensure_is_scalar(metric)
        return metrics

    metric_names = cls.val_metrics.copy()
    metrics = {}
    for name in metric_names:
        api_name = _get_api_metric_name(name, cls.model.loss)
        data = dict(y_true=labels_all_norm,
                    y_pred=preds_all_norm,
                    sample_weight=sample_weight_all,
                    pred_threshold=cls.predict_threshold)

        if name == 'loss' or name == cls.key_metric:
            metrics[name] = _compute_metric(data, metric_fn=cls.key_metric_fn)
            if name == 'loss' or name[-1] == '*':
                metrics[name] += l1l2_weight_loss(cls.model)
        else:
            metrics[name] = _compute_metric(data, metric_name=api_name)

    metrics = _ensure_scalar_metrics(metrics)
    return metrics


def _unroll_into_samples(out_ndim, *arrs):
    ls = []
    for x in arrs:
        if x.shape == out_ndim:  # one batch, nothing to unroll
            ls.append(x)
            continue
        # unroll along non-out (except samples) dims
        x = x.reshape(-1, *x.shape[-(out_ndim - 1):])
        while x.shape[0] == 1:  # collapse non-sample dims
            x = x.squeeze(axis=0)
        ls.append(x)
    return ls if len(ls) > 1 else ls[0]


def _transform_eval_data(cls, labels_all, preds_all, sample_weight_all,
                         class_labels_all, return_as_dict=True,
                         unroll_into_samples=True):
    def _transform_labels_and_preds(labels_all, preds_all, sample_weight_all,
                                    class_labels_all):
        d = _validate_data_shapes(cls, {'labels_all': labels_all,
                                        'preds_all': preds_all})
        labels_all, preds_all = d['labels_all'], d['preds_all']

        # if `loss_weighted_slices_range` but not `pred_weighted_slices_range`,
        # will apply weighted sample weights on non weight-normalized preds
        if cls.pred_weighted_slices_range is not None:
            preds_all_norm = _weighted_normalize_preds(cls, preds_all)
            labels_all_norm = labels_all[:, :1]  # collapse but keep slices dim
            assert (preds_all_norm.max() <= 1) and (preds_all_norm.min() >= 0)
        else:
            preds_all_norm = preds_all
            labels_all_norm = labels_all

        d = _validate_class_data_shapes(cls,
                                        {'sample_weight_all': sample_weight_all,
                                         'class_labels_all': class_labels_all})
        if cls.pred_weighted_slices_range is not None:
            d['sample_weight_all'] = d['sample_weight_all'].mean(axis=1)
            d['class_labels_all'] = d['class_labels_all'][:, :1]
        return (labels_all_norm, preds_all_norm, d['sample_weight_all'],
                d['class_labels_all'])

    (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all,
     ) = _transform_labels_and_preds(labels_all, preds_all, sample_weight_all,
                                     class_labels_all)

    data = (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all)
    if unroll_into_samples:
        data = _unroll_into_samples(len(cls.model.output_shape), *data)

    if return_as_dict:
        names = ('labels_all_norm', 'preds_all_norm', 'sample_weight_all',
                 'class_labels_all')
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

    return preds_norm


def _validate_data_shapes(cls, data, validate_n_slices=True, val=True):
    """Ensures `data` entires are shaped (batches, slices, *model.output_shape)
    """
    def _validate_batch_size(data, outs_shape):
        batch_size = outs_shape[0]
        if batch_size is None:
            batch_size = cls.batch_size or cls._inferred_batch_size
            assert batch_size is not None

        for name, x in data.items():
            assert (batch_size in x.shape), (
                f"`{name}.shape` must include batch_size (={batch_size}) "
                f"{x.shape}")
        return batch_size

    def _validate_iter_ndim(data, slices_per_batch, ndim):
        if slices_per_batch is not None:
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
        x = list(data.values())[0]
        assert all([y.shape == x.shape for y in data.values()])

    def _validate_n_slices(data, slices_per_batch):
        if slices_per_batch is not None:
            x = list(data.values())[0]
            assert (slices_per_batch in x.shape), x.shape

    for name in data:
        data[name] = np.asarray(data[name])

    outs_shape = list(cls.model.output_shape)
    outs_shape[0] = _validate_batch_size(data, outs_shape)
    outs_shape = tuple(outs_shape)

    ndim = len(outs_shape)
    slices_per_batch = getattr(cls.val_datagen if val else cls.datagen,
                               'slices_per_batch', None)
    data = _validate_iter_ndim(data, slices_per_batch, ndim)

    _validate_last_dims_match_outs_shape(data, outs_shape, ndim)
    _validate_equal_shapes(data)
    if validate_n_slices:
        _validate_n_slices(data, slices_per_batch)

    return data if len(data) > 1 else list(data.values())[0]


def _validate_class_data_shapes(cls, data, validate_n_slices=False, val=True):
    """sample_weight and class_labels data"""
    def _validate_batch_size(data, outs_shape):
        batch_size = outs_shape[0]
        if batch_size is None:
            batch_size = cls.batch_size or cls._inferred_batch_size
            assert batch_size is not None

        for name, x in data.items():
            assert (batch_size in x.shape), (
                f"`{name}.shape` must include batch_size (={batch_size}) "
                f"{x.shape}")
        return batch_size

    def _validate_iter_ndim(data, slices_per_batch, ndim):
        if slices_per_batch is not None:
            expected_iter_ndim = ndim + 2  # (batches, slices)+
        else:
            expected_iter_ndim = ndim + 1  # (batches,)+

        for name in data:
            while data[name].ndim < expected_iter_ndim:
                data[name] = np.expand_dims(data[name], 0)
        return data

    def _validate_n_slices(data, slices_per_batch):
        if slices_per_batch is not None:
            for name, x in data.items():
                assert (slices_per_batch in x), (f"{name} -- {x.shape}")

    for k, v in data.items():
        data[k] = np.asarray(v)

    outs_shape = list(cls.model.output_shape)
    batch_size = _validate_batch_size(data, outs_shape)
    outs_shape[0] = batch_size
    outs_shape = tuple(outs_shape)
    ndim = len(outs_shape)
    slices_per_batch = getattr(cls.val_datagen if val else cls.datagen,
                               'slices_per_batch', None)

    data = _validate_iter_ndim(data, slices_per_batch, ndim)
    if validate_n_slices:
        _validate_n_slices(data, slices_per_batch)

    return data if len(data) > 1 else list(data.values())[0]
