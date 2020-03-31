# -*- coding: utf-8 -*-
import numpy as np

from types import LambdaType
from . import WARN, NOTE


def pass_on_error(fn, *args, **kwargs):
    fail_msg = kwargs.pop('fail_msg', None)
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        print(fail_msg)
        print("Errmsg:", e)


def _dict_filter_keys(dc, keys, exclude=True, filter_substr=False):
    def condition(k, keys, exclude, filter_substr):
        if not filter_substr:
            value = k in keys
        else:
            value = any([(key in k) for key in keys])
        return (not value) if exclude else value

    keys = keys if isinstance(keys, (list, tuple)) else [keys]
    return {k:v for k,v in dc.items()
            if condition(k, keys, exclude, filter_substr)}


def ordered_shuffle(*args):
    zipped_args = list(zip(*(a.items() if isinstance(a, dict)
                             else a for a in args)))
    np.random.shuffle(zipped_args)
    return [(_type(data) if _type != np.ndarray else np.asarray(data))
            for _type, data in zip(map(type, args), zip(*zipped_args))]


# TODO: improve case coverage
def _train_on_batch_dummy(model, class_weights={'0':1,'1':6.5},
                          input_as_labels=False):
    """Instantiates trainer & optimizer, but does NOT train (update weights)"""
    def _make_toy_inputs(model):
        return np.random.randn(*model.input_shape)

    def _make_toy_labels(model):
        loss = model.loss
        shape = model.output_shape

        if loss == 'binary_crossentropy':
            return np.random.randint(0, 1, shape)
        elif loss == 'categorical_crossentropy':
            n_classes = shape[-1]
            class_labels = np.random.randint(0, n_classes, shape[0])
            return np.eye(n_classes)[class_labels]
        elif loss == 'sparse_categorical_crossentropy':
            return np.random.randint(0, shape[-1], shape[0])
        elif loss == 'mse':
            return np.random.randn(*shape)
        else:
            raise ValueError("unsupported loss: '{}'".format(loss))

    def _make_sample_weight(toy_labels, class_weights):
        if class_weights is not None:
            return np.array([class_weights[str(l)] for l in toy_labels])
        else:
            return np.ones(toy_labels.shape[0])

    toy_inputs = _make_toy_inputs(model)
    toy_labels = _make_toy_labels(model)
    toy_sample_weight = _make_sample_weight(toy_labels, class_weights)
    if input_as_labels:
        toy_labels = toy_inputs

    model._standardize_user_data(toy_inputs, toy_labels, toy_sample_weight)
    model._make_train_function()


def _validate_traingen_configs(cls):
    def _validate_metrics():
        for name in ('train_metrics', 'val_metrics'):
            value = getattr(cls, name)
            if not isinstance(value, list):
                if isinstance(value, str):
                    setattr(cls, name, [value])
                else:
                    setattr(cls, name, list(value))

        def _from_model(metric):
            return metric != 'loss' and metric not in [
                cls.model.loss, *cls.model.metrics]

        metrics = (*cls.train_metrics, *cls.val_metrics, cls.key_metric)
        supported = cls.BUILTIN_METRICS
        customs = cls.custom_metrics or [None]

        if cls.eval_fn_name == 'predict':
            for metric in metrics:
                metric = metric if metric != 'loss' else cls.model.loss
                if metric not in (*supported, *customs):
                    raise ValueError((
                        "'{0}' metric is not supported; add a function to "
                        "`custom_metrics` as '{0}': func. Supported "
                        "are: {1}").format(metric, ', '.join(supported)))

            if cls.model.loss not in (*supported, *customs):
                raise ValueError((
                    "'{0}' loss is not supported w/ `eval_fn_name = "
                    "'predict'`; add a function to `custom_metrics` "
                    "as '{0}': func, or set `eval_fn_name = 'evaluate'`."
                    " Supported are: {1}").format(
                        cls.model.loss, ', '.join(supported)))

            km = (cls.key_metric if cls.key_metric != 'loss'
                  else cls.model.loss)
            if km not in supported and cls.key_metric_fn is None:
                raise ValueError(("`key_metric = '{}'` is unsupported; set "
                                  "`key_metric_fn = func`. Supported are: {}"
                                  ).format(km, ', '.join(supported)))

        if cls.max_is_best and cls.key_metric == 'loss':
            print(NOTE + "`max_is_best = True` and `key_metric = 'loss'`"
                  "; will consider higher loss to be better")

    def _validate_directories():
        if cls.logs_dir is None and cls.best_models_dir is None:
            print(WARN, "`logs_dir = None` and `best_models_dir = None`; "
                  "logging is OFF")
        elif cls.logs_dir is None:
            print(NOTE, "`logs_dir = None`; will not checkpoint "
                  "periodically")
        elif cls.best_models_dir is None:
            print(NOTE, "`best_models_dir = None`; best models will not "
                  "be checkpointed")


    def _validate_optimizer_saving_configs():
        cfgs = (cls.optimizer_save_configs, cls.optimizer_load_configs)
        for cfg in cfgs:
            if cfg is not None and 'include' in cfg and 'exclude' in cfg:
                raise ValueError("cannot have both 'include' and 'exclude' "
                                 "in `optimizer_save_configs` or "
                                 "`optimizer_load_configs`")

    def _validate_visualizers():
        if (cls.visualizers is not None and cls.eval_fn_name != 'predict'
            and not any([isinstance(x, LambdaType) for x in
                           cls.visualizers])):
            print(WARN, "`eval_fn_name != 'predict'`, cannot use built-in "
                  "`visualizers`; include a custom function")

    def _validate_savelist():
        if cls.input_as_labels and 'labels' in cls.savelist:
            print(NOTE, "will exclude `labels` from saving when "
                  "`input_as_labels=True`; to override, "
                  "supply '{labels}' instead")
            cls.savelist.pop(cls.savelist.index('labels'))
        if '{labels}' in cls.savelist:
            cls.savelist.pop(cls.savelist.index('{labels}'))
            cls.savelist.append('labels')

    def _validate_weighted_slices_range():
        if cls.pred_weighted_slices_range is not None:
            if cls.eval_fn_name != 'predict':
                raise ValueError("`pred_weighted_slices_range` requires "
                                 "`eval_fn_name = 'predict'`")
        if (cls.pred_weighted_slices_range is not None or
            cls.loss_weighted_slices_range is not None):
            if not (hasattr(cls.datagen, 'slices_per_batch') and
                    hasattr(cls.val_datagen, 'slices_per_batch')):
                raise ValueError("to use `weighted_slices_range`, "
                                 "`datagen` and `val_datagen` must have "
                                 "`slices_per_batch` attribute defined "
                                 "(via `preprocessor`).")
            for name in ('datagen', 'val_datagen'):
                dg = getattr(cls, name)
                no_slices = dg.slices_per_batch in {1, None}

                if no_slices:
                    print(WARN, "`%s` uses no (or one) slices; " % name
                          + "setting `pred_weighted_slices_range=None`, "
                          "`loss_weighted_slices_range=None`")
                    cls.pred_weighted_slices_range = None
                    cls.loss_weighted_slices_range = None

    def _validate_class_weights():
        for name in ('class_weights', 'val_class_weights'):
            cw = getattr(cls, name)
            if cw is not None:
                assert all([isinstance(x, int) for x in cw.keys()]), (
                    "`{}` classes must be of type int (got {})"
                    ).format(name, cw)
                assert ((0 in cw and 1 in cw) or cw.sum() > 1), (
                    "`{}` must contain classes 1 and 0, or greater "
                    "(got {})").format(name, cw)

    def _validate_best_subset_size():
        if cls.best_subset_size is not None:
            if cls.batch_size is None:
                raise ValueError("`batch_size` cannot be None to use "
                                 "`best_subset_size`")
            if cls.val_datagen.shuffle_group_samples:
                raise ValueError("`val_datagen` cannot use `shuffle_group_"
                                 "samples` with `best_subset_size`")

    _validate_metrics()
    _validate_directories()
    _validate_optimizer_saving_configs()
    _validate_visualizers()
    _validate_savelist()
    _validate_weighted_slices_range()
    _validate_class_weights()
    _validate_best_subset_size()
