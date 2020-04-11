# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from types import LambdaType
from functools import reduce
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


def nCk(n, k):  # n-Choose-k
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom


# TODO: improve case coverage
def _train_on_batch_dummy(model, class_weights={'0':1,'1':6.5},
                          input_as_labels=False):
    """Instantiates trainer & optimizer, but does NOT train (update weights)"""
    def _make_toy_inputs(batch_size, input_shape):
        return np.random.randn(batch_size, *input_shape[1:])

    def _make_toy_labels(batch_size, output_shape, loss):
        if loss == 'binary_crossentropy':
            return np.random.randint(0, 1, output_shape)
        elif loss == 'categorical_crossentropy':
            n_classes = output_shape[-1]
            class_labels = np.random.randint(0, n_classes, batch_size)
            return np.eye(n_classes)[class_labels]
        elif loss == 'sparse_categorical_crossentropy':
            return np.random.randint(0, output_shape[-1], batch_size)
        elif loss == 'mse':
            return np.random.randn(batch_size, *output_shape[1:])
        else:
            raise ValueError("unsupported loss: '{}'".format(loss))

    def _make_sample_weight(toy_labels, class_weights):
        if class_weights is not None:
            return np.array([class_weights[str(l)] for l in toy_labels])
        else:
            return np.ones(toy_labels.shape[0])

    batch_size = model.output_shape[0]
    if batch_size is None:
        batch_size = 32

    toy_inputs = _make_toy_inputs(batch_size, model.input_shape)
    toy_labels = _make_toy_labels(batch_size, model.output_shape, model.loss)
    toy_sample_weight = _make_sample_weight(toy_labels, class_weights)
    if input_as_labels:
        toy_labels = toy_inputs

    model._standardize_user_data(toy_inputs, toy_labels, toy_sample_weight)
    model._make_train_function()


def _make_plot_configs_from_metrics(cls):
    def _make_colors():
        train_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
        train_defaults.pop(1)  # reserve 'orange' for {'val': 'loss'}
        val_defaults = list(plt.cm.get_cmap('hsv')(np.linspace(.22, 1, 8)))
        train_customs_map = {'loss': train_defaults.pop(0),
                             'accuracy': 'blue'}
        val_customs_map = {'loss': 'orange',
                           'accuracy': 'xkcd:sun yellow',
                           'f1-score': 'purple'}

        colors = []
        for i, metric in enumerate(cls.train_metrics):
            if metric in train_customs_map:
                colors.append(train_customs_map[metric])
            else:
                colors.append(train_defaults[i])

        for metric in cls.val_metrics:
            if metric in val_customs_map:
                colors.append(val_customs_map[metric])
            else:
                colors.append(val_defaults.pop(0))
        return colors

    def _get_val_metric_colors():
        color_cfg = {'f1-score': 'purple'}
        colors = []
        for name in cls.val_metrics[1:]:
            colors.append(color_cfg.get(name, None))
        return tuple(colors)

    plot_configs = {}
    n_train = len(cls.train_metrics)
    n_val = len(cls.val_metrics)

    val_metrics_p1 = cls.val_metrics[:cls.plot_first_pane_max_vals]
    n_val_p1 = len(val_metrics_p1)
    n_total_p1 = n_train + n_val_p1

    colors = _make_colors()
    if cls.key_metric == 'loss':
        mark_best_cfg = {'val': 'loss'}
    else:
        mark_best_cfg = None

    plot_configs['1'] = {
        'metrics':
            {'train': cls.train_metrics,
             'val'  : val_metrics_p1},
        'x_ticks':
            {'train': ['_train_x_ticks'] * n_train,
             'val':   ['_val_train_x_ticks'] * n_val_p1},
        'vhlines'   :
            {'v': '_hist_vlines',
             'h': 1},
        'mark_best_cfg': mark_best_cfg,
        'ylims'        : (0, 2),

        'linewidth': [1.5] * n_total_p1,
        'color'    : colors[:n_total_p1],
        }
    if n_val_p1 <= cls.plot_first_pane_max_vals:
        return plot_configs

    # dedicate separate pane to remainder val_metrics
    if cls.key_metric != 'loss':
        mark_best_cfg = {'val': cls.key_metric}
    else:
        mark_best_cfg = None
    n_val_p2 = n_val - n_val_p1

    plot_configs['2'] = {
        'metrics':
            {'val'  : cls.val_metrics[n_val_p1:]},
        'x_ticks':
            {'val':   ['_val_x_ticks'] * n_val_p2},
        'vhlines'   :
            {'v': '_val_hist_vlines',
             'h': .5},
        'mark_best_cfg': mark_best_cfg,
        'ylims'        : (0, 1),

        'linewidth': [1.5] * n_val_p2,
        'color'    : colors[n_total_p1:],
    }
    return plot_configs


def _validate_traingen_configs(cls):
    def _validate_metrics():
        for name in ('train_metrics', 'val_metrics'):
            value = getattr(cls, name)
            if not isinstance(value, list):
                if isinstance(value, str):
                    setattr(cls, name, [value])
                else:
                    setattr(cls, name, list(value))
            value = getattr(cls, name)
            for alias in value:
                setattr(cls, name, [cls._alias_to_metric_name(alias)])
        cls.key_metric = cls._alias_to_metric_name(cls.key_metric)

        metrics = (*cls.train_metrics, *cls.val_metrics, cls.key_metric)
        supported = cls.BUILTIN_METRICS
        customs = cls.custom_metrics or [None]

        if cls.key_metric not in cls.val_metrics:
            cls.val_metrics.append(cls.key_metric)

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

    def _validate_model_metrics_match():
        def _set_in_matching_order(model_metrics, val):
            """Need metrics in matching order w/ model's to collect history
            """
            _metrics = model_metrics.copy()
            target = cls.val_metrics if val else cls.train_metrics
            if target is None:
                target = model_metrics.copy()
                return

            for metric in target:
                if metric not in _metrics:
                    _metrics.append(metric)
            if val:
                cls.val_metrics = _metrics.copy()
            else:
                cls.train_metrics = _metrics.copy()

        model_metrics = cls.model.metrics_names.copy()
        # ensure api-compatibility, e.g. 'acc' -> 'accuracy'
        model_metrics = [cls._alias_to_metric_name(metric)
                         for metric in model_metrics]

        _set_in_matching_order(model_metrics, val=False)

        if cls.eval_fn_name == 'evaluate':
            if cls.val_metrics is not None:
                for metric in cls.val_metrics:
                    if metric not in model_metrics:
                        print(WARN, "metric {} is not in model.metrics_names, "
                              "w/ `eval_fn_name='evaluate'` - will drop")
            cls.val_metrics = model_metrics.copy()
        else:
            _set_in_matching_order(model_metrics, val=True)

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
        for name in ('optimizer_save_configs', 'optimizer_load_configs'):
            cfg = getattr(cls, name)
            if cfg is not None and 'include' in cfg and 'exclude' in cfg:
                raise ValueError("cannot have both 'include' and 'exclude' "
                                 f"in `{name}`")

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
            # if cls.batch_size is None:
            #     raise ValueError("`batch_size` cannot be None to use "
            #                      "`best_subset_size`")
            if cls.val_datagen.shuffle_group_samples:
                raise ValueError("`val_datagen` cannot use `shuffle_group_"
                                 "samples` with `best_subset_size`")

    def _validate_dynamic_predict_threshold_min_max():
        if cls.dynamic_predict_threshold_min_max is not None:
            if cls.key_metric_fn is None:
                print(WARN, "`key_metric_fn=None` (likely per `eval_fn_name !="
                      " 'predict'`); setting"
                      "`dynamic_predict_threshold_min_max=None`")
                cls.dynamic_predict_threshold_min_max = None
            elif 'pred_threshold' not in cls.key_metric_fn.__code__.co_varnames:
                print(WARN, "`pred_threshold` parameter missing from "
                      "`key_metric_fn`; setting "
                      "`dynamic_predict_threshold_min_max=None`")
                cls.dynamic_predict_threshold_min_max = None

    def _validate_or_make_plot_configs():
        if cls.plot_configs is not None:
            cfg = cls.plot_configs
            assert ('1' in cfg), ("`plot_configs` must number plot panes "
                                  "via strings; see util\configs.py")
            required = ('metrics', 'x_ticks')
            for pane_cfg in cfg.values():
                assert all([name in pane_cfg for name in required]), (
                    "plot pane configs must contain %s" % ', '.join(required))
        else:
            cls.plot_configs = _make_plot_configs_from_metrics(cls)

    _validate_metrics()
    _validate_model_metrics_match()
    _validate_directories()
    _validate_optimizer_saving_configs()
    _validate_visualizers()
    _validate_savelist()
    _validate_weighted_slices_range()
    _validate_class_weights()
    _validate_best_subset_size()
    _validate_dynamic_predict_threshold_min_max()
    _validate_or_make_plot_configs()
