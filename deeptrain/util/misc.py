# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import deeptrain.metrics

from types import LambdaType
from functools import reduce
from inspect import getfullargspec
from ._backend import WARN, NOTE


def pass_on_error(fn, *args, **kwargs):
    fail_msg = kwargs.pop('fail_msg', None)
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        if fail_msg is not None:
            print(fail_msg)
        print("Errmsg:", e)


def argspec(obj):
    return getfullargspec(obj).args


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


def deeplen(item, iterables=(list, tuple, dict, np.ndarray)):
    # return 1 and terminate recursion when `item` is no longer iterable
    if isinstance(item, iterables):
        if isinstance(item, dict):
            item = item.values()
        return sum(deeplen(subitem) for subitem in item)
    else:
        return 1


def nCk(n, k):  # n-Choose-k
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom


def get_module_methods(module):
    output = {}
    for name in dir(module):
        obj = getattr(module, name)
        obj_name = getattr(obj, '__name__', '')
        if ((str(obj).startswith('<function')
             and isinstance(obj, LambdaType)) # is a function
            and module.__name__ == getattr(obj, '__module__', '')  # same module
            and name in str(getattr(obj, '__code__'))  # not a duplicate
            and not (  # not a magic method
                obj_name.startswith('__')
                and obj_name.endswith('__')
                and len(obj_name) >= 5
            )
            and '<lambda>' not in str(getattr(obj, '__code__'))  # not a lambda
        ):
            output[name] = obj
    return output


def _train_on_batch_dummy(model, class_weights=None, input_as_labels=False,
                          alias_to_metric_name_fn=None):
    """Instantiates trainer & optimizer, but does NOT train (update weights)"""
    def _make_toy_inputs(batch_size, input_shape):
        return np.random.uniform(0, 1, (batch_size, *input_shape[1:]))

    def _make_toy_labels(batch_size, output_shape, loss):
        n_classes = output_shape[-1]  # if appicable

        if loss == 'binary_crossentropy':
            return np.random.randint(0, 2, (batch_size, 1))
        elif loss == 'categorical_crossentropy':
            class_labels = np.random.randint(0, n_classes, batch_size)
            return np.eye(n_classes)[class_labels]
        elif loss == 'sparse_categorical_crossentropy':
            return np.random.randint(0, n_classes, (batch_size, 1))
        elif loss in ('mean_squared_error', 'mean_absolute_error',
                      'mean_squared_logarithmic_error',
                      'mean_absolute_percentage_error',
                      'logcosh', 'kullback_leibler_divergence'):
            return np.random.randn(batch_size, 10, 4)
        elif loss in ('squared_hinge', 'hinge', 'categorical_hinge'):
            return np.array([-1, 1])[np.random.randint(0, 2, (batch_size, 1))]
        elif loss in ('poisson', 'cosine_proximity'):
            return np.random.uniform(0, 10, batch_size)
        else:
            raise ValueError("unknown loss: '{}'".format(loss))

    def _make_sample_weight(toy_labels, class_weights, loss):
        if class_weights is None:
            return np.ones(toy_labels.shape[0])
        if loss == 'categorical_crossentropy':
            return np.array([class_weights[int(np.argmax(l))]
                             for l in toy_labels])
        else:
            return np.array([class_weights[int(l)]
                             for l in toy_labels])

    batch_size = model.output_shape[0]
    if batch_size is None:
        batch_size = 32
    loss = model.loss
    if alias_to_metric_name_fn is not None:
        loss = alias_to_metric_name_fn(loss)

    toy_inputs = _make_toy_inputs(batch_size, model.input_shape)
    toy_labels = _make_toy_labels(batch_size, model.output_shape, loss)
    toy_sample_weight = _make_sample_weight(toy_labels, class_weights, loss)
    if input_as_labels:
        toy_labels = toy_inputs

    model._standardize_user_data(toy_inputs, toy_labels, toy_sample_weight)
    model._make_train_function()


def _make_plot_configs_from_metrics(self):
    def _make_colors():
        train_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
        train_defaults.pop(1)  # reserve 'orange' for {'val': 'loss'}
        val_defaults = list(plt.cm.get_cmap('hsv')(np.linspace(.22, 1, 8)))
        train_customs_map = {'loss': train_defaults.pop(0),
                             'accuracy': 'blue'}
        val_customs_map = {'loss': 'orange',
                           'accuracy': 'xkcd:sun yellow',
                           'f1_score': 'purple',
                           'tnr': np.array([0., .503, 1.]),
                           'tpr': 'red'}

        colors = []
        for i, metric in enumerate(self.train_metrics):
            if metric in train_customs_map:
                colors.append(train_customs_map[metric])
            else:
                colors.append(train_defaults[i])

        for metric in self.val_metrics:
            if metric in val_customs_map:
                colors.append(val_customs_map[metric])
            else:
                colors.append(val_defaults.pop(0))
        return colors

    plot_configs = {}
    n_train = len(self.train_metrics)
    n_val = len(self.val_metrics)

    val_metrics_p1 = self.val_metrics[:self.plot_first_pane_max_vals]
    n_val_p1 = len(val_metrics_p1)
    n_total_p1 = n_train + n_val_p1

    colors = _make_colors()
    if self.key_metric == 'loss':
        mark_best_cfg = {'val': 'loss'}
    else:
        mark_best_cfg = None

    plot_configs['1'] = {
        'metrics':
            {'train': self.train_metrics,
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
    if len(self.val_metrics) <= self.plot_first_pane_max_vals:
        return plot_configs

    # dedicate separate pane to remainder val_metrics
    if self.key_metric != 'loss':
        mark_best_cfg = {'val': self.key_metric}
    else:
        mark_best_cfg = None
    n_val_p2 = n_val - n_val_p1

    plot_configs['2'] = {
        'metrics':
            {'val': self.val_metrics[n_val_p1:]},
        'x_ticks':
            {'val': ['_val_x_ticks'] * n_val_p2},
        'vhlines'   :
            {'v': '_val_hist_vlines',
             'h': .5},
        'mark_best_cfg': mark_best_cfg,
        'ylims'        : (0, 1),

        'linewidth': [1.5] * n_val_p2,
        'color'    : colors[n_total_p1:],
    }
    return plot_configs


def _validate_traingen_configs(self):
    def _validate_metrics():
        def _validate(metric, failmsg):
            try:
                # check against alias since converted internally when computing
                getattr(deeptrain.metrics, self._alias_to_metric_name(metric))
            except:
                raise ValueError(failmsg)

        for name in ('train_metrics', 'val_metrics'):
            value = getattr(self, name)
            if not isinstance(value, list):
                if isinstance(value, (str, type(None))):
                    setattr(self, name, [value])
                else:
                    setattr(self, name, list(value))
            value = getattr(self, name)
            for i, maybe_alias in enumerate(value):
                getattr(self, name)[i] = self._alias_to_metric_name(maybe_alias)
        self.key_metric = self._alias_to_metric_name(self.key_metric)

        model_metrics = self.model.metrics_names

        if self.eval_fn_name == 'evaluate':
            basemsg = ("must be in one of metrics returned by model, "
                       "when using `eval_fn_name='evaluate'`. "
                       "(model returns: %s)" % ', '.join(model_metrics))
            for metric in self.val_metrics:
                if metric not in model_metrics:
                    raise ValueError(f"val metric {metric} " + basemsg)
            if self.key_metric not in model_metrics:
                raise ValueError(f"key_metric {self.key_metric} " + basemsg)

        if self.key_metric not in self.val_metrics:
            self.val_metrics.append(self.key_metric)

        if self.eval_fn_name == 'predict':
            for metric in self.val_metrics:
                if metric == 'loss':
                    metric = self.model.loss
                _validate(metric, failmsg=("'{0}' metric is not supported; add "
                                           "a function to `custom_metrics` as "
                                           "'{0}': func.").format(metric))
            _validate(self.model.loss, failmsg=(
                "'{0}' loss is not supported w/ `eval_fn_name = 'predict'`; "
                "add a function to `custom_metrics` as '{0}': func, or set "
                "`eval_fn_name = 'evaluate'`.").format(self.model.loss))

            km = self.key_metric if self.key_metric != 'loss' else self.model.loss
            if self.key_metric_fn is None:
                _validate(km, failmsg=(f"`key_metric = '{km}'` is not supported; "
                                       "set `key_metric_fn = func`."))

        if self.max_is_best and self.key_metric == 'loss':
            print(NOTE + "`max_is_best = True` and `key_metric = 'loss'`"
                  "; will consider higher loss to be better")

    def _validate_model_metrics_match():
        def _set_in_matching_order(model_metrics, val):
            """Need metrics in matching order w/ model's to collect history
            """
            _metrics = model_metrics.copy()
            target_name = 'val_metrics' if val else 'train_metrics'

            for metric in getattr(self, target_name, []):
                if metric not in _metrics:
                    _metrics.append(metric)
            setattr(self, target_name, _metrics.copy())

        model_metrics = self.model.metrics_names.copy()
        # ensure api-compatibility, e.g. 'acc' -> 'accuracy'
        model_metrics = [self._alias_to_metric_name(metric)
                         for metric in model_metrics]

        _set_in_matching_order(model_metrics, val=False)

        if self.eval_fn_name == 'evaluate':
            if self.val_metrics is not None:
                for metric in self.val_metrics:
                    if metric not in model_metrics:
                        raise ValueError(
                            f"metric '{metric}' is not in model.metrics_names, "
                            "with `eval_fn_name='evaluate'`")
            self.val_metrics = model_metrics.copy()
        else:
            _set_in_matching_order(model_metrics, val=True)

    def _validate_directories():
        if self.logs_dir is None and self.best_models_dir is None:
            print(WARN, "`logs_dir = None` and `best_models_dir = None`; "
                  "logging is OFF")
        elif self.logs_dir is None:
            print(NOTE, "`logs_dir = None`; will not checkpoint "
                  "periodically")
        elif self.best_models_dir is None:
            print(NOTE, "`best_models_dir = None`; best models will not "
                  "be checkpointed")

    def _validate_optimizer_saving_configs():
        for name in ('optimizer_save_configs', 'optimizer_load_configs'):
            cfg = getattr(self, name)
            if cfg is not None and 'include' in cfg and 'exclude' in cfg:
                raise ValueError("cannot have both 'include' and 'exclude' "
                                 f"in `{name}`")

    def _validate_savelist():
        if self.input_as_labels and 'labels' in self.savelist:
            print(NOTE, "will exclude `labels` from saving when "
                  "`input_as_labels=True`; to override, "
                  "supply '{labels}' instead")
            self.savelist.pop(self.savelist.index('labels'))
        if '{labels}' in self.savelist:
            self.savelist.pop(self.savelist.index('{labels}'))
            self.savelist.append('labels')
        for required_key in ('datagen', 'val_datagen'):
            if required_key not in self.savelist:
                print(WARN, ("'{}' must be included in `savelist`; will append"
                             ).format(required_key))
                self.savelist.append(required_key)

    def _validate_weighted_slices_range():
        if self.pred_weighted_slices_range is not None:
            if self.eval_fn_name != 'predict':
                raise ValueError("`pred_weighted_slices_range` requires "
                                 "`eval_fn_name = 'predict'`")
        if (self.pred_weighted_slices_range is not None or
            self.loss_weighted_slices_range is not None):
            if not (hasattr(self.datagen, 'slices_per_batch') and
                    hasattr(self.val_datagen, 'slices_per_batch')):
                raise ValueError("to use `loss_weighted_slices_range`, and/or "
                                 "`pred_weighted_slices_range`, "
                                 "`datagen` and `val_datagen` must have "
                                 "`slices_per_batch` attribute defined "
                                 "(via `preprocessor`).")
            for name in ('datagen', 'val_datagen'):
                dg = getattr(self, name)
                no_slices = dg.slices_per_batch in {1, None}

                if no_slices:
                    print(WARN, "`%s` uses no (or one) slices; " % name
                          + "setting `pred_weighted_slices_range=None`, "
                          "`loss_weighted_slices_range=None`")
                    self.pred_weighted_slices_range = None
                    self.loss_weighted_slices_range = None

    def _validate_class_weights():
        for name in ('class_weights', 'val_class_weights'):
            cw = getattr(self, name)
            if cw is not None:
                assert all([isinstance(x, int) for x in cw.keys()]), (
                    "`{}` classes must be of type int (got {})"
                    ).format(name, cw)
                assert ((0 in cw and 1 in cw) or sum(cw.values()) > 1), (
                    "`{}` must contain classes 1 and 0, or greater "
                    "(got {})").format(name, cw)

                if self.model.loss in ('categorical_crossentropy',
                                      'sparse_categorical_crossentropy'):
                    n_classes = self.model.output_shape[-1]
                    for class_label in range(n_classes):
                        if class_label not in cw:
                            getattr(self, name)[name][class_label] = 1

    def _validate_best_subset_size():
        if self.best_subset_size is not None:
            if self.val_datagen.shuffle_group_samples:
                raise ValueError("`val_datagen` cannot use `shuffle_group_"
                                 "samples` with `best_subset_size`")

    def _validate_dynamic_predict_threshold_min_max():
        if self.dynamic_predict_threshold_min_max is not None:
            if self.key_metric_fn is None:
                print(WARN, "`key_metric_fn=None` (likely per `eval_fn_name !="
                      " 'predict'`); setting "
                      "`dynamic_predict_threshold_min_max=None`")
                self.dynamic_predict_threshold_min_max = None
            elif 'pred_threshold' not in argspec(self.key_metric_fn):
                print(WARN, "`pred_threshold` parameter missing from "
                      "`key_metric_fn`; setting "
                      "`dynamic_predict_threshold_min_max=None`")
                self.dynamic_predict_threshold_min_max = None

    def _validate_or_make_plot_configs():
        if self.plot_configs is not None:
            cfg = self.plot_configs
            assert ('1' in cfg), ("`plot_configs` must number plot panes "
                                  "via strings; see util\configs.py")
            required = ('metrics', 'x_ticks')
            for pane_cfg in cfg.values():
                assert all([name in pane_cfg for name in required]), (
                    "plot pane configs must contain %s" % ', '.join(required))
        else:
            self.plot_configs = _make_plot_configs_from_metrics(self)

    def _validate_metric_printskip_configs():
        for name, cfg in self.metric_printskip_configs.items():
            if not isinstance(cfg, list):
                if isinstance(cfg, tuple):
                    self.metric_printskip_configs[name] = list(cfg)
                else:
                    self.metric_printskip_configs[name] = [cfg]

    def _validate_callbacks():
        def _validate_types(cb, stage):
            errmsg = ("`callbacks` stage values types must be callable, "
                      "or list or tuple of callables")
            if not isinstance(cb[stage], (list, tuple)):
                if not isinstance(cb[stage], LambdaType):
                    raise ValueError(errmsg)
                cb[stage] = (cb[stage],)
            else:
                for fn in cb[stage]:
                    if not isinstance(fn, LambdaType):
                        raise ValueError(errmsg)

        supported = ('save', 'load', 'val_end',
                     'train:iter', 'train:batch', 'train:epoch',
                     'val:iter', 'val:batch', 'val:epoch')
        assert isinstance(self.callbacks, dict), ("`callbacks` must be "
                                                 "of type dict")
        for cb in self.callbacks.values():
            assert isinstance(cb, dict), ("`callbacks` values must be "
                                          "of type dict")
            for stage in cb:
                stages = stage if isinstance(stage, tuple) else (stage,)
                if not all(s in supported for s in stages):
                    raise ValueError(f"stage '{stage}' in `callbacks` is not "
                                     "supported; supported are: "
                                     + ', '.join(supported))
                _validate_types(cb, stage)

    _validate_metrics()
    _validate_model_metrics_match()
    _validate_directories()
    _validate_optimizer_saving_configs()
    _validate_savelist()
    _validate_weighted_slices_range()
    _validate_class_weights()
    _validate_best_subset_size()
    _validate_dynamic_predict_threshold_min_max()
    _validate_or_make_plot_configs()
    _validate_metric_printskip_configs()
    _validate_callbacks()
