# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import deeptrain.metrics

from types import LambdaType
from functools import wraps
from inspect import getfullargspec
from copy import deepcopy
from collections.abc import Mapping

from deeptrain.backend import model_utils
from .algorithms import deepmap, deep_isinstance
from .experimental import deepcopy_v2
from .algorithms import builtin_or_npscalar, obj_to_str
from .configs import _PLOT_CFG, _ALIAS_TO_METRIC
from ._backend import WARN, NOTE, TF_KERAS


def pass_on_error(fn, *args, **kwargs):
    errmsg = kwargs.pop('errmsg', None)
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        if errmsg is not None:
            print(errmsg)
        print("Errmsg:", e)


def try_except(try_fn, except_fn):
    try:
        try_fn()
    except:
        if except_fn:  # else pass
            except_fn()


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
    return {k: v for k, v in dc.items()
            if condition(k, keys, exclude, filter_substr)}


def get_module_methods(module):
    output = {}
    for name in dir(module):
        obj = getattr(module, name)
        obj_name = getattr(obj, '__name__', '')
        if ((str(obj).startswith('<function')
             and isinstance(obj, LambdaType)) # is a function
            and module.__name__ == getattr(obj, '__module__', '')  # same module
            and name in str(getattr(obj, '__code__', ''))  # not a duplicate
            and "__%s__" % obj_name.strip('__') != obj_name  # not a magic method
            and '<lambda>' not in str(getattr(obj, '__code__', ''))  # not lambda
        ):
            output[name] = obj
    return output


def capture_args(fn):
    """Capture bound method arguments without changing its input signature.
    Method must have a **kwargs to append captured arguments to.

    Non-literal types and objects will be converted to their string representation
    (or `__qualname__` or `__name__` if they possess it).
    """
    @wraps(fn)
    def wrap(self, *args, **kwargs):
        #### Positional arguments ########
        posarg_names = [arg for arg in argspec(fn)[1:] if arg not in kwargs]
        posargs = {}
        for name, value in zip(posarg_names, args):
            posargs[name] = obj_to_str(value)
        if len(posargs) < len(args):
            varargs = getfullargspec(fn).varargs
            posargs[f'*{varargs}'] = deepmap(args[len(posargs):], obj_to_str)

        #### Keyword arguments ########
        kwargs['_passed_args'] = {}
        if len(kwargs) != 0:
            kwargs['_passed_args'].update(deepcopy_v2(kwargs, obj_to_str))

        kwargs['_passed_args'].update(posargs)
        del kwargs['_passed_args']['_passed_args']
        fn(self, *args, **kwargs)
    return wrap


def _init_optimizer(model, class_weights=None, input_as_labels=False,
                    alias_to_metric_name_fn=None):
    """Instantiates optimizer (and maybe trainer), but does NOT train
       (update weights)"""
    loss = model.loss
    if not isinstance(loss, str):
        if not hasattr(loss, '__name__'):
            raise Exception("unable to instantiate optimizer; open an Issue "
                            "with a minimally-reproducible example")
        loss = loss.__name__

    if alias_to_metric_name_fn is not None:
        loss = alias_to_metric_name_fn(model.loss)
    else:
        loss = _ALIAS_TO_METRIC.get(model.loss, model.loss)

    if hasattr(model, '_make_train_function'):
        model._make_train_function()
    else:
        model.optimizer._create_all_weights(model.trainable_weights)


def _make_plot_configs_from_metrics(self):
    """Makes default `plot_configs`, building on `configs._PLOT_CFG`; see
    :func:`~deeptrain.visuals.get_history_fig`. Validates some configs
    and tries to fill others.

    - Ensures every iterable config is of same `len()` as number of metrics in
      `'metrics'`, by extending *last* value of iterable to match the len. Ex:

      >>> {'metrics': {'val': ['loss', 'accuracy', 'f1']},
      ...  'linestyle': ['--', '-'],  # -> ['--', '-', '-']
      ... }

    - Assigns colors to metrics based on a default cycling coloring scheme,
      with some predefined customs (look for `_customs_map` in source code).
    - Configures up to two plot panes, mediated by `plot_first_pane_max_vals`;
      if number of metrics in `'metrics'` exceeds it, then a second pane is used.
      Can be used to configure how many metrics to draw in first pane; useful
      for managing clutter.
    """
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

    def _get_extend(config, n, tail=False):
        if not isinstance(config, (tuple, list)):
            config = [config]
        cfg = config[n:] if tail else config[:n]
        if len(cfg) < n:
            cfg.extend([cfg[-1]] * (n - len(cfg)))
        return cfg

    plot_configs = []
    n_train = len(self.train_metrics)
    n_val = len(self.val_metrics)

    val_metrics_p1 = self.val_metrics[:self.plot_first_pane_max_vals]
    n_val_p1 = len(val_metrics_p1)
    n_total_p1 = n_train + n_val_p1

    colors = _make_colors()
    mark_best_cfg = {'val': self.key_metric,
                     'max_is_best': self.max_is_best}

    PLOT_CFG = deepcopy(_PLOT_CFG)  # ensure module dict remains unchanged
    CFG = PLOT_CFG[0]
    plot_configs.append({
        'metrics':
            CFG['metrics'] or {'train': self.train_metrics,
                               'val'  : val_metrics_p1},
        'x_ticks':
            CFG['x_ticks'] or {'train': ['_train_x_ticks'] * n_train,
                               'val'  : ['_val_train_x_ticks'] * n_val_p1},
        'vhlines'      : CFG['vhlines'],
        'mark_best_cfg': CFG['mark_best_cfg'] or mark_best_cfg,
        'ylims'        : CFG['ylims'],
        'legend_kw'    : CFG['legend_kw'],

        'linewidth': _get_extend(CFG['linewidth'], n_total_p1),
        'linestyle': _get_extend(CFG['linestyle'], n_total_p1),
        'color'    : _get_extend(CFG['color'] or colors, n_total_p1),
    })
    if len(self.val_metrics) <= self.plot_first_pane_max_vals:
        return plot_configs

    #### dedicate separate pane to remainder val_metrics ######################
    n_val_p2 = n_val - n_val_p1

    CFG = PLOT_CFG[1]
    plot_configs.append({
        'metrics':
            CFG['metrics'] or {'val': self.val_metrics[n_val_p1:]},
        'x_ticks':
            CFG['x_ticks'] or {'val': ['_val_x_ticks'] * n_val_p2},
        'vhlines'      : CFG['vhlines'],
        'mark_best_cfg': CFG['mark_best_cfg'] or mark_best_cfg,
        'ylims'        : CFG['ylims'],
        'legend_kw'    : CFG['legend_kw'],

        'linewidth': _get_extend(CFG['linewidth'], n_val_p2),
        'linestyle': _get_extend(CFG['linestyle'], n_val_p2),
        'color'    : _get_extend(CFG['color'] or colors, n_total_p1, tail=True),
    })
    return plot_configs


def _validate_traingen_configs(self):
    def _validate_metrics():
        def _validate(metric, failmsg):
            if metric == 'accuracy':
                return  # converted internally (training.py)
            try:
                # check against alias since converted internally when computing
                getattr(deeptrain.metrics, self._alias_to_metric_name(metric))
            except:
                if (not self.custom_metrics or
                    (self.custom_metrics and metric not in self.custom_metrics)):
                    raise ValueError(failmsg)

        model_metrics = model_utils.get_model_metrics(self.model)

        vm_and_eval = self.val_metrics and 'evaluate' in self._eval_fn_name
        if self.val_metrics is None or '*' in self.val_metrics or vm_and_eval:
            if self.val_metrics is None or vm_and_eval:
                if vm_and_eval:
                    print(WARN, "will override `val_metrics` with model metrics "
                          "for `eval_fn_name == 'evaluate'`")
                self.val_metrics = model_metrics.copy()
            elif '*' in self.val_metrics:
                for metric in model_metrics:
                    # insert model metrics at wildcard's index
                    self.val_metrics.insert(self.val_metrics.index('*'), metric)
                self.val_metrics.pop(self.val_metrics.index('*'))

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

        if 'evaluate' in self._eval_fn_name:
            basemsg = ("must be in one of metrics returned by model, "
                       "when using 'evaluate' in `eval_fn.__name__`. "
                       "(model returns: %s)" % ', '.join(model_metrics))
            if self.key_metric not in model_metrics:
                raise ValueError(f"key_metric {self.key_metric} " + basemsg)

        if self.key_metric not in self.val_metrics:
            self.val_metrics.append(self.key_metric)

        if 'predict' in self._eval_fn_name:
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

    def _validate_saveskip_list():
        if self.input_as_labels and 'labels' not in self.saveskip_list and (
                '{labels}' not in self.saveskip_list):
            print(NOTE, "will exclude `labels` from saving when "
                  "`input_as_labels=True`; to keep 'labels', add '{labels}'"
                  "to `saveskip_list` instead")
            self.saveskip_list.append('labels')

    def _validate_loadskip_list():
        lsl = self.loadskip_list
        if not isinstance(lsl, list) and lsl not in ('auto', 'none', None):
            raise ValueError("`loadskip_list` must be a list, None, 'auto', "
                             "or 'none'")

    def _validate_weighted_slices_range():
        if self.pred_weighted_slices_range is not None:
            if 'predict' not in self._eval_fn_name:
                raise ValueError("`pred_weighted_slices_range` requires "
                                 "'predict' in `eval_fn_name`")
        if (self.pred_weighted_slices_range is not None or
            self.loss_weighted_slices_range is not None):
            if not (self.datagen.slices_per_batch and
                    self.val_datagen.slices_per_batch):
                raise ValueError("to use `loss_weighted_slices_range`, and/or "
                                 "`pred_weighted_slices_range`, "
                                 "`datagen` and `val_datagen` must have "
                                 "`slices_per_batch` attribute set (not falsy) "
                                 "(via `preprocessor`).")
            for name in ('datagen', 'val_datagen'):
                spb = getattr(self, name).slices_per_batch
                assert (isinstance(spb, int) and spb >= 1
                        ), ("`slices_per_batch` must be None or int >= 1, "
                            "got: %s for %s" % (spb, name))

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
        if self.dynamic_predict_threshold_min_max is None:
            return
        if self.key_metric_fn is None:
            raise ValueError("`key_metric_fn=None` (possibly per 'predict' "
                             "not in `eval_fn_name`); cannot use "
                             "`dynamic_predict_threshold_min_max`")
        elif 'pred_threshold' not in argspec(self.key_metric_fn):
            raise ValueError("`pred_threshold` parameter missing from "
                             "`key_metric_fn`; cannot use "
                             "`dynamic_predict_threshold_min_max`")

    def _validate_or_make_plot_configs():
        if self.plot_configs is not None:
            cfg = self.plot_configs
            required = ('metrics', 'x_ticks')
            for pane_cfg in cfg:
                assert all([name in pane_cfg for name in required]), (
                    "plot pane configs must contain %s" % ', '.join(required))
        else:
            self.plot_configs = _make_plot_configs_from_metrics(self)
        assert all(('metrics' in cfg and 'x_ticks' in cfg)
                   for cfg in self.plot_configs
                   ), ("all dicts in `plot_configs` must include "
                       "'metrics', 'x_ticks'")

    def _validate_metric_printskip_configs():
        for name, cfg in self.metric_printskip_configs.items():
            if not isinstance(cfg, list):
                if isinstance(cfg, tuple):
                    self.metric_printskip_configs[name] = list(cfg)
                else:
                    self.metric_printskip_configs[name] = [cfg]

    def _validate_callbacks():
        def _validate_types(cb, stage):
            if not isinstance(cb[stage], (list, tuple)):
                cb[stage] = (cb[stage],)
            for fn in cb[stage]:
                if not isinstance(fn, LambdaType):
                    raise ValueError("`callbacks` dict values must be "
                                     "functions, or list or tuple of functions.")

        supported = ('save', 'load', 'val_end',
                     'train:iter', 'train:batch', 'train:epoch',
                     'val:iter', 'val:batch', 'val:epoch')
        assert isinstance(self.callbacks, (list, tuple, dict)), (
            "`callbacks` must be list, tuple, or dict "
            "- got %s" % type(self.callbacks))
        if isinstance(self.callbacks, dict):
            self.callbacks = [self.callbacks]

        from ..callbacks import TraingenCallback

        for cb in self.callbacks:
            assert isinstance(cb, (dict, TraingenCallback)), (
                "`callbacks` items must be dict or subclass TraingenCallback, "
                "got %s" % type(cb))
            if isinstance(cb, TraingenCallback):
                continue
            for stage in cb:
                stages = stage if isinstance(stage, tuple) else (stage,)
                if not all(s in supported for s in stages):
                    raise ValueError(f"stage '{stage}' in `callbacks` is not "
                                     "supported; supported are: "
                                     + ', '.join(supported))
                _validate_types(cb, stage)

    def _validate_model_save_kw():
        if self.model_save_kw is None:
            self.model_save_kw = {'include_optimizer': True}
            if TF_KERAS:
                self.model_save_kw['save_format'] = 'h5'
        elif 'save_format' in self.model_save_kw and not TF_KERAS:
            raise ValueError(f"`keras` `model.save()` does not support "
                             "'save_format' kwarg, defaulting to 'h5'")
        if self.model_save_weights_kw is None:
            self.model_save_weights_kw = {'save_format': 'h5'} if TF_KERAS else {}
        elif 'save_format' in self.model_save_weights_kw and not TF_KERAS:
            raise ValueError("f`keras` `model.save_weights()` does not support "
                             "'save_format' kwarg, defaulting to 'h5'")

    def _validate_freq_configs():
        for name in ('val_freq', 'plot_history_freq', 'unique_checkpoint_freq',
                     'temp_checkpoint_freq'):
            attr = getattr(self, name)
            assert isinstance(attr, (dict, type(None))
                              ), f"{name} must be dict or None (got: {attr})"
            if isinstance(attr, dict):
                assert len(attr) <= 1, (
                    f"{name} supports up to one key-value pair (got: {attr})")

    for name, fn in locals().items():
        if name.startswith('_validate_'):
            fn()
