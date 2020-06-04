# -*- coding: utf-8 -*-
## TODO
"""- random seeds management in TrainGenerator?
   - metrics.py
       - .__module__ problem upon superreload
       - pulls sklearn.metrics docs
   - deeptrain.colortext() toggle / setting to set whether NOTE/WARN use color
   - 'min' for `max_is_best=False` in model naming
   - Handle KeyboardInterrupt - with, finally?
   - Safe to interrupt flags print option?
   - MetaTrainer
"""

"""TODO-docs:
    - How's it different from other training frameworks?
       - advanced data pipeline
         - trackable
         - batch size flexibility
         - load speed optimizations
         - stateful option
       - preprocessing
         - batch making
         - class imbalance handling
         - advanced signal timeseries preprocessing
       - advanced train pipeline
         - dynamic hyperparameters; change between epochs, auto-restart session
         - reproducibility; seed tracking & restoring
"""

import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from .util._default_configs import _DEFAULT_TRAINGEN_CFG
from .util.configs  import _TRAINGEN_CFG
from .util._traingen_utils import TraingenUtils
from .util.logging  import _log_init_state
from .util.training import _get_api_metric_name
from .util.misc     import pass_on_error, capture_args
from .introspection import print_dead_weights, print_nan_weights
from .              import metrics as metrics_fns
from .callbacks     import TraingenCallback
from .util._backend import IMPORTS, Unbuffered, NOTE, WARN
from .backend import model_util

sys.stdout = Unbuffered(sys.stdout)


class TrainGenerator(TraingenUtils):
    @capture_args
    def __init__(self, model, datagen, val_datagen,
                 epochs=1,
                 logs_dir=None,
                 best_models_dir=None,
                 loadpath=None,
                 callbacks=None,

                 fit_fn_name='train_on_batch',
                 eval_fn_name='evaluate',
                 key_metric='loss',
                 key_metric_fn=None,
                 val_metrics=None,
                 custom_metrics=None,
                 input_as_labels=False,
                 max_is_best=None,

                 val_freq={'epoch': 1},
                 plot_history_freq={'epoch': 1},
                 unique_checkpoint_freq={'epoch': 1},
                 temp_checkpoint_freq=None,

                 class_weights=None,
                 val_class_weights=None,

                 reset_statefuls=False,
                 iter_verbosity=1,
                 logdir=None,
                 optimizer_save_configs=None,
                 optimizer_load_configs=None,
                 plot_configs=None,
                 model_configs=None,
                 **kwargs):
        super().__init__()

        self.model=model
        self.datagen=datagen
        self.val_datagen=val_datagen
        self.epochs=epochs
        self.logs_dir=logs_dir
        self.best_models_dir=best_models_dir
        self.loadpath=loadpath
        self.callbacks=callbacks or []

        self.fit_fn_name=fit_fn_name
        self.eval_fn_name=eval_fn_name
        self.key_metric=key_metric
        self.key_metric_fn=key_metric_fn
        self.train_metrics = model_util.get_model_metrics(model)
        self.val_metrics=val_metrics
        self.custom_metrics=custom_metrics
        self.input_as_labels=input_as_labels
        if max_is_best is None:
            value = False if self.key_metric == 'loss' else True
            self.max_is_best = value
        else:
            self.max_is_best = max_is_best

        self.val_freq=val_freq
        self.plot_history_freq=plot_history_freq
        self.unique_checkpoint_freq=unique_checkpoint_freq
        self.temp_checkpoint_freq=temp_checkpoint_freq

        self.class_weights=class_weights
        self.val_class_weights=val_class_weights

        self.reset_statefuls=reset_statefuls
        self.iter_verbosity=iter_verbosity
        self.logdir=logdir
        self.optimizer_save_configs=optimizer_save_configs
        self.optimizer_load_configs=optimizer_load_configs
        self.plot_configs=plot_configs
        self.model_configs = model_configs
        self.batch_size=kwargs.pop('batch_size', None) or model.output_shape[0]

        self._passed_args = kwargs.pop('_passed_args', None)
        self._init_and_validate_kwargs(kwargs)
        self._init_class_vars()
        self._init_fit_and_pred_fns()

        if self.logdir or self.logs_dir:
            self._init_logger()
        else:
            print(NOTE, "logging OFF")
            self.logdir = None

        self._init_callbacks_called = False
        if self.loadpath:
            self.load(passed_args=self._passed_args)
        else:
            self._prepare_initial_data()
            self._init_callbacks()  # called in `load`

        if self.logdir:
            savedir = os.path.join(self.logdir, "misc")
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            pass_on_error(_log_init_state, self, kwargs, savedir=savedir,
                          errmsg=WARN + " could not log init state - skipping")

    ########################## MAIN METHODS ##########################
    def train(self):
        """The main loop.
            - Fetches data from `get_data`
            - Fits data via `fin_fn`
            - Processes fit metrics in `_train_postiter_processing`
            - Stores metrics in `history`
            - Applies 'train:iter', 'train:batch', and 'train:epoch' callbacks
            - Calls `validate` when appropriate

        Interruption:
            - *Safe*: during `get_data`, which can be called indefinitely
              without changing any attributes.
            - *Avoid*: during `_train_postiter_processing`, where `fit_fn` is
              applied and weights are updated - but metrics aren't stored, and
              `_has_postiter_processed=False`, restarting the loop without
              recording progress.
        """
        while self.epoch < self.epochs:
            if not self._has_trained:
                if self._has_postiter_processed:
                    x, y, sample_weight = self.get_data(val=False)
                    sw = sample_weight if self.class_weights else None
                    if self.iter_verbosity:
                        self._print_iter_progress()
                    metrics = self.fit_fn(x, y, sample_weight=sw)

                self._has_postiter_processed = False
                self._train_postiter_processing(metrics)  # may call `validate`
                self._has_postiter_processed = True
            else:
                self.validate()
                self.train()

        print("Training has concluded.")

    def validate(self, record_progress=True, clear_cache=True):
        """Validation loop.
            - Fetches data from `get_data`
            - Applies function based on `eval_fn_name`
            - Processes and caches metrics/predictions in
              `_val_postiter_processing`
            - Applies 'val:iter', 'val:batch', and 'val:epoch' callbacks
            - Calls `_on_val_end` at end of validation to compute metrics
              and store them in `val_history`
            - Applies 'val_end' and maybe ('val_end': 'train:epoch') callbacks

        Interruption:
            - *Safe*: during `get_data`, which can be called indefinitely
              without changing any attributes.
            - *Avoid*: during `_val_postiter_processing`. Model remains
              unaffected*, but caches are updated; a restart may yield duplicate
              appending, which will error or yield inaccuracies.
              (* forward pass may consume random seed if random ops are used)
            - *In practice*: prefer interrupting immediately after
              `_print_iter_progress` executes
        """
        txt = ("Validating" if not self._has_validated else
               "Finishing post-val processing")
        print("\n\n{}...".format(txt))

        while not self._has_validated:
            kw = {}
            if self._val_has_postiter_processed:
                x, self._y_true, self._val_sw = self.get_data(val=True)
                sw = self._val_sw if self.val_class_weights else None
                if self.iter_verbosity:
                    self._print_iter_progress(val=True)

                if self.eval_fn_name == 'predict':
                    self._y_preds = self.model.predict(x, batch_size=len(x))
                elif self.eval_fn_name == 'evaluate':
                    kw['metrics'] = self.model.evaluate(
                        x, self._y_true, sample_weight=sw,
                        batch_size=len(x), verbose=0)
                kw['batch_size'] = len(x)

            self._val_has_postiter_processed = False
            self._val_postiter_processing(record_progress, **kw)
            self._val_has_postiter_processed = True

        if self._has_validated:
            self._on_val_end(record_progress, clear_cache)

    ######################### MAIN METHOD HELPERS ########################
    def _train_postiter_processing(self, metrics):
        def _on_iter_end(metrics):
            self._update_temp_history(metrics)
            self._fit_iters += 1
            self.datagen.update_state()
            self._apply_callbacks(stage='train:iter')

        def _on_batch_end():
            self._train_has_notified_of_new_batch = False

            self._batches_fit += 1
            self._train_x_ticks.append(self._batches_fit)
            self._train_val_x_ticks.append(self._times_validated)
            self._set_name_cache.append(self._set_name)
            try:
                self._update_train_history()
                self._print_train_progress()
            except Exception as e:
                # might happen due to incomplete loaded data for updating history
                print(WARN, "could not update and print progress"
                      "- OK if right after load; skipping...\nErrmsg: %s" % e)

            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset ', end='')
            self._apply_callbacks(stage='train:batch')

        def _on_epoch_end(val=False):
            self.temp_history = deepcopy(self._temp_history_empty)
            self.epoch = self.datagen.on_epoch_end()
            decor = "\n_________________________\n\033[4m {}{}{} \033[0m\n"
            print(decor.format("EPOCH ", self.epoch, " -- COMPLETE"))

            self._hist_vlines     += [self._batches_fit]
            self._val_hist_vlines += [self._times_validated]
            self._apply_callbacks(stage='train:epoch')

        def _should_validate():
            return self._should_do(self.val_freq)

        _on_iter_end(metrics)
        if self.datagen.batch_exhausted:
            _on_batch_end()
        if self.datagen.all_data_exhausted:
            _on_epoch_end()

        if _should_validate():
            self._has_postiter_processed = True  # in case val is interrupted
            self._has_trained = True
            self._has_validated = False
            self.validate()

    def _val_postiter_processing(self, record_progress=True, metrics=None,
                                 batch_size=None):
        def _on_iter_end(metrics=None, batch_size=None):
            if metrics is not None:
                self._update_temp_history(metrics, val=True)
            self._val_iters += 1
            if self.eval_fn_name == 'predict':
                self._update_val_iter_cache()
            self.val_datagen.update_state()

            if self.batch_size is None:
                if self._inferred_batch_size is None:
                    self._inferred_batch_size = batch_size
                elif self._inferred_batch_size != batch_size:
                    self._inferred_batch_size = 'varies'
            self._apply_callbacks(stage='val:iter')

        def _on_batch_end():
            self._batches_validated += 1
            self._val_set_name_cache.append(self._val_set_name)

            update = record_progress and self.val_datagen.all_data_exhausted
            if self.iter_verbosity >= 1:
                self._print_val_progress()
            if update:
                self._update_val_history()
            self._val_has_notified_of_new_batch = False

            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset', end=' ')
            self._apply_callbacks(stage='val:batch')

        def _on_epoch_end():
            self._has_validated = True
            self._apply_callbacks(stage='val:epoch')

        _on_iter_end(metrics, batch_size)
        if self.val_datagen.batch_exhausted:
            _on_batch_end()
        if self.val_datagen.all_data_exhausted:
            _on_epoch_end()


    def _on_val_end(self, record_progress, clear_cache):
        def _record_progress():
            self._times_validated += 1
            self.val_epoch = self.val_datagen.on_epoch_end()
            self._val_x_ticks += [self._times_validated]
            self._val_train_x_ticks += [self._batches_fit]

            new_best = bool(self.key_metric_history[-1] > self.best_key_metric)
            if not self.max_is_best:
                new_best = not new_best

            if new_best and self.best_models_dir is not None:
                self._save_best_model(del_previous_best=self.max_one_best_save)

            if self.logdir:
                do_temp = self._should_do(self.temp_checkpoint_freq)
                do_unique = self._should_do(self.unique_checkpoint_freq)
                if do_temp or do_unique:
                    self.checkpoint()

        def _print_best_subset():
            best_nums = ", ".join([str(x) for x in self.best_subset_nums])
            best_size = self.best_subset_size
            print("Best {}-subset: {}".format(best_size, best_nums))

        def _validate_batch_size():
            batch_size = self.batch_size or self._inferred_batch_size
            if not isinstance(batch_size, int):
                raise ValueError(
                    "to use `eval_fn_name = 'predict'`, either (1) `batch_size`"
                    " must be defined, or (2) data fed in `validation()` "
                    "must have same len() / .shape[0] across iterations.")

        _validate_batch_size()
        if self.best_subset_size:
            _print_best_subset()
        if record_progress:
            _record_progress()

        if self._should_do(self.plot_history_freq):
            pass_on_error(self.plot_history, update_fig=record_progress,
                          errmsg=(WARN + " model history could not be "
                                  "plotted; skipping..."))

        if self.datagen.all_data_exhausted:
            self._apply_callbacks(stage=('val_end', 'train:epoch'))
        else:
            self._apply_callbacks(stage='val_end')

        if clear_cache:
            self.clear_cache()
        if self.check_model_health:
            self.check_health()

        self._inferred_batch_size = None  # reset
        self._has_validated = False
        self._has_trained = False

    def clear_cache(self, reset_val_flags=False):  # to `validate` from scratch
        attrs_to_clear = ('_preds_cache', '_labels_cache', '_sw_cache',
                          '_class_labels_cache',
                          '_set_name_cache', '_val_set_name_cache',
                          '_y_true', '_val_sw')
        [setattr(self, attr, []) for attr in attrs_to_clear]
        self.val_temp_history = deepcopy(self._val_temp_history_empty)

        if reset_val_flags:
            self._inferred_batch_size = None
            self._has_validated = False
            self._has_trained = False
            self._val_has_postiter_processed = True

    def _should_do(self, config, forced=False):
        if forced:
            return True
        if config is None:
            return False
        freq_mode, freq_value = list(config.items())[0]

        if freq_mode == 'iter':
            return self._fit_iters % freq_value == 0
        elif freq_mode == 'batch':
            batch_done = self.datagen.batch_exhausted
            return (self._batches_fit % freq_value == 0) and batch_done
        elif freq_mode == 'epoch':
            epoch_done = self.datagen.all_data_exhausted
            return (self.epoch % freq_value == 0) and epoch_done
        elif freq_mode == 'val':
            return (self._times_validated % freq_value == 0)

    ########################## DATA_GET METHODS ##########################
    def get_data(self, val=False):
        def _standardize_shape(class_labels):
            while len(class_labels.shape) < len(self.model.output_shape):
                class_labels = np.expand_dims(class_labels, -1)
            return class_labels

        datagen = self.val_datagen if val else self.datagen
        if datagen.batch_exhausted:
            datagen.advance_batch()
            setattr(self, '_val_set_name' if val else '_set_name',
                    datagen.set_name)

        x, labels = datagen.get()
        y = labels if not self.input_as_labels else x

        class_labels = _standardize_shape(labels)
        slice_idx = getattr(datagen, 'slice_idx', None)
        sample_weight = self.get_sample_weight(class_labels, val, slice_idx)

        return x, y, sample_weight

    ########################## LOG METHODS ################################
    def _update_val_iter_cache(self):
        def _standardize_shapes(*data):
            ls = []
            for x in data:
                while len(x.shape) < len(self.model.output_shape):
                    x = np.expand_dims(x, -1)
                ls.append(x)
            return ls

        y, class_labels, sample_weight = _standardize_shapes(
            self._y_true, self.val_datagen.labels, self._val_sw)

        if getattr(self.val_datagen, 'slice_idx', None) is None:
            self._sw_cache.append(sample_weight)
            self._preds_cache.append(self._y_preds)
            self._labels_cache.append(y)
            self._class_labels_cache.append(class_labels)
            return

        if getattr(self.val_datagen, 'slice_idx', None) == 0:
            self._labels_cache.append([])
            self._class_labels_cache.append([])
            self._sw_cache.append([])
            if self.eval_fn_name == 'predict':
                self._preds_cache.append([])

        self._sw_cache[-1].append(sample_weight)
        self._labels_cache[-1].append(y)
        self._class_labels_cache[-1].append(class_labels)
        if self.eval_fn_name == 'predict':
            self._preds_cache[-1].append(self._y_preds)

    def _update_val_history(self):
        for name, metric in self._get_val_history().items():
            self.val_history[name].append(metric)
        self.key_metric_history.append(self.val_history[self.key_metric][-1])

    def _get_train_history(self):
        return {metric:np.mean(values) for metric, values
                in self.temp_history.items()}

    def _update_train_history(self):
        for metric, value in self._get_train_history().items():
            self.history[metric] += [value]

    def _print_train_progress(self):
        train_metrics = self._get_train_history()
        for name in self.metric_printskip_configs.get('train', []):
            train_metrics.pop(name, None)
        self._print_progress(train_metrics, endchar='')

    def _print_val_progress(self):
        val_metrics = self._get_val_history(for_current_iter=True)
        for name in self.metric_printskip_configs.get('val', []):
            val_metrics.pop(name, None)
        self._print_progress(val_metrics)

    def _print_progress(self, metrics, endchar='\n'):
        names  = [self._metric_name_to_alias(name) for name in metrics]
        values = [v for v in metrics.values()]
        assert len(names) == len(values)

        names_joined  = ', '.join(names)
        values_joined = ', '.join([('%.6f' % v) for v in values])
        if len(names) != 1:
            names_joined  = '(%s)' % names_joined
            values_joined = '(%s)' % values_joined
        print(" {} = {} ".format(names_joined, values_joined), end=endchar)

    def _print_iter_progress(self, val=False):
        if val:
            if not self._val_has_notified_of_new_batch:
                pad = self._val_max_set_name_chars + 3
                padded_num_txt = (self._val_set_name + "...").ljust(pad)
                print(end="Validating set %s" % padded_num_txt)
                self._val_has_notified_of_new_batch = True
            return

        if not self._train_has_notified_of_new_batch:
            pad = self._max_set_name_chars + 3
            padded_num_txt = (self._set_name + "...").ljust(pad)
            print(end="\nFitting set %s" % padded_num_txt)
            self._train_has_notified_of_new_batch = True
        if self.iter_verbosity >= 2:
            print(end='.')

    ########################## VISUAL/CALC METHODS ##########################
    def plot_history(self, update_fig=True, w=1, h=1):
        def _show_closed_fig(fig):
            _fig = plt.figure()
            manager = _fig.canvas.manager
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
            plt.show()

        fig = self._get_history_fig(self.plot_configs, w, h)
        if update_fig:
            self._history_fig = fig
        _show_closed_fig(fig)

    ########################## CALLBACK METHODS ######################
    def _apply_callbacks(self, stage):
        """Callbacks. See examples/callbacks

        Two approaches:
            1. Class-based: inherit deeptrain.callbacks.TraingenCallback,
               define stage-based methods, e.g. on_train_epoch_end. Methods
               also take `stage` argument for further control, e.g. to only
               call on_train_epoch_end when stage == ('val_end', 'train:epoch').
            2. Function-based: make a dict of stage-function call pairs, e.g.:
                {'train:epoch': (fn1, fn2),
                 'val_batch': fn3,
                 ('on_val_end': 'train:epoch"): fn4
                }
               `stage` is controlled such that (fn1, fn2) will also be called
               on stage==('on_val_end', 'train:epoch') - but fn4 won't be,
               on stage=='train:epoch'.
        """
        def _get_matching_stage(cb_keys, stage):
            if stage in cb_keys:
                return stage
            elif stage == ('val_end', 'train:epoch'):
                if 'val_end' in cb_keys:
                    return 'val_end'
                elif 'train:epoch' in cb_keys:
                    return 'train:epoch'
            return None

        if not hasattr(self, 'cb_alias'):
            self.cb_alias = {'train:iter':  'on_train_iter_end',
                             'train:batch': 'on_train_batch_end',
                             'train:epoch': 'on_train_epoch_end',
                             'val:iter':    'on_val_iter_end',
                             'val:batch':   'on_val_batch_end',
                             'val:epoch':   'on_val_epoch_end',
                             'val_end':     'on_val_end',
                             'save':        'on_save',
                             'load':        'on_load'}

        for cb in self.callbacks:
            if isinstance(cb, TraingenCallback):
                _stage = stage if not isinstance(stage, tuple) else stage[0]
                fn = getattr(cb, self.cb_alias[_stage])
                try:
                    fn(stage)
                except NotImplementedError:
                    pass
            elif isinstance(cb, dict):
                _stage = _get_matching_stage(list(cb), stage)
                if _stage is not None:
                    for fn in cb[_stage]:
                        fn(self)
            else:
                raise Exception("unsupported callback type: %s" % type(cb)
                                + "; must be either dict, or subclass "
                                "TraingenCallback.")

    def _init_callbacks(self):
        for cb in self.callbacks:
            if isinstance(cb, TraingenCallback):
                cb.init_with_traingen(self)
        self._init_callbacks_called = True

    ########################## MISC METHODS ##########################
    def check_health(self, dead_threshold=1e-7, dead_notify_above_frac=1e-3,
                     notify_detected_only=True):
        """Check whether any layer weights have 'zeros' or NaN weights;
           very fast.
              - dead_threshold: float. Count values below this as zeros.
              - dead_notify_above_frac: float. If fraction of values exceeds
                    this, print it and the weight's name.
              - notify_detected_only: bool.
                    True  -> print only if dead/NaN found
                    False -> print a 'not found' message
        """
        print_dead_weights(self.model, dead_threshold, dead_notify_above_frac,
                           notify_detected_only)
        print_nan_weights(self.model, notify_detected_only)

    def _alias_to_metric_name(self, alias):
        if alias in self.alias_to_metric:
            return self.alias_to_metric[alias.lower()]
        return alias

    def _metric_name_to_alias(self, metric_name):
        if metric_name in self.metric_to_alias:
            return self.metric_to_alias[metric_name.lower()]
        return metric_name

    def destroy(self, confirm=False, verbose=1):
        """Class 'destructor'. Set own, datagen's, and val_datagen's attributes
        to [] (which can free memory of arrays), then delete them. Also deletes
        'model' attribute, but this has no effect on memory allocation until
        it's dereferenced globally and the TensorFlow/Keras graph is cleared
        (best bet is to restart the Python kernel).
        """
        def _destroy():
            for obj in (self.datagen, self.val_datagen, self):
                attrs = list(vars(obj))
                for attr in attrs:
                    setattr(obj, attr, [])
                    delattr(obj, attr)
                del obj
            gc.collect()
            if verbose:
                print(">>>TrainGenerator SELF-DESTRUCTED")

        if confirm:
            _destroy()
            return
        response = input("!! WARNING !!\nYou are about to destroy TrainGenerator"
                         "; this will wipe all its own and DataGenerator's "
                         "(train & val) shallow-refeerenced data, and delete "
                         "respective attributes. `model` will be dereferenced, "
                         "but not destroyed. Proceed? [y/n]")
        if response == 'y':
            _destroy()

    ########################## INIT METHODS ##########################
    def _prepare_initial_data(self, from_load=False):
        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            if dg.superbatch_set_nums or dg.superbatch_dir:
                dg.preload_superbatch()
            if from_load:
                dg.batch_loaded = False  # load() might've set to True
                dg.preload_labels()      # load() might've changed `labels_path`
            dg.advance_batch()

            pf = '_val' if 'val' in dg_name else ''  # attr prefix
            setattr(self, pf + '_set_name', dg.set_name)
            setattr(self, pf + '_set_num', dg.set_num)

            print("%s initial data prepared" % ("Train" if not pf else "Val"))

    def _init_logger(self):
        if not self.logdir:
            self.model_name = self.get_unique_model_name()
            self.model_num = int(self.model_name.split('__')[0].replace('M', ''))
            self.logdir = os.path.join(self.logs_dir, self.model_name)
            os.mkdir(self.logdir)
            print("Logging ON; directory (new):", self.logdir)
        else:
            print("Logging ON; directory (existing):", self.logdir)

    def _init_and_validate_kwargs(self, kwargs):
        def _validate_kwarg_names(kwargs):
            for kw in kwargs:
                if kw not in _DEFAULT_TRAINGEN_CFG:
                    raise ValueError("unknown kwarg: '{}'".format(kw))

        def _set_kwargs(kwargs):
            class_kwargs = deepcopy(_TRAINGEN_CFG)
            class_kwargs.update(kwargs)

            for attribute in class_kwargs:
                setattr(self, attribute, class_kwargs[attribute])

        def _maybe_set_key_metric_fn():
            if self.eval_fn_name == 'predict' and self.key_metric_fn is None:
                km_name = _get_api_metric_name(self.key_metric, self.model.loss,
                                               self._alias_to_metric_name)
                # if None, will catch in `_validate_traingen_configs`
                self.key_metric_fn = getattr(metrics_fns, km_name, None)

        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)
        _maybe_set_key_metric_fn()
        self._validate_traingen_configs()

    def _init_fit_and_pred_fns(self):
        self.fit_fn_name = self.fit_fn_name or 'train_on_batch'
        self.eval_fn_name = self.eval_fn_name or 'evaluate'

        self.fit_fn = getattr(self.model, self.fit_fn_name)
        self.eval_fn = getattr(self.model, self.eval_fn_name)

    def _init_class_vars(self):
        def _init_misc():
            self.best_key_metric=0 if self.max_is_best else 999
            self.epoch=0
            self.val_epoch=0
            self._set_name=None
            self._val_set_name=None
            self.model_name=self.get_unique_model_name()
            self.model_num=int(self.model_name.split('__')[0].replace('M', ''))
            self._imports = IMPORTS.copy()

            self._history_fig=None
            self._times_validated=0
            self._batches_fit=0
            self._batches_validated=0
            self._fit_iters=0
            self._val_iters=0
            self._has_trained=False
            self._has_validated=False
            self._has_postiter_processed=True
            self._val_has_postiter_processed=True
            self._train_has_notified_of_new_batch=False
            self._val_has_notified_of_new_batch=False
            self._inferred_batch_size=None

            as_empty_list = [
                'key_metric_history', 'best_subset_nums', '_labels',
                '_preds_cache', '_labels_cache', '_sw_cache',
                '_class_labels_cache',
                '_set_name_cache', '_val_set_name_cache',
                '_hist_vlines', '_val_hist_vlines',
                '_train_x_ticks', '_train_val_x_ticks',
                '_val_x_ticks', '_val_train_x_ticks',
                ]
            [setattr(self, name, []) for name in as_empty_list]

        def _init_histories():
            self.history          = {name: [] for name in self.train_metrics}
            self.temp_history     = {name: [] for name in self.train_metrics}
            self.val_history      = {name: [] for name in self.val_metrics}
            self.val_temp_history = {name: [] for name in self.val_metrics}
            self._temp_history_empty     = deepcopy(self.temp_history)
            self._val_temp_history_empty = deepcopy(self.val_temp_history)

        _init_misc()
        _init_histories()
