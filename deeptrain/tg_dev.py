# -*- coding: utf-8 -*-
"""TODO:
    - replace metrics= w/ history=?
    - visualizations
        - layer outs/grads hists w/ 'all' option
    - logging.py ideas:
       - save each class's source code
       - create "init_configs" to log, then also
       getattr(...) for x in init_configs at save time
       - dedicate 'long column'
    - Utils classes (@staticmethod def fn(cls, ..))
    - profiling, configurable (train time, val time, data load time, viz time)
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
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from .util._default_configs import _DEFAULT_TRAINGEN_CFG
from .util.configs  import _TRAINGEN_CFG
from .util.training import _update_temp_history, _get_val_history
from .util.training import _get_sample_weight, _get_api_metric_name
from .util.logging  import _get_unique_model_name
from .util.saving   import save, load, _save_history
from .util.saving   import save_best_model, checkpoint_model_IF
from .util.misc     import pass_on_error, _validate_traingen_configs
from .introspection import print_dead_weights, print_nan_weights
from .introspection import compute_gradient_l2norm
from .visuals import _get_history_fig
from . import metrics as metrics_fns
from .util._backend import IMPORTS, Unbuffered, NOTE, WARN


sys.stdout = Unbuffered(sys.stdout)


class TrainGenerator():
    def __init__(self, model, datagen, val_datagen,
                 epochs=1,
                 logs_dir=None,
                 best_models_dir=None,
                 loadpath=None,
                 callbacks=None,
                 callbacks_init=None,

                 fit_fn_name='train_on_batch',
                 eval_fn_name='evaluate',
                 key_metric='loss',
                 key_metric_fn=None,
                 val_metrics='loss',
                 custom_metrics=None,
                 input_as_labels=False,
                 max_is_best=None,

                 val_freq={'epoch': 1},
                 plot_history_freq={'epoch': 1},
                 viz_freq=None,
                 unique_checkpoint_freq={'epoch': 1},
                 temp_checkpoint_freq=None,

                 class_weights=None,
                 val_class_weights=None,

                 reset_statefuls=False,
                 iter_verbosity=1,
                 optimizer_save_configs=None,
                 optimizer_load_configs=None,
                 plot_configs=None,
                 visualizers=None,
                 model_configs=None,
                 **kwargs):
        self.model=model
        self.datagen=datagen
        self.val_datagen=val_datagen
        self.epochs=epochs
        self.logs_dir=logs_dir
        self.best_models_dir=best_models_dir
        self.loadpath=loadpath
        self.callbacks=callbacks or {}
        self.callbacks_init=callbacks_init or {}

        self.fit_fn_name=fit_fn_name
        self.eval_fn_name=eval_fn_name
        self.key_metric=key_metric
        self.key_metric_fn=key_metric_fn
        self.train_metrics = model.metrics_names.copy()
        self.val_metrics=val_metrics
        self.custom_metrics=custom_metrics
        self.input_as_labels=input_as_labels
        if max_is_best is None:
            value = False if self.key_metric == 'loss' else True
            print(NOTE, "`max_is_best` not set; defaulting to", value)
            self.max_is_best = value
        else:
            self.max_is_best = max_is_best

        self.val_freq=val_freq
        self.plot_history_freq=plot_history_freq
        self.viz_freq=viz_freq or plot_history_freq
        self.unique_checkpoint_freq=unique_checkpoint_freq
        self.temp_checkpoint_freq=temp_checkpoint_freq

        self.class_weights=class_weights
        self.val_class_weights=val_class_weights

        self.reset_statefuls=reset_statefuls
        self.iter_verbosity=iter_verbosity
        self.optimizer_save_configs=optimizer_save_configs
        self.optimizer_load_configs=optimizer_load_configs
        self.plot_configs=plot_configs
        self.visualizers=visualizers
        self.model_configs = model_configs
        self.batch_size=kwargs.pop('batch_size', None) or model.output_shape[0]

        self._init_and_validate_kwargs(kwargs)
        self._init_class_vars()
        self._init_fit_and_pred_fns()
        self._init_callbacks()
        if self.loadpath:
            self.load()  # overwrites model_num, model_name, & others
        else:
            self._prepare_initial_data()
        if self.logs_dir:
            self._init_logger()
        else:
            print(NOTE + "logging OFF")
            self.logdir = None

    ########################## MAIN METHODS ##########################
    def train(self):
        if not self._has_trained:
            while self.epoch < self.epochs:
                if self._has_postiter_processed:
                    x, y, sample_weight = self.get_data(val=False)
                    if self.iter_verbosity:
                        self._print_iter_progress()
                    metrics = self.fit_fn(x, y, sample_weight)
                    self._has_postiter_processed = False

                self._train_postiter_processing(metrics)
                self._has_postiter_processed = True
        else:
            self.validate()
            self.train()

    def validate(self, record_progress=True, clear_cache=True):
        txt = ("Validating" if not self._has_validated else
               "Finishing post-val processing")
        print("\n\n{}...".format(txt))

        while not self._has_validated:
            kw = {}
            if self._val_has_postiter_processed:
                x, self._y_true, self._val_sw = self.get_data(val=True)
                if self.iter_verbosity:
                    self._print_iter_progress(val=True)

                if self.eval_fn_name == 'predict':
                    self._y_preds = self.model.predict(x, batch_size=len(x))
                elif self.eval_fn_name == 'evaluate':
                    kw['metrics'] = self.model.evaluate(
                        x, self._y_true, sample_weight=self._val_sw,
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
            _update_temp_history(self, metrics)
            self._fit_iters += 1
            self.datagen.update_state()
            self._apply_callbacks(stage='train:iter')

        def _on_batch_end():
            self._train_has_notified_of_new_batch = False

            self._batches_fit += 1
            self._train_x_ticks.append(self._batches_fit)
            self._train_val_x_ticks.append(self._times_validated)
            self._set_name_cache.append(self._set_name)
            pass_on_error(self._update_history,
                          print_progress=(self.iter_verbosity >= 1),
                          fail_msg=(
                              WARN + " could not update and print progress - "
                              "OK if right after load; skipping..."))
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
                _update_temp_history(self, metrics, val=True)
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
            self._update_history(val=True, update_val_history=update,
                                 print_progress=(self.iter_verbosity >= 1))
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
            self._checkpoint_model_IF()

        def _clear_cache():
            attrs_to_clear = ('_preds_cache', '_labels_cache', '_sw_cache',
                              '_class_labels_cache',
                              '_set_name_cache', '_val_set_name_cache',
                              '_y_true', '_val_sw')
            [setattr(self, attr, []) for attr in attrs_to_clear]
            self.val_temp_history = deepcopy(self._val_temp_history_empty)

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
                          fail_msg=(WARN + " model history could not be "
                                    "plotted; skipping..."))

        if self.datagen.all_data_exhausted:
            self._apply_callbacks(stage=('val_end', 'train:epoch'))
        else:
            self._apply_callbacks(stage='val_end')

        if clear_cache:
            _clear_cache()
        if self.check_model_health:
            self.check_health()

        self._inferred_batch_size = None  # reset
        self._has_validated = False
        self._has_trained = False

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
            setattr(self, '_val_labels' if val else '_labels',
                    datagen.labels)
            setattr(self, '_val_set_name' if val else '_set_name',
                    datagen.set_name)

        x = datagen.get()
        y = datagen.labels if not self.input_as_labels else x

        class_labels = _standardize_shape(datagen.labels)
        slice_idx = getattr(datagen, 'slice_idx', None)
        sample_weight = _get_sample_weight(self, class_labels, val, slice_idx)

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

    def _get_val_history(self, for_current_iter=False):
        return _get_val_history(self, for_current_iter)

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

    def _update_history(self, val=False, update_val_history=False,
                        print_progress=True):
        if val:
            if print_progress:
                self._print_val_progress()
            if update_val_history:
                self._update_val_history()
        else:
            self._update_train_history()
            if print_progress:
                self._print_train_progress()

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


    ########################## SAVE/LOAD METHODS ##########################
    def _save_best_model(self, del_previous_best=False):
        save_best_model(self, del_previous_best)

    def _checkpoint_model_IF(self, forced=False):
        checkpoint_model_IF(self, forced)

    def save(self, savepath=None):
        save(self, savepath)

    def load(self):
        load(self)

    def _save_history(self, savepath=None):
        _save_history(self, savepath)

    ########################## VISUAL/CALC METHODS ##########################
    def plot_history(self, update_fig=True, w=1, h=1):
        def _show_closed_fig(fig):
            _fig = plt.figure()
            manager = _fig.canvas.manager
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
            plt.show()

        fig = _get_history_fig(self, self.plot_configs, w, h)
        if update_fig:
            self._history_fig = fig
        _show_closed_fig(fig)

    # TODO
    # def show_layer_outputs(self, layer_names=None):
    #     raise NotImplementedError()

    # def show_layer_weights(self, layer_names=None):
    #     raise NotImplementedError()

    # def visualize_gradients(self, on_current_train_batch=True, batch=None,
    #             labels=None, sample_weight=None, learning_phase=0,
    #             slide_size=None, **kwargs):
    #     raise NotImplementedError()

    def compute_gradient_l2norm(self, val=True, learning_phase=0,
                                return_values=False, w=1, h=1):
        return compute_gradient_l2norm(self, val, learning_phase, w, h)


    ########################## CALLBACK METHODS ######################
    def _apply_callbacks(self, stage):
        def _get_matching_stage(cb, stage):
            """Examples:
                1. cb.keys() == ('train:epoch', 'val:batch')
                   stage == ('val_end', 'train:epoch')
                   -> 'train:epoch'
                2. cb.keys() == ('train:epoch', 'val:batch')
                   stage == 'val_end'
                   -> None
                3. cb.keys() == (('val_end', 'train:epoch'), 'train:batch')
                   stage == 'val_end'
                   -> None
                4. cb.keys() == (('val_end', 'train:epoch'), 'train:batch')
                   stage == 'train:batch'
                   -> 'train:batch'
                5. cb.keys() == (('val_end', 'train:epoch'), 'train:batch')
                   stage == ('val_end', 'train:epoch')
                   -> ('train:epoch', 'val_end')
            """
            def _pack_stages(cb, stage):
                cb_stages = []
                for cb_stage in cb:
                    if not isinstance(cb_stage, tuple):
                        cb_stage = (cb_stage,)
                    cb_stages.append(cb_stage)
                stages = stage if isinstance(stage, tuple) else (stage,)
                return tuple(cb_stages), stages

            cb_stages, stages = _pack_stages(cb, stage)
            for cbs in cb_stages:
                if all(x in stages for x in cbs):
                    return cbs if len(cbs) > 1 else cbs[0]

        for name, cb in self.callbacks.items():
            _stage = _get_matching_stage(cb, stage)
            if _stage is None:
                continue
            for fn in cb[_stage]:
                if name in self.callback_objs:
                    fn(self.callback_objs[name])
                else:
                    fn(self)

    def _init_callbacks(self):
        self.callback_objs = {}
        for name, init in self.callbacks_init.items():
            self.callback_objs[name] = init(self)

    ########################## MISC METHODS ##########################
    # very fast, inexpensive
    def check_health(self, dead_threshold=1e-7, dead_notify_above_frac=1e-3,
                     verbose_notify_only=True):
        print_dead_weights(self.model,dead_threshold,
                           dead_notify_above_frac, verbose_notify_only)
        print_nan_weights(self.model, verbose_notify_only)

    def get_unique_model_name(self):
        return _get_unique_model_name(self)

    def _alias_to_metric_name(self, alias):
        if alias in self.alias_to_metric:
            return self.alias_to_metric[alias.lower()]
        return alias

    def _metric_name_to_alias(self, metric_name):
        if metric_name in self.metric_to_alias:
            return self.metric_to_alias[metric_name.lower()]
        return metric_name

    ########################## INIT METHODS ##########################
    def _prepare_initial_data(self):
        if self.datagen.superbatch_set_nums != []:
            self.datagen.preload_superbatch()
        self.datagen.advance_batch()
        self._labels = self.datagen.labels
        self._set_name = self.datagen.set_name
        print("Train initial data prepared")

        if self.val_datagen.superbatch_set_nums != []:
            self.val_datagen.preload_superbatch()
        self.val_datagen.advance_batch()
        self._val_labels = self.val_datagen.labels
        self._val_set_name = self.val_datagen.set_name
        print("Val initial data prepared")

    def _init_logger(self):
        base_name = 'M%s' % self.model_num
        _path = [os.path.join(self.logs_dir, filename) for filename in
                 sorted(os.listdir(self.logs_dir)) if base_name in filename]

        if _path == [] or self.make_new_logdir:
            if self.make_new_logdir:
                self.model_name = self.get_unique_model_name()
                self.model_num = int(self.model_name.split('__')[0].replace(
                    'M', ''))
            _path = os.path.join(self.logs_dir, self.model_name)
            os.makedirs(_path)
            print("Logging ON; directory (new):", _path)
        else:
            print("Logging ON; directory (existing):", _path)

        _path = _path[0] if isinstance(_path, list) else _path
        self.logdir = _path

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
        _validate_traingen_configs(self)

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

        def _init_max_set_name_chars():
            set_names = getattr(self.datagen, 'set_nams_original', None)
            val_set_names = getattr(self.val_datagen, 'set_names_original', None)
            if set_names is not None:
                names_str = map(str, set_names)
                self._max_set_name_chars = max(map(len, names_str))
            else:
                self._max_set_name_chars = 3  # guess
            if val_set_names is not None:
                names_str = map(str, val_set_names)
                self._val_max_set_name_chars = max(map(len, names_str))
            else:
                self._val_max_set_name_chars = 2  # guess

        _init_misc()
        _init_histories()
        _init_max_set_name_chars()

        self._pil_imported = IMPORTS['PIL']
