# -*- coding: utf-8 -*-
"""TODO:
    - replace metrics= w/ history=?
    - visualizations
    - metric aliases
    - unit tests: 
        - images
        - timeseries
        - autoencoders
        - save/load
        - report generator
        - data generators
        - visualizations
        - weighted sample_weight
    - MetaTrainer
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from types import LambdaType

from .util import metrics as metrics_fns
from .util.configs  import _PLOT_CFG, _TRAINGEN_CFG
from .util._default_configs import _DEFAULT_TRAINGEN_CFG
from .util.training import _update_temp_history, _get_val_history
from .util.training import _get_weighted_sample_weights
from .util.logging import _get_unique_model_name
from .util.logging import generate_report
from .util.visuals import get_history_fig, show_predictions_per_iteration
from .util.visuals import show_predictions_distribution
from .util.visuals import comparative_histogram
from .util.saving  import save, load, _save_history
from .util.saving  import save_best_model, checkpoint_model_IF
from .util.misc    import pass_on_error
from .util.introspection import print_dead_weights, print_nan_weights
from .util.introspection import compute_gradient_l2norm
from .util import Unbuffered, NOTE, WARN


sys.stdout = Unbuffered(sys.stdout)


class TrainGenerator():
    ## TODO move elsewhere?
    BUILTIN_METRICS = ('binary_crossentropy', 'categorical_crossentropy',
                       'sparse_categorical_crossentropy', 'f1', 'tnr', 'tpr')

    def __init__(self, model, datagen, val_datagen,
                 epochs=1,
                 logs_dir=None,
                 best_models_dir=None,
                 loadpath=None,
                 
                 fit_fn_name='train_on_batch',
                 eval_fn_name='evaluate',
                 key_metric='loss',
                 key_metric_fn=None,
                 train_metrics=None,
                 val_metrics=None,
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

        self.fit_fn_name=fit_fn_name
        self.eval_fn_name=eval_fn_name
        self.key_metric=key_metric
        self.key_metric_fn=key_metric_fn
        self.train_metrics=train_metrics
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
        self.plot_configs=plot_configs or _PLOT_CFG
        self.visualizers=visualizers
        self.model_configs = model_configs
        self.batch_size=kwargs.pop('batch_size', None) or model.output_shape[0]

        self._init_and_validate_kwargs(kwargs)
        self._init_class_vars()
        if self.loadpath:
            self.load()  # overwrites model_num, model_name, & others
        else:                        
            self._prepare_initial_data()
        if self.logs_dir:
            self._init_logger()
        else:
            print(NOTE + "logging OFF")
            self.logdir = None
        self._init_fit_and_pred_fns()


    ########################## MAIN METHODS ##########################
    def train(self):
        if not self._has_trained:
            while self.epoch < self.epochs:
                if self._has_postiter_processed:
                    x, y, sample_weights = self.get_data(val=False)
                    if self.iter_verbosity:
                        self._print_iter_progress()
                    metrics = self.fit_fn(x, y, sample_weights)
                    self._update_temp_history(metrics)

                self._has_postiter_processed = False
                self._train_postiter_processing()
                self._has_postiter_processed = True
        else:
            self.validate()
            self.train()

    def validate(self, record_progress=True, plot_history=True,
                 do_visualization=True, clear_cache=True):
        txt = ("Validating" if not self._has_validated else
               "Finishing post-val processing")
        print("\n\n{}...".format(txt))

        while not self._has_validated:
            if self._val_has_postiter_processed:
                x, y, self._val_sw = self.get_data(val=True)
                self._y_true = y
                if self.iter_verbosity:
                    self._print_iter_progress(val=True)

                if self.eval_fn_name == 'predict':
                    self._y_preds = self.model.predict(x)
                elif self.eval_fn_name == 'evaluate':
                    metrics = self.model.evaluate(
                        x, y, sample_weight=self._val_sw, verbose=0)
                    self._update_temp_history(metrics, val=True)

            self._val_has_postiter_processed = False
            self._val_postiter_processing(record_progress)
            self._val_has_postiter_processed = True

        if self._has_validated:
            self._on_val_end(record_progress, clear_cache)

    ######################### MAIN METHOD HELPERS ########################
    def _train_postiter_processing(self):
        def _on_iter_end():
            self.datagen.update_state()

        def _on_batch_end():
            self._train_has_notified_of_new_batch = False
            
            self._batches_fit += 1
            self._train_x_ticks.append(self._batches_fit)
            self._train_val_x_ticks.append(self._times_validated)
            self._set_num_cache.append(self._set_num)
            pass_on_error(self._update_history, 
                          print_progress=(self.iter_verbosity >= 1),
                          fail_msg=(
                              WARN + " could not update and print progress - "
                              "OK if right after load; skipping..."))

            self.temp_history = deepcopy(self._temp_history_empty)
    
            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset ', end='')
        
        def _on_epoch_end(val=False):
            self.epoch = self.datagen.on_epoch_end()
            decor = "\n_________________________\n\033[4m {}{}{} \033[0m\n"
            print(decor.format("EPOCH ", self.epoch, " -- COMPLETE"))

            self._hist_vlines     += [self._batches_fit]
            self._val_hist_vlines += [self._times_validated]

            if self.save_post_epoch:
                self._checkpoint_model_IF()

        def _should_validate():
            return self._should_do(self.val_freq)

        _on_iter_end()
        if self.datagen.batch_exhausted:
            _on_batch_end()
        if self.datagen.all_data_exhausted:
            _on_epoch_end()

        if _should_validate():
            self._has_postiter_processed = True  # in case val is interrupted
            self._has_trained = True
            self._has_validated = False 
            self.validate()

    def _val_postiter_processing(self, record_progress=True):
        def _on_iter_end():
            if self.eval_fn_name == 'predict':
                self._update_val_iter_cache()
            self.val_datagen.update_state()

        def _on_batch_end():
            update = record_progress and self.val_datagen.all_data_exhausted
            self._update_history(val=True, update_val_history=update,
                                 print_progress=(self.iter_verbosity >= 1))
            self._batches_validated += 1
            self._val_set_num_cache.append(self._val_set_num)
            self._val_has_notified_of_new_batch = False
            self.val_temp_history = deepcopy(self._val_temp_history_empty)

            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset', end=' ')
        
        def _on_epoch_end():
            self._has_validated = True
            self.val_epoch = self.val_datagen.on_epoch_end()
            self._val_x_ticks += [self._times_validated]
            self._val_train_x_ticks += [self._batches_fit]

        _on_iter_end()
        if self.val_datagen.batch_exhausted:
            _on_batch_end()
        if self.val_datagen.all_data_exhausted:
            _on_epoch_end()
    

    def _on_val_end(self, record_progress, clear_cache):
        def _record_progress():
            self._times_validated += 1
            new_best = bool(self.key_metric_history[-1] > self.best_key_metric)
            if not self.max_is_best:
                new_best = not new_best

            if new_best and self.best_models_dir is not None:
                self._save_best_model(
                    del_previous_best=self.keep_one_best_model)
            self._checkpoint_model_IF()

        def _clear_cache():
            attrs_to_clear = ('_preds_cache', '_labels_cache', '_sw_cache',
                              '_set_num_cache', '_val_set_num_cache',
                              '_y_true', '_val_sw')
            [setattr(self, attr, []) for attr in attrs_to_clear]

        def _should_plot():
            return (self._should_do(self.plot_history_freq),
                    self._should_do(self.viz_freq))

        def _print_best_subset():         
            best_nums = ", ".join([str(x) for x in self.best_subset_nums])
            best_size = self.best_subset_size
            print("Best {}-subset: {}".format(best_size, best_nums))

        if self.best_subset_size != 0:
            _print_best_subset()
        
        plot_history, do_visualization = _should_plot()
        if plot_history or do_visualization:
            self.do_plotting(plot_history, do_visualization)

        if record_progress:
            _record_progress()

        if clear_cache: 
            _clear_cache()

        if self.check_model_health:
            self.check_health()

        self._has_validated = False
        self._has_trained = False
    
    def do_plotting(self, plot_history=True, do_visualization=True, 
                    record_progress=False):
        if plot_history:
            pass_on_error(self.plot_history, update_fig=record_progress,
                          fail_msg=(WARN + " model history could not be "
                                    "plotted; skipping..."))
        if do_visualization:
            # self.show_layer_outputs()
            # self.show_layer_weights()
            if self.eval_fn_name == 'predict':
                self.show_model_outputs()
            
            can_viz = self.visualizers is not None and (
                self.eval_fn_name == 'predict' or 
                any([isinstance(x, LambdaType) for x in self.visualizers]))
            if can_viz:
                for viz in self.visualizers:
                    if viz == 'predictions_per_iteration':            
                        self.show_predictions_per_iteration()
                    elif viz == 'predictions_distribution':
                        self.show_predictions_distribution()
                    elif isinstance(viz, LambdaType):
                        viz(self)

    def _should_do(self, config, forced=False):
        if forced:
            return True
        if config is None:
            return False
        freq_mode, freq_value = list(config.items())[0]

        if freq_mode == 'iter':
            return True
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
        datagen = self.val_datagen if val else self.datagen
        
        if datagen.batch_exhausted:
            datagen.advance_batch()
            setattr(self, '_val_labels' if val else '_labels',   datagen.labels)
            setattr(self, '_val_set_num' if val else '_set_num', datagen.set_num)

        x = datagen.get()
        y = datagen.labels if not self.input_as_labels else x

        class_labels = datagen.labels
        slice_idx = getattr(datagen, 'slice_idx', None)
        sample_weights = self.get_sample_weights(class_labels, val, slice_idx)

        return x, y, sample_weights

    def get_sample_weights(self, labels, val=False, slice_idx=None):
        if self.weighted_slices_range is not None:
            return _get_weighted_sample_weights(
                self, labels, val, self.weighted_slices_range, slice_idx)

        cw = self.val_class_weights if val else self.class_weights
        if cw is not None:
            return np.asarray([cw[str(label)] for label in labels])
        return np.ones(labels.shape[0])

    ########################## LOG METHODS ################################
    def _update_temp_history(self, metrics, val=False):
        _update_temp_history(self, metrics, val=val)

    def _update_val_iter_cache(self):
        if getattr(self.val_datagen, 'slice_idx', None) is None:
            self._sw_cache.append(self._val_sw)
            self._preds_cache.append(np.squeeze(self._y_preds))
            self._labels_cache.append(self.val_datagen.labels)
            return
        
        if getattr(self.val_datagen, 'slice_idx', None) == 0:
            self._labels_cache.append([])
            self._sw_cache.append([])
            if self.eval_fn_name == 'predict':
                self._preds_cache.append([])

        self._sw_cache[-1].append(self._val_sw)
        if self.eval_fn_name == 'predict':
            self._preds_cache[-1].append(np.squeeze(self._y_preds))
            self._labels_cache[-1].append(self._y_true)
        elif self.eval_fn_name == 'evaluate':
            self._labels_cache[-1].append(self.val_datagen.labels)

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
        for name in self.metric_printskip_configs['train']:
            train_metrics.pop(name, None)
        self._print_progress(train_metrics, endchar='')

    def _print_val_progress(self):
        val_metrics = self._get_val_history(for_current_iter=True)
        for name in self.metric_printskip_configs['val']:
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
                pad = self._val_max_set_num_digits + 3
                padded_num_txt = (str(self._val_set_num) + "...").ljust(pad)
                print(end="Validating set %s" % padded_num_txt)
                self._val_has_notified_of_new_batch = True
            return

        if not self._train_has_notified_of_new_batch:
            pad = self._max_set_num_digits + 3
            padded_num_txt = (str(self._set_num) + "...").ljust(pad)
            print(end="\nFitting set %s" % padded_num_txt)
            self._train_has_notified_of_new_batch = True
        if self.iter_verbosity >= 2:
            print(end='.')

    def generate_report(self, savepath):
        generate_report(self, savepath)

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

        fig = self._get_history_fig(self.plot_configs, w, h)
        if update_fig:
            self._history_fig = fig
        _show_closed_fig(fig)

    def _get_history_fig(self, plot_configs=None, w=1, h=1):
        return get_history_fig(self, plot_configs, w, h)
    
    def show_layer_outputs(self, layer_names=None):
        raise NotImplementedError()  #TODO

    def show_layer_weights(self, layer_names=None):
        raise NotImplementedError()  #TODO

    def show_model_outputs(self):
        if self.outputs_visualizer == 'comparative_histogram':
            comparative_histogram(
                self.model,
                layer_name='output',
                data=self.val_datagen.get(),
                vline=self.predict_threshold, 
                xlims=(0,1))
        elif isinstance(self.outputs_visualizer, LambdaType):
            self.outputs_visualizer(self)

    def show_predictions_per_iteration(self):
        show_predictions_per_iteration(self._labels_cache, self._preds_cache)
        
    def show_predictions_distribution(self):
        show_predictions_distribution(self._labels_cache, self._preds_cache,
                                      self.predict_threshold)
            
    def compute_gradient_l2norm(self, val=True, learning_phase=0,
                                return_values=False, w=1, h=1):
        return compute_gradient_l2norm(self, val, learning_phase, w, h)
        
    def visualize_gradients(self, on_current_train_batch=True, batch=None,
                labels=None, sample_weights=None, learning_phase=0,
                slide_size=None, **kwargs):
        raise NotImplementedError()  #TODO

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
        self._set_num = self.datagen.set_num
        print("Train initial data prepared")

        if self.val_datagen.superbatch_set_nums != []:
            self.val_datagen.preload_superbatch()
        self.val_datagen.advance_batch()
        self._val_labels = self.val_datagen.labels
        self._val_set_num = self.val_datagen.set_num
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
        def _set_metrics_from_model():
            model_metrics = ['loss', *self.model.metrics]
            if self.train_metrics is None:
                self.train_metrics = model_metrics
            if self.val_metrics is None:
                self.val_metrics = model_metrics

        def _validate_metrics():
            for name in ('train_metrics', 'val_metrics'):
                value = getattr(self, name)
                if not isinstance(value, list):
                    if isinstance(value, str):
                        setattr(self, name, [value])
                    else:
                        setattr(self, name, list(value))
                        
            def _from_model(metric):
                return metric != 'loss' and metric not in [
                    self.model.loss, *self.model.metrics]

            metrics = (*self.train_metrics, *self.val_metrics, self.key_metric)
            supported = TrainGenerator.BUILTIN_METRICS
            customs = self.custom_metrics or [None]
            
            if self.eval_fn_name == 'predict':
                for metric in metrics:
                    if not _from_model(metric) and metric not in (
                            *supported, *customs):
                        raise ValueError((
                            "'{0}' metric is not supported; add a function to "
                            "`custom_metrics` as '{0}': func. Supported "
                            "are: {1}").format(metric, ', '.join(supported)))
    
                if self.model.loss not in (*supported, *customs):
                        raise ValueError((
                            "'{0}' loss is not supported w/ `eval_fn_name = "
                            "'predict'`; add a function to `custom_metrics` "
                            "as '{0}': func, or set `eval_fn_name = 'evaluate'`."
                            " Supported are: {1}").format(
                                self.model.loss, ', '.join(supported)))

                km = (self.key_metric if self.key_metric != 'loss' 
                      else self.model.loss)
                if km not in supported and self.key_metric_fn is None:
                    raise ValueError(("`key_metric = '{}'` is unsupported; set "
                                      "`key_metric_fn = func`. Supported are: {}"
                                      ).format(km, ', '.join(supported)))

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
            cfgs = (self.optimizer_save_configs, self.optimizer_load_configs)
            for cfg in cfgs:
                if cfg is not None and 'include' in cfg and 'exclude' in cfg:
                    raise ValueError("cannot have both 'include' and 'exclude' "
                                     "in `optimizer_save_configs` or "
                                     "`optimizer_load_configs`")
        
        def _validate_visualizers():
            if (self.visualizers is not None and self.eval_fn_name != 'predict'
                and not any([isinstance(x, LambdaType) for x in
                               self.visualizers])):
                print(WARN, "`eval_fn_name != 'predict'`, cannot use built-in "
                      "`visualizers`; include a custom function")
        
        def _validate_savelist():
            if self.input_as_labels and 'labels' in self.savelist:
                print(NOTE, "will exclude `labels` from saving when "
                      "`input_as_labels=True`; to override, "
                      "supply '{labels}' instead")
                self.savelist.pop(self.savelist.index('labels'))
            if '{labels}' in self.savelist:
                self.savelist.pop(self.savelist.index('{labels}'))
                self.savelist.append('labels')
        
        def _validate_weighted_slices_range():
            if self.weighted_slices_range is not None:
                if not (hasattr(self.datagen, 'weighted_slices_range') and 
                        hasattr(self.val_datagen, 'weighted_slices_range')):
                    raise ValueError("to use `weighted_slices_range`, `datagen`"
                                     " and `val_datagen` must have `weighted_"
                                     "slices_range` attribute defined ( via "
                                     "`preprocessor`).")

                no_slices = self.val_datagen.slices_per_batch in {1, None}
                if no_slices:
                    print(WARN, "BatchGenerator uses no slices; setting "
                          "`weighted_slices_range=None`")
                    self.weighted_slices_range = None

        def _validate_class_weights():
            for name in ('class_weights', 'val_class_weights'):
                cw = getattr(self, name)
                if cw is not None:
                    assert all([isinstance(x, int) for x in cw.keys()]), (
                        "`{}` classes must be of type int (got {})"
                        ).format(name, cw)
                    assert ((0 in cw and 1 in cw) or cw.sum() > 1), (
                        "`{}` must contain classes 1 and 0, or greater "
                        "(got {})").format(name, cw)

        def _validate_best_subset_size():
            if self.best_subset_size is not None and self.batch_size is None:
                raise ValueError("`batch_size` cannot be None to use `best_"
                                 "subset_size`")

        def _validate_kwarg_names(kwargs):
            for kw in kwargs:
                if kw not in _DEFAULT_TRAINGEN_CFG:
                    raise ValueError("unknown kwarg: '{}'".format(kw))

        def _set_kwargs(kwargs):
            class_kwargs = _TRAINGEN_CFG
            class_kwargs.update(kwargs)

            for attribute in class_kwargs:
                setattr(self, attribute, class_kwargs[attribute])
        
        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)

        if self.train_metrics is None or self.val_metrics is None:
            _set_metrics_from_model()
        _validate_metrics()
        _validate_directories()
        _validate_optimizer_saving_configs()
        _validate_visualizers()
        _validate_savelist()
        _validate_weighted_slices_range()
        _validate_class_weights()
        _validate_best_subset_size()
        
        if self.eval_fn_name == 'predict' and self.key_metric_fn is None:
            km = self.key_metric if self.key_metric != 'loss' else (
                self.model.loss)
            self.key_metric_fn = getattr(metrics_fns, km)

    def _init_fit_and_pred_fns(self):
        self.fit_fn_name = self.fit_fn_name or 'train_on_batch'
        self.eval_fn_name = self.eval_fn_name or 'evaluate'  #TODO eval_fn_name

        self.fit_fn = getattr(self.model, self.fit_fn_name)
        self.eval_fn = getattr(self.model, self.eval_fn_name)

    def _init_class_vars(self):
        def _init_misc():
            self.best_key_metric=0 if self.max_is_best else 999
            self.epoch=0
            self.val_epoch=0
            self._set_num=-1
            self._val_set_num=-1
            self.model_name=self.get_unique_model_name()
            self.model_num=int(self.model_name.split('__')[0].replace('M', ''))
    
            self._history_fig=None
            self._times_validated=0
            self._batches_fit=0
            self._batches_validated=0
            self._has_trained=False
            self._has_validated=False
            self._has_postiter_processed=True
            self._val_has_postiter_processed=True
            self._train_has_notified_of_new_batch=False
            self._val_has_notified_of_new_batch=False

            as_empty_list = [
                'key_metric_history', 'best_subset_nums', '_labels',
                '_preds_cache', '_labels_cache', '_sw_cache',
                '_set_num_cache', '_val_set_num_cache',
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
        
        def _init_max_set_num_digits():
            set_nums = getattr(self.datagen, 'set_nums_original', None)
            val_set_nums = getattr(self.val_datagen, 'set_nums_original', None)
            if set_nums is not None:
                nums_str = map(str, set_nums)
                self._max_set_num_digits = max(map(len, nums_str))
            else:
                self._max_set_num_digits = 3  # guess
            if val_set_nums is not None:
                nums_str = map(str, val_set_nums)
                self._val_max_set_num_digits = max(map(len, nums_str))
            else:
                self._val_max_set_num_digits = 2  # guess

        _init_misc()
        _init_histories()
        _init_max_set_num_digits()