# -*- coding: utf-8 -*-
import os
import pickle
import types
import h5py
import tensorflow as tf

from pathlib import Path
from ._backend import K, WARN, NOTE
from ..visuals import _get_history_fig
from .misc import pass_on_error, _train_on_batch_dummy


def _save_best_model(self, del_previous_best=False):
    def _del_previous_best():
        def _get_prev_files():
            return [os.path.join(self.best_models_dir, name)
                    for name in os.listdir(self.best_models_dir)
                    if str(self.model_num) in name]
        prev_files = _get_prev_files()
        if len(prev_files) != 0:
            [os.remove(f) for f in prev_files]

    def _update_best_key_metric_in_model_name(keyword='__max'):
        self.model_name = self.model_name.split(keyword)[0] + keyword + (
            '%.3f' % self.best_key_metric).replace('0.', '.')

    if del_previous_best:
        try:
            _del_previous_best()
        except BaseException as e:
            print(WARN,  "previous best model files could not be deleted; "
                  "skipping")
            print("Errmsg:", e)

    self.best_key_metric = round(self.key_metric_history[-1], 6)
    _update_best_key_metric_in_model_name()

    savepath = os.path.join(self.best_models_dir, self.model_name)
    self.model.save_weights(savepath + '.h5')
    self._history_fig = _get_history_fig(self)
    self._history_fig.savefig(savepath + '.png')

    self.save(savepath + '__state.h5')

    if self._imports.get('PIL', False):
        try:
            self.generate_report(savepath + '__report.png')
        except BaseException as e:
            print(WARN,  "Best model report could not be saved; skipping")
            print("Errmsg", e)
    print("Best model saved to " + savepath)


def checkpoint_model(self):  # TODO maybe clear
    def _checkpoint_type():
        do_temp = self._should_do(self.temp_checkpoint_freq)
        do_unique = self._should_do(self.unique_checkpoint_freq)

        if not (do_temp or do_unique):
            return False
        return do_temp, do_unique

    def _get_savename(do_temp, do_unique):
        if do_temp and not do_unique:  # give latter precedence
            return "_temp_model"

        if self.logs_use_full_model_name:
            return "{}_{}vals__".format(self.model_name, self._times_validated)
        else:
            return "max_{:.3f}_{}vals__".format(self.best_key_metric,
                                                self._times_validated)

    def _clear_logs_IF():
        _paths = [str(x) for x in Path(self.logdir).iterdir() if x.is_file()]
        # (model weights, traingen state, report, history img)
        # TODO infer actual number, do not hard-code
        paths_per_checkpoint = 4
        while len(_paths) / paths_per_checkpoint > self.max_checkpoint_saves:
            [os.remove(_paths.pop(0)) for _ in range(paths_per_checkpoint)]

    do_checkpoint = _checkpoint_type()
    if not do_checkpoint:
        return
    do_temp, do_unique = do_checkpoint

    savename = _get_savename(do_temp, do_unique)
    _path = os.path.join(self.logdir, savename)

    self.model.save_weights(_path + 'weights.h5')
    self.save(_path + 'state.h5')
    self._save_history(_path + 'hist.png')
    self.generate_report(_path + 'report.png')

    try:
        _clear_logs_IF()
    except BaseException as e:
        print(WARN,  "Model logs could not be cleared; skipping")
        print("Errmsg:", e)


def save(self, savepath=None):
    def _get_optimizer_state(opt):
        def _get_attrs_to_save(opt):
            cfg = self.optimizer_save_configs
            all_attrs = [a for a in list(vars(opt).keys()) if a != 'updates']

            if cfg is None:
                return all_attrs

            if 'exclude' in cfg:
                if 'updates' not in cfg['exclude']:
                    print(NOTE, "saving 'updates' is unsupported; will skip")
                return [a for a in all_attrs if a not in cfg['exclude']]

            if 'include' in cfg:
                if 'updates' in cfg['include']:
                    print(WARN, "saving 'updates' is unsupported; skipping")
                attrs = []
                for attr in cfg['include']:
                    if attr in all_attrs:
                        attrs.append(attr)
                    else:
                        print(WARN, ("'{}' attribute not found in optimizer; "
                                     "skipping").format(attr))
                return attrs

        def _get_tensor_value(tensor):
            try:
                return K.get_value(tensor)
            except:
                return K.eval(tensor)

        def _get_optimizer_weights(optimizer):
            weights = K.batch_get_value(optimizer.weights)
            if weights == []:
                print(WARN, "optimizer 'weights' empty, and will not be saved")
            return weights

        state = {}
        to_save = _get_attrs_to_save(opt)
        for name in to_save:
            if name == 'weights':
                weights = _get_optimizer_weights(opt)
                if weights != []:
                    state['weights'] = weights
                continue

            value = getattr(opt, name, None)
            if isinstance(value, tf.Variable):
                state[name] = _get_tensor_value(value)
            else:
                state[name] = value
        return state

    def _cache_datagen_attributes():
        # TODO support configurable caching
        def _cache_then_del_attrs(parent_obj, child_obj_names, to_exclude):
            if not isinstance(child_obj_names, (list, tuple)):
                child_obj_names = [child_obj_names]

            cached_attrs = {}
            for child_name in child_obj_names:
                for attr_name in to_exclude:
                    obj = getattr(parent_obj, child_name)
                    attr_value = getattr(obj, attr_name)

                    cache_name = child_name + '.' + attr_name
                    cached_attrs[cache_name] = attr_value
                    delattr(obj, attr_name)
            return cached_attrs

        cached_attrs = {}
        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            to_exclude = ['batch', 'superbatch']

            for key, val in vars(dg).items():
                if isinstance(val, types.LambdaType):
                    to_exclude.append(key)
                elif key == 'group_batch':
                    to_exclude.append('group_batch')
                elif key == 'hdf5_dataset':
                    dg.hdf5_dataset.close()
                    dg.hdf5_dataset = []
            cached_attrs.update(_cache_then_del_attrs(self, dg_name, to_exclude))
        return cached_attrs

    def _restore_cached_attributes(parent_obj, cached_attrs):
        for obj_attr_name in cached_attrs:
            obj_name, attr_name = obj_attr_name.split('.')
            obj = getattr(parent_obj, obj_name)
            attr_value = cached_attrs[obj_attr_name]
            setattr(obj, attr_name, attr_value)

    savepath = savepath or os.path.join(self.logdir, '_temp_model__state.h5')
    cached_attrs = _cache_datagen_attributes()

    self._apply_callbacks(stage='save')
    savedict = {k:v for k,v in vars(self).items() if k in self.savelist}
    savedict['optimizer_state'] = _get_optimizer_state(self.model.optimizer)

    try:
        with open(savepath, "wb") as savefile:
            pickle.dump(savedict, savefile)
            print("TrainGenerator state saved")
    except BaseException as e:
        print(WARN,  "TrainGenerator state could not be saved; skipping...")
        print("Errmsg:", e)

    _restore_cached_attributes(self, cached_attrs)

    for dg in (self.datagen, self.val_datagen):
        if 'hdf5_dataset' in vars(dg):
            dg.hdf5_dataset = h5py.File(dg.data_dir, 'r')  # re-open


def load(self, filepath=None):
    def _get_filepath(filepath):
        filepath = filepath or self.loadpath
        if filepath is not None:
            return filepath

        basetxt = ("`loadpath` and passed in `filepath` are None")
        txt = None
        if self.logdir is None:
            txt = basetxt + ", and `logdir` is None"
        elif not Path(self.logdir).is_dir():
            txt = basetxt + f", and `logdir` is not a folder ({self.logdir})"
        if txt is not None:
            raise ValueError(txt)

        tempname = '_temp_model__state.h5'
        if tempname not in os.listdir(self.logdir):
            raise ValueError(basetxt + f" and tempfile default {tempname} not "
                             f"found in `logdir` ({self.logdir})")

        filepath = os.path.join(self.logdir, '_temp_model__state.h5')
        return filepath

    def _cache_datagen_attrs():
        def _cache_preprocessor_attrs(pp):
            pp_cache = {}
            for attr in pp.loadskip_list:
                value = getattr(pp, attr)
                if isinstance(value, (list, dict)):
                    value = value.copy()
                pp_cache[attr] = value
            return pp_cache

        caches = {'datagen': {}, 'val_datagen': {}}
        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            caches[dg_name]['preprocessor'] = _cache_preprocessor_attrs(
                dg.preprocessor)

            for attr in dg.loadskip_list:
                value = getattr(dg, attr)
                if isinstance(value, (list, dict)):
                    value = value.copy()
                caches[dg_name][attr] = value
        return caches

    def _restore_cached_attrs(caches):
        def _restore_preprocessor_attrs(pp, pp_cache):
            for attr, value in pp_cache.items():
                setattr(pp, attr, value)

        def _check_and_fix_set_nums(dg):
            if hasattr(dg, 'set_nums_original'):
                if any([(set_num not in dg.set_nums_original) for
                        set_num in dg.set_nums_to_process]):
                    print(WARN,  "found set_num in loaded"
                          "`set_nums_to_process` that isn't in "
                          "the non-loaded `set_nums_original`; setting "
                          "former to `set_nums_original`.")
                    dg.set_nums_to_process = dg.set_nums_original.copy()

        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            pp_cache = caches[dg_name].pop('preprocessor')
            _restore_preprocessor_attrs(dg.preprocessor, pp_cache)

            for attr, value in caches[dg_name].items():
                setattr(dg, attr, value)
            _check_and_fix_set_nums(dg)

    def _unpack_passed_dirs(caches):
        dgs = (self.datagen, self.val_datagen)
        for dg_type, dg in zip(caches, dgs):
            for attr in dg._path_attrs:
                setattr(dg, attr, caches[dg_type][attr])

    filepath = _get_filepath(filepath)
    with open(filepath, "rb") as loadfile:
        loadfile_parsed = pickle.load(loadfile)
        caches = _cache_datagen_attrs()

        # load keys & values w/o deleting existing
        for dg_name in ('datagen', 'val_datagen'):
            for key, value in loadfile_parsed.pop(dg_name).__dict__.items():
                setattr(getattr(self, dg_name), key, value)

        # assign loaded/cached attributes
        self.__dict__.update(loadfile_parsed)

        if self.use_passed_dirs_over_loaded:
            _unpack_passed_dirs(caches)
        if hasattr(self, 'optimizer_state') and self.optimizer_state:
            _load_optimizer_state(self)
        else:
            print(WARN, "'optimizer_state' not found in loadfile; skipping")
        print("TrainGenerator state loaded from", self.loadpath)

    print("--Preloading excluded data based on datagen states ...")
    _restore_cached_attrs(caches)

    for dg_name in ('datagen', 'val_datagen'):
        dg = getattr(self, dg_name)
        dg.batch_loaded = False

        if 'hdf5_dataset' in vars(dg):
            dg.hdf5_dataset = h5py.File(dg.data_dir, 'r')
        if dg.superbatch_set_nums != []:
            dg.preload_superbatch()
        dg.preload_labels()
        dg.advance_batch()

    self._set_num     = self.datagen.set_num
    self._val_set_num = self.val_datagen.set_num

    self._apply_callbacks(stage='load')
    print("... finished--")


def _load_optimizer_state(self):
    def _get_attrs_to_load(opt):
        cfg = self.optimizer_load_configs
        all_attrs = [a for a in list(vars(opt).keys()) if a != 'updates']

        if cfg is None:
            return all_attrs
        if 'exclude' in cfg:
            return [a for a in all_attrs if a not in cfg['exclude']]
        if 'include' in cfg:
            return [a for a in all_attrs if a in cfg['include']]

    opt = self.model.optimizer
    to_load = _get_attrs_to_load(opt)

    for name, value in self.optimizer_state.items():
        if name not in to_load:
            continue
        elif name == 'weights':
            continue  # set later

        if isinstance(getattr(opt, name), tf.Variable):
            K.set_value(getattr(opt, name), value)
        else:
            setattr(opt, name, value)

    _train_on_batch_dummy(self.model, self.class_weights,
                          input_as_labels=self.input_as_labels,
                          alias_to_metric_name_fn=self._alias_to_metric_name)
    if 'weights' in self.optimizer_state:
        self.model.optimizer.set_weights(self.optimizer_state['weights'])

    self.optimizer_state = None  # free up memory
    print("Optimizer state loaded (& cleared from TrainGenerator)")


def _save_history(self, savepath=None):
    def _save_epoch_fig():
        prev_figs = [os.path.join(self.final_fig_dir, name)
            for name in os.listdir(self.final_fig_dir)
            if str(self.model_num) in name]
        if len(prev_figs) != 0:
            [os.remove(fig) for fig in prev_figs]

        _savepath = os.path.join(self.final_fig_dir, self.model_name + '.png')
        self._history_fig.savefig(_savepath)

    _path = savepath or os.path.join(self.logdir, '_temp_model__hist.png')
    if self._history_fig:
        try:
            self._history_fig.savefig(_path + '.png')
        except:
            print(WARN,  "Model history could not be saved; skipping")

    # TODO: rename `final_fig_dir`?
    if self.final_fig_dir:  # keep at most one per model_num
        pass_on_error(_save_epoch_fig, fail_msg=("Epoch fig could not be "
                                                 "saved; skipping"))
