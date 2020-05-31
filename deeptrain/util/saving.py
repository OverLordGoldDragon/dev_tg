# -*- coding: utf-8 -*-
import os
import pickle
import types
import tensorflow as tf

from pathlib import Path
from ._backend import K, WARN
from ..visuals import _get_history_fig
from .misc import pass_on_error, _init_optimizer, exclude_unpickleable
from ..backend import tensor_util


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

    basepath = os.path.join(self.best_models_dir, self.model_name)
    save_fns = self._make_model_save_fns(basepath + '__')
    for path, save_fn in save_fns:
        save_fn(path)

    self._history_fig = _get_history_fig(self)
    self._history_fig.savefig(basepath + '.png')

    self.save(basepath + '__state.h5')

    if self._imports.get('PIL', False):
        try:
            self.generate_report(basepath + '__report.png')
        except BaseException as e:
            print(WARN,  "Best model report could not be saved; skipping")
            print("Errmsg", e)
    print("Best model saved to " + basepath)


def checkpoint(self, forced=False, overwrite=None):
    """overwrite: bool/None. If None, set from `checkpoints_overwrite_duplicates`
                             (else, override it).
    """
    def _get_savename(do_temp, do_unique):
        if do_temp and not do_unique:  # give latter precedence
            return "_temp_model"

        if self.logs_use_full_model_name:
            return "{}_{}vals__".format(self.model_name, self._times_validated)
        else:
            return "max_{:.3f}_{}vals__".format(self.best_key_metric,
                                                self._times_validated)

    def _save(basepath, overwrite):
        def _maybe_save(save_fn, path, overwrite):
            def _make_unique_path(path):
                _dir = Path(path).parent
                stem = Path(path).stem
                ext = Path(path).suffix

                existing_names = [x.name for x in Path(_dir).iterdir()
                                  if x.is_file()]
                new_name = stem + '_v2' + ext
                i = 2
                while new_name in existing_names:
                    i += 1
                    new_name = stem + f'_v{i}' + ext
                return os.path.join(_dir, new_name)

            path_exists = Path(path).is_file()
            if not path_exists or (path_exists and overwrite):
                save_fn(path)
            elif path_exists and not overwrite:
                path = _make_unique_path(path)
                save_fn(path)

        if overwrite not in (None, True, False):
            raise ValueError("`overwrite` must be one of: None, True, False")
        if overwrite is None:
            overwrite = bool(self.checkpoints_overwrite_duplicates)

        save_fns = [(basepath + 'state.h5',   self.save),
                    (basepath + 'hist.png',   self._save_history),
                    (basepath + 'report.png', self.generate_report)]

        _sf = self._make_model_save_fns(basepath)
        save_fns.extend(_sf)

        for path, save_fn in save_fns:
            _maybe_save(save_fn, path, overwrite)

    def _clear_checkpoints_IF():
        def _filter_varying(string):
            """Omit changing chars to infer uniques per checkpoint"""
            # omit digits, which change across `max`, `vals`, etc
            filtered = ''.join(s for s in string if not s.isdigit())

            # omit versions, e.g. _v2, _v3, introduced w/ overwrite=True
            stem, ext = Path(filtered).stem, Path(filtered).suffix
            if stem[-2:] == '_v':  # digit already filtered
                stem = stem[:-2]
            return stem + ext

        paths = [f for f in Path(self.logdir).iterdir() if f.is_file()]
        files_per_checkpoint = len(set(_filter_varying(p.name) for p in paths))
        paths = sorted(paths, key=os.path.getmtime)
        paths = list(map(str, paths))

        while len(paths) / files_per_checkpoint > max(1, self.max_checkpoints):
            # remove oldest first (by creation time)
            [os.remove(paths.pop(0)) for _ in range(files_per_checkpoint)]

    do_temp = self._should_do(self.temp_checkpoint_freq)
    do_unique = self._should_do(self.unique_checkpoint_freq)
    if not (do_temp or do_unique) and not forced:
        return

    savename = _get_savename(do_temp, do_unique)
    basepath = os.path.join(self.logdir, savename)
    _save(basepath, overwrite)

    try:
        _clear_checkpoints_IF()
    except BaseException as e:
        print(WARN,  "Checkpoint files could not be cleared; skipping")
        print("Errmsg:", e)


def _make_model_save_fns(self, basepath):
    save_fns = []
    if 'model' not in self.saveskip_list:
        name = 'model'
        if not self.model_save_kw.get('include_optimizer', True):
            name += '_noopt'
        name += '.' + self.model_save_kw.get('save_format', 'h5')
        save_fns.append((
            basepath + name,
            lambda path: self.model.save(path, **self.model_save_kw)))
    if 'model:weights' not in self.saveskip_list:
        name = 'weights.' + self.model_save_weights_kw.get('save_format', 'h5')
        save_fns.append((
            basepath + name,
            lambda path: self.model.save_weights(
                path, **self.model_save_weights_kw)))
    return save_fns


def save(self, savepath=None):
    def _get_optimizer_state(opt):
        def _get_attrs_to_save(opt):
            cfg = self.optimizer_save_configs
            all_attrs = exclude_unpickleable(vars(opt))
            all_attrs['weights'] = []

            if cfg is None:
                return all_attrs

            if 'exclude' in cfg:
                return [a for a in all_attrs if a not in cfg['exclude']]

            if 'include' in cfg:
                attrs = []
                for attr in cfg['include']:
                    if attr in all_attrs:
                        attrs.append(attr)
                    elif attr in vars(opt):
                        print(WARN, ("'{}' optimizer attribute cannot be "
                                     "pickled; skipping"))
                    else:
                        print(WARN, ("'{}' attribute not found in optimizer; "
                                     "skipping").format(attr))
                return attrs

        state = {}
        to_save = _get_attrs_to_save(opt)
        for name in to_save:
            if name == 'weights':
                weights = opt.get_weights()
                if weights != []:
                    state['weights'] = weights
                continue

            value = getattr(opt, name, None)
            if isinstance(value, tf.Variable):
                state[name] = tensor_util.eval_tensor(value, backend=K)
            else:
                state[name] = value
        return state

    def _cache_datagen_attributes():
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
            to_exclude = dg.saveskip_list.copy()

            for key, val in vars(dg).items():
                if isinstance(val, types.LambdaType):
                    to_exclude.append(key)
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

    skiplist = self.saveskip_list + ['model']  # do not pickle model
    savedict = {k: v for k, v in vars(self).items() if k not in skiplist}
    if 'optimizer_state' not in self.saveskip_list:
        savedict['optimizer_state'] = _get_optimizer_state(self.model.optimizer)

    try:
        with open(savepath, "wb") as savefile:
            pickle.dump(savedict, savefile)
            print("TrainGenerator state saved")
    except BaseException as e:
        print(WARN, "TrainGenerator state could not be saved; skipping...")
        print("Errmsg:", e)
    _restore_cached_attributes(self, cached_attrs)


def load(self, filepath=None, passed_args=None):
    def _get_loadskip_list(passed_args):
        if self.loadskip_list == 'auto' or not self.loadskip_list:
            return list(passed_args) if passed_args else []
        elif '{auto}' in self.loadskip_list:
            lsl = self.loadskip_list.copy()
            if passed_args:
                lsl += list(passed_args)
            lsl.pop(lsl.index('{auto}'))
            return lsl
        elif self.loadskip_list == 'none':
            return []
        else:
            return self.loadskip_list

    def _get_filepath(filepath):
        filepath = filepath or self.loadpath
        if filepath is not None:
            return filepath

        if self.logdir is None:
            raise ValueError("`filepath`, `loadpath`, and `logdir` are None")
        elif not Path(self.logdir).is_dir():
            raise ValueError("`filepath` is None, and `logdir` is not a folder"
                             "(%s)" % self.logdir)

        paths = []
        for path in Path(self.logdir).iterdir():
            if path.name.endswith('__state.h5'):
                paths.append(path)

        if not paths:
            raise ValueError("`filepath` is None, and no __state.h5 files "
                             f"found in `logdir` ({self.logdir})")
        paths.sort(key=os.path.getmtime)
        return paths[-1]  # latest

    def _load_datagen_attrs(loadfile_parsed):
        def _load_preprocessor_attrs(dg, dg_loaded, dg_name):
            pp = dg.preprocessor
            pp_loaded = dg_loaded.preprocessor

            for attr, value in vars(pp_loaded).items():
                if attr not in pp.loadskip_list:
                    setattr(pp, attr, value)

        def _validate_set_nums(dg, dg_loaded):
            if hasattr(dg, 'set_nums_original'):
                if any(set_num not in dg.set_nums_original
                       for set_num in dg_loaded.set_nums_to_process):
                    print(WARN, "found set_num in loaded `set_nums_to_process` "
                          "that isn't in the passed `set_nums_original`; setting "
                          "former to `set_nums_original`.")
                    dg.set_nums_to_process = dg.set_nums_original.copy()

        dg_names = [n for n in ('datagen', 'val_datagen') if n in loadfile_parsed]

        for dg_name in dg_names:
            dg = getattr(self, dg_name)
            dg_loaded = loadfile_parsed.pop(dg_name)
            lsl_loaded = None

            for attr, value in vars(dg_loaded).items():
                if attr not in dg.loadskip_list:
                    if attr == 'set_nums_to_process':
                        _validate_set_nums(dg, dg_loaded)
                    elif attr == 'loadskip_list':
                        # delay setting since it changes iteration logic
                        lsl_loaded = value
                    else:
                        setattr(dg, attr, value)
            _load_preprocessor_attrs(dg, dg_loaded, dg_name)
            if lsl_loaded is not None:
                dg.loadskip_list = lsl_loaded

    filepath = _get_filepath(filepath)
    with open(filepath, 'rb') as loadfile:
        loadfile_parsed = pickle.load(loadfile)
        # drop items in loadskip_list (e.g. to avoid overriding passed kwargs)
        loadskip_list = _get_loadskip_list(passed_args)
        loadskip_list.append('model')  # cannot unpickle model
        for name in loadskip_list:
            loadfile_parsed.pop(name, None)

        if 'datagen' in loadfile_parsed or 'val_datagen' in loadfile_parsed:
            _load_datagen_attrs(loadfile_parsed)

        # assign loaded/cached attributes
        self.__dict__.update(loadfile_parsed)

        if 'optimizer_state' not in loadskip_list:
            if getattr(self, 'optimizer_state', None):
                _load_optimizer_state(self)
            else:
                print(WARN, "'optimizer_state' not found in loadfile; skipping."
                      " (optimizer will still instantiate before .train())")
                _init_optimizer(
                    self.model, self.class_weights,
                    input_as_labels=self.input_as_labels,
                    alias_to_metric_name_fn=self._alias_to_metric_name)
        print("TrainGenerator state loaded from", filepath)

    print("--Preloading excluded data based on datagen states ...")
    self._prepare_initial_data(from_load=True)
    print("... finished--")

    if not getattr(self, 'callback_objs', None):  # if not already initialized
        self._init_callbacks()
    self._apply_callbacks(stage='load')


def _load_optimizer_state(self):
    def _get_attrs_to_load(opt):
        cfg = self.optimizer_load_configs
        all_attrs = [a for a in list(vars(opt)) if a != 'updates']

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

    _init_optimizer(self.model, self.class_weights,
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
            self._history_fig.savefig(_path)
        except Exception as e:
            print(WARN, "Model history could not be saved; skipping",
                  "\nErrmsg:", e)

    if self.final_fig_dir:  # keep at most one per model_num
        pass_on_error(_save_epoch_fig, errmsg=("Epoch fig could not be "
                                                 "saved; skipping"))
