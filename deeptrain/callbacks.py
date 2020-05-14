import os
import pickle

from pathlib import Path
from see_rnn import get_weights, get_outputs, get_gradients

from .visuals import show_predictions_per_iteration
from .visuals import show_predictions_distribution
from .visuals import comparative_histogram, layer_hists
from .util._backend import NOTE
from .util import argspec


def make_callbacks(cb_makers):
    """cb_makers must return: `callbacks` and `callbacks_init`, or `callbacks`.
        callbacks: dict.
          keys: callback names. Names can be of any type, but if values'
                methods use an object, must match name in `callbacks_init`.
          values: dicts of stage-method(s) pairs. Methods can be functions or
                class methods; if latter uses a class instance not defined in
                TrainGenerator, must instantiate it.
        callbacks_init: dict. Instantiate class instances used in `callbacks`,
                which will be packed in `TrainGenerator.callback_objs`.
          keys: callback object names. See `callbacks`.
          values: objects / class instances to be instantiated. TrainGenerator's
                `self` will be passed to the constructor (__init__(self)).
          Can be an empty dict, {}, None, or not returned at all.
    See examples in util.callbacks.
    """
    def _unpack_returned(returned):
        tp = lambda x: type(x).__name__
        if not isinstance(returned, (tuple, list)):
            assert isinstance(returned, dict), (
                "`cb_makers` must return 2 or 1 dicts, or 1 dict and 1 None"
                " - got: %s" % tp(returned))
            return returned, None
        elif len(returned) == 2:
            assert all(isinstance(x, (dict, type(None))) for x in returned), (
                "`cb_makers` must return 2 or 1 dicts, or 1 dict and 1 None"
                " - got: %s, %s" % (tp(returned[0]), tp(returned[1])))
            return returned
        else:
            raise ValueError("`cb_makers` must return 2 or 1 dicts - got "
                             "%s items" % len(returned))

    callbacks, callbacks_init = {}, {}
    for make_cb in cb_makers:
        returned = make_cb()
        cb, cbi = _unpack_returned(returned)
        callbacks.update(cb)
        if isinstance(cbi, dict):
            callbacks_init.update(cbi)
    return callbacks, callbacks_init


def predictions_per_iteration_cb(self):
    show_predictions_per_iteration(self._labels_cache, self._preds_cache)


def predictions_distribution_cb(self):
    show_predictions_distribution(self._labels_cache, self._preds_cache,
                                  self.predict_threshold)

def comparative_histogram_cb(self):
    """Suited for binary classification sigmoid outputs"""
    comparative_histogram(self.model,
                          layer_name=self.model.layers[-1].name,
                          data=self.val_datagen.get(skip_validation=True)[0],
                          vline=self.predict_threshold,
                          xlims=(0, 1))


def make_layer_hists_cb(_id='*', mode='weights', x=None, y=None,
                        omit_names='bias', share_xy=(0, 0),
                        configs=None, **kw):
    def layer_hists_cb(self):
        _x = x or self.val_datagen.batch
        _y = y or (self.val_datagen.labels if not self.input_as_labels else x)
        layer_hists(self.model, _id, mode, _x, _y, omit_names, share_xy,
                    configs, **kw)
    return layer_hists_cb


class TraingenLogger():
    def __init__(self, traingen, savedir, configs,
                 loadpath=None,
                 get_data_fn=None,
                 get_labels_fn=None,
                 gather_fns=None,
                 logname='datalog_',
                 init_log_id=None):
        self.tg=traingen
        self.savedir=savedir
        self.configs=configs
        self.loadpath=loadpath
        self.logname=logname

        self.m = traingen.model
        self.weights = {}
        self.outputs = {}
        self.gradients = {}
        self._loggables = ('weights', 'outputs', 'gradients')

        self._process_args(dict(
            configs=configs,
            get_data_fn=get_data_fn,
            get_labels_fn=get_labels_fn,
            gather_fns=gather_fns,
            init_log_id=init_log_id,
            ))

    def log(self, _id=None):
        def _gather(key, _id):
            def _get_args(key, name_or_idx):
                kw = self.configs.get(f'{key}-kw', {}).copy()
                kw['model'] = self.m
                kw['_id'] = name_or_idx
                kw['as_dict'] = True
                if 'input_data' in argspec(self.gather_fns[key]):
                    kw['input_data'] = self.get_data_fn()
                if 'labels' in argspec(self.gather_fns[key]):
                    kw['labels'] = self.get_labels_fn()
                return kw

            getattr(self, f'{key}')[_id] = []
            for name_or_idx in self.configs[key]:
                args = _get_args(key, name_or_idx)
                getattr(self, f'{key}')[_id].append(
                    self.gather_fns[key](**args))

        if _id is None:
            self._id += 1
            _id = self._id

        for key in self._loggables:
            if key in self.configs:
                _gather(key, _id)

    def save(self, _id=None, clear=False, verbose=1):
        data = {k: getattr(self, f'{k}') for k in self._loggables}
        _id = _id or self._id
        savename = "{}{}.h5".format(self.logname, _id)
        savepath = os.path.join(self.savedir, savename)

        with open(savepath, "wb") as file:
            pickle.dump(data, file)
        if verbose:
            print("TraingenLogger data saved to", savepath)
        if clear:
            self.clear(verbose=verbose)

    def load(self, verbose=1):
        loadpath = self.loadpath
        if verbose and loadpath is None:
            print(NOTE, "`loadpath` is None; fetching last from `savedir`")
            paths = [x for x in Path(self.savedir).iterdir()
                     if (self.logname in x.stem and x.suffix == '.h5')]
            loadpath = sorted(paths, key=os.path.getmtime)[-1]

        with open(loadpath, "rb") as file:
            loadfile = pickle.load(file)
            self.__dict__.update(loadfile)
        if verbose:
            print("TraingenLogger data loaded from", loadpath)

    def clear(self, verbose=1):
        [setattr(self, name, {}) for name in self._loggables]
        if verbose:
            print("TraingenLogger data cleared")

    def _process_args(self, args):
        def _process_configs(configs):
            def _validate_types(configs, key, value):
                if not isinstance(value, list):
                    if isinstance(value, tuple):
                        configs[key] = list(value)
                    else:
                        configs[key] = [value]
                value = configs[key]
                for x in value:
                    assert isinstance(x, (str, int)), (
                        ("loggables specifiers must be of type str or "
                         "int (values of keys: {}) - got: {}").format(
                             ', '.join(self._loggables), value))

            supported = list(self._loggables)
            supported += [f'{n}-kw' for n in supported]
            for key in configs:
                if key not in supported:
                    raise ValueError(f"key {key} in `configs` is not supported; "
                                     "supported are:", ', '.join(supported))
                value = configs[key]
                if '-kw' not in key:
                    _validate_types(configs, key, value)
                else:
                    assert isinstance(value, dict), (
                        "-kw keys must be of type dict (got %s)" % value)

        def _process_get_data_fn(get_data_fn):
            if get_data_fn is not None:
                self.get_data_fn = get_data_fn
                return
            data = self.tg.val_datagen.get()[0]
            self.get_data_fn = lambda: data

        def _process_get_labels_fn(get_labels_fn):
            if get_labels_fn is not None:
                self.get_labels_fn = get_labels_fn
                return
            if self.tg.input_as_labels:
                labels = self.tg.val_datagen.get()[0]
            else:
                labels = self.tg.val_datagen.get()[1]
            self.get_labels_fn = lambda: labels

        def _process_init_log_id(init_log_id):
            assert isinstance(init_log_id, (int, type(None)))
            self._id = init_log_id or -1

        def _process_gather_fns(gather_fns):
            gather_fns_default = {
                'weights': get_weights,
                'outputs': get_outputs,
                'gradients': get_gradients,
                }

            # if None, use defaults - else, add what's missing via defaults
            if gather_fns is None:
                gather_fns = gather_fns_default
            else:
                for name in gather_fns_default:
                    if name not in gather_fns:
                        gather_fns[name] = gather_fns_default[name]
            self.gather_fns = gather_fns

        _process_configs(args['configs'])
        _process_get_data_fn(args['get_data_fn'])
        _process_get_labels_fn(args['get_labels_fn'])
        _process_init_log_id(args['init_log_id'])
        _process_gather_fns(args['gather_fns'])
