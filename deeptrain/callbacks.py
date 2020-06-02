import os
import pickle

from pathlib import Path
from see_rnn import get_weights, get_outputs, get_gradients

from .visuals import show_predictions_per_iteration
from .visuals import show_predictions_distribution
from .visuals import comparative_histogram, layer_hists
from .util._backend import NOTE
from .util.misc import argspec


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


class TraingenCallback():
    def __init__(self):
        pass

    def init_with_traingen(self, traingen=None):
        raise NotImplementedError

    def on_train_iter_end(self, stage=None):
        raise NotImplementedError

    def on_train_batch_end(self, stage=None):
        raise NotImplementedError

    def on_train_epoch_end(self, stage=None):
        raise NotImplementedError

    def on_val_iter_end(self, stage=None):
        raise NotImplementedError

    def on_val_batch_end(self, stage=None):
        raise NotImplementedError

    def on_val_epoch_end(self, stage=None):
        raise NotImplementedError

    def on_val_end(self, stage=None):
        raise NotImplementedError

    def on_save(self, stage=None):
        raise NotImplementedError

    def on_load(self, stage=None):
        raise NotImplementedError


class TraingenLogger(TraingenCallback):
    def __init__(self, savedir, configs,
                 loadpath=None,
                 get_data_fn=None,
                 get_labels_fn=None,
                 gather_fns=None,
                 logname='datalog_',
                 init_log_id=None):
        self.savedir=savedir
        self.configs=configs
        self.loadpath=loadpath
        self.logname=logname

        self.get_data_fn=get_data_fn
        self.get_labels_fn=get_labels_fn
        self.gather_fns=gather_fns
        self.init_log_id=init_log_id
        self.configs=configs

    def init_with_traingen(self, traingen):
        self.tg = traingen
        self.model = traingen.model
        self.weights = {}
        self.outputs = {}
        self.gradients = {}
        self._loggables = ('weights', 'outputs', 'gradients')

        self._process_args(dict(
            configs=self.configs,
            get_data_fn=self.get_data_fn,
            get_labels_fn=self.get_labels_fn,
            gather_fns=self.gather_fns,
            init_log_id=self.init_log_id,
            ))

    def log(self, _id=None):
        def _gather(key, _id):
            def _get_args(key, name_or_idx):
                kw = self.configs.get(f'{key}-kw', {}).copy()
                kw['model'] = self.model
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
