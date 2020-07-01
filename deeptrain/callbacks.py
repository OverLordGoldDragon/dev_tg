import os
import pickle
import random
import numpy as np
import tensorflow as tf

from pathlib import Path
from see_rnn import get_weights, get_outputs, get_gradients

from .visuals import binary_preds_per_iteration
from .visuals import binary_preds_distribution
from .visuals import infer_train_hist, layer_hists
from .util._backend import K, NOTE
from .util.misc import argspec


def binary_preds_per_iteration_cb(self):
    """Suited for binary classification sigmoid outputs.
    See :func:`~deeptrain.visuals.binary_preds_per_iteration`."""
    binary_preds_per_iteration(self._labels_cache, self._preds_cache)


def binary_preds_distribution_cb(self):
    """Suited for binary classification sigmoid outputs.
    See :func:`~deeptrain.visuals.binary_preds_distribution`."""
    binary_preds_distribution(self._labels_cache, self._preds_cache,
                              self.predict_threshold)


def infer_train_hist_cb(self):
    """Suited for binary classification sigmoid outputs.
    See :func:`~deeptrain.visuals.infer_train_hist`."""
    infer_train_hist(
        self.model,
        input_data=self.val_datagen.get(skip_validation=True)[0],
        vline=self.predict_threshold,
        xlims=(0, 1))


def make_layer_hists_cb(_id='*', mode='weights', x=None, y=None,
                        omit_names='bias', share_xy=(0, 0), configs=None, **kw):
    """Layer histograms grid callback.
    See :func:`~deeptrain.visuals.layer_hists`."""
    def layer_hists_cb(self):
        _x = x or self.val_datagen.batch
        _y = y or (self.val_datagen.labels if not self.input_as_labels else x)
        layer_hists(self.model, _id, mode, _x, _y, omit_names, share_xy,
                    configs, **kw)
    return layer_hists_cb


class TraingenCallback():
    """Required base class for `callbacks` objects used by `TrainGenerator`.
    Enables using `TrainGenerator` attributes by assigning it to `self` via
    `init_with_traingen`. Methods are called by `TrainGenerator` at several
    stages: train, validation, save, load, `__init__`.

    `stage` is an optional argument to every method (except `init_with_traingen`)
    to allow finer-grained control, particularly for `('val_end', 'train:epoch')`.

    Methods not implemented by the inheriting class will be skipped by catching
    `NotImplementedError`.

    `__init__` sets a private attribute `_counters`, which can be used along
    a "freq" to call methods every Nth callback, instead of every callback.
    Example: :class:`RandomSeedSetter`.
    """
    def __init__(self):
        # can increment to call methods on N-th callback instead of every,
        # e.g. every 5 iterations or every 2nd epoch. Optional attribute.
        self._counters = {name: 0 for name in
                          ('init', 'train:iter', 'train:batch', 'train:epoch',
                           'val:iter', 'val:batch', 'val:epoch', 'val_end',
                           'save', 'load')}

    def init_with_traingen(self, traingen=None):
        """Called by `TrainGenerator.__init__()`, passing in `self`
        (`TrainGenerator` instance).
        """
        raise NotImplementedError

    def on_train_iter_end(self, stage=None):
        """Called by `_on_iter_end`,
        within :meth:`TrainGenerator._train_postiter_processing`
        with `stage='train:iter'`."""
        raise NotImplementedError

    def on_train_batch_end(self, stage=None):
        """Called by `_on_batch_end`,
        within :meth:`TrainGenerator._train_postiter_processing`,
        with `stage='train:batch'`.
        """
        raise NotImplementedError

    def on_train_epoch_end(self, stage=None):
        """Called by `_on_epoch_end`,
        within :meth:`TrainGenerator._train_postiter_processing`,
        with `stage='train:epoch'`."""
        raise NotImplementedError

    def on_val_iter_end(self, stage=None):
        """Called by `_on_iter_end`,
        within :meth:`TrainGenerator._val_postiter_processing`,
        with `stage='val:iter'`.
        """
        raise NotImplementedError

    def on_val_batch_end(self, stage=None):
        """Called by `_on_batch_end`,
        within :meth:`TrainGenerator._val_postiter_processing`,
        with `stage='val:batch'`."""
        raise NotImplementedError

    def on_val_epoch_end(self, stage=None):
        """Called by `_on_epoch_end`,
        within :meth:`TrainGenerator._val_postiter_processing`,
        with `stage='val:epoch'`."""
        raise NotImplementedError

    def on_val_end(self, stage=None):
        """Called by :meth:`TrainGenerator._on_val_end`, with:

            - `stage=('val_end', 'train:epoch')` if
              `TrainGenerator.datagen.all_data_exhausted`
            - `stage='val_end'` otherwise
        """
        raise NotImplementedError

    def on_save(self, stage=None):
        """Called by :meth:`TrainGenerator.save()` with `stage='save'`.
        """
        raise NotImplementedError

    def on_load(self, stage=None):
        """Called by :meth:`TrainGenerator.load()` with `stage='load'`.
        """
        raise NotImplementedError


class RandomSeedSetter(TraingenCallback):
    """Periodically sets `random`, `numpy`, TensorFlow graph-level, and
    TensorFlow global seeds.

    Arguments:
        seeds: dict / None
            Dict of initial seeds; if None, will default all to 0. E.g.:
            `{'random': 0, 'numpy': 0, 'tf-graph': 1, 'tf-global': 2}`.
        freq: dict
            When to set / reset the seeds. E.g. `{'train:epoch': 2}` (default)
            will set every 2 train epochs. By default, seeds are incremented
            by 1 to avoid repeating random sequences with each setting.

    Methods can be overridden to increment by a different amount or not at all,
    and to clear keras session & reset TF default graph, which doesn't happen
    by default.
    """
    def __init__(self, seeds=None, freq={'train:epoch': 2}):
        super().__init__()
        names = ('random', 'numpy', 'tf-graph', 'tf-global')

        if seeds is None:
            self.seeds = {name: 0 for name in names}
        elif isinstance(seeds, int):
            self.seeds = {name: seeds for name in names}
        elif not isinstance(seeds, dict):
            raise ValueError("`seeds` must be int or dict, got %s" % seeds)
        else:
            self.seeds = seeds
        self.freq = freq

    def set_seeds(self, increment=0, seeds=None, reset_graph=False, verbose=1):
        """Sets seeds. `increment` will add to each of `self.seeds`. If `seeds`
        is not None, will override `increment`.
        """
        if seeds is None:
            seeds = {k: v + increment for k, v in self.seeds.items()}
        self._set_seeds(seeds, reset_graph, verbose)

    @classmethod
    def _set_seeds(self, seeds=None, reset_graph=False, verbose=1):
        """See :meth:`set_seeds`; `_set_seeds` can be used without instantiating
        class, as is used in :func:`deeptrain.set_seeds`.
        """
        if reset_graph:
            self.reset_graph(verbose)
        seeds = seeds or {name: 0 for name in ('random', 'numpy',
                                               'tf-graph', 'tf-global')}
        random.seed(seeds['random'])
        np.random.seed(seeds['numpy'])
        tf.compat.v1.set_random_seed(seeds['tf-graph'])
        if tf.__version__[0] == '2':
            tf.random.set_seed(seeds['tf-global'])
        else:
            tf.set_random_seed(seeds['tf-global'])

        if verbose:
            print(("RANDOM SEEDS RESET (random: {}, numpy: {}, tf-graph: {}, "
                   "tf-global: {})").format(seeds['random'], seeds['numpy'],
                                            seeds['tf-graph'], seeds['tf-global']
                                            ))

    @classmethod
    def reset_graph(self, verbose=1):
        """Clears keras session, and resets TensorFlow default graph.

        NOTE: after calling, best practice is to re-instantiate model, else
        some internal operations may fail due to elements from different graphs
        interacting (of pre-reset model and post-reset tensors).
        """
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    #### Callback calls ######################################################
    def call_on_freq(fn):
        def wrap(self, stage):
            if stage in self.freq:
                self._counters[stage] += 1
                if self._counters[stage] % self.freq[stage] == 0:
                    fn(self, stage)
        return wrap

    @call_on_freq
    def on_train_iter_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_train_batch_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_train_epoch_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_val_iter_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_val_batch_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_val_epoch_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_val_end(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_save(self, stage=None):
        self.set_seeds(increment=1)

    @call_on_freq
    def on_load(self, stage=None):
        self.set_seeds(increment=1)


class TraingenLogger(TraingenCallback):
    """TensorBoard-like logger, gathering layer weights, outputs, and gradients
    over specified periods.

    Arguments:
        savedir: str
            Path to directory where to save data and class instance state.
        configs: dict
            Data mode-layer/weight name pairs for logging. Ex:

            >>> configs = {
            ...     'weights': ['dense_2', 'conv2d_1/kernel:0'],
            ...     'outputs': 'lstm_1',
            ...     'gradients': ('conv2d_1', 'dense_2:/bias:0'),
            ...     'gradients-kw': dict(mode='weights', learning_phase=0),
            ... }

            With `mode='weights'` for `('weights', 'gradients')`,
            complete weight names may be specified - else, all layer weights
            will be included. If layer name is substring, will return earliest
            match. See `see_rnn.inspect_gen` methods.

        loadpath: str / None
            Path to savefile of class instance state, to resume logging.
            If None, will attempt to fetch from `savedir` based on `logname`.
        get_data_fn: function
            Will call to fetch input data to feed for `'gradients'` and
            `'outputs'` data.
            Defaults to `lambda: TrainGenerator.val_datagen.get()[0]`.
        get_labels_fn: function
            Will call to fetch labels to feed for `'gradients'` data.
            Defaults to `lambda: TrainGenerator.val_datagen.get()[1]`
            if `TrainGenerator.input_as_labels == False` - else, to `~get()[0]`.
        logname: str
            Base name of savefiles, prefixing save `_id`.
        init_log_id: int / None
            Initial log `_id`, will increment by 1 with each call to `log()`.
            (unless `_id` was passed to `log()`). Defaults to `-1`.
    """
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
        """Instantiates configs, getter functions, & others; requires a
        `TrainGenerator` instance.
        """
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
        """Gathers data according to `configs` and `gather_fns`.
        """
        def _gather(key, _id):
            def _get_inputs(key, name_or_idx):
                kw = self.configs.get(f'{key}-kw', {}).copy()
                kw['model'] = self.model
                kw['_id'] = name_or_idx
                kw['as_dict'] = True
                if 'input_data' in argspec(self.gather_fns[key]):
                    kw['input_data'] = self.get_data_fn()
                if 'labels' in argspec(self.gather_fns[key]):
                    kw['labels'] = self.get_labels_fn()
                return kw

            getattr(self, key)[_id] = []  # make empty container to append to
            for name_or_idx in self.configs[key]:
                ins = _get_inputs(key, name_or_idx)
                getattr(self, key)[_id].append(self.gather_fns[key](**ins))

        if _id is None:
            self._id += 1
            _id = self._id

        for key in self._loggables:
            if key in self.configs:
                _gather(key, _id)

    def save(self, _id=None, clear=False, verbose=1):
        """Saves data per `_loggables`, but not other class instance attributes.

        Arguments:
            _id: int / str[int] / None
                Appended to `logname` to make savepath in `savedir`.
            clear: bool
                Whether to empty data attributes after saving.
            verbose: bool / int[bool]
                Whether to print a save message with savepath.
        """
        data = {k: getattr(self, k) for k in self._loggables}
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
        """Loads from an .h5 file according to `loadpath`, which can include
        data and other class instance attributes. If `loadpath` is None,
        will attempt to fetch from `savedir` based on `logname`.

        `verbose` == `1`/`True` to print load message with loadpath.
        """
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
        """Set attributes in `_loggables` to `{}`"""
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
