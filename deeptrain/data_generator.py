# -*- coding: utf-8 -*-
"""TODO:
    - require preprocessors to inherit an abstract base class?
    - make preprocessor attributes mutable to avoi synch_to and synch_from
"""
import os
import h5py
import random
import numpy as np

from pathlib import Path
from copy import deepcopy
from types import LambdaType

from .util import Preprocessor, GenericPreprocessor, TimeseriesPreprocessor
from .util import data_loaders, labels_preloaders
from .util.configs import _DATAGEN_CFG
from .util.algorithms import ordered_shuffle
from .util._backend import WARN, IMPORTS
from .util._default_configs import _DEFAULT_DATAGEN_CFG


###############################################################################
class DataGenerator():
    """Central interface between a directory and `TrainGenerator`. Handles data
    loading, preprocessing, shuffling, and batching. Requires only
    `data_dir` to run.

    Arguments:
        data_dir: str
            Path to directory to load data from.
        batch_size: int
            Number of samples to feed the model at once. Can differ from
            size of batches of loaded files; see "Dynamic batching".
        labels_path: str / None
            Path to labels file. None requires `TrainGenerator.input_as_labels`
            to be True, to feed `batch` as `labels` in
            :meth:`TrainGenerator.get_data`.
        preprocessor: None / custom object / str in ('timeseries',)
            Transforms `batch` and `labels` right before both are returned by
            :meth:`.get`. See :meth:`_set_preprocessor`.

            - str: fetches one of API-supported preprocessors.
            - None, uses :class:`GenericPreprocessor`.
            - Custom object: must subclass :class:`Preprocessor`.

        preprocessor_configs: None / dict
            Kwargs to pass to `preprocessor` in case it's None, str, or an
            uninstantiated custom object. Ignored if `preprocessor` is
            instantiated.
        data_loader: None / function
            Custom data loading function, with input signature `(self, set_num)`,
            loading data from directory, based on `set_num` (not explicitly
            required; only returning data in expected format is required).
            If None, defaults to one of defined in :mod:`util.data_loaders`,
            as determined by :meth:`._infer_data_info`.
        labels_preloader: None / function
            Custom labels preloading function, with input signature `(self)`,
            loading labels from `labels_path`, setting `self.all_labels`.
            If None, will default to one of  defined in
            :mod:`util.labels_preloaders', based on `labels_path` file extension.
        base_name: None / str
            Name common to all to non-labels files, to be used internally by
            `data_loader` along `set_num`. If None, is determined within
            :meth:`.infer_data_info` as longest substring common to all filenames.

    **Dynamic batching**:  # TODO "dynamic" implies changing, use another word

    Loaded file's batch size may differ from `batch_size`, so long as former
    is an integer or integer fraction multiple of latter. Ex:

        - `len(loaded) == 32`, `batch_size == 64` -> will load another file
          and concatenate into `len(batch) == 64`.
        - `len(loaded) == 64`, `batch_size == 32` -> will set first half of
          `loaded` as `batch` and cache `loaded`, then repeat for second half.
        - 'Technically', files need not be integer (/ fraction) multiples, as
          the following load order works with `batch_size == 32`:
          `len(loaded) == 31`, `len(loaded) == 1` - but this is *not* recommended,
          as it's bound to fail if using shuffling, or if total number of
          samples isn't divisible by `batch_size`. Other problems may also arise.

    `__init__`:

    Instantiation. ("+" == if certain conditions are met)

        - +Infers missing configs based on args
        - Validates args & kwargs, and tries to correct, printing a"NOTE" or
          "WARNING" message where appropriate
        - +Preloads all labels into `all_labels`
        - Instantiates misc internal parameters to predefiend values (may be
          overridden by `TrainGenerator` loading).
    """
    _BUILTINS = {'preprocessors': (GenericPreprocessor, TimeseriesPreprocessor),
                 'data_loaders': {'numpy', 'numpy-memmap', 'numpy-lz4f',
                                  'hdf5', 'hdf5-dataset'},
                 'data_extensions': {'.npy', '.h5'},
                 'labels_extensions': {'.csv', '.h5'}}

    def __init__(self, data_dir,
                 batch_size=32,
                 labels_path=None,
                 preprocessor=None,
                 preprocessor_configs=None,
                 data_loader=None,
                 labels_preloader=None,
                 base_name=None,
                 shuffle=False,
                 superbatch_dir=None,
                 set_nums=None,
                 superbatch_set_nums=None,
                 data_loader_dtype='float32',
                 data_ext=None,
                 **kwargs):
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.labels_path=labels_path
        self.preprocessor=preprocessor
        self.preprocessor_configs=preprocessor_configs or {}
        self.data_loader=data_loader
        self.labels_preloader=labels_preloader
        self.base_name=base_name
        self.shuffle=shuffle
        self.data_loader_dtype=data_loader_dtype
        self.data_ext=data_ext

        if superbatch_set_nums == 'all':
            self.superbatch_dir = data_dir
        else:
            self.superbatch_dir = superbatch_dir

        info = self._infer_data_info(data_dir, data_ext, data_loader, base_name)
        self.data_loader = info['data_loader']
        self.base_name   = info['base_name']
        self._filenames  = info['filenames']
        self._filepaths  = info['filepaths']
        self.data_ext    = info['data_ext']

        self._init_and_validate_kwargs(kwargs)

        self._set_data_loader(self.data_loader)
        self._set_set_nums(set_nums, superbatch_set_nums)
        self._set_preprocessor(preprocessor, self.preprocessor_configs)

        if labels_preloader is not None or labels_path is not None:
            self.preload_labels()
        else:
            self.all_labels = {}
            self.labels = []

        self._init_class_vars()
        print("DataGenerator initiated")

    ###### MAIN METHODS #######################################################
    def get(self, skip_validation=False):
        """Returns `(batch, labels)` fed to `preprocessor.process()`.

        skip_validation: bool
            - False (default): calls :meth:`_validate_batch`, which will
              :meth:`advance_batch` if `batch_exhausted`, and :meth:`reset_state`
              if `all_data_exhausted`.
            - True: fetch preprocessed `(batch, labels)` without advancing
              any internal states.
        """
        if not skip_validation:
            self._validate_batch()
        return self.preprocessor.process(self.batch, self.labels)

    def advance_batch(self, forced=False, is_recursive=False):
        """Sets next `batch` and `labels`; handles dynamic batching.

            - If `batch_loaded` and not `forced` (and not `is_recursive`),
              prints a warning that batch is loaded, and returns (does nothing)
            - `len(batch) != batch_size`:
                - `< batch_size`: calls :meth:`advance_batch` with
                  `is_recursive = True`. With each such call, `batch` and `labels`
                  are extended (stacked) until matching `batch_size`.
                - `> batch_size`, not integer multiple: raises exception.
                - `> batch_size`, is integer multiple: makes `_group_batch` and
                  `_group_labels`, which are used to set `batch` and `labels`.

            - +If `set_nums_to_process` is empty, will raise Exception; it must
              have been reset beforehand via e.g. :meth:`reset_state`. If it's
              not empty, sets `set_num` by popping from `set_nums_to_process`.
              (+: only if `_group_batch` is None)
            - Sets or extends `batch` via :meth:`_get_next_batch` (by loading,
              or from `_group_batch` or `superbatch`).
            - +Sets or extends `labels` via `all_labels[set_num]` or
              :meth:`_labels_from_group_labels`.
              (+: only if `labels_path` is a path (and not None))
            - Sets `set_name`, used by :class:`TrainGenerator` to print iteration
              messages.
            - Sets `batch_loaded = True`, `batch_exhausted = False`,
              `all_data_exhausted = False`, and `slice_idx` to None if it's
              already None (else to `0`).
        """
        def _handle_batch_size_mismatch(forced):
            if len(self.batch) < self.batch_size:
                self.set_name = self._set_names.pop(0)
                self.advance_batch(forced, is_recursive=True)
                return 'exit'

            n_batches = len(self.batch) / self.batch_size
            if n_batches.is_integer():
                self._make_group_batch_and_labels(n_batches)
            else:
                raise Exception(f"len(batch) = {len(self.batch)} exceeds "
                                "`batch_size` and is its non-integer multiple")

        if not is_recursive:  # recursion is for stacking to match `batch_size`
            if self.batch_loaded and not forced:
                print(WARN, "'batch_loaded'==True; advance_batch() does "
                      "nothing\n(to force next batch, set 'forced'=True')")
                return
            self.batch = []
            self.labels = []

        if self._group_batch is None:
            if len(self.set_nums_to_process) == 0:
                raise Exception("insufficient samples (`set_nums_to_process` "
                                "is empty)")
            self.set_num = self.set_nums_to_process.pop(0)
            self._set_names = [str(self.set_num)]
            if self.labels_path:
                self.labels.extend(self.all_labels[self.set_num])
        elif self.labels_path:
            self.labels.extend(self._labels_from_group_labels())
        self.batch.extend(self._get_next_batch())

        if self.batch_size is not None and len(self.batch) != self.batch_size:
            flag = _handle_batch_size_mismatch(forced)
            if flag == 'exit':
                # exit after completing arbitrary number of recursions, by
                # which time code below would execute as needed
                return

        s = self._set_names.pop(0)
        self.set_name = s if not is_recursive else "%s+%s" % (self.set_name, s)
        self.batch = np.asarray(self.batch)
        if self.labels_path:
            self.labels = np.asarray(self.labels)

        self.batch_loaded = True
        self.batch_exhausted = False
        self.all_data_exhausted = False
        self.slice_idx = None if self.slice_idx is None else 0

    ###### MAIN METHOD HELPERS ################################################
    def load_data(self, set_num):
        """Load and return `batch` data via `data_loader(set_num)`.
        Used by :meth:`_get_next_batch` and :meth:`preload_superbatch`.
        """
        return self.data_loader(self, set_num)

    def _get_next_batch(self, set_num=None, update_state=True, warn=True):
        """Gets batch per `set_num`.

            - `update_state = False`: won't update internal counters.
            - `set_num = None`: will use `self.set_num`.
            - `warn = False`: won't print warning on superbatch not being
              preloaded.
            - If `_group_batch` is not None, will get `batch` from `_group_batch`.
            - If `set_num` is in `superbatch_set_nums`, will get `batch`
              as `superbatch[set_num]` (if `superbatch` exists).
            - By default, gets `batch` via :meth:`load_data`.
        """
        set_num = set_num or self.set_num

        if self._group_batch is not None:
            batch = self._batch_from_group_batch()
            if update_state:
                self._update_group_batch_state()
        elif set_num in self.superbatch_set_nums:
            if self.superbatch:
                batch = self.superbatch[set_num]
            else:
                if warn:
                    print(WARN, f"`set_num` ({set_num}) found in `superbatch_"
                          "set_nums` but `superbatch` is empty; call "
                          "`preload_superbatch()`")
                batch = self.load_data(set_num)
        else:
            batch = self.load_data(set_num)
        return batch

    def _batch_from_group_batch(self):
        """Slices `_group_batch` per `batch_size` and `_group_batch_idx`."""
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_batch[start:end]

    def _labels_from_group_labels(self):
        """Slices `_group_labels` per `batch_size` and `_group_batch_idx`."""
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_labels[start:end]

    def _update_group_batch_state(self):
        """Sets "group" attributes to `None` once sufficient number of batches
        were extracted, else increments `_group_batch_idx`.
        """
        if (self._group_batch_idx + 1
            ) * self.batch_size == len(self._group_batch):
            self._group_batch = None
            self._group_labels = None
            self._group_batch_idx = None
        else:
            self._group_batch_idx += 1

    def on_epoch_end(self):
        """Increments `epoch`, calls `preprocessor.on_epoch_end(epoch)`, then
        :meth:`reset_state`, and returns `epoch`.
        """
        self.epoch += 1
        self.preprocessor.on_epoch_end(self.epoch)
        self.reset_state()
        return self.epoch

    def update_state(self):
        """Calls `preprocessor.update_state()`, and if `batch_exhausted` and
        `set_nums_to_process == []`, sets `all_data_exhausted = True` to signal
        :class:`TrainGenerator` of epoch end.
        """
        self.preprocessor.update_state()
        if self.batch_exhausted and self.set_nums_to_process == []:
            self.all_data_exhausted = True

    def reset_state(self):
        """Calls `preprocessor.reset_state()`, sets `batch_exhausted = True`,
        `batch_loaded = False`, resets `set_nums_to_process` to
        `set_nums_original`, and shuffles `set_nums_to_process` if `shuffle`.
        """
        self.preprocessor.reset_state()
        # ensure below values prevail, in case `preprocessor` sets them to
        # something else; also sets `preprocessor` attributes
        self.batch_exhausted = True
        self.batch_loaded = False
        self.set_nums_to_process = self.set_nums_original.copy()

        if self.shuffle:
            random.shuffle(self.set_nums_to_process)
            print('\nData set_nums shuffled\n')

    ###### MISC METHOS ########################################################
    def _validate_batch(self):
        """If `all_data_exhausted`, calls :meth:`reset_state`.
        If `batch_exhausted`, calls :meth:`advance_batch`.
        """
        if self.all_data_exhausted:
            print(WARN, "all data exhausted; automatically resetting "
                  "datagen state")
            self.reset_state()
        if self.batch_exhausted:
            print(WARN, "batch exhausted; automatically advancing batch")
            self.advance_batch()

    def _make_group_batch_and_labels(self, n_batches):
        """Makes `_group_batch` and `_group_labels` when loaded `len(batch)`
        exceeds `batch_size` as its integer multiple. May shuffle.

            - `_group_batch = np.asarray(batch)`, and
              `_group_labels = np.asarray(labels)`; each's `len() > batch_size`.
            - Shuffles if:
                - `shuffle_group_samples`: shuffles all samples (dim0 slices)
                - `shuffle_group_batches`: groups dim0 slices by `batch_size`,
                  then shuffles the groupings. Ex:

                  >>> batch_size == 32
                  >>> batch.shape == (128, 100)
                  >>> batch = batch.reshape()  # (4, 32, 100) == .shape
                  >>> shuffle(batch)           # 24 (4!) permutations
                  >>> batch = batch.reshape()  # (128, 100)   == .shape

            - Sets `_group_batch_idx = 0`, and calls
              :meth:`_update_group_batch_state`.
        """
        def _maybe_shuffle(gb, lb):
            if self.shuffle_group_samples:
                gb, lb = ordered_shuffle(gb, lb)
            elif self.shuffle_group_batches:
                gb_shape, lb_shape = gb.shape, lb.shape
                gb = gb.reshape(-1, self.batch_size, *gb_shape[1:])
                lb = lb.reshape(-1, self.batch_size, *lb_shape[1:])
                gb, lb = ordered_shuffle(gb, lb)
                gb, lb = gb.reshape(*gb_shape), lb.reshape(*lb_shape)
            return gb, lb

        self._set_names = [f"{self.set_num}-{postfix}" for postfix in
                           "abcdefghijklmnopqrstuvwxyz"[:int(n_batches)]]
        gb = np.asarray(self.batch)
        lb = np.asarray(self.labels)
        assert len(gb) == len(lb), ("len(batch) != len(labels) ({} != {})"
                                    ).format(len(gb), len(lb))

        self.batch, self.labels = [], []  # free memory
        gb, lb = _maybe_shuffle(gb, lb)

        self._group_batch = gb
        self._group_labels = lb
        self._group_batch_idx = 0

        self.batch = self._batch_from_group_batch()
        self.labels = self._labels_from_group_labels()
        self._update_group_batch_state()

    ###### INIT METHODS #######################################################
    def _set_set_nums(self, set_nums, superbatch_set_nums):
        """Sets `set_nums_original`, `set_nums_to_process`, and
        `superbatch_set_nums`.

            - Fetches `set_nums` from `data_dir` or  `_hdf5_path`, depending on
              whether `'hdf5_dataset'` is in `data_loader` function name .
            - Sets `set_nums_to_process` and `set_nums_original`; if `set_nums`
              weren't passed to `__init__`, sets to fetched ones
              (see :func:`~deeptrain.util.data_loaders.hdf5_dataset_loader`).
            - If `set_nums` were passed, validates that they're a subset of
              fetched ones (i.e. can be seen by `data_loader`).
            - Sets `superbatch_set_nums`; if not passed to `__init__`,
              and `== 'all'`, sets to fetched ones. If passed, validates that
              they subset fetched ones.
        """
        def _get_set_nums_to_process():
            def _set_nums_from_dir(_dir):
                def _sort_ascending(ls):
                    return list(map(str, sorted(map(int, ls))))

                nums_from_dir = []
                for filename in os.listdir(_dir):
                    if Path(filename).suffix == self.data_ext and (
                            self.base_name in filename):
                        num = ''.join(x for x in Path(filename).stem.replace(
                            self.base_name, '') if x.isdigit())
                        nums_from_dir.append(num)
                return _sort_ascending(nums_from_dir)

            def _set_nums_from_hdf5_dataset(hdf5_dataset):
                return [num for num in list(hdf5_dataset.keys()) if (
                    num.isdigit() or isinstance(num, (float, int)))]

            if 'hdf5_dataset' in self.data_loader.__name__:
                with h5py.File(self._hdf5_path, 'r') as hdf5_dataset:
                    return _set_nums_from_hdf5_dataset(hdf5_dataset)
            else:
                return _set_nums_from_dir(self.data_dir)

        def _set_and_validate_set_nums(set_nums):
            nums_to_process = _get_set_nums_to_process()

            if not set_nums:
                self.set_nums_original   = nums_to_process.copy()
                self.set_nums_to_process = nums_to_process.copy()
                print(len(nums_to_process), "set nums inferred; if more are "
                      "expected, ensure file names contain a common substring "
                      "w/ a number (e.g. 'train1.npy', 'train2.npy', etc)")
            else:
                if any((num not in nums_to_process) for num in set_nums):
                    raise Exception("a set_num in `set_nums_to_process` was not "
                                    "in set_nums found from `data_dir` filenames")
                self.set_nums_original   = set_nums.copy()
                self.set_nums_to_process = set_nums.copy()

        def _set_and_validate_superbatch_set_nums(superbatch_set_nums):
            if superbatch_set_nums != 'all' and not self.superbatch_dir:
                if superbatch_set_nums:
                    print(WARN, "`superbatch_set_nums` will be ignored, "
                          "since `superbatch_dir` is None")
                self.superbatch_set_nums = []
                return

            nums_to_process = _get_set_nums_to_process()

            if superbatch_set_nums == 'all':
                self.superbatch_set_nums = nums_to_process.copy()
            else:
                if any(num not in nums_to_process for num in superbatch_set_nums):
                    raise Exception("a `set_num` in `superbatch_set_nums` "
                                    "was not in set_nums found from "
                                    "`superbatch_folderpath` filename")
                self.superbatch_set_nums = superbatch_set_nums

        _set_and_validate_set_nums(set_nums)
        _set_and_validate_superbatch_set_nums(superbatch_set_nums)

    def _set_data_loader(self, data_loader):
        """Sets `data_loader`, from `data_loader`:

            - If `None` passed to `__init__`, will set as string in
              :meth:`_infer_data_info` - else, will use whatever passed.
            - If string, will match to a supported builtin.
            - If a function, will set to the function.
        """
        if isinstance(data_loader, LambdaType):  # custom
            self.data_loader = data_loader
        elif data_loader == 'numpy':
            self.data_loader = data_loaders.numpy_loader
        elif data_loader == 'numpy-lz4f':
            if not IMPORTS['LZ4F']:
                raise ImportError("`lz4framed` must be imported for "
                                  "`data_loader = 'numpy-lz4f'`")
            if getattr(self, 'full_batch_shape', None) is None:
                raise ValueError("'numpy-lz4f' data_loader requires "
                                 "`full_batch_shape` attribute defined")
            self.data_loader = data_loaders.numpy_lz4f_loader
        elif data_loader == 'hdf5':
            self.data_loader = data_loaders.hdf5_loader
        elif data_loader == 'hdf5-dataset':
            self.data_loader = data_loaders.hdf5_dataset_loader
        else:
            supported = DataGenerator._BUILTINS['data_loaders']
            raise ValueError(("unsupported data_loader '{}'; must be a custom "
                              "function, or one of {}").format(
                                  data_loader, ', '.join(supported)))

    def _set_preprocessor(self, preprocessor, preprocessor_configs):
        """Sets `preprocessor`, based on `preprocessor` passed to `__init__`:

            - If None, sets to :class:`GenericPreprocessor`, instantiated with
              `preprocessor_configs`.
            - If an uninstantiated class, will validate that it subclasses
              :class:`Preprocessor`, then isntantiate with `preprocessor_configs`.
            - If string, will match to a supported builtin.
            - Validates that the set `preprocessor` subclasses
              :class:`Preprocessor`.
        """
        def _set(preprocessor, preprocessor_configs):
            if preprocessor is None:
                self.preprocessor = GenericPreprocessor(**preprocessor_configs)
            elif isinstance(preprocessor, type):  # uninstantiated
                assert issubclass(preprocessor, Preprocessor
                                  ), "`preprocessor` must subclass `Preprocessor`"
                self.preprocessor = preprocessor(**preprocessor_configs)
            elif preprocessor == 'timeseries':
                self.preprocessor = TimeseriesPreprocessor(**preprocessor_configs)
            else:
                self.preprocessor = preprocessor
            assert isinstance(self.preprocessor, Preprocessor
                              ), "`preprocessor` must subclass `Preprocessor`"

        _set(preprocessor, preprocessor_configs)
        self.preprocessor._validate_configs()

    @property
    def batch_exhausted(self):
        """Is retrieved from and set in `preprocessor`.

        Ex: `self.batch_exhausted = 5` will set
        `self.preprocessor.batch_exhausted = 5`, and `print(self.batch_exhausted)`
        will then print `5` (or something else if `preprocessor` changes it
        internally).
        """
        return self.preprocessor.batch_exhausted

    @batch_exhausted.setter
    def batch_exhausted(self, value):
        self.preprocessor.batch_exhausted = value

    @property
    def batch_loaded(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        """
        return self.preprocessor.batch_loaded

    @batch_loaded.setter
    def batch_loaded(self, value):
        self.preprocessor.batch_loaded = value

    @property
    def slices_per_batch(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        """
        return self.preprocessor.slices_per_batch

    @slices_per_batch.setter
    def slices_per_batch(self, value):
        self.preprocessor.slices_per_batch = value

    @property
    def slice_idx(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        """
        return self.preprocessor.slice_idx

    @slice_idx.setter
    def slice_idx(self, value):
        self.preprocessor.slice_idx = value

    def _infer_data_info(self, data_dir, data_ext=None, data_loader=None,
                         base_name=None):
        """Infers unspecified essential attributes from directory and contained
        files info:

            - Checks that the data directory (`self.data_dir`) isn't empty
              (has files whose names don't start with `'.'`)
            - Retrieves data filenames and gets data extension (to most frequent
              ext in dir, excl. `labels_path` from count if in same dir)
            - Gets `data_loader` based on `data_ext`, if None, Else, checks
              whether it's one of builtins - if not, enforces it to be a callable.
            - Gets `base_name` as longest common substring among files with
              `data_ext` ext
            - Gets filepaths per `data_dir` and filenames. Or, if there's only
              one filename, and `data_ext == '.h5'`, sets `_hdf5_path` to the
              resulting filepath (hdf5_dataset format).
        """
        def _validate_directory(data_dir):
            # not guaranteed to catch hidden files
            nonhidden_files_names = [x for x in os.listdir(data_dir)
                                     if not x.startswith('.')]
            if len(nonhidden_files_names) == 0:
                raise Exception("`data_dir` is empty (%s)" % data_dir)

        def _infer_data_loader(extension, filenames):
            data_loader = {'.npy': 'numpy',
                           '.h5': 'hdf5'}[extension]

            if data_loader == 'hdf5' and len(filenames) == 1:
                data_loader = 'hdf5-dataset'
            return data_loader

        def _get_base_name(extension, filenames):
            def _longest_common_substr(data):
                substr = ''
                ref = data[0]
                for i in range(len(ref)):
                  for j in range(len(ref) - i + 1):
                    if j > len(substr) and all(ref[i:i+j] in x for x in data):
                      substr = ref[i:i+j]
                return substr

            filenames = [x.rstrip(extension) for x in filenames]
            base_name = _longest_common_substr(filenames)
            return base_name

        def _get_filepaths_or_set_hdf5_path(data_dir, filenames):
            if Path(filenames[0]).suffix == '.h5' and len(filenames) == 1:
                # hdf5_dataset format
                if not getattr(self, '_hdf5_path', None):
                    # avoid possibly overwriting via preload_superbatch()
                    self._hdf5_path = os.path.join(data_dir, filenames[0])
                return None

            filepaths = [os.path.join(data_dir, x) for x in filenames]
            print("Discovered %s files with matching format" % len(filepaths))
            return filepaths

        def _get_filenames_and_data_ext(data_dir, data_ext):
            def _infer_extension(data_dir, supported_extensions):
                extensions = [x.suffix for x in Path(data_dir).iterdir()
                              if (x.suffix in supported_extensions and
                                  str(x) != self.labels_path)]
                if len(extensions) == 0:
                    raise Exception("No files found with supported extensions: "
                                    + ', '.join(supported_extensions)
                                    + " in `data_dir` ", data_dir)
                # pick most frequent extension
                data_ext = max(set(extensions), key=extensions.count)

                if len(set(extensions)) > 1:
                    print(WARN, "multiple file extensions found in "
                          "`data_dir`; only", data_ext, "will be used "
                          "(specify `data_ext` if this is false)")
                return data_ext

            supported_extensions = DataGenerator._BUILTINS['data_extensions']
            if data_ext is None:
                data_ext = _infer_extension(data_dir, supported_extensions)

            filenames = [x.name for x in Path(data_dir).iterdir()
                         if (x.suffix == data_ext and str(x) != self.labels_path)]
            return filenames, data_ext

        _validate_directory(data_dir)
        filenames, data_ext = _get_filenames_and_data_ext(data_dir, data_ext)

        supported = DataGenerator._BUILTINS['data_loaders']
        if data_loader is None:
            data_loader = _infer_data_loader(data_ext, filenames)
        elif data_loader not in supported and not isinstance(
                data_loader, LambdaType):
            raise ValueError(("unsupported data_loader '{}'; must be a custom "
                             "function, or one of {}").format(
                                 data_loader, ', '.join(supported)))

        base_name = base_name or _get_base_name(data_ext, filenames)
        filepaths = _get_filepaths_or_set_hdf5_path(data_dir, filenames)

        return dict(data_loader=data_loader, base_name=base_name,
                    filenames=filenames, filepaths=filepaths,
                    data_ext=data_ext)

    def preload_superbatch(self):
        """Loads all data specified by `superbatch_set_nums` via
        :meth:`load_data`, and assigns them to `superbatch` for each `set_num`.
        """
        print(end='Preloading superbatch ... ')

        # get and set `superbatch` variant of data attributes:
        # data_loader, base_name, filenames, filepaths, data_ext (same)
        info = self._infer_data_info(self.superbatch_dir, self.data_ext)
        for name in info:
            alias = '_superbatch_' + name
            setattr(self, alias, info[name])

        self.superbatch = {}  # empty if not empty
        for set_num in self.superbatch_set_nums:
            self.superbatch[set_num] = self.load_data(set_num)
            print(end='.')

        num_samples = sum(len(batch) for batch in self.superbatch.values())
        print(" finished, w/", num_samples, "total samples")

    def preload_labels(self):
        """Loads all labels into `all_labels` using `labels_preloader`.
        See :mod:`~deeptrain.util.labels_preloaders`.

        Selects default `labels_preloader` based on `labels_path` path extension.
        """
        if self.labels_preloader is not None:
            self.labels_preloader(self)
            return

        ext = Path(self.labels_path).suffix
        if ext == '.csv':
            self.labels_preloader = labels_preloaders.csv_preloader
        elif ext == '.h5':
            self.labels_preloader = labels_preloaders.hdf5_preloader
        self.labels_preloader(self)

    def _init_and_validate_kwargs(self, kwargs):
        """Sets and validates `kwargs` passed to `__init__`.

            - Ensures kwargs are functional (compares against names in
              :data:`~deeptrain.util._default_configs._DEFAULT_DATAGEN_CFG`.
            - Sets whichever names were passed with `kwargs`, and defaults
              the rest.
        """
        def _validate_kwarg_names(kwargs):
            for kw in kwargs:
                if kw not in _DEFAULT_DATAGEN_CFG:
                    raise ValueError("unknown kwarg: '{}'".format(kw))

        def _set_kwargs(kwargs):
            class_kwargs = deepcopy(_DATAGEN_CFG)
            class_kwargs.update(kwargs)

            for attribute in class_kwargs:
                setattr(self, attribute, class_kwargs[attribute])

        def _validate_shuffle_group_():
            if self.shuffle_group_batches and self.shuffle_group_samples:
                print(WARN, "`shuffle_group_batches` will be ignored since "
                      "`shuffle_group_samples` is also ==True")

        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)
        _validate_shuffle_group_()

    def _init_class_vars(self):
        """Instantiates various internal attributes. Most of these are saved
        and loaded by :class:`TrainGenerator` by default."""
        _defaults = dict(
            all_data_exhausted=False,
            batch_exhausted=True,
            batch_loaded=False,
            epoch=0,  # managed externally
            superbatch={},
            _group_batch=None,
            _group_labels=None,
            set_num=None,
            set_name=None,
            _set_names=[],
            start_increment=0,
            )
        for k, v in _defaults.items():
            setattr(self, k, getattr(self, k, v))

        # used in saving & report generation
        self._path_attrs = ['data_dir', 'labels_path', 'superbatch_dir']
