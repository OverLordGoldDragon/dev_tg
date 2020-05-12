# -*- coding: utf-8 -*-
"""TODO:
    - profile batch.extend speed
"""
import os
import h5py
import random
import numpy as np
import pandas as pd

from pathlib import Path
from copy import deepcopy
from types import LambdaType

from .pp_dev import GenericPreprocessor, TimeseriesPreprocessor
from .util.misc import ordered_shuffle
from .util.configs import _DATAGEN_CFG
from .util import data_loaders
from .util._backend import WARN, NOTE, IMPORTS
from .util._default_configs import _DEFAULT_DATAGEN_CFG


###############################################################################
class DataGenerator():
    BUILTIN_PREPROCESSORS = (GenericPreprocessor, TimeseriesPreprocessor)
    BUILTIN_DATA_LOADERS = {'numpy', 'numpy-memmap', 'numpy-lz4f',
                            'hdf5', 'hdf5-dataset'}
    SUPPORTED_DATA_EXTENSIONS = {'.npy', '.h5'}

    def __init__(self, data_dir, batch_size,
                 labels_path=None,
                 preprocessor=None,
                 preprocessor_configs=None,
                 data_loader=None,
                 labels_preloader=None,
                 base_name=None,
                 shuffle=False,
                 dtype='float32',
                 data_ext=None,
                 superbatch_dir=None,
                 set_nums=None,
                 superbatch_set_nums=None,
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
        self.dtype=dtype
        self.data_ext=data_ext

        if superbatch_set_nums == 'all':
            self.superbatch_dir = data_dir
        else:
            self.superbatch_dir = superbatch_dir

        info = self._infer_and_get_data_info(data_dir, data_ext, data_loader,
                                             base_name)
        self.data_loader = info['data_loader']
        self.base_name   = info['base_name']
        self._filenames  = info['filenames']
        self._filepaths  = info['filepaths']
        self.data_ext    = info['data_ext']

        self._set_data_loader(self.data_loader)
        self._set_class_params(set_nums, superbatch_set_nums)
        self._set_preprocessor(preprocessor, self.preprocessor_configs)

        if labels_preloader is not None:
            self.labels_preloader(self)
        elif labels_path is not None:
            self.preload_labels()
        else:
            self.all_labels = {}
            self.labels = []
        self._init_and_validate_kwargs(kwargs)
        self._init_class_vars()
        print("DataGenerator initiated")


    ######### Main methods #########
    def get(self, skip_validation=False):
        if not skip_validation:
            self._validate_batch()
        return self.preprocessor.process(self.batch, self.labels)

    def advance_batch(self, forced=False, is_recursive=False):
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

        if not is_recursive:
            if self.batch_loaded and not forced:
                print(WARN, "'batch_loaded'==True; advance_batch() does "
                      "nothing \n(to force next batch, set 'forced'=True')")
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
            self.labels.extend(self._labels_from_group_batch())
        self.batch.extend(self._get_next_batch())

        if self.batch_size is not None and len(self.batch) != self.batch_size:
            flag = _handle_batch_size_mismatch(forced)
            if flag == 'exit':
                return

        s = self._set_names.pop(0)
        self.set_name = s if not is_recursive else "%s+%s" % (self.set_name, s)
        self.batch = np.asarray(self.batch)
        if self.labels_path:
            self.labels = np.asarray(self.labels)

        self.batch_loaded = True
        self.batch_exhausted = False
        self.all_data_exhausted = False
        if hasattr(self, 'slice_idx'):
            self.slice_idx = None if self.slice_idx is None else 0
        self._synch_to_preprocessor(self._SYNCH_ATTRS)

    ######### Main method helpers #########
    def load_data(self, set_num):
        return self.data_loader(self, set_num)

    def _get_next_batch(self):
        if self._group_batch is not None:
            batch = self._batch_from_group_batch()
            self._update_group_batch_state()
        elif self.set_num in self.superbatch_set_nums:
            if self.superbatch:
                batch = self.superbatch[self.set_num]
            else:
                print(WARN, f"`set_num` ({self.set_num}) found in `superbatch_"
                      "set_nums` but `superbatch` is empty; call "
                      "`preload_superbatch()`")
                batch = self.load_data(self.set_num)
        else:
            batch = self.load_data(self.set_num)
        return batch

    def _batch_from_group_batch(self):
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_batch[start:end]

    def _labels_from_group_batch(self):
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_labels[start:end]

    def _update_group_batch_state(self):
        if (self._group_batch_idx + 1
            ) * self.batch_size == len(self._group_batch):
            self._group_batch = None
            self._group_labels = None
            self._group_batch_idx = None
        else:
            self._group_batch_idx += 1

    def on_epoch_end(self):
        self.epoch += 1
        self.preprocessor.on_epoch_end(self.epoch)
        self.reset_state()
        return self.epoch

    def update_state(self):
        self.preprocessor.update_state()
        self._synch_from_preprocessor(self._SYNCH_ATTRS)

        if self.batch_exhausted and self.set_nums_to_process == []:
            self.all_data_exhausted = True

    def reset_state(self):
        self.batch_exhausted = True
        self.batch_loaded = False
        if hasattr(self, 'slice_idx'):
            self.slice_idx = None if self.slice_idx is None else 0
        self.preprocessor.reset_state()
        self.set_nums_to_process = self.set_nums_original.copy()

        if self.shuffle:
            random.shuffle(self.set_nums_to_process)
            print('\nData set_nums shuffled\n')

    ######### Misc methods #########
    def _validate_batch(self):
        if self.all_data_exhausted:
            print(WARN, "all data exhausted; automatically resetting "
                  "datagen state")
            self.reset_state()
        if self.batch_exhausted:
            print(WARN, "batch exhausted; automatically advancing batch")
            self.advance_batch()

    def _make_group_batch_and_labels(self, n_batches):
        self._set_names = [f"{self.set_num}-{postfix}" for postfix in
                           "abcdefghijklmnopqrstuvwxyz"[:int(n_batches)]]
        gb = np.asarray(self.batch)
        lb = np.asarray(self.labels)
        assert len(gb) == len(lb), ("len(batch) != len(labels) ({} != {})"
                                    ).format(len(gb), len(lb))
        self.batch = []  # free memory

        if self.shuffle_group_samples:
            gb, lb = ordered_shuffle(gb, lb)
        elif self.shuffle_group_batches:
            gb_shape, lb_shape = gb.shape, lb.shape
            gb = gb.reshape(-1, self.batch_size, *gb_shape[1:])
            lb = lb.reshape(-1, self.batch_size, *lb_shape[1:])
            gb, lb = ordered_shuffle(gb, lb)
            gb, lb = gb.reshape(*gb_shape), lb.reshape(*lb_shape)

        self._group_batch = gb
        self._group_labels = lb
        self._group_batch_idx = 0

        self.labels = self._labels_from_group_batch()
        self.batch = self._batch_from_group_batch()
        self._update_group_batch_state()

    ######### Init methods #########
    def _set_class_params(self, set_nums, superbatch_set_nums):
        def _get_set_nums_to_load():
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

        def _set_or_validate_set_nums(set_nums):
            nums_to_load = _get_set_nums_to_load()

            if set_nums is None:
                self.set_nums_original   = nums_to_load.copy()
                self.set_nums_to_process = nums_to_load.copy()
                print(len(nums_to_load), "set nums inferred; if more are "
                      "expected, ensure file names contain a common substring "
                      "w/ a number (e.g. 'train1.npy', 'train2.npy', etc)")
            elif any([(num not in nums_to_load) for num in set_nums]):
                raise Exception("a `set_num` in `set_nums_to_process` was not "
                                "in set_nums found from `data_dir` filenames")

        def _set_or_validate_superbatch_set_nums(superbatch_set_nums):
            if self.superbatch_dir is None and superbatch_set_nums != 'all':
                if superbatch_set_nums is not None:
                    print(WARN, "`superbatch_set_nums` will be ignored, "
                          "since `superbatch_dir` is None")
                self.superbatch_set_nums = []
                return

            nums_to_load = _get_set_nums_to_load()

            if superbatch_set_nums is None or superbatch_set_nums == 'all':
                self.superbatch_set_nums = nums_to_load.copy()
            elif any([num not in nums_to_load for num in superbatch_set_nums]):
                raise Exception("a `set_num` in `superbatch_set_nums` "
                                "was not in set_nums found from "
                                "`superbatch_folderpath` filename")

        _set_or_validate_set_nums(set_nums)
        _set_or_validate_superbatch_set_nums(superbatch_set_nums)

    def _set_data_loader(self, data_loader):
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
            raise Exception("unsupported data_loader: '%s'" % data_loader)

    def _set_preprocessor(self, preprocessor, preprocessor_configs):
        def _set(preprocessor, preprocessor_configs):
            _builtins = DataGenerator.BUILTIN_PREPROCESSORS
            if preprocessor is None:
                self.preprocessor = GenericPreprocessor(**preprocessor_configs)
            elif isinstance(preprocessor, (_builtins, type)):
                if isinstance(preprocessor, type):  # uninstantiated
                    self.preprocessor = preprocessor(**preprocessor_configs)
                else:
                    self.preprocessor = preprocessor
            elif preprocessor == 'timeseries':
                self.preprocessor = TimeseriesPreprocessor(**preprocessor_configs)
            else:
                raise ValueError("`preprocessor` must be either string, class, "
                                 "or instantiated object - got: %s" % preprocessor)

        def _validate_preprocessor_attributes():
            def raise_err(name):
                _builtins = ', '.join(
                    (c.__module__ + '.' + c.__name__) for c in
                    DataGenerator.BUILTIN_PREPROCESSORS)
                raise AttributeError(
                    ("`{}` attribute not found in `preprocessor`; required "
                     "are: {}\nSupported builtin preprocessors are: {}"
                     ).format(name, ', '.join(required),
                              "'timeseries', %s" % _builtins))

            required = ('process', 'reset_state', 'update_state',
                        'on_epoch_end') + self._SYNCH_ATTRS
            for name in required:
                if name not in dir(self.preprocessor):
                    raise_err(name)

        _set(preprocessor, preprocessor_configs)

        if isinstance(self.preprocessor, TimeseriesPreprocessor):
            self._SLICE_ATTRS = ('slice_idx', 'slices_per_batch')
        else:
            self._SLICE_ATTRS = ()
        self._BATCH_ATTRS = ('batch_exhausted', 'batch_loaded')
        self._SYNCH_ATTRS = (*self._BATCH_ATTRS, *self._SLICE_ATTRS)

        _validate_preprocessor_attributes()
        self._synch_from_preprocessor(self._SYNCH_ATTRS)


    def _synch_from_preprocessor(self, attrs):
        [setattr(self, x, getattr(self.preprocessor, x)) for x in attrs]

    def _synch_to_preprocessor(self, attrs):
        [setattr(self.preprocessor, x, getattr(self, x)) for x in attrs]

    def _infer_and_get_data_info(self, data_dir, data_ext=None,
                                 data_loader=None, base_name=None):
        def _infer_data_loader(extension, filenames):
            data_loader = {'.npy': 'numpy',
                           '.h5': 'hdf5'}[extension]

            if data_loader == 'numpy':
                print(NOTE, "inferred data format: 'numpy'; "
                      "will load via np.load - if compression / memapping "
                      "is used, specify the exact `data_loader`.")
            elif data_loader == 'hdf5' and len(filenames) == 1:
                print(NOTE, "inferred data format: 'hdf5-dataset' "
                      "(all data in one file, one group key per batch)")
                data_loader = 'hdf5-dataset'
            elif data_loader == 'hdf5':
                print(NOTE, "inferred data format: 'hdf5'. For DataGenerator, "
                      ".h5 files should only have one group key")
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

            print(NOTE, "Inferred `base_name = '%s'`; " % base_name
                  + "if this is false, specify `base_name`.")
            return base_name

        def _get_filepaths(data_dir, filenames):
            if Path(filenames[0]).suffix == '.h5' and len(filenames) == 1:
                self._hdf5_path = os.path.join(data_dir, filenames[0])
                return None

            filepaths = [os.path.join(data_dir, x) for x in filenames]
            print("Discovered %s files with matching format" % len(filepaths))
            return filepaths

        def _get_filenames_and_data_ext(data_dir, data_ext):
            def _infer_extension(data_dir, supported_extensions):
                extensions = [x.suffix for x in Path(data_dir).iterdir()
                               if x.suffix in supported_extensions]
                data_ext = max(set(extensions), key=extensions.count)

                if len(set(extensions)) > 1:
                    print(WARN, "multiple file extensions found in "
                           "`data_dir`; only", data_ext, "will be used "
                           "(specify `data_ext` if this is false)")
                return data_ext

            supported_extensions = DataGenerator.SUPPORTED_DATA_EXTENSIONS
            if data_ext is None:
                data_ext = _infer_extension(data_dir, supported_extensions)

            filenames = [x for x in os.listdir(data_dir)
                         if Path(x).suffix == data_ext]
            if filenames == []:
                raise Exception("No files found with supported extensions: "
                                + ', '.join(supported_extensions)
                                + " in `data_dir` ", data_dir)
            return filenames, data_ext

        filenames, data_ext = _get_filenames_and_data_ext(data_dir, data_ext)

        supported = DataGenerator.BUILTIN_DATA_LOADERS
        if data_loader is None:
            data_loader = _infer_data_loader(data_ext, filenames)
        elif data_loader not in supported:
            msg = "unsupported data_loader '{}'; must be one of {}".format(
                    data_loader, ', '.join(supported))
            raise ValueError(msg)

        base_name = base_name or _get_base_name(data_ext, filenames)
        filepaths = _get_filepaths(data_dir, filenames)

        return dict(data_loader=data_loader, base_name=base_name,
                    filenames=filenames, filepaths=filepaths,
                    data_ext=data_ext)

    def preload_superbatch(self):
        print(end='Preloading superbatch ... ')

        info = self._infer_and_get_data_info(self.superbatch_dir,
                                             self.data_ext)
        name_and_alias = [(name, '_superbatch_' + name) for name in info]
        [setattr(self, alias, info[name]) for name, alias in name_and_alias]

        self.superbatch = {}  # empty if not empty
        for set_num in self.superbatch_set_nums:
            self.superbatch[set_num] = self.load_data(set_num)
            print(end='.')

        num_samples = sum([len(batch) for batch in self.superbatch.values()])
        print(" finished, w/", num_samples, "total samples")

    def preload_labels(self):
        ext = Path(self.labels_path).suffix
        if ext == '.csv':
            df = pd.read_csv(self.labels_path)
            self.all_labels = {}
            for set_num in df:
                self.all_labels[set_num] = df[set_num].to_numpy()
        elif ext == '.h5':
            with h5py.File(self.labels_path, 'r') as f:
                self.all_labels = {k:f[k][:] for k in list(f.keys())}

    def _init_and_validate_kwargs(self, kwargs):
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
                print(NOTE, "`shuffle_group_batches` will be ignored since "
                      "`shuffle_group_samples` is also ==True")

        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)

        _validate_shuffle_group_()

    def _init_class_vars(self):
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
            # attributes to skip in TrainGenerator.load(), i.e. will use
            # those passed to __init__ instead // # TODO put in docstr instead
            loadskip_list=['data_dir', 'labels_path', 'superbatch_dir',
                           'data_loader', 'set_nums_original',
                           'set_nums_to_process', 'superbatch_set_nums'],
            )
        for k, v in _defaults.items():
            setattr(self, k, getattr(self, k, v))

        # used in saving & report generation
        self._path_attrs = ['data_dir', 'labels_path', 'superbatch_dir']
