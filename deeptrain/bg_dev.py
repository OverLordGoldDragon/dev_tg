# -*- coding: utf-8 -*-
"""TODO:
    - label preloaders
    - labels_path non-positional (autoencoders)
    - Ambiguous `data_ext` vs. `data_format`? (try `data_loader`?)
"""
import os
import h5py
import random
import numpy as np
import pandas as pd
from pathlib import Path
from termcolor import colored
from .pp_dev import GenericPreprocessor, TimeseriesPreprocessor


WARN = colored('WARNING:', 'red')
NOTE = colored('NOTE:',    'blue')


try:
    import lz4framed as lz4f
    LZ4F_IMPORTED = True
except:
    LZ4F_IMPORTED = False


###############################################################################
class BatchGenerator():
    SUPPORTED_CATEGORIES = {'image', 'timeseries'}
    SUPPORTED_FORMATS = {'numpy', 'numpy-memmap', 'numpy-lz4f',
                         'hdf5', 'hdf5-dataset'}
    SUPPORTED_EXTENSIONS = {'.npy', '.h5'}

    def __init__(self, data_dir, labels_path, batch_size, data_category,
                 data_format=None,
                 preprocessor_configs=None,
                 base_name=None,
                 shuffle=False,
                 dtype='float32',
                 data_ext=None,
                 superbatch_dir=None,
                 set_nums=None,
                 superbatch_set_nums=None,
                 full_batch_shape=None):  # TODO move to kwargs?
        self.data_dir=data_dir
        self.labels_path=labels_path
        self.batch_size=batch_size
        self.data_format=data_format
        self.preprocessor_configs=preprocessor_configs or {}
        self.data_category=data_category
        self.base_name=base_name
        self.shuffle=shuffle
        self.dtype=dtype
        self.data_ext=data_ext
        self.superbatch_dir=superbatch_dir
        
        info = self._infer_and_get_data_info(data_dir, data_ext)
        name_and_alias = [
            ('data_format', 'data_format'), ('base_name', 'base_name'),
            ('filenames', '_filenames',), ('filepaths', '_filepaths'),
            ('data_ext', 'data_ext')]
        [setattr(self, alias, info[name]) for name, alias in name_and_alias]

        self.load_data = self._get_data_loader(self.data_format)
        self._set_class_params(set_nums, superbatch_set_nums)
        self._set_preprocessor(data_category, self.preprocessor_configs)

        self.preload_labels()
        self.init_class_vars()
        print("BatchGenerator initiated")


    ######### Main methods #########
    def get(self):
        self._validate_batch()
        return self.preprocessor.process(self.batch)

    def advance_batch(self, forced=False, is_recursive=False):
        if not is_recursive:
            if self.batch_loaded and not forced:
                print(WARN, "'batch_loaded'==True; advance_batch() does "
                      "nothing \n(to force next batch, set 'forced'=True')")
                return
            self.batch = []
            self.labels = []        

        self.set_num = self.set_nums_to_process.pop(0)
        self.labels.extend(self.all_labels[self.set_num])
        self.batch.extend(self._get_next_batch())

        if len(self.batch) != self.batch_size:
            if len(self.batch) > self.batch_size:
                raise Exception("len(batch) exceeds batch_size")
            self.advance_batch(forced, is_recursive=True)

        if isinstance(self.batch, list):
            self.batch = np.asarray(self.batch)
            self.labels = np.asarray(self.labels)

        self.batch_loaded = True
        self.batch_exhausted = False
        self.all_data_exhausted = False
        if hasattr(self, 'slice_idx'):
            self.slice_idx = None if self.slice_idx is None else 0
        self._synch_to_preprocessor(self._SYNCH_ATTRS)

    ######### Main method helpers #########        
    def _get_next_batch(self):
        if self.set_num in self.superbatch_set_nums:
            return self.superbatch[self.set_num]
        else:
            return self.load_data(self.set_num)

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

    def verify_state_and_notify(self, forced):
        if self.batch_loaded and not forced:
            print(WARN + "'batch_loaded'==True; advance_batch() does nothing")
            print("(to force next batch, set 'forced'=True')")
            return False, False

        if not hasattr(self, '_verify_state_and_notify'):
            return True, None
        return self._verify_state_and_notify(forced)

    ######### Init methods #########
    def _set_class_params(self, set_nums, superbatch_set_nums):
        def _set_nums_from_dir(_dir):
            def _sort_ascending(ls):
                return list(map(str, sorted(map(int, ls))))

            nums_from_dir = []
            for filename in os.listdir(_dir):
                if Path(filename).suffix == self.data_ext and (
                        self.base_name in filename):
                    num = ''.join(x for x in filename.replace(self.base_name, '')
                                  if x.isdigit())
                    nums_from_dir.append(num)
            return _sort_ascending(nums_from_dir)

        def _set_nums_from_hdf5_dataset(hdf5_dataset):
            return [num for num in list(hdf5_dataset.keys()) if (
                num.isdigit() or isinstance(num, (float, int)))]
            
        def _set_or_validate_set_nums(set_nums):
            if self.data_format == 'hdf5-dataset':
                nums_from_dir = _set_nums_from_hdf5_dataset(self._hdf5_dataset)
            else:
                nums_from_dir = _set_nums_from_dir(self.data_dir)

            if set_nums is None:
                self.set_nums_original   = nums_from_dir.copy()
                self.set_nums_to_process = nums_from_dir.copy()
                print(len(nums_from_dir), "set nums inferred; if more are "
                      "expected, ensure file names contain a common substring "
                      "w/ a number (e.g. 'train1.npy', 'train2.npy', etc)")
            elif any([(num not in nums_from_dir) for num in set_nums]):
                raise Exception("a `set_num` in `set_nums_to_process` was not "
                                "in set_nums found from `data_dir` filenames")

        def _set_or_validate_superbatch_set_nums(superbatch_set_nums):
            if self.superbatch_dir is None:
                if superbatch_set_nums is not None:
                    print(WARN, "`superbatch_set_nums` will be ignored, "
                          "since `superbatch_dir` is None")
                self.superbatch_set_nums = []
                return

            if self.data_format == 'hdf5-dataset':
                nums_from_dir = _set_nums_from_hdf5_dataset(self._hdf5_dataset)
            else:
                nums_from_dir = _set_nums_from_dir(self.superbatch_dir)
 
            if superbatch_set_nums is None:
                self.superbatch_set_nums = nums_from_dir.copy()
            elif any([num not in nums_from_dir for num in superbatch_set_nums]):
                raise Exception("a `set_num` in `superbatch_set_nums` "
                                "was not in set_nums found from "
                                "`superbatch_folderpath` filename")

        _set_or_validate_set_nums(set_nums)
        _set_or_validate_superbatch_set_nums(superbatch_set_nums)

    def _get_data_loader(self, data_format):
        def _get_path(set_num):
            filename = self.base_name + str(set_num) + self.data_ext
            return os.path.join(self.data_dir, filename)

        def numpy_loader(set_num):
            return np.load(_get_path(set_num)).astype(self.dtype)

        def numpy_lz4f_loader(set_num):
            bytes_npy = lz4f.decompress(np.load(_get_path(set_num)))
            return np.frombuffer(bytes_npy, dtype=self.dtype).reshape(
                *self.full_batch_shape)

        def hdf5_loader(set_num):
            with h5py.File(_get_path(set_num), 'r') as hdf5_file:
                a_key = list(hdf5_file.keys())[0]  # only one should be present
                return hdf5_file[a_key][:]

        def hdf5_dataset_loader(self, set_num):
            return self._hdf5_dataset[str(set_num)][:]

        if data_format == 'numpy':
            load_data = numpy_loader
        elif data_format == 'numpy-lz4f':
            if not LZ4F_IMPORTED:
                raise ImportError("`lz4framed` must be imported for "
                                  "`data_format = 'numpy-lz4f'`")
            load_data = numpy_lz4f_loader
        elif data_format == 'hdf5':
            load_data = hdf5_loader
        elif data_format == 'hdf5-dataset':
            load_data = hdf5_dataset_loader
        else:
            raise Exception("unsupported data format: '%s'" % data_format)
        return load_data

    def _set_preprocessor(self, data_category, preprocessor_configs):
        def _validate_preprocessor_attributes(required):
            for name in required:
                if name not in vars(self.preprocessor):
                    raise AttributeError(
                        ("`{}` attribute not found in `preprocessor`; required "
                         "are: {}").format(name, ', '.join(required)))

        supported = BatchGenerator.SUPPORTED_CATEGORIES
        if data_category == 'image':
            self.preprocessor = GenericPreprocessor(**preprocessor_configs)
            self._SLICE_ATTRS = ()
        elif data_category == 'timeseries':
            self.preprocessor = TimeseriesPreprocessor(**preprocessor_configs)
            self._SLICE_ATTRS = ('slice_idx', 'slices_per_batch')
        elif data_category is None:
            print(WARN, "`data_category` not set; defaulting to "
                  "Generic Preprocessor. Supported are:", ', '.join(supported))
        else:
            print(WARN, "unknown `data_category`:", data_category,
                  "; defaulting `preprocessor` to GenericPreprocessor. "
                  "Supported are:", ', '.join(supported))
    
        self._BATCH_ATTRS = ('batch_exhausted', 'batch_loaded')
        self._SYNCH_ATTRS = (*self._BATCH_ATTRS, *self._SLICE_ATTRS)
        _validate_preprocessor_attributes(self._SYNCH_ATTRS)

        self._synch_from_preprocessor(self._SYNCH_ATTRS)

    def _synch_from_preprocessor(self, attrs):
        [setattr(self, x, getattr(self.preprocessor, x)) for x in attrs]

    def _synch_to_preprocessor(self, attrs):
        [setattr(self.preprocessor, x, getattr(self, x)) for x in attrs]
    
    def _infer_and_get_data_info(self, data_dir, data_ext=None, data_format=None):
        def _get_data_format(extension, filenames):
            data_format = {'.npy': 'numpy',
                           '.h5': 'hdf5'}[extension]

            if data_format == 'numpy':
                print(NOTE, "inferred data format: 'numpy'; "
                      "will load via np.load - if compression / memapping "
                      "is used, specify the exact `data_format`.")
            elif data_format == 'hdf5' and len(filenames) == 1:
                print(NOTE, "inferred data format: 'hdf5-dataset' "
                      "(all data in one file, one group key per batch)")
                data_format = 'hdf5-dataset'
            elif data_format == 'hdf5':
                print(NOTE, "inferred data format: 'hdf5'.")
                print(NOTE, "for SimpleBatchgen, .h5 files should only "
                      "have one group key")
            return data_format
        
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
            
            supported_extensions = BatchGenerator.SUPPORTED_EXTENSIONS
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
        
        supported = BatchGenerator.SUPPORTED_FORMATS
        if data_format is None:
            data_format = _get_data_format(data_ext, filenames)
        elif data_format not in supported:
            msg = "unsupported data_format '{}'; must be one of {}".format(
                    data_format, ', '.join(supported))
            raise ValueError(msg)

        base_name = _get_base_name(data_ext, filenames)
        filepaths = _get_filepaths(data_dir, filenames)

        return dict(data_format=data_format, base_name=base_name,
                    filenames=filenames, filepaths=filepaths, 
                    data_ext=data_ext)

    def preload_superbatch(self):
        print(end='Preloading superbatch ... ')

        info = self._infer_and_get_data_info(self.superbatch_dir,
                                             self.data_ext)
        name_and_alias = [(name, '_superbatch_' + name) for name in info]
        [setattr(self, alias, info[name]) for name, alias in name_and_alias]
        load_data = self._get_data_loader(self.data_format)
        
        self.superbatch = {}  # empty if not empty
        for set_num in self.superbatch_set_nums:
            self.superbatch[set_num] = load_data(set_num)
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

    def init_class_vars(self):
        _defaults = dict(
            all_data_exhausted=False,
            batch_exhausted=True,
            batch_loaded=False,
            epoch=0,  # managed externally
            superbatch={},
            set_num=-1,
            start_increment=0,
            # attributes to skip in TrainGenerator.load(), i.e. will use 
            # those passed to __init__ instead // # TODO put in docstr instead
            loadskip_list=['data_dir', 'labels_path', 'superbatch_dir', 
                           'data_format', 'set_nums_original',
                           'set_nums_to_process', 'superbatch_set_nums'],
            )
        for k, v in _defaults.items():
            setattr(self, k, getattr(self, k, v))

        # used in saving & report generation
        self._path_attrs = ['data_dir', 'labels_path', 'superbatch_dir']
