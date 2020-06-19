""":class:`DataGenerator` `batch` loader functions.

For working with a single file containing all training data, the function name
must include hdf5_dataset'`.
"""

import os
import numpy as np
import h5py
from ._backend import lz4f


def _path(self, set_num):
    """Returns absolute file path as `data_dir + base_name + set_num + data_ext`.
    """
    filename = self.base_name + str(set_num) + self.data_ext
    return os.path.join(self.data_dir, filename)


def numpy_loader(self, set_num):
    """For numpy arrays (.npy)."""
    return np.load(_path(self, set_num)).astype(self.data_loader_dtype)


def numpy_lz4f_loader(self, set_num):
    """For numpy arrays (.npy) compressed with `lz4framed`; see
    :func:`preprocessing.numpy_to_lz4f`.
    `self.data_loader_dtype` must be original (save) dtype.
    """
    bytes_npy = lz4f.decompress(np.load(_path(self, set_num)))
    return np.frombuffer(bytes_npy, dtype=self.data_loader_dtype).reshape(
        *self.full_batch_shape)


def hdf5_loader(self, set_num):
    """For hdf5 (.h5) files storing training data, one batch per file. `data_dir`
    in :class:`DataGenerator` must contain more than one non-labels '.h5' file
    to default to this loader.
    """
    with h5py.File(_path(self, set_num), 'r') as hdf5_file:
        a_key = list(hdf5_file.keys())[0]  # only one should be present
        return hdf5_file[a_key][:]


def hdf5_dataset_loader(self, set_num):
    """For a single hdf5 (.h5) file storign all training data, with separate
    batches accessed by string integer keys. `data_dir` in :class:`DataGenerator`
    must contain a single non-labels '.h5' file to default to this loader.
    """
    with h5py.File(self._hdf5_path, 'r') as hdf5_dataset:
        return hdf5_dataset[str(set_num)][:]
