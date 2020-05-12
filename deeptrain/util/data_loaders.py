import os
import numpy as np
import h5py
from ._backend import lz4f


def _path(self, set_num):
    filename = self.base_name + str(set_num) + self.data_ext
    return os.path.join(self.data_dir, filename)


def numpy_loader(self, set_num):
    return np.load(_path(self, set_num)).astype(self.dtype)


def numpy_lz4f_loader(self, set_num):
    bytes_npy = lz4f.decompress(np.load(_path(self, set_num)))
    return np.frombuffer(bytes_npy, dtype=self.dtype).reshape(
        *self.full_batch_shape)


def hdf5_loader(self, set_num):
    with h5py.File(_path(self, set_num), 'r') as hdf5_file:
        a_key = list(hdf5_file.keys())[0]  # only one should be present
        return hdf5_file[a_key][:]


def hdf5_dataset_loader(self, set_num):
    with h5py.File(self._hdf5_path, 'r') as hdf5_dataset:
        return hdf5_dataset[str(set_num)][:]
