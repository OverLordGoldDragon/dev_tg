import os
import numpy as np
import h5py
from ._backend import lz4f


def _path(cls, set_num):
    filename = cls.base_name + str(set_num) + cls.data_ext
    return os.path.join(cls.data_dir, filename)


def numpy_loader(cls, set_num):
    return np.load(_path(cls, set_num)).astype(cls.dtype)


def numpy_lz4f_loader(cls, set_num):
    bytes_npy = lz4f.decompress(np.load(_path(cls, set_num)))
    return np.frombuffer(bytes_npy, dtype=cls.dtype).reshape(
        *cls.full_batch_shape)


def hdf5_loader(cls, set_num):
    with h5py.File(_path(cls, set_num), 'r') as hdf5_file:
        a_key = list(hdf5_file.keys())[0]  # only one should be present
        return hdf5_file[a_key][:]


def hdf5_dataset_loader(cls, set_num):
    with h5py.File(cls._hdf5_path, 'r') as hdf5_dataset:
        return hdf5_dataset[str(set_num)][:]
