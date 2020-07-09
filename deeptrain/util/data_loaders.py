""":class:`DataGenerator` `batch` & `labels` loader functions.

For working with a single file containing all training data, the function name
must include `'_dataset'`.

For :class:`DataGenerator` to default to a `_dataset` loader, `data_path` /
`labels_path` must be a file or a directory containing a single file.
"""

import os
import h5py
import numpy as np
import pandas as pd
from ._backend import lz4f


def _path(self, set_num, name='data'):
    """Returns absolute file path as `dir + base_name + set_num + ext`."""
    _dir = getattr(self, f"{name}_path")
    ext = getattr(self, f"_{name}_ext")
    base_name = getattr(self, f"_{name}_base_name")
    return os.path.join(_dir, base_name + str(set_num) + ext)


def numpy_loader(self, set_num, name='data'):
    """For numpy arrays (.npy)."""
    dtype = getattr(self, f"{name}_loader_dtype")
    return np.load(_path(self, set_num, name)).astype(dtype)


def numpy_dataset_loader(self, set_num, name='data'):
    """For a single numpy array (.npy) file storing all data, with separate
    batches accessed by indexing.

    Shape: `(n_batches, batch_size, *)`, i.e. `(batches, samples, *)`.
    """
    dtype = getattr(self, f"{name}_loader_dtype")
    path = getattr(self, f"_{name}_dataset_path")
    return np.load(path).astype(dtype)


def numpy_lz4f_loader(self, set_num, name='data'):
    """For numpy arrays (.npy) compressed with `lz4framed`; see
    :func:`preprocessing.numpy_to_lz4f`.
    `self.data_loader_dtype` must be original (save) dtype; if there's a mismatch,
    data of wrong value or shape will be decoded.

    Requires `data_batch_shape` / `labels_batch_shape` attribute to be set,
    as compressed representation omits shape info.
    """
    bytes_npy = lz4f.decompress(np.load(_path(self, set_num, name)))
    dtype = getattr(self, f"{name}_loader_dtype")
    shape = getattr(self, f"{name}_batch_shape")
    return np.frombuffer(bytes_npy, dtype=dtype).reshape(*shape)


def numpy_lz4f_dataset_loader(self, set_num, name='data'):
    """For a single numpy array (.npy) file compressed with `lz4framed`,
    storing all data see :func:`preprocessing.numpy_to_lz4f`.
    `self.data_loader_dtype` must be original (save) dtype; if there's a mismatch,
    data of wrong value or shape will be decoded.

    Shape: `(n_batches, batch_size, *)`, i.e. `(batches, samples, *)`.

    Requires `data_batch_shape` / `labels_batch_shape` attribute to be set,
    as compressed representation omits shape info.
    """
    dtype = getattr(self, f"{name}_loader_dtype")
    path = getattr(self, f"_{name}_dataset_path")
    shape = getattr(self, f"{name}_batch_shape")
    bytes_npy = lz4f.decompress(path)
    return np.frombuffer(bytes_npy, dtype=dtype).reshape(*shape)


def hdf5_loader(self, set_num, name='data'):
    """For hdf5 (.h5) files storing data one batch per file. `data_path`
    in :class:`DataGenerator` must contain more than one non-labels '.h5' file
    to default to this loader.
    """
    with h5py.File(_path(self, set_num, name), 'r') as hdf5_file:
        a_key = list(hdf5_file.keys())[0]  # only one should be present
        return hdf5_file[a_key][:]


def hdf5_dataset_loader(self, set_num, name='data'):
    """For a single hdf5 (.h5) file storing all data, with separate batches
    accessed by string integer keys."""
    path = getattr(self, f"_{name}_dataset_path")
    with h5py.File(path, 'r') as hdf5_dataset:
        return hdf5_dataset[str(set_num)][:]


def csv_loader(self, set_num, name='data'):
    """For .csv files (e.g. pandas.DataFrame)."""
    return pd.read_csv(_path(self, set_num, name))[set_num].to_numpy()


def csv_dataset_loader(self, set_num, name='data'):
    """For a single .csv file storing all data, with separate batches accessed
    by string integer key."""
    path = getattr(self, f"_{name}_dataset_path")
    return pd.read_csv(path)[set_num].to_numpy()
