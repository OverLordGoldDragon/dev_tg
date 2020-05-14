# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd

from pathlib import Path
from .util._backend import WARN, NOTE

try:
    import lz4framed as lz4f
except:
    lz4f = None


def numpy2D_to_csv(data, savepath, batch_size=None, columns=None, batch_dim=1,
                   overwrite=None):
    def _process_data(data, batch_size, batch_dim):
        assert data.ndim == 2, "`data` must be 2D"

        batch_size = batch_size or data.shape[1]
        if data.shape[1] != batch_size:
            try:
                # need to 'stack' samples dims, matching format of `data_to_hdf5`
                if batch_dim == 1:
                    data = data.reshape(-1, batch_size, order='C').T
                else:
                    data = data.reshape(batch_size, -1, order='F')
            except Exception as e:
                raise Exception("could not reshape `data`; specify different "
                                "`batch_size`.\nErrmsg: " + str(e))
        return data

    data = _process_data(data, batch_size, batch_dim)
    if columns is None:
        columns = list(map(str, range(data.shape[1])))

    df = pd.DataFrame(data, columns=columns)

    if savepath is not None:
        _validate_savepath(savepath, overwrite)
        df.to_csv(savepath, index=False)
        print(len(df.columns), "batch labels saved to", savepath)
    return df


def numpy_data_to_numpy_sets(data, labels, savedir=None, batch_size=32,
                             shuffle=True, data_basename='batch',
                             oversample_remainder=True, overwrite=None,
                             verbose=1):
    def _process_remainder(remainder, data, labels, oversample_remainder,
                           batch_size):
        action = "will" if oversample_remainder else "will not"
        print(("{} remainder samples for `batch_size={}`; {} oversample"
               ).format(int(remainder), batch_size, action))

        if oversample_remainder:
            to_oversample = batch_size - remainder
            idxs = np.random.randint(0, len(data), to_oversample)
            data = np.vstack([data, data[idxs]])
            labels = labels if labels.ndim > 1 else np.expand_dims(labels, 1)
            labels = np.vstack([labels, labels[idxs]])
        else:
            data = data[:-remainder]
            labels = labels[:-remainder]
        return data, labels

    remainder = len(data) % batch_size
    if remainder != 0:
        data, labels = _process_remainder(remainder, data, labels,
                                          oversample_remainder, batch_size)
    if shuffle:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        data, labels = data[idxs], labels[idxs]
        print("`data` & `labels` samples shuffled")

    n_batches = len(data) / batch_size
    assert (n_batches.is_integer()), ("len(data) must be divisible by "
                                      "`batch_size` ({} / {} = {})".format(
                                          len(data), batch_size, n_batches))
    data = data.reshape(int(n_batches), batch_size, *data.shape[1:])
    labels = labels.reshape(int(n_batches), batch_size, *labels.shape[1:])

    labels_path = os.path.join(savedir, "labels.h5")
    labels_hdf5 = h5py.File(labels_path, mode='w', libver='latest')

    for set_num, (x, y) in enumerate(zip(data, labels)):
        set_num = str(set_num + 1)
        name = "{}__{}.npy".format(data_basename, set_num)
        savepath = os.path.join(savedir, name)

        _validate_savepath(savepath, overwrite)
        np.save(savepath, x)

        labels_hdf5.create_dataset(set_num, data=y, dtype=data.dtype)
        if verbose:
            print("[{}/{}] {}-sample batch {} processed & saved".format(
                set_num, len(data), batch_size, name))

    labels_hdf5.close()
    if verbose:
        print("{} label sets saved to {}".format(len(data), labels_path))


def data_to_hdf5(savepath, batch_size, loaddir=None, data=None,
                 shuffle=False, compression='lzf', dtype='float32',
                 load_fn=None, overwrite=None, verbose=1):
    def _validate_args(savepath, loaddir, data, load_fn):
        def _validate_extensions(loaddir):
            supported = ('.npy',)
            extensions = list(set(x.suffix for x in Path(loaddir).iterdir()))
            if len(extensions) > 1:
                raise ValueError("cannot have more than one file extensions in "
                                 "`loaddir`; found %s" % ', '.join(extensions))
            elif load_fn is None and extensions[0] not in supported:
                raise ValueError(("unsupported file extension {}; supported "
                                  "are: {}. Alternatively, pass in `load_fn` "
                                  "that takes paths & index as arguments"
                                  ).format(extensions[0], ', '.join(supported)))

        if loaddir is None and data is None:
            raise ValueError("one of `loaddir` or `data` must be not None")
        if loaddir is not None and data is not None:
            raise ValueError("can't use both `loaddir` and `data`")
        if data is not None and load_fn is not None:
            print(WARN, "`load_fn` ignored with `data != None`")

        if Path(savepath).suffix != '.h5':
            print(WARN, "`savepath` extension must be '.h5'; will append")
            savepath += '.h5'
        _validate_savepath(savepath, overwrite)

        if loaddir is not None:
            _validate_extensions(loaddir)

        return savepath

    def _get_data_source(loaddir, data, batch_size, compression, shuffle):
        source = data if data is not None else [
            str(x) for x in Path(loaddir).iterdir() if not x.is_dir()]
        if shuffle:
            np.random.shuffle(source)

        if verbose:
            comp = compression if compression is not None else "no"
            shuf = "with" if shuffle else "without"
            print(("Making {}-size batches from {} extractables, using {} "
                   "compression, {} shuffling").format(
                       batch_size, len(source), comp, shuf))
        return source

    def _make_batch(source, j, batch_size, load_fn, verbose):
        def _get_data(source, j, load_fn):
            def _load_data(source, j, load_fn):
                if load_fn is not None:
                    return load_fn(source, j)
                path = source[j]
                if Path(path).suffix == '.npy':
                    return np.load(path)
            try:
                return _load_data(source, j, load_fn)
            except:
                return source[j]
        X = []
        while sum(map(len, X)) < batch_size:
            if j == len(source):
                print(WARN, "insufficient samples in extractable to make "
                      "batch; terminating")
                return None, j
            X.append(_get_data(source, j, load_fn))
            j += 1
            if sum(map(len, X)) > batch_size:
                raise ValueError("`batch_size` exceeded; {} > {}".format(
                    sum(map(len, X)), batch_size))
            if verbose:
                print(end='.')
        return np.vstack(X), j

    def _make_hdf5(hdf5_file, source, batch_size, dtype, load_fn, verbose):
        j, set_num = 0, 0
        while j < len(source):
            batch, j = _make_batch(source, j, batch_size, load_fn, verbose)
            if batch is None:
                break
            hdf5_file.create_dataset(str(set_num), data=batch, dtype=dtype,
                                     chunks=True, compression=compression)
            if verbose:
                print('', set_num, 'done', flush=True)
            set_num += 1
        return set_num - 1

    savepath = _validate_args(savepath, loaddir, data, load_fn)
    source = _get_data_source(loaddir, data, batch_size, compression, shuffle)

    with h5py.File(savepath, mode='w', libver='latest') as hdf5_file:
        last_set_num = _make_hdf5(hdf5_file, source, batch_size, dtype,
                                  load_fn, verbose)
    if verbose:
        print(last_set_num + 1, "batches converted & saved as .hdf5 to",
              savepath)


def numpy_to_lz4f(data, savepath=None, level=9, overwrite=None):
    if lz4f is None:
        raise Exception("cannot convert to lz4f without `lz4framed` installed; "
                        "run `pip install py-lz4framed`")
    data = data.tobytes()
    data = lz4f.compress(data, level=level)

    if savepath is not None:
        if Path(savepath).suffix != '.npy':
            print(WARN, "`savepath` extension must be '.npy'; will append")
            savepath += '.npy'
        _validate_savepath(savepath, overwrite)
        np.save(savepath, data)
        print("lz4f-compressed data saved to", savepath)
    return data


def _validate_savepath(savepath, overwrite):
    if not Path(savepath).is_file():
        return

    if overwrite is None:
        response = input(("Found existing file in `savepath`; "
                          "overwrite?' [y/n]\n"))
        if response == 'y':
            os.remove(savepath)
        else:
            raise SystemExit("program terminated.")
    elif overwrite is True:
        os.remove(savepath)
        print(NOTE, "removed existing file from `savepath`")
    else:
        raise SystemExit(("program terminated. (existing file in "
                          "`savepath` and `overwrite=False`)"))
