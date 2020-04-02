# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np

from pathlib import Path
from . import WARN, NOTE


def numpy_data_to_numpy_sets(savedir, data, labels, batch_size=32,
                             shuffle=True, data_basename='batch',
                             oversample_remainder=True, verbose=1):
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
        np.save(os.path.join(savedir, name), x)

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

        def _validate_savepath(savepath):
            if savepath.split('.')[-1] != '.h5':
                print(WARN, "`savepath` extension must be '.h5'; will append")
                savepath += '.h5'
            if Path(savepath).is_file():
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

        if loaddir is None and data is None:
            raise ValueError("one of `loaddir` or `data` must be not None")
        if loaddir is not None and data is not None:
            raise ValueError("can't use both `loaddir` and `data`")
        if data is not None and load_fn is not None:
            print(WARN, "`load_fn` ignored with `data != None`")

        _validate_savepath(savepath)
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
        print(last_set_num, "batches converted & saved as .hdf5 to", savepath)
