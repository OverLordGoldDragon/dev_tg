# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np


def numpy_data_to_numpy_sets(savedir, data, labels, batch_size=32, 
                             shuffle=True, data_basename='batch',
                             oversample_remainder=True, verbose=1):
    def _process_remainder(data, labels, oversample_remainder, batch_size):
        action = "will" if oversample_remainder else "will not"
        print(("{} remainder samples for `batch_size={}`; {} oversample"
               ).format(int(remainder), batch_size, action))

        if oversample_remainder:
            idxs = np.random.randint(0, len(data), remainder)
            data = np.vstack([data, data[idxs]])
            labels = labels if labels.ndim > 1 else np.expand_dims(labels, 1)
            labels = np.vstack([labels, labels[idxs]])
        else:   
            data = data[:-remainder]
            labels = labels[:-remainder]
        return data, labels
            
    remainder = batch_size - len(data) % batch_size
    if remainder != 0:
        data, labels = _process_remainder(data, labels, oversample_remainder,
                                          batch_size)
    if shuffle:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        data, labels = data[idxs], labels[idxs]
        print("`data` & `labels` samples shuffled")
    
    n_batches = len(data) // batch_size
    data = data.reshape(n_batches, batch_size, *data.shape[1:])
    labels = labels.reshape(n_batches, batch_size, *labels.shape[1:])
    
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
    print("{} label sets saved to {}".format(len(data), labels_path))
