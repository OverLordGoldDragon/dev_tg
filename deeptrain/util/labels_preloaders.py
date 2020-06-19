""":class:`DataGenerator` `all_labels` loader functions.

Custom functions must load all labels from `labels_path` and set them to
`all_labels`.
"""
import h5py
import pandas as pd


def csv_preloader(self):
    """Preloads labels from a single .csv file. Rows must be iterable by set_num.
    See examples/preprocessing/timeseries/val/labels.csv, and
        examples/preprocessing/timeseries.py.
    """
    df = pd.read_csv(self.labels_path)
    self.all_labels = {}
    for set_num in df:
        self.all_labels[set_num] = df[set_num].to_numpy()

def hdf5_preloader(self):
    """Preloads labels from a single .h5 file. Dataset keys must be iterable
    by set_num.
    See examples/preprocessing/mnist/val/labels.h5, and
        examples/preprocessing/mnist.py.
    """
    with h5py.File(self.labels_path, 'r') as f:
        self.all_labels = {k: f[k][:] for k in list(f.keys())}
