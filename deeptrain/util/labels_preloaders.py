import h5py
import pandas as pd


def csv_preloader(self):
    df = pd.read_csv(self.labels_path)
    self.all_labels = {}
    for set_num in df:
        self.all_labels[set_num] = df[set_num].to_numpy()

def hdf5_preloader(self):
    with h5py.File(self.labels_path, 'r') as f:
        self.all_labels = {k: f[k][:] for k in list(f.keys())}
