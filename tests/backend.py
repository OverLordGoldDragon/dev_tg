import os
import contextlib
import shutil
import numpy as np
from deeptrain.util.metrics import f1_score


if os.path.isdir(r"C:\Desktop\School\Deep Learning\DL_code\dev_tg"):
    BASEDIR = r"C:\Desktop\School\Deep Learning\DL_code\dev_tg"
else:
    BASEDIR = ''
TF_KERAS = bool(os.environ.get('TF_KERAS', '0') == '1')


if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras import losses as keras_losses
    from tensorflow.keras import metrics as keras_metrics
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
else:
    from keras import backend as K
    from keras import losses as keras_losses
    from keras import metrics as keras_metrics
    from keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from keras.regularizers import l2
    from keras.models import Model


@contextlib.contextmanager
def tempdir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


class ModelDummy():
    """Proxy model for testing (e.g. methods via `cls`)"""
    def __init__(self):
        self.loss = 'mse'
        self.output_shape = (8, 1)
        self.input_shape = (8, 16, 2)

    def _standardize_user_data(self, *args, **kwargs):
        pass

    def _make_train_function(self, *args, **kwargs):
        pass


class TraingenDummy():
    """Proxy class for testing (e.g. methods via `cls`)"""

    class Datagen():
        def __init__(self):
            self.x = 0

    def __init__(self):
        self.model = ModelDummy()
        self.datagen = TraingenDummy.Datagen()
        self.val_datagen = TraingenDummy.Datagen()

        self.eval_fn_name = 'predict'
        self.key_metric = 'f1_score'
        self.key_metric_fn = f1_score
        self.class_weights = None
        self.val_class_weights = None
        self.batch_size = 8
        self._inferred_batch_size = 8

        self.best_subset_size = None
        self.pred_weighted_slices_range = None
        self.predict_threshold = .5
        self.dynamic_predict_threshold_min_max = None
        self.loss_weighted_slices_range = None
        self.pred_weighted_slices_range = None

        self.val_metrics = []
        self._sw_cache = []

    def set_shapes(self, batch_size, label_dim):
        self.batch_size = batch_size
        self._inferred_batch_size = batch_size
        self.model.output_shape = (batch_size, label_dim)

    def set_cache(self, y_true, y_pred):
        self._labels_cache = y_true.copy()
        self._preds_cache = y_pred.copy()
        self._sw_cache = np.ones(y_true.shape)
        self._class_labels_cache = y_true.copy()
