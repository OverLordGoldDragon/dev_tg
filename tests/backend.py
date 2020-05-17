import os
import contextlib
import shutil
import tempfile
import numpy as np

from pathlib import Path
from termcolor import cprint

from deeptrain import util
from deeptrain import metrics
from deeptrain import TrainGenerator, DataGenerator


BASEDIR = str(Path(__file__).parents[1])
TF_KERAS = bool(os.environ.get('TF_KERAS', '0') == '1')


if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras import losses as keras_losses
    from tensorflow.keras import metrics as keras_metrics
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
else:
    from keras import backend as K
    from keras import losses as keras_losses
    from keras import metrics as keras_metrics
    from keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from keras.regularizers import l2
    from keras.optimizers import Adam
    from keras.models import Model


def _init_session(C, weights_path=None, loadpath=None, model=None,
                  model_fn=None):
    if model is None:
        model = model_fn(weights_path, **C['model'])
    dg  = DataGenerator(**C['datagen'])
    vdg = DataGenerator(**C['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath, **C['traingen'])
    return tg


def destroy_session(tg):
    def _clear_data(tg):
        tg.datagen.batch = []
        tg.datagen.superbatch = {}
        tg.val_datagen.batch = []
        tg.val_datagen.superbatch = {}

    _clear_data(tg)
    [delattr(tg, name) for name in ('model', 'datagen', 'val_datagen')]
    del tg


def _do_test_load(tg, C, init_session_fn):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    destroy_session(tg)

    weights_path, loadpath = _get_latest_paths(logdir)
    tg = init_session_fn(C, weights_path, loadpath)


@contextlib.contextmanager
def tempdir(dirpath=None):
    if dirpath is not None and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)
    elif dirpath is None:
        dirpath = tempfile.mkdtemp()
    else:
        os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def notify(tests_done):
    def wrap(test_fn):
        def _notify(monkeypatch, *args, **kwargs):
            try:
                is_mp = monkeypatch.__class__.__name__ == 'MonkeyPatch'
            except:
                test_fn(*args, **kwargs)

            if ('monkeypatch' in util.argspec(test_fn) and is_mp) or not is_mp:
                test_fn(monkeypatch, *args, **kwargs)
            elif is_mp:
                test_fn(*args, **kwargs)
            else:
                test_fn(monkeypatch, *args, **kwargs)

            name = test_fn.__name__.split('test_')[-1]
            tests_done[name] = True
            print("\n>%s TEST PASSED" % name.upper())

            if all(tests_done.values()):
                test_name = test_fn.__module__.replace(
                    '_', ' ').replace('tests.', '').upper()
                cprint(f"<< {test_name} PASSED >>\n", 'green')
        return _notify
    return wrap


def make_classifier(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'metrics', 'optimizer',
                       'num_classes', 'filters', 'kernel_size',
                       'dropout', 'dense_units')
        return [kw[key] for key in expected_kw]

    (batch_shape, loss, metrics, optimizer, num_classes, filters,
     kernel_size, dropout, dense_units) = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = ipt

    for f, ks in zip(filters, kernel_size):
        x = Conv2D(f, ks, activation='relu', padding='same')(x)

    x   = MaxPooling2D(pool_size=(2, 2))(x)
    x   = Dropout(dropout[0])(x)
    x   = Flatten()(x)
    x   = Dense(dense_units, activation='relu')(x)

    x   = Dropout(dropout[1])(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def make_autoencoder(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'metrics', 'optimizer',
                       'activation', 'filters', 'kernel_size', 'strides',
                       'up_sampling_2d')
        return [kw[key] for key in expected_kw]

    (batch_shape, loss, metrics, optimizer, activation, filters, kernel_size,
     strides, up_sampling_2d) = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = ipt

    configs = (activation, filters, kernel_size, strides, up_sampling_2d)
    for act, f, ks, s, ups in zip(*configs):
        if ups is not None:
            x = UpSampling2D(ups)(x)
        x = Conv2D(f, ks, strides=s, activation=act, padding='same')(x)
    out = x

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def make_timeseries_classifier(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'units', 'optimizer')
        return [kw[key] for key in expected_kw]

    batch_shape, loss, units, optimizer = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, return_sequences=False, stateful=True,
               kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4),
               bias_regularizer=l2(1e-4))(ipt)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


class ModelDummy():
    """Proxy model for testing (e.g. methods via `self`)"""
    def __init__(self):
        self.loss = 'mse'
        self.output_shape = (8, 1)
        self.input_shape = (8, 16, 2)

    def _standardize_user_data(self, *args, **kwargs):
        pass

    def _make_train_function(self, *args, **kwargs):
        pass


class TraingenDummy():
    """Proxy class for testing (e.g. methods via `self`)"""

    class Datagen():
        def __init__(self):
            self.shuffle = False

    def __init__(self):
        self.model = ModelDummy()
        self.datagen = TraingenDummy.Datagen()
        self.val_datagen = TraingenDummy.Datagen()

        self.eval_fn_name = 'predict'
        self.key_metric = 'f1_score'
        self.key_metric_fn = metrics.f1_score
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

        self.logs_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_logs')
        self.best_models_dir = os.path.join(BASEDIR, 'tests', '_outputs',
                                            '_models')
        self.model_configs = None
        self.model_name_configs = None
        self.model_num_continue_from_max = False
        self.model_base_name = 'M'
        self.name_process_key_fn = util.configs._NAME_PROCESS_KEY_FN

    def set_shapes(self, batch_size, label_dim):
        self.batch_size = batch_size
        self._inferred_batch_size = batch_size
        self.model.output_shape = (batch_size, label_dim)

    def set_cache(self, y_true, y_pred):
        self._labels_cache = y_true.copy()
        self._preds_cache = y_pred.copy()
        self._sw_cache = np.ones(y_true.shape)
        self._class_labels_cache = y_true.copy()


for name in ('_transform_eval_data', '_validate_data_shapes',
             '_validate_class_data_shapes', '_compute_metrics',
             '_weighted_normalize_preds'):
    setattr(TraingenDummy, name, getattr(util.training, name))
