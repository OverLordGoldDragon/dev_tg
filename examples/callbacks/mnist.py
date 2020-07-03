# -*- coding: utf-8 -*-
"""TODO: make basic, advanced examples, then skip all the configs setup
& make_model redefinition and just jump into callbacks as in tests?
(and add note to reader to see basic/advanced first)
"""
import os
import numpy as np
from pathlib import Path

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parents[1], "data", "mnist")

from backend import BASEDIR, tempdir, notify
from backend import _init_session, _do_test_load, _get_test_names

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Activation
from tensorflow.keras.models import Model

from see_rnn import get_weights, features_2D
from deeptrain.callbacks import TraingenCallback, TraingenLogger
from deeptrain.callbacks import RandomSeedSetter
from deeptrain.callbacks import make_layer_hists_cb
from deeptrain import set_seeds

#################################################################
batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=[8, 16],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=32,
)
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    superbatch_dir=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}

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
    x   = Dense(num_classes)(x)
    out = Activation('softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model

#### CONFIGURE TESTING #######################################################
tests_done = {}
batch_size, width, height, channels = CONFIGS['model']['batch_shape']
logger_savedir = os.path.join(BASEDIR, 'tests', '_outputs', '_logger_outs')

set_seeds()

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)


def _make_logger_cb(get_data_fn=None, get_labels_fn=None, gather_fns=None):
    log_configs = {
        'weights': ['conv2d'],
        'outputs': 'conv2d',
        'gradients': ('conv2d',),
        'outputs-kw': dict(learning_phase=0),
        'gradients-kw': dict(learning_phase=0),
    }

    def on_save(self, stage=None):
        self.save(_id=self.tg.epoch)

    def on_load(self, stage=None):
        self.clear()
        self.load()

    def on_train_epoch_end(self, stage=None):
        self.log()

    TGL = TraingenLogger
    TGL.on_save = on_save
    TGL.on_load = on_load
    TGL.on_train_epoch_end = on_train_epoch_end
    tgl = TGL(logger_savedir, log_configs,
              get_data_fn=get_data_fn,
              get_labels_fn=get_labels_fn,
              gather_fns=gather_fns)
    return tgl


def _make_2Dviz_cb():
    class Viz2D(TraingenCallback):
        def __init__(self):
            pass

        def init_with_traingen(self, traingen):
            self.tg=traingen

        def on_val_end(self, stage=None):
            if stage == ('val_end', 'train:epoch'):
                self.viz()

        def viz(self):
            data = self._get_data()
            features_2D(data, tight=True, title_mode=False, cmap='hot',
                        norm=None, show_xy_ticks=[0,0], w=1.1, h=.55, n_rows=4)

        def _get_data(self):
            lg = None
            for cb in self.tg.callbacks:
                if getattr(cb.__class__, '__name__', '') == 'TraingenLogger':
                    lg = cb
            if lg is None:
                raise Exception("TraingenLogger not found in `callbacks`")

            last_key = list(lg.outputs.keys())[-1]
            outs = list(lg.outputs[last_key][0].values())[0]
            return outs[0].T

    viz2d = Viz2D()
    return viz2d


layer_hists_cbs = [
    {'train:epoch': [make_layer_hists_cb(mode='gradients:outputs'),
                     make_layer_hists_cb(mode='gradients:weights')]
     },
    # two dicts can be used to organize code, but will execute same as one
    {('val_end', 'train:epoch'): make_layer_hists_cb(mode='weights'),
     'val_end': make_layer_hists_cb(mode='outputs',
                                    configs={'title': dict(fontsize=13),
                                             'plot': dict(annot_kw=None)},
    )},
]

seed_setter = RandomSeedSetter(freq={'train:epoch': 1})

###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']), \
            tempdir(logger_savedir):
        callbacks = [_make_logger_cb(), _make_2Dviz_cb(), *layer_hists_cbs,
                     seed_setter]
        C['traingen']['callbacks'] = callbacks

        tg = init_session(C)
        tg.train()
        _test_load(tg, C)
        # set_seeds(reset_graph=True)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_traingen_logger():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']), \
            tempdir(logger_savedir):
        batch_shape = (batch_size, width, height, channels)

        n_classes = C['model']['num_classes']
        callbacks = [_make_logger_cb(
            get_data_fn=lambda: np.random.randn(*batch_shape),
            get_labels_fn=lambda: np.random.randint(0, 2,
                                                    (batch_size, n_classes)),
            gather_fns={'weights': get_weights},
            )]
        C['traingen']['callbacks'] = callbacks
        tg = init_session(C)
        tg.train()


tests_done.update({name: None for name in _get_test_names(__name__)})

