# -*- coding: utf-8 -*-
import os
import inspect
thisdir = os.path.dirname(inspect.stack()[0][1])

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model

from deeptrain import TrainGenerator, DataGenerator


def init_session(CONFIGS, model_fn):
    model = model_fn(**CONFIGS['model'])
    dg  = DataGenerator(**CONFIGS['datagen'])
    vdg = DataGenerator(**CONFIGS['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, **CONFIGS['traingen'])
    return tg

#### Reusable TrainGenerator, DataGenerator, model configs #####################
def make_autoencoder(batch_shape, optimizer, loss, metrics,
                     filters, kernel_size, strides, activation, up_sampling_2d,
                     input_dropout, preout_dropout):
    ipt = Input(batch_shape=batch_shape)
    x   = Dropout(input_dropout)(ipt)

    configs = (activation, filters, kernel_size, strides, up_sampling_2d)
    for a, f, ks, s, ups in zip(*configs):
        x = UpSampling2D(ups)(x) if ups else x
        x = Conv2D(f, ks, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(a)(x)

    x   = Dropout(preout_dropout)(x)
    x   = Conv2D(1, (3, 3), strides=1, padding='same', activation='sigmoid')(x)
    out = x

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)
    return model

batch_size = 128
width, height, channels = 28, 28, 1
AE_MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,
    optimizer='nadam',
    activation=['relu'] * 5,
    filters=[6, 12, 2, 6, 12],
    kernel_size=[(3, 3)] * 5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
    input_dropout=.5,
    preout_dropout=.4,
)
datadir = os.path.join(thisdir, "dir", "data", "image")
DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    batch_size=batch_size,
    shuffle=True,
    superbatch_set_nums='all',
)
VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    batch_size=batch_size,
    shuffle=False,
    superbatch_set_nums='all',
)
TRAINGEN_CFG = dict(
    epochs=6,
    logs_dir=os.path.join(thisdir, 'dir', 'outputs', 'logs'),
    best_models_dir=os.path.join(thisdir, 'dir', 'outputs', 'models'),
    eval_fn='predict',
)
AE_TRAINGEN_CFG = TRAINGEN_CFG.copy()
# CL_TRAINGEN_CFG = TRAINGEN_CFG.copy()
AE_TRAINGEN_CFG.update({'model_configs': AE_MODEL_CFG,
                        'input_as_labels': True,
                        'max_is_best': False})
# CL_TRAINGEN_CFG.update({'model_configs': CL_MODEL_CFG})

data_cfgs = {'datagen':     DATAGEN_CFG,
             'val_datagen': VAL_DATAGEN_CFG}
AE_CONFIGS = {'model':      AE_MODEL_CFG,
              'traingen':   AE_TRAINGEN_CFG}
# CL_CONFIGS = {'model':      CL_MODEL_CFG,
#               'traingen':   CL_TRAINGEN_CFG}
AE_CONFIGS.update(data_cfgs)
# CL_CONFIGS.update(data_cfgs)
CONFIGS = {'traingen': TRAINGEN_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'model': AE_MODEL_CFG}
