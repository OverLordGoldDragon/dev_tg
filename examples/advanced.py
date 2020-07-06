# -*- coding: utf-8 -*-
"""This example assumes you've read `basic.py`.
   - Multi-phase training
   - Callback streaming images to directory
   - Saving & loading
   - Variable-layer model building
"""
import os
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model

from pathlib import Path
from deeptrain import TrainGenerator, DataGenerator
from deeptrain.callbacks import VizAE2D

#%%# Configuration ###########################################################
# This scheme enables variable number of layers
def make_model(batch_shape, optimizer, loss, metrics,
               filters, kernel_size, strides, activation, up_sampling_2d,
               input_dropout, preout_dropout):
    """28x compression, denoising AutoEncoder."""
    ipt = Input(batch_shape=batch_shape)
    x   = ipt
    x   = Dropout(input_dropout)(x)

    configs = (activation, filters, kernel_size, strides, up_sampling_2d)
    for a, f, ks, s, ups in zip(*configs):
        x = UpSampling2D(ups)(x) if ups else x
        x = Conv2D(f, ks, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(a)(x)

    x   = Dropout(preout_dropout)(x)
    x   = Conv2D(1, (3, 3), 1, padding='same', activation='sigmoid')(x)
    out = x

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)
    return model

batch_size = 128
width, height, channels = 28, 28, 1
MODEL_CFG = dict(
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
datadir = os.path.join("dir", "data", "image")
DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'train'),
    batch_size=batch_size,
    shuffle=True,
    superbatch_set_nums='all',
)
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    batch_size=batch_size,
    shuffle=False,
    superbatch_set_nums='all',
)
# key_metric: the metric that decides the "best" model
# max_is_best: whether greater `key_metric` is better (we seek to minimize loss)
# input_as_labels: y = x, or model.fit(x, x)
# eval_fn: function to use in validation
# val_freq:               how often to validate     (default: every epoch)
# plot_history_freq:      how often to plot history (default: every epoch)
# unique_checkpoint_freq: how often to checkpoint   (default: every epoch)
# model_save_kw: kwargs passed to `model.save()`. Exclude optimizer since
#     we'll save its (and model's) weights separately to load later.
TRAINGEN_CFG = dict(
    epochs=6,
    logs_dir=os.path.join('dir', 'outputs', 'logs'),
    best_models_dir=os.path.join('dir', 'outputs', 'models'),
    model_configs=MODEL_CFG,
    key_metric='mae',
    max_is_best=False,
    input_as_labels=True,
    eval_fn='predict',
    val_freq={'epoch': 2},
    plot_history_freq={'epoch': 2},
    unique_checkpoint_freq={'epoch': 2},
    model_save_kw=dict(include_optimizer=False, save_format='h5'),
    model_name_configs=dict(input_dropout='idp', preout_dropout='pdp',
                            optimizer='', lr='', best_key_metric=None)
)
#%%# Create visualization callback ##########################################
TRAINGEN_CFG['callbacks'] = [VizAE2D(n_images=8, save_images=True)]
#%%# Create training objects ################################################
model = make_model(**MODEL_CFG)
dg    = DataGenerator(**DATAGEN_CFG)
vdg   = DataGenerator(**VAL_DATAGEN_CFG)
tg    = TrainGenerator(model, dg, vdg, **TRAINGEN_CFG)

# do save optimizer weights & attrs to load later
tg.saveskip_list.pop(tg.saveskip_list.index('optimizer_state'))
#%%# Train ##################################################################
tg.train()

#%%# Phase 2 ##########
# switch to Mean Absolute Error loss; greater penalty to smaller errors
# forces better image resolution.
# Internally, TrainGenerator will append 'mae' loss to same list as was 'mse'.
tg.model.compile('nadam', 'mae')
tg.epochs = 12
tg.train()

#%%# New session w/ changed model hyperparams ###############################
# get best save's model weights & TrainGenerator state
best_weights = [str(p) for p in Path(tg.best_models_dir).iterdir()
                if p.name.endswith('__weights.h5')]
best_state   = [str(p) for p in Path(tg.best_models_dir).iterdir()
                if p.name.endswith('__state.h5')]
latest_best_weights = sorted(best_weights, key=os.path.getmtime)[-1]
latest_best_state   = sorted(best_state,   key=os.path.getmtime)[-1]

tg.destroy(confirm=True)
del model, dg, vdg, tg
#%%
# increase preout_dropout to strengthen regularization
MODEL_CFG['preout_dropout'] = .7
MODEL_CFG['loss'] = 'mae'
# `epochs` will load at 12, so need to increase
TRAINGEN_CFG['epochs'] = 16
TRAINGEN_CFG['loadpath'] = latest_best_state
# ensure model_name uses 1 greater model_num, since using new hyperparams
TRAINGEN_CFG['model_num_continue_from_max'] = False
# must re-instantiate callbacks object to hold new TrainGenerator
TRAINGEN_CFG['callbacks'] = [VizAE2D(n_images=8, save_images=True)]

model = make_model(**MODEL_CFG)
model.load_weights(latest_best_weights)

dg    = DataGenerator(**DATAGEN_CFG)
vdg   = DataGenerator(**VAL_DATAGEN_CFG)
tg    = TrainGenerator(model, dg, vdg, **TRAINGEN_CFG)
# can also load via `tg.load`, but passing in `loadpath` and starting a
# new session should work better
#%%
tg.train()
#%%#################
cwd = os.getcwd()
print("Checkpoints can be found in", os.path.join(cwd, tg.logdir))
print("Best model can be found in", os.path.join(cwd, tg.best_models_dir))
print("AE progress can be found in", os.path.join(cwd, tg.logdir, 'misc'))
