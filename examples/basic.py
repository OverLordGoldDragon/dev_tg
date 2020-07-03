# -*- coding: utf-8 -*-
"""TODO: make basic, advanced examples, then skip all the configs setup
& make_model redefinition and just jump into callbacks as in tests?
(and add note to reader to see basic/advanced first)
"""

#### Imports #################################################################
import os
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.models import Model
from deeptrain import TrainGenerator, DataGenerator

#%%# Configuration ###########################################################
# Begin by defining a model maker function.
# Input should specify hyperparameters, optimizer, learning rate, etc.;
# this is the 'blueprint' which is later saved.
def make_model(batch_shape, optimizer, loss, metrics, num_classes,
               filters, kernel_size):
    ipt = Input(batch_shape=batch_shape)

    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(ipt)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)

    out = Activation('softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)
    return model

# Set batch_size and specify MNIST dims (28 x 28 pixel, greyscale)
batch_size = 128
width, height, channels = 28, 28, 1

# Define configs dictionary to feed as **kwargs to `make_model`;
# we'll also pass it to TrainGenerator, which will save it and show in a
# "report" for easy reference
MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=16,
    kernel_size=(3, 3),
    # `activation`, `pool_size`, & others could be set the same way;
    # good idea if we ever plan on changing them.
)
# Configs for (train) DataGenerator
datadir = os.path.join("dir", "data", "image")
DATAGEN_CFG = dict(
    # directory where image data is located
    data_dir=os.path.join(datadir, 'train'),
    # where labels file is located
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    # number of samples to feed at once to model
    batch_size=batch_size,
    # whether to shuffle data at end of each epoch
    shuffle=True,
    # which files to load into a `superbatch`, which holds batches persisently
    # in memory (as opposed to `batch`, which is overwritten after use).
    # Since MNIST is small, we can load it all into RAM.
    superbatch_set_nums='all',
)
# Configs for (validation) DataGenerator
VAL_DATAGEN_CFG = dict(
    data_dir=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
    superbatch_set_nums='all',
)
# Configs for TrainGenerator
TRAINGEN_CFG = dict(
    # number of epochs to train for
    epochs=3,
    # where to save TrainGenerator state, model, report, and history
    logs_dir=os.path.join('dir', 'outputs', 'logs'),
    # where to save model when it achieves new best validation performance
    best_models_dir=os.path.join('dir', 'outputs', 'models'),
    # model configurations dict to save & write to report
    model_configs=MODEL_CFG,
)
#%%# Create training objects ################################################
model       = make_model(**MODEL_CFG)
datagen     = DataGenerator(**DATAGEN_CFG)
val_datagen = DataGenerator(**VAL_DATAGEN_CFG)
traingen    = TrainGenerator(model, datagen, val_datagen, **TRAINGEN_CFG)

#%%# Train ##################################################################
traingen.train()

#%%##############################
cwd = os.getcwd()
print("Checkpoints can be found in", os.path.join(cwd, traingen.logdir))
print("Best model can be found in", os.path.join(cwd, traingen.best_models_dir))
