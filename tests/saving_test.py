# -*- coding: utf-8 -*-
import os
import sys
import inspect
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(inspect.stack()[0][1])
if sys.path[0] != filedir:
    if filedir in sys.path:
        sys.path.pop(sys.path.index(filedir))  # avoid dudplication
    sys.path.insert(0, filedir)

import pytest
import numpy as np

from copy import deepcopy
from pathlib import Path

from backend import K, BASEDIR, tempdir, notify, make_classifier, load_model
from backend import _init_session, _get_test_names


#### CONFIGURE TESTING #######################################################
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
    dynamic_predict_threshold_min_max=(.35, .95),
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {}
classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)
###############################################################################

@notify(tests_done)
def test_model_save():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = init_session(C, model=classifier)

        if 'model:weights' in tg.saveskip_list:
            tg.saveskip_list.pop(tg.saveskip_list.index('model:weights'))
        if 'model' not in tg.saveskip_list:
            tg.saveskip_list.append('model')

        tg.train()
        _validate_save_load(tg, C)


@notify(tests_done)
def test_model_save_weights():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = init_session(C, model=classifier)

        if 'model' in tg.saveskip_list:
            tg.saveskip_list.pop(tg.saveskip_list.index('model'))
        if 'model:weights' not in tg.saveskip_list:
            tg.saveskip_list.append('model:weights')

        tg.train()
        _validate_save_load(tg, C)


def _validate_save_load(tg, C):
    def _get_load_path(tg, logdir):
        for postfix in ('weights', 'model', 'model_noopt'):
            postfix += '.h5'
            path = [str(p) for p in Path(logdir).iterdir()
                    if p.name.endswith(postfix)]
            if path:
                return path[0]
        raise Exception(f"no model save file found in {logdir}")

    # get behavior before saving, to ensure no changes presave-to-postload
    data = np.random.randn(*tg.model.input_shape)

    Wm_save = tg.model.get_weights()
    Wo_save = K.batch_get_value(tg.model.optimizer.weights)
    preds_save = tg.model.predict(data, batch_size=len(data))

    tg.checkpoint()
    logdir = tg.logdir
    tg.destroy(confirm=True)

    C['traingen']['logdir'] = logdir
    path = _get_load_path(tg, logdir)
    if path.endswith('weights.h5'):
        model = make_classifier(**C['model'])
        model.load_weights(path)
    else:
        model = load_model(path)
    tg = init_session(C, model=model)
    tg.load()

    Wm_load = tg.model.get_weights()
    Wo_load = K.batch_get_value(tg.model.optimizer.weights)
    preds_load = tg.model.predict(data, batch_size=len(data))

    for s, l in zip(Wm_save, Wm_load):
        assert np.allclose(s, l), "max absdiff: %s" % np.max(np.abs(s - l))
    for s, l in zip(Wo_save, Wo_load):
        assert np.allclose(s, l), "max absdiff: %s" % np.max(np.abs(s - l))
    assert np.allclose(preds_save, preds_load), (
         "max absdiff: %s" % np.max(np.abs(preds_save - preds_load)))


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
