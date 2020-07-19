# -*- coding: utf-8 -*-
"""This example assumes you've read `callbacks/basic.py`, and covers:
    - Creating advanced custom callbacks
    - Using and modifying builtin callbacks
    - Visualization, data gathering, and random seed setting callbacks
"""
import os
os.environ['TF_KERAS'] = '0'
import sys
from pathlib import Path
filedir = str(Path(Path(__file__).parents[1], "dir"))
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

from utils import make_classifier, init_session, img_labels_paths
from utils import CL_CONFIGS as C
from see_rnn import features_2D
from deeptrain.callbacks import TraingenCallback, TraingenLogger
from deeptrain.callbacks import RandomSeedSetter
from deeptrain.callbacks import make_layer_hists_cb

#%%# CONFIGURE TESTING #######################################################
batch_size, width, height, channels = C['model']['batch_shape']
logger_savedir = os.path.join(filedir, "outputs", "logger")
#%%#
# TraingenLogger gathers data throughout training: weights, outputs, and
# gradients of model layers. We inherit the base class and override
# methods where we wish actions to occur: on save, load, and end of train epoch.
class TraingenLoggerCB(TraingenLogger):
    def __init__(self, savedir, configs, **kwargs):
        super().__init__(savedir, configs, **kwargs)

    def on_save(self, stage=None):
        self.save(_id=self.tg.epoch)  # `tg` will be set inside TrainGenerator

    def on_load(self, stage=None):
        self.clear()
        self.load()

    def on_train_epoch_end(self, stage=None):
        self.log()

log_configs = {
    'weights': ['conv2d'],
    'outputs': 'conv2d',
    'gradients': ('conv2d',),
    'outputs-kw': dict(learning_phase=0),
    'gradients-kw': dict(learning_phase=0),
}
tglogger = TraingenLoggerCB(logger_savedir, log_configs)

#%%#
# Plots model outputs in a heatmap at end of each epoch.
# Relies on `TraingenLogger` being included in `callbacks`, which stores
# model outputs so they aren't recomputed for visualization.
# All callback objects (except funcs in dicts) are required to subclass
# TraingenCallback (TraingenLogger does so)
class Viz2D(TraingenCallback):
    def on_val_end(self, stage=None):
        if stage == ('val_end', 'train:epoch') and (self.tg.epoch % 2) == 0:
            # run `viz` within `TrainGenerator._on_val_end`,
            # and on every other epoch
            self.viz()

    def viz(self):
        data = self._get_data()
        features_2D(data, tight=True, title_mode=False, cmap='hot',
                    norm=None, show_xy_ticks=[0, 0], w=1.1, h=.55, n_rows=4)

    def _get_data(self):
        lg = None
        for cb in self.tg.callbacks:
            if isinstance(cb, TraingenLogger):
                lg = cb
        if lg is None:
            raise Exception("TraingenLogger not found in `callbacks`")

        last_key = list(lg.outputs.keys())[-1]
        outs = list(lg.outputs[last_key][0].values())[0]
        sample = outs[0]                  # (width, height, channels)
        return sample.transpose(2, 0, 1)  # (channels, width, height)

viz2d = Viz2D()

#%%#
# Callbacks can also be configured as str-function dict pairs, where str
# is name of a callback "stage" (see tg._cb_alias after tg.train()).
grad_hists = {'train:epoch': [make_layer_hists_cb(mode='gradients:outputs'),
                              make_layer_hists_cb(mode='gradients:weights')]}
weight_hists = {('val_end', 'train:epoch'): make_layer_hists_cb(mode='weights')}

configs = {'title': dict(fontsize=13), 'plot': dict(annot_kw=None)}
outputs_hists = {'val_end': make_layer_hists_cb(mode='outputs', configs=configs)}
#%%#
# Set new random seeds (`random`, `numpy`, TF-graph, TF-global) every epoch,
# incrementing by 1 from start value (default 0)
seed_setter = RandomSeedSetter(freq={'train:epoch': 2})
#%%###########################################################################
C['traingen']['callbacks'] = [seed_setter, tglogger, viz2d,
                              grad_hists, weight_hists, outputs_hists]
C['traingen']['epochs'] = 4
C['datagen']['labels_path']     = img_labels_paths[0]
C['val_datagen']['labels_path'] = img_labels_paths[1]
tg = init_session(C, make_classifier)
#%%
tg.train()
