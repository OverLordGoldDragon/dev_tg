# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import gc

from ..introspection import get_grads_fn
from ._backend import WARN


#TODO: revamp
# wontdo
def visualize_gradients(cls, on_current_train_batch=True, batch=None,
            labels=None, sample_weight=None, learning_phase=0,
            slide_size=None, **kwargs):
    raise NotImplementedError()  ## TODO

    def _histogram_grid(data, num_cols=4, h=1, w=1, bins=200):
        for idx, entry in enumerate(data):
            if not idx%num_cols:
                plt.show()
                plt.figure(figsize=(14*w,3*h))
            plt.subplot(1,num_cols,idx%num_cols+1)
            _ = plt.hist(np.ndarray.flatten(entry),bins=bins)
        plt.show()

    slide_size = slide_size or cls.timesteps
    if on_current_train_batch:
        if batch or labels or sample_weight:
            print(WARN, "batch', 'labels', and 'sample_weight' args"
                 + " will be overridden, since 'on_current_train_batch'=True;"
                 + " 'set_num' will also be advanced")
        cls.datagen.advance_batch()
        batch  = cls.datagen.batch
        labels = cls._labels
        sample_weight = cls.get_sample_weight(labels)
    else:
        if (not batch) or (not labels) or (not sample_weight):
            raise ("Please supply 'batch','labels','sample_weight', "
                   + "or set 'on_current_train_batch'=True")

    mode = "train" if learning_phase==1 else "inference"
    print( "Visualizing gradients in " + mode + " mode")
    grads_fn = get_grads_fn(cls.model)

    window = 0
    total_grad = []
    while window < (batch.shape[1] / cls.timesteps - 1):
        start = cls.timesteps * window
        end   = start + slide_size
        data  = batch[:, start:end, :]
        grad  = np.asarray(grads_fn([data, sample_weight,
                                     labels, learning_phase]))

        total_grad = (total_grad + grad) if len(total_grad) else grad
        window += 1
    _histogram_grid(total_grad, **kwargs)

    gc.collect()
    return total_grad
