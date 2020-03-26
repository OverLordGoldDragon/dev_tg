# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import gc

from .introspection import get_grads_fn, compute_gradient_l2norm
from . import NOTE, WARN


def _compute_gradient_l2norm(self, val=True, learning_phase=0, w=1, h=1):
    raise NotImplementedError()  # TODO

    datagen_type = "val" if val else "train"
    print(NOTE, datagen_type + " datagen states will be reset")   
    print("'.' = window processed, '|' = batch processed")
    mode = "train" if learning_phase==1 else "inference"
    print("Computing gradient l2-norm over " + datagen_type + " batches, in "
          + mode + " mode")
    datagen = self.val_datagen if val else self.datagen
    datagen.reset_datagen_states()
    
    grads_fn = get_grads_fn(self.model)
    grad_l2norm = []
    batches_processed = 0
    while not datagen.all_data_exhausted:
        data, labels, sample_weights = self.get_data(val=val)
        
        grad_l2norm += [compute_gradient_l2norm(
            data, labels, sample_weights, learning_phase, grads_fn)]
        datagen.update_datagen_states()
        
        # PROGBAR
        if datagen.batch_exhausted:
            prog_mark = '|'
            batches_processed += 1
            if not (batches_processed % np.ceil(datagen.num_batches / 10)):
                prog_mark += "{:}%".format(100 * batches_processed /
                                           datagen.num_batches)
        else:
            prog_mark = '.'
        print(end=prog_mark)

    print(("\nGRADIENT L2-NORM (AVG, MAX) = ({:.2f}, {:.2f}), computed over {} "
           "batches, {} updates").format(
               grad_l2norm.mean(), grad_l2norm.max(), datagen.num_batches,
               datagen_type, len(grad_l2norm)))
        
    bins = len(grad_l2norm) if len(grad_l2norm) < 600 else 600
    plt.hist(grad_l2norm, bins=bins)
    plt.gcf().set_size_inches(9*w, 4*h)
    
    datagen.reset_datagen_states()
    gc.collect()
    
    return grad_l2norm


#TODO: revamp
def visualize_gradients(cls, on_current_train_batch=True, batch=None,
            labels=None, sample_weights=None, learning_phase=0, 
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
        if batch or labels or sample_weights:
            print(WARN, "batch', 'labels', and 'sample_weights' args"
                 + " will be overridden, since 'on_current_train_batch'=True;"
                 + " 'set_num' will also be advanced")
        cls.datagen.advance_batch()
        batch  = cls.datagen.batch
        labels = cls._labels
        sample_weights = cls.get_sample_weights(labels)
    else:
        if (not batch) or (not labels) or (not sample_weights):
            raise ("Please supply 'batch','labels','sample_weights', "
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
        grad  = np.asarray(grads_fn([data, sample_weights,
                                     labels, learning_phase]))
        
        total_grad = (total_grad + grad) if len(total_grad) else grad
        window += 1
    _histogram_grid(total_grad, **kwargs)  
    
    gc.collect()
    return total_grad
