# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from termcolor import cprint
from see_rnn import get_gradients, detect_nans
from .util._backend import K, WARN


def compute_gradients_norm(model, input_data, labels, sample_weight=None,
                           learning_phase=0, mode='weights', norm_fn=np.square):
    grads = get_gradients(model, '*', input_data, labels, sample_weight,
                          learning_phase, mode=mode)
    return np.sqrt(np.sum([np.sum(norm_fn(g)) for g in grads]))


def print_dead_weights(model, dead_threshold=1e-7, notify_above_frac=1e-3,
                       notify_detected_only=False):
    def _print_dead(frac_dead, w_name, notify_above_frac):
        precision = int(np.ceil(-np.log10(notify_above_frac)))
        perc_dead = f'%.{precision}f' % (100 * frac_dead) + '%'

        cprint("{} dead -- '{}')".format(w_name, perc_dead), 'red')

    weight_names   = [w.name for layer in model.layers for w in layer.weights]
    weight_tensors = [w for layer in model.layers for w in layer.weights]
    weight_values  = K.batch_get_value(weight_tensors)

    has_dead_worth_notifying = False
    has_dead = False
    for w_name, w_value in zip(weight_names, weight_values):
        num_dead = np.sum(np.abs(w_value) < dead_threshold)
        if num_dead > 0:
            has_dead = True

        frac_dead = num_dead / w_value.size
        if frac_dead > notify_above_frac:
            has_dead_worth_notifying = True
            _print_dead(frac_dead, w_name, notify_above_frac)

    if has_dead_worth_notifying:
        print("L = layer index, W = weight matrix index")
    elif not notify_detected_only:
        if has_dead:
            _txt = "Dead weights detected, but didn't notify; "
        else:
            _txt = "No dead weights detected in any trainable layers; "
        print(_txt + "(dead_threshold, notify_above_frac) = ({}, {})".format(
            dead_threshold, notify_above_frac))


def print_nan_weights(model, notify_detected_only=False):
    weight_names   = [w.name for layer in model.layers for w in layer.weights]
    weight_tensors = [w for layer in model.layers for w in layer.weights]
    weight_values  = K.batch_get_value(weight_tensors)

    has_nan = False
    for w_name, w_value in zip(weight_names, weight_values):
        num_nan = np.sum(np.isnan(w_value))
        if num_nan > 0:
            has_nan = True
        txt = detect_nans(w_value)
        if txt:
            cprint("\n{} -- '{}'".format(txt, w_name), color='red')

    if has_nan:
        print("L = layer index, W = weight matrix index", end='')
    elif not notify_detected_only:
        print("No NaN weights detected in any trainable layers")


def gradient_norm_over_dataset(cls, val=False, learning_phase=0, mode='weights',
                               w=1, h=1, norm_fn=np.square):
    def _init_notify(dg_name, learning_phase):
        print(WARN, dg_name, "states will be reset")
        print("'.' = slice processed, '|' = batch processed")

        mode = "train" if learning_phase == 1 else "inference"
        print("Computing gradient l2-norm over", dg_name, "batches, in",
              mode, "mode")

    def _print_results(grad_norms, batches_processed, dg_name):
        print(("\nGRADIENT L2-NORM (AVG, MAX) = ({:.2f}, {:.2f}), computed over "
        	   "{} batches, {} {} updates").format(
        		   grad_norms.mean(), grad_norms.max(), batches_processed,
        		   dg_name, len(grad_norms)))

    def _plot_hist(grad_norms, w=1, h=1):
        bins = min(600, len(grad_norms))
        plt.hist(grad_norms.ravel(), bins=bins)
        plt.gcf().set_size_inches(9 * w, 4 * h)

    def _print_progress(dg, batches_processed):
        if not dg.batch_exhausted:
            prog_mark = '.'
        else:
            prog_mark = '|'
            batches_processed += 1
            if batches_processed % 10 == 0:
                prog_mark += str(batches_processed)
        print(end=prog_mark)
        return batches_processed

    def _gather_grad_norms(val):
        dg.all_data_exhausted = False
        grad_norms = []
        batches_processed = 0

        while not dg.all_data_exhausted:
            dg.advance_batch()
            x, y, sw = cls.get_data(val=val)
            grad_norms += [compute_gradients_norm(
                cls.model, x, y, sw, learning_phase, mode=mode, norm_fn=norm_fn)]
            dg.update_state()
            batches_processed = _print_progress(dg, batches_processed)
        return grad_norms, batches_processed

    dg_name = "val_datagen" if val else "datagen"
    dg = cls.val_datagen if val else cls.datagen
    _init_notify(dg_name, learning_phase)

    dg.reset_state()
    grad_norms, batches_processed = _gather_grad_norms(val)
    dg.reset_state()

    grad_norms = np.array(grad_norms)
    _print_results(grad_norms, batches_processed, dg_name)
    _plot_hist(grad_norms)

    return grad_norms
