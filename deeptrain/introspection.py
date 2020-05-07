# -*- coding: utf-8 -*-
import numpy as np
from termcolor import cprint
from see_rnn import get_gradients, detect_nans
from .util._backend import K


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

    has_dead = False
    has_dead_worth_notifying = False

    weight_names   = [w.name for layer in model.layers for w in layer.weights]
    weight_tensors = [w for layer in model.layers for w in layer.weights]
    weight_values  = K.batch_get_value(weight_tensors)

    for w_name, w_value in zip(weight_names, weight_values):
        num_dead = np.sum(np.abs(w_value) < dead_threshold)
        if num_dead > 0:
            has_dead = True

        frac_dead = num_dead / w_value.size
        if frac_dead > notify_above_frac:
            has_dead_worth_notifying = True
            _print_dead(frac_dead, w_value.size, w_name)

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
    has_nan = False

    weight_names   = [w.name for layer in model.layers for w in layer.weights]
    weight_tensors = [w for layer in model.layers for w in layer.weights]
    weight_values  = K.batch_get_value(weight_tensors)

    for w_name, w_value in zip(weight_names, weight_values):
        num_nan = np.sum(np.isnan(w_value))
        if num_nan > 0:
            has_nan = True
        txt = detect_nans(w_value)
        cprint("\n{} -- '{}'".format(txt, w_name), color='red')

    if has_nan:
        print("L = layer index, W = weight matrix index", end='')
    elif not notify_detected_only:
        print("No NaN weights detected in any trainable layers")
