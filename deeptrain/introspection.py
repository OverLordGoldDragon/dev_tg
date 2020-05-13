# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from termcolor import cprint
from see_rnn import get_gradients, features_hist, detect_nans
from see_rnn import get_layer
from see_rnn.inspect_gen import _make_grads_fn, _get_grads
from .util._backend import K, WARN


def compute_gradients_norm(self, input_data, labels, sample_weight=None,
                           learning_phase=0, mode='weights', norm_fn=np.square):
    grads = get_gradients(self.model, '*', input_data, labels, sample_weight,
                          learning_phase, mode=mode, as_dict=False)
    return np.sqrt(np.sum([np.sum(norm_fn(g)) for g in grads]))


def gradient_norm_over_dataset(self, val=False, learning_phase=0, mode='weights',
                               norm_fn=np.square, w=1, h=1, n_iters=None,
                               prog_freq=10):
    def _init_notify(learning_phase, val):
        dg_name = 'val_datagen' if val else 'datagen'
        mode_name = "train" if learning_phase == 1 else "inference"
        print("Computing gradient l2-norm over", dg_name, "batches, in",
              mode_name, "mode")

    def _print_results(grad_norms, batches_processed, iters_processed, val):
        dg_name = 'val_datagen' if val else 'datagen'
        print(("\nGRADIENT L2-NORM (AVG, MAX) = ({:.2f}, {:.2f}), computed over "
        	   "{} batches, {} {} updates").format(
        		   grad_norms.mean(), grad_norms.max(), batches_processed,
        		   iters_processed, dg_name))

    def _plot(grad_norms, w=1, h=1):
        bins = min(600, len(grad_norms))
        plt.hist(grad_norms.ravel(), bins=bins)
        plt.gcf().set_size_inches(9 * w, 4 * h)
        plt.show()

    def _make_gradients_fn(model, learning_phase, mode):
        # make grads_fn only once instead of repeatedly calling `get_gradients`
        # for potentially massive speedup due to not rebuilding graph
        attr = 'output' if mode == 'outputs' else 'trainable_weights'
        _id = []
        for l in model.layers[1:]:  # exclude input
            params = getattr(l, attr, [])
            if (isinstance(params, list) and len(params) > 0) or (
                    not isinstance(params, list)):
                _id.append(l.name)
        layers = get_layer(model, _id)
        _grads_fn, names = _make_grads_fn(model, layers, mode=mode,
                                         return_names=True)
        return lambda x, y, sw: _get_grads(_grads_fn, x, y, sw, learning_phase)

    def _compute_gradients_norm(model, x, y, sw):
        grads = grads_fn(x, y, sw)
        return np.sqrt(np.sum([np.sum(norm_fn(g)) for g in grads]))

    def gather_fn(data, model, x, y, sw):
        newdata = _compute_gradients_norm(model, x, y, sw)
        data.append(newdata)
        return data

    _init_notify(learning_phase, val)
    grads_fn = _make_gradients_fn(self.model, learning_phase, mode)

    grad_norms, batches_processed, iters_processed = _gather_over_dataset(
        self, gather_fn, val, n_iters, prog_freq)
    grad_norms = np.array(grad_norms)

    _print_results(grad_norms, batches_processed, iters_processed, val)
    _plot(grad_norms, w, h)

    return grad_norms, batches_processed, iters_processed


def gradients_sum_over_dataset(self, val=False, learning_phase=0, mode='weights',
                               n_iters=None, prog_freq=10, plot_kw={}):
    def _init_notify(learning_phase, val):
        dg_name = 'val_datagen' if val else 'datagen'
        mode_name = "train" if learning_phase == 1 else "inference"
        print("Computing gradients sum over", dg_name, "batches, in",
              mode_name, "mode")

    def _print_results(grads_sum, batches_processed, iters_processed, val):
        dg_name = 'val_datagen' if val else 'datagen'
        print(("\nGRADIENTS SUM computed over {} batches, {} {} updates:").format(
            batches_processed, iters_processed, dg_name))

    def _plot(grads_sum, plot_kw):
        defaults = {'share_xy': False, 'center_zero': True}
        for k, v in defaults.items():
            if k not in plot_kw:
                plot_kw[k] = v
        data = list(grads_sum.values())
        features_hist(data, annotations=list(grads_sum), **plot_kw)

    def _make_gradients_fn(model, learning_phase, mode):
        # make grads_fn only once instead of repeatedly calling `get_gradients`
        # for potentially massive speedup due to not rebuilding graph
        attr = 'output' if mode == 'outputs' else 'trainable_weights'
        _id = []
        for l in model.layers[1:]:  # exclude input
            params = getattr(l, attr, [])
            if (isinstance(params, list) and len(params) > 0) or (
                    not isinstance(params, list)):
                _id.append(l.name)
        layers = get_layer(model, _id)
        _grads_fn, names = _make_grads_fn(model, layers, mode=mode,
                                         return_names=True)
        def grads_fn(x, y, sw):
            grads = _get_grads(_grads_fn, x, y, sw, learning_phase)
            return {name: x for name, x in zip(names, grads)}
        return grads_fn

    def gather_fn(data, model, x, y, sw):
        newdata = grads_fn(x, y, sw)

        if not data:
            return newdata
        for k, v in newdata.items():
            for i, x in enumerate(v):
                data[k][i] += x
        return data

    _init_notify(learning_phase, val)
    grads_fn = _make_gradients_fn(self.model, learning_phase, mode)

    grads_sum, batches_processed, iters_processed = _gather_over_dataset(
        self, gather_fn, val, n_iters, prog_freq)

    _print_results(grads_sum, batches_processed, iters_processed, val)
    _plot(grads_sum, plot_kw)

    return grads_sum, batches_processed, iters_processed


def _gather_over_dataset(self, gather_fn, val=False, n_iters=None, prog_freq=10):
    def _init_notify(val):
        dg_name = "val_datagen" if val else "datagen"
        print(WARN, dg_name, "states will be reset")
        print("'.' = slice processed, '|' = batch processed")

    def _print_progress(dg, batches_processed, prog_freq):
        if not dg.batch_exhausted:
            prog_mark = '.'
        else:
            prog_mark = '|'
            batches_processed += 1
            if batches_processed % max(1, prog_freq) == 0:
                prog_mark += str(batches_processed)
        print(end=prog_mark)
        return batches_processed

    def _gather(gather_fn, n_iters, val):
        def cond(iters_processed, n_iters, dg):
            if n_iters is None:
                return not dg.all_data_exhausted
            return iters_processed < n_iters

        dg.all_data_exhausted = False
        gathered = []
        batches_processed = 0
        iters_processed = 0

        while cond(iters_processed, n_iters, dg):
            dg.advance_batch()
            x, y, sw = self.get_data(val=val)
            gathered = gather_fn(gathered, self.model, x, y, sw)
            dg.update_state()
            batches_processed = _print_progress(dg, batches_processed, prog_freq)
            iters_processed += 1
        return gathered, batches_processed, iters_processed

    dg = self.val_datagen if val else self.datagen
    _init_notify(val)

    dg.reset_state()
    gathered, batches_processed, iters_processed = _gather(gather_fn,
                                                           n_iters, val)
    dg.reset_state()

    return gathered, batches_processed, iters_processed


def print_dead_weights(model, dead_threshold=1e-7, notify_above_frac=1e-3,
                       notify_detected_only=False):
    def _print_dead(frac_dead, w_name, notify_above_frac):
        precision = int(np.ceil(-np.log10(notify_above_frac)))
        perc_dead = f'%.{precision}f' % (100 * frac_dead) + '%'

        cprint("{} dead -- '{}'".format(perc_dead, w_name), 'red')

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
        num_nan = np.sum(np.isnan(w_value) + np.isinf(w_value))
        txt = detect_nans(w_value)
        if txt:
            if not has_nan:
                print(flush=True)  # newline

            cprint("{} -- '{}'".format(txt, w_name), color='red', flush=True)
        if num_nan > 0:
            has_nan = True

    if has_nan:
        print("L = layer index, W = weight matrix index", end='')
    elif not notify_detected_only:
        print("No NaN weights detected in any trainable layers")
