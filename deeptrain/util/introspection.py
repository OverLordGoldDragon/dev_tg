# -*- coding: utf-8 -*-
import numpy as np
from termcolor import cprint
from . import K


def get_grads_fn(model):
    grad_tensors = model.optimizer.get_gradients(model.total_loss,
                                                 model.trainable_weights)
    return K.function(inputs=[model.inputs[0],  model.sample_weight[0],
                              model.targets[0], K.learning_phase()],
                      outputs=grad_tensors)

def compute_gradient_l2norm(input_data, labels, sample_weight, learning_phase=0,
                            grads_fn=None, model=None):
    if grads_fn is None:
        if model is None:
            raise Exception("Supply at least one of 'get_gradients_function' "
                            "or 'model'")
        grads_fn = get_grads_fn(model)
    gradients = grads_fn([input_data, sample_weight, labels, learning_phase])
    return np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))

# TODO: revamp
def print_dead_weights(model, dead_threshold=1e-7, notify_above_frac=1e-3,
                       verbose_notify_only=False):
    trainable_idxs_and_layers = _get_trainable_layers(
        model, include_indices=True)
    has_dead = False
    has_dead_worth_notifying = False
        
    for layer_idx, layer in trainable_idxs_and_layers:
        for weight_idx, weights in enumerate(layer.get_weights()):
            num_dead = np.sum(np.abs(weights) < dead_threshold)
            if num_dead != 0:
                has_dead = True
                frac_dead = num_dead / weights.size
                if frac_dead > notify_above_frac:
                    has_dead_worth_notifying = True
                    decim_to_show = int(np.ceil(-np.log10(notify_above_frac)))
                    perc_dead = f'%.{decim_to_show}f' % (100 * frac_dead) + '%'
                    
                    cprint("L{} W{} {} dead ('{}')".format(
                        layer_idx, weight_idx, perc_dead, layer.name), 'red')
    
    if has_dead_worth_notifying:
        print("L = layer index, W = weight matrix index")
    elif not verbose_notify_only:
        if has_dead:
            _txt = "Dead weights detected, but didn't notify; "
        else:
            _txt = "No dead weights detected in any trainable layers; "
        print(_txt + "(dead_threshold, notify_above_frac) = ({}, {})".format(
            dead_threshold, notify_above_frac))
                    
# TODO: revamp
def print_nan_weights(model, verbose_notify_only=False):

    trainable_idxs_and_layers = _get_trainable_layers(
        model, include_indices=True)
    has_nan = False
    
    for layer_idx, layer in trainable_idxs_and_layers:
        for weight_idx, weights in enumerate(layer.get_weights()):
            num_nan = np.sum(np.isnan(weights))
            if num_nan:
                has_nan = True
                frac_nan = num_nan / weights.size
                if frac_nan > 0.1: 
                    nan_txt = "%.1f " % (100*frac_nan) + "%"
                else:              
                    nan_txt = str(num_nan)
                cprint('\nL' + str(layer_idx) + 
                       ' W'  + str(weight_idx) + ' ' + nan_txt + ' NaN '
                       + "('" + layer.name + "')", color='red')
    if has_nan:
        print("L = layer index, W = weight matrix index", end='')
    elif not verbose_notify_only:
        print("No NaN weights detected in any trainable layers")
    
def _get_trainable_layers(model, include_names=False, include_indices=False):
    return [( [idx] * int(include_indices) 
            + [layer.name] * int(include_names)
            + [layer] ) for (idx,layer) in enumerate(model.layers) 
                        if  layer._trainable_weights != []]


def l1l2_weight_loss(model): ## TODO: RNN vs TimeDistributed conflict
    l1l2_loss = 0
    for layer in model.layers:
        if hasattr(layer, 'layer') or hasattr(layer, 'cell'):
            if hasattr(layer, 'layer') and 'dense' in str(layer.layer).lower():
                layer = layer.layer
            else:
                l1l2_loss += _l1l2_rnn_loss(layer)
                continue
            
        if 'kernel_regularizer' in layer.__dict__ or \
           'bias_regularizer'   in layer.__dict__:
            l1l2_lambda_k, l1l2_lambda_b = [0,0], [0,0] # defaults
            if layer.__dict__['kernel_regularizer'] is not None:
                l1l2_lambda_k = list(layer.kernel_regularizer.__dict__.values())
            if layer.__dict__['bias_regularizer']   is not None:
                l1l2_lambda_b = list(layer.bias_regularizer.__dict__.values())
                
            if any([(_lambda != 0) for _lambda in (
                    l1l2_lambda_k + l1l2_lambda_b)]):
                W = layer.get_weights()
    
                for idx, _lambda in enumerate(l1l2_lambda_k + l1l2_lambda_b):
                    if _lambda != 0:
                        _pow = 2**(idx % 2) # 1 if idx is even (l1), 2 if odd (l2)
                        l1l2_loss += _lambda*np.sum(np.abs(W[idx//2])**_pow)
    return l1l2_loss

def _l1l2_rnn_loss(layer):
    def _cell_loss(cell):
        cell_loss = 0
        if any([hasattr(cell, f'{name}_regularizer') 
                for name in ('kernel', 'recurrent', 'bias')]):
            l1l2_lambda_k, l1l2_lambda_r, l1l2_lambda_b = [0,0], [0,0], [0,0]
            if getattr(cell, 'kernel_regularizer', None) is not None:
                l1l2_lambda_k = list(cell.kernel_regularizer.__dict__.values())
            if getattr(cell, 'recurrent_regularizer', None) is not None:
                l1l2_lambda_r = list(cell.recurrent_regularizer.__dict__.values())
            if getattr(cell, 'bias_regularizer', None) is not None:
                l1l2_lambda_b = list(cell.bias_regularizer.__dict__.values())

            all_lambda = l1l2_lambda_k + l1l2_lambda_r + l1l2_lambda_b
            if any([(_lambda != 0) for _lambda in all_lambda]):
                W = layer.get_weights()
                idx_incr = len(W)//2 # accounts for 'use_bias'
                
                for idx, _lambda in enumerate(all_lambda):
                    if _lambda != 0:
                        _pow = 2**(idx % 2) # 1 if idx is even (l1), 2 if odd (l2)
                        cell_loss += _lambda*np.sum(np.abs(W[idx//2])**_pow)
                        if IS_BIDIR:
                            cell_loss += _lambda*np.sum(
                                        np.abs(W[idx//2 + idx_incr])**_pow)
        return cell_loss

    if hasattr(layer, 'backward_layer'):
        rnn_type = type(layer.layer).__name__
        IS_BIDIR = True
    else:
        rnn_type = type(layer).__name__
        IS_BIDIR = False
    IS_CUDNN = 'CuDNN' in rnn_type

    if IS_BIDIR:
        l = layer
        forward_cell  = l.forward_layer  if IS_CUDNN else l.forward_layer.cell
        backward_cell = l.backward_layer if IS_CUDNN else l.backward_layer.cell
        return (_cell_loss(forward_cell) +
                _cell_loss(backward_cell))

    cell = layer if IS_CUDNN else layer.cell
    return _cell_loss(cell)

def _rnn_l2_regs(layer):
    _layer = layer.layer if 'backward_layer' in layer.__dict__ else layer
    cell = _layer.cell
    l2_lambda_krb = [None, None, None]
    
    if hasattr(cell, 'kernel_regularizer')    or \
       hasattr(cell, 'recurrent_regularizer') or hasattr(
           cell, 'bias_regularizer'):
        l2_lambda_krb = [getattr(cell, name + '_regularizer', None) for 
                                       name in ['kernel','recurrent','bias']]  
    return [(_lambda.l2 if _lambda is not None else 0) for 
                                        _lambda in l2_lambda_krb]
