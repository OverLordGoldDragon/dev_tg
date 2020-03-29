# -*- coding: utf-8 -*-
import numpy as np


def pass_on_error(fn, *args, **kwargs):
    fail_msg = kwargs.pop('fail_msg', None)
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        print(fail_msg)
        print("Errmsg:", e)


def _dict_filter_keys(dc, keys, exclude=True, filter_substr=False):
    def condition(k, keys, exclude, filter_substr):
        if not filter_substr:
            value = k in keys
        else:
            value = any([(key in k) for key in keys])
        return (not value) if exclude else value

    keys = keys if isinstance(keys, (list, tuple)) else [keys]
    return {k:v for k,v in dc.items() 
            if condition(k, keys, exclude, filter_substr)}


def ordered_shuffle(*args):
    zipped_args = list(zip(*(a.items() if isinstance(a, dict) 
                             else a for a in args)))
    np.random.shuffle(zipped_args)
    return [(_type(data) if _type != np.ndarray else np.asarray(data)) 
            for _type, data in zip(map(type, args), zip(*zipped_args))]


# TODO: improve case coverage
def _train_on_batch_dummy(model, class_weights={'0':1,'1':6.5},
                          input_as_labels=False):
    """Instantiates trainer & optimizer, but does NOT train (update weights)"""
    def _make_toy_inputs(model):
        return np.random.randn(*model.input_shape)
        
    def _make_toy_labels(model):
        loss = model.loss
        shape = model.output_shape

        if loss == 'binary_crossentropy':
            return np.random.randint(0, 1, shape)
        elif loss == 'categorical_crossentropy':
            n_classes = shape[-1]
            class_labels = np.random.randint(0, n_classes, shape[0])
            return np.eye(n_classes)[class_labels]
        elif loss == 'sparse_categorical_crossentropy':
            return np.random.randint(0, shape[-1], shape[0])
        elif loss == 'mse':
            return np.random.randn(*shape)
        else:
            raise ValueError("unsupported loss: '{}'".format(loss))
    
    def _make_sample_weight(toy_labels, class_weights):
        if class_weights is not None:
            return np.array([class_weights[str(l)] for l in toy_labels])
        else:
            return np.ones(toy_labels.shape[0])

    toy_inputs = _make_toy_inputs(model)
    toy_labels = _make_toy_labels(model)
    toy_sample_weight = _make_sample_weight(toy_labels, class_weights)
    if input_as_labels:
        toy_labels = toy_inputs

    model._standardize_user_data(toy_inputs, toy_labels, toy_sample_weight)
    model._make_train_function()
