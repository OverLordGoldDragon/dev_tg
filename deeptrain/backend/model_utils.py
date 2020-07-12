from types import LambdaType
from . import TF_KERAS, TF_EAGER, TF_2


def get_model_metrics(model):
    # TF1, 2, Eager, Graph, keras, and tf.keras store model.compile(metrics)
    # differently
    if TF_2 and TF_KERAS:
        if TF_EAGER:
            metrics = model.compiled_metrics._user_metrics
        else:
            metrics = model._compile_metrics
    else:
        metrics = model.metrics_names

    if metrics and 'loss' in metrics:
        metrics.pop(metrics.index('loss'))
    return ['loss', *metrics] if metrics else ['loss']


def model_loss_name(model):
    if not hasattr(model, 'loss'):
        raise AttributeError("`model` has no attribute 'loss'; did you run "
                             "`model.compile()`?")
    loss = model.loss
    if isinstance(loss, str):
        return loss
    elif isinstance(loss, LambdaType):
        return loss.__name__
    else:
        raise Exception("unable to get name of `model` loss")

# TF2 tf.keras Eager (Graph is same)
"""
_add_unique_metric_name
_cache_output_metric_attributes
_compile_metric_functions
_compile_metrics
_compile_weighted_metrics
_get_existing_metric
_get_training_eval_metrics
_handle_metrics
_handle_per_output_metrics
_metrics
_output_loss_metrics
_per_output_metrics
_per_output_weighted_metrics
_set_metric_attributes
_set_per_output_metric_attributes
add_metric
metrics
metrics_names
reset_metrics
"""
# TF1 tf.keras Graph  (keras is same)
"""
metrics
metrics_names  -- ['loss', 'acc']
metrics_tensors
metrics_updates
stateful_metric_functions
stateful_metric_names
weighted_metrics
"""