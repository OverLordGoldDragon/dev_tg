from . import K, TF_KERAS, TF_EAGER, TF_2


def get_model_metrics(model):
    for attr in ('metrics_names', '_compile_metrics'):
        metrics = getattr(model, attr, None)
        if metrics and all(isinstance(m, str) for m in metrics):
            return metrics if 'loss' in metrics else ['loss', *metrics]

    return ['loss', *model.compiled_metrics._metrics]


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