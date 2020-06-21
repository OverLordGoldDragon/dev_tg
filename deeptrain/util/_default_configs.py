# -*- coding: utf-8 -*-
"""**! DO NOT MODIFY !**
Used internally by classes to validate input arguments.
Effective configurations are in configs.py. Can serve as user reference.
"""
import numpy as np
from .fonts import fontsdir
from .algorithms import builtin_or_npscalar


#:
_DEFAULT_PLOT_CFG = [
{
 'metrics': None,
 'x_ticks': None,
 'vhlines'   :
     {'v': '_hist_vlines',
      'h': 1},
 'mark_best_cfg': None,
 'ylims'        : (0, 2),
 'legend_kw'    : {'fontsize': 13},

 'linewidth': [1.5, 1.5],
 'linestyle': ['-', '-'],
 'color'    : None,
},
{
 'metrics': None,
 'x_ticks': None,
 'vhlines':
     {'v': '_val_hist_vlines',
      'h': .5},
 'mark_best_cfg': None,
 'ylims'        : (0, 1),
 'legend_kw'    : {'fontsize': 13},

 'linewidth': [1.5],
 'linestyle': ['-'],
 'color': None,
}
]


#: order-dependent
_DEFAULT_MODEL_NAME_CFG = dict(
    optimizer       = '',
    lr              = '',
    best_key_metric = '_max',
)


#:
_DEFAULT_REPORT_CFG = {
    'model':
        {},
    'traingen':
        {
        'exclude':
            ['model', 'model_configs', 'logs_use_full_model_name',
             'history_fig', 'plot_configs', 'max_checkpoints',
             'history', 'val_history', 'temp_history', 'val_temp_history',
             'name_process_key_fn', 'report_fontpath', 'model_name_configs',
             'report_configs', 'datagen', 'val_datagen', 'logdir', 'logs_dir',
             'best_models_dir', 'fit_fn', 'eval_fn', 'callbacks',
             '_history_fig', 'metric_printskip_configs', '_inferred_batch_size',
             'plot_first_pane_max_vals', '_imports', 'iter_verbosity',
             '_max_set_name_chars', '_val_max_set_name_chars',
             'metric_to_alias', 'alias_to_metric',
             '*_has_', '*temp_history_empty',
             ],
        'exclude_types':
            [list, np.ndarray, '#best_subset_nums'],
        },
    ('datagen', 'val_datagen'):
        {
        'exclude':
            ['batch', 'group_batch', 'labels', 'all_labels',
             'batch_loaded', 'batch_exhausted', 'set_num', 'set_name',
             '_set_names', 'set_nums_original', 'set_nums_to_process',
             'superbatch_set_nums', 'data_loader', 'data_dir', 'labels_path',
             'loadskip_list', '_path_attrs', 'preprocessor',
             '*_ATTRS', '*superbatch', '*_filepaths', '*_filenames']
        },
}

# TODO note which items are "mandatory"
_DEFAULT_TRAINGEN_SAVESKIP_LIST = [
    'model',
    'callbacks',
    'key_metric_fn',
    'custom_metrics',
    'use_passed_dirs_over_loaded',
    'check_model_health',
    'metric_to_alias',
    'alias_to_metric',
    'name_process_key_fn',
    'fit_fn',
    'eval_fn',
    '_fit_fn',
    '_eval_fn',

    '_labels',
    '_preds',
    '_y_true',
    '_y_preds',
    '_labels_cache',
    '_preds_cache',
    '_sw_cache',

    '_imports',
    '_history_fig',
    '_fit_iters',
    '_val_iters',
    '_val_max_set_name_chars',
    '_max_set_name_chars',
    '_inferred_batch_size',
    '_class_labels_cache',
    '_train_val_x_ticks',
    '_val_train_x_ticks',
    '_temp_history_empty',
    '_val_temp_history_empty',
    '_val_sw',
    '_set_num',
    '_val_set_num',
]

_DEFAULT_TRAINGEN_LOADSKIP_LIST = ['{auto}', 'model_name', 'model_base_name',
                                   'model_num']

_DEFAULT_DATAGEN_SAVESKIP_LIST = ['batch', 'superbatch', 'labels', 'all_labels',
                                  '_group_batch', '_group_labels']
_DEFAULT_DATAGEN_LOADSKIP_LIST = ['data_dir', 'labels_path', 'superbatch_dir',
                                  'data_loader', 'set_nums_original',
                                  'set_nums_to_process', 'superbatch_set_nums']

_DEFAULT_METRIC_PRINTSKIP_CFG = {
    'train': [],
    'val': [],
}

_DEFAULT_METRIC_TO_ALIAS = {
    'loss'    : 'Loss',
    'accuracy': 'Acc',
    'acc'     : 'Acc',
    'f1-score': 'F1',
    'tnr'     : '0-Acc',
    'tpr'     : '1-Acc',
}

_DEFAULT_ALIAS_TO_METRIC = {
    'acc':     'accuracy',
    'mae':     'mean_absolute_error',
    'mse':     'mean_squared_error',
    'mape':    'mean_absolute_percentage_error',
    'msle':    'mean_squared_logarithmic_error',
    'kld':     'kullback_leibler_divergence',
    'cosine':  'cosine_similarity',
    'f1':      'f1_score',
    'f1-score':'f1_score',
}

#:
def _DEFAULT_NAME_PROCESS_KEY_FN(key, alias, attrs):
    def _format_float(val, small_th=1e-2):
        def _format_small_float(val):
            def _decimal_len(val):
                return len(val.split('.')[1].split('e')[0])

            val = ('%.3e' % val).replace('-0', '-')
            while '0e' in val:
                val = val.replace('0e', 'e')
            if _decimal_len(val) == 0:
                val = val.replace('.', '')
            return val

        if abs(val) < small_th:
            return _format_small_float(val)
        elif small_th < abs(val) < 1:
            return ("%.3f" % val).lstrip('0').rstrip('0')
        else:
            return ("%.3f" % val).rstrip('0')

    def _squash_list(ls):
        def _max_reps_from_beginning(ls, reps=1):
            if reps < len(ls) and ls[reps] == ls[0]:
                reps = _max_reps_from_beginning(ls, reps + 1)
            return reps

        _str = ''
        while len(ls) != 0:
            reps = _max_reps_from_beginning(ls)
            if isinstance(ls[0], float):
                val = _format_float(ls[0])
            else:
                val = str(ls[0])
            if reps > 1:
                _str += "{}x{}_".format(val, reps)
            else:
                _str += val + '_'
            ls = ls[reps:]
        return _str.rstrip('_')

    def _process_special_keys(key, val):
        if key == 'timesteps':
            val = val // 1000 if (val / 1000).is_integer() else val / 1000
            val = str(val) + 'k'
        elif key == 'name':
            val = ''
        elif key == 'best_key_metric':
            val = ("%.3f" % val).lstrip('0')
        return val

    def _process_val(key, val):
        if not builtin_or_npscalar(val, include_type_type=False):
            assert hasattr(val, '__name__') or hasattr(type(val), '__name__'), (
                f"cannot encode {val} for model name; `model_configs` values must"
                " be either Python literals (str, int, etc), or objects (or their"
                " classes) with  '__name__' attribute. Alternatively, set custom"
                " `name_process_key_fn`")
            val = val.__name__ if hasattr(val, '__name__') else type(val).__name__
            val = val.split('.')[-1]  # drop packages/modules
        else:
            val = _process_special_keys(key, val)

            if isinstance(val, (list, tuple)):
                val = list(val)  # in case tuple
                val = _squash_list(val)
            if isinstance(val, float):
                val = _format_float(val)
        return val

    val = attrs[key]
    val = _process_val(key, val)

    name = alias if alias is not None else key
    return "-{}{}".format(name, val)


_DEFAULT_TRAINGEN_CFG = dict(
    dynamic_predict_threshold_min_max = None,
    checkpoints_overwrite_duplicates  = True,
    loss_weighted_slices_range  = None,
    pred_weighted_slices_range  = None,
    logs_use_full_model_name    = True,
    model_num_continue_from_max = True,
    dynamic_predict_threshold   = 0.5,  # initial
    plot_first_pane_max_vals    = 2,
    _val_max_set_name_chars     = 2,
    _max_set_name_chars  = 3,
    predict_threshold    = 0.5,
    best_subset_size     = 0,
    check_model_health   = True,
    max_one_best_save    = None,
    max_checkpoints = 5,
    report_fontpath = fontsdir + "consola.ttf",
    model_base_name = "model",
    final_fig_dir   = None,

    loadskip_list = _DEFAULT_TRAINGEN_LOADSKIP_LIST,
    saveskip_list = _DEFAULT_TRAINGEN_SAVESKIP_LIST,
    model_save_kw = None,
    model_save_weights_kw = None,
    metric_to_alias     = _DEFAULT_METRIC_TO_ALIAS,
    alias_to_metric     = _DEFAULT_ALIAS_TO_METRIC,
    report_configs      = _DEFAULT_REPORT_CFG,
    model_name_configs  = _DEFAULT_MODEL_NAME_CFG,
    name_process_key_fn = _DEFAULT_NAME_PROCESS_KEY_FN,
    metric_printskip_configs = _DEFAULT_METRIC_PRINTSKIP_CFG,
)
"""Default :class:`TrainGenerator` configurations. Used within
:meth:`TrainGenerator._init_and_validate_kwargs` to check whether any of args
in `**kwargs` isn't one of keys in this dict (in which case it's unused
internally and will raise an exception).

Parameters:
    dynamic_predict_threshold_min_max: tuple[float, float]
        Range of permitted values for `dynamic_predict_threshold` when setting it.
        Useful for constraining "best subset" search to discourage high classifier
        performance with extreme best thresholds (e.g. 0.99), which might do
        *worse* on larger validation sets.
    checkpoints_overwrite_duplicates: bool
        Default value of `overwrite` in :func:`~deeptrain.util.saving.checkpoint`.
        Controls whether checkpoint will overwrite files if they have same name
        as current checkpoint's; if False, will make unique filenames by
        incrementing as '_v2', '_v3', etc.
    loss_weighted_slices_range: tuple[float, float]
        Passed as `weight_range` to
        :func:`~deeptrain.util.training._get_weighted_sample_weight`. A linear
        scaling of `sample_weight` when using slices. During training, this
        is used over `pred_weighted_slices_range`; during validation, uses latter.
    pred_weighted_slices_range: tuple[float, float]
        Same as `loss_weighted_slices_range`, except is used during validation
        to compute metrics from predictions, and not in scaling train-time
        `sample_weight`.
    logs_use_full_model_name: bool
        Whether to use `model_name` or a minimal name containing number of
        validations done + best key metric, within
        :func:`~deeptrain.util.saving.checkpoint`.
    model_num_continue_from_max: bool
        Used within :func:`~deeptrain.util.logging.get_unique_model_name`.
        If True, will set `model_num` to +1 the max number after `"M"` for
        directory names in `logs_dir`; e.g. if such a directory is
        `"M15__Classifier"`, will use `"M16"`, and set `model_num = 16`.
    dynamic_predict_threshold: float
        `predict_threshold` that is optimized during training to yield best
        `key_metric`. See
        :func:`~deeptrain.util.training._set_predict_threshold`, which is called by
        :func:`~deeptrain.util.training._get_val_history`, and
        :func:`~deeptrain.util.training._get_best_subset_val_history`.
    plot_first_pane_max_vals: int
        Stuff
"""

_DEFAULT_DATAGEN_CFG = dict(
    shuffle_group_batches=False,
    shuffle_group_samples=False,
    full_batch_shape=None,
    saveskip_list=_DEFAULT_DATAGEN_SAVESKIP_LIST,
    loadskip_list=_DEFAULT_DATAGEN_LOADSKIP_LIST,
)
