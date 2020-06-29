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
    best_key_metric = '__max',
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
    'optimizer_state',
    'callbacks',
    'key_metric_fn',
    'custom_metrics',
    'metric_to_alias',
    'alias_to_metric',
    'name_process_key_fn',
    'fit_fn',  # TODO excl. since class properties can't be removed from instance?
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
    '_val_max_set_name_chars',
    '_max_set_name_chars',
    '_inferred_batch_size',
    '_class_labels_cache',
    '_train_val_x_ticks',
    '_val_train_x_ticks',  # ???
    '_temp_history_empty',
    '_val_temp_history_empty',
    '_val_sw',
    '_set_num',
    '_val_set_num',
    # TODO include unlisted as comments: 'model:weights',
]

_DEFAULT_TRAINGEN_LOADSKIP_LIST = ['{auto}', 'model_name', 'model_base_name',
                                   'model_num', 'use_passed_dirs_over_loaded']

_DEFAULT_DATAGEN_SAVESKIP_LIST = ['batch', 'superbatch', 'labels', 'all_labels',
                                  '_group_batch', '_group_labels']
_DEFAULT_DATAGEN_LOADSKIP_LIST = ['data_dir', 'labels_path', 'superbatch_dir',
                                  'data_loader', 'set_nums_original',
                                  'set_nums_to_process', 'superbatch_set_nums']

_DEFAULT_MODEL_SAVE_KW = {'include_optimizer': True, 'save_format': None}
_DEFAULT_MODEL_SAVE_WEIGHTS_KW = {'save_format': None}

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
        if key == 'best_key_metric':
            val = ("%.3f" % val).lstrip('0')
        elif key == 'name':
            val = ''
        elif key == 'timesteps':
            val = val // 1000 if (val / 1000).is_integer() else val / 1000
            val = str(val) + 'k'
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
    if key != 'best_key_metric':
        return "-{}{}".format(name, val)
    else:
        return "{}{}".format(name, val)


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
    best_subset_size     = None,
    check_model_health   = True,
    max_one_best_save    = True,
    max_checkpoints = 5,
    report_fontpath = fontsdir + "consola.ttf",
    model_base_name = "model",
    final_fig_dir   = None,

    loadskip_list = _DEFAULT_TRAINGEN_LOADSKIP_LIST,
    saveskip_list = _DEFAULT_TRAINGEN_SAVESKIP_LIST,
    model_save_kw = _DEFAULT_MODEL_SAVE_KW,
    model_save_weights_kw = _DEFAULT_MODEL_SAVE_WEIGHTS_KW,
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
        Useful for constraining "best subset" search to discourage high binary
        classifier performance with extreme best thresholds (e.g. 0.99), which
        might do *worse* on larger validation sets.
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
    dynamic_predict_threshold: float / None
        `predict_threshold` that is optimized during training to yield best
        `key_metric`. See
        :func:`~deeptrain.util.training._set_predict_threshold`, which's called by
        :func:`~deeptrain.util.training._get_val_history`, and
        :func:`~deeptrain.util.training._get_best_subset_val_history`. If None,
        will only use `predict_threshold`.
    plot_first_pane_max_vals: int
        Maximum number of validation metrics to plot, as set by
        :func:`~deeptrain.util.misc._make_plot_configs_from_metrics`, for
        :func:`~deeptrain.visuals.get_history_fig`. This is a setting for the
        default config maker (first method), which plots all train metrics
        in first pane.
    _val_max_set_name_chars: int
        Padding to use in :meth:`TrainGenerator._print_iter_progress` to justify
        `val_set_name` when printing "Validating set val_set_name"; should be
        set to expected longest set name for vertical alignment. E.g. if
        `'123'`, should set to 3, if `'99'`, to 2.
    _max_set_name_chars: int
        Same as `_val_max_set_name_chars`, but for train `_set_name`.
    predict_threshold: float
        Binary classifier prediction threshold, above which to classify as `'1'`,
        used in :func:`~deeptrain.util.training._compute_metrics`.
        If `dynamic_predict_threshold` and `dynamic_predict_threshold_min_max`
        are not None, it will be set equal to former within bounds of latter.
    best_subset_size: int >= 1 / None
        If not None, will search for `best_subset_size` number of batches yielding
        best validation performance, out of all validation batches (e.g. 5 of 10).
        Useful for model ensembling in specializing member models on different
        parts of data.
        see :func:`~deeptrain.util.training._get_best_subset_val_history`.
    check_model_health: bool
        Whether to call :meth:`TrainGenerator.check_health` at the end of
        validation in :meth:`TrainGenerator._on_val_end`, which checks whether
        any `model` layers have zero/NaN weights. Very fast / inexpensive.
    max_one_best_save: bool
        Whether to keep only one set of save files (model weights,
        `TrainGenerator` state, etc.) in `best_models_dir` when saving best model
        via :func:`~deeptrain.util.saving._save_best_model`.
    max_checkpoints: int
        Maximum sets of checkpoint files (model weights, `TrainGenerator` state,
        etc.) to keep in `logdir`, when checkpointing via
        :func:`~deeptrain.util.saving.checkpoint`.
    report_fontpath: str
        Path to font file for font to use in saving report
        (:func:`~deeptrain.util.logging.save_report); defaults to consola,
        which yields nice vertical & horizontal alignment.
    model_base_name: str
        Name between `"M{model_num}"` and autogenerated string from
        `model_configs`, as `"M{model_num}_{model_base_name}_*"`; see
        :func:`~deeptrain.util.logging.get_unique_model_name`.
    final_fig_dir: str / None
        Path to directory where to save latest metric history using full
        `model_name`, at most one per `model_num`. If None, won't save such a
        figure (but will still save history for best model & checkpoint).
    loadskip_list: list[str]
        List of `TrainGenerator` attribute names to skip from loading. Mainly
        for attributes that should change between different train sessions,
        e.g. `model_num`, or shouldn't have `**kwargs` values overridden by load.
    saveskip_list: list[str]
        List of `TrainGenerator` attribute names to skip from saving. Used to
        exclude e.g. objects that cannot be pickled (e.g. `model`), are large
        and should be loaded separately (e.g. `batch`), or should be
        reinstantiated (e.g. `_imports`).
    model_save_kw: dict / None
        Passed as kwargs to `model.save()`:

            - `overwrite`: bool. Whether to overwrite existing file.
            - `include_optimizer`: bool. Whether to include optimizer weights.
            - `save_format` (tf.keras): str. Savefile format. If None, will
              internally default to `'h5'` if using tf.keras else will drop.
            - others (tf.keras): see `model.save()`.

    model_save_weights_kw: dict / None
        Passed as kwargs to `model.save_weights()`; same as `model_save_kw`,
        excluding `include_optimizer`.
    metric_to_alias: dict
        Dict mapping a metric name to its alias; current use is for controlling
        how metric names are printed (see :meth:`TrainGenerator._print_progress`).
    alias_to_metric: dict
        Dict mapping a metric alias to its TF/Keras/DeepTrain name. If defined
        in TF/Keras, DeepTrain uses the same names - else, they'll match function
        names in :mod:`deeptrain.metrics`.
"""

_DEFAULT_DATAGEN_CFG = dict(
    shuffle_group_batches=False,
    shuffle_group_samples=False,
    full_batch_shape=None,
    saveskip_list=_DEFAULT_DATAGEN_SAVESKIP_LIST,
    loadskip_list=_DEFAULT_DATAGEN_LOADSKIP_LIST,
)
