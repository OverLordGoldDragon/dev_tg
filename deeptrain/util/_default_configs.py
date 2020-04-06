# -*- coding: utf-8 -*-
import numpy as np
from .fonts import fontsdir


_DEFAULT_PLOT_CFG = {
'1': {
    'metrics':
        {'train': ['loss'],
         'val'  : ['loss']},
    'x_ticks':
        {'train': ['_train_x_ticks'],
         'val':   ['_val_train_x_ticks']},
    'vhlines'   :
        {'v': '_hist_vlines',
         'h': 1},
    'mark_best_idx': 1,
    'ylims'        : (0, 2),

    'linewidth': (1.5, 2),
    'linestyle': ('-', '-'),
    'color'    : (None, 'orange'),
},
}

_DEFAULT_BINARY_CLASSIFICATION_PLOT_CFG = {
'1': {
    'metrics':
        {'train': ['loss', 'acc'],
         'val'   : ['loss']},
    'vhlines'   :
        {'v': '_hist_vlines',
         'h': 1},
    'ylims'     : (0, 2),

    'linewidth': (1.5, 1, 2),
    'linestyle': ('-', '--', '-'),
    'color'    : (None, 'b', 'orange'),
    'mark_best_idx': None,
},
'2': {
    'metrics':
        {'val'  : ['tnr', 'tpr', 'f1-score']},
    'vhlines'   :
        {'v': '_val_hist_vlines',
         'h': .5},
    'mark_best_idx': 2,

    'linewidth': (2, 2, 2),
    'linestyle': ('-', '-', '-'),
    'color'    : (None, 'r', 'purple'),
    'ylims'     : (0, 1),
},
}

# order-dependent
_DEFAULT_MODEL_NAME_CFG = dict(
    timesteps       = '',
    init_lr         = '',
    optimizer       = '',
    best_key_metric = '_max',
)


# * == wildcard (match as substring)
_DEFAULT_REPORT_CFG = {
    'model':
        {},
    'traingen':
        {
        'exclude':
            ['model', 'model_configs', 'model_name', 'logs_use_full_model_name',
             'history_fig', 'plot_configs', 'max_checkpoint_saves',
             'history', 'val_history', 'temp_history', 'val_temp_history',
             'name_process_key_fn', 'report_fontpath', 'model_name_configs',
             'report_configs', 'datagen', 'val_datagen',
             'logdir', 'logs_dir', 'best_models_dir', 'fit_fn', 'eval_fn',
             '_history_fig', 'metric_printskip_configs', '_inferred_batch_size',
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
             'superbatch_set_nums', 'load_data', 'data_dir', 'labels_path',
             'loadskip_list', '_path_attrs', 'preprocessor',
             '*_ATTRS', '*superbatch', '*_filepaths', '*_filenames']
        },
}


_DEFAULT_TRAINGEN_SAVE_LIST = [
    'val_freq',
    'plot_history_freq',
    'key_metric',
    'train_metrics',
    'val_metrics',
    'inputs_as_labels',
    'batch_size',

    'datagen',
    'val_datagen',
    'model_name',
    'model_num',
    'epoch',
    'val_epoch',
    'key_metric_history',
    'best_key_metric',
    'labels',
    '_val_labels',
    '_labels_cache',
    '_sw_cache',
    '_y_true',
    '_y_preds',
    'history',
    'val_history',
    'temp_history',
    'val_temp_history',
    'history_fig',
    'predict_threshold',
    '_set_name',
    '_val_set_name',
    '_set_name_cache',
    '_val_set_name_cache',

    '_batches_fit',
    '_batches_validated',
    '_has_trained',
    '_has_validated',
    '_has_postiter_processed',
    '_val_has_postiter_processed',
    '_train_has_notified_of_new_batch',
    '_val_has_notified_of_new_batch',
    '_train_x_ticks',
    '_val_x_ticks',
    '_times_validated',
    '_hist_vlines',
    '_val_hist_vlines',
]

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
    'acc':    'accuracy',
    'mae':    'mean_absolute_error',
    'mse':    'mean_squared_error',
    'mape':   'mean_absolute_percentage_error',
    'msle':   'mean_squared_logarithmic_error',
    'kld':    'kullback_leibler_divergence',
    'cosine': 'cosine_proximity',
    'f1':     'f1-score',
}


def _DEFAULT_NAME_PROCESS_KEY_FN(key, alias, configs):
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
            return ("%.3f" % val).lstrip('0')
        else:
            return "%.3f" % val

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

    val = configs[key]
    val = _process_special_keys(key, val)

    if isinstance(val, (list, tuple)):
        val = list(val)  # in case tuple
        val = _squash_list(val)
    if isinstance(val, float):
        val = _format_float(val)

    return "_{}{}".format(alias, val)


_DEFAULT_TRAINGEN_CFG = dict(
    dynamic_predict_threshold_min_max = None,
    loss_weighted_slices_range  = None,
    pred_weighted_slices_range  = None,
    use_passed_dirs_over_loaded = False,
    visual_outputs_layer_names  = None,
    visual_weights_layer_names  = None,
    logs_use_full_model_name    = True,
    model_num_continue_from_max = True,
    dynamic_predict_threshold   = 0.5,  # initial
    predict_threshold    = 0.5,
    best_subset_size     = 0,
    check_model_health   = True,
    outputs_visualizer   = 'comparative_histogram',
    max_checkpoint_saves = 5,
    max_one_best_save    = None,
    report_fontpath = fontsdir + "consola.ttf",
    model_base_name = "model",
    make_new_logdir = True,
    final_fig_dir   = None,

    savelist = _DEFAULT_TRAINGEN_SAVE_LIST,
    metric_to_alias     = _DEFAULT_METRIC_TO_ALIAS,
    alias_to_metric     = _DEFAULT_ALIAS_TO_METRIC,
    report_configs      = _DEFAULT_REPORT_CFG,
    model_name_configs  = _DEFAULT_MODEL_NAME_CFG,
    name_process_key_fn = _DEFAULT_NAME_PROCESS_KEY_FN,
    metric_printskip_configs = _DEFAULT_METRIC_PRINTSKIP_CFG,
)

_DEFAULT_DATAGEN_CFG = dict(
    shuffle_group_batches=False,
    shuffle_group_samples=False,
    full_batch_shape=None,
)
