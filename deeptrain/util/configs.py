# -*- coding: utf-8 -*-
import numpy as np
from .fonts import fontsdir

# TODO put elsewhere? unused -- or, integrate w/ _make_plot_configs_from_metrics
# e.g. `ylims = (0, 2)` is hard-coded
# keep `None` for configs to be determined automatically by _make_plot...
# TODO add '2' to _PLOT_CFG
_PLOT_CFG = {
'1': {
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
'2': {
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
}

_BINARY_CLASSIFICATION_PLOT_CFG = {
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
    'mark_best_cfg': None,
},
'2': {
    'metrics':
        {'val'  : ['tnr', 'tpr', 'f1-score']},
    'vhlines'   :
        {'v': '_val_hist_vlines',
         'h': .5},
    'mark_best_cfg': {'val': 'f1-score'},

    'linewidth': (2, 2, 2),
    'linestyle': ('-', '-', '-'),
    'color'    : (None, 'r', 'purple'),
    'ylims'     : (0, 1),
},
}

# order-dependent
_MODEL_NAME_CFG = dict(
    timesteps       = '',
    init_lr         = '',
    optimizer       = '',
    best_key_metric = '_max',
)


# * == wildcard (match as substring)
_REPORT_CFG = {
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
             'best_models_dir', 'fit_fn', 'eval_fn',
             'callbacks', 'callbacks_init', 'callback_objs',
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


_TRAINGEN_SAVESKIP_LIST = [
    'model',
    'callbacks',
    'callbacks_init',
    'key_metric_fn',
    'custom_metrics',
    'use_passed_dirs_over_loaded',
    'check_model_health',
    'metric_to_alias',
    'alias_to_metric',
    'name_process_key_fn',
    'fit_fn',
    'eval_fn',
    'callback_objs',
    'optimizer_state',  # is overridden anyway

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

_TRAINGEN_LOADSKIP_LIST = ['{auto}', 'model_name', 'model_base_name',
                           'model_num']

_DATAGEN_SAVESKIP_LIST = ['batch', 'superbatch', 'labels', 'all_labels',
                          '_group_batch', '_group_labels']
_DATAGEN_LOADSKIP_LIST = ['data_dir', 'labels_path', 'superbatch_dir',
                          'data_loader', 'set_nums_original',
                          'set_nums_to_process', 'superbatch_set_nums']

_METRIC_PRINTSKIP_CFG = {
    'train': [],
    'val': [],
}

_METRIC_TO_ALIAS = {
    'loss'    : 'Loss',
    'accuracy': 'Acc',
    'acc'     : 'Acc',
    'f1_score': 'F1',
    'tnr'     : '0-Acc',
    'tpr'     : '1-Acc',
}

_ALIAS_TO_METRIC = {
    'acc':     'accuracy',
    'mae':     'mean_absolute_error',
    'mse':     'mean_squared_error',
    'mape':    'mean_absolute_percentage_error',
    'msle':    'mean_squared_logarithmic_error',
    'kld':     'kullback_leibler_divergence',
    'cosine':  'cosine_proximity',
    'f1':      'f1_score',
    'f1-score':'f1_score',
}


def _NAME_PROCESS_KEY_FN(key, alias, configs):
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

    val = configs[key]
    val = _process_special_keys(key, val)

    if isinstance(val, (list, tuple)):
        val = list(val)  # in case tuple
        val = _squash_list(val)
    if isinstance(val, float):
        val = _format_float(val)

    return "_{}{}".format(alias, val)


_TRAINGEN_CFG = dict(
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
    max_checkpoints = 5,
    max_one_best_save    = None,
    report_fontpath = fontsdir + "consola.ttf",
    model_base_name = "model",
    make_new_logdir = True,
    final_fig_dir   = None,

    loadskip_list = _TRAINGEN_LOADSKIP_LIST,
    saveskip_list = _TRAINGEN_SAVESKIP_LIST,
    metric_to_alias     = _METRIC_TO_ALIAS,
    alias_to_metric     = _ALIAS_TO_METRIC,
    report_configs      = _REPORT_CFG,
    model_name_configs  = _MODEL_NAME_CFG,
    name_process_key_fn = _NAME_PROCESS_KEY_FN,
    metric_printskip_configs = _METRIC_PRINTSKIP_CFG,
)

_DATAGEN_CFG = dict(
    shuffle_group_batches=False,
    shuffle_group_samples=False,
    full_batch_shape=None,
    saveskip_list=_DATAGEN_SAVESKIP_LIST,
    loadskip_list=_DATAGEN_LOADSKIP_LIST,
)
