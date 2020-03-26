# -*- coding: utf-8 -*-
"""Default configs -- DO NOT EDIT"""
import numpy as np


_DEFAULT_PLOT_CFG = {
'1': {
    'metrics':
        {'train': ['loss'],
         'val'   : ['loss']},
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
             'history_fig', 'plot_configs', 'max_checkpoints_to_keep',
             'history', 'val_history', 'temp_history', 'val_temp_history',
             'name_process_key_fn', 'report_fontpath', 'model_name_configs', 
             'report_configs', 'datagen', 'val_datagen',
             'logdir', 'logs_dir', 'best_models_dir', 'fit_fn', 'eval_fn',
             '_history_fig', 'metric_printskip_configs',
             '*_has_', '*temp_history_empty',
             ],
        'exclude_types':
            [list, np.ndarray, '#best_subset_nums'],
        },
    ('datagen', 'val_datagen'):
        {
        'exclude':
            ['batch', 'group_batch', 'labels', 'all_labels',
             'batch_loaded', 'batch_exhausted', 'set_num',
             'set_nums_original', 'set_nums_to_process', 'superbatch_set_nums',
             'load_data', 'data_dir', 'labels_path', 'loadskip_list',
             '_path_attrs', 'preprocessor',
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
    'val_print_omits_key_metric',  #TODO replace w/ configurable

    'datagen',
    'val_datagen',
    'model_name',
    'model_num',
    'epoch',
    'val_epoch',
    'key_metric_history',
    'best_key_metric',
    'labels',  #TODO what if inputs_as_labels==True?
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
    'set_num',
    '_val_set_num',
    
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
    '_temp_history_empty',  #TODO exclude?
    '_val_temp_history_empty',
]

_DEFAULT_METRIC_PRINTSKIP_CFG = {
    'train': [],
    'val': [],
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

        def _format_if_small_decimal(val, th=1e-2):
            if isinstance(val, float) and abs(val) < th:
                return ('%.e' % val).replace('-0', '-') 
            return val

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
        val = val if isinstance(val, list) else [val]
        val = _squash_list(val)
    if isinstance(val, float):
        val = _format_float(val)

    return "_{}{}".format(alias, val)


_path = r"D:\Desktop\School\Deep Learning\DL_code\\"
_DEFAULT_TRAINGEN_CFG = dict(
    dynamic_predict_threshold_min_max = (0.35, 0.90),
    use_dynamic_predict_threshold = False,
    weighted_slices_range       = None,
    use_passed_dirs_over_loaded = False,
    static_predict_threshold    = 0.5,
    dynamic_predict_threshold   = 0.5,  # initial
    visual_outputs_layer_names  = None, 
    visual_weights_layer_names  = None,
    logs_use_full_model_name    = True,
    model_num_continue_from_max = True,
    max_checkpoints_to_keep     = 5,
    best_subset_size  = 0,
    keep_one_best_model   = False,
    save_post_epoch       = False,
    check_model_health = True,
    outputs_visualizer = 'comparative_histogram',
    report_fontpath    = r"C:\Windows\Fonts\consola.ttf",
    model_base_name    = "model",
    make_new_logdir    = True,
    final_fig_dir      = None,

    savelist       = _DEFAULT_TRAINGEN_SAVE_LIST,
    report_configs = _DEFAULT_REPORT_CFG,
    model_name_configs  = _DEFAULT_MODEL_NAME_CFG,
    name_process_key_fn = _DEFAULT_NAME_PROCESS_KEY_FN,
    metric_printskip_configs = _DEFAULT_METRIC_PRINTSKIP_CFG,
)