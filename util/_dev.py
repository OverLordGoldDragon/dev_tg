REPORT_CONFIGS = {
'attributes_to_exclude':
    ({'traingen':
        ({'fullstr':
            ['model','model_configs','model_name_configs',
             'batch','superbatch','batch_loaded','batch_exhausted','group_batch',
             'model_name','pretrained_model_dir','logs_use_full_model_name',
             'history_fig','max_checkpoints_to_keep','do_logging',
             'known_kwargs_keys', 'history', 'val_history', 'temp_history',
             'val_temp_history','plot_configs','encoder_configs',
             'encoder_weights_dir']
        },
         {'substr':
            ['has_', 'path', 'data', 'visual_']
        }),
    },
    {('datagen', 'val_datagen'):
         ({},)
    }),
}

def _process_attributes_to_text_dicts2(cls, report_configs):
    def _get_config_dict(name, is_recursive=False):
        if isinstance(name, tuple) and is_recursive:
            raise ValueError("tuples in `attributes_to_exclude` can only "
                             "contain strings")
        if isinstance(name, tuple):
            names, dicts = [], {}
            for n in name:
                names.append(n)
                dicts[name] = _get_config_dict(name, True)
            return names, dicts

        elif name == 'traingen':
            return cls.__dict__
        elif name == 'model':
            return cls.model_configs
        else:
            return getattr(cls, name).__dict__       

    def _get_attrs_to_exclude(dicts):
        attrs_to_exclude = {}
        for dict_name, _dict in dicts.items():
            to_exclude = _dict['fullstr']

            exclude_substr = _dict.get('substr', None)
            if exclude_substr:
                dincl = _dict_filter_keys(_dict, exclude_substr, 
                                          exclude=False, filter_substr=True)
                to_exclude += list(dincl.keys())
            attrs_to_exclude[dict_name] = to_exclude
        return attrs_to_exclude
    
    dicts = {}
    for name in report_configs['attributes_to_exclude']:
        _dict = _get_config_dict(name)
        if not isinstance(_dict, tuple):
            dicts[name] = _dict
        else:
            for name, dc in _dict.items():
                dicts[name] = dc

    attrs_to_exclude = _get_attrs_to_exclude(dicts)
    for (name, _dict), to_exclude in zip(dicts.items(), attrs_to_exclude):
        dicts[name] = _dict_filter_keys(_dict, to_exclude)

    return dicts
