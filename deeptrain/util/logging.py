# -*- coding: utf-8 -*-
"""IDEAS:
   - save each class's source code
   - create "init_configs" to log, then also
   getattr(...) for x in init_configs at save time
   - dedicate 'long column'
"""

import os
import numpy as np
import textwrap
from ._backend import NOTE, WARN
from .misc import _dict_filter_keys

try:
    from PIL import Image, ImageDraw, ImageFont
except:
    print(NOTE, "could not import PIL, will not generate reports")


def generate_report(cls, savepath):
    def _write_text_image(text, savepath, fontpath, fontsize=15,
                          width=1, height=1):
        img = Image.new('RGB', color='white',
                        size=(int(700 * width), int(300 * height)))
        fnt = ImageFont.truetype(fontpath, fontsize)

        d = ImageDraw.Draw(img)
        d.text((10,30), text, fill=(0,0,0), font=fnt)

        img.save(savepath)

    text = get_report_text(cls)
    longest_line = max(map(len, text.split('\n')))
    num_newlines = len(text.split('\n'))

    savepath = savepath or os.path.join(cls.logdir, '_temp_model__report.png')
    try:
        _write_text_image(text, savepath, cls.report_fontpath,
                          width=longest_line / 80,
                          height=num_newlines / 16)
        print("Model report generated and saved")
    except BaseException as e:
        print(WARN,  "Report could not be generated; skipping")
        print("Errmsg:", e)


def get_report_text(cls):
    def list_to_str_side_by_side_by_side(_list, space_between_cols=0):
        def _split_in_three(_list):
            L = len(_list) // 3
            return _list[:L], _list[L:2*L], _list[2*L:]

        def _exclude_chars(_str, chars):
            return ''.join([c for c in _str if c not in chars])

        list1, list2, list3 = _split_in_three(_list)
        longest_str1 = max(map(len, map(str, list1)))
        longest_str2 = max(map(len, map(str, list2)))

        _str = ''
        for entries in zip(list1, list2, list3):
            left, mid, right = [_exclude_chars(str(x), '[]') for x in entries]
            left += " " * (longest_str1 - len(left) + space_between_cols)
            mid  += " " * (longest_str2 - len(mid)  + space_between_cols)

            _str += left + mid + right + '\n'
        return _str

    def _dict_lists_to_tuples(_dict):
        return {key:tuple(val) for key,val in _dict.items()
                if isinstance(val, list)}

    def _dict_filter_value_types(dc, types):
        if not isinstance(types, tuple):
            types = tuple(types) if isinstance(types, list) else (types,)
        return {key:val for key,val in dc.items() if not isinstance(val, types)}

    def _process_attributes_to_text_dicts(report_configs):
        def _validate_report_configs(cfg):
            def _validate_keys(keys):
                supported = ('model', 'traingen', 'datagen', 'val_datagen')
                for key in keys:
                    if key not in supported:
                        print(WARN,  key + " report_configs key not "
                              "supported, and will be ignored; supported "
                              "are: {}".format(', '.join(supported)))
                        keys.pop(keys.index(key))
                return keys

            def _validate_subkeys(cfg):
                supported = ('include', 'exclude', 'exclude_types')
                for key, val in cfg.items():
                    for subkey, attrs in val.items():
                        if not isinstance(attrs, list):
                            raise ValueError("report_configs subkey values must "
                                             "be lists (e.g. 'exclude' values)")
                        if subkey not in supported:
                            raise ValueError(subkey + " report_configs subkey "
                                             "not supported; must be one of: "
                                             + ', '.join(supported))
                        elif 'include' in subkey and 'exclude' in subkey:
                            raise ValueError("cannot have both 'include' and "
                                             "'exclude' subkeys in "
                                             "report_configs")
                return cfg

            def _unpack_tuple_keys(_dict):
                newdict = {}
                for key, val in _dict.items():
                    if isinstance(key, tuple):
                        for k in key:
                            newdict[k] = val
                    else:
                        newdict[key] = val
                return newdict

            keys = []
            for key in cfg:
                keys.extend([key] if isinstance(key, str) else key)
            keys = _validate_keys(keys)

            cfg = _unpack_tuple_keys(report_configs)
            cfg = {k: v for k, v in cfg.items() if k in keys}
            cfg = _validate_subkeys(cfg)

            return cfg

        def _process_wildcards(txt_dict, obj_dict, obj_cfg, exclude):
            for attr in obj_cfg:
                if attr[0] == '*':
                    from_wildcard = _dict_filter_keys(obj_dict, attr[1:],
                                                      exclude=False,
                                                      filter_substr=True).keys()
                    for key in from_wildcard:
                        if exclude and key in txt_dict:
                            del txt_dict[key]
                        elif not exclude:
                            txt_dict[key] = obj_dict[key]
            return txt_dict

        def _exclude_types(txt_dict, name, exclude_types):
            cache, types = {}, []
            for _type in exclude_types:
                if not isinstance(_type, str):
                    types.append(_type)
                elif _type[0] == '#':
                    cache[_type[1:]] = txt_dict[_type[1:]]
                else:
                    print(WARN,  "str type in report_configs "
                          "is unsupported (unless as an exception specifier "
                          "w/ '#' prepended), and will be skipped "
                          "(recieved '%s')" % _type)

            txt_dict = _dict_filter_value_types(txt_dict, types)
            for attr in cache:
                txt_dict[attr] = cache[attr]  # restore cached
            return txt_dict

        cfg = _validate_report_configs(report_configs)

        txt_dicts = dict(model={}, traingen={}, datagen={}, val_datagen={})
        obj_dicts = (cls.model_configs,
                     *map(vars, (cls, cls.datagen, cls.val_datagen)))

        for name, obj_dict in zip(txt_dicts, obj_dicts):
            if obj_dict is not None and (name not in cfg or not cfg[name]):
                txt_dicts[name] = obj_dict
            elif obj_dict is not None:
                for subkey in cfg[name]:
                    obj_cfg = cfg[name][subkey]
                    if subkey != 'exclude_types':
                        exclude = True if subkey == 'exclude' else False
                        txt_dicts[name] = _dict_filter_keys(
                            obj_dict, obj_cfg, exclude=exclude)
                        txt_dicts[name] = _process_wildcards(
                            txt_dicts[name], obj_dict, obj_cfg, exclude)
                    else:
                        txt_dicts[name] = _exclude_types(
                            txt_dicts[name], name, obj_cfg)
        return txt_dicts

    def _wrap_if_long(dicts_list, len_th=80):
        for i, entry in enumerate(dicts_list):
            if len(entry) == 2 and len(str(entry[1])) > len_th:
                dicts_list[i] = [entry[0], []]
                wrapped = textwrap.wrap(str(entry[1]), width=len_th)
                for line in reversed(wrapped):
                    dicts_list.insert(i + 1, [line])
        return dicts_list

    def _postprocess_text_dicts(txt_dicts):
        all_txt = txt_dicts.pop('model')
        for _dict in txt_dicts.values():
            all_txt += [''] + _dict

        _all_txt = _wrap_if_long(all_txt, len_th=80)
        _all_txt = list_to_str_side_by_side_by_side(_all_txt,
                                                    space_between_cols=0)
        _all_txt = _all_txt.replace("',", "' =" ).replace("0, ", "0," ).replace(
                                    "000,", "k,").replace("000)", "k)")
        return _all_txt

    txt_dicts = _process_attributes_to_text_dicts(cls.report_configs)

    titles = (">>HYPERPARAMETERS", ">>TRAINGEN STATE", ">>TRAIN DATAGEN STATE",
              ">>VAL DATAGEN STATE")
    for (name, _dict), title in zip(txt_dicts.items(), titles):
        txt_dicts[name] = _dict_lists_to_tuples(_dict)
        txt_dicts[name] = list(map(list, _dict.items()))
        txt_dicts[name].insert(0, title)

    _all_txt = _postprocess_text_dicts(txt_dicts)
    return _all_txt


def _get_unique_model_name(cls):
    def _get_model_num():
        filenames = ['M0']
        if cls.logs_dir is not None:
            filenames = [name for name in sorted(os.listdir(cls.logs_dir))
                         if 'M' in name]
        if cls.model_num_continue_from_max:
            if len(filenames) != 0:
                model_num = np.max([int(name.split('__')[0].replace('M', ''))
                                    for name in filenames ]) + 1
            else:
                print(NOTE, "no existing models detected in",
                      cls.logs_dir + "; starting model_num from '0'")

        if not cls.model_num_continue_from_max or len(filenames) == 0:
            model_num = 0; _name='M0'
            while any([(_name in filename) for filename in
                       os.listdir(cls.logs_dir)]):
                model_num += 1
                _name = 'M%s' % model_num
        return model_num

    model_name = "M{}__{}".format(_get_model_num(), cls.model_base_name)

    if cls.model_name_configs:
        configs = cls.__dict__.copy()
        if cls.model_configs:
            configs.update(cls.model_configs.copy())

        for key, alias in cls.model_name_configs.items():
            if '.' in key:
                key = key.split('.')
                dict_config = vars(configs[key[0]])
                if key[1] in dict_config:
                    model_name += cls.name_process_key_fn(
                        key[1], alias, dict_config)
            elif key in configs:
                model_name += cls.name_process_key_fn(key, alias, configs)
    return model_name
