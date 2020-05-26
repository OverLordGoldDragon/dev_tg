# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import textwrap
import pickle
import builtins

from inspect import getsource
from .algorithms import deeplen
from .misc import _dict_filter_keys
from ._backend import NOTE, WARN, Image, ImageDraw, ImageFont


def generate_report(self, savepath):
    def _write_text_image(text, savepath, fontpath, fontsize=15,
                          width=1, height=1):
        img = Image.new('RGB', color='white',
                        size=(int(700 * width), int(300 * height)))
        fnt = ImageFont.truetype(fontpath, fontsize)

        d = ImageDraw.Draw(img)
        d.text((10,30), text, fill=(0, 0, 0), font=fnt)

        img.save(savepath)

    text = get_report_text(self)
    longest_line = max(map(len, text.split('\n')))
    num_newlines = len(text.split('\n'))

    savepath = savepath or os.path.join(self.logdir, '_temp_model__report.png')
    try:
        _write_text_image(text, savepath, self.report_fontpath,
                          width=longest_line / 80,
                          height=num_newlines / 16)
        print("Model report generated and saved")
    except BaseException as e:
        print(WARN,  "Report could not be generated; skipping")
        print("Errmsg:", e)


def get_report_text(self):
    def list_to_str_side_by_side_by_side(_list, space_between_cols=0):
        def _split_in_three(_list):
            def _pad_column_bottom(_list):
                l = len(_list) // 3
                to_fill = 3 - len(_list) % 3

                if to_fill == 1:
                    _list.insert(l, '')
                elif to_fill == 2:
                    _list.insert(l, '')
                    _list.insert(2 * l + 1, '')
                return _list

            L = len(_list) / 3
            if not L.is_integer():
                # L must be integer to preserve all rows
                _list = _pad_column_bottom(_list)
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
                        print(WARN, "'%s' report_configs key not " % key
                              + "supported, and will be ignored; supported "
                              "are: {}".format(', '.join(supported)))
                        keys.pop(keys.index(key))
                return keys

            def _validate_subkeys(cfg):
                supported = ('include', 'exclude', 'exclude_types')
                for key, val in cfg.items():
                    if 'include' in val and 'exclude' in val:
                        raise ValueError("cannot have both 'include' and "
                                         "'exclude' subkeys in report_configs")
                    for subkey, attrs in val.items():
                        if not isinstance(attrs, list):
                            raise ValueError("report_configs subkey values must "
                                             "be lists (e.g. 'exclude' values)")
                        if subkey not in supported:
                            raise ValueError(
                                ("'{}' report_configs subkey not supported; must "
                                 "be one of: {}").format(
                                     subkey, ', '.join(supported)))
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
        obj_dicts = (self.model_configs,
                     *map(vars, (self, self.datagen, self.val_datagen)))

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

    txt_dicts = _process_attributes_to_text_dicts(self.report_configs)

    titles = (">>HYPERPARAMETERS", ">>TRAINGEN STATE", ">>TRAIN DATAGEN STATE",
              ">>VAL DATAGEN STATE")
    for (name, _dict), title in zip(txt_dicts.items(), titles):
        txt_dicts[name] = _dict_lists_to_tuples(_dict)
        txt_dicts[name] = list(map(list, _dict.items()))
        txt_dicts[name].insert(0, title)

    _all_txt = _postprocess_text_dicts(txt_dicts)
    return _all_txt


def get_unique_model_name(self):
    def _get_model_num():
        filenames = ['M0']
        if self.logs_dir is not None:
            filenames = [name for name in sorted(os.listdir(self.logs_dir))
                         if 'M' in name]
        if self.model_num_continue_from_max:
            if len(filenames) != 0:
                model_num = np.max([int(name.split('__')[0].replace('M', ''))
                                    for name in filenames ]) + 1
            else:
                print(NOTE, "no existing models detected in",
                      self.logs_dir + "; starting model_num from '0'")

        if not self.model_num_continue_from_max or len(filenames) == 0:
            model_num = 0; _name='M0'
            while any([(_name in filename) for filename in
                       os.listdir(self.logs_dir)]):
                model_num += 1
                _name = 'M%s' % model_num
        return model_num

    model_name = "M{}__{}".format(_get_model_num(), self.model_base_name)

    if self.model_name_configs:
        configs = self.__dict__.copy()
        if self.model_configs:
            configs.update(self.model_configs.copy())

        for key, alias in self.model_name_configs.items():
            if '.' in key:
                key = key.split('.')
                dict_config = vars(configs[key[0]])
                if key[1] in dict_config:
                    model_name += self.name_process_key_fn(
                        key[1], alias, dict_config)
            elif key in configs:
                model_name += self.name_process_key_fn(key, alias, configs)
    return model_name


def _log_init_state(self, kwargs={}, source_lognames='__main__', savedir=None,
                    to_exclude=[], verbose=0):
    """Extract `self` __dict__ key-value pairs as string, ignoring funcs/methods
    or getting their source codes. May include kwargs passed to __init__ via
    `kwargs`, and __main__ via `source_lognames`.

    Arguments:
        kwargs: kwargs passed to self's __init__.
        source_lognames: str/list of str. Names of self methoda attributes
            to get source code of.
        savedir: str. Path to directory where to save logs. Saves a .json of
               self dict, and .txt of source codes (if any).
    """
    def _save_logs(state, source, savedir, verbose):
        path = os.path.join(savedir, "init_state.h5")
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        if verbose:
            print(str(self), "initial state saved to", path)

        if source != '':
            path = os.path.join(savedir, "init_source.txt")
            with open(path, 'w') as f:
                f.write(source)
            if verbose:
                print(str(self), "source codes saved to", path)

    def builtin_or_npscalar(x):
        return (type(x) in (*vars(builtins).values(), type(None), type(min)) or
                isinstance(x, np.generic))

    def _name(x):
        if hasattr(x, '__name__'):
            return x.__name__
        elif hasattr(x, '__class__'):
            return str(x.__class__).replace("<class ", '').replace(
                ">", '').replace("'", '')
        else:
            return str(x)

    def _filter_objects(state_full, to_exclude):
        state = {}
        for k, v in state_full.items():
            if k in to_exclude:
                continue
            elif builtin_or_npscalar(v):
                if hasattr(v, '__len__') and deeplen(v) > 50:
                    v = _name(v)
                state[k] = str(v)
            else:
                state[k] = _name(v)
        return state

    def _get_source_code(state_full, source_lognames):
        def _get_main_source():
            if not hasattr(sys.modules['__main__'], '__file__'):
                return '', ''
            path = os.path.abspath(sys.modules['__main__'].__file__)
            with open(path, 'r') as f:
                return f.read(), path

        def _get_all_sourceable(keys, source, state_full):
            def not_func(x):
                return getattr(getattr(x, '__class__', None),
                               '__name__', '') not in ('function', 'method')

            to_skip = ['__main__']
            if not isinstance(keys, (list, tuple)):
                keys = [keys]

            for k in keys:
                if k in to_skip:
                    continue
                elif k not in state_full:
                    print(WARN, f"{k} not found in self.__dict__ - will skip")
                    continue
                v = state_full[k]
                if (not builtin_or_npscalar(v) and
                    not isinstance(v, np.ndarray)):
                    if not_func(v):
                        v = v.__class__
                try:
                    source[_name(v)] = getsource(v)
                except Exception as e:
                    print("Failed to log:", k, v, "-- skipping. "
                          "Errmsg: %s" % e)

            return source

        def _to_text(source):
            def _wrap_decor(x):
                """Format as: ## long_text_s ##
                              ## tuff #########"""
                wrapped = textwrap.wrap(x, width=77)
                txt = ''
                for line in wrapped:
                    txt += "## %s\n" % (line + ' ' + "#" * 77)[:80]
                return txt.rstrip('\n')

            txt = ''
            for k, v in source.items():
                txt += "\n\n{}\n{}".format(_wrap_decor(k), v)
            return txt.lstrip('\n')

        source = {}
        if source_lognames == '*':
            source_lognames = list(state_full)
        source = _get_all_sourceable(source_lognames, source, state_full)

        if '__main__' in source_lognames or source_lognames == '*':
            src, path = _get_main_source()
            source[path] = src

        source = _to_text(source)
        return source

    if not isinstance(to_exclude, (list, tuple)):
        to_exclude = [to_exclude]

    state_full = vars(self)
    for k, v in kwargs.items():
        if k not in state_full:
            state_full[k] = v
    state = _filter_objects(state_full, to_exclude)

    if source_lognames is not None:
        source = _get_source_code(state_full, source_lognames)
    else:
        source = ''

    if savedir is not None:
        _save_logs(state, source, savedir, verbose)
    return state, source
