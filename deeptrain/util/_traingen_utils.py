from deeptrain.util import logging, saving, searching, training, misc
from deeptrain import introspection, visuals
from inspect import getfullargspec



class TraingenUtils():
    def __init__(self):
        pass

args = lambda x: getfullargspec(x).args
modules = (logging, saving, searching, training, misc, introspection, visuals)
to_exclude = ['_log_init_state']

for module in modules:
    mm = misc.get_module_methods(module)
    for name, method in mm.items():
        if name in to_exclude or 'self' not in args(method):
            continue
        setattr(TraingenUtils, name, method)
