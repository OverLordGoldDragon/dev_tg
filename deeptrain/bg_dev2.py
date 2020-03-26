from termcolor import colored
from .bg_dev import BatchGenerator

WARN = colored('WARNING: ', 'red')
NOTE = colored('NOTE: ',    'blue')


class SimpleBatchgen(BatchGenerator):
    def __init__(self, data_dir, labels_path, batch_size, data_category,
                 data_format=None,
                 preprocessor_configs=None,
                 base_name=None,
                 shuffle=False,
                 dtype='float32',
                 data_ext=None,
                 superbatch_dir=None,
                 set_nums=None,
                 superbatch_set_nums=None,
                 full_batch_shape=None,  # TODO move to kwargs?
                 **kwargs,
                 ):
        super(SimpleBatchgen, self).__init__(
            data_dir=data_dir,
            labels_path=labels_path,
            batch_size=batch_size,
            data_category=data_category,
            data_format=data_format,
            preprocessor_configs=preprocessor_configs,
            base_name=base_name,
            dtype=dtype,
            data_ext=data_ext,
            superbatch_dir=superbatch_dir,
            set_nums=set_nums,
            superbatch_set_nums=superbatch_set_nums,
            full_batch_shape=full_batch_shape,
            **kwargs
            )



class GroupBatchgen(BatchGenerator):  # TODO
    def __init__(self):
        pass
