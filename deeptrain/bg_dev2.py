from .bg_dev import BatchGenerator


class SimpleBatchgen(BatchGenerator):
    def __init__(self, data_dir, batch_size,
                 labels_path=None,
                 preprocessor=None,
                 preprocessor_configs=None,
                 data_format=None,
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
            batch_size=batch_size,
            labels_path=labels_path,
            preprocessor=preprocessor,
            preprocessor_configs=preprocessor_configs,
            data_format=data_format,
            base_name=base_name,
            shuffle=shuffle,
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
