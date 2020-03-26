import os
import contextlib
import tempfile
import shutil


BASEDIR = ''
TF_KERAS = bool(os.environ['TF_KERAS'] == '1')


if TF_KERAS:
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
    from tensorflow.keras.layers import LSTM, Conv2D, MaxPooling2D
    from tensorflow.keras.models import Model
else:
    from keras.layers import Input, Dense, Dropout
    from keras.layers import LSTM, Conv2D, MaxPooling2D
    from keras.models import Model


@contextlib.contextmanager
def tempdir(prefix=None):
    dirpath = tempfile.mkdtemp(prefix=prefix)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


# with tempdir("logs_") as logs_dir, tempdir("models_") as models_dir:
#     TRAINGEN_CFG['logs_dir'] = logs_dir
#     TRAINGEN_CFG['best_models_dir'] = models_dir
