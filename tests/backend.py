import os
import contextlib
import tempfile
import shutil


BASEDIR = ''
TF_KERAS = bool(os.environ['TF_KERAS'] == '1')


if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras import losses as keras_losses
    from tensorflow.keras import metrics as keras_metrics
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
    from tensorflow.keras.layers import LSTM, Conv2D, MaxPooling2D
    from tensorflow.keras.models import Model
else:
    from keras import backend as K
    from keras import losses as keras_losses
    from keras import metrics as keras_metrics
    from keras.layers import Input, Dense, Dropout, Flatten
    from keras.layers import LSTM, Conv2D, MaxPooling2D
    from keras.models import Model


@contextlib.contextmanager
def tempdir(prefix=None):
    dirpath = tempfile.mkdtemp(prefix=prefix)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)
