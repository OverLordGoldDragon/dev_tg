import os
import contextlib
import shutil


if os.path.isdir(r"C:\Desktop\School\Deep Learning\DL_code\dev_tg"):
    BASEDIR = r"C:\Desktop\School\Deep Learning\DL_code\dev_tg"
else:
    BASEDIR = ''
TF_KERAS = bool(os.environ.get('TF_KERAS', '0') == '1')


if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras import losses as keras_losses
    from tensorflow.keras import metrics as keras_metrics
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
else:
    from keras import backend as K
    from keras import losses as keras_losses
    from keras import metrics as keras_metrics
    from keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from keras.models import Model


@contextlib.contextmanager
def tempdir(dirpath):
    if os.path.isdir(dirpath):
        os.remove(dirpath)
    os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)
