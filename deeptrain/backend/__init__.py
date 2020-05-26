import os
import tensorflow as tf

TF_KERAS = bool(os.environ.get('TF_KERAS', "0") == "1")
TF_EAGER = bool(tf.executing_eagerly())
TF_2 = bool(tf.__version__[0] == '2')

if TF_KERAS:
    import tensorflow.keras.backend as K
else:
    import keras.backend as K

###############################################################################
from . import model_util

from .model_util import *

