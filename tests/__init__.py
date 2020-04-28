import os
import matplotlib

from . import timeseries_test, image_test, autoencoder_test
from . import backend


if not os.environ.get('IS_MAIN', '0') == '1':
    matplotlib.use('template')  # suppress figures for spyder unit testing
