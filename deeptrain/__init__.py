from . import train_generator
from . import data_generator
from . import metrics
from . import callbacks
from . import visuals
from . import introspection
from . import preprocessing
from . import util

from .train_generator import TrainGenerator
from .data_generator import DataGenerator


def set_seeds(seeds=None, reset_graph=False, verbose=1):
    """Sets random seeds and maybe clears keras session and resets TensorFlow
    default graph.

    NOTE: after calling w/ `reset_graph=True`, best practice is to re-instantiate
    model, else some internal operations may fail due to elements from different
    graphs interacting (of pre-reset model and post-reset tensors).
    """
    callbacks.RandomSeedSetter._set_seeds(seeds, reset_graph, verbose)


__version__ = '0.01'
