import os
from termcolor import colored

WARN = colored('WARNING:', 'red')
NOTE = colored('NOTE:',    'blue')
ERR  = colored('ERROR:',   'red')


TF_KERAS = bool(os.environ.get('TF_KERAS', "0") == "1")

if TF_KERAS:
    import tensorflow.keras.backend as K
else:
    import keras.backend as K

from . import configs
from . import logging
from . import metrics
from . import saving
from . import searching
from . import training
from . import visuals
from . import introspection
from . import misc


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
