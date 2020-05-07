import os
from termcolor import colored

WARN = colored('WARNING:', 'red')
NOTE = colored('NOTE:', 'blue')

TF_KERAS = bool(os.environ.get('TF_KERAS', "0") == "1")
IMPORTS = {}

if TF_KERAS:
    import tensorflow.keras.backend as K
else:
    import keras.backend as K

try:
    from PIL import Image, ImageDraw, ImageFont
    IMPORTS['PIL'] = 1
except:
    Image, ImageDraw, ImageFont = None, None, None
    IMPORTS['PIL'] = 0

try:
    import lz4framed as lz4f
    IMPORTS['LZ4F'] = 1
except:
    lz4f = None
    IMPORTS['LZ4F'] = 0


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
