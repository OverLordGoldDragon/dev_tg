from . import K, TF_KERAS, TF_2
from ..util.misc import try_except


def eval_tensor(x, backend):
    K = backend
    te = try_except
    te(lambda x: K.get_value(K.to_dense),
       te(lambda x: K.function([], [x])([])[0],
          te(lambda x: K.eager(K.eval)(x),
             lambda x: K.eval(x))))
