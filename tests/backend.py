import contextlib
import tempfile
import shutil


BASEDIR = ''

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
