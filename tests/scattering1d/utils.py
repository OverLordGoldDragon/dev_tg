# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Methods reused in testing."""
import os, contextlib, tempfile, shutil, warnings


def cant_import(backend_name):
    if backend_name == 'numpy':
        return False
    elif backend_name == 'torch':
        try:
            import torch
        except ImportError:
            warnings.warn("Failed to import torch")
            return True
    elif backend_name == 'tensorflow':
        try:
            import tensorflow
        except ImportError:
            warnings.warn("Failed to import tensorflow")
            return True


@contextlib.contextmanager
def tempdir(dirpath=None):
    if dirpath is not None and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)
    elif dirpath is None:
        dirpath = tempfile.mkdtemp()
    else:
        os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


# tests to skip
SKIPS = {
  'jtfs': 1,
  'visuals': 1,
}
# used to load saved coefficient outputs
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
assert os.path.isdir(TEST_DATA_DIR), TEST_DATA_DIR
assert len(os.listdir(TEST_DATA_DIR)) > 0, os.listdir(TEST_DATA_DIR)
