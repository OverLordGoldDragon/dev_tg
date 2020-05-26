# -*- coding: utf-8 -*-
import os
import sys
import inspect
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(inspect.stack()[0][1])
if sys.path[0] != filedir:
    if filedir in sys.path:
        sys.path.pop(sys.path.index(filedir))  # avoid dudplication
    sys.path.insert(0, filedir)

import pytest
import numpy as np

from backend import notify
from deeptrain.visuals import (
    viz_roc_auc,
    )


tests_done = {name: None for name in ('viz_roc_auc',)}


@notify(tests_done)
def test_viz_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    y_pred = np.random.uniform(0, 1, 32)
    viz_roc_auc(y_true, y_pred)


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
