# -*- coding: utf-8 -*-
import os
os.environ['IS_MAIN'] = '1' * (__name__ == '__main__')
import pytest
import numpy as np

from tests.backend import notify
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
