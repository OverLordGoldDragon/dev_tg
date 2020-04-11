# -*- coding: utf-8 -*-
import pytest
import numpy as np

from deeptrain.util.visuals import (
    viz_roc_auc,
    )


def test_viz_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    y_pred = np.random.uniform(0, 1, 32)
    viz_roc_auc(y_true, y_pred)


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
