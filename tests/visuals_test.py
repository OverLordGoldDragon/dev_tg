# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from pathlib import Path
from termcolor import cprint

from deeptrain.visuals import (
    viz_roc_auc,
    )


tests_done = {name: None for name in ('viz_roc_auc',)}


def test_viz_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    y_pred = np.random.uniform(0, 1, 32)
    viz_roc_auc(y_true, y_pred)

    _notify('viz_roc_auc')


def _notify(name):
    tests_done[name] = True
    print("\n>%s TEST PASSED" % name.upper())

    if all(tests_done.values()):
        test_name = Path(__file__).stem.replace('_', ' ').upper()
        cprint(f"<< {test_name} PASSED >>\n", 'green')


if __name__ == '__main__':
    os.environ['IS_MAIN'] = '1'
    pytest.main([__file__, "-s"])
