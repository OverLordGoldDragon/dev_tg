# -*- coding: utf-8 -*-
import pytest
import numpy as np

from deeptrain.util.metrics import f1_score, f1_score_multi_th


np.random.seed(0)
to_test = ['f1_score',
           'f1_score-multi'
           ]




if __name__ == '__main__':
    pytest.main([__file__, "--capture=sys"])
