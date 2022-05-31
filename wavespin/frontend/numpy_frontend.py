# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- The Kymatio developers
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the Modified BSD License
# (BSD 3-clause; see NOTICE.txt in the WaveSpin root directory for details).
# -----------------------------------------------------------------------------

class ScatteringNumPy:
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/frontend/
    numpy_frontend.py
    """
    def __init__(self):
        self.frontend_name = 'numpy'

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def __call__(self, x):
        """This method is an alias for `scattering`."""

        self.backend.input_checks(x)

        return self.scattering(x)

    _doc_array = 'np.ndarray'
    _doc_array_n = 'n'

    _doc_alias_name = '__call__'

    _doc_alias_call = ''

    _doc_frontend_paragraph = ''

    _doc_sample = 'np.random.randn({shape})'

    _doc_has_shape = True

    _doc_has_out_type = True