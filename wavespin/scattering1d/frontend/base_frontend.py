# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
from ...frontend.base_frontend import ScatteringBase
import math
import numbers
import warnings
from types import FunctionType
from copy import deepcopy

import numpy as np

from ..filter_bank import (scattering_filter_factory, periodize_filter_fourier,
                           energy_norm_filterbank_tm)
from ..filter_bank_jtfs import _FrequencyScatteringBase
from ..utils import (compute_border_indices, compute_padding,
                     compute_minimum_support_to_pad,
                     compute_meta_scattering, compute_meta_jtfs)
from ...toolkit import fill_default_args


class ScatteringBase1D(ScatteringBase):
    """
    This is a modification of
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
    frontend/base_frontend.py
    Kymatio, (C) 2018-present. The Kymatio developers.
    """
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=True,
                 oversampling=0, out_type='array', pad_mode='reflect',
                 max_pad_factor=1, analytic=False, normalize='l1-energy',
                 r_psi=math.sqrt(.5), backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J if isinstance(J, tuple) else (J, J)
        self.shape = shape
        self.Q = Q if isinstance(Q, tuple) else (Q, 1)
        self.T = T
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.out_type = out_type
        self.pad_mode = pad_mode
        self.max_pad_factor = max_pad_factor
        self.analytic = analytic
        self.normalize = (normalize if isinstance(normalize, tuple) else
                          (normalize, normalize))
        self.r_psi = r_psi if isinstance(r_psi, tuple) else (r_psi, r_psi)
        self.backend = backend

    def build(self):
        """Set up padding and filters

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        self.sigma0 = 0.1
        self.alpha = 4.
        self.P_max = 5
        self.eps = 1e-7
        self.criterion_amplitude = 1e-3

        # check the shape
        if isinstance(self.shape, numbers.Integral):
            self.N = self.shape
        elif isinstance(self.shape, tuple):
            self.N = self.shape[0]
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")
        # dyadic scale of N, also min possible padding
        self.N_scale = math.ceil(math.log2(self.N))

        # check `pad_mode`, set `pad_fn`
        if isinstance(self.pad_mode, FunctionType):
            def pad_fn(x):
                return self.pad_mode(x, self.pad_left, self.pad_right)
            self.pad_mode = 'custom'
        elif self.pad_mode not in ('reflect', 'zero'):
            raise ValueError(("unsupported `pad_mode` '{}';\nmust be a "
                              "function, or string, one of: 'zero', 'reflect'."
                              ).format(str(self.pad_mode)))
        else:
            def pad_fn(x):
                return self.backend.pad(x, self.pad_left, self.pad_right,
                                        self.pad_mode)
        self.pad_fn = pad_fn

        # check `normalize`
        supported = ('l1', 'l2', 'l1-energy', 'l2-energy')
        if any(n not in supported for n in self.normalize):
            raise ValueError(("unsupported `normalize`; must be one of: {}\n"
                              "got {}").format(supported, self.normalize))

        # ensure 2**max(J) <= nextpow2(N)
        Np2up = 2**self.N_scale
        if 2**max(self.J) > Np2up:
            raise ValueError(("2**J cannot exceed input length (rounded up to "
                              "pow2) (got {} > {})".format(
                                  2**max(self.J), Np2up)))

        # validate `max_pad_factor`
        # 1/2**J < 1/Np2up so impossible to create wavelet without padding
        if max(self.J) == self.N_scale and self.max_pad_factor == 0:
            raise ValueError("`max_pad_factor` can't be 0 if "
                             "max(J) == log2(nextpow2(N)). Got, "
                             "respectively, %s, %s, %s" % (
                                 self.max_pad_factor, max(self.J), self.N_scale))

        # check T or set default
        if self.T is None:
            self.T = 2**max(self.J)
        elif self.T == 'global':
            self.T == Np2up
        elif self.T > Np2up:
            raise ValueError(("The temporal support T of the low-pass filter "
                              "cannot exceed input length (got {} > {})"
                              ).format(self.T, self.N))
        # log2_T, global averaging
        self.log2_T = math.floor(math.log2(self.T))
        self.average_global_phi = bool(self.T == Np2up)
        self.average_global = bool(self.average_global_phi and self.average)

        # Compute the minimum support to pad (ideally)
        min_to_pad, pad_phi, pad_psi1, pad_psi2 = compute_minimum_support_to_pad(
            self.N, self.J, self.Q, self.T, r_psi=self.r_psi,
            sigma0=self.sigma0, alpha=self.alpha, P_max=self.P_max, eps=self.eps,
            criterion_amplitude=self.criterion_amplitude,
            normalize=self.normalize, pad_mode=self.pad_mode)
        if self.average_global:
            min_to_pad = max(pad_psi1, pad_psi2)  # ignore phi's padding

        J_pad_ideal = math.ceil(math.log2(self.N + 2 * min_to_pad))
        if self.max_pad_factor is None:
            self.J_pad = J_pad_ideal
        else:
            self.J_pad = min(J_pad_ideal, self.N_scale + self.max_pad_factor)
            if J_pad_ideal - self.J_pad > 1:
                extent_txt = ' severe' if J_pad_ideal - self.J_pad > 2 else ''
                warnings.warn(("Insufficient temporal padding, will yield"
                               "{} boundary effects and filter distortion; "
                               "recommended higher `max_pad_factor` or lower "
                               "`J` or `T`.").format(extent_txt))

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, 2**self.J_pad - self.pad_right)

        # record whether configuration yields second order filters
        meta = ScatteringBase1D.meta(self)
        self._no_second_order_filters = (self.max_order < 2 or
                                         bool(np.isnan(meta['n'][-1][1])))

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self.N, self.J_pad, self.J, self.Q, self.T,
            normalize=self.normalize,
            criterion_amplitude=self.criterion_amplitude,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha,
            P_max=self.P_max, eps=self.eps)

        # analyticity
        if self.analytic:
          for psi_fs in (self.psi1_f, self.psi2_f):
            for p in psi_fs:
              for k in p:
                if isinstance(k, int):
                    M = len(p[k])
                    p[k][M//2 + 1:] = 0  # zero negatives
                    p[k][M//2] /= 2      # halve Nyquist

        # energy norm
        # must do after analytic since analytic affects norm
        if any('energy' in n for n in self.normalize):
            energy_norm_filterbank_tm(self.psi1_f, self.psi2_f, phi_f=None,
                                      J=self.J, log2_T=self.log2_T,
                                      normalize=self.normalize)

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_scattering()` with the
        parameters of the transform object.

        Returns
        ------
        meta : dictionary
            See the documentation for `compute_meta_scattering()`.
        """
        return compute_meta_scattering(self.J_pad, self.J, self.Q, self.T,
                                       r_psi=self.r_psi, max_order=self.max_order)

    _doc_shape = 'N'

    _doc_instantiation_shape = {True: 'S = Scattering1D(J, N, Q)',
                                False: 'S = Scattering1D(J, Q)'}

    _doc_param_shape = \
        r"""
         shape : int
            The length of the input signals.
         """

    _doc_attrs_shape = \
        r"""J_pad : int
             The logarithm of the padded length of the signals.
         pad_left : int
             The amount of padding to the left of the signal.
         pad_right : int
             The amount of padding to the right of the signal.
         phi_f : dictionary
             A dictionary containing the lowpass filter at all resolutions. See
             `filter_bank.scattering_filter_factory` for an exact description.
         psi1_f : dictionary
             A dictionary containing all the first-order wavelet filters, each
             represented as a dictionary containing that filter at all
             resolutions. See `filter_bank.scattering_filter_factory` for an
             exact description.
         psi2_f : dictionary
             A dictionary containing all the second-order wavelet filters, each
             represented as a dictionary containing that filter at all
             resolutions. See `filter_bank.scattering_filter_factory` for an
             exact description.
         """

    _doc_param_average = \
        r"""
         average : boolean, optional
             Determines whether the output is averaged in time or not. The
             averaged output corresponds to the standard scattering transform,
             while the un-averaged output skips the last convolution by
             :math:`\phi_J(t)`.  This parameter may be modified after object
             creation. Defaults to `True`.
         """

    _doc_attr_average = \
        r"""
         average : boolean
             Controls whether the output should be averaged (the standard
             scattering transform) or not (resulting in wavelet modulus
             coefficients). Note that to obtain unaveraged output, the
             `vectorize` flag must be set to `False` or `out_type` must be set
             to `'list'`.
         """

    _doc_param_vectorize = \
        r"""
         vectorize : boolean, optional
             Determines wheter to return a vectorized scattering transform
             (that is, a large array containing the output) or a dictionary
             (where each entry corresponds to a separate scattering
             coefficient). This parameter may be modified after object
             creation. Deprecated in favor of `out_type` (see below). Defaults
             to True.
         out_type : str, optional
             The format of the output of a scattering transform. If set to
             `'list'`, then the output is a list containing each individual
             scattering coefficient with meta information. Otherwise, if set to
             `'array'`, the output is a large array containing the
             concatenation of all scattering coefficients. Defaults to
             `'array'`.
         pad_mode : str (default 'reflect') / function, optional
             Name of padding scheme to use, one of (`x = [1, 2, 3]`):
                 - zero:    [0, 0, 0, 1, 2, 3, 0, 0]
                 - reflect: [2, 3, 2, 1, 2, 3, 2, 1]
             Or, pad function with signature `pad_fn(x, pad_left, pad_right)`.
             This sets `self.pad_mode='custom'` (the name of padding is used
             for some internal logic).
         max_pad_factor : int (default 2) / None, optional
             Will pad by at most `2**max_pad_factor` relative to
             `nextpow2(shape)`.
             E.g. if input length is 150, then maximum padding with
             `max_pad_factor=2` is `256 * (2**2) = 1024`.
             The maximum realizable value is `4`: a filter of scale `scale`
             requires `2**(scale + 4)` samples to convolve without boundary
             effects, and with fully decayed wavelets - i.e. x16 the scale,
             and the largest permissible `J` or `log2_T` is `log2(N)`.

             `None` means limitless. A limitation with `analytic=True` is,
             `compute_minimum_support_to_pad` does not account for
             `analytic=True`.
         normalize : str / tuple[str], optional
             Tuple sets first-order and second-order separately, but only the
             first element sets `normalize` for `phi_f`. Supported:

                 - 'l1': bandpass normalization; all filters' amplitude envelopes
                   sum to 1 in time domain (for Morlets makes them peak at 1
                   in frequency domain). `sum(abs(psi)) == 1`.
                 - 'l2': energy normalization; all filters' energies are 1
                   in time domain; not suitable for scattering.
                   `sum(abs(psi)**2) == 1`.
                 - 'l1-energy', 'l2-energy': additionally renormalizes the
                   entire filterbank such that its LP-sum (overlap of
                   frequency-domain energies) is `<=1` (`<=2` for time scattering
                   per using only analytic filters, without anti-analytic).

                   - This improves "even-ness" of input's representation, i.e.
                     no frequency is tiled too great or little (amplified or
                     attenuated).
                   - `l2-energy` is self-defeating, as the `energy` part
                     reverts to `l1`.
                   - `phi` is excluded from norm computations, against the
                     standard. This is because lowpass functions separately from
                     bandpass in coefficient computations, and any `log2_T`
                     that differs from `J` will attenuate or amplify lowest
                     frequencies in an undesired manner. In rare cases, this
                     *is* desired, and can be achieved by calling relevant
                     functions manually.
         r_psi : float / tuple[float], optional
             Should be >0 and <1. Controls the redundancy of the filters
             (the larger r_psi, the larger the overlap between adjacent wavelets),
             and stability against time-warp deformations
             (larger r_psi improves it).
             Defaults to sqrt(0.5).
             Tuple sets separately for first- and second-order filters.
         """

    _doc_attr_vectorize = \
        r"""
         vectorize : boolean
             Controls whether the output should be vectorized into a single
             Tensor or collected into a dictionary. Deprecated in favor of
             `out_type`. For more details, see the documentation for
             `scattering`.
         out_type : str
             Specifices the output format of the transform, which is currently
             one of `'array'` or `'list`'. If `'array'`, the output is a large
             array containing the scattering coefficients. If `'list`', the
             output is a list of dictionaries, each containing a scattering
             coefficient along with meta information. For more information, see
             the documentation for `scattering`.
         pad_mode : str
             One of supported padding modes: 'reflect', 'zero' - or 'custom'
             if a function was passed.
         pad_fn : function
             A backend padding function, or user function (as passed
             to `pad_mode`), with signature `pad_fn(x, pad_left, pad_right)`.
         max_pad_factor : int (default 2) / None, optional
             Will pad by at most `2**max_pad_factor` relative to
             `nextpow2(shape)`.
             E.g. if input length is 150, then maximum padding with
             `max_pad_factor=2` is `256 * (2**2) = 1024`.
             The maximum realizable value is `4`: a filter of scale `scale`
             requires `2**(scale + 4)` samples to convolve without boundary
             effects, and with fully decayed wavelets - i.e. x16 the scale,
             and the largest permissible `J` or `log2_T` is `log2(N)`.

             `None` means limitless. A limitation with `analytic=True` is,
             `compute_minimum_support_to_pad` does not account for
             `analytic=True`.
         analytic : bool (default False)
             If True, will force negative frequencies to zero. Useful if
             strict analyticity is desired, but may worsen time-domain decay.
         average_global_phi : bool
             True if `T == nextpow2(shape)`, i.e. `T` is maximum possible
             and equivalent to global averaging, in which case lowpassing is
             replaced by simple arithmetic mean.

             In case of `average==False`, controls scattering logic for
             `phi_t` pairs in JTFS.
         average_global : bool
             True if `average_global_phi and average_fr`. Same as
             `average_global_phi` if `average_fr==True`.

             In case of `average==False`, controls scattering logic for
             `psi_t` pairs in JTFS.
         """

    _doc_class = \
        r"""
         The 1D scattering transform

         The scattering transform computes a cascade of wavelet transforms
         alternated with a complex modulus non-linearity. The scattering
         transform of a 1D signal :math:`x(t)` may be written as

             $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

         where

             $S_J^{{(0)}} x(t) = x \star \phi_J(t)$,

             $S_J^{{(1)}} x(t, \lambda) = |x \star \psi_\lambda^{{(1)}}|
             \star \phi_J$, and

             $S_J^{{(2)}} x(t, \lambda, \mu) = |\,| x \star \psi_\lambda^{{(1)}}|
             \star \psi_\mu^{{(2)}} | \star \phi_J$.

         In the above formulas, :math:`\star` denotes convolution in time. The
         filters $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
         wavelets with center frequencies $\lambda$ and $\mu$, while
         $\phi_J(t)$ is a real lowpass filter centered at the zero frequency.

         The `Scattering1D` class implements the 1D scattering transform for a
         given set of filters whose parameters are specified at initialization.
         While the wavelets are fixed, other parameters may be changed after
         the object is created, such as whether to compute all of
         :math:`S_J^{{(0)}} x`, $S_J^{{(1)}} x$, and $S_J^{{(2)}} x$ or just
         $S_J^{{(0)}} x$ and $S_J^{{(1)}} x$.
         {frontend_paragraph}
         Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
         number of signals to transform (the batch size) and `N` is the length
         of the signal, we compute its scattering transform by passing it to
         the `scattering` method (or calling the alias `{alias_name}`). Note
         that `B` can be one, in which case it may be omitted, giving an input
         of shape `(N,)`.

         Example
         -------
         ::

             # Set the parameters of the scattering transform.
             J = 6
             N = 2 ** 13
             Q = 8

             # Generate a sample signal.
             x = {sample}

             # Define a Scattering1D object.
             {instantiation}

             # Calculate the scattering transform.
             Sx = S.scattering(x)

             # Equivalently, use the alias.
             Sx = S{alias_call}(x)

         Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while
         the maximum scale ratio of the scattering transform is set to
         :math:`2^J = 2^6 = 64`. The time-frequency resolution of the first-order
         wavelets :math:`\psi_\lambda^{{(1)}}(t)` is set to `Q = 8` wavelets per
         octave. The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` always have
         one wavelet per octave.

         Parameters
         ----------
         J : int / tuple[int]
             Controls the maximum log-scale and number of octaves of the
             scattering transform. There are approx. `Q` wavelets per octave, and
             bandwidth halves with each octave. Hence, largest scale wavelet is
             about `2**J` larger than smallest scale wavelet.

             Tuple sets `J1` and `J2` separately, for first-order and second-order
             scattering, respectively.
         {param_shape}Q : int >= 1 / tuple[int]
             The number of first-order wavelets per octave. Defaults to `1`.
             If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
             second-order wavelets per octave (which defaults to `1`).
                 - Q1: For audio signals, a value of `>= 12` is recommended in
                   order to separate partials.
                 - Q2: Recommended `1` for most (`Scattering1D`) applications.
                 - Greater Q also corresponds to greater scale for all wavelets.
         T : int / str['global']
             Temporal width of low-pass filter, controlling amount of imposed
             time-shift invariance and maximum subsampling.
             'global' for global average pooling (simple arithmetic mean),
             which also eases on padding (ignores `phi_f`'s requirement).
         max_order : int, optional
             The maximum order of scattering coefficients to compute. Must be
             either `1` or `2`. Defaults to `2`.
         {param_average}oversampling : integer >= 0, optional
             Controls the oversampling factor relative to the default as a
             power of two. Since the convolving by wavelets (or lowpass
             filters) and taking the modulus reduces the high-frequency content
             of the signal, we can subsample to save space and improve
             performance. However, this may reduce precision in the
             calculation. If this is not desirable, `oversampling` can be set
             to a large value to prevent too much subsampling. This parameter
             may be modified after object creation.
             Defaults to `0`. Has no effect if `average_global=True`.
         {param_vectorize}
         Attributes
         ----------
         J : int / tuple[int]
             Controls the maximum log-scale and number of octaves of the
             scattering transform. There are approx. `Q` wavelets per octave,
             and bandwidth halves with each octave. Hence, largest scale wavelet
             is about `2**J` larger than smallest scale wavelet.

             Tuple sets `J1` and `J2` separately, for first-order and second-order
             scattering, respectively.
         {param_shape}Q : int >= 1 / tuple[int]
             The number of first-order wavelets per octave. Defaults to `1`.
             If tuple, sets `Q = (Q1, Q2)`, where `Q2` is the number of
             second-order wavelets per octave (which defaults to `1`).
                 - Q1: For audio signals, a value of `>= 12` is recommended in
                   order to separate partials.
                 - Q2: Recommended `1` for most (`Scattering1D`) applications.
         T : int
             Temporal width of low-pass filter, controlling amount of imposed
             time-shift invariance and maximum subsampling.
             'global' for global average pooling (simple arithmetic mean),
             which also eases on padding (ignores `phi_f`'s requirement).
         {attrs_shape}max_order : int
             The maximum scattering order of the transform.
         {attr_average}oversampling : int
             The number of powers of two to oversample the output compared to
             the default subsampling rate determined from the filters.
         {attr_vectorize}

         References
         ----------
         This is a modification of
         https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
         frontend/base_frontend.py
         Kymatio, (C) 2018-present. The Kymatio developers.
         """

    _doc_scattering = \
        """
        Apply the scattering transform

        Given an input `{array}` of size `(B, N)`, where `B` is the batch
        size (it can be potentially an integer or a shape) and `N` is the length
        of the individual signals, this function computes its scattering
        transform. If the `vectorize` flag is set to `True` (or if it is not
        available in this frontend), the output is in the form of a `{array}`
        or size `(B, C, N1)`, where `N1` is the signal length after subsampling
        to the scale :math:`2^J` (with the appropriate oversampling factor to
        reduce aliasing), and `C` is the number of scattering coefficients. If
        `vectorize` is set `False`, however, the output is a dictionary
        containing `C` keys, each a tuple whose length corresponds to the
        scattering order and whose elements are the sequence of filter indices
        used.

        Note that the `vectorize` flag has been deprecated in favor of the
        `out_type` parameter. If this is set to `'array'` (the default), the
        `vectorize` flag is still respected, but if not, `out_type` takes
        precedence. The two current output types are `'array'` and `'list'`.
        The former gives the type of output described above. If set to
        `'list'`, however, the output is a list of dictionaries, each
        dictionary corresponding to a scattering coefficient and its associated
        meta information. The coefficient is stored under the `'coef'` key,
        while other keys contain additional information, such as `'j'` (the
        scale of the filter used) and `'n`' (the filter index).

        Furthermore, if the `average` flag is set to `False`, these outputs
        are not averaged, but are simply the wavelet modulus coefficients of
        the filters.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)`.

        Returns
        -------
        S : tensor or dictionary
            If `out_type` is `'array'` and the `vectorize` flag is `True`, the
            output is a{n} `{array}` containing the scattering coefficients,
            while if `vectorize` is `False`, it is a dictionary indexed by
            tuples of filter indices. If `out_type` is `'list'`, the output is
            a list of dictionaries as described above.

        References
        ----------
        This is a modification of
        https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/
        frontend/base_frontend.py
        Kymatio, (C) 2018-present. The Kymatio developers.
        """

    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_average = cls._doc_param_average if cls._doc_has_out_type else ''
        attr_average = cls._doc_attr_average if cls._doc_has_out_type else ''
        param_vectorize = (cls._doc_param_vectorize if cls._doc_has_out_type else
                           '')
        attr_vectorize = cls._doc_attr_vectorize if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase1D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call,
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_average=param_average,
            attr_average=attr_average,
            param_vectorize=param_vectorize,
            attr_vectorize=attr_vectorize,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        cls.scattering.__doc__ = ScatteringBase1D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


class TimeFrequencyScatteringBase1D():
    SUPPORTED_KWARGS = {'aligned', 'out_3D', 'sampling_filters_fr', 'analytic_fr',
                        'F_kind', 'max_pad_factor_fr', 'pad_mode_fr',
                        'normalize_fr', 'r_psi_fr', 'oversampling_fr',
                        'max_noncqt_fr', 'out_exclude', 'paths_exclude'}
    DEFAULT_KWARGS = dict(
        aligned=None, out_3D=False, sampling_filters_fr=('exclude', 'resample'),
        analytic_fr=True, F_kind='average', max_pad_factor_fr=2,
        pad_mode_fr='zero', normalize_fr='l1-energy',
        r_psi_fr=math.sqrt(.5), oversampling_fr=0, max_noncqt_fr=None,
        out_exclude=None, paths_exclude=None,
    )

    def __init__(self, J_fr=None, Q_fr=2, F=None, average_fr=False,
                 out_type='array', implementation=None, **kwargs):
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.F = F
        self.average_fr = average_fr
        self.out_type = out_type
        self.implementation = implementation
        self.kwargs = kwargs

    def build(self):
        """Check args and instantiate `_FrequencyScatteringBase` object
        (which builds filters).

        Certain internal data, such as the amount of padding and the wavelet
        filters to be used in the scattering transform, need to be computed
        from the parameters given during construction. This function is called
        automatically during object creation and no subsequent calls are
        therefore needed.
        """
        # if config yields no second order coeffs, we cannot do joint scattering
        if self._no_second_order_filters:
            raise ValueError("configuration yields no second-order filters; "
                             "try increasing `J`")

        # handle `implementation` ############################################
        # validate
        if self.implementation is not None:
            if len(self.kwargs) > 0:
                raise ValueError("if `implementation` is passed, `**kwargs` must "
                                 "be empty; got\n%s" % self.kwargs)
            elif not (isinstance(self.implementation, int) and
                      self.implementation in range(1, 6)):
                raise ValueError("`implementation` must be None, or an integer "
                                 "1-5; got %s" % str(self.implementation))

        # fill defaults
        if len(self.kwargs) > 0:
            I = fill_default_args(self.kwargs, self.default_kwargs,
                                  copy_original=True)
        else:
            I = self.default_kwargs
        # handle `None`s
        if I['aligned'] is None:
            not_recalibrate = bool(I['sampling_filters_fr'] not in
                                   ('recalibrate', ('recalibrate', 'recalibrate'))
                                   )
            I['aligned'] = bool(not_recalibrate and I['out_3D'])

        # store for reference
        self.kwargs_filled = deepcopy(I)

        for name in TimeFrequencyScatteringBase1D.SUPPORTED_KWARGS:
            setattr(self, name, I.pop(name))

        # invalid arg check
        if len(I) != 0:
            raise ValueError("unknown kwargs:\n{}\nSupported are:\n{}".format(
                I, TimeFrequencyScatteringBase1D.SUPPORTED_KWARGS))

        # define presets
        self._implementation_presets = {
            1: dict(average_fr=False, aligned=False, out_3D=False,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            2: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='array'),
            3: dict(average_fr=True,  aligned=True,  out_3D=True,
                    sampling_filters_fr=('exclude', 'resample'),
                    out_type='dict:list'),
            4: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('exclude', 'recalibrate'),
                    out_type='array'),
            5: dict(average_fr=True,  aligned=False, out_3D=True,
                    sampling_filters_fr=('recalibrate', 'recalibrate'),
                    out_type='dict:list'),
        }
        # override defaults with presets
        if isinstance(self.implementation, int):
            for k, v in self._implementation_presets[self.implementation].items():
                setattr(self, k, v)

        ######################################################################
        # `out_structure`
        if isinstance(self.implementation, int) and self.implementation in (3, 5):
            self.out_structure = 3
        else:
            self.out_structure = None

        # handle `out_exclude`
        if self.out_exclude is not None:
            if isinstance(self.out_exclude, str):
                self.out_exclude = [self.out_exclude]
            # ensure all names are valid
            supported = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                         'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
            for name in self.out_exclude:
                if name not in supported:
                    raise ValueError(("'{}' is an invalid coefficient name; "
                                      "must be one of: {}").format(
                                          name, ', '.join(supported)))

        # handle `F`
        if self.F is None:
            # default to one octave (Q wavelets per octave, J octaves,
            # approx Q*J total frequency rows, so averaging scale is `Q/total`)
            # F is processed further in `_FrequencyScatteringBase`
            self.F = self.Q[0]

        # handle `max_noncqt_fr`
        if self.max_noncqt_fr is not None:
            if not isinstance(self.max_noncqt_fr, (str, int)):
                raise ValueError("`max_noncqt_fr` must be str, int, or None")
            if self.max_noncqt_fr == 'Q':
                self.max_noncqt_fr = self.Q[0] // 2

        # frequential scattering object ######################################
        self._N_frs = self.get_N_frs()
        # number of psi1 filters
        self._n_psi1_f = len(self.psi1_f)
        max_order_fr = 1

        self.scf = _FrequencyScatteringBase(
            self._N_frs, self.J_fr, self.Q_fr, self.F, max_order_fr,
            self.average_fr, self.aligned, self.oversampling_fr,
            self.sampling_filters_fr, self.out_type, self.out_3D,
            self.max_pad_factor_fr, self.pad_mode_fr, self.analytic_fr,
            self.max_noncqt_fr, self.normalize_fr, self.F_kind, self.r_psi_fr,
            self._n_psi1_f, self.backend)
        self.finish_creating_filters()
        self.handle_paths_exclude()

        # detach __init__ args, instead access `scf`'s via `__getattr__` #####
        # this is so that changes in attributes are reflected here
        init_args = ('J_fr', 'Q_fr', 'F', 'average_fr', 'oversampling_fr',
                     'sampling_filters_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                     'r_psi_fr', 'out_3D')
        for init_arg in init_args:
            delattr(self, init_arg)

        # sanity warning #####################################################
        try:
            self.meta()
        except:
            warnings.warn(("Failed to build meta; the implementation may be "
                           "faulty. Try another configuration, or call "
                           "`jtfs.meta()` to debug."))

    def get_N_frs(self):
        """This is equivalent to `len(x)` along frequency, which varies across
        `psi2`, so we compute for each.
        """
        def is_cqt_if_need_cqt(n1):
            if self.max_noncqt_fr is None:
                return True
            return n_non_cqts[n1] <= self.max_noncqt_fr

        n_non_cqts = np.cumsum([not p['is_cqt'] for p in self.psi1_f])

        N_frs = []
        for n2 in range(len(self.psi2_f)):
            j2 = self.psi2_f[n2]['j']
            max_freq_nrows = 0
            if j2 != 0:
                for n1 in range(len(self.psi1_f)):
                    if j2 > self.psi1_f[n1]['j'] and is_cqt_if_need_cqt(n1):
                        max_freq_nrows += 1

                # add rows for `j2 >= j1` up to `nextpow2` of current number
                # to not change frequential padding scales
                # but account for `cqt_fr`
                max_freq_nrows_at_2gt1 = max_freq_nrows
                p2up_nrows = int(2**math.ceil(math.log2(max_freq_nrows_at_2gt1)))
                for n1 in range(len(self.psi1_f)):
                    if (j2 == self.psi1_f[n1]['j'] and
                            max_freq_nrows < p2up_nrows and
                            is_cqt_if_need_cqt(n1)):
                        max_freq_nrows += 1
            N_frs.append(max_freq_nrows)
        return N_frs

    def finish_creating_filters(self):
        """Handles necessary adjustments in time scattering filters unaccounted
        for in default construction.
        """
        # ensure phi is subsampled up to log2_T for `phi_t * psi_f` pairs
        max_sub_phi = lambda: max(k for k in self.phi_f if isinstance(k, int))
        while max_sub_phi() < self.log2_T:
            self.phi_f[max_sub_phi() + 1] = periodize_filter_fourier(
                self.phi_f[0], nperiods=2**(max_sub_phi() + 1))

        # for early unpadding in joint scattering
        # copy filters, assign to `0` trim (time's `subsample_equiv_due_to_pad`)
        phi_f = {0: [v for k, v in self.phi_f.items() if isinstance(k, int)]}
        # copy meta
        for k, v in self.phi_f.items():
            if not isinstance(k, int):
                phi_f[k] = v

        diff = min(max(self.J) - self.log2_T, self.J_pad - self.N_scale)
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                # subsample in Fourier <-> trim in time
                phi_f[trim_tm] = [v[::2**trim_tm] for v in phi_f[0]]
        self.phi_f = phi_f

        # adjust padding
        ind_start = {0: {k: v for k, v in self.ind_start.items()}}
        ind_end   = {0: {k: v for k, v in self.ind_end.items()}}
        if diff > 0:
            for trim_tm in range(1, diff + 1):
                pad_left, pad_right = compute_padding(self.J_pad - trim_tm,
                                                      self.N)
                start, end = compute_border_indices(
                    self.log2_T, self.J, pad_left, pad_left + self.N)
                ind_start[trim_tm] = start
                ind_end[trim_tm] = end
        self.ind_start, self.ind_end = ind_start, ind_end

    def meta(self):
        """Get meta information on the transform

        Calls the static method `compute_meta_jtfs()` with the parameters of the
        transform object.

        Returns
        ------
        meta : dictionary
            See `help(wavespin.scattering1d.utils.compute_meta_jtfs)`.
        """
        return compute_meta_jtfs(self.J_pad, self.J, self.Q, self.T, self.r_psi,
                                 self.sigma0, self.average, self.average_global,
                                 self.average_global_phi, self.oversampling,
                                 self.out_exclude, self.paths_exclude, self.scf)

    @property
    def fr_attributes(self):
        """Exposes `scf`'s attributes via main object."""
        return ('J_fr', 'Q_fr', 'N_frs', 'N_frs_max', 'N_frs_min',
                'N_fr_scales_max', 'N_fr_scales_min', 'scale_diffs', 'psi_ids',
                'J_pad_frs', 'J_pad_frs_max', 'J_pad_frs_max_init',
                'average_fr', 'average_fr_global', 'aligned', 'oversampling_fr',
                'F', 'log2_F', 'max_order_fr', 'max_pad_factor_fr', 'out_3D',
                'sampling_filters_fr', 'sampling_psi_fr', 'sampling_phi_fr',
                'phi_f_fr', 'psi1_f_fr_up', 'psi1_f_fr_dn')

    @property
    def default_kwargs(self):
        return deepcopy(TimeFrequencyScatteringBase1D.DEFAULT_KWARGS)

    def __getattr__(self, name):
        # access key attributes via frequential class
        # only called when default attribute lookup fails
        # `hasattr` in case called from Scattering1D
        if name in self.fr_attributes and hasattr(self, 'scf'):
            return getattr(self.scf, name)
        raise AttributeError(f"'{type(self).__name__}' object has no "
                             f"attribute '{name}'")  # standard attribute error

    def handle_paths_exclude(self):
        """
          - Ensures `paths_exclude` is dict and doesn't have unsupported keys
          - Ensures the provided n and j aren't out of bounds
          - Handles negative indexing
          - Handles `key: int` (expected `key: list[int]`)
          - "Converts" from j to n (fills all 'n' that have the specified 'j')
        """
        supported = {'n2', 'n1_fr', 'j2', 'j1_fr'}
        if self.paths_exclude is None:
            self.paths_exclude = {nm: [] for nm in supported}
            return
        elif not isinstance(self.paths_exclude, dict):
            raise ValueError("`paths_exclude` must be dict, got %s" % type(
                self.paths_exclude))

        psis = {'n2': self.psi2_f, 'n1_fr': self.psi1_f_fr_up}
        # fill what's missing as we can't change size of dict during iteration
        for nm in supported:
            if nm not in self.paths_exclude:
                self.paths_exclude[nm] = []

        # iterate n's first to avoid duplicate j2=0 warnings and appending
        # to integer values (if user provided them)
        for p_name in ('n2', 'n1_fr', 'j2', 'j1_fr'):
            # ensure all keys are functional
            assert p_name in supported, (p_name, supported)
            # ensure list
            if isinstance(self.paths_exclude[p_name], int):
                self.paths_exclude[p_name] = [self.paths_exclude[p_name]]
            else:
                try:
                    self.paths_exclude[p_name] = list(self.paths_exclude[p_name])
                except:
                    raise ValueError(("`paths_exclude` values must be list[int] "
                                      "or int, got paths_exclude['{}'] type: {}"
                                      ).format(p_name,
                                               type(self.paths_exclude[p_name])))

            # n2, n1_fr ######################################################
            if p_name[0] == 'n':
                for i, n in enumerate(self.paths_exclude[p_name]):
                    # handle negative
                    if n < 0:
                        self.paths_exclude[p_name][i] = len(psis[p_name]) + n

                    # warn if 'n2' already excluded
                    if p_name == 'n2':
                        n = self.paths_exclude[p_name][i]
                        n_j2_0 = [n2 for n2 in range(len(self.psi2_f))
                                  if self.psi2_f[n2]['j'] == 0]
                        if n in n_j2_0:
                            warnings.warn(
                                ("`paths_exclude['n2']` includes `{}`, which "
                                 "is already excluded (alongside {}) per "
                                 "having j2==0."
                                 ).format(n, ', '.join(map(str, n_j2_0))))
            # j2, j1_fr ######################################################
            elif p_name[0] == 'j':
                for i, j in enumerate(self.paths_exclude[p_name]):
                    # fetch all j
                    if p_name == 'j2':
                        j_all = {p['j'] for p in self.psi2_f}
                    elif p_name == 'j1_fr':
                        j_all = set(self.psi1_f_fr_up['j'][0])

                    # handle negative
                    if j < 0:
                        j = max(j_all) + j
                    # forbid out of bounds
                    if j > max(j_all):
                        raise ValueError(("`paths_exclude` exceeded maximum {}: "
                                          "{} > {}\nTo specify max j, use `-1`"
                                          ).format(p_name, j, max(j_all)))
                    # warn if 'j2' already excluded
                    elif p_name == 'j2' and j == 0:
                        warnings.warn(("`paths_exclude['j2']` includes `0`, "
                                       "which is already excluded."))

                    # convert to n ###########################################
                    # fetch all n that have the specified j
                    if p_name == 'j2':
                        n_j_all = [n2 for n2, p in enumerate(self.psi2_f)
                                   if p['j'] == j]
                    elif p_name == 'j1_fr':
                        n_j_all = [n1_fr for n1_fr in
                                   range(len(self.psi1_f_fr_up[0]))
                                   if self.psi1_f_fr_up['j'][0][n1_fr] == j]

                    # append if not already present
                    n_name = 'n2' if p_name == 'j2' else 'n1_fr'
                    for n_j in n_j_all:
                        if n_j not in self.paths_exclude[n_name]:
                            self.paths_exclude[n_name].append(n_j)

    # docs ###################################################################
    @classmethod
    def _document(cls):
        cls.__doc__ = TimeFrequencyScatteringBase1D._doc_class.format(
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_call=cls._doc_alias_call,
            parameters=cls._doc_params,
            attributes=cls._doc_attrs,
            sample=cls._doc_sample.format(shape=cls._doc_shape),
            terminology=cls._terminology,
        )
        cls.scattering.__doc__ = (
            TimeFrequencyScatteringBase1D._doc_scattering.format(
                array=cls._doc_array,
                n=cls._doc_array_n,
            )
        )

    def output_size(self):
        raise NotImplementedError("Not implemented for JTFS.")

    def create_filters(self):
        raise NotImplementedError("Implemented in `_FrequencyScatteringBase`.")

    _doc_class = \
        r"""
        The 1D Joint Time-Frequency Scattering transform.

        JTFS builds on time scattering by convolving first order coefficients
        with joint 2D wavelets along time and frequency, increasing
        discriminability while preserving time-shift invariance and time-warp
        stability. Invariance to frequency transposition can be imposed via
        frequential averaging, while preserving sensitivity to
        frequency-dependent time shifts.

        Joint wavelets are defined separably in time and frequency and permit fast
        separable convolution. Convolutions are followed by complex modulus and
        optionally averaging.

        The JTFS of a 1D signal :math:`x(t)` may be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_{{J, J_{{fr}}}}^{{(0)}} x(t) = x \star \phi_T(t),$

            $S_{{J, J_{{fr}}}}^{{(1)}} x(t, \lambda) =
            |x \star \psi_\lambda^{{(1)}}| \star \phi_T,$ and

            $S_{{J, J_{{fr}}}}^{{(2)}} x(t, \lambda, \mu, l, s) =
            ||x \star \psi_\lambda^{{(1)}}| \star \Psi_{{\mu, l, s}}|
            \star \Phi_{{T, F}}.$

        $\Psi_{{\mu, l, s}}$ comprises of five kinds of joint wavelets:

            $\Psi_{{\mu, l, +1}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(+\lambda)$
            spin up bandpass

            $\Psi_{{\mu, l, -1}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \psi_{{l, s}}(-\lambda)$
            spin down bandpass

            $\Psi_{{\mu, -\infty, 0}}(t, \lambda) =
            \psi_\mu^{{(2)}}(t) \phi_F(\lambda)$
            temporal bandpass, frequential lowpass

            $\Psi_{{-\infty, l, 0}}(t, \lambda) =
            \phi_T(t) \psi_{{l, s}}(\lambda)$
            temporal lowpass, frequential bandpass

            $\Psi_{{-\infty, -\infty, 0}}(t, \lambda)
            = \phi_T(t) \phi_F(\lambda)$
            joint lowpass

        and $\Phi_{{T, F}}$ optionally does temporal and/or frequential averaging:

            $\Phi_{{T, F}}(t, \lambda) = \phi_T(t) \phi_F(\lambda)$

        Above, :math:`\star` denotes convolution in time and/or frequency. The
        filters $\psi_\lambda^{{(1)}}(t)$ and $\psi_\mu^{{(2)}}(t)$ are analytic
        wavelets with center frequencies $\lambda$ and $\mu$, while
        $\phi_T(t)$ is a real lowpass filter centered at the zero frequency.
        $\psi_{{l, s}}(+\lambda)$ is like $\psi_\lambda^{{(1)}}(t)$ but with
        its own parameters (center frequency, support, etc), and an anti-analytic
        complement (spin up is analytic).

        Filters are built at initialization. While the wavelets are fixed, other
        parameters may be changed after the object is created, such as `out_type`.

        {frontend_paragraph}
        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 6
            N = 2 ** 13
            Q = 8

            # Generate a sample signal.
            x = {sample}

            # Define a `TimeFrequencyScattering1D` object.
            jtfs = TimeFrequencyScattering1D(J, N, Q)

            # Calculate the scattering transform.
            Sx = jtfs(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Above, the length of the signal is :math:`N = 2^{{13}} = 8192`, while the
        maximum scale of the scattering transform is set to :math:`2^J = 2^6 =
        64`. The time-frequency resolution of the first-order wavelets
        :math:`\psi_\lambda^{{(1)}}(t)` is set to `Q = 8` wavelets per octave.
        The second-order wavelets :math:`\psi_\mu^{{(2)}}(t)` have one wavelet
        per octave by default, but can be set like `Q = (8, 2)`. Internally,
        `J_fr` and `Q_fr`, the frequential variants of `J` and `Q`, are defaulted,
        but can be specified as well.

        For further description and visuals, refer to:

            - https://dsp.stackexchange.com/a/78623/50076
            - https://dsp.stackexchange.com/a/78625/50076

        {parameters}

        {attributes}

        {terminology}
        """

    _doc_params = \
        r"""
        Parameters
        ----------
        J, shape, T, average, oversampling, pad_mode :
            See `help(wavespin.scattering1d.Scattering1D)`.

            Unlike in time scattering, `T` plays a role even if `average=False`,
            to compute `phi_t` pairs.

        J : int / tuple[int]
            (Extended docs for JTFS)

            Greater `J1` extends time-warp stability to lower frequencies, and
            other desired properties, as greater portion of the transform is CQT
            (fixed `xi` to `sigma` ratio and both exponentially spaced, as opposed
            to fixed `sigma` and linearly spaced `xi`). The number of CQT rows
            is approx `(J1 - 1)*Q1` (last octave is non-CQT), so the ratio of CQT
            to non-CQT is `(J1 - 1)/J1`, which is greater if `J1` is greater.

        Q : int / tuple[int]
            `(Q1, Q2)`, where `Q2=1` if `Q` is int. `Q1` is the number of
            first-order wavelets per octave, and `Q2` the second-order.

              - `Q1`, together with `J`, determines `N_frs_max` and `N_frs`,
                or length of inputs to frequential scattering.
              - `Q2`, together with `J`, determines `N_frs` (via the `j2 >= j1`
                criterion), and total number of joint slices.
              - Greater `Q2` values better capture temporal AM modulations (AMM)
                of multiple rates. Suited for inputs of multirate or intricate AM.
                `Q2=2` is in close correspondence with the mamallian auditory
                cortex: https://asa.scitation.org/doi/full/10.1121/1.1945807
                2 or 1 should work for most purposes.
              - Greater `Q` also corresponds to greater scale for all wavelets.

        J_fr : int
            Controls the maximum log-scale of frequential scattering in joint
            scattering transform, and number of octaves of frequential filters.
            There are approx. `Q_fr` wavelets per octave, and bandwidth halves
            with each octave. Hence, largest scale wavelet is about `2**J_fr`
            larger than smallest scale wavelet.

            Default is determined at instantiation from longest frequential row
            in frequential scattering, set to `log2(nextpow2(N_frs_max)) - 2`,
            i.e. maximum possible minus 2, but no less than 3, and no more than
            max.

        Q_fr : int
            Number of wavelets per octave for frequential scattering.

            Greater values better capture quefrential variations of multiple rates
            - that is, variations and structures along frequency axis of the
            wavelet transform's 2D time-frequency plane. Suited for inputs of many
            frequencies or intricate AM-FM variations. 2 or 1 should work for
            most purposes.

        F : int / str['global'] / None
            Temporal support of frequential low-pass filter, controlling amount of
            imposed frequency transposition invariance and maximum frequential
            subsampling. Defaults to `Q1`, i.e. one octave.

              - If `'global'`, sets to maximum possible `F` based on `N_frs_max`.
              - Used even with `average_fr=False` (see its docs); this is likewise
                true of `T` for `phi_t * phi_f` and `phi_t * psi_f` pairs.

        average_fr : bool (default False)
            Whether to average (lowpass) along frequency axis.

            If `False`, `phi_t * phi_f` and `psi_t * phi_f` pairs are still
            computed.

        out_type : str, optional
            Affects output format (but not how coefficients are computed).
            See `help(TimeFrequencyScattering1D.scattering)` for further info.

                - 'list': coeffs are packed in a list of dictionaries, each dict
                  storing meta info, and output tensor keyed by `'coef.`.
                - 'array': concatenated along slices (`out_3D=True`) or mixed
                  slice-frequency dimension (`out_3D=False`). Both require
                  `average=True` (and `out_3D=True` additionally
                  `average_fr=True`).
                - 'dict:list' || 'dict:array': same as 'array' and 'list', except
                  coefficients will not be concatenated across pairs - e.g.
                  tensors from `'S1'` will be kept separate from those from
                  `'phi_t * psi_f'`.
                - See `out_3D` for all behavior controlled by `out_3D`, and
                  `aligned` for its behavior and interactions with `out_3D`.

        kwargs : dict
            Keyword arguments controlling advanced configurations.
            See `help(TimeFrequencyScattering1D.SUPPORTED_KWARGS)`.
            These args are documented below.

        implementation : int / None / dict
            Preset configuration to use. Overrides the following parameters:

                - `average_fr, aligned, out_3D, sampling_filters_fr, out_type`

            Defaults to `None`, and any `None` argument above will default to
            that of `TimeFrequencyScatteringBase1D.DEFAULT_KWARGS`.
            See `help(wavespin.toolkit.pack_coeffs_jtfs)` for further information.

            **Implementations:**

                1: Standard for 1D convs. `(n1_fr * n2 * n1, t)`.
                  - average_fr = False
                  - aligned = False
                  - out_3D = False
                  - sampling_psi_fr = 'exclude'
                  - sampling_phi_fr = 'resample'

                2: Standard for 2D convs. `(n1_fr * n2, n1, t)`.
                  - average_fr = True
                  - aligned = True
                  - out_3D = True
                  - sampling_psi_fr = 'exclude'
                  - sampling_phi_fr = 'resample'

                3: Standard for 3D/4D convs. `(n1_fr, n2, n1, t)`. [2] but
                  - out_structure = 3

                4: Efficient for 2D convs. [2] but
                  - aligned = False
                  - sampling_phi_fr = 'recalibrate'

                5: Efficient for 3D convs. [3] but
                  - aligned = False
                  - sampling_psi_fr = 'recalibrate'
                  - sampling_phi_fr = 'recalibrate'

            `'exclude'` in `sampling_psi_fr` can be replaced with `'resample'`,
            which yields significantly more coefficients and doens't lose
            information (which `'exclude'` strives to minimize), but is slower
            and the coefficients are mostly "synthetic zeros" and uninformative.

            `out_structure` refers to packing output coefficients via
            `pack_coeffs_jtfs(..., out_structure)`. This zero-pads and reshapes
            coefficients, but does not affect their values or computation in any
            way. (Thus, 3==2 except for shape). Requires `out_type` 'dict:list'
            (default) or 'dict:array'; if 'dict:array' is passed, will use it
            instead.

            `5` also makes sense with `sampling_phi_fr = 'resample'` and small `F`
            (small enough to let `J_pad_frs` drop below max), but the argument
            will only set `'recalibrate'`.

        aligned : bool / None
            Defaults to True if `sampling_filters_fr != 'recalibrate'` and
            `out_3D=True`.

            If True, rows of joint slices index to same frequency for all slices.
            E.g. `S_2[3][5]` and `S_2[4][5]` (fifth row of third and fourth joint
            slices) correspond to same frequency. With `aligned=True`:

              - `out_3D=True`: all slices are zero-padded to have same number of
                rows. Earliest (low `n2`, i.e. high second-order freq) slices are
                likely to be mostly zero per `psi2` convolving with minority of
                first-order coefficients.
              - `out_3D=False`: all slices are padded by minimal amount needed to
                avert boundary effects.
                  - `average_fr=True`: number of output frequency rows will vary
                    across slices but be same *per `psi2_f`*.
                  - `average_fr=False`: number of rows will vary across and within
                    slices (`psi1_f_fr_up`-to-`psi1_f_fr_up`, and down).

            For any config, `aligned=True` enforces same total frequential stride
            for all slices, while `aligned=False` uses stride that maximizes
            information richness and density.
            See "Compute logic: stride, padding" in `core`, specifically
            'recalibrate'

            Note: `sampling_psi_fr = 'recalibrate'` breaks global alignment per
            shifting `xi_frs`, but preserves it on per-`N_fr_scale` basis.

            **Illustration**:

            Intended usage is `aligned=True` && `sampling_filters_fr='resample'`
            and `aligned=False` && `sampling_filters_fr='recalibrate'`. Below
            example assumes these.

            `x` == zero; `0, 4, ...` == indices of actual (nonpadded) data.
            That is, `x` means the convolution kernel (wavelet or lowpass) is
            centered in the padded region and contains less (or no) information,
            whereas `4 ` centers at `input[4]`. And `input` is `U1`, so the
            numbers are indexing `xi1` (i.e. are `n1`).

            ::

                data -> padded
                16   -> 128
                64   -> 128

                False:
                  [0,  4,  8, 16]  # stride 4
                  [0, 16, 32, 48]  # stride 16

                True:
                  [0,  x,  x,  x]  # stride 16
                  [0, 16, 32, 48]  # stride 16

            `False` is more information rich, containing fewer `x`. Further,
            with `out_3D=False`, it allows `stride_fr > log2_F`, making it more
            information dense
            (same info with fewer datapoints <=> non-oversampled).

            In terms of unpadding with `out_3D=True`:

                - `aligned=True`: we always have fr stride == `log2_F`, with
                  which we index `ind_start_fr_max` and `ind_end_fr_max`
                  (i.e. take maximum of all unpads across `n2` from this factor
                  and reuse it across all other `n2`).
                - `aligned=False`: decided from `N_fr_scales_max` case, where we
                  compute `unpad_len_common_at_max_fr_stride`. Other `N_fr_scales`
                  use that quantity to compute `min_stride_to_unpad_like_max`.
                  See "Compute logic: stride, padding" in `core`, specifically
                  'recalibrate'.

            The only exception is with `average_fr_global_phi and not average_fr`:
            spinned pairs will have zero stride, but `phi_f` pairs will have max.

        out_3D : bool (default False)
            `True` (requires `average_fr=True`) adjusts frequential scattering
            to enable concatenation along joint slices dimension, as opposed to
            flattening (mixing slices and frequencies):

                - `False` will unpad freq by exact amounts for each joint slice,
                  whereas `True` will unpad by minimum amount common to all
                  slices at a given subsampling factor to enable concatenation.
                  See `scf_compute_padding_fr()`.
                - See `aligned` for its interactions with `out_3D` (also below).

            Both `True` and `False` can still be concatenated into the 'true' JTFS
            4D structure; see `help(wavespin.toolkit.pack_coeffs_jtfs)` for a
            complete description. The difference is in how values are computed,
            especially near boundaries. More importantly, `True` enforces
            `aligned=True` on *per-`n2`* basis, enabling 3D convs even with
            `aligned=False`.

            `aligned` and `out_3D`
            ----------------------
            From an information/classification standpoint,

              - `True` is more information-rich. The 1D equivalent case is
                unpadding by 3, instead of by 6 and then zero-padding by 3: same
                final length, but former fills gaps with partial convolutions
                where latter fills with zeros.
              - `False` is the "latter" case.

            We emphasize the above distinction. `out_3D=True` && `aligned=True`
            imposes a large compute overhead by padding all `N_frs` maximally.
            If a given `N_fr` is treated as a complete input, then unpadding
            anything more than `N_fr/stride` includes convolutions from completely
            outside of this input, which we never do elsewhere.

              - However, we note, if `N_fr=20` for `n2=2` and `N_frs_max=100`,
                what this really says is, we *expect* the 80 lowest frequency rows
                to yield negligible energy after convolving with `psi2_f`.
                That is, zeros (i.e. padding) are the *true* continuation of the
                input (hence why 'conj-reflect-zero'), and hence, unpadding by
                more than `N_fr/stride` is actually within bounds.
              - Hence, unpadding by `N_fr/stride` and then re-padding (i.e.
                treating `N_fr` as a complete input) is actually a distortion and
                is incorrect.
                Namely, the complete scattering, without any
                shortcuts/optimizations on stride or padding, is consistent with
                unpadding `> N_fr/stride`.
                At the same time, depending on our feature goals, especially if
                slices are processed independently, such distortion might be
                preferable to avoid air-packing (see "Illustration" in `aligned`).
              - The described re-padding happens with `aligned=True` &&
                `out_3D=False` packed into a 3D/4D tensor; even without
                re-padding, this config tosses out valid outputs
                (unpads to `N_fr/stride`), though less informative ones.

        sampling_filters_fr : str / tuple[str]
            Controls filter properties for frequential input lengths (`N_frs`)
            below maximum.

              - 'resample': preserve physical dimensionality
                (center frequeny, width) at every length (trimming in time
                domain).
                E.g. `psi = psi_fn(N/2) == psi_fn(N)[N/4:-N/4]`.

              - 'recalibrate': recalibrate filters to each length.

                - widths (in time): widest filter is halved in width, narrowest is
                  kept unchanged, and other widths are re-distributed from the
                  new minimum to same maximum.
                - center frequencies: all redistribute between new min and max.
                  New min is set as `2 / new_length`
                  (old min was `2 / max_length`).
                  New max is set by halving distance between old max and 0.5
                  (greatest possible), e.g. 0.44 -> 0.47, then 0.47 -> 0.485, etc.

              - 'exclude': same as 'resample' except filters wider than
                `widest / 2` are excluded. (and `widest / 4` for next
                `N_fr_scales`, etc).

            Tuple can set separately `(sampling_psi_fr, sampling_phi_fr)`, else
            both set to same value.

            From an information/classification standpoint:

                - 'resample' enforces freq invariance imposed by `phi_f_fr` and
                  physical scale of extracted modulations by `psi1_f_fr_up`
                  (& down). This is consistent with scattering theory and is the
                  standard used in existing applied literature.
                - 'recalibrate' remedies a problem with 'resample'. 'resample'
                  calibrates all filters relative to longest input; when the
                  shortest input is very different in comparison, it makes most
                  filters appear lowpass-ish. In contrast, recalibration enables
                  better exploitation of fine structure over the smaller interval
                  (which is the main motivation behind wavelets,
                  a "multi-scale zoom".)
                - 'exclude' circumvents the problem by simply excluding wide
                  filters. 'exclude' is simply a subset of 'resample', preserving
                  all center frequencies and widths - a 3D/4D coefficient packing
                  will zero-pad to compensate
                  (see `help(wavespin.toolkit.pack_coeffs_jtfs)`).

            Note: `sampling_phi_fr = 'exclude'` will re-set to `'resample'`, as
            `'exclude'` isn't a valid option (there must exist a lowpass for every
            fr input length).

        analytic_fr : bool (default True)
            If True, will enforce strict analyticity/anti-analyticity:
                - zero negative frequencies for temporal and spin up bandpasses
                - zero positive frequencies for spin down bandpasses
                - halve the Nyquist bin for both spins

            `True` improves FDTS-discriminability, especially for
            `r_psi > sqrt(.5)`, but may slightly worsen wavelet time decay.

        F_kind : str['average', 'decimate']
            Kind of lowpass filter to use for spinned coefficients:

                - 'average': Gaussian, standard for scattering. Imposes time-shift
                  invariance.
                - 'decimate': Hamming-windowed sinc (~brickwall in freq domain).
                  Decimates coefficients: used for unaliased downsampling,
                  without imposing invariance.
                   - Preserves more information along frequency than 'average'.
                   - Ignores padding specifications and pads its own way
                     (future TODO)
                   - Corrects negative outputs via absolute value; the negatives
                     are possible since the kernel contains negatives, but are in
                     minority and are small in magnitude.

            Does not interact with other parameters in any way - that is, won't
            affect stride, padding, etc - only changes the lowpass filter for
            spinned pairs. `phi_f` pairs will still use Gaussian, and `phi_f_fr`
            remains Gaussian but is used only for `phi_f` pairs. Has no effect
            with `average_fr=False`.

            'decimate' is an experimental but tested feature:
                - 'torch' backend:
                    - will assume GPU use and move built filters to GPU
                    - lacks `register_filters` support, so filters are invisible
                      to `nn.Module`
                - filters are built dynamically, on per-requested basis. The first
                  run is slower than the rest as a result
                - `oversampling_fr != 0` is not supported
                - is differentiable

            Info preservation
            -----------------
            'decimate'
              - 1) Increases amount of information preserved.
                  - Its cutoff spills over the alias threshold, and there's
                    notable amount of aliasing (subject to future improvement).
                  - Its main lobe is narrower than Gauss's, hence better
                    preserving component separation along frequency, at expense
                    of longer tails.
                  - Limited reconstruction experiments did not reveal a definitive
                    advantage over Gaussian: either won depending on transform and
                    optimizer configurations. Further study is required.

              - 2) Reduces distortion of preserved information.
                  - The Gaussian changes relative scalings of bins, progressively
                    attenuating higher frequencies, whereas windowed sinc is ~flat
                    in frequency until reaching cutoff (i.e. it copies input's
                    spectrum). As a result, Gaussian blurs, while sinc faithfully
                    captures the original.
                  - At the same time, sinc increases distortion per aliasing, but
                    the net effect is a benefit.

              - 3) Increases distortion of preserved information.
                  - Due to notable aliasing. Amount of energy aliased is ~1/110 of
                    total energy, while for Kymatio's Gaussian, it's <1/1000000.
                  - Due to the time-domain kernel having negatives, which
                    sometimes outputs negatives for a non-negative input,
                    requiring correction.
                  - 2) benefits much more than 3) harms

            2) is the main advantage and is the main motivation for 'decimate': we
            want a smaller unaveraged output, that resembles the full original.

        max_pad_factor_fr : int / None (default) / list[int], optional
            `max_pad_factor` for frequential axis in frequential scattering.

                - None: unrestricted; will pad as much as needed.
                - list[int]: controls max padding for each `N_fr_scales`
                  separately, in reverse order (max to min).
                    - Values may not be such that they yield increasing
                      `J_pad_frs`
                    - If the list is insufficiently long (less than number of
                      scales), will extend list with the last provided value
                      (e.g. `[1, 2] -> [1, 2, 2, 2]`).
                    - Indexed by `scale_diff == N_fr_scales_max - N_fr_scales`
                - int: will convert to list[int] of same value.

            Specified values aren't guaranteed to be realized. They override some
            padding values, but are overridden by others.

            Overrides:
                - Padding that lessens boundary effects and wavelet distortion
                  (`min_to_pad`).

            Overridden by:
                - `J_pad_frs_min_limit_due_to_phi`
                - `J_pad_frs_min_limit_due_to_psi`
                - Will not allow any `J_pad_fr > J_pad_frs_max_init`
                - With `sampling_psi_fr = 'resample'`, will not allow `J_pad_fr`
                  that yields a pure sinusoid wavelet (raises `ValueError` in
                  `filter_bank.get_normalizing_factor`).

            A limitation of `None` with`analytic=True` is,
            `compute_minimum_support_to_pad` does not account for it.

        pad_mode_fr : str['zero', 'conj-reflect-zero'] / function
            Name of frequential padding mode to use, one of: 'zero',
            'conj-reflect-zero'.
            Or, function with signature `pad_fn_fr(x, pad_fr, scf, B)`;
            see `_right_pad` in
            `wavespin.scattering1d.core.timefrequency_scattering1d`.

            If using `pad_mode = 'reflect'` and `average = True`, reflected
            portions will be automatically conjugated before frequential
            scattering to avoid spin cancellation. For same reason, there isn't
            `pad_mode_fr = 'reflect'`.

            'zero' is default only because it's faster; in general, if
            `J_fr >= log2(N_frs_max) - 3`, 'conj-reflect-zero' should be
            preferred.
            See https://github.com/kymatio/kymatio/discussions/
            752#discussioncomment-864234

            Also, note that docs and comments tend to mention only `J, J_fr` and
            `T, F`, but `Q, Q_fr` also significantly affect max scale: higher ->
            greater max scale.

        normalize_fr : str
            See `normalize` in `help(wavespin.scattering1d.Scattering1D)`.
            Applies to `psi1_f_fr_up`, `psi1_f_fr_dn`, `phi_f_fr`.

        r_psi_fr : float
            See `r_psi` in `help(wavespin.scattering1d.Scattering1D)`.
            See `help(wavespin.scattering1d.utils.calibrate_scattering_filters)`.

        oversampling_fr : int (default 0)
            How much to oversample along frequency axis (with respect to
            `2**J_fr`).
            Also see `oversampling` in `Scattering1D`.
            Has no effect if `average_fr_global=True`.

        max_noncqt_fr : int / None / str['Q']
            Maximum non-CQT rows (`U1` vectors) to include in frequential
            scattering, i.e. rows derived from `not psi1_f[n1]['is_cqt']`.

              - `0` means CQT-only; `3` means *up to* 3 rows (rather than
                *at least*)
                for any given `N_fr` (see `N_frs`).
              - `None` means all non-CQT are permitted
              - `'Q'` means up to `Q1//2` non-CQT are permitted

            Non-CQT rows are sub-ideal for frequential scattering, as they violate
            the basic assumption of convolution that the input is uniformly
            spaced.
            CQT rows are uniformly spaced in log-space, non-CQT in linear space,
            so the two aren't directly compatible and form a discontinuity
            boundary.

              - This lowers FDTS discriminability, albeit not considerably.
              - It also affects frequency transposition invariance and time-warp
                stability, as a shift in log space is a shift by different amount
                in linear (& fixed wavelet bandwidth) space. The extent is again
                acceptable.
              - At the same time, excluding such rows loses information.
              - `max_noncqt_fr` can control this tradeoff, but in general, `None`
                (the default) is recommended.
              - Higher `J` (namely `J1`) increases the CQT portion (see `J`),
                mediating aforementioned effects.

        out_exclude : list/tuple[str] / None
            Will exclude coefficients with these names from computation and output
            (except for `S1`, which always computes but still excludes from
            output).
            All names (JTFS pairs, except 'S0', 'S1'):

                - 'S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
                  'psi_t * psi_f_up', 'psi_t * psi_f_dn'

        paths_exclude : dict[str: list[int]] / dict[str: int] / None
            Will exclude coefficients with these paths from computation and
            output.
            Supported keys: 'n2', 'n1_fr', 'j2', 'j1_fr'. E.g.:

                - {'n2': [2, 3, 5], 'n1_fr': [0, -1]}
                - {'j2': [1], 'j1_fr': [3, 1]}
                - {'n2': [0, 1], 'j2': [-1]}

            Excluding `j2==1` paths yields greatest speedup, and is recommended
            in compute-restricted settings, as they're the lowest energy paths
            (i.e. generally least informative).

            `dict[str: int]` will convert to `dict[str: list[int]`.
        """

    _doc_attrs = \
        r"""
        Attributes
        ----------
        scf : `_FrequencyScatteringBase`
            Frequential scattering object, storing pertinent attributes and
            filters. Temporal scattering's attributes are accessed directly via
            `self`.

            "scf" abbreviates "scattering frequency" (i.e. frequential
            scattering).

        N_frs : list[int]
            List of lengths of frequential columns (i.e. numbers of frequential
            rows) in joint scattering, indexed by `n2` (second-order temporal
            wavelet idx).
            E.g. `N_frs[3]==52` means 52 highest-frequency vectors from
            first-order time scattering are fed to `psi2_f[3]` (effectively, a
            multi-input network).

        N_frs_max : int
            `== max(N_frs)`.

        N_frs_min : int
            `== min(N_frs_realized)`

        N_frs_realized: list[int]
            `N_frs` without `0`s.
            Unaffected by `paths_exclude` to allow `paths_exclude` to be
            dynamically configurable.

        N_frs_max_all : int
            `== _n_psi1_f`. Used to compute `_J_pad_frs_fo` (unused quantity),
            and `n_zeros` in `_pad_conj_reflect_zero` (`core/timefreq...`).

        N_fr_scales : list[int]
            `== nextpow2(N_frs)`. Filters are calibrated relative to these
            (for 'exclude' & 'recalibrate' `sampling_psi_fr`).

        N_fr_scales_max : int
            `== max(N_fr_scales)`. Used to set `J_pad_frs_max` and
            `J_pad_frs_max_init`.

                - `J_fr` default is set using this value, and `J_fr` cannot
                  exceed it. If `F == 2**J_fr`, then `average_fr_global=True`.
                - Used in `compute_J_pad_fr()` and `psi_fr_factory()`.

        N_fr_scales_min : int
            `== min(N_fr_scales)`.

            Used in `scf._compute_J_pad_frs_min_limit_due_to_psi`.

        N_fr_scales_unique : list[int]
            `N_fr_scales` without duplicate entries.

        scale_diffs : list[int]
            `scale_diff == N_fr_scales_max - N_fr_scale`.
            0-indexed surrogate for `N_fr_scale`, indexing multi-length logic
            for building filterbanks and computing stride and padding.

        scale_diffs_unique : list[int]
            `scale_diffs` without duplicate entries.

        scale_diff_max_recalibrate : int / None
            Max permitted `scale_diff`, per `sampling_psi_fr='recalibrate'`
            and `sigma_max_to_min_max_ratio`. Build terminates to avoid filters
            more time-localized than the most time-localized original wavelet
            (max sigma in freq domain), within set tolerance, as a quality check.

        total_conv_stride_over_U1s : dict[int: list[int]]
            Stores total strides for frequential scattering (`psi_f` pairs,
            followed by `phi_f_fr`):
                {scale_diff: [stride0, stride1, ...]}  # list indexed by `n1_fr`

            `J_pad_frs` is built to accomodate stride.
            See `help(scf.compute_stride_fr)`.
            See "Compute logic: stride, padding" in
            `core.timefrequency_scattering1d`.

            `over_U1` seeks to emphasize that it is the stride over first order
            coefficients.

        total_conv_stride_over_U1s_phi : dict[int: int]
            Stores strides for frequential scattering (`phi_f` pairs):
                {scale_diff: stride}

            Derives from `total_conv_stride_over_U1s`, differently depending on
            `average_fr`, `aligned`, and `sampling_phi_fr`.
            See "Stride, padding: `phi_f` pairs" in
            `core.timefrequency_scattering1d`.

        n1_fr_subsamples : dict[str: dict[int: list[int]]]
            Stores strides for frequential scattering (`psi_f` pairs).
            Accounts for both `j1_fr` and `log2_F_phi`, so subsampling won't alias
            the lowpass.

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            See `scf._compute_scale_and_stride_logic`.

        log2_F_phis : dict[str: dict[int: list[int]]]
            `log2_F`-equivalent - that is, maximum permitted subsampling, and
            dyadic scale of invariance - of lowpass filters used for a given
            pair, `N_fr_scale`, and `n1_fr`:

                {'spinned: {scale_diff: [...]},
                 'phi':    {scale_diff: [...]}}

            Equals `log2_F` everywhere with `sampling_phi_fr='resample'`.
            Is `None` for 'spinned' if
            `not (average_fr and not average_fr_global)`
            (since convolution with lowpass isn't used).

        log2_F_phi_diffs : dict[str: dict[int: list[int]]]
            `== log2_F - log2_F_phi`. See `log2_F_phis`.

        unpad_len_common_at_max_fr_stride : int
            Unpad length at `N_fr_scales_max`, with whatever frequential stride
            happens to be there. Used when `out_3D=True` && `aligned=False` to set
            unpad length for other `N_fr_scales`, via
            `min_stride_to_unpad_like_max`.

            See "Compute logic: stride, padding" in
            `core.timefrequency_scattering1d`, specifically 'recalibrate'

        phi_f_fr : dict[int: dict[int: list[tensor[float]]],
                        str: dict[int: dict[int: list[int]], float]]
            Contains the frequential lowpass filter at all resolutions.
            See `help(wavespin.scattering1d.filter_bank.phi_fr_factory)`.

        psi1_f_fr_up : dict[int: list[tensor[float]],
                            str: dict[int: list[int/float]]]
            List of dictionaries containing all frequential scattering filters
            with "up" spin.
            See `help(wavespin.scattering1d.filter_bank.psi_fr_factory)`.

        psi1_f_fr_dn : dict[int: list[tensor[float]],
                            str: dict[int: list[int/float]]]
            `psi1_f_fr_up`, but with "down" spin, forming a complementary pair.

        psi_ids : dict[int: int]
            See `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

        psi_fr_params : dict[int:dict[str:list]]
            Parameters used to build filterbanks for frequential scattering.
            See `help(scf._compute_psi_fr_params)` and
            `help(wavespin.scattering1d.filter_bank_jtfs.psi_fr_factory)`.

        average_fr_global_phi : bool
            True if `F == nextpow2(N_frs_max)`, i.e. `F` is maximum possible
            and equivalent to global averaging, in which case lowpassing is
            replaced by simple arithmetic mean.

            If True, `sampling_phi_fr` has no effect.

            In case of `average_fr==False`, controls scattering logic for
            `phi_f` pairs.

        average_fr_global : bool
            True if `average_fr_global_phi and average_fr`. Same as
            `average_fr_global_phi` if `average_fr==True`.

              - In case of `average_fr==False`, controls scattering logic for
                `psi_f` pairs.
              - If `True`, `phi_fr` filters are never used (but are still
                created).
              - Results are very close to lowpassing w/ `F == 2**N_fr_scales_max`.
                Unlike with such lowpassing, `psi_fr` filters are allowed to be
                created at lower `J_pad_fr` than shortest `phi_fr` (which also is
                where greatest deviation with `not average_fr_global` occurs).

        log2_F : int
            Equal to `log2(prevpow2(F))`; is the maximum frequential subsampling
            factor if `average_fr=True` (otherwise that factor is up to `J_fr`).

        J_pad_frs : list[int]
            log2 of padding lengths of frequential columns in joint scattering
            (column lengths given by `N_frs`). See `scf.compute_padding_fr()`.

        J_pad_frs_max_init : int
            Set as reference for computing other `J_pad_fr`.

            Serves to create the initial frequential filterbank, and equates to
            `J_pad_frs_max` with `sampling_psi_fr='resample'` &&
            `sampling_phi_fr='resample'`. Namely, it is the maximum padding under
            "standard" frequential scattering configurations.

        J_pad_frs_max : int
            `== max(J_pad_frs)`.

        J_pad_frs_min : int
            `== min(J_pad_frs)` (excluding -1).

        J_pad_frs_min_limit_due_to_psi: int / None
            Controls minimal padding.
            Prevents severe filterbank distortions due to insufficient padding.
            See docs for `_compute_J_pad_frs_min_limit_due_to_psi`,
            in `filter_bank_jtfs.py`.

        _J_pad_fr_fo : int
            Padding for the `phi_t` pairs. Used only in edge case testing,
            and to warn of an edge case handling in `core`.

            `phi_t` pairs reuse spinned pairs' largest padding, yet the `N_fr` of
            `phi_t` pairs is always greater than or equal to that of spinned's,
            which at times otherwise yields greater padding.
            This is done to simplify implementation, with minimal or negligible
            effect on `phi_t` pairs.

            `core` edge case: `max_pad_factor_fr=0` with
            `N_fr_scales_max < N_fr_scale_fo` means the padded length will be
            less than `_n_psi1_f`. Accounting for this requires changing
            `J_pad_frs_max_init`, yet `compute_padding_fr` doesn't reuse
            `J_pad_frs_max_init`, hence accounting for this is complicated and
            unworthwhile. Instead, will only include up to `2**N_fr_scales_max`
            rows from `U1`.

        min_to_pad_fr_max : int
            `min_to_pad` from `compute_minimum_support_to_pad(N=N_frs_max)`.
            Used in computing `J_pad_fr`. See `scf.compute_J_pad_fr()`.

        unrestricted_pad_fr : bool
            `True` if `max_pad_factor is None`. Affects padding computation and
            filter creation:

              - `phi_f_fr` w/ `sampling_phi_fr=='resample'`:

                - `True`: will limit the shortest `phi_f_fr` to avoid distorting
                  its time-domain shape
                - `False`: will compute `phi_f_fr` at every `J_pad_fr`

              - `psi_f_fr` w/ `sampling_psi_fr=='resample'`: same as for phi

        subsample_equiv_relative_to_max_pad_init : int
            Amount of *equivalent subsampling* of frequential padding relative to
            `J_pad_frs_max_init`, indexed by `n2`.
            See `help(scf.compute_padding_fr())`.

        scale_diff_max_to_build: int / None
            Largest `scale_diff` (smallest `N_fr_scale`) for which a filterbank
            will be built; lesser `N_fr_scales` will reuse it. Used alongside
            other attributes to control said building, also as an additional
            sanity and clarity layer.
            Prevents severe filterbank distortions due to insufficient padding.

              - Affected by `sampling_psi_fr`, padding, and filterbank param
                choices.
                See docs for `_compute_J_pad_frs_min_limit_due_to_psi`,
                in `filter_bank_jtfs.py`.
              - With 'recalibrate', `scale_diff_max_to_build=None` if build didn't
                terminate per `sigma_max_to_min_max_ratio`.

        sigma_max_to_min_max_ratio : float >= 1
            Largest permitted `max(sigma) / min(sigma)`. Used with 'recalibrate'
            `sampling_psi_fr` to restrict how large the smallest sigma can get.

            Worst cases (high `subsample_equiv_due_to_pad`):
              - A value of `< 1` means a lower center frequency will have
                the narrowest temporal width, which is undesired.
              - A value of `1` means all center frequencies will have the same
                temporal width, which is undesired.
              - The `1.2` default was chosen arbitrarily as a seemingly good
                compromise between not overly restricting sigma and closeness to
                `1`.

        _n_phi_f_fr : int
            `== len(phi_f_fr)`.
            Used for setting `max_subsample_equiv_before_phi_fr`.

        pad_left_fr : int
            Amount of padding to left  of frequential columns
            (or top of joint matrix). Unused in implementation; can be used
            by user if `pad_mode` is a function.

        pad_right_fr : int
            Amount of padding to right of frequential columns
            (or bottom of joint matrix).

        ind_start_fr : list[list[int]]
            Frequential unpadding start index, indexed by `n2` (`N_fr`) and
            stride:

                `ind_start_fr[n2][stride]`

            See `help(scf.compute_padding_fr)` and `scf.compute_unpadding_fr`.

        ind_end_fr : list[list[int]]
            Frequential unpadding end index. See `ind_start_fr`.

        ind_start_fr_max : list[int]
            Frequential unpadding start index common to all `N_frs` for
            `out_3D=True`, determined from `N_frs_max` case, indexed by stride:

                `ind_start_fr_max[stride]`

            See `ind_start_fr`.

        ind_end_fr_max : list[int]
            Frequential unpadding end index common to all `N_frs` for
            `out_3D=True`.
            See `ind_start_fr_max`.

        r_psi : tuple[float]
            Temporal redundancy, first- and second-order.

        r_psi_fr : float
            Frequential redundancy.

        max_order_fr : int == 1
            Frequential scattering's `max_order`. Unused.
        """

    _terminology = \
        r"""
        Terminoloy
        ----------
        FDTS :
            Frequency-Dependent Time Shift. JTFS's main purpose is to detect
            these. Up spin wavelet resonates with up chirp (rising; right-shifts
            with increasing freq), down spin with down chirp (left-shifts with
            increasing freq).

            In convolution (cross-correlation with flipped kernel), the roles are
            reversed; the implementation will yield high values for up chirp
            from down spin.

        Frequency transposition :
            i.e. frequency shift, except in context of wavelet transform (hence
            scattering) it means log-frequency shift.

        n1_fr_subsample, n2 : int, int
            See `help(wavespin.scattering1d.core.timefrequency_scattering1d)`.
            Not attributes. Summary:

                - n1_fr_subsample: subsampling done after convolving with `psi_fr`
                - n2: index of temporal wavelet in joint scattering, like
                  `psi2[n2]`.
        """

    _doc_scattering = \
        """
        Apply the Joint Time-Frequency Scattering transform.

        Given an input `{array}` of size `(B, N)`, where `B` is the batch size
        and `N` is the length of the individual signals, computes its JTFS.

        Output format is specified by `out_type`: a list, array, tuple, or
        dictionary of lists or arrays with keys specifying coefficient names as
        follows:

        ::

            {{'S0': ...,                # (time)  zeroth order
             'S1': ...,                # (time)  first order
             'phi_t * phi_f': ...,     # (joint) joint lowpass
             'phi_t * psi_f': ...,     # (joint) time lowpass (w/ freq bandpass)
             'psi_t * phi_f': ...,     # (joint) freq lowpass (w/ time bandpass)
             'psi_t * psi_f_up': ...,  # (joint) spin up
             'psi_t * psi_f_dn': ...,  # (joint) spin down
             }}

        Coefficient structure depends on `average, average_fr, aligned, out_3D`,
        and `sampling_filters_fr`. See `help(wavespin.toolkit.pack_coeffs_jtfs)`
        for a complete description.

        Parameters
        ----------
        x : {array}
            An input `{array}` of size `(B, N)` or `(N,)`.

        Returns
        -------
        S : dict[tensor/list] / tensor/list / tuple of former two
            See above.
        """


__all__ = ['ScatteringBase1D', 'TimeFrequencyScatteringBase1D']
