# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Convenience utilities."""
import numpy as np
import scipy.signal
import warnings
from scipy.fft import fft, ifft
from itertools import zip_longest, chain
from copy import deepcopy


def drop_batch_dim_jtfs(Scx, sample_idx=0):
    """Index into dim0 with `sample_idx` for every JTFS coefficient, and
    drop that dimension.

    Doesn't modify input:
        - dict/list: new list/dict (with copied meta if applicable)
        - array: new object but shared storage with original array (so original
          variable reference points to unindexed array).
    """
    fn = lambda x: x[sample_idx]
    return _iterate_apply(Scx, fn)


def jtfs_to_numpy(Scx):
    """Convert PyTorch/TensorFlow tensors to numpy arrays, with meta copied,
    and without affecting original data structures.
    """
    B = ExtendedUnifiedBackend(Scx)
    return _iterate_apply(Scx, B.numpy)


def _iterate_apply(Scx, fn):
    def get_meta(s):
        return {k: v for k, v in s.items() if not hasattr(v, 'ndim')}

    if isinstance(Scx, dict):
        out = {}  # don't modify source dict
        for pair in Scx:
            if isinstance(Scx[pair], list):
                out[pair] = []
                for i, s in enumerate(Scx[pair]):
                    out[pair].append(get_meta(s))
                    out[pair][i]['coef'] = fn(s['coef'])
            else:
                out[pair] = fn(Scx[pair])
    elif isinstance(Scx, list):
        out = []  # don't modify source list
        for s in Scx:
            o = get_meta(s)
            o['coef'] = fn(s['coef'])
            out.append(o)
    elif isinstance(Scx, tuple):  # out_type=='array' && out_3D==True
        out = (fn(Scx[0]), fn(Scx[1]))
    elif hasattr(Scx, 'ndim'):
        out = fn(Scx)
    else:
        raise ValueError(("unrecognized input type: {}; must be as returned by "
                          "`jtfs(x)`.").format(type(Scx)))
    return out


def normalize(X, mean_axis=(1, 2), std_axis=(1, 2), C=None, mu=1, C_mult=None):
    """Log-normalize + (optionally) standardize coefficients for learning
    algorithm suitability.

    Is a modification of Eq. 10 of https://arxiv.org/pdf/2007.10926.pdf
    For exact match (minus temporal global averaging), set
    `mean_axis=std_axis=(0, 2)`.

    Parameters
    ----------
    X : tensor
        Nonnegative tensor with dimensions `(samples, features, spatial)`.
        If there's more than one `features` or `spatial` dimensions, flatten
        before passing.
        (Obtain tensor via e.g. `pack_coeffs_jtfs(Scx)`, or `out_type='array'`.)

    std_axis : tuple[int] / int / None
        If not None, will unit-variance after `rscaling` along specified axes.

    mean_axis : tuple[int] / int / None
        If not None, will zero-mean before `rscaling` along specified axes.

    C : float / None
        `log(1 + X * C / median)`.
        Greater will bring more disparate values closer. Too great will equalize
        too much, too low will have minimal effect.

        Defaults to `5 / sparse_mean(abs(X / mu))`, which should yield moderate
        contraction for a variety of signals. This was computed on a mixture
        of random processes, with outliers, and may not generalize to all signals.

            - `sparse_mean` takes mean over non-negligible points, aiding
              consistency between representations. A scalogram with an extra
              octave, for example, may capture nothing in the new octave,
              while a simple mean would lower the output, attenuating existing
              values.

    mu : float / None
        In case precomputed; See "Online computation".

        `mu=None` will compute `mu` for per-channel normalization, while
        `mu=1` essentially disables `mu` and preserves channels' relative scaling;
        see "Relative scaling".

    C_mult : float / None
        Multiplies `C`. Useful if the default `C` compute scheme is appropriate
        but needs adjusting. Defaults to `5` if `C` is None, else to `1`.

    Returns
    -------
    Xnorm : tensor
        Normalized `X`.

    Relative scaling
    ----------------

    Scaling `features` independently changes the relative norms bewteen them.

      - If a signal rarely has high frequencies and low are dominant, for example,
        then post-normalization this nuance is lost and highs and lows are brought
        to a common norm - which may be undesired.
      - SNR is lowered, as low signal contents that are dominated by noise
        or float inaccuracies are amplified.
      - Convolutions over `features` dims are invalidated (as it's akin to
        standardizing individual time steps in 1D convolution); e.g. if
        normalizing on per-`n1` basis, then we can no longer do 2D convs
        over the joint `(n1, time)` pairs.
      - To keep convs valid, all spatial dims that are convolved over must be
        standardized by the same factor - i.e. same `mean` and `std`. `rscaling`
        also accounts for rescaling due to log.

    Regardless, this "channel normalization" has been used with success in
    variuous settings; above are but points worth noting.

    To preserve relative scaling, set `mu=1`.

    Online computation
    ------------------

    Any computation with `axis` that includes `0` requires simultaneous access
    to all samples. This poses a problem in two settings:

        1. Insufficient RAM. The solution is to write an *equivalent* computation
           that aggregates statistics one sample at a time. E.g. for `mu`:

               Xsum = []
               for x in dataset:
                   Xsum.append(B.sum(x, axis=-1, keepdims=True))
               mu = B.median(B.vstack(Xsum), axis=0, keepdims=True)

        2. Streaming / new samples. In this case we must reuse parameters computed
           over e.g. entire train set.

    Computations over all axes *except* `0` are done on per-sample basis, which
    means not having to rely on other samples - but also an inability to do so
    (i.e. to precompute and reuse params).
    """
    # validate args & set defaults ###########################################
    if X.ndim != 3:
        raise ValueError("input must be 3D, `(samples, features, spatial)` - "
                         "got %s" % str(X.shape))
    B = ExtendedUnifiedBackend(X)

    # check input values
    if B.min(X) < 0:
        warnings.warn("`X` must be non-negative; will take modulus.")
        X = B.abs(X)
    # convert axes to positive
    axes = [mean_axis, std_axis]
    for i, ax in enumerate(axes):
        if ax is None:
            continue
        ax = ax if isinstance(ax, (list, tuple)) else [ax]
        ax = list(ax)
        for j, a in enumerate(ax):
            if a < 0:
                ax[j] = X.ndim + a
        axes[i] = tuple(ax)
    mean_axis, std_axis = axes

    # check input dims
    dim_ones = tuple(d for d in range(X.ndim) if X.shape[d] == 1)
    if dim_ones != ():
        def check_dims(g, name):
            g = g if isinstance(g, (tuple, list)) else (g,)
            if all(dim in dim_ones for dim in g):
                raise ValueError("input dims cannot be `1` along same dims as "
                                 "`{}` (gives NaNs); got X.shape == {}, "
                                 "{} = {}".format(name, X.shape, name, mean_axis))

        check_dims(mean_axis, 'mean_axis')
        check_dims(std_axis,  'std_axis')
        # check mu
        if mu is None and 0 in dim_ones and 2 in dim_ones:
            raise ValueError("input dims cannot be `1` along dims 0 and 2 "
                             "if `mu` is None (gives NaNs); "
                             "got X.shape == {}".format(X.shape))

    # main transform #########################################################
    if mu is None:
        # spatial sum (integral)
        Xsum = B.sum(X, axis=-1, keepdims=True)
        # sample median
        mu = B.median(Xsum, axis=0, keepdims=True)

    def sparse_mean(x, div=100, iters=4):
        """Mean of non-negligible points"""
        m = x.mean()
        for _ in range(iters - 1):
            m = x[x > m / div].mean()
        return m

    # rescale
    Xnorm = X / mu
    # contraction factor
    if C_mult is None:
        C_mult = 5 if C is None else 1
    if C is None:
        C = 1 / sparse_mean(B.abs(Xnorm), iters=4)
    C *= C_mult
    # log
    Xnorm = B.log(1 + Xnorm * C)

    # standardization ########################################################
    if mean_axis is not None:
        Xnorm -= B.mean(Xnorm, axis=mean_axis, keepdims=True)

    if std_axis is not None:
        Xnorm /= B.std(Xnorm, axis=std_axis, keepdims=True)

    return Xnorm


def pack_coeffs_jtfs(Scx, meta, structure=1, sample_idx=None,
                     separate_lowpass=None, sampling_psi_fr=None, out_3D=None,
                     reverse_n1=False, debug=False, recursive=False):
    """Packs efficiently JTFS coefficients into one of valid 4D structures.

    Parameters
    ----------
    Scx : tensor/list/dict
        JTFS output. Must have `out_type` 'dict:array' or 'dict:list',
        and `average=True`.

    meta : dict
        JTFS meta.

    structure : int / None
        Structure to pack `Scx` into (see "Structures" below), integer 1 to 4.
        Will pack into a structure even if not suitable for convolution (as
        determined by JTFS parameters); see "Structures" if convs are relevant.

          - If can pack into one structure, can pack into any other (1 to 5).
          - 6 to 9 aren't implemented since they're what's already returned
            as output.
          - This method is only needed for 3D or 4D convolutions, for which
            only structure=5 with `out_3D=True` and `aligned=True` is fully valid
            (see below); 1D convolutions can be done on any JTFS with
            `average=True`, and 2D on any `out_3D=True`.

    sample_idx : int / None
        Index of sample in batched input to pack. If None (default), will
        pack all samples.
        Returns 5D if `not None` *and* there's more than one sample.

    separate_lowpass : None / bool
        If True, will pack spinned (`psi_t * psi_f_up`, `psi_t * psi_f_dn`)
        and lowpass (`phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`) pairs
        separately. Recommended for convolutions (see Structures & Uniformitym).

        Defaults to False if `structure != 5`. `structure = 5` requires True.

    sampling_psi_fr : str / None
        Used for sanity check for padding along `n1_fr`.
        Must match what was passed to `TimeFrequencyScattering1D`.
        If None, will assume library default.

    out_3D : bool / None
        Used for sanity check for padding along `n1`
        (enforces same number of `n1`s per `n2`).

    reverse_n1 : bool (default False)
        If True, will reverse ordering of `n1`. By default, low n1 <=> high freq
        (as directly output by `timefrequency_scattering1d`).

    debug : bool (defualt False)
        If True, coefficient values will be replaced by meta `n` values for
        debugging purposes, where the last dim is size 4 and contains
        `(n1_fr, n2, n1, time)` assuming `structure == 1`.

     recursive : bool (default False)
        Internal argument for handling batch_size > 1, do not use.

    Returns
    -------
    out: tensor / tuple[tensor]
        Packed `Scx`, depending on `structure` and `separate_lowpass`:

          - 1: `out` if False else
               `(out, out_phi_f, out_phi_t)`
          - 2: same as 1
          - 3: `(out_up, out_dn, out_phi_f)` if False else
               `(out_up, out_dn, out_phi_f, out_phi_t)`
          - 4: `(out_up, out_dn)` if False else
               `(out_up, out_dn, out_phi_t)`
          - 5: `(out_up, out_dn, out_phi_f, out_phi_t, out_phi)`

        `out_phi_t` is `phi_t * psi_f` and `phi_t * phi_f` concatenated.
        `out_phi_f` is `psi_t * phi_f` for all configs except
        `3, True`, where it is concatenated with `phi_t * phi_f`.

        For further info, see "Structures", "Parameter effects", and "Notes".

    Structures
    ----------
    Assuming `aligned=True`, then for `average, average_fr`, the following form
    valid convolution structures:

      1. `True, True*`:  3D/4D*, `(n1_fr, n2, n1, time)`
      2. `True, True*`:  2D/4D*, `(n2, n1_fr, n1, time)`
      3. `True, True*`:  4D,     `(n2, n1_fr//2,     n1, time)`*2,
                                 `(n2, 1, n1, time)`
      4. `True, True*`:  2D/4D*, `(n2, n1_fr//2 + 1, n1, time)`*2
      5. `True, True*`:  4D,     `(n2, n1_fr//2,     n1, time)`*2,
                                 `(n2, 1, n1, time)`,
                                 `(1, n1_fr, n1, time)`,
                                 `(1, 1, n1, time)`
      6. `True, True*`:  2D/3D*, `(n2 * n1_fr, n1, time)`
      7. `True, False`:  1D/2D*, `(n2 * n1_fr * n1, time)`
      8. `False, True`:  list of variable length 1D tensors
      9. `False, False`: list of variable length 1D tensors

    **Indexing/units**:

      - n1: frequency [Hz], first-order temporal variation
      - n2: frequency [Hz], second-order temporal variation
        (frequency of amplitude modulation)
      - n1_fr: quefrency [cycles/octave], first-order frequential variation
        (frequency of frequency modulation bands, roughly. More precisely,
         correlates with frequential bands (independent components/modes) of
         varying widths, decay factors, and recurrences, per temporal slice)
      - time: time [sec]
      - The actual units are discrete, "Hz" and "sec" are an example.
        To convert, multiply by sampling rate `fs`.
      - The `n`s are indexings of the output array, also indexings of wavelets
        once accounting for stride and order reversal (n1_reverse).
          - E.g. `n1=2` may index `psi1_f[2*log2_F]` - or, generally,
            `psi1_f[2*total_conv_stride_over_U1_realized]` (see `core`).
          - With `aligned=False`, `n1` striding varies on per-`n2` basis.
            `n1` is the only "uncertain" index in this regard, and only `n1` and
            `t` are subject to stride; `n2` always means `psi2_f[n2]`, and
            `n1_fr` always means `psi1_f_fr_up[n1_fr]` (or down).
          - Hence, the frequency in "n2: frequency [Hz]" is obtained via
            `psi2_f[n2]['xi']`.
          - Higher n <=> higher center frequency. That is, coeffs are packed in
            order of decreasing frequency, just as in computation.
            Exceptions: 1) structure `1` or `2`, where spin down's `n1_fr` axis
            is reversed, and 2) if `n1_reverse=True`.

    **Convolution-validity**:

      - Structure 3 is 3D/4D-valid only if one deems valid the disjoint
        representation with separate convs over spinned and lowpassed
        (thus convs over lowpassed-only coeffs are deemed valid) - or if one
        opts to exclude the lowpassed pairs.
      - Structure 4 is 3D/4D-valid only if one deems valid convolving over both
        lowpassed and spinned coefficients.
      - Structure 5 is completely valid.
      - For convolutions, first dim is assumed to be channels (unless doing
        4D convs).
      - `True*` indicates a "soft requirement"; as long as `aligned=True`,
        `False` can be fully compensated with padding.
        Since 5 isn't implemented with `False`, it can be obtained from `False`
        by reshaping one of 1-4.
      - `2D/4D*` means 3D/4D convolutions aren't strictly valid for convolving
        over trailing (last) dimensions (see below), but 1D/2D are.
        `3D` means 1D, 2D, 3D are all valid.

    Structure interpretations for convolution
    -----------------------------------------
    Interpretations for convolution (and equivalently, spatial coherence)
    are as follows:

        1. The true JTFS structure. `(n2, n1, time)` are uniform and thus
           valid dimensions for 3D convolution (if all `phi` pairs are excluded,
           which isn't default behavior; see "Uniformity").
        2. It's a dim-permuted 1, but last three dimensions are no longer uniform
           and don't necessarily form a valid convolution pair.
           This is the preferred structure for conceptualizing or debugging as
           it's how the computation graph unfolds (and so does information
           density, as `N_fr` varies along `n2`).
        3. It's 2, but split into uniform pairs - `out_up, out_dn, out_phi`
           suited for convolving over last three dims. These still include
           `phi_t * psi_f` and `phi_t * phi_f` pairs, so for strict uniformity
           these slices should drop (e.g. `out_up[1:]`).
        4. It's 3, but only `out_up, out_dn`, and each includes `psi_t * phi_f`.
           If this "soft uniformity" is acceptable then `phi_t * psi_f` pairs
           should be kept.
        5. Completely valid convolutional structure.
           Every pair is packed separately. The only role of `pack_coeffs_jtfs`
           here is to reshape the pairs into 4D tensors, and pad.
        6. `n2` and `n1_fr` are flattened into one dimension. The resulting
           3D structure is suitable for 2D convolutions along `(n1, time)`.
        7. `n2`, `n1_fr`, and `n1` are flattened into one dimension. The resulting
           2D structure is suitable for 1D convolutions along `time`.
        8. `time` is variable; structue not suitable for convolution.
        9. `time` and `n1` are variable; structure not suitable for convolution.

    Structures not suited for convolutions may be suited for other transforms,
    e.g. Dense or Graph Neural Networks (or graph convolutions).

    Helpful visuals:  # TODO relink
      https://github.com/kymatio/kymatio/discussions/708#discussioncomment-1624521

    Uniformity
    ----------
    Coefficients are "uniform" if their generating wavelets are spaced uniformly
    (that is, equally incremented/spaced apart) in log space. The lowpass filter
    is equivalently an infinite scale wavelet, thus it breaks uniformity
    (it'd take infinite number of wavelets to be one increment away from lowpass).
    Opposite spins require stepping over the lowpass and are hence disqualified.

    Above is strictly true in continuous time. In a discrete setting, however,
    the largest possible non-dc scale is far from infinite. A 2D lowpass wavelet
    is somewhat interpretable as a subsequent scaling and rotation of the
    largest scale bandpass, as the bandpass itself is such a scaling and rotation
    of its preceding bandpass (emphasis on "somewhat", as this is wrong in
    important ways).

    Nonetheless, a lowpass is an averaging rather than modulation extracting
    filter: its physical units differ, and it has zero FDTS sensitivity - and
    this is a stronger objection for convolution. Further, when convolving over
    modulus of wavelet transform (as frequential scattering does), the dc bin
    is most often dominant, and by a lot - thus without proper renormalization
    it will drown out the bandpass coefficients in concatenation.

    The safest configuration for convolution thus excludes all lowpass pairs:
    `phi_t * phi_f`, `phi_t * psi_f`, and `psi_t * phi_f`; these can be convolved
    over separately. The bandpass and lowpass concatenations aren't recommended
    as anything but experimental.

    Parameter effects
    -----------------
    `average` and `average_fr` are described in "Structures". Additionally:

      - aligned:
        - True: enables the true JTFS structure (every structure in 1-7 is
          as described).
        - False: yields variable stride along `n1`, disqualifying it from
          3D convs along `(n2, n1, time)`. However, assuming semi-uniformity
          is acceptable, then each `n2` slice in `(n2, n1_fr, n1, time)`, i.e.
          `(n1_fr, n1, time)`, has the same stride, and forms valid conv pair
          (so use 3 or 4). Other structures require similar accounting.
          Rules out structure 1 for 3D/4D convs.
      - out_3D:
        - True: enforces same freq conv stride on *per-`n2`* basis, enabling
          3D convs even if `aligned=False`.
      - sampling_psi_fr:
        - 'resample': enables the true JTFS structure.
        - 'exclude': enables the true JTFS structure (it's simply a subset of
          'resample'). However, this involves large amounts of zero-padding to
          fill the missing convolutions and enable 4D concatenation.
        - 'recalibrate': breaks the true JTFS structure. `n1_fr` frequencies
          and widths now vary with `n2`, which isn't spatially coherent in 4D.
          It also renders `aligned=True` a pseudo-alignment.
          Like with `aligned=False`, alignment and coherence is preserved on
          per-`n2` basis, retaining the true structure in a piecewise manner.
          Rules out structure 1 for 3D/4D convs.
      - average:
        - It's possible to support `False` the same way `average_fr=False` is
          supported, but this isn't implemented.

    Notes
    -----
      1. Method requires `out_exclude=None` if `not separate_lowpass` - else,
         the following are allowed to be excluded: 'phi_t * psi_f',
         'phi_t * phi_f', and if `structure != 4`, 'psi_t * phi_f'.

      2. The built-in energy renormalization includes doubling the energy
         of `phi_t * psi_f` pairs to compensate for computing only once (for
         just one spin since it's identical to other spin), while here it may
         be packed twice (structure=`1` or `2`, or structure=`3` or `4` and
         `not separate_lowpass`); to compensate, its energy is halved before
         packing.

      3. Energy duplication isn't avoided for all configs:
          - `3, separate_lowpass`: packs the `phi_t * phi_f` pair twice -
            with `phi_t * psi_f`, and with `psi_t * phi_f`.
            `out_phi_f` always concats with `phi_t * phi_f` for `3` since
            `phi_f` is never concat with spinned, so it can't concat with
            `phi_t` pairs as usual.
          - `4, not separate_lowpass`: packs `phi_t * phi_f` and `psi_t * phi_f`
            pairs twice, once for each spin.
          - `4, separate_lowpass`: packs `psi_t * phi_f` pairs twice, once for
            each spin.
          - Note both `3` and `4` pack `phi_t * psi_f` pairs twice if
            `not separate_lowpass`, but the energy is halved anyway and hence
            not duped.
         This is intentional, as the idea is to treat each packing as an
         independent unit.
    """
    B = ExtendedUnifiedBackend(Scx)

    def combined_to_tensor(combined_all, recursive):
        def process_dims(o):
            if recursive:
                assert o.ndim == 5, o.shape
            else:
                assert o.ndim == 4, o.shape
                o = o[None]
            return o

        def not_none(x):
            return (x is not None if not recursive else
                    all(_x is not None for _x in x))

        # fetch combined params
        if structure in (1, 2):
            combined, combined_phi_t, combined_phi_f, combined_phi = combined_all
        else:
            (combined_up, combined_dn, combined_phi_t, combined_phi_f,
             combined_phi) = combined_all

        # compute pad params
        cbs = [(cb[0] if recursive else cb) for cb in combined_all
               if not_none(cb)]
        n_n1s_max = max(len(cb[n2_idx][n1_fr_idx])
                        for cb in cbs
                        for n2_idx in range(len(cb))
                        for n1_fr_idx in range(len(cb[n2_idx])))
        pad_value = 0 if not debug else -2
        # left pad along `n1` if `reverse_n1`
        left_pad_axis = (-2 if reverse_n1 else None)
        general = False  # use routine optimized for JTFS
        kw = dict(pad_value=pad_value, left_pad_axis=left_pad_axis,
                  general=general)

        # `phi`s #############################################################
        out_phi_t, out_phi_f, out_phi = None, None, None
        # ensure `phi`s and spinned pad to the same number of `n1`s
        ref_shape = ((None, None, n_n1s_max, None) if not recursive else
                     (None, None, None, n_n1s_max, None))
        # this will pad along `n1`
        if not_none(combined_phi_t):
            out_phi_t = tensor_padded(combined_phi_t, ref_shape=ref_shape, **kw)
            out_phi_t = process_dims(out_phi_t)
        if not_none(combined_phi_f):
            out_phi_f = tensor_padded(combined_phi_f, ref_shape=ref_shape, **kw)
            out_phi_f = process_dims(out_phi_f)
        if not_none(combined_phi):
            out_phi = tensor_padded(combined_phi, ref_shape=ref_shape, **kw)
            out_phi = process_dims(out_phi)

        # spinned ############################################################
        # don't need `ref_shape` here since by implementation max `n1`s
        # should be found in spinned (`phi`s are trimmed to ensure this)
        if structure in (1, 2):
            out = tensor_padded(combined, **kw)
            out = process_dims(out)

            if structure == 1:
                tp_shape = (0, 2, 1, 3, 4)
                out = B.transpose(out, tp_shape)
                if separate_lowpass:
                    if out_phi_t is not None:
                        out_phi_t = B.transpose(out_phi_t, tp_shape)
                    if out_phi_f is not None:
                        out_phi_f = B.transpose(out_phi_f, tp_shape)

            out = (out if not separate_lowpass else
                   (out, out_phi_f, out_phi_t))

        elif structure in (3, 4):
            out_up = tensor_padded(combined_up, **kw)
            out_dn = tensor_padded(combined_dn, **kw)
            out_up = process_dims(out_up)
            out_dn = process_dims(out_dn)

            if structure == 3:
                out = ((out_up, out_dn, out_phi_f) if not separate_lowpass else
                       (out_up, out_dn, out_phi_f, out_phi_t))
            else:
                if not separate_lowpass:
                    out = (out_up, out_dn)
                else:
                    out = (out_up, out_dn, out_phi_t)

        elif structure == 5:
            out = (out_up, out_dn, out_phi_f, out_phi_t, out_phi)

        # sanity checks ##########################################################
        phis = dict(out_phi_t=out_phi_t, out_phi_f=out_phi_f, out_phi=out_phi)
        ref = out[0] if isinstance(out, tuple) else out
        for name, op in phis.items():
            if op is not None:
                errmsg = (name, op.shape, ref.shape)
                # `t`s must match
                assert op.shape[-1] == ref.shape[-1], errmsg
                # number of `n1`s must match
                assert op.shape[-2] == ref.shape[-2], errmsg
                # number of samples must match
                assert op.shape[0]  == ref.shape[0],  errmsg

                # due to transpose
                fr_dim = -3 if structure != 1 else -4
                if name in ('out_phi_f', 'out_phi'):
                    assert op.shape[fr_dim] == 1, op.shape
                    if name == 'out_phi':
                        # only for structure=5, which has `n2` at `shape[-4]`
                        assert op.shape[-4] == 1, op.shape
                    continue

                # phi_t only #################################################
                # compute `ref_fr_len`
                if structure in (1, 2, 5):
                    ref_fr_len = ref.shape[fr_dim]
                elif structure == 3:
                    # separate spins have half of total `n1_fr`s, but
                    # we also pack `phi_t` only once
                    ref_fr_len = ref.shape[fr_dim] * 1
                elif structure == 4:
                    # above + having `psi_t * phi_f`
                    # (i.e. fr_len_4 = fr_len_3 + 1)
                    ref_fr_len = (ref.shape[fr_dim] - 1) * 1
                # due to `phi_t * phi_f` being present only in `out_phi_t`
                ref_fr_len = (ref_fr_len if not separate_lowpass else
                              ref_fr_len + 1)

                # assert
                assert op.shape[fr_dim] == ref_fr_len, (
                    "{} != {} | {} | {}, {}".format(op.shape[fr_dim], ref_fr_len,
                                                    name, op.shape, ref.shape))
        if structure in (3, 4, 5):
            assert out_up.shape == out_dn.shape, (out_up.shape, out_dn.shape)

        if not recursive:
            # drop batch dim; `None` in case of `out_exclude`
            if isinstance(out, tuple):
                out = tuple((o[0] if o is not None else o) for o in out)
            else:
                out = out[0]
        return out

    # pack full batch recursively ############################################
    if not isinstance(Scx, dict):
        raise ValueError("must use `out_type` 'dict:array' or 'dict:list' "
                         "for `pack_coeffs_jtfs` (got `type(Scx) = %s`)" % (
                             type(Scx)))

    # infer batch size
    ref_pair = list(Scx)[0]
    if isinstance(Scx[ref_pair], list):
        n_samples = Scx[ref_pair][0]['coef'].shape[0]
    else:  # tensor
        n_samples = Scx[ref_pair].shape[0]
    n_samples = int(n_samples)

    # handle recursion, if applicable
    if n_samples > 1 and sample_idx is None:
        combined_phi_t_s, combined_phi_f_s, combined_phi_s = [], [], []
        if structure in (1, 2):
            combined_s = []
        elif structure in (3, 4):
            combined_up_s, combined_dn_s = [], []

        for sample_idx in range(n_samples):
            combined_all = pack_coeffs_jtfs(Scx, meta, structure, sample_idx,
                                            separate_lowpass, sampling_psi_fr,
                                            debug, recursive=True)

            combined_phi_t_s.append(combined_all[-3])
            combined_phi_f_s.append(combined_all[-2])
            combined_phi_s.append(combined_all[-1])
            if structure in (1, 2):
                combined_s.append(combined_all[0])
            elif structure in (3, 4):
                combined_up_s.append(combined_all[0])
                combined_dn_s.append(combined_all[1])

        phis = (combined_phi_t_s, combined_phi_f_s, combined_phi_s)
        if structure in (1, 2):
            combined_all_s = (combined_s, *phis)
        elif structure in (3, 4):
            combined_all_s = (combined_up_s, combined_dn_s, *phis)
        out = combined_to_tensor(combined_all_s, recursive=True)
        return out

    ##########################################################################

    # validate `structure` / set default
    structures_available = {1, 2, 3, 4, 5}
    if structure is None:
        structure = structures_available[0]
    elif structure not in structures_available:
        raise ValueError(
            "invalid `structure={}`; Available are: {}".format(
                structure, ','.join(map(str, structures_available))))

    if separate_lowpass is None:
        separate_lowpass = False if structure != 5 else True
    elif separate_lowpass is True and structure == 5:
        raise ValueError("`structure=5` requires `separate_lowpass=True`.")

    # unpack coeffs for further processing
    Scx_unpacked = {}
    list_coeffs = isinstance(list(Scx.values())[0], list)
    if sample_idx is None and not recursive and n_samples == 1:
        sample_idx = 0

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)
    t_ref = None
    for pair in Scx:
        is_joint = bool(pair not in ('S0', 'S1'))
        if not is_joint:
            continue
        Scx_unpacked[pair] = []
        for coef in Scx[pair]:
            if list_coeffs and (isinstance(coef, dict) and 'coef' in coef):
                coef = coef['coef']
            if t_ref is None:
                t_ref = coef.shape[-1]
            assert coef.shape[-1] == t_ref, (coef.shape, t_ref,
                                             "(if using average=False, set "
                                             "oversampling=99)")

            if coef.ndim == 2:
                Scx_unpacked[pair].extend(coef)
            elif coef.ndim == 1:
                Scx_unpacked[pair].append(coef)
            else:
                raise ValueError("expected `coef.ndim` of 1 or 2, got "
                                 "shape = %s" % str(coef.shape))

    # check that all necessary pairs are present
    pairs = ('psi_t * psi_f_up', 'psi_t * psi_f_dn', 'psi_t * phi_f',
             'phi_t * psi_f', 'phi_t * phi_f')
    # structure 4 requires `psi_t * phi_f`
    okay_to_exclude_if_sep_lp = (pairs[-3:] if structure != 4 else
                                 pairs[-2:])
    Scx_pairs = list(Scx)
    for p in pairs:
      if p not in Scx_pairs:
        if (not separate_lowpass or
            (separate_lowpass and p not in okay_to_exclude_if_sep_lp)):
          raise ValueError(("configuration requires pair '%s', which is "
                            "missing") % p)

    # for later; controls phi_t pair energy norm
    phi_t_packed_twice = bool((structure in (1, 2)) or
                              (structure in (3, 4) and not separate_lowpass))

    # pack into dictionary indexed by `n1_fr`, `n2` ##########################
    packed = {}
    ns = meta['n']
    n_n1_frs_max = 0
    for pair in pairs:
        if pair not in Scx_pairs:
            continue
        packed[pair] = []
        nsp = ns[pair].astype(int).reshape(-1, 3)

        idx = 0
        n2s_all = nsp[:, 0]
        n2s = np.unique(n2s_all)

        for n2 in n2s:
            n1_frs_all = nsp[n2s_all == n2, 1]
            packed[pair].append([])
            n1_frs = np.unique(n1_frs_all)
            n_n1_frs_max = max(n_n1_frs_max, len(n1_frs))

            for n1_fr in n1_frs:
                packed[pair][-1].append([])
                n1s_done = 0

                if out_3D:
                    # same number of `n1`s for all frequential slices *per-`n2`*
                    n_n1s = len(n1_frs_all)
                    n_n1s_in_n1_fr = n_n1s // len(n1_frs)
                    assert (n_n1s / len(n1_frs)
                            ).is_integer(), (n_n1s, len(n1_frs))
                else:
                    n_n1s_in_n1_fr = len(nsp[n2s_all == n2, 2
                                             ][n1_frs_all == n1_fr])
                if debug:
                    # pack meta instead of coeffs
                    n1s = nsp[n2s_all == n2, 2][n1_frs_all == n1_fr]
                    coef = [[n2, n1_fr, n1, 0] for n1 in n1s]
                    # ensure coef.shape[-1] == t
                    while Scx_unpacked[pair][0].shape[-1] > len(coef[0]):
                        for i in range(len(coef)):
                            coef[i].append(0)
                    coef = np.array(coef)

                    packed[pair][-1][-1].extend(coef)
                    assert len(coef) == n_n1s_in_n1_fr
                    idx += len(coef)
                    n1s_done += len(coef)

                else:
                    while idx < len(nsp) and n1s_done < n_n1s_in_n1_fr:
                        try:
                            coef = Scx_unpacked[pair][idx]
                        except Exception as e:
                            print(pair, idx)
                            raise e
                        if pair == 'phi_t * psi_f' and phi_t_packed_twice:
                            # see "Notes" in docs
                            coef = coef / B.sqrt(2., dtype=coef.dtype)
                        packed[pair][-1][-1].append(coef)
                        idx += 1
                        n1s_done += 1

    # pad along `n1_fr`
    if sampling_psi_fr is None:
        sampling_psi_fr = 'exclude'
    pad_value = 0 if not debug else -2
    for pair in packed:
        if 'psi_f' not in pair:
            continue
        for n2_idx in range(len(packed[pair])):
            if len(packed[pair][n2_idx]) < n_n1_frs_max:
                assert sampling_psi_fr == 'exclude'  # should not occur otherwise
            else:
                continue

            # make a copy to avoid modifying `packed`
            ref = list(tensor_padded(packed[pair][n2_idx][0]))
            # assumes last dim is same (`average=False`)
            # and is 2D, `(n1, t)` (should always be true)
            for i in range(len(ref)):
                if debug:
                    # n2 will be same, everything else variable
                    ref[i][1:] = ref[i][1:] * 0 + pad_value
                else:
                    ref[i] = ref[i] * 0

            while len(packed[pair][n2_idx]) < n_n1_frs_max:
                packed[pair][n2_idx].append(list(ref))

    # pack into list ready to convert to 4D tensor ###########################
    # current indexing: `(n2, n1_fr, n1, time)`
    # c = combined
    c_up    = packed['psi_t * psi_f_up']
    c_dn    = packed['psi_t * psi_f_dn']
    c_phi_t = packed['phi_t * psi_f'] if 'phi_t * psi_f' in Scx_pairs else None
    c_phi_f = packed['psi_t * phi_f'] if 'psi_t * phi_f' in Scx_pairs else None
    c_phi   = packed['phi_t * phi_f'] if 'phi_t * phi_f' in Scx_pairs else None

    can_make_c_phi_t = bool(c_phi_t is not None and c_phi is not None)

    # `deepcopy` below is to ensure same structure packed repeatedly in different
    # places isn't modified in both places when it's modified in one place.
    # `None` set to variables means they won't be tensored and returned.

    if structure in (1, 2):
        # structure=2 is just structure=1 transposed, so pack them same
        # and transpose later.
        # instantiate total combined
        combined = c_up
        c_up = None

        # append phi_f ####
        if not separate_lowpass:
            for n2 in range(len(c_phi_f)):
                for n1_fr in range(len(c_phi_f[n2])):
                    c = c_phi_f[n2][n1_fr]
                    combined[n2].append(c)
            c_phi_f = None
            # assert that appending phi_f only increased dim1 by 1
            l0, l1 = len(combined[0]), len(c_dn[0])
            assert l0 == l1 + 1, (l0, l1)

        # append down ####
        # assert that so far dim0 hasn't changed
        assert len(combined) == len(c_dn), (len(combined), len(c_dn))

        # dn: reverse `psi_f` ordering
        for n2 in range(len(c_dn)):
            c_dn[n2] = c_dn[n2][::-1]

        for n2 in range(len(combined)):
            combined[n2].extend(c_dn[n2])
        c_dn = None

        # pack phi_t ####
        if not separate_lowpass or can_make_c_phi_t:
            c_phi_t = deepcopy(c_phi_t)
            c_phi_t[0].append(c_phi[0][0])
            # phi_t: reverse `psi_f` ordering
            c_phi_t[0].extend(packed['phi_t * psi_f'][0][::-1])
            c_phi = None

        # append phi_t ####
        if not separate_lowpass:
            combined.append(c_phi_t[0])
            c_phi_t = None

    elif structure == 3:
        # pack spinned ####
        if not separate_lowpass:
            c_up.append(c_phi_t[0])
            c_dn.append(deepcopy(c_phi_t[0]))
            c_phi_t = None

        # pack phi_t ####
        if separate_lowpass and can_make_c_phi_t:
            c_phi_t[0].append(deepcopy(c_phi[0][0]))

        # pack phi_f ####
        # structure=3 won't pack `phi_f` with `psi_f`, so can't pack
        # `phi_t * phi_f` along `phi_t * psi_f` (unless `separate_lowpass=True`
        # where `phi_t` isn't packed with `psi_f`), must pack with `psi_t * phi_f`
        # instead
        if not separate_lowpass or (c_phi_f is not None and c_phi is not None):
            c_phi_f.append(c_phi[0])
            c_phi = None

    elif structure == 4:
        # pack phi_f ####
        for n2 in range(len(c_phi_f)):
            # structure=4 joins `psi_t * phi_f` with spinned
            c = c_phi_f[n2][0]
            c_up[n2].append(c)
            c_dn[n2].append(c)
        c_phi_f = None
        # assert up == dn along dim1
        l0, l1 = len(c_up[0]), len(c_dn[0])
        assert l0 == l1, (l0, l1)

        # pack phi_t ####
        if separate_lowpass and can_make_c_phi_t:
            # pack `phi_t * phi_f` with `phi_t * psi_f`
            c_phi_t[0].append(c_phi[0][0])
        elif not separate_lowpass:
            # pack `phi_t * phi_f` with `phi_t * psi_f`, packed with each spin
            # phi_t, append `n1_fr` slices via `n2`
            c_up.append(deepcopy(c_phi_t[0]))
            c_dn.append(c_phi_t[0])
            # phi, append one `n2, n1_fr` slice
            c_up[-1].append(deepcopy(c_phi[0][0]))
            c_dn[-1].append(c_phi[0][0])

            c_phi_t, c_phi = None, None
        # assert up == dn along dim0, and dim1
        assert len(c_up) == len(c_dn), (len(c_up), len(c_dn))
        l0, l1 = len(c_up[0]), len(c_dn[0])
        assert l0 == l1, (l0, l1)

    elif structure == 5:
        pass  # all packed

    # reverse ordering of `n1` ###############################################
    if reverse_n1:
        # pack all into `cbs`
        if c_up is not None:
            cbs = [c_up, c_dn]
        else:
            cbs = [combined]
        if c_phi_t is not None:
            cbs.append(c_phi_t)
        if c_phi_f is not None:
            cbs.append(c_phi_f)
        if c_phi is not None:
            cbs.append(c_phi)

        # reverse `n1`
        cbs_new = []
        for i, cb in enumerate(cbs):
            cbs_new.append([])
            for n2_idx in range(len(cb)):
                cbs_new[i].append([])
                for n1_fr_idx in range(len(cb[n2_idx])):
                    cbs_new[i][n2_idx].append(cb[n2_idx][n1_fr_idx][::-1])

        # unpack all from `cbs`
        if c_up is not None:
            c_up = cbs_new.pop(0)
            c_dn = cbs_new.pop(0)
        else:
            combined = cbs_new.pop(0)
        if c_phi_t is not None:
            c_phi_t = cbs_new.pop(0)
        if c_phi_f is not None:
            c_phi_f = cbs_new.pop(0)
        if c_phi is not None:
            c_phi = cbs_new.pop(0)
        assert len(cbs_new) == 0, len(cbs_new)

    # finalize ###############################################################
    phis = (c_phi_t, c_phi_f, c_phi)
    combined_all = ((combined, *phis) if c_up is None else
                    (c_up, c_dn, *phis))
    if recursive:
        return combined_all
    return combined_to_tensor(combined_all, recursive=False)


def coeff_energy(Scx, meta, pair=None, aggregate=True, correction=False,
                 kind='l2'):
    """Computes energy of JTFS coefficients.

    Current implementation permits computing energy directly via
    `sum(abs(coef)**2)`, hence this method isn't necessary.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose energy to compute.
        If None, will compute for all.

    aggregate: bool (default True)
        True: return one value per `pair`, the sum of all its coeffs' energies
        False: return `(E_flat, E_slices)`, where:
            - E_flat = energy of every coefficient, flattened into a list
              (but still organized pair-wise)
            - E_slices = energy of every joint slice (if not `'S0', 'S1'`),
              in a pair. That is, sum of coeff energies on per-`(n2, n1_fr)`
              basis.

    correction : bool (default False)
        Whether to apply stride and filterbank norm correction factors:
            - stride: if subsampled by 2, energy will reduce by 2
            - filterbank: since input is assumed real, we convolve only over
              positive frequencies, getting half the energy

        Current JTFS implementation accounts for both so default is `False`
        (in fact `True` isn't implemented with any configuration due to
        forced LP sum renormalization - though it's possible to implement).

        Filterbank energy correction is as follows:

            - S0 -> 1
            - U1 -> 2 (because psi_t is only analytic)
            - phi_t * phi_f -> 2 (because U1)
            - psi_t * phi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)
            - phi_t * psi_f -> 4 (because U1 and psi_f is only for one spin)
            - psi_t * psi_f -> 4 (because U1 and another psi_t that's
                                  only analytic)

        For coefficient correction (e.g. distance computation) we instead
        scale the coefficients by square root of these values.

    kind: str['l1', 'l2']
        Kind of energy to compute. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)`
        (so actually L2^2).

    Returns
    -------
    E: float / tuple[list]
        Depends on `pair`, `aggregate`.

    Rationale
    ---------
    Simply `sum(abs(coef)**2)` won't work because we must account for subsampling.

        - Subsampling by `k` will reduce energy (both L1 and L2) by `k`
          (assuming no aliasing).
        - Must account for variable subsampling factors, both in time
          (if `average=False` for S0, S1) and frequency.
          This includes if only seeking ratio (e.g. up vs down), since
          `(a*x0 + b*x1) / (a*y0 + b*y1) != (x0 + x1) / (y0 + y1)`.
    """
    if pair is None or isinstance(pair, (tuple, list)):
        # compute for all (or multiple) pairs
        pairs = pair
        E_flat, E_slices = {}, {}
        for pair in Scx:
            if pairs is not None and pair not in pairs:
                continue
            E_flat[pair], E_slices[pair] = coeff_energy(
                Scx, meta, pair, aggregate=False, kind=kind)
        if aggregate:
            E = {}
            for pair in E_flat:
                E[pair] = np.sum(E_flat[pair])
            return E
        return E_flat, E_slices

    elif not isinstance(pair, str):
        raise ValueError("`pair` must be string, list/tuple of strings, or None "
                         "(got %s)" % pair)

    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: energy(c, kind=kind)
    norm_fn = lambda total_joint_stride: (2**total_joint_stride
                                          if correction else 1)

    E_flat, E_slices = _iterate_coeffs(Scx, meta, pair, fn, norm_fn, factor)
    Es = []
    for s in E_slices:
        Es.append(np.sum(s))
    E_slices = Es

    if aggregate:
        return np.sum(E_flat)
    return E_flat, E_slices


def coeff_distance(Scx0, Scx1, meta0, meta1=None, pair=None, correction=False,
                   kind='l2'):
    """Computes L2 or L1 relative distance between `Scx0` and `Scx1`.

    Current implementation permits computing distance directly between
    coefficients, as `toolkit.rel_l2(coef0, coef1)`.

    Parameters
    ----------
    Scx0, Scx1: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta0, meta1: dict[dict[np.ndarray]]
        `jtfs.meta()`. If `meta1` is None, will set equal to `meta0`.
        Note that scattering objects responsible for `Scx0` and `Scx1` cannot
        differ in any way that alters coefficient shapes.

    pair: str / list/tuple[str] / None
        Name(s) of coefficient pairs whose distances to compute.
        If None, will compute for all.

    kind: str['l1', 'l2']
        Kind of distance to compute. L1==`sum(abs(x))`,
        L2==`sqrt(sum(abs(x)**2))`. L1 is not implemented for `correction=False`.

    correction: bool (default False)
        See `help(wavespin.toolkit.coeff_energy)`.

    Returns
    -------
    reldist_flat : list[float]
        Relative distances between individual frequency rows, i.e. per-`n1`.

    reldist_slices : list[float]
        Relative distances between joint slices, i.e. per-`(n2, n1_fr)`.
    """
    if not correction and kind == 'l1':
        raise NotImplementedError

    if meta1 is None:
        meta1 = meta0
    # compute compensation factor (see `correction` docs)
    factor = _get_pair_factor(pair, correction)
    fn = lambda c: c

    def norm_fn(total_joint_stride):
        if not correction:
            return 1
        return (2**(total_joint_stride / 2) if kind == 'l2' else
                2**total_joint_stride)

    c_flat0, c_slices0 = _iterate_coeffs(Scx0, meta0, pair, fn, norm_fn, factor)
    c_flat1, c_slices1 = _iterate_coeffs(Scx1, meta1, pair, fn, norm_fn, factor)

    # make into array and assert shapes are as expected
    c_flat0, c_flat1 = np.asarray(c_flat0), np.asarray(c_flat1)
    c_slices0 = [np.asarray(c) for c in c_slices0]
    c_slices1 = [np.asarray(c) for c in c_slices1]

    assert c_flat0.ndim == c_flat1.ndim == 2, (c_flat0.shape, c_flat1.shape)
    is_joint = bool(pair not in ('S0', 'S1'))
    if is_joint:
        shapes = [np.array(c).shape for cs in (c_slices0, c_slices1) for c in cs]
        # effectively 3D
        assert all(len(s) == 2 for s in shapes), shapes

    d_fn = lambda x: l2(x) if kind == 'l2' else l1(x)
    ref0, ref1 = d_fn(c_flat0), d_fn(c_flat1)
    eps = _eps(ref0, ref1)
    ref = (ref0 + ref1) / 2 + eps

    def D(x0, x1, axis):
        if isinstance(x0, list):
            return [D(_x0, _x1, axis) for (_x0, _x1) in zip(x0, x1)]

        if kind == 'l2':
            return np.sqrt(np.sum(np.abs(x0 - x1)**2, axis=axis)) / ref
        return np.sum(np.abs(x0 - x1), axis=axis) / ref

    # do not collapse `freq` dimension
    reldist_flat   = D(c_flat0,   c_flat1,   axis=-1)
    reldist_slices = D(c_slices0, c_slices1, axis=(-1, -2) if is_joint else -1)

    # return tuple consistency; we don't do "slices" here
    return reldist_flat, reldist_slices


def coeff_energy_ratios(Scx, meta, down_to_up=True, max_to_eps_ratio=10000):
    """Compute ratios of coefficient slice energies, spin down vs up.
    Statistically robust alternative measure to ratio of total energies.

    Parameters
    ----------
    Scx : dict[list] / dict[tensor]
        `jtfs(x)`.

    meta : dict[dict[np.ndarray]]
        `jtfs.meta()`.

    down_to_up : bool (default True)
        Whether to take `E_dn / E_up` (True) or `E_up / E_dn` (False).
        Note, the actual similarities are opposite, as "down" means convolution
        with down, which is cross-correlation with up.

    max_to_eps_ratio : int
        `eps = max(E_pair0, E_pair1) / max_to_eps_ratio`. Epsilon term
        to use in ratio: `E_pair0 / (E_pair1 + eps)`.

    Returns
    -------
    Ratios of coefficient energies.
    """
    # handle args
    assert isinstance(Scx, dict), ("`Scx` must be dict (got %s); " % type(Scx)
                                   + "set `out_type='dict:array'` or 'dict:list'")

    # compute ratios
    l2s = {}
    pairs = ('psi_t * psi_f_dn', 'psi_t * psi_f_up')
    if not down_to_up:
        pairs = pairs[::-1]
    for pair in pairs:
        _, E_slc0 = coeff_energy(Scx, meta, pair=pair, aggregate=False, kind='l2')
        l2s[pair] = np.asarray(E_slc0)

    a, b = l2s.values()
    mx = np.vstack([a, b]).max(axis=0) / max_to_eps_ratio
    eps = np.clip(mx, mx.max() / (max_to_eps_ratio * 1000), None)
    r = a / (b + eps)

    return r


def _get_pair_factor(pair, correction):
    if pair == 'S0' or not correction:
        factor = 1
    elif 'psi' in pair:
        factor = 4
    else:
        factor = 2
    return factor


def _iterate_coeffs(Scx, meta, pair, fn=None, norm_fn=None, factor=None):
    coeffs = drop_batch_dim_jtfs(Scx)[pair]
    out_list = bool(isinstance(coeffs, list))

    # infer out_3D
    if out_list:
        out_3D = bool(coeffs[0]['coef'].ndim == 3)
    else:
        out_3D = bool(coeffs.ndim == 3)

    # fetch backend
    B = ExtendedUnifiedBackend(coeffs)

    # completely flatten into (*, time)
    if out_list:
        coeffs_flat = []
        for coef in coeffs:
            c = coef['coef']
            coeffs_flat.extend(c)
    else:
        if out_3D:
            coeffs = B.reshape(coeffs, (-1, coeffs.shape[-1]))
        coeffs_flat = coeffs

    # prepare for iterating
    meta = deepcopy(meta)  # don't change external dict
    if out_3D:
        meta['stride'][pair] = meta['stride'][pair].reshape(-1, 2)
        meta['n'][pair] = meta['n'][pair].reshape(-1, 3)

    assert (len(coeffs_flat) == len(meta['stride'][pair])), (
        "{} != {} | {}".format(len(coeffs_flat), len(meta['stride'][pair]), pair))

    # define helpers #########################################################
    def get_total_joint_stride(meta_idx):
        n_freqs = 1
        m_start, m_end = meta_idx[0], meta_idx[0] + n_freqs
        stride = meta['stride'][pair][m_start:m_end]
        assert len(stride) != 0, pair

        stride[np.isnan(stride)] = 0
        total_joint_stride = stride.sum()
        meta_idx[0] = m_end  # update idx
        return total_joint_stride

    def n_current():
        i = meta_idx[0]
        m = meta['n'][pair]
        return (m[i] if i <= len(m) - 1 else
                np.array([-3, -3]))  # reached end; ensure equality fails

    def n_is_equal(n0, n1):
        n0, n1 = n0[:2], n1[:2]  # discard U1
        n0[np.isnan(n0)], n1[np.isnan(n1)] = -2, -2  # NaN -> -2
        return bool(np.all(n0 == n1))

    # append energies one by one #############################################
    fn = fn or (lambda c: c)
    norm_fn = norm_fn or (lambda total_joint_stride: 2**total_joint_stride)
    factor = factor or 1

    is_joint = bool(pair not in ('S0', 'S1'))
    E_flat = []
    E_slices = [] if not is_joint else [[]]
    meta_idx = [0]  # make mutable to avoid passing around
    for c in coeffs_flat:
        if hasattr(c, 'numpy'):
            if hasattr(c, 'cpu') and 'torch' in str(type(c)):
                c = c.cpu()
            c = c.numpy()  # TF/torch
        n_prev = n_current()
        assert c.ndim == 1, (c.shape, pair)
        total_joint_stride = get_total_joint_stride(meta_idx)

        E = norm_fn(total_joint_stride) * fn(c) * factor
        E_flat.append(E)

        if not is_joint:
            E_slices.append(E)  # append to list of coeffs
        elif n_is_equal(n_current(), n_prev):
            E_slices[-1].append(E)  # append to slice
        else:
            E_slices[-1].append(E)  # append to slice
            E_slices.append([])

    # in case loop terminates early
    if isinstance(E_slices[-1], list) and len(E_slices[-1]) == 0:
        E_slices.pop()

    # ensure they sum to same
    Es_sum = np.sum([np.sum(s) for s in E_slices])
    adiff = abs(np.sum(E_flat) - Es_sum)
    assert np.allclose(np.sum(E_flat), Es_sum), "MAE=%.3f" % adiff

    return E_flat, E_slices


def est_energy_conservation(x, sc=None, T=None, F=None, J=None, J_fr=None,
                            Q=None, Q_fr=None, max_pad_factor=None,
                            max_pad_factor_fr=None, pad_mode=None,
                            pad_mode_fr=None, average=None, average_fr=None,
                            sampling_filters_fr=None, r_psi=None, analytic=None,
                            out_3D=None, aligned=None, jtfs=False, backend=None,
                            verbose=True, get_out=False):
    """Estimate energy conservation given scattering configurations, especially
    scale of averaging. With default settings, passing only `T`/`F`, computes the
    upper bound.

    Limitations:
      - For time scattering (`jtfs=False`) and non-dyadic length `x`, the
        estimation will be inaccurate per not accounting for energy loss due to
        unpadding.
      - With `jtfs=True`, energies are underestimated per lacking support for
        `out_3D and not average_fr`. That is, convolutions over zero-padded
        regions aren't included with `out_3D=False`. those are regions with
        assumed negligible energy that are nonetheless part of actual
        frequential input. See `out_3D` docs.

    Parameters
    ----------
    x : tensor
        1D input.

    sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
        Scattering object to use. If None, will create per parameters.

    T, F, J, J_fr, Q, Q_fr, max_pad_factor, max_pad_factor_fr, pad_mode,
    pad_mode_fr, average, average_fr, sampling_filters_fr, r_psi, analytic,
    out_3D, aligned:
        Scattering parameters.

    jtfs : bool (default False)
        Whether to estimate per JTFS; if False, does time scattering.
        Must pass also with `sc` to indicate which object it is.
        If `sc` is passed, won't use unaveraged variant where appropriate,
        which won't provide upper bound on energy if `sc(average_fr=True)`.

    backend : None / str
        Backend to use (defaults to torch w/ GPU if available).

    verbose : bool (default True)
        Whether to print results to console.

    get_out : bool (default False)
        Whether to return computed coefficients and scattering objects alongside
        energy ratios.

    Returns
    -------
    ESr : dict[float]
        Energy ratios.

    Scx : tensor / dict[tensor]
        Scattering output (if `get_out==True`).

    sc : `Scattering1D` / `TimeFrequencyScattering1D`
        Scattering object (if `get_out==True`).
    """
    # warn if passing params alongside `sc`
    _kw = dict(T=T, F=F, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr,
               max_pad_factor=max_pad_factor, max_pad_factor_fr=max_pad_factor_fr,
               pad_mode=pad_mode, pad_mode_fr=pad_mode_fr,
               average=average, average_fr=average_fr,
               sampling_filters_fr=sampling_filters_fr,
               out_3D=out_3D, aligned=aligned)
    tm_params = ('T', 'J', 'Q', 'max_pad_factor', 'pad_mode', 'average')
    fr_params = ('F', 'J_fr', 'Q_fr', 'max_pad_factor_fr', 'pad_mode_fr',
                 'average_fr', 'sampling_filters_fr', 'out_3D', 'aligned')
    all_params = (*tm_params, *fr_params)
    if sc is not None and any(_kw[arg] is not None for arg in all_params):
        warnings.warn("`sc` object provided - parametric arguments ignored.")
    elif not jtfs and any(_kw[arg] is not None for arg in fr_params):
        warnings.warn("passed JTFS parameters with `jtfs=False` -- ignored.")

    # create scattering object, if not provided
    if sc is not None:
        if jtfs:
            sc_u = sc_a = sc
    else:
        if jtfs:
            from wavespin import TimeFrequencyScattering1D as SC
        else:
            from wavespin import Scattering1D as SC

        # handle args & pack parameters
        N = x.shape[-1]
        if Q is None:
            Q = (8, 3)
        if pad_mode is None:
            pad_mode = 'reflect'
        if r_psi is None:
            r_psi = (.9, .9)
            r_psi_fr = .9 if jtfs else None
        if backend is None:
            try:
                import torch
                backend = 'torch'
            except:
                backend = 'numpy'
        elif backend == 'torch':
            import torch
        kw = dict(shape=N, J=int(np.log2(N)), T=T, max_pad_factor=max_pad_factor,
                  pad_mode=pad_mode, Q=Q, frontend=backend, r_psi=r_psi)
        if not jtfs:
            if average is None:
                average = True
            if analytic is None:
                analytic = False  # library default
            kw.update(**dict(average=average, analytic=analytic, out_type='list'))
        else:
            # handle `J_fr` & `F`
            if J_fr is None:
                if F is None:
                    sc_temp = SC(**kw)
                    n_psi1 = len(sc_temp.psi1_f)
                    J_fr = int(np.log2(n_psi1)) - 1
                    F = 2**J_fr
                else:
                    J_fr = int(np.log2(F))
            elif F is None:
                F = 2**J_fr
            # handle other args
            if pad_mode_fr is None:
                pad_mode_fr = 'conj-reflect-zero'
            if average_fr is None:
                average_fr = False
            if analytic is None:
                analytic = True  # library default
            if aligned is None:
                aligned = True
            if out_3D is None:
                out_3D = False
            if sampling_filters_fr is None:
                sampling_filters_fr = 'resample'
            if Q_fr is None:
                Q_fr = 4

            # pack JTFS args
            kw.update(**dict(max_pad_factor_fr=max_pad_factor_fr, F=F,
                             pad_mode_fr=pad_mode_fr, average_fr=average_fr,
                             analytic=analytic, Q_fr=Q_fr, out_type='dict:list',
                             sampling_filters_fr=sampling_filters_fr,
                             out_3D=out_3D, aligned=aligned, r_psi_fr=r_psi_fr))
            if average is None:
                kw_u = dict(**kw, average=False)
                kw_a = dict(**kw, average=True)
            else:
                kw_u = kw_a = dict(**kw, average=average)

        # create scattering object
        if backend == 'torch':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not jtfs:
            sc = SC(**kw)
            if backend == 'torch':
                sc = sc.to(device)
            meta = sc.meta()
        else:
            sc_u, sc_a = SC(**kw_u), SC(**kw_a)
            if backend == 'torch':
                sc_u, sc_a = sc_u.to(device), sc_a.to(device)

    # scatter
    if not jtfs:
        Scx = sc(x)
        Scx = jtfs_to_numpy(Scx)
    else:
        Scx_u = sc_u(x)
        Scx_a = sc_a(x)
        Scx_u, Scx_a = jtfs_to_numpy(Scx_u), jtfs_to_numpy(Scx_a)

    # compute energies
    # input energy
    Ex = energy(x)
    if not jtfs and average:
        Ex /= 2**sc.log2_T

    # scattering energy & ratios
    ES = {}
    if not jtfs:
        for o in (0, 1, 2):
            ES[f'S{o}'] = np.sum([energy(Scx[int(i)]['coef']) for i in
                                  np.where(meta['order'] == o)[0]])
    else:
        for pair in Scx_u:
            Scx = Scx_u if pair not in ('S0', 'S1') else Scx_a
            ES[pair] = np.sum([energy(c['coef']) for c in Scx[pair]])

    ESr = {k: v/Ex for k, v in ES.items()}
    if not jtfs:
        ESr['total'] = np.sum(list(ES.values())) / Ex
    else:
        E_common = sum(ES[pair] for pair in ('S0', 'psi_t * phi_f',
                                             'psi_t * psi_f_up',
                                             'psi_t * psi_f_dn'))
        E_v1 = E_common + ES['phi_t * phi_f'] + ES['phi_t * psi_f']
        E_v2 = E_common + ES['S1']
        ESr['total_v1'], ESr['total_v2'] = E_v1 / Ex, E_v2 / Ex

    # print energies
    if T is None:
        T = (sc_a if jtfs else sc).T
    _txt = f", F={F}" if jtfs else ""
    print(f"E(Sx)/E(x) | T={T}{_txt}")
    for k, v in ESr.items():
        print("{:.4f} -- {}".format(v, k))

    if jtfs:
        sc = (sc_u, sc_a)
    return (ESr, Scx, sc) if get_out else ESr


def compute_lp_sum(psi_fs, phi_f=None, J=None, log2_T=None,
                   fold_antianalytic=False):
    lp_sum = 0
    for psi_f in psi_fs:
        lp_sum += np.abs(psi_f)**2
    if phi_f is not None and (
            # else lowest frequency bandpasses are too attenuated
            log2_T is not None and J is not None and log2_T >= J):
        lp_sum += np.abs(phi_f)**2

    if fold_antianalytic:
        lp_sum = fold_lp_sum(lp_sum, analytic_part=True)
    return lp_sum


def fold_lp_sum(lp_sum, analytic_part=True):
    if analytic_part:
        # reflect anti-analytic part onto analytic;
        # goal is energy conservation - if this is ignored and we
        # normalize analytic part perfectly to 2, the non-zero negative
        # freqs will make the filterbank energy-expansive

        # sum onto positives, excluding DC and Nyquist,
        # from negatives, excluding Nyquist
        lp_sum[1:len(lp_sum)//2] += lp_sum[len(lp_sum)//2 + 1:][::-1]
        # zero what we just carried over to not duplicate later by accident
        lp_sum[len(lp_sum)//2 + 1:] = 0
        # with `analytic=True`, this has no effect (all negatives == 0)
        # (note, "analytic" in "analytic_only" includes pseudo-analytic)
    else:
        # above, but in reverse
        lp_sum[len(lp_sum)//2 + 1:] += lp_sum[1:len(lp_sum)//2][::-1]
        lp_sum[1:len(lp_sum)//2] = 0
    return lp_sum


def make_jtfs_pair(N, pair='up', xi0=4, sigma0=1.35):
    """Creates a 2D JTFS wavelet. Used in `wavespin.visuals`."""
    from .scattering1d.filter_bank import morlet_1d, gauss_1d
    from scipy.fft import ifftshift, ifft

    morl = morlet_1d(N, xi=xi0/N, sigma=sigma0/N).squeeze()
    gaus = gauss_1d(N, sigma=sigma0/N).squeeze()

    if pair in ('up', 'dn'):
        i0, i1 = 0, 0
    elif pair == 'phi_f':
        i0, i1 = 1, 0
    elif pair in ('phi_t', 'phi_t_dn'):
        i0, i1 = 0, 1
    elif pair == 'phi':
        i0, i1 = 1, 1
    else:
        supported = {'up', 'dn', 'phi_f', 'phi_t', 'phi', 'phi_t_dn'}
        raise ValueError("unknown pair %s; supported are %s" % (
            pair, '\n'.join(supported)))

    pf_f = (morl, gaus)[i0]
    pt_f = (morl, gaus)[i1]
    pf_f, pt_f = pf_f.copy(), pt_f.copy()
    if pair in ('dn', 'phi_t_dn'):
        # time reversal
        pf_f[1:] = pf_f[1:][::-1]
    pf, pt = [ifftshift(ifft(p)) for p in (pf_f, pt_f)]

    Psi = pf[:, None] * pt[None]
    return Psi


#### Validating 1D filterbank ################################################
def validate_filterbank_tm(sc=None, psi1_f=None, psi2_f=None, phi_f=None,
                           criterion_amplitude=1e-3, verbose=True):
    """Runs `validate_filterbank()` on temporal filters; supports `Scattering1D`
    and `TimeFrequencyScattering1D`.

    Parameters
    ----------
    sc : `Scattering1D` / `TimeFrequencyScattering1D` / None
        If None, then `psi1_f_fr_up`, `psi1_f_fr_dn`, and `phi_f_fr` must
        be not None.

    psi1_f : list[tensor] / None
        First-order bandpasses in frequency domain.
        Overridden if `sc` is not None.

    psi2_f : list[tensor] / None
        Second-order bandpasses in frequency domain.
        Overridden if `sc` is not None.

    phi_f : tensor / None
        Lowpass filter in frequency domain.
        Overridden if `sc` is not None.

    criterion_amplitude : float
        Used for various thresholding in `validate_filterbank()`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data1, data2 : dict, dict
        Returns from `validate_filterbank()` for `psi1_f` and `psi2_f`.
    """
    if sc is None:
        assert not any(arg is None for arg in (psi1_f, psi2_f, phi_f))
    else:
        psi1_f, psi2_f, phi_f = [getattr(sc, k) for k in
                                 ('psi1_f', 'psi2_f', 'phi_f')]
    psi1_f, psi2_f = [[p[0] for p in ps] for ps in (psi1_f, psi2_f)]
    phi_f = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]

    if verbose:
        print("\n// FIRST-ORDER")
    data1 = validate_filterbank(psi1_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    if verbose:
        print("\n\n// SECOND-ORDER")
    data2 = validate_filterbank(psi2_f, phi_f, criterion_amplitude,
                                verbose=verbose,
                                for_real_inputs=True, unimodal=True)
    return data1, data2


def validate_filterbank_fr(sc=None, psi1_f_fr_up=None, psi1_f_fr_dn=None,
                           phi_f_fr=None, psi_id=0, criterion_amplitude=1e-3,
                           verbose=True):
    """Runs `validate_filterbank()` on frequential filters of JTFS.

    Parameters
    ----------
    sc : `TimeFrequencyScattering1D` / None
        JTFS instance. If None, then `psi1_f_fr_up`, `psi1_f_fr_dn`, and
        `phi_f_fr` must be not None.

    psi1_f_fr_up : list[tensor] / None
        Spin up bandpasses in frequency domain.
        Overridden if `sc` is not None.

    psi1_f_fr_dn : list[tensor] / None
        Spin down bandpasses in frequency domain.
        Overridden if `sc` is not None.

    phi_f_fr : tensor / None
        Lowpass filter in frequency domain.
        Overridden if `sc` is not None.

    psi_id : int
        See `psi_id` in `filter_bank_jtfs.psi_fr_factory`.

    criterion_amplitude : float
        Used for various thresholding in `validate_filterbank()`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data_up, data_dn : dict, dict
        Returns from `validate_filterbank()` for `psi1_f_fr_up` and
        `psi1_f_fr_dn`.
    """
    if sc is None:
        assert not any(arg is None for arg in
                       (psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr))
    else:
        psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr = [
            getattr(sc, k) for k in
            ('psi1_f_fr_up', 'psi1_f_fr_dn', 'phi_f_fr')]

    psi1_f_fr_up, psi1_f_fr_dn = psi1_f_fr_up[psi_id], psi1_f_fr_dn[psi_id]
    phi_f_fr = phi_f_fr[0][0][0]

    if verbose:
        print("\n// SPIN UP")
    data_up = validate_filterbank(psi1_f_fr_up, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    if verbose:
        print("\n\n// SPIN DOWN")
    data_dn = validate_filterbank(psi1_f_fr_dn, phi_f_fr, criterion_amplitude,
                                  verbose=verbose,
                                  for_real_inputs=False, unimodal=True)
    return data_up, data_dn


def validate_filterbank(psi_fs, phi_f=None, criterion_amplitude=1e-3,
                        for_real_inputs=True, unimodal=True, is_time_domain=False,
                        verbose=True):
    """Checks whether the wavelet filterbank is well-behaved against several
    criterion:

        1. Analyticity:
          - A: Whether analytic *and* anti-analytic filters are present
               (input should contain only one)
          - B: Extent of (anti-)analyticity - whether there's components
               on other side of Nyquist
          - C: Whether the Nyquist bin is halved

        2. Aliasing:
          - A. Whether peaks are sorted (left to right or right to left).
               If not, it's possible aliasing (or sloppy user input).
          - B. Whether peaks are distributed exponentially or linearly.
               If neither, it's possible aliasing. (Detection isn't foulproof.)

        3. Zero-mean: whether filters are zero-mean (in time domain)

        4. Zero-phase: whether filters are zero-phase

        5. Frequency coverage: whether filters capture every frequency,
           and whether they do so excessively or insufficiently.
             - Measured with Littlewood-Paley sum (sum of energies),
               the "energy transfer function".
             - Also measured with sum of LP sum, in case of imperfect
               analyticity not being accounted for (must fold leaked frequencies,
               see `help(toolkit.compute_lp_sum)`, `fold_antianalytic`).

        6. Frequency-bandwidth tiling: whether upper quarters of frequencies
           follow CQT (fixed `xi/sigma = (center freq) / bandwidth`), and
           whether all wavelet peak frequencies are distributed either
           exponentially or linearly.

           Only upper quarters (i.e. not `0 to N//4`) is checked for CQT because
           the non-CQT portion could be in the majority, but unlikely for it to
           ever span the upper half.

        7. Redundancy: whether filters overlap excessively (this isn't
           necessarily bad).
             - Measured as ratio of product of energies to sum of energies
               of adjacent filters
             - Also measured as peak duplication in frequency domain. Note,
               it's possible to exceed redundancy thershold without duplicating
               peaks, and vice versa (but latter is more difficult).

        8. Decay:
          - A: Whether any filter is a pure sine (occurs if we try to sample
               a wavelet at too low of a center frequency)
          - B: Whether filters decay sufficiently in time domain to avoid
               boundary effects
          - C: Whether filters decay sufficiently in frequency domain
               (bandwidth isn't the entire signal), and whether they decay
               permanently (don't rise again after decaying)
          B may fail for same reason as 8A & 8B (see these).

        9. Temporal peaks:
          - A: Whether peak is at t==0
          - B: Whether there is only one peak
          - C: Whether decay is smooth (else will incur inflection points)
          A and B may fail to hold for lowest xi due to Morlet's corrective
          term; this is proper behavior.
          See https://www.desmos.com/calculator/ivd7t3mjn8

    Parameters
    ----------
    psi_fs : list[tensor]
        Wavelet filterbank, by default in frequency domain (if in time domain,
        set `in_time_domain=True`.
        Analytic or pseudo-analytic, or anti- of either; does not support
        real-valued wavelets (in time domain).

        If `psi_fs` aren't all same length, will pad in time domain and
        center about `n=0` (DFT-symmetrically), with original length's center
        placed at index 0.

        Note, if `psi_fs` are provided in time domain or aren't all same length,
        they're padded such that FFT convolution matches
        `np.convolve(, mode='full')`. If wavelets are properly centered for FFT
        convolution - that is, either at `n=0` or within `ifftshift` or `n=0`,
        then for even lengths, `np.convolve` *will not* produce correct
        results - which is what happens with `scipy.cwt`.

    phi_f : tensor
        Lowpass filter in frequency domain, of same length as `psi_fs`.

    criterion_amplitude : float
        Used for various thresholding.

    for_real_inputs : bool (default True)
        Whether the filterbank is intended for real-only inputs.
        E.g. `False` for spinned bandpasses in JTFS.

    unimodal : bool (default True)
        Whether the wavelets have a single peak in frequency domain.
        If `False`, some checks are omitted, and others might be inaccurate.
        Always `True` for Morlet wavelets.

    in_time_domain : bool (default False)
        Whether `psi_fs` are in time domain. See notes in `psi_fs`.

    verbose : bool (default True)
        Whether to print the report.

    Returns
    -------
    data : dict
        Aggregated testing info, along with the report. For keys, see
        `print(list(data))`. Note, for entries that describe individual filters,
        the indexing corresponds to `psi_fs` sorted in order of decreasing
        peak frequency.
    """
    def pop_if_no_header(report, did_atleast_one_header):
        """`did_atleast_one_header` sets to `False` after every `title()` call,
        whereas `did_header` before every subsection, i.e. a possible
        `if not did_header: report += []`. Former is to pop titles, latter
        is to avoid repeatedly appending subsection text.
        """
        if not did_atleast_one_header:
            report.pop(-1)

    # handle `psi_fs` domain and length ######################################
    # squeeze all for convenience
    psi_fs = [p.squeeze() for p in psi_fs]
    # fetch max length
    max_len = max(len(p) for p in psi_fs)

    # take to freq or pad to max length
    _psi_fs = []  # store processed filters
    # also handle lowpass
    if phi_f is not None:
        psi_fs.append(phi_f)

    for p in psi_fs:
        if len(p) != max_len:
            if not is_time_domain:
                p = ifft(p)
            # right-pad
            orig_len = len(p)
            p = np.pad(p, [0, max_len - orig_len])
            # odd case: circularly-center about n=0; equivalent to `ifftshift`
            # even case: center such that first output index of FFT convolution
            # corresponds to `sum(x, p[::-1][-len(p)//2:])`, where `p` is in
            # time domain. This is what `np.convolve` does, and it's *not*
            # equivalent to FFT convolution after `ifftshift`
            center_idx = orig_len // 2
            p = np.roll(p, -(center_idx - 1))
            # take to freq-domain
            p = fft(p)
        elif is_time_domain:
            center_idx = len(p) // 2
            p = np.roll(p, -(center_idx - 1))
            p = fft(p)
        _psi_fs.append(p)
    psi_fs = _psi_fs
    # recover & detach phi_f
    if phi_f is not None:
        phi_f = psi_fs.pop(-1)

    ##########################################################################

    # set reference filter
    psi_f_0 = psi_fs[0]
    # fetch basic metadata
    N = len(psi_f_0)

    # assert all inputs are same length
    # note, above already guarantees this, but we keep the code logic in case
    # something changes in the future
    for n, p in enumerate(psi_fs):
        assert len(p) == N, (len(p), N)
    if phi_f is not None:
        assert len(phi_f) == N, (len(phi_f), N)

    # initialize report
    report = []
    data = {k: {} for k in ('analytic_a_ratio', 'nonzero_mean', 'sine', 'decay',
                            'imag_mean', 'time_peak_idx', 'n_inflections',
                            'redundancy', 'peak_duplicates')}
    data['opposite_analytic'] = []

    def title(txt):
        return ("\n== {} " + "=" * (80 - len(txt)) + "\n").format(txt)
    # for later
    w_pos = np.linspace(0, N//2, N//2 + 1, endpoint=True).astype(int)
    w_neg = - w_pos[1:-1][::-1]
    w = np.hstack([w_pos, w_neg])
    eps = np.finfo(psi_f_0.dtype).eps

    peak_idxs = np.array([np.argmax(np.abs(p)) for p in psi_fs])
    peak_idxs_sorted = np.sort(peak_idxs)
    if unimodal and not (np.all(peak_idxs == peak_idxs_sorted) or
                         np.all(peak_idxs == peak_idxs_sorted[::-1])):
        warnings.warn("`psi_fs` peak locations are not sorted; a possible reason "
                      "is aliasing. Will sort, breaking mapping with input's.")
        data['not_sorted'] = True
        peak_idxs = peak_idxs_sorted

    # Analyticity ############################################################
    # check if there are both analytic and anti-analytic bandpasses ##########
    report += [title("ANALYTICITY")]
    did_header = did_atleast_one_header = False

    peak_idx_0 = np.argmax(psi_f_0)
    if peak_idx_0 == N // 2:  # ambiguous case; check next filter
        peak_idx_0 = np.argmax(psi_fs[1])
    analytic_0 = bool(peak_idx_0 < N//2)
    # assume entire filterbank is per psi_0
    analyticity = "analytic" if analytic_0 else "anti-analytic"

    # check whether all is analytic or anti-analytic
    found_counteranalytic = False
    for n, p in enumerate(psi_fs[1:]):
        peak_idx_n = np.argmax(np.abs(p))
        analytic_n = bool(peak_idx_n < N//2)
        if not (analytic_0 is analytic_n):
            if not did_header:
                report += [("Found analytic AND anti-analytic filters in same "
                            "filterbank! psi_fs[0] is {}, but the following "
                            "aren't:\n").format(analyticity)]
                did_header = did_atleast_one_header = True
            report += [f"psi_fs[{n}]\n"]
            data['opposite_analytic'].append(n)
            found_counteranalytic = True

    # set `is_analytic` based on which there are more of
    if not found_counteranalytic:
        is_analytic = analytic_0
    else:
        n_analytic     = sum(np.argmax(np.abs(p)) <= N//2 for p in psi_fs)
        n_antianalytic = sum(np.argmax(np.abs(p)) >= N//2 for p in psi_fs)
        if n_analytic > n_antianalytic or n_analytic == n_antianalytic:
            is_analytic = True
        else:
            is_analytic = False
        report += [("\nIn total, there are {} analytic and {} anti-analytic "
                    "filters\n").format(n_analytic, n_antianalytic)]

    # determine whether the filterbank is strictly analytic/anti-analytic
    if is_analytic:
        negatives_all_zero = False
        for p in psi_fs:
            # exclude Nyquist as it's both in analytic and anti-analytic
            if not np.allclose(p[len(p)//2 + 1:], 0.):
                break
        else:
            negatives_all_zero = True
        strict_analyticity = negatives_all_zero
    else:
        positives_all_zero = False
        for p in psi_fs:
            # exclude DC, one problem at a time; exclude Nyquist
            if not np.allclose(p[1:len(p)//2], 0.):
                break
        else:
            positives_all_zero = True
        strict_analyticity = positives_all_zero

    # determine whether the Nyquist bin is halved
    if strict_analyticity:
        did_header = False
        pf = psi_fs[0]
        if is_analytic:
            nyquist_halved = bool(pf[N//2 - 1] / pf[N//2] > 2)
        else:
            nyquist_halved = bool(pf[N//2 + 1] / pf[N//2] > 2)
        if not nyquist_halved:
            report += [("Nyquist bin isn't halved for strictly analytic wavelet; "
                        "yields improper analyticity with bad time decay.\n")]
            did_header = did_atleast_one_header = True

    # check if any bandpass isn't strictly analytic/anti- ####################
    did_header = False
    th_ratio = (1 / criterion_amplitude)
    for n, p in enumerate(psi_fs):
        ap = np.abs(p)
        # assume entire filterbank is per psi_0
        if is_analytic:
            # Nyquist is *at* N//2, so to include in sum, index up to N//2 + 1
            a_ratio = (ap[:N//2 + 1].sum() / (ap[N//2 + 1:].sum() + eps))
        else:
            a_ratio = (ap[N//2:].sum() / (ap[:N//2].sum() + eps))
        if a_ratio < th_ratio:
            if not did_header:
                report += [("\nFound not strictly {} filter(s); threshold for "
                            "ratio of `spectral sum` to `spectral sum past "
                            "Nyquist` is {} - got (less is worse):\n"
                            ).format(analyticity, th_ratio)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, a_ratio)]
            data['analytic_a_ratio'][n] = a_ratio

    # check if any bandpass isn't zero-mean ##################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("ZERO-MEAN")]
    did_header = did_atleast_one_header = False

    for n, p in enumerate(psi_fs):
        if p[0] != 0:
            if not did_header:
                report += ["Found non-zero mean filter(s)!:\n"]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}][0] == {:.2e}\n".format(n, p[0])]
            data['nonzero_mean'][n] = p[0]

    # Littlewood-Paley sum ###################################################
    def report_lp_sum(report, phi):
        with_phi = not isinstance(phi, int)
        s = "with" if with_phi else "without"
        report += [title("LP-SUM (%s phi)" % s)]
        did_header = did_atleast_one_header = False

        # compute parameters #################################################
        # finish computing lp sum
        lp_sum = lp_sum_psi + np.abs(phi)**2
        lp_sum = (lp_sum[:N//2 + 1] if is_analytic else
                  lp_sum[N//2:])
        if with_phi:
            data['lp'] = lp_sum
        else:
            data['lp_no_phi'] = lp_sum
        if not with_phi and is_analytic:
            lp_sum = lp_sum[1:]  # exclude dc

        # excess / underflow
        diff_over  = lp_sum - th_lp_sum_over
        diff_under = th_lp_sum_under - lp_sum
        diff_over_max, diff_under_max = diff_over.max(), diff_under.max()
        excess_over  = np.where(diff_over  > th_sum_excess)[0]
        excess_under = np.where(diff_under > th_sum_excess)[0]
        if not is_analytic:
            excess_over  += N//2
            excess_under += N//2
        elif is_analytic and not with_phi:
            excess_over += 1
            excess_under += 1  # dc

        # lp sum sum
        lp_sum_sum = lp_sum.sum()
        # `1` per bin, minus
        #   - DC bin, since no phi
        #   - half of Nyquist bin, since `analytic=True` cannot ever get a full
        #     Nyquist (Nyquist bin is halved, so even in best case of the peak
        #     placed at Nyquist, we get 0.5). Unclear if any correction is due
        #     on this.
        # negligible adjustments if `N` is large (JTFS N_frs can be small enough)
        expected_sum = N
        if not with_phi:
            expected_sum -= 1
        if strict_analyticity:
            expected_sum -= .5

        # scale according to tolerance.
        # tolerances determined empirically from the most conservative case;
        # see `tests.test_jtfs.test_lp_sum`
        th_sum_above = .01
        th_sum_below = .15
        expected_above = expected_sum * (1 + th_sum_above)
        expected_below = expected_sum * (1 - th_sum_below)

        # append report entries ##############################################
        input_kind = "real" if for_real_inputs else "complex"
        if len(excess_over) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_over) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum exceeds threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_over, input_kind, diff_over_max,
                                 N//2, s)]
            report += ["{}\n\n".format(w[excess_over][::stride])]
            did_header = did_atleast_one_header = True
            if with_phi:
                data['lp_excess_over'] = excess_over
                data['lp_excess_over_max'] = diff_over_max
            else:
                data['lp_no_phi_excess_over'] = excess_over
                data['lp_no_phi_excess_over_max'] = diff_over_max

        if len(excess_under) > 0:
            # show at most 30 values
            stride = max(int(round(len(excess_under) / 30)), 1)
            s = f", shown skipping every {stride-1} values" if stride != 1 else ""
            report += [("LP sum falls below threshold of {} (for {} inputs) by "
                        "at most {:.3f} (more is worse; ~{} implies ~zero "
                        "capturing of the frequency!) at following frequency "
                        "bin indices (0 to {}{}):\n"
                        ).format(th_lp_sum_under, input_kind, diff_under_max,
                                 th_lp_sum_under, N//2, s)]
            # w_show = np.round(w[excess_under][::stride], 3)
            report += ["{}\n\n".format(w[excess_under][::stride])]
            did_header = did_atleast_one_header = True
            if with_phi:
                data['lp_excess_under'] = excess_under
                data['lp_excess_under_max'] = diff_under_max
            else:
                data['lp_no_phi_excess_under'] = excess_under
                data['lp_no_phi_excess_under_max'] = diff_under_max

        if lp_sum_sum > expected_above:
            report += [("LP sum sum exceeds expected: {} > {}. If LP sum "
                        "otherwise has no excess, then there may be leakage due "
                        "to imperfect analyticity, corrected by folding; see "
                        "help(toolkit.fold_lp_sum)\n").format(lp_sum_sum,
                                                              expected_above)]
            did_header = did_atleast_one_header = True
            diff = lp_sum_sum - expected_above
            if with_phi:
                data['lp_sum_sum_excess_over'] = diff
            else:
                data['lp_sum_sum_no_phi_excess_over'] = diff

        if lp_sum_sum < expected_below:
            report += [("LP sum sum falls short of expected: {} < {}. If LP sum "
                        "otherwise doesn't fall short, then there may be leakage "
                        "due to imperfect analyticity, corrected by folding; see "
                        "help(toolkit.fold_lp_sum)\n").format(lp_sum_sum,
                                                              expected_below)]
            did_header = did_atleast_one_header = True
            diff = expected_below - lp_sum_sum
            if with_phi:
                data['lp_sum_sum_excess_under'] = diff
            else:
                data['lp_sum_sum_no_phi_excess_under'] = diff

        if did_header:
            stdev = np.abs(lp_sum[lp_sum >= th_lp_sum_under] -
                           th_lp_sum_under).std()
            report += [("Mean absolute deviation from tight frame: {:.2f}\n"
                        "Standard deviation from tight frame: {:.2f} "
                        "(excluded LP sum values below {})\n").format(
                            np.abs(diff_over).mean(), stdev, th_lp_sum_under)]

        pop_if_no_header(report, did_atleast_one_header)

    pop_if_no_header(report, did_atleast_one_header)
    th_lp_sum_over = 2 if for_real_inputs else 1
    th_lp_sum_under = th_lp_sum_over / 2
    th_sum_excess = (1 + criterion_amplitude)**2 - 1
    lp_sum_psi = np.sum([np.abs(p)**2 for p in psi_fs], axis=0)
    # fold opposite frequencies to ensure leaks are accounted for
    lp_sum_psi = fold_lp_sum(lp_sum_psi, analytic_part=is_analytic)

    # do both cases
    if phi_f is not None:
        report_lp_sum(report, phi=phi_f)
    report_lp_sum(report, phi=0)

    # Redundancy #############################################################
    from .scattering1d.filter_bank import compute_filter_redundancy

    report += [title("REDUNDANCY")]
    did_header = did_atleast_one_header = False
    max_to_print = 20

    # overlap ####
    th_r = .4 if for_real_inputs else .2

    printed = 0
    for n in range(len(psi_fs) - 1):
        r = compute_filter_redundancy(psi_fs[n], psi_fs[n + 1])
        data['redundancy'][(n, n + 1)] = r
        if r > th_r:
            if not did_header:
                report += [("Found filters with redundancy exceeding {} (energy "
                            "overlap relative to sum of individual energies) "
                            "-- This isn't necessarily bad. Showing up to {} "
                            "filters:\n").format(th_r, max_to_print)]
                did_header = did_atleast_one_header = True
            if printed < max_to_print:
                report += ["psi_fs[{}] & psi_fs[{}]: {:.3f}\n".format(
                    n, n + 1, r)]
                printed += 1

    # peak duplication ####
    did_header = False

    printed = 0
    for n, peak_idx in enumerate(peak_idxs):
        if np.sum(peak_idx == peak_idxs) > 1:
            data['peak_duplicates'][n] = peak_idx
            if not did_header:
                spc = "\n" if did_atleast_one_header else ""
                report += [("{}Found filters with duplicate peak frequencies! "
                            "Showing up to {} filters:\n").format(spc,
                                                                  max_to_print)]
                did_header = did_atleast_one_header = True
            if printed < max_to_print:
                report += ["psi_fs[{}], peak_idx={}\n".format(n, peak_idx)]
                printed += 1

    # Decay: check if any bandpass is a pure sine ############################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (check for pure sines)")]
    did_header = did_atleast_one_header = False
    th_ratio_max_to_next_max = (1 / criterion_amplitude)

    for n, p in enumerate(psi_fs):
        psort = np.sort(np.abs(p))  # least to greatest
        ratio = psort[-1] / (psort[-2] + eps)
        if ratio > th_ratio_max_to_next_max:
            if not did_header:
                report += [("Found filter(s) that are pure sines! Threshold for "
                            "ratio of Fourier peak to next-highest value is {} "
                            "- got (more is worse):\n"
                            ).format(th_ratio_max_to_next_max)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.2e}\n".format(n, ratio)]
            data['sine'][n] = ratio

    # Decay: frequency #######################################################
    from .scattering1d.filter_bank import compute_bandwidth

    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (frequency)")]
    did_header = did_atleast_one_header = False

    # compute bandwidths
    bandwidths = [compute_bandwidth(pf, criterion_amplitude)
                  for pf in psi_fs]

    excess_bw = N//2 if strict_analyticity else N
    for n, bw in enumerate(bandwidths):
        if bw == excess_bw:
            if not did_header:
                report += [("Found filter(s) that never sufficiently decay "
                            "in frequency:\n")]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}], bandwidth={}\n".format(n, bw)]

    # handle case where a filter first decays and then rises again
    if unimodal:
        def decayed_then_rose(epf):
            criterion_energy = criterion_amplitude**2
            decay_idxs = np.where(epf < criterion_energy)[0]
            if len(decay_idxs) == 0:
                # never decayed
                return False

            first_decay_idx = decay_idxs[0]
            bound = len(epf)//2  # exclude opposite half
            rise_idxs = np.where(epf[first_decay_idx + 1:bound + 1] >
                                 criterion_energy)[0]
            return bool(len(rise_idxs) > 0)

        did_header = False
        for n, pf in enumerate(psi_fs):
            # center about n=0 to handle left & right separately
            pf = np.roll(pf, -np.argmax(np.abs(pf)))
            epf = np.abs(pf)**2

            dtr_right = decayed_then_rose(epf)
            # frequency-reverse
            epf[1:] = epf[1:][::-1]
            dtr_left = decayed_then_rose(epf)

            # both apply regardless of `strict_analyticity`
            # (since one of them should be impossible if it's `True`)
            if dtr_left or dtr_right:
                if not did_header:
                    report += [("Found filter(s) that decay then rise again in "
                                "frequency:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]\n".format(n)]

    # Decay: boundary effects ################################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("DECAY (boundary effects)")]
    did_header = did_atleast_one_header = False
    th_ratio_max_to_min = (1 / criterion_amplitude)

    psis = [np.fft.ifft(p) for p in psi_fs]
    apsis = [np.abs(p) for p in psis]
    for n, ap in enumerate(apsis):
        ratio = ap.max() / (ap.min() + eps)
        if ratio < th_ratio_max_to_min:
            if not did_header:
                report += [("Found filter(s) with incomplete decay (will incur "
                            "boundary effects), with following ratios of "
                            "amplitude max to edge (less is worse; threshold "
                            "is {}):\n").format(1 / criterion_amplitude)]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1f}\n".format(n, ratio)]
            data['decay'][n] = ratio

    # check lowpass
    if phi_f is not None:
        aphi = np.abs(np.fft.ifft(phi_f))
        ratio = aphi.max() / (aphi.min() + eps)
        if ratio < th_ratio_max_to_min:
            nl = "\n" if did_header else ""
            report += [("{}Lowpass filter has incomplete decay (will incur "
                        "boundary effects), with following ratio of amplitude "
                        "max to edge: {:.1f} > {}\n").format(nl, ratio,
                                                             th_ratio_max_to_min)]
            did_header = did_atleast_one_header = True
            data['decay'][-1] = ratio

    # Phase ##################################################################
    pop_if_no_header(report, did_atleast_one_header)
    report += [title("PHASE")]
    did_header = did_atleast_one_header = False
    th_imag_mean = eps

    for n, p in enumerate(psi_fs):
        imag_mean = np.abs(p.imag).mean()
        if imag_mean > th_imag_mean:
            if not did_header:
                report += [("Found filters with non-zero phase, with following "
                            "absolute mean imaginary values:\n")]
                did_header = did_atleast_one_header = True
            report += ["psi_fs[{}]: {:.1e}\n".format(n, imag_mean)]
            data['imag_mean'][n] = imag_mean

    # Aliasing ###############################################################
    def diff_extend(diff, th, cond='gt', order=1):
        # the idea is to take `diff` without losing samples, if the goal is
        # `where(diff == 0)`; `diff` is forward difference, and two samples
        # participated in producing the zero, where later one's index is dropped
        # E.g. detecting duplicate peak indices:
        # [0, 1, 3, 3, 5] -> diff gives [2], so take [2, 3]
        # but instead of adding an index, replace next sample with zero such that
        # its `where == 0` produces that index
        if order > 1:
            diff_e = diff_extend(diff, th)
            for o in range(order - 1):
                diff_e = diff_e(diff_e, th)
            return diff_e

        diff_e = []
        d_extend = 2*th if cond == 'gt' else th
        prev_true = False
        for d in diff:
            if prev_true:
                diff_e.append(d_extend)
                prev_true = False
            else:
                diff_e.append(d)
            if (cond == 'gt' and np.abs(d) > th or
                cond == 'eq' and np.abs(d) == th):
                prev_true = True
        if prev_true:
            # last sample was zero; extend
            diff_e.append(d_extend)
        return np.array(diff_e)

    if unimodal:
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("ALIASING")]
        did_header = did_atleast_one_header = False
        eps_big = eps * 100  # ease threshold for "zero"

        if len(peak_idxs) < 6:
            warnings.warn("Alias detector requires at least 6 wavelets to "
                          "work properly, per repeated `np.diff`")

        # check whether peak locations follow a linear or exponential
        # distribution, progressively dropping those that do to see if any remain

        # x[n] = A^n + C; x[n] - x[n - 1] = A^n - A^(n-1) = A^n*(1 - A) = A^n*C
        # log(A^n*C) = K + n; diff(diff(K + n)) == 0
        # `abs` for anti-analytic case with descending idxs
        logdiffs = np.diff(np.log(np.abs(np.diff(peak_idxs))), 2)
        # In general it's impossible to determine whether a rounded sequence
        # samples an exponential, since if the exponential rate (A in A^n) is
        # sufficiently small, its rounded values will be linear over some portion.
        # However, it cannot be anything else, and we are okay with linear
        # (unless constant, i.e. duplicate, captured elsewhere) - thus the net
        # case of `exp + lin` is still captured. The only uncertainty is in
        # the threshold; assuming deviation by at most 1 sample, we set it to 1.
        # A test is:
        # `for b in linspace(1.2, 6.5, 500): x = round(b**arange(10) + 50)`
        # with `if any(abs(diff, o).min() == 0 for o in (1, 2, 3)): continue`,
        # Another with: `linspace(.2, 1, 500)` and `round(256*b**arange(10) + 50)`
        # to exclude `x` with repeated or linear values
        # However, while this has no false positives (never misses an exp/lin),
        # it can also count some non-exp/lin as exp/lin, but this is rare.
        # To be safe, per above test, we use the empirical value of 0.9
        logdiffs_extended = diff_extend(logdiffs, .9)
        if len(logdiffs_extended) > len(logdiffs) + 2:
            # this could be `assert` but not worth erroring over this
            warnings.warn("`len(logdiffs_extended) > len(logdiffs) + 2`; will "
                          "use more conservative estimate on peaks distribution")
            logdiffs_extended = logdiffs
        keep = np.where(np.abs(logdiffs_extended) > .9)
        # note due to three `diff`s we artificially exclude 3 samples
        peak_idxs_remainder = peak_idxs[keep]

        # now constant (diff_order==1) and linear (diff_order==2)
        for diff_order in (1, 2):
            idxs_diff2 = np.diff(peak_idxs_remainder, diff_order)
            keep = np.where(np.abs(idxs_diff2) > eps_big)
            peak_idxs_remainder = peak_idxs_remainder[keep]

        # if anything remains, it's neither
        if len(peak_idxs_remainder) > 0:
            report += [("Found Fourier peaks that are spaced neither "
                        "exponentially nor linearly, suggesting possible "
                        "aliasing.\npsi_fs[n], n={}\n"
                        ).format(peak_idxs_remainder)]
            data['alias_peak_idxs'] = peak_idxs_remainder
            did_header = did_atleast_one_header = True

    # Frequency-bandwidth tiling; CQT ########################################
    # note, we checked for linear/exponential spacing in "Aliasing" section
    if unimodal:
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("FREQUENCY-BANDWIDTH TILING")]
        did_header = did_atleast_one_header = False

        def isnt_lower_quarter(pidx):
            return ((is_analytic and pidx > N//8) or
                    (not is_analytic and pidx < (N - N//8)))

        got_peaks_above_first_quarter = any(isnt_lower_quarter(peak_idx)
                                            for peak_idx in peak_idxs)
        if got_peaks_above_first_quarter:
            # idxs must reflect distance from DC
            if is_analytic:
                peak_idxs_dist = peak_idxs
            else:
                peak_idxs_dist = [N - peak_idx for peak_idx in peak_idxs]

            # compute bandwidths, accounting for strict analyticity;
            # can infer full intended bandwidth from just one half
            if strict_analyticity:
                if is_analytic:
                    # right is trimmed
                    bandwidths = [compute_bandwidth(pf, criterion_amplitude,
                                                    left_only=True)
                                  for pf in psi_fs]
                else:
                    # left is trimmed
                    bandwidths = [compute_bandwidth(pf, criterion_amplitude,
                                                    right_only=True)
                                  for pf in psi_fs]
            else:
                bandwidths = [compute_bandwidth(pf, criterion_amplitude)
                              for pf in psi_fs]

            Qs_upper_quarters = {n: peak_idx_dist / bw
                                 for n, (peak_idx_dist, bw)
                                 in enumerate(zip(peak_idxs_dist, bandwidths))
                                 # must still use original peak idxs here
                                 if isnt_lower_quarter(peak_idxs[n])}

            Qs_values = list(Qs_upper_quarters.values())
            tolerance = .01  # abs relative difference tolerance 1%
            # pick most favorable reference
            Qs_diffs = np.abs(np.diff(Qs_values))
            Q_ref = Qs_values[np.argmin(Qs_diffs) + 1]

            non_cqts = []
            for n, Q in Qs_upper_quarters.items():
                if abs(Q - Q_ref) / Q_ref > tolerance:
                    non_cqts.append((n, Q))

            if len(non_cqts) > 0:
                non_cqt_strs = ["psi_fs[{}], Q={}".format(n, Q)
                                for n, Q in zip(*zip(*non_cqts))]
                report += [("Found non-CQT wavelets in upper quarters of "
                            "frequencies - i.e., `(center freq) / bandwidth` "
                            "isn't constant: \n{}\n"
                            ).format("\n".join(non_cqt_strs))]
                data['non_cqts'] = non_cqts
                did_header = did_atleast_one_header = True

    # Temporal peak ##########################################################
    if unimodal:
        # check that temporal peak is at t==0 ################################
        pop_if_no_header(report, did_atleast_one_header)
        report += [title("TEMPORAL PEAK")]
        did_header = did_atleast_one_header = False

        for n, ap in enumerate(apsis):
            peak_idx = np.argmax(ap)
            if peak_idx != 0:
                if not did_header:
                    report += [("Found filters with temporal peak not at t=0!, "
                                "with following peak locations:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]: {}\n".format(n, peak_idx)]
                data['time_peak_idx'][n] = peak_idx

        # check that there is only one temporal peak #########################
        did_header = False
        for n, ap in enumerate(apsis):
            # count number of inflection points (where sign of derivative changes)
            # exclude very small values
            # center for proper `diff`
            ap = np.fft.ifftshift(ap)
            inflections = np.diff(np.sign(np.diff(ap[ap > 10*eps])))
            n_inflections = sum(np.abs(inflections) > eps)

            if n_inflections > 1:
                if not did_header:
                    report += [("\nFound filters with multiple temporal peaks "
                                "(or incomplete/non-smooth decay)! "
                                "(more precisely, >1 inflection points) with "
                                "following number of inflection points:\n")]
                    did_header = did_atleast_one_header = True
                report += ["psi_fs[{}]: {}\n".format(n, n_inflections)]
                data['n_inflections'] = n_inflections
    else:
        pop_if_no_header(report, did_atleast_one_header)

    # Print report ###########################################################
    report = ''.join(report)
    data['report'] = report
    if verbose:
        if len(report) == 0:
            print("Perfect filterbank!")
        else:
            print(report)
    return data


#### energy & distance #######################################################
def energy(x, axis=None, kind='l2'):
    """Compute energy. L1==`sum(abs(x))`, L2==`sum(abs(x)**2)` (so actually L2^2).
    """
    x = x['coef'] if isinstance(x, dict) else x
    B = ExtendedUnifiedBackend(x)
    out = (B.norm(x, ord=1, axis=axis) if kind == 'l1' else
           B.norm(x, ord=2, axis=axis)**2)
    if np.prod(out.shape) == 1:
        out = float(out)
    return out


def l2(x, axis=None, keepdims=True):
    """`sqrt(sum(abs(x)**2))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=2, axis=axis, keepdims=keepdims)

def rel_l2(x0, x1, axis=None, adj=False):
    """Coeff distance measure; Eq 2.24 in
    https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf
    """
    ref = l2(x0, axis) if not adj else (l2(x0, axis) + l2(x1, axis)) / 2
    return l2(x1 - x0, axis) / ref


def l1(x, axis=None, keepdims=True):
    """`sum(abs(x))`."""
    B = ExtendedUnifiedBackend(x)
    return B.norm(x, ord=1, axis=axis, keepdims=keepdims)

def rel_l1(x0, x1, adj=False, axis=None):
    ref = l1(x0, axis) if not adj else (l1(x0, axis) + l1(x1, axis)) / 2
    return l1(x1 - x0, axis) / ref


def rel_ae(x0, x1, eps=None, ref_both=True):
    """Relative absolute error."""
    B = ExtendedUnifiedBackend(x0)
    if ref_both:
        if eps is None:
            eps = _eps(x0, x1)
        ref = (x0 + x1)/2 + eps
    else:
        if eps is None:
            eps = _eps(x0)
        ref = x0 + eps
    return B.abs(x0 - x1) / ref


def _eps(x0, x1=None):
    B = ExtendedUnifiedBackend(x0)
    if x1 is None:
        eps = B.abs(x0).max() / 1000
    else:
        eps = (B.abs(x0).max() + B.abs(x1).max()) / 2000
    eps = max(eps, 10 * np.finfo(B.numpy(x0).dtype).eps)
    return eps

#### test signals ###########################################################
def echirp(N, fmin=1, fmax=None, tmin=0, tmax=1):
    """https://overlordgolddragon.github.io/test-signals/ (bottom)"""
    fmax = fmax or N // 2
    t = np.linspace(tmin, tmax, N)

    phi = _echirp_fn(fmin, fmax, tmin, tmax)(t)
    return np.cos(phi)


def _echirp_fn(fmin, fmax, tmin=0, tmax=1):
    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)
    phi = lambda t: 2*np.pi * (a/np.log(b)) * (b**t - b**tmin)
    return phi


def fdts(N, n_partials=2, total_shift=None, f0=None, seg_len=None,
         partials_f_sep=1.6, global_shift=0, brick_spectrum=False,
         endpoint=False):
    """Generate windowed tones with Frequency-dependent Time Shifts (FDTS)."""
    def brick(g):
        gf = np.fft.rfft(g)

        # center at dc
        ct = np.argmax(np.abs(gf))
        gf_ct = np.roll(gf, -ct)
        agf_ct = np.abs(gf_ct)
        # brickwall width = ~support width
        # decays slower so pick smaller criterion_amplitude
        width = np.where(agf_ct < agf_ct.max() / 10000)[0][0]
        brick_f = np.zeros(len(g)//2 + 1)
        brick_f[:width] = 1
        brick_f[-width:] = 1
        gf_ct *= brick_f

        gf_bricked = np.roll(gf_ct, ct)
        g_bricked = np.fft.irfft(gf_bricked)
        return g_bricked

    total_shift = total_shift or N//16
    f0 = f0 or N//12
    seg_len = seg_len or N//8

    t = np.linspace(0, 1, N, endpoint=endpoint)
    window = scipy.signal.tukey(seg_len, alpha=0.5)
    pad_right = (N - len(window)) // 2
    pad_left = N - len(window) - pad_right
    window = np.pad(window, [pad_left, pad_right])

    x = np.zeros(N)
    xs = x.copy()
    for p in range(0, n_partials):
        f_shift = partials_f_sep**p
        x_partial = np.sin(2*np.pi * f0 * f_shift * t) * window
        if brick_spectrum:
            x_partial = brick(x_partial)

        partial_shift = int(total_shift * np.log2(f_shift) / np.log2(n_partials))
        xs_partial = np.roll(x_partial, partial_shift)
        x += x_partial
        xs += xs_partial

    if global_shift:
        x = np.roll(x, global_shift)
        xs = np.roll(xs, global_shift)
    return x, xs

#### misc ###################################################################
def tensor_padded(seq, pad_value=0, init_fn=None, cast_fn=None, ref_shape=None,
                  left_pad_axis=None, general=True):
    """Make tensor from variable-length `seq` (e.g. nested list) padded with
    `fill_value`.

    Parameters
    ----------
    seq : list[tensor]
        Nested list of tensors.

    pad_value : float
        Value to pad with.

    init_fn : function / None
        Instantiates packing tensor, e.g. `lambda shape: torch.zeros(shape)`.
        Defaults to `backend.full`.

    cast_fn : function / None
        Function to cast tensor values before packing, e.g.
        `lambda x: torch.tensor(x)`.

    ref_shape : tuple[int] / None
        Tensor output shape to pack into, instead of what's inferred for `seq`,
        as long as >= that shape. Shape inferred from `seq` is the minimal size
        determined from longest list at each nest level, i.e. we can't go lower
        without losing elements.

        Tuple can contain `None`: `(None, 3)` will pad dim0 per `seq` and dim1
        to 3, unless `seq`'s dim1 is 4, then will pad to 4.

        Recommended to pass this argument if applying `tensor_padded` multiple
        times, as time to infer shape is significant, especially relative to
        GPU computation.

    left_pad_axis : int / tuple[int] / None
        Along these axes, will pad from left instead of the default right.
        Not implemented for dim0 (`0 in left_pad_axis`).

    general : bool (default True)
        If `False`, will use a much faster routine that's for JTFS.

    Not implemented for TensorFlow: will convert to numpy array then rever to
    TF tensor.

    Code borrows from: https://stackoverflow.com/a/27890978/10133797
    """
    iter_axis = [0]
    prev_axis = [iter_axis[0]]

    def fill_tensor(arr, seq, fill_value=0):
        if iter_axis[0] != prev_axis[0]:
            prev_axis[0] = iter_axis[0]

        if arr.ndim == 1:
            try:
                len_ = len(seq)
            except TypeError:
                len_ = 0

            if len_ == 0:
                pass
            elif len(shape) not in left_pad_axis:  # right pad
                arr[:len_] = cast_fn(seq)
            else:  # left pad
                arr[-len_:] = cast_fn(seq)
        else:
            iter_axis[0] += 1

            left_pad = bool(iter_axis[0] in left_pad_axis)
            if left_pad:
                seq = IterWithDelay(seq, len(arr) - len(seq), fillvalue=())

            for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
                fill_tensor(subarr, subseq, fill_value)
                if subarr.ndim != 1:
                    iter_axis[0] -= 1

    # infer `init_fn` and `cast_fn` from `seq`, if not provided ##############
    backend, backend_name = _infer_backend(seq, get_name=True)
    is_tf = bool(backend_name == 'tensorflow')
    if is_tf:
        tf = backend
        backend = np
        backend_name = 'numpy'

    # infer dtype & whether on GPU
    sq = seq
    while isinstance(sq, list):
        sq = sq[0]
    dtype = sq.dtype if hasattr(sq, 'dtype') else type(sq)
    if backend_name == 'torch':
        device = sq.device
    else:
        device = None

    if init_fn is None:
        if backend_name == 'numpy':
            if is_tf:
                dtype = dtype.name
            init_fn = lambda s: np.full(s, pad_value, dtype=dtype)
        elif backend_name == 'torch':
            init_fn = lambda s: backend.full(s, pad_value, dtype=dtype,
                                             device=device)

    if cast_fn is None:
        if is_tf:
            cast_fn = lambda x: x.numpy()
        elif backend_name == 'numpy':
            cast_fn = lambda x: x
        elif backend_name == 'torch':
            cast_fn = lambda x: (backend.tensor(x, device=device)
                                 if not isinstance(x, backend.Tensor) else x)

    ##########################################################################
    # infer shape if not provided
    if ref_shape is None or any(s is None for s in ref_shape):
        shape = list(find_shape(seq, fast=not general))
        # override shape with `ref_shape` where provided
        if ref_shape is not None:
            for i, s in enumerate(ref_shape):
                if s is not None and s >= shape[i]:
                    shape[i] = s
    else:
        shape = ref_shape
    shape = tuple(shape)

    # handle `left_pad_axis`
    if left_pad_axis is None:
        left_pad_axis = ()
    elif isinstance(left_pad_axis, int):
        left_pad_axis = [left_pad_axis]
    elif isinstance(left_pad_axis, tuple):
        left_pad_axis = list(left_pad_axis)
    # negatives -> positives
    for i, a in enumerate(left_pad_axis):
        if a < 0:
            # +1 since counting index depth which goes `1, 2, ...`
            left_pad_axis[i] = (len(shape) + 1) + a
    if 0 in left_pad_axis:
        raise NotImplementedError("`0` in `left_pad_axis`")

    # fill
    arr = init_fn(shape)
    fill_tensor(arr, seq, fill_value=pad_value)

    # revert if needed
    if is_tf:
        arr = tf.convert_to_tensor(arr)
    return arr


def find_shape(seq, fast=False):
    """Finds shape to pad a variably nested list to.
    `fast=True` uses an implementation optimized for JTFS.
    """
    if fast:
        """Assumes 4D/5D and only variable length lists are in dim3."""
        flat = chain.from_iterable
        try:
            dim4 = len(seq[0][0][0][0])  # 5D
            dims = (len(seq), len(seq[0]), len(seq[0][0]),
                    max(map(len, flat(flat(seq)))), dim4)
        except:
            dims = (len(seq), len(seq[0]),
                    max(map(len, flat(seq))), len(seq[0][0][0]))
    else:
        dims = _find_shape_gen(seq)
    return dims


def _find_shape_gen(seq):
    """Code borrows from https://stackoverflow.com/a/27890978/10133797"""
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [_find_shape_gen(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in
                           zip_longest(*shapes, fillvalue=1))


class IterWithDelay():
    """Allows implementing left padding by delaying iteration of a sequence."""
    def __init__(self, x, delay, fillvalue=()):
        self.x = x
        self.delay = delay
        self.fillvalue = fillvalue

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx > self.delay - 1:
            idx = self.idx - self.delay
            if idx < len(self.x):
                out = self.x[idx]
            else:
                raise StopIteration
        else:
            out = self.fillvalue
        self.idx += 1
        return out


def fill_default_args(cfg, defaults, copy_original=True,
                      check_against_defaults=False):
    """If a key is present in `defaults` but not in `cfg`, then copies
    the key-value pair from `defaults` onto `cfg`. Also applies to nests.

    `check_against_defaults` will raise Exception is there's keys in `cfg`
    that aren't in `defaults`.
    """
    if cfg is None or cfg == {}:
        return defaults
    elif not isinstance(cfg, dict):
        raise ValueError("`cfg` must be dict or None, got %s" % type(cfg))

    if copy_original:
        cfg = deepcopy(cfg)  # don't affect external

    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        else:
            if isinstance(v, dict):
                cfg[k] = fill_default_args(cfg[k], v)

    if check_against_defaults:
        for k in cfg:
            if k not in defaults:
                raise ValueError("unknown kwarg: '{}', supported are:\n{}".format(
                    k, '\n'.join(list(cfg))))
    return cfg


def get_phi_for_psi_id(jtfs, psi_id):
    """Returns `phi_f_fr` at appropriate length, but always of scale `log2_F`."""
    scale_diff = list(jtfs.psi_ids.values()).index(psi_id)
    pad_diff = jtfs.J_pad_frs_max_init - jtfs.J_pad_frs[scale_diff]
    return jtfs.phi_f_fr[0][pad_diff][0]


# decimate object ############################################################
class Decimate():
    def __init__(self, backend='numpy', gpu=None, dtype=None,
                 sign_correction='abs', cutoff_mult=1.):
        """Windowed-sinc decimation.

        Parameters
        ----------
        backend : str['numpy', 'torch', 'tensorflow'] / module
            Name of module, or module object, to use as backend.
              - 'torch' defaults to using GPU and single precision.
              - 'tensorflow' is not supported.

        gpu : bool / None
            Whether to use GPU (torch/tensorflow backends only). For 'torch'
            backend, defaults to True.

        dtype : str['float32', 'float64'] / None
            Whether to compute and store filters in single or double precision.

        sign_correction: str / None
            None: no correction
            'abs': `abs(out)`.
                   An explored alternative was `out -= out.min()`, but it's not
                   favored per
                     - shifting the entire output (dc bias), while the negatives
                       don't result from such a shift
                     - the negatives are in minority and vary with "noisy" factors
                       such as boundary effects and signal regularity, making
                       the process itself noisy and sensitive to outliers
        """
        # input checks
        assert sign_correction in (None, 'abs'), sign_correction
        if not isinstance(dtype, (str, type(None))):
            dtype = str(dtype).split('.')[-1]  # e.g. 'torch.float32'
        assert dtype in (None, 'float32', 'float64'), dtype

        self.dtype = dtype
        self.sign_correction = sign_correction
        self.cutoff_mult = cutoff_mult

        # handle `backend`
        if isinstance(backend, str):
            self.backend_name = backend
            import importlib
            backend = importlib.import_module('wavespin.scattering1d.backend.'
                                              + self.backend_name + "_backend",
                                              'backend').backend
        else:
            self.backend_name = backend.__module__.split('.')[-1].rstrip(
                '_backend')
        self.Bk = backend

        # complete module of backend
        if self.backend_name == 'torch':
            import torch
            self.B = torch
        elif self.backend_name == 'tensorflow':
            raise NotImplementedError("currently only 'numpy' and 'torch' "
                                      "backends are supported.")
            # import tensorflow as tf
            # self.B = tf
        else:
            self.B = np

        # handle `gpu`
        if gpu is None:
            gpu = bool(self.backend_name != 'numpy')
        elif gpu and self.backend_name == 'numpy':
            self._err_backend()
        self.gpu = gpu

        # instantiate reusables
        self.filters = {}
        self.unpads = {}
        self.pads = {}

    def __call__(self, x, factor, axis=-1, x_is_fourier=False):
        """Decimate input (anti-alias filter + subsampling).

        Parameters
        ----------
        x : tensor
            n-dim tensor.

        factor : int
            Subsampling factor, must be power of 2.

        axis : int
            Axis along which to decimate. Negative supported.

        x_is_fourier : bool (default False)
            Whether `x` is already in frequency domain.
            If possible, it's more performant to pass in `x` in time domain
            as it's passed to time domain anyway before padding (unless it
            won't require padding, which is possible).

        Returns
        -------
        o : tensor
            `x` decimated along `axis` axis by `factor` factor.
        """
        assert np.log2(factor).is_integer()
        key = (factor, x.shape[axis])
        if key not in self.filters:
            self.make_filter(key)
        return self.decimate(x, key, axis, x_is_fourier)

    def decimate(self, x, key, axis=-1, x_is_fourier=False):
        xf, filtf, factor, ind_start, ind_end = self._handle_input(
            x, key, axis, x_is_fourier)

        # convolve, subsample, unpad
        of = xf * filtf
        of = self.Bk.subsample_fourier(of, factor, axis=axis)
        o = self.Bk.irfft(of, axis=axis)
        o = self.Bk.unpad(o, ind_start, ind_end, axis=axis)

        # sign correction
        if self.sign_correction == 'abs':
            o = self.B.abs(o)

        return o

    def _handle_input(self, x, key, axis, x_is_fourier):
        # from `key` get filter & related info
        factor, N = key
        filtf = self.filters[key]
        ind_start, ind_end = self.unpads[key]
        pad_left, pad_right = self.pads[key]

        # pad `x` if necessary
        if pad_left != 0 or pad_right != 0:
            if x_is_fourier:
                xf = x
                x = self.Bk.ifft(xf, axis=axis)
            xp = self.Bk.pad(x, pad_left, pad_right, pad_mode='zero', axis=axis)
            xf = self.Bk.fft(xp, axis=axis)
        elif not x_is_fourier:
            xf = self.Bk.fft(x, axis=axis)
        else:
            xf = x

        # broadcast filter to input's shape
        broadcast = [None] * x.ndim
        broadcast[axis] = slice(None)
        filtf = filtf[tuple(broadcast)]

        return xf, filtf, factor, ind_start, ind_end

    def make_filter(self, key):
        """Create windowed sinc, centered at n=0 and padded to a power of 2,
        and compute pad and unpad parameters.

        The filters are keyed by `key = (factor, N)`, where `factor` and `N`
        are stored with successive calls to `Decimate`, yielding dynamic
        creation and storage of filters.
        """
        q, N = key
        half_len = 10 * q
        n = int(2 * half_len)
        cutoff = (1. / q) * self.cutoff_mult

        filtf, unpads, pads = self._make_decimate_filter(n + 1, cutoff, q, N)
        self.filters[key] = filtf
        self.unpads[key] = unpads
        self.pads[key] = pads

    # helpers ################################################################
    def _make_decimate_filter(self, numtaps, cutoff, q, N):
        h = self._windowed_sinc(numtaps, cutoff)

        # for FFT conv
        ((pad_left_x, pad_right_x), (pad_left_filt, pad_right_filt)
         ) = self._compute_pad_amount(N, h)
        h = np.pad(h, [pad_left_filt, pad_right_filt])

        # time-center filter about 0 (in DFT sense, n=0)
        h = np.roll(h, -np.argmax(h))
        # take to fourier
        hf = np.fft.fft(h)
        # assert zero phase (imag part zero)
        assert hf.imag.mean() < 1e-15, hf.imag.mean()
        # keep only real part
        hf = hf.real

        # backend, device, dtype
        hf = self._handle_backend_device_dtype(hf)

        # account for additional padding
        ind_start = int(np.ceil(pad_left_x / q))
        ind_end = int(np.ceil((N + pad_left_x) / q))

        return hf, (ind_start, ind_end), (pad_left_x, pad_right_x)

    def _compute_pad_amount(self, N, h):
        # don't concern with whether it decays to zero sooner, assume worst case
        support = len(h)
        # since we zero-pad, can halve (else we'd pad by `support` on each side)
        to_pad = support
        # pow2 for fast FFT conv
        padded_pow2 = int(2**np.ceil(np.log2(N + to_pad)))

        # compute padding for input
        pad_right_x = padded_pow2 - N
        pad_left_x = 0
        # compute padding for filter
        pad_right_filt = padded_pow2 - len(h)
        pad_left_filt = 0

        return (pad_left_x, pad_right_x), (pad_left_filt, pad_right_filt)

    def _windowed_sinc(self, numtaps, cutoff):
        """Sample & normalize windowed sinc, in time domain"""
        win = scipy.signal.get_window("hamming", numtaps, fftbins=False)

        # sample, window, & norm sinc
        alpha = 0.5 * (numtaps - 1)
        m = np.arange(0, numtaps) - alpha
        h = win * cutoff * np.sinc(cutoff * m)
        h /= h.sum()  # L1 norm

        return h

    def _handle_backend_device_dtype(self, hf):
        if self.backend_name == 'numpy':
            if self.dtype == 'float32':
                hf = hf.astype('float32')
            if self.gpu:
                self._err_backend()

        elif self.backend_name == 'torch':
            hf = self.B.from_numpy(hf)
            if self.dtype == 'float32':
                hf = hf.float()
            if self.gpu:
                hf = hf.cuda()

        elif self.backend_name == 'tensorflow':
            raise NotImplementedError

        return hf

    def _err_backend(self):
        raise ValueError("`gpu=True` requires `backend` that's 'torch' "
                         "or 'tensorflow' (got %s)" % str(self.backend_name))

# backend ####################################################################
class ExtendedUnifiedBackend():
    """Extends existing WaveSpin backend with functionality."""
    def __init__(self, x_or_backend_name):
        if isinstance(x_or_backend_name, str):
            backend_name = x_or_backend_name
        else:
            backend_name = _infer_backend(x_or_backend_name, get_name=True)[1]
        self.backend_name = backend_name
        if backend_name == 'torch':
            import torch
            self.B = torch
        elif backend_name == 'tensorflow':
            import tensorflow as tf
            self.B = tf
        else:
            self.B = np
        self.Bk = get_wavespin_backend(backend_name)

    def __getattr__(self, name):
        # fetch from wavespin.backend if possible
        if hasattr(self.Bk, name):
            return getattr(self.Bk, name)
        raise AttributeError(f"'{self.Bk.__name__}' object has no "
                             f"attribute '{name}'")

    def abs(self, x):
        return self.B.abs(x)

    def log(self, x):
        if self.backend_name == 'numpy':
            out = np.log(x)
        elif self.backend_name == 'torch':
            out = self.B.log(x)
        else:
            out = self.B.math.log(x)
        return out

    def sum(self, x, axis=None, keepdims=False):
        if self.backend_name == 'numpy':
            out = np.sum(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.sum(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.reduce_sum(x, axis=axis, keepdims=keepdims)
        return out

    def norm(self, x, ord=2, axis=None, keepdims=True):
        if self.backend_name == 'numpy':
            if ord == 1:
                out = np.sum(np.abs(x), axis=axis, keepdims=keepdims)
            elif ord == 2:
                out = np.linalg.norm(x, ord=None, axis=axis, keepdims=keepdims)
            else:
                out = np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.norm(x, p=ord, dim=axis, keepdim=keepdims)
        else:
            out = self.B.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        return out

    def median(self, x, axis=None, keepdims=None):
        if keepdims is None and self.backend_name != 'tensorflow':
            keepdims = True
        if self.backend_name == 'numpy':
            out = np.median(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.median(x, dim=axis, keepdim=keepdims)
            # torch may return `values` and `indices` if `axis is not None`
            if isinstance(out.values, self.B.Tensor):
                out = out.values
        else:
            if axis is not None or keepdims is not None:
                raise ValueError("`axis` and `keepdims` for `median` in "
                                 "TensorFlow backend are not implemented.")
            v = self.B.reshape(x, [-1])
            m = v.get_shape()[0]//2
            out = self.B.reduce_min(self.B.nn.top_k(v, m, sorted=False).values)
        return out

    def std(self, x, axis=None, keepdims=True):
        if self.backend_name == 'numpy':
            out = np.std(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            out = self.B.std(x, dim=axis, keepdim=keepdims)
        else:
            out = self.B.math.reduce_std(x, axis=axis, keepdims=keepdims)
        return out

    def min(self, x, axis=None, keepdims=False):
        if self.backend_name == 'numpy':
            out = np.min(x, axis=axis, keepdims=keepdims)
        elif self.backend_name == 'torch':
            kw = {'dim': axis} if axis is not None else {}
            if keepdims:
                kw['keepdim'] = True
            out = self.B.min(x, **kw)
        else:
            out = self.B.math.reduce_min(x, axis=axis, keepdims=keepdims)
        return out

    def numpy(self, x):
        if self.backend_name == 'numpy':
            out = x
        else:
            if hasattr(x, 'to') and 'cpu' not in x.device.type:
                x = x.cpu()
            if getattr(x, 'requires_grad', False):
                x = x.detach()
            out = x.numpy()
        return out


def _infer_backend(x, get_name=False):
    while isinstance(x, (dict, list)):
        if isinstance(x, dict):
            if 'coef' in x:
                x = x['coef']
            else:
                x = list(x.values())[0]
        else:
            x = x[0]

    module = type(x).__module__.split('.')[0]

    if module == 'numpy':
        backend = np
    elif module == 'torch':
        import torch
        backend = torch
    elif module == 'tensorflow':
        import tensorflow
        backend = tensorflow
    elif isinstance(x, (int, float)):
        # list of lists, fallback to numpy
        module = 'numpy'
        backend = np
    else:
        raise ValueError("could not infer backend from %s" % type(x))
    return (backend if not get_name else
            (backend, module))


def get_wavespin_backend(backend_name):
    if backend_name == 'numpy':
        from .backend.numpy_backend import NumpyBackend as B
    elif backend_name == 'torch':
        from .backend.torch_backend import TorchBackend as B
    elif backend_name == 'tensorflow':
        from .backend.tensorflow_backend import TensorFlowBackend as B
    return B
