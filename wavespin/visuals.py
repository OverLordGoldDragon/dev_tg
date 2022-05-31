# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Convenience visual methods."""
import os
import numpy as np
import warnings
from scipy.fft import ifft, ifftshift
from copy import deepcopy
from .toolkit import (coeff_energy, coeff_distance, energy, drop_batch_dim_jtfs,
                      _eps, fill_default_args, pack_coeffs_jtfs,
                      get_phi_for_psi_id, make_jtfs_pair)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    warnings.warn("`wavespin.visuals` requires `matplotlib` installed.")


def filterbank_heatmap(scattering, first_order=None, second_order=False,
                       frequential=None, parts='all', psi_id=0, **plot_kw):
    """Visualize scattering filterbank as heatmap of all bandpasses.

    Parameters
    ----------
    scattering : wavespin.scattering1d.Scattering1D,
                wavespin.scattering1d.TimeFrequencyScattering1D
        Scattering object.

    first_order : bool / None
        Whether to show first-order filterbank. Defaults to `True` if
        `scattering` is non-JTFS.

    second_order : bool (default False)
        Whether to show second-order filterbank.

    frequential : bool / tuple[bool]
        Whether to show frequential filterbank (requires JTFS `scattering`).
        Tuple specifies `(up, down)` spins separately. Defaults to `(False, True)`
        if `scattering` is JTFS and `first_order` is `False` or `None` and
        `second_order == False`. If bool, becomes `(False, frequential)`.

    parts : str / tuple / list
        One of: 'abs', 'real', 'imag', 'freq'. First three refer to time-domain,
        'freq' is abs of frequency domain.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    plot_kw : None / dict
        Will pass to `wavespin.visuals.plot(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        scattering = Scattering1D(shape=2048, J=10, Q=16)
        filterbank_heatmap(scattering, first_order=True, second_order=True)
    """
    def to_time_and_viz(psi_fs, name, get_psi):
        # move wavelets to time domain
        psi_fs = [get_psi(p) for p in psi_fs]
        psi_fs = [p for p in psi_fs if p is not None]
        psi1s = [ifftshift(ifft(p)) for p in psi_fs]
        psi1s = np.array([p / np.abs(p).max() for p in psi1s])

        # handle kwargs
        pkw = deepcopy(plot_kw)
        user_kw = list(plot_kw)
        if 'xlabel' not in user_kw:
            pkw['xlabel'] = 'time [samples]'
        if 'ylabel' not in user_kw:
            pkw['ylabel'] = 'wavelet index'
        if 'interpolation' not in user_kw and len(psi1s) < 30:
            pkw['interpolation'] = 'none'

        # plot
        if 'abs' in parts:
            apsi1s = np.abs(psi1s)
            imshow(apsi1s, abs=1, **pkw, title=(f"{name} filterbank | modulus | "
                                                "scaled to same amp."),)
        if 'real' in parts:
            imshow(psi1s.real, **pkw,
                   title=f"{name} filterbank | real part | scaled same amp.")
        if 'imag' in parts:
            imshow(psi1s.imag, **pkw,
                   title=f"{name} filterbank | imag part | scale same amp.")
        if 'freq' in parts:
            if 'xlabel' not in user_kw:
                pkw['xlabel'] = 'frequencies [samples] | dc, +, -'
            psi_fs = np.array(psi_fs)
            imshow(psi_fs, abs=1, **pkw, title=f"{name} filterbank | freq-domain")

    # process `parts`
    supported = ('abs', 'real', 'imag', 'freq')
    if parts == 'all':
        parts = supported
    else:
        for p in parts:
            if p not in supported:
                raise ValueError(("unsupported `parts` '{}'; must be one of: {}"
                                  ).format(p, ', '.join(parts)))

    # process visuals selection
    is_jtfs = bool(hasattr(scattering, 'scf'))
    if first_order is None:
        first_order = not is_jtfs
    if frequential is None:
        # default to frequential only if is jtfs and first_order wasn't requested
        frequential = (False, is_jtfs and not (first_order or second_order))
    elif isinstance(frequential, (bool, int)):
        frequential = (False, bool(frequential))
    if all(not f for f in frequential):
        frequential = False
    if frequential and not is_jtfs:
        raise ValueError("`frequential` requires JTFS `scattering`.")
    if not any(arg for arg in (first_order, second_order, frequential)):
        raise ValueError("Nothing to visualize! (got False for all of "
                         "`first_order`, `second_order`, `frequential`)")

    # visualize
    if first_order or second_order:
        get_psi = lambda p: (p[0] if not hasattr(p[0], 'cpu') else
                             p[0].cpu().numpy())
        if first_order:
            to_time_and_viz(scattering.psi1_f, 'First-order', get_psi)
        if second_order:
            to_time_and_viz(scattering.psi2_f, 'Second-order', get_psi)
    if frequential:
        get_psi = lambda p: ((p if not hasattr(p, 'cpu') else
                              p.cpu().numpy()).squeeze())
        if frequential[0]:
            to_time_and_viz(scattering.psi1_f_fr_up[psi_id], 'Frequential up',
                            get_psi)
        if frequential[1]:
            to_time_and_viz(scattering.psi1_f_fr_dn[psi_id], 'Frequential down',
                            get_psi)


def filterbank_scattering(scattering, zoom=0, filterbank=True, lp_sum=False,
                          lp_phi=True, first_order=True, second_order=False,
                          plot_kw=None):
    """Visualize temporal filterbank in frequency domain, 1D.

    Parameters
    ----------
    scattering: wavespin.scattering1d.Scattering1D,
                wavespin.scattering1d.TimeFrequencyScattering1D
        Scattering object.

    zoom: int
        Will zoom plots by this many octaves.
        If -1, will show full frequency axis (including negatives),
        and both spins.

    filterbank : bool (default True)
        Whether to plot the filterbank.

    lp_sum: bool (default False)
        Whether to plot Littlewood-Paley sum of the filterbank.

    lp_phi : bool (default True)
        Whether to include the lowpass filter in LP-sum visual.
        Has no effect if `lp_sum == False`.

    first_order : bool (default True)
        Whether to plot the first-order filters.

    second_order : bool (default False)
        Whether to plot the second-order filters.

    plot_kw: None / dict
        Will pass to `wavespin.visuals.plot(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        scattering = Scattering1D(shape=2048, J=8, Q=8)
        filterbank_scattering(scattering)
    """
    def _plot_filters(ps, p0, lp, J, title):
        # determine plot parameters ##########################################
        Nmax = len(ps[0][0])
        # x-axis zoom
        if 'xlims' in user_plot_kw_names:
            xlims = plot_kw['xlims']
        else:
            if zoom == -1:
                xlims = (-.02 * Nmax, 1.02 * Nmax)
            else:
                xlims = (-.01 * Nmax/ 2**zoom, .55 * Nmax / 2**zoom)

        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = (title, {'fontsize': 18})
        if 'w' not in plot_kw:
            plot_kw['w'] = .68
        elif 'h' not in plot_kw:
            plot_kw['h'] = .85

        # plot filterbank ####################################################
        _, ax = plt.subplots(1, 1)
        if filterbank:
            # Morlets
            for p in ps:
                j = p['j']
                plot(p[0], color=colors[j], linestyle=linestyles[j])
            # vertical lines (octave bounds)
            plot([], vlines=([Nmax//2**j for j in range(1, J + 2)],
                             dict(color='k', linewidth=1)), ax=ax)
            # lowpass
            if isinstance(p0[0], list):
                p0 = p0[0]
            plot([], vlines=(Nmax//2, dict(color='k', linewidth=1)))
            plot(p0[0], color='k', **plot_kw, ax=ax)

        N = len(p[0])
        _filterbank_style_axes(ax, N, xlims)
        plt.show()

        # plot LP sum ########################################################
        if lp_sum:
            if 'title' not in user_plot_kw_names:
                plot_kw['title'] = ("Littlewood-Paley sum", {'fontsize': 18})
            fig, ax = plt.subplots(1, 1)
            plot(lp, **plot_kw, show=0, ax=ax,
                 hlines=(2, dict(color='tab:red', linestyle='--')),
                 vlines=(Nmax//2, dict(color='k', linewidth=1)))
            _filterbank_style_axes(ax, N, xlims, ymax=lp.max()*1.03)
            plt.show()

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)

    # define colors & linestyles
    colors = [f"tab:{c}" for c in ("blue orange green red purple brown pink "
                                   "gray olive cyan".split())]
    linestyles = ('-', '--', '-.')
    nc = len(colors)
    nls = len(linestyles)

    # support J up to nc * nls
    colors = colors * nls
    linestyles = [ls_set for ls in "- -- -.".split() for ls_set in [ls]*nc]

    # shorthand references
    p0 = scattering.phi_f
    p1 = scattering.psi1_f
    if second_order:
        p2 = scattering.psi2_f

    # compute LP sum
    lp1, lp2 = 0, 0
    if lp_sum:
        # it's list for JTFS
        p0_longest = p0[0] if not isinstance(p0[0], list) else p0[0][0]
        for p in p1:
            lp1 += np.abs(p[0])**2
        if lp_phi:
            lp1 += np.abs(p0_longest)**2

        if second_order:
            for p in p2:
                lp2 += np.abs(p[0])**2
            if lp_phi:
                lp2 += np.abs(p0_longest)**2

    # title & plot
    (Q0, Q1), (J0, J1) = scattering.Q, scattering.J
    if first_order:
        title = "First-order filterbank | J, Q1, T = {}, {}, {}".format(
            J0, Q0, scattering.T)
        _plot_filters(p1, p0, lp1, J0, title=title)

    if second_order:
        title = "Second-order filterbank | J, Q2, T = {}, {}, {}".format(
            J1, Q1, scattering.T)
        _plot_filters(p2, p0, lp2, J1, title=title)


def filterbank_jtfs_1d(jtfs, zoom=0, psi_id=0, filterbank=True, lp_sum=False,
                       lp_phi=True, center_dc=None, both_spins=True,
                       plot_kw=None):
    """Visualize JTFS frequential filterbank in frequency domain, 1D.

    Parameters
    ----------
    jtfs : wavespin.scattering1d.TimeFrequencyScattering1D
        Scattering object.

    zoom : int
        Will zoom plots by this many octaves.
        If -1, will show full frequency axis (including negatives),
        and both spins.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    filterbank : bool (default True)
        Whether to plot the filterbank.

    lp_sum : bool (default False)
        Whether to plot Littlewood-Paley sum of the filterbank.

    lp_phi : bool (default True)
        Whether to include the lowpass filter in LP-sum visual.
        Has no effect if `lp_sum == False`.

    center_dc : bool / None
        If True, will `ifftshift` to center the dc bin.
        Defaults to `True` if `zoom == -1`.

    both_spins : bool (default True)
        Whether to plot both up and down, or only up.

    plot_kw : None / dict
        Will pass to `plot(**plot_kw)`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        jtfs = TimeFrequencyScattering1D(shape=2048, J=8, Q=8)
        filterbank_jtfs_1d(jtfs)
    """
    def _plot_filters(ps, p0, lp, fig0, ax0, fig1, ax1, title_base, up):
        # determine plot parameters ##########################################
        # vertical lines (octave bounds)
        Nmax = len(ps[psi_id][0])
        j_dists = np.array([Nmax//2**j for j in range(1, jtfs.J_fr + 1)])
        if up and not (up and zoom == -1 and center_dc and not both_spins):
            vlines = (Nmax//2 - j_dists if center_dc else
                      j_dists)
        else:
            vlines = (Nmax//2 + j_dists if center_dc else
                      Nmax - j_dists)
        # x-axis zoom
        if 'xlims' in user_plot_kw_names:
            xlims = plot_kw['xlims']
        else:
            if zoom == -1:
                xlims = (-.02 * Nmax, 1.02 * Nmax)
            else:
                xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)
                if not up:
                    xlims = (Nmax - xlims[1], Nmax - .2 * xlims[0])

        # title
        if zoom != -1:
            title = title_base % "up" if up else title_base % "down"
        else:
            title = title_base

        # handle `plot_kw`
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = title

        # plot filterbank ####################################################
        if filterbank:
            # bandpasses
            for n1_fr, p in enumerate(ps[psi_id]):
                j = ps['j'][psi_id][n1_fr]
                pplot = p.squeeze()
                if center_dc:
                    pplot = ifftshift(pplot)
                plot(pplot, color=colors[j], linestyle=linestyles[j], ax=ax0)
            # lowpass
            p0plot = get_phi_for_psi_id(jtfs, psi_id)
            if center_dc:
                p0plot = ifftshift(p0plot)
            plot(p0plot, color='k', **plot_kw, ax=ax0, fig=fig0,
                 vlines=(vlines, dict(color='k', linewidth=1)))

        N = len(p)
        _filterbank_style_axes(ax0, N, xlims, zoom=zoom, is_jtfs=True)

        # plot LP sum ########################################################
        plot_kw_lp = {}
        if 'title' not in user_plot_kw_names:
            plot_kw['title'] = ("Littlewood-Paley sum" +
                                " (no phi)" * int(not lp_phi))
        if 'ylims' not in user_plot_kw_names:
            plot_kw_lp['ylims'] = (0, None)

        if lp_sum and not (zoom == -1 and not up):
            lpplot = ifftshift(lp) if center_dc else lp
            plot(lpplot, **plot_kw, **plot_kw_lp, ax=ax1, fig=fig1,
                 hlines=(1, dict(color='tab:red', linestyle='--')),
                 vlines=(Nmax//2, dict(color='k', linewidth=1)))

            _filterbank_style_axes(ax1, N, xlims, ymax=lp.max()*1.03,
                                   zoom=zoom, is_jtfs=True)

    # handle `plot_kw`
    if plot_kw is not None:
        # don't alter external dict
        plot_kw = deepcopy(plot_kw)
    else:
        plot_kw = {}
    user_plot_kw_names = list(plot_kw)
    # handle `center_dc`
    if center_dc is None:
        center_dc = bool(zoom == -1)

    # define colors & linestyles
    colors = [f"tab:{c}" for c in ("blue orange green red purple brown pink "
                                   "gray olive cyan".split())]
    linestyles = ('-', '--', '-.')
    nc = len(colors)
    nls = len(linestyles)

    # support J up to nc * nls
    colors = colors * nls
    linestyles = [ls_set for ls in "- -- -.".split() for ls_set in [ls]*nc]

    # shorthand references
    p0 = jtfs.scf.phi_f_fr
    pup = jtfs.psi1_f_fr_up
    pdn = jtfs.psi1_f_fr_dn

    # compute LP sum
    lp = 0
    if lp_sum:
        psi_fs = (pup, pdn) if both_spins else (pup,)
        for psi1_f in psi_fs:
            for p in psi1_f[psi_id]:
                lp += np.abs(p)**2
        if lp_phi:
            p0 = get_phi_for_psi_id(jtfs, psi_id)
            lp += np.abs(p0)**2

    # title
    params = (jtfs.J_fr, jtfs.Q_fr, jtfs.F)
    if zoom != -1:
        title_base = ("Frequential filterbank | spin %s | J_fr, Q_fr, F = "
                      "{}, {}, {}").format(*params)
    else:
        title_base = ("Frequential filterbank | J_fr, Q_fr, F = "
                      "{}, {}, {}").format(*params)

    # plot ###################################################################
    def make_figs():
        return ([plt.subplots(1, 1) for _ in range(2)] if lp_sum else
                (plt.subplots(1, 1), (None, None)))

    (fig0, ax0), (fig1, ax1) = make_figs()
    _plot_filters(pup, p0, lp, fig0, ax0, fig1, ax1, title_base=title_base,
                  up=True)
    if zoom != -1:
        plt.show()
        if both_spins:
            (fig0, ax0), (fig1, ax1) = make_figs()

    if both_spins:
        _plot_filters(pdn, p0, lp, fig0, ax0, fig1, ax1, title_base=title_base,
                      up=False)
    plt.show()


def gif_jtfs_2d(Scx, meta, savedir='', base_name='jtfs2d', images_ext='.png',
                overwrite=False, save_images=None, show=None, cmap='turbo',
                norms=None, skip_spins=False, skip_unspinned=False, sample_idx=0,
                inf_token=-1, verbose=False, gif_kw=None):
    """Slice heatmaps of JTFS outputs.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    savedir : str
        Path of directory to save GIF/images to. Defaults to current
        working directory.

    base_name : str
        Will save gif with this name, and images with same name enumerated.

    images_ext : str
        Generates images with this format. '.png' (default) is lossless but takes
        more space, '.jpg' is compressed.

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    save_images : bool (default False)
        Whether to save images. Images are always saved if `savepath` is not None,
        but deleted after if `save_images=False`.
        If `True` and `savepath` is None, will save images to current working
        directory (but not gif).

    show : None / bool
        Whether to display images to console. If `savepath` is None, defaults
        to True.

    cmap : str
        Passed to `imshow`.

    norms: None / tuple
        Plot color norms for 1) `psi_t * psi_f`, 2) `psi_t * phi_f`, and
        3) `phi_t * psi_f` pairs, respectively.
        Tuple of three (upper limits only, lower assumed 0).
        If None, will norm to `.5 * max(coeffs)`, where coeffs = all joint
        coeffs except `phi_t * phi_f`.

    skip_spins: bool (default False)
        Whether to skip `psi_t * psi_f` pairs.

    skip_unspinned: bool (default False)
        Whether to skip `phi_t * phi_f`, `phi_t * psi_f`, `psi_t * phi_f`
        pairs.

    sample_idx : int (default 0)
        Index of sample in batched input to visualize.

    inf_token: int / np.nan
        Placeholder used in `meta` to denote infinity.

    verbose : bool (default False)
        Whether to print to console the location of save file upon success.

    gif_kw : dict / None
        Passed as kwargs to `wavespin.visuals.make_gif`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(J, N, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        meta = jtfs.meta()

        gif_jtfs_2d(Scx, meta)
    """
    def _title(meta_idx, pair, spin):
        txt = r"$|\Psi_{%s, %s, %s} \star \mathrm{U1}|$"
        values = ns[pair][meta_idx[0]]
        assert values.ndim == 1, values
        mu, l, _ = [int(n) if (float(n).is_integer() and n >= 0) else '\infty'
                    for n in values]
        return (txt % (mu, l, spin), {'fontsize': 20})

    def _n_n1s(pair):
        n2, n1_fr, _ = ns[pair][meta_idx[0]]
        return np.sum(np.all(ns[pair][:, :2] == np.array([n2, n1_fr]), axis=1))

    def _get_coef(i, pair, meta_idx):
        n_n1s = _n_n1s(pair)
        start, end = meta_idx[0], meta_idx[0] + n_n1s
        if out_list:
            coef = Scx[pair][i]['coef']
        elif out_3D:
            coef = Scx[pair][i]
        else:
            coef = Scx[pair][start:end]
        assert len(coef) == n_n1s
        return coef

    def _save_image():
        path = os.path.join(savedir, f'{base_name}{img_idx[0]}{images_ext}')
        if os.path.isfile(path) and overwrite:
            os.unlink(path)
        if not os.path.isfile(path):
            plt.savefig(path, bbox_inches='tight')
        img_paths.append(path)
        img_idx[0] += 1

    def _viz_spins(Scx, i, norm):
        kup = 'psi_t * psi_f_up'
        kdn = 'psi_t * psi_f_dn'
        sup = _get_coef(i, kup, meta_idx)
        sdn = _get_coef(i, kdn, meta_idx)

        _, axes = plt.subplots(1, 2, figsize=(14, 7))
        kw = dict(abs=1, ticks=0, show=0, norm=norm)

        imshow(sup, ax=axes[0], **kw, title=_title(meta_idx, kup, '+1'))
        imshow(sdn, ax=axes[1], **kw, title=_title(meta_idx, kdn, '-1'))
        plt.subplots_adjust(wspace=0.01)
        if save_images or do_gif:
            _save_image()
        if show:
            plt.show()
        plt.close()

        meta_idx[0] += len(sup)

    def _viz_simple(Scx, pair, i, norm):
        coef = _get_coef(i, pair, meta_idx)

        _kw = dict(abs=1, ticks=0, show=0, norm=norm, w=14/12, h=7/12,
                   title=_title(meta_idx, pair, '0'))
        if do_gif:
            # make spacing consistent with up & down
            _, axes = plt.subplots(1, 2, figsize=(14, 7))
            imshow(coef, ax=axes[0], **_kw)
            plt.subplots_adjust(wspace=0.01)
            axes[1].set_frame_on(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        else:
            # optimize spacing for single image
            imshow(coef, **_kw)
        if save_images or do_gif:
            _save_image()
        if show:
            plt.show()
        plt.close()

        meta_idx[0] += len(coef)

    # handle args & check if already exists (if so, delete if `overwrite`)
    savedir, savepath, images_ext, save_images, show, do_gif = _handle_gif_args(
        savedir, base_name, images_ext, save_images, overwrite, show=False)

    # set params
    out_3D = bool(meta['n']['psi_t * phi_f'].ndim == 3)
    out_list = isinstance(Scx['S0'], list)
    ns = {pair: meta['n'][pair].reshape(-1, 3) for pair in meta['n']}

    Scx = drop_batch_dim_jtfs(Scx, sample_idx)

    if isinstance(norms, (list, tuple)):
        norms = [(0, n) for n in norms]
    elif isinstance(norms, float):
        norms = [(0, norms) for _ in range(3)]
    else:
        # set to .5 times the max of any joint coefficient (except phi_t * phi_f)
        mx = np.max([(c['coef'] if out_list else c).max()
                     for pair in Scx for c in Scx[pair]
                     if pair not in ('S0', 'S1', 'phi_t * phi_f')])
        norms = [(0, .5 * mx)] * 5

    # spinned pairs ##########################################################
    img_paths = []
    img_idx = [0]
    meta_idx = [0]
    if not skip_spins:
        i = 0
        while True:
            _viz_spins(Scx, i, norms[0])
            i += 1
            if meta_idx[0] > len(ns['psi_t * psi_f_up']) - 1:
                break

    # unspinned pairs ########################################################
    if not skip_unspinned:
        pairs = ('psi_t * phi_f', 'phi_t * psi_f', 'phi_t * phi_f')
        for j, pair in enumerate(pairs):
            meta_idx = [0]
            i = 0
            while True:
                _viz_simple(Scx, pair, i, norms[1 + j])
                i += 1
                if meta_idx[0] > len(ns[pair]) - 1:
                    break

    # make gif & cleanup #####################################################
    try:
        if do_gif:
            if gif_kw is None:
                gif_kw = {}
            make_gif(loaddir=savedir, savepath=savepath, ext=images_ext,
                     overwrite=overwrite, delimiter=base_name, verbose=verbose,
                     **gif_kw)
    finally:
        if not save_images:
            # guarantee cleanup
            for path in img_paths:
                if os.path.isfile(path):
                    os.unlink(path)


def gif_jtfs_3d(Scx, jtfs=None, preset='spinned', savedir='',
                base_name='jtfs3d', images_ext='.png', cmap='turbo', cmap_norm=.5,
                axes_labels=('xi2', 'xi1_fr', 'xi1'), overwrite=False,
                save_images=False, width=800, height=800, surface_count=30,
                opacity=.2, zoom=1, angles=None, verbose=True, gif_kw=None):
    """Generate and save GIF of 3D JTFS slices.

    Parameters
    ----------
    Scx : dict / tensor, 4D
        Output of `jtfs(x)` with `out_type='dict:array'` or `'dict:list'`,
        or output of `wavespin.toolkit.pack_coeffs_jtfs`.

    jtfs : TimeFrequencyScattering1D
        Required if `preset` is not `None`.

    preset : str['spinned', 'all'] / None
        If `Scx = jtfs(x)`, then
            - 'spinned': show only `psi_t * psi_f_up` and `psi_t * psi_f_dn` pairs
            - 'all': show all pairs
        `None` is for when `Scx` is already packed via `pack_coeffs_jtfs`.

    savedir, base_name, images_ext, overwrite :
        See `help(wavespin.visuals.gif_jtfs)`.

    cmap : str
        Colormap to use.

    cmap_norm : float
        Colormap norm to use, as fraction of maximum value of `packed`
        (i.e. `norm=(0, cmap_norm * packed.max())`).

    axes_labels : tuple[str]
        Names of last three dimensions of `packed`. E.g. `structure==2`
        (in `pack_coeffs_jtfs`) will output `(n2, n1_fr, n1, t)`, so
        `('xi2', 'xi1_fr', 'xi1')` (default).

    width : int
        2D width of each image (GIF frame).

    height : int
        2D height of each image (GIF frame).

    surface_count : int
        Greater improves 3D detail of each frame, but takes longer to render.

    opacity : float
        Lesser makes 3D surfaces more transparent, exposing more detail.

    zoom : float (default=1) / None
        Zoom factor on each 3D frame. If None, won't modify `angles`.
        If not None, will first divide by L2 norm of `angles`, then by `zoom`.

    angles : None / np.ndarray / list/tuple[np.ndarray] / str['rotate']
        Controls display angle of the GIF.

          - None: default angle that faces the line extending from min to max
            of `xi1`, `xi2`, and `xi1_fr` (assuming default `axes_labels`).
          - Single 1D array: will reuse for each frame.
          - 'rotate': will use a preset that rotates the display about the
            default angle.

        Resulting array is passed to `go.Figure.update_layout()` as
        `'layout_kw': {'scene_camera': 'center': dict(x=e[0], y=e[1], z=e[2])}`,
        where `e = angles[0]` up to `e = angles[len(packed) - 1]`.

    verbose : bool (default True)
        Whether to print GIF generation progress.

    gif_kw : dict / None
        Passed as kwargs to `wavespin.visuals.make_gif`.

    Example
    -------
    Also see `examples/visuals_tour.py`.
    ::

        N, J, Q = 2049, 7, 16
        x = toolkit.echirp(N)

        jtfs = TimeFrequencyScattering1D(J, N, Q, J_fr=4, Q_fr=2,
                                         out_type='dict:list')
        Scx = jtfs(x)
        gif_jtfs_3d(Scx, jtfs, savedir='', preset='spinned')
    """
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        print("\n`plotly.graph_objs` is needed for `gif_jtfs_3d`.")
        raise e

    # handle args & check if already exists (if so, delete if `overwrite`)
    savedir, savepath_gif, images_ext, save_images, *_ = _handle_gif_args(
        savedir, base_name, images_ext, save_images, overwrite, show=False)
    if preset not in ('spinned', 'all', None):
        raise ValueError("`preset` must be 'spinned', 'all', or None (got %s)" % (
            preset))

    # handle input tensor
    if not isinstance(Scx, (dict, np.ndarray)):
        raise ValueError("`Scx` must be dict or numpy array (need `out_type` "
                         "'dict:array' or 'dict:list'). Got %s" % type(Scx))
    elif isinstance(Scx, dict):
        ckw = dict(Scx=Scx, meta=jtfs.meta(), reverse_n1=False,
                   out_3D=jtfs.out_3D,
                   sampling_psi_fr=jtfs.sampling_psi_fr)
        if preset == 'spinned':
            _packed = pack_coeffs_jtfs(structure=2, separate_lowpass=True, **ckw)
            _packed = _packed[0]  # spinned only
        elif preset == 'all':
            _packed = pack_coeffs_jtfs(structure=2, separate_lowpass=False, **ckw)
        else:
            raise ValueError("dict `Scx` requires string `preset` (got %s)" % (
                preset))
        packed = _packed.transpose(-1, 0, 1, 2)  # time first
    elif isinstance(Scx, np.ndarray):
        packed = Scx

    # handle labels
    supported = ('t', 'xi2', 'xi1_fr', 'xi1')
    for label in axes_labels:
        if label not in supported:
            raise ValueError(("unsupported `axes_labels` element: {} -- must "
                              "be one of: {}").format(
                                  label, ', '.join(supported)))
    frame_label = [label for label in supported if label not in axes_labels][0]

    # 3D meshgrid
    def slc(i, g):
        label = axes_labels[i]
        start = {'xi1': .5, 'xi2': .5, 't': 0, 'xi1_fr': .5}[label]
        end   = {'xi1': 0., 'xi2': 0., 't': 1, 'xi1_fr': -.5}[label]
        return slice(start, end, g*1j)

    a, b, c = packed.shape[1:]
    X, Y, Z = np.mgrid[slc(0, a), slc(1, b), slc(2, c)]

    # handle `angles`; camera focus
    if angles is None:
        eye = np.array([2.5, .3, 2])
        eye /= np.linalg.norm(eye)
        eyes = [eye] * len(packed)
    elif (isinstance(angles, (list, tuple)) or
          (isinstance(angles, np.ndarray) and angles.ndim == 2)):
        eyes = angles
    elif isinstance(angles, str):
        assert angles == 'rotate', angles
        n_pts = len(packed)

        def gauss(n_pts, mn, mx, width=20):
            t = np.linspace(0, 1, n_pts)
            g = np.exp(-(t - .5)**2 * width)
            g *= (mx - mn)
            g += mn
            return g

        x = np.logspace(np.log10(2.5), np.log10(8.5), n_pts, endpoint=1)
        y = np.logspace(np.log10(0.3), np.log10(6.3), n_pts, endpoint=1)
        z = np.logspace(np.log10(2.0), np.log10(2.0), n_pts, endpoint=1)

        x, y, z = [gauss(n_pts, mn, mx) for (mn, mx)
                   in [(2.5, 8.5), (0.3, 6.3), (2, 2)]]

        eyes = np.vstack([x, y, z]).T
    else:
        eyes = [angles] * len(packed)
    assert len(eyes) == len(packed), (len(eyes), len(packed))

    # camera zoom
    if zoom is not None:
        for i in range(len(eyes)):
            eyes[i] /= (np.linalg.norm(eyes[i]) * .5 * zoom)
    # colormap norm
    mx = cmap_norm * packed.max()

    # gif configs
    volume_kw = dict(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        colorscale=cmap,
        showscale=False,
        cmin=0,
        cmax=mx,
    )
    layout_kw = dict(
        margin_pad=0,
        margin_l=0,
        margin_r=0,
        margin_t=0,
        title_pad_t=0,
        title_pad_b=0,
        margin_autoexpand=False,
        scene_aspectmode='cube',
        width=width,
        height=height,
        scene=dict(
            xaxis_title=axes_labels[0],
            yaxis_title=axes_labels[1],
            zaxis_title=axes_labels[2],
        ),
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        ),
    )

    # generate gif frames ####################################################
    img_paths = []
    for k, vol4 in enumerate(packed):
        fig = go.Figure(go.Volume(value=vol4.flatten(), **volume_kw))

        eye = dict(x=eyes[k][0], y=eyes[k][1], z=eyes[k][2])
        layout_kw['scene_camera']['eye'] = eye
        fig.update_layout(
            **layout_kw,
            title={'text': f"{frame_label}={k}",
                   'x': .5, 'y': .09,
                   'xanchor': 'center', 'yanchor': 'top'}
        )

        savepath = os.path.join(savedir, f'{base_name}{k}{images_ext}')
        if os.path.isfile(savepath) and overwrite:
            os.unlink(savepath)
        fig.write_image(savepath)
        img_paths.append(savepath)
        if verbose:
            print("{}/{} frames done".format(k + 1, len(packed)), flush=True)

    # make gif ###############################################################
    try:
        if gif_kw is None:
            gif_kw = {}
        make_gif(loaddir=savedir, savepath=savepath_gif, ext=images_ext,
                 delimiter=base_name, overwrite=overwrite, verbose=verbose,
                 **gif_kw)
    finally:
        if not save_images:
            # guarantee cleanup
            for path in img_paths:
                if os.path.isfile(path):
                    os.unlink(path)


def energy_profile_jtfs(Scx, meta, x=None, pairs=None, kind='l2', flatten=False,
                        plots=True, **plot_kw):
    """Plot & print relevant energy information across coefficient pairs.
    Works for all `'dict' in out_type` and `out_exclude`.
    Also see `help(wavespin.toolkit.coeff_energy)`.

    Parameters
    ----------
    Scx: dict[list] / dict[np.ndarray]
        `jtfs(x)`.

    meta: dict[dict[np.ndarray]]
        `jtfs.meta()`.

    x : tensor, optional
        Original input to print `E_out / E_in`.

    pairs: None / list/tuple[str]
        Computes energies for these pairs in provided order. None will compute
        for all in default order:
            ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_dn')

    kind : str['l1', 'l2']
        - L1: `sum(abs(x))`
        - L2: `sum(abs(x)**2)` -- actually L2^2

    flatten : bool (default False)
        If True, will return quantities on per-`n1` (per frequency row) basis,
        rather than per-`(n2, n1_fr)` (per joint slice).

    plots : bool (default True)
        Whether to visualize the energies and print statistics
        (will print E_out / E_in if `x` is passed regardless).

    plot_kw : kwargs
        Will pass to `wavespin.visuals.plot()`.

    Returns
    -------
    energies: list[float]
        List of coefficient energies.

    pair_energies: dict[str: float]
        Keys are pairs, values are sums of all pair's coefficient energies.
    """
    if not isinstance(Scx, dict):
        raise NotImplementedError("input must be dict. Set out_type='dict:array' "
                                  "or 'dict:list'.")
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target="L1 norm" if kind == 'l1' else "Energy")
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_energy(
        Scx, meta, pair, aggregate=False, kind=kind)

    # compute, plot, print
    energies, pair_energies = _iterate_coeff_pairs(
        Scx, meta, fn, pairs, plots=plots, flatten=flatten,
        titles=titles, **plot_kw)

    # E_out / E_in
    if x is not None:
        e_total = np.sum(energies)
        print("E_out / E_in = %.3f" % (e_total / energy(x)))
    return energies, pair_energies


def coeff_distance_jtfs(Scx0, Scx1, meta0, meta1=None, pairs=None, kind='l2',
                        flatten=False, plots=True, **plot_kw):
    """Computes relative distance between JTFS coefficients.

    Parameters
    ----------
    Scx0, Scx1: dict[list] / dict[np.ndarray]
        `jtfs(x0)`, `jtfs(x1)` (or `jtfs0` vs `jtfs1`, but see `meta1`).

    meta0: dict[dict[np.ndarray]]
        `jtfs.meta()` for `Scx0`.

    meta1: dict[dict[np.ndarray]] / None
        `jtfs.meta()` for `Scx1`. Configuration cannot differ in any way
        that alters coefficient shapes.

    pairs: None / list/tuple[str]
        Computes distances for these pairs in provided order. None will compute
        for all in default order:
            ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f',
             'psi_t * psi_f_up', 'psi_t * psi_f_dn')

    kind : str['l1', 'l2']
        - L1: `sum(abs(x))`
        - L2: `sum(abs(x)**2)` -- actually L2^2, i.e. energy

    flatten : bool (default False)
        If True, will return quantities on per-`n1` (per frequency row) basis,
        rather than per-`(n2, n1_fr)` (per joint slice).

    plots : bool (default True)
        Whether to visualize the distances.

    plot_kw : kwargs
        Will pass to `wavespin.visuals.plot()`.

    Returns
    -------
    distances : list[float]
        List of coefficient distances.

    pair_distances : dict[str: float]
        Keys are pairs, values are sums of all pair's coefficient distances.
    """
    if not all(isinstance(Scx, dict) for Scx in (Scx0, Scx1)):
        raise NotImplementedError("inputs must be dict. Set "
                                  "out_type='dict:array' or 'dict:list'.")
    if meta1 is None:
        meta1 = meta0

    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)
    # make `titles`
    titles = _make_titles_jtfs(compute_pairs,
                               target=("Absolute reldist" if kind == 'l1'
                                       else "Euclidean reldist"))
    # make `fn`
    fn = lambda Scx, meta, pair: coeff_distance(*Scx, *meta, pair, kind=kind)

    # compute, plot, print
    distances, pair_distances = _iterate_coeff_pairs(
        (Scx0, Scx1), (meta0, meta1), fn, pairs, plots=plots,
        titles=titles, **plot_kw)

    return distances, pair_distances


def _iterate_coeff_pairs(Scx, meta, fn, pairs=None, flatten=False, plots=True,
                         titles=None, **plot_kw):
    # in case multiple meta passed
    meta0 = meta[0] if isinstance(meta, tuple) else meta
    # enforce pair order
    compute_pairs = _get_compute_pairs(pairs, meta0)

    # extract energy info
    energies = []
    pair_energies = {}
    idxs = [0]
    for pair in compute_pairs:
        if pair not in meta0['n']:
            continue
        E_flat, E_slices = fn(Scx, meta, pair)
        data = E_flat if flatten else E_slices
        # flip to order freqs low-to-high
        pair_energies[pair] = data[::-1]
        energies.extend(data[::-1])
        # don't repeat 0
        idxs.append(len(energies) - 1 if len(energies) != 1 else 1)

    # format & plot ##########################################################
    energies = np.array(energies)
    ticks = np.arange(len(energies))
    vlines = (idxs, {'color': 'tab:red', 'linewidth': 1})

    if titles is None:
        titles = ['', '']
    if plots:
        scat(ticks[idxs], energies[idxs], s=20)
        plot_kw['ylims'] = plot_kw.get('ylims', (0, None))
        plot(energies, vlines=vlines, title=titles[0], show=1, **plot_kw)

    # cumulative sum
    energies_cs = np.cumsum(energies)

    if plots:
        scat(ticks[idxs], energies_cs[idxs], s=20)
        plot(energies_cs, vlines=vlines, title=titles[1], show=1, **plot_kw)

    # print report ###########################################################
    def sig_figs(x, n_sig=3):
        s = str(x)
        nondecimals = len(s.split('.')[0]) - int(s[0] == '0')
        decimals = max(n_sig - nondecimals, 0)
        return s.lstrip('0')[:decimals + nondecimals + 1].rstrip('.')

    e_total = np.sum(energies)
    pair_energies_sum = {pair: np.sum(pair_energies[pair])
                         for pair in pair_energies}
    nums = [sig_figs(e, n_sig=3) for e in pair_energies_sum.values()]
    longest_num = max(map(len, nums))

    if plots:
        i = 0
        for pair in compute_pairs:
            E_pair = pair_energies_sum[pair]
            eps = _eps(e_total)
            e_perc = sig_figs(E_pair / (e_total + eps) * 100, n_sig=3)
            print("{} ({}%) -- {}".format(
                nums[i].ljust(longest_num), str(e_perc).rjust(4), pair))
            i += 1
    return energies, pair_energies


def compare_distances_jtfs(pair_distances, pair_distances_ref, plots=True,
                           verbose=True, title=None):
    """Compares distances as per-coefficient ratios, as a generally more viable
    alternative to the global L2 measure.

    Parameters
    ----------
    pair_distances : dict[tensor]
        (second) Output of `wavespin.visuals.coeff_distance_jtfs`, or alike.
        The numerator of the ratio.

    pair_distances_ref : dict[tensor]
        (second) Output of `wavespin.visuals.coeff_distance_jtfs`, or alike.
        The denominator of the ratio.

    plots : bool (default True)
        Whether to plot the ratios.

    verbose : bool (default True)
        Whether to print a summary of ratio statistics.

    title : str / None
        Will append to pre-made title.

    Returns
    -------
    ratios : dict[tensor]
        Distance ratios, keyed by pairs.

    stats : dict[tensor]
        Mean, minimum, and maximum of ratios along pairs, respectively,
        keyed by pairs.
    """
    # don't modify external
    pd0, pd1 = deepcopy(pair_distances), deepcopy(pair_distances_ref)

    ratios, stats = {}, {}
    for pair in pd0:
        p0, p1 = np.asarray(pd0[pair]), np.asarray(pd1[pair])
        # threshold out small points
        idxs = np.where((p0 < .001*p0.max()).astype(int) +
                        (p1 < .001*p1.max()).astype(int))[0]
        p0[idxs], p1[idxs] = 1, 1
        R = p0 / p1
        ratios[pair] = R
        stats[pair] = dict(mean=R.mean(), min=R.min(), max=R.max())

    if plots:
        if title is None:
            title = ''
        _title = _make_titles_jtfs(list(ratios), f"Distance ratios | {title}")[0]
        vidxs = np.cumsum([len(r) for r in ratios.values()])
        ratios_flat = np.array([r for rs in ratios.values() for r in rs])
        plot(ratios_flat, ylims=(0, None), title=_title,
             hlines=(1,     dict(color='tab:red', linestyle='--')),
             vlines=(vidxs, dict(color='k', linewidth=1)))
        scat(idxs, ratios_flat[idxs], color='tab:red', show=1)
    if verbose:
        print("Ratios (Sx/Sx_ref):")
        print("mean  min   max   | pair")
        for pair in ratios:
            print("{:<5.2f} {:<5.2f} {:<5.2f} | {}".format(
                *list(stats[pair].values()), pair))
    return ratios, stats


def scalogram(x, sc, fs=None, show_x=False, w=1., h=1., plot_cfg=None):
    """Compute and plot scalogram. Optionally plots `x`, separately.

    Parameters
    ----------
    x : np.ndarray
        Input, 1D.

    sc : `Scattering1D` instance
        Must be from NumPy backend, and have `average=False`. Will internally
        set `sc.oversampling=999` and `sc.max_order=1`.

    fs : None / int
        Sampling rate. If provided, will display physical units (Hz), else
        discrete (cycles/sample).

    show_x : bool (default False)
        Whether to plot `x` in time domain.

    w, h : float, float
        Scale width and height, separately.

    plot_cfg : None / dict
        Configures plotting. Will fill for missing values from defaults
        (see `plot_cfg_defaults` in source code). Supported key-values:

            'label_kw_xy' : dict
                Passed to all `ax.set_xlabel` and `ax.set_ylabel`.

            'title_kw' : dict
                Passed to all `ax.set_title.

            'tick_params_kw' : dict
                Passed to all `ax.tick_params`.

            'title_x' : str
                Title to show for plot of `x`, if applicable.

            'title_scalogram' : str
                Title to show for plot of scalogram.
    """
    # sanity checks
    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 1, x.shape
    assert not sc.average
    assert 'numpy' in sc.__module__, sc.__module__

    # `plot_cfg`, defaults
    plot_cfg_defaults = {
        'label_kw_xy': dict(weight='bold', fontsize=18),
        'title_kw':    dict(weight='bold', fontsize=20),
        'tick_params_kw': dict(labelsize=16),
        'title_x': 'x',
        'title_scalogram': 'Scalogram',
    }
    C = fill_default_args(plot_cfg, plot_cfg_defaults, copy_original=True)

    # extract basic params, configure `sc`
    N = len(x)
    sc.oversampling = 999
    sc.max_order = 1

    # compute scalogram
    Scx = sc(x)
    meta = sc.meta()
    S1 = np.array([c['coef'].squeeze() for c in Scx])[meta['order'] == 1]

    # ticks & units
    if fs is not None:
        f_units = "[Hz]"
        t_units = "[sec]"
    else:
        f_units = "[cycles/sample]"
        t_units = "[samples]"

    yticks = np.array([p['xi'] for p in sc.psi1_f])
    if fs is not None:
        t = np.linspace(0, N/fs, N, endpoint=False)
        yticks *= fs
    else:
        t = np.arange(N)

    # axis labels
    xlabel  = (f"Time {t_units}",      C['label_kw_xy'])
    ylabel0 = ("Amplitude",            C['label_kw_xy'])
    ylabel1 = (f"Frequency {f_units}", C['label_kw_xy'])
    # titles
    title0 = (C['title_x'],         C['title_kw'])
    title1 = (C['title_scalogram'], C['title_kw'])
    # format yticks (limit # of shown decimal digits, and round the rest)
    yticks = _format_ticks(yticks)

    # plot ###################################################################
    if show_x:
        fig, axes = plt.subplots(1, 2, figsize=(16*w, 6*h))
        ax0, ax1 = axes

        plot(t, x, xlabel=xlabel, ylabel=ylabel0, fig=fig, ax=ax0, title=title0,
             show=0)
        ax0.tick_params(**C['tick_params_kw'])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8*w, 6*h))

    imshow(S1, abs=1, xlabel=xlabel, ylabel=ylabel1, title=title1,
           yticks=yticks, xticks=t, show=0, fig=fig, ax=ax1)
    ax1.tick_params(**C['tick_params_kw'])
    plt.show()


def viz_jtfs_2d(jtfs, Scx=None, viz_filterbank=True, viz_coeffs=None,
                viz_spins=(True, True), axis_labels=True, fs=None, psi_id=0,
                w=1., h=1., show=True, savename=None, plot_cfg=None):
    """Visualize JTFS filterbank and/or coefficients in their true 4D structure,
    via 2D time-frequency slices laid out in a 2D time-(log-quefrency) grid.

    Method accounts for `paths_exclude`.

    Parameters
    ----------
    jtfs : TimeFrequencyScattering1D
        JTFS instance. Requires `jtfs.out_type` that's 'dict:array' or
        'dict:list'.

    Scx : None / dict / np.ndarray
        Coefficients to visualize. Requires:

            - `jtfs.out_type` to be`'dict:list'` or `'dict:array'`. Or,
            - `Scx` to be a 4D numpy array packed with `pack_coeffs_jtfs` and
              `structure=2` (which is what it will otherwise do internally).

    viz_filterbank : bool (default True)
        Whether to visualize the filterbank.

        Note, each 2D wavelet's plot is color-normed to the wavelet's maxima
        (otherwise most wavelets vanish).

    viz_coeffs : bool / None
        Whether to visualize the coefficients (requires `Scx`).

        The coefficients and filters are placed in same slots in the 2D grid,
        so if both are visualized, we see which wavelet produced which
        coefficient. An exception is `sampling_psi_fr='recalibrate'`, as the
        visual supports only one `psi_id`, while `'recalibrate'` varies it
        with `xi2`.

        Defaults to True if `Scx` is not None.

    viz_spins : tuple[bool]
        `viz_spin_up, viz_spin_dn = viz_spins` -- can use to visualize only
        one of the two spinned pairs.

    axis_labels : bool (default True)
        If True, will label plot with title, axis labels, and units.

    fs : None / int
        Sampling rate. If provided, will display physical units (Hz), else
        discrete (cycles/sample).

    savename : str / None
        If str, will save as `savename + '0.png'` and `savename + '1.png'`,
        for filterbank and coeffs respectively.

    psi_id : int
        Indexes `jtfs.psi1_f_fr_up` & `_dn` - the ID of the filterbank
        (lower = tailored to larger input along frequency).

    w, h : int, int
        Scale plot width and height, respectively.

    show : bool (default True)
        Whether to display generated plots. Else, will `plt.close(fig)`
        (after saving, if applicable).

    plot_cfg : None / dict
        Configures plotting. Will fill for missing values from defaults
        (see `plot_cfg_defaults` in source code). Will not warn if an argument
        is unused (e.g. per `viz_coeffs=False`). Supported key-values:

            'phi_t_blank' : bool
              If True, draws `phi_t * psi_f` pairs only once (since up == down).
              Can't be `True` with `phi_t_loc='both'`.

            'phi_t_loc' : str['top', 'bottom', 'both']
              'top' places `phi_t * psi_f` pairs alongside "up" spin,
              'bottom' alongside "down", and 'both' places them in both spots.
              Additionally, 'top' and 'bottom' will scale coefficients by
              `sqrt(2)` for energy norm (since they're shown in half of all
              places).

            'filter_part' : str['real', 'imag', 'complex', 'abs']
              Part of each filter to plot.

            'filter_label' : bool (default False)
              Whether to label each filter plot with its index/meta info.

            'filter_label_kw' : dict / None
              Passed to `ax.annotate` for filterbank visuals.

            'label_kw_xy' : dict
                Passed to all `ax.set_xlabel` and `ax.set_ylabel`.

            'title_kw' : dict
                Passed to all `fig.suptitle`.

            'suplabel_kw_x' : dict
                Passed to all `fig.supxlabel`.

            'suplabel_kw_y' : dict
                Passed to all `fig.supylabel`.

            'imshow_kw_filterbank' : dict
                Passed to all `ax.imshow` for filterbank visuals.

            'imshow_kw_coeffs' : dict
                Passed to all `ax.imshow` for coefficient visuals.

            'subplots_adjust_kw' : dict
                Passed to all `fig.subplots_adjust`.

            'savefig_kw': dict
                Passed to all `fig.savefig`.

            'filterbank_zoom': float / int
                Zoom factor for filterbank visual.
                  - >1: zoom in
                  - <1: zoom out.
                  - -1: zoom every wavelet to its own support. With 'resample',
                        all wavelets within the same pair should look the same,
                        per wavelet self-similarity.

            'coeff_color_max_mult' : float
                Scales plot color norm via
                    `ax.imshow(, vmin=0, vmax=coeff_color_max_mult * Scx.max())`
                `<1` will pronounce lower-valued coefficients and clip the rest.

    Note: `xi1_fr` units
    --------------------
    Meta stores discrete, [cycles/sample].
    Want [cycles/octave].
    To get physical, we do `xi * fs`, where `fs [samples/second]`.
    Hence, find `fs` equivalent for `octaves`.

    If `Q1` denotes "number of first-order wavelets per octave", we
    realize that "samples" of `psi_fr` are actually "first-order wavelets":
        `xi [cycles/(first-order wavelets)]`
    Hence, set
        `fs [(first-order wavelets)/octave]`
    and so
        `xi1_fr = xi*fs = xi*Q1 [cycles/octave]`

     - This is consistent with raising `Q1` being equivalent of raising
       the physical sampling rate (i.e. sample `n1` more densely without
       changing the number of octaves).
     - Raising `J1` is then equivalent to increasing physical duration
       (seconds) without changing sampling rate, so `xi1_fr` is only a
       function of `Q1`.
    """
    # handle args ############################################################
    # `jtfs`, `Scx` sanity checks; set `viz_coeffs`
    if 'dict:' not in jtfs.out_type:
        raise ValueError("`jtfs.out_type` must be 'dict:array' or 'dict:list' "
                         "(got %s)" % str(jtfs.out_type))
    if Scx is not None:
        if not isinstance(Scx, dict):
            assert isinstance(Scx, np.ndarray), type(Scx)
            assert Scx.ndim == 4, Scx.shape
        else:
            assert isinstance(Scx, dict), type(Scx)
        if viz_coeffs is None:
            viz_coeffs = True
        elif not viz_coeffs:
            warnings.warn("Passed `Scx` and `viz_coeffs=False`; won't visualize!")
    elif viz_coeffs:
        raise ValueError("`viz_coeffs=True` requires passing `Scx`.")
    # `viz_coeffs`, `viz_filterbank` sanity check
    if not viz_coeffs and not viz_filterbank:
        raise ValueError("Nothing to visualize! (viz_coeffs and viz_filterbank "
                         "aren't True")
    # `psi_id` sanity check
    psi_ids_max = max(jtfs.psi_ids.values())
    if psi_id > psi_ids_max:
        raise ValueError("`psi_id` exceeds max existing value ({} > {})".format(
            psi_id, psi_ids_max))
    elif psi_id > 0 and jtfs.sampling_psi_fr == 'exclude':
        raise ValueError("`psi_id > 0` with `sampling_psi_fr = 'exclude'` "
                         "is not supported; to see which filters are excluded, "
                         "check which coefficients are zero.")

    # `plot_cfg`, defaults
    plot_cfg_defaults = {
        'phi_t_blank': None,
        'phi_t_loc': 'bottom',

        'filter_part': 'real',
        'filter_label': False,
        'filter_label_kw': dict(weight='bold', fontsize=26, xy=(.05, .82),
                                xycoords='axes fraction'),

        'label_kw_xy':   dict(fontsize=20),
        'title_kw':      dict(weight='bold', fontsize=26, y=1.025),
        'suplabel_kw_x': dict(weight='bold', fontsize=24, y=-.05),
        'suplabel_kw_y': dict(weight='bold', fontsize=24, x=-.066),
        'imshow_kw_filterbank': dict(aspect='auto', cmap='bwr'),
        'imshow_kw_coeffs':     dict(aspect='auto', cmap='turbo'),
        'subplots_adjust_kw': dict(left=0, right=1, bottom=0, top=1,
                                   wspace=.02, hspace=.02),
        'savefig_kw': dict(bbox_inches='tight'),

        'filterbank_zoom': .9,
        'coeff_color_max_mult': .8,
    }
    C = fill_default_args(plot_cfg, plot_cfg_defaults, copy_original=True,
                          check_against_defaults=True)

    # viz_spin; phi_t_loc; phi_t_blank
    viz_spin_up, viz_spin_dn = viz_spins

    assert C['phi_t_loc'] in ('top', 'bottom', 'both')
    if C['phi_t_loc'] == 'both':
        if C['phi_t_blank']:
            warnings.warn("`phi_t_blank` does nothing if `phi_t_loc='both'`")
            C['phi_t_blank'] = 0
    elif C['phi_t_blank'] is None:
        C['phi_t_blank'] = 1

    # fs
    if fs is not None:
        f_units = "[Hz]"
    else:
        f_units = "[cycles/sample]"

    # pack `Scx`, get meta ###################################################
    jmeta = jtfs.meta()
    if Scx is not None:
        if isinstance(Scx, dict):
            Scx = pack_coeffs_jtfs(Scx, jmeta, structure=2, out_3D=jtfs.out_3D,
                                   sampling_psi_fr=jtfs.sampling_psi_fr,
                                   reverse_n1=False)
            # reverse psi_t ordering
            Scx = Scx[::-1]

    # unpack filters and relevant meta #######################################
    n2s    = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 0])
    n1_frs = np.unique(jmeta['n']['psi_t * psi_f_up'][..., 1])
    n_n2s, n_n1_frs = len(n2s), len(n1_frs)

    psi2s = [p for n2, p in enumerate(jtfs.psi2_f) if n2 in n2s]
    psis_up, psis_dn = [[p for n1_fr, p in enumerate(psi1_f_fr[psi_id])
                         if n1_fr in n1_frs]
                        for psi1_f_fr in (jtfs.psi1_f_fr_up, jtfs.psi1_f_fr_dn)]
    # must time-reverse to plot, so that
    #     low plot idx <=> high wavelet idx,    i.e.
    #                  <=> high spatial sample, i.e.
    #                  <=> high log-frequency
    # Up is time-reversed down, so just swap
    psis_dn, psis_up = psis_up, psis_dn
    pdn_meta = {field: [value for n1_fr, value in
                        enumerate(jtfs.psi1_f_fr_dn[field][psi_id])
                        if n1_fr in n1_frs]
                for field in jtfs.psi1_f_fr_dn if isinstance(field, str)}

    # Visualize ################################################################
    # helpers ###################################
    def show_filter(pt, pf, row_idx, col_idx, label_axis_fn=None,
                    n2_idx=None, n1_fr_idx=None, mx=None, up=None, skip=False):
        # style first so we can exit early if needed
        ax0 = axes0[row_idx, col_idx]
        no_border(ax0)
        if axis_labels and label_axis_fn is not None:
            label_axis_fn(ax0)

        if skip:
            return

        # trim to zoom on wavelet
        if zoom_each:
            if n2_idx == -1:
                stz = jtfs.phi_f['support'] // 2
            else:
                stz = psi2s[len(psi2s) - n2_idx - 1]['support'][0] // 2
            if n1_fr_idx == -1:
                scale_diff = list(jtfs.psi_ids.values()).index(psi_id)
                pad_diff = jtfs.J_pad_frs_max_init - jtfs.J_pad_frs[scale_diff]
                sfz = jtfs.phi_f_fr['support'][0][pad_diff][0] // 2
            else:
                supps = (jtfs.psi1_f_fr_up if up else
                         jtfs.psi1_f_fr_dn)['support'][psi_id]
                ix = (n1_frs if up else n1_frs[::-1])[n1_fr_idx]
                sfz = supps[ix] // 2
            pt = pt[ct - stz:ct + stz + 1]
            pf = pf[cf - sfz:cf + sfz + 1]
        else:
            pt = pt[ct - st:ct + st + 1]
            pf = pf[cf - sf:cf + sf + 1]

        Psi = pf[:, None] * pt[None]
        if mx is None:
            mx = np.abs(Psi).max()
        mn = -mx
        if C['filter_part'] == 'real':
            Psi = Psi.real
        elif C['filter_part'] == 'imag':
            Psi = Psi.imag
        elif C['filter_part'] == 'complex':
            Psi = _colorize_complex(Psi)
        elif C['filter_part'] == 'abs':
            Psi = np.abs(Psi)
            mn = 0

        ax0.imshow(Psi, vmin=mn, vmax=mx, **C['imshow_kw_filterbank'])
        if C['filter_label']:
            psi_txt = get_filter_label(n2_idx, n1_fr_idx, up)
            ax0.annotate(psi_txt, **C['filter_label_kw'])

    def get_filter_label(n2_idx, n1_fr_idx, up=None):
        if n2_idx != -1:
            n_t_psi = int(n2s[n_n2s - n2_idx - 1])
        if n1_fr_idx != -1:
            n_f_psi = int(n1_frs[n1_fr_idx] if up else
                          n1_frs[n_n1_frs - n1_fr_idx - 1])

        if n2_idx == -1 and n1_fr_idx == -1:
            info = ("\infty", "\infty", 0)
        elif n2_idx == -1:
            info = ("\infty", n_f_psi, 0)
        elif n1_fr_idx == -1:
            info = (n_t_psi, "\infty", 0)
        else:
            info = (n_t_psi, n_f_psi, '+1' if up else '-1')

        psi_txt = r"$\Psi_{%s, %s, %s}$" % info
        return psi_txt

    def no_border(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    def to_time(p_f):
        while isinstance(p_f, (dict, list)):
            p_f = p_f[0]
        return ifftshift(ifft(p_f.squeeze()))

    # generate canvas ###########################
    if viz_spin_up and viz_spin_dn:
        n_rows = 2*n_n1_frs + 1
    else:
        n_rows = n_n1_frs + 1
    n_cols = n_n2s + 1

    width  = 11 * w
    height = 11 * n_rows / n_cols * h

    if viz_filterbank:
        fig0, axes0 = plt.subplots(n_rows, n_cols, figsize=(width, height))
    if viz_coeffs:
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(width, height))

    # compute common params to zoom on wavelets based on largest wavelet
    # centers
    n1_fr_largest = n_n1_frs - 1
    n2_largest = n_n2s - 1
    pf_f = psis_dn[n1_fr_largest].squeeze()
    pt_f = psi2s[n2_largest][0].squeeze()
    ct = len(pt_f) // 2
    cf = len(pf_f) // 2

    zoom_each = bool(C['filterbank_zoom'] == -1)
    if not zoom_each:
        # supports
        # `1/2` is base zoom since 'support' is total support, so halving
        # allows using it like `psi[center-st:center+st]`.
        # `min` to not allow excess zoom out that indexes outside the array
        global_zoom = (1 / 2) / C['filterbank_zoom']
        st = min(int(psi2s[n2_largest]['support'][0]    * global_zoom), ct)
        sf = min(int(pdn_meta['support'][n1_fr_largest] * global_zoom), cf)

    # coeff max
    if Scx is not None:
        cmx = Scx.max() * C['coeff_color_max_mult']

    # plot pairs ################################
    def plot_spinned(up):
        def label_axis(ax, n1_fr_idx, n2_idx):
            at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
            if at_border:
                xi2 = psi2s[::-1][n2_idx]['xi']
                if fs is not None:
                    xi2 = xi2 * fs
                xi2 = _format_ticks(xi2)
                ax.set_xlabel(xi2, **C['label_kw_xy'])

        if up:
            psi1_frs = psis_up
        else:
            psi1_frs = psis_dn[::-1]

        for n2_idx, pt_f in enumerate(psi2s[::-1]):
            for n1_fr_idx, pf_f in enumerate(psi1_frs):
                # compute axis & coef indices ################################
                if up:
                    row_idx = n1_fr_idx
                    coef_n1_fr_idx = n1_fr_idx
                else:
                    if viz_spin_up:
                        row_idx = n1_fr_idx + 1 + n_n1_frs
                    else:
                        row_idx = n1_fr_idx + 1
                    coef_n1_fr_idx = n1_fr_idx + n_n1_frs + 1
                col_idx = n2_idx + 1
                coef_n2_idx = n2_idx + 1

                # visualize ##################################################
                # filterbank
                if viz_filterbank:
                    pt = to_time(pt_f)
                    pf = to_time(pf_f)
                    # if both spins, viz only on down
                    if (((viz_spin_up and viz_spin_dn) and not up) or
                        not (viz_spin_up and viz_spin_dn)):
                        label_axis_fn = lambda ax0: label_axis(ax0, n1_fr_idx,
                                                               n2_idx)
                    else:
                        label_axis_fn = None
                    show_filter(pt, pf, row_idx, col_idx, label_axis_fn,
                                n2_idx, n1_fr_idx, up=up)

                # coeffs
                if viz_coeffs:
                    c = Scx[coef_n2_idx, coef_n1_fr_idx]

                    ax1 = axes1[row_idx, col_idx]
                    ax1.imshow(c, vmin=0, vmax=cmx,
                               **C['imshow_kw_coeffs'])

                    # axis styling
                    no_border(ax1)
                    if axis_labels:
                        label_axis(ax1, n1_fr_idx, n2_idx)

    if viz_spin_up:
        plot_spinned(up=True)
    if viz_spin_dn:
        plot_spinned(up=False)

    # psi_t * phi_f ##########################################################
    if viz_filterbank:
        phif = to_time(get_phi_for_psi_id(jtfs, psi_id))

    if viz_spin_up:
        row_idx = n_n1_frs
    else:
        row_idx = 0
    coef_n1_fr_idx = n_n1_frs

    for n2_idx, pt_f in enumerate(psi2s[::-1]):
        # compute axis & coef indices
        col_idx = n2_idx + 1
        coef_n2_idx = n2_idx + 1

        # filterbank
        if viz_filterbank:
            pt = to_time(pt_f)
            show_filter(pt, phif, row_idx, col_idx, None, n2_idx, n1_fr_idx=-1)

        # coeffs
        if viz_coeffs:
            ax1 = axes1[row_idx, col_idx]
            c = Scx[coef_n2_idx, coef_n1_fr_idx]
            ax1.imshow(c, vmin=0, vmax=cmx, **C['imshow_kw_coeffs'])
            no_border(ax1)

    # phi_t * psi_f ##########################################################
    def plot_phi_t(up):
        def label_axis(ax, n1_fr_idx):
            if up:
                filter_n1_fr_idx = n1_fr_idx
            else:
                filter_n1_fr_idx = n_n1_frs - n1_fr_idx - 1

            xi1_fr = pdn_meta['xi'][filter_n1_fr_idx] * jtfs.Q[0]
            if not up:
                xi1_fr = -xi1_fr
            xi1_fr = _format_ticks(xi1_fr)
            ax.set_ylabel(xi1_fr, **C['label_kw_xy'])

            at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
            if at_border and axis_labels:
                ax.set_xlabel("0", **C['label_kw_xy'])

        if C['phi_t_loc'] == 'top' or (C['phi_t_loc'] == 'both' and up):
            if up:
                psi1_frs = psis_up
                assert not viz_spin_dn or (viz_spin_up and viz_spin_dn)
            else:
                if viz_spin_up and viz_spin_dn:
                    # don't show stuff if both spins given
                    psi1_frs = [p*0 for p in psis_up]
                else:
                    psi1_frs = psis_dn[::-1]
        elif C['phi_t_loc'] == 'bottom' or (C['phi_t_loc'] == 'both' and not up):
            if up:
                if viz_spin_up and viz_spin_dn:
                    # don't show stuff if both spins given
                    psi1_frs = [p*0 for p in psis_up]
                else:
                    psi1_frs = psis_up
            else:
                psi1_frs = psis_dn[::-1]
                assert not viz_spin_up or (viz_spin_up and viz_spin_dn)

        col_idx = 0
        coef_n2_idx = 0
        for n1_fr_idx, pf_f in enumerate(psi1_frs):
            if up:
                row_idx = n1_fr_idx
                coef_n1_fr_idx = n1_fr_idx
            else:
                if viz_spin_up and viz_spin_dn:
                    row_idx = n1_fr_idx + 1 + n_n1_frs
                else:
                    row_idx = n1_fr_idx + 1
                coef_n1_fr_idx = n1_fr_idx + 1 + n_n1_frs

            if viz_filterbank:
                pf = to_time(pf_f)

                # determine color `mx` and whether to skip
                skip = False
                if C['phi_t_loc'] != 'both':
                    # energy norm (no effect if color norm adjusted to Psi)
                    pf *= np.sqrt(2)

                if C['phi_t_loc'] == 'top':
                    if not up and (viz_spin_up and viz_spin_dn):
                        # actually zero but that defaults the plot to max negative
                        skip = True
                elif C['phi_t_loc'] == 'bottom':
                    if up and (viz_spin_up and viz_spin_dn):
                        # actually zero but that defaults the plot to max negative
                        skip = True
                elif C['phi_t_loc'] == 'both':
                    pass

                # show
                label_axis_fn = lambda ax0: label_axis(ax0, n1_fr_idx)
                show_filter(phit, pf, row_idx, col_idx, label_axis_fn,
                            n2_idx=-1, n1_fr_idx=n1_fr_idx, skip=skip, up=up)

            if viz_coeffs:
                ax1 = axes1[row_idx, col_idx]
                skip_coef = bool(
                    C['phi_t_blank'] and ((C['phi_t_loc'] == 'top' and not up) or
                                          (C['phi_t_loc'] == 'bottom' and up)))

                if not skip_coef:
                    c = Scx[coef_n2_idx, coef_n1_fr_idx]
                    if C['phi_t_loc'] != 'both':
                        # energy norm since we viz only once;
                        # did /= sqrt(2) in pack_coeffs_jtfs
                        c = c * np.sqrt(2)
                    if C['phi_t_loc'] == 'top':
                        if not up and (viz_spin_up and viz_spin_dn):
                            c = c * 0  # viz only once
                    elif C['phi_t_loc'] == 'bottom':
                        if up and (viz_spin_up and viz_spin_dn):
                            c = c * 0  # viz only once
                    ax1.imshow(c, vmin=0, vmax=cmx,
                               **C['imshow_kw_coeffs'])

                # axis styling
                no_border(ax1)
                if axis_labels:
                    label_axis(ax1, n1_fr_idx)

    if viz_filterbank:
        phit = to_time(jtfs.phi_f)

    if viz_spin_up:
        plot_phi_t(up=True)
    if viz_spin_dn:
        plot_phi_t(up=False)

    # phi_t * phi_f ##############################################################
    def label_axis(ax):
        ax.set_ylabel("0", **C['label_kw_xy'])

    if viz_spin_up:
        row_idx = n_n1_frs
    else:
        row_idx = 0
    col_idx = 0
    coef_n2_idx = 0
    coef_n1_fr_idx = n_n1_frs

    # filterbank
    if viz_filterbank:
        label_axis_fn = label_axis
        show_filter(phit, phif, row_idx, col_idx, label_axis_fn,
                    n2_idx=-1, n1_fr_idx=-1)

    # coeffs
    if viz_coeffs:
        c = Scx[coef_n2_idx, coef_n1_fr_idx]
        ax1 = axes1[row_idx, col_idx]
        ax1.imshow(c, vmin=0, vmax=cmx, **C['imshow_kw_coeffs'])

        # axis styling
        no_border(ax1)
        if axis_labels:
            label_axis(ax1)

    # finalize ###############################################################
    def fig_adjust(fig):
        if axis_labels:
            fig.supxlabel(f"Temporal modulation {f_units}",
                          **C['suplabel_kw_x'])
            fig.supylabel("Freqential modulation [cycles/octave]",
                          **C['suplabel_kw_y'])
        fig.subplots_adjust(**C['subplots_adjust_kw'])

    if viz_filterbank:
        fig_adjust(fig0)
        if axis_labels:
            if C['filter_part'] in ('real', 'imag'):
                info_txt = "%s part" % C['filter_part']
            elif C['filter_part'] == 'complex':
                info_txt = "complex"
            elif C['filter_part'] == 'abs':
                info_txt = "modulus"
            if zoom_each:
                info_txt += ", zoomed"
            fig0.suptitle("JTFS filterbank (%s)" % info_txt, **C['title_kw'])
    if viz_coeffs:
        fig_adjust(fig1)
        if axis_labels:
            fig1.suptitle("JTFS coefficients", **C['title_kw'])

    if savename is not None:
        if viz_filterbank:
            fig0.savefig(f'{savename}0.png', **C['savefig_kw'])
        if viz_coeffs:
            fig1.savefig(f'{savename}1.png', **C['savefig_kw'])

    if show:
        plt.show()
    else:
        if viz_filterbank:
            plt.close(fig0)
        if viz_coeffs:
            plt.close(fig1)

#### Demonstrataive ##########################################################
# visuals likelier for one-time use rather than filterbank/coeff introspection

def viz_spin_2d(pair_waves=None, pairs=None, preset=None, axis_labels=None,
                pair_labels=True, fps=60, savepath='spin2d.mp4', verbose=True):
    """Visualize the complete 4D behavior of 2D (1D-separable) complex Morlet
    wavelets, with the time dimension unrolled. Also supports all JTFS pairs.

    Also supports a general 2D complex input. For a time-domain signal, apply
    `fft(fft(fftshift(fftshift(x, axes=0), axes=1), axis=0), axis=1)`
    before passing in, as this method does
    `ifftshift(ifftshift(ifft(ifft(psi_f, axis=0), axis=1), axes=0), axes=1)`
    internally.

    Parameters
    ----------
    pair_waves : dict / None
        Wavelets/lowpasses to use to generate pairs. If not provided,
        will use defaults. Supported keys:

            - 'up' for psi_f_up
            - 'dn' for psi_f_dn
            - 'psi_t' for psi_t
            - 'phi_t' for phi_t
            - 'phi_f' for phi_f

        Must provide all keys that are provided in `pairs`, except `phi_t_dn`
        which instead requires 'dn'.

    pairs : None / tuple[str['up', 'dn', 'phi_t', 'phi_f', 'phi', 'phi_t_dn']]
        Pairs to visualize. Number of specified pairs must be 1, 2, or 6.
        Will ignore pairs in `pair_waves` if they aren't in `pairs`;
        `pairs` defaults to either what's in `preset` or what's in `pair_waves`.

    preset : None / int[0, 1, 2]
        Animation preset to use:

            - 0: pairs=('up',)
            - 1: pairs=('up', 'dn')
            - 2: pairs=('up', 'dn', 'phi_t', 'phi_f', 'phi', 'phi_t_dn')

        If wavelets/lowpasses aren't passed in, will generate them
        (if `pairs_preset` is not None).

    axis_labels : None / bool
        If False, will omit axis tick labels, axis labels, and axis planes.
        Defaults to True if `len(pair_waves) == 1`.

    pair_labels : bool (default True)
        If True, will title plot with name of pair being plotted, with LaTeX.

    fps : int
        Frames per second of the animation.

    savepath : str
        Path to save the animation to, as .mp4.

    verbose : bool (default True)
        Whether to print where the animation is saved.
    """
    # handle arguments #######################################################
    pair_presets = {0: ('up',),
                    1: ('up', 'dn'),
                    2: ('up', 'phi_f', 'dn', 'phi_t', 'phi', 'phi_t_dn')}
    # handle `preset`
    if preset is None:
        preset = 0
    elif preset not in pair_presets:
        raise ValueError("`preset` %s is unsupported, must be one of %s" % (
            preset, list(pair_presets)))

    # handle `pairs`
    if pairs is None:
        if pair_waves is not None:
            pairs = list(pair_waves)
        else:
            pairs = pair_presets[preset]
    elif isinstance(pairs, str):
        pairs = (pairs,)

    # handle `pair_waves`
    if pair_waves is None:
        N, xi0, sigma0 = 256, 4., 1.35
        pair_waves = {pair: make_jtfs_pair(N, pair, xi0, sigma0)
                      for pair in pairs}
    else:
        pair_waves = pair_waves.copy()  # don't affect external keys
        for pair in pairs:
            if pair == 'phi_t_dn':
                if not ('dn' in pair_waves and 'phi_t' in pair_waves):
                    raise ValueError("pair 'phi_t_dn'` requires 'dn' and 'phi_t' "
                                     "in `pair_waves`")
                pair_waves['phi_t_dn'] = (pair_waves['dn'][:, None] *
                                          pair_waves['phi_t'][None])
            elif pair not in pair_waves:
                raise ValueError("missing pair from pair_waves: %s" % pair)

        passed_pairs = list(pair_waves)
        for pair in passed_pairs:
            if pair not in pairs:
                del pair_waves[pair]

        # convert to time, center
        for pair in pair_waves:
            pair_waves[pair] = ifftshift(ifftshift(
                ifft(ifft(pair_waves[pair], axis=0), axis=-1), axes=0), axes=-1)
            if pair == 'phi':
                pair_waves['phi'] = pair_waves['phi'].real

    # handle `axis_labels`
    if len(pair_waves) > 1 and axis_labels:
        raise ValueError("`axis_labels=True` is only supported for "
                         "`len(pair_waves) == 1`")
    elif axis_labels is None and len(pair_waves) == 1:
        axis_labels = True

    # visualize ##############################################################
    if not savepath.endswith('.mp4'):
        savepath += '.mp4'
    savepath = os.path.abspath(savepath)
    ani = SpinAnimator2D(pair_waves, axis_labels, pair_labels)
    ani.save(savepath, fps=fps, savefig_kwargs=dict(pad_inches=0))
    plt.close()

    if verbose:
        print("Saved animation to", savepath)


def viz_spin_1d(psi_f=None, fps=33, savepath='spin1d.mp4', end_pause=None,
                w=None, h=None, subplots_adjust_kw=None, verbose=True):
    """Visualize the complete 3D behavior of 1D complex Morlet wavelets.

    Also supports a general 1D complex input. For a time-domain signal, apply
    `fft(fftshift(x))` before passing in, as this method does
    `ifftshift(ifft(psi_f))` internally.

    Parameters
    ----------
    psi_f : tensor / None
        1D complex Morlet wavelet. If None, will make a default.

    fps : int
        Frames per second of the animation.

    savepath : str
        Path to save the animation to, as .mp4.

    end_pause : int / None
        Number of frames to insert at the end of animation that duplicate the
        last frame, effectively "pausing" the animation at finish.
        Defaults to `fps`, i.e. one second.

    w, h : float / None
        Animation width and height scaling factors.
        Act via `subplots(, figsize=(width*w, height*h))`.

        Defaults motivated same as `subplots_adjust_kw`.

    subplots_adjust_kw : dict / None
        Passed to `fig.subplots_adjust()`.

        Defaults strive for a `plt.tight()` layout, with presets for
        `len(psi_f)=1` and `=2`.

    verbose : bool (default True)
        Whether to print where the animation is saved.
    """
    from .scattering1d.filter_bank import morlet_1d

    # handle arguments #######################################################
    if end_pause is None:
        end_pause = fps
    if psi_f is None:
        N, xi0, sigma0 = 256, 4., 1.35
        psi_f = morlet_1d(N, xi=xi0/N, sigma=sigma0/N).squeeze()
    if not isinstance(psi_f, (list, tuple)):
        psi_f = [psi_f]
    psi_t = [ifftshift(ifft(p)) for p in psi_f]

    # visualize ##############################################################
    if not savepath.endswith('.mp4'):
        savepath += '.mp4'
    savepath = os.path.abspath(savepath)

    ani = SpinAnimator1D(psi_t, end_pause=end_pause)
    ani.save(savepath, fps=fps, savefig_kwargs=dict(pad_inches=0))
    plt.close()

    if verbose:
        print("Saved animation to", savepath)


class SpinAnimator2D(animation.TimedAnimation):
    def __init__(self, pair_waves, axis_labels=False, pair_labels=True):
        assert isinstance(pair_waves, dict), type(pair_waves)
        assert len(pair_waves) in (1, 2, 6), len(pair_waves)
        assert not (len(pair_waves) > 1 and axis_labels)

        self.pair_waves = pair_waves
        self.axis_labels = axis_labels
        self.pair_labels = pair_labels

        self.ref = list(pair_waves.values())[0]
        self.plot_frames = list(pair_waves.values())
        self.n_pairs = len(pair_waves)

        # make titles
        titles = {'up':       r"$\psi(t) \psi(+\lambda) \uparrow$",
                  'phi_f':    r"$\psi(t) \phi(\lambda)$",
                  'dn':       r"$\psi(t) \psi(-\lambda) \downarrow$",
                  'phi_t':    r"$\phi(t) \psi(+\lambda)$",
                  'phi':      r"$\phi(t) \phi(\lambda)$",
                  'phi_t_dn': r"$\phi(t) \psi(-\lambda)$"}
        self.titles = [titles[pair] for pair in self.pair_waves]

        # get quantities from reference
        self.n_f, self.N = self.ref.shape
        self.n_frames = len(self.ref)
        self.z = np.arange(self.n_f) / self.n_f
        self.T_all = np.arange(self.N) / self.N

        # get axis limits
        mx = max(np.abs(p).max() for p in list(pair_waves.values()))
        z_max  = self.z.max()

        # configure label args
        self.title_kw = dict(y=.83, weight='bold', fontsize=18)
        self.txt_kw = dict(x=3*mx, y=25*mx, z=-2*z_max, s="", ha="left")
        self.label_kw = dict(weight='bold', fontsize=18)

        # create figure & axes
        fig = plt.figure(figsize=(16, 8))
        axes = []
        subplot_args = {1: [(1, 1, 1)],
                        2: [(1, 2, i) for i in range(1, 2+1)],
                        6: [(2, 3, i) for i in range(1, 6+1)]}[self.n_pairs]
        for arg in subplot_args:
            axes.append(fig.add_subplot(*arg, projection='3d'))

        # initialize plots ###################################################
        def init_plot(i):
            ax = axes[i]
            # plot ####
            xc = self.plot_frames[i][0]
            line = ax.plot(xc.real, xc.imag, label='parametric curve',
                           linewidth=2)[0]
            line.set_data(xc.real, xc.imag)
            line.set_3d_properties(self.z)
            setattr(self, f'lines{i}', [line])

            # axes styling ####
            xlims = (-mx, mx)
            ylims = (-mx, mx)
            zlims = (0, z_max)

            ax.set_xlim3d(xlims)
            ax.set_ylim3d(ylims)
            ax.set_zlim3d(zlims)
            if self.pair_labels:
                ax.set_title(self.titles[i], **self.title_kw)

            if not axis_labels:
                # no border, panes, spines; 0 margin ####
                for anm in ('x', 'y', 'z'):
                    getattr(ax, f'set_{anm}ticks')([])
                    getattr(ax, f'{anm}axis').set_pane_color((1, 1, 1, 0))
                    getattr(ax, f'{anm}axis').line.set_color((1, 1, 1, 0))
                    getattr(ax, f'set_{anm}margin')(0)
                    ax.patch.set_alpha(0.)
            else:
                ax.set_xlabel("real", **self.label_kw)
                ax.set_ylabel("imag", **self.label_kw)
                ax.set_zlabel(r"$\lambda$", **self.label_kw)
                setattr(self, f'txt{i}',
                        ax.text(transform=ax.transAxes, **self.txt_kw,
                                fontsize=18))

        for i in range(len(axes)):
            init_plot(i)

        # finalize #######################################################
        wspace = -.65 if self.n_pairs == 6 else -.45
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=-.4, wspace=wspace)
        animation.TimedAnimation.__init__(self, fig, blit=True)

    def _draw_frame(self, frame_idx):
        # plot ###############################################################
        lines, txts = [], []
        for i in range(self.n_pairs):
            xc = self.plot_frames[i][frame_idx]
            line = getattr(self, f'lines{i}')
            line[0].set_data(xc.real, xc.imag)
            line[0].set_3d_properties(self.z)
            lines.append(*line)

            if self.axis_labels:
                T_sec = self.T_all[frame_idx]
                txt = getattr(self, f'txt{i}')
                txt.set_text("t=%.3f" % T_sec)
                txts.append(txt)

        # finalize ###########################################################
        self._drawn_artists = [*lines, *txts]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass


class SpinAnimator1D(animation.TimedAnimation):
    def __init__(self, plot_frames, end_pause=0, w=None, h=None,
                 subplots_adjust_kw=None):
        self.plot_frames = plot_frames
        self.end_pause = end_pause
        n_plots = len(plot_frames)
        self.n_plots = n_plots
        ref = plot_frames[0]

        # handle `subplots_adjust_kw`
        sakw_defaults = {
            1: dict(top=1, bottom=0, right=.975, left=.075, hspace=.1,
                    wspace=.1),
            2: dict(top=1, bottom=0, right=.975, left=.075, hspace=-.7,
                    wspace=.1),
        }[n_plots]
        subplots_adjust_kw = fill_default_args(subplots_adjust_kw, sakw_defaults)
        # handle `w, h`
        if w is None:
            w = {1: 1, 2: 1}[n_plots]
        if h is None:
            h = {1: 1, 2: 1.1}[n_plots]

        # get quantities from reference
        self.n_frames = len(ref)
        self.n_frames_total = self.n_frames + self.end_pause
        self.z = np.arange(len(ref)) / len(ref)
        self.T_all = np.arange(self.n_frames) / self.n_frames

        # get axis limits
        zmax = self.z.max()
        mx = max(np.abs(p).max() for p in plot_frames)

        # configure labels
        self.txt_kw = dict(x=-.25*mx, y=1.03*mx, s="", ha="left")
        self.label_kw = dict(weight='bold', fontsize=18)

        # create figure & axes
        width, height = 13/1.02, 16/1.1
        width *= w
        height *= h
        n_rows_base = 6
        fig, axes = plt.subplots(n_rows_base*n_plots, 7, figsize=(width, height))

        # gridspec object allows treating multiple axes as one
        gs = axes[0, 0].get_gridspec()
        # remove existing axes
        for ax in axes.flat:
            ax.remove()

        def init_plot(i):
            # create two axes with greater height and width ratio for the 2D
            # plot, since 3D is mainly padding
            inc = i * n_rows_base  # index increment
            ax0 = fig.add_subplot(gs[(inc + 2):(inc + 4), :3])
            ax1 = fig.add_subplot(gs[(inc + 0):(inc + n_rows_base), 3:],
                                  projection='3d')

            # initialize plots ###############################################
            plot_frames = self.plot_frames[i]
            xc = plot_frames[0]
            color = np.array([[102, 0, 204]])/256
            dot0 = ax0.scatter(xc.real, xc.imag, c=color)
            setattr(self, f'dots{i}0', [dot0])

            xcl = plot_frames[:1]
            line1 = ax1.plot(xcl.real, xcl.imag, label='parametric curve')[0]
            line1.set_data(xcl.real, xcl.imag)
            line1.set_3d_properties(0.)

            dot1 = ax1.scatter(xc.real, xc.imag, 0., c=color)
            dot1.set_3d_properties(0., 'z')
            setattr(self, f'lines{i}1', [line1, dot1])

            # styling ####
            # limits
            ax0.set_xlim(-mx, mx)
            ax0.set_ylim(-mx, mx)

            ax1.set_xlim(-mx, mx)
            ax1.set_ylim(-mx, mx)
            ax1.set_zlim(0, zmax)

            # labels
            ax0.set_xlabel("real", **self.label_kw)
            ax0.set_ylabel("imag", **self.label_kw)
            setattr(self, f'txt{i}', ax0.text(**self.txt_kw, fontsize=18))

            ax1.set_xlabel("real", **self.label_kw)
            ax1.set_ylabel("imag", **self.label_kw)
            ax1.set_zlabel(r"$t$", **self.label_kw)

        for i in range(self.n_plots):
            init_plot(i)

        # finalize #######################################################
        fig.subplots_adjust(**subplots_adjust_kw)
        animation.TimedAnimation.__init__(self, fig, blit=True)

    def _draw_frame(self, frame_idx):
        if frame_idx < self.n_frames:
            self._drawn_artists = []
            for i in range(self.n_plots):
                plot_frames = self.plot_frames[i]
                # plot #######################################################
                # dot
                name = f'dots{i}0'
                dotsi0 = getattr(self, name)

                xc = plot_frames[frame_idx]
                xc = np.array([[xc.real, xc.imag]])
                dotsi0[0].set_offsets(xc)

                setattr(self, name, dotsi0)
                self._drawn_artists.append(dotsi0)

                # spiral
                name = f'lines{i}1'
                linesi1 = getattr(self, name)

                xcl = plot_frames[:frame_idx+1]
                linesi1[0].set_data(xcl.real, xcl.imag)
                linesi1[0].set_3d_properties(self.z[:frame_idx+1])

                linesi1[1].set_offsets(xc)
                linesi1[1].set_3d_properties(frame_idx/self.n_frames, 'z')

                setattr(self, name, linesi1)
                self._drawn_artists.append(linesi1)

                # text
                name = f'txt{i}'
                txti = getattr(self, name)

                T_sec = self.T_all[frame_idx]
                txti.set_text("t=%.3f" % T_sec)

                setattr(self, name, txti)
                self._drawn_artists.append(txti)
        else:
            # repeat the last frame
            pass

    def new_frame_seq(self):
        return iter(range(self.n_frames_total))

    def _init_draw(self):
        pass


def make_gif(loaddir, savepath, duration=250, start_end_pause=0, ext='.png',
             delimiter='', overwrite=False, delete_images=False, HD=None,
             verbose=False):
    """Makes gif out of images in `loaddir` directory with `ext` extension,
    and saves to `savepath`.

    Parameters
    ----------
    loaddir : str
        Path to directory from which to fetch images to use as GIF frames.

    savepath : path
        Save path, must end with '.gif'.

    duration : int
        Interval between each GIF frame, in milliseconds.

    start_end_pause : int / tuple[int]
        Number of times to repeat the start and end frames, which multiplies
        their `duration`; if tuple, first element is for start, second for end.

    ext : str
        Images filename extension.

    delimiter : str
        Substring common to all iamge filenames, e.g. 'img' for 'img0.png',
        'img1.png', ... .

    overwrite : bool (default False)
        If True and file at `savepath` exists, will overwrite it.

    HD : bool / None
        If True, will preserve image quality in GIFs and use `imageio`.
        Defaults to True if `imageio` is installed, else falls back on
        `PIL.Image`.

    delete_images : bool (default False)
        Whether to delete the images used to make the GIF.

    verbose : bool (default False)
        Whether to print to console the location of save file upon success.
    """
    # handle `HD`
    if HD or HD is None:
        try:
            import imageio
            HD = True
        except ImportError as e:
            if HD:
                print("`HD=True` requires `imageio` installed")
                raise e
            else:
                try:
                    from PIL import Image
                except ImportError as e:
                    print("`make_gif` requires `imageio` or `PIL` installed.")
                    raise e
                HD = False

    # fetch frames
    loaddir = os.path.abspath(loaddir)
    names = [n for n in os.listdir(loaddir)
             if (n.startswith(delimiter) and n.endswith(ext))]
    names = sorted(names, key=lambda p: int(
        ''.join(s for s in p.split(os.sep)[-1] if s.isdigit())))
    paths = [os.path.join(loaddir, n) for n in names]
    frames = [(imageio.imread(p) if HD else Image.open(p))
              for p in paths]

    # handle frame duplication to increase their duration
    if start_end_pause is not None:
        if not isinstance(start_end_pause, (tuple, list)):
            start_end_pause = (start_end_pause, start_end_pause)
        for repeat_start in range(start_end_pause[0]):
            frames.insert(0, frames[0])
        for repeat_end in range(start_end_pause[1]):
            frames.append(frames[-1])

    if os.path.isfile(savepath) and overwrite:
        # delete if exists
        os.unlink(savepath)
    # save
    if HD:
        imageio.mimsave(savepath, frames, fps=1000/duration)
    else:
        frame_one = frames[0]
        frame_one.save(savepath, format="GIF", append_images=frames,
                       save_all=True, duration=duration, loop=0)
    if verbose:
        print("Saved gif to", savepath)

    if delete_images:
        for p in paths:
            os.unlink(p)
        if verbose:
            print("Deleted images used in making the GIF (%s total)" % len(paths))


#### Visuals primitives ## messy code ########################################
def imshow(x, title=None, show=True, cmap=None, norm=None, abs=0,
           w=None, h=None, ticks=True, borders=True, aspect='auto',
           ax=None, fig=None, yticks=None, xticks=None, xlabel=None, ylabel=None,
           **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if norm is None:
        mx = np.max(np.abs(x))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    if cmap == 'none':
        cmap = None
    elif cmap is None:
        cmap = 'turbo' if abs else 'bwr'
    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        ax.imshow(np.abs(x), **_kw)
    else:
        ax.imshow(x.real, **_kw)

    _handle_ticks(ticks, xticks, yticks, ax)

    if title is not None:
        _title(title, ax=ax)
    if w or h:
        fig.set_size_inches(12 * (w or 1), 12 * (h or 1))

    _scale_plot(fig, ax, show=False, w=None, h=None, xlabel=xlabel,
                ylabel=ylabel, auto_xlims=False)

    if not borders:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)

    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, complex=0, abs=0, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None,
         xlabel=None, ylabel=None, xticks=None, yticks=None, ticks=True,
         ax=None, fig=None, squeeze=True, auto_xlims=None, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        y = y if isinstance(y, list) or not squeeze else y.squeeze()
        x = np.arange(len(y))
    elif y is None:
        x = x if isinstance(x, list) or not squeeze else x.squeeze()
        y = x
        x = np.arange(len(x))
    x = x if isinstance(x, list) or not squeeze else x.squeeze()
    y = y if isinstance(y, list) or not squeeze else y.squeeze()

    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)

    # styling
    if vlines:
        vhlines(vlines, kind='v', ax=ax)
    if hlines:
        vhlines(hlines, kind='h', ax=ax)

    _handle_ticks(ticks, xticks, yticks, ax)

    if title is not None:
        _title(title, ax=ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def scat(x, y=None, title=None, show=0, s=18, w=None, h=None,
         xlims=None, ylims=None, vlines=None, hlines=None, ticks=1,
         complex=False, abs=False, xlabel=None, ylabel=None, ax=None, fig=None,
         auto_xlims=None, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    if complex:
        ax.scatter(x, y.real, s=s, **kw)
        ax.scatter(x, y.imag, s=s, **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.scatter(x, y, s=s, **kw)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        _title(title, ax=ax)
    if vlines:
        vhlines(vlines, kind='v', ax=ax)
    if hlines:
        vhlines(hlines, kind='h', ax=ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def plotscat(*args, **kw):
    show = kw.pop('show', False)
    plot(*args, **kw)
    scat(*args, **kw)
    if show:
        plt.show()


def hist(x, bins=500, title=None, show=0, stats=0, ax=None, fig=None,
         w=1, h=1, xlims=None, ylims=None, xlabel=None, ylabel=None):
    """Histogram. `stats=True` to print mean, std, min, max of `x`."""
    def _fmt(*nums):
        return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
                 ("%.3f" % n)) for n in nums]

    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    x = np.asarray(x)
    _ = ax.hist(x.ravel(), bins=bins)
    _title(title, ax)
    _scale_plot(fig, ax, show=show, w=w, h=h, xlims=xlims, ylims=ylims,
                xlabel=xlabel, ylabel=ylabel)
    if show:
        plt.show()

    if stats:
        mu, std, mn, mx = (x.mean(), x.std(), x.min(), x.max())
        print("(mean, std, min, max) = ({}, {}, {}, {})".format(
            *_fmt(mu, std, mn, mx)))
        return mu, std, mn, mx


def vhlines(lines, kind='v', ax=None):
    lfn = getattr(plt if ax is None else ax, f'ax{kind}line')

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)

    for line in lines:
        lfn(line, **lkw)

#### misc / utils ############################################################
def _ticks(xticks, yticks, ax):
    def fmt(ticks):
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.2f")

    if yticks is not None:
        if not hasattr(yticks, '__len__') and not yticks:
            ax.set_yticks([])
        else:
            if isinstance(yticks, tuple):
                yticks, ykw = yticks
            else:
                ykw = {}

            idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
            yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
            ax.set_yticks(idxs)
            ax.set_yticklabels(yt, **ykw)
    if xticks is not None:
        if not hasattr(xticks, '__len__') and not xticks:
            ax.set_xticks([])
        else:
            if isinstance(xticks, tuple):
                xticks, xkw = xticks
            else:
                xkw = {}
            idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
            xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
            ax.set_xticks(idxs)
            ax.set_xticklabels(xt, **xkw)


def _handle_ticks(ticks, xticks, yticks, ax):
    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)


def _title(title, ax=None):
    if title is None:
        return
    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    defaults = dict(loc='left', fontsize=17, weight='bold')
    for k, v in defaults.items():
        kw[k] = kw.get(k, v)

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, xlabel=None, ylabel=None,
                auto_xlims=True):
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin + .02 * rng, xmax - .02 * rng)

    if ylims:
        ax.set_ylim(*ylims)
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        if isinstance(xlabel, tuple):
            xlabel, xkw = xlabel
        else:
            xkw = dict(weight='bold', fontsize=15)
        ax.set_xlabel(xlabel, **xkw)
    if ylabel is not None:
        if isinstance(ylabel, tuple):
            ylabel, ykw = ylabel
        else:
            ykw = dict(weight='bold', fontsize=15)
        ax.set_ylabel(ylabel, **ykw)
    if show:
        plt.show()


def _colorize_complex(z):
    """Map complex `z` to 3D array suitable for complex image visualization.

    Borrowed from https://stackoverflow.com/a/20958684/10133797
    """
    from colorsys import hls_to_rgb
    z = z / np.abs(z).max()
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 / (1 + r)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)
    c = np.array(c)
    c = c.swapaxes(0, 2).transpose(1, 0, 2)
    return c


def _get_compute_pairs(pairs, meta):
    # enforce pair order
    if pairs is None:
        pairs_all = ('S0', 'S1', 'phi_t * phi_f', 'phi_t * psi_f',
                     'psi_t * phi_f', 'psi_t * psi_f_up', 'psi_t * psi_f_dn')
    else:
        pairs_all = pairs if not isinstance(pairs, str) else [pairs]
    compute_pairs = []
    for pair in pairs_all:
        if pair in meta['n']:
            compute_pairs.append(pair)
    return compute_pairs


def _filterbank_style_axes(ax, N, xlims, ymax=None, zoom=None, is_jtfs=False):
    if zoom != -1:
        xticks = np.linspace(0, N, 9, endpoint=1).astype(int)
        # x limits and labels
        w = np.linspace(0, 1, len(xticks), 1)
        w[w > .5] -= 1
        ax.set_xticks(xticks[:-1])
        ax.set_xticklabels(w[:-1])
        ax.set_xlim(*xlims)
    else:
        xticks = np.linspace(0, N, 9, endpoint=1).astype(int)
        w = [-.5, -.375, -.25, -.125, 0, .125, .25, .375, .5]
        ax.set_xticks(xticks)
        ax.set_xticklabels(w)

    # y limits
    ax.set_ylim(-.05, ymax)


def _make_titles_jtfs(compute_pairs, target):
    """For energies and distances."""
    # make `titles`
    titles = []
    pair_aliases = {'psi_t * phi_f': '* phi_f', 'phi_t * psi_f': 'phi_t *',
                    'psi_t * psi_f_up': 'up', 'psi_t * psi_f_dn': 'down'}
    title = "%s | " % target
    for pair in compute_pairs:
        if pair in pair_aliases:
            title += "{}, ".format(pair_aliases[pair])
        else:
            title += "{}, ".format(pair)
    title = title.rstrip(', ')
    titles.append(title)

    title = "cumsum(%s)" % target
    titles.append(title)
    return titles


def _handle_gif_args(savedir, base_name, images_ext, save_images, overwrite,
                     show):
    do_gif = bool(savedir is not None)
    if save_images is None:
        if savedir is None:
            save_images = bool(not show)
        else:
            save_images = False
    if show is None:
        show = bool(not save_images and not do_gif)

    if savedir is None and save_images:
        savedir = ''
    if savedir is not None:
        savedir = os.path.abspath(savedir)

    if not images_ext.startswith('.'):
        images_ext = '.' + images_ext

    if not base_name.endswith('.gif'):
        base_name += '.gif'
    savepath = os.path.join(savedir, base_name)
    _check_savepath(savepath, overwrite)
    return savedir, savepath, images_ext, save_images, show, do_gif


def _format_ticks(ticks, max_digits=3):
    # `max_digits` not strict
    not_iterable = bool(not isinstance(ticks, (tuple, list, np.ndarray)))
    if not_iterable:
        ticks = [ticks]
    _ticks = []
    for tk in ticks:
        negative = False
        if tk < 0:
            negative = True
            tk = abs(tk)

        n_nondecimal = np.log10(tk)
        if n_nondecimal < 0:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)) + 1)
            n_total = n_nondecimal + 2
            tk = f"%.{n_total - 1}f" % tk
        else:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)))
            n_decimal = max(0, max_digits - n_nondecimal)
            tk = round(tk, n_decimal)
            tk = f"%.{n_decimal}f" % tk

        if negative:
            tk = "-" + tk
        _ticks.append(tk)
    if not_iterable:
        _ticks = _ticks[0]
    return _ticks


def _check_savepath(savepath, overwrite):
    if os.path.isfile(savepath):
        if not overwrite:
            raise RuntimeError("File already exists at `savepath`; "
                               "set `overwrite=True` to overwrite.\n"
                               "%s" % str(savepath))
        else:
            # delete if exists
            os.unlink(savepath)
