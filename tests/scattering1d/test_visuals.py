# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
"""Test that wavespin/visuals.py methods run without error."""
import pytest, os, warnings
from wavespin import Scattering1D, TimeFrequencyScattering1D
from wavespin.toolkit import echirp, pack_coeffs_jtfs
from wavespin import visuals as v
from utils import tempdir, SKIPS

# backend to use for most tests
default_backend = 'numpy'
# set True to execute all test functions without pytest
run_without_pytest = 0
# set True to disable matplotlib plots
# (done automatically for CI via `conftest.py`, but `False` here takes precedence)
no_plots = 1
# set True to skip this file entirely
skip_all = SKIPS['visuals']

# disable plots for pytest unit testing
# restart kernel when changing `no_plots`
if no_plots:
    import matplotlib
    matplotlib.use('template')
    matplotlib.pyplot.ioff()

# set up reusable references in global scope
sc_tms, jtfss, sc_all = [], [], []
metas = []
xs = []
out_tms, out_jtfss, out_all = [], [], []


def make_reusables():
    # run after __main__ so test doesn't fail during collection
    # reusable scattering objects
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    N = 512
    kw0 = dict(shape=N, J=9, Q=8, frontend=default_backend)
    sc_tms.extend([Scattering1D(**kw0, out_type='array')])

    sfs = [('resample', 'resample'), ('exclude', 'resample'),
           ('recalibrate', 'recalibrate')]
    kw1 = dict(out_type='dict:array', **kw0)
    jtfss.extend([
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[0]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[1]),
        TimeFrequencyScattering1D(**kw1, sampling_filters_fr=sfs[2]),
    ])
    sc_all.extend([*sc_tms, *jtfss])

    # reusable input
    xs.append(echirp(N))
    # reusable outputs
    out_tms.extend([sc(xs[0]) for sc in sc_tms])
    out_jtfss.extend([jtfs(xs[0]) for jtfs in jtfss])
    out_all.extend([*out_tms, *out_jtfss])
    # metas
    metas.extend([sc.meta() for sc in sc_all])

    return sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all


#### Tests ###################################################################

def test_filterbank_heatmap(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    for i, sc in enumerate(sc_all):
        frequential = bool(i > 0)
        v.filterbank_heatmap(sc, first_order=True, second_order=True,
                             frequential=frequential)


def test_filterbank_scattering(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    sc_all = G['sc_all']
    for sc in sc_all:
        v.filterbank_scattering(sc, second_order=1, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        v.filterbank_scattering(sc_tms[0], second_order=1, lp_sum=1, zoom=zoom)


def test_filterbank_jtfs_1d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    jtfss = G['jtfss']
    for jtfs in jtfss:
        v.filterbank_jtfs_1d(jtfs, lp_sum=1, zoom=0)
    for zoom in (4, -1):
        v.filterbank_jtfs_1d(jtfs, lp_sum=0, lp_phi=0, zoom=zoom)


def test_viz_jtfs_2d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    jtfss = G['jtfss']
    out_jtfss = G['out_jtfss']
    v.viz_jtfs_2d(jtfss[1], Scx=out_jtfss[1])


def test_gif_jtfs_2d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    out_jtfss, metas = G['out_jtfss'], G['metas']
    savename = 'jtfs2d.gif'
    fn = lambda savedir: v.gif_jtfs_2d(out_jtfss[1], metas[2],
                                       savedir=savedir, base_name=savename)
    _run_with_cleanup(fn, savename)


def test_gif_jtfs_3d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    try:
        import plotly
    except ImportError:
        warnings.warn("Skipped `test_gif_jtfs_3d` since `plotly` not installed.")
        return

    out_jtfss, metas = G['out_jtfss'], G['metas']
    packed = pack_coeffs_jtfs(out_jtfss[1], metas[2], structure=2,
                              sampling_psi_fr='exclude')

    savename = 'jtfs3d.gif'
    fn = lambda savedir: v.gif_jtfs_3d(packed, savedir=savedir, base_name=savename,
                                       images_ext='.png', verbose=False)
    _run_with_cleanup_handle_exception(fn, savename)


def test_energy_profile_jtfs(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = v.energy_profile_jtfs(Scx, metas[1 + i], **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def test_coeff_distance_jtfs(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    out_jtfss = G['out_jtfss']
    for i, Scx in enumerate(out_jtfss):
      for flatten in (False, True):
        for pairs in (None, ('phi_t * psi_f',)):
          test_params = dict(flatten=flatten, pairs=pairs)
          try:
              _ = v.coeff_distance_jtfs(Scx, Scx, metas[1 + i], **test_params)
          except Exception as e:
              test_params['i'] = i
              print('\n'.join(f'{k}={v}' for k, v in test_params.items()))
              raise e


def test_viz_spin_1d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    savename = 'spin_1d.mp4'
    fn = lambda savedir: v.viz_spin_1d(savepath=os.path.join(savedir, savename),
                                       verbose=0)
    _run_with_cleanup_handle_exception(fn, savename)


def test_viz_spin_2d(G):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    savename = 'spin_2d.mp4'
    fn = lambda savedir: v.viz_spin_2d(savepath=os.path.join(savedir, savename),
                                       preset=2, verbose=0)
    _run_with_cleanup_handle_exception(fn, savename)


def _run_with_cleanup(fn, savename):
    if skip_all:
        return None if run_without_pytest else pytest.skip()
    with tempdir() as savedir:
        try:
            fn(savedir)
            path = os.path.join(savedir, savename)
            # assert file was created
            assert os.path.isfile(path), path
            os.unlink(path)
        finally:
            # clean up images, if any were made
            paths = [os.path.join(savedir, n) for n in os.listdir(savedir)
                     if (n.startswith(savename) and n.endswith('.png'))]
            for p in paths:
                os.unlink(p)


def _run_with_cleanup_handle_exception(fn, savename):
    try:
        _run_with_cleanup(fn, savename)
    except Exception as e:
        if 'ffmpeg' not in str(e):
            # automated testing has some issues with this
            raise e


# create testing objects #####################################################
if run_without_pytest:
    sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
        make_reusables())
    G = dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
             xs=xs, out_tms=out_tms, out_jtfss=out_jtfss, out_all=out_all)
else:
    mr = [False]
    @pytest.fixture(scope='module')
    def G():
        if skip_all:
            return
        if not mr[0]:
            sc_tms, jtfss, sc_all, metas, xs, out_tms, out_jtfss, out_all = (
                make_reusables())
            mr[0] = True
        return dict(sc_tms=sc_tms, jtfss=jtfss, sc_all=sc_all, metas=metas,
                    xs=xs, out_tms=out_tms, out_jtfss=out_jtfss,
                    out_all=out_all)


# run tests ##################################################################
if __name__ == '__main__':
    if run_without_pytest and not skip_all:
        test_filterbank_heatmap(G)
        test_filterbank_scattering(G)
        test_filterbank_jtfs_1d(G)
        test_viz_jtfs_2d(G)
        test_gif_jtfs_2d(G)
        test_gif_jtfs_3d(G)
        test_energy_profile_jtfs(G)
        test_coeff_distance_jtfs(G)
        test_viz_spin_1d(G)
        test_viz_spin_2d(G)
    else:
        pytest.main([__file__, "-s"])
