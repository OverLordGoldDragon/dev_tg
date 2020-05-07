# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from see_rnn import get_weights, get_outputs, get_gradients
from see_rnn import features_hist
from .util._backend import K


def show_predictions_per_iteration(_labels_cache, _preds_cache):
    for batch_idx, label in enumerate(_labels_cache):
        preds_per_batch = len(_preds_cache[batch_idx])

        f, axes = plt.subplots(preds_per_batch + 1, 1,
                     gridspec_kw={'height_ratios': [2] + [1]*preds_per_batch})
        f.set_size_inches(14, 0.75)
        label_2d = np.atleast_2d(np.asarray(label).T[:, 0])
        axes[0].imshow(label_2d, cmap='bwr')
        axes[0].set_axis_off()
        axes[0].axis('tight')

        for pred_idx, y_preds in enumerate(_preds_cache[batch_idx]):
            preds_2d = np.atleast_2d(np.asarray(y_preds).T)
            axes[pred_idx + 1].imshow(preds_2d, cmap='bwr', vmin=0, vmax=1)
            axes[pred_idx + 1].set_axis_off()
            axes[pred_idx + 1].axis('tight')

        plt.show()


def show_predictions_distribution(_labels_cache, _preds_cache, pred_th):
    def _get_pred_colors(labels_f):
        labels_f = np.expand_dims(labels_f, -1)
        N = len(labels_f)
        red  = np.array([1, 0, 0]*N).reshape(N, 3)
        blue = np.array([0, 0, 1]*N).reshape(N, 3)
        return labels_f * red + (1 - labels_f) * blue

    def _make_alignment_array(labels_f, n_lines=10):
        N = len(labels_f)
        k = N / n_lines
        while not k.is_integer():
            n_lines -= 1
            k = N / n_lines
        return np.array(list(range(n_lines)) * int(k))

    def _plot(preds_f, pred_th, alignment_arr, colors):
        _, ax = plt.subplots(1, 1, figsize=(13, 4))
        ax.axvline(pred_th, color='black', linewidth=4)
        ax.scatter(preds_f, alignment_arr, c=colors)
        ax.set_yticks([])
        ax.set_xlim(-.02, 1.02)
        plt.show()

    preds_flat = np.asarray(_preds_cache).ravel()
    labels_flat = np.asarray(_labels_cache).ravel()
    colors = _get_pred_colors(labels_flat)
    alignment_arr = _make_alignment_array(labels_flat, n_lines=10)

    _plot(preds_flat, pred_th, alignment_arr, colors)


def _get_history_fig(cls, plot_configs=None, w=1, h=1):
    def _unpack_plot_kws(config):
        reserved_keys = ('metrics', 'x_ticks', 'vhlines',
                         'mark_best_cfg', 'ylims')
        metric_keys = list(config['metrics'].keys())  # 'train', 'val'
        values_per_key = sum(len(config['metrics'][x]) for x in metric_keys)

        plot_kws = []
        for i in range(values_per_key):
            plot_kws.append({key:config[key][i]
                             for key in config
                             if key not in reserved_keys})
        return plot_kws

    def _equalize_ticks_range(x_ticks, metrics):
        max_value = max([np.max(ticks) for ticks in x_ticks if len(ticks) > 0])

        for idx, ticks, metric in zip(range(len(x_ticks)), x_ticks, metrics):
            if (len(ticks) == 0) or (ticks[-1] < max_value):
                if len(metric) == 1:
                    x_ticks[idx] = [max_value]
                else:
                    x_ticks[idx] = list(np.linspace(1, max_value, len(metric)))

        assert all([ticks[-1] == max_value for ticks in x_ticks])
        assert all([len(t) == len(m)       for t, m in zip(x_ticks, metrics)])
        return x_ticks

    def _equalize_metric_names(config):
        metrics = config['metrics']

        if 'train' in metrics:
            for idx, name in enumerate(metrics['train']):
                metrics['train'][idx] = cls._alias_to_metric_name(name)
        if 'val'   in metrics:
            for idx, name in enumerate(metrics['val']):
                metrics['val'][idx]   = cls._alias_to_metric_name(name)
        return config

    def _unpack_vhlines(config):
        vhlines = {'v':[], 'h':[]}
        for vh in vhlines:
            vhline = config['vhlines'][vh]
            if isinstance(vhline, (float, int, list, tuple, np.ndarray)):
                vhlines[vh] = vhline
            elif 'val_hist_vlines' in vhline:
                vhlines[vh] = cls._val_hist_vlines or None
            elif   'hist_vlines' in vhline:
                vhlines[vh] = cls._hist_vlines     or None
            else:
                raise ValueError("unsupported `vhlines` in `plot_configs`:",
                                 vhline)
        return vhlines

    def _unpack_ticks_and_metrics(config):
        def _get_mark_best_idx(metrics, name, mark_best_cfg, val):
            expects_val = bool('val' in mark_best_cfg)
            expected_name = list(mark_best_cfg.values())[0]

            if not val and expects_val:
                return
            elif val and expects_val and name == expected_name:
                return len(metrics) - 1
            elif not val and not expects_val and name == expected_name:
                return len(metrics) - 1

        x_ticks, metrics = [], []
        mark_best_cfg = config.get('mark_best_cfg', None)
        mark_best_idx = None

        if 'train' in config['metrics']:
            for i, name in enumerate(config['metrics']['train']):
                metrics.append(cls.history[name])
                x_ticks.append(getattr(cls, config['x_ticks']['train'][i]))
                if mark_best_cfg is not None and mark_best_idx is None:
                    mark_best_idx = _get_mark_best_idx(metrics, name,
                                                       mark_best_cfg, val=False)
        if 'val' in config['metrics']:
            for i, name in enumerate(config['metrics']['val']):
                metrics.append(cls.val_history[name])
                x_ticks.append(getattr(cls, config['x_ticks']['val'][i]))
                if mark_best_cfg is not None and mark_best_idx is None:
                    mark_best_idx = _get_mark_best_idx(metrics, name,
                                                       mark_best_cfg, val=True)
        return x_ticks, metrics, mark_best_idx

    if plot_configs is None:
        plot_configs = cls.plot_configs

    fig, axes = plt.subplots(len(plot_configs), 1)
    axes = np.atleast_1d(axes)

    for config, axis in zip(plot_configs.values(), axes):
        config = _equalize_metric_names(config)
        x_ticks, metrics, mark_best_idx = _unpack_ticks_and_metrics(config)
        x_ticks = _equalize_ticks_range(x_ticks, metrics)

        plot_kws = _unpack_plot_kws(config)
        if config.get('vhlines', None) is not None:
            vhlines  = _unpack_vhlines(config)
        ylims = config.get('ylims', (0, 2))

        _plot_metrics(x_ticks, metrics, plot_kws, mark_best_idx,
                      axis=axis, vhlines=vhlines, ylims=ylims,
                      key_metric=cls.key_metric)

    subplot_scaler = .5 * len(axes)
    fig.set_size_inches(14 * w, 11 * h * subplot_scaler)
    plt.close(fig)
    return fig


def _plot_metrics(x_ticks, metrics, plot_kws, mark_best_idx=None, axis=None,
                  vhlines={'v':None, 'h':None}, ylims=(0, 2), key_metric='loss'):
    if axis is not None:
        ax = axis
    else:
        _, ax = plt.subplots()

    def _plot_vhlines(vhlines, ax):
        def non_iterable(x):
            return not isinstance(x, (list, tuple, np.ndarray))
        vlines, hlines = vhlines.get('v', None), vhlines.get('h', None)

        if vlines is not None:
            if non_iterable(vlines):
                vlines = [vlines]
            [ax.axvline(l, color='k', linewidth=2) for l in vlines if l]
        if hlines is not None:
            if non_iterable(hlines):
                hlines = [hlines]
            [ax.axhline(l, color='k', linewidth=2) for l in hlines if l]

    def _mark_best_metric(x_ticks, metrics, mark_best_idx, ax):
        metric = metrics[mark_best_idx]

        best_fn = np.min if key_metric=='loss' else np.max
        x_best_idx = np.where(metric == best_fn(metric))[0][0]
        x_best = x_ticks[mark_best_idx][x_best_idx]

        ax.plot(x_best, best_fn(metric), 'o', color=[.3, .95, .3],
                markersize=15, markeredgewidth=4, markerfacecolor='none')

    def _plot_main(x_ticks, metrics, plot_kws, ax):
        for ticks, metric, kws in zip(x_ticks, metrics, plot_kws):
            ax.plot(ticks, metric, **kws)

    _plot_main(x_ticks, metrics, plot_kws, ax)
    if vhlines is not None:
        _plot_vhlines(vhlines, ax)

    if mark_best_idx is not None:
        _mark_best_metric(x_ticks, metrics, mark_best_idx, ax)

    xmin = min([np.min(ticks) for ticks in x_ticks])
    xmax = max([np.max(ticks) for ticks in x_ticks])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(*ylims)


# TODO: multiclass
def comparative_histogram(model, layer_name, data, keep_borders=True,
                          bins=100, xlims=(0, 1), fontsize=14, vline=None,
                          w=1, h=1):
    outs = get_outputs(model, layer_name, data)

    _, axes = plt.subplots(2, 1, sharex=True, sharey=True,
                           figsize=(13 * w, 6 * h))
    for i, (ax, out) in enumerate(zip(axes.flat, outs)):
        ax.hist(np.asarray(out).ravel(), bins=bins)

        mode = "ON" if i == 0 else "OFF"
        ax.set_title("Train mode " + mode, weight='bold')
        if not keep_borders:
            ax.box(on=None)
        if vline:
            ax.axvline(vline, color='r', linewidth=2)
        ax.set_xlim(*xlims)
    plt.show()


def layer_hists(model, _id='*', mode='weights', input_data=None, labels=None,
                omit_names='bias', share_xy=(0, 0), configs=None, **kw):
    def _process_configs(configs, mode):
        defaults = {}
        if 'gradients' in mode or mode == 'outputs':
            defaults.update({'plot': dict(peaks_to_clip=2, annot_kw=None),
                             'subplot': dict(sharex=False, sharey=False)})
        if not configs:
            return defaults
        for k, v in defaults.items():
            if k not in configs:
                configs[k] = v
        return configs

    def _prevalidate(mode, input_data, labels):
        if features_hist is None:
            raise Exception("`weights_hists_cb` requires `see-rnn` "
                            "(!pip install see-rnn)")

        supported = ('weights', 'outputs', 'gradients',
                     'gradients:outputs', 'gradients:weights')
        if mode not in supported:
            raise ValueError(("unsupported `mode`: '{}'; supported are: {}"
                             ).format(mode, ', '.join(supported)))
        if 'gradients' in mode and (input_data is None or labels is None):
            raise ValueError("`input_data` or `labels` cannot be None for "
                             "'gradients'-based `mode`")
        if mode == 'outputs' and input_data is None:
            raise ValueError("`input_data` cannot be None for `mode='outputs'`")

    def _get_data(model, _id, mode, input_data, labels, omit_names, kw):
        if mode == 'weights':
            data = get_weights(model, _id, omit_names, as_dict=True)
        elif 'gradients' in mode:
            if mode in ('gradients', 'gradients:outputs'):
                data = get_gradients(model, _id, input_data, labels,
                                     mode='outputs', as_dict=True)
            else:
                data = get_gradients(model, _id, input_data, labels,
                                     mode='weights', as_dict=True)
        elif mode == 'outputs':
            data = get_outputs(model, _id, input_data, as_dict=True)

        data_flat = [x.ravel() for x in data.values()]
        return data_flat, list(data)

    configs = _process_configs(configs, mode)
    _prevalidate(mode, input_data, labels)
    data_flat, data_names = _get_data(model, _id, mode, input_data, labels,
                                      omit_names, kw)
    features_hist(data_flat, annotations=data_names, configs=configs,
                  share_xy=share_xy, **kw)


def viz_roc_auc(y_true, y_pred):
    def _compute_roc_auc(y_true, y_pred):
        i_x = [(i, x) for (i, x) in enumerate(y_pred)]
        i_xs = list(sorted(i_x, key=lambda x: x[1], reverse=True))
        idxs = [d[0] for d in i_xs]
        xs = y_pred[idxs]
        ys = y_true[idxs]

        p_inv = 1 / ys.sum()
        n_inv = 1 / (len(ys) - ys.sum())
        pts = np.zeros((len(ys) + 1, 2))

        for i, (x, y) in enumerate(zip(xs, ys)):
            inc = p_inv if y == 1 else n_inv
            if y == 1:
                pts[i + 1] = [pts[i][0], pts[i][1] + inc]
            else:
                pts[i + 1] = [pts[i][0] + inc, pts[i][1]]

        score = np.trapz(pts[:, 1], pts[:, 0])
        return pts, score

    def _plot(pts, score):
        kw = dict(fontsize=12, weight='bold')
        plt.scatter(pts[:, 0], pts[:, 1])

        plt.title("Receiver Operating Characteristic (AUC = %.3f)" % score, **kw)
        plt.xlabel("1 - specificity", **kw)
        plt.ylabel("sensitivity", **kw)
        plt.gcf().set_size_inches(6, 6)
        plt.show()

    # standardize
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    pts, score = _compute_roc_auc(y_true, y_pred)
    _plot(pts, score)
