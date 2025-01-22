from __future__ import annotations

import contextlib
import numbers
from pathlib import Path
import typing
import warnings

from .api.structs import Capture
from .api import util, xarray_ops

if typing.TYPE_CHECKING:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import numpy as np
    import xarray as xr
    import iqwaveform
else:
    xr = util.lazy_import('xarray')
    mpl = util.lazy_import('matplotlib')
    plt = util.lazy_import('matplotlib.pyplot')
    np = util.lazy_import('numpy')
    iqwaveform = util.lazy_import('iqwaveform')


class CapturePlotter:
    def __init__(
        self,
        interactive: bool = True,
        output_dir: str|Path = None,
        subplot_by_channel: bool = True,
        col_wrap=2,
        title_fmt="Channel {channel}",
        suptitle_fmt="{center_frequency}",
        filename_fmt="{name} {center_frequency}.svg"
    ):
        self.interactive: bool = interactive
        if subplot_by_channel:
            self.facet_col = "capture"
        else:
            self.facet_col = None
        self.col_wrap = col_wrap
        self.output_dir = output_dir
        self._suptitle_fmt = suptitle_fmt
        self._title_fmt = title_fmt
        self._filename_fmt = filename_fmt

    @contextlib.contextmanager
    def _plot_context(self, data, name: str, x: str = None, y: str = None, hue: str = None):
        if self.facet_col is None:
            fig, axs = plt.subplots()
        else:
            fig, axs = None, []

        warning_ctx = warnings.catch_warnings()
        warning_ctx.__enter__()
        warnings.filterwarnings('ignore', category=UserWarning, message=r'.*figure layout has changed.*')

        yield

        if fig is None:
            fig = plt.gcf()
            axs = fig.get_axes()
            fig.tight_layout()
        if not isinstance(axs, (list, tuple)):
            axs = [axs]

        if self._suptitle_fmt is not None:
            suptitle = label_by_coord(data, self._suptitle_fmt, title_case=True, name=name)[0]
            fig.suptitle(suptitle)

        if self._title_fmt is not None:
            titles = label_by_coord(data, self._title_fmt, title_case=True, name=name)
            if len(titles) > len(axs):
                raise ValueError(f'data has {len(titles)} captures but plotted {len(axs)} plots')
            for ax, title in zip(axs, titles):
                ax.set_title(title)

        if self.output_dir is not None:
            filename = set(label_by_coord(data, self._filename_fmt, name=name, title_case=True))
            if len(filename) > 1:
                raise ValueError('select a filename format that does not depend on channel index')
            path = Path(self.output_dir) / list(filename)[0]
            plt.savefig(path, dpi=300)

        if x is not None:
            for ax in axs:
                label_axis("x", data[x], ax=ax)

        if y is not None:
            label_axis("y", data[y])

        if hue is not None:
            label_legend(data, coord_name=hue)

        warning_ctx.__exit__(None, None, None)
        
        if not self.interactive:
            plt.close()

    def _line(
        self,
        data: xr.DataArray,
        name: str,
        sel: dict = {},
        *,
        x: str,
        hue: str | None = None,
        rasterized: bool = True,
        sharey: bool = True,
    ):
        kws = dict(x=x, hue=hue, rasterized=rasterized)

        if data.capture.size == 1:
            # iterate across the one capture
            seq = [data]
        elif self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            kws.update(col=self.facet_col, sharey=sharey, col_wrap=self.col_wrap)
        else:
            seq = data

        for sub in seq:
            with self._plot_context(sub, name=name, x=x, hue=hue):
                data.sel(sel).plot.line(**kws)

    def _heatmap(
        self,
        data: xr.DataArray,
        name: str,
        sel: dict = {},
        *,
        x: str,
        y: str,
        rasterized: bool = True,
        sharey: bool = True,
        transpose: bool = True,
        **kws
    ):
        kws.update(x=x, y=y, rasterized=rasterized)

        if data.capture.size == 1:
            # iterate across the one capture
            seq = [data]
        elif self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            kws.update(col=self.facet_col, sharey=sharey, col_wrap=self.col_wrap)
        else:
            seq = data

        for sub in seq:
            with self._plot_context(sub, name=name, x=x, y=y):
                spg = data.sel(sel)
                if transpose:
                    spg = spg.T
                spg.dropna(x).dropna(y).plot(**kws)

    def cellular_cyclic_autocorrelation(self, data: xr.Dataset, hue='link_direction', **sel):
        key = self.cellular_cyclic_autocorrelation.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x="cyclic_sample_lag",
            hue=hue,
        )

    def channel_power_histogram(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x="channel_power_bin",
            hue=hue,
        )

    def channel_power_time_series(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_time_series.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x="time_elapsed",
            hue=hue,
        )

    def persistence_spectrum(self, data: xr.Dataset, hue='persistence_statistic', **sel):
        key = self.persistence_spectrum.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x="baseband_frequency",
            hue=hue,
        )

    def spectrogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram.__name__
        self._heatmap(
            data[key].sel(sel),
            name=key,
            x="spectrogram_time",
            y="spectrogram_baseband_frequency",
        )

    def spectrogram_histogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x="spectrogram_power_bin",
            # hue=hue,
        )

    def cyclic_channel_power(self, data: xr.Dataset, **sel):
        data_across_facets = data.cyclic_channel_power.sel(**sel)
        with self._plot_context(data, name="cyclic_channel_power", x='cyclic_lag'):
            if self.facet_col is not None:
                facets = data[self.facet_col]
                fig, axs = plt.subplots(1, len(facets))
            else:
                facets = [None]
                fig = plt.gcf()
                axs = fig.get_axes()

            for facet, ax in zip(facets, axs):
                if facet is not None:
                    cyclic_power = data_across_facets.sel({self.facet_col: facet})
                else:
                    cyclic_power = data_across_facets

                plot_cyclic_channel_power(cyclic_power, ax=ax)


def capture_to_dicts(capture: xr.DataArray, title_case=False) -> dict[str]:
    if capture.ndim > 0:
        return [capture_to_dicts(c, title_case)[0] for c in capture]

    coords = capture.coords.to_dataset().to_dict("list")["coords"]
    d = {}
    for k, v in coords.items():
        if isinstance(v["data"], numbers.Number):
            d[k] = xarray_ops.describe_value(v["data"], v["attrs"])
        elif isinstance(v["data"], str):
            d[k] = v["data"].replace("_", " ")
            if title_case:
                d[k] = d[k].title()
        else:
            d[k] = v["data"]

    return [d]


def label_by_coord(data: xr.DataArray, fmt: str, *, title_case=True, **extra_fields):
    coords = capture_to_dicts(data.capture, title_case=title_case)
    return [fmt.format(**c, **extra_fields) for c in coords]


def summarize_metadata(
    source: 'xr.Dataset',
    capture_type: type[Capture],
    array: 'xr.DataArray ' = None,
    *,
    as_str: bool = False,
):
    meta = {
        k: source.attrs[k] for k in capture_type.__struct_fields__ if k in source.attrs
    }

    if array is not None:
        meta.update(
            {
                name: coord.item()
                for name, coord in array.coords.items()
                if coord.size == 1
            }
        )

    if as_str:
        return '\n'.join([f'{k}: {v}' for k, v in meta.items()])
    else:
        return meta


def plot_cyclic_channel_power(
    cyclic_channel_power: 'xr.DataArray',
    center_statistic='mean',
    bound_statistics=('min', 'max'),
    dB=True,
    ax=None,
    colors=None,
):
    if ax is None:
        _, ax = plt.subplots()

    time = cyclic_channel_power.cyclic_lag

    if 'capture' in cyclic_channel_power.dims:
        cyclic_channel_power = cyclic_channel_power.squeeze('capture')

    for i, detector in enumerate(cyclic_channel_power.power_detector.values):
        a = cyclic_channel_power.sel(power_detector=detector)

        if not dB:
            a = iqwaveform.dBtopow(a)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
        )

    for i, detector in enumerate(cyclic_channel_power.power_detector.values):
        a = cyclic_channel_power.sel(power_detector=detector)

        if not dB:
            a = iqwaveform.dBtopow(a)

        ax.fill_between(
            time,
            a.sel(cyclic_statistic=bound_statistics[0]),
            a.sel(cyclic_statistic=bound_statistics[1]),
            color=f'C{i}' if colors is None else colors[i],
            alpha=0.25,
            lw=0,
            rasterized=True,
        )

    label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    label_legend(cyclic_channel_power.power_detector, ax=ax)


def label_axis(
    which_axis: typing.Literal['x'] | typing.Literal['y'],
    data: typing.Union['xr.DataArray', 'xr.Dataset'],
    *,
    coord_name: typing.Optional['xr.Coordinates'] = None,
    tick_units=True,
    ax: typing.Optional['mpl.axes.Ax'] = None,
):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.

    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    if ax is None:
        ax = plt.gca()

    if which_axis == 'x':
        axis = ax.xaxis
    elif which_axis == 'y':
        axis = ax.yaxis

    if coord_name is None:
        # label = a.attrs.get('standard_name', None)
        units = data.attrs.get('units', None)
    else:
        # label = a[dimension].attrs.get('label', None)
        units = data[coord_name].attrs.get('units', None)

    # if label is not None:
    #     if units is not None and not tick_units:
    #         label = f'{label} ({units})'
    #     axis.set_label_text(label)
    if units is not None and tick_units:
        axis.set_major_formatter(mpl.ticker.EngFormatter(unit=units))
        axis.set_label_text(data.standard_name or data.name)
    elif units is not None:
        axis.set_label_text(f'{data.standard_name or data.name} ({units})')
    else:
        axis.set_label_text(data.standard_name or data.name)


def label_legend(
    data: typing.Union['xr.DataArray', 'xr.Dataset'],
    *,
    coord_name: str = None,
    tick_units=True,
    ax: typing.Optional['mpl.axes._axes.Axes'] = None,
):
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""

    if ax is None:
        ax = plt.gca()

    if coord_name is None:
        obj = data
    else:
        obj = data[coord_name]
    standard_name = obj.attrs.get('standard_name', None)
    units = obj.attrs.get('units', None)
    values = obj.values

    if standard_name is not None:
        if units is not None and not tick_units:
            standard_name = f'{standard_name} ({units})'
    if units is not None:
        # TODO: implement tick_units
        formatter = mpl.ticker.EngFormatter(unit=units)
        values = [formatter(v) for v in values]

    ax.legend(values, title=standard_name)


def label_selection(
    sel: typing.Union['xr.DataArray', 'xr.Dataset'],
    ax: typing.Optional['mpl.axes._axes.Axes'] = None,
    attrs=True,
):
    if ax is None:
        ax = plt.gca()
    coord_names = {}
    for name, coord in sel.coords.items():
        if name == 'capture' or name in sel.indexes or coord.values.size == 0:
            continue

        units = coord.attrs.get('units', None)

        label = coord.attrs.get('standard_name', coord.attrs.get('name', name))
        values = np.atleast_1d(coord.values)
        if units is not None:
            formatter = mpl.ticker.EngFormatter(unit=units)
            coord_names[label] = ', '.join([formatter(v) for v in values])
        else:
            coord_names[label] = ', '.join([str(v) for v in values])

    coord_title = ', '.join(f'{k}: {v}' for k, v in coord_names.items())

    if attrs:
        attr_title = ', '.join(
            [
                f'{k}: {v}'
                for k, v in sel.attrs.items()
                if k not in ('units', 'name', 'standard_name')
            ]
        )
    else:
        attr_title = ''

    if len(attr_title) > 0:
        attr_title = f'\nAnalysis: {attr_title}'
    ax.set_title(f'{coord_title}{attr_title}')
