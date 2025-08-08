from __future__ import annotations

import contextlib
import functools
import math
import numbers
from pathlib import Path
import typing
import warnings

from .lib.specs import Capture
from .lib import dataarrays

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import iqwaveform

# avoid lazy loading, since this module isn't imported with striqt.analysis
# and lazy loading seems to lead to problems with matplotlib in some cases
import iqwaveform.figures  # noqa: F401


_FORCE_UNIT_PREFIXES = {'center_frequency': 'M'}


class MissingDataError(AttributeError):
    pass


class FixedEngFormatter(mpl.ticker.EngFormatter):
    """Behave as mpl.ticker.EngFormatter, but also support an
    invariant the unit suffix across the entire axis"""

    def __init__(
        self,
        unit='',
        unitInTick=True,
        places=None,
        sep=None,
        *,
        usetex=None,
        useMathText=None,
        **kws,
    ):
        self.unitInTick = unitInTick

        if sep is not None:
            pass
        if unit is None or (len(unit) == 1 and not unit.isalnum()):
            sep = ''
        else:
            sep = ' '

        super().__init__(
            unit,
            places,
            sep,
            usetex=usetex,
            useMathText=useMathText,
            **kws,
        )

    def format_data(self, value):
        sign = 1
        fmt = 'g' if self.places is None else f'.{self.places:d}f'

        if value < 0:
            sign = -1
            value = -value

        elif value == float('inf'):
            return '∞'

        elif value == float('-inf'):
            return '-∞'

        if value != 0:
            pow10 = int(math.floor(math.log10(value) / 3) * 3)
        else:
            pow10 = 0
            # Force value to zero, to avoid inconsistencies like
            # format_eng(-0) = "0" and format_eng(0.0) = "0"
            # but format_eng(-0.0) = "-0.0"
            value = 0.0

        if self.unitInTick:
            pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))
        else:
            pow10 = self.orderOfMagnitude

        mant = sign * value / (10.0**pow10)
        # Taking care of the cases like 999.9..., which may be rounded to 1000
        # instead of 1 k.  Beware of the corner case of values that are beyond
        # the range of SI prefixes (i.e. > 'Y').
        if (
            abs(float(format(mant, fmt))) >= 1000
            and pow10 < max(self.ENG_PREFIXES)
            and self.unitInTick
        ):
            mant /= 1000
            pow10 += 3

        unit_prefix = self.ENG_PREFIXES[int(pow10)]
        if self.unitInTick and (self.unit or unit_prefix):
            suffix = f'{self.sep}{unit_prefix}{self.unit}'
        else:
            suffix = ''
        if self._usetex or self._useMathText:
            return f'${mant:{fmt}}${suffix}'
        else:
            return f'{mant:{fmt}}{suffix}'

    def get_axis_unit_suffix(self, vmin, vmax):
        if self.unitInTick:
            return ''

        orderOfMagnitude = math.floor(math.log(vmax - vmin, 1000)) * 3
        unit_prefix = self.ENG_PREFIXES[int(orderOfMagnitude)]
        return f'{self.sep}({unit_prefix}{self.unit})'


def _maybe_skip_missing(func):
    """maybe skip runs on CapturePlotter methods that refer to missing data"""

    @functools.wraps(func)
    def wrapped(obj: CapturePlotter, data, *args, **kws):
        if obj._ignore_missing and func.__name__ not in data.data_vars:
            return

        try:
            func(obj, data, *args, **kws)
        except BaseException as ex:
            new_text = f'while plotting {func.__name__}, {ex.args[0]}'
            ex.args = (new_text,) + ex.args[1:]
            raise ex

    return wrapped


class CapturePlotter:
    def __init__(
        self,
        style=None,
        interactive: bool = True,
        output_dir: str | Path = None,
        subplot_by_channel: bool = True,
        col_wrap=2,
        title_fmt='Channel {channel}',
        suptitle_fmt='{center_frequency}',
        filename_fmt='{name} {center_frequency}.svg',
        ignore_missing=False,
    ):
        self.interactive: bool = interactive
        if subplot_by_channel:
            self.facet_col = dataarrays.CAPTURE_DIM
        else:
            self.facet_col = None

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        self._col_wrap = col_wrap
        self.output_dir = output_dir
        self._suptitle_fmt = suptitle_fmt
        self._title_fmt = title_fmt
        self._filename_fmt = filename_fmt
        self._ignore_missing = ignore_missing
        self._style = style

    @contextlib.contextmanager
    def _plot_context(
        self,
        data,
        name: str,
        x: str = None,
        y: str = None,
        hue: str = None,
        xticklabelunits=True,
    ):
        if self._style is not None:
            plt.style.use(self._style)

        if self.facet_col is None:
            fig, axs = plt.subplots()
            facet_count = 1
        else:
            fig, axs = None, []
            facet_count = data[self.facet_col].shape[0]

        warning_ctx = warnings.catch_warnings()
        warning_ctx.__enter__()
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=r'.*figure layout has changed.*'
        )

        yield

        if fig is None:
            fig = plt.gcf()
            axs = fig.get_axes()
        else:
            fig.tight_layout()

        if not isinstance(axs, (list, tuple)):
            axs = [axs]

        if self._suptitle_fmt is not None:
            suptitle = label_by_coord(
                data, self._suptitle_fmt, title_case=True, name=name
            )[0]
            fig.suptitle(suptitle)

        if self._title_fmt is not None:
            titles = label_by_coord(data, self._title_fmt, title_case=True, name=name)
            if len(titles) > len(axs):
                raise ValueError(
                    f'data has {len(titles)} captures but plotted {len(axs)} plots'
                )
            for ax, title in zip(axs, titles):
                ax.set_title(title)

        if x is not None:
            for ax in axs[:facet_count]:
                label_axis('x', data[x], ax=ax, tick_units=xticklabelunits)

        if y is not None:
            label_axis('y', data[y], ax=axs[0], tick_units=False)

        if hue is not None:
            fig.legends[0].remove()
            label_legend(data, coord_name=hue, ax=axs[0])

        if len(axs) > facet_count:
            # assume heatmap
            clabel_ax = axs[-1]
            ylabel = clabel_ax.get_ylabel().replace('\n', '\N{THIN SPACE}')
            clabel_ax.set_ylabel(ylabel, rotation=90)

        warning_ctx.__exit__(None, None, None)

        if self.output_dir is not None:
            filename = set(
                label_by_coord(data, self._filename_fmt, name=name, title_case=True)
            )
            if len(filename) > 1:
                raise ValueError(
                    'select a filename format that does not depend on channel index'
                )
            path = Path(self.output_dir) / list(filename)[0]
            plt.savefig(path, dpi=300)

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
        xticklabelunits: bool = True,
    ):
        kws = dict(x=x, hue=hue, rasterized=rasterized)
        ctx_kws = dict(name=name, x=x, hue=hue, xticklabelunits=xticklabelunits)

        if self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            kws.update(col=self.facet_col, sharey=sharey, col_wrap=self._col_wrap)
        else:
            seq = data

        for sub in seq:
            with self._plot_context(sub, **ctx_kws):
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
        **kws,
    ):
        kws.update(x=x, y=y, rasterized=rasterized)

        if self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            kws.update(col=self.facet_col, sharey=sharey, col_wrap=self._col_wrap)
        else:
            seq = data

        for sub in seq:
            with self._plot_context(sub, name=name, x=x, y=y):
                spg = data.sel(sel)
                if transpose:
                    spg = spg.T
                spg.plot.pcolormesh(**kws)

    def cellular_cyclic_autocorrelation(
        self, data: xr.Dataset, hue='link_direction', **sel
    ):
        key = self.cellular_cyclic_autocorrelation.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='cyclic_sample_lag',
            hue=hue,
        )

    def channel_power_histogram(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='channel_power_bin',
            hue=hue,
            xticklabelunits=False,
        )

    @_maybe_skip_missing
    def channel_power_time_series(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_time_series.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='time_elapsed',
            hue=hue,
        )

    @_maybe_skip_missing
    def power_spectral_density(self, data: xr.Dataset, hue='time_statistic', **sel):
        key = self.power_spectral_density.__name__

        if (
            key not in data.data_vars
            and 'persistence_spectrum' in data.data_vars
            and 'persistence_statistics' in data
            and hue == 'time_statistic'
        ):
            # legacy

            hue = 'persistence_statistics'
            key = 'persistence_spectrum'

        return self._line(
            data[key].sel(sel),
            name=key,
            x='baseband_frequency',
            hue=hue,
        )

    @_maybe_skip_missing
    def spectrogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram.__name__
        self._heatmap(
            data[key].sel(sel).dropna('spectrogram_baseband_frequency'),
            name=key,
            x='spectrogram_time',
            y='spectrogram_baseband_frequency',
        )

    @_maybe_skip_missing
    def spectrogram_histogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='spectrogram_power_bin',
            xticklabelunits=False,
            # hue=hue,
        )

    @_maybe_skip_missing
    def cellular_resource_power_histogram(self, data: xr.Dataset, **sel):
        key = self.cellular_resource_power_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='cellular_resource_power_bin',
            xticklabelunits=False,
            # hue=hue,
        )

    @_maybe_skip_missing
    def spectrogram_ratio_histogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram_ratio_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='spectrogram_ratio_power_bin',
            xticklabelunits=False,
            # hue=hue,
        )

    @_maybe_skip_missing
    def cyclic_channel_power(self, data: xr.Dataset, **sel):
        data_across_facets = data.cyclic_channel_power.sel(**sel)
        with self._plot_context(data, name='cyclic_channel_power', x='cyclic_lag'):
            if self.facet_col is not None:
                facets = data[self.facet_col]
                fig, (axs,) = plt.subplots(1, len(facets), squeeze=False, sharey=True)
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

    coords = capture.coords.to_dataset().to_dict('list')['coords']
    d = {}
    for k, v in coords.items():
        if isinstance(v['data'], numbers.Number):
            prefix = _FORCE_UNIT_PREFIXES.get(k, None)
            d[k] = dataarrays.describe_value(v['data'], v['attrs'], unit_prefix=prefix)
        elif isinstance(v['data'], str):
            d[k] = v['data'].replace('_', ' ')
            if title_case:
                d[k] = d[k].title()
        else:
            d[k] = v['data']

    return [d]


def label_by_coord(data: xr.DataArray, fmt: str, *, title_case=True, **extra_fields):
    coords = capture_to_dicts(data.capture, title_case=title_case)
    return [fmt.format(**c, **extra_fields) for c in coords]


def summarize_metadata(
    source: xr.Dataset,
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
    cyclic_channel_power: xr.DataArray,
    center_statistic='mean',
    bound_statistics=('min', 'max'),
    dB=True,
    ax=None,
    colors=None,
    steps=True,
    plot_kws={},
):
    if ax is None:
        _, ax = plt.subplots()

    time = cyclic_channel_power.cyclic_lag

    fill_kws = {}
    if steps:
        plot_kws = plot_kws | {'drawstyle': 'steps-post'}
        fill_kws['step'] = 'post'

    if dataarrays.CAPTURE_DIM in cyclic_channel_power.dims:
        cyclic_channel_power = cyclic_channel_power.squeeze(dataarrays.CAPTURE_DIM)

    for i, detector in enumerate(cyclic_channel_power.power_detector.values):
        a = cyclic_channel_power.sel(power_detector=detector)

        if not dB:
            a = iqwaveform.dBtopow(a)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
            **plot_kws,
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
            **fill_kws,
        )

    label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    label_legend(cyclic_channel_power.power_detector, ax=ax)


def label_axis(
    which_axis: typing.Literal['x'] | typing.Literal['y'] | typing.Literal['colorbar'],
    ax_data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: typing.Optional['xr.Coordinates'] = None,
    tick_units=True,
    short=False,
    ax: typing.Optional['mpl.axes.Ax'|list['mpl.axes.Ax']] = None,
    fig: typing.Optional['mpl.figure.FigureBase'] = None
):
    """apply axis labeling based on label and unit metadata in the specified dimension of `a`.

    If dimension is None, then labeling is applied from metadata in a.attrs
    """

    if which_axis not in ('x', 'y', 'colorbar'):
        raise ValueError("which_axis must be one of 'x', 'y', 'colorbar'")

    if fig is None:
        fig = plt.gcf()
        do_suplabel = False
    else:
        do_suplabel = False if which_axis == 'colorbar' else True

    if ax is None:
        if which_axis == 'colorbar':
            colorbars = [ax for ax in fig.axes if 'colorbar' in repr(ax)]
            if len(colorbars) == 0:
                raise ValueError('no colorbars found')
            else:
                ax = colorbars[0]
        else:
            if do_suplabel:
                ax = fig.axes
            else:
                ax = fig.gca()

    if hasattr(ax, '__len__') or hasattr(ax, '__iter__'):
        axs = list(ax)
    else:
        axs = [ax]

    if which_axis == 'x':
        target_axs = [a.xaxis for a in axs]
    elif which_axis in ('y', 'colorbar'):
        target_axs = [a.yaxis for a in axs]

    if coord_name is None:
        # label = a.attrs.get('standard_name', None)
        units = ax_data.attrs.get('units', None)
    else:
        # label = a[dimension].attrs.get('label', None)
        units = ax_data[coord_name].attrs.get('units', None)

    standard_name = ax_data.attrs.get('standard_name', None) or ax_data.name
    long_name = ax_data.attrs.get('long_name', None) or standard_name

    if short:
        desc_text = standard_name
    else:
        desc_text = long_name

    if units is not None:
        formatter = FixedEngFormatter(unit=units, unitInTick=tick_units)

        for target in target_axs:
            target.set_major_formatter(formatter)
        ax_finite_data = ax_data.values[np.isfinite(ax_data.values)]

        if len(ax_finite_data) > 0:
            unit_suffix = formatter.get_axis_unit_suffix(
                ax_finite_data.min(), ax_finite_data.max()
            )
            label_str = f'{desc_text}{unit_suffix}'
        else:
            label_str = None
    else:
        label_str = desc_text

    if do_suplabel:
        if which_axis == 'x':
            fig.supxlabel(label_str)
        elif which_axis == 'y':
            fig.supylabel(label_str)

        for target in target_axs:
            target.label.set_visible(False)
    else:
        for target in target_axs:
            target.set_label_text(label_str)


def label_legend(
    data: typing.Union[xr.DataArray, xr.Dataset],
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
    sel: typing.Union[xr.DataArray, xr.Dataset],
    ax: typing.Optional['mpl.axes._axes.Axes'] = None,
    attrs=True,
):
    if ax is None:
        ax = plt.gca()
    coord_names = {}
    for name, coord in sel.coords.items():
        if name == dataarrays.CAPTURE_DIM:
            continue
        elif name in sel.indexes or coord.values.size == 0:
            continue

        units = coord.attrs.get('units', None)

        label = coord.attrs.get('standard_name', coord.attrs.get('name', name))
        values = np.atleast_1d(coord.values)
        if units is not None:
            formatter = FixedEngFormatter(unit=units)
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
