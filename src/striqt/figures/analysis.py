from __future__ import annotations as __

import contextlib
import functools
import math
import numbers
import typing
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import ticker

import striqt.waveform as sw

from ..analysis.lib import dataarrays
from ..analysis.specs import Capture

_FORCE_UNIT_PREFIXES = {'center_frequency': 'M'}


class MissingDataError(AttributeError):
    pass


class FixedEngFormatter(ticker.EngFormatter):
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
            sep = ' '

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
            # new_text = f'{ex.args[0]} (encountered while plotting {func.__name__})'
            ex.args = (*ex.args, f'while plotting {func.__name__}')
            raise ex

    return wrapped


def _color_levels(data: xr.DataArray, step_size, plot_kws):
    if 'vmin' in plot_kws and plot_kws['vmin'] is not None:
        vmin = plot_kws['vmin']
    else:
        vmin = float(data.min())

    if 'vmax' in plot_kws and plot_kws['vmax'] is not None:
        vmax = plot_kws['vmax']
    else:
        vmax = float(data.max())

    vmax_edge = math.ceil(vmax / step_size) * step_size
    vmin_edge = math.floor(vmin / step_size) * step_size

    n = round((vmax_edge - vmin_edge) / step_size) + 1

    levels = np.linspace(vmin_edge, vmax_edge, n).tolist()

    return levels


def _count_facets(facet_col, data) -> int:
    if facet_col is None:
        return 1
    else:
        return data[facet_col].shape[0]


def _fix_axes(data: xr.DataArray, grid, x, y=None, xticklabelunits=True):
    axs = list(np.array(grid.axes).flatten())

    label_axis('x', data[x], fig=grid.fig, tick_units=xticklabelunits)

    if y is None:
        label_axis('y', data, fig=grid.fig, tick_units=False)
    else:
        label_axis('y', data[y], fig=grid.fig, tick_units=False)


class CapturePlotter:
    def __init__(
        self,
        style=None,
        interactive: bool = True,
        output_dir: str | Path = None,
        subplot_by_port: bool = True,
        col_wrap: int = 2,
        title_fmt='Port {port}',
        suptitle_fmt='{center_frequency}',
        filename_fmt='{name} {center_frequency}.svg',
        ignore_missing=False,
    ):
        self.interactive: bool = interactive

        if subplot_by_port:
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

    def process_setup(self):
        if not self.interactive:
            mpl.use('agg')

        if self._style is not None:
            plt.style.use(self._style)

        warning_ctx = warnings.catch_warnings()
        warning_ctx.__enter__()
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=r'.*figure layout has changed.*'
        )
        warnings.filterwarnings(
            'ignore', category=UserWarning, message='.*artists with labels.*'
        )

    @contextlib.contextmanager
    def _plot_context(
        self,
        data,
        name: str,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        xticklabelunits=True,
        meta: dict = {},
    ):
        if self.facet_col is None:
            facet_count = 1
        else:
            facet_count = data[self.facet_col].shape[0]

        if facet_count == 1:
            fig, axs = plt.subplots()
        else:
            fig, axs = None, []

        yield fig

        if fig is None:
            fig = plt.gcf()
            axs = fig.get_axes()
        else:
            fig.tight_layout()
        fig.set_size_inches(mpl.rcParams['figure.figsize'])

        if not isinstance(axs, (list, tuple)):
            axs = [axs]

        if self._suptitle_fmt is not None:
            left = fig.subplotpars.left
            right = fig.subplotpars.right
            center_x = (left + right) / 2.0

            suptitle = label_by_coord(
                data, self._suptitle_fmt, title_case=True, name=name, **meta
            )[0]
            fig.suptitle(suptitle, x=center_x, ha='center', va='bottom')

        if self._title_fmt is not None:
            titles = label_by_coord(
                data, self._title_fmt, title_case=True, name=name, **meta
            )
            if len(titles) > len(axs):
                raise ValueError(
                    f'data has {len(titles)} captures but plotted {len(axs)} plots'
                )
            for ax, title in zip(axs, titles):
                ax.set_title(title)

        label_axis('x', data[x], fig=fig, tick_units=xticklabelunits)

        if hue is not None and fig.legends:
            fig.legends[0].remove()
            label_legend(data, coord_name=hue, ax=axs[0])

        if len(axs) > facet_count:
            # assume heatmap
            clabel_ax = axs[-1]
            ylabel = clabel_ax.get_ylabel().replace('\n', ' ')
            clabel_ax.set_ylabel(ylabel, rotation=90)

        if self.output_dir is not None:
            filename = set(
                label_by_coord(
                    data, self._filename_fmt, name=name, title_case=True, **meta
                )
            )
            # if len(filename) > 1:
            #     raise ValueError(
            #         'select a filename format that does not depend on port index'
            #     )
            path = Path(self.output_dir) / list(filename)[0]
            fig.savefig(path, dpi=300)

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
        rasterized: bool = False,
        sharey: bool = True,
        xticklabelunits: bool = True,
        meta: dict = {},
        plot_kws: dict[str, typing.Any] = {},
    ):
        kws = dict(x=x, hue=hue, rasterized=rasterized)
        ctx_kws = dict(
            meta=meta, name=name, x=x, hue=hue, xticklabelunits=xticklabelunits
        )

        if self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            col_wrap = min(_count_facets(self.facet_col, data), self._col_wrap)
            kws.update(col=self.facet_col, sharey=sharey, col_wrap=col_wrap)
        else:
            seq = data

        fig = None
        for sub in seq:
            with self._plot_context(sub, **ctx_kws):
                kws['figsize'] = mpl.rcParams['figure.figsize']
                grid = data.sel(sel).plot.line(**kws, **plot_kws)
                _fix_axes(data=data, grid=grid, x=x, xticklabelunits=xticklabelunits)
                fig = grid.fig
                fig.tight_layout()

        return fig

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
        meta: dict = {},
        facet_as_row=False,
        cmap='cubehelix',
        **kws,
    ):
        from matplotlib import colors

        levels = _color_levels(data, 2, kws)
        cmap = plt.get_cmap(cmap, len(levels) - 1)
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N)
        kws.pop('vmin', None)
        kws.pop('vmax', None)

        kws.update(x=x, y=y, rasterized=rasterized, cmap=cmap, norm=norm)

        if self.facet_col is not None:
            # treat the sequence of multiple captures in one plot
            seq = [data]
            col_wrap = min(_count_facets(self.facet_col, data), self._col_wrap)

            if facet_as_row:
                kws.update(row=self.facet_col, sharey=sharey, col_wrap=col_wrap)
            else:
                kws.update(col=self.facet_col, sharey=sharey, col_wrap=col_wrap)
        else:
            seq = data

        for sub in seq:
            with self._plot_context(sub, name=name, x=x, y=y, meta=meta):
                spg = data.sel(sel)
                if transpose:
                    spg = spg.T
                kws['figsize'] = mpl.rcParams['figure.figsize']
                grid = spg.plot.pcolormesh(**kws)
                _fix_axes(data=data, grid=grid, x=x, y=y)

    def cellular_cyclic_autocorrelation(
        self, data: xr.Dataset, hue='link_direction', dB: bool = False, **sel
    ):
        key = self.cellular_cyclic_autocorrelation.__name__
        sub = data[key].sel(sel)

        if hue == 'link_direction':
            scs_peaks = sub.max(['capture', 'link_direction', 'cyclic_sample_lag'])
            iscs = int(scs_peaks.argmax())  # type: ignore
            sub = sub.isel(subcarrier_spacing=iscs)
        else:
            sub = sub.sel(link_direction='downlink')

        if dB:
            sub = sw.powtodB(sub)

        return self._line(
            sub, name=key, x='cyclic_sample_lag', hue=hue, meta=data.attrs
        )

    @_maybe_skip_missing
    def cellular_5g_pss_correlation(
        self, data: xr.Dataset, hue='cellular_ssb_beam_index', dB=True, **sel
    ):
        key = self.cellular_5g_pss_correlation.__name__

        Rpss = data[key].sel(sel).isel(cellular_ssb_start_time=0)

        power = sw.envtopow(Rpss)

        if hue == 'cellular_cell_id2':
            power = power.mean('cellular_ssb_beam_index', keep_attrs=True)
        elif hue == 'cellular_ssb_beam_index':
            power = power.mean('cellular_cell_id2', keep_attrs=True)
        else:
            raise KeyError('invalid hue')

        if dB:
            power = sw.powtodB(power)

        fig = self._line(
            power,
            name=key,
            x='cellular_ssb_lag',
            hue=hue,
            rasterized=False,
            meta=data.attrs,
            plot_kws={'lw': 1, 'add_legend': False},
        )

        if fig is not None:
            fig.legend(ncol=4)

    @_maybe_skip_missing
    def cellular_5g_ssb_spectrogram(self, data: xr.Dataset, **sel):
        key = self.cellular_5g_ssb_spectrogram.__name__
        return self._heatmap(
            data[key]
            .sel(sel)
            .isel(cellular_ssb_index=0)
            .dropna('cellular_ssb_baseband_frequency'),
            name=key,
            x='cellular_ssb_symbol_index',
            y='cellular_ssb_baseband_frequency',
            meta=data.attrs,
        )

    def channel_power_histogram(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='channel_power_bin',
            hue=hue,
            xticklabelunits=False,
            meta=data.attrs,
        )

    @_maybe_skip_missing
    def channel_power_time_series(self, data: xr.Dataset, hue='power_detector', **sel):
        key = self.channel_power_time_series.__name__
        return self._line(
            data[key].sel(sel), name=key, x='time_elapsed', hue=hue, meta=data.attrs
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
            meta=data.attrs,
        )

    @_maybe_skip_missing
    def spectrogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram.__name__

        if 'system_noise' in data.data_vars:
            noise_power = data.system_noise + sw.powtodB(
                data.spectrogram.attrs['noise_bandwidth']
            )
            vmin = float(noise_power.min() - 6)
        else:
            vmin = None

        return self._heatmap(
            data[key].sel(sel).dropna('spectrogram_baseband_frequency'),
            name=key,
            x='spectrogram_time',
            y='spectrogram_baseband_frequency',
            meta=data.attrs,
            vmin=vmin,
        )

    @_maybe_skip_missing
    def spectrogram_histogram(self, data: xr.Dataset, **sel):
        key = self.spectrogram_histogram.__name__
        return self._line(
            data[key].sel(sel),
            name=key,
            x='spectrogram_power_bin',
            xticklabelunits=False,
            meta=data.attrs,
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
            meta=data.attrs,
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
            meta=data.attrs,
            # hue=hue,
        )

    @_maybe_skip_missing
    def cyclic_channel_power(self, data: xr.Dataset, **sel):
        data_across_facets = data.cyclic_channel_power.sel(**sel)
        with self._plot_context(
            data, name='cyclic_channel_power', x='cyclic_lag', meta=data.attrs
        ):
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

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

        if not dB:
            a = sw.dBtopow(a)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
            **plot_kws,
        )

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

        if not dB:
            a = sw.dBtopow(a)

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
    which_axis: typing.Literal['x', 'y', 'colorbar'],
    ax_data: typing.Union[xr.DataArray, xr.Dataset],
    *,
    coord_name: typing.Optional['xr.Coordinates'] = None,
    tick_units=True,
    short=False,
    ax: typing.Optional['mpl.axes.Ax' | list['mpl.axes.Ax']] = None,
    fig: typing.Optional['mpl.figure.FigureBase'] = None,
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
        target_axs = [a.xaxis for a in axs if not hasattr(a, '_colorbar')]
    elif which_axis == 'y':
        target_axs = [a.yaxis for a in axs if not hasattr(a, '_colorbar')]
    elif which_axis == 'colorbar':
        target_axs = [a.yaxis for a in axs if hasattr(a, '_colorbar')]

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

        ax_finite_data = ax_data.data[np.isfinite(ax_data.data)]
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
            assert fig is not None
            left = fig.subplotpars.left
            right = fig.subplotpars.right
            center_x = (left + right) / 2.0
            fig.supxlabel(label_str, x=center_x)
        elif which_axis == 'y':
            assert fig is not None
            top = fig.subplotpars.top
            bot = fig.subplotpars.bottom
            center_y = (top + bot) / 2

            fig.supylabel(label_str, y=center_y)

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
) -> 'mpl.legends':
    """apply legend labeling based on label and unit metadata in the specified dimension of `a`"""

    if ax is None:
        ax = plt.gca()

    if coord_name is None:
        obj = data
    else:
        obj = data[coord_name]
    standard_name = obj.attrs.get('standard_name', None)
    units = obj.attrs.get('units', None)
    data = obj.data

    if standard_name is not None:
        if units is not None and not tick_units:
            standard_name = f'{standard_name} ({units})'
    if units is not None:
        # TODO: implement tick_units
        formatter = mpl.ticker.EngFormatter(unit=units)
        data = [formatter(v) for v in data]

    return ax.legend(data, title=standard_name)


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
        elif name in sel.indexes or coord.data.size == 0:
            continue

        units = coord.attrs.get('units', None)

        label = coord.attrs.get('standard_name', coord.attrs.get('name', name))
        values = np.atleast_1d(coord.data)
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
