from __future__ import annotations as __

import math
from pathlib import Path
import typing

import striqt.waveform as sw
import striqt.analysis as sa

from . import specs


if typing.TYPE_CHECKING:
    import matplotlib as mpl
    from matplotlib import colors
    from matplotlib import pyplot as plt
    import numpy as np
    import xarray as xr
    import xarray.plot
    import xarray.core.types

    class DataVariablePlotter(typing.Protocol):
        def __call__(
            self,
            plotter: CapturePlotter,
            data: xr.Dataset,
            *args,
            **kwargs,
        ) -> typing.Any: ...

        __name__: str

    _TVP = typing.TypeVar('_TVP', bound=DataVariablePlotter)

else:
    mpl = sw.util.lazy_import('matplotlib')
    plt = sw.util.lazy_import('matplotlib.pyplot')
    xr = sw.util.lazy_import('xarray')
    np = sw.util.lazy_import('numpy')


class _HeatmapKwArgs(typing.TypedDict):
    norm: 'colors.BoundaryNorm | None'
    cmap: 'colors.Colormap | str'


class _BasePlotKws(typing.TypedDict):
    x: str
    y: typing.NotRequired[str | None]
    hue: typing.NotRequired[str | None]


class _PlotKwArgs(_BasePlotKws):
    col: str
    col_wrap: typing.NotRequired[int | None]
    row: typing.NotRequired[str | None]
    figsize: typing.NotRequired[list[float]]


def quantize_heatmap_kws(
    data,
    cmap: str = 'cubehelix',
    vmin: float | None = None,
    vmax: float | None = None,
    vstep: float | None = None,
) -> _HeatmapKwArgs:
    from matplotlib import colors

    if vstep is None:
        norm = None
        _cmap = cmap
    else:
        levels = _color_levels(data, vmin, vmax, vstep)
        _cmap = plt.get_cmap(cmap, len(levels) - 1)
        norm = colors.BoundaryNorm(levels, ncolors=_cmap.N)

    return _HeatmapKwArgs(norm=norm, cmap=_cmap)


def get_system_noise(data: xr.Dataset, var_name: str, margin=None):
    da = data[var_name]
    if 'system_noise' not in data:
        return None

    if 'noise_bandwidth' in da.attrs:
        B = da.attrs['noise_bandwidth']
    elif da.attrs.get('units', None) != 'dBm':
        return None
    elif 'analysis_bandwidth' in data.coords:
        B = data.coords['analysis_bandwidth']
    else:
        return None

    noise = data.system_noise + sw.powtodB(B)

    if margin is None:
        return noise
    else:
        return float(noise.min() - margin)


data_var_plotters: dict[str, DataVariablePlotter] = {}


def _register_data_var_plot(func: '_TVP') -> '_TVP':
    name = func.__name__
    func = sa.util.stopwatch(name, 'analysis', logger_level=sa.util.WARNING)(func)
    data_var_plotters[name] = func
    return func


def _select_dpi(grid, x_data, y_data, min_=0):
    dpi = min_
    scale_inv = grid.fig.dpi_scale_trans.inverted()
    for ax in grid.fig.axes:
        if grid.cbar is not None and ax is grid.cbar.ax:
            continue
        bbox = ax.get_window_extent().transformed(scale_inv)
        width, height = bbox.width, bbox.height
        dpi = max(dpi, x_data.size / width, y_data.size / height)

    return dpi


class CapturePlotter:
    opts: specs.SharedPlotOptions

    def __init__(
        self,
        opts: specs.SharedPlotOptions,
        output_dir: Path | None,
        *,
        interactive: bool = False,
    ):
        self.opts = opts
        self.output_dir = output_dir
        self.interactive = interactive

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def kwargs(self, **base: typing.Unpack[_BasePlotKws]) -> _PlotKwArgs:
        if self.opts.col is None and self.opts.row is None:
            col = '_view'
        else:
            col = self.opts.col

        return {
            **base,
            'figsize': mpl.rcParams['figure.figsize'],
            'col_wrap': self.opts.col_wrap,
            'row': self.opts.row,
            'col': col,  # type: ignore
        }

    def finish(
        self,
        grid: 'xarray.plot.FacetGrid',
        coords: _PlotKwArgs,
        xticklabelunits: bool | typing.Literal['auto'] = 'auto',
        rasterized=False,
    ):
        from . import labels

        data = grid.data
        x = coords['x']
        y = coords.get('y', None)

        if self.opts.suptitle_fmt is not None:
            suptitle = labels.label_by_coord(
                data,
                self.opts.suptitle_fmt,
                title_case=True,
                name=data.name,
                meta=data.attrs,
            )[0]
            grid.fig.suptitle(
                suptitle, x=labels._get_fig_center_x(grid.fig), ha='center', va='bottom'
            )

        # axes
        labels.label_axis('x', data[x], fig=grid.fig, tick_units=xticklabelunits)
        if y is None:
            labels.label_axis('y', data, fig=grid.fig, tick_units=False)
        else:
            labels.label_axis('y', data[y], fig=grid.fig, tick_units=False)

        # row/col titles
        grid.set_titles('{value}')
        if self.opts.col is None and self.opts.col_label_format is not None:
            for ax in grid.fig.axes:
                if grid.cbar is None or ax is not grid.cbar.ax:
                    ax.title.set_visible(False)
        else:
            col_text = labels.label_by_coord(
                data,
                self.opts.col_label_format,
                title_case=True,
                name=data.name,
                **data.attrs,
            )
            for s, ax in zip(col_text, grid.axs.flat):
                ax.set_title(s)
        if self.opts.row is not None and self.opts.row_label_format is not None:
            row_text = labels.label_by_coord(
                data,
                self.opts.row_label_format,
                title_case=True,
                name=data.name,
                **data.attrs,
            )
            for s, label in zip(row_text, grid.row_labels):
                assert label is not None
                label.set_text(s)

        # rasterization
        if grid.cbar is not None and grid.cbar.solids is not None:
            grid.cbar.solids.set_rasterized(rasterized)
        dpi = _select_dpi(grid, data[x], data if y is None else data[y], min_=100)
        grid.fig.set_dpi(dpi)

        if grid.cbar is not None:
            ylabel = grid.cbar.ax.get_ylabel().replace('\n', ' ')
            grid.cbar.ax.set_ylabel(ylabel, rotation=90)

        if self.output_dir is not None:
            filename = labels.label_by_coord(
                data, self.opts.filename_fmt, name=data.name, **data.attrs
            )
            path = Path(self.output_dir) / filename[0]
            grid.fig.savefig(path, dpi=dpi)

        if not self.interactive:
            plt.close('all')


@_register_data_var_plot
def cellular_cyclic_autocorrelation(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue: str = 'link_direction',
    dB: bool = False,
    **sel,
):
    sub = data.cellular_cyclic_autocorrelation.sel(sel).pipe(
        _coerce_griddable_data, plotter
    )

    if hue == 'link_direction':
        scs_peaks = sub.max(['capture', 'link_direction', 'cyclic_sample_lag'])
        iscs = int(scs_peaks.argmax())  # type: ignore
        sub = sub.isel(subcarrier_spacing=iscs)
    elif hue == 'subcarrier_spacing':
        sub = sub.sel(link_direction='downlink')
    else:
        raise TypeError('invalid hue coordinate')

    if dB:
        sub = sw.powtodB(sub)

    coords = plotter.kwargs(x='cyclic_sample_lag', hue=hue)
    grid = sub.plot.line(**coords)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def cellular_5g_pss_correlation(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='cellular_ssb_beam_index',
    dB=True,
    **sel,
):
    R = data.cellular_5g_pss_correlation.sel(sel).isel(cellular_ssb_start_time=0)
    pow = sw.envtopow(R)

    if hue == 'cellular_cell_id2':
        pow = pow.mean('cellular_ssb_beam_index', keep_attrs=True)
    elif hue == 'cellular_ssb_beam_index':
        pow = pow.mean('cellular_cell_id2', keep_attrs=True)
    else:
        raise KeyError('invalid hue coordinate')

    if dB:
        pow = sw.powtodB(pow)

    coords = plotter.kwargs(x='cellular_ssb_lag', hue=hue)
    grid = pow.plot.line(**coords)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def cellular_5g_ssb_spectrogram(plotter: CapturePlotter, data: xr.Dataset, **sel):
    key = cellular_5g_ssb_spectrogram.__name__
    sub = data[key].sel(sel).isel(cellular_ssb_index=0)
    coords = plotter.kwargs(
        x='cellular_ssb_symbol_index', y='cellular_ssb_baseband_frequency'
    )
    vmin = get_system_noise(data, key, 6)
    colors = quantize_heatmap_kws(sub, vmin=vmin, vstep=2)
    grid = sub.plot.imshow(**coords, **colors, rasterized=True, interpolation='nearest')
    for ax in grid.axs.flat:
        ax.grid(False)
    plotter.finish(grid, coords, rasterized=True)


@_register_data_var_plot
def cellular_resource_power_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='link_direction',
    noise_line=True,
    yscale: 'xarray.core.types.ScaleOptions' = 'linear',
    **sel,
):
    key = cellular_resource_power_histogram.__name__
    coords = plotter.kwargs(x='cellular_resource_power_bin', hue=hue)
    sub = _select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(yscale=yscale, **coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def channel_power_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='power_detector',
    noise_line=True,
    **sel,
):
    key = channel_power_histogram.__name__
    coords = plotter.kwargs(x='channel_power_bin', hue=hue)
    sub = _select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(**coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def channel_power_time_series(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='power_detector',
    noise_line=True,
    **sel,
):
    key = channel_power_time_series.__name__
    coords = plotter.kwargs(x='time_elapsed', hue=hue)
    grid = data[key].sel(sel).plot.line(**coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=True)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def cyclic_channel_power(
    plotter: CapturePlotter, data: xr.Dataset, noise_line=True, **sel
):
    from xarray.plot import FacetGrid

    key = cyclic_channel_power.__name__
    sub = data[key].sel(**sel)
    grid = FacetGrid(sub, **plotter.kwargs(), sharey=True)  # type: ignore
    coords = plotter.kwargs(x='cyclic_lag')

    for col, ax in zip(data[coords['col']], grid.axs.flat):
        plot_cyclic_channel_power(sub.sel({coords['col']: col}), ax=ax)

    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=True)

    return plotter.finish(grid, coords)


@_register_data_var_plot
def power_spectral_density(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='time_statistic',
    noise_line=True,
    **sel,
):
    key = power_spectral_density.__name__

    if (
        key not in data.data_vars
        and 'persistence_spectrum' in data.data_vars
        and 'persistence_statistics' in data
        and hue == 'time_statistic'
    ):
        # legacy
        hue = 'persistence_statistics'
        key = 'persistence_spectrum'
    coords = plotter.kwargs(x='baseband_frequency')
    grid = data[key].sel(sel).plot.line(**coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=True)

    return plotter.finish(grid, coords)


@_register_data_var_plot
def spectrogram(plotter: CapturePlotter, data: xr.Dataset, **sel):
    from . import ticker

    key = spectrogram.__name__
    sub = data[key].sel(sel).dropna('spectrogram_baseband_frequency')
    coords = plotter.kwargs(x='spectrogram_time', y='spectrogram_baseband_frequency')
    vmin = get_system_noise(data, key, 6)
    colors = quantize_heatmap_kws(sub, vmin=vmin, vstep=2)
    grid = sub.plot.imshow(**coords, **colors, rasterized=True, interpolation='nearest')
    for ax in grid.axs.flat:
        ax.grid(False)
    plotter.finish(grid, coords, rasterized=True)


@_register_data_var_plot
def spectrogram_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    yscale: 'xarray.core.types.ScaleOptions' = 'linear',
    noise_line=True,
    hue: str | None = None,
    **sel,
):
    key = spectrogram_histogram.__name__
    coords = plotter.kwargs(x='spectrogram_power_bin', hue=hue)
    sub = _select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(yscale=yscale, **coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


@_register_data_var_plot
def spectrogram_ratio_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    yscale: 'xarray.core.types.ScaleOptions' = 'linear',
    noise_line=True,
    hue: str | None = None,
    **sel,
):
    key = spectrogram_ratio_histogram.__name__
    coords = plotter.kwargs(x='spectrogram_ratio_power_bin', hue=hue)
    sub = _select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(yscale=yscale, **coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


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
    from . import labels

    if ax is None:
        _, ax = plt.subplots()

    time = cyclic_channel_power.cyclic_lag

    fill_kws = {}
    if steps:
        plot_kws = plot_kws | {'drawstyle': 'steps-post'}
        fill_kws['step'] = 'post'

    if sa.dataarrays.CAPTURE_DIM in cyclic_channel_power.dims:
        cyclic_channel_power = cyclic_channel_power.squeeze(sa.dataarrays.CAPTURE_DIM)

    if not dB:
        cyclic_channel_power = sw.dBtopow(cyclic_channel_power)

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

        ax.plot(
            time,
            (a.sel(cyclic_statistic=center_statistic)),
            color=f'C{i}' if colors is None else colors[i],
            **plot_kws,
        )

    for i, detector in enumerate(cyclic_channel_power.power_detector.data):
        a = cyclic_channel_power.sel(power_detector=detector)

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

    labels.label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    labels.label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    labels.label_legend(cyclic_channel_power.power_detector, ax=ax)


def _color_levels(data: xr.DataArray, vmin, vmax, vstep):
    if vmin is None:
        vmin = float(data.min())

    if vmax is None:
        vmax = float(data.max())

    vmax_edge = math.ceil(vmax / vstep) * vstep
    vmin_edge = math.floor(vmin / vstep) * vstep

    n = round((vmax_edge - vmin_edge) / vstep) + 1

    levels = np.linspace(vmin_edge, vmax_edge, n).tolist()

    return levels


def _plot_noise_line(
    data: xr.Dataset, var_name: str, grid: 'xarray.plot.FacetGrid', horizontal=True
):
    noise = get_system_noise(data, var_name)
    if noise is None:
        return
    assert not isinstance(noise, (int, float))
    for pow, ax in zip(noise.values.flat, grid.axs.flat):
        if horizontal:
            ax.axhline(pow, color='k', linestyle=':')
        else:
            ax.axvline(pow, color='k', linestyle=':')


def _coerce_griddable_data(data: xr.DataArray, plotter: CapturePlotter) -> xr.DataArray:
    opts = plotter.opts
    if opts.col is None and opts.row is None:
        return data.extend_dims(_view=[''])
    else:
        return data


def _select_histogram_bins(data, var_name, bin_name, pad_low=12, pad_hi=3):
    xmin = get_system_noise(data, bin_name, margin=pad_low)
    sub = data[var_name]
    xmax = sub.cumsum(bin_name).idxmax(bin_name).max() + pad_hi
    return sub.sel({bin_name: slice(xmin, xmax)})
