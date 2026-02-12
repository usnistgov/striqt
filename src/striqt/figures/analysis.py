from __future__ import annotations as __

from pathlib import Path
import typing
import warnings

import striqt.waveform as sw
import striqt.analysis as sa

import matplotlib as mpl
from matplotlib import pyplot as plt
import msgspec
import xarray as xr

from . import util

if typing.TYPE_CHECKING:
    from typing_extensions import ParamSpec

    _P = ParamSpec('_P')
    _R = typing.TypeVar('_R', covariant=True)
    import xarray.plot
    import xarray.core.types

    class PlotCallerProtocol(typing.Protocol[_P, _R]):
        def __call__(
            self,
            data: xr.DataArray | xr.Dataset,
            name: str,
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _R: ...

        __name__: str

    class InitKwArgsCallable(typing.Protocol[_P, _R]):
        def __call__(
            self,
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _R: ...

        __name__: str

    class KwArgsOnly(typing.Protocol[_P, _R]):
        def __call__(
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _R: ...

        __name__: str


def spec_as_init_kwargs(
    spec_cls: typing.Callable[_P, typing.Any],
) -> typing.Callable[[typing.Callable[..., _R]], 'InitKwArgsCallable[_P, _R]']:
    """fill in type hints for the analysis parameters"""
    return lambda f: f  # type: ignore


class CapturePlotterOptions(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    unstack: list[str]
    style: str | None = None
    interactive: bool = True
    output_dir: Path | None = None
    col: str | None = 'port'
    row: str | None = None
    col_label_format: str | None = 'Port {port}'
    row_label_format: str | None = None
    col_wrap: int = 2
    title_fmt: str = 'Port {port}'
    suptitle_fmt: str = '{center_frequency}'
    filename_fmt: str = '{name} {center_frequency}.svg'
    ignore_missing: bool = False


def coerce_griddable_plot_data(
    data: xr.DataArray, plotter: CapturePlotter
) -> xr.DataArray:
    opts = plotter.opts
    if opts.col is None and opts.row is None:
        return data.extend_dims(_view=[''])
    else:
        return data


class CapturePlotter:
    @spec_as_init_kwargs(CapturePlotterOptions)
    def __init__(self, *args, **kwargs):
        self.opts = msgspec.convert(kwargs, CapturePlotterOptions, strict=False)

        if self.opts.output_dir is not None:
            self.opts.output_dir.mkdir(parents=True, exist_ok=True)

    def process_setup(self):
        if not self.opts.interactive:
            mpl.use('agg')
        if self.opts.style is not None:
            plt.style.use(self.opts.style)

        warning_ctx = warnings.catch_warnings()
        warning_ctx.__enter__()
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=r'.*figure layout has changed.*'
        )
        warnings.filterwarnings(
            'ignore', category=UserWarning, message='.*artists with labels.*'
        )

    def kwargs(self, **base: typing.Unpack[util._BasePlotKws]) -> util._PlotKwArgs:
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
        coords: util._PlotKwArgs,
        xticklabelunits: bool | typing.Literal['auto'] = 'auto',
        rasterized=False,
    ):
        data = grid.data
        x = coords['x']
        y = coords.get('y', None)

        if self.opts.suptitle_fmt is not None:
            suptitle = util.label_by_coord(
                data,
                self.opts.suptitle_fmt,
                title_case=True,
                name=data.name,
                meta=data.attrs,
            )[0]
            grid.fig.suptitle(
                suptitle, x=util._get_fig_center_x(grid.fig), ha='center', va='bottom'
            )

        # axes
        util.label_axis('x', data[x], fig=grid.fig, tick_units=xticklabelunits)
        if y is None:
            util.label_axis('y', data, fig=grid.fig, tick_units=False)
        else:
            util.label_axis('y', data[y], fig=grid.fig, tick_units=False)

        # row/col titles
        grid.set_titles('{value}')
        if self.opts.col is None and self.opts.col_label_format is not None:
            for ax in grid.fig.axes:
                if grid.cbar is None or ax is not grid.cbar.ax:
                    ax.title.set_visible(False)
        else:
            col_text = util.label_by_coord(
                data,
                self.opts.col_label_format,
                title_case=True,
                name=data.name,
                **data.attrs,
            )
            for s, ax in zip(col_text, grid.axs.flat):
                ax.set_title(s)
        if self.opts.row is not None and self.opts.row_label_format is not None:
            row_text = util.label_by_coord(
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
        dpi = 0
        if grid.cbar is not None and grid.cbar.solids is not None:
            grid.cbar.solids.set_rasterized(rasterized)
        for ax in grid.fig.axes:
            if grid.cbar is None or ax is not grid.cbar.ax:
                bbox = ax.get_window_extent().transformed(
                    grid.fig.dpi_scale_trans.inverted()
                )
                width, height = bbox.width, bbox.height
                dpi = max(
                    dpi,
                    max(
                        data[x].size / width,
                        (data if y is None else data[y]).size / height,
                    ),
                )
        grid.fig.set_dpi(max(100, dpi))

        if grid.cbar is not None:
            ylabel = grid.cbar.ax.get_ylabel().replace('\n', ' ')
            grid.cbar.ax.set_ylabel(ylabel, rotation=90)

        if self.opts.output_dir is not None:
            filename = set(
                util.label_by_coord(
                    data,
                    self.opts.filename_fmt,
                    name=data.name,
                    title_case=True,
                    **data.attrs,
                )
            )
            path = Path(self.opts.output_dir) / list(filename)[0]
            grid.fig.savefig(path, dpi=dpi)

        if not self.opts.interactive:
            plt.close(grid.fig)


def cellular_cyclic_autocorrelation(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue: str = 'link_direction',
    dB: bool = False,
    **sel,
):
    sub = data.cellular_cyclic_autocorrelation.sel(sel).pipe(
        coerce_griddable_plot_data, plotter
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


def cellular_5g_ssb_spectrogram(plotter: CapturePlotter, data: xr.Dataset, **sel):
    key = cellular_5g_ssb_spectrogram.__name__
    sub = data[key].sel(sel).isel(cellular_ssb_index=0)
    coords = plotter.kwargs(
        x='cellular_ssb_symbol_index', y='cellular_ssb_baseband_frequency'
    )
    vmin = util._get_system_noise(data, key, 6)
    colors = util.quantize_heatmap(sub, vmin=vmin, vstep=2)
    grid = sub.plot.pcolormesh(**coords, **colors, rasterized=True)
    plotter.finish(grid, coords, rasterized=True)


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
    sub = util._select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(yscale=yscale, **coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


def channel_power_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue='power_detector',
    noise_line=True,
    **sel,
):
    key = channel_power_histogram.__name__
    coords = plotter.kwargs(x='channel_power_bin', hue=hue)
    sub = util._select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(**coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


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


def spectrogram(plotter: CapturePlotter, data: xr.Dataset, **sel):
    key = spectrogram.__name__
    sub = data[key].sel(sel).dropna('spectrogram_baseband_frequency')
    coords = plotter.kwargs(x='spectrogram_time', y='spectrogram_baseband_frequency')
    vmin = util._get_system_noise(data, key, 6)
    colors = util.quantize_heatmap(sub, vmin=vmin, vstep=2)
    grid = sub.plot.pcolormesh(**coords, **colors, rasterized=True)
    plotter.finish(grid, coords, rasterized=True)


def spectrogram_histogram(
    plotter: CapturePlotter,
    data: xr.Dataset,
    hue=None,
    yscale='linear',
    noise_line=True,
    **sel,
):
    key = spectrogram_histogram.__name__
    coords = plotter.kwargs(x='spectrogram_power_bin', hue=hue)
    sub = util._select_histogram_bins(data, key, coords['x']).sel(**sel)
    grid = sub.plot.line(yscale=yscale, **coords)
    if noise_line:
        _plot_noise_line(data, key, grid, horizontal=False)
    return plotter.finish(grid, coords)


def spectrogram_ratio_histogram(plotter: CapturePlotter, data: xr.Dataset, **sel):
    key = spectrogram_ratio_histogram.__name__
    return plotter.line(
        data[key].sel(sel),
        name=key,
        x='spectrogram_ratio_power_bin',
        xticklabelunits=False,
        meta=data.attrs,
        # hue=hue,
    )


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

    if sa.dataarrays.CAPTURE_DIM in cyclic_channel_power.dims:
        cyclic_channel_power = cyclic_channel_power.squeeze(sa.dataarrays.CAPTURE_DIM)

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

    util.label_axis('x', cyclic_channel_power.cyclic_lag, ax=ax)
    util.label_axis('y', cyclic_channel_power, tick_units=False, ax=ax)
    util.label_legend(cyclic_channel_power.power_detector, ax=ax)


def _plot_noise_line(
    data: xr.Dataset, key: str, grid: 'xarray.plot.FacetGrid', horizontal=True
):
    noise = util._get_system_noise(data, key)
    if noise is None:
        return
    for pow, ax in zip(noise.values.flat, grid.axs.flat):
        if horizontal:
            ax.axhline(pow, color='k', linestyle=':')
        else:
            ax.axvline(pow, color='k', linestyle=':')
