from __future__ import annotations as __

import typing as _typing
import striqt.waveform as _sw
import striqt.analysis as _sa
from . import util


if _typing.TYPE_CHECKING:
    from xarray import Dataset as _DS
    from xarray.core.types import ScaleOptions as _ScaleOptions
    from .backend import PlotBackend as _PlotBackend

    class _DataVariablePlotter(_typing.Protocol):
        def __call__(
            self, data: '_DS', plotter: _PlotBackend, /, *args, **kwargs
        ) -> _typing.Any: ...

        __name__: str

    _TVP = _typing.TypeVar('_TVP', bound=_DataVariablePlotter)

else:
    _xr = _sw.util.lazy_import('xarray')


_data_plots: dict[str, _DataVariablePlotter] = {}


def _register_data_var_plot(func: '_TVP') -> '_TVP':
    name = func.__name__
    func = _sa.util.stopwatch(name, 'analysis', logger_level=_sa.util.WARNING)(func)
    _data_plots[name] = func
    return func


@_register_data_var_plot
def cellular_cyclic_autocorrelation(
    data: '_DS',
    plotter: '_PlotBackend',
    *,
    hue: str = 'link_direction',
    dB: bool = False,
):
    from . import backend

    name = _sa.measurements.cellular_cyclic_autocorrelation.__name__
    sub = backend.coerce_column(data[name], plotter)
    if hue == 'link_direction':
        scs_peaks = sub.max([n for n in sub.dims if n != 'subcarrier_spacing'])
        iscs = int(scs_peaks.argmax())  # type: ignore
        sub = sub.isel(subcarrier_spacing=iscs)
    elif hue == 'subcarrier_spacing':
        sub = sub.sel(link_direction='downlink')
    else:
        raise TypeError('invalid hue coordinate name')
    if dB:
        sub = _sw.powtodB(sub)
    grid = plotter.line(sub, x='cyclic_sample_lag', hue=hue)
    return plotter.finish(grid)


@_register_data_var_plot
def cellular_5g_pss_correlation(
    data: '_DS',
    plotter: '_PlotBackend',
    *,
    hue: str = 'cellular_ssb_beam_index',
    dB: bool = True,
):
    name = _sa.measurements.cellular_5g_pss_correlation.__name__
    R = data[name].isel(cellular_ssb_start_time=0)
    pow = _sw.envtopow(R)

    if hue == 'cellular_cell_id2':
        pow = pow.mean('cellular_ssb_beam_index', keep_attrs=True)
    elif hue == 'cellular_ssb_beam_index':
        pow = pow.mean('cellular_cell_id2', keep_attrs=True)
    else:
        raise KeyError('invalid hue coordinate')

    if dB:
        pow = _sw.powtodB(pow)

    grid = plotter.line(pow, x='cellular_ssb_lag', hue=hue)
    return plotter.finish(grid)


@_register_data_var_plot
def cellular_5g_ssb_spectrogram(data: '_DS', plotter: '_PlotBackend', noise_line=True):
    name = _sa.measurements.cellular_5g_ssb_spectrogram.__name__
    x = 'cellular_ssb_symbol_index'
    y = 'cellular_ssb_baseband_frequency'
    sub = data[name].isel(cellular_ssb_index=0)
    vmin = util.get_system_noise(data, name, 6)
    grid = plotter.heatmap(sub, x=x, y=y, vmin=vmin, vstep=2)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='colorbar')
    plotter.finish(grid)


@_register_data_var_plot
def cellular_resource_power_histogram(
    data: '_DS',
    plotter: '_PlotBackend',
    *,
    hue='link_direction',
    noise_line=True,
    yscale: '_ScaleOptions' = 'linear',
):
    name = _sa.measurements.cellular_resource_power_histogram.__name__
    x = 'cellular_resource_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale, hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='y')
    return plotter.finish(grid)


@_register_data_var_plot
def channel_power_histogram(
    data: '_DS',
    plotter: '_PlotBackend',
    hue='power_detector',
    noise_line=True,
    yscale: '_ScaleOptions' = 'linear',
):
    name = _sa.measurements.channel_power_histogram.__name__
    x = 'channel_power_bin'
    sub = util.select_histogram_bins(data, name, x)

    grid = plotter.line(sub, x=x, yscale=yscale, hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='y')
    return plotter.finish(grid)


@_register_data_var_plot
def channel_power_time_series(
    data: '_DS',
    plotter: '_PlotBackend',
    *,
    hue='power_detector',
    noise_line=True,
):
    name = _sa.measurements.channel_power_time_series.__name__
    grid = plotter.line(data[name], x='time_elapsed', hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def cyclic_channel_power(data: '_DS', plotter: '_PlotBackend', *, noise_line=True):
    # TODO: refactor to make this implementable with plotter method calls

    from xarray.plot import FacetGrid
    from . import backend

    name = _sa.measurements.cyclic_channel_power.__name__
    sub = data[name]
    grid = FacetGrid(sub, **plotter._coord_kws(), sharey=True)  # type: ignore
    coords = plotter._coord_kws(x='cyclic_lag')
    grid._x_var = coords['x']  # type: ignore

    for col, ax in zip(data[coords['col']], grid.axs.flat):
        backend.plot_cyclic_channel_power(sub.sel({coords['col']: col}), ax=ax)

    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')

    return plotter.finish(grid)


@_register_data_var_plot
def power_spectral_density(
    data: '_DS', plotter: '_PlotBackend', *, hue='time_statistic', noise_line=True
):
    name = _sa.measurements.power_spectral_density.__name__
    x = _sa.measurements._power_spectral_density.baseband_frequency.__name__
    # legacy support tweaks
    if name not in data.data_vars and 'persistence_spectrum' in data.data_vars:
        name = 'persistence_spectrum'
    if hue == 'time_statistic' and 'persistence_statistics' in data:
        hue = 'persistence_statistics'
    grid = plotter.line(data[name], x=x, hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def spectrogram(data: '_DS', plotter: '_PlotBackend', noise_line=True):
    name = _sa.measurements.spectrogram.__name__
    sub = data[name].dropna('spectrogram_baseband_frequency')
    x = _sa.measurements._spectrogram.spectrogram_time.__name__
    y = _sa.measurements.shared.spectrogram_baseband_frequency.__name__
    vmin = util.get_system_noise(data, name, 6)
    grid = plotter.heatmap(sub, x=x, y=y, vmin=vmin, vstep=2)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='colorbar')
    plotter.finish(grid)


@_register_data_var_plot
def spectrogram_histogram(
    data: '_DS',
    plotter: '_PlotBackend',
    *,
    yscale: '_ScaleOptions' = 'linear',
    noise_line=True,
):
    name = _sa.measurements.spectrogram_histogram.__name__
    x = 'spectrogram_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale)
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def spectrogram_ratio_histogram(
    data: '_DS', plotter: '_PlotBackend', *, yscale: '_ScaleOptions' = 'linear'
):
    name = _sa.measurements.spectrogram_ratio_histogram.__name__
    x = 'spectrogram_ratio_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale)
    return plotter.finish(grid)
