from __future__ import annotations as __

import typing as _typing

import striqt.analysis as sa
import striqt.waveform as _sw

from . import util

if _typing.TYPE_CHECKING:
    from pathlib import Path as _Path

    from xarray import DataArray as _DA, Dataset as _DS
    from xarray.core.types import ScaleOptions as _ScaleOptions

    from .backend import PlotBackend, StepStyle as _StepStyle

    _PlotBackend = PlotBackend[_typing.Any]

    class _DataVariablePlotter(_typing.Protocol):
        def __call__(
            self, data: _DS, plotter: _PlotBackend, /, *args, **kwargs
        ) -> _typing.Any: ...

        __name__: str

    _TVP = _typing.TypeVar('_TVP', bound=_DataVariablePlotter)


_data_plots: dict[str, _DataVariablePlotter] = {}


def _coord_names(spec_type: type[sa.specs.Analysis]) -> tuple[str, ...]:
    """return the coordinate names of the measurement registered for `spec_type`.

    The order is that of `coord_factories` in the registry entry.
    """
    factories = sa.registry[spec_type].coord_factories
    return tuple(sa.registry.coordinates[f].name for f in factories)


def _register_data_var_plot(func: _TVP) -> _TVP:
    name = func.__name__
    func = sa.util.stopwatch(name, 'analysis', logger_level=sa.util.PERFORMANCE_INFO)(
        func
    )
    _data_plots[name] = func
    return func


@_register_data_var_plot
def cellular_cyclic_autocorrelation(
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue: str = 'link_direction',
    dB: bool = False,
):
    from . import backend

    name = sa.measurements.cellular_cyclic_autocorrelation.__name__
    sub = backend.coerce_column(data[name], plotter)
    if hue == 'link_direction':
        scs_peaks = sub.max([n for n in sub.dims if n != 'subcarrier_spacing'])
        iscs = int(_typing.cast('_DA', scs_peaks.argmax()))
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
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue: str = 'cellular_ssb_beam_index',
    dB: bool = True,
):
    name = sa.measurements.cellular_5g_pss_correlation.__name__
    R = data[name].isel(cellular_ssb_start_time=0)
    pow = _sw.envtopow(R)

    if hue == 'cellular_cell_id2':
        pow = pow.mean('cellular_ssb_beam_index', keep_attrs=True)
    elif hue == 'cellular_ssb_beam_index':
        pow = pow.mean('cellular_cell_id2', keep_attrs=True)
    else:
        raise KeyError('invalid hue coordinate')

    low_median = pow.median(set(pow.dims) - {'cellular_ssb_beam_index'}).min()
    ymin = float(low_median / 5)

    if dB:
        pow = _sw.powtodB(pow)
        ymin = _sw.powtodB(ymin)

    grid = plotter.line(pow, x='cellular_ssb_lag', hue=hue, ylim=(ymin, None))
    return plotter.finish(grid)


@_register_data_var_plot
def cellular_5g_ssb_spectrogram(data: _DS, plotter: _PlotBackend, noise_line=True):
    name = sa.measurements.cellular_5g_ssb_spectrogram.__name__
    x = 'cellular_ssb_symbol_index'
    y = 'cellular_ssb_baseband_frequency'
    sub = data[name].isel(cellular_ssb_index=0)
    vmin = util.system_noise_floor(data, name, 6)
    grid = plotter.heatmap(sub, x=x, y=y, vmin=vmin, vstep=2)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='colorbar')
    plotter.finish(grid)


@_register_data_var_plot
def cellular_resource_power_histogram(
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue='link_direction',
    noise_line=True,
    yscale: _ScaleOptions = 'linear',
):
    name = sa.measurements.cellular_resource_power_histogram.__name__
    x = 'cellular_resource_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale, hue=hue, sharex=False)
    plotter.limit_facets(grid, *util.histogram_bin_limits(data, name, x))
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='y')
    return plotter.finish(grid)


@_register_data_var_plot
def channel_power_histogram(
    data: _DS,
    plotter: _PlotBackend,
    hue='power_detector',
    noise_line=True,
    yscale: _ScaleOptions = 'linear',
):
    name = sa.measurements.channel_power_histogram.__name__
    x = 'channel_power_bin'
    sub = util.select_histogram_bins(data, name, x)

    grid = plotter.line(sub, x=x, yscale=yscale, hue=hue, sharex=False)
    plotter.limit_facets(grid, *util.histogram_bin_limits(data, name, x))
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='y')
    return plotter.finish(grid)


@_register_data_var_plot
def channel_power_time_series(
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue='power_detector',
    noise_line=True,
):
    name = sa.measurements.channel_power_time_series.__name__
    grid = plotter.line(data[name], x='time_elapsed', hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def cyclic_channel_power(
    data: _DS,
    plotter: _PlotBackend,
    *,
    center_statistic: str | float = 'mean',
    bound_statistics: tuple[str | float, str | float] = ('min', 'max'),
    hue: str = 'power_detector',
    dB: bool = True,
    steps: bool = True,
    noise_line: bool = True,
) -> _Path | None:
    """plot the `center_statistic` line per `hue` value and shade `bound_statistics`.

    Args:
        data: the dataset with the `cyclic_channel_power` variable
        plotter: the backend that draws the figure
        center_statistic: the `cyclic_statistic` value drawn as the line
        bound_statistics: the `cyclic_statistic` values at the lower and upper edge of
            the shaded band
        hue: the dimension that selects one line and band per value
        dB: False converts the values from dB to linear power before drawing
        steps: True holds each value until the next `cyclic_lag` sample
        noise_line: True marks the system noise level on each facet

    Raises:
        ValueError: if `center_statistic` or either of `bound_statistics` is not a
            `cyclic_statistic` value of `data`
    """
    name = sa.measurements.cyclic_channel_power.__name__
    sub = data[name]

    available = sub.cyclic_statistic.values.tolist()
    missing = [s for s in (center_statistic, *bound_statistics) if s not in available]
    if missing:
        raise ValueError(
            f'cyclic_statistic values {missing!r} are not in {available!r}'
        )

    if not dB:
        sub = _sw.dBtopow(sub)
    step: _StepStyle | None = 'post' if steps else None

    center = sub.sel(cyclic_statistic=center_statistic)
    low = sub.sel(cyclic_statistic=bound_statistics[0])
    high = sub.sel(cyclic_statistic=bound_statistics[1])

    grid = plotter.line(center, x='cyclic_lag', hue=hue, step=step)
    plotter.fill_between(low, high, x='cyclic_lag', hue=hue, step=step, target=grid)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def power_spectral_density(
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue: str = 'time_statistic',
    noise_line: bool = True,
) -> _Path | None:
    name = sa.measurements.power_spectral_density.__name__
    _, x = _coord_names(sa.specs.PowerSpectralDensity)
    # a store may name these 'persistence_spectrum' and 'persistence_statistics'
    if name not in data.data_vars and 'persistence_spectrum' in data.data_vars:
        name = 'persistence_spectrum'
    if hue == 'time_statistic' and 'persistence_statistics' in data:
        hue = 'persistence_statistics'
    grid = plotter.line(data[name], x=x, hue=hue)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='x')
    return plotter.finish(grid)


@_register_data_var_plot
def spectrogram(data: _DS, plotter: _PlotBackend, noise_line: bool = True) -> None:
    name = sa.measurements.spectrogram.__name__
    x, y = _coord_names(sa.specs.Spectrogram)
    sub = data[name].dropna(y)
    vmin = util.system_noise_floor(data, name, 6)
    grid = plotter.heatmap(sub, x=x, y=y, vmin=vmin, vstep=2)
    if noise_line:
        plotter.mark_noise_level(data, name, grid, where='colorbar')
    plotter.finish(grid)


@_register_data_var_plot
def spectrogram_histogram(
    data: _DS,
    plotter: _PlotBackend,
    *,
    yscale: _ScaleOptions = 'linear',
    noise_line=True,
):
    name = sa.measurements.spectrogram_histogram.__name__
    x = 'spectrogram_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale, sharex=False)
    plotter.limit_facets(grid, *util.histogram_bin_limits(data, name, x))
    if noise_line:
        plotter.mark_noise_level(data, x, grid, where='y')
    return plotter.finish(grid)


@_register_data_var_plot
def spectrogram_ratio_histogram(
    data: _DS, plotter: _PlotBackend, *, yscale: _ScaleOptions = 'linear'
):
    name = sa.measurements.spectrogram_ratio_histogram.__name__
    x = 'spectrogram_ratio_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale)
    return plotter.finish(grid)
