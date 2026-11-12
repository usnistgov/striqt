from __future__ import annotations as __

import types as _types
import typing as _typing

import striqt.analysis as sa
import striqt.waveform as _sw

from . import util

if _typing.TYPE_CHECKING:
    from collections.abc import Mapping as _Mapping
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


def data_var_plotters() -> _Mapping[str, _DataVariablePlotter]:
    """return a read-only mapping from data variable name to plotting function.

    Each key is the name of a data variable in an analysis dataset. Each value is
    the function that draws the variable, called as ``plotter(ds, backend,
    **options)``: `ds` is the dataset that contains the variable, `backend` a
    `PlotBackend`, and `options` the keyword arguments of the function.

    Returns:
        a read-only view of the registered functions; item assignment raises
        `TypeError`
    """
    return _types.MappingProxyType(_data_plots)


@_register_data_var_plot
def cellular_cyclic_autocorrelation(
    data: _DS,
    plotter: _PlotBackend,
    *,
    hue: str = 'link_direction',
    dB: bool = False,
) -> _Path | None:
    """plot `cellular_cyclic_autocorrelation` of `data` against `cyclic_sample_lag`.

    Args:
        data: the dataset with the `cellular_cyclic_autocorrelation` variable and its
            `link_direction`, `subcarrier_spacing`, and `cyclic_sample_lag` coordinates
        plotter: the backend that draws the figure
        hue: ``'link_direction'`` draws one line per link direction at the
            `subcarrier_spacing` with the largest peak; ``'subcarrier_spacing'`` draws
            one line per spacing of the ``'downlink'`` row
        dB: True converts the values from linear power to dB before drawing

    Returns:
        the path of the figure file, or `None` when the backend writes no file

    Raises:
        TypeError: if `hue` is neither ``'link_direction'`` nor
            ``'subcarrier_spacing'``
    """
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
) -> _Path | None:
    """plot `cellular_5g_pss_correlation` of `data` as power against `cellular_ssb_lag`.

    The values are the squared magnitude of the correlation in the first sync block
    (`cellular_ssb_start_time` index 0). The lower y limit is 7 dB below the smallest
    median power over lags of the `hue` values.

    Args:
        data: the dataset with the `cellular_5g_pss_correlation` variable and its
            `cellular_cell_id2`, `cellular_ssb_start_time`,
            `cellular_ssb_beam_index`, and `cellular_ssb_lag` coordinates
        plotter: the backend that draws the figure
        hue: ``'cellular_ssb_beam_index'`` draws one line per beam index, averaged
            over `cellular_cell_id2`; ``'cellular_cell_id2'`` draws one line per cell
            sector ID, averaged over `cellular_ssb_beam_index`
        dB: True draws the power in dB, False in linear units

    Returns:
        the path of the figure file, or `None` when the backend writes no file

    Raises:
        KeyError: if `hue` is neither ``'cellular_ssb_beam_index'`` nor
            ``'cellular_cell_id2'``
    """
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
def cellular_5g_ssb_spectrogram(
    data: _DS, plotter: _PlotBackend, noise_line: bool = True
) -> None:
    """plot `cellular_5g_ssb_spectrogram` of `data` as a heatmap of symbol by frequency.

    The first burst set of each capture (`cellular_ssb_index` 0) is drawn. The color
    scale starts 6 dB below the lowest system noise level over the captures, in 2 dB
    bands; without a `system_noise` variable in `data` the scale starts at the data
    minimum.

    Args:
        data: the dataset with the `cellular_5g_ssb_spectrogram` variable and its
            `cellular_ssb_index`, `cellular_ssb_symbol_index`, and
            `cellular_ssb_baseband_frequency` coordinates
        plotter: the backend that draws the figure
        noise_line: True marks the system noise level on the colorbar, from
            `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of the
            variable; nothing is marked without `system_noise`
    """
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
    hue: str = 'link_direction',
    noise_line: bool = True,
    yscale: _ScaleOptions = 'linear',
) -> _Path | None:
    """plot `cellular_resource_power_histogram` of `data` against the power bins.

    The x axis spans from 12 dB below the system noise level of each capture, or from
    the lowest bin without a `system_noise` variable in `data`, to 3 dB above the
    highest populated bin over the captures.

    Args:
        data: the dataset with the `cellular_resource_power_histogram` variable and
            its `link_direction` and `cellular_resource_power_bin` coordinates
        plotter: the backend that draws the figure
        hue: the dimension of the histogram that selects one line per value
        noise_line: True marks the system noise level as a vertical line on each
            facet, from `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of
            the power bins; nothing is marked without `system_noise`
        yscale: the y axis scale, ``'linear'`` or ``'log'``

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
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
    hue: str = 'power_detector',
    noise_line: bool = True,
    yscale: _ScaleOptions = 'linear',
) -> _Path | None:
    """plot `channel_power_histogram` of `data` against `channel_power_bin`.

    The x axis spans from 12 dB below the system noise level of each capture, or from
    the lowest bin without a `system_noise` variable in `data`, to 3 dB above the
    highest populated bin over the captures.

    Args:
        data: the dataset with the `channel_power_histogram` variable and its
            `power_detector` and `channel_power_bin` coordinates
        plotter: the backend that draws the figure
        hue: the dimension of the histogram that selects one line per value
        noise_line: True marks the system noise level as a vertical line on each
            facet, from `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of
            the power bins; nothing is marked without `system_noise`
        yscale: the y axis scale, ``'linear'`` or ``'log'``

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
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
    hue: str = 'power_detector',
    noise_line: bool = True,
) -> _Path | None:
    """plot `channel_power_time_series` of `data` against `time_elapsed`.

    Args:
        data: the dataset with the `channel_power_time_series` variable and its
            `power_detector` and `time_elapsed` coordinates
        plotter: the backend that draws the figure
        hue: the dimension that selects one line per value
        noise_line: True marks the system noise level as a horizontal line on each
            facet, from `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of
            the variable; nothing is marked without `system_noise`

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
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
    """draw the `center_statistic` line and the `bound_statistics` band per `hue` value.

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

    Returns:
        the path of the figure file, or `None` when the backend writes no file

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
    """plot `power_spectral_density` of `data` against `baseband_frequency`.

    Args:
        data: the dataset with the `power_spectral_density` variable and its
            `time_statistic` and `baseband_frequency` coordinates; the variable and
            the statistic coordinate are also read under the names
            `persistence_spectrum` and `persistence_statistics`
        plotter: the backend that draws the figure
        hue: the dimension that selects one line per value
        noise_line: True marks the system noise level as a horizontal line on each
            facet, from `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of
            the variable; nothing is marked without `system_noise`

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
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
    """plot `spectrogram` of `data` as a heatmap of time by baseband frequency.

    A frequency bin with a `nan` value in any pixel is not drawn. The color scale
    starts 6 dB below the lowest system noise level over the captures, in 2 dB bands;
    without a `system_noise` variable in `data` the scale starts at the data minimum.

    Args:
        data: the dataset with the `spectrogram` variable and its `spectrogram_time`
            and `spectrogram_baseband_frequency` coordinates
        plotter: the backend that draws the figure
        noise_line: True marks the system noise level on the colorbar, from
            `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of the
            variable; nothing is marked without `system_noise`
    """
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
    noise_line: bool = True,
) -> _Path | None:
    """plot `spectrogram_histogram` of `data` against `spectrogram_power_bin`.

    The x axis spans from 12 dB below the system noise level of each capture, or from
    the lowest bin without a `system_noise` variable in `data`, to 3 dB above the
    highest populated bin over the captures.

    Args:
        data: the dataset with the `spectrogram_histogram` variable and its
            `spectrogram_power_bin` coordinate
        plotter: the backend that draws the figure
        yscale: the y axis scale, ``'linear'`` or ``'log'``
        noise_line: True marks the system noise level as a vertical line on each
            facet, from `system_noise` (dBm/Hz) of `data` in the `noise_bandwidth` of
            the power bins; nothing is marked without `system_noise`

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
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
) -> _Path | None:
    """plot `spectrogram_ratio_histogram` of `data` against the power-ratio bins.

    The x axis spans from the lowest bin to 3 dB above the highest populated bin over
    the captures.

    Args:
        data: the dataset with the `spectrogram_ratio_histogram` variable and its
            `spectrogram_ratio_power_bin` coordinate
        plotter: the backend that draws the figure
        yscale: the y axis scale, ``'linear'`` or ``'log'``

    Returns:
        the path of the figure file, or `None` when the backend writes no file
    """
    name = sa.measurements.spectrogram_ratio_histogram.__name__
    x = 'spectrogram_ratio_power_bin'
    sub = util.select_histogram_bins(data, name, x)
    grid = plotter.line(sub, x=x, yscale=yscale)
    return plotter.finish(grid)
