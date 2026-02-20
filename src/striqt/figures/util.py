from __future__ import annotations as __

import math
import striqt.analysis as sa
import striqt.waveform as sw
import striqt.sensor as ss
import typing

from . import specs

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    _T = typing.TypeVar('_T', bound=xr.Dataset)
else:
    np = sw.util.lazy_import('numpy')
    xr = sw.util.lazy_import('xarray')


@typing.overload
def get_system_noise(
    data: 'xr.Dataset', var_name: str, margin: float
) -> float | None: ...


@typing.overload
def get_system_noise(
    data: 'xr.Dataset', var_name: str, margin: None = None
) -> xr.DataArray | float | None: ...


def get_system_noise(
    data: 'xr.Dataset', var_name: str, margin: float | None = None
) -> xr.DataArray | float | None:
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


def select_histogram_bins(
    data: 'xr.Dataset', var_name, bin_dim, pad_low: float = 12, pad_hi: float = 3
) -> 'xr.DataArray':
    """return a downselected range of histogram power bins from extrema and cal data"""
    xnoise = get_system_noise(data, bin_dim, margin=pad_low)
    sub = data[var_name]
    xmax = sub.cumsum(bin_dim).idxmax(bin_dim).max() + pad_hi
    return sub.sel({bin_dim: slice(xnoise, xmax)})


def quantized_value_range(data: xr.DataArray, vmin, vmax, vstep):
    """return an array of bin edges suitable for a quantized colorbar"""
    if vmin is None:
        vmin = float(data.min())

    if vmax is None:
        vmax = float(data.max())

    vmax_edge = math.ceil(vmax / vstep) * vstep
    vmin_edge = math.floor(vmin / vstep) * vstep

    n = round((vmax_edge - vmin_edge) / vstep) + 1

    levels = np.linspace(vmin_edge, vmax_edge, n).tolist()

    return levels


def get_groupby_fields(ds: 'xr.Dataset', opts: specs.PlotOptions) -> list[str]:
    """guess a list of fields that produce groups compatible with the plot options"""
    from striqt import figures as sf

    loops = ss.lib.compute.get_looped_coords(ds)
    fields = sf.specs.get_format_fields(opts.plotter.filename_fmt, exclude=('name',))
    return sa.util.ordered_set_union(opts.data.groupby_dims, fields, loops, ['port'])


def guess_index_coords(ds: 'xr.Dataset', opts: specs.PlotOptions) -> list[str | None]:
    idx_coords = get_groupby_fields(ds, opts) + [opts.plotter.col]
    if opts.plotter.row:
        idx_coords = idx_coords + [opts.plotter.row]
    return idx_coords


def query_match_at_index(ds: '_T', dim: str, var_name: str, index: int) -> '_T':
    return ds.query({dim: f'{var_name} == {var_name}[{index}]'})  # .squeeze(var_name)
