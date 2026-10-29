from __future__ import annotations as __

import ast
import math
import typing

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

from . import specs

if typing.TYPE_CHECKING:
    _T = typing.TypeVar('_T', bound=xr.Dataset)

get_format_fields = ss.specs.paths.get_format_fields


def _owner_noise_bandwidth(data: xr.Dataset, coord_name: str) -> float | None:
    """return the `noise_bandwidth` attr (Hz) of a variable spanning `coord_name`.

    None when `coord_name` is a data variable or a scalar, or when no data variable on
    the dimension of `coord_name` has the attr.
    """
    coord = data[coord_name]
    if coord_name in data.data_vars or coord.ndim == 0:
        return None
    dim = coord.dims[0]
    for var in data.data_vars.values():
        if dim in var.dims and 'noise_bandwidth' in var.attrs:
            return float(var.attrs['noise_bandwidth'])
    return None


def noise_bandwidth(data: xr.Dataset, var_name: str) -> xr.DataArray | float | None:
    """return the noise bandwidth (in Hz) of a data variable or power-bin coordinate.

    The first matching rule applies:

    * `var_name` has a `noise_bandwidth` attr: that value;
    * `var_name` is a coordinate: the `noise_bandwidth` attr of a data variable on
      the dimension of the coordinate;
    * `var_name` has `units` beginning with dBm or √mW: the `analysis_bandwidth`
      coordinate where finite, and the `sample_rate` coordinate where infinite;
    * otherwise: None.

    Returns:
        a float, a DataArray over the captures when the bandwidth is a capture
        coordinate, or None
    """
    da = data[var_name]
    if 'noise_bandwidth' in da.attrs:
        return float(da.attrs['noise_bandwidth'])

    from_owner = _owner_noise_bandwidth(data, var_name)
    if from_owner is not None:
        return from_owner

    units = str(da.attrs.get('units', ''))
    if not units.startswith(('dBm', '√mW')):
        return None
    if 'analysis_bandwidth' in data.variables:
        bw = data['analysis_bandwidth']
        if 'sample_rate' in data.variables:
            bw = bw.where(np.isfinite(bw), data['sample_rate'])
        return bw
    if 'sample_rate' in data.variables:
        return data['sample_rate']
    return None


@typing.overload
def get_system_noise(
    data: xr.Dataset, var_name: str, margin: float
) -> float | None: ...


@typing.overload
def get_system_noise(
    data: xr.Dataset, var_name: str, margin: None = None
) -> xr.DataArray | None: ...


def get_system_noise(
    data: xr.Dataset, var_name: str, margin: float | None = None
) -> xr.DataArray | float | None:
    """return the calibrated system noise (in dBm) in the noise bandwidth of `var_name`.

    Per capture, the level is `system_noise` plus ``10*log10(B)``, with `B` from
    `noise_bandwidth` for the data variable or power-bin coordinate `var_name`. With
    a terminated input, a histogram of the dB values of a Gamma-distributed bin power
    has a mode at this level.

    Args:
        data: a dataset with the `system_noise` variable (dBm/Hz) on the capture
            dimension
        var_name: a data variable or power-bin coordinate name
        margin: None returns the level per capture; a value (in dB) returns the
            minimum over captures minus `margin`

    Returns:
        a DataArray over the captures with ``units='dBm'``, or the scalar minimum
        with `margin`; None without `system_noise` or without a noise bandwidth for
        `var_name`
    """
    if 'system_noise' not in data:
        return None

    B = noise_bandwidth(data, var_name)
    if B is None:
        return None

    noise = (data.system_noise + sw.powtodB(B)).assign_attrs(
        standard_name='System noise', units='dBm'
    )
    if isinstance(B, float):
        bandwidth = sa.dataarrays.format_units(B, 'Hz')
        noise = noise.assign_attrs(
            noise_bandwidth=B, long_name=f'System noise (dBm/{bandwidth})'
        )

    if margin is None:
        return noise
    else:
        return float(noise.min() - margin)


def system_noise_floor(data: xr.Dataset, var_name: str, margin: float) -> float | None:
    """return the minimum over captures of `get_system_noise` minus `margin` (in dB), or None"""
    return get_system_noise(data, var_name, margin)


def check_noise_floor(data: xr.Dataset, var_name: str) -> xr.Dataset:
    """compare the mode of each power histogram with the calibrated system noise.

    The mode of a histogram of the dB values of a Gamma-distributed bin power equals
    the dB of the mean power. With a terminated input and a correct calibration,
    `offset_dB` is within a few tenths of a dB of 0. A signal in the capture makes
    `offset_dB` positive. The ``-inf`` and ``inf`` catch-all bins are excluded from
    the mode.

    Args:
        data: a dataset with `system_noise` and the histogram `var_name`
        var_name: a power histogram variable with one ``*_power_bin`` dimension
            (`spectrogram_histogram`, `cellular_resource_power_histogram`, or
            `channel_power_histogram`)

    Returns:
        a Dataset over the dimensions of the histogram other than the bin, with
        `observed_dBm` (the center of the mode bin), `predicted_dBm`
        (`get_system_noise`), and `offset_dB` (observed minus predicted)

    Raises:
        ValueError: if `var_name` does not have exactly one ``*_power_bin``
            dimension, or `data` has no `system_noise` or no noise bandwidth for
            `var_name`
    """
    hist = data[var_name]
    bin_dims = [str(d) for d in hist.dims if str(d).endswith('_power_bin')]
    if len(bin_dims) != 1:
        raise ValueError(
            f'{var_name!r} must have one *_power_bin dimension, found {bin_dims}'
        )
    bin_dim = bin_dims[0]

    predicted = get_system_noise(data, var_name)
    if predicted is None:
        # a histogram variable without a noise bandwidth can still have one on the
        # bin coordinate
        predicted = get_system_noise(data, bin_dim)
    if predicted is None:
        raise ValueError(
            f'no system_noise or noise bandwidth to predict the floor of {var_name!r}'
        )

    bins = data[bin_dim]
    counts = hist.fillna(0.0).where(np.isfinite(bins), -1.0)
    observed = bins.isel({bin_dim: counts.argmax(bin_dim)})
    observed = observed.drop_vars(bin_dim, errors='ignore').assign_attrs(
        standard_name='Histogram mode', units='dBm'
    )
    predicted = predicted.broadcast_like(observed)
    offset = (observed - predicted).assign_attrs(
        standard_name='Mode above system noise', units='dB'
    )

    return xr.Dataset({
        'observed_dBm': observed,
        'predicted_dBm': predicted,
        'offset_dB': offset,
    })


def literal_eval(s) -> typing.Any:
    """evaluate a literal, also allowing a slice object"""
    return _SliceLiteralEval.evaluate(s)


def histogram_bin_limits(
    data: xr.Dataset,
    var_name: str,
    bin_dim: str,
    pad_low: float = 12,
    pad_hi: float = 3,
) -> tuple[xr.DataArray | None, float]:
    """return the plot limits (in dBm) of a power histogram.

    Args:
        data: the dataset with the histogram `var_name` and the coordinate `bin_dim`
        var_name: the power histogram variable
        bin_dim: the power-bin coordinate (dBm) of `var_name`
        pad_low: the margin (dB) below the system noise level
        pad_hi: the margin (dB) above the highest populated bin

    Returns:
        ``(low, high)``: `low` is the system noise level per capture minus
        `pad_low`, or None when `data` has no noise level for `bin_dim`; `high` is
        the highest populated bin over all captures plus `pad_hi`
    """
    noise = get_system_noise(data, bin_dim)
    low = None if noise is None else noise - pad_low
    sub = data[var_name]
    high = float(sub.cumsum(bin_dim).idxmax(bin_dim).max()) + pad_hi
    return low, high


def select_histogram_bins(
    data: xr.Dataset, var_name, bin_dim, pad_low: float = 12, pad_hi: float = 3
) -> xr.DataArray:
    """return the histogram `var_name` trimmed to the bins within `histogram_bin_limits`"""
    low, high = histogram_bin_limits(data, var_name, bin_dim, pad_low, pad_hi)
    xmin = None if low is None else float(low.min())
    return data[var_name].sel({bin_dim: slice(xmin, high)})


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


def get_groupby_fields(ds: xr.Dataset, opts: specs.PlotOptions) -> list[str]:
    """guess a list of fields that produce groups compatible with the plot options"""
    loops = ss.lib.compute.get_looped_coords(ds)
    fields = get_format_fields(opts.plotter.filename_fmt, exclude=('name',))
    return sa.util.ordered_set_union(opts.data.groupby_dims, fields, loops)


def guess_index_coords(ds: xr.Dataset, opts: specs.PlotOptions) -> list[str | None]:
    idx_coords = get_groupby_fields(ds, opts) + [opts.plotter.col]
    if opts.plotter.row:
        idx_coords = idx_coords + [opts.plotter.row]
    return idx_coords


def query_match_at_index(ds: _T, dim: str, var_name: str, index: int) -> _T:
    return ds.query({dim: f'{var_name} == {var_name}[{index}]'})  # .squeeze(var_name)


class _SliceLiteralEval(ast.NodeVisitor):
    def __init__(self):
        self.allowed_constants = {
            type(None),
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            frozenset,
        }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        # Python 3.8 and later parse literal values as ast.Constant nodes
        if type(node.value) in self.allowed_constants or isinstance(node.value, tuple):
            return node.value
        raise ValueError(f'Malformed node or string: {node}')

    def visit_Name(self, node):
        if node.id == 'None':
            return None
        elif node.id == 'True':
            return True
        elif node.id == 'False':
            return False
        raise ValueError(f'Malformed node or string: {node}')

    def visit_Tuple(self, node):
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_List(self, node):
        return [self.visit(elt) for elt in node.elts]

    def visit_Dict(self, node):
        keys = [self.visit(k) for k in node.keys]
        values = [self.visit(v) for v in node.values]
        return dict(zip(keys, values))

    def visit_Set(self, node):
        return {self.visit(elt) for elt in node.elts}

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'slice':
            if node.keywords:
                raise ValueError(
                    'Keyword arguments are not supported in slice() for literal evaluation'
                )
            args = [self.visit(arg) for arg in node.args]
            if len(args) > 3 or len(args) < 1:
                raise ValueError('slice() takes 1 to 3 arguments')
            return slice(*args)
        raise ValueError(f'Malformed node or string: {node}')

    @classmethod
    def evaluate(cls, expression: str):
        node = ast.parse(expression, mode='eval')
        return cls().visit(node)
