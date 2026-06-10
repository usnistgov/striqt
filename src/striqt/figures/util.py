from __future__ import annotations as __

import ast
import math
import string
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


def literal_eval(s) -> typing.Any:
    """evaluate a literal, also allowing a slice object"""
    return _SliceLiteralEval.evaluate(s)


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


@sa.util.lru_cache()
def get_format_fields(fmt, exclude=()):
    all_exclude = exclude + (None,)
    return [l[1] for l in string.Formatter().parse(fmt) if l[1] not in all_exclude]


def get_groupby_fields(ds: 'xr.Dataset', opts: specs.PlotOptions) -> list[str]:
    """guess a list of fields that produce groups compatible with the plot options"""
    loops = ss.lib.compute.get_looped_coords(ds)
    fields = get_format_fields(opts.plotter.filename_fmt, exclude=('name',))
    return sa.util.ordered_set_union(opts.data.groupby_dims, fields, loops)


def guess_index_coords(ds: 'xr.Dataset', opts: specs.PlotOptions) -> list[str | None]:
    idx_coords = get_groupby_fields(ds, opts) + [opts.plotter.col]
    if opts.plotter.row:
        idx_coords = idx_coords + [opts.plotter.row]
    return idx_coords


def query_match_at_index(ds: '_T', dim: str, var_name: str, index: int) -> '_T':
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
        # Python 3.8+ handles literal values as Constants
        if type(node.value) in self.allowed_constants or isinstance(node.value, tuple):
            return node.value
        raise ValueError(f'Malformed node or string: {node}')

    def visit_Name(self, node):
        # Handle single standalone literals like None, True, False
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
