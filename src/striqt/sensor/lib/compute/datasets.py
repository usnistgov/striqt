"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

from collections import defaultdict
import dataclasses
import logging
from typing import Any, Literal, TYPE_CHECKING

from ... import specs
from .. import util

import striqt.analysis as sa
from striqt.analysis.lib.dataarrays import CAPTURE_DIM
from ..typing import Callable, Sequence, TAR

import msgspec

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr

else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')


SOURCE_ID_NAME = 'source_id'


class EvaluationOptions(sa.EvaluationOptions[TAR], kw_only=True):
    sweep_spec: specs.Sweep
    extra_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    correction: bool = False
    cache_callback: Callable | None = None
    expand_dims: Sequence[str] = (CAPTURE_DIM,)

    def __post_init__(self):
        super().__post_init__()


@dataclasses.dataclass
class DelayedDataset:
    delayed: dict[str, sa.dataarrays.DelayedDataArray]
    capture: specs.SensorCapture
    extra_coords: specs.AcquisitionInfo
    extra_data: dict[str, Any]
    config: EvaluationOptions


def from_delayed(dd: DelayedDataset) -> 'xr.Dataset':
    """complete any remaining calculations, transfer from the device, and build an output dataset"""

    from .logs import log_info

    with sa.util.stopwatch(
        'package xarray',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        analysis = sa.dataarrays.package_analysis(
            dd.capture, dd.delayed, expand_dims=(CAPTURE_DIM,)
        )

    if isinstance(dd.config.sweep_spec.source, specs.NoSource):
        pass

    log_info(dd)

    with sa.util.stopwatch(
        'build coords',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        coords = build_capture_coords(
            dd.capture, dd.extra_coords, dd.config.sweep_spec.loops
        )

        analysis = analysis.assign_coords(coords)

    # don't duplicate coords as attrs
    if coords is not None:
        for name in coords.keys():
            analysis.attrs.pop(name, None)

    analysis.attrs.update(dd.config.extra_attrs)

    with sa.util.stopwatch(
        'add peripheral data',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        if dd.extra_data is not None:
            new_arrays = {}
            allowed_capture_shapes = (0, 1, analysis.capture.size)

            for k, v in dd.extra_data.items():
                ndim = getattr(v, 'ndim', 0)

                if not isinstance(v, xr.DataArray):
                    if ndim > 0:
                        dims = [CAPTURE_DIM] + [f'{k}_dim{n}' for n in range(1, ndim)]
                    else:
                        dims = []
                    v = xr.DataArray(v, dims=dims)

                if ndim == 0 or v.dims[0] != CAPTURE_DIM:
                    v = v.expand_dims({CAPTURE_DIM: analysis.capture.size})

                if v.sizes[CAPTURE_DIM] not in allowed_capture_shapes:
                    raise ValueError(
                        f'size of first axis of extra data "{k}" must be one of {allowed_capture_shapes}'
                    )

                new_arrays[k] = v

            analysis = analysis.assign(new_arrays)

    return analysis


def concat_time_dim(datasets: list['xr.Dataset'], time_dim: str) -> 'xr.Dataset':
    """concatenate captured datasets into one along a time axis.

    This can be used to e.g. transform a contiguous sequence
    of spectrogram captures into a single spectrogram.

    Preconditions:
    - all datasets share the same dimension and type.
    - time coordinates based on time_dim are uniformly spaced

    """
    pad_dims = {time_dim: (0, len(datasets[0][time_dim]) * (len(datasets) - 1))}
    ds = datasets[0].pad(pad_dims, constant_values=float('nan'))

    for data_name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            continue
        else:
            axis = var.dims[1:].index(time_dim)

        values = np.concatenate(
            [sub[data_name].isel(capture=0).data for sub in datasets], axis=axis
        )
        var.data[:] = values

    for coord_name, coord in ds.coords.items():
        if time_dim not in coord.dims:
            continue
        time_step = float(coord[1] - coord[0])
        ds[coord_name] = pd.RangeIndex(ds.sizes[coord_name]) * time_step

    return ds


def build_dataset_attrs(sweep: specs.Sweep):
    attrs: dict[str, Any] = {}
    as_dict = sweep.to_dict(unfreeze=True)

    if isinstance(sweep.description, str):
        attrs['description'] = sweep.description
    else:
        attrs['description'] = sweep.description.to_dict()

    attrs['loops'] = [l.to_dict(True) for l in sweep.loops]
    attrs['captures'] = [c.to_dict(True) for c in sweep.captures]

    for field, entry in as_dict.items():
        if field == 'adjust_captures':
            # label specs with tuple keys are not supported by zarr (at least in 2.x)
            continue
        else:
            attrs[field] = entry

    return attrs


def build_capture_coords(
    capture: specs.SensorCapture,
    extras: specs.AcquisitionInfo,
    loops: tuple[specs.LoopBase, ...],
) -> 'xr.Coordinates|None':
    captures = specs.helpers.split_capture_ports(capture)

    if len(captures) == 0:
        return None

    coords = _coords_template(
        type(captures[0]), type(extras), loops, port_count=len(captures)
    )
    coords = coords.copy(deep=True)
    changes = defaultdict(list)

    extra_coords = extras.to_dict()

    for c in captures:
        capture_entries = c.to_dict() | extra_coords
        adjust_analysis = capture_entries.pop('adjust_analysis', {})

        for name, value in capture_entries.items():
            if name not in coords:
                continue
            changes[name].append(value)

        for loop in loops:
            if loop.isin == 'analysis':
                changes[loop.field].append(adjust_analysis[loop.field])

    for name, values in changes.items():
        if name in coords:
            coords[name].data[:] = np.array(values)
        else:
            raise KeyError(f'unsupported field name {name!r}')
    return coords


def get_looped_coords(ds: 'xr.Dataset', include_repeats=False) -> list[str]:
    """return a list of coordinates that were looped"""
    if 'loops' not in ds.attrs:
        raise AttributeError('no loops metadata in data attributes')
    return [
        l['field']
        for l in ds.attrs['loops']
        if include_repeats or l['kind'] != 'repeat'
    ]


def index_dataset(
    ds: 'xr.Dataset',
    index_coords: list[str] = ['start_time', 'port'],
    *,
    chunks: int | Literal['auto'] | None = None,
) -> 'xr.Dataset':
    """Return an dataset with indexes applied across multiple axes. Apply multiple-coordinate indexing to the dataset.

    Args:
        ds: the sensor dataset to index
        index_coords:
            A list of coordinate names to index, matching index entries from the sweep
            sweep `captures` specification, or 'sweep_start_time'. The set of specified
            coordinates must uniquely specify exactly one capture from captures list.
        chunks:
            If not None, the returned dataset will be backed by dask arrays for
            multiprocessing (see `xarray.Dataset.chunk`)
    Returns:
        A xr.Dataset containing the data and coordinates of `ds` with indexing applied.
    """

    _check_coord_indexes(ds, index_coords)  # must be before ds.chunk(...)
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds.set_xindex(index_coords)


def unstack_dataset(
    ds: 'xr.Dataset',
    dim_coords: list[str] = ['start_time', 'port'],
    *,
    chunks: int | Literal['auto'] | None = None,
) -> 'xr.Dataset':
    """Unstack a dataset from a flat list of captures into multiple dimensions.

    The dimensions are `['sweep_start_time', *extra_coord_dims, 'port']`.
    `loop_vars` are introspected by the sweep loop specification from ds.attrs['loops'].

    Args:
        ds: the sensor dataset to unstack
        dim_coords:
            A list of coordinate names to transform into dimensions, which must fulfill
            the conditions given for the `index_coords` parameter of `index_dataset`.
        chunks:
            If not None, the returned dataset will be backed by dask arrays for
            multiprocessing (see `xarray.Dataset.chunk`)

    Returns:
        The unstacked dataset.
    """
    idx_ds = index_dataset(ds, dim_coords, chunks=chunks)
    unstacked = idx_ds.unstack()

    if 'sweep_start_time' in dim_coords:
        sweep_dim = 'sweep_start_time'
    elif 'sweep_index' in dim_coords:
        sweep_dim = 'sweep_index'
    else:
        sweep_dim = None

    if sweep_dim is not None:
        # the sweep dimension is for repeats. assume capture coordinates
        # are invariant and remove this dimension
        other_dims = [d for d in dim_coords if d != sweep_dim]
        for name, coord in unstacked.isel(sweep_start_time=0).coords.items():
            if name == sweep_dim:
                continue
            elif any(d in coord.dims for d in other_dims):
                unstacked[name] = coord

    return unstacked


@sa.util.lru_cache()
def _coords_template(
    capture_cls: type[specs.SensorCapture],
    info_cls: type[specs.AcquisitionInfo],
    loops: tuple[specs.LoopBase, ...],
    *,
    port_count: int,
) -> 'xr.Coordinates':
    """returns a cached xr.Coordinates object to use as a template for data results"""

    coord_vars = {}

    def make_var(field) -> xr.Variable:
        attrs, default = sa.specs.helpers.infer_coord_info(field.type)

        v = xr.Variable(
            (CAPTURE_DIM,), data=port_count * [default], attrs=attrs, fastpath=True
        )

        if isinstance(default, str):
            return v.astype(object)
        else:
            return v

    # templates for the capture and acquisition info type fields
    for spec_cls in (capture_cls, info_cls):
        for field in msgspec.structs.fields(spec_cls):
            if field.name.startswith('_') or field.name == 'adjust_analysis':
                continue
            coord_vars[field.name] = make_var(field)

    # templates for analysis loops
    for loop in loops:
        if loop.isin != 'analysis':
            continue
        if loop.field not in sa.registry.parameter_fields:
            raise KeyError(f'loop field {loop.field!r} is not supported')
        if loop.field in coord_vars:
            msg = f'analysis loop on {loop.field!r} would override capture/acquisition'
            raise KeyError(msg)
        coord_vars[loop.field] = make_var(sa.registry.parameter_fields[loop.field])

    return xr.Coordinates(coord_vars)


def _check_coord_indexes(ds, index_coords: list[str]):
    if _xarray_version() < (2024, 9, 0):
        sa.util.get_logger('analysis').warning(
            'xarray is too old to to validate coord indexes'
        )
        return

    coords_ds = ds.capture.coords.to_dataset()
    for _, coords_group in coords_ds.groupby(index_coords):
        break
    else:
        raise ValueError(f'no groups matched groupby fields {index_coords}')

    sub = coords_group.reset_coords()  # .drop_vars('start_time')
    if sub.sizes['capture'] == 1:
        # success! the specified index fields indexed a unique value
        # in the first group
        return

    # check for easy alternatives
    counts = {n: np.unique(da).tolist() for n, da in sub.data_vars.items()}
    counts = {n: c for n, c in counts.items() if len(c) == sub.sizes['capture']}

    if len(counts) == 0:
        raise ValueError(
            f'coords {index_coords} are insufficient to unstack and least two more may be needed'
        )
    else:
        suggest = set(counts.keys())
        raise ValueError(
            f'coords {index_coords} are insufficient to unstack. consider adding one of {suggest}'
        )


def _xarray_version() -> tuple[int, int, int]:
    maj, min, rel = tuple(int(v) for v in xr.__version__.split('.'))
    return maj, min, rel
