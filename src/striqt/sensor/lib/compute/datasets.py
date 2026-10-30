"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

import msgspec

import striqt.analysis as sa
from striqt.analysis.lib.dataarrays import CAPTURE_DIM
from striqt.analysis.lib.util import np, pd, xr

from ... import specs
from ..typing import TAR, Callable, Sequence

SOURCE_ID_NAME = 'source_id'


class EvaluationOptions(sa.EvaluationOptions[TAR], kw_only=True):
    sweep_spec: specs.Sweep
    extra_attrs: dict[str, Any] = msgspec.field(default_factory=dict)
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


def from_delayed(dd: DelayedDataset) -> xr.Dataset:
    """return the materialized dataset of the capture in `dd`.

    The dataset has one ``capture`` entry per port of `dd.capture`, a coordinate per
    capture field and per acquisition info field, one variable per measurement and
    per `dd.extra_data` entry, and the attrs ``dd.config.extra_attrs``. Timings log
    at DEBUG on the ``analysis`` logger.

    Raises:
        ValueError: if the first axis of an `extra_data` array has a size other
            than 0, 1, or the port count
    """

    with sa.util.stopwatch(
        'package xarray',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        analysis = sa.dataarrays.package_analysis(
            dd.capture, dd.delayed, expand_dims=(CAPTURE_DIM,)
        )

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
        for name in coords:
            analysis.attrs.pop(name, None)

    analysis.attrs.update(dd.config.extra_attrs)

    with sa.util.stopwatch(
        'add peripheral data',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        if dd.extra_data is not None:
            analysis = analysis.assign(
                extra_data_arrays(dd.extra_data, analysis.capture.size)
            )

    return analysis


def extra_data_arrays(
    extra_data: dict[str, Any], port_count: int
) -> dict[str, xr.DataArray]:
    """return the data variables of the `extra_data` of one capture.

    A scalar becomes a ``capture`` vector of `port_count` copies. An array of
    ``ndim`` axes takes the dims ``capture`` and ``<key>_dim1`` through
    ``<key>_dim<ndim-1>``. An `xarray.DataArray` keeps its dims and coordinates and
    gains a leading ``capture`` axis of `port_count` entries unless the first dim is
    already ``capture``.

    Raises:
        ValueError: if the ``capture`` axis of a value has a size other than 0, 1,
            or `port_count`
    """
    allowed_capture_shapes = (0, 1, port_count)
    arrays = {}

    for k, v in extra_data.items():
        ndim = getattr(v, 'ndim', 0)

        if not isinstance(v, xr.DataArray):
            if ndim > 0:
                dims = [CAPTURE_DIM] + [f'{k}_dim{n}' for n in range(1, ndim)]
            else:
                dims = []
            v = xr.DataArray(v, dims=dims)

        if ndim == 0 or v.dims[0] != CAPTURE_DIM:
            v = v.expand_dims({CAPTURE_DIM: port_count})

        if v.sizes[CAPTURE_DIM] not in allowed_capture_shapes:
            raise ValueError(
                f'size of first axis of extra data "{k}" must be one of {allowed_capture_shapes}'
            )

        arrays[k] = v

    return arrays


def concat_time_dim(datasets: list[xr.Dataset], time_dim: str) -> xr.Dataset:
    """concatenate captured datasets into one along a time axis.

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


def build_capture_coords(
    capture: specs.SensorCapture,
    extras: specs.AcquisitionInfo,
    loops: tuple[specs.LoopBase, ...],
) -> xr.Coordinates | None:
    captures = specs.captures.split_capture_ports(capture)

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


def get_looped_coords(ds: xr.Dataset, include_repeats=False) -> list[str]:
    """return a list of coordinates that were looped"""
    if 'loops' not in ds.attrs:
        raise AttributeError('no loops metadata in data attributes')
    return [
        l['field']
        for l in ds.attrs['loops']
        if include_repeats or l['kind'] != 'repeat'
    ]


def index_dataset(
    ds: xr.Dataset,
    index_coords: Sequence[str] = ('start_time', 'port'),
    *,
    chunks: int | Literal['auto'] | None = None,
) -> xr.Dataset:
    """return `ds` with a multi-index over `index_coords`.

    Args:
        ds: the sensor dataset to index
        index_coords: capture field names, or ``'sweep_start_time'``; together the
            coordinates must select exactly one capture
        chunks: a chunk size or ``'auto'`` backs the result with dask arrays
            (`xarray.Dataset.chunk`); None keeps the arrays of `ds`

    Returns:
        the data and coordinates of `ds` with the multi-index set

    Raises:
        ValueError: if `index_coords` do not select exactly one capture
    """

    index_coords = list(index_coords)
    _check_coord_indexes(ds, index_coords)  # must be before ds.chunk(...)
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds.set_xindex(index_coords)


def unstack_dataset(
    ds: xr.Dataset,
    dim_coords: Sequence[str] = ('start_time', 'port'),
    *,
    chunks: int | Literal['auto'] | None = None,
) -> xr.Dataset:
    """Unstack a dataset from a flat list of captures into multiple dimensions.

    The dimensions are `['sweep_start_time', *extra_coord_dims, 'port']`.

    Args:
        ds: the sensor dataset to unstack
        dim_coords:
            A list of coordinate names to transform into dimensions, which must fulfill
            the conditions given for the `index_coords` parameter of `index_dataset`.
        chunks: a chunk size or ``'auto'`` backs the result with dask arrays
            (`xarray.Dataset.chunk`); None keeps the arrays of `ds`

    Returns:
        The unstacked dataset.
    """
    dim_coords = list(dim_coords)
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
) -> xr.Coordinates:
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

    # templates for the capture and acquisition info type fields; adjust_analysis is
    # a dict and signal_trigger a Trigger, and neither is a coordinate
    skipped = {'adjust_analysis', 'signal_trigger'}
    for spec_cls in (capture_cls, info_cls):
        for field in msgspec.structs.fields(spec_cls):
            if field.name.startswith('_') or field.name in skipped:
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
        # the specified index fields select a unique capture in the first group
        return

    # suggest coordinates that would make the index unique
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


# %% Store templates without data
MIN_STRING_WIDTH = 16
TIME_ENCODING = {
    'units': 'nanoseconds since 1970-01-01',
    'dtype': 'int64',
    '_FillValue': np.iinfo(np.int64).min,
}


@dataclasses.dataclass(frozen=True)
class VariableGrid:
    """the output geometry of one measurement over every capture of a sweep."""

    name: str
    """the measurement name, which is the data variable name"""
    dims: tuple[str, ...]
    """the dims of one row, after the ``capture`` axis"""
    shape: tuple[int, ...]
    """the length of each of `dims` on the union grid"""
    dtype: np.dtype
    """the storage dtype: the registered dtype, promoted to a float when `mixed`
    and the registered dtype is an integer or bool"""
    coords: dict[str, np.ndarray]
    """the coordinate values of each of `dims`, the union over the captures"""
    coord_attrs: dict[str, dict[str, Any]]
    """the attrs of each coordinate, without the attrs that differ between captures"""
    attrs: dict[str, Any]
    """the registered and spec attrs, without the attrs that differ between captures"""
    mixed: bool
    """True when a dim has different coordinate values on 2 captures"""


def _equivalent(a: Any, b: Any) -> bool:
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except (TypeError, ValueError):
        return False


def drop_conflicts(attrs_list: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """return the union of the attrs, without a key that differs between 2 mappings.

    This is the ``combine_attrs='drop_conflicts'`` rule of `xarray.concat`: a key
    present in some mappings is kept with its value, and a key with 2 unequal
    values is removed.
    """
    result: dict[str, Any] = {}
    dropped: set[str] = set()
    for attrs in attrs_list:
        for key, value in attrs.items():
            if key in dropped:
                continue
            if key not in result:
                result[key] = value
            elif not _equivalent(result[key], value):
                del result[key]
                dropped.add(key)
    return result


def sweep_row_count(sweep: specs.Sweep, source_id: str) -> int:
    """return the row count of a store of `sweep`: one per port of every expanded capture.

    The captures are those of `specs.sequencing.repeat_captures`, so a leading
    `Repeat` loop multiplies the count.
    """
    captures = specs.sequencing.repeat_captures(sweep, source_id)
    return sum(len(specs.captures.split_capture_ports(c)) for c in captures)


def _distinct_analyses(
    sweep: specs.Sweep, captures: Sequence[specs.SensorCapture]
) -> list[tuple[specs.SensorCapture, specs.AnalysisGroup]]:
    pairs: dict[tuple[Any, Any], None] = {}
    for capture in captures:
        analysis = specs.sequencing.adjust_analysis(
            sweep.analysis, capture.adjust_analysis
        )
        pairs.setdefault((capture, analysis), None)
    return [pair for pair in pairs]


def _fill_value(dtype: np.dtype) -> Any:
    if dtype.kind in 'fc':
        return np.nan
    if dtype.kind in 'mM':
        return np.datetime64('NaT')
    return 0


def measurement_grids(
    sweep: specs.Sweep, captures: Sequence[specs.SensorCapture]
) -> dict[str, VariableGrid]:
    """return the output geometry of each enabled measurement over `captures`.

    The coordinate factories of each measurement run once per distinct pair of
    capture and adjusted analysis. A dim with equal coordinate values on every pair
    keeps those values. Otherwise the coordinate is the sorted union of the values, which is
    the outer join of `xarray.concat`, and the grid is `mixed`.

    Raises:
        NotImplementedError: if a coordinate with more than one dim differs between
            captures, or if a dim of a measurement has no coordinate factory
    """
    pairs = _distinct_analyses(sweep, captures)
    names = [name for name, kws in sweep.analysis.to_dict().items() if kws]
    grids: dict[str, VariableGrid] = {}

    for name in names:
        per_pair = []
        for capture, analysis in pairs:
            # the registry wrapper hands the validated spec to the factories and
            # the attrs, so a Fraction default reads back as a string
            spec = getattr(analysis, name).validate()
            info = sa.registry[type(spec)]
            coords = sa.dataarrays.measurement_coords(info, capture, spec)
            per_pair.append((coords, spec))
        info = sa.registry[type(per_pair[0][1])]
        dims = sa.dataarrays.measurement_dims(info)

        union: dict[str, np.ndarray] = {}
        coord_attrs: dict[str, dict[str, Any]] = {}
        mixed = False
        for dim in dims:
            if dim not in per_pair[0][0]:
                raise NotImplementedError(
                    f'dimension {dim!r} of measurement {name!r} has no coordinate factory'
                )
            arrays = [coords[dim][0] for coords, _ in per_pair]
            if all(_equivalent(arrays[0], a) for a in arrays[1:]):
                union[dim] = arrays[0]
            elif any(a.ndim != 1 for a in arrays):
                raise NotImplementedError(
                    f'coordinate {dim!r} of measurement {name!r} differs between '
                    'captures and has more than one dim'
                )
            else:
                union[dim] = np.unique(np.concatenate(arrays))
                mixed = True
            coord_attrs[dim] = drop_conflicts([
                coords[dim][1] for coords, _ in per_pair
            ])

        dtype = np.dtype(info.dtype)
        if mixed and dtype.kind in 'biu':
            dtype = np.promote_types(dtype, np.float32)
        attrs = drop_conflicts([
            {**info.attrs, **spec.to_dict()} for _, spec in per_pair
        ])
        grids[name] = VariableGrid(
            name=name,
            dims=dims,
            shape=tuple(len(union[d]) for d in dims),
            dtype=dtype,
            coords=union,
            coord_attrs=coord_attrs,
            attrs=attrs,
            mixed=mixed,
        )

    return grids


def row_index(
    grid: VariableGrid, capture: specs.SensorCapture, spec: sa.specs.Analysis
) -> tuple[slice | np.ndarray, ...]:
    """return the position of the coordinates of one capture on the union grid.

    Each entry is ``slice(None)`` when the capture's coordinate values equal the
    union, and otherwise an integer index array, one per value.

    Raises:
        ValueError: if a coordinate value of the capture is not on the union grid
    """
    info = sa.registry[type(spec)]
    coords = sa.dataarrays.measurement_coords(info, capture, spec)
    index: list[slice | np.ndarray] = []
    for dim in grid.dims:
        values = coords[dim][0]
        union = grid.coords[dim]
        if _equivalent(values, union):
            index.append(slice(None))
            continue
        if union.dtype.kind in 'OUS':
            lookup = {v: i for i, v in enumerate(union.tolist())}
            try:
                pos = np.array([lookup[v] for v in values.tolist()], dtype=np.intp)
            except KeyError as ex:
                raise ValueError(
                    f'coordinate value {ex.args[0]!r} of {dim!r} is not on the grid'
                ) from None
        else:
            pos = np.searchsorted(union, values)
            pos = np.minimum(pos, len(union) - 1)
            if not _equivalent(union[pos], values):
                raise ValueError(f'coordinate values of {dim!r} are not on the grid')
        index.append(pos)
    return tuple(index)


def scatter_rows(
    grid: VariableGrid, result: Any, index: tuple[slice | np.ndarray, ...]
) -> np.ndarray:
    """return the rows of one capture placed on the union grid of `grid`.

    `result` has one leading axis of rows (ports) and the dims of `grid` after it.
    When every entry of `index` is ``slice(None)``, `result` is returned as an
    array without a copy. Otherwise the rows are placed into a block of the grid
    shape filled with NaN (float and complex), NaT (datetime), or 0.
    """
    result = np.asarray(result)
    if all(isinstance(i, slice) for i in index):
        return result
    block = np.full((result.shape[0], *grid.shape), _fill_value(grid.dtype), grid.dtype)
    axes = [
        np.arange(n) if isinstance(i, slice) else i for i, n in zip(index, grid.shape)
    ]
    block[(slice(None), *np.ix_(*axes))] = result
    return block


def _encode_times(values: np.ndarray) -> np.ndarray:
    """return `values` as int64 nanoseconds since the epoch, NaT as the int64 minimum."""
    arr = np.asarray(values)
    if arr.dtype.kind != 'M':
        # numpy's conversion of Timestamp objects truncates to microseconds
        arr = pd.to_datetime(arr.ravel().tolist()).values.reshape(arr.shape)
    return arr.astype('datetime64[ns]').view('int64')


def capture_coord_storage(
    values: np.ndarray, known: Sequence[Any] = ()
) -> tuple[np.dtype, dict[str, Any], Callable[[np.ndarray], np.ndarray]]:
    """return the storage dtype, encoding, and row encoder of a per-capture coordinate.

    Strings are stored as fixed-width unicode wide enough for `values` and `known`,
    and at least `MIN_STRING_WIDTH` characters. Timestamps are stored as int64
    nanoseconds with `TIME_ENCODING`. Any other dtype is stored unchanged.

    Returns:
        the storage dtype, the xarray encoding of the variable (empty when none is
        needed), and a function from the values of one capture to storage values
    """
    sample = np.asarray(values)
    if sample.dtype.kind == 'O':
        items = sample.tolist()
        if any(isinstance(v, (pd.Timestamp, np.datetime64)) for v in items):
            sample = np.array(items, dtype='datetime64[ns]')
        else:
            sample = np.array(items)
            if sample.dtype.kind == 'O':
                sample = np.array([str(v) for v in items])
    if sample.dtype.kind in 'US':
        widths = [len(str(v)) for v in (*sample.tolist(), *known)]
        width = max((*widths, MIN_STRING_WIDTH))
        dtype = np.dtype(f'<U{width}')
        return dtype, {}, lambda v: np.asarray(v).astype(dtype)
    if sample.dtype.kind == 'M':
        return np.dtype('datetime64[ns]'), dict(TIME_ENCODING), _encode_times
    return sample.dtype, {}, lambda v: np.asarray(v, dtype=sample.dtype)


def capture_coords_template(
    sweep: specs.Sweep,
    captures: Sequence[specs.SensorCapture],
    source_id: str,
    info_cls: type[specs.AcquisitionInfo],
) -> tuple[xr.Coordinates, dict[str, dict[str, Any]], dict[str, Callable]]:
    """return the zero-row per-capture coordinates of a sweep and the storage rules.

    The coordinate set is that of `build_capture_coords` for the capture type of
    `captures` and `info_cls`. String widths cover every value of `captures` and
    `source_id`.

    Returns:
        the coordinates with no entries along ``capture``, the xarray encoding per
        coordinate that needs one, and the row encoder per coordinate
    """
    template = _coords_template(type(captures[0]), info_cls, sweep.loops, port_count=1)
    known: dict[str, list[Any]] = defaultdict(list)
    for capture in captures:
        for name, value in capture.to_dict().items():
            if name in template:
                known[name].extend(value if isinstance(value, tuple) else [value])
    known[SOURCE_ID_NAME].append(source_id)

    coords: dict[str, Any] = {}
    encoding: dict[str, dict[str, Any]] = {}
    encoders: dict[str, Callable] = {}
    for key, var in template.items():
        name = str(key)
        dtype, enc, encoder = capture_coord_storage(var.values, known.get(name, ()))
        coords[name] = (var.dims, np.empty((0,), dtype=dtype), dict(var.attrs))
        if enc:
            encoding[name] = enc
        encoders[name] = encoder
    return xr.Coordinates(coords), encoding, encoders


def sweep_template(
    sweep: specs.Sweep,
    source_id: str,
    info_cls: type[specs.AcquisitionInfo],
    *,
    attrs: Mapping[str, Any],
) -> tuple[
    xr.Dataset, dict[str, VariableGrid], dict[str, dict[str, Any]], dict[str, Callable]
]:
    """return the zero-row dataset of a sweep, built from the specs without IQ.

    The data variables are the enabled measurements on the union grids, with the
    fixed coordinates and attrs of `measurement_grids`. The per-capture coordinates
    are those of `capture_coords_template`. The dataset attrs are `attrs` over the
    capture fields that are equal on every capture and are not coordinates.

    Returns:
        the dataset, the grid of each data variable, the xarray encoding per
        coordinate that needs one, and the row encoder per per-capture coordinate

    Raises:
        NotImplementedError: as `measurement_grids`
        ValueError: if 2 measurements share a dim with different union coordinates
    """
    captures = specs.sequencing.loop_captures(sweep, source_id=source_id)
    grids = measurement_grids(sweep, captures)

    coords: dict[str, Any] = {}
    data_vars: dict[str, Any] = {}
    for name, grid in grids.items():
        for dim in grid.dims:
            entry = (dim, grid.coords[dim], grid.coord_attrs[dim])
            if dim in coords and not _equivalent(coords[dim][1], entry[1]):
                raise ValueError(f'measurements disagree on the coordinate {dim!r}')
            coords.setdefault(dim, entry)
        data_vars[name] = (
            (CAPTURE_DIM, *grid.dims),
            np.empty((0, *grid.shape), dtype=grid.dtype),
            grid.attrs,
        )
    capture_coords, encoding, encoders = capture_coords_template(
        sweep, captures, source_id, info_cls
    )

    capture_attrs = drop_conflicts([
        {
            **c.to_dict(),
            **(
                {'analysis_filter': c.analysis_filter.to_dict()}
                if isinstance(c, sa.specs.FilteredCapture)
                else {}
            ),
        }
        for c in captures
    ])
    for name in capture_coords:
        capture_attrs.pop(str(name), None)

    ds = xr.Dataset(data_vars, coords=coords, attrs={**capture_attrs, **attrs})
    ds = ds.assign_coords(capture_coords)
    return ds, grids, encoding, encoders
