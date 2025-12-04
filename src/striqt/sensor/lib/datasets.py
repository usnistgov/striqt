"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations

import dataclasses
import logging
import typing

import msgspec

from striqt.analysis import dataarrays, Capture
from striqt.analysis.lib.dataarrays import CAPTURE_DIM, PORT_DIM  # noqa: F401
from striqt.analysis.lib.util import is_cupy_array

from . import captures, sources, specs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr

    from striqt.waveform._typing import ArrayType

else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')


RADIO_ID_NAME = 'source_id'


class EvaluationOptions(dataarrays.EvaluationOptions[dataarrays._TA], kw_only=True):
    sweep_spec: specs.Sweep
    extra_attrs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    correction: bool = False
    cache_callback: typing.Callable | None = None
    expand_dims: typing.Sequence[str] = (dataarrays.CAPTURE_DIM,)

    def __post_init__(self):
        super().__post_init__()


@dataclasses.dataclass
class DelayedDataset:
    delayed: dict[str, dataarrays.DelayedDataArray]
    capture: specs.ResampledCapture
    extra_coords: specs.AcquisitionInfo
    extra_data: dict[str, typing.Any]
    config: EvaluationOptions


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
            [sub[data_name].isel(capture=0).values for sub in datasets], axis=axis
        )
        var.values[:] = values

    for coord_name, coord in ds.coords.items():
        if time_dim not in coord.dims:
            continue
        time_step = float(coord[1] - coord[0])
        ds[coord_name] = pd.RangeIndex(ds.sizes[coord_name]) * time_step

    return ds


def _msgspec_type_to_coord_info(type_: msgspec.inspect.Type) -> tuple[dict, typing.Any]:
    """returns an (attrs, default_value) pair for the given msgspec field type"""
    from msgspec import inspect as mi

    BUILTINS = {mi.FloatType: 0.0, mi.BoolType: False, mi.IntType: 0, mi.StrType: ''}

    if not isinstance(type_, mi.Type):
        type_ = mi.type_info(type_)

    if isinstance(type_, tuple(BUILTINS.keys())):
        # dicey if subclasses show up
        return {}, BUILTINS[type(type_)]
    elif isinstance(type_, mi.CustomType):
        if issubclass(type_.cls, pd.Timestamp):
            return {}, pd.Timestamp(0)
        else:
            try:
                return {}, type_.cls()
            except Exception as ex:
                name = type_.cls.__qualname__
                raise TypeError(f'failed to make default for type {name!r}') from ex
    elif isinstance(type_, mi.Metadata):
        return type_.extra or {}, _msgspec_type_to_coord_info(type_.type)[1]
    elif isinstance(type_, mi.LiteralType):
        return {}, type(type_.values[0])
    elif isinstance(type_, mi.UnionType):
        UNION_SKIP = (mi.NoneType, mi.VarTupleType)
        types = [t for t in type_.types if not isinstance(t, UNION_SKIP)]
        if len(types) == 1:
            return _msgspec_type_to_coord_info(types[0])
        else:
            names = tuple(type(t).__qualname__ for t in types)
            raise TypeError(
                f'cannot determine xarray type for union of msgspec types {names!r}'
            )
    else:
        raise TypeError(f'unsupported msgspec field type {type_!r}')


@util.lru_cache()
def _coord_template(
    capture_cls: type[specs.ResampledCapture],
    info_cls: type[specs.AcquisitionInfo],
    port_count: int,
    **alias_dtypes: 'np.dtype',
) -> 'xr.Coordinates':
    """returns a cached xr.Coordinates object to use as a template for data results"""

    capture_fields = msgspec.structs.fields(capture_cls)
    info_fields = msgspec.structs.fields(info_cls)
    vars = {}

    for field in capture_fields + info_fields:
        attrs, default = _msgspec_type_to_coord_info(field.type)

        vars[field.name] = xr.Variable(
            (CAPTURE_DIM,),
            data=port_count * [default],
            fastpath=True,
            attrs=attrs,
        )

        if isinstance(default, str):
            vars[field.name] = vars[field.name].astype(object)

    for field, dtype in alias_dtypes.items():
        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            data=port_count * [dtype.type()],
            fastpath=True,
        ).astype(dtype)

    return xr.Coordinates(vars)


@util.lru_cache()
def _get_alias_dtypes(output: specs.Sink) -> dict[str, typing.Any]:
    aliases = output.coord_aliases

    alias_dtypes = {}
    for field, entries in aliases.items():
        alias_dtypes[field] = np.array(list(entries.keys())).dtype

    return alias_dtypes


@util.lru_cache()
def get_attrs(struct: type[specs.SpecBase], field: str) -> dict[str, str]:
    """introspect an attrs dict for xarray from the specified field in `struct`"""
    hints = typing.get_type_hints(struct, include_extras=True)

    try:
        metas = hints[field].__metadata__
    except (AttributeError, KeyError):
        return {}

    if len(metas) == 0:
        return {}
    elif len(metas) == 1 and isinstance(metas[0], msgspec.Meta):
        return metas[0].extra
    else:
        raise TypeError(
            'Annotated[] type hints must contain exactly one msgspec.Meta object'
        )


def build_dataset_attrs(sweep: specs.Sweep):
    FIELDS = [
        'analysis',
        'extensions',
        'peripherals',
        'sink',
        'source',
    ]

    attrs = {}

    if isinstance(sweep.description, str):
        attrs['description'] = sweep.description
    else:
        attrs['description'] = sweep.description.todict()

    attrs['loops'] = {l.field: l.get_points() for l in sweep.loops}

    for field in FIELDS:
        obj = getattr(sweep, field)
        new_attrs = obj.todict()
        attrs.update(new_attrs)

    return attrs


def build_capture_coords(
    capture: specs.ResampledCapture, output: specs.Sink, info: specs.AcquisitionInfo
):
    alias_dtypes = _get_alias_dtypes(output)

    if isinstance(capture.port, tuple):
        port_count = len(capture.port)
    else:
        port_count = 1

    coords = _coord_template(type(capture), type(info), port_count, **alias_dtypes)
    coords = coords.copy(deep=True)

    updates = {}

    for c in captures.split_capture_ports(capture):
        alias_hits = captures.evaluate_aliases(
            c, source_id=info.source_id, output=output
        )

        for field in coords.keys():
            if field == RADIO_ID_NAME:
                updates.setdefault(field, []).append(info.source_id)
                continue

            assert isinstance(field, str)

            try:
                value = captures.get_field_value(field, c, info, alias_hits)
            except KeyError:
                continue

            updates.setdefault(field, []).append(value)

    for field, values in updates.items():
        coords[field].values[:] = np.array(values)

    return coords


@typing.overload
def analyze(
    iq: sources.AcquiredIQ,
    options: EvaluationOptions[typing.Literal[True]],
) -> 'xr.Dataset': ...


@typing.overload
def analyze(
    iq: sources.AcquiredIQ,
    options: EvaluationOptions[typing.Literal['delayed']],
) -> DelayedDataset: ...


@typing.overload
def analyze(
    iq: sources.AcquiredIQ,
    options: EvaluationOptions[typing.Literal[False]],
) -> 'dict[str, ArrayType]': ...


@util.stopwatch('', 'analysis')
def analyze(
    iq: sources.AcquiredIQ,
    options: EvaluationOptions,
) -> 'dict[str, ArrayType] | xr.Dataset | DelayedDataset':
    """convenience function to analyze a waveform from a specification.

    The waveform may be transformed with resampling and calibration
    corrections before evaluation. Acquisition data and metadata
    are included in the returned results.
    """

    # wait to import until here to avoid a circular import
    from . import iq_corrections

    overwrite_x = not options.sweep_spec.info.reuse_iq

    assert iq.capture is not None

    with options.registry.cache_context(iq.capture, options.cache_callback):
        if options.correction:
            with util.stopwatch('resampling filter', logger_level=logging.DEBUG):
                iq = iq_corrections.resampling_correction(iq, overwrite_x=overwrite_x)
                assert iq.capture is not None

        opts = msgspec.structs.replace(options, as_xarray='delayed')
        opts = typing.cast(
            dataarrays.EvaluationOptions[typing.Literal['delayed']], opts
        )
        da_delayed = dataarrays.analyze_by_spec(
            iq, options.sweep_spec.analysis, iq.capture, opts
        )

    if 'unscaled_iq_peak' in iq.extra_data:
        peak = iq.extra_data['unscaled_iq_peak']
        if is_cupy_array(peak):
            peak = peak.get()
            iq.extra_data['unscaled_iq_peak'] = peak

    if not options.as_xarray:
        return da_delayed

    assert isinstance(iq.capture, specs.ResampledCapture)

    ds_delayed = DelayedDataset(
        delayed=da_delayed,
        capture=iq.capture,
        config=options,
        extra_coords=iq.info,
        extra_data=iq.extra_data,
    )

    if options.as_xarray == 'delayed':
        return ds_delayed
    else:
        return from_delayed(ds_delayed)


def from_delayed(dd: DelayedDataset):
    """complete any remaining calculations, transfer from the device, and build an output dataset"""

    with util.stopwatch(
        'package xarray',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        analysis = dataarrays.package_analysis(
            dd.capture, dd.delayed, expand_dims=(CAPTURE_DIM,)
        )

    unscaled_peak = dd.extra_data.get('unscaled_iq_peak', None)
    if unscaled_peak is not None and not isinstance(
        dd.config.sweep_spec.source, specs.NoSource
    ):
        descs = ','.join(f'{p:0.0f}' for p in unscaled_peak)
        logger = util.get_logger('analysis')
        logger.info(f'({descs}) dBfs ADC peak')

    with util.stopwatch(
        'build coords',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        coords = build_capture_coords(
            dd.capture, dd.config.sweep_spec.sink, dd.extra_coords
        )
        analysis = analysis.assign_coords(coords)

        # don't duplicate coords as attrs
        for name in coords.keys():
            analysis.attrs.pop(name, None)

        analysis.attrs.update(dd.config.extra_attrs)

    with util.stopwatch(
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
