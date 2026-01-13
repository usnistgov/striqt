"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

import dataclasses
import logging
import typing

import msgspec

from striqt.analysis import dataarrays, measurements
from striqt.analysis.lib.dataarrays import CAPTURE_DIM, PORT_DIM  # noqa: F401
from striqt.analysis.lib.util import is_cupy_array
from ..specs import helpers
from .. import specs
from striqt.waveform import util as waveform_util
from . import sources, util

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from typing_extensions import TypeAlias

    from striqt.waveform._typing import ArrayType
    from ..specs import _TS, _TC, _TP

    WarmupSweep: TypeAlias = specs.Sweep[specs.NoSource, specs.NoPeripherals, _TC]

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
            [sub[data_name].isel(capture=0).data for sub in datasets], axis=axis
        )
        var.data[:] = values

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
def _get_alias_dtypes(desc: specs.Description) -> dict[str, typing.Any]:
    aliases = desc.coord_aliases

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

    attrs: dict[str, typing.Any] = {}

    if isinstance(sweep.description, str):
        attrs['description'] = sweep.description
    else:
        attrs['description'] = sweep.description.to_dict()

    attrs['loops'] = {l.field: l.get_points() for l in sweep.loops}

    for field in FIELDS:
        obj = getattr(sweep, field)
        new_attrs = obj.to_dict()
        attrs.update(new_attrs)

    return attrs


def build_capture_coords(
    capture: specs.ResampledCapture,
    desc: specs.Description,
    info: specs.AcquisitionInfo,
):
    alias_dtypes = _get_alias_dtypes(desc)

    if isinstance(capture.port, tuple):
        port_count = len(capture.port)
    else:
        port_count = 1

    coords = _coord_template(type(capture), type(info), port_count, **alias_dtypes)
    coords = coords.copy(deep=True)

    updates = {}

    for c in helpers.split_capture_ports(capture):
        alias_hits = helpers.evaluate_aliases(c, source_id=info.source_id, desc=desc)

        for field in coords.keys():
            if field == RADIO_ID_NAME:
                updates.setdefault(field, []).append(info.source_id)
                continue

            assert isinstance(field, str)

            try:
                value = helpers.get_field_value(field, c, info, alias_hits)
            except KeyError:
                continue

            updates.setdefault(field, []).append(value)

    for field, values in updates.items():
        coords[field].data[:] = np.array(values)

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
    from . import resampling

    overwrite_x = not options.sweep_spec.options.reuse_iq

    assert iq.capture is not None

    with options.registry.cache_context(iq.capture, options.cache_callback):
        if options.correction:
            with util.stopwatch('resampling filter', logger_level=logging.DEBUG):
                iq = resampling.resampling_correction(iq, overwrite_x=overwrite_x)
                assert iq.capture is not None

        opts = msgspec.structs.replace(options, as_xarray='delayed')
        opts = typing.cast(
            dataarrays.EvaluationOptions[typing.Literal['delayed']], opts
        )
        da_delayed = dataarrays.analyze_by_spec(
            iq, options.sweep_spec.analysis, iq.capture, opts
        )

    if 'adc_overload' in iq.extra_data:
        peak = iq.extra_data['adc_overload']
        if is_cupy_array(peak):
            peak = peak.get()
            iq.extra_data['adc_overload'] = peak

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

    if isinstance(dd.config.sweep_spec.source, specs.NoSource):
        pass
    elif 'adc_overload' not in dd.extra_data:
        pass
    elif any(dd.extra_data['adc_overload']):
        overloads = dd.extra_data['adc_overload']
        overload_ports = [i for i, flag in enumerate(overloads) if flag]
        logger = util.get_logger('analysis')
        logger.warning(f'ADC overload on port {overload_ports}')

    with util.stopwatch(
        'build coords',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        coords = build_capture_coords(
            dd.capture, dd.config.sweep_spec.description, dd.extra_coords
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


def sweep_touches_gpu(sweep: specs.Sweep) -> bool:
    """returns True if the sweep would benefit from the GPU"""

    if sweep.source.calibration is not None:
        return True

    analysis_dict = sweep.analysis.to_dict()
    if tuple(analysis_dict.keys()) != (measurements.iq_waveform.__name__,):
        # everything except iq_clipping requires a warmup
        return True

    # check the inner loop (explicit) values
    for capture in sweep.captures:
        if capture.host_resample or capture.analysis_bandwidth is not None:
            return True

    # check any values specified in outer loops
    for loop in sweep.loops:
        if loop.field == 'host_resample' and True in loop.get_points():
            return True
        elif loop.field == 'analysis_bandwidth' and not all(loop.get_points()):
            return True

    return False


def build_warmup_sweep(sweep: specs.Sweep[_TS, _TP, _TC]) -> WarmupSweep:
    """derive a warmup sweep specification derived from sweep.

    This is meant to trigger expensive python imports and warm up JIT caches. The goal
    is to avoid analysis slowdowns later during the execution of the first captures.

    The derived sweep has the following characteristics:
        - It is bound to NoPeripheral and NoSink
        - It contains only one capture, with no loops
    """

    from .bindings import mock_binding
    from ..bindings import warmup

    # check loops to select an easy case for warmup
    updates = {}
    ports = []
    for loop in sweep.loops:
        if loop.field in ('duration', 'sample_rate', 'backend_sample_rate'):
            updates[loop.field] = min(loop.get_points())
        if loop.field == 'host_resample':
            updates[loop.field] = any(loop.get_points())
        if loop.field == 'port':
            ports.extend(loop.get_points())

    captures = [c.replace(**updates) for c in sweep.captures]
    by_size = {round(c.sample_rate * c.duration): c for c in captures}

    num_rx_ports = 0
    for port in ports + [c.port for c in captures]:
        if port is None:
            continue
        if isinstance(port, tuple):
            n = max(port)
        else:
            n = port
        if n > num_rx_ports:
            num_rx_ports = n

    if len(captures) > 1:
        captures = [by_size[min(by_size.keys())]]

    b = mock_binding(sweep.__bindings__, 'warmup', register=False)

    source = warmup.schema.source(
        num_rx_ports=num_rx_ports,
        master_clock_rate=sweep.source.master_clock_rate,
        trigger_strobe=None,
        signal_trigger=sweep.source.signal_trigger,
    )

    return b.sweep_spec(
        source=source,
        captures=tuple(captures),
        loops=(),
        analysis=sweep.analysis,
        sink=sweep.sink,
    )


def import_compute_modules(cupy=False):
    """import expensive compute modules.

    If used at all, this needs to be run from _at least_ the main thread.
    The use of util.safe_imports means that child threads will wait for
    the main thread to import these.
    """
    if cupy:
        # this order is important on some versions/platforms!
        # https://github.com/numba/numba/issues/6131
        util.safe_import('numba.cuda')
        waveform_util.cp = util.safe_import('cupy')
        util.configure_cupy()
        # safe_import('cupyx')
        # safe_import('cupyx.scipy')

    util.safe_import('scipy')
    util.safe_import('numpy')
    util.safe_import('pandas')
    util.safe_import('xarray')
    util.safe_import('numba')


def prepare_compute(input_spec: specs.Sweep, skip_warmup: bool = False):
    import_compute_modules(cupy=input_spec.source.array_backend == 'cupy')

    if skip_warmup or input_spec.options.skip_warmup:
        return

    from .. import bindings
    from . import execute, resources, sinks

    spec = build_warmup_sweep(input_spec)

    if len(spec.captures) == 0:
        return

    res = resources.Resources(
        sweep_spec=spec,
        source=bindings.warmup.source.from_spec(spec.source),
        peripherals=bindings.warmup.peripherals(spec),
        sink=sinks.NoSink(spec),
        calibration=None,
        alias_func=None,
    )

    with res['source']:
        sweep = execute.iterate_sweep(res, always_yield=True, calibration=None)

        for _ in sweep:
            pass
