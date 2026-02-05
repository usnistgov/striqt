"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

from collections import defaultdict
import dataclasses
import logging
import typing
import warnings

from .. import specs
from . import sources, util

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.dataarrays import CAPTURE_DIM, PORT_DIM  # noqa: F401

import msgspec

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


SOURCE_ID_NAME = 'source_id'


class EvaluationOptions(sa.EvaluationOptions[sa.dataarrays._TA], kw_only=True):
    sweep_spec: specs.Sweep
    extra_attrs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    correction: bool = False
    cache_callback: typing.Callable | None = None
    expand_dims: typing.Sequence[str] = (CAPTURE_DIM,)

    def __post_init__(self):
        super().__post_init__()


@dataclasses.dataclass
class DelayedDataset:
    delayed: dict[str, sa.dataarrays.DelayedDataArray]
    capture: specs.SensorCapture
    extra_coords: specs.SourceCoordinates
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


@sa.util.lru_cache()
def _coord_template(
    capture_cls: type[specs.SensorCapture],
    info_cls: type[specs.SourceCoordinates],
    port_count: int,
) -> 'xr.Coordinates':
    """returns a cached xr.Coordinates object to use as a template for data results"""

    vars = {}

    for spec_cls in (capture_cls, info_cls):
        attrs, defaults = specs.helpers.field_template_values(spec_cls)

        for name in attrs.keys():
            if name.startswith('_') or name == 'adjust_analysis':
                continue

            vars[name] = xr.Variable(
                (CAPTURE_DIM,),
                data=port_count * [defaults[name]],
                fastpath=True,
                attrs=attrs[name],
            )

            if isinstance(defaults[name], str):
                vars[name] = vars[name].astype(object)

    return xr.Coordinates(vars)


def build_dataset_attrs(sweep: specs.Sweep):
    attrs: dict[str, typing.Any] = {}
    as_dict = sweep.to_dict(skip_private=True, unfreeze=True)

    if isinstance(sweep.description, str):
        attrs['description'] = sweep.description
    else:
        attrs['description'] = sweep.description.to_dict()

    attrs['loops'] = [l.to_dict(True) for l in sweep.loops]
    attrs['captures'] = [c.to_dict(True) for c in sweep.captures]

    for field, entry in as_dict.items():
        if field in attrs or field == 'adjust_captures':
            # label specs with tuple keys are not supported by zarr (at least in 2.x)
            continue
        else:
            attrs[field] = entry

    return attrs


def build_capture_coords(
    capture: specs.SensorCapture,
    info: specs.SourceCoordinates,
) -> 'xr.Coordinates|None':
    captures = specs.helpers.split_capture_ports(capture)

    if len(captures) == 0:
        return None

    coords = _coord_template(type(captures[0]), type(info), len(captures))
    coords = coords.copy(deep=True)
    changes = defaultdict(list)

    info_dict = info.to_dict()

    for c in captures:
        capture_entries = info_dict | c.to_dict(skip_private=True)
        del capture_entries['adjust_analysis']

        for field, value in capture_entries.items():
            changes[field].append(value)

    for field, values in changes.items():
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


@sa.util.stopwatch('', 'analysis')
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

    capture = iq.capture

    assert isinstance(capture, specs.SensorCapture)

    with options.registry.cache_context(capture, options.cache_callback):
        if options.correction:
            with sa.util.stopwatch('resampling filter', logger_level=logging.DEBUG):
                iq = resampling.resampling_correction(iq, overwrite_x=overwrite_x)

        opts = msgspec.structs.replace(options, as_xarray='delayed')
        opts = typing.cast(sa.EvaluationOptions[typing.Literal['delayed']], opts)
        analysis = specs.helpers.adjust_analysis(
            options.sweep_spec.analysis, capture.adjust_analysis
        )
        da_delayed = sa.analyze_by_spec(iq, analysis, capture, opts)

    if iq.source_spec.array_backend == 'cupy':
        for name, value in list(iq.extra_data.items()):
            if sw.util.is_cupy_array(value):
                iq.extra_data[name] = value.get()

    if not options.as_xarray:
        return da_delayed

    assert isinstance(iq.capture, specs.SensorCapture)

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


def _adc_overload_message(
    extra_data: dict[str, typing.Sequence[float]], capture: specs.SensorCapture
) -> str | None:
    if 'adc_headroom' in extra_data and isinstance(capture, specs.SoapyCapture):
        headroom = extra_data['adc_headroom']
        caps = specs.helpers.split_capture_ports(capture)
    else:
        return None

    overload_ports = []
    for c, hr in zip(caps, headroom):
        if hr > 0:
            continue
        else:
            assert not isinstance(c.center_frequency, tuple)
        msg = f'port {c.port} ({c.center_frequency / 1e6:0.0f} MHz)'
        overload_ports.append(msg)

    if len(overload_ports) > 0:
        return 'adc overload on ' + ', '.join(overload_ports)
    else:
        return None


def _if_overload_message(
    extra_data: dict[str, typing.Sequence[float]],
    capture: _TC,
    sweep_spec: specs.Sweep[_TS, _TP, _TC],
) -> str | None:
    if 'if_headroom' in extra_data:
        if_headroom = extra_data['if_headroom']
    else:
        return None

    if not isinstance(capture, specs.SoapyCapture):
        return None
    else:
        captures = typing.cast(tuple[specs.SoapyCapture, ...], sweep_spec.captures)

    gains = specs.helpers.max_by_frequency('gain', captures, sweep_spec.loops)
    caps = specs.helpers.split_capture_ports(capture)

    ol_cases = {}
    for c, hr in zip(caps, if_headroom):
        # estimate IM3 levels in other channels
        ol_cases.setdefault(c.port, set())

        for fc, gain in gains[c.port].items():
            im3_headroom = hr + (2 / 3 * (c.gain - gain))

            if im3_headroom > 0:
                continue

            ol_cases[c.port].add(fc)

    ol_labels = []
    for port, freqs in ol_cases.items():
        if len(freqs) > 0:
            freqs_MHz = ', '.join([f'{f / 1e6:0.0f}' for f in sorted(freqs)])
            ol_labels.append(f'port {port} (onto {freqs_MHz} MHz)')

    if len(ol_labels) > 0:
        return 'if overload at ' + ' and '.join(ol_labels)
    else:
        return None


def from_delayed(dd: DelayedDataset):
    """complete any remaining calculations, transfer from the device, and build an output dataset"""

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

    # log overload messages
    overload_msgs = []
    adc_ol_info = _adc_overload_message(dd.extra_data, dd.capture)
    if adc_ol_info is not None:
        overload_msgs.append(adc_ol_info)
    if_ol_info = _if_overload_message(dd.extra_data, dd.capture, dd.config.sweep_spec)
    if if_ol_info is not None:
        overload_msgs.append(if_ol_info)
    if len(overload_msgs) > 0:
        sa.util.get_logger('analysis').warning(', '.join(overload_msgs))

    with sa.util.stopwatch(
        'build coords',
        'analysis',
        threshold=10e-3,
        logger_level=logging.DEBUG,
    ):
        coords = build_capture_coords(dd.capture, dd.extra_coords)
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


def sweep_touches_gpu(sweep: specs.Sweep) -> bool:
    """returns True if the sweep would benefit from the GPU"""

    if sweep.source.calibration is not None:
        return True

    analysis_dict = sweep.analysis.to_dict()
    if tuple(analysis_dict.keys()) != (sa.measurements.iq_waveform.__name__,):
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
    args = 'sweep', 1, sa.util.INFO

    if cupy:
        with sa.util.stopwatch('import cuda computing packages', *args):
            # this order is important on some versions/platforms!
            # https://github.com/numba/numba/issues/6131
            np.__version__  # reify
            util.safe_import('numba.cuda', service=False)
            sw.util.cp = util.safe_import('cupy', service=False)
            # safe_import('cupyx')
            # safe_import('cupyx.scipy')

    with sa.util.stopwatch('import general computing packages', *args):
        util.safe_import('scipy')
        util.safe_import('numpy')
        util.safe_import('pandas')
        util.safe_import('xarray')
        util.safe_import('numba')

    if cupy:
        sa.util.configure_cupy()

def prepare_compute(input_spec: specs.Sweep, skip_warmup: bool = False):
    import_compute_modules(cupy=input_spec.source.array_backend == 'cupy')

    warnings.filterwarnings(
        'ignore',
        category=RuntimeWarning,
        message='Mean of empty slice.*',
    )

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
