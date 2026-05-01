"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

import logging
from typing import cast, Literal, overload, TYPE_CHECKING
import warnings

from ... import specs
from .. import sources, util
from . import corrections, gpu, datasets

import striqt.analysis as sa
import striqt.waveform as sw

import msgspec

if TYPE_CHECKING:
    from ..typing import Array

    import numpy as np
    import xarray as xr

else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')


SOURCE_ID_NAME = 'source_id'


@overload
def get_trigger_from_spec(setup: specs.Source, analysis: None = None) -> None: ...


@overload
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup
) -> sa.Trigger: ...


@sa.util.lru_cache()
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup | None = None
) -> sa.Trigger | None:
    name = _get_signal_trigger_name(setup)
    if name is None:
        return None

    if analysis is None and isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger

    if analysis is None:
        meas_name = sa.register.get_signal_trigger_measurement_name(name, sa.registry)
        raise ValueError(
            f'signal_trigger {meas_name!r} requires an analysis specification for {setup.signal_trigger!r}'
        )
    elif isinstance(analysis, specs.AnalysisGroup):
        return sa.Trigger.from_spec(name, analysis, registry=sa.registry)
    elif isinstance(analysis, Analysis):
        return sa.Trigger(setup.signal_trigger, analysis, registry=sa.registry)


@sa.util.lru_cache()
def _get_signal_trigger_name(setup: specs.Source) -> str | None:
    if isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger
        meas = {
            name: meas for name, meas in analysis.to_dict().items() if meas is not None
        }
        if len(meas) != 1:
            raise ValueError(
                'specify exactly one trigger for an explicit signal_trigger'
            )
        return list(meas.keys())[0]
    elif isinstance(setup.signal_trigger, str):
        return setup.signal_trigger
    else:
        return None


@overload
def analyze(
    iq: sources.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[True]],
) -> 'xr.Dataset': ...


@overload
def analyze(
    iq: sources.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal['delayed']],
) -> datasets.DelayedDataset: ...


@overload
def analyze(
    iq: sources.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[False]],
) -> 'dict[str, Array]': ...


@sa.util.stopwatch('', 'analysis')
def analyze(
    iq: sources.AcquiredIQ,
    options: datasets.EvaluationOptions,
) -> 'dict[str, Array] | xr.Dataset | datasets.DelayedDataset':
    """convenience function to analyze a waveform from a specification.

    The waveform may be transformed with resampling and calibration
    corrections before evaluation. Acquisition data and metadata
    are included in the returned results.
    """

    if isinstance(iq.capture, specs.SensorCapture):
        capture = iq.capture
    else:
        raise TypeError('iq.capture must be a SensorCapture')

    analysis = specs.helpers.adjust_analysis(
        options.sweep_spec.analysis, capture.adjust_analysis
    )

    trigger = get_trigger_from_spec(iq.source_spec, analysis)
    overwrite_x = not options.sweep_spec.options.reuse_iq

    with options.registry.cache_context(capture, options.cache_callback):
        if options.correction:
            with sa.util.stopwatch(
                'resampling filter',
                threshold=capture.duration / 3,
                logger_level=sa.util.PERFORMANCE_INFO,
            ):
                iq = corrections.correct_iq(
                    iq, signal_trigger=trigger, overwrite_x=overwrite_x
                )

        opts = msgspec.structs.replace(options, as_xarray='delayed')
        opts = cast(sa.EvaluationOptions[Literal['delayed']], opts)
        da_delayed = sa.analyze_by_spec(iq, analysis, capture, opts)

    if iq.source_spec.array_backend == 'cupy':
        for name, value in list(iq.extra_data.items()):
            if sw.is_cupy_array(value):
                iq.extra_data[name] = value.get()

    if not options.as_xarray:
        return cast('dict[str, Array]', da_delayed)

    assert isinstance(iq.capture, specs.SensorCapture)

    ds_delayed = datasets.DelayedDataset(
        delayed=da_delayed,
        capture=iq.capture,
        config=options,
        extra_coords=iq.info,
        extra_data=iq.extra_data,
    )

    if options.as_xarray == 'delayed':
        return ds_delayed
    else:
        return datasets.from_delayed(ds_delayed)


def prepare_compute(spec: specs.Sweep, skip_warmup: bool = False):
    if spec.source.array_backend == 'cupy':
        with sa.util.stopwatch('import cuda computing packages', 'sweep', 2):
            # this order is important on some versions/platforms!
            # https://github.com/numba/numba/issues/6131
            np.__version__  # reify
            import numba.cuda  # pyright: ignore
            import cupy  # type: ignore

        with sa.util.stopwatch('configure cupy', 'sweep', 1):
            sw.arrays.configure_cupy()

    warnings.filterwarnings(
        'ignore',
        category=RuntimeWarning,
        message='Mean of empty slice.*',
    )

    if skip_warmup or spec.options.skip_warmup or not gpu.sweep_touches_gpu(spec):
        yield from (None,)
        return

    from ... import bindings
    from .. import execute, resources, sinks

    spec = gpu.build_warmup_sweep(spec)

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
            yield _
