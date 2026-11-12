"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import xr

from ... import specs
from . import corrections, datasets

if TYPE_CHECKING:
    from collections.abc import Generator

    from .datasets import DelayedDataset


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

    # the analysis spec inside a struct-form signal_trigger replaces the sweep analysis
    if isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger

    if analysis is None:
        meas_name = sa.register.get_signal_trigger_measurement_name(name, sa.registry)
        raise ValueError(
            f'signal_trigger {meas_name!r} requires an analysis specification for {setup.signal_trigger!r}'
        )
    elif isinstance(analysis, specs.AnalysisGroup):
        return sa.Trigger.from_spec(name, analysis, registry=sa.registry)
    elif isinstance(analysis, sa.specs.Analysis):
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
        return next(iter(meas.keys()))
    elif isinstance(setup.signal_trigger, str):
        return setup.signal_trigger
    else:
        return None


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[True]],
) -> xr.Dataset: ...


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal['delayed']],
) -> datasets.DelayedDataset: ...


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[False]],
) -> dict[str, sa.dataarrays.DelayedDataArray]: ...


@sa.util.stopwatch('', 'analysis')
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions,
) -> dict[str, sa.dataarrays.DelayedDataArray] | xr.Dataset | datasets.DelayedDataset:
    """run the analysis of ``options.sweep_spec`` on `iq`.

    The analysis is ``options.sweep_spec.analysis`` adjusted by
    ``iq.capture.adjust_analysis``. The signal trigger is resolved from that adjusted
    group. ``options.select_analysis``, when set, receives the capture and the
    adjusted group and returns the group to evaluate. A measurement that prefers
    aligned input aligns whether or not the returned group enables the trigger
    measurement. The resampling stage consumes `iq.x` in place unless
    ``options.sweep_spec.options.reuse_iq`` is set. cupy arrays in `iq.extra_data`
    are replaced in place by numpy copies. ``options.cache_callback`` runs each time
    a measurement fills a cache.

    Returns:
        the result selected by `options.as_xarray`: a dict of `DelayedDataArray`
        by measurement name for False, a `DelayedDataset` for ``'delayed'``, or an
        `xr.Dataset` for True. The `DelayedDataset` and the `xr.Dataset` carry the
        capture fields and acquisition info as coordinates and `iq.extra_data` as
        variables.

    Raises:
        TypeError: if `iq.capture` is not a `SensorCapture`, or if `select_analysis`
            returns an object that is not an instance of the adjusted group type
        ValueError: if ``iq.source_spec.signal_trigger`` names a measurement that
            the adjusted analysis does not enable
    """

    if isinstance(iq.capture, specs.SensorCapture):
        capture = iq.capture
    else:
        raise TypeError('iq.capture must be a SensorCapture')

    analysis = specs.sequencing.adjust_analysis(
        options.sweep_spec.analysis, capture.adjust_analysis
    )

    trigger = get_trigger_from_spec(iq.source_spec, analysis)
    overwrite_x = not options.sweep_spec.options.reuse_iq

    if options.select_analysis is not None:
        selected = options.select_analysis(capture, analysis)
        if not isinstance(selected, type(analysis)):
            raise TypeError(
                f'select_analysis returned {type(selected).__name__}, not '
                f'{type(analysis).__name__}'
            )
        analysis = selected

    with options.registry.cache_context(capture, options.cache_callback):
        iq = corrections.attach_corrections(
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
        return da_delayed

    assert isinstance(iq.capture, specs.SensorCapture)

    ds_delayed = datasets.DelayedDataset(
        delayed=da_delayed,
        capture=iq.capture,
        config=options,
        extra_coords=iq.info,
        extra_data=iq.extra_data,
        pending_extra_data=iq.pending_extra_data,
    )

    if options.as_xarray == 'delayed':
        return ds_delayed
    else:
        return datasets.from_delayed(ds_delayed)


def prepare_compute(
    spec: specs.Sweep[Any, Any, Any, Any], skip_warmup: bool = False
) -> Generator[xr.Dataset | DelayedDataset | None]:
    """yield the results of the warmup sweep of `spec`.

    The cupy imports and configuration run before the first yield when
    ``spec.source.array_backend`` is ``'cupy'``. A `RuntimeWarning` matching
    ``Mean of empty slice`` is filtered process-wide.

    Args:
        skip_warmup: True selects no warmup sweep

    Yields:
        None once when `skip_warmup` is set or `warmup_sweep_for` returns None.
        Otherwise each yield of a `SweepRunner` over the warmup sweep with
        `always_yield` set, and nothing when the warmup sweep has no captures.
    """
    if spec.source.array_backend == 'cupy':
        with sa.util.stopwatch('configure cupy', 'sweep', 1):
            sw.arrays.configure_cupy()

    warnings.filterwarnings(
        'ignore',
        category=RuntimeWarning,
        message='Mean of empty slice.*',
    )

    from ..actions import sweep as sweep_action, warmup

    if skip_warmup:
        warm = None
    else:
        warm = warmup.warmup_sweep_for(spec)

    if warm is None:
        yield from (None,)
        return

    if len(warm.captures) == 0:
        return

    warm_runner = sweep_action.SweepRunner(warm)
    res = warmup.open_warmup_resources(warm_runner)

    with res['source']:
        yield from warm_runner.iterate(res, always_yield=True)
