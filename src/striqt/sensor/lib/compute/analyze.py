"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

from typing import Any, cast, Literal, overload, TYPE_CHECKING
import warnings

from ... import specs
from . import corrections, datasets

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

import msgspec

if TYPE_CHECKING:
    from collections.abc import Generator

    from ..typing import Array
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

    # the spec inside a struct-form trigger takes precedence over the sweep analysis
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
        return list(meas.keys())[0]
    elif isinstance(setup.signal_trigger, str):
        return setup.signal_trigger
    else:
        return None


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[True]],
) -> 'xr.Dataset': ...


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal['delayed']],
) -> datasets.DelayedDataset: ...


@overload
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions[Literal[False]],
) -> 'dict[str, Array]': ...


@sa.util.stopwatch('', 'analysis')
def analyze(
    iq: specs.AcquiredIQ,
    options: datasets.EvaluationOptions,
) -> 'dict[str, Array] | xr.Dataset | datasets.DelayedDataset':
    """run the analysis of ``options.sweep_spec`` on `iq`.

    The analysis is ``options.sweep_spec.analysis`` adjusted by
    ``iq.capture.adjust_analysis``. Each measurement selects a conditioning of `iq`
    through `preprocess`, and a conditioning stage runs on the first selection that
    needs the stage. The resampling stage consumes `iq.x` in place unless
    ``options.sweep_spec.options.reuse_iq`` is set. cupy arrays in `iq.extra_data`
    are replaced in place by numpy copies. The `cache_callback` of `options` runs
    each time a measurement fills a cache.

    Returns:
        as `options.as_xarray` selects: a dict of arrays by measurement name
        (False), a `DelayedDataset` (``'delayed'``), or an `xr.Dataset` (True); the
        latter 2 have the capture fields and acquisition info as coordinates and
        `iq.extra_data` as variables

    Raises:
        TypeError: if `iq.capture` is not a `SensorCapture`
        ValueError: if ``iq.source_spec.signal_trigger`` names a measurement that
            the analysis does not enable
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


def prepare_compute(
    spec: specs.Sweep[Any, Any, Any, Any], skip_warmup: bool = False
) -> Generator[xr.Dataset | DelayedDataset | None]:
    """import the compute backend of `spec`, then run the warmup sweep.

    The cupy imports and configuration run before the first yield when
    ``spec.source.array_backend`` is ``'cupy'``. The ``Mean of empty slice``
    `RuntimeWarning` is filtered out process-wide.

    Args:
        skip_warmup: True runs no warmup sweep

    Yields:
        one None when `skip_warmup` is set or `warmup_sweep_for` returns None;
        otherwise the yields of a `SweepRunner` over the warmup sweep with
        `always_yield` set, and nothing when the warmup sweep has no captures
    """
    if spec.source.array_backend == 'cupy':
        with sa.util.stopwatch('import cuda computing packages', 'sweep', 2):
            # numba.cuda must be imported before cupy on some versions/platforms
            # (https://github.com/numba/numba/issues/6131)
            np.__version__  # reify
            import numba.cuda  # noqa: F401
            import cupy  # type: ignore  # noqa: F401

        with sa.util.stopwatch('configure cupy', 'sweep', 1):
            sw.arrays.configure_cupy()

    warnings.filterwarnings(
        'ignore',
        category=RuntimeWarning,
        message='Mean of empty slice.*',
    )

    from ..actions import sweep as sweep_action
    from ..actions import warmup

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
