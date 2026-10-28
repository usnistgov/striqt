"""striqt.sensor.lib.compute.analyze: packaging around the conditioning of AcquiredIQ
and the measurement registry, trigger selection, and the warmup decision
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import xarray as xr
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    IQ_ONLY,
    SPECTROGRAM,
    acquire_corrected,
    filtered,
    preset_sweep,
    tone_captures,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute

(CAPTURE,) = tone_captures((1e6,))
SWEEP = preset_sweep('single_tone', (CAPTURE,), analysis=IQ_ONLY)

PSS = ss.specs.BuiltinAnalyses.from_dict({
    'cellular_5g_pss_sync': {'subcarrier_spacing': 30e3}
})
PSS_BUNDLE = ss.specs.BuiltinTriggers(cellular_5g_pss_sync=PSS.cellular_5g_pss_sync)


def options(**kws):
    kws = {'sweep_spec': SWEEP, 'registry': sa.registry, 'extra_attrs': {}, **kws}
    return compute.EvaluationOptions(**kws)


@pytest.fixture(scope='module')
def stages():
    return acquire_corrected('single_tone', CAPTURE, analysis=IQ_ONLY)


def fresh_raw(stages):
    """return a copy of the raw acquisition of `stages` with `sweep_index` set to 0"""
    return dataclasses.replace(
        stages.raw,
        x=stages.raw.x.copy(),
        info=stages.raw.info.replace(sweep_index=0),
    )


# %% analyze


@pytest.mark.parametrize(
    'as_xarray, container',
    [('delayed', compute.DelayedDataset), (True, xr.Dataset)],
    ids=['delayed', 'dataset'],
)
def test_analyze_result_matches_the_filtered_selection(stages, as_xarray, container):
    result = compute.analyze(fresh_raw(stages), options(as_xarray=as_xarray))
    assert isinstance(result, container)
    ds = result if as_xarray is True else compute.from_delayed(result)
    assert np.array_equal(ds.iq_waveform.values, filtered(stages.corrected))


def test_as_xarray_false_returns_the_delayed_arrays(stages):
    """current behavior, not a contract: the values are DelayedDataArray results
    keyed by measurement name, not bare arrays as the annotation says"""
    result = compute.analyze(fresh_raw(stages), options(as_xarray=False))
    assert isinstance(result, dict)
    assert set(result) == {'iq_waveform'}
    assert isinstance(result['iq_waveform'], sa.dataarrays.DelayedDataArray)


def test_analyze_rejects_a_bare_capture(stages):
    iq = dataclasses.replace(fresh_raw(stages), capture=sa.specs.Capture())
    with pytest.raises(TypeError, match='SensorCapture'):
        compute.analyze(iq, options(as_xarray=True))


def test_a_source_only_capture_field_reuses_the_analysis_caches(stages):
    """the analysis-side caches are keyed on a projection of the capture through
    `lru_cache_on_converted`. A sweep that loops over a field no measurement reads,
    here the `snr` of the synthetic source, misses each cache once."""
    spectrum = sa.measurements.spectrum
    cached = (spectrum.validated_spectrogram_sizing, spectrum.spectrogram_freqs)
    for func in cached:
        func.cache_clear()

    opts = options(
        sweep_spec=preset_sweep('single_tone', (CAPTURE,), analysis=SPECTROGRAM),
        as_xarray=True,
    )
    for snr in (None, 10.0):
        raw = fresh_raw(stages)
        iq = dataclasses.replace(raw, capture=raw.capture.replace(snr=snr))
        compute.analyze(iq, opts)

    for func in cached:
        info = func.cache_info()
        assert info.misses == 1, f'{func.__name__} ran again for the second capture'
        assert info.hits > 0


# %% get_trigger_from_spec

TRIGGER_CASES = {
    'no_trigger': (SOURCE, IQ_ONLY, None),
    'name_in_analysis': (
        SOURCE.replace(signal_trigger='cellular_5g_pss_sync'),
        PSS,
        'pss',
    ),
    'name_missing_from_analysis': (
        SOURCE.replace(signal_trigger='cellular_5g_pss_sync'),
        IQ_ONLY,
        ValueError,
    ),
    'name_without_analysis': (
        SOURCE.replace(signal_trigger='cellular_5g_pss_sync'),
        None,
        ValueError,
    ),
    'bundle_without_analysis': (
        SOURCE.replace(signal_trigger=PSS_BUNDLE),
        None,
        'pss',
    ),
    'bundle_with_other_analysis': (
        SOURCE.replace(signal_trigger=PSS_BUNDLE),
        IQ_ONLY,
        'pss',
    ),
}


@pytest.mark.parametrize('case', list(TRIGGER_CASES), ids=list(TRIGGER_CASES))
def test_get_trigger_from_spec(case):
    source, analysis, expected = TRIGGER_CASES[case]
    if expected is ValueError:
        with pytest.raises(ValueError):
            compute.get_trigger_from_spec(source, analysis)
    elif expected is None:
        assert compute.get_trigger_from_spec(source, analysis) is None
    else:
        trigger = compute.get_trigger_from_spec(source, analysis)
        assert isinstance(trigger, sa.Trigger)
        assert trigger.info.name == 'cellular_5g_pss_sync'
        assert trigger.meas_spec == PSS.cellular_5g_pss_sync


# %% prepare_compute

WARMUP_ENABLED = ss.specs.SweepOptions(skip_warmup=False)


@pytest.mark.parametrize(
    'skip_warmup, sweep',
    [
        (True, preset_sweep('single_tone', (CAPTURE,), options=WARMUP_ENABLED)),
        (False, preset_sweep('single_tone', (CAPTURE,), options=WARMUP_ENABLED)),
        (False, preset_sweep('single_tone', (CAPTURE,))),
    ],
    ids=['skip_argument', 'numpy_backend', 'skip_option'],
)
def test_prepare_compute_yields_once_without_a_warmup(skip_warmup, sweep):
    assert list(compute.prepare_compute(sweep, skip_warmup=skip_warmup)) == [None]


@pytest.mark.namespaces('cupy')
def test_prepare_compute_runs_a_warmup_sweep_on_cupy(array_backend):
    """the warmup is one capture through ``iterate_sweep(always_yield=True)``. The 3
    pipeline stages yield 2 placeholders and 1 result."""
    source = SOURCE.replace(array_backend=array_backend)
    sweep = preset_sweep(
        'single_tone',
        (CAPTURE,),
        analysis=ANALYSIS,
        source=source,
        options=WARMUP_ENABLED,
    )
    items = list(compute.prepare_compute(sweep, skip_warmup=False))
    assert [item is None for item in items] == [True, True, False]
    assert isinstance(items[-1], compute.DelayedDataset)
