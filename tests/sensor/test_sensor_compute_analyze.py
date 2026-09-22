"""striqt.sensor.lib.compute.analyze: analyze() packaging around correct_iq and the
measurement registry, trigger selection from the source spec, and the warmup
decision of prepare_compute"""

from __future__ import annotations

import dataclasses

import msgspec
import numpy as np
import pytest
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    IQ_ONLY,
    SPECTROGRAM,
    acquire_corrected,
    make_sweep,
    tone_captures,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute

(CAPTURE,) = tone_captures((1e6,))
SWEEP = make_sweep('single_tone', (CAPTURE,), analysis=IQ_ONLY)

PSS = ss.specs.BundledAnalysis.from_dict({
    'cellular_5g_pss_sync': {'subcarrier_spacing': 30e3}
})


def options(**kws):
    kws = {'sweep_spec': SWEEP, 'registry': sa.registry, 'extra_attrs': {}, **kws}
    return compute.EvaluationOptions(**kws)


@pytest.fixture(scope='module')
def stages():
    return acquire_corrected('single_tone', CAPTURE, analysis=IQ_ONLY)


def fresh_raw(stages):
    """a copy of the raw acquisition, indexed as iterate_sweep would before analysis"""
    return dataclasses.replace(
        stages.raw,
        pre_align=stages.raw.pre_align.copy(),
        info=stages.raw.info.replace(sweep_index=0),
    )


# %% analyze


def test_delayed_result_matches_correct_iq(stages):
    dd = compute.analyze(
        fresh_raw(stages), options(correction=True, as_xarray='delayed')
    )
    assert isinstance(dd, compute.DelayedDataset)
    ds = compute.from_delayed(dd)
    assert np.array_equal(ds.iq_waveform.values, stages.corrected.pre_align)


def test_as_xarray_true_returns_a_dataset(stages):
    import xarray as xr

    ds = compute.analyze(fresh_raw(stages), options(correction=True, as_xarray=True))
    assert isinstance(ds, xr.Dataset)
    assert np.array_equal(ds.iq_waveform.values, stages.corrected.pre_align)


def test_as_xarray_false_returns_the_delayed_arrays(stages):
    """current behaviour, not a contract: the values are DelayedDataArray results
    keyed by measurement name, not bare arrays as the annotation says"""
    result = compute.analyze(
        fresh_raw(stages), options(correction=True, as_xarray=False)
    )
    assert isinstance(result, dict)
    assert set(result) == {'iq_waveform'}
    assert isinstance(result['iq_waveform'], sa.dataarrays.DelayedDataArray)


def test_without_correction_the_iq_passes_through(stages):
    iq = dataclasses.replace(
        stages.corrected, info=stages.raw.info.replace(sweep_index=0)
    )
    ds = compute.analyze(iq, options(correction=False, as_xarray=True))
    assert np.array_equal(ds.iq_waveform.values, stages.corrected.pre_align)


def test_analyze_rejects_a_bare_capture(stages):
    iq = dataclasses.replace(fresh_raw(stages), capture=sa.specs.Capture())
    with pytest.raises(TypeError, match='SensorCapture'):
        compute.analyze(iq, options(correction=False, as_xarray=True))


def test_a_source_only_capture_field_reuses_the_analysis_caches(stages):
    """the analysis-side caches are keyed on a projection of the capture
    (`sa.specs.helpers.lru_cache_on_converted`), so a sweep that loops over a field no
    measurement reads - here the synthetic source's `snr` - pays for them once"""
    shared = sa.measurements.shared
    cached = (shared.validated_spectrogram_sizing, shared.spectrogram_freqs)
    for func in cached:
        func.cache_clear()

    opts = options(
        sweep_spec=make_sweep('single_tone', (CAPTURE,), analysis=SPECTROGRAM),
        correction=True,
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
        assert trigger.meas_spec == PSS.cellular_5g_pss_sync


@pytest.mark.xfail(
    strict=True,
    raises=msgspec.ValidationError,
    reason='Source.signal_trigger annotates the group form as the fieldless '
    'AnalysisGroup base, so validation rejects every measurement key and the '
    'AnalysisGroup branches of get_trigger_from_spec are unreachable '
    '(sensor/specs/structs.py:105)',
)
def test_get_trigger_from_an_analysis_group_trigger():
    trigger = compute.get_trigger_from_spec(SOURCE.replace(signal_trigger=PSS), None)
    assert trigger.meas_spec == PSS.cellular_5g_pss_sync


# %% prepare_compute

WARMUP_ENABLED = ss.specs.SweepOptions(skip_warmup=False)


@pytest.mark.parametrize(
    'skip_warmup, sweep',
    [
        (True, make_sweep('single_tone', (CAPTURE,), options=WARMUP_ENABLED)),
        (False, make_sweep('single_tone', (CAPTURE,), options=WARMUP_ENABLED)),
        (False, make_sweep('single_tone', (CAPTURE,))),
    ],
    ids=['skip_argument', 'numpy_backend', 'skip_option'],
)
def test_prepare_compute_yields_once_without_a_warmup(skip_warmup, sweep):
    assert list(compute.prepare_compute(sweep, skip_warmup=skip_warmup)) == [None]


@pytest.mark.namespaces('cupy')
def test_prepare_compute_runs_a_warmup_sweep_on_cupy(array_backend):
    """the warmup is one capture through iterate_sweep(always_yield=True), so its
    three pipeline stages yield two placeholders and one result"""
    source = SOURCE.replace(array_backend=array_backend)
    sweep = make_sweep(
        'single_tone',
        (CAPTURE,),
        analysis=ANALYSIS,
        source=source,
        options=WARMUP_ENABLED,
    )
    items = list(compute.prepare_compute(sweep, skip_warmup=False))
    assert [item is None for item in items] == [True, True, False]
    assert isinstance(items[-1], compute.DelayedDataset)
