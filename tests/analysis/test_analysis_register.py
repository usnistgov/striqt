"""striqt.analysis.lib.register: the `validate=` and `tolerance=` hooks on measurement
registration, the `AnalysisRegistry.validate`/`.tolerances` walks over an analysis
group, and the guarantee the live registry's validators make: a (capture, spec) pair
that validates is one the measurement can run.

The hook tests build their own `AnalysisRegistry`; mutating the live
`register.registry` would leak a measurement into every other test module. The
measurements are called with `as_xarray=False` because the xarray path resolves
coordinate factories through the *global* registry.
"""

from __future__ import annotations

from fractions import Fraction

import msgspec
import numpy as np
import pytest
from analysis_strategies import VALIDATOR_DOMAINS
from hypothesis import given, settings
from hypothesis import strategies as st

import striqt.analysis as sa
from striqt.analysis.lib import register
from striqt.analysis.specs.helpers import SpecValidationError

CAPTURE = sa.specs.Capture(duration=1e-4, sample_rate=1e6)


class ToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    nfft: int = 8


class OtherToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    scale: float = 1.0


def build_registry(validate=None, *, spec_type=ToySpec, calls=None, tolerance=None):
    """a registry holding one measurement that records its (capture, spec) calls"""
    registry = register.AnalysisRegistry()

    @registry.measurement(
        spec_type,
        dtype='float32',
        dims=('toy',),
        validate=validate,
        tolerance=tolerance,
    )
    def toy(iq, capture, **kwargs):
        if calls is not None:
            calls.append(('body', capture, kwargs))
        return np.zeros((1, 2), dtype='float32')

    return registry, toy


# %% validate= on measurement registration


def test_validate_receives_the_projected_capture_and_the_decoded_spec():
    seen = []

    def validate(capture, spec):
        seen.append((capture, spec))

    _, toy = build_registry(validate)
    toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False, nfft=16)

    ((capture, spec),) = seen
    assert type(capture) is sa.specs.AnalysisCapture
    assert capture == sa.specs.helpers.to_analysis_capture(CAPTURE)
    assert spec == ToySpec(nfft=16)


def test_validate_runs_before_the_measurement_body():
    calls = []

    def validate(capture, spec):
        calls.append(('validate', capture, spec))

    _, toy = build_registry(validate, calls=calls)
    toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert [name for name, *_ in calls] == ['validate', 'body']


def test_a_validator_failure_skips_the_measurement_body():
    calls = []

    def validate(capture, spec):
        raise ValueError('no')

    _, toy = build_registry(validate, calls=calls)
    with pytest.raises(SpecValidationError):
        toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert calls == []


@pytest.mark.parametrize(
    'raised',
    [
        ValueError('nfft too large'),
        TypeError('nfft too large'),
        SpecValidationError('nfft too large', ('.nfft',)),
    ],
    ids=['value_error', 'type_error', 'spec_validation_error'],
)
def test_a_validator_failure_is_annotated_with_the_measurement_name(raised):
    def validate(capture, spec):
        raise raised

    _, toy = build_registry(validate)
    with pytest.raises(SpecValidationError) as excinfo:
        toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert excinfo.value.path[0] == '.toy'
    assert str(excinfo.value).startswith('$.toy')
    assert str(excinfo.value).endswith(': nfft too large')


def test_validate_defaults_to_none_and_the_measurement_still_runs():
    registry, toy = build_registry()
    assert registry[ToySpec].validate is None

    data, _ = toy(
        np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False, nfft=16
    )
    assert data.shape == (1, 2)


# %% AnalysisRegistry.validate


def build_group_registry(*validators, tolerances=(None, None)):
    """a registry of two measurements, each with the given validator and tolerance"""
    registry = register.AnalysisRegistry()
    toy_validate, other_validate = validators
    toy_tolerance, other_tolerance = tolerances

    @registry.measurement(
        ToySpec,
        dtype='float32',
        dims=('toy',),
        validate=toy_validate,
        tolerance=toy_tolerance,
    )
    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    @registry.measurement(
        OtherToySpec,
        dtype='float32',
        dims=('other',),
        validate=other_validate,
        tolerance=other_tolerance,
    )
    def other_toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    return registry


def test_registry_validate_calls_only_the_requested_measurements():
    seen = []
    registry = build_group_registry(
        lambda capture, spec: seen.append('toy'),
        lambda capture, spec: seen.append('other_toy'),
    )
    group = registry.tospec()(other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_skips_measurements_without_a_validator():
    seen = []
    registry = build_group_registry(
        None, lambda capture, spec: seen.append('other_toy')
    )
    group = registry.tospec()(toy=ToySpec(), other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_annotates_the_analysis_field_path():
    def validate(capture, spec):
        raise SpecValidationError('nfft must be even', ('.nfft',))

    registry = build_group_registry(validate, None)
    group = registry.tospec()(toy=ToySpec())

    with pytest.raises(SpecValidationError) as excinfo:
        registry.validate(CAPTURE, group)

    assert excinfo.value.path == ('.analysis', '.toy', '.nfft')
    assert str(excinfo.value) == '$.analysis.toy.nfft: nfft must be even'


def test_registry_validate_passes_the_projected_capture():
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(capture), None)
    group = registry.tospec()(toy=ToySpec())

    registry.validate(CAPTURE, group)
    assert seen == [sa.specs.helpers.to_analysis_capture(CAPTURE)]


def test_registry_validate_is_cached_on_the_projected_capture():
    """the sensor walker relies on this to keep a long sweep off the critical path"""
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(spec), None)
    group = registry.tospec()(toy=ToySpec())

    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE.replace(analysis_bandwidth=5e5), group)

    assert len(seen) == 2


def test_registry_is_hashable():
    registry = build_group_registry(None, None)
    assert hash(registry) == hash(registry)
    assert isinstance(hash(registry), int)


def test_validator_errors_are_msgspec_validation_errors():
    """the sweep decode path reports these, so they must survive msgspec's own
    error handling and be catchable as a decode failure"""
    registry = build_group_registry(
        lambda capture, spec: (_ for _ in ()).throw(ValueError('nope')), None
    )
    group = registry.tospec()(toy=ToySpec())

    with pytest.raises(msgspec.ValidationError):
        registry.validate(CAPTURE, group)


# %% tolerance= on measurement registration

TOY_TOLERANCE = sa.specs.Tolerance(
    units='dB', rtol=1e-6, on_peak=sa.specs.ErrorBound(rms=1e-3, peak=1e-2)
)


class WideCapture(sa.specs.Capture, frozen=True, kw_only=True):
    """a sensor-style capture with a field the analysis layer never reads"""

    gain: float = 0.0


def recording_tolerance(seen):
    def tolerance(capture, spec, **kwargs):
        seen.append((capture, spec, kwargs))
        return TOY_TOLERANCE

    return tolerance


def test_tolerance_is_stored_on_the_info_and_defaults_to_none():
    registry, _ = build_registry()
    assert registry[ToySpec].tolerance is None

    tolerance = recording_tolerance([])
    registry, _ = build_registry(tolerance=tolerance)
    assert registry[ToySpec].tolerance is tolerance


def test_registry_tolerances_is_empty_when_nothing_declares_one():
    registry = build_group_registry(None, None)
    group = registry.tospec()(toy=ToySpec(), other_toy=OtherToySpec())

    assert registry.tolerances(CAPTURE, group) == {}


def test_registry_tolerances_keys_declared_measurements_by_name():
    registry = build_group_registry(
        None, None, tolerances=(None, recording_tolerance([]))
    )
    group = registry.tospec()(toy=ToySpec(), other_toy=OtherToySpec())

    result = registry.tolerances(CAPTURE, group)

    assert result == {'other_toy': TOY_TOLERANCE}
    assert isinstance(result['other_toy'], sa.specs.Tolerance)


def test_registry_tolerances_skips_measurements_not_in_the_group():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), recording_tolerance(seen))
    )
    group = registry.tospec()(toy=ToySpec(nfft=4))

    registry.tolerances(CAPTURE, group)

    ((capture, spec, _),) = seen
    assert capture == sa.specs.helpers.to_analysis_capture(CAPTURE)
    assert spec == ToySpec(nfft=4)


def test_registry_tolerances_forwards_backend_and_input_error_as_keywords():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), None)
    )
    group = registry.tospec()(toy=ToySpec())

    registry.tolerances(CAPTURE, group)
    registry.tolerances(CAPTURE, group, array_backend='cupy', input_error=1e-4)

    assert [kws for *_, kws in seen] == [
        {'array_backend': 'numpy', 'input_error': 0.0},
        {'array_backend': 'cupy', 'input_error': 1e-4},
    ]


def test_registry_tolerances_is_cached_on_the_projected_capture():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), None)
    )
    group = registry.tospec()(toy=ToySpec())

    registry.tolerances(WideCapture(duration=1e-4, sample_rate=1e6, gain=0.0), group)
    registry.tolerances(WideCapture(duration=1e-4, sample_rate=1e6, gain=10.0), group)
    registry.tolerances(CAPTURE.replace(analysis_bandwidth=5e5), group)

    assert len(seen) == 2


# %% a validator that passes means the measurement runs

VALIDATED = [info for info in sa.registry.values() if info.validate is not None]
VALIDATED_IDS = [info.name for info in VALIDATED]
# the SSS correlators search all 1008 cell ids per accepted example, so their draws
# cost roughly ten times the others'
SSS = [info for info in VALIDATED if 'sss' in info.name]
OTHERS = [info for info in VALIDATED if 'sss' not in info.name]


def test_every_validated_measurement_has_a_strategy():
    """the guarantee below is only as broad as the domains it draws from"""
    assert set(VALIDATOR_DOMAINS) == set(VALIDATED_IDS)


def check_accepted_pair_runs(info, data):
    """`Sweep.__post_init__` runs these validators on every capture before anything
    is acquired, so a pair they accept must not fail on shape or scalar arithmetic
    once IQ arrives. Pairs they reject are out of scope here; the spot checks below
    cover those."""
    capture, spec = data.draw(VALIDATOR_DOMAINS[info.name], label=info.name)
    group = sa.registry.tospec()(**{info.name: spec})
    try:
        sa.registry.validate(capture, group)
    except SpecValidationError:
        return

    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    iq = sa.testing.noise(
        capture.duration,
        capture.sample_rate,
        noise_psd=1 / capture.sample_rate,
        ports=ports,
    )
    try:
        info.func(iq, capture, as_xarray=False, **spec.to_dict())
    except (ValueError, IndexError, TypeError) as ex:
        pytest.fail(f'{info.name} accepted {capture!r} with {spec!r} but raised {ex!r}')


@pytest.mark.parametrize('info', OTHERS, ids=[i.name for i in OTHERS])
@given(data=st.data())
@settings(max_examples=50)
def test_a_validator_that_passes_means_the_measurement_runs(info, data):
    check_accepted_pair_runs(info, data)


@pytest.mark.parametrize('info', SSS, ids=[i.name for i in SSS])
@given(data=st.data())
@settings(max_examples=15)
def test_a_validator_that_passes_means_the_sss_measurement_runs(info, data):
    check_accepted_pair_runs(info, data)


FS = 1.024e6
NFFT = 8
SAMPLES = 64
SPG = {'window': 'boxcar', 'frequency_resolution': FS / NFFT}
CELL_FS = 3.84e6
FRAME = round(10e-3 * CELL_FS)
PSS = {'subcarrier_spacing': 30e3, 'sample_rate': CELL_FS, 'symbol_indexes': 'c'}
BINS = {'power_low': -40.0, 'power_high': 10.0, 'power_resolution': 1.0}


def capture_of(samples, sample_rate, **kwargs) -> sa.specs.Capture:
    return sa.specs.Capture(
        duration=samples / sample_rate, sample_rate=sample_rate, **kwargs
    )


# (measurement, capture, spec, the spec or capture field the message must name)
REJECTED = {
    'spectrogram_shorter_than_nfft': (
        capture_of(NFFT - 1, FS),
        sa.specs.Spectrogram(**SPG),
        'duration',
    ),
    'spectrogram_lo_bandstop_on_odd_nfft': (
        capture_of(SAMPLES, FS),
        sa.specs.Spectrogram(
            window='boxcar', frequency_resolution=FS / 7, lo_bandstop=FS / 7
        ),
        'lo_bandstop',
    ),
    'spectrogram_analysis_bandwidth_above_sample_rate': (
        capture_of(SAMPLES, FS, analysis_bandwidth=1.5 * FS),
        sa.specs.Spectrogram(**SPG),
        'analysis_bandwidth',
    ),
    'spectrogram_trim_on_odd_nfft': (
        capture_of(SAMPLES, FS, analysis_bandwidth=FS / 2),
        sa.specs.Spectrogram(window='boxcar', frequency_resolution=FS / 7),
        'analysis_bandwidth',
    ),
    'spectrogram_integration_bandwidth_above_sample_rate': (
        capture_of(SAMPLES, FS),
        sa.specs.Spectrogram(**SPG, integration_bandwidth=2 * FS),
        'integration_bandwidth',
    ),
    'spectrogram_time_aperture_longer_than_capture': (
        capture_of(NFFT, FS),
        sa.specs.Spectrogram(**SPG, time_aperture=2 * NFFT / FS),
        'time_aperture',
    ),
    'psd_unknown_statistic': (
        capture_of(SAMPLES, FS),
        sa.specs.PowerSpectralDensity(**SPG, time_statistic=('mean', 'bogus')),
        'time_statistic',
    ),
    'psd_quantile_above_one': (
        capture_of(SAMPLES, FS),
        sa.specs.PowerSpectralDensity(**SPG, time_statistic=(1.5,)),
        'time_statistic',
    ),
    'channel_power_unknown_detector': (
        capture_of(100, 1e6),
        sa.specs.ChannelPowerTimeSeries(
            detector_period=Fraction(1, 100_000), power_detectors=('rms', 'bogus')
        ),
        'power_detectors',
    ),
    'channel_power_histogram_unknown_detector': (
        capture_of(100, 1e6),
        sa.specs.ChannelPowerHistogram(
            detector_period=Fraction(1, 100_000),
            power_detectors=('bogus',),
            **BINS,
        ),
        'power_detectors',
    ),
    'cyclic_power_quantile_below_zero': (
        capture_of(100, 1e6),
        sa.specs.CyclicChannelPower(
            cyclic_period=1e-4,
            detector_period=Fraction(1, 100_000),
            cyclic_statistics=('min', -0.5),
        ),
        'cyclic_statistics',
    ),
    'pss_odd_sample_count': (
        capture_of(FRAME + 1, CELL_FS),
        sa.specs.Cellular5GNRPSSCorrelator(**PSS),
        'duration',
    ),
    'pss_frequency_offset_off_the_resampler_grid': (
        capture_of(FRAME, CELL_FS),
        sa.specs.Cellular5GNRPSSCorrelator(**PSS, frequency_offset=150.0),
        'frequency_offset',
    ),
    'pss_frequency_offset_shifts_past_the_band': (
        capture_of(FRAME, CELL_FS),
        sa.specs.Cellular5GNRPSSCorrelator(
            subcarrier_spacing=15e3, sample_rate=1.92e6, frequency_offset=CELL_FS / 2
        ),
        'frequency_offset',
    ),
    'sss_partial_frame': (
        capture_of(FRAME + FRAME // 2, CELL_FS),
        sa.specs.Cellular5GNRSSSCorrelator(**PSS),
        'duration',
    ),
    'pss_sync_partial_frame': (
        capture_of(FRAME + FRAME // 2, CELL_FS),
        sa.specs.Cellular5GNPSSSync(**PSS),
        'duration',
    ),
    'sss_sync_odd_sample_count': (
        capture_of(FRAME + 1, CELL_FS),
        sa.specs.Cellular5GNSSSSync(**PSS),
        'duration',
    ),
    'ssb_spectrogram_frequency_offset_off_the_subcarrier_grid': (
        capture_of(8400, 420e3),
        sa.specs.Cellular5GNRSSBSpectrogram(
            subcarrier_spacing=30e3, sample_rate=120e3, frequency_offset=15e3
        ),
        'frequency_offset',
    ),
    'ssb_spectrogram_sample_rate_above_capture': (
        capture_of(8400, 420e3),
        sa.specs.Cellular5GNRSSBSpectrogram(subcarrier_spacing=30e3, sample_rate=840e3),
        'sample_rate',
    ),
    'ssb_spectrogram_partial_burst_set': (
        capture_of(8400 + 210, 420e3),
        sa.specs.Cellular5GNRSSBSpectrogram(subcarrier_spacing=30e3, sample_rate=120e3),
        'duration',
    ),
    'autocorrelation_symbol_range_past_the_slot': (
        capture_of(FRAME, CELL_FS),
        sa.specs.CellularCyclicAutocorrelator(
            subcarrier_spacings=30e3, symbol_range=(14, 15)
        ),
        'symbol_range',
    ),
    'autocorrelation_no_downlink_slot': (
        capture_of(FRAME, CELL_FS),
        sa.specs.CellularCyclicAutocorrelator(
            subcarrier_spacings=30e3, frame_slots='u'
        ),
        'frame_slots',
    ),
    'autocorrelation_frame_range_past_the_capture': (
        capture_of(FRAME, CELL_FS),
        sa.specs.CellularCyclicAutocorrelator(
            subcarrier_spacings=30e3, frame_range=(0, 2)
        ),
        'frame_range',
    ),
    'resource_grid_analysis_bandwidth_above_sample_rate': (
        capture_of(420, 210e3, analysis_bandwidth=1.5 * 210e3),
        sa.specs.CellularResourcePowerHistogram(
            window='hamming', subcarrier_spacing=15e3, **BINS
        ),
        'analysis_bandwidth',
    ),
}


@pytest.mark.parametrize('case', list(REJECTED), ids=list(REJECTED))
def test_validator_rejects_at_the_analysis_path_naming_the_field(case):
    """each failure mode the runtime kernels would raise on is rejected first by the
    registered validator, located at the measurement key and worded in spec terms"""
    capture, spec, field = REJECTED[case]
    name = sa.registry[type(spec)].name
    group = sa.registry.tospec()(**{name: spec})

    with pytest.raises(SpecValidationError) as excinfo:
        sa.registry.validate(capture, group)

    assert excinfo.value.path[:2] == ('.analysis', f'.{name}')
    assert str(excinfo.value).startswith(f'$.analysis.{name}')
    assert field in excinfo.value.message
