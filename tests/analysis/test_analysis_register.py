"""striqt.analysis.lib.register: measurement registration hooks, the validation and
tolerance walks over an analysis group, and the contract every registered measurement
follows (dtype, dims, coords, and attrs reach the DataArray; the raw path returns the
same values).

The hook tests build their own `AnalysisRegistry`, since mutating the live registry
would leak a measurement into every other test module.
"""

# ruff: noqa: UP045  msgspec resolves the ToyGroup annotations at runtime on py3.9

from __future__ import annotations

import weakref
from fractions import Fraction
from typing import Optional

import msgspec
import numpy as np
import pytest
from analysis_strategies import (
    POWER_BINS,
    VALIDATOR_DOMAINS,
    capture_of,
    registered_tolerance,
)
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


class ToyGroup(sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True):
    toy: Optional[ToySpec] = None
    other_toy: Optional[OtherToySpec] = None


def build_registry(validate=None, *, spec_type=ToySpec, calls=None, tolerance=None):
    """a registry with one measurement that records its (capture, spec) calls"""
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
    """a registry of 2 measurements, each with the given validator and tolerance"""
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
    group = ToyGroup(other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_skips_measurements_without_a_validator():
    seen = []
    registry = build_group_registry(
        None, lambda capture, spec: seen.append('other_toy')
    )
    group = ToyGroup(toy=ToySpec(), other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_annotates_the_analysis_field_path():
    def validate(capture, spec):
        raise SpecValidationError('nfft must be even', ('.nfft',))

    registry = build_group_registry(validate, None)
    group = ToyGroup(toy=ToySpec())

    with pytest.raises(SpecValidationError) as excinfo:
        registry.validate(CAPTURE, group)

    assert excinfo.value.path == ('.analysis', '.toy', '.nfft')
    assert str(excinfo.value) == '$.analysis.toy.nfft: nfft must be even'


def test_registry_validate_passes_the_projected_capture():
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(capture), None)
    group = ToyGroup(toy=ToySpec())

    registry.validate(CAPTURE, group)
    assert seen == [sa.specs.helpers.to_analysis_capture(CAPTURE)]


def test_registry_validate_is_cached_on_the_projected_capture():
    """the sensor walker relies on the cache to keep a long sweep off the critical
    path"""
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(spec), None)
    group = ToyGroup(toy=ToySpec())

    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE.replace(analysis_bandwidth=5e5), group)

    assert len(seen) == 2


def test_validator_errors_are_msgspec_validation_errors():
    """the sweep decode path reports these, so they must survive msgspec's own
    error handling and be catchable as a decode failure"""
    registry = build_group_registry(
        lambda capture, spec: (_ for _ in ()).throw(ValueError('nope')), None
    )
    group = ToyGroup(toy=ToySpec())

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


def test_registry_tolerances_is_empty_when_nothing_declares_one():
    registry = build_group_registry(None, None)
    group = ToyGroup(toy=ToySpec(), other_toy=OtherToySpec())

    assert registry.tolerances(CAPTURE, group) == {}


def test_registry_tolerances_keys_declared_measurements_by_name():
    registry = build_group_registry(
        None, None, tolerances=(None, recording_tolerance([]))
    )
    group = ToyGroup(toy=ToySpec(), other_toy=OtherToySpec())

    result = registry.tolerances(CAPTURE, group)

    assert result == {'other_toy': TOY_TOLERANCE}
    assert isinstance(result['other_toy'], sa.specs.Tolerance)


def test_registry_tolerances_skips_measurements_not_in_the_group():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), recording_tolerance(seen))
    )
    group = ToyGroup(toy=ToySpec(nfft=4))

    registry.tolerances(CAPTURE, group)

    ((capture, spec, _),) = seen
    assert capture == sa.specs.helpers.to_analysis_capture(CAPTURE)
    assert spec == ToySpec(nfft=4)


def test_registry_tolerances_forwards_backend_and_input_error_as_keywords():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), None)
    )
    group = ToyGroup(toy=ToySpec())

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
    group = ToyGroup(toy=ToySpec())

    registry.tolerances(WideCapture(duration=1e-4, sample_rate=1e6, gain=0.0), group)
    registry.tolerances(WideCapture(duration=1e-4, sample_rate=1e6, gain=10.0), group)
    registry.tolerances(CAPTURE.replace(analysis_bandwidth=5e5), group)

    assert len(seen) == 2


# %% a validator that passes means the measurement runs

VALIDATED = [info for info in sa.registry.values() if info.validate is not None]
VALIDATED_IDS = [info.name for info in VALIDATED]
# the SSS correlators share the PSS validator and search all 1008 cell ids per
# accepted example, so the PSS measurements alone draw from the shared domain
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
    group = sa.specs.BuiltinAnalyses(**{info.name: spec})
    try:
        sa.registry.validate(capture, group)
    except SpecValidationError:
        return

    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    iq = sa.testing.circular_awgn(
        capture.duration, capture.sample_rate, power=1.0, ports=ports
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


FS = 1.024e6
NFFT = 8
SAMPLES = 64
SPG = {'window': 'boxcar', 'frequency_resolution': FS / NFFT}

# (capture, spec, the spec or capture field the message must name). Every registered
# validator raises a bare ValueError, so `registry.validate` has one mechanism to
# exercise here: `validation_path` wrapping it at the measurement key. One row names
# a capture field and one a spec field; the per-measurement rejection tables are in
# the test_analysis_measurements_* modules.
REJECTED = {
    'spectrogram_shorter_than_nfft': (
        capture_of(NFFT - 1, FS),
        sa.specs.Spectrogram(**SPG),
        'duration',
    ),
    'psd_unknown_statistic': (
        capture_of(SAMPLES, FS),
        sa.specs.PowerSpectralDensity(**SPG, time_statistic=('mean', 'bogus')),
        'time_statistic',
    ),
}


@pytest.mark.parametrize('case', list(REJECTED), ids=list(REJECTED))
def test_validator_rejects_at_the_analysis_path_naming_the_field(case):
    """a pair the registered validator rejects is reported at the measurement key
    under `$.analysis`, worded in terms of the spec or capture field at fault"""
    capture, spec, field = REJECTED[case]
    name = sa.registry[type(spec)].name
    group = sa.specs.BuiltinAnalyses(**{name: spec})

    with pytest.raises(SpecValidationError) as excinfo:
        sa.registry.validate(capture, group)

    assert excinfo.value.path[:2] == ('.analysis', f'.{name}')
    assert str(excinfo.value).startswith(f'$.analysis.{name}')
    assert field in excinfo.value.message


# %% the contract shared by every measurement

CONTRACT_FS = 210e3
CONTRACT_SCS = 15e3
CONTRACT_DETECTOR_PERIOD = Fraction(1, 10000)
CONTRACT_CAPTURE = sa.specs.Capture(
    duration=4e-3, sample_rate=CONTRACT_FS, analysis_bandwidth=150e3
)

# 4 ms at 210 kHz is 1 discovery period of 56 symbols at 15 kHz subcarriers and a
# whole number of detector and cyclic periods. The first 28 symbols are the SSB burst
# set.
CONTRACT_SPECS = (
    sa.specs.Spectrogram(window='hamming', frequency_resolution=7.5e3),
    sa.specs.PowerSpectralDensity(window='hamming', frequency_resolution=7.5e3),
    sa.specs.Cellular5GNRSSBSpectrogram(
        subcarrier_spacing=CONTRACT_SCS,
        sample_rate=CONTRACT_FS / 2,
        discovery_periodicity=CONTRACT_CAPTURE.duration,
    ),
    sa.specs.ChannelPowerTimeSeries(detector_period=CONTRACT_DETECTOR_PERIOD),
    sa.specs.CyclicChannelPower(
        cyclic_period=1e-3, detector_period=CONTRACT_DETECTOR_PERIOD
    ),
    sa.specs.IQWaveform(),
    sa.specs.ChannelPowerHistogram(
        detector_period=CONTRACT_DETECTOR_PERIOD, **POWER_BINS
    ),
    sa.specs.SpectrogramHistogram(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.SpectrogramHistogramRatio(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.CellularResourcePowerHistogram(
        window='hamming', subcarrier_spacing=CONTRACT_SCS, **POWER_BINS
    ),
)

SPEC_IDS = [type(spec).__name__ for spec in CONTRACT_SPECS]


def measure(spec, as_xarray):
    """run the measurement registered for `type(spec)` the way the registry calls it"""
    info = sa.registry[type(spec)]
    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    iq = sa.testing.tone(
        CONTRACT_CAPTURE.duration, CONTRACT_CAPTURE.sample_rate, ports=ports
    )
    return info, info.func(iq, CONTRACT_CAPTURE, as_xarray=as_xarray, **spec.to_dict())


def registered_dims(info):
    dims = ['port']
    for factory in info.coord_factories:
        for dim in sa.registry.coordinates[factory].dims:
            if dim not in dims:
                dims.append(dim)
    return tuple(dims)


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_dtype_and_dims(spec):
    info, da = measure(spec, as_xarray=True)

    assert da.dtype == info.dtype
    if info.dims is None:
        assert da.dims == registered_dims(info)
    else:
        assert da.dims == ('port',) + info.dims
    assert set(da.coords) == set(da.dims) - {'port'}


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_attrs(spec):
    """the registered attrs and every spec field reach the DataArray, which is what
    names them in a saved zarr store"""
    info, da = measure(spec, as_xarray=True)

    for name, value in info.attrs.items():
        assert da.attrs[name] == value

    # the wrapper decodes the keywords into a spec before calling, so the attrs are
    # the validated field values rather than the ones passed in
    for name, value in spec.validate().to_dict().items():
        assert da.attrs[name] == value


ABSOLUTE_POWER_UNITS = ('dBm', '√mW')


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_absolute_power_outputs_name_their_noise_bandwidth(spec):
    """every output and power-bin coordinate in absolute power units has a
    `noise_bandwidth` in Hz and a `long_name` that names the bandwidth, except the
    outputs whose bandwidth is the per-capture `analysis_bandwidth`, which set
    neither; ratios in dB have neither"""
    info, da = measure(spec, as_xarray=True)

    def check(attrs, what):
        units = attrs.get('units')
        if str(what) in sa.lib.io._CAPTURE_BANDWIDTH_NAMES:
            assert 'noise_bandwidth' not in attrs, what
            assert '/' not in str(attrs.get('long_name', '')), what
        elif units in ABSOLUTE_POWER_UNITS:
            assert isinstance(attrs.get('noise_bandwidth'), float), what
            assert attrs['noise_bandwidth'] > 0, what
            assert f'({units}/' in attrs['long_name'], what
            assert 'dBm/' not in units, what
        elif units == 'dB':
            assert 'noise_bandwidth' not in attrs, what

    check(da.attrs, info.name)
    for name in da.coords:
        if str(name).endswith('_power_bin'):
            check(da[name].attrs, name)
            assert ('noise_bandwidth' in da[name].attrs) == (
                'noise_bandwidth' in da.attrs
            ), name


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_without_xarray(spec):
    _, (data, attrs) = measure(spec, as_xarray=False)
    _, da = measure(spec, as_xarray=True)

    assert isinstance(attrs, dict)
    data = np.asarray(data)
    assert data.shape == da.shape
    # the raw array is not held to the registered dtype: power_spectral_density
    # returns float16 where it registers float32
    assert np.array_equal(data.astype(da.dtype), da.values, equal_nan=True)


# the only measurement that validates nothing: iq_waveform clamps its start/stop
# bounds into the capture by design rather than rejecting them
NO_VALIDATOR = {'iq_waveform'}


def test_every_measurement_has_a_validator():
    """a measurement without a validator silently skips pre-sweep validation, so a
    new one must either register `validate=` or be named here"""
    missing = {
        info.name for info in sa.registry.values() if info.validate is None
    } - NO_VALIDATOR

    assert missing == set()
    assert NO_VALIDATOR <= {info.name for info in sa.registry.values()}


CONTRACT_TOLERANCE_SPECS = [
    s for s in CONTRACT_SPECS if sa.registry[type(s)].tolerance is not None
]
CONTRACT_TOLERANCE_IDS = [type(spec).__name__ for spec in CONTRACT_TOLERANCE_SPECS]


@pytest.mark.parametrize('spec', CONTRACT_TOLERANCE_SPECS, ids=CONTRACT_TOLERANCE_IDS)
def test_registered_tolerance_is_a_dB_budget_that_grows_with_input_error(spec):
    """the registered budget bounds a level in dB with a peak at or above its rms.
    An amplitude error in the IQ widens both."""
    exact = registered_tolerance(CONTRACT_CAPTURE, spec, input_error=0.0)
    assert isinstance(exact, sa.specs.Tolerance)
    assert exact.units == 'dB'
    # iq_waveform copies its input, so its exact-input bound is 0
    assert exact.on_peak.peak >= exact.on_peak.rms >= 0

    perturbed = registered_tolerance(CONTRACT_CAPTURE, spec, input_error=1e-4)
    assert perturbed.on_peak.rms > exact.on_peak.rms
    assert perturbed.on_peak.peak > exact.on_peak.peak


# %% check_group_fields


def test_check_group_fields_accepts_a_matching_group():
    expected = {'toy': ToySpec, 'other_toy': OtherToySpec}
    assert register.check_group_fields(ToyGroup, expected) is None


def test_check_group_fields_reports_a_missing_field():
    expected = {'toy': ToySpec, 'other_toy': OtherToySpec, 'third': ToySpec}
    with pytest.raises(TypeError, match=r"missing fields \['third'\]"):
        register.check_group_fields(ToyGroup, expected)


def test_check_group_fields_reports_an_unregistered_field():
    with pytest.raises(TypeError, match=r"unregistered fields \['other_toy'\]"):
        register.check_group_fields(ToyGroup, {'toy': ToySpec})


def test_check_group_fields_reports_a_wrong_spec_type():
    expected = {'toy': OtherToySpec, 'other_toy': OtherToySpec}
    with pytest.raises(TypeError, match=r"wrong spec type \['toy'\]"):
        register.check_group_fields(ToyGroup, expected)


def test_check_group_fields_rejects_a_field_that_is_not_optional():
    class Bad(sa.specs.AnalysisGroup, frozen=True, kw_only=True):
        toy: ToySpec = ToySpec()

    with pytest.raises(TypeError, match='Optional'):
        register.check_group_fields(Bad, {'toy': ToySpec})


def test_registry_and_any_analysis_agree():
    """the import-time guard in striqt.analysis.measurements, asserted by name"""
    expected = {info.name: spec_type for spec_type, info in sa.registry.items()}
    register.check_group_fields(sa.specs.BuiltinAnalyses, expected)
    assert set(sa.specs.BuiltinAnalyses.__struct_fields__) == set(expected)


def test_trigger_registry_and_any_signal_trigger_agree():
    expected = {
        info.name: info.meas_spec_type
        for key, info in sa.registry.signal_trigger.items()
        if isinstance(key, str)
    }
    register.check_group_fields(sa.specs.BuiltinTriggers, expected)
    triggers = set(sa.specs.BuiltinTriggers.__struct_fields__)
    assert triggers <= set(sa.specs.BuiltinAnalyses.__struct_fields__)


# %% KwArgCache


def build_counting_cache():
    """a KwArgCache keyed on `nfft` around a function that counts its calls"""
    cache = register.KwArgCache(['nfft'])
    calls = []

    @cache.apply
    def helper(iq, capture, *, nfft):
        calls.append((id(iq), nfft))
        return np.zeros(nfft)

    return cache, helper, calls


def test_kwargcache_hits_on_the_same_array_and_keywords():
    cache, helper, calls = build_counting_cache()
    x = np.ones((1, 4))

    with cache:
        first = helper(x, CAPTURE, nfft=8)
        second = helper(x, CAPTURE, nfft=8)

    assert second is first
    assert len(calls) == 1


def test_kwargcache_misses_on_a_different_array_with_the_same_keywords():
    cache, helper, calls = build_counting_cache()
    x = np.ones((1, 4))
    y = x.copy()

    with cache:
        helper(x, CAPTURE, nfft=8)
        helper(y, CAPTURE, nfft=8)

    assert [iq_id == id(y) for iq_id, _ in calls] == [False, True]


def test_kwargcache_holds_the_array_until_exit():
    cache, helper, _ = build_counting_cache()
    x = np.ones((1, 4))
    ref = weakref.ref(x)

    with cache:
        helper(x, CAPTURE, nfft=8)
        del x
        assert ref() is not None

    assert ref() is None


# %% check_group_registered

GROUP_CHECKS = {
    'fields': lambda group: register.check_group_fields(
        group, {'toy': ToySpec, 'other_toy': OtherToySpec}
    ),
    'registered': lambda group: register.check_group_registered(
        group, build_group_registry(None, None)
    ),
}


def test_check_group_registered_accepts_a_group_of_registered_measurements():
    registry = build_group_registry(None, None)
    assert register.check_group_registered(ToyGroup, registry) is None


def test_check_group_registered_accepts_a_subset_of_the_registry():
    class Subset(sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True):
        toy: Optional[ToySpec] = None

    assert (
        register.check_group_registered(Subset, build_group_registry(None, None))
        is None
    )


def test_check_group_registered_rejects_an_unregistered_spec_type():
    registry, _ = build_registry()
    assert OtherToySpec not in registry
    with pytest.raises(TypeError, match='other_toy: OtherToySpec is not a registered'):
        register.check_group_registered(ToyGroup, registry)


def test_check_group_registered_rejects_a_field_named_unlike_the_measurement():
    class Renamed(
        sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
    ):
        renamed: Optional[ToySpec] = None

    with pytest.raises(TypeError, match="renamed must be named 'toy'"):
        register.check_group_registered(Renamed, build_group_registry(None, None))


@pytest.mark.parametrize('check', GROUP_CHECKS.values(), ids=list(GROUP_CHECKS))
def test_group_checks_reject_a_group_without_omit_defaults(check):
    """`to_dict` and the saved attrs name only the enabled measurements, which needs
    ``omit_defaults=True`` on the group"""

    class Kept(sa.specs.AnalysisGroup, frozen=True, kw_only=True):
        toy: Optional[ToySpec] = None
        other_toy: Optional[OtherToySpec] = None

    with pytest.raises(TypeError, match='omit_defaults'):
        check(Kept)


@pytest.mark.parametrize('check', GROUP_CHECKS.values(), ids=list(GROUP_CHECKS))
def test_group_checks_reject_a_field_that_is_not_optional(check):
    class Bad(sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True):
        toy: ToySpec = ToySpec()
        other_toy: Optional[OtherToySpec] = None

    with pytest.raises(TypeError, match='Optional'):
        check(Bad)
